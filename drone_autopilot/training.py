"""Training and offline evaluation pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data import ActionStats, PilotManifestDataset
from .exceptions import MissingOptionalDependencyError
from .manifest import read_manifest
from .models import ModelConfig, build_model, save_checkpoint
from .core_types import ACTION_NAMES, DatasetManifestRecord

BEST_VAL_SCORE_WEIGHTS = {
    "vx": 1.0,
    "vy": 0.0,
    "vz": 1.0,
    "yaw_rate": 1.0,
}


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise MissingOptionalDependencyError("torch", "training") from exc
    return torch, DataLoader


@dataclass(frozen=True)
class TrainingConfig:
    manifest_path: Path
    data_root: Path
    output_path: Path = Path("checkpoints/rgbd_pilot.pt")
    best_output_path: Path | None = None
    backbone: str = "mobilenet_v3_small"
    modality: str = "rgbd"
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 1e-4
    num_workers: int = 0
    max_depth_m: float = 50.0
    vy_weight: float = 0.25
    device: str = "auto"
    multi_gpu: bool = False
    distributed: bool = False


@dataclass(frozen=True)
class MetricTable:
    mae: dict[str, float]
    rmse: dict[str, float]
    per_source_mae: dict[str, dict[str, float]]


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _device_type(device: object) -> str:
    return getattr(device, "type", str(device).split(":", maxsplit=1)[0])


def _should_use_data_parallel(torch_module: Any, device: object, multi_gpu: bool) -> bool:
    return (
        multi_gpu
        and _device_type(device) == "cuda"
        and torch_module.cuda.device_count() > 1
    )


def _distributed_env() -> DistributedContext:
    return DistributedContext(
        enabled=False,
        rank=int(os.environ.get("RANK", "0")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
    )


def _setup_distributed(torch_module: Any, requested: bool) -> tuple[DistributedContext, object]:
    env = _distributed_env()
    if not requested and env.world_size <= 1:
        return DistributedContext(), "auto"
    if env.world_size <= 1:
        raise ValueError(
            "Distributed training requires torchrun with --nproc_per_node > 1"
        )
    if not torch_module.cuda.is_available():
        raise ValueError("Distributed training currently requires CUDA GPUs")

    torch_module.cuda.set_device(env.local_rank)
    if not torch_module.distributed.is_initialized():
        torch_module.distributed.init_process_group(backend="nccl")
    return (
        DistributedContext(
            enabled=True,
            rank=env.rank,
            local_rank=env.local_rank,
            world_size=env.world_size,
        ),
        torch_module.device("cuda", env.local_rank),
    )


def _cleanup_distributed(torch_module: Any, context: DistributedContext) -> None:
    if context.enabled and torch_module.distributed.is_initialized():
        torch_module.distributed.destroy_process_group()


def _barrier(torch_module: Any, context: DistributedContext) -> None:
    if not context.enabled:
        return
    try:
        torch_module.distributed.barrier(device_ids=[context.local_rank])
    except TypeError:
        torch_module.distributed.barrier()


def _unwrap_distributed_model(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def _default_best_output_path(output_path: Path | str) -> Path:
    output_path = Path(output_path)
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}_best{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}_best")


def _validation_score(
    row: dict[str, float],
    normalization: dict[str, float] | None = None,
) -> float | None:
    """Weighted sum of per-output validation MAE.

    ``normalization`` divides each MAE by its action's training-split std
    before weighting, so linear (m/s) and angular (rad/s) errors become
    comparable relative-error terms instead of being summed as raw units.
    """
    score = 0.0
    used = False
    for name, weight in BEST_VAL_SCORE_WEIGHTS.items():
        if weight == 0.0:
            continue
        key = f"val_mae_{name}"
        if key not in row:
            continue
        scale = normalization.get(name, 1.0) if normalization else 1.0
        scale = scale if scale > 1e-8 else 1.0
        score += (row[key] / scale) * weight
        used = True
    return score if used else None


def _reduce_training_loss(
    torch_module: Any,
    *,
    running_loss: float,
    batches: int,
    device: object,
    context: DistributedContext,
) -> tuple[float, int]:
    if not context.enabled:
        return running_loss, batches

    totals = torch_module.tensor(
        [running_loss, float(batches)],
        dtype=torch_module.float64,
        device=device,
    )
    torch_module.distributed.all_reduce(
        totals,
        op=torch_module.distributed.ReduceOp.SUM,
    )
    return float(totals[0].item()), int(totals[1].item())


def _split_records(
    records: Sequence[DatasetManifestRecord],
) -> tuple[list[DatasetManifestRecord], list[DatasetManifestRecord], list[DatasetManifestRecord]]:
    train = [record for record in records if record.split == "train"]
    val = [record for record in records if record.split == "val"]
    test = [record for record in records if record.split == "test"]
    if not train:
        raise ValueError("Manifest has no train records")
    return train, val, test


def masked_huber_loss(prediction, target, mask, action_weights):  # type: ignore[no-untyped-def]
    import torch.nn.functional as F

    losses = F.smooth_l1_loss(prediction, target, reduction="none")
    weighted_mask = mask * action_weights
    denominator = weighted_mask.sum().clamp_min(1.0)
    return (losses * weighted_mask).sum() / denominator


def evaluate_model(
    model,
    records: Sequence[DatasetManifestRecord],
    *,
    data_root: Path,
    action_stats: ActionStats,
    batch_size: int = 32,
    image_size: int = 224,
    max_depth_m: float = 50.0,
    modality: str = "rgbd",
    device: str = "cpu",
) -> MetricTable:
    torch, DataLoader = _require_torch()
    if not records:
        return MetricTable(mae={}, rmse={}, per_source_mae={})

    dataset = PilotManifestDataset(
        records,
        data_root=data_root,
        image_size=image_size,
        max_depth_m=max_depth_m,
        modality=modality,
        action_stats=action_stats,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    mean = torch.tensor(action_stats.mean, dtype=torch.float32, device=device)
    std = torch.tensor(action_stats.std, dtype=torch.float32, device=device)

    errors: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    sources: list[str] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)
            target = batch["action"].to(device)
            mask = batch["action_mask"].to(device)
            prediction = model(rgb, depth)
            prediction_raw = (prediction * std) + mean
            target_raw = (target * std) + mean
            errors.append((prediction_raw - target_raw).detach().cpu().numpy())
            masks.append(mask.detach().cpu().numpy())
            sources.extend(batch["source"])

    error_matrix = np.concatenate(errors, axis=0)
    mask_matrix = np.concatenate(masks, axis=0).astype(bool)
    mae: dict[str, float] = {}
    rmse: dict[str, float] = {}
    for dim, name in enumerate(ACTION_NAMES):
        values = error_matrix[mask_matrix[:, dim], dim]
        if len(values) == 0:
            continue
        mae[name] = float(np.mean(np.abs(values)))
        rmse[name] = float(np.sqrt(np.mean(values**2)))

    per_source_mae: dict[str, dict[str, float]] = {}
    source_array = np.asarray(sources, dtype=object)
    for source in sorted(set(sources)):
        source_mask = source_array == source
        source_metrics: dict[str, float] = {}
        for dim, name in enumerate(ACTION_NAMES):
            dim_mask = source_mask & mask_matrix[:, dim]
            values = error_matrix[dim_mask, dim]
            if len(values) > 0:
                source_metrics[name] = float(np.mean(np.abs(values)))
        per_source_mae[source] = source_metrics

    return MetricTable(mae=mae, rmse=rmse, per_source_mae=per_source_mae)


def train(config: TrainingConfig) -> dict[str, object]:
    torch, DataLoader = _require_torch()
    context, distributed_device = _setup_distributed(torch, config.distributed)
    records = read_manifest(config.manifest_path)
    train_records, val_records, test_records = _split_records(records)
    action_stats = ActionStats.from_records(train_records)
    score_normalization = dict(zip(ACTION_NAMES, action_stats.std))

    train_dataset = PilotManifestDataset(
        train_records,
        data_root=config.data_root,
        image_size=config.image_size,
        max_depth_m=config.max_depth_m,
        modality=config.modality,
        action_stats=action_stats,
    )
    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
        )
        if context.enabled
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.num_workers,
    )

    device = distributed_device if context.enabled else config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_config = ModelConfig(backbone=config.backbone)
    model = build_model(model_config).to(device)
    if context.enabled:
        print(
            f"Using DistributedDataParallel rank={context.rank} "
            f"local_rank={context.local_rank} world_size={context.world_size}"
        )
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
        )
    if not context.enabled and _should_use_data_parallel(torch, device, config.multi_gpu):
        print(f"Using DataParallel with {torch.cuda.device_count()} CUDA devices")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    action_weights = torch.tensor(
        [1.0, config.vy_weight, 1.0, 1.0],
        dtype=torch.float32,
        device=device,
    )

    history: list[dict[str, float]] = []
    best_output_path = config.best_output_path or _default_best_output_path(config.output_path)
    best_epoch: float | None = None
    best_val_score: float | None = None
    try:
        for epoch in range(config.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            running_loss = 0.0
            batches = 0
            for batch in train_loader:
                rgb = batch["rgb"].to(device)
                depth = batch["depth"].to(device)
                target = batch["action"].to(device)
                mask = batch["action_mask"].to(device)
                prediction = model(rgb, depth)
                loss = masked_huber_loss(prediction, target, mask, action_weights)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.detach().cpu())
                batches += 1

            running_loss, batches = _reduce_training_loss(
                torch,
                running_loss=running_loss,
                batches=batches,
                device=device,
                context=context,
            )
            train_loss = running_loss / max(batches, 1)
            row = {"epoch": float(epoch + 1), "train_loss": train_loss}
            if context.is_main and val_records:
                val_metrics = evaluate_model(
                    _unwrap_distributed_model(model),
                    val_records,
                    data_root=config.data_root,
                    action_stats=action_stats,
                    batch_size=config.batch_size,
                    image_size=config.image_size,
                    max_depth_m=config.max_depth_m,
                    modality=config.modality,
                    device=device,
                )
                for name, value in val_metrics.mae.items():
                    row[f"val_mae_{name}"] = value
                val_score = _validation_score(row, score_normalization)
                if val_score is not None:
                    row["val_score"] = val_score
                    if best_val_score is None or val_score < best_val_score:
                        best_val_score = val_score
                        best_epoch = row["epoch"]
                        best_output_path.parent.mkdir(parents=True, exist_ok=True)
                        save_checkpoint(
                            best_output_path,
                            model=_unwrap_distributed_model(model),
                            model_config=model_config,
                            action_stats=action_stats,
                            extra={
                                "history": [*history, row],
                                "modality": config.modality,
                                "checkpoint_kind": "best",
                                "best_epoch": best_epoch,
                                "best_val_score": best_val_score,
                                "best_metric_weights": BEST_VAL_SCORE_WEIGHTS,
                            },
                        )
            if context.is_main:
                history.append(row)
                print(row)
            if context.enabled:
                _barrier(torch, context)

        result: dict[str, object]
        if context.is_main:
            config.output_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                config.output_path,
                model=_unwrap_distributed_model(model),
                model_config=model_config,
                action_stats=action_stats,
                extra={
                    "history": history,
                    "modality": config.modality,
                    "checkpoint_kind": "last",
                    "best_checkpoint": (
                        str(best_output_path) if best_val_score is not None else None
                    ),
                    "best_epoch": best_epoch,
                    "best_val_score": best_val_score,
                    "best_metric_weights": BEST_VAL_SCORE_WEIGHTS,
                },
            )

            test_metrics = evaluate_model(
                _unwrap_distributed_model(model),
                test_records,
                data_root=config.data_root,
                action_stats=action_stats,
                batch_size=config.batch_size,
                image_size=config.image_size,
                max_depth_m=config.max_depth_m,
                modality=config.modality,
                device=device,
            )
            result = {
                "checkpoint": str(config.output_path),
                "best_checkpoint": (
                    str(best_output_path) if best_val_score is not None else None
                ),
                "best_epoch": best_epoch,
                "best_val_score": best_val_score,
                "history": history,
                "test_mae": test_metrics.mae,
                "test_rmse": test_metrics.rmse,
                "test_per_source_mae": test_metrics.per_source_mae,
            }
        else:
            result = {"_suppress_cli_output": True}

        if context.enabled:
            _barrier(torch, context)
        return result
    finally:
        _cleanup_distributed(torch, context)
