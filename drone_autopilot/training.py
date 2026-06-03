"""Training and offline evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data import ActionStats, PilotManifestDataset
from .exceptions import MissingOptionalDependencyError
from .manifest import read_manifest
from .models import ModelConfig, build_model, save_checkpoint
from .core_types import ACTION_NAMES, DatasetManifestRecord


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


@dataclass(frozen=True)
class MetricTable:
    mae: dict[str, float]
    rmse: dict[str, float]
    per_source_mae: dict[str, dict[str, float]]


def _device_type(device: object) -> str:
    return getattr(device, "type", str(device).split(":", maxsplit=1)[0])


def _should_use_data_parallel(torch_module: Any, device: object, multi_gpu: bool) -> bool:
    return (
        multi_gpu
        and _device_type(device) == "cuda"
        and torch_module.cuda.device_count() > 1
    )


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
    records = read_manifest(config.manifest_path)
    train_records, val_records, test_records = _split_records(records)
    action_stats = ActionStats.from_records(train_records)

    train_dataset = PilotManifestDataset(
        train_records,
        data_root=config.data_root,
        image_size=config.image_size,
        max_depth_m=config.max_depth_m,
        modality=config.modality,
        action_stats=action_stats,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_config = ModelConfig(backbone=config.backbone)
    model = build_model(model_config).to(device)
    if _should_use_data_parallel(torch, device, config.multi_gpu):
        print(f"Using DataParallel with {torch.cuda.device_count()} CUDA devices")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    action_weights = torch.tensor(
        [1.0, config.vy_weight, 1.0, 1.0],
        dtype=torch.float32,
        device=device,
    )

    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
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

        train_loss = running_loss / max(batches, 1)
        row = {"epoch": float(epoch + 1), "train_loss": train_loss}
        if val_records:
            val_metrics = evaluate_model(
                model,
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
        history.append(row)
        print(row)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        config.output_path,
        model=model,
        model_config=model_config,
        action_stats=action_stats,
        extra={"history": history, "modality": config.modality},
    )

    test_metrics = evaluate_model(
        model,
        test_records,
        data_root=config.data_root,
        action_stats=action_stats,
        batch_size=config.batch_size,
        image_size=config.image_size,
        max_depth_m=config.max_depth_m,
        modality=config.modality,
        device=device,
    )
    return {
        "checkpoint": str(config.output_path),
        "history": history,
        "test_mae": test_metrics.mae,
        "test_rmse": test_metrics.rmse,
        "test_per_source_mae": test_metrics.per_source_mae,
    }
