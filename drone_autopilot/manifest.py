"""Dataset manifest construction and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .core_types import ACTION_NAMES, ACTION_UNITS, DatasetManifestRecord, VelocityCommand


VALID_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class ManifestSummary:
    rows: int
    sources: dict[str, int]
    splits: dict[str, int]
    labeled_actions: int
    action_mean: tuple[float, float, float, float] | None
    action_std: tuple[float, float, float, float] | None


def _stable_fraction(value: str) -> float:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def split_for_episode(
    source: str,
    episode_id: str,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> str:
    """Deterministically assign an entire episode/source pair to one split."""

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    fraction = _stable_fraction(f"{source}:{episode_id}")
    if fraction < train_ratio:
        return "train"
    if fraction < train_ratio + val_ratio:
        return "val"
    return "test"


def _relative_or_absolute(path: Path, path_root: Path | None) -> str:
    if path_root is None:
        return str(path)
    try:
        return str(path.relative_to(path_root))
    except ValueError:
        return str(path)


def build_airsim_seed_manifest(
    dataset_root: Path | str,
    *,
    source: str = "airsim_obstacle_avoidance_seed",
    path_root: Path | str | None = None,
    episode_length: int = 500,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> list[DatasetManifestRecord]:
    """Build a manifest for the local AirSim RGB/depth/commands dataset.

    Expected layout:

    - rgb/<frame>.png
    - depth/<frame>.npy
    - commands/<frame>.npy, with [vx, vy, vz, yaw_rate_deg_s]
    """

    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(root)
    if episode_length <= 0:
        raise ValueError("episode_length must be positive")

    rgb_dir = root / "rgb"
    depth_dir = root / "depth"
    command_dir = root / "commands"
    for required in (rgb_dir, depth_dir, command_dir):
        if not required.is_dir():
            raise FileNotFoundError(f"Missing dataset directory: {required}")

    resolved_path_root = Path(path_root) if path_root is not None else root
    records: list[DatasetManifestRecord] = []

    for rgb_path in sorted(rgb_dir.glob("*.png")):
        frame_id = rgb_path.stem
        depth_path = depth_dir / f"{frame_id}.npy"
        command_path = command_dir / f"{frame_id}.npy"
        if not depth_path.exists() or not command_path.exists():
            continue

        command = VelocityCommand.from_iterable(np.load(command_path), yaw_unit="deg/s")
        frame_index = int(frame_id)
        episode_id = f"{source}_chunk_{frame_index // episode_length:04d}"
        split = split_for_episode(
            source,
            episode_id,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

        records.append(
            DatasetManifestRecord(
                source=source,
                episode_id=episode_id,
                frame_id=frame_id,
                rgb_path=_relative_or_absolute(rgb_path, resolved_path_root),
                depth_path=_relative_or_absolute(depth_path, resolved_path_root),
                action=tuple(float(v) for v in command.to_numpy(np.float64)),  # type: ignore[arg-type]
                action_mask=(True, True, True, True),
                split=split,
                action_frame="body",
                units=ACTION_UNITS,
            )
        )

    return records


def records_to_dataframe(records: Sequence[DatasetManifestRecord]):
    import pandas as pd

    return pd.DataFrame([record.to_row() for record in records], columns=DatasetManifestRecord.COLUMNS)


def write_manifest(records: Sequence[DatasetManifestRecord], path: Path | str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = records_to_dataframe(records)
    if output.suffix == ".parquet":
        df.to_parquet(output, index=False)
    elif output.suffix in {".csv", ".txt"}:
        df.to_csv(output, index=False)
    else:
        raise ValueError("Manifest path must end with .parquet or .csv")


def read_manifest(path: Path | str) -> list[DatasetManifestRecord]:
    import pandas as pd

    manifest_path = Path(path)
    if manifest_path.suffix == ".parquet":
        df = pd.read_parquet(manifest_path)
    elif manifest_path.suffix in {".csv", ".txt"}:
        string_columns = {
            "source",
            "episode_id",
            "frame_id",
            "rgb_path",
            "depth_path",
            "pose",
            "action",
            "action_mask",
            "split",
            "action_frame",
            "units",
        }
        df = pd.read_csv(
            manifest_path,
            dtype={column: "string" for column in string_columns},
        )
    else:
        raise ValueError("Manifest path must end with .parquet or .csv")

    missing = set(DatasetManifestRecord.COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    return [DatasetManifestRecord.from_row(row) for row in df.to_dict(orient="records")]


def validate_alignment(records: Iterable[DatasetManifestRecord], *, data_root: Path | str = ".") -> list[str]:
    """Return validation errors for missing files or malformed rows."""

    root = Path(data_root)
    errors: list[str] = []
    for index, record in enumerate(records):
        if record.split not in VALID_SPLITS:
            errors.append(f"row {index}: invalid split {record.split!r}")
        rgb_path = root / record.rgb_path
        if not rgb_path.exists():
            errors.append(f"row {index}: missing rgb file {rgb_path}")
        if record.depth_path:
            depth_path = root / record.depth_path
            if not depth_path.exists():
                errors.append(f"row {index}: missing depth file {depth_path}")
        if record.action is None and any(record.action_mask):
            errors.append(f"row {index}: action_mask is true but action is missing")
    return errors


def ensure_episode_split_integrity(records: Iterable[DatasetManifestRecord]) -> list[str]:
    """Return errors when the same source/episode appears in multiple splits."""

    seen: dict[tuple[str, str], str] = {}
    errors: list[str] = []
    for record in records:
        key = (record.source, record.episode_id)
        previous = seen.get(key)
        if previous is None:
            seen[key] = record.split
        elif previous != record.split:
            errors.append(
                f"{record.source}/{record.episode_id} appears in both {previous!r} and {record.split!r}"
            )
    return errors


def summarize_manifest(records: Sequence[DatasetManifestRecord]) -> ManifestSummary:
    sources: dict[str, int] = {}
    splits: dict[str, int] = {}
    actions: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for record in records:
        sources[record.source] = sources.get(record.source, 0) + 1
        splits[record.split] = splits.get(record.split, 0) + 1
        if record.action is not None:
            actions.append(np.asarray(record.action, dtype=np.float64))
            masks.append(np.asarray(record.action_mask, dtype=bool))

    if not actions:
        return ManifestSummary(len(records), sources, splits, 0, None, None)

    action_matrix = np.stack(actions)
    mask_matrix = np.stack(masks)
    mean_values: list[float] = []
    std_values: list[float] = []
    for dim in range(len(ACTION_NAMES)):
        values = action_matrix[mask_matrix[:, dim], dim]
        if len(values) == 0:
            mean_values.append(float("nan"))
            std_values.append(float("nan"))
        else:
            mean_values.append(float(values.mean()))
            std_values.append(float(values.std(ddof=0)))

    return ManifestSummary(
        rows=len(records),
        sources=sources,
        splits=splits,
        labeled_actions=len(actions),
        action_mean=tuple(mean_values),  # type: ignore[arg-type]
        action_std=tuple(std_values),  # type: ignore[arg-type]
    )
