"""Manifest-backed RGB-D PyTorch dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .exceptions import MissingOptionalDependencyError
from .core_types import DatasetManifestRecord

try:
    from torch.utils.data import Dataset as _TorchDataset
except ModuleNotFoundError:
    _TorchDataset = object  # type: ignore[assignment]


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise MissingOptionalDependencyError("torch", "training") from exc
    return torch


@dataclass(frozen=True)
class ActionStats:
    mean: tuple[float, float, float, float]
    std: tuple[float, float, float, float]

    @classmethod
    def from_records(cls, records: Sequence[DatasetManifestRecord]) -> "ActionStats":
        actions: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for record in records:
            if record.action is None:
                continue
            actions.append(np.asarray(record.action, dtype=np.float64))
            masks.append(np.asarray(record.action_mask, dtype=bool))

        if not actions:
            return cls(mean=(0.0, 0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0, 1.0))

        action_matrix = np.stack(actions)
        mask_matrix = np.stack(masks)
        means: list[float] = []
        stds: list[float] = []
        for dim in range(4):
            values = action_matrix[mask_matrix[:, dim], dim]
            if len(values) == 0:
                means.append(0.0)
                stds.append(1.0)
                continue
            std = float(values.std(ddof=0))
            means.append(float(values.mean()))
            stds.append(std if std > 1e-6 else 1.0)
        return cls(mean=tuple(means), std=tuple(stds))  # type: ignore[arg-type]


class PilotManifestDataset(_TorchDataset):  # type: ignore[misc]
    """Loads RGB, depth, action, and action masks from a manifest."""

    def __init__(
        self,
        records: Sequence[DatasetManifestRecord],
        *,
        data_root: Path | str = ".",
        image_size: int = 224,
        max_depth_m: float = 50.0,
        modality: str = "rgbd",
        action_stats: ActionStats | None = None,
    ) -> None:
        torch = _require_torch()
        super().__init__()
        if modality not in {"rgb", "depth", "rgbd"}:
            raise ValueError("modality must be one of: rgb, depth, rgbd")
        self.torch = torch
        self.records = list(records)
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.max_depth_m = max_depth_m
        self.modality = modality
        self.action_stats = action_stats

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        rgb = self._load_rgb(record.rgb_path)
        depth = self._load_depth(record.depth_path)
        action = np.zeros(4, dtype=np.float32)
        action_mask = np.zeros(4, dtype=np.float32)
        if record.action is not None:
            action = np.asarray(record.action, dtype=np.float32)
            action_mask = np.asarray(record.action_mask, dtype=np.float32)
        if self.action_stats is not None:
            mean = np.asarray(self.action_stats.mean, dtype=np.float32)
            std = np.asarray(self.action_stats.std, dtype=np.float32)
            action = (action - mean) / std

        torch = self.torch
        return {
            "rgb": torch.from_numpy(rgb),
            "depth": torch.from_numpy(depth),
            "action": torch.from_numpy(action),
            "action_mask": torch.from_numpy(action_mask),
            "source": record.source,
            "frame_id": record.frame_id,
        }

    def _load_rgb(self, path: str) -> np.ndarray:
        image = Image.open(self.data_root / path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        if self.modality == "depth":
            return np.zeros_like((array - mean) / std, dtype=np.float32)
        return ((array - mean) / std).astype(np.float32)

    def _load_depth(self, path: str | None) -> np.ndarray:
        if path is None or self.modality == "rgb":
            return np.zeros((1, self.image_size, self.image_size), dtype=np.float32)

        depth = np.load(self.data_root / path).astype(np.float32)
        depth = np.nan_to_num(depth, nan=self.max_depth_m, posinf=self.max_depth_m, neginf=0.0)
        depth = np.clip(depth, 0.0, self.max_depth_m) / self.max_depth_m
        image = Image.fromarray((depth * 65535.0).astype(np.uint16))
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        resized = np.asarray(image, dtype=np.float32) / 65535.0
        return resized[None, :, :].astype(np.float32)
