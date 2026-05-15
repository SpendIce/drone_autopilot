"""Policy wrappers for running trained models in simulators."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .data import ActionStats
from .exceptions import MissingOptionalDependencyError
from .models import load_checkpoint
from .core_types import VelocityCommand


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise MissingOptionalDependencyError("torch", "training") from exc
    return torch


class TorchPilotPolicy:
    """Model-backed policy that maps RGB-D observations to velocity commands."""

    def __init__(
        self,
        checkpoint_path: Path | str,
        *,
        image_size: int = 224,
        max_depth_m: float = 50.0,
        device: str = "auto",
    ) -> None:
        torch = _require_torch()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.device = device
        self.image_size = image_size
        self.max_depth_m = max_depth_m
        self.model, checkpoint = load_checkpoint(checkpoint_path, map_location=device)
        self.model.to(device)
        stats = checkpoint["action_stats"]
        self.action_stats = ActionStats(mean=tuple(stats["mean"]), std=tuple(stats["std"]))  # type: ignore[arg-type]

    def predict(self, rgb: np.ndarray, depth_m: np.ndarray) -> VelocityCommand:
        rgb_tensor = self._rgb_tensor(rgb)
        depth_tensor = self._depth_tensor(depth_m)
        mean = self.torch.tensor(self.action_stats.mean, dtype=self.torch.float32, device=self.device)
        std = self.torch.tensor(self.action_stats.std, dtype=self.torch.float32, device=self.device)
        self.model.eval()
        with self.torch.no_grad():
            prediction = self.model(rgb_tensor, depth_tensor)[0]
            raw = (prediction * std) + mean
        return VelocityCommand.from_iterable(raw.detach().cpu().numpy())

    def _rgb_tensor(self, rgb: np.ndarray):
        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        array = ((array - mean) / std)[None, :, :, :]
        return self.torch.from_numpy(array).to(self.device)

    def _depth_tensor(self, depth_m: np.ndarray):
        depth = np.asarray(depth_m, dtype=np.float32)
        depth = np.nan_to_num(depth, nan=self.max_depth_m, posinf=self.max_depth_m, neginf=0.0)
        depth = np.clip(depth, 0.0, self.max_depth_m) / self.max_depth_m
        image = Image.fromarray((depth * 65535.0).astype(np.uint16))
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        array = (np.asarray(image, dtype=np.float32) / 65535.0)[None, None, :, :]
        return self.torch.from_numpy(array).to(self.device)
