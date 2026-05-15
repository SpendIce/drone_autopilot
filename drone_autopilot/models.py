"""RGB-D pilot networks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import MissingOptionalDependencyError

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def _require_torch():
    if torch is None or nn is None:
        raise MissingOptionalDependencyError("torch", "training")
    return torch, nn


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "mobilenet_v3_small"
    output_dim: int = 4
    dropout: float = 0.1


class _TinyEncoder(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, in_channels: int, feature_dim: int = 128) -> None:
        _torch, module = _require_torch()
        super().__init__()
        self.feature_dim = feature_dim
        self.net = module.Sequential(
            module.Conv2d(in_channels, 24, kernel_size=5, stride=2, padding=2),
            module.BatchNorm2d(24),
            module.ReLU(inplace=True),
            module.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            module.BatchNorm2d(48),
            module.ReLU(inplace=True),
            module.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            module.BatchNorm2d(96),
            module.ReLU(inplace=True),
            module.AdaptiveAvgPool2d(1),
            module.Flatten(),
            module.Linear(96, feature_dim),
            module.ReLU(inplace=True),
        )

    def forward(self, x):  # type: ignore[no-untyped-def]
        return self.net(x)


def _torchvision_encoder(backbone: str, in_channels: int):
    _torch, module = _require_torch()
    try:
        import torchvision.models as models
    except ModuleNotFoundError as exc:
        raise MissingOptionalDependencyError("torchvision", "training") from exc

    if backbone == "resnet18":
        model = models.resnet18(weights=None)
        feature_dim = model.fc.in_features
        model.fc = module.Identity()
    elif backbone == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        feature_dim = model.classifier[0].in_features
        model.classifier = module.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    if in_channels == 3:
        return model, feature_dim

    adapter = module.Sequential(module.Conv2d(in_channels, 3, kernel_size=1), model)
    return adapter, feature_dim


class RgbdPilotNet(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Two-stream RGB-D CNN that predicts [vx, vy, vz, yaw_rate]."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        _torch, module = _require_torch()
        super().__init__()
        self.config = config or ModelConfig()
        if self.config.backbone == "tiny":
            self.rgb_encoder = _TinyEncoder(3)
            self.depth_encoder = _TinyEncoder(1)
            feature_dim = 128
        else:
            self.rgb_encoder, feature_dim = _torchvision_encoder(self.config.backbone, 3)
            self.depth_encoder, _ = _torchvision_encoder(self.config.backbone, 1)

        self.head = module.Sequential(
            module.Linear(feature_dim * 2, 256),
            module.ReLU(inplace=True),
            module.Dropout(self.config.dropout),
            module.Linear(256, self.config.output_dim),
        )

    def forward(self, rgb, depth):  # type: ignore[no-untyped-def]
        rgb_features = self.rgb_encoder(rgb)
        depth_features = self.depth_encoder(depth)
        return self.head(torch.cat([rgb_features, depth_features], dim=1))  # type: ignore[union-attr]


def build_model(config: ModelConfig | None = None) -> RgbdPilotNet:
    return RgbdPilotNet(config)


def save_checkpoint(
    path: Path | str,
    *,
    model: RgbdPilotNet,
    model_config: ModelConfig,
    action_stats: Any,
    extra: dict[str, Any] | None = None,
) -> None:
    torch_module, _module = _require_torch()
    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": model_config.__dict__,
        "action_stats": {
            "mean": tuple(action_stats.mean),
            "std": tuple(action_stats.std),
        },
        "extra": extra or {},
    }
    torch_module.save(checkpoint, path)


def load_checkpoint(path: Path | str, *, map_location: str = "cpu"):
    torch_module, _module = _require_torch()
    checkpoint = torch_module.load(path, map_location=map_location)
    config = ModelConfig(**checkpoint["model_config"])
    model = RgbdPilotNet(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint
