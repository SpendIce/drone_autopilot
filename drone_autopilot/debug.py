"""Debug helpers for simulator observations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .simulators.base import Observation


def save_observation_snapshot(
    observation: Observation,
    output_dir: Path | str,
    *,
    depth_vis_max_m: float = 10.0,
) -> dict[str, Any]:
    """Save RGB and depth visualization files for a simulator observation."""
    if depth_vis_max_m <= 0.0:
        raise ValueError("depth_vis_max_m must be positive")

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    rgb = np.asarray(observation.rgb, dtype=np.uint8)
    depth = np.asarray(observation.depth_m, dtype=np.float32)
    finite_positive = depth[np.isfinite(depth) & (depth > 0.0)]
    min_depth = float(finite_positive.min()) if finite_positive.size else None

    rgb_path = directory / "rgb.png"
    depth_path = directory / "depth.png"
    Image.fromarray(rgb).convert("RGB").save(rgb_path)
    Image.fromarray(_depth_to_uint8(depth, depth_vis_max_m)).save(depth_path)

    return {
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "rgb_shape": list(rgb.shape),
        "depth_shape": list(depth.shape),
        "min_depth_m": min_depth,
    }


def _depth_to_uint8(depth_m: np.ndarray, max_depth_m: float) -> np.ndarray:
    finite = np.nan_to_num(depth_m, nan=max_depth_m, posinf=max_depth_m, neginf=max_depth_m)
    clipped = np.clip(finite, 0.0, max_depth_m)
    near_is_bright = 1.0 - (clipped / max_depth_m)
    return (near_is_bright * 255.0).astype(np.uint8)
