from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from drone_autopilot.debug import save_observation_snapshot
from drone_autopilot.simulators.base import Observation


def test_save_observation_snapshot_writes_rgb_depth_and_summary(tmp_path) -> None:
    observation = Observation(
        rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        depth_m=np.asarray([[5.0, 2.0, np.nan], [1.5, 12.0, 3.0]], dtype=np.float32),
    )

    summary = save_observation_snapshot(observation, tmp_path, depth_vis_max_m=10.0)

    assert summary["rgb_path"] == str(tmp_path / "rgb.png")
    assert summary["depth_path"] == str(tmp_path / "depth.png")
    assert summary["rgb_shape"] == [2, 3, 3]
    assert summary["depth_shape"] == [2, 3]
    assert summary["min_depth_m"] == pytest.approx(1.5)
    assert Image.open(tmp_path / "rgb.png").size == (3, 2)
    assert Image.open(tmp_path / "depth.png").size == (3, 2)


def test_save_observation_snapshot_rejects_non_positive_depth_visualization_range(tmp_path) -> None:
    observation = Observation(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_m=np.ones((2, 2), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="positive"):
        save_observation_snapshot(observation, tmp_path, depth_vis_max_m=0.0)
