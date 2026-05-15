from __future__ import annotations

import numpy as np
import pytest

from drone_autopilot.safety import SafetyConfig, SafetyFilter
from drone_autopilot.core_types import VelocityCommand


def test_safety_filter_rejects_invalid_predictions() -> None:
    safety = SafetyFilter()

    result = safety.filter([float("nan"), 0.0, 0.0, 0.0])

    assert result.emergency_stop
    assert result.reason == "invalid_prediction"
    assert result.command == VelocityCommand.hover()


def test_safety_filter_clamps_and_smooths_commands() -> None:
    safety = SafetyFilter(
        SafetyConfig(
            max_vx_mps=2.0,
            max_vy_mps=1.0,
            max_vz_mps=1.0,
            max_yaw_rate_radps=0.5,
            smoothing_alpha=0.5,
        )
    )

    result = safety.filter([10.0, -10.0, 10.0, 10.0])

    assert not result.emergency_stop
    assert result.command == VelocityCommand(1.0, -0.5, 0.5, 0.25)


def test_safety_filter_hovers_on_close_depth() -> None:
    safety = SafetyFilter(SafetyConfig(emergency_depth_m=1.0))
    depth = np.asarray([[2.0, 0.9], [3.0, 4.0]], dtype=np.float32)

    result = safety.filter([1.0, 0.0, 0.0, 0.0], depth_m=depth)

    assert result.emergency_stop
    assert result.reason == "close_obstacle"
    assert result.command == VelocityCommand.hover()
    assert result.min_depth_m == pytest.approx(0.9)
