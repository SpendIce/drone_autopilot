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


def test_safety_filter_close_obstacle_zeros_forward_only_command() -> None:
    safety = SafetyFilter(SafetyConfig(emergency_depth_m=1.0))
    depth = np.asarray([[2.0, 0.9], [3.0, 4.0]], dtype=np.float32)

    result = safety.filter([1.0, 0.0, 0.0, 0.0], depth_m=depth)

    assert result.emergency_stop
    assert result.reason == "close_obstacle"
    assert result.command == VelocityCommand.hover()
    assert result.min_depth_m == pytest.approx(0.9)


def test_safety_filter_close_obstacle_allows_escape_yaw_and_lateral() -> None:
    safety = SafetyFilter(
        SafetyConfig(
            emergency_depth_m=1.0,
            max_vy_mps=1.0,
            max_yaw_rate_radps=1.0,
            smoothing_alpha=1.0,
        )
    )
    depth = np.full((4, 4), 0.5, dtype=np.float32)

    result = safety.filter([0.8, -0.3, 0.0, 0.6], depth_m=depth)

    assert result.emergency_stop
    assert result.reason == "close_obstacle"
    assert result.command.vx == 0.0
    assert result.command.vy == pytest.approx(-0.3)
    assert result.command.yaw_rate == pytest.approx(0.6)


def test_safety_filter_close_obstacle_forbids_forward_approach() -> None:
    safety = SafetyFilter(SafetyConfig(emergency_depth_m=1.0, smoothing_alpha=1.0))
    depth = np.full((4, 4), 0.5, dtype=np.float32)

    result = safety.filter([2.0, 0.0, 0.0, 0.0], depth_m=depth)

    assert result.command.vx == 0.0


def test_safety_filter_close_obstacle_permits_backing_away() -> None:
    safety = SafetyFilter(
        SafetyConfig(emergency_depth_m=1.0, max_vx_mps=2.0, smoothing_alpha=1.0)
    )
    depth = np.full((4, 4), 0.5, dtype=np.float32)

    result = safety.filter([-1.5, 0.0, 0.0, 0.0], depth_m=depth)

    assert result.command.vx == pytest.approx(-1.5)


def test_safety_filter_close_obstacle_steers_by_raw_reactive_command() -> None:
    safety = SafetyFilter(
        SafetyConfig(emergency_depth_m=1.0, max_yaw_rate_radps=1.0, smoothing_alpha=1.0)
    )
    depth = np.full((4, 4), 0.5, dtype=np.float32)

    result = safety.filter(
        [0.5, 0.0, 0.0, 0.1],
        depth_m=depth,
        reactive=[0.5, 0.0, 0.0, 0.6],
    )

    assert result.command.yaw_rate == pytest.approx(0.6)


def test_safety_filter_close_obstacle_honors_raw_retreat_over_diluted_blend() -> None:
    """A mission-blended command can cancel a policy's raw retreat intent
    (blend of a negative reactive vx with a positive goal.vx nets near zero).
    The escape must still retreat if the raw source wants to."""
    safety = SafetyFilter(
        SafetyConfig(emergency_depth_m=1.0, max_vx_mps=2.0, smoothing_alpha=1.0)
    )
    depth = np.full((4, 4), 0.5, dtype=np.float32)

    result = safety.filter(
        [0.05, 0.0, 0.0, 0.0],  # blended command: goal.vx diluted the retreat to near zero
        depth_m=depth,
        reactive=[-0.3, 0.0, 0.0, 0.0],  # raw policy actively wants to back away
    )

    assert result.command.vx == pytest.approx(-0.3)


def test_safety_filter_close_obstacle_falls_back_without_reactive_command() -> None:
    safety = SafetyFilter(
        SafetyConfig(emergency_depth_m=1.0, max_yaw_rate_radps=1.0, smoothing_alpha=1.0)
    )
    depth = np.full((4, 4), 0.5, dtype=np.float32)

    result = safety.filter([0.5, 0.0, 0.0, 0.3], depth_m=depth)

    assert result.command.yaw_rate == pytest.approx(0.3)


def test_safety_filter_uses_configured_depth_roi_for_emergency_stop() -> None:
    safety = SafetyFilter(
        SafetyConfig(
            emergency_depth_m=1.0,
            depth_roi_bottom=0.7,
        )
    )
    depth = np.full((10, 10), 3.0, dtype=np.float32)
    depth[7:, :] = 0.5

    result = safety.filter([1.0, 0.0, 0.0, 0.0], depth_m=depth)

    assert not result.emergency_stop
    assert result.reason == "ok"
    assert result.min_depth_m == pytest.approx(3.0)


def test_safety_filter_rejects_invalid_depth_roi() -> None:
    with pytest.raises(ValueError, match="depth ROI"):
        SafetyFilter(SafetyConfig(depth_roi_top=0.8, depth_roi_bottom=0.7))


def test_safety_filter_zeros_targets_below_deadbands() -> None:
    safety = SafetyFilter(
        SafetyConfig(
            smoothing_alpha=1.0,
            vx_deadband_mps=0.1,
            vy_deadband_mps=0.1,
            vz_deadband_mps=0.1,
            yaw_rate_deadband_radps=0.1,
        )
    )

    result = safety.filter([0.09, -0.09, 0.09, -0.09])

    assert not result.emergency_stop
    assert result.command == VelocityCommand.hover()


def test_safety_filter_keeps_commands_above_deadbands() -> None:
    safety = SafetyFilter(
        SafetyConfig(
            smoothing_alpha=1.0,
            vx_deadband_mps=0.1,
            yaw_rate_deadband_radps=0.1,
        )
    )

    result = safety.filter([0.11, 0.0, 0.0, -0.11])

    assert result.command == VelocityCommand(0.11, 0.0, 0.0, -0.11)


def test_safety_filter_deadband_does_not_block_smoothed_ramp() -> None:
    safety = SafetyFilter(
        SafetyConfig(
            max_yaw_rate_radps=0.18,
            smoothing_alpha=0.15,
            yaw_rate_deadband_radps=0.08,
        )
    )

    first = safety.filter([0.0, 0.0, 0.0, 0.43])
    second = safety.filter([0.0, 0.0, 0.0, 0.43])

    assert first.command.yaw_rate == pytest.approx(0.027)
    assert second.command.yaw_rate == pytest.approx(0.04995)
