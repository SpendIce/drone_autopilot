from __future__ import annotations

import numpy as np
import pytest

from drone_autopilot.expert_policy import ReactiveAvoidanceConfig, ReactiveAvoidancePolicy


def _depth(value: float, *, shape: tuple[int, int] = (40, 60)) -> np.ndarray:
    return np.full(shape, value, dtype=np.float32)


def _rgb(shape: tuple[int, int] = (40, 60)) -> np.ndarray:
    return np.zeros((*shape, 3), dtype=np.uint8)


def test_cruises_straight_when_clear() -> None:
    policy = ReactiveAvoidancePolicy()

    command = policy.predict(_rgb(), _depth(10.0))

    assert command.vx == pytest.approx(policy.config.cruise_vx_mps)
    assert command.vy == 0.0
    assert command.yaw_rate == 0.0


def test_steers_left_away_from_closer_right_obstacle() -> None:
    policy = ReactiveAvoidancePolicy()
    depth = _depth(2.0)
    depth[:, depth.shape[1] // 2 :] = 1.5  # right half closer

    command = policy.predict(_rgb(), depth)

    assert command.vy < 0.0
    assert command.yaw_rate < 0.0


def test_steers_right_away_from_closer_left_obstacle() -> None:
    policy = ReactiveAvoidancePolicy()
    depth = _depth(2.0)
    depth[:, : depth.shape[1] // 2] = 1.5  # left half closer

    command = policy.predict(_rgb(), depth)

    assert command.vy > 0.0
    assert command.yaw_rate > 0.0


def test_slows_down_as_center_depth_approaches_minimum() -> None:
    policy = ReactiveAvoidancePolicy()

    near_stop = policy.predict(_rgb(), _depth(policy.config.min_forward_depth_m))
    mid_caution = policy.predict(
        _rgb(),
        _depth((policy.config.min_forward_depth_m + policy.config.caution_depth_m) / 2),
    )

    assert near_stop.vx == pytest.approx(0.0, abs=1e-6)
    assert 0.0 < mid_caution.vx < policy.config.cruise_vx_mps


def test_hovers_when_depth_entirely_invalid() -> None:
    policy = ReactiveAvoidancePolicy()
    depth = np.full((40, 60), -1.0, dtype=np.float32)

    command = policy.predict(_rgb(), depth)

    assert command == pytest_approx_hover()


def pytest_approx_hover():
    from drone_autopilot.core_types import VelocityCommand

    return VelocityCommand(0.0, 0.0, 0.0, 0.0)


def test_respects_custom_config() -> None:
    config = ReactiveAvoidanceConfig(cruise_vx_mps=1.2, caution_depth_m=5.0)
    policy = ReactiveAvoidancePolicy(config)

    command = policy.predict(_rgb(), _depth(10.0))

    assert command.vx == pytest.approx(1.2)
