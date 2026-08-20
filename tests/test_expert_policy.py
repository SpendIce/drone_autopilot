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


def test_retreats_below_retreat_depth() -> None:
    policy = ReactiveAvoidancePolicy()

    command = policy.predict(_rgb(), _depth(policy.config.retreat_depth_m / 2))

    assert command.vx == pytest.approx(policy.config.retreat_vx_mps)


def test_retreat_ramps_between_min_forward_and_retreat_depth() -> None:
    policy = ReactiveAvoidancePolicy()
    midpoint = (policy.config.min_forward_depth_m + policy.config.retreat_depth_m) / 2

    command = policy.predict(_rgb(), _depth(midpoint))

    assert policy.config.retreat_vx_mps < command.vx < 0.0


def test_urgency_exponent_below_one_front_loads_the_turn() -> None:
    front_loaded = ReactiveAvoidancePolicy(ReactiveAvoidanceConfig(urgency_exponent=0.5))
    linear = ReactiveAvoidancePolicy(ReactiveAvoidanceConfig(urgency_exponent=1.0))
    depth = _depth(2.0)
    depth[:, depth.shape[1] // 2 :] = 1.5  # right half closer, still mid-caution range

    front_loaded_command = front_loaded.predict(_rgb(), depth)
    linear_command = linear.predict(_rgb(), depth)

    assert abs(front_loaded_command.yaw_rate) > abs(linear_command.yaw_rate)


def test_rejects_retreat_depth_at_or_above_min_forward_depth() -> None:
    with pytest.raises(ValueError, match="retreat_depth_m"):
        ReactiveAvoidanceConfig(min_forward_depth_m=1.0, retreat_depth_m=1.0)


def test_rejects_positive_retreat_speed() -> None:
    with pytest.raises(ValueError, match="retreat_vx_mps"):
        ReactiveAvoidanceConfig(retreat_vx_mps=0.1)


def test_rejects_urgency_exponent_out_of_range() -> None:
    with pytest.raises(ValueError, match="urgency_exponent"):
        ReactiveAvoidanceConfig(urgency_exponent=0.0)
    with pytest.raises(ValueError, match="urgency_exponent"):
        ReactiveAvoidanceConfig(urgency_exponent=1.5)


def test_respects_custom_config() -> None:
    config = ReactiveAvoidanceConfig(cruise_vx_mps=1.2, caution_depth_m=5.0)
    policy = ReactiveAvoidancePolicy(config)

    command = policy.predict(_rgb(), _depth(10.0))

    assert command.vx == pytest.approx(1.2)
