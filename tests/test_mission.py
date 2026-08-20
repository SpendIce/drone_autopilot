from __future__ import annotations

import math

import pytest

from drone_autopilot.core_types import VelocityCommand
from drone_autopilot.mission import MissionConfig, MissionPlanner, Waypoint


def test_mission_planner_biases_reactive_command_toward_waypoint() -> None:
    planner = MissionPlanner(
        [Waypoint(10.0, 0.0)],
        MissionConfig(
            cruise_speed_mps=1.0,
            waypoint_radius_m=0.5,
            slow_radius_m=5.0,
            position_blend=1.0,
            yaw_blend=1.0,
            heading_gain=1.0,
            max_goal_yaw_rate_radps=0.5,
        ),
    )

    result = planner.update(
        VelocityCommand.hover(),
        {"x": 0.0, "y": 0.0, "yaw": 0.0},
    )

    assert not result.mission_complete
    assert result.active_index == 0
    assert result.distance_to_waypoint_m == pytest.approx(10.0)
    assert result.yaw_error_rad == pytest.approx(0.0)
    assert result.command == VelocityCommand(1.0, 0.0, 0.0, 0.0)


def test_mission_planner_converts_world_goal_to_body_frame() -> None:
    planner = MissionPlanner(
        [Waypoint(0.0, 10.0)],
        MissionConfig(
            cruise_speed_mps=1.0,
            waypoint_radius_m=0.5,
            slow_radius_m=5.0,
            position_blend=1.0,
            yaw_blend=1.0,
            heading_gain=1.0,
            max_goal_yaw_rate_radps=0.5,
        ),
    )

    result = planner.update(
        VelocityCommand.hover(),
        {"x": 0.0, "y": 0.0, "yaw": 0.0},
    )

    assert result.command.vx == pytest.approx(0.0, abs=1e-6)
    assert result.command.vy == pytest.approx(1.0)
    assert result.command.yaw_rate == pytest.approx(0.5)


def test_mission_planner_advances_waypoints_and_hovers_at_final() -> None:
    planner = MissionPlanner(
        [Waypoint(1.0, 0.0), Waypoint(2.0, 0.0)],
        MissionConfig(
            cruise_speed_mps=1.0,
            waypoint_radius_m=0.5,
            slow_radius_m=5.0,
            position_blend=1.0,
        ),
    )

    first = planner.update(VelocityCommand(0.2, 0.0, 0.0, 0.0), {"x": 0.8, "y": 0.0, "yaw": 0.0})
    second = planner.update(VelocityCommand(0.2, 0.0, 0.0, 0.0), {"x": 2.0, "y": 0.0, "yaw": 0.0})

    assert first.reached_waypoint
    assert first.active_index == 1
    assert not first.mission_complete
    assert second.reached_waypoint
    assert second.mission_complete
    assert second.command == VelocityCommand.hover()


def test_mission_planner_wraps_yaw_error() -> None:
    planner = MissionPlanner(
        [Waypoint(-1.0, 0.0)],
        MissionConfig(
            cruise_speed_mps=1.0,
            waypoint_radius_m=0.1,
            slow_radius_m=1.0,
            position_blend=1.0,
            yaw_blend=1.0,
            heading_gain=1.0,
            max_goal_yaw_rate_radps=10.0,
        ),
    )

    result = planner.update(
        VelocityCommand.hover(),
        {"x": 0.0, "y": 0.0, "yaw": -math.pi + 0.1},
    )

    assert result.yaw_error_rad == pytest.approx(-0.1)
    assert result.command.yaw_rate == pytest.approx(-0.1)


def test_mission_planner_leaves_command_when_state_is_missing() -> None:
    planner = MissionPlanner([Waypoint(1.0, 0.0)])
    reactive = VelocityCommand(0.2, 0.0, 0.0, 0.1)

    result = planner.update(reactive, {"x": 0.0, "y": 0.0})

    assert result.command == reactive
    assert result.reason == "missing_state"


def test_mission_planner_avoidance_urgency_tapers_goal_contribution() -> None:
    planner = MissionPlanner(
        [Waypoint(10.0, 0.0)],
        MissionConfig(
            cruise_speed_mps=1.0,
            waypoint_radius_m=0.5,
            slow_radius_m=5.0,
            position_blend=1.0,
            yaw_blend=1.0,
            heading_gain=1.0,
            max_goal_yaw_rate_radps=0.5,
        ),
    )
    reactive = VelocityCommand(-0.3, 0.0, 0.0, 0.0)

    full_urgency = planner.update(
        reactive, {"x": 0.0, "y": 0.0, "yaw": 0.0}, avoidance_urgency=1.0
    )

    assert full_urgency.command.vx == pytest.approx(-0.3)


def test_mission_planner_avoidance_urgency_zero_matches_default_blend() -> None:
    config = MissionConfig(
        cruise_speed_mps=1.0,
        waypoint_radius_m=0.5,
        slow_radius_m=5.0,
        position_blend=0.5,
        yaw_blend=0.5,
    )
    reactive = VelocityCommand(0.1, 0.0, 0.0, 0.2)

    default_urgency = MissionPlanner([Waypoint(10.0, 0.0)], config).update(
        reactive, {"x": 0.0, "y": 0.0, "yaw": 0.0}
    )
    explicit_zero = MissionPlanner([Waypoint(10.0, 0.0)], config).update(
        reactive, {"x": 0.0, "y": 0.0, "yaw": 0.0}, avoidance_urgency=0.0
    )

    assert default_urgency.command == explicit_zero.command


def test_mission_planner_avoidance_urgency_clips_out_of_range_values() -> None:
    planner = MissionPlanner(
        [Waypoint(10.0, 0.0)],
        MissionConfig(cruise_speed_mps=1.0, waypoint_radius_m=0.5, slow_radius_m=5.0, position_blend=1.0),
    )
    reactive = VelocityCommand(-0.3, 0.0, 0.0, 0.0)

    over_one = planner.update(reactive, {"x": 0.0, "y": 0.0, "yaw": 0.0}, avoidance_urgency=5.0)

    assert over_one.command.vx == pytest.approx(-0.3)


def test_mission_planner_rejects_negative_heading_gain() -> None:
    with pytest.raises(ValueError, match="heading_gain"):
        MissionPlanner([Waypoint(1.0, 0.0)], MissionConfig(heading_gain=-1.0))
