from __future__ import annotations

import numpy as np
import pytest

from drone_autopilot.core_types import VelocityCommand
from drone_autopilot.mission import MissionConfig, MissionPlanner, Waypoint
from drone_autopilot.safety import SafetyConfig, SafetyFilter
from drone_autopilot.simulators.base import Observation, SimulatorAdapter, run_closed_loop


class _Adapter(SimulatorAdapter):
    def __init__(self) -> None:
        self.sent: list[VelocityCommand] = []
        self.hover_count = 0
        self.state_calls = 0

    def capture_observation(self) -> Observation:
        return Observation(
            rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            depth_m=np.ones((8, 8), dtype=np.float32) * 5.0,
        )

    def send_velocity(self, command: VelocityCommand, *, duration_s: float) -> None:
        self.sent.append(command)

    def hover(self, *, duration_s: float) -> None:
        self.hover_count += 1

    def capture_state(self) -> dict[str, object]:
        self.state_calls += 1
        return {
            "x": 0.0,
            "y": 0.0,
            "z": -3.0,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "yaw": 0.0,
            "collided": False,
            "collision_object": "",
        }


class _Policy:
    def predict(self, rgb: np.ndarray, depth_m: np.ndarray) -> VelocityCommand:
        return VelocityCommand(0.2, 0.0, 0.0, 0.1)


def test_closed_loop_writes_command_log_and_yaw_metrics(tmp_path) -> None:
    log_path = tmp_path / "commands.csv"

    metrics = run_closed_loop(
        _Adapter(),
        _Policy(),
        SafetyFilter(SafetyConfig(smoothing_alpha=1.0)),
        steps=2,
        command_duration_s=0.1,
        command_log_path=log_path,
    )

    assert metrics.steps == 2
    summary = metrics.to_dict()
    assert summary["mean_abs_predicted_yaw_rate"] == pytest.approx(0.1)
    assert summary["mean_step_s"] > 0.0
    header = log_path.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("step,")
    assert "state_z" in header
    assert "capture_s" in header


def test_closed_loop_applies_mission_planner_and_logs_mission_fields(tmp_path) -> None:
    log_path = tmp_path / "mission.csv"
    adapter = _Adapter()
    planner = MissionPlanner(
        [Waypoint(10.0, 0.0)],
        MissionConfig(
            cruise_speed_mps=1.0,
            waypoint_radius_m=0.5,
            slow_radius_m=5.0,
            position_blend=1.0,
            yaw_blend=1.0,
        ),
    )

    metrics = run_closed_loop(
        adapter,
        _Policy(),
        SafetyFilter(SafetyConfig(smoothing_alpha=1.0)),
        steps=1,
        command_duration_s=0.1,
        command_log_path=log_path,
        mission_planner=planner,
    )

    assert metrics.steps == 1
    summary = metrics.to_dict()
    assert summary["mission_complete"] is False
    assert summary["mean_abs_predicted_yaw_rate"] == pytest.approx(0.1)
    assert adapter.sent == [VelocityCommand(1.0, 0.0, 0.0, 0.0)]
    assert adapter.state_calls >= 1
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert "planned_vx" in lines[0]
    assert "mission_target_x" in lines[0]
    assert "mission_distance_m" in lines[0]
    assert "mission_state_yaw" in lines[0]
