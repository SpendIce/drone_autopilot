from __future__ import annotations

import numpy as np
import pytest

from drone_autopilot.core_types import VelocityCommand
from drone_autopilot.safety import SafetyFilter
from drone_autopilot.simulators.base import Observation, SimulatorAdapter, run_closed_loop


class _Adapter(SimulatorAdapter):
    def __init__(self) -> None:
        self.sent: list[VelocityCommand] = []
        self.hover_count = 0

    def capture_observation(self) -> Observation:
        return Observation(
            rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            depth_m=np.ones((8, 8), dtype=np.float32) * 5.0,
        )

    def send_velocity(self, command: VelocityCommand, *, duration_s: float) -> None:
        self.sent.append(command)

    def hover(self, *, duration_s: float) -> None:
        self.hover_count += 1


class _Policy:
    def predict(self, rgb: np.ndarray, depth_m: np.ndarray) -> VelocityCommand:
        return VelocityCommand(0.2, 0.0, 0.0, 0.1)


def test_closed_loop_writes_command_log_and_yaw_metrics(tmp_path) -> None:
    log_path = tmp_path / "commands.csv"

    metrics = run_closed_loop(
        _Adapter(),
        _Policy(),
        SafetyFilter(),
        steps=2,
        command_duration_s=0.1,
        command_log_path=log_path,
    )

    assert metrics.steps == 2
    assert metrics.to_dict()["mean_abs_predicted_yaw_rate"] == pytest.approx(0.1)
    header = log_path.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("step,")
    assert "state_z" in header
