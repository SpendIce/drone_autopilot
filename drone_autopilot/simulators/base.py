"""Simulator-neutral closed-loop control interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from ..safety import SafetyFilter, SafetyFilterResult
from ..core_types import VelocityCommand


@dataclass(frozen=True)
class Observation:
    rgb: np.ndarray
    depth_m: np.ndarray
    timestamp: float | None = None


class PilotPolicy(Protocol):
    def predict(self, rgb: np.ndarray, depth_m: np.ndarray) -> VelocityCommand:
        ...


class SimulatorAdapter(ABC):
    @abstractmethod
    def capture_observation(self) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def send_velocity(self, command: VelocityCommand, *, duration_s: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def hover(self, *, duration_s: float) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None


@dataclass
class ClosedLoopMetrics:
    steps: int = 0
    emergency_stops: int = 0
    min_depth_m: float | None = None
    command_smoothness: float = 0.0
    predicted_yaw_abs_sum: float = 0.0
    command_yaw_abs_sum: float = 0.0
    command_yaw_sign_changes: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def update(
        self,
        *,
        prediction: VelocityCommand,
        result: SafetyFilterResult,
        previous_command: VelocityCommand,
    ) -> None:
        self.steps += 1
        self.reasons[result.reason] = self.reasons.get(result.reason, 0) + 1
        if result.emergency_stop:
            self.emergency_stops += 1
        if result.min_depth_m is not None:
            if self.min_depth_m is None:
                self.min_depth_m = result.min_depth_m
            else:
                self.min_depth_m = min(self.min_depth_m, result.min_depth_m)
        delta = result.command.to_numpy(np.float64) - previous_command.to_numpy(np.float64)
        self.command_smoothness += float(np.linalg.norm(delta))
        self.predicted_yaw_abs_sum += abs(prediction.yaw_rate)
        self.command_yaw_abs_sum += abs(result.command.yaw_rate)
        previous_sign = np.sign(previous_command.yaw_rate)
        current_sign = np.sign(result.command.yaw_rate)
        if previous_sign != 0.0 and current_sign != 0.0 and previous_sign != current_sign:
            self.command_yaw_sign_changes += 1

    def to_dict(self) -> dict[str, object]:
        mean_abs_predicted_yaw = self.predicted_yaw_abs_sum / max(self.steps, 1)
        mean_abs_command_yaw = self.command_yaw_abs_sum / max(self.steps, 1)
        return {
            "steps": self.steps,
            "emergency_stops": self.emergency_stops,
            "min_depth_m": self.min_depth_m,
            "command_smoothness": self.command_smoothness,
            "mean_abs_predicted_yaw_rate": mean_abs_predicted_yaw,
            "mean_abs_command_yaw_rate": mean_abs_command_yaw,
            "command_yaw_sign_changes": self.command_yaw_sign_changes,
            "reasons": self.reasons,
        }


def run_closed_loop(
    adapter: SimulatorAdapter,
    policy: PilotPolicy,
    safety_filter: SafetyFilter,
    *,
    steps: int,
    command_duration_s: float = 0.1,
    command_log_path: Path | str | None = None,
) -> ClosedLoopMetrics:
    metrics = ClosedLoopMetrics()
    previous = VelocityCommand.hover()
    log_file = None
    log_writer = None
    if command_log_path is not None:
        path = Path(command_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        log_file = path.open("w", encoding="utf-8", newline="")
        log_writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "step",
                "predicted_vx",
                "predicted_vy",
                "predicted_vz",
                "predicted_yaw_rate",
                "command_vx",
                "command_vy",
                "command_vz",
                "command_yaw_rate",
                "emergency_stop",
                "reason",
                "min_depth_m",
            ],
        )
        log_writer.writeheader()
    try:
        for _ in range(steps):
            observation = adapter.capture_observation()
            prediction = policy.predict(observation.rgb, observation.depth_m)
            result = safety_filter.filter(prediction, depth_m=observation.depth_m)
            if log_writer is not None:
                log_writer.writerow(
                    {
                        "step": metrics.steps + 1,
                        "predicted_vx": prediction.vx,
                        "predicted_vy": prediction.vy,
                        "predicted_vz": prediction.vz,
                        "predicted_yaw_rate": prediction.yaw_rate,
                        "command_vx": result.command.vx,
                        "command_vy": result.command.vy,
                        "command_vz": result.command.vz,
                        "command_yaw_rate": result.command.yaw_rate,
                        "emergency_stop": result.emergency_stop,
                        "reason": result.reason,
                        "min_depth_m": result.min_depth_m,
                    }
                )
            if result.emergency_stop:
                adapter.hover(duration_s=command_duration_s)
            else:
                adapter.send_velocity(result.command, duration_s=command_duration_s)
            metrics.update(prediction=prediction, result=result, previous_command=previous)
            previous = result.command
    finally:
        adapter.hover(duration_s=command_duration_s)
        if log_file is not None:
            log_file.close()
    return metrics
