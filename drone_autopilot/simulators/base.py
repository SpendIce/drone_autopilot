"""Simulator-neutral closed-loop control interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
import csv
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Protocol

import numpy as np

from ..mission import MissionOutput, MissionPlanner
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

    def capture_state(self) -> dict[str, Any]:
        return {}

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
    elapsed_s: float = 0.0
    capture_elapsed_s: float = 0.0
    predict_elapsed_s: float = 0.0
    filter_elapsed_s: float = 0.0
    command_elapsed_s: float = 0.0
    state_elapsed_s: float = 0.0
    step_elapsed_s: float = 0.0
    mission_complete: bool = False
    mission_active_index: int | None = None
    mission_last_distance_m: float | None = None
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

    def record_timing(
        self,
        *,
        capture_s: float,
        predict_s: float,
        filter_s: float,
        command_s: float,
        state_s: float,
        step_s: float,
    ) -> None:
        self.capture_elapsed_s += capture_s
        self.predict_elapsed_s += predict_s
        self.filter_elapsed_s += filter_s
        self.command_elapsed_s += command_s
        self.state_elapsed_s += state_s
        self.step_elapsed_s += step_s

    def record_mission(self, mission: MissionOutput | None) -> None:
        if mission is None:
            return
        self.mission_complete = mission.mission_complete
        self.mission_active_index = mission.active_index
        self.mission_last_distance_m = mission.distance_to_waypoint_m

    def to_dict(self) -> dict[str, object]:
        mean_abs_predicted_yaw = self.predicted_yaw_abs_sum / max(self.steps, 1)
        mean_abs_command_yaw = self.command_yaw_abs_sum / max(self.steps, 1)
        mean_step_hz = float(self.steps / self.elapsed_s) if self.elapsed_s > 0.0 else 0.0
        divisor = max(self.steps, 1)
        return {
            "steps": self.steps,
            "elapsed_s": self.elapsed_s,
            "mean_step_hz": mean_step_hz,
            "mean_capture_s": self.capture_elapsed_s / divisor,
            "mean_predict_s": self.predict_elapsed_s / divisor,
            "mean_filter_s": self.filter_elapsed_s / divisor,
            "mean_command_s": self.command_elapsed_s / divisor,
            "mean_state_s": self.state_elapsed_s / divisor,
            "mean_step_s": self.step_elapsed_s / divisor,
            "emergency_stops": self.emergency_stops,
            "min_depth_m": self.min_depth_m,
            "command_smoothness": self.command_smoothness,
            "mean_abs_predicted_yaw_rate": mean_abs_predicted_yaw,
            "mean_abs_command_yaw_rate": mean_abs_command_yaw,
            "command_yaw_sign_changes": self.command_yaw_sign_changes,
            "mission_complete": self.mission_complete,
            "mission_active_index": self.mission_active_index,
            "mission_last_distance_m": self.mission_last_distance_m,
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
    mission_planner: MissionPlanner | None = None,
) -> ClosedLoopMetrics:
    metrics = ClosedLoopMetrics()
    previous = VelocityCommand.hover()
    started_at = time.perf_counter()
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
                "planned_vx",
                "planned_vy",
                "planned_vz",
                "planned_yaw_rate",
                "command_vx",
                "command_vy",
                "command_vz",
                "command_yaw_rate",
                "mission_active_index",
                "mission_target_x",
                "mission_target_y",
                "mission_distance_m",
                "mission_yaw_error_rad",
                "mission_reached_waypoint",
                "mission_complete",
                "mission_reason",
                "mission_state_x",
                "mission_state_y",
                "mission_state_yaw",
                "emergency_stop",
                "reason",
                "min_depth_m",
                "capture_s",
                "predict_s",
                "filter_s",
                "command_s",
                "state_s",
                "step_s",
                "state_x",
                "state_y",
                "state_z",
                "state_vx",
                "state_vy",
                "state_vz",
                "state_collided",
                "state_collision_object",
            ],
        )
        log_writer.writeheader()
    try:
        for _ in range(steps):
            step_started_at = time.perf_counter()
            phase_started_at = step_started_at
            observation = adapter.capture_observation()
            capture_s = time.perf_counter() - phase_started_at
            phase_started_at = time.perf_counter()
            prediction = policy.predict(observation.rgb, observation.depth_m)
            predict_s = time.perf_counter() - phase_started_at
            mission_state: dict[str, Any] = {}
            mission_output: MissionOutput | None = None
            planned = prediction
            if mission_planner is not None:
                phase_started_at = time.perf_counter()
                mission_state = adapter.capture_state()
                mission_output = mission_planner.update(prediction, mission_state)
                planned = mission_output.command
                mission_state_s = time.perf_counter() - phase_started_at
            else:
                mission_state_s = 0.0
            phase_started_at = time.perf_counter()
            result = safety_filter.filter(planned, depth_m=observation.depth_m, reactive=prediction)
            filter_s = time.perf_counter() - phase_started_at
            phase_started_at = time.perf_counter()
            if result.emergency_stop:
                adapter.hover(duration_s=command_duration_s)
            else:
                adapter.send_velocity(result.command, duration_s=command_duration_s)
            command_s = time.perf_counter() - phase_started_at
            state_s = 0.0
            state: dict[str, Any] = {}
            if log_writer is not None:
                phase_started_at = time.perf_counter()
                state = adapter.capture_state()
                state_s = time.perf_counter() - phase_started_at
            state_s += mission_state_s
            step_s = time.perf_counter() - step_started_at
            if log_writer is not None:
                target = mission_output.target if mission_output is not None else None
                log_writer.writerow(
                    {
                        "step": metrics.steps + 1,
                        "predicted_vx": prediction.vx,
                        "predicted_vy": prediction.vy,
                        "predicted_vz": prediction.vz,
                        "predicted_yaw_rate": prediction.yaw_rate,
                        "planned_vx": planned.vx,
                        "planned_vy": planned.vy,
                        "planned_vz": planned.vz,
                        "planned_yaw_rate": planned.yaw_rate,
                        "command_vx": result.command.vx,
                        "command_vy": result.command.vy,
                        "command_vz": result.command.vz,
                        "command_yaw_rate": result.command.yaw_rate,
                        "mission_active_index": mission_output.active_index if mission_output is not None else None,
                        "mission_target_x": target.x if target is not None else None,
                        "mission_target_y": target.y if target is not None else None,
                        "mission_distance_m": mission_output.distance_to_waypoint_m if mission_output is not None else None,
                        "mission_yaw_error_rad": mission_output.yaw_error_rad if mission_output is not None else None,
                        "mission_reached_waypoint": mission_output.reached_waypoint if mission_output is not None else None,
                        "mission_complete": mission_output.mission_complete if mission_output is not None else None,
                        "mission_reason": mission_output.reason if mission_output is not None else None,
                        "mission_state_x": mission_state.get("x"),
                        "mission_state_y": mission_state.get("y"),
                        "mission_state_yaw": mission_state.get("yaw"),
                        "emergency_stop": result.emergency_stop,
                        "reason": result.reason,
                        "min_depth_m": result.min_depth_m,
                        "capture_s": capture_s,
                        "predict_s": predict_s,
                        "filter_s": filter_s,
                        "command_s": command_s,
                        "state_s": state_s,
                        "step_s": step_s,
                        "state_x": state.get("x"),
                        "state_y": state.get("y"),
                        "state_z": state.get("z"),
                        "state_vx": state.get("vx"),
                        "state_vy": state.get("vy"),
                        "state_vz": state.get("vz"),
                        "state_collided": state.get("collided"),
                        "state_collision_object": state.get("collision_object"),
                    }
                )
            metrics.update(prediction=prediction, result=result, previous_command=previous)
            metrics.record_mission(mission_output)
            metrics.record_timing(
                capture_s=capture_s,
                predict_s=predict_s,
                filter_s=filter_s,
                command_s=command_s,
                state_s=state_s,
                step_s=step_s,
            )
            previous = result.command
    finally:
        metrics.elapsed_s = time.perf_counter() - started_at
        adapter.hover(duration_s=command_duration_s)
        if log_file is not None:
            log_file.close()
    return metrics
