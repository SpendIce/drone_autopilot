"""Simulator-neutral closed-loop control interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
    reasons: dict[str, int] = field(default_factory=dict)

    def update(self, result: SafetyFilterResult, previous_command: VelocityCommand) -> None:
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


def run_closed_loop(
    adapter: SimulatorAdapter,
    policy: PilotPolicy,
    safety_filter: SafetyFilter,
    *,
    steps: int,
    command_duration_s: float = 0.1,
) -> ClosedLoopMetrics:
    metrics = ClosedLoopMetrics()
    previous = VelocityCommand.hover()
    try:
        for _ in range(steps):
            observation = adapter.capture_observation()
            prediction = policy.predict(observation.rgb, observation.depth_m)
            result = safety_filter.filter(prediction, depth_m=observation.depth_m)
            if result.emergency_stop:
                adapter.hover(duration_s=command_duration_s)
            else:
                adapter.send_velocity(result.command, duration_s=command_duration_s)
            metrics.update(result, previous)
            previous = result.command
    finally:
        adapter.hover(duration_s=command_duration_s)
    return metrics
