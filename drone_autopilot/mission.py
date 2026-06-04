"""Waypoint and heading guidance for mission-level navigation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .core_types import VelocityCommand


@dataclass(frozen=True)
class Waypoint:
    """A 2D world-frame waypoint in simulator coordinates."""

    x: float
    y: float


@dataclass(frozen=True)
class MissionConfig:
    """Tuning for the waypoint tracker that biases the reactive pilot."""

    cruise_speed_mps: float = 0.45
    waypoint_radius_m: float = 1.5
    slow_radius_m: float = 4.0
    position_blend: float = 0.45
    yaw_blend: float = 0.5
    heading_gain: float = 0.8
    max_goal_yaw_rate_radps: float = 0.25


@dataclass(frozen=True)
class MissionOutput:
    command: VelocityCommand
    active_index: int | None
    target: Waypoint | None
    distance_to_waypoint_m: float | None
    yaw_error_rad: float | None
    reached_waypoint: bool
    mission_complete: bool
    reason: str = "ok"


class MissionPlanner:
    """Track waypoints by blending goal heading with a local reactive command."""

    def __init__(
        self,
        waypoints: Sequence[Waypoint],
        config: MissionConfig | None = None,
    ) -> None:
        if not waypoints:
            raise ValueError("mission requires at least one waypoint")
        self.waypoints = tuple(waypoints)
        self.config = config or MissionConfig()
        self._validate_config()
        self._active_index = 0
        self._mission_complete = False

    @property
    def active_index(self) -> int | None:
        if self._mission_complete:
            return None
        return self._active_index

    def update(
        self,
        reactive_command: VelocityCommand,
        state: Mapping[str, object],
    ) -> MissionOutput:
        if self._mission_complete:
            return self._complete_output()

        current = _state_pose(state)
        if current is None:
            return MissionOutput(
                command=reactive_command,
                active_index=self.active_index,
                target=self._active_waypoint(),
                distance_to_waypoint_m=None,
                yaw_error_rad=None,
                reached_waypoint=False,
                mission_complete=False,
                reason="missing_state",
            )

        x, y, yaw = current
        reached = self._advance_reached_waypoints(x, y)
        if self._mission_complete:
            return self._complete_output(reached_waypoint=reached)

        target = self._active_waypoint()
        assert target is not None
        distance = math.hypot(target.x - x, target.y - y)
        bearing = math.atan2(target.y - y, target.x - x)
        yaw_error = _wrap_angle(bearing - yaw)
        desired_speed = self._desired_speed(distance)
        goal = VelocityCommand(
            vx=desired_speed * math.cos(yaw_error),
            vy=desired_speed * math.sin(yaw_error),
            vz=reactive_command.vz,
            yaw_rate=float(
                np.clip(
                    self.config.heading_gain * yaw_error,
                    -self.config.max_goal_yaw_rate_radps,
                    self.config.max_goal_yaw_rate_radps,
                )
            ),
        )
        command = VelocityCommand(
            vx=_blend(reactive_command.vx, goal.vx, self.config.position_blend),
            vy=_blend(reactive_command.vy, goal.vy, self.config.position_blend),
            vz=reactive_command.vz,
            yaw_rate=_blend(reactive_command.yaw_rate, goal.yaw_rate, self.config.yaw_blend),
        )
        return MissionOutput(
            command=command,
            active_index=self._active_index,
            target=target,
            distance_to_waypoint_m=distance,
            yaw_error_rad=yaw_error,
            reached_waypoint=reached,
            mission_complete=False,
        )

    def _advance_reached_waypoints(self, x: float, y: float) -> bool:
        reached = False
        while self._active_index < len(self.waypoints):
            waypoint = self.waypoints[self._active_index]
            if math.hypot(waypoint.x - x, waypoint.y - y) > self.config.waypoint_radius_m:
                break
            reached = True
            self._active_index += 1
        if self._active_index >= len(self.waypoints):
            self._mission_complete = True
        return reached

    def _active_waypoint(self) -> Waypoint | None:
        if self._mission_complete:
            return None
        return self.waypoints[self._active_index]

    def _desired_speed(self, distance_m: float) -> float:
        if distance_m <= self.config.waypoint_radius_m:
            return 0.0
        if self.config.slow_radius_m <= self.config.waypoint_radius_m:
            return self.config.cruise_speed_mps
        scale = (distance_m - self.config.waypoint_radius_m) / (
            self.config.slow_radius_m - self.config.waypoint_radius_m
        )
        return self.config.cruise_speed_mps * float(np.clip(scale, 0.0, 1.0))

    def _complete_output(self, *, reached_waypoint: bool = False) -> MissionOutput:
        return MissionOutput(
            command=VelocityCommand.hover(),
            active_index=None,
            target=None,
            distance_to_waypoint_m=0.0,
            yaw_error_rad=0.0,
            reached_waypoint=reached_waypoint,
            mission_complete=True,
            reason="mission_complete",
        )

    def _validate_config(self) -> None:
        if self.config.cruise_speed_mps < 0.0:
            raise ValueError("cruise_speed_mps must be non-negative")
        if self.config.waypoint_radius_m <= 0.0:
            raise ValueError("waypoint_radius_m must be positive")
        if self.config.slow_radius_m <= 0.0:
            raise ValueError("slow_radius_m must be positive")
        if not 0.0 <= self.config.position_blend <= 1.0:
            raise ValueError("position_blend must be in [0, 1]")
        if not 0.0 <= self.config.yaw_blend <= 1.0:
            raise ValueError("yaw_blend must be in [0, 1]")
        if self.config.heading_gain < 0.0:
            raise ValueError("heading_gain must be non-negative")
        if self.config.max_goal_yaw_rate_radps < 0.0:
            raise ValueError("max_goal_yaw_rate_radps must be non-negative")


def parse_waypoint(value: str) -> Waypoint:
    """Parse ``x,y`` CLI waypoint text."""

    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected waypoint as x,y, got {value!r}")
    return Waypoint(float(parts[0]), float(parts[1]))


def _blend(reactive: float, goal: float, weight: float) -> float:
    return ((1.0 - weight) * reactive) + (weight * goal)


def _state_pose(state: Mapping[str, object]) -> tuple[float, float, float] | None:
    try:
        x = float(state["x"])
        y = float(state["y"])
        yaw = float(state["yaw"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x, y, yaw)):
        return None
    return x, y, yaw


def _wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))
