"""Runtime safety filtering for model-produced velocity commands."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core_types import VelocityCommand


@dataclass(frozen=True)
class SafetyConfig:
    max_vx_mps: float = 2.0
    max_vy_mps: float = 1.0
    max_vz_mps: float = 1.0
    max_yaw_rate_radps: float = 0.8
    smoothing_alpha: float = 0.35
    vx_deadband_mps: float = 0.0
    vy_deadband_mps: float = 0.0
    vz_deadband_mps: float = 0.0
    yaw_rate_deadband_radps: float = 0.0
    emergency_depth_m: float = 0.8
    invalid_depth_fraction_limit: float = 0.8
    depth_roi_top: float = 0.0
    depth_roi_bottom: float = 1.0
    depth_roi_left: float = 0.0
    depth_roi_right: float = 1.0


@dataclass(frozen=True)
class SafetyFilterResult:
    command: VelocityCommand
    emergency_stop: bool
    reason: str
    min_depth_m: float | None = None


class SafetyFilter:
    """Clamp, smooth, and fail closed before commands reach a simulator."""

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()
        self._validate_depth_roi()
        self._previous = VelocityCommand.hover()

    def reset(self) -> None:
        self._previous = VelocityCommand.hover()

    def filter(
        self,
        predicted: VelocityCommand | np.ndarray | list[float] | tuple[float, ...],
        *,
        depth_m: np.ndarray | None = None,
        reactive: VelocityCommand | np.ndarray | list[float] | tuple[float, ...] | None = None,
    ) -> SafetyFilterResult:
        command = self._coerce_command(predicted)
        if command is None or not command.is_finite():
            self._previous = VelocityCommand.hover()
            return SafetyFilterResult(
                command=VelocityCommand.hover(),
                emergency_stop=True,
                reason="invalid_prediction",
            )

        min_depth = self._min_valid_depth(depth_m)
        if min_depth is not None and min_depth < self.config.emergency_depth_m:
            reactive_command = self._coerce_command(reactive) if reactive is not None else None
            steering_source = reactive_command if reactive_command is not None and reactive_command.is_finite() else command
            escape = self._escape_command(command, steering_source)
            self._previous = escape
            return SafetyFilterResult(
                command=escape,
                emergency_stop=True,
                reason="close_obstacle",
                min_depth_m=min_depth,
            )

        clamped = command.clamp(
            max_vx=self.config.max_vx_mps,
            max_vy=self.config.max_vy_mps,
            max_vz=self.config.max_vz_mps,
            max_yaw_rate=self.config.max_yaw_rate_radps,
        )
        target = self._apply_deadbands(clamped)
        filtered = target.smooth_toward(self._previous, self.config.smoothing_alpha)
        self._previous = filtered
        return SafetyFilterResult(
            command=filtered,
            emergency_stop=False,
            reason="ok",
            min_depth_m=min_depth,
        )

    def _escape_command(
        self, command: VelocityCommand, steering_source: VelocityCommand
    ) -> VelocityCommand:
        """Command used while inside the emergency depth radius.

        A blind hover() here freezes the drone permanently: depth stays under
        the threshold forever with zero velocity, so no avoidance (learned or
        planned) can ever execute again once triggered (see closed-loop runs
        where the drone locks at the same position for the rest of the
        episode). Instead, forbid further approach (clamp vx to <= 0, i.e.
        brake or back away, never accelerate toward the obstacle) while still
        clamping and passing through lateral/yaw motion so a turn-and-strafe
        escape started before the emergency threshold isn't cut off.

        `steering_source` supplies vy/yaw_rate, and also participates in the vx
        veto — pass the policy's raw, pre-mission-blend prediction here when
        available. Measured runs show the mission planner's goal-seeking blend
        roughly halves the network's turn-rate intent (it keeps pulling toward
        a waypoint whose bearing is exactly what's blocked); the same dilution
        hits vx just as hard — a policy actively wanting to reverse can have
        that cancelled by the blended-in positive goal.vx, leaving the escape
        with nothing but "brake to zero" and no way to actually gain distance.
        Taking min(blended, raw, 0.0) means either source retreating is
        enough; the goal's pull can only ever hold vx at 0, never override an
        active retreat with forward motion.
        """
        clamped_forward = command.clamp(
            max_vx=self.config.max_vx_mps,
            max_vy=self.config.max_vy_mps,
            max_vz=self.config.max_vz_mps,
            max_yaw_rate=self.config.max_yaw_rate_radps,
        )
        clamped_steering = steering_source.clamp(
            max_vx=self.config.max_vx_mps,
            max_vy=self.config.max_vy_mps,
            max_vz=self.config.max_vz_mps,
            max_yaw_rate=self.config.max_yaw_rate_radps,
        )
        escape = VelocityCommand(
            vx=min(clamped_forward.vx, clamped_steering.vx, 0.0),
            vy=clamped_steering.vy,
            vz=0.0,
            yaw_rate=clamped_steering.yaw_rate,
        )
        target = self._apply_deadbands(escape)
        return target.smooth_toward(self._previous, self.config.smoothing_alpha)

    def _coerce_command(
        self,
        predicted: VelocityCommand | np.ndarray | list[float] | tuple[float, ...],
    ) -> VelocityCommand | None:
        if isinstance(predicted, VelocityCommand):
            return predicted
        try:
            return VelocityCommand.from_iterable(predicted)
        except (TypeError, ValueError):
            return None

    def _min_valid_depth(self, depth_m: np.ndarray | None) -> float | None:
        if depth_m is None:
            return None
        values = self._depth_roi(np.asarray(depth_m, dtype=np.float32))
        finite_positive = values[np.isfinite(values) & (values > 0.0)]
        total = max(int(values.size), 1)
        invalid_fraction = 1.0 - (float(finite_positive.size) / float(total))
        if invalid_fraction > self.config.invalid_depth_fraction_limit:
            return 0.0
        if finite_positive.size == 0:
            return 0.0
        return float(finite_positive.min())

    def _depth_roi(self, values: np.ndarray) -> np.ndarray:
        height, width = values.shape[:2]
        top = int(round(height * self.config.depth_roi_top))
        bottom = int(round(height * self.config.depth_roi_bottom))
        left = int(round(width * self.config.depth_roi_left))
        right = int(round(width * self.config.depth_roi_right))
        return values[top:bottom, left:right]

    def _validate_depth_roi(self) -> None:
        roi = (
            self.config.depth_roi_top,
            self.config.depth_roi_bottom,
            self.config.depth_roi_left,
            self.config.depth_roi_right,
        )
        if any(value < 0.0 or value > 1.0 for value in roi):
            raise ValueError("depth ROI bounds must be within [0.0, 1.0]")
        if self.config.depth_roi_top >= self.config.depth_roi_bottom:
            raise ValueError("depth ROI top must be less than bottom")
        if self.config.depth_roi_left >= self.config.depth_roi_right:
            raise ValueError("depth ROI left must be less than right")

    def _apply_deadbands(self, command: VelocityCommand) -> VelocityCommand:
        return VelocityCommand(
            vx=self._zero_below(command.vx, self.config.vx_deadband_mps),
            vy=self._zero_below(command.vy, self.config.vy_deadband_mps),
            vz=self._zero_below(command.vz, self.config.vz_deadband_mps),
            yaw_rate=self._zero_below(command.yaw_rate, self.config.yaw_rate_deadband_radps),
        )

    def _zero_below(self, value: float, threshold: float) -> float:
        if threshold < 0.0:
            raise ValueError("deadband thresholds must be non-negative")
        return 0.0 if abs(value) < threshold else value
