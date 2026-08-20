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
    stuck_streak_threshold: int = 15
    stuck_depth_improvement_m: float = 0.05
    stuck_escape_steps: int = 20
    stuck_escape_vx_mps: float = -0.6
    caution_depth_m: float = 3.0

    def __post_init__(self) -> None:
        if self.stuck_streak_threshold <= 0:
            raise ValueError("stuck_streak_threshold must be positive")
        if self.stuck_escape_steps <= 0:
            raise ValueError("stuck_escape_steps must be positive")
        if self.stuck_escape_vx_mps > 0.0:
            raise ValueError("stuck_escape_vx_mps must be <= 0 (it is a reverse speed)")
        if self.caution_depth_m <= self.emergency_depth_m:
            raise ValueError("caution_depth_m must be greater than emergency_depth_m")


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
        self._reset_stuck_tracking()

    def reset(self) -> None:
        self._previous = VelocityCommand.hover()
        self._reset_stuck_tracking()

    def _reset_stuck_tracking(self) -> None:
        self._stuck_streak = 0
        self._stuck_best_depth: float | None = None
        self._stuck_escape_remaining = 0
        self._stuck_direction_sign = 1.0

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

            if self._stuck_escape_remaining <= 0:
                self._update_stuck_tracking(min_depth, steering_source)

            if self._stuck_escape_remaining > 0:
                escape = self._stuck_escape_command()
                self._stuck_escape_remaining -= 1
                reason = "stuck_escape"
            else:
                escape = self._escape_command(command, steering_source)
                reason = "close_obstacle"

            self._previous = escape
            return SafetyFilterResult(
                command=escape,
                emergency_stop=True,
                reason=reason,
                min_depth_m=min_depth,
            )

        self._reset_stuck_tracking()
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

    def urgency_for_depth(self, depth_m: np.ndarray | None) -> float:
        """0..1 proximity signal for MissionPlanner.update(avoidance_urgency=...).

        0 at/beyond caution_depth_m, 1 at/below emergency_depth_m, linear in
        between. Stateless (depends only on config and the given frame) so
        it's safe to call before mission_planner.update()/filter() in the
        same step, ahead of computing the mission-blended command.
        """
        min_depth = self._min_valid_depth(depth_m)
        if min_depth is None:
            return 0.0
        span = max(self.config.caution_depth_m - self.config.emergency_depth_m, 1e-6)
        return float(np.clip((self.config.caution_depth_m - min_depth) / span, 0.0, 1.0))

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

    def _update_stuck_tracking(self, min_depth: float, steering_source: VelocityCommand) -> None:
        """Escalate to a committed escape maneuver if the per-step reactive
        response isn't actually gaining clearance.

        `_escape_command` reacts proportionally every step — enough for most
        obstacles, but a closed-loop run against a concave corner showed depth
        can sit exactly frozen for hundreds of steps while that per-step
        signal stays at full authority: something (contact resolution against
        the collision mesh) is absorbing it, and reacting the same way every
        step never escalates. This is the classic bug-algorithm fix: once
        stuck long enough, stop re-deciding each step and commit to a fixed,
        sustained maneuver in one direction for a while instead.
        """
        cfg = self.config
        if self._stuck_best_depth is None or min_depth > self._stuck_best_depth + cfg.stuck_depth_improvement_m:
            self._stuck_best_depth = min_depth
            self._stuck_streak = 0
            return

        self._stuck_streak += 1
        if self._stuck_streak >= cfg.stuck_streak_threshold:
            self._stuck_direction_sign = 1.0 if steering_source.yaw_rate >= 0.0 else -1.0
            self._stuck_escape_remaining = cfg.stuck_escape_steps
            self._stuck_streak = 0
            self._stuck_best_depth = None

    def _stuck_escape_command(self) -> VelocityCommand:
        """Fixed, sustained maneuver for the committed-escape window: full
        reverse plus full lateral/yaw authority in whichever direction was
        indicated when the commit triggered. Deliberately not smoothed
        toward `_previous` — the point is to break out of a state the
        smoothed, proportional response got stuck in, not ease into it."""
        cfg = self.config
        sign = self._stuck_direction_sign
        return VelocityCommand(
            vx=cfg.stuck_escape_vx_mps,
            vy=sign * cfg.max_vy_mps,
            vz=0.0,
            yaw_rate=sign * cfg.max_yaw_rate_radps,
        )

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
