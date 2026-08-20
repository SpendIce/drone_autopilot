"""Deterministic depth-based reactive avoidance policy.

Not used at inference time by the trained pilot. It exists to generate imitation
demonstrations that include real obstacle-avoidance behavior, which the offline
seed dataset does not reliably cover in every environment (see the closed-loop
ablation in the informe: the learned network alone drove into a wall and the
runtime SafetyFilter only hovers on close depth, it never steers away).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core_types import VelocityCommand


@dataclass(frozen=True)
class ReactiveAvoidanceConfig:
    cruise_vx_mps: float = 0.6
    caution_depth_m: float = 4.5
    min_forward_depth_m: float = 1.2
    retreat_depth_m: float = 0.6
    retreat_vx_mps: float = -0.35
    max_lateral_vy_mps: float = 0.6
    max_avoid_yaw_radps: float = 0.5
    urgency_exponent: float = 0.5
    depth_roi_top: float = 0.25
    depth_roi_bottom: float = 0.85

    def __post_init__(self) -> None:
        if self.retreat_depth_m >= self.min_forward_depth_m:
            raise ValueError("retreat_depth_m must be below min_forward_depth_m")
        if self.retreat_vx_mps > 0.0:
            raise ValueError("retreat_vx_mps must be <= 0 (it is a reverse speed)")
        if not 0.0 < self.urgency_exponent <= 1.0:
            raise ValueError("urgency_exponent must be in (0, 1]; <1 front-loads the turn")


class ReactiveAvoidancePolicy:
    """Depth-only lateral avoidance: steers away from whichever side (left/right
    half of the depth ROI) is closer as soon as an obstacle enters the caution
    band, and slows only as a secondary effect of that same urgency signal —
    the priority is turning early and hard enough to never need to stop.
    `urgency_exponent` < 1 front-loads the turn/lateral response (steep at
    moderate range, not just once almost touching), and `caution_depth_m`
    defaults wide (4.5 m) so there is real distance to redirect the
    trajectory before the emergency band.

    Reversing (below `retreat_depth_m`, once forward speed already hit zero)
    is a last-resort fallback, not the primary strategy: yaw changes heading
    but not position, so a turn alone doesn't increase separation from
    whatever is directly ahead — only vx/vy translation does (see the
    closed-loop run where the drone froze at identical (x, y) for hundreds of
    steps while still commanding yaw_rate). With early, hard turning this
    band should rarely trigger; it exists so the controller isn't stuck
    approach-braking with nothing left to give if it does.

    Body frame is forward-right-down (AirSim convention): positive vy/yaw_rate
    steer right, negative steer left.
    """

    def __init__(self, config: ReactiveAvoidanceConfig | None = None) -> None:
        self.config = config or ReactiveAvoidanceConfig()

    def predict(self, rgb: np.ndarray, depth_m: np.ndarray) -> VelocityCommand:
        cfg = self.config
        height, width = depth_m.shape[:2]
        top = int(height * cfg.depth_roi_top)
        bottom = int(height * cfg.depth_roi_bottom)
        band = depth_m[top:bottom, :]

        center_min = self._min_valid(band)
        if center_min is None:
            return VelocityCommand(0.0, 0.0, 0.0, 0.0)
        if center_min >= cfg.caution_depth_m:
            return VelocityCommand(cfg.cruise_vx_mps, 0.0, 0.0, 0.0)

        left_min = self._min_valid(band[:, : width // 2])
        right_min = self._min_valid(band[:, width // 2 :])
        left_min = left_min if left_min is not None else float("inf")
        right_min = right_min if right_min is not None else float("inf")

        span = max(cfg.caution_depth_m - cfg.min_forward_depth_m, 1e-6)
        vx = self._vx_for_depth(center_min)

        linear_urgency = float(np.clip((cfg.caution_depth_m - center_min) / span, 0.0, 1.0))
        urgency = linear_urgency**cfg.urgency_exponent
        steer_right = left_min < right_min
        sign = 1.0 if steer_right else -1.0
        vy = sign * cfg.max_lateral_vy_mps * urgency
        yaw_rate = sign * cfg.max_avoid_yaw_radps * urgency

        return VelocityCommand(vx=vx, vy=vy, vz=0.0, yaw_rate=yaw_rate)

    def _vx_for_depth(self, center_min: float) -> float:
        """Piecewise-linear forward speed: cruise above min_forward_depth_m,
        braking to zero down to retreat_depth_m, reversing below that."""
        cfg = self.config
        if center_min >= cfg.min_forward_depth_m:
            brake_span = max(cfg.caution_depth_m - cfg.min_forward_depth_m, 1e-6)
            clearance = center_min - cfg.min_forward_depth_m
            return cfg.cruise_vx_mps * float(np.clip(clearance / brake_span, 0.0, 1.0))

        reverse_span = max(cfg.min_forward_depth_m - cfg.retreat_depth_m, 1e-6)
        overrun = cfg.min_forward_depth_m - center_min
        return cfg.retreat_vx_mps * float(np.clip(overrun / reverse_span, 0.0, 1.0))

    @staticmethod
    def _min_valid(values: np.ndarray) -> float | None:
        finite_positive = values[np.isfinite(values) & (values > 0.0)]
        if finite_positive.size == 0:
            return None
        return float(finite_positive.min())
