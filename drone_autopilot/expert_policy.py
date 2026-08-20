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
    caution_depth_m: float = 3.0
    min_forward_depth_m: float = 1.2
    max_lateral_vy_mps: float = 0.6
    max_avoid_yaw_radps: float = 0.5
    depth_roi_top: float = 0.25
    depth_roi_bottom: float = 0.85


class ReactiveAvoidancePolicy:
    """Depth-only lateral avoidance: slows and steers away from whichever side
    (left/right half of the depth ROI) is closer once an obstacle enters the
    caution band; cruises straight forward otherwise.

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
        clearance = max(center_min - cfg.min_forward_depth_m, 0.0)
        vx = cfg.cruise_vx_mps * float(np.clip(clearance / span, 0.0, 1.0))

        urgency = float(np.clip((cfg.caution_depth_m - center_min) / span, 0.0, 1.0))
        steer_right = left_min < right_min
        sign = 1.0 if steer_right else -1.0
        vy = sign * cfg.max_lateral_vy_mps * urgency
        yaw_rate = sign * cfg.max_avoid_yaw_radps * urgency

        return VelocityCommand(vx=vx, vy=vy, vz=0.0, yaw_rate=yaw_rate)

    @staticmethod
    def _min_valid(values: np.ndarray) -> float | None:
        finite_positive = values[np.isfinite(values) & (values > 0.0)]
        if finite_positive.size == 0:
            return None
        return float(finite_positive.min())
