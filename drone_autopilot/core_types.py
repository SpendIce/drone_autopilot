"""Core data types for commands and manifest rows."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, ClassVar, Iterable

import numpy as np


ACTION_NAMES = ("vx", "vy", "vz", "yaw_rate")
ACTION_UNITS = ("m/s", "m/s", "m/s", "rad/s")


@dataclass(frozen=True)
class VelocityCommand:
    """High-level body-frame velocity command.

    Internal units are meters per second for linear axes and radians per
    second for yaw rate.
    """

    vx: float
    vy: float
    vz: float
    yaw_rate: float

    @classmethod
    def hover(cls) -> "VelocityCommand":
        return cls(0.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_iterable(
        cls,
        values: Iterable[float],
        *,
        yaw_unit: str = "rad/s",
    ) -> "VelocityCommand":
        raw = [float(v) for v in values]
        if len(raw) != 4:
            raise ValueError(f"Expected 4 command values, got {len(raw)}")

        if yaw_unit in {"deg/s", "degree/s", "degrees/s"}:
            raw[3] = math.radians(raw[3])
        elif yaw_unit not in {"rad/s", "radian/s", "radians/s"}:
            raise ValueError(f"Unsupported yaw unit: {yaw_unit}")

        return cls(*raw)

    def to_numpy(self, dtype: np.dtype | type = np.float32) -> np.ndarray:
        return np.asarray([self.vx, self.vy, self.vz, self.yaw_rate], dtype=dtype)

    def is_finite(self) -> bool:
        return bool(np.isfinite(self.to_numpy(np.float64)).all())

    def clamp(
        self,
        *,
        max_vx: float,
        max_vy: float,
        max_vz: float,
        max_yaw_rate: float,
    ) -> "VelocityCommand":
        return VelocityCommand(
            vx=float(np.clip(self.vx, -max_vx, max_vx)),
            vy=float(np.clip(self.vy, -max_vy, max_vy)),
            vz=float(np.clip(self.vz, -max_vz, max_vz)),
            yaw_rate=float(np.clip(self.yaw_rate, -max_yaw_rate, max_yaw_rate)),
        )

    def smooth_toward(self, previous: "VelocityCommand", alpha: float) -> "VelocityCommand":
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        current = self.to_numpy(np.float64)
        prev = previous.to_numpy(np.float64)
        smoothed = (alpha * current) + ((1.0 - alpha) * prev)
        return VelocityCommand.from_iterable(smoothed)


@dataclass(frozen=True)
class DatasetManifestRecord:
    """One aligned RGB-D/action frame in the dataset manifest."""

    source: str
    episode_id: str
    frame_id: str
    rgb_path: str
    depth_path: str | None = None
    pose: dict[str, Any] | None = None
    action: tuple[float, float, float, float] | None = None
    action_mask: tuple[bool, bool, bool, bool] = (False, False, False, False)
    timestamp: float | None = None
    split: str = "train"
    action_frame: str = "body"
    units: tuple[str, str, str, str] = ACTION_UNITS

    COLUMNS: ClassVar[tuple[str, ...]] = (
        "source",
        "episode_id",
        "frame_id",
        "rgb_path",
        "depth_path",
        "pose",
        "action",
        "action_mask",
        "timestamp",
        "split",
        "action_frame",
        "units",
    )

    def __post_init__(self) -> None:
        if self.action is not None and len(self.action) != 4:
            raise ValueError("action must have four values")
        if len(self.action_mask) != 4:
            raise ValueError("action_mask must have four values")
        if len(self.units) != 4:
            raise ValueError("units must have four values")

    def to_row(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "episode_id": self.episode_id,
            "frame_id": self.frame_id,
            "rgb_path": self.rgb_path,
            "depth_path": self.depth_path,
            "pose": json.dumps(self.pose, sort_keys=True) if self.pose is not None else None,
            "action": json.dumps(list(self.action)) if self.action is not None else None,
            "action_mask": json.dumps([bool(v) for v in self.action_mask]),
            "timestamp": self.timestamp,
            "split": self.split,
            "action_frame": self.action_frame,
            "units": json.dumps(list(self.units)),
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "DatasetManifestRecord":
        def is_missing(value: Any) -> bool:
            if value is None:
                return True
            if isinstance(value, str):
                return value == "" or value == "<NA>"
            if isinstance(value, float) and math.isnan(value):
                return True
            return type(value).__name__ == "NAType"

        def parse_json(value: Any, default: Any) -> Any:
            if is_missing(value):
                return default
            if isinstance(value, str):
                return json.loads(value)
            return value

        action = parse_json(row.get("action"), None)
        action_mask = parse_json(row.get("action_mask"), [False, False, False, False])
        units = parse_json(row.get("units"), list(ACTION_UNITS))
        pose = parse_json(row.get("pose"), None)
        depth_path = row.get("depth_path")
        if is_missing(depth_path):
            depth_path = None
        timestamp = row.get("timestamp")
        if is_missing(timestamp):
            timestamp = None

        return cls(
            source=str(row["source"]),
            episode_id=str(row["episode_id"]),
            frame_id=str(row["frame_id"]),
            rgb_path=str(row["rgb_path"]),
            depth_path=str(depth_path) if depth_path else None,
            pose=pose,
            action=tuple(float(v) for v in action) if action is not None else None,  # type: ignore[arg-type]
            action_mask=tuple(bool(v) for v in action_mask),  # type: ignore[arg-type]
            timestamp=float(timestamp) if timestamp is not None else None,
            split=str(row.get("split", "train")),
            action_frame=str(row.get("action_frame", "body")),
            units=tuple(str(v) for v in units),  # type: ignore[arg-type]
        )
