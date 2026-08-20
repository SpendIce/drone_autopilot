"""Record closed-loop episodes as imitation-learning demonstrations.

Writes frames in the layout `build_airsim_seed_manifest` already expects
(`rgb/<frame>.png`, `depth/<frame>.npy`, `commands/<frame>.npy` with
`[vx, vy, vz, yaw_rate_deg_s]`), so recorded episodes need no new ingestion
code. The recorded label is the command that actually executed (after
mission blending and the safety filter), matching standard imitation
learning practice of cloning the expert's realized actions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .core_types import VelocityCommand
from .mission import MissionPlanner
from .safety import SafetyFilter
from .simulators.base import PilotPolicy, SimulatorAdapter


@dataclass
class RecordingResult:
    frames_written: int
    next_frame_id: int
    emergency_stops: int
    mission_complete: bool


def next_frame_id(output_dir: Path | str) -> int:
    """Highest existing frame id in output_dir, plus one (0 if empty)."""
    rgb_dir = Path(output_dir) / "rgb"
    if not rgb_dir.is_dir():
        return 0
    existing = [int(p.stem) for p in rgb_dir.glob("*.png") if p.stem.isdigit()]
    return max(existing, default=-1) + 1


def record_episode(
    adapter: SimulatorAdapter,
    expert_policy: PilotPolicy,
    safety_filter: SafetyFilter,
    *,
    steps: int,
    command_duration_s: float = 0.1,
    output_dir: Path | str,
    start_frame_id: int | None = None,
    mission_planner: MissionPlanner | None = None,
) -> RecordingResult:
    if steps <= 0:
        raise ValueError("steps must be positive")

    output_dir = Path(output_dir)
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    command_dir = output_dir / "commands"
    for directory in (rgb_dir, depth_dir, command_dir):
        directory.mkdir(parents=True, exist_ok=True)

    frame_id = start_frame_id if start_frame_id is not None else next_frame_id(output_dir)
    emergency_stops = 0
    mission_complete = False

    for _ in range(steps):
        observation = adapter.capture_observation()
        prediction = expert_policy.predict(observation.rgb, observation.depth_m)

        planned = prediction
        if mission_planner is not None:
            state = adapter.capture_state()
            mission_output = mission_planner.update(prediction, state)
            planned = mission_output.command
            mission_complete = mission_output.mission_complete

        result = safety_filter.filter(planned, depth_m=observation.depth_m, reactive=prediction)
        if result.emergency_stop:
            emergency_stops += 1
            adapter.hover(duration_s=command_duration_s)
        else:
            adapter.send_velocity(result.command, duration_s=command_duration_s)

        _write_frame(
            rgb_dir=rgb_dir,
            depth_dir=depth_dir,
            command_dir=command_dir,
            frame_id=frame_id,
            observation=observation,
            command=result.command,
        )
        frame_id += 1

    return RecordingResult(
        frames_written=steps,
        next_frame_id=frame_id,
        emergency_stops=emergency_stops,
        mission_complete=mission_complete,
    )


def _write_frame(
    *,
    rgb_dir: Path,
    depth_dir: Path,
    command_dir: Path,
    frame_id: int,
    observation: Any,
    command: VelocityCommand,
) -> None:
    frame_name = f"{frame_id:06d}"
    rgb = np.asarray(observation.rgb, dtype=np.uint8)
    Image.fromarray(rgb).convert("RGB").save(rgb_dir / f"{frame_name}.png")
    np.save(depth_dir / f"{frame_name}.npy", np.asarray(observation.depth_m, dtype=np.float32))
    command_array = np.array(
        [command.vx, command.vy, command.vz, math.degrees(command.yaw_rate)],
        dtype=np.float32,
    )
    np.save(command_dir / f"{frame_name}.npy", command_array)
