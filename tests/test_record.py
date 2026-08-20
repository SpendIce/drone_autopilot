from __future__ import annotations

import numpy as np

from drone_autopilot.cli import build_parser
from drone_autopilot.core_types import VelocityCommand
from drone_autopilot.manifest import build_airsim_seed_manifest
from drone_autopilot.mission import MissionConfig, MissionPlanner, Waypoint
from drone_autopilot.record import next_frame_id, record_episode
from drone_autopilot.safety import SafetyConfig, SafetyFilter
from drone_autopilot.simulators.base import Observation, SimulatorAdapter


def test_record_expert_cli_exposes_retreat_and_urgency_flags() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "record-expert",
            "--output-dir",
            "runs/expert",
            "--retreat-depth",
            "0.4",
            "--retreat-vx",
            "-0.5",
            "--urgency-exponent",
            "0.3",
        ]
    )

    assert args.retreat_depth == 0.4
    assert args.retreat_vx == -0.5
    assert args.urgency_exponent == 0.3


def test_record_expert_cli_defaults_match_config_defaults() -> None:
    from drone_autopilot.expert_policy import ReactiveAvoidanceConfig

    parser = build_parser()
    defaults = ReactiveAvoidanceConfig()

    args = parser.parse_args(["record-expert", "--output-dir", "runs/expert"])

    assert args.caution_depth == defaults.caution_depth_m
    assert args.retreat_depth == defaults.retreat_depth_m
    assert args.retreat_vx == defaults.retreat_vx_mps
    assert args.urgency_exponent == defaults.urgency_exponent


class _Adapter(SimulatorAdapter):
    def __init__(self) -> None:
        self.sent: list[VelocityCommand] = []
        self.hover_count = 0

    def capture_observation(self) -> Observation:
        return Observation(
            rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            depth_m=np.ones((8, 8), dtype=np.float32) * 5.0,
        )

    def send_velocity(self, command: VelocityCommand, *, duration_s: float) -> None:
        self.sent.append(command)

    def hover(self, *, duration_s: float) -> None:
        self.hover_count += 1

    def capture_state(self) -> dict[str, object]:
        return {"x": 0.0, "y": 0.0, "z": -3.0, "yaw": 0.0}


class _Policy:
    def predict(self, rgb: np.ndarray, depth_m: np.ndarray) -> VelocityCommand:
        return VelocityCommand(0.4, 0.1, 0.0, 0.05)


def test_record_episode_writes_manifest_compatible_layout(tmp_path) -> None:
    result = record_episode(
        _Adapter(),
        _Policy(),
        SafetyFilter(SafetyConfig(smoothing_alpha=1.0)),
        steps=3,
        output_dir=tmp_path,
    )

    assert result.frames_written == 3
    assert result.next_frame_id == 3
    assert sorted(p.name for p in (tmp_path / "rgb").glob("*.png")) == [
        "000000.png",
        "000001.png",
        "000002.png",
    ]
    assert sorted(p.name for p in (tmp_path / "depth").glob("*.npy")) == [
        "000000.npy",
        "000001.npy",
        "000002.npy",
    ]
    command_array = np.load(tmp_path / "commands" / "000000.npy")
    assert command_array.shape == (4,)

    records = build_airsim_seed_manifest(tmp_path, episode_length=10)
    assert len(records) == 3
    assert records[0].action_frame == "body"


def test_record_episode_appends_after_existing_frames(tmp_path) -> None:
    record_episode(_Adapter(), _Policy(), SafetyFilter(SafetyConfig(smoothing_alpha=1.0)), steps=2, output_dir=tmp_path)

    assert next_frame_id(tmp_path) == 2

    second = record_episode(
        _Adapter(),
        _Policy(),
        SafetyFilter(SafetyConfig(smoothing_alpha=1.0)),
        steps=2,
        output_dir=tmp_path,
    )

    assert second.next_frame_id == 4
    assert len(list((tmp_path / "rgb").glob("*.png"))) == 4


def test_record_episode_uses_mission_planner_blended_command(tmp_path) -> None:
    adapter = _Adapter()
    planner = MissionPlanner(
        [Waypoint(10.0, 0.0)],
        MissionConfig(cruise_speed_mps=1.0, waypoint_radius_m=0.5, slow_radius_m=5.0, position_blend=1.0, yaw_blend=1.0),
    )

    record_episode(
        adapter,
        _Policy(),
        SafetyFilter(SafetyConfig(smoothing_alpha=1.0)),
        steps=1,
        output_dir=tmp_path,
        mission_planner=planner,
    )

    assert adapter.sent == [VelocityCommand(1.0, 0.0, 0.0, 0.0)]
