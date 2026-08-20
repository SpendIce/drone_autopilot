"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .manifest import (
    build_airsim_seed_manifest,
    ensure_episode_split_integrity,
    merge_manifests,
    read_manifest,
    summarize_manifest,
    validate_alignment,
    write_manifest,
)
from .core_types import ACTION_NAMES
from .mission import Waypoint, parse_waypoint


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _parse_waypoint_arg(value: str) -> Waypoint:
    try:
        return parse_waypoint(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def cmd_build_airsim_manifest(args: argparse.Namespace) -> int:
    dataset_root = Path(args.dataset_root)
    output = Path(args.output) if args.output else dataset_root / "manifest.parquet"
    records = build_airsim_seed_manifest(
        dataset_root,
        source=args.source,
        path_root=Path(args.path_root) if args.path_root else dataset_root,
        episode_length=args.episode_length,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    write_manifest(records, output)
    summary = summarize_manifest(records)
    _print_json(
        {
            "output": str(output),
            "rows": summary.rows,
            "sources": summary.sources,
            "splits": summary.splits,
            "labeled_actions": summary.labeled_actions,
        }
    )
    return 0


def cmd_merge_manifests(args: argparse.Namespace) -> int:
    records = merge_manifests(args.manifests)
    output = Path(args.output)
    write_manifest(records, output)
    summary = summarize_manifest(records)
    _print_json(
        {
            "output": str(output),
            "rows": summary.rows,
            "sources": summary.sources,
            "splits": summary.splits,
            "labeled_actions": summary.labeled_actions,
        }
    )
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    records = read_manifest(args.manifest)
    summary = summarize_manifest(records)
    payload: dict[str, Any] = {
        "rows": summary.rows,
        "sources": summary.sources,
        "splits": summary.splits,
        "labeled_actions": summary.labeled_actions,
    }
    if summary.action_mean is not None and summary.action_std is not None:
        payload["action_mean"] = dict(zip(ACTION_NAMES, summary.action_mean, strict=True))
        payload["action_std"] = dict(zip(ACTION_NAMES, summary.action_std, strict=True))
    _print_json(payload)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    records = read_manifest(args.manifest)
    errors = validate_alignment(records, data_root=args.data_root)
    errors.extend(ensure_episode_split_integrity(records))
    if errors:
        _print_json({"ok": False, "errors": errors})
        return 1
    _print_json({"ok": True, "rows": len(records)})
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .training import TrainingConfig, train

    result = train(
        TrainingConfig(
            manifest_path=Path(args.manifest),
            data_root=Path(args.data_root),
            output_path=Path(args.output),
            best_output_path=Path(args.best_output) if args.best_output else None,
            backbone=args.backbone,
            modality=args.modality,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            num_workers=args.num_workers,
            max_depth_m=args.max_depth_m,
            vy_weight=args.vy_weight,
            device=args.device,
            multi_gpu=args.multi_gpu,
            distributed=args.distributed,
        )
    )
    if not result.pop("_suppress_cli_output", False):
        _print_json(result)
    return 0


def cmd_airsim_loop(args: argparse.Namespace) -> int:
    from .inference import TorchPilotPolicy
    from .mission import MissionConfig, MissionPlanner
    from .safety import SafetyConfig, SafetyFilter
    from .simulators.airsim_adapter import AirSimAdapter
    from .simulators.base import run_closed_loop

    policy = TorchPilotPolicy(
        args.checkpoint,
        image_size=args.image_size,
        max_depth_m=args.max_depth_m,
        device=args.device,
    )
    safety_filter = SafetyFilter(
        SafetyConfig(
            max_vx_mps=args.max_vx,
            max_vy_mps=args.max_vy,
            max_vz_mps=args.max_vz,
            max_yaw_rate_radps=args.max_yaw_rate,
            smoothing_alpha=args.smoothing_alpha,
            vx_deadband_mps=args.vx_deadband,
            vy_deadband_mps=args.vy_deadband,
            vz_deadband_mps=args.vz_deadband,
            yaw_rate_deadband_radps=args.yaw_deadband,
            emergency_depth_m=args.emergency_depth,
            depth_roi_top=args.depth_roi_top,
            depth_roi_bottom=args.depth_roi_bottom,
            depth_roi_left=args.depth_roi_left,
            depth_roi_right=args.depth_roi_right,
        )
    )
    adapter = AirSimAdapter(
        vehicle_name=args.vehicle_name,
        scene_camera=args.scene_camera,
        depth_camera=args.depth_camera,
        invert_z=args.invert_z,
        hold_altitude=args.hold_altitude,
        wait_for_commands=not args.async_commands,
        depth_capture_interval=args.depth_interval,
        release_control_on_close=not args.keep_api_control,
    )
    adapter.connect(
        arm=args.arm,
        takeoff=args.takeoff,
        takeoff_altitude_m=args.takeoff_altitude,
        takeoff_velocity_mps=args.takeoff_velocity,
    )
    if any(
        value is not None
        for value in (args.start_x, args.start_y, args.start_z, args.start_yaw_deg)
    ):
        adapter.set_pose(
            x=args.start_x,
            y=args.start_y,
            z=args.start_z,
            yaw_rad=math.radians(args.start_yaw_deg)
            if args.start_yaw_deg is not None
            else None,
        )
    mission_planner = None
    waypoints = args.waypoint or []
    if waypoints:
        mission_planner = MissionPlanner(
            waypoints,
            MissionConfig(
                cruise_speed_mps=args.mission_cruise_speed,
                waypoint_radius_m=args.mission_waypoint_radius,
                slow_radius_m=args.mission_slow_radius,
                position_blend=args.mission_position_blend,
                yaw_blend=args.mission_yaw_blend,
                heading_gain=args.mission_heading_gain,
                max_goal_yaw_rate_radps=args.mission_max_yaw_rate,
            ),
        )
    metrics = run_closed_loop(
        adapter,
        policy,
        safety_filter,
        steps=args.steps,
        command_duration_s=args.command_duration,
        command_log_path=Path(args.command_log) if args.command_log else None,
        mission_planner=mission_planner,
    )
    _print_json(metrics.to_dict())
    return 0


def cmd_record_expert(args: argparse.Namespace) -> int:
    from .expert_policy import ReactiveAvoidanceConfig, ReactiveAvoidancePolicy
    from .mission import MissionConfig, MissionPlanner
    from .record import record_episode
    from .safety import SafetyConfig, SafetyFilter
    from .simulators.airsim_adapter import AirSimAdapter

    policy = ReactiveAvoidancePolicy(
        ReactiveAvoidanceConfig(
            cruise_vx_mps=args.cruise_vx,
            caution_depth_m=args.caution_depth,
            min_forward_depth_m=args.min_forward_depth,
            retreat_depth_m=args.retreat_depth,
            retreat_vx_mps=args.retreat_vx,
            max_lateral_vy_mps=args.max_lateral_vy,
            max_avoid_yaw_radps=args.max_avoid_yaw_rate,
            urgency_exponent=args.urgency_exponent,
        )
    )
    safety_filter = SafetyFilter(
        SafetyConfig(
            max_vx_mps=args.max_vx,
            max_vy_mps=args.max_vy,
            max_vz_mps=args.max_vz,
            max_yaw_rate_radps=args.max_yaw_rate,
            smoothing_alpha=args.smoothing_alpha,
            emergency_depth_m=args.emergency_depth,
            depth_roi_top=args.depth_roi_top,
            depth_roi_bottom=args.depth_roi_bottom,
            depth_roi_left=args.depth_roi_left,
            depth_roi_right=args.depth_roi_right,
        )
    )
    adapter = AirSimAdapter(
        vehicle_name=args.vehicle_name,
        scene_camera=args.scene_camera,
        depth_camera=args.depth_camera,
        invert_z=args.invert_z,
        hold_altitude=args.hold_altitude,
        wait_for_commands=not args.async_commands,
        depth_capture_interval=args.depth_interval,
        release_control_on_close=not args.keep_api_control,
    )
    adapter.connect(
        arm=args.arm,
        takeoff=args.takeoff,
        takeoff_altitude_m=args.takeoff_altitude,
        takeoff_velocity_mps=args.takeoff_velocity,
    )
    if any(
        value is not None
        for value in (args.start_x, args.start_y, args.start_z, args.start_yaw_deg)
    ):
        adapter.set_pose(
            x=args.start_x,
            y=args.start_y,
            z=args.start_z,
            yaw_rad=math.radians(args.start_yaw_deg)
            if args.start_yaw_deg is not None
            else None,
        )
    mission_planner = None
    waypoints = args.waypoint or []
    if waypoints:
        mission_planner = MissionPlanner(
            waypoints,
            MissionConfig(
                cruise_speed_mps=args.mission_cruise_speed,
                waypoint_radius_m=args.mission_waypoint_radius,
                slow_radius_m=args.mission_slow_radius,
                position_blend=args.mission_position_blend,
                yaw_blend=args.mission_yaw_blend,
                heading_gain=args.mission_heading_gain,
                max_goal_yaw_rate_radps=args.mission_max_yaw_rate,
            ),
        )
    result = record_episode(
        adapter,
        policy,
        safety_filter,
        steps=args.steps,
        command_duration_s=args.command_duration,
        output_dir=Path(args.output_dir),
        mission_planner=mission_planner,
    )
    _print_json(
        {
            "frames_written": result.frames_written,
            "next_frame_id": result.next_frame_id,
            "emergency_stops": result.emergency_stops,
            "mission_complete": result.mission_complete,
            "output_dir": args.output_dir,
        }
    )
    return 0


def cmd_airsim_snapshot(args: argparse.Namespace) -> int:
    from .debug import save_observation_snapshot
    from .simulators.airsim_adapter import AirSimAdapter

    adapter = AirSimAdapter(
        vehicle_name=args.vehicle_name,
        scene_camera=args.scene_camera,
        depth_camera=args.depth_camera,
        invert_z=args.invert_z,
    )
    adapter.connect(
        arm=args.arm,
        takeoff=args.takeoff,
        takeoff_altitude_m=args.takeoff_altitude,
        takeoff_velocity_mps=args.takeoff_velocity,
    )
    observation = adapter.capture_observation()
    summary = save_observation_snapshot(
        observation,
        Path(args.output_dir),
        depth_vis_max_m=args.depth_vis_max,
    )
    _print_json(summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="drone-autopilot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-airsim-manifest")
    build.add_argument("dataset_root")
    build.add_argument("--output")
    build.add_argument("--source", default="airsim_obstacle_avoidance_seed")
    build.add_argument("--path-root")
    build.add_argument("--episode-length", type=int, default=500)
    build.add_argument("--train-ratio", type=float, default=0.8)
    build.add_argument("--val-ratio", type=float, default=0.1)
    build.set_defaults(func=cmd_build_airsim_manifest)

    merge = subparsers.add_parser("merge-manifests")
    merge.add_argument("manifests", nargs="+")
    merge.add_argument("--output", required=True)
    merge.set_defaults(func=cmd_merge_manifests)

    stats = subparsers.add_parser("stats")
    stats.add_argument("manifest")
    stats.set_defaults(func=cmd_stats)

    validate = subparsers.add_parser("validate")
    validate.add_argument("manifest")
    validate.add_argument("--data-root", default=".")
    validate.set_defaults(func=cmd_validate)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("manifest")
    train_parser.add_argument("--data-root", default=".")
    train_parser.add_argument("--output", default="checkpoints/rgbd_pilot.pt")
    train_parser.add_argument("--best-output")
    train_parser.add_argument("--backbone", default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet18", "tiny"])
    train_parser.add_argument("--modality", default="rgbd", choices=["rgb", "depth", "rgbd"])
    train_parser.add_argument("--image-size", type=int, default=224)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--max-depth-m", type=float, default=50.0)
    train_parser.add_argument("--vy-weight", type=float, default=0.25)
    train_parser.add_argument("--device", default="auto")
    gpu_mode = train_parser.add_mutually_exclusive_group()
    gpu_mode.add_argument("--distributed", action="store_true")
    gpu_mode.add_argument("--multi-gpu", dest="multi_gpu", action="store_true")
    gpu_mode.add_argument("--no-multi-gpu", dest="multi_gpu", action="store_false")
    train_parser.set_defaults(distributed=False, multi_gpu=False)
    train_parser.set_defaults(func=cmd_train)

    airsim = subparsers.add_parser("airsim-loop")
    airsim.add_argument("checkpoint")
    airsim.add_argument("--steps", type=int, default=200)
    airsim.add_argument("--command-duration", type=float, default=0.1)
    airsim.add_argument("--vehicle-name", default="")
    airsim.add_argument("--scene-camera", default="0")
    airsim.add_argument("--depth-camera", default="0")
    airsim.add_argument("--image-size", type=int, default=224)
    airsim.add_argument("--max-depth-m", type=float, default=50.0)
    airsim.add_argument("--device", default="auto")
    airsim.add_argument("--max-vx", type=float, default=2.0)
    airsim.add_argument("--max-vy", type=float, default=1.0)
    airsim.add_argument("--max-vz", type=float, default=1.0)
    airsim.add_argument("--max-yaw-rate", type=float, default=0.8)
    airsim.add_argument("--smoothing-alpha", type=float, default=0.2)
    airsim.add_argument("--vx-deadband", type=float, default=0.0)
    airsim.add_argument("--vy-deadband", type=float, default=0.02)
    airsim.add_argument("--vz-deadband", type=float, default=0.02)
    airsim.add_argument("--yaw-deadband", type=float, default=0.05)
    airsim.add_argument("--emergency-depth", type=float, default=0.8)
    airsim.add_argument("--depth-roi-top", type=float, default=0.0)
    airsim.add_argument("--depth-roi-bottom", type=float, default=1.0)
    airsim.add_argument("--depth-roi-left", type=float, default=0.0)
    airsim.add_argument("--depth-roi-right", type=float, default=1.0)
    airsim.add_argument("--command-log")
    airsim.add_argument("--invert-z", action="store_true")
    airsim.add_argument("--hold-altitude", action="store_true")
    airsim.add_argument("--async-commands", action="store_true")
    airsim.add_argument("--keep-api-control", action="store_true")
    airsim.add_argument("--depth-interval", type=int, default=1)
    airsim.add_argument("--arm", action="store_true")
    airsim.add_argument("--takeoff", action="store_true")
    airsim.add_argument("--takeoff-altitude", type=float)
    airsim.add_argument("--takeoff-velocity", type=float, default=1.0)
    airsim.add_argument("--start-x", type=float)
    airsim.add_argument("--start-y", type=float)
    airsim.add_argument("--start-z", type=float, help="AirSim NED z coordinate; altitude 3m is -3.0")
    airsim.add_argument("--start-yaw-deg", type=float)
    airsim.add_argument("--waypoint", action="append", type=_parse_waypoint_arg)
    airsim.add_argument("--mission-cruise-speed", type=float, default=0.45)
    airsim.add_argument("--mission-waypoint-radius", type=float, default=1.5)
    airsim.add_argument("--mission-slow-radius", type=float, default=4.0)
    airsim.add_argument("--mission-position-blend", type=float, default=0.45)
    airsim.add_argument("--mission-yaw-blend", type=float, default=0.5)
    airsim.add_argument("--mission-heading-gain", type=float, default=0.8)
    airsim.add_argument("--mission-max-yaw-rate", type=float, default=0.25)
    airsim.set_defaults(func=cmd_airsim_loop)

    snapshot = subparsers.add_parser("airsim-snapshot")
    snapshot.add_argument("--output-dir", default="runs/airsim_snapshot")
    snapshot.add_argument("--vehicle-name", default="")
    snapshot.add_argument("--scene-camera", default="0")
    snapshot.add_argument("--depth-camera", default="0")
    snapshot.add_argument("--depth-vis-max", type=float, default=10.0)
    snapshot.add_argument("--invert-z", action="store_true")
    snapshot.add_argument("--arm", action="store_true")
    snapshot.add_argument("--takeoff", action="store_true")
    snapshot.add_argument("--takeoff-altitude", type=float)
    snapshot.add_argument("--takeoff-velocity", type=float, default=1.0)
    snapshot.set_defaults(func=cmd_airsim_snapshot)

    record = subparsers.add_parser("record-expert")
    record.add_argument("--output-dir", required=True)
    record.add_argument("--steps", type=int, default=200)
    record.add_argument("--command-duration", type=float, default=0.1)
    record.add_argument("--vehicle-name", default="")
    record.add_argument("--scene-camera", default="0")
    record.add_argument("--depth-camera", default="0")
    record.add_argument("--device", default="auto")
    record.add_argument("--cruise-vx", type=float, default=0.6)
    record.add_argument("--caution-depth", type=float, default=4.5)
    record.add_argument("--min-forward-depth", type=float, default=1.2)
    record.add_argument("--retreat-depth", type=float, default=0.6)
    record.add_argument("--retreat-vx", type=float, default=-0.35)
    record.add_argument("--urgency-exponent", type=float, default=0.5)
    record.add_argument("--max-lateral-vy", type=float, default=0.6)
    record.add_argument("--max-avoid-yaw-rate", type=float, default=0.5)
    record.add_argument("--max-vx", type=float, default=2.0)
    record.add_argument("--max-vy", type=float, default=1.0)
    record.add_argument("--max-vz", type=float, default=1.0)
    record.add_argument("--max-yaw-rate", type=float, default=0.8)
    record.add_argument("--smoothing-alpha", type=float, default=0.2)
    record.add_argument("--emergency-depth", type=float, default=0.8)
    record.add_argument("--depth-roi-top", type=float, default=0.0)
    record.add_argument("--depth-roi-bottom", type=float, default=1.0)
    record.add_argument("--depth-roi-left", type=float, default=0.0)
    record.add_argument("--depth-roi-right", type=float, default=1.0)
    record.add_argument("--invert-z", action="store_true")
    record.add_argument("--hold-altitude", action="store_true")
    record.add_argument("--async-commands", action="store_true")
    record.add_argument("--keep-api-control", action="store_true")
    record.add_argument("--depth-interval", type=int, default=1)
    record.add_argument("--arm", action="store_true")
    record.add_argument("--takeoff", action="store_true")
    record.add_argument("--takeoff-altitude", type=float)
    record.add_argument("--takeoff-velocity", type=float, default=1.0)
    record.add_argument("--start-x", type=float)
    record.add_argument("--start-y", type=float)
    record.add_argument("--start-z", type=float)
    record.add_argument("--start-yaw-deg", type=float)
    record.add_argument("--waypoint", action="append", type=_parse_waypoint_arg)
    record.add_argument("--mission-cruise-speed", type=float, default=0.6)
    record.add_argument("--mission-waypoint-radius", type=float, default=1.5)
    record.add_argument("--mission-slow-radius", type=float, default=4.0)
    record.add_argument("--mission-position-blend", type=float, default=0.6)
    record.add_argument("--mission-yaw-blend", type=float, default=0.6)
    record.add_argument("--mission-heading-gain", type=float, default=0.8)
    record.add_argument("--mission-max-yaw-rate", type=float, default=0.25)
    record.set_defaults(func=cmd_record_expert)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
