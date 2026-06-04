"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .manifest import (
    build_airsim_seed_manifest,
    ensure_episode_split_integrity,
    read_manifest,
    summarize_manifest,
    validate_alignment,
    write_manifest,
)
from .core_types import ACTION_NAMES


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


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
        )
    )
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
    metrics = run_closed_loop(
        adapter,
        policy,
        safety_filter,
        steps=args.steps,
        command_duration_s=args.command_duration,
        command_log_path=Path(args.command_log) if args.command_log else None,
    )
    _print_json(metrics.to_dict())
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
    airsim.add_argument("--command-log")
    airsim.add_argument("--invert-z", action="store_true")
    airsim.add_argument("--arm", action="store_true")
    airsim.add_argument("--takeoff", action="store_true")
    airsim.add_argument("--takeoff-altitude", type=float)
    airsim.add_argument("--takeoff-velocity", type=float, default=1.0)
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
