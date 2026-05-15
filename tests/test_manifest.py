from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from drone_autopilot.manifest import (
    build_airsim_seed_manifest,
    ensure_episode_split_integrity,
    read_manifest,
    validate_alignment,
    write_manifest,
)


def _make_seed_dataset(root: Path, count: int = 4) -> None:
    for subdir in ("rgb", "depth", "commands"):
        (root / subdir).mkdir(parents=True)
    for index in range(count):
        frame_id = f"{index:06d}"
        Image.new("RGB", (8, 8), color=(index, 0, 0)).save(root / "rgb" / f"{frame_id}.png")
        np.save(root / "depth" / f"{frame_id}.npy", np.ones((8, 8), dtype=np.float16) * 3.0)
        np.save(
            root / "commands" / f"{frame_id}.npy",
            np.asarray([1.0, 0.0, -0.5, 90.0], dtype=np.float32),
        )


def test_build_airsim_manifest_converts_yaw_and_keeps_episode_splits(tmp_path: Path) -> None:
    _make_seed_dataset(tmp_path)

    records = build_airsim_seed_manifest(tmp_path, episode_length=2, train_ratio=0.7, val_ratio=0.2)

    assert len(records) == 4
    assert records[0].rgb_path == "rgb/000000.png"
    assert records[0].depth_path == "depth/000000.npy"
    assert records[0].action is not None
    assert records[0].action[:3] == (1.0, 0.0, -0.5)
    assert math.isclose(records[0].action[3], math.pi / 2.0, rel_tol=1e-6)
    assert not ensure_episode_split_integrity(records)


def test_manifest_roundtrip_and_alignment_validation(tmp_path: Path) -> None:
    _make_seed_dataset(tmp_path)
    records = build_airsim_seed_manifest(tmp_path)
    output = tmp_path / "manifest.csv"

    write_manifest(records, output)
    loaded = read_manifest(output)

    assert len(loaded) == len(records)
    assert loaded[0].action_mask == (True, True, True, True)
    assert validate_alignment(loaded, data_root=tmp_path) == []
