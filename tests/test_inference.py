from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from drone_autopilot.inference import TorchPilotPolicy


def _bare_policy(modality: str) -> TorchPilotPolicy:
    """Build a TorchPilotPolicy without loading a checkpoint — only the
    modality-zeroing tensor helpers are under test here."""
    policy = object.__new__(TorchPilotPolicy)
    policy.torch = torch
    policy.device = "cpu"
    policy.image_size = 8
    policy.max_depth_m = 50.0
    policy.modality = modality
    return policy


def _rgb_frame() -> np.ndarray:
    return (np.random.rand(8, 8, 3) * 255).astype(np.uint8)


def _depth_frame() -> np.ndarray:
    return np.full((8, 8), 3.0, dtype=np.float32)


def test_rejects_invalid_modality() -> None:
    with pytest.raises(ValueError, match="modality"):
        TorchPilotPolicy.__init__(
            object.__new__(TorchPilotPolicy),
            checkpoint_path="unused",
            modality="not-a-modality",
        )


def test_rgbd_modality_keeps_both_tensors_nonzero() -> None:
    policy = _bare_policy("rgbd")

    rgb_tensor = policy._rgb_tensor(_rgb_frame())
    depth_tensor = policy._depth_tensor(_depth_frame())

    assert torch.count_nonzero(rgb_tensor) > 0
    assert torch.count_nonzero(depth_tensor) > 0


def test_depth_modality_zeros_rgb_tensor() -> None:
    policy = _bare_policy("depth")

    rgb_tensor = policy._rgb_tensor(_rgb_frame())
    depth_tensor = policy._depth_tensor(_depth_frame())

    assert torch.count_nonzero(rgb_tensor) == 0
    assert torch.count_nonzero(depth_tensor) > 0


def test_rgb_modality_zeros_depth_tensor() -> None:
    policy = _bare_policy("rgb")

    rgb_tensor = policy._rgb_tensor(_rgb_frame())
    depth_tensor = policy._depth_tensor(_depth_frame())

    assert torch.count_nonzero(rgb_tensor) > 0
    assert torch.count_nonzero(depth_tensor) == 0


def test_depth_tensor_shape_matches_rgbd_when_zeroed_for_rgb_modality() -> None:
    policy = _bare_policy("rgb")

    depth_tensor = policy._depth_tensor(_depth_frame())

    assert tuple(depth_tensor.shape) == (1, 1, policy.image_size, policy.image_size)
