from __future__ import annotations

import pytest

from drone_autopilot.simulators.airsim_adapter import _depth_image_type


class _ImageTypePlanar:
    DepthPlanar = 1
    DepthPlanner = 2


class _ImageTypePlanner:
    DepthPlanner = 2


class _ImageTypeMissing:
    Scene = 0


class _AirSimPlanar:
    ImageType = _ImageTypePlanar


class _AirSimPlanner:
    ImageType = _ImageTypePlanner


class _AirSimMissing:
    ImageType = _ImageTypeMissing


def test_depth_image_type_prefers_current_depth_planar_name() -> None:
    assert _depth_image_type(_AirSimPlanar) == _ImageTypePlanar.DepthPlanar


def test_depth_image_type_falls_back_to_legacy_depth_planner_name() -> None:
    assert _depth_image_type(_AirSimPlanner) == _ImageTypePlanner.DepthPlanner


def test_depth_image_type_fails_when_no_depth_type_exists() -> None:
    with pytest.raises(RuntimeError, match="DepthPlanar"):
        _depth_image_type(_AirSimMissing)
