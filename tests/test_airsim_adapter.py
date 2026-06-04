from __future__ import annotations

import pytest

from drone_autopilot.simulators import airsim_adapter
from drone_autopilot.simulators.airsim_adapter import AirSimAdapter, _depth_image_type


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


class _AsyncResult:
    def __init__(self) -> None:
        self.joined = False

    def join(self) -> None:
        self.joined = True


class _Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def confirmConnection(self) -> None:
        self.calls.append(("confirmConnection", (), {}))

    def enableApiControl(self, enabled: bool, *, vehicle_name: str) -> None:
        self.calls.append(("enableApiControl", (enabled,), {"vehicle_name": vehicle_name}))

    def armDisarm(self, armed: bool, *, vehicle_name: str) -> None:
        self.calls.append(("armDisarm", (armed,), {"vehicle_name": vehicle_name}))

    def takeoffAsync(self, *, vehicle_name: str) -> _AsyncResult:
        self.calls.append(("takeoffAsync", (), {"vehicle_name": vehicle_name}))
        return _AsyncResult()

    def moveToZAsync(self, z: float, velocity: float, *, vehicle_name: str) -> _AsyncResult:
        self.calls.append(("moveToZAsync", (z, velocity), {"vehicle_name": vehicle_name}))
        return _AsyncResult()


def test_depth_image_type_prefers_current_depth_planar_name() -> None:
    assert _depth_image_type(_AirSimPlanar) == _ImageTypePlanar.DepthPlanar


def test_depth_image_type_falls_back_to_legacy_depth_planner_name() -> None:
    assert _depth_image_type(_AirSimPlanner) == _ImageTypePlanner.DepthPlanner


def test_depth_image_type_fails_when_no_depth_type_exists() -> None:
    with pytest.raises(RuntimeError, match="DepthPlanar"):
        _depth_image_type(_AirSimMissing)


def test_connect_moves_to_requested_takeoff_altitude(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(airsim_adapter, "_require_airsim", lambda: _AirSimPlanar)
    client = _Client()
    adapter = AirSimAdapter(client=client, vehicle_name="Drone1")

    adapter.connect(
        arm=True,
        takeoff=True,
        takeoff_altitude_m=3.0,
        takeoff_velocity_mps=1.5,
    )

    assert client.calls == [
        ("confirmConnection", (), {}),
        ("enableApiControl", (True,), {"vehicle_name": "Drone1"}),
        ("armDisarm", (True,), {"vehicle_name": "Drone1"}),
        ("takeoffAsync", (), {"vehicle_name": "Drone1"}),
        ("moveToZAsync", (-3.0, 1.5), {"vehicle_name": "Drone1"}),
    ]
