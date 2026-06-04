from __future__ import annotations

import pytest

from drone_autopilot.simulators import airsim_adapter
from drone_autopilot.simulators.airsim_adapter import AirSimAdapter, _depth_image_type
from drone_autopilot.core_types import VelocityCommand


class _ImageTypePlanar:
    Scene = 0
    DepthPlanar = 1
    DepthPlanner = 2


class _ImageTypePlanner:
    DepthPlanner = 2


class _ImageTypeMissing:
    Scene = 0


class _YawMode:
    def __init__(self, *, is_rate: bool, yaw_or_rate: float) -> None:
        self.is_rate = is_rate
        self.yaw_or_rate = yaw_or_rate


class _ImageRequest:
    def __init__(self, camera: str, image_type: int, pixels_as_float: bool, compress: bool) -> None:
        self.camera = camera
        self.image_type = image_type
        self.pixels_as_float = pixels_as_float
        self.compress = compress


class _AirSimPlanar:
    ImageType = _ImageTypePlanar
    YawMode = _YawMode
    ImageRequest = _ImageRequest


class _AirSimPlanner:
    ImageType = _ImageTypePlanner


class _AirSimMissing:
    ImageType = _ImageTypeMissing


class _AsyncResult:
    def __init__(self) -> None:
        self.joined = False

    def join(self) -> None:
        self.joined = True


class _Vector:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x_val = x
        self.y_val = y
        self.z_val = z


class _Kinematics:
    def __init__(self) -> None:
        self.position = _Vector(1.0, 2.0, -3.0)
        self.linear_velocity = _Vector(0.1, 0.2, -0.3)


class _State:
    def __init__(self) -> None:
        self.kinematics_estimated = _Kinematics()


class _Collision:
    has_collided = False
    object_name = ""


class _ImageResponse:
    def __init__(self, *, image_type: int, depth_value: float = 1.0) -> None:
        self.height = 2
        self.width = 2
        self.time_stamp = 123
        if image_type == _ImageTypePlanar.Scene:
            self.image_data_uint8 = bytes([0, 0, 0] * self.height * self.width)
            self.image_data_float = []
        else:
            self.image_data_uint8 = b""
            self.image_data_float = [depth_value] * self.height * self.width


class _Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.last_async_result: _AsyncResult | None = None
        self.depth_value = 1.0

    def _async_result(self) -> _AsyncResult:
        self.last_async_result = _AsyncResult()
        return self.last_async_result

    def confirmConnection(self) -> None:
        self.calls.append(("confirmConnection", (), {}))

    def enableApiControl(self, enabled: bool, *, vehicle_name: str) -> None:
        self.calls.append(("enableApiControl", (enabled,), {"vehicle_name": vehicle_name}))

    def armDisarm(self, armed: bool, *, vehicle_name: str) -> None:
        self.calls.append(("armDisarm", (armed,), {"vehicle_name": vehicle_name}))

    def takeoffAsync(self, *, vehicle_name: str) -> _AsyncResult:
        self.calls.append(("takeoffAsync", (), {"vehicle_name": vehicle_name}))
        return self._async_result()

    def moveToZAsync(self, z: float, velocity: float, *, vehicle_name: str) -> _AsyncResult:
        self.calls.append(("moveToZAsync", (z, velocity), {"vehicle_name": vehicle_name}))
        return self._async_result()

    def hoverAsync(self, *, vehicle_name: str) -> _AsyncResult:
        self.calls.append(("hoverAsync", (), {"vehicle_name": vehicle_name}))
        return self._async_result()

    def moveByVelocityZBodyFrameAsync(
        self,
        vx: float,
        vy: float,
        z: float,
        duration: float,
        *,
        yaw_mode: _YawMode,
        vehicle_name: str,
    ) -> _AsyncResult:
        self.calls.append(
            (
                "moveByVelocityZBodyFrameAsync",
                (vx, vy, z, duration, yaw_mode.yaw_or_rate),
                {"vehicle_name": vehicle_name},
            )
        )
        return self._async_result()

    def getMultirotorState(self, *, vehicle_name: str) -> _State:
        self.calls.append(("getMultirotorState", (), {"vehicle_name": vehicle_name}))
        return _State()

    def simGetCollisionInfo(self, *, vehicle_name: str) -> _Collision:
        self.calls.append(("simGetCollisionInfo", (), {"vehicle_name": vehicle_name}))
        return _Collision()

    def simGetImages(self, requests: list[_ImageRequest], *, vehicle_name: str):
        self.calls.append(
            (
                "simGetImages",
                (tuple(request.image_type for request in requests),),
                {"vehicle_name": vehicle_name},
            )
        )
        responses = []
        for request in requests:
            responses.append(_ImageResponse(image_type=request.image_type, depth_value=self.depth_value))
            if request.image_type != _ImageTypePlanar.Scene:
                self.depth_value += 1.0
        return responses


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


def test_hover_uses_airsim_hover_async(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(airsim_adapter, "_require_airsim", lambda: _AirSimPlanar)
    monkeypatch.setattr(airsim_adapter.time, "sleep", lambda duration: None)
    client = _Client()
    adapter = AirSimAdapter(client=client, vehicle_name="Drone1")

    adapter.hover(duration_s=0.1)

    assert client.calls == [("hoverAsync", (), {"vehicle_name": "Drone1"})]


def test_hold_altitude_uses_takeoff_altitude_for_body_frame_velocity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(airsim_adapter, "_require_airsim", lambda: _AirSimPlanar)
    client = _Client()
    adapter = AirSimAdapter(client=client, vehicle_name="Drone1", hold_altitude=True)
    adapter.connect(takeoff_altitude_m=3.0)

    adapter.send_velocity(VelocityCommand(0.4, 0.1, -0.5, 0.2), duration_s=0.05)

    assert client.calls[-1] == (
        "moveByVelocityZBodyFrameAsync",
        (0.4, 0.1, -3.0, 0.05, pytest.approx(11.459155902616466)),
        {"vehicle_name": "Drone1"},
    )
    assert client.last_async_result is not None
    assert client.last_async_result.joined


def test_async_commands_do_not_wait_for_velocity_future(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(airsim_adapter, "_require_airsim", lambda: _AirSimPlanar)
    client = _Client()
    adapter = AirSimAdapter(
        client=client,
        vehicle_name="Drone1",
        hold_altitude=True,
        wait_for_commands=False,
    )
    adapter.connect(takeoff_altitude_m=3.0)

    adapter.send_velocity(VelocityCommand(0.4, 0.1, 0.0, 0.2), duration_s=0.5)

    assert client.last_async_result is not None
    assert not client.last_async_result.joined


def test_capture_state_returns_pose_velocity_and_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(airsim_adapter, "_require_airsim", lambda: _AirSimPlanar)
    adapter = AirSimAdapter(client=_Client(), vehicle_name="Drone1")

    state = adapter.capture_state()

    assert state == {
        "x": 1.0,
        "y": 2.0,
        "z": -3.0,
        "vx": 0.1,
        "vy": 0.2,
        "vz": -0.3,
        "collided": False,
        "collision_object": "",
    }


def test_capture_observation_reuses_depth_until_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(airsim_adapter, "_require_airsim", lambda: _AirSimPlanar)
    client = _Client()
    adapter = AirSimAdapter(client=client, vehicle_name="Drone1", depth_capture_interval=3)

    observations = [adapter.capture_observation() for _ in range(5)]

    image_calls = [call for call in client.calls if call[0] == "simGetImages"]
    assert [call[1][0] for call in image_calls] == [
        (_ImageTypePlanar.Scene, _ImageTypePlanar.DepthPlanar),
        (_ImageTypePlanar.Scene,),
        (_ImageTypePlanar.Scene,),
        (_ImageTypePlanar.Scene, _ImageTypePlanar.DepthPlanar),
        (_ImageTypePlanar.Scene,),
    ]
    assert observations[0].depth_m[0, 0] == pytest.approx(1.0)
    assert observations[2].depth_m[0, 0] == pytest.approx(1.0)
    assert observations[3].depth_m[0, 0] == pytest.approx(2.0)


def test_depth_capture_interval_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(airsim_adapter, "_require_airsim", lambda: _AirSimPlanar)

    with pytest.raises(ValueError, match="at least 1"):
        AirSimAdapter(client=_Client(), depth_capture_interval=0)
