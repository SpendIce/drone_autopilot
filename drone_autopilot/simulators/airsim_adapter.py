"""AirSim simulator adapter."""

from __future__ import annotations

import math
import time

import numpy as np

from ..exceptions import MissingOptionalDependencyError
from ..core_types import VelocityCommand
from .base import Observation, SimulatorAdapter


def _require_airsim():
    try:
        import airsim
    except ModuleNotFoundError as exc:
        raise MissingOptionalDependencyError("airsim", "sim") from exc
    return airsim


def _depth_image_type(airsim_module):  # type: ignore[no-untyped-def]
    image_type = airsim_module.ImageType
    if hasattr(image_type, "DepthPlanar"):
        return image_type.DepthPlanar
    if hasattr(image_type, "DepthPlanner"):
        return image_type.DepthPlanner
    raise RuntimeError("AirSim ImageType has neither DepthPlanar nor DepthPlanner")


class AirSimAdapter(SimulatorAdapter):
    """Capture RGB/depth and send body-frame velocity commands in AirSim."""

    def __init__(
        self,
        *,
        vehicle_name: str = "",
        scene_camera: str = "0",
        depth_camera: str = "0",
        client=None,
        invert_z: bool = False,
        hold_altitude: bool = False,
        wait_for_commands: bool = True,
        depth_capture_interval: int = 1,
    ) -> None:
        if depth_capture_interval < 1:
            raise ValueError("depth_capture_interval must be at least 1")
        self.airsim = _require_airsim()
        self.vehicle_name = vehicle_name
        self.scene_camera = scene_camera
        self.depth_camera = depth_camera
        self.client = client or self.airsim.MultirotorClient()
        self.invert_z = invert_z
        self.hold_altitude = hold_altitude
        self.wait_for_commands = wait_for_commands
        self.depth_capture_interval = depth_capture_interval
        self._hold_z_ned: float | None = None
        self._last_depth_m: np.ndarray | None = None
        self._capture_count = 0

    def connect(
        self,
        *,
        arm: bool = False,
        takeoff: bool = False,
        takeoff_altitude_m: float | None = None,
        takeoff_velocity_mps: float = 1.0,
    ) -> None:
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        if arm:
            self.client.armDisarm(True, vehicle_name=self.vehicle_name)
        if takeoff:
            self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        if takeoff_altitude_m is not None:
            z_ned = -abs(takeoff_altitude_m)
            self.client.moveToZAsync(
                z_ned,
                takeoff_velocity_mps,
                vehicle_name=self.vehicle_name,
            ).join()
            if self.hold_altitude:
                self._hold_z_ned = z_ned
        elif self.hold_altitude:
            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            self._hold_z_ned = float(state.kinematics_estimated.position.z_val)

    def capture_observation(self) -> Observation:
        should_capture_depth = (
            self._last_depth_m is None
            or self.depth_capture_interval == 1
            or self._capture_count % self.depth_capture_interval == 0
        )
        requests = [
            self.airsim.ImageRequest(self.scene_camera, self.airsim.ImageType.Scene, False, False),
        ]
        if should_capture_depth:
            requests.append(self.airsim.ImageRequest(self.depth_camera, _depth_image_type(self.airsim), True, False))
        responses = self.client.simGetImages(requests, vehicle_name=self.vehicle_name)
        scene_response = responses[0]
        rgb = self._scene_response_to_rgb(scene_response)
        if should_capture_depth:
            self._last_depth_m = self._depth_response_to_array(responses[1])
        if self._last_depth_m is None:
            raise RuntimeError("AirSim depth capture did not produce a reusable depth frame")
        depth = self._last_depth_m
        self._capture_count += 1
        timestamp = getattr(scene_response, "time_stamp", None)
        return Observation(rgb=rgb, depth_m=depth, timestamp=float(timestamp) if timestamp else None)

    def send_velocity(self, command: VelocityCommand, *, duration_s: float) -> None:
        yaw_deg_s = math.degrees(command.yaw_rate)
        yaw_mode = self.airsim.YawMode(is_rate=True, yaw_or_rate=yaw_deg_s)
        if self._hold_z_ned is not None:
            task = self.client.moveByVelocityZBodyFrameAsync(
                command.vx,
                command.vy,
                self._hold_z_ned,
                duration_s,
                yaw_mode=yaw_mode,
                vehicle_name=self.vehicle_name,
            )
            if self.wait_for_commands:
                task.join()
            return
        vz = -command.vz if self.invert_z else command.vz
        task = self.client.moveByVelocityBodyFrameAsync(
            command.vx,
            command.vy,
            vz,
            duration_s,
            yaw_mode=yaw_mode,
            vehicle_name=self.vehicle_name,
        )
        if self.wait_for_commands:
            task.join()

    def hover(self, *, duration_s: float) -> None:
        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
        if duration_s > 0.0:
            time.sleep(duration_s)

    def capture_state(self) -> dict[str, object]:
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        position = state.kinematics_estimated.position
        velocity = state.kinematics_estimated.linear_velocity
        collision = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
        return {
            "x": float(position.x_val),
            "y": float(position.y_val),
            "z": float(position.z_val),
            "vx": float(velocity.x_val),
            "vy": float(velocity.y_val),
            "vz": float(velocity.z_val),
            "collided": bool(collision.has_collided),
            "collision_object": str(collision.object_name),
        }

    def close(self) -> None:
        self.client.enableApiControl(False, vehicle_name=self.vehicle_name)

    def _scene_response_to_rgb(self, response) -> np.ndarray:  # type: ignore[no-untyped-def]
        if response.height <= 0 or response.width <= 0:
            raise RuntimeError("AirSim returned an empty scene image")
        array = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        return array.reshape(response.height, response.width, 3)

    def _depth_response_to_array(self, response) -> np.ndarray:  # type: ignore[no-untyped-def]
        if response.height <= 0 or response.width <= 0:
            raise RuntimeError("AirSim returned an empty depth image")
        return np.asarray(response.image_data_float, dtype=np.float32).reshape(response.height, response.width)
