"""AirSim simulator adapter."""

from __future__ import annotations

import math

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
    ) -> None:
        self.airsim = _require_airsim()
        self.vehicle_name = vehicle_name
        self.scene_camera = scene_camera
        self.depth_camera = depth_camera
        self.client = client or self.airsim.MultirotorClient()
        self.invert_z = invert_z

    def connect(self, *, arm: bool = False, takeoff: bool = False) -> None:
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        if arm:
            self.client.armDisarm(True, vehicle_name=self.vehicle_name)
        if takeoff:
            self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()

    def capture_observation(self) -> Observation:
        requests = [
            self.airsim.ImageRequest(self.scene_camera, self.airsim.ImageType.Scene, False, False),
            self.airsim.ImageRequest(self.depth_camera, self.airsim.ImageType.DepthPlanner, True, False),
        ]
        scene_response, depth_response = self.client.simGetImages(requests, vehicle_name=self.vehicle_name)
        rgb = self._scene_response_to_rgb(scene_response)
        depth = self._depth_response_to_array(depth_response)
        timestamp = getattr(scene_response, "time_stamp", None)
        return Observation(rgb=rgb, depth_m=depth, timestamp=float(timestamp) if timestamp else None)

    def send_velocity(self, command: VelocityCommand, *, duration_s: float) -> None:
        vz = -command.vz if self.invert_z else command.vz
        yaw_deg_s = math.degrees(command.yaw_rate)
        yaw_mode = self.airsim.YawMode(is_rate=True, yaw_or_rate=yaw_deg_s)
        self.client.moveByVelocityBodyFrameAsync(
            command.vx,
            command.vy,
            vz,
            duration_s,
            yaw_mode=yaw_mode,
            vehicle_name=self.vehicle_name,
        ).join()

    def hover(self, *, duration_s: float) -> None:
        self.send_velocity(VelocityCommand.hover(), duration_s=duration_s)

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
