"""Safety gates for a future real-drone adapter.

This module intentionally does not send MAVSDK, MAVLink, or ROS commands.
It records the minimum gates that must be satisfied before any real vehicle
adapter is allowed to arm or send high-level velocity setpoints.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RealDroneSafetyChecklist:
    simulator_validated: bool = False
    manual_override_ready: bool = False
    geofence_configured: bool = False
    bench_test_passed: bool = False
    prop_guard_low_speed_test_passed: bool = False
    safety_pilot_present: bool = False
    high_level_setpoints_only: bool = True

    def missing_items(self) -> list[str]:
        checks = {
            "simulator_validated": self.simulator_validated,
            "manual_override_ready": self.manual_override_ready,
            "geofence_configured": self.geofence_configured,
            "bench_test_passed": self.bench_test_passed,
            "prop_guard_low_speed_test_passed": self.prop_guard_low_speed_test_passed,
            "safety_pilot_present": self.safety_pilot_present,
            "high_level_setpoints_only": self.high_level_setpoints_only,
        }
        return [name for name, passed in checks.items() if not passed]

    def assert_ready(self) -> None:
        missing = self.missing_items()
        if missing:
            raise RuntimeError(f"Real-drone adapter is not enabled. Missing gates: {', '.join(missing)}")
