from __future__ import annotations

import pytest

from drone_autopilot.real_drone import RealDroneSafetyChecklist


def test_real_drone_gate_blocks_until_all_safety_items_are_true() -> None:
    checklist = RealDroneSafetyChecklist()

    with pytest.raises(RuntimeError, match="simulator_validated"):
        checklist.assert_ready()

    ready = RealDroneSafetyChecklist(
        simulator_validated=True,
        manual_override_ready=True,
        geofence_configured=True,
        bench_test_passed=True,
        prop_guard_low_speed_test_passed=True,
        safety_pilot_present=True,
        high_level_setpoints_only=True,
    )
    ready.assert_ready()
