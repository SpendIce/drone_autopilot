"""Second expert-avoidance recording campaign, after fixing three SafetyFilter
bugs (permanent hover freeze, blended-goal diluting yaw, blended-goal diluting
retreat) and redesigning ReactiveAvoidancePolicy to turn earlier/harder with
retreat as a last resort.

Reuses the same start-pose x heading grid as the first campaign
(record_expert_campaign.py) minus the diagonal heading (kept the campaign
under a tighter time budget), plus repeated attempts at the deliberately
obstructed route that motivated the safety-filter fixes — its outcome is
probabilistic (AirSim sometimes lets the drone graze past, sometimes wedges
it on contact), so multiple attempts are worth it even though not all will
complete cleanly. Frames accumulate into a new output dir (kept separate from
the first campaign's runs/expert_dataset) so the two can be inspected/merged
independently.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "runs" / "expert_dataset_v2"
SUMMARY_PATH = REPO_ROOT / "runs" / "expert_campaign_v2_summary.jsonl"

START_POINTS = [
    (-10.0, -30.0),
    (6.0, -30.0),
    (22.0, -30.0),
    (-10.0, -14.0),
    (6.0, -14.0),
    (22.0, -14.0),
    (-10.0, 2.0),
    (6.0, 2.0),
]

HEADINGS = [
    (10.0, 0.0),
    (0.0, -10.0),
]

Z = -3.0
STEPS = 135
DEPTH_INTERVAL = 2

# The obstructed route: (10,-14) -> (34,-14), straight line blocked by
# TemplateCube_Rounded_77. Repeated because outcome is probabilistic.
OBSTRUCTED_REPEATS = 3


def build_grid_episodes() -> list[dict]:
    episodes = []
    for sx, sy in START_POINTS:
        for dx, dy in HEADINGS:
            wx, wy = sx + dx, sy + dy
            yaw_deg = math.degrees(math.atan2(dy, dx))
            episodes.append(
                {
                    "start_x": sx,
                    "start_y": sy,
                    "start_yaw_deg": yaw_deg,
                    "waypoint": f"{wx},{wy}",
                    "steps": STEPS,
                }
            )
    return episodes


def build_obstructed_episodes() -> list[dict]:
    return [
        {
            "start_x": 10.0,
            "start_y": -14.0,
            "start_yaw_deg": 0.0,
            "waypoint": "34.0,-14.0",
            "steps": 250,
        }
        for _ in range(OBSTRUCTED_REPEATS)
    ]


def run_episode(python_bin: str, episode: dict, index: int, total: int) -> dict:
    cmd = [
        python_bin,
        "-m",
        "drone_autopilot.cli",
        "record-expert",
        "--output-dir",
        str(OUTPUT_DIR),
        "--steps",
        str(episode["steps"]),
        "--command-duration",
        "0.8",
        "--max-vx",
        "0.80",
        "--max-vy",
        "0.60",
        "--max-vz",
        "0.0",
        "--max-yaw-rate",
        "0.6",
        "--smoothing-alpha",
        "0.15",
        "--emergency-depth",
        "0.8",
        "--depth-roi-bottom",
        "0.55",
        "--arm",
        "--takeoff",
        "--takeoff-altitude",
        "3.0",
        "--hold-altitude",
        "--async-commands",
        "--depth-interval",
        str(DEPTH_INTERVAL),
        "--keep-api-control",
        "--start-x",
        str(episode["start_x"]),
        "--start-y",
        str(episode["start_y"]),
        "--start-z",
        str(Z),
        "--start-yaw-deg",
        str(episode["start_yaw_deg"]),
        "--waypoint",
        episode["waypoint"],
        "--mission-cruise-speed",
        "0.6",
        "--mission-waypoint-radius",
        "2.5",
        "--mission-position-blend",
        "0.5",
        "--mission-yaw-blend",
        "0.5",
        "--mission-max-yaw-rate",
        "0.3",
        # caution/retreat/urgency left at CLI defaults (match ReactiveAvoidanceConfig).
    ]
    print(
        f"[{index}/{total}] start=({episode['start_x']},{episode['start_y']}) "
        f"waypoint={episode['waypoint']} yaw={episode['start_yaw_deg']:.1f} "
        f"steps={episode['steps']}",
        flush=True,
    )
    started = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed = time.time() - started
    record = {"index": index, **episode, "elapsed_s": elapsed, "returncode": result.returncode}
    if result.returncode == 0:
        try:
            payload = json.loads(result.stdout[result.stdout.index("{") :])
            record.update(payload)
        except (ValueError, IndexError):
            record["parse_error"] = True
            record["stdout_tail"] = result.stdout[-2000:]
    else:
        record["stderr_tail"] = result.stderr[-2000:]
    return record


def main() -> int:
    python_bin = sys.executable
    episodes = build_grid_episodes() + build_obstructed_episodes()
    total = len(episodes)
    with SUMMARY_PATH.open("w", encoding="utf-8") as summary_file:
        for index, episode in enumerate(episodes, start=1):
            record = run_episode(python_bin, episode, index, total)
            summary_file.write(json.dumps(record) + "\n")
            summary_file.flush()
            print(f"  -> {record}", flush=True)
            if record["returncode"] != 0:
                print("  !! non-zero exit, stopping campaign", flush=True)
                return 1
    print("Campaign v2 complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
