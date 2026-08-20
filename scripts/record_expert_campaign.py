"""Generate and run a batch of expert-avoidance recording episodes in AirSim.

Grid of start poses x headings inside FlyingExampleMap (Blocks), yaw pointed
toward each episode's waypoint so the RGB-D camera faces the direction of
travel. Frames accumulate into a single output dataset dir (frame ids continue
across episodes), matching the layout build_airsim_seed_manifest expects.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "runs" / "expert_dataset"
SUMMARY_PATH = REPO_ROOT / "runs" / "expert_campaign_summary.jsonl"

# (start_x, start_y) grid, meters, AirSim NED x/y.
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

# (dx, dy) heading offsets, meters, applied per start point.
HEADINGS = [
    (10.0, 0.0),
    (0.0, -10.0),
    (7.0, -7.0),
]

Z = -3.0
STEPS = 135


def build_episodes() -> list[dict]:
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
                }
            )
    return episodes


def run_episode(python_bin: str, episode: dict, index: int, total: int) -> dict:
    cmd = [
        python_bin,
        "-m",
        "drone_autopilot.cli",
        "record-expert",
        "--output-dir",
        str(OUTPUT_DIR),
        "--steps",
        str(STEPS),
        "--command-duration",
        "0.8",
        "--max-vx",
        "0.50",
        "--max-vy",
        "0.40",
        "--max-vz",
        "0.0",
        "--max-yaw-rate",
        "0.5",
        "--smoothing-alpha",
        "0.12",
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
        "5",
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
        "0.50",
        "--mission-waypoint-radius",
        "2.5",
        "--mission-position-blend",
        "0.6",
        "--mission-yaw-blend",
        "0.6",
        "--mission-max-yaw-rate",
        "0.25",
        "--cruise-vx",
        "0.6",
        "--caution-depth",
        "3.0",
        "--min-forward-depth",
        "1.2",
        "--max-lateral-vy",
        "0.5",
        "--max-avoid-yaw-rate",
        "0.4",
    ]
    print(f"[{index}/{total}] start=({episode['start_x']},{episode['start_y']}) "
          f"waypoint={episode['waypoint']} yaw={episode['start_yaw_deg']:.1f}", flush=True)
    started = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed = time.time() - started
    record = {"index": index, **episode, "elapsed_s": elapsed, "returncode": result.returncode}
    if result.returncode == 0:
        try:
            payload = json.loads(result.stdout.strip().splitlines()[-1])
            record.update(payload)
        except (ValueError, IndexError):
            record["parse_error"] = True
            record["stdout_tail"] = result.stdout[-2000:]
    else:
        record["stderr_tail"] = result.stderr[-2000:]
    return record


def main() -> int:
    python_bin = sys.executable
    episodes = build_episodes()
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
    print("Campaign complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
