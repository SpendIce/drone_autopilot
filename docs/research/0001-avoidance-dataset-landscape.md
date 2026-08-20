# Research: RGB-D obstacle-avoidance dataset landscape (2026-08-20)

## Question

We have ~15.2k imitation-learning frames total:
- ~10k frames from the Kaggle seed dataset `lukpellant/droneflight-obs-avoidanceairsimrgbdepth10k-320x320` (AirSim, RGB + depth + velocity commands).
- ~5.2k frames self-recorded with our scripted expert policy (`drone_autopilot/expert_policy.py`) in AirSim's official "Blocks" map.

Manifest format required per frame (`drone_autopilot/manifest.py:75-81`):
```
rgb/<frame>.png
depth/<frame>.npy        # raw depth in meters, float32
commands/<frame>.npy     # [vx, vy, vz, yaw_rate_deg_s], degrees/s for yaw
```

We have ~26 free Kaggle GPU-hours this week. Question: is there a freely available dataset or AirSim map that gives a genuinely good return versus just training on what we already have?

## Findings

### 1. Published RGB-D drone datasets with action/control labels

**None of the well-known synthetic drone datasets found include continuous velocity/action labels alongside RGB+depth.** They are built for perception tasks (depth estimation, segmentation), not imitation learning.

| Dataset | Source | License | Size | RGB | Depth | Velocity/action labels | Notes |
|---|---|---|---|---|---|---|---|
| Mid-Air | [midair.ulg.ac.be](https://midair.ulg.ac.be/) ([download page](https://midair.ulg.ac.be/download.html)) | CC BY-NC-SA 4.0 | 420k+ frames, split across many per-trajectory archives ("sheer size", exact GB not published on the page) | Yes (JPEG, left/right/down cameras) | Yes (16-bit float, meters, lossless PNG) | **No** — trajectories come with IMU/GPS/attitude flight-record data, not drone-frame velocity commands in our schema | Two environments only: "Kite" (4 weather variants) and "PLE" (3 seasons) — still synthetic outdoor flight, but not AirSim-generated and not obstacle-avoidance-oriented (built for SLAM/localization benchmarking) |
| DDOS (Drone Depth and Obstacle Segmentation Dataset) | [huggingface.co/datasets/benediktkol/DDOS](https://huggingface.co/datasets/benediktkol/DDOS), paper [arxiv.org/abs/2312.12494](https://arxiv.org/html/2312.12494v1) | CC BY-NC 4.0 | 136 GB, 34k images (30k train / 2k val / 2k test) | Yes | Yes (uint16 PNG, 0–100 m linear range — needs rescale to float32 meters) | **No** — segmentation-focused (10 obstacle classes) + depth + optical flow + normals, no control labels | **Is** generated with AirSim, which is promising for visual-domain match, but the HF card doesn't name which map(s); no velocity data means it cannot be used for imitation learning without inventing pseudo-labels (defeats the purpose) |
| `ziya07/uav-autonomous-navigation-dataset` (Kaggle) | [kaggle.com/datasets/ziya07/uav-autonomous-navigation-dataset](https://www.kaggle.com/datasets/ziya07/uav-autonomous-navigation-dataset) | CC0 | **609,499 bytes total** (confirmed via Kaggle public API `totalBytesNullable`) | Unverified but implausible | Unverified but implausible | Unverified | At ~0.6 MB total, this cannot contain any meaningful volume of RGB+depth imagery — ruled out without further inspection |
| EuRoC MAV | ETH ASL | BSD-ish/academic | ~11 sequences, real-world | Yes (real, greyscale stereo) | No dense depth (sparse via IMU/vicon) | No velocity commands, only ground-truth pose | Real-world (not AirSim), stereo greyscale not RGB, sensor-suite mismatch (MAV computer/rig, not our drone model), no continuous velocity action labels — ruled out on format grounds without needing deeper investigation |

**Conclusion for Q1**: no candidate provides RGB + depth + continuous velocity/yaw-rate labels. Every option that has action-like data (IMU/pose/flight-record) uses a different schema (attitude/GPS logs, not commanded body-frame velocities), and every option that has depth+RGB in an AirSim-native format (DDOS) drops action labels entirely. Bridging that gap means either (a) running a policy/heuristic over someone else's pre-recorded frame sequence to invent plausible `[vx,vy,vz,yaw_rate]` labels post-hoc, which is not label recovery but label fabrication and would inject noise directly into the imitation target, or (b) using the frames for an unsupervised/self-supervised pretraining stage unrelated to the current imitation-learning manifest — out of scope for "complement our IL dataset."

### 2. Additional free AirSim map packages (classic OSS AirSim API)

Confirmed directly from the GitHub Releases API (`api.github.com/repos/microsoft/AirSim/releases`), release tag `v1.8.1` (named "v1.8.1 - Linux", published 2022-07-18) — this is the `microsoft/AirSim` classic open-source repo, i.e. the `airsim` Python package / `moveByVelocityBodyFrameAsync`-style API, **not** the newer incompatible "Project AirSim" product (which lives in a separate `CodexLabs*`/`iRobot` ecosystem, not `microsoft/AirSim`).

Linux binaries available in `v1.8.1`, with exact asset sizes from the release API:

| Environment | Zip size |
|---|---|
| Blocks (already used) | 0.14 GB |
| AbandonedPark | 1.66 GB |
| Africa_Savannah | 1.11 GB |
| AirSimNH (small urban neighborhood) | 2.11 GB |
| Building_99 | (listed, size not captured — small) |
| LandscapeMountains | 1.15 GB |
| MSBuild2018 (soccer field) | 0.79 GB |
| TrapCamera (split archive, .001/.002) | 2.10 GB + 1.75 GB |
| ZhangJiajie | 0.88 GB |

Source: [github.com/microsoft/AirSim/releases](https://github.com/microsoft/AirSim/releases), tag `v1.8.1`.

These are legitimate, free, official binaries that work with the same `airsim` Python client our `expert_policy.py`/`record.py` already talk to — genuinely different visual/geometric content from Blocks (Africa_Savannah = outdoor natural terrain with vegetation/uneven ground, LandscapeMountains = open mountainous terrain, AirSimNH/AbandonedPark = structured urban/building obstacles). This is a real path to more diverse, correctly-labeled data, because recording in them reuses our own expert-policy pipeline and produces frames already in our exact manifest format — no conversion work at all.

**Caveat that matters for this decision**: recording in a new map requires running the AirSim/Unreal Engine binary interactively (a windowed/headless-rendered simulation) and driving it with our scripted policy in real time. That is a **local-machine or dedicated-VM task**, not something that fits inside a Kaggle notebook's headless GPU-hours allocation — Kaggle notebooks don't support running an Unreal Engine simulation binary with AirSim's TCP API in a normal session. So while this is the best "more diverse, correctly-labeled data" option, it does not consume the 26 Kaggle-hour budget at all; it would be a separate, non-Kaggle recording session (multi-GB download + local disk + wall-clock time to fly enough episodes, likely more than one afternoon of work for a modest frame count).

## Ranked short-list

1. **Not pursuing any external dataset for the imitation-learning manifest.** No candidate found (Mid-Air, DDOS, EuRoC, the Kaggle ziya07 listing) carries both depth and genuine velocity/action commands in a form convertible without fabricating labels.
2. **(Separate from the Kaggle budget) Recording more demonstrations in another official AirSim map** (Africa_Savannah or LandscapeMountains are the most visually distinct from Blocks) using the existing `expert_policy.py`/`record.py` pipeline — zero format-conversion cost, but it's a local recording task, not a Kaggle-GPU task.

## Recommendation

**Don't spend the 26 Kaggle GPU-hours chasing external datasets or new environments.** The honest read:

- Every dataset with the right visual modality (AirSim-native RGB+depth, like DDOS) lacks action labels; every dataset with action-like data (Mid-Air, EuRoC) isn't in our control schema and isn't AirSim-native. Converting either means synthesizing commands we don't actually have ground truth for — that's not "complementing" the dataset, it's injecting fabricated imitation targets, which is a real risk to model quality, not just wasted effort.
- The one genuine option for more diverse, correctly-labeled data — recording in another official AirSim map — is real and cheap in licensing/conversion terms, but it doesn't map onto a Kaggle notebook's GPU-hour budget at all (it needs an interactive simulator session, not headless training compute).
- So there is no candidate that is both (a) a genuine quality improvement and (b) a good use of this specific 26-hour Kaggle GPU budget.

Best use of the 26 Kaggle hours: **spend them training/tuning on the ~15.2k frames we already have** (seed Kaggle set + Blocks recordings). If more visual diversity is wanted later, treat "record in Africa_Savannah or LandscapeMountains" as a separate, local task — not a research or Kaggle-hour item.
