# Simulator-first RGB-D Drone Pilot

This project is a Python scaffold for a reactive drone pilot:

`RGB + depth -> safe body-frame velocity command`

The first target is closed-loop simulation. Real drone use is intentionally gated behind a safety checklist and should only send high-level velocity setpoints through a future adapter.

## Dataset strategy

The local AirSim RGB/depth/commands dataset is treated as the supervised seed dataset because it has exact `[vx, vy, vz, yaw_rate]` labels. The manifest builder converts yaw rate from degrees per second to radians per second and stores commands in the internal standard:

- `vx`, `vy`, `vz`: meters per second
- `yaw_rate`: radians per second
- `action_frame`: `body`

Mid-Air and DDOS are intended as later manifest sources: Mid-Air can contribute pseudo-actions from pose deltas, while DDOS is better used for depth/perception robustness or auxiliary tasks.

## Quick start

Run commands from the project root, not from inside `drone_autopilot/`:

```bash
cd /home/spendice/Documents/Archivos_Facu/IA
```

Build a manifest for the local AirSim seed dataset:

```bash
python3 -m drone_autopilot.cli build-airsim-manifest \
  data_collected_potential_final_v58_mod25_320x320_cmds
```

Inspect action statistics:

```bash
python3 -m drone_autopilot.cli stats \
  data_collected_potential_final_v58_mod25_320x320_cmds/manifest.parquet
```

Install training dependencies before model training:

```bash
python3 -m pip install -e ".[training,dev]"
```

Train a small RGB-D pilot:

```bash
python3 -m drone_autopilot.cli train \
  data_collected_potential_final_v58_mod25_320x320_cmds/manifest.parquet \
  --data-root data_collected_potential_final_v58_mod25_320x320_cmds \
  --epochs 10 \
  --backbone mobilenet_v3_small
```

The training code masks unavailable labels, normalizes actions from train-split statistics only, and reports per-output plus per-source metrics.

## Safety model

The simulator adapter always runs predictions through `SafetyFilter` before a command is sent. The filter:

- rejects NaN or infinite predictions
- clamps velocities and yaw rate
- smooths successive commands
- switches to hover/stop when depth indicates a close obstacle

The AirSim adapter is optional and imported only when used. If `airsim` is not installed, CLI simulator commands fail with a clear dependency message instead of breaking manifest and test workflows.

## Real drone policy

Real drones are out of scope for v1. A future MAVSDK/MAVLink/ROS adapter must keep these gates:

- simulator validation first
- manual override
- geofence
- bench tests
- low-speed prop-guard tests
- safety pilot
- high-level velocity setpoints only

## Verification

Local checks that do not require PyTorch:

```bash
python3 -m pytest
```

Optional syntax check:

```bash
python3 -m compileall drone_autopilot tests
```
