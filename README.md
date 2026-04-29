# Realistic Sumo 3D Simulation

[![CI](https://github.com/USERNAME/Realistic_Sumo_3D_Simulation/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/Realistic_Sumo_3D_Simulation/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

PyBullet-based mini-sumo arena for training a Reinforcement Learning policy that deploys to an Arduino Nano. The agent learns to push a hardcoded `davo_sirad` opponent off a 70 cm dohyo, then its weights are exported as a PROGMEM C++ header for on-device inference.

## What's in the repo

| File | Purpose |
|------|---------|
| `sumo_env.py` | Gymnasium env: PyBullet world, robot URDF loading, sensor raycasts, reward shaping, davo_sirad enemy controller. |
| `robot.urdf` | Mini-sumo robot: 75×100×50 mm chassis, 25 mm front wedge (21° scoop), two driven rear wheels, frictionless front caster. |
| `train.py` | SAC trainer with a 2.0× → 2.5× → 3.0× enemy-torque curriculum, periodic checkpoints, best-model tracking. Exports `neural_net.h` at the end. |
| `test_policy.py` | Loads `sac_actor.zip` and runs deterministic episodes in the GUI; supports keyboard-controlled blue robot for human-vs-AI play. |
| `main.py` | Random-action smoke test for the env. |
| `neural_net.h` | Auto-generated PROGMEM weights + `predict_motors(...)` C++ forward pass. Drop into the Arduino sketch. |

## Architecture

* **Observation (4D):** `[front_norm, left_norm, right_norm, last_seen_dir]` — three normalised laser distances and the last side that picked up the opponent.
* **Action (2D, continuous):** `[left_motor, right_motor]` in [-1, +1].
* **Network:** SAC, MLP `[32, 32]` ReLU + tanh on the actor mean.
* **Domain randomisation:** dohyo friction ±20 %, robot mass ±10 %, motor slop ±15 %, 20 % chance one sensor is dead per episode.

## Setup

The codebase needs PyBullet (which is finicky on Windows + Python 3.13). Easiest path is conda-forge:

```powershell
conda create -n sumo -c conda-forge python=3.12 pybullet numpy gymnasium -y
conda activate sumo
pip install stable-baselines3 torch==2.5.1
```

Windows-specific quirks (already handled by the scripts, but FYI):
* `KMP_DUPLICATE_LIB_OK=TRUE` to silence the OpenMP runtime conflict between MKL and torch.
* `import torch` **before** `import pybullet` to avoid `fbgemm.dll` load failures.

## Run

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# Train (3 phases × 200k steps = 600k total, ~2 hours).
python train.py

# Watch the trained policy fight davo_sirad.
python test_policy.py
```

Training writes `sac_actor.zip`, `sac_actor_best.zip`, `neural_net.h`, `sac_actor_weights.npz`, and a `checkpoints/` directory with one snapshot every 25k steps.

## Human-vs-AI mode

`test_policy.py` is currently configured to let you drive the blue robot:

* **W** / **↑** — forward
* **S** / **↓** — reverse
* **A** / **←** — pivot left
* **D** / **→** — pivot right

Click the PyBullet window first so it captures keyboard events.

## Deploying to Arduino

1. Drop `neural_net.h` into the Arduino sketch (next to `davo_sirad.ino` style firmware).
2. Compute the same 4 normalised inputs from the Arduino's three VL53L1X sensors.
3. Call `mini_sumo_ai::predict_motors(front, left, right, last_seen, out_left, out_right)`.
4. Map `out_left`, `out_right` (in [-1, +1]) to PWM (typically `int pwm = (int)(out * 255.0f)`).

## Best results

After 600k SAC steps with the torque curriculum, deterministic eval gave the **final 600k checkpoint an 81 % win-rate against davo_sirad at 3.0× torque** (100-episode evaluation). Training-time rolling win-rate looked lower (~40 %) because SAC keeps Gaussian exploration noise on actions during rollout.

## Author

Narek Stepanyan

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
