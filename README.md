# Realistic Sumo 3D Simulation

PyBullet-based mini-sumo arena for training a reinforcement-learning policy that deploys to an Arduino Nano. The agent learns to push 6 different scripted opponents off a 70 cm dohyo, then its weights are exported as a PROGMEM C++ header for on-device inference.

![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)
![PyBullet](https://img.shields.io/badge/sim-PyBullet-orange.svg)
![PyTorch](https://img.shields.io/badge/RL-PyTorch%20DQN-EE4C2C.svg)
![Arduino](https://img.shields.io/badge/deploy-Arduino%20Nano-00979D.svg)

## What's in the repo

```
sumo_env.py           Gymnasium env: PyBullet world, IR raycasts, reward stack, DR
combat_policy.py      60-line scripted policy used as BC teacher
train_dqn_3d.py       BC pretrain + 5-phase online RL pipeline
export_weights.py     PyTorch state-dict -> PROGMEM C++ header
opponents/            6 zoo controllers (dodger, spinner, rammer, wedger, novamax, charger)
assets/               robot.urdf, novamax.urdf, STL meshes
scripts/              entry-point scripts (watch, play, eval, human data collect)
tests/audit_3d.py     49-test correctness suite (run after any env change)
checkpoints/          committed model weights
firmware/             Arduino headers + sketches
firmware/v3_deploy/   ready-to-flash Arduino Nano sketch for the v3 model
```

## Architecture

**Observation (12D):**

| idx | feature | range |
|---|---|---|
| 0-2 | front / left / right IR normalised distance | [0, 1] |
| 3 | `last_seen_dir` latch | {-1, 0, +1} |
| 4-5 | rear-left / rear-right line sensors over white border | {0, 1} |
| 6-7 | previous tick's raw motor command | [-1, +1] |
| 8 | engagement timer (consecutive ticks with front IR < 0.15) | [0, 1] |
| 9 | yaw-rate proxy (decayed L-R accumulator) | [-1, +1] |
| 10 | front-IR temporal delta | [-1, +1] |
| 11 | lateral-IR temporal delta | [-1, +1] |

**Action:** `Discrete(9)` — Cartesian product of `(-1, 0, +1)` on each wheel.

**Network:** Dueling Double DQN, MLP trunk `[48, 48]`, n-step returns (n=3), Polyak target (τ=0.005).

**Reward stack:** terminal split (win +10, push_loss -15, mutual_out -20, self_out **-50**, timeout -10) + dense shaping (approach, engage, wedge, anti-flicker, bearing-tracking, Narek action-conditioned).

**Domain randomisation per episode:** chassis mass ±5%, dohyo friction 0.4–0.7, motor deadzone 0.20–0.35, battery sag 85–100%, action latency 1 step, IR Gaussian noise σ=0.016 m, 1% per-sensor per-step dropout.

## Physics fixes (3D PyBullet)

The original env had two failure modes that prevented learning:
- **Orbital deadlock** — both bots tracked each other but never closed (rotational symmetry). Fixed: `CHASSIS_FRICTION` 0.4 → 0.05 + stuck-detector that injects a lateral kick after 8 ticks of stationary contact.
- **Push impotence** — even when contact happened, the agent couldn't transfer momentum. Fixed in three places:
  - `WHEEL_FRICTION` 1.0 → 2.0 (thrust margin over braking opponent)
  - `anisotropicFriction=[1.0, 0.3, 1.0]` on wheels (high in roll direction, low along axle — side pushes work)
  - Idle motors free-spin (force=0 when |cmd|<0.05) — eliminates the artificial 24 N active-brake that no real DC motor exhibits

See [tests/audit_3d.py](tests/audit_3d.py) — verifies these in `test_rear_push`, `test_side_push`, `test_idle_push`, `test_stuck_detector`.

## Setup

```powershell
conda create -n sumo -c conda-forge python=3.12 pybullet numpy gymnasium -y
conda activate sumo
pip install torch==2.5.1
```

Windows quirks:
- `KMP_DUPLICATE_LIB_OK=TRUE` to silence MKL/OpenMP conflict.
- Import `torch` before `pybullet` to avoid `fbgemm.dll` race.
- Launch via `cmd /c "call C:\Users\User\miniforge3\Scripts\activate.bat sumo && python ..."` so DLL paths are set.

## Run

```bash
# Train (1M steps, ~3 hr): BC pretrain + 5-phase online RL curriculum.
train.bat

# Sanity-check the env (~30 s, should print "PASS: 49  FAIL: 0").
python tests/audit_3d.py

# Watch the trained policy fight the zoo.
python scripts/watch_3d.py --ckpt checkpoints/dqn_3d_bc_actor_best.pt --mult 1.0

# Hardest matchup, full novamax torque.
python scripts/watch_3d.py --ckpt checkpoints/dqn_3d_bc_actor_best.pt --opp novamax --mult 3.0

# Play yourself (WASD).
python scripts/play_vs_dqn_3d.py

# Headless eval (no GUI, just numbers).
python scripts/eval_best.py --ckpt checkpoints/dqn_3d_bc_actor_best.pt --n-eps 30 --mult 3.0

# Deterministic "clean" world (no DR noise / dropout / deadzone).
python scripts/watch_3d.py --ckpt checkpoints/dqn_3d_bc_actor_best.pt --clean
```

## Training pipeline

1. **Phase 0a — collection** (~5 min): `combat_policy.py` plays 20–40k env-steps against each of 6 opponents at mult=1.0; only **winning** transitions are kept. ~38k pairs total cached to `data/bc_dataset_3d_v3.npz`.
2. **Phase 0b — BC pretrain** (~5 min): 20 epochs of MSE regression of the dueling-Q net's argmax onto CombatPolicy's discretised actions over the winners. Loss converges ~0.16.
3. **Phase 1-5 — online RL** (~3 hr): ε-greedy fine-tune across a torque curriculum 0.7 → 1.0 → 1.5 → 2.0 → 3.0, 100k / 200k / 200k / 200k / 300k = 1M steps. ε bumps back to 0.20 at every phase boundary.
4. **Export**: `_emit_dqn_header(BEST, firmware/neural_net_v6_3d.h)` writes the advantage head + LUT + self-test pairs as a single-file C++ header for AVR inference.

## Results (v3 BEST snapshot, eval at mult=3.0)

| opp | WR |
|---|---|
| dodger | 76% |
| spinner | 41% |
| novamax | 23% |
| rammer | 27% |
| wedger | 18% |
| **overall** | **40%** |

Compare to the SAC-era baseline (pre-physics-fixes): trackers (novamax/rammer/wedger) were all 0–10%. The combination of correct physics + BC-from-CombatPolicy warm start makes the trackers winnable.

## Deploying to Arduino Nano

The trained policy ships as a single C++ header (`firmware/neural_net_v6_3d.h`) containing weights in PROGMEM + a `predict_action()` forward pass. Copy that header next to the sketch and flash:

1. Open Arduino IDE.
2. File → Open → `firmware/v3_deploy/v3_deploy.ino`.
3. Tools → Board → Arduino Nano (ATmega328P, Old Bootloader if your bot uses CH340).
4. Plug in bot, select COM port, Upload.
5. Serial Monitor at 115200 — should print `model self-test mismatches: 0` then `ready (v3, 3D-trained)`.

**Hardware mapped in [firmware/v3_deploy/v3_deploy.ino](firmware/v3_deploy/v3_deploy.ino):**
- 3× VL53L0X ToF sensors (front / left / right), XSHUT on D2 / D3 / D4
- TB6612FNG motor driver: left A1/A2=D10/D9 PWM=D11, right B1/B2=D6/D7 PWM=D5, STBY=D8
- 2× QTR-1A reflectance sensors (rear-left = A0, rear-right = A1) for the white border ring

The sketch also implements a 1.5 s search-spin watchdog and a self-test that refuses to drive motors if the weights are corrupted in flash.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Author

Narek Stepanyan
