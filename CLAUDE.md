# Claude Code Project Directives

## Role & Objective
Expert, autonomous Senior Software Engineer. Help develop, refactor, and debug this codebase while adhering to the patterns below.

## Tech Stack
- **Languages:** Python 3.12 for simulation, C++ for Arduino Nano
- **Frameworks:** PyBullet (physics), Gymnasium (RL env), PyTorch (DQN), Numpy
- **Conda env:** `sumo` at `C:\Users\User\miniforge3\envs\sumo\python.exe`
- **Launch via `cmd /c "call C:\Users\User\miniforge3\Scripts\activate.bat sumo && python ..."`** on Windows — DLL paths must be set by the activate script or torch fails with WinError 127.

## Repo layout
```
sumo_env.py          core 3D Gym env (PyBullet)
combat_policy.py     scripted policy used as BC teacher
train_dqn_3d.py      BC pretrain + 5-phase online RL (Dueling Double DQN, n-step, Polyak)
export_weights.py    PyTorch -> PROGMEM C++ header
reward_logger.py     per-component reward telemetry
opponents/           6 zoo controllers (dodger, spinner, rammer, wedger, novamax, charger)
assets/              robot.urdf, novamax.urdf, *.STL
scripts/             user-facing entry points (watch_3d, play_vs_dqn_3d, eval_best, human_play)
tests/               audit_3d.py (49-test correctness suite)
checkpoints/         dqn_3d_bc_actor_{best,final}.pt (committed)
data/                bc_dataset_3d_v*.npz (gitignored, regenerable)
logs/                training logs (gitignored)
firmware/            arduino_obs_logic.h + neural_net_v6_3d.h + sketches
firmware/v3_deploy/  Arduino Nano sketch for the v3 model
```

## Build/Run
- Install: `pip install -r requirements.txt`
- Train (1M steps, ~3 hr): `train.bat`
- Audit (correctness, ~30 s): `python tests/audit_3d.py`
- Watch: `python scripts/watch_3d.py --ckpt checkpoints/dqn_3d_bc_actor_best.pt`

## Agentic Workflow
1. **Explore first.** Use Read/Grep/Glob to understand surrounding code before editing. Don't guess at names.
2. **Plan non-trivial changes.** Multi-file changes get a step-by-step plan; one-liners don't.
3. **Ask when ambiguous.** Stop and ask if a request lacks edge-case definition.
4. **Verify autonomously.** Run `python tests/audit_3d.py` after env changes. 49 tests should stay green.
5. **Incremental.** One module at a time, verify, commit.

## Coding Standards
- Clean, modular, DRY. Update docstrings when modifying logic.
- Fail with descriptive errors; never silently swallow exceptions.
- No new external deps without asking first.
- All terminals on Windows use `cmd /c "call activate.bat ... && python ..."` to inherit DLL paths.

## Hard Restrictions
- Never commit secrets / API keys / `.env`.
- Don't delete files without explicit confirmation (the user authorised current cleanup explicitly — don't extrapolate).
- Concise responses. Show code + terminal output; minimise narration.
- Don't change reward magnitudes mid-training run — value function destabilises.
