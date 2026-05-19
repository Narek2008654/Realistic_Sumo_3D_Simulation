# Contributing

Thanks for your interest in improving the project. PRs and issues welcome.

## Dev setup

```bash
conda create -n sumo -c conda-forge python=3.12 pybullet numpy gymnasium -y
conda activate sumo
pip install torch==2.5.1
```

Windows: launch via `cmd /c "call activate.bat sumo && python ..."` so PyTorch's
DLL paths inherit from the conda env (otherwise you'll hit `fbgemm.dll`
`WinError 127`).

## Workflow

1. Fork and create a topic branch off `main`.
2. Make your change. Keep diffs focused — one concern per PR.
3. Smoke-test:
   - **Env changes:** `python tests/audit_3d.py` — 49 tests, exit code 0 if green.
   - **Policy / training changes:** quick CombatPolicy sanity at `python scripts/watch_combat_3d.py --opp novamax --mult 1.0` and an `eval_best.py` run on the new checkpoint.
4. Open a PR describing the motivation, the change, and how you tested it.

## Code style

- Python 3.12+, type hints where they help readability.
- Google-style docstrings on public functions.
- No new runtime dependencies without discussion.
- Don't commit training artifacts (`data/*.npz`, `logs/`, `timings_*.json`) — `.gitignore` already excludes them. `checkpoints/dqn_3d_bc_actor_best.pt` is the one weight committed deliberately so the firmware export is reproducible from the repo.

## Reporting bugs

Open an issue with: PyBullet / torch / Python versions, OS, the command you ran, and the full traceback.
