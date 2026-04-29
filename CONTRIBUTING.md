# Contributing

Thanks for your interest in improving the project. PRs and issues welcome.

## Dev setup

```bash
conda create -n sumo -c conda-forge python=3.12 pybullet numpy gymnasium -y
conda activate sumo
pip install stable-baselines3 torch==2.5.1
```

## Workflow

1. Fork and create a topic branch off `main`.
2. Make your change. Keep diffs focused — one concern per PR.
3. Smoke-test: `python main.py` (random-action env check) and, if you touched the policy or training, `python test_policy.py`.
4. Open a PR describing the motivation, the change, and how you tested it.

## Code style

- Python 3.11+, type hints where they help readability.
- Google-style docstrings on public functions.
- No new runtime dependencies without discussion.
- Don't commit training artifacts (`checkpoints/`, `*.npz`, `timings_*.json`) — `.gitignore` already excludes them.

## Reporting bugs

Open an issue with: PyBullet/torch/Python versions, OS, the command you ran, and the full traceback.
