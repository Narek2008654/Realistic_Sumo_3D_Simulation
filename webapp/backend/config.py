"""Repo-root-relative filesystem paths for the LITE backend.

No configuration files, no env vars, no DB — every path is derived from the
repository root so the app works from any working directory. The ``data/``
sub-trees are created lazily on first use.
"""

from __future__ import annotations

from pathlib import Path

# webapp/backend/config.py -> repo root is three parents up.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

CHECKPOINTS: Path = REPO_ROOT / "checkpoints"
DATA: Path = REPO_ROOT / "data"

# Per-model registry cards live here, one ``<id>.json`` per checkpoint.
REGISTRY_DIR: Path = DATA / "registry"

# Job artifacts (trajectories, etc.) live under ``data/jobs/<job>/``.
JOBS_DIR: Path = DATA / "jobs"


def ensure_dirs() -> None:
    """Create the writable data sub-trees if they don't already exist."""
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
