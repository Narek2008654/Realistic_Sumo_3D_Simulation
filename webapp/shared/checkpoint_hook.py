"""Periodic checkpoint / eval / trajectory hook for the trainers (E1f).

Called from inside a trainer's main loop ONLY when a ``TrainingConfig`` is
active (i.e. ``SUMO_RUN_CONFIG`` was set). :func:`maybe_fire` snapshots the
live net every ``cfg.eval_every`` steps and launches a DETACHED subprocess
(:mod:`webapp.shared.eval_and_record`) that evaluates the snapshot and
records a trajectory — so training never blocks on eval.

Design mirrors ``train_ppo_3d.launch_watch``: save a checkpoint atomically,
then ``subprocess.Popen`` a separate process. To keep CUDA/MKL DLL paths
correct on Windows, the subprocess re-activates the conda env via a tiny
temp ``.bat`` (the inline ``cmd /c "call ... && python"`` form is flaky
through the tooling). On non-Windows the python interpreter is invoked
directly.

The whole body is wrapped in try/except: a failure here must NEVER take down
a multi-hour training run. On any error we log and return the state
unchanged.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _save_snapshot_atomic(net, path: Path) -> None:
    """Atomic torch.save (temp + os.replace) so a concurrent eval reader
    never sees a half-written file (Windows torch.save isn't atomic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(net.state_dict(), str(tmp))
    os.replace(str(tmp), str(path))


def _launch_eval(snapshot: Path, cfg, step: int):
    """Spawn the detached eval/record subprocess; return the Popen handle.

    The args JSON is materialised next to the snapshot so the child process
    is fully self-describing. On Windows we route through a temp ``.bat`` so
    the conda activate script sets the DLL search path before python starts.
    """
    job_dir = Path(cfg.job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    # Eval against EVERY opponent in the training mix — built-in zoo ids AND
    # custom (user-authored) ids. ``eval_and_record`` resolves a custom id via
    # opponents_store and fights it on its own chassis (skipping any id it can't
    # resolve), so the per-opponent win-rates cover the full mix. Falls back to
    # novamax when no mix is set (the CLI/default path).
    if cfg.opponent_weights:
        # Only opponents actually in the mix (positive weight) — a zero-weight id
        # (e.g. held-out feinter/orbiter) isn't sampled in training, so don't
        # spend eval wall-clock on it.
        opponents = [o for o, w in cfg.opponent_weights.items() if w > 0] or ["novamax"]
    else:
        opponents = ["novamax"]
    hw = cfg.hardware_spec.to_dict() if cfg.hardware_spec is not None else None
    args = {
        "snapshot": str(snapshot),
        "job_dir": str(job_dir),
        "step": int(step),
        "opponents": opponents,
        "mult": 3.0,
        "n_eval": 5 if os.environ.get("DQN_SMOKE") or os.environ.get("PPO_SMOKE")
        else 10,
        "seed": int(cfg.seed),
        "hardware_spec": hw,
    }
    args_path = job_dir / "snapshots" / f"eval_args_{step}.json"
    args_path.parent.mkdir(parents=True, exist_ok=True)
    args_path.write_text(json.dumps(args), encoding="utf-8")

    module = "webapp.shared.eval_and_record"
    if os.name == "nt":
        activate = r"C:\Users\User\miniforge3\Scripts\activate.bat"
        bat = job_dir / "snapshots" / f"eval_{step}.bat"
        bat.write_text(
            f'@echo off\r\n'
            f'call "{activate}" sumo\r\n'
            f'set KMP_DUPLICATE_LIB_OK=TRUE\r\n'
            f'cd /d "{_REPO_ROOT}"\r\n'
            f'python -m {module} "{args_path}"\r\n',
            encoding="utf-8",
        )
        cmd = ["cmd", "/c", str(bat)]
    else:
        cmd = [sys.executable, "-m", module, str(args_path)]

    proc = subprocess.Popen(
        cmd, cwd=str(_REPO_ROOT),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return proc


def maybe_fire(net, step, cfg, last_eval_step, eval_proc):
    """Maybe snapshot + launch a detached eval/record subprocess.

    Fires when ``step - last_eval_step >= cfg.eval_every``. If the previous
    ``eval_proc`` is still running, SKIP launching a new one (the snapshot is
    still saved, and ``last_eval_step`` advances so we don't thrash). Never
    raises into the caller — any error is logged and the state is returned
    unchanged.

    Returns ``(new_last_eval_step, eval_proc)``.
    """
    try:
        if cfg is None or cfg.job_dir is None:
            return last_eval_step, eval_proc
        if step - last_eval_step < cfg.eval_every:
            return last_eval_step, eval_proc

        snapshot = Path(cfg.job_dir) / "snapshots" / f"ckpt_{step}.pt"
        _save_snapshot_atomic(net, snapshot)

        if eval_proc is not None and eval_proc.poll() is None:
            print(f"  [hook] step {step}: previous eval still running — "
                  f"snapshot saved, skipping new eval launch", flush=True)
            return step, eval_proc

        proc = _launch_eval(snapshot, cfg, step)
        print(f"  [hook] step {step}: snapshot {snapshot.name} + eval "
              f"subprocess launched (pid {proc.pid})", flush=True)
        return step, proc
    except Exception as exc:  # never break the training loop
        print(f"  [hook] step {step}: checkpoint hook error (ignored): "
              f"{exc!r}", flush=True)
        return last_eval_step, eval_proc
