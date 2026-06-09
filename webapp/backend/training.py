"""Single-job training orchestration for the LITE backend (E1f launcher).

This is the launcher side of the E1f seam: it builds a
:class:`webapp.shared.run_config.TrainingConfig`, writes it to
``data/jobs/<job_id>/run_config.json``, and spawns the appropriate trainer
(``train_dqn_3d.py`` / ``train_ppo_3d.py``) as a DETACHED subprocess with
``SUMO_RUN_CONFIG`` pointing at that file. The trainer's periodic checkpoint
hook then writes ``progress.jsonl`` / ``trajectories/`` / ``snapshots/`` under
the job dir, which :func:`status` reads back.

LITE constraints: at most ONE active job at a time (a new start while one runs
is rejected). No DB — every fact about a job lives in files under its job dir
(``job.json`` meta, ``run_config.json``, ``console.log``, ``progress.jsonl``).

Windows: the trainer is launched through a temp ``.bat`` that re-activates the
conda env (so torch's DLL search path is correct) and sets
``KMP_DUPLICATE_LIB_OK``; the bat is deleted once the process has been
spawned. ``stop()`` kills the whole process tree via ``taskkill /T`` so the
detached eval subprocesses the trainer spawns die with it.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from webapp.backend import config, opponents_store, registry, robots_store
from webapp.shared.hardware_spec import HardwareSpec
from webapp.shared.run_config import TrainingConfig

__all__ = [
    "start_job",
    "status",
    "stop",
    "list_jobs",
    "active_job_id",
    "JobError",
    "JobBusyError",
]

_REPO_ROOT = config.REPO_ROOT
_ACTIVATE_BAT = r"C:\Users\User\miniforge3\Scripts\activate.bat"
_CONDA_ENV = "sumo"

_TRAINER_SCRIPT = {
    "dqn": "train_dqn_3d.py",
    "ppo": "train_ppo_3d.py",
}
_SMOKE_ENV = {
    "dqn": "DQN_SMOKE",
    "ppo": "PPO_SMOKE",
}


class JobError(Exception):
    """A training-job request could not be fulfilled (maps to 400)."""


class JobBusyError(JobError):
    """A job is already running (maps to 409)."""


# ---------------------------------------------------------------------------
# Single-job in-process state. Guarded by a lock so the (single-threaded by
# design, but FastAPI is threaded) start/stop paths can't race.
# ---------------------------------------------------------------------------
@dataclass
class _ActiveJob:
    job_id: str
    proc: subprocess.Popen
    algo: str
    mode: str
    job_dir: Path


_lock = threading.Lock()
_active: Optional[_ActiveJob] = None


def active_job_id() -> Optional[str]:
    """The id of the currently-running job, or ``None`` if idle.

    Prefers the in-process handle; falls back to a disk re-adopt so the active
    job is still found after a backend reload/restart.
    """
    with _lock:
        if _active is not None and _active.proc.poll() is None:
            return _active.job_id
    return _disk_running_job()


# ---------------------------------------------------------------------------
# Config building
# ---------------------------------------------------------------------------
def _resolve_hardware_spec(req: dict[str, Any]) -> HardwareSpec:
    """Resolve the job's :class:`HardwareSpec` from the request.

    Precedence: ``robot_id`` (a saved robot) -> inline ``hardware_spec`` dict
    -> the default spec. An unknown ``robot_id`` or an invalid inline spec is
    a :class:`JobError` (400).
    """
    robot_id = req.get("robot_id")
    if robot_id:
        record = robots_store.get_robot(robot_id)
        if record is None:
            raise JobError(f"unknown robot_id: {robot_id}")
        try:
            return HardwareSpec.from_dict(record["hardware_spec"])
        except (KeyError, TypeError, ValueError) as exc:
            raise JobError(f"saved robot has invalid spec: {exc}") from exc

    spec_dict = req.get("hardware_spec")
    if spec_dict is not None:
        if not isinstance(spec_dict, dict):
            raise JobError("hardware_spec must be an object")
        try:
            return HardwareSpec.from_dict(spec_dict)
        except (KeyError, TypeError, ValueError) as exc:
            raise JobError(f"invalid hardware_spec: {exc}") from exc

    return HardwareSpec.default()


def _resolve_resume_path(base_model_id: str) -> str:
    """Absolute path to the base model's ``.pt`` (finetune resume).

    ``base_model_id`` is a registry id (filename stem under ``checkpoints/``).
    Raises :class:`JobError` if the model is unknown.
    """
    card = registry.get_model(base_model_id)
    if card is None:
        raise JobError(f"unknown base_model_id: {base_model_id}")
    pt = config.CHECKPOINTS / card["filename"]
    if not pt.is_file():
        raise JobError(f"checkpoint file missing for {base_model_id}")
    return str(pt)


def _resolve_custom_opponents(
    opponent_weights: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve the custom (non-zoo) ids in ``opponent_weights`` to their saved
    records, returning ``[{id, behavior, behavior_dsl?, hardware_spec}, ...]``.

    ``behavior`` is the zoo|dsl object the trainer builds its controller from
    via the shared factory, so a "Heavy Dodger" (zoo behavior on a heavy
    chassis) is sampled with the dodger controller on the heavy body. For
    back-compat we also carry ``behavior_dsl`` for DSL behaviors so an older
    trainer build keeps working.

    A built-in zoo id is left for the trainer to sample normally. Any id that
    is neither a built-in nor a saved custom opponent raises :class:`JobError`
    (400). Zero-weight custom ids are skipped (they'd never be sampled and
    don't need a factory). Returns ``[]`` when there are no custom opponents,
    so the unset/zoo-only path is byte-identical.
    """
    if not opponent_weights:
        return []
    # Lazy import: the zoo registry pulls sumo_env, so keep it off module load.
    from opponents import OPPONENT_REGISTRY

    out: list[dict[str, Any]] = []
    for opp_id, weight in opponent_weights.items():
        if opp_id in OPPONENT_REGISTRY:
            continue  # built-in zoo opponent — the trainer samples it directly
        record = opponents_store.get_opponent(opp_id)
        if record is None:
            raise JobError(
                f"unknown opponent id in opponent_weights: {opp_id!r} "
                f"(not a built-in zoo opponent nor a saved custom opponent)"
            )
        try:
            w = float(weight)
        except (TypeError, ValueError):
            w = 0.0
        if w <= 0.0:
            continue  # never sampled — no factory needed
        behavior = opponents_store.normalize_behavior(record)
        entry: dict[str, Any] = {
            "id": record["id"],
            "behavior": behavior,
            "hardware_spec": record.get("hardware_spec"),
        }
        if behavior.get("kind") == "dsl":  # back-compat mirror for old trainers
            entry["behavior_dsl"] = behavior["dsl"]
        out.append(entry)
    return out


def _build_config(req: dict[str, Any], job_dir: Path) -> TrainingConfig:
    """Translate an API request into a :class:`TrainingConfig`.

    Validates ``algo`` / ``mode`` and resolves the hardware spec + resume
    path. Output checkpoints are written under the job dir so jobs never
    clobber the committed ``checkpoints/``.
    """
    algo = str(req.get("algo", "dqn")).lower()
    if algo not in _TRAINER_SCRIPT:
        raise JobError(f"algo must be one of {sorted(_TRAINER_SCRIPT)}, got {algo!r}")

    mode = str(req.get("mode", "scratch")).lower()
    if mode not in ("scratch", "finetune"):
        raise JobError(f"mode must be 'scratch' or 'finetune', got {mode!r}")

    spec = _resolve_hardware_spec(req)

    total_steps = int(req.get("total_steps") or 1_000_000)
    if total_steps <= 0:
        raise JobError("total_steps must be positive")
    eval_every = int(req.get("eval_every") or 100_000)
    if eval_every <= 0:
        raise JobError("eval_every must be positive")

    resume_path: Optional[str] = None
    if mode == "finetune":
        base_model_id = req.get("base_model_id")
        if not base_model_id:
            raise JobError("mode 'finetune' requires base_model_id")
        resume_path = _resolve_resume_path(str(base_model_id))

    opponent_weights = req.get("opponent_weights")
    if opponent_weights is not None and not isinstance(opponent_weights, dict):
        raise JobError("opponent_weights must be an object or null")

    # Resolve any opponent ids in the mix that are NOT built-in zoo ids: they
    # must be saved custom opponents, whose DSL+spec we thread into the run
    # config so the trainer can sample them (via extra_opponents). An id that
    # is neither a zoo id nor a saved opponent is a 400.
    custom_opponents = _resolve_custom_opponents(opponent_weights)

    start_mult = req.get("start_mult")
    if start_mult is not None:
        try:
            start_mult = float(start_mult)
        except (TypeError, ValueError) as exc:
            raise JobError("start_mult must be a number or null") from exc
        if start_mult <= 0:
            raise JobError("start_mult must be positive")

    hyperparams = req.get("hyperparams") or {}
    if not isinstance(hyperparams, dict):
        raise JobError("hyperparams must be an object or null")

    # ADAPTIVE opponent weighting (PPO only, default OFF => byte-identical path).
    adaptive_opponents = bool(req.get("adaptive_opponents", False))

    return TrainingConfig(
        algo=algo,
        total_steps=total_steps,
        eval_every=eval_every,
        output_best_path=str(job_dir / "best.pt"),
        output_final_path=str(job_dir / "final.pt"),
        job_dir=str(job_dir),
        opponent_weights=opponent_weights,
        resume_path=resume_path,
        hardware_spec=spec,
        seed=int(req.get("seed") or 42),
        start_mult=start_mult,
        hyperparams=dict(hyperparams),
        custom_opponents=custom_opponents,
        adaptive_opponents=adaptive_opponents,
        adaptive_builtin_share=float(
            req.get("adaptive_builtin_share", 0.55)),
        adaptive_floor=float(req.get("adaptive_floor", 0.01)),
        adaptive_cap_mult=float(req.get("adaptive_cap_mult", 2.5)),
        adaptive_ema=float(req.get("adaptive_ema", 0.5)),
    )


# ---------------------------------------------------------------------------
# Process launch
# ---------------------------------------------------------------------------
def _spawn_trainer(
    algo: str, run_config_path: Path, job_dir: Path, smoke: bool
) -> subprocess.Popen:
    """Launch the trainer detached, teeing stdout+stderr to console.log.

    On Windows we route through a temp ``.bat`` so the conda activate script
    sets the DLL search path before python starts (the inline
    ``cmd /c "call ... && python"`` form is flaky through tooling). The bat is
    removed once the process is spawned. A new process GROUP is created so the
    whole tree can be signalled/killed on stop.
    """
    script = _TRAINER_SCRIPT[algo]
    console_log = job_dir / "console.log"
    log_fh = open(console_log, "w", encoding="utf-8")

    env = os.environ.copy()
    env["SUMO_RUN_CONFIG"] = str(run_config_path)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONUNBUFFERED"] = "1"
    if smoke:
        env[_SMOKE_ENV[algo]] = "1"

    if os.name == "nt":
        bat = job_dir / "_launch.bat"
        smoke_line = (
            f"set {_SMOKE_ENV[algo]}=1\r\n" if smoke else ""
        )
        bat.write_text(
            f'@echo off\r\n'
            f'call "{_ACTIVATE_BAT}" {_CONDA_ENV}\r\n'
            f'set KMP_DUPLICATE_LIB_OK=TRUE\r\n'
            f'set PYTHONUNBUFFERED=1\r\n'
            f'set SUMO_RUN_CONFIG={run_config_path}\r\n'
            f'{smoke_line}'
            f'cd /d "{_REPO_ROOT}"\r\n'
            f'python "{_REPO_ROOT / script}"\r\n',
            encoding="utf-8",
        )
        proc = subprocess.Popen(
            ["cmd", "/c", str(bat)],
            cwd=str(_REPO_ROOT),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        # The bat has been read by cmd by the time Popen returns its handle;
        # leave it on disk for debuggability (it's inside the gitignored job
        # dir) — removing it immediately can race the cmd interpreter.
    else:
        proc = subprocess.Popen(
            [sys.executable, str(_REPO_ROOT / script)],
            cwd=str(_REPO_ROOT),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

    # Keep the log file handle attached to the child only; close our copy once
    # the child owns it (the OS keeps the underlying fd alive for the child).
    log_fh.close()
    return proc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def start_job(req: dict[str, Any]) -> str:
    """Start a single training job; return its ``job_id``.

    Rejects with :class:`JobBusyError` (409) if a job is already running.
    Builds + persists the run config and job meta, then spawns the trainer.
    """
    global _active
    with _lock:
        if _active is not None and _active.proc.poll() is None:
            raise JobBusyError(
                f"a job is already running: {_active.job_id}"
            )
        # Disk re-adopt: a job from a previous server lifetime may still be
        # running even when _active is None (after a reload). Don't start a
        # second trainer alongside it.
        disk_running = _disk_running_job()
        if disk_running is not None:
            raise JobBusyError(f"a job is already running: {disk_running}")

        config.ensure_dirs()
        job_id = uuid.uuid4().hex[:12]
        job_dir = config.JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        cfg = _build_config(req, job_dir)
        run_config_path = job_dir / "run_config.json"
        run_config_path.write_text(cfg.to_json(), encoding="utf-8")

        smoke = bool(req.get("smoke"))
        proc = _spawn_trainer(cfg.algo, run_config_path, job_dir, smoke)

        meta = {
            "id": job_id,
            "algo": cfg.algo,
            "mode": str(req.get("mode", "scratch")).lower(),
            "smoke": smoke,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pid": proc.pid,
            "config": cfg.to_dict(),
        }
        (job_dir / "job.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
        )

        _active = _ActiveJob(
            job_id=job_id, proc=proc, algo=cfg.algo,
            mode=meta["mode"], job_dir=job_dir,
        )
        return job_id


def _read_progress(job_dir: Path) -> list[dict[str, Any]]:
    """Parse ``progress.jsonl`` into a list of events (skip bad lines).

    The file is append-only one-JSON-object-per-line, written by the detached
    eval subprocess; a partially-written trailing line is simply skipped.
    """
    path = job_dir / "progress.jsonl"
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # half-written tail line — ignore
    return events


def _job_meta(job_dir: Path) -> Optional[dict[str, Any]]:
    path = job_dir / "job.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _pid_alive(pid: Optional[int]) -> bool:
    """Best-effort check that OS process ``pid`` currently exists."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                capture_output=True, text=True, check=False,
            ).stdout
        except OSError:
            return False
        return f'"{pid}"' in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _disk_running_job() -> Optional[str]:
    """Newest job whose launcher pid is still alive, re-adopted from disk.

    Lets job tracking survive a backend reload/restart (e.g. uvicorn --reload
    after an edit): a job counts as running if its ``job.json`` pid is alive
    and it has not been explicitly stopped.
    """
    if not config.JOBS_DIR.is_dir():
        return None
    candidates: list[tuple[str, str]] = []
    for child in config.JOBS_DIR.iterdir():
        if not child.is_dir() or (child / "stopped.marker").exists():
            continue
        meta = _job_meta(child)
        if meta is None or not _pid_alive(meta.get("pid")):
            continue
        candidates.append((meta.get("started_at") or "", meta.get("id", child.name)))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _is_alive(job_id: str) -> bool:
    """Whether ``job_id`` is still running — via the in-process handle when we
    own it, else by checking the job's launcher pid on disk (so tracking
    survives a backend reload/restart)."""
    with _lock:
        if (_active is not None and _active.job_id == job_id
                and _active.proc.poll() is None):
            return True
    job_dir = config.JOBS_DIR / job_id
    if (job_dir / "stopped.marker").exists():
        return False
    meta = _job_meta(job_dir)
    return _pid_alive(meta.get("pid") if meta else None)


def _exit_code(job_id: str) -> Optional[int]:
    with _lock:
        if _active is not None and _active.job_id == job_id:
            return _active.proc.poll()
    return None


def status(job_id: Optional[str] = None) -> dict[str, Any]:
    """Status of ``job_id`` (or the active job when ``job_id`` is None).

    Returns ``{state, running, config, events, latest_checkpoint, job_id}``.
    ``state`` is one of ``running`` / ``done`` / ``error`` / ``stopped`` /
    ``unknown``. ``events`` is the parsed ``progress.jsonl``;
    ``latest_checkpoint`` is the most recent ``checkpoint`` event (or None).
    """
    if job_id is None:
        job_id = active_job_id()
        if job_id is None:
            return {
                "state": "idle", "running": False, "config": None,
                "events": [], "latest_checkpoint": None, "job_id": None,
            }

    job_dir = config.JOBS_DIR / job_id
    meta = _job_meta(job_dir)
    if meta is None:
        return {
            "state": "unknown", "running": False, "config": None,
            "events": [], "latest_checkpoint": None, "job_id": job_id,
        }

    events = _read_progress(job_dir)
    checkpoints = [e for e in events if e.get("t") == "checkpoint"]
    latest = checkpoints[-1] if checkpoints else None

    running = _is_alive(job_id)
    if running:
        state = "running"
    else:
        # Not the active in-process job, or it has exited. Distinguish a clean
        # finish from an error / explicit stop using the exit code + a marker.
        if (job_dir / "stopped.marker").exists():
            state = "stopped"
        else:
            code = _exit_code(job_id)
            if code is None:
                # Process belongs to a previous server lifetime; infer from
                # whether a final checkpoint exists.
                state = "done" if (job_dir / "final.pt").exists() or latest \
                    else "unknown"
            elif code == 0:
                state = "done"
            else:
                state = "error"

    return {
        "state": state,
        "running": running,
        "config": meta.get("config"),
        "events": events,
        "latest_checkpoint": latest,
        "job_id": job_id,
    }


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill ``proc`` and its whole descendant tree (Windows-safe).

    On Windows we shell out to ``taskkill /T /F`` so the detached eval
    subprocesses the trainer spawns die too; elsewhere we signal the process
    group. Best-effort: a dead process is fine.
    """
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            proc.kill()
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


def stop() -> bool:
    """Stop the active job's process tree; return True if one was running.

    Writes a ``stopped.marker`` in the job dir so :func:`status` reports
    ``stopped`` (not ``error``) for the killed run.
    """
    global _active
    with _lock:
        if _active is None or _active.proc.poll() is not None:
            return False
        job = _active

    _kill_tree(job.proc)
    try:
        (job.job_dir / "stopped.marker").write_text("stopped\n", encoding="utf-8")
    except OSError:
        pass
    with _lock:
        if _active is not None and _active.job_id == job.job_id:
            _active = None
    return True


def list_jobs() -> list[dict[str, Any]]:
    """Summaries of all jobs on disk (newest first by ``started_at``)."""
    config.ensure_dirs()
    out: list[dict[str, Any]] = []
    for child in sorted(config.JOBS_DIR.iterdir()):
        if not child.is_dir():
            continue
        meta = _job_meta(child)
        if meta is None:
            continue
        job_id = meta.get("id", child.name)
        out.append({
            "id": job_id,
            "algo": meta.get("algo"),
            "mode": meta.get("mode"),
            "smoke": meta.get("smoke", False),
            "started_at": meta.get("started_at"),
            "running": _is_alive(job_id),
        })
    out.sort(key=lambda j: j.get("started_at") or "", reverse=True)
    return out
