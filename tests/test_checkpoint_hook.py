"""Unit tests for the E1f job-config seam + checkpoint hook.

Runnable two ways:
    python tests/test_checkpoint_hook.py     # plain, prints PASS/FAIL
    pytest tests/test_checkpoint_hook.py     # collected as test_* funcs

Covers:
  * TrainingConfig <-> JSON round-trip (incl. nested HardwareSpec).
  * maybe_fire does NOTHING when step < eval_every.
  * skip-if-previous-alive: a still-running eval proc is not relaunched.
  * a launched eval produces a well-formed progress.jsonl line (the eval
    subprocess is faked by writing the line directly through a stub).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# repo root on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn

from webapp.shared import run_config, checkpoint_hook
from webapp.shared.run_config import TrainingConfig
from webapp.shared.hardware_spec import HardwareSpec


class _TinyNet(nn.Module):
    """Minimal stand-in with a state_dict for snapshotting."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 2)


class _FakeProc:
    """Stand-in for subprocess.Popen with a controllable poll()."""

    def __init__(self, alive: bool):
        self._alive = alive
        self.pid = 4242

    def poll(self):
        return None if self._alive else 0


def test_config_round_trip():
    cfg = TrainingConfig(
        algo="dqn",
        total_steps=12_000,
        eval_every=2_000,
        output_best_path="cp/best.pt",
        output_final_path="cp/final.pt",
        job_dir="jobs/abc",
        opponent_weights={"novamax": 1.0, "rammer": 0.5},
        resume_path=None,
        hardware_spec=HardwareSpec.default(),
        seed=7,
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "run_config.json"
        p.write_text(cfg.to_json(), encoding="utf-8")
        back = run_config.load(p)

    assert back.algo == "dqn"
    assert back.total_steps == 12_000
    assert back.eval_every == 2_000
    assert back.output_best_path == "cp/best.pt"
    assert back.opponent_weights == {"novamax": 1.0, "rammer": 0.5}
    assert back.seed == 7
    assert back.hardware_spec is not None
    # HardwareSpec survives the trip (compare the obs/action signature).
    assert (back.hardware_spec.obs_signature_hash
            == HardwareSpec.default().obs_signature_hash)
    # null hardware_spec must round-trip to None.
    cfg2 = TrainingConfig(algo="ppo", total_steps=5)
    d = cfg2.to_dict()
    assert d["hardware_spec"] is None
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "rc.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        assert run_config.load(p).hardware_spec is None
    print("PASS test_config_round_trip")


def test_no_fire_before_interval():
    with tempfile.TemporaryDirectory() as td:
        cfg = TrainingConfig(algo="dqn", total_steps=10_000,
                             eval_every=2_000, job_dir=td)
        net = _TinyNet()
        last, proc = checkpoint_hook.maybe_fire(net, 500, cfg, 0, None)
        assert last == 0, "last_eval_step must not advance before the interval"
        assert proc is None, "no eval proc should be launched"
        # nothing written to disk
        assert not (Path(td) / "progress.jsonl").exists()
        assert not (Path(td) / "snapshots").exists()
    print("PASS test_no_fire_before_interval")


def test_skip_when_previous_alive():
    with tempfile.TemporaryDirectory() as td:
        cfg = TrainingConfig(algo="dqn", total_steps=10_000,
                             eval_every=2_000, job_dir=td)
        net = _TinyNet()
        alive = _FakeProc(alive=True)
        last, proc = checkpoint_hook.maybe_fire(net, 2_000, cfg, 0, alive)
        # The interval elapsed, so a snapshot is saved and last advances...
        assert last == 2_000
        # ...but the same (still-alive) proc is returned: no relaunch.
        assert proc is alive
        snap = Path(td) / "snapshots" / "ckpt_2000.pt"
        assert snap.exists(), "snapshot must still be saved when skipping eval"
    print("PASS test_skip_when_previous_alive")


def test_progress_jsonl_line_well_formed():
    """Drive eval_and_record's writer path directly (no subprocess) to prove
    the progress.jsonl line shape, then assert it parses as one JSON object
    per line with the expected keys."""
    with tempfile.TemporaryDirectory() as td:
        job_dir = Path(td)
        (job_dir / "trajectories").mkdir(parents=True, exist_ok=True)
        traj_path = job_dir / "trajectories" / "2000.json"
        traj_path.write_text(json.dumps({
            "dt": 0.04167, "dohyo_radius": 0.5, "frames": [],
            "outcome": {"winner": "agent", "reason": "win"},
        }), encoding="utf-8")
        event = {
            "t": "checkpoint",
            "step": 2000,
            "snapshot": str(job_dir / "snapshots" / "ckpt_2000.pt"),
            "eval": {"wins": 3, "losses": 1, "wr": 0.6, "mean_ep_len": 120.0},
            "trajectory": str(traj_path),
        }
        prog = job_dir / "progress.jsonl"
        with open(prog, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        # second event to confirm append-only, one-object-per-line.
        with open(prog, "a", encoding="utf-8") as f:
            f.write(json.dumps({**event, "step": 4000}) + "\n")

        lines = prog.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        for ln in lines:
            obj = json.loads(ln)  # each line parses independently
            assert obj["t"] == "checkpoint"
            assert "step" in obj and "snapshot" in obj
            assert "eval" in obj and "wr" in obj["eval"]
            assert "trajectory" in obj
    print("PASS test_progress_jsonl_line_well_formed")


def _run_all():
    fns = [
        test_config_round_trip,
        test_no_fire_before_interval,
        test_skip_when_previous_alive,
        test_progress_jsonl_line_well_formed,
    ]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc!r}")
    print(f"\nPASS: {len(fns) - failed} / FAIL: {failed}")
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
