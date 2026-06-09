"""Tests for promoting a finished training run into the Models registry.

config.CHECKPOINTS / JOBS_DIR / REGISTRY_DIR are redirected to a throwaway temp
sandbox so the real registry is never touched. Covers the happy path plus the
edge cases the adversarial review flagged: job_id path traversal, a corrupt .pt
not 500-ing the listing, name conflicts, protected names, and validation.

Runnable two ways::

    python tests/test_promote.py     # plain PASS/FAIL harness
    pytest tests/test_promote.py
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: F401,E402  before numpy (Windows DLL order)
from webapp.backend import config, registry  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
_VALID_PT = _REPO / "checkpoints" / "ppo_robust_best.pt"
_ORIG = (config.CHECKPOINTS, config.JOBS_DIR, config.REGISTRY_DIR)
_SANDBOXES: list[Path] = []


def _sandbox() -> Path:
    """Fresh sandbox with one finished job (valid best.pt + a checkpoint line)."""
    tmp = Path(tempfile.mkdtemp(prefix="promote_test_"))
    _SANDBOXES.append(tmp)
    config.CHECKPOINTS = tmp / "checkpoints"
    config.JOBS_DIR = tmp / "jobs"
    config.REGISTRY_DIR = tmp / "registry"
    for d in (config.CHECKPOINTS, config.JOBS_DIR, config.REGISTRY_DIR):
        d.mkdir(parents=True)
    job = config.JOBS_DIR / "job_demo"
    job.mkdir()
    shutil.copy2(_VALID_PT, job / "best.pt")
    (job / "job.json").write_text(json.dumps(
        {"algo": "ppo", "mode": "finetune", "started_at": "2026-06-08T00:00:00"}))
    (job / "progress.jsonl").write_text(json.dumps(
        {"t": "checkpoint", "step": 1000,
         "eval": {"wr": 0.7, "self_out": 3, "n": 10}}) + "\n")
    return tmp


def _expect(exc_type, fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def test_list_runs() -> None:
    _sandbox()
    runs = registry.list_runs()
    assert len(runs) == 1, runs
    r = runs[0]
    assert r["job_id"] == "job_demo" and r["has_best"] and r["best_wr"] == 0.7


def test_promote_happy() -> None:
    _sandbox()
    card = registry.promote_job_model("job_demo", "best", "My Finetune v1")
    assert card["id"] == "my-finetune-v1", card["id"]
    assert any(m["id"] == "my-finetune-v1" for m in registry.list_models())


def test_promote_conflict() -> None:
    _sandbox()
    registry.promote_job_model("job_demo", "best", "dupe")
    _expect(registry.PromoteConflict,
            lambda: registry.promote_job_model("job_demo", "best", "dupe"))


def test_promote_protected() -> None:
    _sandbox()
    for nm in ("ppo_robust_best", "ppo-robust-best"):
        _expect(registry.PromoteConflict,
                lambda nm=nm: registry.promote_job_model("job_demo", "best", nm))


def test_unknown_job_and_bad_args() -> None:
    _sandbox()
    _expect(registry.PromoteNotFound,
            lambda: registry.promote_job_model("nope", "best", "x"))
    _expect(registry.PromoteInvalid,
            lambda: registry.promote_job_model("job_demo", "weird", "x"))
    _expect(registry.PromoteInvalid,
            lambda: registry.promote_job_model("job_demo", "best", "!!!"))


def test_path_traversal_blocked() -> None:
    tmp = _sandbox()
    outside = tmp / "outside"
    outside.mkdir()
    shutil.copy2(_VALID_PT, outside / "best.pt")
    _expect(registry.PromoteNotFound,
            lambda: registry.promote_job_model("../outside", "best", "evil"))
    assert not (config.CHECKPOINTS / "evil.pt").exists()


def test_corrupt_pt_does_not_break_listing() -> None:
    _sandbox()
    registry.promote_job_model("job_demo", "best", "good")
    (config.CHECKPOINTS / "broken.pt").write_bytes(b"not a torch archive")
    ids = {m["id"] for m in registry.list_models()}  # must not raise
    assert "good" in ids and "broken" not in ids, ids


def _main() -> int:
    tests = [
        ("list_runs", test_list_runs),
        ("promote_happy", test_promote_happy),
        ("promote_conflict", test_promote_conflict),
        ("promote_protected", test_promote_protected),
        ("unknown_job_and_bad_args", test_unknown_job_and_bad_args),
        ("path_traversal_blocked", test_path_traversal_blocked),
        ("corrupt_pt_does_not_break_listing", test_corrupt_pt_does_not_break_listing),
    ]
    failed = []
    try:
        for name, fn in tests:
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as exc:  # noqa: BLE001
                failed.append(name)
                print(f"  FAIL  {name}: {exc!r}")
    finally:
        config.CHECKPOINTS, config.JOBS_DIR, config.REGISTRY_DIR = _ORIG
        for d in _SANDBOXES:
            shutil.rmtree(d, ignore_errors=True)
    print(f"\nPASS: {len(tests) - len(failed)} / FAIL: {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
