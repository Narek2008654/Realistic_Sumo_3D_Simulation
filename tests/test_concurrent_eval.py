"""Regression: two concurrent model evaluations must not 500.

PyBullet is single-client per process, so two ``evaluate_model`` calls running
at once used to corrupt each other's env (internal server error when a user
clicked *Evaluate* on two models). ``webapp.backend.pybullet_lock`` serializes
the in-process rollouts; this test fires two evaluations on different models
from two threads and asserts BOTH return metrics with no exception.

Run: cmd /c "call ...activate.bat sumo && set KMP_DUPLICATE_LIB_OK=TRUE &&
            python tests/test_concurrent_eval.py"
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webapp.backend import registry, config


def _two_compatible_model_ids() -> list[str]:
    """Two committed 21/9 checkpoints, falling back to whatever is on disk."""
    preferred = ["ppo_robust_best", "dqn3d_stack_stageA_best"]
    present = [m for m in preferred if (config.CHECKPOINTS / f"{m}.pt").exists()]
    if len(present) >= 2:
        return present[:2]
    ids = [c["id"] for c in registry.list_models()
           if c["obs_dim"] == 21 and c["action_dim"] == 9]
    assert len(ids) >= 2, f"need >=2 21/9 checkpoints, found {ids}"
    return ids[:2]


def test_concurrent_evaluate_does_not_crash() -> None:
    import shutil
    import tempfile
    from webapp.backend import config

    # Shrink the eval so the test is quick; the concurrency path is unchanged.
    registry._EVAL_OPPONENTS = ("dodger",)
    registry._EVAL_N = 2

    # Redirect the registry cache to a throwaway dir so this fast, tiny-n eval
    # never pollutes the real data/registry/ cards the UI reads.
    orig_registry_dir = config.REGISTRY_DIR
    config.REGISTRY_DIR = Path(tempfile.mkdtemp(prefix="reg_eval_test_"))

    ids = _two_compatible_model_ids()
    results: dict[str, object] = {}
    errors: dict[str, BaseException] = {}

    def run(model_id: str) -> None:
        try:
            results[model_id] = registry.evaluate_model(model_id)
        except BaseException as exc:  # noqa: BLE001 - capture any crash
            errors[model_id] = exc

    try:
        threads = [threading.Thread(target=run, args=(mid,)) for mid in ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        shutil.rmtree(config.REGISTRY_DIR, ignore_errors=True)
        config.REGISTRY_DIR = orig_registry_dir

    assert not errors, f"concurrent evaluate raised: {errors}"
    for mid in ids:
        card = results.get(mid)
        assert card is not None, f"{mid} returned None"
        assert card.get("metrics") is not None, f"{mid} has no metrics"
        assert "win_rate" in card["metrics"], f"{mid} metrics missing win_rate"
    print(f"PASS concurrent evaluate OK for {ids}")


if __name__ == "__main__":
    test_concurrent_evaluate_does_not_crash()
    print("all tests passed")
