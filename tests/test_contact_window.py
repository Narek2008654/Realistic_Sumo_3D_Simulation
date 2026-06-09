"""Regression test for the self-out vs push-loss CONTACT window.

The bug: a 2-step (~83 ms) "recent contact" window expired during a heavy
opponent's pre-falloff disengage (it rams the agent to the brink, then backs
off for its own edge-recovery several steps before the agent finally slides
off), so clear PUSHES were logged as self_out — inflating self-out metrics and
over-penalising training (lose_self vs lose_push). Fix: one shared constant
``CONTACT_RECENT_STEPS`` (sumo_env), raised to ~12 steps and imported by
battle.py / agent_vs_agent.py instead of each hardcoding its own window.

Runnable two ways::

    python tests/test_contact_window.py     # plain PASS/FAIL harness
    pytest tests/test_contact_window.py
"""
from __future__ import annotations

import inspect
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: F401,E402  # before numpy/pybullet (Windows DLL order)
import pybullet as p  # noqa: E402
from sumo_env import CONTACT_RECENT_STEPS, DISCRETE_ACTION_MAP, FALL_Z  # noqa: E402


def _reason_for_gap(gap: int) -> str:
    """Terminate the agent off-edge with the last enemy contact ``gap`` steps
    ago; return the env's termination_reason."""
    from train_dqn_3d import build_env

    env = build_env(gui=False, seed=3)
    base = env.unwrapped
    try:
        env.reset(seed=3)
        qrn = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        # Agent already below the platform (falls this step); enemy parked far
        # away so no fresh contact overwrites the backdated stamp.
        p.resetBasePositionAndOrientation(base.robot_id, [0.0, 0.0, FALL_Z - 0.2], qrn)
        p.resetBasePositionAndOrientation(base.enemy_id, [5.0, 5.0, 0.05], qrn)
        base._last_contact_step = base._steps - gap
        idle = next(i for i, a in enumerate(DISCRETE_ACTION_MAP) if a == (0.0, 0.0))
        _, _, _, _, info = env.step(idle)
        return info.get("termination_reason", "?")
    finally:
        env.close()


def test_recent_push_is_push_loss() -> None:
    # Contact a few steps before falloff (within the window) -> push_loss.
    assert _reason_for_gap(CONTACT_RECENT_STEPS - 4) == "push_loss"


def test_old_contact_is_self_out() -> None:
    # No contact for far longer than the window -> genuine self_out.
    assert _reason_for_gap(CONTACT_RECENT_STEPS + 12) == "self_out"


def test_window_unified_and_adequate() -> None:
    # Long enough to outlast a pre-falloff disengage (the bug was 2 steps)...
    assert CONTACT_RECENT_STEPS >= 8, CONTACT_RECENT_STEPS
    # ...and the OTHER classifiers share the single source of truth.
    from scripts.agent_vs_agent import play_match

    assert (
        inspect.signature(play_match).parameters["contact_window"].default
        == CONTACT_RECENT_STEPS
    )
    import webapp.backend.battle as battle

    src = Path(battle.__file__).read_text(encoding="utf-8")
    assert "contact_window = CONTACT_RECENT_STEPS" in src


def _main() -> int:
    tests = [
        ("recent_push_is_push_loss", test_recent_push_is_push_loss),
        ("old_contact_is_self_out", test_old_contact_is_self_out),
        ("window_unified_and_adequate", test_window_unified_and_adequate),
    ]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name}: {exc!r}")
    print(f"\nPASS: {len(tests) - len(failed)} / FAIL: {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
