"""Smoke tests for the ARENA battle backend.

Runnable two ways (mirrors tests/test_backend_lite.py):
    python tests/test_battle_backend.py    # plain-python harness (PASS/FAIL)
    pytest tests/test_battle_backend.py     # standard pytest

Uses FastAPI's TestClient (or the anyio fallback) in-process — it NEVER binds
port 8000, so it is safe to run while a training job owns the live server.

Battles run real physics, so rounds are kept tiny (1) and only a couple of
opponents are touched to minimize CPU contention. Every battle this suite
creates is cleaned up from ``data/battles/`` afterward.
"""

from __future__ import annotations

import importlib.util
import json as _json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webapp.backend import config  # noqa: E402
from webapp.backend.app import app  # noqa: E402

# Reuse the in-process client harness from the lite suite.
from tests.test_backend_lite import _make_client  # noqa: E402

client = _make_client()

A_MODEL = "ppo_robust_best"
B_MODEL = "dqn3d_stack_stageA_best"


def _assert_trajectory(traj: dict[str, Any]) -> None:
    assert isinstance(traj, dict), traj
    assert "dt" in traj and traj["dt"] > 0, traj
    assert "dohyo_radius" in traj and traj["dohyo_radius"] > 0, traj
    assert "outcome" in traj and "reason" in traj["outcome"], traj
    frames = traj["frames"]
    assert isinstance(frames, list) and len(frames) >= 2, len(frames)
    f0 = frames[0]
    for side in ("agent", "enemy"):
        assert side in f0, f0
        assert len(f0[side]["p"]) == 3, f0[side]
        assert len(f0[side]["q"]) == 4, f0[side]


def _assert_stats(stats: dict[str, Any], rounds: int) -> None:
    for k in ("rounds", "a_wins", "b_wins", "draws", "timeouts",
              "a_self_out", "b_self_out"):
        assert k in stats, stats
    assert stats["rounds"] == rounds, stats
    decided = stats["a_wins"] + stats["b_wins"] + stats["draws"] + stats["timeouts"]
    assert decided == rounds, stats


def _cleanup(battle_id: str | None) -> None:
    if not battle_id:
        return
    d = config.BATTLES_DIR / battle_id
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def test_battle_vs_opponent() -> None:
    battle_id = None
    try:
        resp = client.post("/api/battle", json={
            "a_model_id": A_MODEL, "b_opponent_id": "dodger",
            "rounds": 1, "mult": 3.0, "seed": 4242,
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        battle_id = body["battle_id"]
        _assert_stats(body["stats"], 1)
        _assert_trajectory(body["trajectory"])

        # GET the recorded trajectory back.
        got = client.get(f"/api/battle/{battle_id}/trajectory")
        assert got.status_code == 200, got.text
        _assert_trajectory(got.json())
    finally:
        _cleanup(battle_id)


def test_battle_vs_model() -> None:
    battle_id = None
    try:
        resp = client.post("/api/battle", json={
            "a_model_id": A_MODEL, "b_model_id": B_MODEL,
            "rounds": 1, "mult": 3.0, "seed": 4242,
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        battle_id = body["battle_id"]
        _assert_stats(body["stats"], 1)
        _assert_trajectory(body["trajectory"])
    finally:
        _cleanup(battle_id)


def test_battle_requires_one_opponent() -> None:
    # Neither b_model_id nor b_opponent_id -> 422.
    resp = client.post("/api/battle", json={"a_model_id": A_MODEL})
    assert resp.status_code == 422, resp.text
    # Both -> 422.
    resp = client.post("/api/battle", json={
        "a_model_id": A_MODEL, "b_model_id": B_MODEL, "b_opponent_id": "dodger",
    })
    assert resp.status_code == 422, resp.text


def test_battle_unknown_model() -> None:
    resp = client.post("/api/battle", json={
        "a_model_id": "does_not_exist", "b_opponent_id": "dodger", "rounds": 1,
    })
    assert resp.status_code == 404, resp.text


def test_battle_unknown_trajectory() -> None:
    resp = client.get("/api/battle/deadbeef/trajectory")
    assert resp.status_code == 404, resp.text


# ---------------------------------------------------------------------------
# Plain-python harness (no pytest required).
# ---------------------------------------------------------------------------
def _main() -> int:
    tests = [
        ("battle_requires_one_opponent", test_battle_requires_one_opponent),
        ("battle_unknown_model", test_battle_unknown_model),
        ("battle_unknown_trajectory", test_battle_unknown_trajectory),
        ("battle_vs_opponent", test_battle_vs_opponent),
        ("battle_vs_model", test_battle_vs_model),
    ]
    passed: list[str] = []
    failed: list[tuple[str, str]] = []
    for name, fn in tests:
        try:
            fn()
            passed.append(name)
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append((name, repr(exc)))
            print(f"  FAIL  {name}: {exc!r}")
    print(f"\nPASS: {len(passed)} / FAIL: {len(failed)}")
    for name, err in failed:
        print(f"    - {name}: {err}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(_main())
