"""Unit tests for the SAFE opponent behavior DSL + its runtime interpreter.

Runnable two ways (mirrors tests/test_backend_lite.py):
    python tests/test_opponent_dsl.py    # plain-python harness (PASS/FAIL)
    pytest tests/test_opponent_dsl.py     # standard pytest

Pure logic — no PyBullet, no torch, no server. Safe to run anytime.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opponents.dsl_runtime import DslOpponent  # noqa: E402
from webapp.shared.opponent_dsl import (  # noqa: E402
    ACTION_WHEELS,
    DSL_SPEED,
    OpponentDSL,
    validate,
)

# A representative "good" program: charge when the front beam sees a target,
# turn toward a flank hit, otherwise spin-search.
GOOD_DSL = {
    "rules": [
        {"when": "front_hit", "do": "forward"},
        {"when": {"any": ["left_hit", "side_left_hit"]}, "do": "spin_left"},
        {"when": {"any": ["right_hit", "side_right_hit"]}, "do": "spin_right"},
        {"when": {"all": ["no_target", {"timer": {"every": 2}}]}, "do": "arc_left"},
    ],
    "default": "spin_right",
}

_NO_HITS = {"fc": False, "fl": False, "fr": False, "sl": False, "sr": False}


def _ir(**kw) -> dict[str, bool]:
    d = dict(_NO_HITS)
    d.update(kw)
    return d


def test_good_dsl_validates() -> None:
    assert validate(GOOD_DSL) == []
    # bare list form is accepted too
    assert validate([{"when": "front_hit", "do": "forward"}]) == []


def test_unknown_predicate_rejected() -> None:
    errors = validate({"rules": [{"when": "rear_hit", "do": "forward"}]})
    assert errors, "unknown predicate must produce an error"
    assert any("rear_hit" in e for e in errors), errors


def test_unknown_action_rejected() -> None:
    errors = validate({"rules": [{"when": "front_hit", "do": "teleport"}]})
    assert errors
    assert any("teleport" in e for e in errors), errors


def test_unknown_combinator_and_keys_rejected() -> None:
    assert validate({"rules": [{"when": {"xor": ["front_hit"]}, "do": "stop"}]})
    assert validate({"rules": [{"when": "front_hit", "do": "stop", "x": 1}]})
    assert validate({"rules": [{"when": "front_hit", "do": "stop"}], "junk": 1})
    assert validate({"rules": [{"when": "front_hit", "do": "stop"}], "default": "nope"})
    assert validate({"rules": []})  # empty rule list rejected
    assert validate("not a dsl")


def test_timer_validation() -> None:
    assert validate({"rules": [{"when": {"timer": {"every": 3}}, "do": "stop"}]}) == []
    assert validate({"rules": [{"when": {"timer": {"every": 0}}, "do": "stop"}]})
    assert validate({"rules": [{"when": {"timer": {"x": 3}}, "do": "stop"}]})


def test_roundtrip() -> None:
    dsl = OpponentDSL.from_dict(GOOD_DSL)
    assert dsl.to_dict()["default"] == "spin_right"
    assert len(dsl.to_dict()["rules"]) == 4
    # re-parse the serialised form
    assert validate(dsl.to_dict()) == []


def test_runtime_returns_clamped_tuples() -> None:
    opp = DslOpponent(GOOD_DSL)
    opp.reset()

    # front hit -> forward
    l, r = opp.decide(_ir(fc=True), False, False)
    assert (l, r) == ACTION_WHEELS["forward"]
    assert all(abs(v) <= DSL_SPEED + 1e-9 for v in (l, r))

    # left flank -> spin_left
    l, r = opp.decide(_ir(sl=True), False, False)
    assert (l, r) == ACTION_WHEELS["spin_left"]

    # right beam -> spin_right
    l, r = opp.decide(_ir(fr=True), False, False)
    assert (l, r) == ACTION_WHEELS["spin_right"]

    # no target: arc_left fires only when timer (every 2) lands; otherwise the
    # default spin_right runs. Either way the output is one of the action rows.
    seen = set()
    for _ in range(6):
        out = opp.decide(_ir(), False, False)
        assert all(abs(v) <= DSL_SPEED + 1e-9 for v in out)
        seen.add(out)
    assert ACTION_WHEELS["arc_left"] in seen or ACTION_WHEELS["spin_right"] in seen


def test_edge_recovery_takes_priority() -> None:
    # Even with a front hit, an edge trigger must drive the recovery manoeuvre
    # (reverse) instead of the matching rule's forward.
    opp = DslOpponent(GOOD_DSL)
    opp.reset()
    l, r = opp.decide(_ir(fc=True), edge_left=True, edge_right=False)
    assert l < 0 and r < 0, (l, r)  # reverse phase
    assert opp.is_edge_braking


def test_default_action_when_no_rule_matches() -> None:
    dsl = {"rules": [{"when": "front_hit", "do": "forward"}], "default": "stop"}
    opp = DslOpponent(dsl)
    opp.reset()
    assert opp.decide(_ir(), False, False) == ACTION_WHEELS["stop"]


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def _main() -> int:
    failures = 0
    for fn in _TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {fn.__name__}: {exc}")
    total = len(_TESTS)
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
