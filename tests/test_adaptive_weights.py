"""Unit tests for the pure adaptive opponent re-weighting helper.

Runnable two ways:
    python tests/test_adaptive_weights.py     # plain, prints PASS/FAIL
    pytest tests/test_adaptive_weights.py     # collected as test_* funcs

These are PURE tests (no env, no torch) — they pin the three guardrails that
prevent the catastrophic-forgetting failure:
  * reserved pool shares: the built-in zoo never drops below ``builtin_share``
    (minus a tiny epsilon for cap/EMA interaction);
  * the per-opponent cap: a 0%%-winrate opponent gets MORE weight than a 90%%
    one but cannot dominate its pool;
  * EMA: weights move gradually, not in one lurch;
  * a missing opponent keeps a sane (non-zero) weight; the mix sums to 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from webapp.shared.adaptive_weights import AdaptiveCfg, recompute_weights

_BUILTIN = {"novamax", "rammer", "wedger", "dodger", "spinner", "tracker", "charger"}
_EPS = 1e-9


def _approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_sums_to_one():
    prev = {"novamax": 0.5, "rammer": 0.5}
    wr = {"novamax": 0.3, "rammer": 0.8}
    out = recompute_weights(prev, wr, _BUILTIN)
    assert _approx(sum(out.values()), 1.0), f"sum={sum(out.values())}"
    print("PASS test_sums_to_one")


def test_loses_more_gets_more_but_capped():
    """A 0%% opponent gets MORE weight than a 90%% one, but the cap keeps it from
    dominating the pool (it must be well under the full reserved share)."""
    prev = {"novamax": 0.25, "rammer": 0.25, "wedger": 0.25, "dodger": 0.25}
    wr = {"novamax": 0.0, "rammer": 0.9, "wedger": 0.5, "dodger": 0.5}
    cfg = AdaptiveCfg(builtin_share=1.0, ema=1.0)  # all built-in, no smoothing
    out = recompute_weights(prev, wr, _BUILTIN, cfg)
    assert out["novamax"] > out["rammer"], "0%% must outweigh 90%%"
    # Cap: with cap_mult=2.5 and 4 opponents the 0%% loser cannot take the whole
    # pool — it stays comfortably below the reserved share (here 1.0).
    assert out["novamax"] < 0.6, f"0%% opponent dominated pool: {out['novamax']}"
    assert _approx(sum(out.values()), 1.0)
    print("PASS test_loses_more_gets_more_but_capped")


def test_cap_bounds_single_loser():
    """Explicit cap check: one 0%% opponent among many high-WR ones is bounded
    by cap_mult * pool_mean, not allowed to swallow the pool."""
    prev = {oid: 1.0 / 6 for oid in
            ["novamax", "rammer", "wedger", "dodger", "spinner", "tracker"]}
    wr = {"novamax": 0.0, "rammer": 0.95, "wedger": 0.95, "dodger": 0.95,
          "spinner": 0.95, "tracker": 0.95}
    cfg = AdaptiveCfg(builtin_share=1.0, ema=1.0, cap_mult=2.5, floor=0.01)
    out = recompute_weights(prev, wr, _BUILTIN, cfg)
    # priorities: loser=1.01, others=0.06; mean=(1.01+5*0.06)/6=0.218; cap=0.546.
    # capped loser share = 0.546 / (0.546 + 5*0.06) = ~0.645. Assert it is
    # capped (without the cap it would be 1.01/(1.01+0.30)=~0.77).
    assert out["novamax"] < 0.70, f"cap failed: loser={out['novamax']}"
    assert out["novamax"] > out["rammer"]
    print("PASS test_cap_bounds_single_loser")


def test_reserved_zoo_share_holds():
    """With a brutal (0%%) custom opponent and winnable zoo, the zoo pool keeps
    its reserved share — it is NEVER starved (the key guardrail)."""
    prev = {"novamax": 0.2, "rammer": 0.15, "heavy-custom": 0.65}
    wr = {"novamax": 0.8, "rammer": 0.8, "heavy-custom": 0.0}
    cfg = AdaptiveCfg(builtin_share=0.55, ema=1.0)
    out = recompute_weights(prev, wr, _BUILTIN, cfg)
    zoo = sum(w for o, w in out.items() if o in _BUILTIN)
    extra = sum(w for o, w in out.items() if o not in _BUILTIN)
    assert zoo >= 0.55 - _EPS, f"zoo starved: zoo_share={zoo}"
    assert _approx(zoo, 0.55, 1e-6), f"zoo_share drifted: {zoo}"
    assert _approx(extra, 0.45, 1e-6), f"extra_share drifted: {extra}"
    assert _approx(sum(out.values()), 1.0)
    print("PASS test_reserved_zoo_share_holds")


def test_ema_moves_gradually():
    """EMA blends toward the target rather than jumping to it: a half-EMA step
    must land strictly between the previous and the no-smoothing target."""
    prev = {"novamax": 0.5, "rammer": 0.5}
    wr = {"novamax": 0.0, "rammer": 1.0}  # novamax should gain
    full = recompute_weights(prev, wr, _BUILTIN, AdaptiveCfg(builtin_share=1.0, ema=1.0))
    half = recompute_weights(prev, wr, _BUILTIN, AdaptiveCfg(builtin_share=1.0, ema=0.5))
    # novamax gains in both, but the half-EMA gain is smaller than the full one.
    assert full["novamax"] > half["novamax"] > prev["novamax"], (
        f"EMA not gradual: prev={prev['novamax']} half={half['novamax']} "
        f"full={full['novamax']}"
    )
    assert _approx(sum(half.values()), 1.0)
    print("PASS test_ema_moves_gradually")


def test_missing_opponent_keeps_weight():
    """An opponent absent from the win-rate dict keeps a sane (non-zero) weight
    — it is carried forward, not zeroed."""
    prev = {"novamax": 0.4, "rammer": 0.3, "wedger": 0.3}
    wr = {"novamax": 0.5}  # rammer + wedger have no datum this round
    out = recompute_weights(prev, wr, _BUILTIN, AdaptiveCfg(ema=1.0))
    assert out["rammer"] > 0.0, "missing opponent was zeroed"
    assert out["wedger"] > 0.0, "missing opponent was zeroed"
    assert _approx(sum(out.values()), 1.0)
    print("PASS test_missing_opponent_keeps_weight")


def test_empty_wr_is_noop():
    """The FIRST eval (no win-rates) leaves the static mix unchanged (just
    renormalized)."""
    prev = {"novamax": 0.6, "rammer": 0.4}
    out = recompute_weights(prev, {}, _BUILTIN)
    assert _approx(out["novamax"], 0.6) and _approx(out["rammer"], 0.4)
    print("PASS test_empty_wr_is_noop")


def test_zero_weight_ids_stay_zero():
    """A held-out id with weight 0 stays 0 and is preserved in the output."""
    prev = {"novamax": 0.5, "rammer": 0.5, "feinter": 0.0}
    wr = {"novamax": 0.5, "rammer": 0.5}
    out = recompute_weights(prev, wr, _BUILTIN, AdaptiveCfg(ema=1.0))
    assert _approx(out.get("feinter", 0.0), 0.0), "zero-weight id became nonzero"
    assert "feinter" in out
    print("PASS test_zero_weight_ids_stay_zero")


def test_builtin_only_mix_works():
    """A built-in-only mix (no custom pool) still adapts + sums to 1."""
    prev = {"novamax": 0.25, "rammer": 0.25, "wedger": 0.25, "dodger": 0.25}
    wr = {"novamax": 0.2, "rammer": 0.4, "wedger": 0.6, "dodger": 0.8}
    out = recompute_weights(prev, wr, _BUILTIN, AdaptiveCfg(ema=1.0))
    assert _approx(sum(out.values()), 1.0)
    assert out["novamax"] > out["dodger"], "loses-more should outweigh"
    print("PASS test_builtin_only_mix_works")


def _run_all():
    fns = [
        test_sums_to_one,
        test_loses_more_gets_more_but_capped,
        test_cap_bounds_single_loser,
        test_reserved_zoo_share_holds,
        test_ema_moves_gradually,
        test_missing_opponent_keeps_weight,
        test_empty_wr_is_noop,
        test_zero_weight_ids_stay_zero,
        test_builtin_only_mix_works,
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
