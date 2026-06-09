"""Adaptive opponent-mix re-weighting (pure, gated).

The training opponent mix can re-weight itself from the per-opponent greedy
win-rates produced by each checkpoint eval. The goal is curriculum pressure —
spend more training episodes on the opponents the agent currently *loses* to —
WITHOUT the catastrophic-forgetting failure a naive "fight what you lose"
scheme caused (a near-unwinnable custom heavy body soaked up 65% of the mix and
the zoo win-rate collapsed 73% -> 60%).

Three guardrails make that failure impossible:

  1. **Reserved pool shares.** The built-in zoo and the custom (extra) pool each
     get a FIXED fraction of the total weight (``builtin_share`` and
     ``1 - builtin_share``). Adaptation only redistributes weight *within* a
     pool, so the zoo is never starved no matter how unwinnable the customs are.
  2. **Per-opponent cap.** Within a pool each opponent's priority is capped at
     ``cap_mult * pool_mean_priority``, so one near-0%% opponent cannot dominate
     its pool.
  3. **EMA.** The new weights are blended with the previous weights
     (``ema * target + (1 - ema) * prev``), so the mix drifts gradually rather
     than lurching every eval on noisy small-n win-rates.

Pure + stdlib-only (no numpy, no eval/exec) so it is trivially unit-testable
and safe to import from the trainers. :func:`recompute_weights` is the single
entry point; everything else is a private helper.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptiveCfg:
    """Knobs for :func:`recompute_weights` (all with safe defaults).

    ``builtin_share`` is the FIXED fraction of total weight reserved for the
    built-in zoo pool (the custom pool gets ``1 - builtin_share``). ``floor`` is
    the per-opponent priority floor (every opponent keeps a sliver of weight).
    ``cap_mult`` caps each priority at ``cap_mult * pool_mean``. ``ema`` is the
    blend toward the freshly-computed target (1.0 == no smoothing).
    """

    builtin_share: float = 0.55
    floor: float = 0.01
    cap_mult: float = 2.5
    ema: float = 0.5


def _normalize(weights: dict[str, float], target_sum: float) -> dict[str, float]:
    """Scale ``weights`` so they sum to ``target_sum``. An all-zero/empty input
    returns an even split (or an empty dict)."""
    total = sum(weights.values())
    if not weights:
        return {}
    if total <= 0.0:
        even = target_sum / len(weights)
        return {k: even for k in weights}
    scale = target_sum / total
    return {k: v * scale for k, v in weights.items()}


def _pool_target(
    ids: list[str],
    prev: dict[str, float],
    per_opponent_wr: dict[str, float],
    reserved: float,
    floor: float,
    cap_mult: float,
) -> dict[str, float]:
    """Compute one pool's TARGET weights (summing to ``reserved``).

    * An opponent WITH an eval datum gets priority ``floor + max(0, 1 - wr)`` —
      more weight on what it loses — then capped at ``cap_mult * mean_priority``.
    * An opponent WITHOUT an eval datum this round keeps its previous weight
      (renormalized into the pool), so a missing/odd id never gets zeroed.

    The two groups are then jointly renormalized to ``reserved``.
    """
    if not ids:
        return {}

    scored = {i: per_opponent_wr[i] for i in ids if i in per_opponent_wr}
    missing = [i for i in ids if i not in per_opponent_wr]

    target: dict[str, float] = {}

    if scored:
        prio = {i: floor + max(0.0, 1.0 - float(wr)) for i, wr in scored.items()}
        mean = sum(prio.values()) / len(prio)
        cap = cap_mult * mean
        prio = {i: min(p, cap) for i, p in prio.items()}
        target.update(prio)

    # Opponents with no datum this round carry their previous weight forward so
    # they are never zeroed; if they had no previous weight either, seed them at
    # the floor so they stay sampleable.
    for i in missing:
        target[i] = max(float(prev.get(i, 0.0)), floor)

    return _normalize(target, reserved)


def recompute_weights(
    prev_weights: dict[str, float],
    per_opponent_wr: dict[str, float],
    builtin_ids,
    cfg: AdaptiveCfg | None = None,
) -> dict[str, float]:
    """Re-weight the opponent mix from per-opponent greedy win-rates.

    ``prev_weights`` is the live mix (``{id: weight}``, every sampled id, both
    pools). ``per_opponent_wr`` maps ``id -> win_rate`` from the latest eval
    (ids absent from this dict keep a sane weight). ``builtin_ids`` is the set
    of built-in zoo ids (everything else in ``prev_weights`` is a custom/extra
    opponent). Returns a NEW ``{id: weight}`` dict summing to 1.0.

    The built-in pool is reserved ``cfg.builtin_share`` and the custom pool
    ``1 - cfg.builtin_share`` — FIXED, so the zoo is never starved. Within each
    pool, weight flows toward the opponents being lost to (priority
    ``floor + max(0, 1 - wr)``), each priority capped at ``cap_mult * pool_mean``.
    The result is EMA-blended with ``prev_weights`` and renormalized to 1.0.

    Pure: no eval/exec, no I/O. When ``per_opponent_wr`` is empty the result is
    just ``prev_weights`` renormalized (a no-op mix), so the FIRST eval — before
    any win-rates exist — leaves the static mix untouched.
    """
    cfg = cfg or AdaptiveCfg()
    builtin_set = set(builtin_ids)

    # Only consider ids the mix actually samples (positive weight) plus any id
    # that has an eval datum — a zero-weight id stays zero.
    ids = [i for i, w in prev_weights.items() if w > 0.0]
    for i in per_opponent_wr:
        if i in prev_weights and prev_weights[i] > 0.0 and i not in ids:
            ids.append(i)

    if not ids:
        return dict(prev_weights)

    builtin_pool = [i for i in ids if i in builtin_set]
    extra_pool = [i for i in ids if i not in builtin_set]

    # Reserved shares. If a pool is EMPTY its reserved share is handed to the
    # other pool (so a built-in-only or custom-only mix still sums to 1).
    if builtin_pool and extra_pool:
        builtin_reserved = cfg.builtin_share
        extra_reserved = 1.0 - cfg.builtin_share
    elif builtin_pool:
        builtin_reserved, extra_reserved = 1.0, 0.0
    else:
        builtin_reserved, extra_reserved = 0.0, 1.0

    target: dict[str, float] = {}
    target.update(_pool_target(
        builtin_pool, prev_weights, per_opponent_wr,
        builtin_reserved, cfg.floor, cfg.cap_mult,
    ))
    target.update(_pool_target(
        extra_pool, prev_weights, per_opponent_wr,
        extra_reserved, cfg.floor, cfg.cap_mult,
    ))

    # EMA blend toward the target, using each id's previous weight (0 for a
    # brand-new id). Then renormalize the whole mix to 1.0.
    ema = cfg.ema
    blended = {
        i: ema * target.get(i, 0.0) + (1.0 - ema) * float(prev_weights.get(i, 0.0))
        for i in ids
    }
    result = _normalize(blended, 1.0)

    # Carry any zero-weight ids from the previous mix through as 0.0 so the
    # returned dict has the same id set (keeps the env's weight map stable).
    for i, w in prev_weights.items():
        result.setdefault(i, 0.0)
    return result
