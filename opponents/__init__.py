"""Opponent zoo registry. ``sample_opponent(np_random)`` picks one
uniformly and returns ``(name, instance)``. The env stores the name in
``info["opponent_id"]`` so eval can slice metrics per-opponent.

NovamaxController lives in sumo_env.py to avoid moving a long-standing
class; we import it lazily in the factory so this package can be
imported without triggering the full sumo_env load.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .charger import Charger
from .dodger import Dodger
from .rammer import Rammer
from .spinner import Spinner
from .wedger import Wedger


def _novamax_factory():
    # Imported lazily because sumo_env imports from this package, and
    # NovamaxController is defined in sumo_env. The lazy import means
    # the package init does not depend on sumo_env being fully loaded.
    from sumo_env import NovamaxController
    return NovamaxController()


OPPONENT_REGISTRY: dict[str, Callable[[], object]] = {
    "novamax": _novamax_factory,
    "rammer":  Rammer,
    "dodger":  Dodger,
    "spinner": Spinner,
    "wedger":  Wedger,
    "charger": Charger,
}
OPPONENT_IDS: tuple[str, ...] = tuple(OPPONENT_REGISTRY.keys())

# Run 2: aggressive opponents over-represented so the agent doesn't
# farm passive idlers. Run 1 ended at 0% vs novamax/rammer/wedger but
# 20–28% vs dodger/spinner — this rebalance forces the train
# distribution toward the eval/deploy distribution.
# Run 11: "charger" added (mirror of charge_policy) for offline-data
# scenario 1 (agent fights its own logic). Weight is 0 by default so
# the standard zoo sampling distribution stays unchanged; charger is
# only used when explicitly pinned via force_opponent_id.
OPPONENT_WEIGHTS: dict[str, float] = {
    "novamax": 0.30,
    "rammer":  0.20,
    "wedger":  0.20,
    "dodger":  0.15,
    "spinner": 0.15,
    "charger": 0.0,
}
assert abs(sum(OPPONENT_WEIGHTS.values()) - 1.0) < 1e-9, (
    f"OPPONENT_WEIGHTS must sum to 1.0, got {sum(OPPONENT_WEIGHTS.values())}"
)
assert set(OPPONENT_WEIGHTS) == set(OPPONENT_IDS), (
    "OPPONENT_WEIGHTS keys must match OPPONENT_IDS exactly"
)


def make_opponent(name: str):
    """Instantiate a named opponent. Raises KeyError if unknown."""
    return OPPONENT_REGISTRY[name]()


def sample_opponent(
    np_random: np.random.Generator,
) -> tuple[str, object]:
    """Weighted draw over OPPONENT_IDS using OPPONENT_WEIGHTS.
    Returns (name, fresh instance)."""
    weights = np.array(
        [OPPONENT_WEIGHTS[name] for name in OPPONENT_IDS],
        dtype=np.float64,
    )
    weights /= weights.sum()
    idx = int(np_random.choice(len(OPPONENT_IDS), p=weights))
    name = OPPONENT_IDS[idx]
    return name, make_opponent(name)
