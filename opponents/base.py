"""Common opponent controller interface + shared edge-recovery helper.

Every controller in the opponent zoo must implement:
    reset()                                       -> None
    decide(ir_hits, edge_left, edge_right)        -> (l_omega, r_omega) in rad/s
    is_edge_braking : bool                          (property)
    force_edge_brake(edge_left, edge_right)       -> None

The env's per-substep watchdog calls ``force_edge_brake`` to push the
controller into its reverse phase from outside. NovamaxController in
sumo_env.py implements the same surface area (duck-typed) so the env
can swap any of them in interchangeably.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class EdgeRecovery:
    """Two-phase recovery: reverse → tank-spin away from the triggered
    side. Speeds match NovaMax's EDGE_SAFE_SPEED so the body keeps
    traction during the manoeuvre.
    """

    EDGE_SAFE_SPEED = 30.0       # rad/s ≈ 286 RPM
    EDGE_REVERSE_STEPS = 2       # ~92 ms @ 24 Hz
    EDGE_SPIN_STEPS = 4          # ~184 ms @ 24 Hz

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._state: Optional[str] = None
        self._timer = 0
        self._spin_dir = 0

    @property
    def active(self) -> bool:
        return self._state is not None

    def force(self, edge_left: bool, edge_right: bool) -> None:
        """Inject the reverse phase from the env's watchdog."""
        if self._state is None and (edge_left or edge_right):
            self._state = "reverse"
            self._timer = self.EDGE_REVERSE_STEPS
            self._spin_dir = +1 if edge_left else -1

    def step(
        self, edge_left: bool, edge_right: bool,
    ) -> Optional[tuple[float, float]]:
        """Advance the state machine one tick. Returns (l, r) if the
        controller is currently braking, else None (caller runs its own
        decision policy)."""
        if self._state is None and (edge_left or edge_right):
            self._state = "reverse"
            self._timer = self.EDGE_REVERSE_STEPS
            self._spin_dir = +1 if edge_left else -1
        if self._state == "reverse":
            self._timer -= 1
            if self._timer <= 0:
                self._state = "spin"
                self._timer = self.EDGE_SPIN_STEPS
            v = self.EDGE_SAFE_SPEED
            return -v, -v
        if self._state == "spin":
            self._timer -= 1
            if self._timer <= 0:
                self._state = None
            v = self.EDGE_SAFE_SPEED
            return (v, -v) if self._spin_dir > 0 else (-v, v)
        return None


class OpponentController(ABC):
    """Base for the new opponents in the zoo. Composes an EdgeRecovery
    so each subclass only writes its `decide` body."""

    def __init__(self) -> None:
        self._edge = EdgeRecovery()

    def reset(self) -> None:
        self._edge.reset()

    @property
    def is_edge_braking(self) -> bool:
        return self._edge.active

    def force_edge_brake(self, edge_left: bool, edge_right: bool) -> None:
        self._edge.force(edge_left, edge_right)

    @abstractmethod
    def decide(
        self,
        ir_hits: dict[str, bool],
        edge_left: bool,
        edge_right: bool,
    ) -> tuple[float, float]:
        ...
