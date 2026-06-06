"""Feinter: bait-and-lunge. Fakes a retreat when it first sees the agent,
then charges once the agent commits to the chase.

Held-out opponent (never trained against) used to test anticipation /
the policy's temporal memory: a memoryless agent reads the retreat as
"opponent fleeing", drives in, and gets rammed when the feint reverses.
A stacked-distance policy should see the closing-rate sign flip and hold
position instead.
"""

from __future__ import annotations

from .base import OpponentController


class Feinter(OpponentController):
    FULL = 83.78           # lunge speed (clipped by _novamax_caps)
    BAIT = 50.0            # retreat speed during the feint
    BAIT_STEPS = 4         # ~167 ms backing up
    LUNGE_STEPS = 8        # ~333 ms charging

    def __init__(self) -> None:
        super().__init__()
        self._phase: str | None = None   # None | "bait" | "lunge"
        self._timer = 0

    def reset(self) -> None:
        super().reset()
        self._phase = None
        self._timer = 0

    def decide(
        self,
        ir_hits: dict[str, bool],
        edge_left: bool,
        edge_right: bool,
    ) -> tuple[float, float]:
        edge = self._edge.step(edge_left, edge_right)
        if edge is not None:
            self._phase = None
            return edge

        sees_front = ir_hits.get("fc") or ir_hits.get("fl") or ir_hits.get("fr")

        if self._phase is None and sees_front:
            self._phase = "bait"
            self._timer = self.BAIT_STEPS

        if self._phase == "bait":
            self._timer -= 1
            if self._timer <= 0:
                self._phase = "lunge"
                self._timer = self.LUNGE_STEPS
            return -self.BAIT, -self.BAIT          # back away to bait the chase

        if self._phase == "lunge":
            self._timer -= 1
            if self._timer <= 0:
                self._phase = None
            return self.FULL, self.FULL            # commit the ram

        # No detection yet: slow search spin.
        return self.FULL * 0.5, -self.FULL * 0.5
