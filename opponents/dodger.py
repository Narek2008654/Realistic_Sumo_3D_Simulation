"""Dodger: tank-spin perpendicular when the agent is in front; otherwise idle."""

from __future__ import annotations

from .base import OpponentController


class Dodger(OpponentController):
    FULL = 83.78
    SPIN = 83.78

    def decide(
        self,
        ir_hits: dict[str, bool],
        edge_left: bool,
        edge_right: bool,
    ) -> tuple[float, float]:
        edge = self._edge.step(edge_left, edge_right)
        if edge is not None:
            return edge

        # Front threat: spin AWAY from the side that sees the agent.
        if ir_hits.get("fl"):
            return self.SPIN, -self.SPIN
        if ir_hits.get("fr"):
            return -self.SPIN, self.SPIN
        if ir_hits.get("fc"):
            return self.SPIN, -self.SPIN  # pick a direction on a head-on hit

        # Side / rear contact: floor it forward to escape.
        if ir_hits.get("sl") or ir_hits.get("sr"):
            return self.FULL, self.FULL

        # Oblivious — idle so the agent has to chase us down.
        return 0.0, 0.0
