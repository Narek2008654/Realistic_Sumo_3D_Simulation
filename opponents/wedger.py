"""Wedger: arc-flank approach + close-range push, mirroring our agent's style."""

from __future__ import annotations

from .base import OpponentController


class Wedger(OpponentController):
    FULL = 83.78
    ARC = 40.0   # inside-wheel speed during an arc turn (~380 RPM)
    SPIN = 60.0

    def decide(
        self,
        ir_hits: dict[str, bool],
        edge_left: bool,
        edge_right: bool,
    ) -> tuple[float, float]:
        edge = self._edge.step(edge_left, edge_right)
        if edge is not None:
            return edge

        # Close the gap with an asymmetric arc when the target is off-axis.
        if ir_hits.get("fc"):
            return self.FULL, self.FULL
        if ir_hits.get("fl"):
            return self.ARC, self.FULL
        if ir_hits.get("fr"):
            return self.FULL, self.ARC

        # Side hits: spin to bring the target into the front cone.
        if ir_hits.get("sl"):
            return -self.SPIN, self.SPIN
        if ir_hits.get("sr"):
            return self.SPIN, -self.SPIN

        # Default: continuous arc-search so we sweep the ring.
        return self.ARC, self.FULL * 0.7
