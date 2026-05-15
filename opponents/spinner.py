"""Spinner: tank-spin in place, brief charge when the agent walks into front cone."""

from __future__ import annotations

from .base import OpponentController


class Spinner(OpponentController):
    FULL = 83.78
    # Slightly under full so the body keeps traction while spinning in place.
    SPIN = 70.0

    def decide(
        self,
        ir_hits: dict[str, bool],
        edge_left: bool,
        edge_right: bool,
    ) -> tuple[float, float]:
        edge = self._edge.step(edge_left, edge_right)
        if edge is not None:
            return edge

        if ir_hits.get("fc"):
            return self.FULL, self.FULL  # one-shot lunge straight into agent

        # Default: tank-spin clockwise.
        return self.SPIN, -self.SPIN
