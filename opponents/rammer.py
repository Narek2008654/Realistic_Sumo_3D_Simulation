"""Rammer: charge full speed forward, no targeting, no dodging."""

from __future__ import annotations

from .base import OpponentController


class Rammer(OpponentController):
    FULL = 83.78  # 800 RPM, will be clipped by the env's _novamax_caps.

    def decide(
        self,
        ir_hits: dict[str, bool],
        edge_left: bool,
        edge_right: bool,
    ) -> tuple[float, float]:
        edge = self._edge.step(edge_left, edge_right)
        if edge is not None:
            return edge
        return self.FULL, self.FULL
