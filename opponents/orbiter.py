"""Orbiter: continuous fixed-radius strafe around the agent.

Held-out opponent (never trained against). Instead of charging, it
drives forward with a steady turn bias so it circles, steering to keep
the agent off to one side rather than dead ahead. Tests two things the
training zoo never demands: defending against a flank-from-the-side and
breaking out of a sustained circle instead of a head-on wedge.
"""

from __future__ import annotations

from .base import OpponentController


class Orbiter(OpponentController):
    FWD = 60.0     # forward component of the orbit
    TURN = 28.0    # wheel differential setting the orbit curvature

    def decide(
        self,
        ir_hits: dict[str, bool],
        edge_left: bool,
        edge_right: bool,
    ) -> tuple[float, float]:
        edge = self._edge.step(edge_left, edge_right)
        if edge is not None:
            return edge

        # Curve so the agent stays on a side, not the nose. If it drifts
        # to the left sensors, curve right (and vice versa); on a head-on
        # hit, veer to convert the approach into an orbit.
        if ir_hits.get("fl") or ir_hits.get("sl"):
            return self.FWD + self.TURN, self.FWD - self.TURN   # curve right
        if ir_hits.get("fr") or ir_hits.get("sr"):
            return self.FWD - self.TURN, self.FWD + self.TURN   # curve left

        # Default / head-on: hold the strafing orbit.
        return self.FWD + self.TURN, self.FWD - self.TURN
