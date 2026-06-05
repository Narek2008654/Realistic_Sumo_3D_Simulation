"""Charger: mirror of bc_warmstart.charge_policy.

Logic (kept in lock-step with charge_policy):
  * front-center IR hit -> full charge (FULL, FULL)
  * front-left IR hit   -> soft turn left (ARC inside wheel)
  * front-right IR hit  -> soft turn right (ARC inside wheel)
  * side-left IR hit    -> tank-spin to face the agent
  * side-right IR hit   -> tank-spin to face the agent
  * no hit yet          -> search spin (default arc)

Used to expose the DQN's offline-pretrain dataset to "tracker-vs-
tracker" dynamics: both bots are running the same fight policy, so
the winning trajectories teach the network what *successful* mirror
combat looks like (wedge timing, side approach, head-on outcomes).
"""

from __future__ import annotations

from .base import OpponentController


class Charger(OpponentController):
    FULL = 83.78  # 800 RPM, clipped by _novamax_caps
    ARC = 40.0    # ~380 RPM, inside wheel during soft turn

    def __init__(self) -> None:
        super().__init__()
        self.last_seen = 0  # -1 left, 0 unseen, +1 right

    def reset(self) -> None:
        super().reset()
        self.last_seen = 0

    def decide(
        self,
        ir_hits: dict[str, bool],
        edge_left: bool,
        edge_right: bool,
    ) -> tuple[float, float]:
        # Priority 1: edge survival (matches base zoo behavior; the env
        # watchdog will also force this if the rear sensor fires).
        edge = self._edge.step(edge_left, edge_right)
        if edge is not None:
            return edge

        # Priority 2: attack via tracking.
        if ir_hits.get("fc"):
            return self.FULL, self.FULL
        if ir_hits.get("fl"):
            self.last_seen = -1
            return self.ARC, self.FULL
        if ir_hits.get("fr"):
            self.last_seen = +1
            return self.FULL, self.ARC
        if ir_hits.get("sl"):
            self.last_seen = -1
            return -self.FULL, self.FULL
        if ir_hits.get("sr"):
            self.last_seen = +1
            return self.FULL, -self.FULL

        # Priority 3: search in last-seen direction (matches the spin
        # direction charge_policy emits when last_seen_dir is set).
        if self.last_seen == -1:   # agent last seen on the LEFT -> spin left to re-acquire
            return -self.FULL * 0.6, self.FULL * 0.6
        if self.last_seen == +1:   # agent last seen on the RIGHT -> spin right to re-acquire
            return self.FULL * 0.6, -self.FULL * 0.6
        # Default: search spin to find the agent. Mirrors charge_policy's
        # (+0.6, -0.6) when no detection has happened yet.
        return self.FULL * 0.6, -self.FULL * 0.6
