"""Davo: the user's scripted tracker robot (davo_sirad.ino) as an opponent.

Distinct from Charger: the real robot has only THREE forward range sensors
(right / middle / left) and NO side sensors, so it is blind to its own
flanks (flankable). Behaviour mirrors the sketch's loop():
  * middle (front-center) hit -> charge straight (and stick to the target
    for a few ticks, matching the sketch's CONFIRM_MISSES=10 front grace)
  * right hit  -> pivot right toward it
  * left hit   -> pivot left toward it
  * nothing    -> spin-search in the last-seen direction

The sketch's distance hysteresis (DETECT 350 / HOLD 450 mm) is approximated
by the env's boolean IR hits; side hits (sl/sr) are intentionally ignored
because the real robot cannot see its sides.
"""

from __future__ import annotations

from .base import OpponentController


class Davo(OpponentController):
    FULL = 83.78           # full speed (clipped by _novamax_caps)
    FRONT_HOLD_TICKS = 6   # sticky front pursuit (~sketch CONFIRM_MISSES)

    def __init__(self) -> None:
        super().__init__()
        self.last_dir = 1      # +1 = right, -1 = left (sketch spinDir seed)
        self.hold = 0

    def reset(self) -> None:
        super().reset()
        self.last_dir = 1
        self.hold = 0

    def decide(
        self,
        ir_hits: dict[str, bool],
        edge_left: bool,
        edge_right: bool,
    ) -> tuple[float, float]:
        # Priority 1: edge survival (shared zoo behaviour + env watchdog).
        edge = self._edge.step(edge_left, edge_right)
        if edge is not None:
            return edge

        # Priority 2: pursue via the three forward sensors (no sides).
        if ir_hits.get("fc"):
            self.hold = self.FRONT_HOLD_TICKS
            return self.FULL, self.FULL
        if ir_hits.get("fr"):
            self.last_dir = 1
            return self.FULL, -self.FULL          # pivot right
        if ir_hits.get("fl"):
            self.last_dir = -1
            return -self.FULL, self.FULL          # pivot left

        # Priority 3: sticky front-hold — keep charging briefly after losing
        # the centered target (the sketch holds the last front reading).
        if self.hold > 0:
            self.hold -= 1
            return self.FULL, self.FULL

        # Priority 4: spin-search in the last-seen direction.
        if self.last_dir >= 0:
            return self.FULL, -self.FULL          # spin right
        return -self.FULL, self.FULL              # spin left
