"""Combat policy: scripted controller for offline DQN data collection.

Designed to break the orbital deadlock that defeats reactive variants
of bc_warmstart.charge_policy against aggressive trackers (rammer /
wedger / novamax / charger). Bench v1/v2/v3 confirmed that policies
which tank-spin in place (charge_policy's search) get stuck mirroring
the opponent's reactive turn — both bots orbit, neither closes,
episode times out (~85% timeout rate vs trackers at mult=0.7).

Core design:
  1. NEVER pure tank-spin in place — always keep a non-trivial forward
     component. Constant forward translation forces convergence even
     when both bots are turning toward each other.
  2. Steer via IR imbalance (slow the inner wheel proportionally),
     not stop-and-turn.
  3. Commit at a generous IR threshold (any IR < 0.8) so the agent
     starts charging before the opponent finishes its first turn.
  4. Rear-line edge override: if line sensors fire, force full
     forward. Rear-line means our back is at the white border so
     reversing would self-out.

The policy is stateless. ``reset()`` exists only so callers can use
the same per-episode interface as a stateful policy without changes.
"""

from __future__ import annotations


class CombatPolicy:
    """Drop-in replacement for bc_warmstart.charge_policy with stronger
    engagement behavior.
    """

    BASE_FWD = 0.85        # base forward speed (both wheels) when engaged
    STEER_GAIN = 0.55      # asymmetric wheel bias for IR imbalance steering
    COMMIT_THRESH = 0.8    # commit forward when any IR < this (normalised)
    SEARCH_FWD = 0.85      # search-phase forward speed (no detection yet)
    SEARCH_BIAS = 0.20     # inner wheel speed during last-seen search curve

    def __init__(self) -> None:
        return  # stateless

    def reset(self) -> None:
        return

    def __call__(self, obs) -> tuple[float, float]:
        front = float(obs[0])
        left = float(obs[1])
        right = float(obs[2])
        last_seen = float(obs[3])
        line_l = float(obs[4])
        line_r = float(obs[5])

        # 1) Edge override: rear line sensor over white border. Reversing
        # would push us further off; forward is the only safe direction.
        if line_l > 0.5 or line_r > 0.5:
            return 1.0, 1.0

        # 2) Any IR detection within threshold -> commit forward with
        # proportional steering toward the strongest side.
        if (front < self.COMMIT_THRESH
                or left < self.COMMIT_THRESH
                or right < self.COMMIT_THRESH):
            s_l = max(0.0, 1.0 - left)
            s_r = max(0.0, 1.0 - right)
            bias = s_l - s_r   # positive -> enemy more on left -> turn left
            l_cmd = self.BASE_FWD - self.STEER_GAIN * bias
            r_cmd = self.BASE_FWD + self.STEER_GAIN * bias
            return (
                max(-1.0, min(1.0, l_cmd)),
                max(-1.0, min(1.0, r_cmd)),
            )

        # 3) No detection: search while still moving forward. last_seen
        # tells us which side the opponent was last on.
        if last_seen < -0.5:
            return self.SEARCH_BIAS, self.SEARCH_FWD
        if last_seen > 0.5:
            return self.SEARCH_FWD, self.SEARCH_BIAS
        # Default: forward + slight right curve (wide arc-search),
        # NOT in-place spin.
        return self.SEARCH_FWD, self.SEARCH_BIAS
