"""Raw-distance frame stacking for the mini-sumo policy.

The deployed policy is otherwise memoryless: it sees a single 12-D
observation per step. The three VL53L0X distance channels (obs[0:3] =
front/left/right) carry the only signal about *where the opponent is
moving*; the other nine features are already engineered temporal cues
(last_seen latch, engagement timer, yaw-rate proxy, two IR deltas, the
previous motor command). Stacking the whole vector would therefore stack
deltas-of-deltas — mostly redundant. Instead we stack ONLY the three raw
distance channels over the last ``K`` frames and keep the nine engineered
features single-frame. That hands the network the short distance
*trajectory* (so it can read a fast opponent's pivot rhythm and cut it
off) at minimal cost.

Layout (the single source of truth — the 21-D Arduino firmware mirrors this
byte-for-byte via a K-frame ring buffer in firmware/v5_deploy/v5_deploy.ino,
and firmware/v4_deploy for the DQN Stage-A model):

    [ d_{t-K+1} , ... , d_{t-1} , d_t , engineered_9 ]   (oldest distances first)

where each ``d`` is the 3-vector (front, left, right) and ``engineered_9``
is the *current* frame's obs[3:12]. For K=4 the stacked vector is 21-D.

On reset the ring is filled by replicating the first frame's distances
(zero initial motion), exactly as the firmware seeds its ring on setup /
watchdog-clear, so there is no sim-to-real start transient.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# obs[0:3] are the three raw VL distances; obs[3:12] are engineered.
DIST_DIM = 3
ENGINEERED_DIM = 9
BASE_OBS_DIM = DIST_DIM + ENGINEERED_DIM  # 12
DEFAULT_STACK_K = 4


def stacked_dim(k: int = DEFAULT_STACK_K) -> int:
    """Length of the stacked observation for a stack depth of ``k``."""
    return DIST_DIM * k + ENGINEERED_DIM


class RawDistanceStack(gym.ObservationWrapper):
    """Stack the three raw distance channels over the last ``k`` frames.

    The engineered features (obs[3:12]) are passed through from the
    current frame only. The wrapped env's own ``observation_space`` is
    left untouched at 12-D; this wrapper exposes the stacked 21-D space.
    """

    def __init__(self, env: gym.Env, k: int = DEFAULT_STACK_K) -> None:
        super().__init__(env)
        if int(k) < 1:
            raise ValueError(f"stack depth k must be >= 1, got {k!r}")
        self.k = int(k)

        # E1b: derive the distance / base widths from the wrapped env's
        # HardwareSpec when present (n_distance raw channels repeated K times
        # + the single-frame engineered tail). Falls back to the legacy
        # 3-distance / 12-D layout if the env exposes no spec, so the default
        # robot is unchanged.
        spec = getattr(env.unwrapped, "hw_spec", None)
        self._n_dist = getattr(spec, "n_distance", None) or DIST_DIM
        base = env.observation_space
        self._base_dim = (
            getattr(spec, "base_obs_dim", None) or int(base.shape[0])
        )
        if base.shape != (self._base_dim,):
            raise ValueError(
                f"RawDistanceStack expects a {self._base_dim}-D base obs, "
                f"got shape {base.shape}"
            )
        stacked = self._n_dist * self.k + (self._base_dim - self._n_dist)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(stacked,), dtype=np.float32,
        )
        self._ring: deque[np.ndarray] = deque(maxlen=self.k)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        dist = np.asarray(obs[:self._n_dist], dtype=np.float32)
        self._ring.clear()
        for _ in range(self.k):
            self._ring.append(dist.copy())
        return self._assemble(obs), info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # Called by ObservationWrapper.step() once per env step.
        self._ring.append(np.asarray(obs[:self._n_dist], dtype=np.float32))
        return self._assemble(obs)

    def _assemble(self, obs: np.ndarray) -> np.ndarray:
        engineered = np.asarray(obs[self._n_dist:self._base_dim], dtype=np.float32)
        return np.concatenate(list(self._ring) + [engineered]).astype(np.float32)


def stack_from_trajectory(
    obs: np.ndarray, dones: np.ndarray, k: int = DEFAULT_STACK_K
) -> np.ndarray:
    """Re-window a flat (N, 12) buffer of single-frame obs into (N, stacked).

    Stacking never crosses an episode boundary: ``dones[i] == True`` means
    the transition at ``i`` was terminal, so ``obs[i+1]`` begins a fresh
    episode. Frames before an episode start are filled by replicating that
    episode's first frame — identical to ``RawDistanceStack.reset``.

    Used as a fallback to re-window a 12-D dataset; the primary path
    collects through ``RawDistanceStack`` and stores 21-D obs natively.

    O(N*K) pure-Python — offline re-windowing only; never call on the hot path.
    """
    obs = np.asarray(obs, dtype=np.float32)
    dones = np.asarray(dones).astype(bool)
    n = obs.shape[0]
    if obs.shape[1] != BASE_OBS_DIM:
        raise ValueError(f"expected (N, {BASE_OBS_DIM}) obs, got {obs.shape}")
    out = np.empty((n, stacked_dim(k)), dtype=np.float32)
    for i in range(n):
        frames = [obs[i, :DIST_DIM]]          # newest first
        j = i
        while len(frames) < k and j - 1 >= 0 and not dones[j - 1]:
            j -= 1
            frames.append(obs[j, :DIST_DIM])
        while len(frames) < k:                # replicate episode-start frame
            frames.append(frames[-1])
        frames.reverse()                      # oldest first
        out[i] = np.concatenate(frames + [obs[i, DIST_DIM:BASE_OBS_DIM]])
    return out
