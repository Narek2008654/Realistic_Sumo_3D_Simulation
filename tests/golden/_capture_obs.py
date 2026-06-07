"""Golden-master capture for the spec-driven obs refactor (feature E1b).

Builds the default env, runs a FIXED deterministic action sequence across
multiple episodes, and records:
  * the stacked 21-D observation per step  -> obs array  [steps, obs_dim]
  * the current-frame raw distance channels -> raw array [steps, n_distance]

Run with NO args to (re)generate the committed golden files:
    python tests/golden/_capture_obs.py

Importable: ``rollout()`` returns ``(obs, raw)`` so the regression test can
regenerate the same rollout against the refactored code and diff it against
the golden ``.npy`` files.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Import torch FIRST on Windows: importing numpy first can trigger a
# DLL-search-order race against torch's fbgemm.dll (WinError 127). The
# rollout pulls in train_dqn_3d (which needs torch), so seed the loader
# here exactly as the trainer does.
import torch  # noqa: F401  (DLL-order side effect only)

import sys

import numpy as np

# Repo root on sys.path so `train_dqn_3d` / `sumo_env` import cleanly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

SEED = 12345
N_STEPS = 300

OBS_GOLDEN = os.path.join(_HERE, "obs_default.npy")
RAW_GOLDEN = os.path.join(_HERE, "raw_default.npy")


def rollout(n_steps: int = N_STEPS, seed: int = SEED):
    """Deterministic rollout of the default env.

    Cycles actions 0..8 repeatedly. Resets with an incrementing seed on
    ``done`` so the run spans >= 2 episodes. Returns ``(obs, raw)`` where
    ``obs`` is ``[n_steps, obs_dim]`` (stacked) and ``raw`` is
    ``[n_steps, n_distance]`` (current-frame raw distance channels =
    the newest distance frame in the stack).
    """
    from train_dqn_3d import build_env  # noqa: WPS433 (import after sys.path)

    env = build_env(gui=False, seed=seed, force_opponent_id="novamax")
    # n_distance is the env spec's distance count (3 for default). On the
    # pre-refactor env, `hw_spec` is absent, so fall back to 3; the refactored
    # env stores a HardwareSpec at `.hw_spec` with `.n_distance`.
    spec = getattr(env.unwrapped, "hw_spec", None)
    n_dist = getattr(spec, "n_distance", None) or 3
    k = env.k

    obs, _ = env.reset(seed=seed)
    obs_buf = np.empty((n_steps, obs.shape[0]), dtype=np.float32)
    raw_buf = np.empty((n_steps, n_dist), dtype=np.float32)

    reset_seed = seed
    for t in range(n_steps):
        action = t % 9
        obs, _reward, terminated, truncated, _info = env.step(action)
        obs_buf[t] = obs
        # Newest distance frame = last distance block before the engineered
        # tail. Stacked layout: [d_0 ... d_{k-1}, engineered]; newest is
        # block index k-1.
        raw_buf[t] = obs[n_dist * (k - 1): n_dist * k]
        if terminated or truncated:
            reset_seed += 1
            obs, _ = env.reset(seed=reset_seed)

    env.close()
    return obs_buf, raw_buf


def main() -> None:
    obs, raw = rollout()
    np.save(OBS_GOLDEN, obs)
    np.save(RAW_GOLDEN, raw)
    n_eps = 1  # at least 1; report rough episode coverage by counting resets
    print(f"saved {OBS_GOLDEN}  shape={obs.shape}")
    print(f"saved {RAW_GOLDEN}  shape={raw.shape}")
    print(f"obs dtype={obs.dtype}  raw dtype={raw.dtype}")


if __name__ == "__main__":
    main()
