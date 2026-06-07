"""Golden-master capture for the spec-driven dynamics refactor (E1c+E1d).

Builds the default env, runs a FIXED deterministic action sequence across
multiple episodes, and records per step:
  * the scalar reward                            -> rewards    [steps]
  * the agent base position (x, y, z)            -> agent_pos  [steps, 3]
  * the enemy base position (x, y, z)            -> enemy_pos  [steps, 3]

These exercise BOTH the AGENT motor limits (drivetrain torque/omega caps,
which shape every agent move and therefore every base position) and the
TERMINAL reward values (win / lose_push / lose_mutual / lose_self / timeout,
which dominate the per-episode reward spikes).

Run with NO args to (re)generate the committed golden file:
    python tests/golden/_capture_dynamics.py

Importable: ``rollout()`` returns ``(rewards, agent_pos, enemy_pos)`` so the
regression test can regenerate the same rollout against the refactored code
and diff it against the golden ``.npz``.
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

SEED = 2024
N_STEPS = 400

GOLDEN = os.path.join(_HERE, "dynamics_default.npz")


def rollout(n_steps: int = N_STEPS, seed: int = SEED):
    """Deterministic rollout of the default env.

    Cycles actions 0..8 repeatedly. Resets with an incrementing seed on
    ``done`` so the run spans multiple episodes. Returns
    ``(rewards, agent_pos, enemy_pos)`` where ``rewards`` is ``[n_steps]``
    and the two pose arrays are ``[n_steps, 3]`` (base x, y, z read via
    ``getBasePositionAndOrientation`` after each step).
    """
    import pybullet as p  # noqa: WPS433 (import after sys.path)

    from train_dqn_3d import build_env  # noqa: WPS433

    env = build_env(gui=False, seed=seed, force_opponent_id="novamax")
    base = env.unwrapped

    env.reset(seed=seed)

    rewards = np.empty((n_steps,), dtype=np.float64)
    agent_pos = np.empty((n_steps, 3), dtype=np.float64)
    enemy_pos = np.empty((n_steps, 3), dtype=np.float64)

    reset_seed = seed
    for t in range(n_steps):
        action = t % 9
        _obs, reward, terminated, truncated, _info = env.step(action)
        rewards[t] = float(reward)
        a_xyz, _ = p.getBasePositionAndOrientation(base.robot_id)
        e_xyz, _ = p.getBasePositionAndOrientation(base.enemy_id)
        agent_pos[t] = a_xyz
        enemy_pos[t] = e_xyz
        if terminated or truncated:
            reset_seed += 1
            env.reset(seed=reset_seed)

    env.close()
    return rewards, agent_pos, enemy_pos


def main() -> None:
    rewards, agent_pos, enemy_pos = rollout()
    np.savez(
        GOLDEN,
        rewards=rewards,
        agent_pos=agent_pos,
        enemy_pos=enemy_pos,
    )
    print(f"saved {GOLDEN}")
    print(
        f"  rewards shape={rewards.shape} "
        f"agent_pos shape={agent_pos.shape} enemy_pos shape={enemy_pos.shape}"
    )
    print(
        f"  reward[min,max]=[{rewards.min():.3f},{rewards.max():.3f}] "
        f"nonzero={int((rewards != 0).sum())}"
    )


if __name__ == "__main__":
    main()
