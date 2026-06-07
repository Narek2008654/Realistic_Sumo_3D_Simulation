"""Regression gate for the spec-driven dynamics refactor (E1c+E1d).

The env's AGENT motor caps (drivetrain torque/omega) and its TERMINAL reward
magnitudes (win / lose_push / lose_mutual / lose_self / timeout) are now driven
by a ``HardwareSpec`` (``HardwareSpec.default()`` for the default robot). This
test re-runs the exact deterministic rollout captured in
``tests/golden/_capture_dynamics.py`` and asserts the refactored env reproduces
the pre-refactor golden stream to < 1e-9 for:
  * the per-step scalar reward (exercises the terminal reward table), and
  * the agent AND enemy base-position traces (exercise the agent motor caps,
    which shape every move; the enemy is a cross-check that nothing else moved).

The reward sequence is additionally asserted EXACTLY equal (==).

Run directly:
    python tests/test_dynamics_faithfulness_spec.py
or under pytest:
    pytest tests/test_dynamics_faithfulness_spec.py
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Import torch FIRST on Windows (DLL-search-order race vs numpy); the rollout
# pulls in train_dqn_3d, which requires torch.
import torch  # noqa: F401

import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

sys.path.insert(0, os.path.join(_HERE, "golden"))
from _capture_dynamics import rollout, GOLDEN  # noqa: E402

TOL = 1e-9


def test_dynamics_faithfulness_spec() -> None:
    """Refactored env == golden master, to < 1e-9, for rewards and poses."""
    assert os.path.exists(GOLDEN), (
        f"missing golden {GOLDEN}; regenerate with "
        f"`python tests/golden/_capture_dynamics.py` on a known-good revision"
    )

    g = np.load(GOLDEN)
    g_rewards = g["rewards"]
    g_agent = g["agent_pos"]
    g_enemy = g["enemy_pos"]

    rewards, agent_pos, enemy_pos = rollout()

    assert rewards.shape == g_rewards.shape, (
        f"rewards shape {rewards.shape} != golden {g_rewards.shape}"
    )
    assert agent_pos.shape == g_agent.shape, (
        f"agent_pos shape {agent_pos.shape} != golden {g_agent.shape}"
    )
    assert enemy_pos.shape == g_enemy.shape, (
        f"enemy_pos shape {enemy_pos.shape} != golden {g_enemy.shape}"
    )

    # Terminal rewards are the headline of this refactor: require EXACT match.
    assert np.array_equal(rewards, g_rewards), (
        "reward sequence not exactly equal to golden; "
        f"max abs diff = {float(np.max(np.abs(rewards - g_rewards))):.3e}"
    )

    agent_diff = float(np.max(np.abs(agent_pos - g_agent)))
    enemy_diff = float(np.max(np.abs(enemy_pos - g_enemy)))

    assert agent_diff < TOL, f"agent pose max abs diff {agent_diff:.3e} >= {TOL:.0e}"
    assert enemy_diff < TOL, f"enemy pose max abs diff {enemy_diff:.3e} >= {TOL:.0e}"


if __name__ == "__main__":
    test_dynamics_faithfulness_spec()
    g = np.load(GOLDEN)
    rewards, agent_pos, enemy_pos = rollout()
    print(
        "PASS  reward max|diff| =",
        f"{float(np.max(np.abs(rewards - g['rewards']))):.3e}",
        " agent max|diff| =",
        f"{float(np.max(np.abs(agent_pos - g['agent_pos']))):.3e}",
        " enemy max|diff| =",
        f"{float(np.max(np.abs(enemy_pos - g['enemy_pos']))):.3e}",
    )
