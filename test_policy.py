"""Run the trained SAC policy in the GUI to watch how it plays.

Loads ``sac_actor_novamax.zip`` (saved by ``train_novamax.py``) and runs
``MiniSumoEnv`` with ``gui=True`` using the deterministic mean action
(``tanh(mu)``), i.e. no exploration noise. Domain randomisation stays on
so you see behaviour under realistic noise.
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: F401  must import before pybullet on Windows
from pathlib import Path

import pybullet as p
from stable_baselines3 import SAC

from sumo_env import MiniSumoEnv


MODEL_PATH = Path(__file__).with_name("sac_actor_novamax_v2.zip")
EPISODES = 30


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH.name} not found — run train_novamax.py first."
        )

    env = MiniSumoEnv(gui=True, seed=0, novamax_level=3)
    model = SAC.load(str(MODEL_PATH), device="cpu")

    wins = losses = 0
    try:
        for episode in range(EPISODES):
            if not env.connected:
                print("GUI window closed — exiting.")
                break
            try:
                obs, _ = env.reset()
            except p.error:
                print("GUI window closed — exiting.")
                break

            terminated = truncated = False
            episode_return = 0.0
            steps = 0

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                if info.get("disconnected"):
                    print("GUI window closed — exiting.")
                    return
                episode_return += reward
                steps += 1

            won = episode_return > 500.0
            outcome = "WIN " if won else "LOSE"
            if won:
                wins += 1
            else:
                losses += 1
            print(
                f"[ep {episode:02d}] {outcome}  steps={steps:3d}  "
                f"return={episode_return:+.1f}  "
                f"running_wr={wins / (episode + 1):.1%}"
            )
    finally:
        env.close()

    total = wins + losses
    if total:
        print(f"\nGreedy win-rate: {wins}/{total} = {wins / total:.1%}")


if __name__ == "__main__":
    main()
