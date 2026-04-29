"""Visual smoke-test for ``MiniSumoEnv``.

Spins up the environment in GUI mode, picks random actions, and prints the
discretized observation + reward each tick. The window stays open until you
close it (or until ``MAX_EPISODES`` finish).
"""

from __future__ import annotations

import pybullet as p

from sumo_env import MiniSumoEnv


MAX_EPISODES = 20


def main() -> None:
    env = MiniSumoEnv(gui=True, seed=0)
    try:
        for episode in range(MAX_EPISODES):
            if not env.connected:
                print("GUI window closed — exiting.")
                break
            try:
                obs, _ = env.reset()
            except p.error:
                # Window was closed between connectivity check and reset.
                print("GUI window closed — exiting.")
                break
            total_reward = 0.0
            steps = 0
            terminated = truncated = False

            while not (terminated or truncated):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                if info.get("disconnected"):
                    print("GUI window closed — exiting.")
                    return

            outcome = "WIN" if total_reward > 0 else ("LOSE" if total_reward <= -100 else "DRAW")
            print(
                f"[episode {episode:02d}] steps={steps:3d} "
                f"return={total_reward:+.0f} outcome={outcome} "
                f"last_obs={obs.tolist()}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
