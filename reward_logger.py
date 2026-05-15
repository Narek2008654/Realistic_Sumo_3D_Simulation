"""Per-component reward bookkeeping for tensorboard.

The env emits ``info["reward_components_step"]`` every step and
``info["reward_components_episode"]`` on terminal/truncated. This
module exposes:

* ``RewardLoggerWrapper`` — gymnasium Wrapper that is otherwise
  transparent; it can also write a CSV trace if ``csv_path`` is set.
* ``RewardLoggerCallback`` — SB3 BaseCallback that reads the per-
  episode component totals out of info dicts during training and
  writes per-component running means to tensorboard under
  ``rew/<component>``.

Wire-up in train.py:

    env = MiniSumoEnv(...)
    env = RewardLoggerWrapper(env)
    model = SAC("MlpPolicy", env, tensorboard_log="runs/", ...)
    model.learn(..., callback=RewardLoggerCallback())
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback


class RewardLoggerWrapper(gym.Wrapper):
    """Pass-through wrapper that optionally writes per-episode component
    totals to a CSV. Tensorboard logging is handled separately by
    ``RewardLoggerCallback`` so we don't depend on the SB3 logger here.
    """

    def __init__(self, env: gym.Env, csv_path: Optional[str] = None):
        super().__init__(env)
        self._csv_path = Path(csv_path) if csv_path else None
        self._csv_keys: list[str] = []
        self._episode_idx = 0
        if self._csv_path is not None:
            self._csv_path.parent.mkdir(parents=True, exist_ok=True)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if (terminated or truncated) and self._csv_path is not None:
            comps = info.get("reward_components_episode", {})
            self._write_csv_row(comps)
            self._episode_idx += 1
        return obs, reward, terminated, truncated, info

    def _write_csv_row(self, comps: dict[str, float]) -> None:
        new_keys = sorted(set(self._csv_keys) | set(comps.keys()))
        write_header = (
            not self._csv_path.exists() or new_keys != self._csv_keys
        )
        self._csv_keys = new_keys
        with self._csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["episode"] + new_keys)
            writer.writerow(
                [self._episode_idx] + [comps.get(k, 0.0) for k in new_keys]
            )


class RewardLoggerCallback(BaseCallback):
    """SB3 callback that pulls per-component cumulative reward out of
    each terminal info dict and records running means to tensorboard
    under ``rew/<component>``.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        dones = self.locals.get("dones") or []
        for info, done in zip(infos, dones):
            if not done:
                continue
            comps = info.get("reward_components_episode")
            if comps:
                for key, value in comps.items():
                    self.logger.record_mean(f"rew/{key}", float(value))
                self.logger.record_mean(
                    "rew/episode_total_shaping",
                    float(sum(v for k, v in comps.items() if k != "terminal")),
                )
                self.logger.record_mean(
                    "rew/episode_terminal", float(comps.get("terminal", 0.0))
                )
            # Run 6: episode-level mechanics diagnostics (NOT rewards).
            # Surfaces wedge engagement frequency / signed score so we
            # can tell if the agent is actually achieving wedge contact
            # in addition to the weighted rew/wedge magnitude.
            diag = info.get("episode_diag")
            if diag:
                for key, value in diag.items():
                    self.logger.record_mean(f"diag/{key}", float(value))
        return True
