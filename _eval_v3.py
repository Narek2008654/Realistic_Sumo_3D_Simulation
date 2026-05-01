"""Pick the V3 checkpoint with the best deterministic win-rate at level 3."""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import shutil
import torch  # noqa: F401
from pathlib import Path

from stable_baselines3 import SAC

from sumo_env import MiniSumoEnv

CKPT_DIR = Path(__file__).with_name("checkpoints_v3")
N_EPISODES = 100
DEPLOY_LEVEL = 3


def evaluate(model: SAC, env: MiniSumoEnv, n: int) -> float:
    wins = 0
    for _ in range(n):
        obs, _ = env.reset()
        terminated = truncated = False
        ep_return = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
        if ep_return > 500.0:
            wins += 1
    return wins / n


def main() -> None:
    candidates = sorted(int(p.stem.split("_")[-2])
                        for p in CKPT_DIR.glob("sac_actor_v3_*_steps.zip"))
    if not candidates:
        print(f"No checkpoints in {CKPT_DIR.name}")
        return

    env = MiniSumoEnv(gui=False, seed=42, novamax_level=DEPLOY_LEVEL)
    results: dict[int, float] = {}
    try:
        for step in candidates:
            ckpt = CKPT_DIR / f"sac_actor_v3_{step}_steps.zip"
            model = SAC.load(str(ckpt), device="cpu")
            wr = evaluate(model, env, N_EPISODES)
            results[step] = wr
            print(f"step {step:>7d}: {wr:.1%}  ({N_EPISODES} eps @ level {DEPLOY_LEVEL})",
                  flush=True)
    finally:
        env.close()

    best_step = max(results, key=results.get)
    print(f"\nBEST: step {best_step} -> {results[best_step]:.1%}", flush=True)

    src = CKPT_DIR / f"sac_actor_v3_{best_step}_steps.zip"
    dst_best = Path(__file__).with_name("sac_actor_v3_best.zip")
    dst_main = Path(__file__).with_name("sac_actor_v3.zip")
    shutil.copy2(src, dst_best)
    shutil.copy2(src, dst_main)
    print(f"Copied {src.name} -> sac_actor_v3_best.zip and sac_actor_v3.zip",
          flush=True)

    from train_novamax import export_neural_net
    out_header = Path(__file__).with_name("neural_net_v3.h")
    best_model = SAC.load(str(dst_best), device="cpu")
    export_neural_net(best_model, out_header)
    print(f"Re-exported {out_header.name} from step {best_step}", flush=True)


if __name__ == "__main__":
    main()
