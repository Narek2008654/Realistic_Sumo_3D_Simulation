"""Train SAC from scratch in the noise-injected NovaMax environment.

The 'Born in Chaos' run — every episode the agent ever sees has spawn yaw
jitter, spawn-radius jitter, NovaMax motor jitter, and a 5% sensor-blink,
so the policy can't learn the deterministic head-on timing hack that
plagued the v1 fine-tune. SAC starts from random weights with
ent_coef=auto_0.1 for steady exploration.

Curriculum (4 phases × 200k each = 800k total, each phase calls
model.learn with reset_num_timesteps=False):
    Phase 1: novamax_torque_mult = 1.0  (matches agent)
    Phase 2: novamax_torque_mult = 1.5
    Phase 3: novamax_torque_mult = 2.0
    Phase 4: novamax_torque_mult = 3.0

Outputs land in ``sac_actor_novamax_v2.zip`` / ``_best.zip``,
``neural_net_novamax_v2.h``, and ``checkpoints_novamax_v2/`` so the v1
NovaMax artefacts (``sac_actor_novamax.zip``, ``neural_net_novamax.h``)
stay intact.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# torch must be imported before pybullet on Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: F401
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from sumo_env import MiniSumoEnv
from train import (
    NET_ARCH,
    WinRateCallback,
)
from train_novamax import export_neural_net


# ---------------------------------------------------------------------------
# Hyperparameters / curriculum
# ---------------------------------------------------------------------------
CURRICULUM: list[tuple[float, int]] = [
    (1.0, 200_000),
    (1.5, 200_000),
    (2.0, 200_000),
    (3.0, 200_000),
]
TOTAL_TIMESTEPS = sum(steps for _, steps in CURRICULUM)
SEED = 42
CHECKPOINT_EVERY = 25_000

HERE = Path(__file__).parent
OUTPUT_MODEL = HERE / "sac_actor_novamax_v2.zip"
BEST_MODEL = HERE / "sac_actor_novamax_v2_best.zip"
OUTPUT_HEADER = HERE / "neural_net_novamax_v2.h"
CHECKPOINT_DIR = HERE / "checkpoints_novamax_v2"


def main() -> None:
    env = MiniSumoEnv(
        gui=False, seed=SEED,
        novamax_torque_mult=CURRICULUM[0][0],
    )

    print("Starting Born-in-Chaos curriculum from scratch (no checkpoint)…",
          flush=True)
    model = SAC(
        "MlpPolicy", env,
        policy_kwargs=dict(net_arch=NET_ARCH),
        ent_coef="auto_0.1",
        batch_size=256,
        verbose=1,
        seed=SEED,
        device="cpu",
    )

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    win_callback = WinRateCallback(report_every=10_000, best_path=BEST_MODEL)
    ckpt_callback = CheckpointCallback(
        save_freq=CHECKPOINT_EVERY,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="sac_actor_novamax_v2",
    )
    callback = CallbackList([win_callback, ckpt_callback])

    try:
        for phase_idx, (mult, steps) in enumerate(CURRICULUM, start=1):
            model.get_env().env_method("__setattr__", "novamax_torque_mult", mult)
            print(
                f"\n--- PHASE {phase_idx}/{len(CURRICULUM)}: "
                f"novamax_torque_mult={mult}x  ({steps} steps) ---",
                flush=True,
            )
            model.learn(
                total_timesteps=steps,
                callback=callback,
                progress_bar=False,
                reset_num_timesteps=False,
            )
    finally:
        env.close()

    if win_callback.total_episodes:
        overall = win_callback.total_wins / win_callback.total_episodes
        recent = (
            sum(win_callback.recent_outcomes) / len(win_callback.recent_outcomes)
            if win_callback.recent_outcomes else 0.0
        )
        print(
            f"\nBorn-in-Chaos run complete: {win_callback.total_wins} wins / "
            f"{win_callback.total_losses} losses ({overall:.2%} overall, "
            f"{recent:.2%} on last {len(win_callback.recent_outcomes)}, "
            f"best score {win_callback.best_wr:.3f})",
            flush=True,
        )

    if BEST_MODEL.exists():
        export_model = SAC.load(str(BEST_MODEL), device="cpu")
        shutil.copy2(BEST_MODEL, OUTPUT_MODEL)
        print(f"Exporting BEST snapshot from {BEST_MODEL.name}.", flush=True)
    else:
        export_model = model
        model.save(str(OUTPUT_MODEL))
        print("No best snapshot — exporting final model.", flush=True)

    export_neural_net(export_model, OUTPUT_HEADER)
    print(
        f"Wrote {OUTPUT_HEADER.name}, {OUTPUT_MODEL.name}, "
        f"and {BEST_MODEL.name}",
        flush=True,
    )


if __name__ == "__main__":
    main()
