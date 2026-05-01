"""Fine-tune the v2 NovaMax SAC against the upgraded bullet NovaMax.

The v2 model (sac_actor_novamax_v2.zip, 50% deterministic) was trained
against a 400 RPM NovaMax with weak torque. NovaMax now packs:
    - 800 RPM motors (83.78 rad/s, 1.38 m/s top speed)
    - 0.50 N·m stall torque (4× the agent)
    - 0.45 kg chassis (heavier hits)
plus the new instant-stop / per-substep line-sensor watchdog so it doesn't
ballistically launch off the dohyo when the edge fires.

Curriculum (3 phases × 200k = 600k total):
    Phase 1: novamax_level = 1 (matches agent)
    Phase 2: novamax_level = 2 (medium 40 rad/s, 0.25 N·m)
    Phase 3: novamax_level = 3 (full real spec, 83.78 rad/s, 0.50 N·m)

Outputs land in ``sac_actor_novamax_v3.zip`` / ``_best.zip``,
``neural_net_novamax_v3.h``, and ``checkpoints_novamax_v3/`` so v2 stays
on disk untouched.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

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
CURRICULUM: list[tuple[int, int]] = [
    (1, 200_000),    # bootstrap on the new physics
    (2, 200_000),
    (3, 200_000),
]
TOTAL_TIMESTEPS = sum(steps for _, steps in CURRICULUM)
SEED = 42
CHECKPOINT_EVERY = 25_000

HERE = Path(__file__).parent
SOURCE_MODEL = HERE / "sac_actor_novamax_v2.zip"   # v2 — read only
OUTPUT_MODEL = HERE / "sac_actor_novamax_v3.zip"
BEST_MODEL = HERE / "sac_actor_novamax_v3_best.zip"
OUTPUT_HEADER = HERE / "neural_net_novamax_v3.h"
CHECKPOINT_DIR = HERE / "checkpoints_novamax_v3"


def main() -> None:
    if not SOURCE_MODEL.exists():
        raise FileNotFoundError(
            f"{SOURCE_MODEL.name} not found — train_novamax_v2.py must finish first."
        )

    env = MiniSumoEnv(
        gui=False, seed=SEED,
        novamax_level=CURRICULUM[0][0],
    )

    print(f"Loading v2 NovaMax model from {SOURCE_MODEL.name}…", flush=True)
    # Resetting ent_coef to auto_0.1 so SAC actively re-explores against
    # the new bullet physics instead of inheriting v2's converged value.
    model = SAC.load(
        str(SOURCE_MODEL),
        env=env,
        device="cpu",
        custom_objects={"ent_coef": "auto_0.1"},
    )

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    win_callback = WinRateCallback(report_every=10_000, best_path=BEST_MODEL)
    ckpt_callback = CheckpointCallback(
        save_freq=CHECKPOINT_EVERY,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="sac_actor_novamax_v3",
    )
    callback = CallbackList([win_callback, ckpt_callback])

    try:
        for phase_idx, (level, steps) in enumerate(CURRICULUM, start=1):
            model.get_env().env_method("__setattr__", "novamax_level", level)
            # Make sure novamax_torque_mult override is off so the level
            # table is the active difficulty knob.
            model.get_env().env_method("__setattr__", "novamax_torque_mult", None)
            print(
                f"\n--- PHASE {phase_idx}/{len(CURRICULUM)}: "
                f"novamax_level={level}  ({steps} steps) ---",
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
            f"\nv3 fine-tune complete: {win_callback.total_wins} wins / "
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
