"""V3 fine-tune: load the noise-hardened V2 champion and train it against
the fixed Level-3 NovaMax (800 RPM, 0.50 N·m, controlled edge recovery).

* Source: sac_actor_novamax_v2_best.zip
* Curriculum: 150k steps at novamax_level=3 (full bullet spec)
* ent_coef reset to 'auto_0.1' so SAC actively re-explores against the
  new opponent physics instead of inheriting V2's converged value.
* Outputs land in train_v3-suffixed files; V2 artefacts stay intact.
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
# Hyperparameters
# ---------------------------------------------------------------------------
TOTAL_TIMESTEPS = 150_000
NOVAMAX_LEVEL = 3
SEED = 42
CHECKPOINT_EVERY = 10_000

HERE = Path(__file__).parent
SOURCE_MODEL = HERE / "sac_actor_novamax_v2_best.zip"   # noise-hardened champion
OUTPUT_MODEL = HERE / "sac_actor_v3.zip"
BEST_MODEL = HERE / "sac_actor_v3_best.zip"
OUTPUT_HEADER = HERE / "neural_net_v3.h"
CHECKPOINT_DIR = HERE / "checkpoints_v3"


def main() -> None:
    if not SOURCE_MODEL.exists():
        raise FileNotFoundError(
            f"{SOURCE_MODEL.name} not found. Run train_novamax_v2.py first."
        )

    env = MiniSumoEnv(gui=False, seed=SEED, novamax_level=NOVAMAX_LEVEL)

    print(f"Loading V2 champion from {SOURCE_MODEL.name}…", flush=True)
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
        name_prefix="sac_actor_v3",
    )
    callback = CallbackList([win_callback, ckpt_callback])

    print(f"\n--- V3 fine-tune: novamax_level={NOVAMAX_LEVEL} "
          f"({TOTAL_TIMESTEPS} steps) ---", flush=True)

    try:
        # Force the level-3 difficulty knob and clear any stray override.
        model.get_env().env_method("__setattr__", "novamax_level", NOVAMAX_LEVEL)
        model.get_env().env_method("__setattr__", "novamax_torque_mult", None)
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
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
            f"\nV3 fine-tune complete: {win_callback.total_wins} wins / "
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
