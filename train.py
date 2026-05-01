"""SAC trainer for the 4D-reactive ``MiniSumoEnv`` with a torque curriculum.

[32, 32] MLP actor:  4 inputs  -> Linear(32) ReLU
                                -> Linear(32) ReLU
                                -> Linear(2)  tanh
The 2 outputs are: [left_motor, right_motor].

The opponent torque ramps over the run: each phase fine-tunes the same
SAC checkpoint at progressively harder enemy strength. After training we
extract weights from ``model.policy.actor.latent_pi`` / ``actor.mu`` and
emit ``neural_net.h`` with ``predict_motors(...)``.
"""

from __future__ import annotations

import os
from pathlib import Path

# KMP_DUPLICATE_LIB_OK and torch-before-pybullet must come first on Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
import numpy as np

from sumo_env import MiniSumoEnv


# ---------------------------------------------------------------------------
# Hyperparameters / curriculum
# ---------------------------------------------------------------------------
# Start where SAC already wins (~50% at 2.0×) and ramp the opponent to
# 3.0× over the run. Each phase fine-tunes the same SAC checkpoint.
CURRICULUM_TORQUES = [2.0, 2.5, 3.0]
PHASE_TIMESTEPS = 200_000
TOTAL_TIMESTEPS = PHASE_TIMESTEPS * len(CURRICULUM_TORQUES)
NET_ARCH = [32, 32]
SEED = 42

OBS_DIM = 4
ACTION_DIM = 2

# Save a recoverable checkpoint every N steps. With 600k total steps and
# 25k between snapshots this gives ~24 rollback points.
CHECKPOINT_EVERY = 25_000

OUTPUT_HEADER = Path(__file__).with_name("neural_net.h")
OUTPUT_NPZ = Path(__file__).with_name("sac_actor_weights.npz")
OUTPUT_MODEL = Path(__file__).with_name("sac_actor.zip")
BEST_MODEL = Path(__file__).with_name("sac_actor_best.zip")
CHECKPOINT_DIR = Path(__file__).with_name("checkpoints")


class WinRateCallback(BaseCallback):
    """Tracks win/loss outcomes from terminal rewards and prints summaries."""

    def __init__(self, report_every: int = 10_000, best_path: Path | None = None,
                 best_min_episodes: int = 200):
        super().__init__(verbose=0)
        self.report_every = report_every
        self.recent_outcomes: list[int] = []
        self.total_wins = 0
        self.total_losses = 0
        self.total_episodes = 0
        self._next_report = report_every
        self.best_path = best_path
        self.best_min_episodes = best_min_episodes
        self.best_wr = -1.0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        rewards = self.locals.get("rewards")
        infos = self.locals.get("infos") or []
        if dones is None:
            return True
        for done, reward, _info in zip(dones, rewards, infos):
            if not done:
                continue
            # +100 win / -100 loss dominates the terminal reward;
            # threshold at +50 cleanly discriminates.
            if reward > 50.0:
                self.total_wins += 1
                self.recent_outcomes.append(1)
            else:
                self.total_losses += 1
                self.recent_outcomes.append(0)
            self.total_episodes += 1
            if len(self.recent_outcomes) > 200:
                self.recent_outcomes.pop(0)

        if self.num_timesteps >= self._next_report:
            recent = (
                sum(self.recent_outcomes) / len(self.recent_outcomes)
                if self.recent_outcomes else 0.0
            )
            total = self.total_wins / max(1, self.total_episodes)

            # Save best model on torque-weighted score: rolling_wr * torque.
            # A 51% win-rate at 2.5× (score 1.275) is genuinely stronger
            # than a 55% win-rate at 2.0× (score 1.10), even though the
            # raw rate is lower.
            # Prefer novamax_level (per-tier difficulty) — fall back to
            # the legacy enemy_torque_mult for the davo_sirad runs.
            try:
                level = int(self.model.get_env().get_attr("novamax_level")[0])
                difficulty = float(level)
            except Exception:
                try:
                    difficulty = float(
                        self.model.get_env().get_attr("enemy_torque_mult")[0]
                    )
                except Exception:
                    difficulty = 1.0
            score = recent * difficulty

            new_best = False
            if (self.best_path is not None
                    and len(self.recent_outcomes) >= self.best_min_episodes
                    and score > self.best_wr):
                self.best_wr = score
                self.model.save(str(self.best_path))
                new_best = True

            tag = "  *BEST*" if new_best else ""
            print(
                f"step {self.num_timesteps:6d}/{TOTAL_TIMESTEPS}  "
                f"episodes={self.total_episodes:4d}  "
                f"rolling_wr({len(self.recent_outcomes)})={recent:.2%}  "
                f"total_wr={total:.2%}  "
                f"score(wr*diff{difficulty:.1f})={score:.3f}{tag}",
                flush=True,
            )
            self._next_report += self.report_every
        return True


def _fmt(x: float) -> str:
    return f"{x:+.6e}f"


def _matrix_lines(mat: np.ndarray) -> str:
    rows = []
    R, C = mat.shape
    for r in range(R):
        cells = ", ".join(_fmt(float(mat[r, c])) for c in range(C))
        suffix = "," if r < R - 1 else ""
        rows.append(f"    {{ {cells} }}{suffix}")
    return "\n".join(rows)


def _vector_line(vec: np.ndarray) -> str:
    cells = ", ".join(_fmt(float(vec[i])) for i in range(vec.shape[0]))
    return f"    {cells}"


def export_neural_net(model: SAC, path: Path) -> None:
    """Pull weights from the SAC actor and emit a self-contained C++ header."""
    actor = model.policy.actor
    linears = [m for m in actor.latent_pi if isinstance(m, torch.nn.Linear)]
    if len(linears) != 2:
        raise RuntimeError(
            f"expected 2 hidden Linear layers (net_arch={NET_ARCH}), "
            f"got {len(linears)}"
        )
    L1, L2 = linears
    LOUT = actor.mu

    W1 = L1.weight.detach().cpu().numpy()                  # (32, 4)
    B1 = L1.bias.detach().cpu().numpy()                    # (32,)
    W2 = L2.weight.detach().cpu().numpy()                  # (32, 32)
    B2 = L2.bias.detach().cpu().numpy()                    # (32,)
    W_OUT = LOUT.weight.detach().cpu().numpy()             # (2, 32)
    B_OUT = LOUT.bias.detach().cpu().numpy()               # (2,)

    h1, in_dim = W1.shape
    h2, _ = W2.shape
    out_dim, _ = W_OUT.shape
    if (in_dim, h1, h2, out_dim) != (OBS_DIM, NET_ARCH[0], NET_ARCH[1], ACTION_DIM):
        raise RuntimeError(
            f"actor shape mismatch: W1={W1.shape} W2={W2.shape} W_OUT={W_OUT.shape}"
        )

    np.savez(
        OUTPUT_NPZ,
        W1=W1, B1=B1, W2=W2, B2=B2, W_OUT=W_OUT, B_OUT=B_OUT,
    )

    header = f"""// Auto-generated by train.py - do not edit by hand.
// SAC actor: input(4) -> Linear({h1}) ReLU -> Linear({h2}) ReLU -> Linear({out_dim}) tanh.
//
// Inputs (must be normalised the same way the env does):
//     front     in [0, 1]  (laser distance / 0.80 m, 1.0 = no hit)
//     left      in [0, 1]
//     right     in [0, 1]
//     last_seen in {{-1, 0, +1}}  (Left, Front, Right last detection)
//
// Outputs (tanh-squashed mean of the SAC policy distribution):
//     out_left, out_right  in [-1, +1]  (motor commands, map to PWM)

#pragma once
#include <avr/pgmspace.h>
#include <math.h>

namespace mini_sumo_ai {{

constexpr uint8_t INPUT_DIM  = {OBS_DIM};
constexpr uint8_t H1_DIM     = {h1};
constexpr uint8_t H2_DIM     = {h2};
constexpr uint8_t OUTPUT_DIM = {out_dim};

const float W1[{h1}][{in_dim}] PROGMEM = {{
{_matrix_lines(W1)}
}};
const float B1[{h1}] PROGMEM = {{
{_vector_line(B1)}
}};

const float W2[{h2}][{h1}] PROGMEM = {{
{_matrix_lines(W2)}
}};
const float B2[{h2}] PROGMEM = {{
{_vector_line(B2)}
}};

const float W_OUT[{out_dim}][{h2}] PROGMEM = {{
{_matrix_lines(W_OUT)}
}};
const float B_OUT[{out_dim}] PROGMEM = {{
{_vector_line(B_OUT)}
}};

inline float relu(float x) {{ return x > 0.0f ? x : 0.0f; }}

inline void predict_motors(
    float front, float left, float right, float last_seen,
    float &out_left, float &out_right
) {{
    const float in[INPUT_DIM] = {{ front, left, right, last_seen }};

    float h1_act[H1_DIM];
    for (uint8_t i = 0; i < H1_DIM; ++i) {{
        float s = pgm_read_float(&B1[i]);
        for (uint8_t j = 0; j < INPUT_DIM; ++j) {{
            s += pgm_read_float(&W1[i][j]) * in[j];
        }}
        h1_act[i] = relu(s);
    }}

    float h2_act[H2_DIM];
    for (uint8_t i = 0; i < H2_DIM; ++i) {{
        float s = pgm_read_float(&B2[i]);
        for (uint8_t j = 0; j < H1_DIM; ++j) {{
            s += pgm_read_float(&W2[i][j]) * h1_act[j];
        }}
        h2_act[i] = relu(s);
    }}

    float out[OUTPUT_DIM];
    for (uint8_t i = 0; i < OUTPUT_DIM; ++i) {{
        float s = pgm_read_float(&B_OUT[i]);
        for (uint8_t j = 0; j < H2_DIM; ++j) {{
            s += pgm_read_float(&W_OUT[i][j]) * h2_act[j];
        }}
        out[i] = tanhf(s);
    }}

    out_left  = out[0];
    out_right = out[1];
}}

}}  // namespace mini_sumo_ai
"""
    path.write_text(header, encoding="ascii")


def main() -> None:
    env = MiniSumoEnv(
        gui=False, seed=SEED,
        enemy_torque_multiplier=CURRICULUM_TORQUES[0],
    )

    model = SAC(
        "MlpPolicy", env,
        policy_kwargs=dict(net_arch=NET_ARCH),
        learning_rate=7.3e-4,
        buffer_size=100_000,
        batch_size=256,
        ent_coef="auto_0.1",
        gamma=0.99,
        verbose=1,
        seed=SEED,
        device="cpu",
    )

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    win_callback = WinRateCallback(report_every=10_000, best_path=BEST_MODEL)
    ckpt_callback = CheckpointCallback(
        save_freq=CHECKPOINT_EVERY,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="sac_actor",
    )
    callback = CallbackList([win_callback, ckpt_callback])

    try:
        for phase_idx, torque in enumerate(CURRICULUM_TORQUES, start=1):
            model.get_env().env_method("__setattr__", "enemy_torque_mult", torque)
            print(
                f"\n--- STARTING CURRICULUM PHASE {phase_idx}/"
                f"{len(CURRICULUM_TORQUES)}: Torque {torque}x "
                f"({PHASE_TIMESTEPS} steps) ---",
                flush=True,
            )
            model.learn(
                total_timesteps=PHASE_TIMESTEPS,
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
            f"\nTraining complete: {win_callback.total_wins} wins / "
            f"{win_callback.total_losses} losses ({overall:.2%} overall, "
            f"{recent:.2%} on last {len(win_callback.recent_outcomes)}, "
            f"best rolling {win_callback.best_wr:.2%})",
            flush=True,
        )

    # Export the BEST model (peak rolling win-rate), not the final one —
    # SAC sometimes collapses near the end. Fall back to current model if
    # we never saved a best (e.g. <200 episodes total).
    if BEST_MODEL.exists():
        export_model = SAC.load(str(BEST_MODEL), device="cpu")
        print(f"Exporting BEST model from {BEST_MODEL.name} (rolling "
              f"{win_callback.best_wr:.2%}).", flush=True)
    else:
        export_model = model
        print("No best snapshot — exporting final model.", flush=True)

    model.save(str(OUTPUT_MODEL))
    export_neural_net(export_model, OUTPUT_HEADER)
    print(
        f"Wrote {OUTPUT_HEADER.name}, {OUTPUT_NPZ.name}, "
        f"{OUTPUT_MODEL.name}, and {BEST_MODEL.name}",
        flush=True,
    )

    for stale in ("policy.h", "q_table.h", "q_table.npy",
                  "ppo_actor.zip", "ppo_actor_weights.npz"):
        path = Path(__file__).with_name(stale)
        if path.exists():
            path.unlink()


if __name__ == "__main__":
    main()
