"""Training-config recommender — simple, documented heuristics.

Given a :class:`HardwareSpec` and a training ``mode`` ("scratch" | "finetune"),
suggest a sensible starting :class:`TrainingConfig`-shaped dict for the UI:
algorithm, step budget, eval cadence, net architecture, the curriculum
starting torque-multiplier, a small hyperparameter block, and a wall-clock
estimate.

These are deliberately coarse rules of thumb, not tuned constants — the UI
shows them as editable defaults. Everything is pure arithmetic over the
spec's derived dims, so this module is dependency-light and trivially
testable.
"""

from __future__ import annotations

from typing import Any

from webapp.shared.hardware_spec import HardwareSpec

__all__ = ["recommend"]

# Reference NovaMax drivetrain torque the online curriculum is calibrated
# against — the default robot's max_torque_nm (sumo_env.py:148). The start
# curriculum multiplier scales inversely with how strong the trained robot is
# relative to this reference: a weaker robot starts against a gentler opponent.
_REFERENCE_TORQUE_NM = HardwareSpec.default().drivetrain.max_torque_nm  # 0.12

# Throughput assumption for the wall-clock estimate (env steps / second on a
# typical CPU box). Coarse; the UI shows it as an estimate only.
_ENV_STEPS_PER_SEC = 250.0

# Step budgets per mode (env steps).
_SCRATCH_STEPS = 1_000_000
_FINETUNE_STEPS = 400_000
_DEFAULT_EVAL_EVERY = 100_000


def _net_arch(spec: HardwareSpec) -> list[int]:
    """Pick a two-layer MLP width from the obs/action dims.

    Default contract (21 obs / 9 actions) -> 32x32, matching the committed
    Stage-A checkpoints. Wider obs or action spaces bump the width one notch
    so capacity scales with the problem, capped to keep firmware-export
    feasible on the Nano.
    """
    width = 32
    if spec.obs_dim > 32 or spec.action_dim > 12:
        width = 48
    if spec.obs_dim > 64 or spec.action_dim > 24:
        width = 64
    return [width, width]


def _start_mult(spec: HardwareSpec) -> float:
    """Curriculum starting torque-multiplier from drivetrain strength.

    A robot weaker than the reference faces a proportionally weaker opponent
    to start (mult < 1), a stronger one can start harder. Clamped to a sane
    [0.5, 3.0] band so the recommendation is always trainable.
    """
    torque = spec.drivetrain.max_torque_nm
    if torque <= 0:
        return 1.0
    ratio = torque / _REFERENCE_TORQUE_NM
    return round(max(0.5, min(3.0, ratio)), 3)


def _algo(mode: str) -> str:
    """DQN for scratch (BC pretrain bootstraps it well); PPO for finetune
    polishing is also viable, but we default both to DQN for the LITE flow
    because its BC-pretrain path is the most reliable end-to-end."""
    return "dqn"


def recommend(spec: HardwareSpec, mode: str) -> dict[str, Any]:
    """Return a recommended training configuration for ``spec`` + ``mode``.

    ``mode`` is ``"scratch"`` (train a fresh net) or ``"finetune"`` (continue
    from a base model). Unknown modes are treated as scratch.

    The returned dict has stable keys::

        {
          "algo": "dqn" | "ppo",
          "total_steps": int,
          "eval_every": int,
          "net_arch": [int, int],
          "start_mult": float,
          "hyperparams": {...},
          "est_minutes": float,
        }
    """
    mode = (mode or "scratch").lower()
    is_finetune = mode == "finetune"

    total_steps = _FINETUNE_STEPS if is_finetune else _SCRATCH_STEPS
    eval_every = _DEFAULT_EVAL_EVERY
    net_arch = _net_arch(spec)
    algo = _algo(mode)

    # Finetune runs at full power (the curriculum's top torque-mult), matching
    # the trainers' resume default; scratch starts from the drivetrain-derived
    # mult so a weaker robot faces a gentler opponent to begin with.
    start_mult = 3.0 if is_finetune else _start_mult(spec)

    # A small, documented hyperparameter block of the knobs the trainer seam
    # honours. Defaults mirror the trainer module constants (train_dqn_3d.py:
    # ONLINE_LR/GAMMA/N_STEP/TAU; train_ppo_3d.py: LR/GAMMA/ENT_COEF/CLIP).
    # Finetune uses a lower LR since it refines rather than learns afresh.
    hyperparams = {
        "lr": 1e-4 if is_finetune else 3e-4,
        "net_arch": net_arch,
        "start_mult": start_mult,
        "gamma": 0.99,
        # DQN-specific (train_dqn_3d.py defaults).
        "n_step": 3,
        "tau": 0.005,
        # PPO-specific (train_ppo_3d.py defaults).
        "ent_coef": 0.02,
        "clip": 0.2,
    }

    est_minutes = round(total_steps / _ENV_STEPS_PER_SEC / 60.0, 1)

    return {
        "algo": algo,
        "total_steps": total_steps,
        "eval_every": eval_every,
        "net_arch": net_arch,
        "start_mult": start_mult,
        "hyperparams": hyperparams,
        "est_minutes": est_minutes,
    }
