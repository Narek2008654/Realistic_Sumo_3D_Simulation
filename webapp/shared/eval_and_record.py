"""Detached eval + trajectory recorder for a training job (E1f).

Invoked by :mod:`webapp.shared.checkpoint_hook` as a subprocess::

    python -m webapp.shared.eval_and_record <args.json>

The args JSON is written by the hook and carries everything the subprocess
needs to re-do its work without inheriting the trainer's state::

    {
      "snapshot": "<job_dir>/snapshots/ckpt_<step>.pt",
      "job_dir": "<job_dir>",
      "step": <int>,
      "opponents": ["novamax", ...],
      "mult": 3.0,
      "n_eval": 10,
      "seed": 4242,
      "hardware_spec": {<HardwareSpec.to_dict()>} | null
    }

It:
  1. loads the snapshot (architecture inferred from the weights, exactly as
     ``eval_best`` does, so a PPO or DQN checkpoint both load);
  2. builds a headless env from the job's hardware_spec;
  3. runs ``eval_best.run_eval`` over the opponent set at ``mult`` and
     aggregates the metrics;
  4. records ONE greedy rollout's kinematic trajectory to
     ``<job_dir>/trajectories/<step>.json``;
  5. appends a ``{"t": "checkpoint", ...}`` JSONL line to
     ``<job_dir>/progress.jsonl``.

This module imports torch/pybullet lazily inside ``main`` so the hook (which
only needs to spawn it) and the unit tests stay light.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Repo root on path so `train_dqn_3d`, `scripts.eval_best`, `sumo_env`
# import cleanly whether launched as a module or a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _record_trajectory(env, model, max_steps: int = 600) -> dict:
    """Run ONE greedy rollout and capture per-frame agent + enemy poses.

    Returns a JSON-able dict with ``dt``, ``dohyo_radius``, ``frames``
    (each ``{"agent": {"p", "q"}, "enemy": {"p", "q"}}``) and ``outcome``
    (``{"winner", "reason"}``). Kinematics only — geometry is the
    frontend's job.
    """
    import pybullet as p
    from sumo_env import STEP_DT_SECONDS

    base = env.unwrapped

    def _pose(body_id):
        pos, orn = p.getBasePositionAndOrientation(body_id)
        return {"p": [float(v) for v in pos], "q": [float(v) for v in orn]}

    frames: list[dict] = []
    obs, _ = env.reset()
    reason = "unknown"
    # Capture the initial frame, then one frame after each step.
    frames.append({"agent": _pose(base.robot_id), "enemy": _pose(base.enemy_id)})
    for _ in range(max_steps):
        a = model.act_greedy(obs)
        obs, _, terminated, truncated, info = env.step(a)
        frames.append(
            {"agent": _pose(base.robot_id), "enemy": _pose(base.enemy_id)}
        )
        if terminated or truncated:
            reason = info.get("termination_reason", "unknown")
            break

    winner = "agent" if reason == "win" else (
        "enemy" if reason in ("push_loss", "self_out") else None
    )
    return {
        "dt": float(STEP_DT_SECONDS),
        # Actual ring size from the env's spec, so a resized dohyo replays at
        # the right size rather than the module default.
        "dohyo_radius": float(base.hw_spec.dohyo.radius_m),
        "frames": frames,
        "outcome": {"winner": winner, "reason": reason},
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m webapp.shared.eval_and_record <args.json>",
              file=sys.stderr)
        return 2

    args = json.loads(Path(argv[0]).read_text(encoding="utf-8"))

    import torch
    from train_dqn_3d import DuelingQNet, build_env
    from scripts.eval_best import run_eval
    from webapp.shared.hardware_spec import HardwareSpec

    snapshot = Path(args["snapshot"])
    job_dir = Path(args["job_dir"])
    step = int(args["step"])
    opponents = args.get("opponents") or ["novamax"]
    mult = float(args.get("mult", 1.0))
    n_eval = int(args.get("n_eval", 10))
    seed = int(args.get("seed", 4242))
    hw_dict = args.get("hardware_spec")
    hw_spec = HardwareSpec.from_dict(hw_dict) if hw_dict else None

    # Infer architecture from the checkpoint so DQN/PPO of any width loads.
    state = torch.load(str(snapshot), map_location="cpu", weights_only=True)
    h1, obs_dim = state["trunk.0.weight"].shape
    h2 = state["trunk.2.weight"].shape[0]
    n_act = state["advantage_head.weight"].shape[0]
    model = DuelingQNet(obs_dim, n_act, hidden=(h1, h2))
    model.load_state_dict(state)
    model.eval()

    # --- metrics: aggregate run_eval over the opponent set ---
    per_opp = {}
    tot_w = tot_l = tot_s = tot_n = 0
    len_sum = 0.0
    for opp in opponents:
        r = run_eval(model, opp, mult, n_eval, seed)
        per_opp[opp] = r
        tot_w += r["wins"]
        tot_l += r["losses"]
        tot_s += r["self_out"]
        tot_n += r["n"]
        len_sum += r["mean_ep_len"] * r["n"]
    metrics = {
        "wins": tot_w,
        "losses": tot_l,
        "self_out": tot_s,
        "n": tot_n,
        "wr": (tot_w / tot_n) if tot_n else 0.0,
        "mean_ep_len": (len_sum / tot_n) if tot_n else 0.0,
        "per_opponent": {k: v for k, v in per_opp.items()},
        "mult": mult,
    }

    # --- trajectory: one greedy rollout vs the first opponent ---
    traj_dir = job_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_path = traj_dir / f"{step}.json"
    env = build_env(
        gui=False, seed=seed,
        novamax_torque_mult=mult, force_opponent_id=opponents[0],
        narek_reward=False,
        **({"hardware_spec": hw_spec} if hw_spec is not None else {}),
    )
    try:
        traj = _record_trajectory(env, model)
    finally:
        env.close()
    traj_path.write_text(json.dumps(traj), encoding="utf-8")

    # --- progress event (append-only JSONL, one object per line) ---
    event = {
        "t": "checkpoint",
        "step": step,
        "snapshot": str(snapshot),
        "eval": metrics,
        "trajectory": str(traj_path),
    }
    progress_path = job_dir / "progress.jsonl"
    with open(progress_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

    print(f"[eval_and_record] step={step} wr={metrics['wr']:.2%} "
          f"-> {progress_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
