"""Quick eval: load dqn_actor_best.pt, run deterministic greedy
rollouts vs every opponent, print per-opponent WR.

Usage:
    python eval_best.py                          # default 20 eps, mult=1.0
    python eval_best.py --n-eps 50 --mult 3.0    # tougher eval
    python eval_best.py --opp novamax            # single opponent

Safe to run while train_dqn.py is still training — torch.save is
atomic so we won't race against a checkpoint write.
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# --- sys.path shim (added by reorg) ---
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))


import torch
import argparse
from pathlib import Path

import numpy as np
from sumo_env import MiniSumoEnv
from train_dqn_3d import DuelingQNet, OBS_DIM, N_ACTIONS, NET_ARCH, BEST_PATH


OPPONENTS = ("dodger", "spinner", "rammer", "wedger", "novamax", "charger")


def run_eval(model: DuelingQNet, opp: str, mult: float, n: int, seed: int):
    env = MiniSumoEnv(
        gui=False, seed=seed,
        novamax_torque_mult=mult, force_opponent_id=opp,
        action_space_kind="discrete", narek_reward=False,
    )
    wins = 0
    losses = 0
    timeouts = 0
    ep_lens = []
    obs, _ = env.reset(seed=seed)
    for ep in range(n):
        if ep > 0:
            obs, _ = env.reset()
        terminated = truncated = False
        for k in range(600):
            a = model.act_greedy(obs)
            obs, _, terminated, truncated, info = env.step(a)
            if terminated or truncated:
                break
        ep_lens.append(k + 1)
        reason = info.get("termination_reason", "unknown")
        if reason == "win":
            wins += 1
        elif reason == "timeout":
            timeouts += 1
        else:
            losses += 1
    return {
        "wins": wins, "losses": losses, "timeouts": timeouts,
        "n": n, "wr": wins / n, "mean_ep_len": float(np.mean(ep_lens)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eps", type=int, default=20)
    ap.add_argument("--mult", type=float, default=1.0)
    ap.add_argument("--opp", default=None, help="single opponent name; default = all")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument(
        "--ckpt", default=str(BEST_PATH),
        help=f"path to .pt checkpoint (default {BEST_PATH.name})",
    )
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    model = DuelingQNet(OBS_DIM, N_ACTIONS, hidden=NET_ARCH)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {ckpt_path.name}")
    print(f"Eval: {args.n_eps} eps per opponent at mult={args.mult}, seed={args.seed}\n")

    opps = (args.opp,) if args.opp else OPPONENTS

    print(f"{'opp':10s}  {'WR':>5s}  {'W/L/T':>9s}  {'mean_len':>9s}")
    print("-" * 40)
    overall_wins = 0
    overall_n = 0
    for opp in opps:
        r = run_eval(model, opp, args.mult, args.n_eps, args.seed)
        print(
            f"{opp:10s}  {r['wr']:>5.0%}  "
            f"{r['wins']:>2d}/{r['losses']:>2d}/{r['timeouts']:>2d}  "
            f"{r['mean_ep_len']:>9.0f}"
        )
        overall_wins += r["wins"]
        overall_n += r["n"]
    print("-" * 40)
    print(f"{'overall':10s}  {overall_wins/overall_n:>5.0%}  "
          f"{overall_wins:>2d}/{overall_n-overall_wins:>2d}")


if __name__ == "__main__":
    main()
