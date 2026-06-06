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
from collections import Counter
from pathlib import Path

import numpy as np
from train_dqn_3d import DuelingQNet, BEST_PATH, build_env
from opponents import HELD_OUT_OPPONENT_IDS


# Trained-against ("seen") zoo vs the held-out opponents used to measure
# zero-shot generalization. Eval runs with a clean env (no opponent/hard
# sensor DR) so win-rates are reproducible and comparable to the baseline.
SEEN_OPPONENTS = ("dodger", "spinner", "rammer", "wedger", "novamax", "charger")
HELD_OUT = HELD_OUT_OPPONENT_IDS


def run_eval(model: DuelingQNet, opp: str, mult: float, n: int, seed: int,
             safety: bool = False):
    env = build_env(
        gui=False, seed=seed,
        novamax_torque_mult=mult, force_opponent_id=opp,
        narek_reward=False, safety_override=safety,
    )
    reasons: Counter = Counter()
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
        reasons[info.get("termination_reason", "unknown")] += 1
    wins = reasons["win"]
    timeouts = reasons["timeout"]
    losses = n - wins - timeouts
    return {
        "wins": wins, "losses": losses, "timeouts": timeouts,
        # self_out = drove itself off the edge (the recoverable failure mode)
        "self_out": reasons["self_out"], "push_loss": reasons["push_loss"],
        "mutual_out": reasons["mutual_out"],
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
    ap.add_argument("--safety", action="store_true",
                    help="apply the hardcoded safety override at eval")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    # Infer architecture from the checkpoint so any net (e.g. 32x32 vs
    # 48x48) loads regardless of the trainer's current NET_ARCH.
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    h1, obs_dim = state["trunk.0.weight"].shape
    h2 = state["trunk.2.weight"].shape[0]
    n_act = state["advantage_head.weight"].shape[0]
    model = DuelingQNet(obs_dim, n_act, hidden=(h1, h2))
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {ckpt_path.name}  (obs={obs_dim}, arch=({h1},{h2}))")
    print(f"Eval: {args.n_eps} eps per opponent at mult={args.mult}, seed={args.seed}\n")

    if args.opp:
        groups = [("opponent", (args.opp,))]
    else:
        groups = [("seen", SEEN_OPPONENTS), ("held-out", HELD_OUT)]

    print(f"{'opp':10s}  {'WR':>5s}  {'W/L/T':>9s}  {'selfout':>7s}  {'mean_len':>9s}")
    for label, opps in groups:
        print(f"--- {label} " + "-" * (35 - len(label)))
        group_wins = group_n = group_self = 0
        for opp in opps:
            r = run_eval(model, opp, args.mult, args.n_eps, args.seed, args.safety)
            print(
                f"{opp:10s}  {r['wr']:>5.0%}  "
                f"{r['wins']:>2d}/{r['losses']:>2d}/{r['timeouts']:>2d}  "
                f"{r['self_out']:>7d}  {r['mean_ep_len']:>9.0f}"
            )
            group_wins += r["wins"]
            group_n += r["n"]
            group_self += r["self_out"]
        print(f"{label + ' mean':10s}  {group_wins/group_n:>5.0%}  "
              f"{group_wins:>2d}/{group_n-group_wins:>2d}  "
              f"{group_self:>7d}  (self-outs / {group_n})")


if __name__ == "__main__":
    main()
