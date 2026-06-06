"""Watch one model fight EVERY opponent, N rounds each, in one GUI window.

The opponent is switched between episodes via env.force_opponent_id, so a
single PyBullet window cycles the whole zoo + held-out set. Greedy actions.

Usage:
  python scripts/watch_gauntlet.py --ckpt checkpoints/ppo_ent0_best.pt \
      --rounds 6 --mult 3.0
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
import time

import torch  # noqa: F401  # before numpy (Windows DLL order)
import pybullet as p

from train_dqn_3d import DuelingQNet, build_env

OPPONENTS = ["dodger", "spinner", "rammer", "wedger", "novamax", "charger",
             "tracker", "feinter", "orbiter"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="checkpoints/ppo_ent0_best.pt")
    ap.add_argument("--rounds", type=int, default=6, help="episodes per opponent")
    ap.add_argument("--mult", type=float, default=3.0)
    ap.add_argument("--guard", action="store_true",
                    help="apply the deployed hardcoded action overrides "
                         "(safety + opening charge + spawn guard + anti-stall)")
    ap.add_argument("--opp", default=None,
                    help="fight only this opponent (default: the whole list)")
    ap.add_argument("--same-chassis", action="store_true",
                    help="spawn the opponent on the agent chassis (robot.urdf)")
    args = ap.parse_args()

    opponents = [args.opp] if args.opp else OPPONENTS

    ckpt = pathlib.Path(args.ckpt)
    if not ckpt.exists():
        sys.exit(f"checkpoint not found: {ckpt}")
    state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    h1, obs_dim = state["trunk.0.weight"].shape
    h2 = state["trunk.2.weight"].shape[0]
    n_act = state["advantage_head.weight"].shape[0]
    model = DuelingQNet(obs_dim, n_act, hidden=(h1, h2))
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {ckpt.name} (obs={obs_dim}, arch=({h1},{h2}))")
    print(f"Gauntlet: {args.rounds} rounds vs each of {len(opponents)} "
          f"opponents at mult={args.mult}. Q/Esc or close window to quit.\n")

    env = build_env(gui=True, seed=int(time.time()),
                    novamax_torque_mult=args.mult,
                    force_opponent_id=opponents[0], narek_reward=False,
                    safety_override=args.guard, opening_charge=args.guard,
                    spawn_guard=args.guard, antistall=args.guard,
                    enemy_as_agent=args.same_chassis)
    if args.same_chassis:
        print("opponent on the AGENT chassis (robot.urdf)")
    if args.guard:
        print("deployed action overrides ON (safety + opening + spawn + anti-stall)")
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)

    def quit_requested():
        keys = p.getKeyboardEvents()
        esc = p.B3G_ESCAPE if hasattr(p, "B3G_ESCAPE") else 27
        return any(keys.get(k, 0) & p.KEY_IS_DOWN for k in (ord("q"), ord("Q"), esc))

    try:
        for opp in opponents:
            env.unwrapped.force_opponent_id = opp
            wins = 0
            for r in range(args.rounds):
                if not p.isConnected(env.unwrapped._client_id):
                    print("GUI window closed — exiting.")
                    return
                obs, _ = env.reset()
                terminated = truncated = False
                while not (terminated or truncated):
                    if not p.isConnected(env.unwrapped._client_id):
                        print("GUI window closed — exiting.")
                        return
                    if quit_requested():
                        print("\nQuit signal received.")
                        return
                    obs, _, terminated, truncated, info = env.step(model.act_greedy(obs))
                reason = info.get("termination_reason", "?")
                if reason == "win":
                    wins += 1
                print(f"  {opp:8s} round {r+1}/{args.rounds}: {reason}")
            print(f"== {opp:8s}: {wins}/{args.rounds} wins ==\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
