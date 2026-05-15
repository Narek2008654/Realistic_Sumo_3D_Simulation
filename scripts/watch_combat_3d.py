"""Watch the scripted CombatPolicy play inside the 3D PyBullet sim.

Same shape as watch_3d.py but uses the 60-line hand-coded controller
in combat_policy.py — no neural network. Useful for sanity-checking
the env (engagement, edge handling) and for comparing against the
trained DQN viewer.

Usage:
    python watch_combat_3d.py                          # random zoo, mult=1.0
    python watch_combat_3d.py --opp novamax --mult 3.0
    python watch_combat_3d.py --n-episodes 20
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# --- sys.path shim (added by reorg) ---
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))


import argparse
import sys
import time

import torch  # noqa: F401  # import first to avoid Windows fbgemm.dll race
import numpy as np
import pybullet as p

from sumo_env import MiniSumoEnv
from combat_policy import CombatPolicy


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--opp", default=None, help="pin opponent (else zoo)")
    ap.add_argument("--mult", type=float, default=1.0)
    ap.add_argument(
        "--n-episodes", type=int, default=0,
        help="stop after N episodes (default 0 = run forever)",
    )
    args = ap.parse_args()

    policy = CombatPolicy()
    print(f"Opening PyBullet GUI: opp={args.opp or 'zoo'} mult={args.mult}\n")

    env = MiniSumoEnv(
        gui=True, seed=int(time.time()),
        novamax_torque_mult=args.mult,
        force_opponent_id=args.opp,
        action_space_kind="continuous",
        narek_reward=False,
    )
    # View-only window: suppress PyBullet's built-in keyboard shortcuts
    # and mouse picking so accidental clicks can't grab the bots.
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)

    wins = 0
    losses = 0
    timeouts = 0

    try:
        ep = 0
        while True:
            if args.n_episodes and ep >= args.n_episodes:
                break
            obs, _ = env.reset()
            policy.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                if not p.isConnected(env._client_id):
                    print("GUI window closed — exiting.")
                    return
                keys = p.getKeyboardEvents()
                if (
                    keys.get(ord("q"), 0) & p.KEY_IS_DOWN
                    or keys.get(ord("Q"), 0) & p.KEY_IS_DOWN
                    or keys.get(p.B3G_ESCAPE if hasattr(p, "B3G_ESCAPE") else 27, 0)
                    & p.KEY_IS_DOWN
                ):
                    print("\nQuit signal received.")
                    return
                l, r = policy(obs)
                obs, _, terminated, truncated, info = env.step(
                    np.array([l, r], dtype=np.float32)
                )
            ep += 1
            reason = info.get("termination_reason", "?")
            opp = info.get("opponent_id", "?")
            if reason == "win":
                wins += 1
            elif reason == "timeout":
                timeouts += 1
            else:
                losses += 1
            total = wins + losses + timeouts
            wr = wins / total if total else 0.0
            print(
                f"ep {ep:3d}  vs {opp:8s}  result={reason:10s}  "
                f"running: W={wins}  L={losses}  T={timeouts}  WR={wr:.0%}",
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
