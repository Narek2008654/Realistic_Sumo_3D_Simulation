"""Watch the trained DQN play inside the 3D PyBullet sim.

Loads ``dqn_actor_best.pt`` (the 2D-trained model copied here) and
opens MiniSumoEnv with ``gui=True``, which spawns PyBullet's debug
window. The env already paces stepSimulation at real time when
gui=True, so playback runs at the env's 25 Hz tick rate naturally.

Controls:
    Q          quit cleanly (also Ctrl-C in the terminal)
    Window-X   close window also quits

Usage:
    python watch_3d.py                          # zoo, mult=1.0
    python watch_3d.py --opp novamax --mult 3.0
    python watch_3d.py --ckpt dqn_actor_final.pt
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
from pathlib import Path

import torch  # noqa: F401  # import first to avoid Windows fbgemm.dll race
import numpy as np
import pybullet as p

from sumo_env import MiniSumoEnv
from train_dqn_3d import DuelingQNet, OBS_DIM, N_ACTIONS, NET_ARCH, BEST_PATH


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--opp", default=None, help="pin opponent (else zoo)")
    ap.add_argument("--mult", type=float, default=1.0)
    ap.add_argument("--ckpt", default=str(BEST_PATH))
    ap.add_argument(
        "--n-episodes", type=int, default=0,
        help="stop after N episodes (default 0 = run forever)",
    )
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        sys.exit(f"checkpoint not found: {ckpt_path}")

    model = DuelingQNet(OBS_DIM, N_ACTIONS, hidden=NET_ARCH)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {ckpt_path.name}")
    print(f"Opening PyBullet GUI: opp={args.opp or 'zoo'} mult={args.mult}\n")

    env = MiniSumoEnv(
        gui=True, seed=int(time.time()),
        novamax_torque_mult=args.mult,
        force_opponent_id=args.opp,
        action_space_kind="discrete",
        narek_reward=False,
    )
    # Suppress PyBullet's built-in keyboard shortcuts (W/G/etc) and
    # mouse picking so the window is "view only" — no accidental
    # interaction with the bots.
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
            terminated = truncated = False
            while not (terminated or truncated):
                if not p.isConnected(env._client_id):
                    print("GUI window closed — exiting.")
                    return
                # Quit on Q held inside the window
                keys = p.getKeyboardEvents()
                if (
                    keys.get(ord("q"), 0) & p.KEY_IS_DOWN
                    or keys.get(ord("Q"), 0) & p.KEY_IS_DOWN
                    or keys.get(p.B3G_ESCAPE if hasattr(p, "B3G_ESCAPE") else 27, 0) & p.KEY_IS_DOWN
                ):
                    print("\nQuit signal received.")
                    return
                a = model.act_greedy(obs)
                obs, _, terminated, truncated, info = env.step(a)
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
