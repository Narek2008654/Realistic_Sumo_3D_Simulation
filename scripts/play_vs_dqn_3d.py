"""Play human (blue, WASD) vs trained DQN (red) inside the 3D PyBullet sim.

Loads ``dqn_3d_bc_actor_best.pt`` (or ``--ckpt <path>``) and opens a
PyBullet GUI. Red is driven by the DQN's argmax action each tick; blue
is driven by the keyboard via the env's existing ``human_enemy=True``
mode.

Controls:
    W/A/S/D or arrows   forward / left / back / right
    Q or Esc            quit
    close window        quit

Notes:
- Safe to run while train_dqn_3d.py is still training: torch.save
  writes atomically, so the load races without corruption. Worst case
  you load a checkpoint that gets overwritten one tick later — fine.
- Discrete action space (9 actions on 3×3 motor grid), same as training.

Usage:
    python play_vs_dqn_3d.py                                 # default ckpt
    python play_vs_dqn_3d.py --ckpt dqn_3d_actor_best.pt     # smoke run
    python play_vs_dqn_3d.py --ckpt dqn_3d_smoke_best.pt     # 200k smoke
    python play_vs_dqn_3d.py --mult 3.0                      # NovaMax level
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

import torch  # noqa: F401
import pybullet as p

from train_dqn_3d import (
    DuelingQNet, NET_OBS_DIM, N_ACTIONS, NET_ARCH, BEST_PATH, build_env,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=str(BEST_PATH))
    ap.add_argument("--mult", type=float, default=1.0,
                    help="opponent torque mult (only matters if you "
                         "lose; the human IS the opponent here)")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        sys.exit(f"checkpoint not found: {ckpt_path}")

    model = DuelingQNet(NET_OBS_DIM, N_ACTIONS, hidden=NET_ARCH)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {ckpt_path.name}")
    print("Opening PyBullet GUI: human (blue, WASD) vs DQN (red).\n")

    env = build_env(
        gui=True, seed=int(time.time()),
        novamax_torque_mult=args.mult,
        human_enemy=True,
        narek_reward=False,
    )
    # No PyBullet built-in shortcuts / mouse picking — WASD only.
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)

    wins = losses = timeouts = 0

    try:
        ep = 0
        while True:
            obs, _ = env.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                if not p.isConnected(env.unwrapped._client_id):
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
                a = model.act_greedy(obs)
                obs, _, terminated, truncated, info = env.step(a)
            ep += 1
            reason = info.get("termination_reason", "?")
            # Reframe the result from the human's perspective: DQN's
            # "win" means YOU lost; DQN's loss means YOU won.
            if reason == "win":
                losses += 1
                human_result = "DQN WINS"
            elif reason in ("push_loss", "self_out", "mutual_out"):
                wins += 1
                human_result = "YOU WIN"
            else:  # timeout
                timeouts += 1
                human_result = "TIMEOUT"
            total = wins + losses + timeouts
            wr = wins / total if total else 0.0
            print(
                f"ep {ep:3d}  result={human_result:8s} ({reason:10s})  "
                f"running: YOU={wins}  DQN={losses}  T={timeouts}  "
                f"your WR={wr:.0%}",
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
