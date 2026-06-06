"""Human play data collector for the DQN offline pretrain.

Opens a separate PyBullet GUI window. You drive the agent with WASD
(or arrow keys); the opponent is whichever scripted bot you pin via
the --opp flag. Per-tick (obs, action_idx, reward, next_obs, done)
transitions are buffered. On a WINNING episode, the buffer is
appended to ``bc_dataset_human.npz`` (separate file from the
scripted-collected ``bc_dataset_v5.npz`` so it stays intact as
backup). On loss/timeout, the buffer is discarded.

Keyboard mapping (matches sumo_env.DISCRETE_ACTION_MAP, 9 discrete
actions). Action = (left_motor, right_motor) in {-1, 0, +1}:

    (no keys)    -> ( 0,  0)  idx 4   idle
    W            -> (+1, +1)  idx 8   forward
    S            -> (-1, -1)  idx 0   reverse
    A            -> (-1, +1)  idx 2   tank-pivot left
    D            -> (+1, -1)  idx 6   tank-pivot right
    W + A        -> ( 0, +1)  idx 5   forward + curve left
    W + D        -> (+1,  0)  idx 7   forward + curve right
    S + A        -> ( 0, -1)  idx 3   reverse + curve left
    S + D        -> (-1,  0)  idx 1   reverse + curve right

Press Q (or ESC) inside the GUI window to quit cleanly.

Usage:
    conda activate sumo
    python human_play.py --opp novamax --mult 1.0 --n-wins 20
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# --- sys.path shim (added by reorg) ---
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))


# torch is imported BEFORE pybullet to avoid the Windows fbgemm.dll
# load-order race we hit earlier; even though this script doesn't
# need torch directly, the import side-effects matter.
import torch  # noqa: F401

import argparse
import time
from pathlib import Path

import numpy as np
import pybullet as p

from sumo_env import MiniSumoEnv, DISCRETE_ACTION_MAP
from obs_stack import RawDistanceStack, DEFAULT_STACK_K


# ---------------------------------------------------------------------------
# Keyboard -> action index mapping
# ---------------------------------------------------------------------------
KEY_W = ord("w")
KEY_W_UP = ord("W")
KEY_S = ord("s")
KEY_S_UP = ord("S")
KEY_A = ord("a")
KEY_A_UP = ord("A")
KEY_D = ord("d")
KEY_D_UP = ord("D")
KEY_Q = ord("q")
KEY_Q_UP = ord("Q")
KEY_ESC = 27  # PyBullet maps Escape to ASCII 27


def _idx(pair: tuple[float, float]) -> int:
    return DISCRETE_ACTION_MAP.index(pair)


def read_action(keys: dict) -> int:
    """Translate the current PyBullet keyboard event dict into one of
    the 9 discrete action indices."""
    DOWN = p.KEY_IS_DOWN

    def held(*codes: int) -> bool:
        return any((keys.get(c, 0) & DOWN) != 0 for c in codes)

    fwd = held(KEY_W, KEY_W_UP, p.B3G_UP_ARROW)
    bwd = held(KEY_S, KEY_S_UP, p.B3G_DOWN_ARROW)
    lft = held(KEY_A, KEY_A_UP, p.B3G_LEFT_ARROW)
    rgt = held(KEY_D, KEY_D_UP, p.B3G_RIGHT_ARROW)

    if fwd and not bwd:
        if lft and not rgt:
            return _idx((0.0, 1.0))   # forward + curve left
        if rgt and not lft:
            return _idx((1.0, 0.0))   # forward + curve right
        return _idx((1.0, 1.0))        # straight forward
    if bwd and not fwd:
        if lft and not rgt:
            return _idx((0.0, -1.0))  # reverse + curve left
        if rgt and not lft:
            return _idx((-1.0, 0.0))  # reverse + curve right
        return _idx((-1.0, -1.0))     # straight reverse
    if lft and not rgt:
        return _idx((-1.0, 1.0))      # tank-pivot left
    if rgt and not lft:
        return _idx((1.0, -1.0))      # tank-pivot right
    return _idx((0.0, 0.0))            # idle


def quit_requested(keys: dict) -> bool:
    DOWN = p.KEY_IS_DOWN
    if (keys.get(KEY_Q, 0) & DOWN) or (keys.get(KEY_Q_UP, 0) & DOWN):
        return True
    if keys.get(KEY_ESC, 0) & DOWN:
        return True
    return False


# ---------------------------------------------------------------------------
# Dataset persistence (append mode)
# ---------------------------------------------------------------------------
def append_to_npz(
    path: Path,
    obs: np.ndarray, act: np.ndarray, rew: np.ndarray,
    next_obs: np.ndarray, done: np.ndarray,
) -> None:
    """Append the given arrays to an existing .npz, or create it.

    The schema matches train_dqn.py's BC_DATASET_PATH format so the
    Phase-0a loader can read this file directly (or the merged loader
    can concatenate it with bc_dataset_v5.npz).
    """
    if path.exists():
        old = np.load(path)
        obs = np.concatenate([old["obs"], obs], axis=0)
        act = np.concatenate([old["act"], act], axis=0)
        rew = np.concatenate([old["rew"], rew], axis=0)
        next_obs = np.concatenate([old["next"], next_obs], axis=0)
        done = np.concatenate([old["done"], done], axis=0)
    np.savez(path, obs=obs, act=act, rew=rew, next=next_obs, done=done)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--opp", default="novamax",
        choices=("dodger", "spinner", "rammer", "wedger", "novamax", "charger"),
        help="opponent to fight (default novamax)",
    )
    ap.add_argument(
        "--mult", type=float, default=3.0,
        help="opponent torque multiplier (0.3-3.0). Default 3.0 = "
             "full bullet (800 RPM, 0.50 N·m), human agent stays at "
             "spec 400 RPM, 0.12 N·m — deploy-realistic mismatch.",
    )
    ap.add_argument(
        "--out", default="bc_dataset_human.npz",
        help="output .npz filename (relative to script dir). "
             "Default bc_dataset_human.npz",
    )
    ap.add_argument(
        "--n-wins", type=int, default=20,
        help="stop after this many winning episodes (default 20)",
    )
    args = ap.parse_args()

    here = Path(__file__).parent
    out_path = here / args.out

    # Wrapped in the K-frame raw-distance stack so recorded obs are 21-D,
    # matching the policy network / BC dataset layout.
    env = RawDistanceStack(
        MiniSumoEnv(
            gui=True, seed=int(time.time()),
            novamax_torque_mult=args.mult,
            force_opponent_id=args.opp,
            action_space_kind="discrete",
            narek_reward=True,
        ),
        k=DEFAULT_STACK_K,
    )

    # Disable PyBullet's built-in keyboard shortcuts (W = wireframe,
    # G = toggle GUI panel, etc.) so WASD ONLY drives the robot.
    # Also disable mouse picking so we can't accidentally grab a body.
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
    # Hide the side parameter panel — full focus on the dohyo view.
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    print(f"\n=== Human play vs {args.opp} @ mult={args.mult} ===")
    print(f"Saving wins to: {out_path}")
    print(f"Target: {args.n_wins} winning episodes")
    print("Controls: W A S D (or arrow keys); Q / ESC to quit cleanly")
    print("Only WINNING-episode transitions are saved. Losses/timeouts discarded.\n")

    n_wins = 0
    n_episodes = 0
    n_pairs_total = 0

    try:
        while n_wins < args.n_wins:
            obs, _ = env.reset()
            ep_obs: list[np.ndarray] = []
            ep_act: list[int] = []
            ep_rew: list[float] = []
            ep_next: list[np.ndarray] = []
            ep_done: list[bool] = []
            terminated = truncated = False
            user_quit = False

            while not (terminated or truncated):
                keys = p.getKeyboardEvents()
                if quit_requested(keys):
                    user_quit = True
                    break
                a_idx = read_action(keys)
                next_obs, reward, terminated, truncated, info = env.step(a_idx)
                ep_obs.append(obs.copy())
                ep_act.append(a_idx)
                ep_rew.append(float(reward))
                ep_next.append(next_obs.copy())
                ep_done.append(bool(terminated or truncated))
                obs = next_obs

            if user_quit:
                print("\nQuit signal received. Closing.")
                break

            n_episodes += 1
            reason = info.get("termination_reason", "unknown")
            if reason == "win":
                n_wins += 1
                n_added = len(ep_obs)
                n_pairs_total += n_added
                append_to_npz(
                    out_path,
                    np.array(ep_obs, dtype=np.float32),
                    np.array(ep_act, dtype=np.int64),
                    np.array(ep_rew, dtype=np.float32),
                    np.array(ep_next, dtype=np.float32),
                    np.array(ep_done, dtype=np.float32),
                )
                print(
                    f"WIN #{n_wins:3d}/{args.n_wins}  ep={n_episodes:3d}  "
                    f"+{n_added:3d} pairs  (total {n_pairs_total} pairs)"
                )
            else:
                print(
                    f"  {reason:10s}  ep={n_episodes:3d}  (discarded)"
                )

    finally:
        env.close()

    print(
        f"\nDone. {n_wins} winning eps / {n_episodes} total. "
        f"{n_pairs_total} pairs saved to {out_path.name}."
    )


if __name__ == "__main__":
    main()
