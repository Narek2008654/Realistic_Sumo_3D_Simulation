"""Head-to-head: two policies, the SAME robot chassis, fighting directly.

Drives both robots in MiniSumoEnv by their own greedy policy + their own
21-D observation. Faithfulness: the per-robot observation is built by the
env's OWN methods (_raw_distances / _update_last_seen / _build_obs) via a
save/restore swap of robot_id + obs state, so there is no duplicated obs
logic. A validation gate compares the swap-built agent obs against the env's
native obs before any match runs. Both robots are spawned as the agent URDF
(robot.urdf) so the comparison isolates the AGENT, not the chassis.

Usage:
  python scripts/agent_vs_agent.py --a checkpoints/ppo_stack_best.pt \
      --b checkpoints/dqn3d_stack_stageA_best.pt --n 60
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
from collections import deque, Counter

import torch  # noqa: F401  # before numpy (Windows DLL order)
import numpy as np
import pybullet as p

import sumo_env
from train_dqn_3d import DuelingQNet, build_env
from obs_stack import DEFAULT_STACK_K
from sumo_env import (
    DISCRETE_ACTION_MAP, SUBSTEPS_PER_STEP, WHEEL_OMEGA_FWD, WHEEL_MAX_TORQUE,
    FORWARD_SIGN, FALL_Z, ENGAGEMENT_FRONT_THRESHOLD, ENEMY_RGBA,
)

# env obs-state attributes swapped per body so the env's own obs methods
# build a faithful observation for whichever robot we point them at.
SWAP = ["last_seen_dir", "_steps_since_last_hit", "_prev_action",
        "_engagement_timer", "_yaw_rate_proxy", "_prev_front_norm",
        "_prev_min_lateral"]
K = DEFAULT_STACK_K


def load_policy(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    h1, obs = sd["trunk.0.weight"].shape
    h2 = sd["trunk.2.weight"].shape[0]
    n = sd["advantage_head.weight"].shape[0]
    net = DuelingQNet(obs, n, hidden=(h1, h2))
    net.load_state_dict(sd)
    net.eval()
    return net


def fresh_state():
    return {"last_seen_dir": 0.0, "_steps_since_last_hit": 0,
            "_prev_action": (0.0, 0.0), "_engagement_timer": 0,
            "_yaw_rate_proxy": 0.0, "_prev_front_norm": 1.0,
            "_prev_min_lateral": 1.0, "frames": None}


def robot_obs(env_u, body_id, st, raw_l, raw_r, first=False):
    """Build the 21-D stacked obs for `body_id` using the env's own methods.
    Mirrors step(): action-derived updates (prev_action, yaw) first, then
    post-physics distance-derived updates (engagement, last_seen, deltas)."""
    st["_prev_action"] = (raw_l, raw_r)
    yaw = (st["_yaw_rate_proxy"] + (raw_l - raw_r) * 0.1) * 0.9
    st["_yaw_rate_proxy"] = max(-1.0, min(1.0, yaw))

    saved_id = env_u.robot_id
    saved = {a: getattr(env_u, a) for a in SWAP}
    env_u.robot_id = body_id
    for a in SWAP:
        setattr(env_u, a, st[a])

    dists = env_u._raw_distances()
    if env_u._norm(dists[0]) < ENGAGEMENT_FRONT_THRESHOLD:
        env_u._engagement_timer += 1
    else:
        env_u._engagement_timer = 0
    env_u._update_last_seen(dists)
    obs12 = env_u._build_obs(dists)

    for a in SWAP:
        st[a] = getattr(env_u, a)
    env_u.robot_id = saved_id
    for a in SWAP:
        setattr(env_u, a, saved[a])

    if first or st["frames"] is None:
        st["frames"] = deque([obs12[:3].copy() for _ in range(K)], maxlen=K)
    else:
        st["frames"].append(obs12[:3].copy())
    return np.concatenate(list(st["frames"]) + [obs12[3:12]]).astype(np.float32)


def drive(left_idx, right_idx, body_id, raw_l, raw_r):
    cap = WHEEL_OMEGA_FWD
    for cmd, jidx in ((raw_l, left_idx), (raw_r, right_idx)):
        force = WHEEL_MAX_TORQUE if abs(cmd) > 0.05 else 0.0
        p.setJointMotorControl2(
            bodyUniqueId=body_id, jointIndex=jidx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=FORWARD_SIGN * cmd * cap, force=force,
        )


def respawn_enemy_as_agent_robot(env_u):
    """Replace the NovaMax enemy with a second agent-chassis robot so the
    match isolates the policy, not the chassis."""
    pos, orn = p.getBasePositionAndOrientation(env_u.enemy_id)
    yaw = p.getEulerFromQuaternion(orn)[2]
    p.removeBody(env_u.enemy_id)
    env_u.enemy_id, joints = env_u._spawn_robot(list(pos), yaw, ENEMY_RGBA)
    env_u._enemy_left_idx = joints["left_wheel_joint"]
    env_u._enemy_right_idx = joints["right_wheel_joint"]


def play_match(env_u, net_agent, net_enemy, max_steps=600, contact_window=10):
    """Returns (winner, loser_exit) where winner in {A,B,draw,timeout} and
    loser_exit in {self_out, push, ""}. A loss with no contact in the last
    `contact_window` steps is a self-out; otherwise the opponent pushed."""
    stA, stB = fresh_state(), fresh_state()
    obsA = robot_obs(env_u, env_u.robot_id, stA, 0.0, 0.0, first=True)
    obsB = robot_obs(env_u, env_u.enemy_id, stB, 0.0, 0.0, first=True)
    last_contact = -10_000
    for step in range(max_steps):
        rawA = DISCRETE_ACTION_MAP[net_agent.act_greedy(obsA)]
        rawB = DISCRETE_ACTION_MAP[net_enemy.act_greedy(obsB)]
        drive(env_u._left_wheel_idx, env_u._right_wheel_idx, env_u.robot_id, *rawA)
        drive(env_u._enemy_left_idx, env_u._enemy_right_idx, env_u.enemy_id, *rawB)
        for _ in range(SUBSTEPS_PER_STEP):
            p.stepSimulation()
        if p.getContactPoints(bodyA=env_u.robot_id, bodyB=env_u.enemy_id):
            last_contact = step
        za = p.getBasePositionAndOrientation(env_u.robot_id)[0][2]
        zb = p.getBasePositionAndOrientation(env_u.enemy_id)[0][2]
        a_out, b_out = za < FALL_Z, zb < FALL_Z
        if a_out or b_out:
            if a_out and b_out:
                return "draw", "mutual"
            exit_kind = "push" if (step - last_contact) <= contact_window else "self_out"
            return ("B", exit_kind) if a_out else ("A", exit_kind)
        obsA = robot_obs(env_u, env_u.robot_id, stA, *rawA)
        obsB = robot_obs(env_u, env_u.enemy_id, stB, *rawB)
    return "timeout", ""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="policy A checkpoint")
    ap.add_argument("--b", required=True, help="policy B checkpoint")
    ap.add_argument("--n", type=int, default=60, help="matches")
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    net_a = load_policy(args.a)
    net_b = load_policy(args.b)
    # Deterministic, fair, validatable: disable per-step ToF noise for ALL
    # resets (reset() re-reads this module global each episode).
    sumo_env.DR_TOF_NOISE_SIGMA_PCT = 0.0
    env = build_env(gui=False, seed=args.seed, narek_reward=False)
    env_u = env.unwrapped

    # Faithfulness gate: swap-built agent obs must equal the env's native obs.
    native = env.reset(seed=args.seed)[0]
    st = fresh_state()
    mine = robot_obs(env_u, env_u.robot_id, st, 0.0, 0.0, first=True)
    drift = float(np.abs(native - mine).max())
    print(f"obs faithfulness check: max|native - swap| = {drift:.2e}")
    if drift > 1e-5:
        raise SystemExit("obs mismatch — harness would be unfaithful, aborting")

    print(f"A = {pathlib.Path(args.a).name}")
    print(f"B = {pathlib.Path(args.b).name}")
    print(f"playing {args.n} matches (same chassis, greedy policies)\n")

    res = Counter()
    a_selfout = b_selfout = 0
    for i in range(args.n):
        env.reset(seed=args.seed + i)
        respawn_enemy_as_agent_robot(env_u)
        winner, exit_kind = play_match(env_u, net_a, net_b)
        res[winner] += 1
        if winner == "B" and exit_kind == "self_out":   # A drove itself off
            a_selfout += 1
        if winner == "A" and exit_kind == "self_out":   # B drove itself off
            b_selfout += 1

    env.close()
    a, b, d, t = res["A"], res["B"], res["draw"], res["timeout"]
    decided = a + b
    wr = a / decided if decided else 0.0
    print(f"A wins: {a}   B wins: {b}   draws: {d}   timeouts: {t}")
    print(f"A win-rate over decided matches: {wr:.0%}")
    print(f"A losses by self-out: {a_selfout}/{b}   "
          f"B losses by self-out: {b_selfout}/{a}")


if __name__ == "__main__":
    main()
