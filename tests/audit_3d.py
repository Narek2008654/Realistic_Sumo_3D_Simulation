"""Comprehensive correctness audit for the 3D mini-sumo env.

Runs 67 self-contained tests across categories A-S and prints PASS/FAIL
for each. Categories:

  A. Constants sanity (mass / speed / dimensions match spec)
  B. Termination logic (win, push_loss, self_out, mutual_out, timeout)
  C. Push physics (rear, side, stuck-detector)
  D. Wheel kinematics (top speed, reverse, tank-spin yaw)
  E. Observation vector (shape, range per index)
  F. Sensor model (IR, line sensors)
  G. Opponent zoo (instantiation, decide() output)
  H. Discrete action space (size, mapping, edge cases)
  I. Domain randomization (does it actually vary?)
  J. last_seen_dir state machine
  K. Engagement timer + engage reward
  L. Action latency queue
  M. Reward components
  N. yaw_rate_proxy
  O. IR sensor noise + dropout DR
  P. Flank reward (rear/side engagement, gating, episode cap)
  Q. Frame stacking (RawDistanceStack + offline windowing)
  R. Behavioral opponent DR + hard sensor-noise DR
  S. Held-out opponent zoo (feinter / orbiter)

Run:
    python tests/audit_3d.py

Exit code is 0 if all pass, 1 otherwise.
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# --- sys.path shim (added by reorg) ---
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

import math
import sys
import traceback

import torch  # noqa
import numpy as np
import pybullet as p

# Make stdout encoding-safe: a few detail strings below contain
# non-ASCII (Delta). On Windows a redirected/piped stdout defaults
# to cp1252, which cannot encode them and would crash the suite
# mid-run. UTF-8 with a safe error handler keeps output readable.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except (AttributeError, ValueError):
    pass

# Test bookkeeping --------------------------------------------------------------
_passed = []
_failed = []
def check(name: str, ok: bool, detail: str = ""):
    tag = "PASS" if ok else "FAIL"
    line = f"  [{tag}] {name}" + (f"   {detail}" if detail else "")
    print(line, flush=True)
    (_passed if ok else _failed).append(name)

def section(s: str):
    print(f"\n=== {s} ===", flush=True)

# Imports under test ------------------------------------------------------------
from sumo_env import (
    MiniSumoEnv,
    AGENT_MAX_RAD, AGENT_MAX_FORCE, WHEEL_RADIUS,
    DOHYO_RADIUS, INNER_RADIUS, SPAWN_RADIUS, FALL_Z,
    STEP_DT_SECONDS, MAX_EPISODE_STEPS,
    DISCRETE_ACTION_MAP,
    REWARD_WIN, REWARD_LOSE_PUSH, REWARD_LOSE_MUTUAL, REWARD_LOSE_SELF,
    REWARD_TIMEOUT, REWARD_TIME,
    REWARD_ENGAGE_PER_TICK, REWARD_APPROACH, REWARD_WEDGE_PER_M,
    ENEMY_FAR_DIST, ENGAGEMENT_FRONT_THRESHOLD,
    CONTACT_RECENT_STEPS,
    WHEEL_FRICTION, CHASSIS_FRICTION,
    STUCK_DETECTION_TICKS, STUCK_VELOCITY_THRESHOLD, STUCK_KICK_SPEED,
    DR_MASS_RANGE, DR_FRICTION_RANGE, DR_VELOCITY_RANGE,
    DR_DEADZONE_RANGE, DR_ACTION_LATENCY_CHOICES,
    SIM_TIMESTEP,
    FLANK_EPISODE_CAP,
    DR_TOF_NOISE_SIGMA_PCT, DR_TOF_NOISE_SIGMA_PCT_HARD,
    OPP_DR_SPEED_RANGE, OPP_DR_TRACKING_RANGE, OPP_DR_REACTION_CHOICES,
)
from obs_stack import (
    RawDistanceStack, stack_from_trajectory, stacked_dim, DEFAULT_STACK_K,
)
from opponents import (
    OPPONENT_REGISTRY, OPPONENT_WEIGHTS, HELD_OUT_OPPONENT_IDS,
)


# =============================================================================
# A. Constants sanity
# =============================================================================
section("A. Constants sanity")

check("AGENT_MAX_RAD = 41.88 (400 RPM)", abs(AGENT_MAX_RAD - 41.88) < 0.01,
      f"got {AGENT_MAX_RAD}")
check("WHEEL_RADIUS = 0.010 m", abs(WHEEL_RADIUS - 0.010) < 1e-6,
      f"got {WHEEL_RADIUS}")
check("DOHYO_RADIUS = 0.35 m", abs(DOHYO_RADIUS - 0.35) < 1e-6,
      f"got {DOHYO_RADIUS}")
check("SPAWN_RADIUS < INNER_RADIUS", SPAWN_RADIUS < INNER_RADIUS,
      f"spawn={SPAWN_RADIUS} inner={INNER_RADIUS}")
check("STEP_DT_SECONDS ~= 1/24", abs(STEP_DT_SECONDS - 1/24) < 5e-3,
      f"got {STEP_DT_SECONDS:.5f}")
check("Loss split: self < mutual < push < 0",
      REWARD_LOSE_SELF < REWARD_LOSE_MUTUAL < REWARD_LOSE_PUSH < 0,
      f"self={REWARD_LOSE_SELF} mutual={REWARD_LOSE_MUTUAL} push={REWARD_LOSE_PUSH}")
check("REWARD_WIN positive, terminals dominate shaping",
      REWARD_WIN > 0 and abs(REWARD_WIN) > 10 * REWARD_ENGAGE_PER_TICK,
      f"win={REWARD_WIN} engage_per_tick={REWARD_ENGAGE_PER_TICK}")
check("Physics top speed ~= 0.42 m/s",
      abs(AGENT_MAX_RAD * WHEEL_RADIUS - 0.4188) < 1e-3,
      f"computed {AGENT_MAX_RAD * WHEEL_RADIUS:.4f}")


# =============================================================================
# Helpers to spin up envs for the rest of the tests
# =============================================================================
def make_env(**kwargs):
    defaults = dict(
        gui=False, seed=42, novamax_torque_mult=1.0,
        force_opponent_id="rammer",
        action_space_kind="discrete",
        narek_reward=False,
    )
    defaults.update(kwargs)
    return MiniSumoEnv(**defaults)


# =============================================================================
# B. Termination logic + per-reason reward
# =============================================================================
section("B. Termination logic + per-reason reward")

# B1: timeout fires after MAX_EPISODE_STEPS in a no-contact stalemate.
# Pin to dodger and keep teleporting both bots back to opposite corners
# every step so neither falls off. Verifies that _steps counter and the
# truncated path work correctly.
def test_timeout():
    env = make_env(force_opponent_id="dodger")
    env.reset(seed=0)
    qrn = p.getQuaternionFromEuler([0, 0, 0])
    last_info = {}
    n_steps = 0
    idx_idle = next(i for i, a in enumerate(DISCRETE_ACTION_MAP)
                    if a == (0.0, 0.0))
    for _ in range(MAX_EPISODE_STEPS + 5):
        # Pin both bots well inside the dohyo at opposite ends so
        # neither can self_out or contact within one step.
        p.resetBasePositionAndOrientation(env.robot_id,
            [-0.15, 0, 0.04], qrn)
        p.resetBasePositionAndOrientation(env.enemy_id,
            [+0.15, 0, 0.04], qrn)
        p.resetBaseVelocity(env.robot_id, [0, 0, 0], [0, 0, 0])
        p.resetBaseVelocity(env.enemy_id, [0, 0, 0], [0, 0, 0])
        _, _, term, trunc, info = env.step(idx_idle)
        last_info = info
        n_steps += 1
        if term or trunc:
            break
    env.close()
    return last_info.get("termination_reason"), n_steps

reason, nsteps = test_timeout()
check("Stalemate (no contact) -> timeout at MAX_EPISODE_STEPS",
      reason == "timeout" and nsteps == MAX_EPISODE_STEPS,
      f"reason={reason} after {nsteps} steps (max={MAX_EPISODE_STEPS})")

# B2: drive agent off the dohyo without ever contacting enemy -> self_out
def test_self_out():
    env = make_env()
    env.reset(seed=1)
    # Teleport agent near the edge facing outward
    qrn = p.getQuaternionFromEuler([0, 0, 0])
    p.resetBasePositionAndOrientation(env.robot_id,
        [INNER_RADIUS - 0.01, 0, 0.04], qrn)
    # Push enemy far away so no contact possible
    p.resetBasePositionAndOrientation(env.enemy_id,
        [-INNER_RADIUS + 0.01, 0, 0.04], qrn)
    # Idle action wouldn't move, so we need to drive forward
    idx_fwd = next(i for i, a in enumerate(DISCRETE_ACTION_MAP)
                   if a == (1.0, 1.0))
    last_info = {}
    last_rew = 0.0
    for _ in range(100):
        _, r, term, trunc, info = env.step(idx_fwd)
        last_info, last_rew = info, r
        if term or trunc:
            break
    env.close()
    return last_info.get("termination_reason"), last_rew

reason, rew = test_self_out()
check("Drive agent off edge -> self_out", reason == "self_out",
      f"reason={reason} reward={rew:.2f}")
# In the env, the per-step reward INCLUDES terminal + components, so we
# verify the terminal magnitude is at least the constant (could be more
# negative due to time/shaping additions).
check("self_out reward <= REWARD_LOSE_SELF + small slack",
      rew <= REWARD_LOSE_SELF + 0.1,
      f"got {rew:.2f}; expected near {REWARD_LOSE_SELF}")


# =============================================================================
# C. Push physics
# =============================================================================
section("C. Push physics")

def push_test(agent_xyz_yaw, enemy_xyz_yaw, agent_omega, n_ticks):
    env = make_env()
    env.reset(seed=2)
    ax, ay, az, ayaw = agent_xyz_yaw
    ex, ey, ez, eyaw = enemy_xyz_yaw
    p.resetBasePositionAndOrientation(env.robot_id,
        [ax, ay, az], p.getQuaternionFromEuler([0, 0, ayaw]))
    p.resetBasePositionAndOrientation(env.enemy_id,
        [ex, ey, ez], p.getQuaternionFromEuler([0, 0, eyaw]))
    p.resetBaseVelocity(env.robot_id, [0, 0, 0], [0, 0, 0])
    p.resetBaseVelocity(env.enemy_id, [0, 0, 0], [0, 0, 0])

    def wheel_joints(body_id):
        lookup = {p.getJointInfo(body_id, j)[1].decode(): j
                  for j in range(p.getNumJoints(body_id))}
        return lookup["left_wheel_joint"], lookup["right_wheel_joint"]
    a_l, a_r = wheel_joints(env.robot_id)
    e_l, e_r = wheel_joints(env.enemy_id)
    start_pos, _ = p.getBasePositionAndOrientation(env.enemy_id)
    for _ in range(n_ticks):
        p.setJointMotorControl2(env.robot_id, a_l, p.VELOCITY_CONTROL,
                                targetVelocity=agent_omega, force=AGENT_MAX_FORCE)
        p.setJointMotorControl2(env.robot_id, a_r, p.VELOCITY_CONTROL,
                                targetVelocity=agent_omega, force=AGENT_MAX_FORCE)
        p.setJointMotorControl2(env.enemy_id, e_l, p.VELOCITY_CONTROL,
                                targetVelocity=0.0, force=AGENT_MAX_FORCE)
        p.setJointMotorControl2(env.enemy_id, e_r, p.VELOCITY_CONTROL,
                                targetVelocity=0.0, force=AGENT_MAX_FORCE)
        p.stepSimulation()
    end_pos, _ = p.getBasePositionAndOrientation(env.enemy_id)
    env.close()
    return start_pos, end_pos

ticks_1p5s = int(1.5 / SIM_TIMESTEP)

# C1: rear push moves enemy > 2 cm in 1.5s
s, e = push_test(
    agent_xyz_yaw=(-0.12, 0, 0.04, 0),
    enemy_xyz_yaw=(0, 0, 0.04, 0),
    agent_omega=+AGENT_MAX_RAD, n_ticks=ticks_1p5s)
dx = e[0] - s[0]
check("Rear push moves enemy > 0.02 m in 1.5 s", dx > 0.02,
      f"Δx={dx*1000:.1f} mm")

# C2: side push moves enemy > 0.05 m in 1.5s
s, e = push_test(
    agent_xyz_yaw=(0, -0.12, 0.04, math.pi/2),
    enemy_xyz_yaw=(0, 0, 0.04, 0),
    agent_omega=+AGENT_MAX_RAD, n_ticks=ticks_1p5s)
dy = e[1] - s[1]
check("Side push moves enemy > 0.05 m in 1.5 s", dy > 0.05,
      f"Δy={dy*1000:.1f} mm")

# C2b: push idle opponent along its own forward axis (the bug from the
# screenshot — agent pressed against opp's flank near the edge couldn't
# shove them off). With the free-spin fix, idle wheels do NOT brake.
def idle_push_test(n_ticks):
    env = make_env()
    env.reset(seed=11)
    # Agent behind enemy, both facing +x. Enemy is fully idle (no motor
    # control commanded), only ground friction holds them.
    p.resetBasePositionAndOrientation(env.robot_id,
        [-0.12, 0, 0.04], p.getQuaternionFromEuler([0, 0, 0]))
    p.resetBasePositionAndOrientation(env.enemy_id,
        [0.00, 0, 0.04], p.getQuaternionFromEuler([0, 0, 0]))
    p.resetBaseVelocity(env.robot_id, [0, 0, 0], [0, 0, 0])
    p.resetBaseVelocity(env.enemy_id, [0, 0, 0], [0, 0, 0])
    def wjoints(b):
        lk = {p.getJointInfo(b, j)[1].decode(): j
              for j in range(p.getNumJoints(b))}
        return lk["left_wheel_joint"], lk["right_wheel_joint"]
    al, ar = wjoints(env.robot_id)
    el, er = wjoints(env.enemy_id)
    start_x = 0.0
    for _ in range(n_ticks):
        # Agent: drive forward.
        p.setJointMotorControl2(env.robot_id, al, p.VELOCITY_CONTROL,
            targetVelocity=+AGENT_MAX_RAD, force=AGENT_MAX_FORCE)
        p.setJointMotorControl2(env.robot_id, ar, p.VELOCITY_CONTROL,
            targetVelocity=+AGENT_MAX_RAD, force=AGENT_MAX_FORCE)
        # Enemy: TRULY idle — target=0, force=0 (free-spin model).
        p.setJointMotorControl2(env.enemy_id, el, p.VELOCITY_CONTROL,
            targetVelocity=0.0, force=0.0)
        p.setJointMotorControl2(env.enemy_id, er, p.VELOCITY_CONTROL,
            targetVelocity=0.0, force=0.0)
        p.stepSimulation()
    end_pos, _ = p.getBasePositionAndOrientation(env.enemy_id)
    env.close()
    return end_pos[0] - start_x

dx_idle = idle_push_test(ticks_1p5s)
check("Idle opponent gets pushed > 0.05 m in own forward direction",
      dx_idle > 0.05,
      f"Δx={dx_idle*1000:.1f} mm (was the screenshot-bug regime)")

# C3: stuck-detector fires when both bots stationary in contact.
# The detector itself is internal; verify both halves of its contract:
#   (1) the counter increments while bots are in contact + stationary
#   (2) the kick fires (counter resets to 0 after reaching the threshold)
#   (3) at fire, both bots gain non-trivial linear velocity in one tick
def test_stuck_detector():
    # Both bots fully passive (we monkey-patch the enemy control to a
    # no-op so it doesn't drive). Place them in clean contact and let
    # them sit. The stuck-counter should accumulate; once it hits the
    # threshold, the detector fires a lateral kick.
    env = make_env(force_opponent_id="rammer")
    env.reset(seed=3)
    env._apply_enemy_control = lambda: None  # noqa - silence the enemy
    p.resetBasePositionAndOrientation(env.robot_id,
        [-0.06, 0, 0.04], p.getQuaternionFromEuler([0, 0, 0]))
    p.resetBasePositionAndOrientation(env.enemy_id,
        [+0.06, 0, 0.04], p.getQuaternionFromEuler([0, 0, math.pi]))
    p.resetBaseVelocity(env.robot_id, [0, 0, 0], [0, 0, 0])
    p.resetBaseVelocity(env.enemy_id, [0, 0, 0], [0, 0, 0])
    idx_fwd  = next(i for i, a in enumerate(DISCRETE_ACTION_MAP) if a == (1.0, 1.0))
    idx_idle = next(i for i, a in enumerate(DISCRETE_ACTION_MAP) if a == (0.0, 0.0))
    # Drive agent briefly to make contact, then idle so both bots are
    # passive and in contact.
    for _ in range(6):
        env.step(idx_fwd)
    counter_max = 0
    fire_seen = False
    prev_stuck = 0
    prev_speed = 0.0
    for _ in range(40):
        env.step(idx_idle)
        a_lv, _ = p.getBaseVelocity(env.robot_id)
        a_speed = math.hypot(a_lv[0], a_lv[1])
        stuck = env._stuck_ticks
        counter_max = max(counter_max, stuck)
        if prev_stuck >= STUCK_DETECTION_TICKS - 1 and stuck == 0 \
                and a_speed > 0.02 and prev_speed < 0.05:
            fire_seen = True
        prev_stuck = stuck
        prev_speed = a_speed
    env.close()
    return counter_max, fire_seen

counter_max, fire_seen = test_stuck_detector()
check("Stuck counter accumulates >= STUCK_DETECTION_TICKS-1 ticks",
      counter_max >= STUCK_DETECTION_TICKS - 1,
      f"counter peaked at {counter_max} (threshold {STUCK_DETECTION_TICKS})")
check("Stuck-detector kick fires (counter resets + speed jumps)",
      fire_seen,
      "kick observed" if fire_seen else "no counter-reset + velocity-spike observed")


# =============================================================================
# D. Wheel kinematics
# =============================================================================
section("D. Wheel kinematics")

# D1: top forward speed reaches ~0.42 m/s within ~0.5 s
def test_top_speed():
    env = make_env()
    env.reset(seed=4)
    # Clear initial-charge: re-zero everything
    p.resetBasePositionAndOrientation(env.robot_id,
        [0, 0, 0.04], p.getQuaternionFromEuler([0, 0, 0]))
    p.resetBasePositionAndOrientation(env.enemy_id,
        [10, 10, 0.04], p.getQuaternionFromEuler([0, 0, 0]))  # far away
    p.resetBaseVelocity(env.robot_id, [0, 0, 0], [0, 0, 0])
    idx_fwd = next(i for i, a in enumerate(DISCRETE_ACTION_MAP)
                   if a == (1.0, 1.0))
    speeds = []
    for _ in range(30):
        env.step(idx_fwd)
        lv, _ = p.getBaseVelocity(env.robot_id)
        speeds.append(math.hypot(lv[0], lv[1]))
    env.close()
    return max(speeds)

top = test_top_speed()
expected = AGENT_MAX_RAD * WHEEL_RADIUS  # 0.4188
check("Forward top speed within 70-130% of spec",
      0.7 * expected <= top <= 1.3 * expected,
      f"top={top:.3f} m/s (spec {expected:.3f})")

# D2: reverse works (negative speed achievable)
def test_reverse():
    env = make_env()
    env.reset(seed=5)
    p.resetBasePositionAndOrientation(env.robot_id,
        [0, 0, 0.04], p.getQuaternionFromEuler([0, 0, 0]))
    p.resetBasePositionAndOrientation(env.enemy_id,
        [10, 10, 0.04], p.getQuaternionFromEuler([0, 0, 0]))
    p.resetBaseVelocity(env.robot_id, [0, 0, 0], [0, 0, 0])
    idx_rev = next(i for i, a in enumerate(DISCRETE_ACTION_MAP)
                   if a == (-1.0, -1.0))
    start, _ = p.getBasePositionAndOrientation(env.robot_id)
    for _ in range(20):
        env.step(idx_rev)
    end, _ = p.getBasePositionAndOrientation(env.robot_id)
    env.close()
    return end[0] - start[0]

dx_rev = test_reverse()
check("Reverse action moves bot backward", dx_rev < -0.02,
      f"Δx={dx_rev*1000:.1f} mm (should be negative)")


# =============================================================================
# E. Observation vector
# =============================================================================
section("E. Observation vector")

def test_obs_ranges():
    env = make_env()
    obs, _ = env.reset(seed=6)
    issues = []
    if obs.shape != (12,):
        issues.append(f"shape={obs.shape}")
    else:
        if not (0.0 <= obs[0] <= 1.0): issues.append(f"obs[0]={obs[0]}")
        if not (0.0 <= obs[1] <= 1.0): issues.append(f"obs[1]={obs[1]}")
        if not (0.0 <= obs[2] <= 1.0): issues.append(f"obs[2]={obs[2]}")
        if obs[3] not in (-1.0, 0.0, 1.0): issues.append(f"obs[3]={obs[3]}")
        if obs[4] not in (0.0, 1.0):       issues.append(f"obs[4]={obs[4]}")
        if obs[5] not in (0.0, 1.0):       issues.append(f"obs[5]={obs[5]}")
        if not (-1.0 <= obs[6] <= 1.0):    issues.append(f"obs[6]={obs[6]}")
        if not (-1.0 <= obs[7] <= 1.0):    issues.append(f"obs[7]={obs[7]}")
        if not (0.0 <= obs[8] <= 1.0):     issues.append(f"obs[8]={obs[8]}")
        if not (-1.0 <= obs[9] <= 1.0):    issues.append(f"obs[9]={obs[9]}")
        if not (-1.0 <= obs[10] <= 1.0):   issues.append(f"obs[10]={obs[10]}")
        if not (-1.0 <= obs[11] <= 1.0):   issues.append(f"obs[11]={obs[11]}")
    env.close()
    return issues

issues = test_obs_ranges()
check("Reset obs has shape (12,)", "shape" not in str(issues),
      str(issues))
check("All obs indices in legal ranges at reset",
      len(issues) == 0, "; ".join(issues))


# =============================================================================
# F. Sensor model
# =============================================================================
section("F. Sensor model")

# F1: with no enemy in IR range, obs[0..2] should saturate at 1.0
def test_ir_no_target():
    env = make_env()
    env.reset(seed=7)
    p.resetBasePositionAndOrientation(env.robot_id,
        [0, 0, 0.04], p.getQuaternionFromEuler([0, 0, 0]))
    p.resetBasePositionAndOrientation(env.enemy_id,
        [10, 10, 0.04], p.getQuaternionFromEuler([0, 0, 0]))
    idx_idle = next(i for i, a in enumerate(DISCRETE_ACTION_MAP)
                    if a == (0.0, 0.0))
    obs, _, _, _, _ = env.step(idx_idle)
    env.close()
    return obs[0], obs[1], obs[2]

f, l, r = test_ir_no_target()
check("IR saturates near 1.0 with target out of range",
      f > 0.9 and l > 0.9 and r > 0.9,
      f"front={f:.2f} left={l:.2f} right={r:.2f}")

# F2: with enemy directly in front, front IR should drop below 0.3
def test_ir_close_target():
    env = make_env()
    env.reset(seed=8)
    p.resetBasePositionAndOrientation(env.robot_id,
        [0, 0, 0.04], p.getQuaternionFromEuler([0, 0, 0]))
    p.resetBasePositionAndOrientation(env.enemy_id,
        [0.10, 0, 0.04], p.getQuaternionFromEuler([0, 0, 0]))  # 10 cm ahead
    idx_idle = next(i for i, a in enumerate(DISCRETE_ACTION_MAP)
                    if a == (0.0, 0.0))
    obs, _, _, _, _ = env.step(idx_idle)
    env.close()
    return obs[0]

f = test_ir_close_target()
check("Front IR < 0.3 when enemy 10 cm ahead", f < 0.3,
      f"front={f:.3f}")


# =============================================================================
# G. Opponent zoo
# =============================================================================
section("G. Opponent zoo")

OPPONENTS = ("dodger", "spinner", "rammer", "wedger", "novamax", "charger")
for opp in OPPONENTS:
    try:
        env = make_env(force_opponent_id=opp)
        env.reset(seed=hash(opp) & 0xffff)
        idx_idle = next(i for i, a in enumerate(DISCRETE_ACTION_MAP)
                        if a == (0.0, 0.0))
        for _ in range(10):
            _, _, term, trunc, _ = env.step(idx_idle)
            if term or trunc:
                break
        env.close()
        check(f"Opponent {opp} runs 10 steps without crash", True)
    except Exception as exc:
        check(f"Opponent {opp} runs 10 steps without crash", False,
              str(exc))


# =============================================================================
# H. Discrete action space
# =============================================================================
section("H. Discrete action space")

env = make_env()
check("action_space.n == 9", env.action_space.n == 9,
      f"got {env.action_space.n}")
check("DISCRETE_ACTION_MAP len 9", len(DISCRETE_ACTION_MAP) == 9,
      f"got {len(DISCRETE_ACTION_MAP)}")
unique = set(DISCRETE_ACTION_MAP)
check("All 9 actions unique", len(unique) == 9,
      f"got {len(unique)} unique")
for (l, r) in DISCRETE_ACTION_MAP:
    if l not in (-1.0, 0.0, 1.0) or r not in (-1.0, 0.0, 1.0):
        check(f"Action ({l},{r}) maps to {{-1,0,+1}} grid", False)
        break
else:
    check("All 9 actions on the {-1,0,+1} grid", True)

# Out-of-range action raises
env.reset(seed=9)
try:
    env.step(99)
    check("step(99) raises ValueError", False, "no exception raised")
except ValueError:
    check("step(99) raises ValueError", True)
except Exception as e:
    check("step(99) raises ValueError", False, f"raised {type(e).__name__}")
env.close()


# =============================================================================
# I. Domain randomization
# =============================================================================
section("I. Domain randomization")

# Probe DR via the PyBullet dynamics API + persistent env attributes.
# DR per reset randomizes: (a) dohyo surface friction, (b) agent +
# enemy chassis mass, (c) motor deadzone (kept as self._motor_deadzone).
def collect_dr_samples(n=30):
    env = make_env()
    masses, deadzones = [], []
    for s in range(n):
        env.reset(seed=s)
        # Mass of agent's base link via dynamics info
        info = p.getDynamicsInfo(env.robot_id, -1)
        masses.append(info[0])  # mass
        deadzones.append(env._motor_deadzone)
    env.close()
    return masses, deadzones

try:
    masses, deadzones = collect_dr_samples()
    mass_std = float(np.std(masses))
    dz_std = float(np.std(deadzones))
    check("Mass varies per reset (DR_MASS_RANGE)",
          mass_std > 0.001,
          f"std={mass_std:.4f} kg, range={min(masses):.3f}-{max(masses):.3f}")
    check("Deadzone varies per reset (DR_DEADZONE_RANGE)",
          dz_std > 0.01,
          f"std={dz_std:.3f}, range={min(deadzones):.3f}-{max(deadzones):.3f}")
except Exception as e:
    check("DR probe", False, str(e))


# =============================================================================
# J. last_seen_dir state machine
# =============================================================================
section("J. last_seen_dir state machine")

def get_last_seen(env_obs):
    return env_obs[3]

# J1: enemy on right → last_seen latches to +1
def test_last_seen_right():
    env = make_env()
    env.reset(seed=20)
    p.resetBasePositionAndOrientation(env.robot_id,
        [0, 0, 0.04], p.getQuaternionFromEuler([0, 0, 0]))
    # Place enemy at +30° offset to the right at 15 cm.
    ex = 0.15 * math.cos(math.radians(-30))
    ey = 0.15 * math.sin(math.radians(-30))
    p.resetBasePositionAndOrientation(env.enemy_id,
        [ex, ey, 0.04], p.getQuaternionFromEuler([0, 0, 0]))
    idx_idle = next(i for i, a in enumerate(DISCRETE_ACTION_MAP) if a == (0.0, 0.0))
    obs, _, _, _, _ = env.step(idx_idle)
    env.close()
    return get_last_seen(obs)

ls = test_last_seen_right()
check("Enemy on right -> last_seen = +1", ls == 1.0, f"got {ls}")

# J2: enemy on left → last_seen = -1
def test_last_seen_left():
    env = make_env()
    env.reset(seed=21)
    p.resetBasePositionAndOrientation(env.robot_id,
        [0, 0, 0.04], p.getQuaternionFromEuler([0, 0, 0]))
    ex = 0.15 * math.cos(math.radians(+30))
    ey = 0.15 * math.sin(math.radians(+30))
    p.resetBasePositionAndOrientation(env.enemy_id,
        [ex, ey, 0.04], p.getQuaternionFromEuler([0, 0, 0]))
    idx_idle = next(i for i, a in enumerate(DISCRETE_ACTION_MAP) if a == (0.0, 0.0))
    obs, _, _, _, _ = env.step(idx_idle)
    env.close()
    return get_last_seen(obs)

ls = test_last_seen_left()
check("Enemy on left -> last_seen = -1", ls == -1.0, f"got {ls}")

# J3: enemy ahead → last_seen = 0
def test_last_seen_front():
    env = make_env()
    env.reset(seed=22)
    p.resetBasePositionAndOrientation(env.robot_id,
        [0, 0, 0.04], p.getQuaternionFromEuler([0, 0, 0]))
    p.resetBasePositionAndOrientation(env.enemy_id,
        [0.15, 0.0, 0.04], p.getQuaternionFromEuler([0, 0, 0]))
    idx_idle = next(i for i, a in enumerate(DISCRETE_ACTION_MAP) if a == (0.0, 0.0))
    obs, _, _, _, _ = env.step(idx_idle)
    env.close()
    return get_last_seen(obs)

ls = test_last_seen_front()
check("Enemy directly ahead -> last_seen = 0", ls == 0.0, f"got {ls}")

# J4: enemy disappears → last_seen decays to 0 after LAST_SEEN_DECAY_STEPS
def test_last_seen_decay():
    env = make_env()
    env.reset(seed=23)
    p.resetBasePositionAndOrientation(env.robot_id,
        [0, 0, 0.04], p.getQuaternionFromEuler([0, 0, 0]))
    # Place enemy briefly on right, latch dir=+1.
    p.resetBasePositionAndOrientation(env.enemy_id,
        [0.15 * math.cos(math.radians(-30)),
         0.15 * math.sin(math.radians(-30)), 0.04],
        p.getQuaternionFromEuler([0, 0, 0]))
    idx_idle = next(i for i, a in enumerate(DISCRETE_ACTION_MAP) if a == (0.0, 0.0))
    obs, _, _, _, _ = env.step(idx_idle)
    assert obs[3] == 1.0, f"latch failed, got {obs[3]}"
    # Now move enemy far away — should decay over ~12 ticks.
    p.resetBasePositionAndOrientation(env.enemy_id,
        [10, 10, 0.04], p.getQuaternionFromEuler([0, 0, 0]))
    decayed = False
    for tick in range(25):
        obs, _, _, _, _ = env.step(idx_idle)
        if obs[3] == 0.0:
            decayed = True
            break
    env.close()
    return decayed, tick

decayed, tick = test_last_seen_decay()
check("last_seen decays to 0 within 25 ticks once enemy gone",
      decayed, f"decayed at tick {tick}")


# =============================================================================
# K. Engagement timer + engage reward
# =============================================================================
section("K. Engagement timer + engage reward")

def test_engagement_timer():
    # Disable enemy control so the enemy stays put. Re-pin positions
    # each tick so wedge slip doesn't separate them. Check timer counts up.
    env = make_env()
    env.reset(seed=24)
    env._apply_enemy_control = lambda: None  # noqa
    idx_idle = next(i for i, a in enumerate(DISCRETE_ACTION_MAP) if a == (0.0, 0.0))
    qrn = p.getQuaternionFromEuler([0, 0, 0])
    for _ in range(5):
        p.resetBasePositionAndOrientation(env.robot_id, [0, 0, 0.04], qrn)
        p.resetBasePositionAndOrientation(env.enemy_id, [0.10, 0, 0.04], qrn)
        p.resetBaseVelocity(env.robot_id, [0, 0, 0], [0, 0, 0])
        p.resetBaseVelocity(env.enemy_id, [0, 0, 0], [0, 0, 0])
        env.step(idx_idle)
    eng_close = env._engagement_timer
    # Now teleport enemy far → timer should reset on next step
    p.resetBasePositionAndOrientation(env.enemy_id, [10, 10, 0.04], qrn)
    env.step(idx_idle)
    eng_far = env._engagement_timer
    env.close()
    return eng_close, eng_far

eng_close, eng_far = test_engagement_timer()
check("engagement_timer accumulates while close", eng_close >= 4,
      f"got {eng_close} after 5 close ticks")
check("engagement_timer resets when enemy far", eng_far == 0,
      f"got {eng_far} after teleport away")


# =============================================================================
# L. Action latency queue
# =============================================================================
section("L. Action latency queue")

# With DR_ACTION_LATENCY_CHOICES=(1,), tick 0's command should be queued
# and tick 1 applies it. So first call to step() with forward action
# results in zero motor torque (queue empty → emits 0,0).
def test_action_latency():
    env = make_env()
    env.reset(seed=25)
    p.resetBasePositionAndOrientation(env.robot_id,
        [0, 0, 0.04], p.getQuaternionFromEuler([0, 0, 0]))
    p.resetBasePositionAndOrientation(env.enemy_id,
        [10, 10, 0.04], p.getQuaternionFromEuler([0, 0, 0]))
    p.resetBaseVelocity(env.robot_id, [0, 0, 0], [0, 0, 0])
    idx_fwd = next(i for i, a in enumerate(DISCRETE_ACTION_MAP) if a == (1.0, 1.0))
    # Drain the queue first so we start from known state. The env
    # warmup (initial charge) already populated it, so the queue may
    # be hot. We just check the OBSERVATION reflects the delay: after
    # one fwd step, prev_left (obs[6]) should equal the COMMANDED
    # value, not the delayed one (prev_action is the policy's raw cmd).
    obs, _, _, _, _ = env.step(idx_fwd)
    prev_l, prev_r = obs[6], obs[7]
    # Latency queue length should == DR_ACTION_LATENCY_CHOICES[0]
    qlen = len(env._action_queue)
    env.close()
    return prev_l, prev_r, qlen

prev_l, prev_r, qlen = test_action_latency()
check("After (1,1) action, obs[6:8] reflects RAW cmd (1,1)",
      prev_l == 1.0 and prev_r == 1.0,
      f"prev_left={prev_l} prev_right={prev_r}")
check("Action queue length matches DR_ACTION_LATENCY_CHOICES",
      qlen == DR_ACTION_LATENCY_CHOICES[0],
      f"qlen={qlen} expected {DR_ACTION_LATENCY_CHOICES[0]}")


# =============================================================================
# M. Reward components (approach, engage, narek, wedge)
# =============================================================================
section("M. Reward components")

def test_components(narek=False):
    env = make_env(narek_reward=narek)
    env.reset(seed=26)
    env._apply_enemy_control = lambda: None  # noqa - freeze enemy
    qrn = p.getQuaternionFromEuler([0, 0, 0])
    qrn_back = p.getQuaternionFromEuler([0, 0, math.pi])
    p.resetBasePositionAndOrientation(env.robot_id, [-0.20, 0, 0.04], qrn)
    p.resetBasePositionAndOrientation(env.enemy_id, [+0.10, 0, 0.04], qrn_back)
    p.resetBaseVelocity(env.robot_id, [0, 0, 0], [0, 0, 0])
    p.resetBaseVelocity(env.enemy_id, [0, 0, 0], [0, 0, 0])
    idx_fwd  = next(i for i, a in enumerate(DISCRETE_ACTION_MAP) if a == (1.0, 1.0))
    idx_idle = next(i for i, a in enumerate(DISCRETE_ACTION_MAP) if a == (0.0, 0.0))
    components_seen = set()
    # Drive forward (closes distance), capture components per step.
    for _ in range(20):
        obs, r, term, trunc, info = env.step(idx_fwd)
        comps = info.get("reward_components_step", {}) or {}
        components_seen.update(comps.keys())
        if term or trunc:
            break
    # One idle step at close range to surface engage / narek_idle.
    p.resetBasePositionAndOrientation(env.robot_id, [-0.05, 0, 0.04], qrn)
    p.resetBasePositionAndOrientation(env.enemy_id, [+0.05, 0, 0.04], qrn_back)
    obs, r, term, trunc, info = env.step(idx_idle)
    comps = info.get("reward_components_step", {}) or {}
    components_seen.update(comps.keys())
    env.close()
    return components_seen

comps = test_components(narek=False)
check("approach component fires during closing", "approach" in comps,
      f"saw {sorted(comps)}")
check("engage component fires when close", "engage" in comps,
      f"saw {sorted(comps)}")

comps_narek = test_components(narek=True)
check("narek component fires when narek_reward=True",
      "narek" in comps_narek, f"saw {sorted(comps_narek)}")

# Time component fires every step
env_t = make_env()
env_t.reset(seed=27)
idx_idle = next(i for i, a in enumerate(DISCRETE_ACTION_MAP) if a == (0.0, 0.0))
_, _, _, _, info = env_t.step(idx_idle)
env_t.close()
comps_t = info.get("reward_components_step", {}) or {}
check("time component fires every step", "time" in comps_t,
      f"saw {list(comps_t.keys())}")


# =============================================================================
# N. yaw_rate_proxy responds to tank-spin
# =============================================================================
section("N. yaw_rate_proxy")

def test_yaw_rate():
    env = make_env()
    env.reset(seed=28)
    # Tank-spin: left forward, right reverse → strong yaw signal
    idx_spin = next(i for i, a in enumerate(DISCRETE_ACTION_MAP)
                    if a == (1.0, -1.0))
    obs0, _, _, _, _ = env.step(idx_spin)
    obs1, _, _, _, _ = env.step(idx_spin)
    obs2, _, _, _, _ = env.step(idx_spin)
    env.close()
    # obs[9] is yaw_rate_proxy. Tank-spin (+1, -1) → diff=+2 →
    # value += 2*0.1=0.2, *=0.9 → 0.18, then 0.18+0.2=0.38 *0.9=0.342...
    # After 3 steps should be > 0.3 in magnitude.
    return abs(obs2[9])

magnitude = test_yaw_rate()
check("yaw_rate_proxy responds to sustained tank-spin",
      magnitude > 0.3, f"|obs[9]| after 3 spin steps = {magnitude:.3f}")


# =============================================================================
# O. IR sensors: noise + dropout DR
# =============================================================================
section("O. IR sensor DR")

def test_ir_noise():
    # Sample many obs at fixed geometry. Noise + dropout should produce
    # std > 0 across samples for the front IR.
    env = make_env()
    env.reset(seed=29)
    idx_idle = next(i for i, a in enumerate(DISCRETE_ACTION_MAP) if a == (0.0, 0.0))
    fronts = []
    for s in range(50):
        # Re-pin geometry each step so randomness is sensor-only.
        p.resetBasePositionAndOrientation(env.robot_id,
            [0, 0, 0.04], p.getQuaternionFromEuler([0, 0, 0]))
        p.resetBasePositionAndOrientation(env.enemy_id,
            [0.20, 0, 0.04], p.getQuaternionFromEuler([0, 0, 0]))
        obs, _, _, _, _ = env.step(idx_idle)
        fronts.append(obs[0])
    env.close()
    return float(np.std(fronts)), float(np.mean(fronts))

std, mean = test_ir_noise()
check("IR sensor noise produces sample variance",
      std > 0.001, f"std={std:.4f} mean={mean:.3f} over 50 samples")


# =============================================================================
# P. Flank reward (alg/improvment)
# =============================================================================
section("P. Flank reward")

def stage_rear_push(env, facing_out: bool):
    """Pin the enemy near the +x edge and the agent just behind it on the
    center side, both facing +x. With the enemy facing OUTWARD the agent
    is behind it and a forward push drives enemy_r up (flank should fire);
    facing the agent (inward) makes the same push a frontal contact."""
    env._apply_enemy_control = lambda: None  # freeze enemy
    qa = p.getQuaternionFromEuler([0, 0, 0.0])               # agent faces +x
    qe = p.getQuaternionFromEuler([0, 0, 0.0 if facing_out else math.pi])
    p.resetBasePositionAndOrientation(env.robot_id, [0.16, 0, 0.04], qa)
    p.resetBasePositionAndOrientation(env.enemy_id, [0.25, 0, 0.04], qe)
    p.resetBaseVelocity(env.robot_id, [0, 0, 0], [0, 0, 0])
    p.resetBaseVelocity(env.enemy_id, [0, 0, 0], [0, 0, 0])

def flank_rollout(facing_out, flank_on, n=40):
    env = make_env(flank_reward=flank_on)
    env.reset(seed=31)
    stage_rear_push(env, facing_out)
    idx_fwd = next(i for i, a in enumerate(DISCRETE_ACTION_MAP) if a == (1.0, 1.0))
    saw = set()
    for _ in range(n):
        _, _, term, trunc, info = env.step(idx_fwd)
        saw.update((info.get("reward_components_step") or {}).keys())
        if term or trunc:
            break
    paid = env._flank_paid
    env.close()
    return saw, paid

saw_rear, paid_rear = flank_rollout(facing_out=True, flank_on=True)
check("flank fires when pushing the enemy out from behind",
      "flank" in saw_rear and paid_rear > 0.0, f"flank_paid={paid_rear:.3f}")

saw_front, _ = flank_rollout(facing_out=False, flank_on=True)
check("flank does NOT fire on a frontal push",
      "flank" not in saw_front, f"saw {sorted(saw_front)}")

saw_off, paid_off = flank_rollout(facing_out=True, flank_on=False)
check("flank silent when flank_reward=False",
      "flank" not in saw_off and paid_off == 0.0, f"saw {sorted(saw_off)}")

check("flank reward never exceeds the per-episode cap",
      paid_rear <= FLANK_EPISODE_CAP + 1e-9,
      f"flank_paid={paid_rear:.3f} <= cap={FLANK_EPISODE_CAP}")

# Cap clamp: with the budget already spent, a paying step adds nothing.
def flank_cap_clamp():
    env = make_env(flank_reward=True)
    env.reset(seed=32)
    stage_rear_push(env, facing_out=True)
    idx_fwd = next(i for i, a in enumerate(DISCRETE_ACTION_MAP) if a == (1.0, 1.0))
    for _ in range(6):                       # establish contact + a push reference
        env.step(idx_fwd)
    env._flank_paid = FLANK_EPISODE_CAP      # exhaust the budget
    _, _, _, _, info = env.step(idx_fwd)
    comps = info.get("reward_components_step") or {}
    env.close()
    return comps.get("flank", 0.0)
flank_after_cap = flank_cap_clamp()
check("flank pays nothing once the episode cap is reached",
      flank_after_cap == 0.0, f"flank={flank_after_cap}")

env_fl = make_env(flank_reward=True, tracking_reward=True)
check("flank_reward forces the conflicting tracking_reward off",
      env_fl.flank_reward and not env_fl.tracking_reward)
env_fl.close()

# Observable-proxy edge avoidance: penalize a blind forward charge (no
# target in the cone) and the rear line sensors over the border. Helper
# stages the agent at (px,py) facing +x (yaw 0) with the enemy at enemy_xy
# and runs one step of `action`; returns the named edge components.
def edge_components(px, py, action, enemy_xy=(10.0, 10.0), yaw=0.0):
    env = make_env(edge_avoid_reward=True)
    env.reset(seed=40)
    env._apply_enemy_control = lambda: None
    qa = p.getQuaternionFromEuler([0, 0, yaw])
    p.resetBasePositionAndOrientation(env.robot_id, [px, py, 0.04], qa)
    p.resetBasePositionAndOrientation(
        env.enemy_id, [enemy_xy[0], enemy_xy[1], 0.04],
        p.getQuaternionFromEuler([0, 0, 0]))
    idx = next(i for i, a in enumerate(DISCRETE_ACTION_MAP) if a == action)
    _, _, _, _, info = env.step(idx)
    env.close()
    return info.get("reward_components_step") or {}

FWD = (1.0, 1.0)
IDLE = (0.0, 0.0)
# Center, driving forward, nothing detected -> blind-charge penalty.
check("blind-charge penalty fires (forward + no target)",
      edge_components(0.0, 0.0, FWD).get("blindcharge", 0.0) < 0.0)
# Same but idle -> no blind-charge (not driving forward).
check("blind-charge silent when not driving forward",
      "blindcharge" not in edge_components(0.0, 0.0, IDLE))
# Opponent detected close ahead -> no blind-charge (target in the cone).
check("blind-charge silent when a target is detected",
      "blindcharge" not in edge_components(0.0, 0.0, FWD, enemy_xy=(0.30, 0.0)))
# Agent backed toward the +x rim (facing -x so its rear is over the border)
# -> rear-edge penalty.
check("rear-edge penalty fires when a rear line sensor is over the border",
      edge_components(0.31, 0.0, IDLE, yaw=math.pi).get("rearedge", 0.0) < 0.0)

def edge_off_default():
    env = make_env()  # edge_avoid_reward defaults False
    env.reset(seed=1)
    _, _, _, _, info = env.step(0)
    env.close()
    comps = info.get("reward_components_step") or {}
    return "blindcharge" in comps or "rearedge" in comps
check("edge proxy reward off by default", not edge_off_default())

# Hardcoded safety override (observable, deployable).
def safety_env():
    env = make_env(safety_override=True)
    env.reset(seed=44)
    env.last_seen_dir = 0.0
    return env

clear = [None, None, None]   # no hit -> all distances clear
# Anti-blind-charge: forward + nothing detected + last_seen lost -> scan-spin.
_se = safety_env()
check("safety: blind forward charge -> scan-spin",
      _se._apply_safety_override(1.0, 1.0, clear) == (1.0, -1.0))
# Target detected ahead -> pass the forward command through.
check("safety: forward passes when a target is detected",
      _se._apply_safety_override(1.0, 1.0, [0.1, None, None]) == (1.0, 1.0))
_se.close()
# Rear line sensor over the border -> drive inward to recover (priority).
_se2 = make_env(safety_override=True)
_se2.reset(seed=44)
qb = p.getQuaternionFromEuler([0, 0, math.pi])  # face -x so rear is at +x rim
p.resetBasePositionAndOrientation(_se2.robot_id, [0.31, 0, 0.04], qb)
check("safety: rear-edge reflex -> drive inward",
      _se2._apply_safety_override(1.0, -1.0, clear) == (1.0, 1.0))
_se2.close()


# =============================================================================
# Q. Frame stacking (obs_stack)
# =============================================================================
section("Q. Frame stacking")

def test_frame_stack():
    base = make_env()
    env = RawDistanceStack(base, k=DEFAULT_STACK_K)
    ok_space = env.observation_space.shape == (stacked_dim(DEFAULT_STACK_K),)
    obs, _ = env.reset(seed=33)
    # On reset the ring is filled with frame-0 distances -> all K distance
    # blocks identical; engineered tail passes through from the base obs.
    d0 = obs[:3]
    blocks_equal = all(np.allclose(obs[3 * j:3 * j + 3], d0) for j in range(DEFAULT_STACK_K))
    # Step and confirm the newest block updates while old blocks shift.
    obs1, _, _, _, _ = env.step(0)
    shape_ok = obs1.shape == (stacked_dim(DEFAULT_STACK_K),)
    env.close()
    return ok_space, blocks_equal, shape_ok

space_ok, reset_ok, step_ok = test_frame_stack()
check("RawDistanceStack exposes the 21-D stacked space", space_ok,
      f"expected ({stacked_dim(DEFAULT_STACK_K)},)")
check("reset replicates frame-0 across all stack slots", reset_ok)
check("stacked obs keeps shape after a step", step_ok)

# Offline windowing matches the documented oldest-first layout + episode reset.
def test_windowing():
    # Two 3-step episodes; distances tick up so blocks are distinguishable.
    obs = np.zeros((6, 12), dtype=np.float32)
    for i in range(6):
        obs[i, :3] = (i % 3) * 0.1          # 0.0,0.1,0.2 | 0.0,0.1,0.2
    dones = np.array([0, 0, 1, 0, 0, 1], dtype=bool)
    out = stack_from_trajectory(obs, dones, k=4)
    # Row 0 (episode start): all four blocks replicate frame 0 (=0.0).
    start_ok = np.allclose(out[0, :12], 0.0)
    # Row 2 (3rd frame of ep0): blocks = [d0,d0,d1,d2] = [0,0,.1,.2].
    mid_ok = np.allclose(out[2, :12], [0, 0, 0, 0, 0, 0, .1, .1, .1, .2, .2, .2])
    # Row 3 is a fresh episode start -> must NOT see ep0 frames.
    reset_ok = np.allclose(out[3, :12], 0.0)
    return start_ok and mid_ok and reset_ok

check("stack_from_trajectory respects layout + episode boundaries",
      test_windowing())


# =============================================================================
# R. Behavioral + hard-sensor DR (alg/improvment)
# =============================================================================
section("R. Behavioral + hard-sensor DR")

def opp_dr_samples(dr_on, n=12):
    env = make_env(opponent_dr=dr_on)
    speeds, gains, lats = set(), set(), set()
    for s in range(n):
        env.reset(seed=100 + s)
        speeds.add(round(env._enemy_speed_mult, 5))
        gains.add(round(env._enemy_tracking_gain, 5))
        lats.add(env._enemy_action_latency)
    env.close()
    return speeds, gains, lats

s_on, g_on, l_on = opp_dr_samples(True)
check("opponent_dr varies enemy speed/turn/latency across episodes",
      len(s_on) > 1 and len(g_on) > 1,
      f"speeds={len(s_on)} gains={len(g_on)} lats={sorted(l_on)}")
check("opponent_dr stays inside the declared ranges",
      all(OPP_DR_SPEED_RANGE[0] <= x <= OPP_DR_SPEED_RANGE[1] for x in s_on)
      and all(OPP_DR_TRACKING_RANGE[0] <= x <= OPP_DR_TRACKING_RANGE[1] for x in g_on)
      and l_on.issubset(set(OPP_DR_REACTION_CHOICES)))

s_off, g_off, l_off = opp_dr_samples(False)
check("opponent_dr OFF is identity (speed=gain=1, no skew)",
      s_off == {1.0} and g_off == {1.0})

def hard_dr_probe(hard_on):
    env = make_env(sensor_hard_dr=hard_on)
    env.reset(seed=7)
    sigma = env._tof_noise_sigma
    bias_nonzero = bool(np.any(env._tof_calib_bias != 0.0))
    flip = env._line_flip_prob
    lat = env._action_latency
    env.close()
    return sigma, bias_nonzero, flip, lat

sig_on, bias_on, flip_on, _ = hard_dr_probe(True)
sig_off, bias_off, flip_off, lat_off = hard_dr_probe(False)
check("hard sensor DR widens ToF sigma",
      abs(sig_on - DR_TOF_NOISE_SIGMA_PCT_HARD * ENEMY_FAR_DIST) < 1e-9
      and abs(sig_off - DR_TOF_NOISE_SIGMA_PCT * ENEMY_FAR_DIST) < 1e-9,
      f"on={sig_on:.4f} off={sig_off:.4f}")
check("hard sensor DR adds calibration bias + line flips (off=clean)",
      bias_on and flip_on > 0.0 and not bias_off and flip_off == 0.0)
check("standard latency profile is pinned to 1 when hard DR off",
      lat_off == 1, f"lat_off={lat_off}")


# =============================================================================
# S. Held-out opponent zoo (alg/improvment)
# =============================================================================
section("S. Held-out zoo")

check("zoo registry/weights stay consistent with held-out entries",
      abs(sum(OPPONENT_WEIGHTS.values()) - 1.0) < 1e-9
      and set(OPPONENT_WEIGHTS) == set(OPPONENT_REGISTRY)
      and all(OPPONENT_WEIGHTS[h] == 0.0 for h in HELD_OUT_OPPONENT_IDS),
      f"held_out={HELD_OUT_OPPONENT_IDS}")

def heldout_runs(name):
    env = make_env(force_opponent_id=name)
    env.reset(seed=5)
    ok = True
    for _ in range(25):
        _, _, term, trunc, info = env.step(env.action_space.sample())
        ok = ok and (info["opponent_id"] == name)
        if term or trunc:
            env.reset()
    env.close()
    return ok

check("held-out opponents instantiate and drive in-env",
      all(heldout_runs(n) for n in HELD_OUT_OPPONENT_IDS))


# =============================================================================
# Summary
# =============================================================================
section("Summary")
print(f"  PASS: {len(_passed)}", flush=True)
print(f"  FAIL: {len(_failed)}", flush=True)
if _failed:
    print("\n  Failures:")
    for n in _failed:
        print(f"    - {n}")
sys.exit(0 if not _failed else 1)
