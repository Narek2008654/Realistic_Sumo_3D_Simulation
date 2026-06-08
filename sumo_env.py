"""Mini-sumo dohyo — 4D obs, 2D continuous action, original reward stack.

This is the env shape that produced the 45-55% SAC win-rate. One RL step
≈ 45 ms (11 physics ticks at 240 Hz). The opponent is now the Novamax
Professional Mini Sumo Kit (heavy steel, 5 IR sensors + 2 line sensors,
400 RPM motors); see NovamaxController for the firmware port.

* Observation space — Box(low=-1, high=1, shape=(12,), float32):
    [front_norm, left_norm, right_norm, last_seen_dir, line_l, line_r,
     prev_left, prev_right, engagement, yaw_rate_proxy, front_ir_delta,
     lateral_ir_delta]

* Action space — discrete 9-action grid (default, used by the DQN) OR
    continuous Box(2,); both decode to (left_motor, right_motor) in [-1, 1].

* Rewards (per step):
    terminal win / loss-by-reason (push_loss, self_out, mutual_out) /
    timeout, plus a positive-shaping stack: approach, engage, wedge
    (outward enemy displacement), optional narek / tracking /
    action-consistency, and a small per-step time penalty.
    (See the REWARD_* constants for magnitudes.)
"""

from __future__ import annotations

import math
import os
import time
from typing import Optional

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from webapp.shared.hardware_spec import HardwareSpec


# ---------------------------------------------------------------------------
# World / robot constants (meters / seconds).
# ---------------------------------------------------------------------------
DOHYO_DIAMETER = 0.70
DOHYO_RADIUS = DOHYO_DIAMETER / 2.0
BORDER_WIDTH = 0.025
INNER_RADIUS = DOHYO_RADIUS - BORDER_WIDTH
DOHYO_THICKNESS = 0.02
BLACK_TOP_THICKNESS = DOHYO_THICKNESS * 0.05
DOHYO_TOP_Z = DOHYO_THICKNESS + BLACK_TOP_THICKNESS

ROBOT_WIDTH = 0.098
ROBOT_FRONT_EXTENT = 0.0496
WHEEL_RADIUS = 0.010
WHEEL_TRACK_HALF = 0.0401
ROBOT_URDF = os.path.join(os.path.dirname(__file__), "assets", "robot.urdf")

# Novamax Professional Mini Sumo Kit (opponent).
NOVAMAX_URDF = os.path.join(os.path.dirname(__file__), "assets", "novamax.urdf")
# 800 RPM gearmotors, 16.5 mm wheel radius -> ~1.38 m/s top speed.
# At 0.45 kg total mass, that's ~0.62 kg·m/s of momentum -- ~9× the agent's.
NOVAMAX_RPM = 800.0
NOVAMAX_WHEEL_OMEGA = (NOVAMAX_RPM / 60.0) * 2.0 * math.pi  # 83.78 rad/s
# 5 IR distance sensors (60 cm range). Each entry is
# (start_x_body, start_y_body, dir_x_body, dir_y_body) in the enemy's
# local frame; +X forward, +Y left.
NOVAMAX_IR_RANGE = 0.60
_INV_SQRT2 = math.sqrt(0.5)
NOVAMAX_IR_SENSORS: dict[str, tuple[float, float, float, float]] = {
    "fc": (0.0495,  0.000,        1.0,         0.0),
    "fl": (0.0495,  0.045,  _INV_SQRT2,  _INV_SQRT2),
    "fr": (0.0495, -0.045,  _INV_SQRT2, -_INV_SQRT2),
    "sl": (0.0000,  0.045,        0.0,         1.0),
    "sr": (0.0000, -0.045,        0.0,        -1.0),
}
# 2 downward QTR line sensors at the front-left/right plow corners,
# expressed as (body_x, body_y) in the enemy's local frame.
NOVAMAX_LINE_SENSORS: tuple[tuple[float, float], tuple[float, float]] = (
    (0.0495,  0.045),
    (0.0495, -0.045),
)

# Agent line sensors: two downward QTRs at the rear of the chassis
# bottom face, as (body_x, body_y) in the agent's local frame. They
# return 1.0 when over the white border ring of the dohyo.
AGENT_LINE_SENSORS: tuple[tuple[float, float], tuple[float, float]] = (
    (-0.02384,  0.0404),   # rear-left
    (-0.02384, -0.0404),   # rear-right
)

AGENT_RGBA = (0.85, 0.15, 0.15, 1.0)
ENEMY_RGBA = (0.15, 0.30, 0.85, 1.0)

ENEMY_SENSOR_MAX_M = 1.5
SPAWN_RADIUS = 0.25

SIM_TIMESTEP = 1.0 / 240.0
# 1 RL step = SUBSTEPS_PER_STEP physics ticks at 240 Hz.
# 10 ticks ≈ 41.67 ms ≈ 24 Hz, targeting a 25 Hz Arduino loop. Physics
# tick stays at 240 Hz so the per-substep edge watchdog still runs at
# ~4.2 ms granularity.
SUBSTEPS_PER_STEP = 10
STEP_DT_SECONDS = SUBSTEPS_PER_STEP * SIM_TIMESTEP   # 0.04167 s
TARGET_ACTION_HZ = 25.0
# Tolerance is ~2 ms because 240 Hz / 25 Hz = 9.6 substeps is not an
# integer; 10 substeps lands at 24 Hz, which is within 4% of target.
assert abs(STEP_DT_SECONDS - 1.0 / TARGET_ACTION_HZ) < 2.0e-3, (
    f"env.step() is {STEP_DT_SECONDS * 1000:.2f} ms; "
    f"target {1000.0 / TARGET_ACTION_HZ:.2f} ms ({TARGET_ACTION_HZ} Hz)"
)

# alg/improvment: hardcoded opening charge — drive straight forward for the
# first OPENING_CHARGE_SECONDS of each match (the standard sumo opening; the
# user's robot and the zoo openers do the same). Opt-in; mirrored 1:1 in
# firmware. ~100 ms quantizes to 2 env steps (2 * 41.7 ms = 83 ms).
OPENING_CHARGE_SECONDS = 0.100
OPENING_CHARGE_STEPS = max(1, round(OPENING_CHARGE_SECONDS / STEP_DT_SECONDS))

# alg/improvment: spawn guard — for the first SPAWN_GUARD_SECONDS of a match,
# if the policy commands net-backward, drive forward instead (otherwise leave
# the action alone). The robot spawns close to the rear rim, and a self-out
# timing study found ~⅓ of self-outs happen in the first ~1 s from backing
# into the edge; after that, reversing can be legitimate. Mirrored in firmware.
SPAWN_GUARD_SECONDS = 1.0
SPAWN_GUARD_STEPS = max(1, round(SPAWN_GUARD_SECONDS / STEP_DT_SECONDS))

# alg/improvment: anti-stall override — if the policy commands (near-)idle for
# ANTISTALL_STEPS in a row, force a forward charge instead. A hard backstop on
# top of the soft still-penalty, so the bot never freezes mid-match. Mirrored
# in firmware. ~1 s of standing still = too long.
ANTISTALL_SECONDS = 1.0
ANTISTALL_STEPS = max(1, round(ANTISTALL_SECONDS / STEP_DT_SECONDS))

# Initial agent-only forward charge at episode start (500 ms). The red
# robot drives both wheels full forward; the enemy stays still until
# the policy takes over.
# Initial-charge handicap-removal phase. Was 500 ms (the agent drove
# full forward while the enemy was commanded stationary) but with a
# 25 cm spawn radius the agent crossed the dohyo and rammed the
# stationary enemy before tick 0 of training, scattering both bots
# in random directions. Disabled in Run 7 — both controllers run from
# tick 0, both bots start stationary at SPAWN_RADIUS facing centre.
INITIAL_CHARGE_MS = 0
INITIAL_CHARGE_TICKS = int(round((INITIAL_CHARGE_MS / 1000.0) / SIM_TIMESTEP))

# --- Hyper-realistic motor physics --------------------------------
# Agent motors: N20 12 V, 400 RPM, ~1.2 kg·cm stall torque.
#   max angular vel = 400 * 2*pi / 60 = 41.888 rad/s
#   max stall force = 0.12 N·m
AGENT_MAX_RAD = 41.88
AGENT_MAX_FORCE = 0.12

# NovaMax motors: 16 mm gearmotor, 800 RPM, ~5 kg·cm stall torque.
#   max angular vel = 800 * 2*pi / 60 = 83.776 rad/s
#   max stall force = 0.50 N·m  (out-pushes the agent's 0.12 N·m)
NOVAMAX_MAX_RAD = NOVAMAX_WHEEL_OMEGA   # 83.776 rad/s
NOVAMAX_MAX_FORCE = 0.50

# Curriculum levels for the NovaMax opponent. Each entry is
# (max_velocity_rad_s, max_stall_torque_Nm). Level 1 matches the agent's
# N20 spec exactly; level 3 is full real spec — fast AND stronger than
# the agent in pushing contests.
NOVAMAX_LEVELS: dict[int, tuple[float, float]] = {
    1: (9.95,   0.12),    # nerfed — matches agent
    2: (40.0,   0.25),    # medium speed, ~2× agent torque
    3: (83.78,  0.50),    # boss / real spec (800 RPM, 4× agent torque)
}

# Legacy aliases kept so existing call sites don't break; these are
# the AGENT's limits, used only when applying the agent's wheel motors.
WHEEL_MAX_TORQUE = AGENT_MAX_FORCE
WHEEL_OMEGA_FWD = AGENT_MAX_RAD
FORWARD_SIGN = 1.0

# Raised 1.0 → 2.0 to fix the rear-push bug: at 1.0 the per-wheel
# thrust margin over a braking opponent was only ~0.4 N, so even a
# stationary enemy resisted being pushed. Real silicone tire on wood
# is μ ≈ 1.5–2.0, so this stays in physical range.
WHEEL_FRICTION = 2.0
# Dropped from 0.4 → 0.05 to fix bot-on-bot push-stalemate. PyBullet's
# pairwise friction is the geometric mean of the two surfaces; with 0.4
# on both chassis (μ_pair = 0.4) tip-to-tip contact locks up because
# the lateral component resists side-slip. At 0.05 contact slides off
# instead of wedging. Wheels keep μ=1.0 for ground grip.
CHASSIS_FRICTION = 0.05

# Stuck-detector: if both bots are in continuous contact with near-zero
# net motion for STUCK_DETECTION_TICKS, inject equal-and-opposite
# lateral velocity deltas perpendicular to the contact normal. Breaks
# symmetric deadlocks that survive the friction drop above.
STUCK_DETECTION_TICKS    = 8       # ~333 ms @ 24 Hz
STUCK_VELOCITY_THRESHOLD = 0.05    # m/s; below this counts as "stationary"
STUCK_KICK_SPEED         = 0.10    # m/s lateral Δv applied to each bot

# Domain randomization (Step 4 — exposed as named constants so any
# script tweaking realism can find them in one place).
DR_FRICTION_RANGE = (0.4, 0.7)         # silicone wheels on smooth wooden dohyo (measured μ ≈ 0.55, ±0.15)
DR_MASS_RANGE = (0.95, 1.05)           # uniform mult on each robot's chassis mass
DR_VELOCITY_RANGE = (0.85, 1.0)        # uniform per-episode "battery sag" on max wheel ω
DR_DEADZONE_RANGE = (0.20, 0.35)       # uniform per-episode PWM dead zone (sym, both wheels)
DR_TOF_NOISE_SIGMA_PCT = 0.02          # gaussian σ as fraction of ENEMY_FAR_DIST
# alg/improvment: disabled (was 0.01). Random per-step sensor "turn-off"
# blinds the agent for a tick and triggers some self-outs / unstable play;
# the real VL53L0X rarely drops, so 0 is both more stable and more
# deploy-realistic. Mild gaussian ToF noise is kept for sim2real.
DR_TOF_DROPOUT_PROB = 0.0              # per-sensor per-step "max-range" return
# Pinned to a single value for the first 1M-step run so we don't debug
# convergence + variable latency at the same time. After convergence is
# confirmed, widen to (1, 2) for a robustness pass.
DR_ACTION_LATENCY_CHOICES = (1,)       # discrete uniform queue depth (steps)

# alg/improvment: a harder sensor-noise profile for training robustness,
# opt-in via the env's sensor_hard_dr flag. Eval/audit keep the STANDARD
# profile above so per-opponent win-rates stay comparable to the committed
# baseline (the new model and the old one are scored under identical eval
# noise). These model real VL53L0X / QTR failure modes the firmware can't
# pre-filter; they are NOT mirrored in arduino_obs_logic.h because the
# physical sensors produce them for free.
DR_ACTION_LATENCY_CHOICES_HARD = (1, 2)   # widen latency under hard DR
DR_TOF_NOISE_SIGMA_PCT_HARD = 0.03        # wider gaussian σ (≈24 mm)
DR_TOF_CALIB_BIAS_SIGMA = 0.008           # per-channel per-episode bias (m)
DR_TOF_STUCK_PROB = 0.05                  # per-episode chance one ToF channel sticks
DR_TOF_STUCK_HOLD_PROB = 0.5              # while stuck, per-step chance to hold last
DR_LINE_FLIP_PROB = 0.005                 # per-step per-line-sensor bit flip

# alg/improvment: behavioral opponent domain randomization (training
# only — disabled for eval so per-opponent win-rates stay reproducible).
# Applied at the env boundary so the controller scripts stay pure: each
# episode draws a wheel-speed multiplier (decouples "fast" from the
# torque cap), a turn-sharpness multiplier (varies arc/spin tightness),
# and an independent reaction latency. Turns the 5 fixed zoo scripts into
# a continuum so the policy generalizes to robots it never trained on.
OPP_DR_SPEED_RANGE = (0.7, 1.0)        # uniform mult on enemy wheel speed
OPP_DR_TRACKING_RANGE = (0.8, 1.2)     # uniform mult on enemy turn sharpness
OPP_DR_REACTION_CHOICES = (1, 2, 3)    # enemy action-queue depth (steps)

# Legacy aliases (kept so external scripts importing these still work,
# but they now map onto the new DR_* knobs).
DOHYO_FRICTION_RANGE = DR_FRICTION_RANGE
ROBOT_MASS_NOMINAL = 0.5
ROBOT_MASS_VARIATION = 0.05            # = (1.05 - 0.95) / 2 from DR_MASS_RANGE
SENSOR_DEAD_PROB = 0.0                 # superseded by DR_TOF_DROPOUT_PROB

# Sensor max range / observation normalization scale.
ENEMY_FAR_DIST = 0.80

# last_seen_dir state machine (Step 5). Hysteresis prevents thrashing
# when two sensors are near-equal; decay zeroes the latch when we lose
# track of the opponent for > LAST_SEEN_DECAY_SECONDS. Mirrored byte-
# for-byte in arduino_obs_logic.h.
LAST_SEEN_HYSTERESIS_RATIO = 1.1
LAST_SEEN_DECAY_SECONDS = 0.5
LAST_SEEN_DECAY_STEPS = max(
    1, int(round(LAST_SEEN_DECAY_SECONDS / STEP_DT_SECONDS))
)

# Run 3: collapsed reward stack — terminal + ONE shaping signal.
#
# The run-1/run-2 stack had 8+ competing shaping signals (edge penalty,
# coward, idle, flank, push_delta, focus, hunter, track, push) all
# pulling at once. Net per-episode shaping went *negative* in run 1
# (-18 mean) and the policy learned to minimise punishment by not
# engaging. Run 3 strips everything except a single dense approach
# signal that is positive-only — retreating gives exactly zero, not a
# negative reward, so there is no deadlock between "don't reverse" and
# "stay away from the edge".
# Run 9: reward magnitudes rescaled by 100× to bring Q-values into the
# O(1)-O(10) range SAC/TD3 critic optimisation is conditioned for. With
# ±1000 terminals, target Q-values ran in the hundreds and critic_loss
# oscillated 3-3000; at ±10 the same policy decisions produce well-
# behaved gradients. Reward *ratios* are unchanged so policy preference
# is identical.
REWARD_WIN = 10.0             # opponent off ring (any cause)
# Loss rewards now split by termination_reason. push_loss is the
# "neutral" baseline — opponent earned it. self_out is punished hardest
# because the policy has full control over not driving off the edge.
REWARD_LOSE_PUSH   = -15.0    # opponent pushed us off
REWARD_LOSE_MUTUAL = -20.0    # both bots off in same step
# alg/improvment: self-outs are the dominant loss at mult 3.0 (~82% of
# losses, 92% of them forward drive-offs). The forward edge isn't
# observable (rear-only line sensors), so reward shaping couldn't fix it;
# this run instead HEAVILY punishes the terminal self-out (-50 -> -100,
# 10x the win) to push the policy toward more conservative forward play
# from scratch. Watch for over-correction into passive timeout-stalling
# (timeout -10 becomes a "safe haven" vs a -100 self-out).
REWARD_LOSE_SELF   = -50.0    # walked off without recent contact (HEAVY).
                              # -100 was tried and did not reduce self-outs
                              # (the forward edge is unobservable) — reverted
                              # to the proven -50 to avoid timid play.
REWARD_TIMEOUT     = -10.0
# alg/improvment: per-step penalty for standing still (barely moving while not
# in contact) — discourages freezing, timeout-stalling, and in-place spinning.
REWARD_STILL_PENALTY = -0.05
STILL_SPEED_THRESHOLD = 0.05   # m/s linear speed below this counts as "still"
# alg/improvment: slight per-step penalty for going backward (net-reverse
# command) — discourages retreating. Smaller than the still penalty.
REWARD_BACKWARD_PENALTY = -0.02

# Bearing-based tracking reward (Run 14 / 2D port). Fires only when the
# opponent is within REWARD_TRACK_RANGE (close enough that orientation
# matters). Rewards keeping the opponent in the forward cone, punishes
# letting it get behind.
REWARD_TRACK_FRONT   = +0.05
REWARD_TRACK_SIDE    = -0.10
REWARD_TRACK_BEHIND  = -0.15
REWARD_TRACK_RANGE   = 0.50               # only fire when dist < 50 cm
TRACK_FRONT_HALF_RAD = math.radians(30)
TRACK_SIDE_HALF_RAD  = math.radians(90)

# Anti-flicker (Run 16 / 2D port). Penalises any tick where the
# commanded motor pair differs from the previous tick's commanded pair.
# Same magnitude as REWARD_TIME so flicker pays the same rate as time.
REWARD_ACTION_CHANGE_PENALTY = -0.002
# Small per-step cost: -0.005 * 600 = -3 max additional bias (was -300).
REWARD_TIME = -0.005
# Self-out vs push-loss vs mutual-out are distinguished in
# info["termination_reason"] AND now in the reward magnitude itself.
CONTACT_RECENT_STEPS = 2

# Wedge-engagement reward (Run 7 — outward Δr signal).
#
# Lineage:
#   v1 (Run 6): contactNormalOnB.z thresholded — sign was inverted,
#               fired on tip-to-tip collisions without real lift.
#   v2:        enemy_z > rest + 2 mm — mechanically honest lift detector
#               but rewarded sustained holding regardless of progress
#               toward the edge. Diagnostic showed agent could wedge
#               for 232+ ticks without moving the opponent radially.
#   v3 (this): per-tick CHANGE in opponent radial distance while in
#               contact. Captures both sustained pushing AND tip-and-
#               flick (brief contact, opponent flies outward). A fast
#               flick that displaces the enemy 5 cm in 3 ticks scores
#               +150 (≫ a 200-tick stalemate that moves them 1 cm =
#               +30). The terminal +1000 win remains the dominant
#               signal; this just tells the policy "any progress
#               toward the edge while in contact is good".
#
# Scaling: 10 reward per metre = 0.10 reward per cm of outward push.
# An episode that flicks the enemy fully off (35 cm outward) tops
# out at +3.5; a stalemate-while-wedged scores ~0 because Δr is
# small per tick. Win signal +10 still dominates.
REWARD_WEDGE_PER_M = 10.0

# Approach reward (post-rescale): 0.75 still rewards closing distance
# but stays subordinate to the wedge engagement signal.
REWARD_APPROACH = 0.75
APPROACH_MIN = 0.001          # filter pure float-noise reward only

# Engagement timer obs feature AND reward source. Counts consecutive
# ticks where the front laser sees the enemy at wedge-contact range
# (front_norm < ENGAGEMENT_FRONT_THRESHOLD). The normalised count
# goes to the policy as obs[8]; the same condition pays a per-tick
# REWARD_ENGAGE_PER_TICK shaping bonus so the policy gets a dense,
# positive gradient toward "be in wedge-engagement position",
# independent of whether physical contact actually happens.
#
# Run 10 motivation: Run 9 ended phase 1 with wedge_engaged_ticks=0
# averaged across episodes — the policy never made contact at all,
# so the wedge reward never fired and the only engagement signal
# was the (small, terminal-dominated) win bonus. Without a dense
# positioning reward there's no gradient that says "getting closer
# to wedge contact is good"; with REWARD_ENGAGE_PER_TICK=0.10 the
# policy gets +3 max per fully-engaged episode, comparable to the
# ±10 terminal but cheap to earn — turns approach-then-engage from
# a sparse-reward bandit problem into a dense gradient ascent.
ENGAGEMENT_FRONT_THRESHOLD = 0.15
ENGAGEMENT_MAX_STEPS = 30          # ~1.25 s @ 24 Hz
REWARD_ENGAGE_PER_TICK = 0.10

# alg/improvment: flank shaping (opt-in). Rewards driving the enemy
# outward while positioned at its rear/side, so the policy learns to
# attack from the back/left/right instead of only head-on. It fires
# ONLY when the wedge push is already paying (enemy_r rising during
# contact) AND the agent is heading into the enemy, so it cannot be
# farmed by orbiting at range or while being pushed back. Capped per
# episode well below a win, so it can only bias positioning into the
# wedge/win path, never dominate the terminal outcome.
REWARD_FLANK_PER_TICK = 0.08       # < REWARD_ENGAGE_PER_TICK (0.10)
FLANK_RANGE = 0.18                 # m; near-contact only
FLANK_MIN_RAD = math.radians(75)   # agent must be >75° off the enemy's nose
FLANK_HEADING_HALF_RAD = math.radians(75)  # agent front within ±75° of enemy
FLANK_EPISODE_CAP = 2.0            # << REWARD_WIN (10), << |REWARD_LOSE_SELF| (50)

# alg/improvment: OBSERVABLE-PROXY edge avoidance (opt-in). An earlier
# version used the agent's true radius/velocity — privileged state the
# deployed 21-D policy never sees, so it could not transfer (a critique
# confirmed it as reward-hacking, and two finetunes moved self-outs 0%).
# Instrumentation showed 92% of self-outs are FORWARD drive-offs: the agent
# charges straight ahead off the front rim, usually after losing the
# opponent. There is no forward edge sensor, so we penalize the observable
# BEHAVIOR that precedes it — driving straight forward with NOTHING detected
# in the forward cone (nudging the policy to turn and re-acquire instead of
# charging into the void) — plus the rear line sensors crossing the border
# (the one direct, observable "at the edge" cue). Both use only features the
# deployed policy observes, so the learned avoidance transfers to hardware.
REWARD_BLIND_CHARGE = -0.06        # per-step: driving forward with no target
REWARD_REAR_EDGE = -0.30           # per-step: a rear line sensor over the border
PROXY_CLEAR_NORM = 0.5625          # ~45 cm (0.45/0.80): a target must be within
                                   # ~40-50 cm to count as "detected"; charging
                                   # forward with everything farther = blind.

# alg/improvment: HARDCODED safety override (opt-in, deployable in firmware
# 1:1 since it reads only observable signals). Reward shaping cannot stop
# self-outs because the forward edge is unobservable; this deterministic
# layer prevents the two self-out behaviors directly, BEFORE the command
# reaches the motors. Active in both training (the policy co-adapts) and at
# deploy. (1) Anti-blind-charge: if the policy commands net-forward but
# NOTHING is detected (all 3 distances clear) and last_seen is lost, replace
# it with an in-place scan-spin (re-acquire instead of charging into the
# void). (2) Rear-edge reflex (priority): if a rear line sensor is over the
# border, drive inward to recover. Mirrors the NovamaxController/CombatPolicy
# edge reflexes the agent never had.
SAFETY_CLEAR_NORM = 0.90           # nothing within ~0.72 m in a sensor = clear

# Run 11 (DQN port): Narek-style discrete action map + semantic
# action-conditioned reward shaping. The 9-action grid is the
# Cartesian product of (left_motor, right_motor) each ∈ {-1, 0, +1},
# matching the discrete keyboard inputs that produced the offline-
# trained DQN at github.com/Narek2008654/Simulator. The shaped
# rewards reproduce that repo's reward function:
#   idle (0, 0)                              -> -0.2
#   any -1 wheel AND front_norm < threshold  -> -0.5  (retreat-close)
#   any +1 wheel AND front_norm < threshold  -> +0.3  (attack-close)
# Spinning in place (one wheel +1, the other -1) triggers BOTH gates
# at once; net is -0.2 — same as idle, mild discouragement.
DISCRETE_ACTION_MAP: tuple[tuple[float, float], ...] = (
    (-1.0, -1.0), (-1.0,  0.0), (-1.0, +1.0),
    ( 0.0, -1.0), ( 0.0,  0.0), ( 0.0, +1.0),
    (+1.0, -1.0), (+1.0,  0.0), (+1.0, +1.0),
)
NAREK_CLOSE_THRESHOLD = ENGAGEMENT_FRONT_THRESHOLD
NAREK_IDLE_PENALTY    = -0.2
NAREK_RETREAT_PENALTY = -0.5
NAREK_ATTACK_BONUS    = +0.3

# Curriculum.
# `novamax_torque_mult` is the canonical curriculum knob: a scalar in
# [NOVAMAX_TORQUE_MULT_MIN, 3.0]. The lerp anchors at mult=1.0 (matched
# agent strength) and mult=3.0 (full bullet); for mult < 1.0 we LINEARLY
# EXTRAPOLATE below the matched endpoint so phase 0 of the reverse
# curriculum (Run 6) can produce sub-agent opponents the agent's
# wedge can reliably overpower. The MIN floor is a safety bound to
# stop pathological negative-torque values.
# Run 12: floor lowered from 0.5 to 0.3 so offline data collection can
# use very weak opponents (force ~0.01 N·m, near-zero pushing power)
# against which the scripted combat policy reliably wins. The agent
# learns engagement basics from these wins; online curriculum ramps
# back up to mult=3.0.
NOVAMAX_TORQUE_MULT_MIN = 0.3
NOVAMAX_TORQUE_MULT_REF = 1.0   # reference point: opponent matches agent
NOVAMAX_TORQUE_MULT_MAX = 3.0

FALL_Z = 0.0
MAX_EPISODE_STEPS = 600   # 25 s @ 24 Hz; longer episodes let the agent recover from a failed engagement


class NovamaxController:
    """Novamax Professional Mini Sumo Kit firmware port.

    State machine, evaluated once per RL step (~46 ms):
      Priority 1 (survival): if a downward line sensor sees the white edge,
                             reverse for ~200 ms then tank-spin away from
                             the triggered side for another ~200 ms.
      Priority 2 (attack):   route on the 5 IR sensors — front-centre charge,
                             front-side arc, side spin to bring enemy back
                             into the front cone.
      Priority 3 (search):   tank-spin in last-seen direction, or arc-search
                             if the enemy hasn't been seen yet.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.last_seen_side = 0       # -1 left, 0 unseen, +1 right
        self._edge_state: str | None = None      # None / "reverse" / "spin"
        self._edge_timer = 0
        self._edge_spin_dir = 0       # +1 spin right, -1 spin left

    @property
    def is_edge_braking(self) -> bool:
        """True while the controller is reversing/spinning away from edge."""
        return self._edge_state is not None

    def force_edge_brake(self, edge_left: bool, edge_right: bool) -> None:
        """Watchdog hook: jump straight into the reverse phase.

        The env's per-substep watchdog calls this when it detects an
        edge before the next ``decide()`` runs, so the bullet doesn't
        skate off the dohyo in the gap between RL steps.
        """
        if self._edge_state is None and (edge_left or edge_right):
            self._edge_state = "reverse"
            self._edge_timer = self.EDGE_REVERSE_STEPS
            self._edge_spin_dir = +1 if edge_left else -1

    # Real-world NovaMax wheel speeds (rad/s) for the ATTACK state. The env
    # clips them to the active level's max_rad, so level-1 / 2 still nerf
    # the absolute speed.
    FULL = 83.78        # full 800 RPM
    ARC = 40.0          # inside wheel during an arc turn (~380 RPM)
    SPIN = 83.78        # full opposite-direction spin (attack mode only)

    # EDGE-RECOVERY speeds. Spinning at 800 RPM in place creates so much
    # centrifugal force that the silicone wheels lose traction and the
    # chassis slides off the dohyo like a hockey puck. The recovery uses
    # a slow controlled speed, matching a real Arduino's edge PWM.
    EDGE_SAFE_SPEED = 30.0   # rad/s ≈ 286 RPM, ~0.5 m/s
    EDGE_REVERSE_STEPS = 2   # ~92 ms of controlled reverse
    EDGE_SPIN_STEPS = 4      # ~184 ms of controlled spin

    def decide(
        self, ir_hits: dict[str, bool], edge_left: bool, edge_right: bool,
    ) -> tuple[float, float]:
        # ---------- Priority 1: edge avoidance --------------------------
        # Two-phase recovery at SAFE_SPEED so the body keeps traction:
        #   1) reverse for EDGE_REVERSE_STEPS ticks
        #   2) tank-spin away for EDGE_SPIN_STEPS ticks
        if self._edge_state is None and (edge_left or edge_right):
            self._edge_state = "reverse"
            self._edge_timer = self.EDGE_REVERSE_STEPS
            # Spin AWAY from the triggered side. Left edge -> spin right
            # (l=+, r=-) so the body rotates clockwise / faces centre.
            self._edge_spin_dir = +1 if edge_left else -1

        if self._edge_state == "reverse":
            self._edge_timer -= 1
            if self._edge_timer <= 0:
                self._edge_state = "spin"
                self._edge_timer = self.EDGE_SPIN_STEPS
            v = self.EDGE_SAFE_SPEED
            return -v, -v

        if self._edge_state == "spin":
            self._edge_timer -= 1
            if self._edge_timer <= 0:
                self._edge_state = None
            v = self.EDGE_SAFE_SPEED
            return (v, -v) if self._edge_spin_dir > 0 else (-v, v)

        # ---------- Priority 2: attack ----------------------------------
        if ir_hits.get("fc"):
            return self.FULL, self.FULL
        if ir_hits.get("fl"):
            self.last_seen_side = -1
            return self.ARC, self.FULL
        if ir_hits.get("fr"):
            self.last_seen_side = +1
            return self.FULL, self.ARC
        if ir_hits.get("sl"):
            self.last_seen_side = -1
            return -self.SPIN, self.SPIN
        if ir_hits.get("sr"):
            self.last_seen_side = +1
            return self.SPIN, -self.SPIN

        # ---------- Priority 3: search ----------------------------------
        if self.last_seen_side == -1:
            return -self.SPIN, self.SPIN     # tank-spin left
        if self.last_seen_side == +1:
            return self.SPIN, -self.SPIN     # tank-spin right
        # Default wide arc-search if we've never seen the agent.
        return self.ARC * 0.8, self.FULL


class MiniSumoEnv(gym.Env):
    """Fast reactive continuous-control mini-sumo (no macro-actions)."""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        gui: bool = False,
        seed: Optional[int] = None,
        enemy_torque_multiplier: float = 2.0,
        human_enemy: bool = False,
        novamax_level: int = 3,
        novamax_torque_mult: Optional[float] = None,
        force_opponent_id: Optional[str] = None,
        action_space_kind: str = "continuous",
        narek_reward: bool = False,
        tracking_reward: bool = False,
        action_consistency_reward: bool = False,
        flank_reward: bool = False,
        edge_avoid_reward: bool = False,
        safety_override: bool = False,
        opening_charge: bool = False,
        spawn_guard: bool = False,
        antistall: bool = False,
        still_penalty: bool = False,
        backward_penalty: bool = False,
        enemy_as_agent: bool = False,
        enemy_chassis_dr: bool = False,
        mult_dr_range: Optional[tuple] = None,
        opponent_dr: bool = False,
        sensor_hard_dr: bool = False,
        opponent_weights: Optional[dict] = None,
        hardware_spec: Optional[HardwareSpec] = None,
        enemy_hardware_spec: Optional[HardwareSpec] = None,
        extra_opponents: Optional[dict] = None,
        extra_opponent_specs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        # E1b: the robot's sensing + observation contract is driven by a
        # HardwareSpec. None => HardwareSpec.default(), which encodes TODAY's
        # robot byte-identically (3 ToF rays, 2 line sensors, 9 engineered
        # features). Stored as `self.hw_spec` (NOT `self.spec`, which is
        # Gymnasium's reserved EnvSpec slot — shadowing it breaks str(env),
        # gym.make registration, and Wrapper.spec).
        self.hw_spec = hardware_spec if hardware_spec is not None else HardwareSpec.default()
        # add/ui-local: the dohyo (ring) geometry is now spec-driven. The
        # instance attrs default to the module constants when the spec carries
        # the default radius/border (DOHYO_RADIUS / BORDER_WIDTH), so the
        # default spec is byte-identical to today. A non-default radius scales
        # the platform, its border ring, the line-trigger annulus, AND the
        # spawn radius (so the bots stay proportionally placed inside the ring).
        self._dohyo_radius = float(self.hw_spec.dohyo.radius_m)
        self._border_width = float(self.hw_spec.dohyo.border_width_m)
        self._inner_radius = self._dohyo_radius - self._border_width
        # Spawn radius keeps its default 0.25 m for the default ring and scales
        # with the ring otherwise (SPAWN_RADIUS / DOHYO_RADIUS is the fraction).
        self._spawn_radius = SPAWN_RADIUS * (self._dohyo_radius / DOHYO_RADIUS)
        # E1c: the AGENT's wheel motor caps come from the spec's drivetrain.
        # Defaults equal the module constants AGENT_MAX_FORCE/AGENT_MAX_RAD
        # (== WHEEL_MAX_TORQUE/WHEEL_OMEGA_FWD), so behaviour is unchanged.
        # NOTE: opponent (NovaMax) caps are NOT driven by this — they stay on
        # the NOVAMAX_* constants (a later opponent-hardware feature).
        self._agent_max_torque = self.hw_spec.drivetrain.max_torque_nm
        self._agent_max_omega = self.hw_spec.drivetrain.max_omega_rad_s
        # E1d: terminal reward magnitudes come from the spec's reward table.
        # Defaults equal REWARD_WIN/LOSE_PUSH/LOSE_MUTUAL/LOSE_SELF/TIMEOUT.
        self._rw = dict(self.hw_spec.reward.terminal)
        # alg/improvment: dense edge-avoidance shaping to cut self-outs
        # (the dominant loss mode at high mult). Off by default.
        self.edge_avoid_reward = bool(edge_avoid_reward)
        # alg/improvment: hardcoded observable safety override (anti-blind-
        # charge + rear-edge reflex), applied to the action before motors.
        self.safety_override = bool(safety_override)
        # alg/improvment: hardcoded opening forward charge (first ~100 ms).
        self.opening_charge = bool(opening_charge)
        # alg/improvment: spawn guard — early net-backward -> forward.
        self.spawn_guard = bool(spawn_guard)
        # alg/improvment: anti-stall — too many idle commands -> forward.
        self.antistall = bool(antistall)
        # alg/improvment: spawn the opponent on the agent chassis (robot.urdf).
        self.enemy_as_agent = bool(enemy_as_agent)
        # alg/improvment: per-episode opponent hardware/power randomization.
        self.enemy_chassis_dr = bool(enemy_chassis_dr)
        self.mult_dr_range = mult_dr_range
        # alg/improvment: per-step penalty for standing still (see step()).
        self.still_penalty = bool(still_penalty)
        # alg/improvment: slight per-step penalty for going backward.
        self.backward_penalty = bool(backward_penalty)
        # alg/improvment: optional per-episode opponent sampling weights
        # (e.g. a novamax-heavy mix for a targeted finetune). None = the
        # zoo default. Ignored when force_opponent_id pins one opponent.
        self.opponent_weights = opponent_weights
        # alg/improvment: per-episode behavioral randomization of the
        # opponent (training only). Off for eval/audit so WR is reproducible.
        self.opponent_dr = bool(opponent_dr)
        # alg/improvment: harder sensor-noise profile (training only). Off
        # for eval/audit so the env matches the committed baseline exactly.
        self.sensor_hard_dr = bool(sensor_hard_dr)
        # Flank shaping rewards rear/side engagement; the bearing-based
        # tracking reward penalizes the enemy being off the agent's nose,
        # which directly conflicts. When flank is on, tracking is forced off.
        self.flank_reward = bool(flank_reward)
        self.tracking_reward = bool(tracking_reward) and not self.flank_reward
        self.action_consistency_reward = bool(action_consistency_reward)

        self.gui = gui
        # Legacy knob kept so old call-sites still work; novamax_level
        # supersedes it for the actual physics caps.
        self.enemy_torque_mult = float(enemy_torque_multiplier)
        # NovaMax curriculum tier (1 = nerfed, 3 = real spec). Set by the
        # trainer between phases via env.unwrapped.novamax_level = N.
        self.novamax_level = int(novamax_level)
        # Fine-grained override: when set, NovaMax velocity & torque caps
        # = AGENT_MAX_RAD/FORCE * mult (e.g. 1.5x for an intermediate phase).
        # When None, the level table above is used.
        self.novamax_torque_mult: Optional[float] = (
            float(novamax_torque_mult) if novamax_torque_mult is not None else None
        )
        # Eval / debugging override: pin the opponent to a specific zoo
        # entry. When None (default), reset() uniformly samples each
        # episode.  Set to e.g. "spinner" to lock in one controller.
        self.force_opponent_id: Optional[str] = force_opponent_id
        # add/ui-local: optional user-authored opponent factories merged into
        # the sampling pool. Maps a custom opponent id -> a zero-arg callable
        # returning a fresh OpponentController. None (default) => behaviour is
        # byte-identical to the built-in zoo (audit-green). A custom id can be
        # pinned via force_opponent_id or weighted via opponent_weights.
        self.extra_opponents: Optional[dict] = (
            dict(extra_opponents) if extra_opponents else None
        )
        # add/ui-local: when set, the ENEMY fights on ITS OWN hardware — the
        # chassis/wheels/wedge come from a URDF generated off this spec and the
        # enemy's motor caps (force/omega) come from spec.drivetrain instead of
        # the NOVAMAX_* constants. None (default) => the enemy spawns/drives
        # EXACTLY as today (novamax / agent chassis), so the env is byte-
        # identical and tests/audit_3d.py stays green. The opponent CONTROLLER
        # (DSL/zoo) and its sensor model (5-key ir suite, line sensors) are
        # unchanged; only the physical body + drive caps differ.
        self.enemy_hw_spec: Optional[HardwareSpec] = enemy_hardware_spec
        # Lazily-created temp dir holding the generated enemy URDF; cleaned up
        # in close(). Only allocated when enemy_hw_spec is provided.
        self._enemy_urdf_tmpdir = None
        self._enemy_urdf_path: Optional[str] = None
        # add/ui-local: PER-EPISODE custom enemy hardware for TRAINING. Maps a
        # custom opponent id -> the HardwareSpec the enemy spawns on when THAT
        # opponent is the episode's opponent. None (default) => behaviour is
        # byte-identical to today (the enemy spawns/drives exactly as the
        # single-spec / novamax path; audit stays green). Built-in zoo ids (no
        # entry here) spawn the enemy exactly as today. Each custom spec's URDF
        # is generated + CACHED once below (id -> temp path) so resets only
        # removeBody+loadURDF, never regenerate URDF text.
        self.extra_opponent_specs: Optional[dict] = (
            dict(extra_opponent_specs) if extra_opponent_specs else None
        )
        # Lazy cache: custom opponent id -> generated URDF path. The backing
        # TemporaryDirectory objects are held in _enemy_spec_tmpdirs and all
        # cleaned up in close().
        self._enemy_spec_urdf_cache: dict[str, str] = {}
        self._enemy_spec_tmpdirs: list = []
        # The HardwareSpec the enemy is fighting on for the CURRENT episode
        # (set in reset()): the per-episode custom spec when the sampled
        # opponent has one, else the single-battle enemy_hw_spec, else None
        # (novamax / agent chassis). Drives _novamax_caps() + base mass.
        self._active_enemy_spec: Optional[HardwareSpec] = enemy_hardware_spec
        # When True, the blue robot is driven by keyboard (WASD / arrows)
        # instead of the NovamaxController. Requires gui=True.
        self.human_enemy = bool(human_enemy)
        self._client_id = -1
        self._connect()

        # Run 11: action space is either continuous (default — used by
        # TD3 / SAC) or discrete 9-action grid (used by DQN). The
        # discrete grid maps idx → (left_motor, right_motor) via
        # DISCRETE_ACTION_MAP. step() handles the decode internally so
        # all downstream physics/reward code sees floats.
        if action_space_kind not in ("continuous", "discrete"):
            raise ValueError(
                f"action_space_kind must be 'continuous' or 'discrete', "
                f"got {action_space_kind!r}"
            )
        self.action_space_kind = action_space_kind
        self.narek_reward = bool(narek_reward)
        if action_space_kind == "discrete":
            self.action_space = spaces.Discrete(len(DISCRETE_ACTION_MAP))
        else:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
            )
        # 12D obs (Run 9): 3 forward laser distances, last_seen_dir, 2
        # line sensors, prev_left/prev_right, engagement_timer,
        # yaw_rate_proxy, front_ir_delta, lateral_ir_delta. The two
        # delta features are temporal derivatives of the IR distances:
        # they tell the policy whether the opponent is closing in
        # (negative) or escaping (positive) and at what rate, which the
        # 10D-instantaneous obs cannot express. Critical for predicting
        # Dodger-style pivot evasion.
        # E1b: single-frame obs width = spec.base_obs_dim
        # (n_distance + len(engineered)). Default spec => 3 + 9 = 12.
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.hw_spec.base_obs_dim,), dtype=np.float32,
        )

        self.robot_id: Optional[int] = None
        self.enemy_id: Optional[int] = None
        self._left_wheel_idx: Optional[int] = None
        self._right_wheel_idx: Optional[int] = None
        self._caster_idx: Optional[int] = None
        self._enemy_left_idx: Optional[int] = None
        self._enemy_right_idx: Optional[int] = None

        # Default opponent; reset() draws a fresh one from the zoo.
        self._enemy_ctrl = NovamaxController()
        self._opponent_id: str = "novamax"
        self._disabled_sensor: Optional[int] = None

        self.last_seen_dir: float = 0.0
        self._steps_since_last_hit: int = 0
        # Run 2: prev_action latch (raw policy output, before deadzone /
        # latency) and engagement timer (close-front consecutive ticks).
        self._prev_action: tuple[float, float] = (0.0, 0.0)
        self._engagement_timer: int = 0
        # Run 6: yaw-rate proxy (decayed accumulator of left-right diff).
        self._yaw_rate_proxy: float = 0.0
        # Run 9: previous-frame normalised IR readings, used to compute
        # the two temporal-derivative obs features (front_ir_delta and
        # lateral_ir_delta). Seeded to 1.0 = "no hit", so the first delta
        # is measured against max-range; it is only ~zero if the enemy
        # also starts out of IR range (at spawn it usually isn't, so the
        # first delta is typically non-zero). This matches the firmware
        # IrDeltaState reset, so there's no sim-to-real drift.
        self._prev_front_norm: float = 1.0
        self._prev_min_lateral: float = 1.0
        # alg/improvment: cumulative flank reward paid this episode (cap).
        self._flank_paid: float = 0.0
        # alg/improvment: per-episode opponent DR state (defaults = no DR).
        self._enemy_speed_mult: float = 1.0
        self._enemy_tracking_gain: float = 1.0
        self._enemy_action_latency: int = 1
        # alg/improvment: per-episode hard-sensor-DR state (defaults = none).
        self._tof_calib_bias: np.ndarray = np.zeros(
            self.hw_spec.n_distance, dtype=np.float64,
        )
        self._tof_stuck_channel: Optional[int] = None
        self._tof_stuck_last: Optional[float] = None
        self._line_flip_prob: float = 0.0
        self._last_contact_step: int = -10_000
        self._steps = 0
        self._np_random: np.random.Generator = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _connect(self) -> None:
        mode = p.GUI if self.gui else p.DIRECT
        self._client_id = p.connect(mode)
        if self._client_id < 0:
            raise RuntimeError("Failed to connect to PyBullet server.")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.2,
                cameraYaw=45.0,
                cameraPitch=-50.0,
                cameraTargetPosition=[0.0, 0.0, 0.0],
            )

    def _build_dohyo(self, surface_friction: float) -> None:
        white_visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER, radius=self._dohyo_radius,
            length=DOHYO_THICKNESS, rgbaColor=(1.0, 1.0, 1.0, 1.0),
        )
        white_collision = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER, radius=self._dohyo_radius,
            height=DOHYO_THICKNESS,
        )
        white_id = p.createMultiBody(
            baseMass=0.0, baseCollisionShapeIndex=white_collision,
            baseVisualShapeIndex=white_visual,
            basePosition=[0.0, 0.0, DOHYO_THICKNESS / 2.0],
        )
        p.changeDynamics(white_id, -1, lateralFriction=surface_friction)

        black_visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER, radius=self._inner_radius,
            length=BLACK_TOP_THICKNESS, rgbaColor=(0.05, 0.05, 0.05, 1.0),
        )
        black_collision = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER, radius=self._inner_radius,
            height=BLACK_TOP_THICKNESS,
        )
        black_id = p.createMultiBody(
            baseMass=0.0, baseCollisionShapeIndex=black_collision,
            baseVisualShapeIndex=black_visual,
            basePosition=[0.0, 0.0, DOHYO_THICKNESS + BLACK_TOP_THICKNESS / 2.0],
        )
        p.changeDynamics(black_id, -1, lateralFriction=surface_friction)

    def _spawn_robot(
        self, position, yaw: float, chassis_rgba,
        urdf_path: str = ROBOT_URDF,
    ) -> tuple[int, dict]:
        orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        body_id = p.loadURDF(
            urdf_path, basePosition=list(position), baseOrientation=orn,
            useFixedBase=False, flags=p.URDF_USE_INERTIA_FROM_FILE,
        )
        p.changeVisualShape(body_id, -1, rgbaColor=chassis_rgba)

        joint_lookup = {
            p.getJointInfo(body_id, j)[1].decode(): j
            for j in range(p.getNumJoints(body_id))
        }
        for jname in ("left_wheel_joint", "right_wheel_joint"):
            if jname in joint_lookup:
                # Anisotropic friction: high in the rolling direction
                # (so the wheel grips for propulsion + braking) and low
                # in the rotation-axis direction (so the bot CAN be
                # pushed sideways). Without this the wheel grips equally
                # in all tangential directions — opponent's lateral
                # grip exactly cancels agent's forward thrust → side
                # pushes have zero net effect. The 0.3 factor on the
                # axle axis lets a sustained ~2 N lateral force slip
                # the wheel; well above incidental contact noise but
                # below committed pushing.
                p.changeDynamics(body_id, joint_lookup[jname],
                                 lateralFriction=WHEEL_FRICTION,
                                 anisotropicFriction=[1.0, 0.3, 1.0])
        p.changeDynamics(body_id, -1, lateralFriction=CHASSIS_FRICTION)
        if "front_caster_joint" in joint_lookup:
            p.changeDynamics(
                body_id, joint_lookup["front_caster_joint"],
                lateralFriction=0.0, rollingFriction=0.0, spinningFriction=0.0,
            )
        return body_id, joint_lookup

    # ------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------
    def _ray_distance(
        self, from_body: int, body_dx: float, body_dy: float,
        start_offset: float, max_dist: float, target_id: Optional[int] = None,
    ) -> Optional[float]:
        pos, orn = p.getBasePositionAndOrientation(from_body)
        yaw = p.getEulerFromQuaternion(orn)[2]
        cy, sy = math.cos(yaw), math.sin(yaw)
        wx = cy * body_dx - sy * body_dy
        wy = sy * body_dx + cy * body_dy
        sensor_z = pos[2] + 0.022
        start = [pos[0] + wx * start_offset, pos[1] + wy * start_offset, sensor_z]
        end = [start[0] + wx * max_dist, start[1] + wy * max_dist, sensor_z]
        hit_uid, _, hit_fraction, _, _ = p.rayTest(start, end)[0]
        if hit_uid < 0 or hit_uid == from_body:
            return None
        if target_id is not None and hit_uid != target_id:
            return None
        return hit_fraction * max_dist

    def _ray_from_body_point(
        self, from_body: int,
        start_x: float, start_y: float,
        dir_x: float, dir_y: float,
        max_dist: float, target_id: Optional[int] = None,
    ) -> Optional[float]:
        """Like _ray_distance, but the start point is an arbitrary body-frame
        offset instead of `start_offset` along the ray direction.

        Useful for sensor mounts that are NOT on the body's centre line
        (e.g. side IRs at the left/right edges of the chassis).
        """
        pos, orn = p.getBasePositionAndOrientation(from_body)
        yaw = p.getEulerFromQuaternion(orn)[2]
        cy, sy = math.cos(yaw), math.sin(yaw)
        sx_w = cy * start_x - sy * start_y
        sy_w = sy * start_x + cy * start_y
        dx_w = cy * dir_x - sy * dir_y
        dy_w = sy * dir_x + cy * dir_y
        sensor_z = pos[2] + 0.022
        start = [pos[0] + sx_w, pos[1] + sy_w, sensor_z]
        end = [start[0] + dx_w * max_dist, start[1] + dy_w * max_dist, sensor_z]
        hit_uid, _, hit_fraction, _, _ = p.rayTest(start, end)[0]
        if hit_uid < 0 or hit_uid == from_body:
            return None
        if target_id is not None and hit_uid != target_id:
            return None
        return hit_fraction * max_dist

    def _novamax_edge_watchdog(self) -> bool:
        """High-rate edge check, intended to run every physics substep.

        At 1600 RPM NovaMax covers ~12 cm per RL step, so checking line
        sensors only at the RL boundary lets it skate past the dohyo edge.
        This routine samples both line sensors and, if either fires AND
        the controller isn't already in its brake state, performs the
        instant-stop (zero wheels + zero chassis linear vel) and forces
        the controller into brake state. Returns True if it intervened.
        """
        if self._enemy_ctrl.is_edge_braking:
            return False
        edge_left = self._novamax_line_triggered(*NOVAMAX_LINE_SENSORS[0])
        edge_right = self._novamax_line_triggered(*NOVAMAX_LINE_SENSORS[1])
        if not (edge_left or edge_right):
            return False

        # Push the controller into the reverse phase. Each opponent
        # implements force_edge_brake(); the next decide() call continues
        # the reverse → spin sequence at safe speed so the chassis keeps
        # traction.
        self._enemy_ctrl.force_edge_brake(edge_left, edge_right)

        # Instant stop: kill linear momentum and wheel rotation. Keep
        # angular velocity so the body can pivot during the spin phase.
        _, ang_vel = p.getBaseVelocity(self.enemy_id)
        p.resetBaseVelocity(
            self.enemy_id,
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=list(ang_vel),
        )
        p.resetJointState(
            self.enemy_id, self._enemy_left_idx,
            targetValue=0.0, targetVelocity=0.0,
        )
        p.resetJointState(
            self.enemy_id, self._enemy_right_idx,
            targetValue=0.0, targetVelocity=0.0,
        )
        # Drive both wheels at -EDGE_SAFE_SPEED (controlled reverse) for
        # the rest of this RL step. Next RL tick re-enters the main
        # control branch which will continue the reverse → spin sequence
        # at EDGE_SAFE_SPEED so the chassis keeps traction.
        max_force_brake = 2.0
        v = NovamaxController.EDGE_SAFE_SPEED
        p.setJointMotorControl2(
            bodyUniqueId=self.enemy_id, jointIndex=self._enemy_left_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=-FORWARD_SIGN * v,
            force=max_force_brake,
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.enemy_id, jointIndex=self._enemy_right_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=-FORWARD_SIGN * v,
            force=max_force_brake,
        )
        return True

    def _line_triggered_at(
        self, body_id: int, body_x: float, body_y: float,
    ) -> bool:
        """Return True when the body-frame point (body_x, body_y) on the
        given body is over the dohyo's white border ring."""
        pos, orn = p.getBasePositionAndOrientation(body_id)
        yaw = p.getEulerFromQuaternion(orn)[2]
        cy, sy = math.cos(yaw), math.sin(yaw)
        wx = pos[0] + cy * body_x - sy * body_y
        wy = pos[1] + sy * body_x + cy * body_y
        return math.hypot(wx, wy) > self._inner_radius

    def _novamax_line_triggered(self, body_x: float, body_y: float) -> bool:
        """Backwards-compat wrapper for the enemy's line sensors."""
        return self._line_triggered_at(self.enemy_id, body_x, body_y)

    def _sample_with_extra(self, sample_opponent, extra: dict):
        """Weighted draw over the built-in zoo PLUS user-authored opponents.

        ``extra`` maps custom id -> zero-arg controller factory. Built-in ids
        keep their default/overridden weights; a custom id participates only
        when ``opponent_weights`` assigns it a positive weight (default 0), so
        merely registering a factory does not perturb the zoo distribution.
        Returns ``(id, controller)``.
        """
        from opponents import OPPONENT_WEIGHTS

        wmap = self.opponent_weights if self.opponent_weights is not None else OPPONENT_WEIGHTS
        # Total weight assigned to custom ids in this episode's weight map.
        custom_w = {cid: float(wmap.get(cid, 0.0)) for cid in extra}
        custom_total = sum(custom_w.values())
        if custom_total <= 0.0:
            # No custom weight => fall back to the pure built-in draw (default
            # distribution, byte-identical to no-extra behaviour).
            return sample_opponent(self._np_random, self.opponent_weights)

        def _draw_custom():
            ids = list(custom_w)
            probs = [custom_w[i] / custom_total for i in ids]
            idx = int(self._np_random.choice(len(ids), p=probs))
            cid = ids[idx]
            return cid, extra[cid]()

        # Custom-only mix: the built-in weight map is all-zero, which
        # sample_opponent rejects — draw straight from the custom ids (lets a
        # roster of only custom opponents, e.g. several Heavy Dodgers, train).
        builtin_total = sum(float(wmap.get(name, 0.0)) for name in OPPONENT_WEIGHTS)
        if builtin_total <= 0.0:
            return _draw_custom()

        # Mixed: built-in side keeps its own weights; pick custom-vs-built-in by
        # their relative total weights.
        builtin_id, builtin_ctrl = sample_opponent(self._np_random, self.opponent_weights)
        grand = builtin_total + custom_total
        if self._np_random.random() < (builtin_total / grand):
            return builtin_id, builtin_ctrl
        return _draw_custom()

    def _select_opponent(self) -> None:
        """Draw this episode's opponent id + controller into
        ``self._opponent_id`` / ``self._enemy_ctrl`` — exactly ONCE per reset.

        Factored out of reset() so the per-episode custom-hardware path can
        pick the opponent BEFORE spawning the enemy body (to choose the body)
        while the legacy call-site at the end of the spawn block reuses the
        result instead of re-drawing. The ``_opponent_selected`` guard makes
        the second call a no-op, so the RNG-draw sequence is unchanged versus
        the single-call baseline.
        """
        if getattr(self, "_opponent_selected", False):
            return
        from opponents import make_opponent, sample_opponent
        extra = self.extra_opponents
        if self.force_opponent_id is not None:
            self._opponent_id = self.force_opponent_id
            # Prefer a user-authored factory when the pinned id is custom;
            # fall back to the built-in zoo otherwise (default path unchanged).
            if extra is not None and self._opponent_id in extra:
                self._enemy_ctrl = extra[self._opponent_id]()
            else:
                self._enemy_ctrl = make_opponent(self._opponent_id)
        elif extra:
            # Merge custom factories into the weighted draw. Built-in ids keep
            # their OPPONENT_WEIGHTS; custom ids only participate when given a
            # positive weight in opponent_weights (default 0 => never sampled),
            # so adding factories alone does not change the zoo distribution.
            self._opponent_id, self._enemy_ctrl = self._sample_with_extra(
                sample_opponent, extra,
            )
        else:
            self._opponent_id, self._enemy_ctrl = sample_opponent(
                self._np_random, self.opponent_weights,
            )
        self._opponent_selected = True

    def _ensure_enemy_urdf(self, spec: HardwareSpec) -> str:
        """Generate (once) a URDF for the custom-hardware enemy and return its
        path. The file lives in a per-env temp dir cleaned up in close().

        Imported lazily so the core env module can load without the web stack
        on the import path; urdf_gen only depends on hardware_spec (no cycle).
        """
        if self._enemy_urdf_path is not None:
            return self._enemy_urdf_path
        import tempfile
        from webapp.shared.urdf_gen import write_urdf

        self._enemy_urdf_tmpdir = tempfile.TemporaryDirectory(
            prefix="sumo_enemy_urdf_"
        )
        path = os.path.join(self._enemy_urdf_tmpdir.name, "enemy.urdf")
        write_urdf(spec, path)
        self._enemy_urdf_path = path
        return path

    def _ensure_custom_opponent_urdf(self, opp_id: str, spec: HardwareSpec) -> str:
        """Generate (ONCE per custom opponent id) a URDF for that opponent's
        own hardware and return its cached path.

        add/ui-local: used by the per-episode custom-enemy path. The URDF text
        is written exactly once into a dedicated temp dir; subsequent episodes
        with the same opponent id reuse the cached path (only removeBody +
        loadURDF happen per reset, never URDF regeneration). All temp dirs are
        held in ``self._enemy_spec_tmpdirs`` and cleaned up in close().
        """
        cached = self._enemy_spec_urdf_cache.get(opp_id)
        if cached is not None:
            return cached
        import tempfile
        from webapp.shared.urdf_gen import write_urdf

        tmpdir = tempfile.TemporaryDirectory(prefix=f"sumo_opp_{opp_id}_urdf_")
        self._enemy_spec_tmpdirs.append(tmpdir)
        path = os.path.join(tmpdir.name, "enemy.urdf")
        write_urdf(spec, path)
        self._enemy_spec_urdf_cache[opp_id] = path
        return path

    def _novamax_caps(self) -> tuple[float, float]:
        """Resolve the enemy's current (max_rad, max_force).

        add/ui-local: when the enemy fights on its OWN hardware
        (``enemy_hw_spec`` set), the caps come straight from that spec's
        drivetrain (max_omega_rad_s -> velocity clip, max_torque_nm -> motor
        force), NOT the NOVAMAX_* curriculum. Battery sag still scales omega
        symmetrically. The NovaMax curriculum path below is unchanged when no
        enemy spec is supplied (default / audit path byte-identical).

        ``self.novamax_torque_mult`` is the canonical curriculum knob.
        The lerp anchors on mult=1.0 (matched agent strength) and
        mult=3.0 (full bullet); for mult < 1.0 we LINEARLY EXTRAPOLATE
        below the matched endpoint so phase 0 of the reverse curriculum
        can produce a sub-agent opponent. The input mult is clamped
        to [NOVAMAX_TORQUE_MULT_MIN, 3.5] for safety.

        Run-6 worked example: mult=0.7 → t=-0.15 →
            max_rad ≈ 35.6 rad/s (~340 RPM)
            max_force ≈ 0.063 N·m (~53% of agent's 0.12 N·m)
        """
        if self._active_enemy_spec is not None:
            dt = self._active_enemy_spec.drivetrain
            max_rad = max(1.0, float(dt.max_omega_rad_s))
            max_force = max(0.01, float(dt.max_torque_nm))
            return max_rad * self._velocity_factor, max_force
        if self.novamax_torque_mult is not None:
            mult = max(NOVAMAX_TORQUE_MULT_MIN,
                       min(3.5, float(self.novamax_torque_mult)))
            # Anchored at REF=1.0 (matched), span = MAX - REF = 2.0.
            # NO clamp on `t`: linear extrapolation for mult < 1.0
            # gives genuinely weaker-than-agent opponents.
            t = (mult - NOVAMAX_TORQUE_MULT_REF) / (
                NOVAMAX_TORQUE_MULT_MAX - NOVAMAX_TORQUE_MULT_REF
            )
            max_rad = AGENT_MAX_RAD + t * (NOVAMAX_MAX_RAD - AGENT_MAX_RAD)
            max_force = AGENT_MAX_FORCE + t * (NOVAMAX_MAX_FORCE - AGENT_MAX_FORCE)
            # Floor both at small positive values — at MIN=0.5 the
            # extrapolated force lands ~0.025 N·m which is fine, but
            # defend against future MIN tweaks producing negatives.
            max_rad = max(1.0, max_rad)
            max_force = max(0.01, max_force)
        else:
            max_rad, max_force = NOVAMAX_LEVELS.get(
                self.novamax_level, NOVAMAX_LEVELS[3],
            )
        # Battery sag affects both bots symmetrically.
        return max_rad * self._velocity_factor, max_force

    def _raw_distances(self) -> list[Optional[float]]:
        # E1b: cast one ray per spec.distance_sensors entry. Each sensor's
        # body-frame mount (mount_xyz x,y) and ray yaw (angle_rad, +CCW from
        # +X) drive a _ray_from_body_point call; the ray direction is the
        # unit (cos angle, sin angle). The default spec reproduces the legacy
        # 3-ray cone exactly: front (yaw+0), left (yaw+30°), right (yaw-30°),
        # all mounted at x≈0.045 just behind the wedge tip, range ENEMY_FAR_DIST.
        readings: list[Optional[float]] = []
        for sensor in self.hw_spec.distance_sensors:
            mx, my, _mz = sensor.mount_xyz
            dir_x = math.cos(sensor.angle_rad)
            dir_y = math.sin(sensor.angle_rad)
            readings.append(
                self._ray_from_body_point(
                    self.robot_id, mx, my, dir_x, dir_y, sensor.range_m,
                )
            )
        if self._disabled_sensor is not None:
            readings[self._disabled_sensor] = None
        # Per-step ToF noise + dropout (Step 4 domain randomization). The
        # per-episode calibration bias (zero unless hard sensor DR) is
        # folded in here so it rides through the same clip.
        if self._tof_noise_sigma > 0.0 or self._tof_dropout_prob > 0.0:
            for i, r in enumerate(readings):
                if r is None:
                    continue
                if self._np_random.random() < self._tof_dropout_prob:
                    readings[i] = None
                    continue
                noisy = (
                    r
                    + float(self._tof_calib_bias[i])
                    + float(self._np_random.normal(0.0, self._tof_noise_sigma))
                )
                # Clip to a physical range. Above ENEMY_FAR_DIST = no-hit.
                if noisy >= ENEMY_FAR_DIST:
                    readings[i] = None
                else:
                    readings[i] = max(0.0, noisy)
        # alg/improvment: stuck channel (hard sensor DR) — intermittently
        # freezes one channel at its last value, modelling an I2C hold.
        if self._tof_stuck_channel is not None:
            ch = self._tof_stuck_channel
            if (
                self._tof_stuck_last is not None
                and self._np_random.random() < DR_TOF_STUCK_HOLD_PROB
            ):
                readings[ch] = self._tof_stuck_last
            else:
                self._tof_stuck_last = readings[ch]
        return readings

    @staticmethod
    def _norm(distance: Optional[float]) -> float:
        if distance is None:
            return 1.0
        return min(1.0, max(0.0, distance / ENEMY_FAR_DIST))

    @staticmethod
    def _sensor_strength(d: Optional[float]) -> float:
        """Convert raw distance (m) to strength in [0, 1]. Closer = stronger.
        ``None`` or readings >= ENEMY_FAR_DIST count as zero strength."""
        if d is None or d >= ENEMY_FAR_DIST:
            return 0.0
        if d <= 0.0:
            return 1.0
        return 1.0 - (d / ENEMY_FAR_DIST)

    def _update_last_seen(self, distances: list[Optional[float]]) -> None:
        """Argmax over (front, left, right) strengths with hysteresis +
        decay. Keep this in lock-step with arduino_obs_logic.h."""
        front, left, right = distances
        s_front = self._sensor_strength(front)
        s_left = self._sensor_strength(left)
        s_right = self._sensor_strength(right)
        strengths = (s_front, s_left, s_right)
        max_s = max(strengths)

        # All sensors at max range / dropped out → start the decay timer.
        if max_s <= 0.0:
            self._steps_since_last_hit += 1
            if self._steps_since_last_hit >= LAST_SEEN_DECAY_STEPS:
                self.last_seen_dir = 0.0
            return
        self._steps_since_last_hit = 0

        # Map current latched direction to a sensor index.
        # 0 = front, 1 = left, 2 = right.
        if self.last_seen_dir < -0.5:
            prev_idx = 1
        elif self.last_seen_dir > 0.5:
            prev_idx = 2
        else:
            prev_idx = 0
        prev_s = strengths[prev_idx]

        # Tie-break: prefer the previous winner if it's tied with the max.
        # Otherwise pick the first tied index in declared order
        # (front, left, right) — matches the C++ short-circuit chain.
        tied = [i for i, s in enumerate(strengths) if s == max_s]
        winner = prev_idx if prev_idx in tied else tied[0]

        # Hysteresis: only switch if the new winner is meaningfully
        # stronger. prev_s == 0 ⇒ any non-zero strength clears the bar.
        if winner != prev_idx:
            if max_s <= LAST_SEEN_HYSTERESIS_RATIO * prev_s:
                return
        self.last_seen_dir = (0.0, -1.0, 1.0)[winner]

    def _agent_line_sensors(self) -> tuple[float, ...]:
        """Read every QTR line sensor as a float (1.0 over white, 0.0 over black).

        E1b: one reading per ``spec.line_sensors`` entry, in spec order. The
        default spec's two sensors (rear-left, rear-right) reproduce the
        legacy ``AGENT_LINE_SENSORS`` pair byte-identically — same mount xy,
        same per-step bit-flip DR draw order.
        """
        lines: list[float] = [
            1.0 if self._line_triggered_at(self.robot_id, *ls.mount_xy) else 0.0
            for ls in self.hw_spec.line_sensors
        ]
        # alg/improvment: per-step line-sensor bit flip (hard sensor DR),
        # modelling a dirty / edge-lit QTR. Zero-prob in eval. One RNG draw
        # per sensor, in order — matches the legacy two-draw sequence.
        if self._line_flip_prob > 0.0:
            for i in range(len(lines)):
                if self._np_random.random() < self._line_flip_prob:
                    lines[i] = 1.0 - lines[i]
        return tuple(lines)

    def _build_obs(self, distances: list[Optional[float]]) -> np.ndarray:
        """Assemble the single-frame observation from the HardwareSpec.

        E1b: the vector is ``[ d_0 ... d_{N-1} , <engineered in spec order> ]``
        where ``N = spec.n_distance``. Each distance is normalised to [0, 1]
        by ``ENEMY_FAR_DIST``; the engineered channels are looked up by name
        from ``spec.engineered``. For the default spec (3 distances, 9
        engineered) this is byte-identical to the legacy hand-written vector:
            [front, left, right, last_seen_dir, line_l, line_r,
             prev_left, prev_right, engagement, yaw_rate_proxy,
             front_ir_delta, lateral_ir_delta]
        """
        # --- raw distance channels (normalised), spec order ---------------
        norms = [self._norm(d) for d in distances]
        # front = first sensor; lateral = the closest of the remaining
        # (side) sensors. For the default these are the left/right pair, so
        # `min_lateral` matches the legacy left/right minimum exactly.
        front_norm = norms[0]
        if len(norms) > 1:
            min_lateral = min(norms[1:])
        else:
            min_lateral = front_norm

        # --- line sensors -------------------------------------------------
        lines = self._agent_line_sensors()

        # --- prev action latch + engagement timer -------------------------
        prev_l, prev_r = self._prev_action
        engagement = min(1.0, self._engagement_timer / ENGAGEMENT_MAX_STEPS)

        # --- Run 9 closing-rate (IR delta) features -----------------------
        # Negative front_ir_delta = enemy approaching head-on; positive =
        # escaping. lateral_ir_delta uses the closer side sensor so pivot
        # evasion is visible regardless of which side it spun away on.
        raw_front_delta = front_norm - self._prev_front_norm
        raw_lateral_delta = min_lateral - self._prev_min_lateral
        front_ir_delta = max(-1.0, min(1.0, raw_front_delta * 2.0))
        lateral_ir_delta = max(-1.0, min(1.0, raw_lateral_delta * 2.0))
        # Cache AFTER computing the delta (this frame vs the previous one).
        self._prev_front_norm = front_norm
        self._prev_min_lateral = min_lateral

        # --- name -> value table for the engineered channels --------------
        feature_values: dict[str, float] = {
            "last_seen_dir": float(self.last_seen_dir),
            "prev_left": float(prev_l),
            "prev_right": float(prev_r),
            "engagement": float(engagement),
            "yaw_rate_proxy": float(self._yaw_rate_proxy),
            "front_ir_delta": float(front_ir_delta),
            "lateral_ir_delta": float(lateral_ir_delta),
        }
        # Line-sensor channels: the conventional names (line_l, line_r) map
        # positionally to the spec's line sensors; each sensor id is also
        # registered so a non-default spec can name them by id.
        for i, ls in enumerate(self.hw_spec.line_sensors):
            feature_values[ls.id] = float(lines[i])
        if len(lines) >= 1:
            feature_values.setdefault("line_l", float(lines[0]))
        if len(lines) >= 2:
            feature_values.setdefault("line_r", float(lines[1]))

        try:
            engineered = [feature_values[name] for name in self.hw_spec.engineered]
        except KeyError as exc:  # pragma: no cover - config error path
            raise KeyError(
                f"HardwareSpec.engineered names an unknown feature {exc.args[0]!r}; "
                f"known: {sorted(feature_values)}"
            ) from exc

        return np.array(norms + engineered, dtype=np.float32)

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_deadzone(action: float, dz: float) -> float:
        """PWM dead-zone model: |a| < dz produces no torque, otherwise
        rescale linearly so a = ±1 still maps to ±1 effective output.
        """
        if dz <= 0.0:
            return action
        if dz >= 1.0:
            return 0.0
        mag = max(0.0, abs(action) - dz) / (1.0 - dz)
        return math.copysign(mag, action) if action != 0.0 else 0.0

    def _apply_safety_override(
        self, left: float, right: float, distances: list,
    ) -> tuple[float, float]:
        """Hardcoded, observable safety layer (deployable in firmware 1:1).
        Reads only the live distance + rear line sensors + last_seen latch.
        (1) Rear-edge reflex (priority): a rear line sensor over the border
            -> drive inward to recover. (2) Anti-blind-charge: net-forward
            command with nothing detected and last_seen lost -> in-place
            scan-spin instead of charging off the (unobservable) front rim.
        """
        rear_l = self._line_triggered_at(self.robot_id, *AGENT_LINE_SENSORS[0])
        rear_r = self._line_triggered_at(self.robot_id, *AGENT_LINE_SENSORS[1])
        if rear_l or rear_r:
            return 1.0, 1.0
        no_target = (
            self.last_seen_dir == 0.0
            and self._norm(distances[0]) > SAFETY_CLEAR_NORM
            and self._norm(distances[1]) > SAFETY_CLEAR_NORM
            and self._norm(distances[2]) > SAFETY_CLEAR_NORM
        )
        if no_target and left > 0.5 and right > 0.5:
            return 1.0, -1.0
        return left, right

    def _apply_motor_velocities(self, left_cmd: float, right_cmd: float) -> None:
        # Per-episode "battery sag" on the wheel-velocity cap.
        cap = self._agent_max_omega * self._velocity_factor
        left_omega = left_cmd * cap
        right_omega = right_cmd * cap

        # Idle = free-spin (force=0). Real DC motors at zero voltage have
        # only back-EMF resistance — they do NOT actively brake. Sending
        # force=WHEEL_MAX_TORQUE with targetVelocity=0 was modelling an
        # active-brake that exceeded the agent's push capacity, making it
        # impossible to shove an idle opponent off the dohyo. Threshold
        # tolerates float noise in continuous commands (NovaMax outputs).
        left_force  = self._agent_max_torque if abs(left_cmd)  > 0.05 else 0.0
        right_force = self._agent_max_torque if abs(right_cmd) > 0.05 else 0.0
        p.setJointMotorControl2(
            bodyUniqueId=self.robot_id, jointIndex=self._left_wheel_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=FORWARD_SIGN * left_omega, force=left_force,
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.robot_id, jointIndex=self._right_wheel_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=FORWARD_SIGN * right_omega, force=right_force,
        )

    def _apply_enemy_control(self) -> None:
        # 5 IR sensors -> bool dict (True if any hit on the agent within range).
        ir_hits: dict[str, bool] = {}
        for name, (sx, sy, dx, dy) in NOVAMAX_IR_SENSORS.items():
            d = self._ray_from_body_point(
                self.enemy_id, sx, sy, dx, dy,
                NOVAMAX_IR_RANGE, target_id=self.robot_id,
            )
            ir_hits[name] = d is not None

        # 2 downward line sensors at the front-left/right plow corners.
        edge_left = self._novamax_line_triggered(*NOVAMAX_LINE_SENSORS[0])
        edge_right = self._novamax_line_triggered(*NOVAMAX_LINE_SENSORS[1])

        max_rad, max_force = self._novamax_caps()

        # Did the edge state machine kick in this tick? (Track the
        # transition None -> active so we only instant-stop once per
        # detection, not every tick of the 2+4 brake sequence.)
        was_braking = self._enemy_ctrl.is_edge_braking
        left_omega, right_omega = self._enemy_ctrl.decide(
            ir_hits, edge_left, edge_right,
        )
        edge_just_triggered = (not was_braking) and self._enemy_ctrl.is_edge_braking

        if edge_just_triggered:
            # NovaMax's 0.05 N·m stall torque can't bleed off ~1.25 kg·m/s
            # of forward momentum in 2 ticks, so the robot would slide off
            # the dohyo before the reverse maneuver can take effect.
            # Instant-stop: zero the wheels and the chassis linear velocity
            # the moment the line sensor fires (preserve angular velocity
            # so the body can still pivot during the spin phase).
            _, ang_vel = p.getBaseVelocity(self.enemy_id)
            p.resetBaseVelocity(
                self.enemy_id,
                linearVelocity=[0.0, 0.0, 0.0],
                angularVelocity=list(ang_vel),
            )
            p.resetJointState(
                self.enemy_id, self._enemy_left_idx,
                targetValue=0.0, targetVelocity=0.0,
            )
            p.resetJointState(
                self.enemy_id, self._enemy_right_idx,
                targetValue=0.0, targetVelocity=0.0,
            )

        # Run 9: SYMMETRIC LATENCY+DEADZONE on the opponent's command.
        # Only applied to the regular decide() output — edge-braking is
        # a survival manoeuvre and must reach the motors at full
        # authority (it's what stops the bullet from skating off-ring).
        if not self._enemy_ctrl.is_edge_braking and max_rad > 1e-6:
            # alg/improvment: behavioral DR. Skew the wheel differential
            # (turn sharpness) then scale overall speed. Both are 1.0 when
            # opponent_dr is off, so eval behavior is unchanged. Edge-
            # braking already bypasses this block (survival authority).
            mean = (left_omega + right_omega) * 0.5
            diff = (left_omega - right_omega) * 0.5 * self._enemy_tracking_gain
            left_omega = (mean + diff) * self._enemy_speed_mult
            right_omega = (mean - diff) * self._enemy_speed_mult
            norm_l = float(left_omega) / max_rad
            norm_r = float(right_omega) / max_rad
            self._enemy_action_queue.append((norm_l, norm_r))
            if len(self._enemy_action_queue) > self._enemy_action_latency:
                delayed_l, delayed_r = self._enemy_action_queue.pop(0)
            else:
                delayed_l, delayed_r = 0.0, 0.0
            delayed_l = self._apply_deadzone(delayed_l, self._enemy_motor_deadzone)
            delayed_r = self._apply_deadzone(delayed_r, self._enemy_motor_deadzone)
            left_omega = delayed_l * max_rad
            right_omega = delayed_r * max_rad

        left_omega = max(-max_rad, min(max_rad, float(left_omega)))
        right_omega = max(-max_rad, min(max_rad, float(right_omega)))
        # While braking, give the motors a much higher effective torque
        # cap so the reverse/spin commands actually have authority over
        # the body's momentum.
        torque = max(max_force, 2.0) if self._enemy_ctrl.is_edge_braking else max_force
        # Idle = free-spin (see _apply_motor_velocities comment).
        l_torque = torque if abs(left_omega)  > 0.05 * max_rad else 0.0
        r_torque = torque if abs(right_omega) > 0.05 * max_rad else 0.0
        p.setJointMotorControl2(
            bodyUniqueId=self.enemy_id, jointIndex=self._enemy_left_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=FORWARD_SIGN * left_omega, force=l_torque,
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.enemy_id, jointIndex=self._enemy_right_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=FORWARD_SIGN * right_omega, force=r_torque,
        )

    def _apply_human_enemy_control(self) -> None:
        """Drive the blue robot from the keyboard (WASD or arrow keys)."""
        keys = p.getKeyboardEvents()
        DOWN = p.KEY_IS_DOWN

        def held(*codes) -> bool:
            return any(keys.get(c, 0) & DOWN for c in codes)

        forward = held(ord("w"), ord("W"), p.B3G_UP_ARROW)
        backward = held(ord("s"), ord("S"), p.B3G_DOWN_ARROW)
        left = held(ord("a"), ord("A"), p.B3G_LEFT_ARROW)
        right = held(ord("d"), ord("D"), p.B3G_RIGHT_ARROW)

        # Translate the held keys into per-wheel commands in [-1, +1].
        l_cmd = r_cmd = 0.0
        if forward:
            l_cmd += 1.0
            r_cmd += 1.0
        if backward:
            l_cmd -= 1.0
            r_cmd -= 1.0
        if left:
            l_cmd -= 1.0
            r_cmd += 1.0
        if right:
            l_cmd += 1.0
            r_cmd -= 1.0
        l_cmd = max(-1.0, min(1.0, l_cmd))
        r_cmd = max(-1.0, min(1.0, r_cmd))

        max_rad, max_force = self._novamax_caps()
        l_omega = l_cmd * max_rad
        r_omega = r_cmd * max_rad
        # Idle = free-spin (see comment in _apply_motor_velocities).
        l_torque = max_force if abs(l_cmd) > 0.05 else 0.0
        r_torque = max_force if abs(r_cmd) > 0.05 else 0.0
        p.setJointMotorControl2(
            bodyUniqueId=self.enemy_id, jointIndex=self._enemy_left_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=FORWARD_SIGN * l_omega, force=l_torque,
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.enemy_id, jointIndex=self._enemy_right_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=FORWARD_SIGN * r_omega, force=r_torque,
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        # add/ui-local: per-reset guard so _select_opponent() draws exactly
        # ONCE (the per-episode custom-hardware path calls it early to pick the
        # body; the legacy call-site then reuses the result).
        self._opponent_selected = False

        p.resetSimulation()
        p.setGravity(0.0, 0.0, -9.81)
        p.setTimeStep(SIM_TIMESTEP)

        # ---- Per-episode domain randomization (Step 4) ----
        surface_friction = float(self._np_random.uniform(*DR_FRICTION_RANGE))
        self._build_dohyo(surface_friction)

        # Battery sag: U(0.85, 1.0) on the wheel-velocity cap, applied
        # symmetrically to agent and NovaMax via _apply_motor_velocities
        # / _novamax_caps.
        self._velocity_factor = float(self._np_random.uniform(*DR_VELOCITY_RANGE))

        # PWM dead zone: U(0.20, 0.35), applied at the top of step().
        self._motor_deadzone = float(self._np_random.uniform(*DR_DEADZONE_RANGE))

        # Action latency: discrete uniform over the active choice set
        # (widened under hard sensor DR).
        latency_choices = (
            DR_ACTION_LATENCY_CHOICES_HARD
            if self.sensor_hard_dr
            else DR_ACTION_LATENCY_CHOICES
        )
        idx = int(self._np_random.integers(0, len(latency_choices)))
        self._action_latency = int(latency_choices[idx])
        self._action_queue: list[tuple[float, float]] = []
        # Run 9: enemy gets its own action FIFO and deadzone, same DR
        # distributions as the agent. Symmetric handicap kills the
        # fairness gap that let opponents (especially Dodger) react in
        # 0 ms while the agent paid ~40 ms latency. Edge-braking
        # bypasses these layers — see _apply_enemy_control.
        self._enemy_action_queue: list[tuple[float, float]] = []
        self._enemy_motor_deadzone = float(
            self._np_random.uniform(*DR_DEADZONE_RANGE)
        )

        # alg/improvment: per-episode behavioral opponent DR (training
        # only). Decouples speed from torque, varies turn sharpness, and
        # gives the enemy its own reaction latency. Disabled for eval so
        # the held-out controllers present a fixed, reproducible behavior.
        if self.opponent_dr:
            self._enemy_speed_mult = float(
                self._np_random.uniform(*OPP_DR_SPEED_RANGE)
            )
            self._enemy_tracking_gain = float(
                self._np_random.uniform(*OPP_DR_TRACKING_RANGE)
            )
            idx = int(self._np_random.integers(0, len(OPP_DR_REACTION_CHOICES)))
            self._enemy_action_latency = int(OPP_DR_REACTION_CHOICES[idx])
        else:
            self._enemy_speed_mult = 1.0
            self._enemy_tracking_gain = 1.0
            self._enemy_action_latency = self._action_latency

        # ToF noise + dropout (per-step) parameters. Sigma widens under
        # hard sensor DR; the standard profile keeps eval comparable.
        sigma_pct = (
            DR_TOF_NOISE_SIGMA_PCT_HARD
            if self.sensor_hard_dr
            else DR_TOF_NOISE_SIGMA_PCT
        )
        self._tof_noise_sigma = sigma_pct * ENEMY_FAR_DIST
        self._tof_dropout_prob = DR_TOF_DROPOUT_PROB

        # alg/improvment: hard-sensor-DR per-episode draws (train only).
        n_dist = self.hw_spec.n_distance
        if self.sensor_hard_dr:
            self._tof_calib_bias = self._np_random.normal(
                0.0, DR_TOF_CALIB_BIAS_SIGMA, size=n_dist,
            )
            if self._np_random.random() < DR_TOF_STUCK_PROB:
                self._tof_stuck_channel = int(self._np_random.integers(0, n_dist))
            else:
                self._tof_stuck_channel = None
            self._line_flip_prob = DR_LINE_FLIP_PROB
        else:
            self._tof_calib_bias = np.zeros(n_dist, dtype=np.float64)
            self._tof_stuck_channel = None
            self._line_flip_prob = 0.0
        self._tof_stuck_last = None

        # Legacy episode-long dead sensor (still respected; default 0%).
        if self._np_random.random() < SENSOR_DEAD_PROB:
            self._disabled_sensor = int(self._np_random.integers(0, 3))
        else:
            self._disabled_sensor = None

        angle = float(self._np_random.uniform(0.0, 2.0 * math.pi))
        spawn_z = DOHYO_TOP_Z + 0.001

        # Per-robot yaw and spawn-radius jitter so the agent can't memorise
        # a deterministic head-on charge. Yaw ±10° (Run-7 fix: was ±30°,
        # which caused ~40% of episodes to spawn the bots so far off-axis
        # they passed each other entirely without colliding).
        yaw_jitter_a = float(self._np_random.uniform(-math.radians(10),
                                                      math.radians(10)))
        yaw_jitter_e = float(self._np_random.uniform(-math.radians(10),
                                                      math.radians(10)))
        radius_a = self._spawn_radius + float(self._np_random.uniform(-0.05, 0.05))
        radius_e = self._spawn_radius + float(self._np_random.uniform(-0.05, 0.05))

        agent_pos = (
            radius_a * math.cos(angle),
            radius_a * math.sin(angle),
            spawn_z,
        )
        agent_yaw = angle + math.pi + yaw_jitter_a

        # Symmetric spawn: enemy across from agent, both at SPAWN_RADIUS
        # with independent yaw / radius jitter. The 25% edge-spawn cheese
        # was removed in Step 3 in favour of a smoother torque curriculum.
        enemy_pos = (
            -radius_e * math.cos(angle),
            -radius_e * math.sin(angle),
            spawn_z,
        )
        enemy_yaw = angle + yaw_jitter_e

        self.robot_id, agent_joints = self._spawn_robot(agent_pos, agent_yaw, AGENT_RGBA)
        self._left_wheel_idx = agent_joints["left_wheel_joint"]
        self._right_wheel_idx = agent_joints["right_wheel_joint"]
        self._caster_idx = agent_joints["front_caster_joint"]
        # Independent mass perturbations for each robot (Step 4).
        agent_mass_mult = float(self._np_random.uniform(*DR_MASS_RANGE))
        p.changeDynamics(
            self.robot_id, -1, mass=ROBOT_MASS_NOMINAL * agent_mass_mult,
        )

        # alg/improvment: per-episode power randomization (different torque
        # mults) so the agent faces a range of opponent strengths.
        if self.mult_dr_range is not None:
            self.novamax_torque_mult = float(
                self._np_random.uniform(*self.mult_dr_range))

        # alg/improvment: enemy chassis. enemy_as_agent forces the AGENT
        # chassis (robot.urdf); enemy_chassis_dr randomizes it 50/50 per
        # episode, so the agent meets opponents on different hardware (and
        # learns the equal-power dynamics, not just the novamax chassis).
        # add/ui-local: a custom-hardware opponent fights on ITS OWN body — a
        # URDF generated off enemy_hw_spec (same named wheel joints as the
        # agent, so _spawn_robot's lookup is unchanged). This takes precedence
        # over the chassis-DR knobs. The mass comes from the spec (the heavy
        # chassis stays heavy); the NOVAMAX-style enemy mass-DR is skipped so
        # the custom body's physical identity is preserved.
        #
        # PER-EPISODE custom hardware (TRAINING): when extra_opponent_specs is
        # set, the opponent is selected FIRST (here, gated so the default
        # RNG-draw order is untouched), so the enemy body for this episode can
        # be THAT opponent's own chassis/motors. _select_opponent() performs
        # the same draw the legacy block below would, and caches its result so
        # that block does not re-draw. self._active_enemy_spec is then the
        # per-episode custom spec (or enemy_hw_spec, or None).
        self._active_enemy_spec = self.enemy_hw_spec
        if self.extra_opponent_specs is not None:
            self._select_opponent()
            ep_spec = self.extra_opponent_specs.get(self._opponent_id)
            if ep_spec is not None:
                self._active_enemy_spec = ep_spec

        if self._active_enemy_spec is not None:
            # Spawn on the active spec's body. For the per-episode path this
            # is the sampled opponent's cached URDF; for the single-battle
            # path it stays the enemy_hw_spec URDF (unchanged).
            if self._active_enemy_spec is self.enemy_hw_spec:
                enemy_urdf = self._ensure_enemy_urdf(self.enemy_hw_spec)
            else:
                enemy_urdf = self._ensure_custom_opponent_urdf(
                    self._opponent_id, self._active_enemy_spec,
                )
        else:
            use_agent_chassis = self.enemy_as_agent or (
                self.enemy_chassis_dr and self._np_random.random() < 0.5)
            enemy_urdf = ROBOT_URDF if use_agent_chassis else NOVAMAX_URDF
        self.enemy_id, enemy_joints = self._spawn_robot(
            enemy_pos, enemy_yaw, ENEMY_RGBA, urdf_path=enemy_urdf,
        )
        self._enemy_left_idx = enemy_joints["left_wheel_joint"]
        self._enemy_right_idx = enemy_joints["right_wheel_joint"]
        if self._active_enemy_spec is not None:
            # Keep the spec's true base mass (no DR jitter) so the custom
            # chassis's push/inertia matches what the user designed.
            p.changeDynamics(
                self.enemy_id, -1, mass=self._active_enemy_spec.chassis.mass_kg,
            )
        else:
            enemy_mass_mult = float(self._np_random.uniform(*DR_MASS_RANGE))
            p.changeDynamics(
                self.enemy_id, -1, mass=ROBOT_MASS_NOMINAL * enemy_mass_mult,
            )

        # Step 8: draw a fresh opponent from the zoo each episode (or
        # honour `force_opponent_id` when the eval / debugging caller
        # has pinned a specific controller). When the per-episode custom
        # path above already selected the opponent, this is a no-op reuse.
        self._select_opponent()
        self._enemy_ctrl.reset()

        self.last_seen_dir = 0.0
        self._steps_since_last_hit = 0
        self._prev_action = (0.0, 0.0)
        self._engagement_timer = 0
        self._yaw_rate_proxy = 0.0
        # Reset previous-frame IR caches so the first delta is 0.
        self._prev_front_norm = 1.0
        self._prev_min_lateral = 1.0
        # Run 6: per-episode wedge-engagement diagnostics.
        self._wedge_total_score = 0.0
        self._wedge_engaged_ticks = 0
        self._wedge_being_wedged_ticks = 0
        self._flank_paid = 0.0
        self._opening_charge_left = OPENING_CHARGE_STEPS if self.opening_charge else 0
        self._idle_streak = 0
        self._steps = 0
        # Step 6: tracks the env-step at which the last agent↔enemy
        # contact was observed. Used to distinguish push-loss from
        # self-out at termination.
        self._last_contact_step = -10_000
        # Stuck-detector counter (see STUCK_DETECTION_TICKS).
        self._stuck_ticks = 0

        # Initial agent-only charge: DISABLED (INITIAL_CHARGE_MS = 0 ->
        # INITIAL_CHARGE_TICKS == 0, so the loop runs 0 iterations). It
        # only runs if INITIAL_CHARGE_MS is set > 0, in which case red
        # drives both wheels full forward while blue holds still (zero
        # target velocity). The opponent controller's hysteresis only ticks
        # while we're calling _apply_enemy_control, which we skip here, so it
        # just observes the runup.
        for _ in range(INITIAL_CHARGE_TICKS):
            if not p.isConnected(self._client_id):
                break
            self._apply_motor_velocities(1.0, 1.0)
            initial_force = self._novamax_caps()[1]
            p.setJointMotorControl2(
                bodyUniqueId=self.enemy_id, jointIndex=self._enemy_left_idx,
                controlMode=p.VELOCITY_CONTROL, targetVelocity=0.0,
                force=initial_force,
            )
            p.setJointMotorControl2(
                bodyUniqueId=self.enemy_id, jointIndex=self._enemy_right_idx,
                controlMode=p.VELOCITY_CONTROL, targetVelocity=0.0,
                force=initial_force,
            )
            p.stepSimulation()
            if self.gui:
                time.sleep(SIM_TIMESTEP)

        # Distance tracker initialised AFTER the initial charge so the
        # approach reward isn't paid for motion that happened before
        # step() ever runs.
        agent_pos_now, _ = p.getBasePositionAndOrientation(self.robot_id)
        enemy_pos_now, _ = p.getBasePositionAndOrientation(self.enemy_id)
        self.prev_agent_to_enemy = math.hypot(
            enemy_pos_now[0] - agent_pos_now[0],
            enemy_pos_now[1] - agent_pos_now[1],
        )
        # Rest-z reference, kept for diagnostic scripts (trace_lift)
        # though no longer used by the wedge reward in v3.
        self._rest_agent_z = float(agent_pos_now[2])
        self._rest_enemy_z = float(enemy_pos_now[2])
        # Wedge v3: previous enemy radial distance during the current
        # contact bout. Reset to None when contact lapses so we don't
        # get a discontinuous Δr when contact resumes far away.
        self._prev_enemy_r_for_wedge: float | None = None
        # Per-component cumulative reward — emitted in info on terminal
        # so RewardLoggerCallback can write per-component TB scalars.
        # In Run 3 there are only two keys: "approach" and "terminal".
        self._reward_components: dict[str, float] = {}

        distances = self._raw_distances()
        self._update_last_seen(distances)
        return self._build_obs(distances), {}

    def step(self, action):
        if not p.isConnected(self._client_id):
            zero_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return zero_obs, 0.0, False, True, {"disconnected": True}

        # Run 11: decode discrete action int to (left, right) pair via
        # DISCRETE_ACTION_MAP. Continuous action passes through unchanged.
        if self.action_space_kind == "discrete":
            try:
                action_idx = int(np.asarray(action).item())
            except (ValueError, TypeError):
                raise ValueError(
                    f"discrete action must be a scalar int, got {action!r}"
                )
            if not (0 <= action_idx < len(DISCRETE_ACTION_MAP)):
                raise ValueError(
                    f"discrete action {action_idx} out of range "
                    f"[0, {len(DISCRETE_ACTION_MAP)})"
                )
            raw_left, raw_right = DISCRETE_ACTION_MAP[action_idx]
        else:
            action = np.asarray(action, dtype=np.float32).flatten()
            if action.shape != (2,):
                raise ValueError(f"action must be (2,), got {action.shape}")
            raw_left = float(np.clip(action[0], -1.0, 1.0))
            raw_right = float(np.clip(action[1], -1.0, 1.0))

        # alg/improvment: hardcoded safety override. Reads the live sensors
        # (same as the firmware would, just before driving) and replaces the
        # commanded action when it would blind-charge or sit on the rear edge.
        # Applied here so raw_left/raw_right (motors, reward gates, the
        # prev_action obs latch) all reflect the action actually executed.
        if self.safety_override:
            raw_left, raw_right = self._apply_safety_override(
                raw_left, raw_right, self._raw_distances(),
            )

        # alg/improvment: hardcoded opening charge — force straight forward
        # for the first ~100 ms of the match (deterministic, mirrored in
        # firmware). Applied last so it overrides policy + safety at the start.
        if self._opening_charge_left > 0:
            raw_left, raw_right = 1.0, 1.0
            self._opening_charge_left -= 1

        # alg/improvment: spawn guard — early in the match, a net-backward
        # command drives the robot into the rear rim (it spawns near it), so
        # go forward instead; later, reversing is left alone. Mirrored in
        # firmware. self._steps is 0-indexed at the top of step().
        if (self.spawn_guard and self._steps < SPAWN_GUARD_STEPS
                and (raw_left + raw_right) < 0.0):
            raw_left, raw_right = 1.0, 1.0

        # alg/improvment: anti-stall — if the policy keeps commanding (near-)
        # idle, force a forward charge so it never freezes mid-match. Mirrored
        # in firmware.
        if self.antistall:
            if abs(raw_left) < 0.5 and abs(raw_right) < 0.5:
                self._idle_streak += 1
            else:
                self._idle_streak = 0
            if self._idle_streak >= ANTISTALL_STEPS:
                raw_left, raw_right = 1.0, 1.0
                self._idle_streak = 0

        # Domain randomization (Step 4): action latency → dead zone.
        # The new action enters the FIFO; the action that left the FIFO
        # (or zeros, while the queue is warming up) is what physically
        # reaches the wheels this tick. Reward gates and motor commands
        # both see the *delayed* value so attribution stays consistent.
        self._action_queue.append((raw_left, raw_right))
        if len(self._action_queue) > self._action_latency:
            delayed_left, delayed_right = self._action_queue.pop(0)
        else:
            delayed_left, delayed_right = 0.0, 0.0
        left_cmd = self._apply_deadzone(delayed_left, self._motor_deadzone)
        right_cmd = self._apply_deadzone(delayed_right, self._motor_deadzone)

        self._apply_motor_velocities(left_cmd, right_cmd)
        if self.human_enemy:
            self._apply_human_enemy_control()
        else:
            self._apply_enemy_control()

        for _ in range(SUBSTEPS_PER_STEP):
            if not p.isConnected(self._client_id):
                zero_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
                return zero_obs, 0.0, False, True, {"disconnected": True}
            # Fast-path edge check (≈4 ms granularity vs 46 ms at the RL
            # boundary): catches the line sensor before NovaMax flies off
            # the dohyo at 2.77 m/s.
            if not self.human_enemy:
                self._novamax_edge_watchdog()
            p.stepSimulation()
            if self.gui:
                time.sleep(SIM_TIMESTEP)

        agent_pos, agent_orn = p.getBasePositionAndOrientation(self.robot_id)
        enemy_pos, enemy_orn = p.getBasePositionAndOrientation(self.enemy_id)
        agent_out = agent_pos[2] < FALL_Z
        enemy_out = enemy_pos[2] < FALL_Z

        # Single contact query reused for recency tracking AND the
        # Run-6 wedge-engagement reward. cheaper than two calls and
        # keeps the two consumers from drifting on what counts as
        # "contact this tick".
        contacts = p.getContactPoints(
            bodyA=self.robot_id, bodyB=self.enemy_id,
        )
        if contacts:
            self._last_contact_step = self._steps

        # Stuck-detector: when both bots are in continuous contact with
        # near-zero linear motion, the iterative solver has reached a
        # symmetric wedge stalemate that wheels alone won't break. After
        # STUCK_DETECTION_TICKS of this state, inject equal-and-opposite
        # lateral Δv perpendicular to the contact normal — breaks the
        # symmetry without favoring either bot, so trained policies can't
        # game it. Skip when bots are at rest pre-engagement (no contact).
        if contacts and not (agent_out or enemy_out):
            a_lv, a_ang = p.getBaseVelocity(self.robot_id)
            e_lv, e_ang = p.getBaseVelocity(self.enemy_id)
            a_speed = math.hypot(a_lv[0], a_lv[1])
            e_speed = math.hypot(e_lv[0], e_lv[1])
            if a_speed < STUCK_VELOCITY_THRESHOLD and e_speed < STUCK_VELOCITY_THRESHOLD:
                self._stuck_ticks += 1
            else:
                self._stuck_ticks = 0
            if self._stuck_ticks >= STUCK_DETECTION_TICKS:
                normal = contacts[0][7]  # contactNormalOnB
                perp_x, perp_y = -normal[1], normal[0]
                sign = 1.0 if self._np_random.random() < 0.5 else -1.0
                dvx = perp_x * STUCK_KICK_SPEED * sign
                dvy = perp_y * STUCK_KICK_SPEED * sign
                p.resetBaseVelocity(
                    self.robot_id,
                    linearVelocity=[a_lv[0] + dvx, a_lv[1] + dvy, a_lv[2]],
                    angularVelocity=a_ang,
                )
                p.resetBaseVelocity(
                    self.enemy_id,
                    linearVelocity=[e_lv[0] - dvx, e_lv[1] - dvy, e_lv[2]],
                    angularVelocity=e_ang,
                )
                self._stuck_ticks = 0
        else:
            self._stuck_ticks = 0

        # Run 3: collapsed reward — terminal + ONE shaping signal
        # (per-tick approach delta). All shaping branches from runs 1/2
        # (hunter, track, edge, push_delta, focus, push, flank, idle,
        # coward, monotonic-approach) are gone. Self-out / push-loss /
        # mutual-out distinction kept in info but rewards are unified.
        components: dict[str, float] = {}
        terminal = 0.0
        terminated = False
        termination_reason: Optional[str] = None

        if agent_out or enemy_out:
            terminated = True
            distances = [None, None, None]
            if enemy_out and not agent_out:
                terminal = self._rw["win"]
                termination_reason = "win"
            else:
                # Loss reward varies by reason: self-out is hardest punished
                # because the agent had full control over not driving off.
                if agent_out and enemy_out:
                    termination_reason = "mutual_out"
                    terminal = self._rw["lose_mutual"]
                elif self._steps - self._last_contact_step <= CONTACT_RECENT_STEPS:
                    termination_reason = "push_loss"
                    terminal = self._rw["lose_push"]
                else:
                    termination_reason = "self_out"
                    terminal = self._rw["lose_self"]
        else:
            distances = self._raw_distances()
            # front_norm drives the engagement timer (obs feature). The
            # other two normalised readings only matter for _build_obs,
            # which calls _norm() itself when it builds the obs vector.
            front_norm = self._norm(distances[0])
            if front_norm < ENGAGEMENT_FRONT_THRESHOLD:
                self._engagement_timer += 1
                # Run 10: pay a per-tick proximity bonus while the front
                # IR has the enemy in wedge-contact range. Dense gradient
                # for "get into engagement position" — fills the dead
                # signal between approach (rewarded only on Δdistance)
                # and wedge (rewarded only on physical contact + push).
                components["engage"] = REWARD_ENGAGE_PER_TICK
            else:
                self._engagement_timer = 0

            # alg/improvment: standing-still penalty. Punish barely moving
            # while NOT in contact (so genuine push stalemates are exempt —
            # the stuck-detector handles those). Discourages freezing,
            # timeout-stalling, and useless in-place spinning (low linear
            # speed). Opt-in so future finetunes can include it.
            if self.still_penalty and not contacts:
                a_lv, _ = p.getBaseVelocity(self.robot_id)
                if math.hypot(a_lv[0], a_lv[1]) < STILL_SPEED_THRESHOLD:
                    components["still"] = REWARD_STILL_PENALTY

            # alg/improvment: slight penalty for a net-backward (reverse)
            # command — discourages retreating. Uses the executed action so
            # the opening charge / safety override are correctly exempt.
            if self.backward_penalty and (raw_left + raw_right) < 0.0:
                components["backward"] = REWARD_BACKWARD_PENALTY

            # Run 11: Narek-style action-conditioned reward shaping.
            # Mirrors github.com/Narek2008654/Simulator/data_reader.py.
            # Computed on the COMMANDED action (raw_left/raw_right pre-
            # deadzone) so the policy sees a consistent signal about
            # its own choices, not the post-DR filtered output.
            if self.narek_reward:
                close = front_norm < NAREK_CLOSE_THRESHOLD
                is_idle = (raw_left == 0.0) and (raw_right == 0.0)
                has_neg = (raw_left == -1.0) or (raw_right == -1.0)
                has_pos = (raw_left == +1.0) or (raw_right == +1.0)
                narek_sum = 0.0
                if is_idle:
                    narek_sum += NAREK_IDLE_PENALTY
                if has_neg and close:
                    narek_sum += NAREK_RETREAT_PENALTY
                if has_pos and close:
                    narek_sum += NAREK_ATTACK_BONUS
                if narek_sum != 0.0:
                    components["narek"] = narek_sum

            # 2D-ported Run 16: action persistence (anti-flicker).
            # Fires before tracking so both signals can hit the same tick.
            if self.action_consistency_reward:
                prev_l, prev_r = self._prev_action
                if (raw_left != prev_l) or (raw_right != prev_r):
                    components["consistency"] = REWARD_ACTION_CHANGE_PENALTY

            # 2D-ported Run 14: bearing-based tracking reward. Uses the
            # agent's world-frame yaw (PyBullet quaternion → euler[2])
            # and the opponent's world-frame xy. Encourages the policy
            # to keep the enemy in its forward cone when within 50 cm.
            if self.tracking_reward:
                dx = enemy_pos[0] - agent_pos[0]
                dy = enemy_pos[1] - agent_pos[1]
                dist_xy = math.hypot(dx, dy)
                if dist_xy < REWARD_TRACK_RANGE:
                    _, agent_orn = p.getBasePositionAndOrientation(self.robot_id)
                    agent_yaw = p.getEulerFromQuaternion(agent_orn)[2]
                    bearing_world = math.atan2(dy, dx)
                    bl = (
                        bearing_world - agent_yaw + math.pi
                    ) % (2.0 * math.pi) - math.pi
                    abs_b = abs(bl)
                    if abs_b < TRACK_FRONT_HALF_RAD:
                        track_val = REWARD_TRACK_FRONT
                    elif abs_b < TRACK_SIDE_HALF_RAD:
                        track_val = REWARD_TRACK_SIDE
                    else:
                        track_val = REWARD_TRACK_BEHIND
                    components["track"] = track_val

            # Wedge engagement reward v3 — outward enemy displacement.
            # Per tick during contact: reward proportional to Δenemy_r.
            # Positive Δr (enemy pushed outward) ⇒ positive reward;
            # negative Δr ⇒ negative reward (agent being pushed back
            # while in contact, e.g. while being wedged with Rammer
            # driving the agent toward center). When contact lapses
            # the previous-r reference is cleared so we don't credit
            # discontinuous jumps when contact resumes elsewhere.
            wedge_delta_r = 0.0
            if contacts:
                enemy_r_now = math.hypot(enemy_pos[0], enemy_pos[1])
                if self._prev_enemy_r_for_wedge is not None:
                    wedge_delta_r = enemy_r_now - self._prev_enemy_r_for_wedge
                    if abs(wedge_delta_r) > APPROACH_MIN:
                        components["wedge"] = wedge_delta_r * REWARD_WEDGE_PER_M
                self._prev_enemy_r_for_wedge = enemy_r_now
            else:
                self._prev_enemy_r_for_wedge = None
            # Per-episode wedge diagnostics — accumulate signed Δr
            # totals and tick counts so RewardLoggerCallback can dump
            # to TB and we can tell "lots of small pushes" from
            # "few big flicks".
            self._wedge_total_score += wedge_delta_r * REWARD_WEDGE_PER_M
            if wedge_delta_r > APPROACH_MIN:
                self._wedge_engaged_ticks += 1
            elif wedge_delta_r < -APPROACH_MIN:
                self._wedge_being_wedged_ticks += 1

            # Approach reward: raw per-step delta, halved coefficient
            # for Run 6 (wedge is now primary shaping). Positive-only.
            current_agent_to_enemy = math.hypot(
                enemy_pos[0] - agent_pos[0], enemy_pos[1] - agent_pos[1],
            )
            approach_delta = self.prev_agent_to_enemy - current_agent_to_enemy
            if approach_delta > APPROACH_MIN:
                components["approach"] = approach_delta * REWARD_APPROACH
            self.prev_agent_to_enemy = current_agent_to_enemy

            # alg/improvment: flank shaping. Only when the wedge push is
            # already paying (enemy_r rising during contact) — that means
            # the agent is actively driving the enemy outward, not being
            # pushed and not merely orbiting. Reward scales with how far
            # behind/beside the enemy the agent is, and requires the agent
            # to be heading into the enemy. Capped per episode.
            if (
                self.flank_reward
                and wedge_delta_r > APPROACH_MIN
                and self._flank_paid < FLANK_EPISODE_CAP
            ):
                fdx = agent_pos[0] - enemy_pos[0]
                fdy = agent_pos[1] - enemy_pos[1]
                if math.hypot(fdx, fdy) < FLANK_RANGE:
                    enemy_yaw = p.getEulerFromQuaternion(enemy_orn)[2]
                    # Agent's bearing in the enemy's heading frame:
                    # 0 = in front of the enemy, ±π = directly behind it.
                    rel = (
                        math.atan2(fdy, fdx) - enemy_yaw + math.pi
                    ) % (2.0 * math.pi) - math.pi
                    pos_q = (abs(rel) - FLANK_MIN_RAD) / (math.pi - FLANK_MIN_RAD)
                    if pos_q > 0.0:
                        agent_yaw = p.getEulerFromQuaternion(agent_orn)[2]
                        head_err = (
                            math.atan2(-fdy, -fdx) - agent_yaw + math.pi
                        ) % (2.0 * math.pi) - math.pi
                        if abs(head_err) < FLANK_HEADING_HALF_RAD:
                            pay = min(
                                REWARD_FLANK_PER_TICK * min(1.0, pos_q),
                                FLANK_EPISODE_CAP - self._flank_paid,
                            )
                            components["flank"] = pay
                            self._flank_paid += pay

            # alg/improvment: OBSERVABLE-PROXY edge avoidance. Uses only
            # signals the deployed policy sees, so the learned behavior
            # transfers. (1) Penalize a blind forward charge — both wheels
            # forward while nothing is detected anywhere in the forward cone
            # (no last-seen latch and all three distances clear): this is the
            # pattern that precedes the 92%-dominant forward self-out, and the
            # penalty nudges the policy to turn and re-acquire instead.
            # (2) Penalize the rear line sensors over the border — the one
            # direct observable "at the edge" cue.
            if self.edge_avoid_reward:
                left_norm = self._norm(distances[1])
                right_norm = self._norm(distances[2])
                no_target = (
                    self.last_seen_dir == 0.0
                    and front_norm > PROXY_CLEAR_NORM
                    and left_norm > PROXY_CLEAR_NORM
                    and right_norm > PROXY_CLEAR_NORM
                )
                if no_target and raw_left > 0.5 and raw_right > 0.5:
                    components["blindcharge"] = REWARD_BLIND_CHARGE
                rear_l = self._line_triggered_at(self.robot_id, *AGENT_LINE_SENSORS[0])
                rear_r = self._line_triggered_at(self.robot_id, *AGENT_LINE_SENSORS[1])
                if rear_l or rear_r:
                    components["rearedge"] = REWARD_REAR_EDGE

        # Step counter advances here so MAX_EPISODE_STEPS is checked
        # against the count *after* this transition.
        self._steps += 1
        truncated = (not terminated) and (self._steps >= MAX_EPISODE_STEPS)
        if truncated:
            terminal = self._rw["timeout"]
            termination_reason = "timeout"

        # Run 8: per-step time cost added to break the "do nothing"
        # attractor. Only paid on non-terminal steps (terminal already
        # captures the episode-end signal); a passive episode now nets
        # REWARD_TIMEOUT + 600 * REWARD_TIME = -13, strictly worse
        # than any episode that accumulates positive shaping.
        if not (terminated or truncated):
            components["time"] = REWARD_TIME
        reward = sum(components.values()) + terminal

        # Per-component cumulative bookkeeping for RewardLoggerCallback.
        for k, v in components.items():
            self._reward_components[k] = self._reward_components.get(k, 0.0) + v
        if terminal:
            self._reward_components["terminal"] = (
                self._reward_components.get("terminal", 0.0) + terminal
            )

        self._update_last_seen(distances)
        # Latch prev_action *before* _build_obs so the obs returned by
        # step() reflects the action just taken. raw_left/raw_right are
        # post-clip but pre-deadzone / pre-latency, matching what the
        # firmware can replicate without those layers.
        self._prev_action = (raw_left, raw_right)
        # Yaw-rate proxy: decayed accumulator of (left - right). Mirror
        # firmware exactly: add then decay then clamp.
        self._yaw_rate_proxy = (
            self._yaw_rate_proxy + (raw_left - raw_right) * 0.1
        ) * 0.9
        if self._yaw_rate_proxy > 1.0:
            self._yaw_rate_proxy = 1.0
        elif self._yaw_rate_proxy < -1.0:
            self._yaw_rate_proxy = -1.0
        obs = self._build_obs(distances)
        info = {
            "agent_pos": agent_pos,
            "enemy_pos": enemy_pos,
            "opponent_id": self._opponent_id,
            "reward_components_step": components,
        }
        if terminated or truncated:
            info["reward_components_episode"] = dict(self._reward_components)
            info["termination_reason"] = termination_reason
            info["terminal_reward"] = float(terminal)
            # Run 6: per-episode wedge mechanics diagnostics. RAW
            # tick counts and signed score so TB can show whether the
            # agent actually achieves engagement (vs. just the
            # weighted reward, which mixes magnitude with frequency).
            info["episode_diag"] = {
                "wedge_score_total": float(self._wedge_total_score),
                "wedge_engaged_ticks": int(self._wedge_engaged_ticks),
                "wedge_being_wedged_ticks": int(self._wedge_being_wedged_ticks),
            }
        return obs, float(reward), terminated, truncated, info

    @property
    def connected(self) -> bool:
        return p.isConnected(self._client_id)

    def render(self) -> None:
        """No-op: rendering is driven by the GUI client when ``gui=True``."""

    def close(self) -> None:
        if p.isConnected(self._client_id):
            p.disconnect(self._client_id)
        self._client_id = -1
        # add/ui-local: drop the generated enemy URDF temp dir (if any).
        if self._enemy_urdf_tmpdir is not None:
            try:
                self._enemy_urdf_tmpdir.cleanup()
            except OSError:  # pragma: no cover - best-effort cleanup
                pass
            self._enemy_urdf_tmpdir = None
            self._enemy_urdf_path = None
        # add/ui-local: drop every per-episode custom-opponent URDF temp dir.
        for tmpdir in self._enemy_spec_tmpdirs:
            try:
                tmpdir.cleanup()
            except OSError:  # pragma: no cover - best-effort cleanup
                pass
        self._enemy_spec_tmpdirs = []
        self._enemy_spec_urdf_cache = {}
