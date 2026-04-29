"""Mini-sumo dohyo — 4D obs, 2D continuous action, original reward stack.

This is the env shape that produced the 45-55% SAC win-rate against the
2.0× davo_sirad opponent. One RL step ≈ 45 ms (11 physics ticks at 240 Hz).

* Observation space — Box(low=-1, high=1, shape=(4,), float32):
    [front_norm, left_norm, right_norm, last_seen_dir]

* Action space — Box(low=-1, high=1, shape=(2,), float32):
    [left_motor, right_motor]

* Rewards (per RL step):
    +100   win, -100 loss               (terminal — original scale)
    -0.5   per-step time penalty
    +2     hunter      front_norm<1 AND both motors>0.1
    +1     tracking-left   left_norm<1  AND L<R
    +1     tracking-right  right_norm<1 AND L>R
    -5     coward     L<-0.5 AND R<-0.5
    +5     pushing    front_norm<0.2 AND L>0.5 AND R>0.5
    +15    flanking   pushing AND outside enemy frontal cone (dot<0.5)
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

ROBOT_WIDTH = 0.10
ROBOT_FRONT_EXTENT = 0.0625
WHEEL_RADIUS = 0.014
WHEEL_TRACK_HALF = 0.028
ROBOT_URDF = os.path.join(os.path.dirname(__file__), "robot.urdf")

AGENT_RGBA = (0.85, 0.15, 0.15, 1.0)
ENEMY_RGBA = (0.15, 0.30, 0.85, 1.0)

ENEMY_SENSOR_MAX_M = 1.5
SPAWN_RADIUS = 0.25

SIM_TIMESTEP = 1.0 / 240.0
# 1 RL step = SUBSTEPS_PER_STEP physics ticks at 240 Hz.
# 11 ticks ≈ 45.8 ms — fast enough to react to davo_sirad's pivots.
SUBSTEPS_PER_STEP = 11

# Initial agent-only forward charge at episode start (500 ms). The red
# robot drives both wheels full forward; the enemy stays still until
# the policy takes over.
INITIAL_CHARGE_MS = 500
INITIAL_CHARGE_TICKS = int(round((INITIAL_CHARGE_MS / 1000.0) / SIM_TIMESTEP))

WHEEL_MAX_TORQUE = 2.0
LINEAR_SPEED = 0.40
WHEEL_OMEGA_FWD = LINEAR_SPEED / WHEEL_RADIUS
FORWARD_SIGN = 1.0

# Enemy: 3x torque so head-on pushing contests are decisively the
# enemy's win. The agent must learn to flank/sidestep, not engage
# head-on. Base speed otherwise.
ENEMY_WHEEL_TORQUE_MULT = 3.0
ENEMY_FORWARD_OMEGA = WHEEL_OMEGA_FWD
ENEMY_PIVOT_OMEGA = WHEEL_OMEGA_FWD
ENEMY_WHEEL_TORQUE = WHEEL_MAX_TORQUE * ENEMY_WHEEL_TORQUE_MULT

WHEEL_FRICTION = 1.0
CHASSIS_FRICTION = 0.4

# Domain randomization.
DOHYO_FRICTION_RANGE = (0.8, 1.2)
ROBOT_MASS_NOMINAL = 0.4
ROBOT_MASS_VARIATION = 0.10
MOTOR_SLOP_RANGE = (0.85, 1.15)
SENSOR_DEAD_PROB = 0.20

# Sensor max range / observation normalization scale.
ENEMY_FAR_DIST = 0.80
LAST_SEEN_THRESHOLD = 0.5

# Original reward stack that produced 45-55% win-rate.
REWARD_WIN = 100.0
REWARD_LOSE = -100.0
REWARD_TIME = -0.5
REWARD_HUNTER = 2.0
REWARD_TRACK = 1.0
REWARD_COWARD = -5.0
REWARD_PUSH = 5.0
REWARD_FLANK = 15.0

# Trigger thresholds.
HUNTER_MOTOR_THRESHOLD = 0.1
COWARD_MOTOR_THRESHOLD = -0.5
PUSH_FRONT_THRESHOLD = 0.2
PUSH_MOTOR_THRESHOLD = 0.5
FLANK_DOT_THRESHOLD = 0.5

# Curriculum.
CURRICULUM_PROB = 0.25
EDGE_SPAWN_RADIUS = 0.30

FALL_Z = 0.0
MAX_EPISODE_STEPS = 250


class EnemyController:
    """Python port of the davo_sirad.ino opponent's behavior loop."""

    OUTLIER_MAX = 8000
    DETECT_DIST = 350
    HOLD_DIST = 450
    CONFIRM_HITS = 1
    CONFIRM_MISSES = (2, 10, 2)

    FORWARD = 0
    LEFT = 1
    RIGHT = 2

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._last_valid = [500, 500, 500]
        self._has_ever_seen = [False, False, False]
        self._currently_seen = [False, False, False]
        self._hit_streak = [0, 0, 0]
        self._miss_streak = [0, 0, 0]
        self._spin_dir_right = False

    def _filtered(self, idx: int, raw_mm: int) -> int:
        valid = 0 < raw_mm < self.OUTLIER_MAX
        if not valid:
            self._hit_streak[idx] = 0
            if self._miss_streak[idx] < 255:
                self._miss_streak[idx] += 1
            if self._miss_streak[idx] >= self.CONFIRM_MISSES[idx]:
                self._currently_seen[idx] = False
                self._has_ever_seen[idx] = False
                return self.OUTLIER_MAX
            return self._last_valid[idx] if self._has_ever_seen[idx] else self.OUTLIER_MAX

        self._miss_streak[idx] = 0
        if self._hit_streak[idx] < 255:
            self._hit_streak[idx] += 1
        if self._hit_streak[idx] >= self.CONFIRM_HITS:
            self._currently_seen[idx] = True
        self._has_ever_seen[idx] = True
        self._last_valid[idx] = raw_mm
        return raw_mm

    def decide(self, raw_right_mm: int, raw_front_mm: int, raw_left_mm: int) -> int:
        d0 = self._filtered(0, raw_right_mm)
        d1 = self._filtered(1, raw_front_mm)
        d2 = self._filtered(2, raw_left_mm)

        limit0 = self.HOLD_DIST if self._currently_seen[0] else self.DETECT_DIST
        limit1 = self.HOLD_DIST if self._currently_seen[1] else self.DETECT_DIST
        limit2 = self.HOLD_DIST if self._currently_seen[2] else self.DETECT_DIST

        s0 = self._currently_seen[0] and d0 <= limit0
        s1 = self._currently_seen[1] and d1 <= limit1
        s2 = self._currently_seen[2] and d2 <= limit2

        if s0 and s1 and s2:
            if d1 <= d0 and d1 <= d2:
                return self.FORWARD
            if d0 <= d2:
                self._spin_dir_right = True
                return self.RIGHT
            self._spin_dir_right = False
            return self.LEFT
        if s1:
            return self.FORWARD
        if s0:
            self._spin_dir_right = True
            return self.RIGHT
        if s2:
            self._spin_dir_right = False
            return self.LEFT
        return self.RIGHT if self._spin_dir_right else self.LEFT


class MiniSumoEnv(gym.Env):
    """Fast reactive continuous-control mini-sumo (no macro-actions)."""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        gui: bool = False,
        seed: Optional[int] = None,
        enemy_torque_multiplier: float = 2.0,
        human_enemy: bool = False,
    ) -> None:
        super().__init__()

        self.gui = gui
        # Curriculum knob — set by the trainer between phases via
        # env.unwrapped.enemy_torque_mult = X. Multiplies the wheel torque
        # cap applied to the davo_sirad opponent.
        self.enemy_torque_mult = float(enemy_torque_multiplier)
        # When True, the blue robot is driven by keyboard (WASD / arrows)
        # instead of the davo_sirad controller. Requires gui=True.
        self.human_enemy = bool(human_enemy)
        self._client_id = -1
        self._connect()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32,
        )

        self.robot_id: Optional[int] = None
        self.enemy_id: Optional[int] = None
        self._left_wheel_idx: Optional[int] = None
        self._right_wheel_idx: Optional[int] = None
        self._caster_idx: Optional[int] = None
        self._enemy_left_idx: Optional[int] = None
        self._enemy_right_idx: Optional[int] = None

        self._enemy_ctrl = EnemyController()
        self._disabled_sensor: Optional[int] = None

        self.last_seen_dir: float = 0.0
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
            shapeType=p.GEOM_CYLINDER, radius=DOHYO_RADIUS,
            length=DOHYO_THICKNESS, rgbaColor=(1.0, 1.0, 1.0, 1.0),
        )
        white_collision = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER, radius=DOHYO_RADIUS, height=DOHYO_THICKNESS,
        )
        white_id = p.createMultiBody(
            baseMass=0.0, baseCollisionShapeIndex=white_collision,
            baseVisualShapeIndex=white_visual,
            basePosition=[0.0, 0.0, DOHYO_THICKNESS / 2.0],
        )
        p.changeDynamics(white_id, -1, lateralFriction=surface_friction)

        black_visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER, radius=INNER_RADIUS,
            length=BLACK_TOP_THICKNESS, rgbaColor=(0.05, 0.05, 0.05, 1.0),
        )
        black_collision = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER, radius=INNER_RADIUS,
            height=BLACK_TOP_THICKNESS,
        )
        black_id = p.createMultiBody(
            baseMass=0.0, baseCollisionShapeIndex=black_collision,
            baseVisualShapeIndex=black_visual,
            basePosition=[0.0, 0.0, DOHYO_THICKNESS + BLACK_TOP_THICKNESS / 2.0],
        )
        p.changeDynamics(black_id, -1, lateralFriction=surface_friction)

    def _spawn_robot(self, position, yaw: float, chassis_rgba) -> tuple[int, dict]:
        orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        body_id = p.loadURDF(
            ROBOT_URDF, basePosition=list(position), baseOrientation=orn,
            useFixedBase=False, flags=p.URDF_USE_INERTIA_FROM_FILE,
        )
        p.changeVisualShape(body_id, -1, rgbaColor=chassis_rgba)

        joint_lookup = {
            p.getJointInfo(body_id, j)[1].decode(): j
            for j in range(p.getNumJoints(body_id))
        }
        for jname in ("left_wheel_joint", "right_wheel_joint"):
            p.changeDynamics(body_id, joint_lookup[jname], lateralFriction=WHEEL_FRICTION)
        p.changeDynamics(body_id, -1, lateralFriction=CHASSIS_FRICTION)
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

    def _enemy_distance_mm(self, body_dx: float, body_dy: float, start_offset: float) -> int:
        dist = self._ray_distance(
            self.enemy_id, body_dx, body_dy, start_offset,
            ENEMY_SENSOR_MAX_M, target_id=self.robot_id,
        )
        if dist is None:
            return EnemyController.OUTLIER_MAX
        return int(dist * 1000.0)

    def _raw_distances(self) -> list[Optional[float]]:
        front_offset = ROBOT_FRONT_EXTENT + 0.005
        side_offset = ROBOT_WIDTH / 2.0 + 0.005
        readings: list[Optional[float]] = [
            self._ray_distance(self.robot_id, 1.0, 0.0, front_offset, ENEMY_FAR_DIST),
            self._ray_distance(self.robot_id, 0.0, 1.0, side_offset, ENEMY_FAR_DIST),
            self._ray_distance(self.robot_id, 0.0, -1.0, side_offset, ENEMY_FAR_DIST),
        ]
        if self._disabled_sensor is not None:
            readings[self._disabled_sensor] = None
        return readings

    @staticmethod
    def _norm(distance: Optional[float]) -> float:
        if distance is None:
            return 1.0
        return min(1.0, max(0.0, distance / ENEMY_FAR_DIST))

    def _update_last_seen(self, distances: list[Optional[float]]) -> None:
        front, left, right = distances
        if left is not None and self._norm(left) < LAST_SEEN_THRESHOLD:
            self.last_seen_dir = -1.0
        if right is not None and self._norm(right) < LAST_SEEN_THRESHOLD:
            self.last_seen_dir = 1.0
        if front is not None and self._norm(front) < LAST_SEEN_THRESHOLD:
            self.last_seen_dir = 0.0

    def _build_obs(self, distances: list[Optional[float]]) -> np.ndarray:
        front, left, right = distances
        return np.array(
            [self._norm(front), self._norm(left), self._norm(right), self.last_seen_dir],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------
    def _apply_motor_velocities(self, left_cmd: float, right_cmd: float) -> None:
        left_omega = left_cmd * WHEEL_OMEGA_FWD
        right_omega = right_cmd * WHEEL_OMEGA_FWD
        left_omega *= float(self._np_random.uniform(*MOTOR_SLOP_RANGE))
        right_omega *= float(self._np_random.uniform(*MOTOR_SLOP_RANGE))

        p.setJointMotorControl2(
            bodyUniqueId=self.robot_id, jointIndex=self._left_wheel_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=FORWARD_SIGN * left_omega, force=WHEEL_MAX_TORQUE,
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.robot_id, jointIndex=self._right_wheel_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=FORWARD_SIGN * right_omega, force=WHEEL_MAX_TORQUE,
        )

    def _apply_enemy_control(self) -> None:
        front_offset = ROBOT_FRONT_EXTENT + 0.005
        side_offset = ROBOT_WIDTH / 2.0 + 0.005
        right_mm = self._enemy_distance_mm(0.0, -1.0, side_offset)
        front_mm = self._enemy_distance_mm(1.0, 0.0, front_offset)
        left_mm = self._enemy_distance_mm(0.0, 1.0, side_offset)

        cmd = self._enemy_ctrl.decide(right_mm, front_mm, left_mm)

        if cmd == EnemyController.FORWARD:
            left_omega, right_omega = ENEMY_FORWARD_OMEGA, ENEMY_FORWARD_OMEGA
        elif cmd == EnemyController.RIGHT:
            left_omega, right_omega = ENEMY_PIVOT_OMEGA, -ENEMY_PIVOT_OMEGA
        else:
            left_omega, right_omega = -ENEMY_PIVOT_OMEGA, ENEMY_PIVOT_OMEGA

        # Curriculum-controlled enemy torque cap.
        torque = WHEEL_MAX_TORQUE * self.enemy_torque_mult
        p.setJointMotorControl2(
            bodyUniqueId=self.enemy_id, jointIndex=self._enemy_left_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=FORWARD_SIGN * left_omega, force=torque,
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.enemy_id, jointIndex=self._enemy_right_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=FORWARD_SIGN * right_omega, force=torque,
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

        l_omega = l_cmd * WHEEL_OMEGA_FWD
        r_omega = r_cmd * WHEEL_OMEGA_FWD
        torque = WHEEL_MAX_TORQUE * self.enemy_torque_mult
        p.setJointMotorControl2(
            bodyUniqueId=self.enemy_id, jointIndex=self._enemy_left_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=FORWARD_SIGN * l_omega, force=torque,
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.enemy_id, jointIndex=self._enemy_right_idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=FORWARD_SIGN * r_omega, force=torque,
        )

    @staticmethod
    def _flank_dot(agent_pos, enemy_pos, enemy_yaw: float) -> float:
        dx = agent_pos[0] - enemy_pos[0]
        dy = agent_pos[1] - enemy_pos[1]
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return 1.0
        vx, vy = dx / norm, dy / norm
        fx, fy = math.cos(enemy_yaw), math.sin(enemy_yaw)
        return vx * fx + vy * fy

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        p.resetSimulation()
        p.setGravity(0.0, 0.0, -9.81)
        p.setTimeStep(SIM_TIMESTEP)

        surface_friction = float(self._np_random.uniform(*DOHYO_FRICTION_RANGE))
        self._build_dohyo(surface_friction)

        if self._np_random.random() < SENSOR_DEAD_PROB:
            self._disabled_sensor = int(self._np_random.integers(0, 3))
        else:
            self._disabled_sensor = None

        angle = float(self._np_random.uniform(0.0, 2.0 * math.pi))
        spawn_z = DOHYO_TOP_Z + 0.001
        agent_pos = (
            SPAWN_RADIUS * math.cos(angle),
            SPAWN_RADIUS * math.sin(angle),
            spawn_z,
        )
        agent_yaw = angle + math.pi

        if self._np_random.random() < CURRICULUM_PROB:
            enemy_angle = float(self._np_random.uniform(0.0, 2.0 * math.pi))
            enemy_pos = (
                EDGE_SPAWN_RADIUS * math.cos(enemy_angle),
                EDGE_SPAWN_RADIUS * math.sin(enemy_angle),
                spawn_z,
            )
            enemy_yaw = enemy_angle
        else:
            enemy_pos = (
                -SPAWN_RADIUS * math.cos(angle),
                -SPAWN_RADIUS * math.sin(angle),
                spawn_z,
            )
            enemy_yaw = angle

        self.robot_id, agent_joints = self._spawn_robot(agent_pos, agent_yaw, AGENT_RGBA)
        self._left_wheel_idx = agent_joints["left_wheel_joint"]
        self._right_wheel_idx = agent_joints["right_wheel_joint"]
        self._caster_idx = agent_joints["front_caster_joint"]
        mass_factor = 1.0 + float(self._np_random.uniform(
            -ROBOT_MASS_VARIATION, ROBOT_MASS_VARIATION
        ))
        p.changeDynamics(self.robot_id, -1, mass=ROBOT_MASS_NOMINAL * mass_factor)

        self.enemy_id, enemy_joints = self._spawn_robot(enemy_pos, enemy_yaw, ENEMY_RGBA)
        self._enemy_left_idx = enemy_joints["left_wheel_joint"]
        self._enemy_right_idx = enemy_joints["right_wheel_joint"]
        self._enemy_ctrl.reset()

        self.last_seen_dir = 0.0
        self._steps = 0

        # 500 ms forward charge: red drives both wheels full forward,
        # blue holds still (zero target velocity). davo_sirad's hysteresis
        # only ticks while we're calling _apply_enemy_control, which we
        # skip here, so it just observes the runup.
        for _ in range(INITIAL_CHARGE_TICKS):
            if not p.isConnected(self._client_id):
                break
            self._apply_motor_velocities(1.0, 1.0)
            p.setJointMotorControl2(
                bodyUniqueId=self.enemy_id, jointIndex=self._enemy_left_idx,
                controlMode=p.VELOCITY_CONTROL, targetVelocity=0.0,
                force=WHEEL_MAX_TORQUE * self.enemy_torque_mult,
            )
            p.setJointMotorControl2(
                bodyUniqueId=self.enemy_id, jointIndex=self._enemy_right_idx,
                controlMode=p.VELOCITY_CONTROL, targetVelocity=0.0,
                force=WHEEL_MAX_TORQUE * self.enemy_torque_mult,
            )
            p.stepSimulation()
            if self.gui:
                time.sleep(SIM_TIMESTEP)

        distances = self._raw_distances()
        self._update_last_seen(distances)
        return self._build_obs(distances), {}

    def step(self, action):
        if not p.isConnected(self._client_id):
            zero_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return zero_obs, 0.0, False, True, {"disconnected": True}

        action = np.asarray(action, dtype=np.float32).flatten()
        if action.shape != (2,):
            raise ValueError(f"action must be (2,), got {action.shape}")
        left_cmd = float(np.clip(action[0], -1.0, 1.0))
        right_cmd = float(np.clip(action[1], -1.0, 1.0))

        self._apply_motor_velocities(left_cmd, right_cmd)
        if self.human_enemy:
            self._apply_human_enemy_control()
        else:
            self._apply_enemy_control()

        for _ in range(SUBSTEPS_PER_STEP):
            if not p.isConnected(self._client_id):
                zero_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
                return zero_obs, 0.0, False, True, {"disconnected": True}
            p.stepSimulation()
            if self.gui:
                time.sleep(SIM_TIMESTEP)

        agent_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        enemy_pos, enemy_orn = p.getBasePositionAndOrientation(self.enemy_id)
        agent_out = agent_pos[2] < FALL_Z
        enemy_out = enemy_pos[2] < FALL_Z

        reward = REWARD_TIME
        terminated = False

        if agent_out or enemy_out:
            if enemy_out and not agent_out:
                reward += REWARD_WIN
            else:
                reward += REWARD_LOSE
            terminated = True
            distances = [None, None, None]
        else:
            distances = self._raw_distances()
            front_norm = self._norm(distances[0])
            left_norm = self._norm(distances[1])
            right_norm = self._norm(distances[2])

            # Hunter: charge forward when something is in front.
            if (front_norm < 1.0
                    and left_cmd > HUNTER_MOTOR_THRESHOLD
                    and right_cmd > HUNTER_MOTOR_THRESHOLD):
                reward += REWARD_HUNTER

            # Tracking: turn toward whichever side sensor is hit.
            if left_norm < 1.0 and left_cmd < right_cmd:
                reward += REWARD_TRACK
            if right_norm < 1.0 and left_cmd > right_cmd:
                reward += REWARD_TRACK

            # Coward: heavy reverse on both motors.
            if left_cmd < COWARD_MOTOR_THRESHOLD and right_cmd < COWARD_MOTOR_THRESHOLD:
                reward += REWARD_COWARD

            # Pushing + flank: close-range pressure, bonus for side/rear.
            if (front_norm < PUSH_FRONT_THRESHOLD
                    and left_cmd > PUSH_MOTOR_THRESHOLD
                    and right_cmd > PUSH_MOTOR_THRESHOLD):
                reward += REWARD_PUSH
                enemy_yaw = p.getEulerFromQuaternion(enemy_orn)[2]
                if self._flank_dot(agent_pos, enemy_pos, enemy_yaw) < FLANK_DOT_THRESHOLD:
                    reward += REWARD_FLANK

        self._steps += 1
        truncated = (not terminated) and (self._steps >= MAX_EPISODE_STEPS)

        self._update_last_seen(distances)
        obs = self._build_obs(distances)
        info = {"agent_pos": agent_pos, "enemy_pos": enemy_pos}
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
