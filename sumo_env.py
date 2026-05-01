"""Mini-sumo dohyo — 4D obs, 2D continuous action, original reward stack.

This is the env shape that produced the 45-55% SAC win-rate. One RL step
≈ 45 ms (11 physics ticks at 240 Hz). The opponent is now the Novamax
Professional Mini Sumo Kit (heavy steel, 5 IR sensors + 2 line sensors,
400 RPM motors); see NovamaxController for the firmware port.

* Observation space — Box(low=-1, high=1, shape=(4,), float32):
    [front_norm, left_norm, right_norm, last_seen_dir]

* Action space — Box(low=-1, high=1, shape=(2,), float32):
    [left_motor, right_motor]

* Rewards (per RL step):
    +1000  win, -1000 loss              (terminal)
    -0.5   per-step time penalty
    +2     hunter   front_norm<1 AND both motors>0.1
    +5     pushing  front_norm<0.2 AND L>0.5 AND R>0.5
    +15    flanking pushing AND outside enemy frontal cone (dot<0.5)
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

# Novamax Professional Mini Sumo Kit (opponent).
NOVAMAX_URDF = os.path.join(os.path.dirname(__file__), "novamax.urdf")
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

# --- Hyper-realistic motor physics --------------------------------
# Agent motors: N20 12 V, 95 RPM, 1.2 kg·cm stall torque.
#   max angular vel = 95 * 2*pi / 60 = 9.948 rad/s
#   max stall force = 0.12 N·m
AGENT_MAX_RAD = 9.95
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

WHEEL_FRICTION = 1.0
CHASSIS_FRICTION = 0.4

# Domain randomization.
DOHYO_FRICTION_RANGE = (0.8, 1.2)
ROBOT_MASS_NOMINAL = 0.5
ROBOT_MASS_VARIATION = 0.10
MOTOR_SLOP_RANGE = (0.85, 1.15)
SENSOR_DEAD_PROB = 0.20

# Sensor max range / observation normalization scale.
ENEMY_FAR_DIST = 0.80
LAST_SEEN_THRESHOLD = 0.5

# "Peak 50%" reward stack scaled up for stronger terminal signal.
REWARD_WIN = 1000.0
REWARD_LOSE = -1000.0
REWARD_TIME = 0.0
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
MAX_EPISODE_STEPS = 500


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
    ) -> None:
        super().__init__()

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
        # When True, the blue robot is driven by keyboard (WASD / arrows)
        # instead of the NovamaxController. Requires gui=True.
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

        self._enemy_ctrl = NovamaxController()
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
                p.changeDynamics(body_id, joint_lookup[jname], lateralFriction=WHEEL_FRICTION)
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

        # Manually push the controller into the reverse phase. The next
        # decide() call will continue the reverse → spin sequence at
        # EDGE_SAFE_SPEED so the chassis keeps traction.
        self._enemy_ctrl._edge_state = "reverse"
        self._enemy_ctrl._edge_timer = self._enemy_ctrl.EDGE_REVERSE_STEPS
        self._enemy_ctrl._edge_spin_dir = +1 if edge_left else -1

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

    def _novamax_line_triggered(self, body_x: float, body_y: float) -> bool:
        """Return True when the corner (body_x, body_y) on the enemy chassis
        is over the dohyo's white border ring (i.e. line sensor fires)."""
        pos, orn = p.getBasePositionAndOrientation(self.enemy_id)
        yaw = p.getEulerFromQuaternion(orn)[2]
        cy, sy = math.cos(yaw), math.sin(yaw)
        wx = pos[0] + cy * body_x - sy * body_y
        wy = pos[1] + sy * body_x + cy * body_y
        return math.hypot(wx, wy) > INNER_RADIUS

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

        # Resolve the per-tier (max_rad, max_force). novamax_torque_mult
        # (when not None) overrides the level lookup so train_novamax_v2
        # can use intermediate steps like 1.5× / 2.0×.
        if self.novamax_torque_mult is not None:
            mult = float(self.novamax_torque_mult)
            max_rad = AGENT_MAX_RAD * mult
            max_force = AGENT_MAX_FORCE * mult
        else:
            max_rad, max_force = NOVAMAX_LEVELS.get(
                self.novamax_level, NOVAMAX_LEVELS[3],
            )

        # Did the edge state machine kick in this tick? (Track the
        # transition None -> active so we only instant-stop once per
        # detection, not every tick of the 5+5 brake sequence.)
        was_braking = self._enemy_ctrl.is_edge_braking
        left_omega, right_omega = self._enemy_ctrl.decide(
            ir_hits, edge_left, edge_right,
        )
        edge_just_triggered = (not was_braking) and self._enemy_ctrl.is_edge_braking

        if edge_just_triggered:
            # NovaMax's 0.05 N·m stall torque can't bleed off ~1.25 kg·m/s
            # of forward momentum in 5 ticks, so the robot would slide off
            # the dohyo before the reverse maneuver can take effect.
            # Instant-stop: zero the wheels and the chassis linear velocity
            # the moment the line sensor fires (preserve angular velocity
            # so the body can still pivot during the spin phase).
            _, ang_vel = p.getBaseVelocity(self.enemy_id)
            base_pos, base_orn = p.getBasePositionAndOrientation(self.enemy_id)
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

        left_omega = max(-max_rad, min(max_rad, float(left_omega)))
        right_omega = max(-max_rad, min(max_rad, float(right_omega)))
        # While braking, give the motors a much higher effective torque
        # cap so the reverse/spin commands actually have authority over
        # the body's momentum.
        torque = max(max_force, 2.0) if self._enemy_ctrl.is_edge_braking else max_force
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

        max_rad, max_force = NOVAMAX_LEVELS.get(
            self.novamax_level, NOVAMAX_LEVELS[3],
        )
        l_omega = l_cmd * max_rad
        r_omega = r_cmd * max_rad
        torque = max_force
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

        # Per-robot yaw and spawn-radius jitter so the agent can't memorise
        # a deterministic head-on charge. Yaw ±30°, radius ±5 cm.
        yaw_jitter_a = float(self._np_random.uniform(-math.radians(30),
                                                      math.radians(30)))
        yaw_jitter_e = float(self._np_random.uniform(-math.radians(30),
                                                      math.radians(30)))
        radius_a = SPAWN_RADIUS + float(self._np_random.uniform(-0.05, 0.05))
        radius_e = SPAWN_RADIUS + float(self._np_random.uniform(-0.05, 0.05))

        agent_pos = (
            radius_a * math.cos(angle),
            radius_a * math.sin(angle),
            spawn_z,
        )
        agent_yaw = angle + math.pi + yaw_jitter_a

        if self._np_random.random() < CURRICULUM_PROB:
            enemy_angle = float(self._np_random.uniform(0.0, 2.0 * math.pi))
            edge_radius = EDGE_SPAWN_RADIUS + float(
                self._np_random.uniform(-0.05, 0.05)
            )
            enemy_pos = (
                edge_radius * math.cos(enemy_angle),
                edge_radius * math.sin(enemy_angle),
                spawn_z,
            )
            enemy_yaw = enemy_angle + yaw_jitter_e
        else:
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
        mass_factor = 1.0 + float(self._np_random.uniform(
            -ROBOT_MASS_VARIATION, ROBOT_MASS_VARIATION
        ))
        p.changeDynamics(self.robot_id, -1, mass=ROBOT_MASS_NOMINAL * mass_factor)

        self.enemy_id, enemy_joints = self._spawn_robot(
            enemy_pos, enemy_yaw, ENEMY_RGBA, urdf_path=NOVAMAX_URDF,
        )
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
                force=NOVAMAX_LEVELS.get(self.novamax_level, NOVAMAX_LEVELS[3])[1],
            )
            p.setJointMotorControl2(
                bodyUniqueId=self.enemy_id, jointIndex=self._enemy_right_idx,
                controlMode=p.VELOCITY_CONTROL, targetVelocity=0.0,
                force=NOVAMAX_LEVELS.get(self.novamax_level, NOVAMAX_LEVELS[3])[1],
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
            # Fast-path edge check (≈4 ms granularity vs 46 ms at the RL
            # boundary): catches the line sensor before NovaMax flies off
            # the dohyo at 2.77 m/s.
            if not self.human_enemy:
                self._novamax_edge_watchdog()
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
