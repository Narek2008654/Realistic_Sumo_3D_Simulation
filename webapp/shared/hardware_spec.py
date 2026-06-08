"""Hardware configuration contract for the mini-sumo robot.

A ``HardwareSpec`` is a frozen, JSON-serialisable description of a single
physical (or simulated) robot: its chassis, drivetrain, sensor layout,
observation encoding, action space, and reward schedule. It is the single
source of truth that the training env, the firmware exporter, and the web
UI all agree on. Two specs with the same ``obs_signature_hash`` produce
byte-compatible observation vectors and action indices, so a checkpoint
trained against one is loadable against the other.

This module is pure Python-3.12 stdlib (``dataclasses``, ``hashlib``,
``json``, ``math``, ``typing``) — no third-party deps — so it can be
imported by the core training env without dragging in the web stack.

``HardwareSpec.default()`` encodes TODAY's robot exactly. Every number is
copied from the live code; the citation for each is given inline next to
the value. Source files (relative to repo root), as read on 2026-06-07:

  sumo_env.py:
    147  AGENT_MAX_RAD   = 41.88     -> drivetrain.max_omega_rad_s
    148  AGENT_MAX_FORCE = 0.12      -> drivetrain.max_torque_nm
    168  WHEEL_MAX_TORQUE = AGENT_MAX_FORCE  (alias of :148)
    169  WHEEL_OMEGA_FWD  = AGENT_MAX_RAD    (alias of :147)
    176  WHEEL_FRICTION   = 2.0      -> chassis.wheel_friction
    182  CHASSIS_FRICTION = 0.05     -> chassis.chassis_friction
    242  ENEMY_FAR_DIST   = 0.80     -> distance_sensors[*].range_m
    198  DR_TOF_NOISE_SIGMA_PCT = 0.02 ; sigma = 0.02 * ENEMY_FAR_DIST
            = 0.016 m                -> distance_sensors[*].noise_sigma
    83-86 AGENT_LINE_SENSORS         -> line_sensors mount_xy
            (-0.02384, +0.0404) rear-left, (-0.02384, -0.0404) rear-right
    270  REWARD_WIN         =  10.0  -> reward.terminal["win"]
    274  REWARD_LOSE_PUSH   = -15.0  -> reward.terminal["lose_push"]
    275  REWARD_LOSE_MUTUAL = -20.0  -> reward.terminal["lose_mutual"]
    283  REWARD_LOSE_SELF   = -50.0  -> reward.terminal["lose_self"]
    287  REWARD_TIMEOUT     = -10.0  -> reward.terminal["timeout"]
    423-427 DISCRETE_ACTION_MAP (9 entries) -> action_space.grid
    578-588 reward shaping bool kwargs       -> reward.shaping_flags
    981-1006 _raw_distances(): sensor_x=0.045, sensor_y=0.0;
            front yaw+0, left yaw+30deg, right yaw-30deg, range ENEMY_FAR_DIST
            -> distance_sensors mount_xyz / angle_rad / range_m
    9-11  obs layout (engineered order)      -> engineered tuple

  obs_stack.py:
    38   DIST_DIM     = 3            -> n_distance (default)
    39   ENGINEERED_DIM = 9          -> len(engineered) (default)
    41   DEFAULT_STACK_K = 4         -> stack_k
    44-46 stacked_dim = DIST_DIM*k + ENGINEERED_DIM = 21 for k=4 -> obs_dim

  assets/robot.urdf:
    13   chassis box 0.0669 x 0.098 x 0.050 m -> chassis length/width/height
            (length uses the chassis box X extent, 0.0669 m)
    45-46 CoM xyz=(-0.0001, 0.0, 0.007), mass=0.45 kg -> chassis.com_xyz/mass
    182  wheel cylinder radius 0.010 m        -> drivetrain.wheel_radius_m
    206  left axle y=+0.03975 -> track_width_m = 0.0795, x=-0.035 offset
    19,162 wedge 28.82 mm long, pitch 0.5113 rad -> chassis.wedge_*
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Leaf component types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DistanceSensor:
    """A forward-facing time-of-flight (VL53L0X/L1X) range finder.

    ``mount_xyz`` is the sensor origin in the robot body frame (metres,
    +X forward / +Y left / +Z up). ``angle_rad`` is the yaw of the ray
    relative to +X (positive = toward +Y / left). ``range_m`` is the
    max reported distance; readings at or beyond it mean "no hit".
    ``noise_sigma`` is the gaussian std-dev (metres) added per step.
    """

    id: str
    mount_xyz: tuple[float, float, float]
    angle_rad: float
    range_m: float
    noise_sigma: float


@dataclass(frozen=True)
class LineSensor:
    """A downward QTR reflectance sensor that fires over the border ring.

    ``mount_xy`` is the (body_x, body_y) position on the chassis bottom
    face, in metres.
    """

    id: str
    mount_xy: tuple[float, float]


@dataclass(frozen=True)
class Drivetrain:
    """Differential drive parameters (one motor per side)."""

    wheel_radius_m: float
    track_width_m: float
    wheel_x_offset_m: float
    max_torque_nm: float
    max_omega_rad_s: float


@dataclass(frozen=True)
class Dohyo:
    """The competition ring (dohyo) geometry.

    ``radius_m`` is the outer radius of the white playing surface (metres);
    a robot whose body centre is carried beyond this falls off the edge.
    ``border_width_m`` is the width of the white border ring at the rim;
    the black inner disc therefore has radius ``radius_m - border_width_m``,
    and a downward line sensor over that annulus reads "border".

    NOTE: the dohyo size is a WORLD parameter, not part of the robot's
    observation/action contract, so it is deliberately EXCLUDED from
    ``HardwareSpec.obs_signature_hash`` — changing the ring does not change
    obs/action byte-compatibility.
    """

    radius_m: float
    border_width_m: float


@dataclass(frozen=True)
class Chassis:
    """Rigid-body and contact parameters of the robot body + wedge.

    The chassis box is described by ``length_m`` / ``width_m`` / ``height_m``,
    where ``length_m`` is the box X extent WITHOUT the wedge (the wedge is a
    separate slab mounted at the front face). ``body_length_m`` is a read-only
    alias of ``length_m`` for callers that want the explicit "body, no wedge"
    name.

    The wedge is described by its horizontal run (``wedge_length_m``) plus the
    heights of its two edges above the floor: ``wedge_low_height_m`` is the
    front lip / sharp tip (thin edge), ``wedge_high_height_m`` is the tall
    rear edge where it joins the chassis. The slope angle is derived from
    these as the ``wedge_pitch_rad`` property; pass ``wedge_pitch_override_rad``
    only to force a specific pitch (rare).
    """

    length_m: float
    width_m: float
    height_m: float
    mass_kg: float
    com_xyz: tuple[float, float, float]
    chassis_friction: float
    wheel_friction: float
    wedge_present: bool
    wedge_length_m: float
    wedge_low_height_m: float
    wedge_high_height_m: float
    # Optional explicit pitch override (radians). When None, the pitch is
    # derived from the low/high edge heights and the wedge length.
    wedge_pitch_override_rad: float | None = None

    @property
    def body_length_m(self) -> float:
        """Chassis box X extent WITHOUT the wedge (alias of ``length_m``)."""
        return self.length_m

    @property
    def wedge_pitch_rad(self) -> float:
        """Slope angle of the wedge about +Y (radians).

        Derived from the edge heights and the horizontal run as
        ``atan2(high - low, wedge_length_m)`` unless an explicit
        ``wedge_pitch_override_rad`` is supplied.
        """
        if self.wedge_pitch_override_rad is not None:
            return self.wedge_pitch_override_rad
        return math.atan2(
            self.wedge_high_height_m - self.wedge_low_height_m,
            self.wedge_length_m,
        )


@dataclass(frozen=True)
class ActionSpace:
    """The policy's action interface.

    ``kind`` is ``"discrete"`` or ``"continuous"``. For discrete, ``grid``
    is the ordered tuple of ``(left, right)`` normalised motor commands —
    the action index selects a row. For continuous, ``grid`` is empty and
    actions are the raw 2-vector ``(left, right)``.
    """

    kind: str
    grid: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if self.kind not in ("discrete", "continuous"):
            raise ValueError(
                f"ActionSpace.kind must be 'discrete' or 'continuous', "
                f"got {self.kind!r}"
            )
        if self.kind == "continuous" and self.grid:
            raise ValueError("continuous ActionSpace must have an empty grid")
        if self.kind == "discrete" and not self.grid:
            raise ValueError("discrete ActionSpace must have a non-empty grid")


@dataclass(frozen=True)
class RewardSpec:
    """Terminal reward magnitudes plus which shaping signals are enabled.

    ``terminal`` keys: win, lose_push, lose_mutual, lose_self, timeout.
    ``shaping_flags`` keys are the env's boolean reward kwargs.
    """

    terminal: dict[str, float]
    shaping_flags: dict[str, bool]


# ---------------------------------------------------------------------------
# Top-level spec
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HardwareSpec:
    """Complete description of one robot configuration.

    ``engineered`` is the ordered tuple of single-frame engineered feature
    names, EXCLUDING the per-frame raw distances. Only the distance
    channels are repeated across ``stack_k`` frames in the observation;
    engineered features (including the line-sensor channels) are
    single-frame. See ``obs_dim``.
    """

    name: str
    chassis: Chassis
    drivetrain: Drivetrain
    distance_sensors: tuple[DistanceSensor, ...]
    line_sensors: tuple[LineSensor, ...]
    stack_k: int
    action_space: ActionSpace
    reward: RewardSpec
    engineered: tuple[str, ...]
    dohyo: Dohyo

    # -- derived sizes -----------------------------------------------------
    @property
    def n_distance(self) -> int:
        """Number of distance (ToF) sensors = raw distance channels."""
        return len(self.distance_sensors)

    @property
    def n_line(self) -> int:
        """Number of downward line sensors."""
        return len(self.line_sensors)

    @property
    def base_obs_dim(self) -> int:
        """Single-frame observation width before frame stacking.

        ``n_distance`` raw distance channels + the engineered features
        (which already exclude the distances). For the default this is
        ``3 + 9 = 12`` (== obs_stack.BASE_OBS_DIM).
        """
        return self.n_distance + len(self.engineered)

    @property
    def obs_dim(self) -> int:
        """Stacked observation width fed to the policy.

        Only the distance channels are repeated across ``stack_k`` frames;
        engineered features are single-frame:
            n_distance * stack_k + len(engineered)
        For the default (3, 4, 9) this is ``3*4 + 9 = 21`` — matching
        ``obs_stack.stacked_dim(DEFAULT_STACK_K)``.
        """
        return self.n_distance * self.stack_k + len(self.engineered)

    @property
    def action_dim(self) -> int:
        """Number of discrete actions (grid rows); 2 for continuous."""
        if self.action_space.kind == "discrete":
            return len(self.action_space.grid)
        return 2

    @property
    def obs_signature_hash(self) -> str:
        """Short, process-stable hex hash of the obs/action contract.

        Computed over a canonical JSON of the fields that determine
        observation/action byte-compatibility. Uses ``hashlib.sha1`` (NOT
        builtin ``hash()``) so it is identical across processes regardless
        of ``PYTHONHASHSEED``.
        """
        payload = {
            "n_distance": self.n_distance,
            "n_line": self.n_line,
            "stack_k": self.stack_k,
            "engineered": list(self.engineered),
            "action_kind": self.action_space.kind,
            "action_dim": self.action_dim,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]

    # -- JSON round-trip ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain (JSON-ready) dict.

        ``dataclasses.asdict`` recurses into the nested dataclasses; tuples
        become lists, which ``from_dict`` re-tuples on the way back.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HardwareSpec":
        """Rebuild a spec from ``to_dict`` output (tuples restored)."""
        chassis_d = dict(data["chassis"])
        chassis_d["com_xyz"] = tuple(chassis_d["com_xyz"])
        # ``wedge_pitch_rad`` is a derived PROPERTY (not a field) -> drop it if a
        # legacy/serialised dict carries it. ``body_length_m`` is likewise an
        # alias property; drop any stray copy too.
        chassis_d.pop("wedge_pitch_rad", None)
        chassis_d.pop("body_length_m", None)
        chassis = Chassis(**chassis_d)

        drivetrain = Drivetrain(**data["drivetrain"])

        dohyo_d = dict(data["dohyo"])
        dohyo = Dohyo(
            radius_m=float(dohyo_d["radius_m"]),
            border_width_m=float(dohyo_d["border_width_m"]),
        )

        distance_sensors = tuple(
            DistanceSensor(
                id=s["id"],
                mount_xyz=tuple(s["mount_xyz"]),
                angle_rad=s["angle_rad"],
                range_m=s["range_m"],
                noise_sigma=s["noise_sigma"],
            )
            for s in data["distance_sensors"]
        )
        line_sensors = tuple(
            LineSensor(id=s["id"], mount_xy=tuple(s["mount_xy"]))
            for s in data["line_sensors"]
        )

        action_d = data["action_space"]
        action_space = ActionSpace(
            kind=action_d["kind"],
            grid=tuple(tuple(row) for row in action_d["grid"]),
        )

        reward_d = data["reward"]
        reward = RewardSpec(
            terminal={k: float(v) for k, v in reward_d["terminal"].items()},
            shaping_flags={
                k: bool(v) for k, v in reward_d["shaping_flags"].items()
            },
        )

        return cls(
            name=data["name"],
            chassis=chassis,
            drivetrain=drivetrain,
            distance_sensors=distance_sensors,
            line_sensors=line_sensors,
            stack_k=int(data["stack_k"]),
            action_space=action_space,
            reward=reward,
            engineered=tuple(data["engineered"]),
            dohyo=dohyo,
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "HardwareSpec":
        """Rebuild a spec from a JSON string."""
        return cls.from_dict(json.loads(text))

    # -- the canonical robot ----------------------------------------------
    @classmethod
    def default(cls) -> "HardwareSpec":
        """The spec encoding TODAY's robot exactly.

        Every constant is copied from the live code; see the module
        docstring for the source file:line of each value.
        """
        # _raw_distances(): all three ToF sensors share the same mount
        # (sumo_env.py:986-987), differing only in ray yaw.
        mount = (0.045, 0.0, 0.007)  # x,y from _raw_distances; z = CoM height
        far = 0.80                   # sumo_env.py:242 ENEMY_FAR_DIST
        sigma = 0.02 * far           # sumo_env.py:198 DR_TOF_NOISE_SIGMA_PCT
        deg30 = math.radians(30.0)   # sumo_env.py:988-989 side rays +/-30deg

        distance_sensors = (
            DistanceSensor("front", mount, 0.0, far, sigma),
            DistanceSensor("left", mount, +deg30, far, sigma),
            DistanceSensor("right", mount, -deg30, far, sigma),
        )

        # sumo_env.py:83-86 AGENT_LINE_SENSORS (rear-left, rear-right).
        line_sensors = (
            LineSensor("line_l", (-0.02384, 0.0404)),
            LineSensor("line_r", (-0.02384, -0.0404)),
        )

        drivetrain = Drivetrain(
            wheel_radius_m=0.010,    # robot.urdf:182 cylinder radius
            track_width_m=0.0795,    # robot.urdf:206 axle y=+/-0.03975
            wheel_x_offset_m=-0.035,  # robot.urdf:206 axle x
            max_torque_nm=0.12,      # sumo_env.py:148 AGENT_MAX_FORCE
            max_omega_rad_s=41.88,   # sumo_env.py:147 AGENT_MAX_RAD
        )

        # Wedge edge heights: keep the front lip on the floor (low = 0) and
        # set the back edge so the DERIVED pitch reproduces today's 0.5113 rad
        # (robot.urdf:162) for the 28.82 mm run, i.e. high = run * tan(0.5113).
        # This makes generate_urdf(default) byte-identical to the prior wedge.
        wedge_run = 0.02882          # robot.urdf:19 wedge 28.82 mm long
        wedge_low = 0.0              # front lip skims the floor
        wedge_high = wedge_run * math.tan(0.5113)  # back edge -> pitch 0.5113
        chassis = Chassis(
            length_m=0.0669,         # robot.urdf:13/63 chassis box X (body, no wedge)
            width_m=0.098,           # robot.urdf:13/63 chassis box Y
            height_m=0.050,          # robot.urdf:13/63 chassis box Z
            mass_kg=0.45,            # robot.urdf:46 mass value
            com_xyz=(-0.0001, 0.0, 0.007),  # robot.urdf:45 inertial origin
            chassis_friction=0.05,   # sumo_env.py:182 CHASSIS_FRICTION
            wheel_friction=2.0,      # sumo_env.py:176 WHEEL_FRICTION
            wedge_present=True,       # robot.urdf:108-163 nose_wedge link
            wedge_length_m=wedge_run,
            wedge_low_height_m=wedge_low,
            wedge_high_height_m=wedge_high,
            # No override: wedge_pitch_rad property derives 0.5113 from the above.
        )

        # sumo_env.py:423-427 DISCRETE_ACTION_MAP (9 entries).
        action_space = ActionSpace(
            kind="discrete",
            grid=(
                (-1.0, -1.0), (-1.0, 0.0), (-1.0, +1.0),
                (0.0, -1.0), (0.0, 0.0), (0.0, +1.0),
                (+1.0, -1.0), (+1.0, 0.0), (+1.0, +1.0),
            ),
        )

        reward = RewardSpec(
            terminal={
                "win": 10.0,          # sumo_env.py:270 REWARD_WIN
                "lose_push": -15.0,   # sumo_env.py:274 REWARD_LOSE_PUSH
                "lose_mutual": -20.0,  # sumo_env.py:275 REWARD_LOSE_MUTUAL
                "lose_self": -50.0,   # sumo_env.py:283 REWARD_LOSE_SELF
                "timeout": -10.0,     # sumo_env.py:287 REWARD_TIMEOUT
            },
            # sumo_env.py:578-588 reward shaping bool kwargs (default off).
            shaping_flags={
                "tracking_reward": False,
                "flank_reward": False,
                "still_penalty": False,
                "backward_penalty": False,
                "edge_avoid_reward": False,
                "narek_reward": False,
                "action_consistency_reward": False,
            },
        )

        # sumo_env.py:9-11 obs layout, engineered channels = obs[3:12],
        # i.e. everything after the 3 raw distances.
        engineered = (
            "last_seen_dir",
            "line_l",
            "line_r",
            "prev_left",
            "prev_right",
            "engagement",
            "yaw_rate_proxy",
            "front_ir_delta",
            "lateral_ir_delta",
        )

        # sumo_env.py:43-46 DOHYO_RADIUS = 0.70/2 = 0.35, BORDER_WIDTH = 0.025
        # (INNER_RADIUS = 0.325 is derived as radius_m - border_width_m).
        dohyo = Dohyo(radius_m=0.35, border_width_m=0.025)

        return cls(
            name="default_v3",
            chassis=chassis,
            drivetrain=drivetrain,
            distance_sensors=distance_sensors,
            line_sensors=line_sensors,
            stack_k=4,               # obs_stack.py:41 DEFAULT_STACK_K
            action_space=action_space,
            reward=reward,
            engineered=engineered,
            dohyo=dohyo,
        )


# ---------------------------------------------------------------------------
# Mini-sumo class limits (regulation): <= 500 g, <= 10 x 10 cm footprint.
# Enforced on the chassis box (length_m x width_m) + mass at authoring time
# (robot/opponent save + /hardware/validate). The default and the faithful
# novamax both fit, so the gate never rejects the baseline.
# ---------------------------------------------------------------------------
MINI_SUMO_MAX_MASS_KG = 0.5
MINI_SUMO_MAX_FOOTPRINT_M = 0.10


def mini_sumo_violations(spec: "HardwareSpec") -> list[str]:
    """Human-readable reasons ``spec`` breaks the mini-sumo class limits.

    Empty list == compliant. Checks the chassis box footprint
    (``length_m`` x ``width_m``) and ``mass_kg`` against the 10x10 cm / 500 g
    regulation, with a tiny tolerance for float noise.
    """
    tol = 1e-6
    c = spec.chassis
    out: list[str] = []
    if c.mass_kg > MINI_SUMO_MAX_MASS_KG + tol:
        out.append(
            f"mass {c.mass_kg * 1000:.0f} g exceeds the mini-sumo limit of "
            f"{MINI_SUMO_MAX_MASS_KG * 1000:.0f} g"
        )
    if c.length_m > MINI_SUMO_MAX_FOOTPRINT_M + tol:
        out.append(
            f"body length {c.length_m * 100:.1f} cm exceeds the mini-sumo limit "
            f"of {MINI_SUMO_MAX_FOOTPRINT_M * 100:.0f} cm"
        )
    if c.width_m > MINI_SUMO_MAX_FOOTPRINT_M + tol:
        out.append(
            f"body width {c.width_m * 100:.1f} cm exceeds the mini-sumo limit "
            f"of {MINI_SUMO_MAX_FOOTPRINT_M * 100:.0f} cm"
        )
    return out
