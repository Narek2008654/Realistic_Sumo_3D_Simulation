// TypeScript mirror of webapp/shared/hardware_spec.py (HardwareSpec.to_dict()).
// Tuples on the Python side serialise to JSON arrays; we type them as fixed
// or variable-length number arrays accordingly.

export type Vec3 = [number, number, number];

// 3D preview camera mode (frontend-only UI state).
export type PreviewView = 'TOP' | 'UNDERSIDE';

export interface DistanceSensor {
  id: string;
  mount_xyz: Vec3;
  angle_rad: number;
  range_m: number;
  noise_sigma: number;
}

export interface LineSensor {
  id: string;
  mount_xy: [number, number];
}

export interface Drivetrain {
  wheel_radius_m: number;
  track_width_m: number;
  wheel_x_offset_m: number;
  max_torque_nm: number;
  max_omega_rad_s: number;
}

export interface Chassis {
  length_m: number; // body length (chassis box X extent, no wedge)
  width_m: number;
  height_m: number;
  mass_kg: number;
  com_xyz: Vec3;
  chassis_friction: number;
  wheel_friction: number;
  wedge_present: boolean;
  wedge_length_m: number;
  wedge_low_height_m: number;
  wedge_high_height_m: number;
  // Optional explicit pitch override (radians); null -> derived from edges.
  wedge_pitch_override_rad: number | null;
}

export interface Dohyo {
  radius_m: number;
  border_width_m: number;
}

export interface ActionSpace {
  kind: 'discrete' | 'continuous';
  grid: [number, number][];
}

export interface RewardSpec {
  terminal: Record<string, number>;
  shaping_flags: Record<string, boolean>;
}

export interface HardwareSpec {
  name: string;
  chassis: Chassis;
  drivetrain: Drivetrain;
  distance_sensors: DistanceSensor[];
  line_sensors: LineSensor[];
  stack_k: number;
  action_space: ActionSpace;
  reward: RewardSpec;
  engineered: string[];
  dohyo: Dohyo;
}

// ---- Saved robots ----------------------------------------------------------

export interface RobotSummary {
  id: string;
  name: string;
  created_at: string;
  obs_dim: number;
  action_dim: number;
  obs_signature_hash: string | null;
}

export interface RobotRecord extends RobotSummary {
  hardware_spec: HardwareSpec;
}

// ---- API response shapes ---------------------------------------------------

export interface ValidateResult {
  obs_dim: number;
  action_dim: number;
  obs_signature_hash: string;
  urdf_valid: boolean;
  errors: string[];
  finetune_candidates: ModelCard[];
}

export type GeomShape = 'box' | 'cylinder' | 'sphere' | 'mesh';

export interface GeomLink {
  name: string;
  shape: GeomShape;
  size?: Vec3; // box
  radius?: number; // cylinder / sphere
  length?: number; // cylinder
  filename?: string; // mesh
  scale?: Vec3;
  origin_xyz: Vec3;
  origin_rpy: Vec3;
  rgba: [number, number, number, number];
}

export interface GeomJoint {
  name: string;
  type: string;
  parent: string | null;
  child: string | null;
  origin_xyz: Vec3;
  origin_rpy: Vec3;
  axis: Vec3 | null;
}

export interface Geometry {
  links: GeomLink[];
  joints: GeomJoint[];
}

export interface ModelMetrics {
  [key: string]: number | string | null;
}

export interface ModelCard {
  id: string;
  filename: string;
  algo: string;
  obs_dim: number;
  action_dim: number;
  net_arch: [number, number];
  param_count: number;
  obs_signature_hash: string | null;
  metrics: ModelMetrics | null;
  created_at: string;
}

// ---- Training ---------------------------------------------------------------

export type TrainAlgo = 'dqn' | 'ppo';
export type TrainMode = 'scratch' | 'finetune';

export interface StartTrainBody {
  robot_id?: string;
  hardware_spec?: HardwareSpec;
  algo: TrainAlgo;
  mode: TrainMode;
  base_model_id?: string;
  total_steps?: number;
  eval_every?: number;
  opponent_weights?: Record<string, number> | null;
  smoke?: boolean;
  seed?: number;
}

export interface TrainHyperparams {
  lr: number;
  gamma: number;
  batch_size: number;
  eps_start: number;
  eps_end: number;
}

export interface RecommendResult {
  algo: TrainAlgo;
  total_steps: number;
  eval_every: number;
  net_arch: [number, number];
  start_mult: number;
  hyperparams: TrainHyperparams;
  est_minutes: number;
}

/** Per-opponent eval block from scripts.eval_best.run_eval. */
export interface OpponentEval {
  wins: number;
  losses: number;
  timeouts?: number;
  self_out: number;
  push_loss?: number;
  mutual_out?: number;
  n: number;
  mean_ep_len: number;
  [key: string]: number | undefined;
}

/** Aggregated eval metrics carried by a checkpoint event. */
export interface CheckpointEval {
  wins: number;
  losses: number;
  self_out: number;
  n: number;
  wr: number;
  mean_ep_len: number;
  per_opponent?: Record<string, OpponentEval>;
  mult?: number;
}

export interface CheckpointEvent {
  t: 'checkpoint';
  step: number;
  snapshot: string;
  eval: CheckpointEval;
  trajectory: string;
}

export interface TrainEvent {
  t: string;
  step?: number;
  [key: string]: unknown;
}

export type TrainState =
  | 'idle'
  | 'running'
  | 'done'
  | 'error'
  | 'stopped'
  | 'unknown';

export interface TrainStatus {
  state: TrainState;
  running: boolean;
  config: Record<string, unknown> | null;
  events: TrainEvent[];
  latest_checkpoint: CheckpointEvent | null;
  job_id: string | null;
}

export interface TrainJobSummary {
  id: string;
  algo: string | null;
  mode: string | null;
  smoke: boolean;
  started_at: string | null;
  running: boolean;
}

// ---- Trajectory replay ------------------------------------------------------

export interface FramePose {
  p: Vec3;
  q: [number, number, number, number];
}

export interface TrajectoryFrame {
  agent: FramePose;
  enemy: FramePose;
}

export interface TrajectoryOutcome {
  winner: 'agent' | 'enemy' | null;
  reason: string;
}

export interface Trajectory {
  dt: number;
  dohyo_radius: number;
  frames: TrajectoryFrame[];
  outcome: TrajectoryOutcome;
}
