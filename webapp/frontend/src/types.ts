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

export type EvalMode = 'quick' | 'full';

/** Per-opponent eval block carried in a model card's metrics.per_opponent. */
export interface PerOpponentMetric {
  wr: number;
  wins: number;
  losses: number;
  timeouts: number;
  self_out: number;
  mean_ep_len: number;
}

export interface ModelMetrics {
  // Populated by POST /api/models/{id}/evaluate?mode=quick|full. Older cached
  // cards may lack the newer fields — all are optional so the UI degrades.
  mode?: EvalMode;
  win_rate?: number;
  self_outs?: number;
  self_out_rate?: number;
  n_episodes?: number;
  mult?: number;
  opponents?: string[];
  per_opponent?: Record<string, PerOpponentMetric>;
  evaluated_at?: string;
  // Keep the loose index so legacy keys (winrate, wr, …) still read.
  [key: string]: number | string | string[] | Record<string, PerOpponentMetric> | null | undefined;
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
  protected?: boolean; // deployed/canonical model — not deletable via the UI
}

// ---- Training ---------------------------------------------------------------

export type TrainAlgo = 'dqn' | 'ppo';
export type TrainMode = 'scratch' | 'finetune';

/** Editable subset of hyperparameters POSTed in `hyperparams`. All optional so
 *  the backend falls back to its own constants for anything omitted. */
export interface TrainHyperparamOverrides {
  lr?: number;
  gamma?: number;
  net_arch?: [number, number];
  n_step?: number; // DQN
  tau?: number; // DQN
  ent_coef?: number; // PPO
  clip?: number; // PPO
}

export interface StartTrainBody {
  robot_id?: string;
  hardware_spec?: HardwareSpec;
  algo: TrainAlgo;
  mode: TrainMode;
  base_model_id?: string;
  total_steps?: number;
  eval_every?: number;
  start_mult?: number;
  hyperparams?: TrainHyperparamOverrides;
  opponent_weights?: Record<string, number> | null;
  smoke?: boolean;
  seed?: number;
}

/** Full hyperparameter block returned by /api/train/recommend. */
export interface TrainHyperparams {
  lr: number;
  net_arch: [number, number];
  start_mult: number;
  gamma: number;
  n_step: number;
  tau: number;
  ent_coef: number;
  clip: number;
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

// ---- Training opponents ------------------------------------------------------

export interface TrainOpponent {
  id: string;
  default_weight: number;
  held_out: boolean;
}

export interface TrainOpponentsResult {
  opponents: TrainOpponent[];
  weights_normalized: boolean;
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

// ---- Custom opponents (rule-DSL) --------------------------------------------

/** The fixed action vocabulary (maps to bounded wheel commands on the backend). */
export type OpponentAction =
  | 'forward'
  | 'reverse'
  | 'spin_left'
  | 'spin_right'
  | 'arc_left'
  | 'arc_right'
  | 'stop';

/** The fixed predicate vocabulary (pure booleans over the sensor inputs). */
export type OpponentPredicate =
  | 'front_hit'
  | 'left_hit'
  | 'right_hit'
  | 'side_left_hit'
  | 'side_right_hit'
  | 'edge_left'
  | 'edge_right'
  | 'no_target';

/**
 * A condition node. The backend accepts a bare predicate name, a single-key
 * combinator dict (`all`/`any` over a list, `not` over one, `timer {every:N}`),
 * recursively. The builder UI emits a pragmatic subset of this tree.
 */
export type OpponentCond =
  | OpponentPredicate
  | { all: OpponentCond[] }
  | { any: OpponentCond[] }
  | { not: OpponentCond }
  | { timer: { every: number } };

export interface OpponentRule {
  when: OpponentCond;
  do: OpponentAction;
}

export interface OpponentDsl {
  rules: OpponentRule[];
  default: OpponentAction;
}

/** Summary row from GET /api/opponents. `n_rules` is derived client-side. */
export interface CustomOpponentSummary {
  id: string;
  name: string;
  created_at: string;
}

/** Full record from GET /api/opponents/{id} (and POST /api/opponents). */
export interface CustomOpponent extends CustomOpponentSummary {
  hardware_spec: HardwareSpec;
  behavior_dsl: OpponentDsl;
  notes?: string;
}

/** Body for POST /api/opponents. */
export interface CreateOpponentBody {
  name: string;
  hardware_spec: HardwareSpec;
  behavior_dsl: OpponentDsl;
}

/** Result of POST /api/opponents/validate. */
export interface OpponentValidateResult {
  ok: boolean;
  errors: string[];
}

// ---- Arena battles ----------------------------------------------------------

/** Body for POST /api/battle. Exactly one of `b_model_id` / `b_opponent_id`. */
export interface BattleBody {
  a_model_id: string;
  b_model_id?: string;
  b_opponent_id?: string;
  rounds?: number;
  mult?: number;
  seed?: number;
  a_spec?: HardwareSpec;
  b_spec?: HardwareSpec;
}

/** Aggregated win/loss stats across the battle's rounds. */
export interface BattleStats {
  rounds: number;
  a_wins: number;
  b_wins: number;
  draws: number;
  timeouts: number;
  a_self_out: number;
  b_self_out: number;
}

export interface BattleResult {
  battle_id: string;
  stats: BattleStats;
  trajectory: Trajectory;
  notes?: string;
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
