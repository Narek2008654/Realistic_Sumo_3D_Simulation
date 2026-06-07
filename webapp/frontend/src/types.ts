// TypeScript mirror of webapp/shared/hardware_spec.py (HardwareSpec.to_dict()).
// Tuples on the Python side serialise to JSON arrays; we type them as fixed
// or variable-length number arrays accordingly.

export type Vec3 = [number, number, number];

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
  length_m: number;
  width_m: number;
  height_m: number;
  mass_kg: number;
  com_xyz: Vec3;
  chassis_friction: number;
  wheel_friction: number;
  wedge_present: boolean;
  wedge_length_m: number;
  wedge_pitch_rad: number;
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
