// Tiny typed client for the LITE FastAPI backend. All requests go through the
// Vite dev proxy (/api -> http://127.0.0.1:8000), so paths are relative.

import type {
  EvalMode,
  Geometry,
  HardwareSpec,
  ModelCard,
  RecommendResult,
  RobotRecord,
  RobotSummary,
  StartTrainBody,
  TrainJobSummary,
  TrainOpponentsResult,
  TrainMode,
  TrainStatus,
  Trajectory,
  ValidateResult,
} from './types';

export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(status: number, detail: unknown, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) {
    let detail: unknown = res.statusText;
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      detail = body?.detail ?? body;
      if (typeof detail === 'string') message = detail;
    } catch {
      // non-JSON error body — keep the status text
    }
    throw new ApiError(res.status, detail, message);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => request<{ status: string }>('/api/health'),

  hardwareDefault: () => request<HardwareSpec>('/api/hardware/default'),

  validate: (spec: HardwareSpec) =>
    request<ValidateResult>('/api/hardware/validate', {
      method: 'POST',
      body: JSON.stringify(spec),
    }),

  geometry: (spec: HardwareSpec) =>
    request<Geometry>('/api/hardware/geometry', {
      method: 'POST',
      body: JSON.stringify(spec),
    }),

  models: () => request<ModelCard[]>('/api/models'),

  model: (id: string) => request<ModelCard>(`/api/models/${id}`),

  finetuneCandidates: (id: string) =>
    request<ModelCard[]>(`/api/models/${id}/finetune-candidates`),

  // Slow: runs a real PyBullet eval on the backend and returns the card with
  // metrics populated. Triggered explicitly from the UI, never on list load.
  // mode=quick (default) = 3-opponent probe; mode=full = whole zoo + held-out.
  evaluate: (id: string, mode: EvalMode = 'quick') =>
    request<ModelCard>(`/api/models/${id}/evaluate?mode=${mode}`, {
      method: 'POST',
    }),

  // Delete a checkpoint + its cached card. 409 for a protected/deployed model.
  deleteModel: (id: string) =>
    request<{ deleted: boolean; id: string }>(`/api/models/${id}`, {
      method: 'DELETE',
    }),

  // ---- Saved robots --------------------------------------------------------
  saveRobot: (name: string, spec: HardwareSpec) =>
    request<RobotRecord>('/api/robots', {
      method: 'POST',
      body: JSON.stringify({ name, hardware_spec: spec }),
    }),

  listRobots: () => request<RobotSummary[]>('/api/robots'),

  getRobot: (id: string) => request<RobotRecord>(`/api/robots/${id}`),

  // URDF endpoint returns text/plain, so bypass the JSON request helper.
  getRobotUrdf: async (id: string): Promise<string> => {
    const res = await fetch(`/api/robots/${id}/urdf`);
    if (!res.ok) {
      let message = `${res.status} ${res.statusText}`;
      try {
        const body = await res.json();
        if (typeof body?.detail === 'string') message = body.detail;
      } catch {
        // non-JSON error body — keep the status text
      }
      throw new ApiError(res.status, message, message);
    }
    return res.text();
  },

  deleteRobot: (id: string) =>
    request<{ deleted: boolean }>(`/api/robots/${id}`, { method: 'DELETE' }),

  // ---- Training ------------------------------------------------------------
  // Start a job. Throws ApiError(409) if one is already running.
  startTrain: (body: StartTrainBody) =>
    request<{ job_id: string }>('/api/train', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  // Status of the active job (no id) or a specific job by id.
  trainStatus: (jobId?: string) =>
    request<TrainStatus>(
      jobId ? `/api/train/status/${jobId}` : '/api/train/status',
    ),

  stopTrain: () =>
    request<{ stopped: boolean }>('/api/train/stop', { method: 'POST' }),

  trainJobs: () => request<TrainJobSummary[]>('/api/train/jobs'),

  recommendTrain: (hardware_spec: HardwareSpec | null, mode: TrainMode) =>
    request<RecommendResult>('/api/train/recommend', {
      method: 'POST',
      body: JSON.stringify({ hardware_spec, mode }),
    }),

  trainOpponents: () =>
    request<TrainOpponentsResult>('/api/train/opponents'),

  // Trajectory JSON for a job's checkpoint step (kinematics only).
  getTrajectory: (job: string, step: number | string) =>
    request<Trajectory>(`/api/trajectories/${job}/${step}`),
};
