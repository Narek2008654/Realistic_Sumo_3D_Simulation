// TRAIN — configure a run, start/stop it, watch live metrics, replay the
// latest checkpoint's battle in 3D.
//
// Two modes in one page:
//  • SETUP (idle): pick a source (saved robot or current/default spec), algo
//    (DQN/PPO) + mode (scratch/finetune). For finetune, pick a compatible base
//    model (filtered by the source's obs signature). Hyperparameters are
//    pre-filled from POST /api/train/recommend (editable; ETA shown prominently
//    with a plain-language CPU note). TRAIN POSTs and transitions to running.
//  • RUNNING / DONE dashboard: polls GET /api/train/status every ~2.5s (stops on
//    done/error/stopped). Shows state + step progress + ETA, a STOP button,
//    inline SVG charts of win-rate + self-out across checkpoints, a checkpoint
//    timeline, per-opponent win-rates, and the latest checkpoint's trajectory in
//    the TrajectoryPlayer.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { api, ApiError } from '../api';
import { Panel, Reveal, StatusPill } from '../components/ui';
import { TrajectoryPlayer } from '../components/TrajectoryPlayer';
import type {
  CheckpointEvent,
  HardwareSpec,
  ModelCard,
  RecommendResult,
  RobotSummary,
  StartTrainBody,
  TrainAlgo,
  TrainMode,
  TrainStatus,
  Trajectory,
} from '../types';

const POLL_MS = 2500;
const TERMINAL = new Set(['done', 'error', 'stopped', 'unknown']);

// ---------------------------------------------------------------------------
// Small primitives
// ---------------------------------------------------------------------------
function Seg<T extends string>({
  options,
  value,
  onChange,
}: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div className="flex overflow-hidden rounded border" style={{ borderColor: 'var(--line)' }}>
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={o.value}
            type="button"
            onClick={() => onChange(o.value)}
            className="font-display uppercase"
            style={{
              flex: 1,
              fontSize: 12,
              letterSpacing: '.06em',
              padding: '7px 10px',
              border: 'none',
              cursor: 'pointer',
              color: active ? 'var(--bg-0)' : 'var(--fg-1)',
              background: active ? 'var(--accent)' : 'var(--bg-2)',
            }}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

function HpField({
  label,
  value,
  onChange,
  step = 1,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="micro text-fg-2" style={{ fontSize: 9 }}>
        {label}
      </span>
      <input
        type="number"
        className="ctl num"
        style={{ height: 30, fontSize: 12 }}
        value={Number.isFinite(value) ? value : ''}
        step={step}
        onChange={(e) => {
          const v = parseFloat(e.target.value);
          if (Number.isFinite(v)) onChange(v);
        }}
      />
    </label>
  );
}

// ---------------------------------------------------------------------------
// Inline SVG line chart (win-rate + self-out across checkpoints)
// ---------------------------------------------------------------------------
function MetricChart({ checkpoints }: { checkpoints: CheckpointEvent[] }) {
  const W = 520;
  const H = 180;
  const PAD = { l: 36, r: 12, t: 14, b: 24 };
  const iw = W - PAD.l - PAD.r;
  const ih = H - PAD.t - PAD.b;

  const pts = checkpoints.map((c) => {
    const wr = c.eval.wr ?? 0;
    const so = c.eval.n ? c.eval.self_out / c.eval.n : 0;
    return { step: c.step, wr: wr > 1 ? wr / 100 : wr, so };
  });

  if (pts.length === 0) {
    return (
      <div className="flex h-[180px] items-center justify-center">
        <span className="micro text-fg-2">NO CHECKPOINTS YET</span>
      </div>
    );
  }

  const steps = pts.map((p) => p.step);
  const minStep = Math.min(...steps);
  const maxStep = Math.max(...steps);
  const spanStep = Math.max(1, maxStep - minStep);

  const x = (step: number) =>
    PAD.l + (pts.length === 1 ? iw / 2 : ((step - minStep) / spanStep) * iw);
  const y = (v: number) => PAD.t + (1 - Math.max(0, Math.min(1, v))) * ih;

  const line = (key: 'wr' | 'so') =>
    pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${x(p.step)},${y(p[key])}`).join(' ');

  const grid = [0, 0.25, 0.5, 0.75, 1];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 200 }}>
      {grid.map((g) => (
        <g key={g}>
          <line
            x1={PAD.l}
            x2={W - PAD.r}
            y1={y(g)}
            y2={y(g)}
            stroke="var(--line)"
            strokeWidth={0.5}
          />
          <text
            x={PAD.l - 6}
            y={y(g) + 3}
            textAnchor="end"
            fill="var(--fg-2)"
            fontSize={8}
            fontFamily="'IBM Plex Mono', monospace"
          >
            {(g * 100).toFixed(0)}
          </text>
        </g>
      ))}

      <path d={line('so')} fill="none" stroke="var(--loss)" strokeWidth={1.6} opacity={0.85} />
      <path d={line('wr')} fill="none" stroke="var(--accent)" strokeWidth={2} />

      {pts.map((p) => (
        <g key={`wr-${p.step}`}>
          <circle cx={x(p.step)} cy={y(p.wr)} r={2.6} fill="var(--accent)" />
          <circle cx={x(p.step)} cy={y(p.so)} r={2.2} fill="var(--loss)" />
        </g>
      ))}

      {/* x labels: first + last step */}
      <text
        x={PAD.l}
        y={H - 8}
        fill="var(--fg-2)"
        fontSize={8}
        fontFamily="'IBM Plex Mono', monospace"
      >
        {minStep.toLocaleString()}
      </text>
      <text
        x={W - PAD.r}
        y={H - 8}
        textAnchor="end"
        fill="var(--fg-2)"
        fontSize={8}
        fontFamily="'IBM Plex Mono', monospace"
      >
        {maxStep.toLocaleString()}
      </text>
    </svg>
  );
}

function MiniBar({ frac, color }: { frac: number; color: string }) {
  const pct = Math.max(0, Math.min(1, frac));
  return (
    <div className="h-1.5 w-full overflow-hidden rounded" style={{ background: 'var(--bg-3)' }}>
      <div className="h-full rounded" style={{ width: `${pct * 100}%`, background: color }} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Setup panel
// ---------------------------------------------------------------------------
type SourceKind = 'current' | 'robot';

function Setup({ onStarted }: { onStarted: (jobId: string) => void }) {
  const [robots, setRobots] = useState<RobotSummary[]>([]);
  const [sourceKind, setSourceKind] = useState<SourceKind>('current');
  const [robotId, setRobotId] = useState<string>('');

  const [algo, setAlgo] = useState<TrainAlgo>('dqn');
  const [mode, setMode] = useState<TrainMode>('scratch');
  const [smoke, setSmoke] = useState(false);

  const [defaultSpec, setDefaultSpec] = useState<HardwareSpec | null>(null);
  const [robotSpec, setRobotSpec] = useState<HardwareSpec | null>(null);
  const [sourceSig, setSourceSig] = useState<string | null>(null);

  const [models, setModels] = useState<ModelCard[]>([]);
  const [baseModelId, setBaseModelId] = useState<string>('');

  const [rec, setRec] = useState<RecommendResult | null>(null);
  const [totalSteps, setTotalSteps] = useState(0);
  const [evalEvery, setEvalEvery] = useState(0);

  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load robots + models + the default spec once.
  useEffect(() => {
    api.listRobots().then(setRobots).catch(() => {});
    api.models().then(setModels).catch(() => {});
    api.hardwareDefault().then(setDefaultSpec).catch(() => {});
  }, []);

  // The active source spec (current/default vs the picked robot).
  const sourceSpec: HardwareSpec | null =
    sourceKind === 'robot' ? robotSpec : defaultSpec;

  // When a robot is chosen, fetch its full record (spec + signature).
  useEffect(() => {
    if (sourceKind !== 'robot' || !robotId) {
      setRobotSpec(null);
      return;
    }
    api
      .getRobot(robotId)
      .then((r) => {
        setRobotSpec(r.hardware_spec);
        setSourceSig(r.obs_signature_hash);
      })
      .catch(() => setRobotSpec(null));
  }, [sourceKind, robotId]);

  // For the current/default source, resolve its signature via validate (so we
  // can filter finetune candidates). Cheap; runs only when needed.
  useEffect(() => {
    if (sourceKind === 'current' && defaultSpec) {
      api
        .validate(defaultSpec)
        .then((v) => setSourceSig(v.obs_signature_hash))
        .catch(() => setSourceSig(null));
    }
  }, [sourceKind, defaultSpec]);

  // Pull recommended hyperparameters whenever source spec or mode changes.
  useEffect(() => {
    api
      .recommendTrain(sourceSpec, mode)
      .then((r) => {
        setRec(r);
        setAlgo(r.algo);
        setTotalSteps(r.total_steps);
        setEvalEvery(r.eval_every);
      })
      .catch(() => {});
  }, [sourceSpec, mode]);

  // Compatible base models for finetune: same obs signature as the source.
  const compatibleModels = useMemo(
    () =>
      sourceSig
        ? models.filter((m) => m.obs_signature_hash === sourceSig)
        : models,
    [models, sourceSig],
  );

  useEffect(() => {
    // Reset base pick if it's no longer compatible.
    if (baseModelId && !compatibleModels.some((m) => m.id === baseModelId)) {
      setBaseModelId('');
    }
  }, [compatibleModels, baseModelId]);

  const needsBase = mode === 'finetune';
  const baseMissing = needsBase && !baseModelId;
  const robotMissing = sourceKind === 'robot' && !robotId;

  const etaMinutes = useMemo(() => {
    if (!rec) return null;
    if (rec.total_steps <= 0) return rec.est_minutes;
    // Scale the recommender's ETA by the edited step budget.
    return (rec.est_minutes * totalSteps) / rec.total_steps;
  }, [rec, totalSteps]);

  function fmtEta(min: number | null): string {
    if (min == null) return '—';
    if (smoke) return '< 1 min (SMOKE)';
    if (min < 1) return '< 1 min';
    if (min < 90) return `~${min.toFixed(0)} min`;
    return `~${(min / 60).toFixed(1)} hr`;
  }

  async function start() {
    setStarting(true);
    setError(null);
    const body: StartTrainBody = {
      algo,
      mode,
      total_steps: totalSteps || undefined,
      eval_every: evalEvery || undefined,
      smoke: smoke || undefined,
    };
    if (sourceKind === 'robot') body.robot_id = robotId;
    else if (sourceSpec) body.hardware_spec = sourceSpec;
    if (mode === 'finetune') body.base_model_id = baseModelId;

    try {
      const { job_id } = await api.startTrain(body);
      onStarted(job_id);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        setError('A job is already running. Open the dashboard to monitor it.');
      } else {
        setError(e instanceof ApiError ? e.message : 'Failed to start training.');
      }
      setStarting(false);
    }
  }

  return (
    <div className="grid grid-cols-1 gap-5 xl:grid-cols-[minmax(0,1fr)_420px]">
      <Reveal index={0}>
        <Panel title="Run Configuration" live ticks bodyClassName="flex flex-col gap-5 p-5">
          {/* Source */}
          <div className="flex flex-col gap-2">
            <span className="micro text-fg-2" style={{ fontSize: 10 }}>
              SOURCE HARDWARE
            </span>
            <Seg
              value={sourceKind}
              onChange={setSourceKind}
              options={[
                { value: 'current', label: 'Current Spec' },
                { value: 'robot', label: 'Saved Robot' },
              ]}
            />
            {sourceKind === 'robot' && (
              <select
                className="ctl num"
                value={robotId}
                onChange={(e) => setRobotId(e.target.value)}
                style={{ fontSize: 12 }}
              >
                <option value="">— select a saved robot —</option>
                {robots.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.name} · {r.obs_dim}obs/{r.action_dim}act
                  </option>
                ))}
              </select>
            )}
            {sourceKind === 'current' && (
              <span className="num text-fg-2" style={{ fontSize: 11 }}>
                Trains the canonical builder spec
                {defaultSpec ? ` (${defaultSpec.name})` : ''}.
              </span>
            )}
          </div>

          {/* Algo + Mode */}
          <div className="grid grid-cols-2 gap-4">
            <div className="flex flex-col gap-2">
              <span className="micro text-fg-2" style={{ fontSize: 10 }}>
                ALGORITHM
              </span>
              <Seg
                value={algo}
                onChange={setAlgo}
                options={[
                  { value: 'dqn', label: 'DQN' },
                  { value: 'ppo', label: 'PPO' },
                ]}
              />
            </div>
            <div className="flex flex-col gap-2">
              <span className="micro text-fg-2" style={{ fontSize: 10 }}>
                MODE
              </span>
              <Seg
                value={mode}
                onChange={setMode}
                options={[
                  { value: 'scratch', label: 'Scratch' },
                  { value: 'finetune', label: 'Finetune' },
                ]}
              />
            </div>
          </div>

          {/* Finetune base picker */}
          {needsBase && (
            <div className="flex flex-col gap-2">
              <span className="micro text-fg-2" style={{ fontSize: 10 }}>
                BASE MODEL · COMPATIBLE WITH SOURCE
              </span>
              {compatibleModels.length === 0 ? (
                <span className="num" style={{ fontSize: 11, color: 'var(--warn)' }}>
                  No models match this source's observation contract. Train from
                  scratch, or pick a different source.
                </span>
              ) : (
                <select
                  className="ctl num"
                  value={baseModelId}
                  onChange={(e) => setBaseModelId(e.target.value)}
                  style={{ fontSize: 12 }}
                >
                  <option value="">— select a base model —</option>
                  {compatibleModels.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.id} · {m.algo} · {m.net_arch[0]}×{m.net_arch[1]}
                    </option>
                  ))}
                </select>
              )}
            </div>
          )}

          {/* Hyperparameters */}
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="micro text-fg-2" style={{ fontSize: 10 }}>
                HYPERPARAMETERS · RECOMMENDED
              </span>
              {rec && (
                <button
                  type="button"
                  className="micro"
                  onClick={() => {
                    setTotalSteps(rec.total_steps);
                    setEvalEvery(rec.eval_every);
                  }}
                  style={{
                    fontSize: 9,
                    color: 'var(--fg-2)',
                    background: 'transparent',
                    border: 'none',
                    cursor: 'pointer',
                  }}
                  title="Reset to recommended"
                >
                  ⟲ RESET
                </button>
              )}
            </div>
            <div className="grid grid-cols-2 gap-3">
              <HpField
                label="TOTAL STEPS"
                value={totalSteps}
                onChange={setTotalSteps}
                step={10000}
              />
              <HpField
                label="EVAL EVERY"
                value={evalEvery}
                onChange={setEvalEvery}
                step={10000}
              />
            </div>
            {rec && (
              <div className="grid grid-cols-3 gap-2 pt-1">
                <RecChip label="NET" value={`${rec.net_arch[0]}×${rec.net_arch[1]}`} />
                <RecChip label="START MULT" value={rec.start_mult.toFixed(2)} />
                <RecChip label="LR" value={rec.hyperparams.lr.toExponential(0)} />
              </div>
            )}
          </div>

          {/* Smoke toggle */}
          <label className="flex cursor-pointer items-center gap-2">
            <input
              type="checkbox"
              checked={smoke}
              onChange={(e) => setSmoke(e.target.checked)}
              style={{ accentColor: 'var(--accent)' }}
            />
            <span className="num" style={{ fontSize: 11, color: 'var(--fg-1)' }}>
              SMOKE TEST — tiny run to validate the pipeline end-to-end
            </span>
          </label>

          {error && (
            <div className="num" style={{ fontSize: 11, color: 'var(--loss)' }}>
              {error}
            </div>
          )}

          <button
            className="btn btn-primary"
            disabled={starting || robotMissing || baseMissing}
            onClick={start}
            style={{ height: 38, fontSize: 14 }}
          >
            {starting ? 'LAUNCHING…' : 'TRAIN'}
          </button>
        </Panel>
      </Reveal>

      {/* ETA / brief */}
      <Reveal index={1}>
        <Panel title="Estimated Cost" live ticks bodyClassName="flex flex-col gap-4 p-5">
          <div>
            <span className="micro text-fg-2" style={{ fontSize: 10 }}>
              WALL-CLOCK ETA
            </span>
            <div
              className="num"
              style={{ fontSize: 40, color: 'var(--accent)', lineHeight: 1.1 }}
            >
              {fmtEta(etaMinutes)}
            </div>
          </div>
          <p className="num text-fg-1" style={{ fontSize: 12, lineHeight: 1.5 }}>
            Training runs locally and{' '}
            <span style={{ color: 'var(--warn)' }}>pins the CPU</span> for the
            full duration — expect fans to spin up and other work to slow down
            while it runs. You can STOP it at any time from the dashboard.
          </p>
          <div
            className="h-px w-full"
            style={{ background: 'linear-gradient(90deg, var(--accent-dim), transparent)' }}
          />
          <span className="micro text-fg-2" style={{ fontSize: 9 }}>
            Estimate scales with the step budget at ~250 env-steps/s. Smoke runs
            finish in seconds.
          </span>
        </Panel>
      </Reveal>
    </div>
  );
}

function RecChip({ label, value }: { label: string; value: string }) {
  return (
    <div
      className="flex flex-col rounded border px-2 py-1.5"
      style={{ borderColor: 'var(--line)', background: 'var(--bg-2)' }}
    >
      <span className="micro text-fg-2" style={{ fontSize: 8 }}>
        {label}
      </span>
      <span className="num" style={{ fontSize: 12, color: 'var(--cyan)' }}>
        {value}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Running / done dashboard
// ---------------------------------------------------------------------------
function Dashboard({
  jobId,
  onReset,
}: {
  jobId: string;
  onReset: () => void;
}) {
  const [status, setStatus] = useState<TrainStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stopping, setStopping] = useState(false);
  const [traj, setTraj] = useState<Trajectory | null>(null);
  const loadedStepRef = useRef<number | null>(null);

  const poll = useCallback(async () => {
    try {
      const s = await api.trainStatus(jobId);
      setStatus(s);
      return s.state;
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'status poll failed');
      return null;
    }
  }, [jobId]);

  // Poll loop: every POLL_MS, stop once terminal.
  useEffect(() => {
    let active = true;
    let timer: number | undefined;
    const tick = async () => {
      const state = await poll();
      if (!active) return;
      if (state && !TERMINAL.has(state)) {
        timer = window.setTimeout(tick, POLL_MS);
      }
    };
    tick();
    return () => {
      active = false;
      if (timer) window.clearTimeout(timer);
    };
  }, [poll]);

  const checkpoints = useMemo(
    () =>
      (status?.events.filter((e) => e.t === 'checkpoint') ??
        []) as unknown as CheckpointEvent[],
    [status],
  );
  const latest = status?.latest_checkpoint ?? null;

  // Load the latest checkpoint's trajectory when a new one arrives.
  useEffect(() => {
    if (!latest) return;
    if (loadedStepRef.current === latest.step) return;
    loadedStepRef.current = latest.step;
    api
      .getTrajectory(jobId, latest.step)
      .then(setTraj)
      .catch(() => {});
  }, [latest, jobId]);

  async function stop() {
    setStopping(true);
    try {
      await api.stopTrain();
      await poll();
    } catch {
      /* ignore — next poll reflects state */
    } finally {
      setStopping(false);
    }
  }

  const cfg = status?.config as
    | { total_steps?: number; algo?: string; mode?: string }
    | null
    | undefined;
  const totalSteps = (cfg?.total_steps as number) ?? 0;
  const curStep = latest?.step ?? 0;
  const progress = totalSteps ? Math.min(1, curStep / totalSteps) : 0;

  const state = status?.state ?? 'unknown';
  const running = state === 'running';
  const done = state === 'done';
  const terminal = TERMINAL.has(state);

  const bestWr = useMemo(() => {
    let best = 0;
    for (const c of checkpoints) {
      const wr = c.eval.wr > 1 ? c.eval.wr / 100 : c.eval.wr;
      if (wr > best) best = wr;
    }
    return best;
  }, [checkpoints]);

  const pill = running
    ? { status: 'accent', label: 'TRAINING', pulse: true }
    : done
      ? { status: 'win', label: 'COMPLETE', pulse: false }
      : state === 'stopped'
        ? { status: 'idle', label: 'STOPPED', pulse: false }
        : state === 'error'
          ? { status: 'loss', label: 'ERROR', pulse: false }
          : { status: 'idle', label: state.toUpperCase(), pulse: false };

  // Per-opponent win rates from the latest checkpoint.
  const perOpp = latest?.eval.per_opponent ?? null;

  return (
    <div className="flex flex-col gap-5">
      {/* Header / control strip */}
      <Reveal index={0}>
        <Panel live bodyClassName="flex flex-wrap items-center justify-between gap-4 p-4">
          <div className="flex items-center gap-4">
            <StatusPill status={pill.status} label={pill.label} pulse={pill.pulse} />
            <span className="num text-fg-2" style={{ fontSize: 11 }}>
              job {jobId} · {cfg?.algo?.toUpperCase() ?? '—'} · {cfg?.mode ?? '—'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            {running ? (
              <button
                className="btn btn-secondary"
                onClick={stop}
                disabled={stopping}
                style={{ color: 'var(--loss)', borderColor: 'var(--loss)' }}
              >
                {stopping ? 'STOPPING…' : '■ STOP'}
              </button>
            ) : (
              <button className="btn btn-secondary" onClick={onReset}>
                ← NEW RUN
              </button>
            )}
          </div>
        </Panel>
      </Reveal>

      {/* Progress + key stats */}
      <Reveal index={1}>
        <Panel title="Run Progress" live bodyClassName="flex flex-col gap-3 p-4">
          <div className="flex items-end justify-between">
            <span className="num" style={{ fontSize: 13 }}>
              {curStep.toLocaleString()}
              <span className="text-fg-2"> / {totalSteps.toLocaleString()} steps</span>
            </span>
            <span className="num text-fg-2" style={{ fontSize: 11 }}>
              {(progress * 100).toFixed(0)}%
            </span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded" style={{ background: 'var(--bg-3)' }}>
            <div
              className="h-full rounded"
              style={{
                width: `${progress * 100}%`,
                background: 'var(--accent)',
                boxShadow: '0 0 12px var(--accent-glow)',
                transition: 'width .4s ease',
              }}
            />
          </div>
          <div className="grid grid-cols-3 gap-2 pt-1">
            <Stat label="CHECKPOINTS" value={String(checkpoints.length)} tone="cyan" />
            <Stat
              label="LATEST WR"
              value={latest ? `${((latest.eval.wr > 1 ? latest.eval.wr / 100 : latest.eval.wr) * 100).toFixed(0)}%` : '—'}
              tone="accent"
            />
            <Stat label="BEST WR" value={checkpoints.length ? `${(bestWr * 100).toFixed(0)}%` : '—'} tone="win" />
          </div>
        </Panel>
      </Reveal>

      {error && (
        <div className="num" style={{ fontSize: 11, color: 'var(--loss)' }}>
          {error}
        </div>
      )}

      {/* Terminal banner */}
      {terminal && (
        <Reveal index={2}>
          <Panel live ticks bodyClassName="flex flex-col gap-3 p-5">
            <span className="micro" style={{ color: pill.status === 'win' ? 'var(--win)' : 'var(--loss)' }}>
              ◇ {done ? 'TRAINING COMPLETE' : state === 'stopped' ? 'TRAINING STOPPED' : 'TRAINING ENDED'}
            </span>
            <div className="num" style={{ fontSize: 13, color: 'var(--fg-1)' }}>
              {checkpoints.length} checkpoints · best win-rate{' '}
              <span style={{ color: 'var(--accent)' }}>{(bestWr * 100).toFixed(0)}%</span>
            </div>
            <Link
              to="/models"
              className="btn btn-secondary"
              style={{ alignSelf: 'flex-start', textDecoration: 'none' }}
            >
              VIEW MODELS →
            </Link>
          </Panel>
        </Reveal>
      )}

      {/* Charts + replay */}
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
        <Reveal index={3}>
          <Panel title="Win-Rate / Self-Out · per checkpoint" live ticks bodyClassName="p-4">
            <MetricChart checkpoints={checkpoints} />
            <div className="mt-2 flex gap-4">
              <span className="micro num" style={{ fontSize: 9, color: 'var(--accent)' }}>
                ── WIN-RATE
              </span>
              <span className="micro num" style={{ fontSize: 9, color: 'var(--loss)' }}>
                ── SELF-OUT
              </span>
            </div>
          </Panel>
        </Reveal>

        <Reveal index={4} className="min-h-[420px]">
          <TrajectoryPlayer
            traj={traj}
            label={latest ? `STEP ${latest.step.toLocaleString()}` : undefined}
          />
        </Reveal>
      </div>

      {/* Per-opponent + timeline */}
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
        {perOpp && Object.keys(perOpp).length > 0 && (
          <Reveal index={5}>
            <Panel title="Per-Opponent Win-Rate · latest" live bodyClassName="flex flex-col gap-2 p-4">
              {Object.entries(perOpp).map(([opp, r]) => {
                const wr = r.n ? r.wins / r.n : 0;
                return (
                  <div key={opp} className="flex items-center gap-3">
                    <span className="num text-fg-1" style={{ fontSize: 11, minWidth: 88 }}>
                      {opp}
                    </span>
                    <MiniBar frac={wr} color="var(--cyan)" />
                    <span className="num" style={{ fontSize: 11, color: 'var(--cyan)', minWidth: 34, textAlign: 'right' }}>
                      {(wr * 100).toFixed(0)}%
                    </span>
                  </div>
                );
              })}
            </Panel>
          </Reveal>
        )}

        <Reveal index={6}>
          <Panel title="Checkpoint Timeline" live bodyClassName="p-0">
            <div className="max-h-[260px] overflow-y-auto">
              {checkpoints.length === 0 ? (
                <div className="p-4">
                  <span className="micro text-fg-2">AWAITING FIRST CHECKPOINT…</span>
                </div>
              ) : (
                <table className="w-full border-collapse">
                  <thead>
                    <tr>
                      {['STEP', 'WR', 'SELF-OUT', 'EP LEN'].map((h) => (
                        <th
                          key={h}
                          className="micro px-3 py-2 text-left text-fg-2"
                          style={{ fontSize: 9, borderBottom: '1px solid var(--line)' }}
                        >
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {[...checkpoints].reverse().map((c, i) => {
                      const wr = c.eval.wr > 1 ? c.eval.wr / 100 : c.eval.wr;
                      const so = c.eval.n ? c.eval.self_out / c.eval.n : 0;
                      return (
                        <tr key={c.step} style={{ background: i % 2 ? 'var(--bg-2)' : 'transparent' }}>
                          <td className="num px-3 py-1.5" style={{ fontSize: 11 }}>
                            {c.step.toLocaleString()}
                          </td>
                          <td className="num px-3 py-1.5" style={{ fontSize: 11, color: 'var(--accent)' }}>
                            {(wr * 100).toFixed(0)}%
                          </td>
                          <td className="num px-3 py-1.5" style={{ fontSize: 11, color: 'var(--loss)' }}>
                            {(so * 100).toFixed(0)}%
                          </td>
                          <td className="num px-3 py-1.5 text-fg-1" style={{ fontSize: 11 }}>
                            {c.eval.mean_ep_len.toFixed(0)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              )}
            </div>
          </Panel>
        </Reveal>
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone: 'cyan' | 'accent' | 'win';
}) {
  const color =
    tone === 'cyan' ? 'var(--cyan)' : tone === 'accent' ? 'var(--accent)' : 'var(--win)';
  return (
    <div
      className="flex flex-col rounded border px-2 py-1.5"
      style={{ borderColor: 'var(--line)', background: 'var(--bg-2)' }}
    >
      <span className="micro text-fg-2" style={{ fontSize: 9 }}>
        {label}
      </span>
      <span className="num" style={{ fontSize: 16, color }}>
        {value}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------
export default function Train() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [resolved, setResolved] = useState(false);

  // On mount, adopt an already-running job so a refresh lands on the dashboard.
  useEffect(() => {
    api
      .trainStatus()
      .then((s) => {
        if (s.job_id && s.state !== 'idle') setJobId(s.job_id);
      })
      .catch(() => {})
      .finally(() => setResolved(true));
  }, []);

  if (!resolved) {
    return (
      <Panel title="Training" live>
        <span className="micro animate-pulse" style={{ color: 'var(--cyan)' }}>
          PROBING TRAINER…
        </span>
      </Panel>
    );
  }

  if (jobId) {
    return <Dashboard jobId={jobId} onReset={() => setJobId(null)} />;
  }
  return <Setup onStarted={setJobId} />;
}
