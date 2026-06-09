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
import type { Dispatch, ReactNode, SetStateAction } from 'react';
import { Link } from 'react-router-dom';
import { api, ApiError } from '../api';
import { Panel, Reveal, StatusPill } from '../components/ui';
import { SliderField } from '../components/fields';
import { TrajectoryPlayer } from '../components/TrajectoryPlayer';
import {
  clearTrainSetup,
  loadTrainSetup,
  saveTrainSetup,
  type OppChoice,
  type SourceKind,
  type TrainSetup,
} from '../store/trainSetup';
import type {
  CheckpointEvent,
  CustomOpponentSummary,
  HardwareSpec,
  LogEvent,
  ModelCard,
  RecommendResult,
  RobotSummary,
  StartTrainBody,
  TrainAlgo,
  TrainHyperparamOverrides,
  TrainMode,
  TrainOpponent,
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

// ---------------------------------------------------------------------------
// Entropy sparkline (PPO only — DQN has no policy entropy)
// ---------------------------------------------------------------------------
function EntropyChart({ logs }: { logs: LogEvent[] }) {
  const W = 520;
  const H = 140;
  const PAD = { l: 36, r: 12, t: 14, b: 24 };
  const iw = W - PAD.l - PAD.r;
  const ih = H - PAD.t - PAD.b;
  // Cap the axis at ~ln(9) ≈ 2.2 (uniform over the 9 discrete actions).
  const TOP = 2.2;

  const pts = logs
    .filter((l) => typeof l.entropy === 'number' && Number.isFinite(l.entropy))
    .map((l) => ({ step: l.step, ent: l.entropy as number }));

  if (pts.length === 0) {
    return (
      <div className="flex h-[140px] items-center justify-center">
        <span className="micro text-fg-2">NO ENTROPY YET (PPO ONLY)</span>
      </div>
    );
  }

  const steps = pts.map((p) => p.step);
  const minStep = Math.min(...steps);
  const maxStep = Math.max(...steps);
  const spanStep = Math.max(1, maxStep - minStep);

  const x = (step: number) =>
    PAD.l + (pts.length === 1 ? iw / 2 : ((step - minStep) / spanStep) * iw);
  const y = (v: number) => PAD.t + (1 - Math.max(0, Math.min(1, v / TOP))) * ih;

  const path = pts
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${x(p.step)},${y(p.ent)}`)
    .join(' ');
  const grid = [0, 0.5, 1, 1.5, 2];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 160 }}>
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
            {g.toFixed(1)}
          </text>
        </g>
      ))}

      <path d={path} fill="none" stroke="var(--cyan)" strokeWidth={1.8} />
      {pts.length <= 80 &&
        pts.map((p) => (
          <circle key={p.step} cx={x(p.step)} cy={y(p.ent)} r={2} fill="var(--cyan)" />
        ))}

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
// Opponent mix — included weights
// ---------------------------------------------------------------------------
// Raw weights of the INCLUDED (checked) opponents — the user's exact values.
// Unchecked opponents are dropped. These are sent as-is; the UI requires their
// sum to equal 1.0 (weightSum / weightsValid) rather than auto-normalizing, so
// the user keeps full control of the mix.
function includedWeights(
  opps: Record<string, OppChoice>,
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [id, c] of Object.entries(opps)) {
    if (c.on) out[id] = Math.max(0, c.weight);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Setup panel
// ---------------------------------------------------------------------------
function IntPairField({
  label,
  value,
  onChange,
  disabled = false,
  note,
}: {
  label: string;
  value: [number, number];
  onChange: (v: [number, number]) => void;
  disabled?: boolean;
  note?: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="micro text-fg-2" style={{ fontSize: 9 }}>
        {label}
        {note && <span style={{ color: 'var(--fg-2)' }}> · {note}</span>}
      </span>
      <div className="flex items-center gap-2">
        {[0, 1].map((i) => (
          <input
            key={i}
            type="number"
            className="ctl num"
            disabled={disabled}
            style={{ height: 30, fontSize: 12, opacity: disabled ? 0.55 : 1 }}
            value={value[i]}
            step={8}
            min={4}
            onChange={(e) => {
              const v = parseInt(e.target.value, 10);
              if (!Number.isFinite(v)) return;
              const next: [number, number] = [...value] as [number, number];
              next[i] = v;
              onChange(next);
            }}
          />
        ))}
      </div>
    </div>
  );
}

// Restore once at module-eval so initial state can seed from localStorage
// without a flash of defaults. Merged field-by-field below (missing fields fall
// back to defaults / fresh recommend).
function Setup({ onStarted }: { onStarted: (jobId: string) => void }) {
  const savedRef = useRef<Partial<TrainSetup> | null>(loadTrainSetup());
  const saved = savedRef.current;

  const [robots, setRobots] = useState<RobotSummary[]>([]);
  const [sourceKind, setSourceKind] = useState<SourceKind>(saved?.sourceKind ?? 'current');
  const [robotId, setRobotId] = useState<string>(saved?.robotId ?? '');

  const [algo, setAlgo] = useState<TrainAlgo>(saved?.algo ?? 'dqn');
  const [mode, setMode] = useState<TrainMode>(saved?.mode ?? 'scratch');
  const [smoke, setSmoke] = useState(saved?.smoke ?? false);
  const [adaptiveOpponents, setAdaptiveOpponents] = useState(
    saved?.adaptiveOpponents ?? false,
  );

  const [defaultSpec, setDefaultSpec] = useState<HardwareSpec | null>(null);
  const [robotSpec, setRobotSpec] = useState<HardwareSpec | null>(null);
  const [sourceSig, setSourceSig] = useState<string | null>(null);

  const [models, setModels] = useState<ModelCard[]>([]);
  const [baseModelId, setBaseModelId] = useState<string>(saved?.baseModelId ?? '');

  const [rec, setRec] = useState<RecommendResult | null>(null);
  const [totalSteps, setTotalSteps] = useState(saved?.totalSteps ?? 0);
  const [evalEvery, setEvalEvery] = useState(saved?.evalEvery ?? 0);

  // Editable hyperparameters (pre-filled from recommend; restored from save).
  const [startMult, setStartMult] = useState(saved?.startMult ?? 1.0);
  const [lr, setLr] = useState(saved?.hyperparams?.lr ?? 3e-4);
  const [gamma, setGamma] = useState(saved?.hyperparams?.gamma ?? 0.99);
  const [nStep, setNStep] = useState(saved?.hyperparams?.n_step ?? 3);
  const [tau, setTau] = useState(saved?.hyperparams?.tau ?? 5e-3);
  const [entCoef, setEntCoef] = useState(saved?.hyperparams?.ent_coef ?? 0.02);
  const [clip, setClip] = useState(saved?.hyperparams?.clip ?? 0.2);
  const [netArch, setNetArch] = useState<[number, number]>(
    saved?.hyperparams?.net_arch ?? [32, 32],
  );
  const [advanced, setAdvanced] = useState(false);

  // Opponent mix: id -> {on, weight}. Seeded from /api/train/opponents (held-out
  // unchecked) once loaded, then merged with any saved choices.
  const [oppList, setOppList] = useState<TrainOpponent[]>([]);
  // User-authored custom opponents (DSL). Listed under a CUSTOM group; default
  // UNCHECKED so the mix is unchanged unless explicitly opted in.
  const [customList, setCustomList] = useState<CustomOpponentSummary[]>([]);
  const [opps, setOpps] = useState<Record<string, OppChoice>>(saved?.opponents ?? {});

  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // True only while the recommend response should overwrite the live HP fields.
  // We DON'T clobber values the user restored from a save; recommend still fills
  // anything the save lacked (handled in applyRecommend).
  const recAppliedRef = useRef(saved != null);

  // Load robots + models + the default spec once.
  useEffect(() => {
    api.listRobots().then(setRobots).catch(() => {});
    api.models().then(setModels).catch(() => {});
    api.hardwareDefault().then(setDefaultSpec).catch(() => {});
  }, []);

  // Load the opponent zoo + seed include/weight. Default-check seen opponents
  // with their default_weight; show held-out (feinter/orbiter) UNCHECKED. Merge
  // saved choices on top so a restored setup wins.
  useEffect(() => {
    Promise.all([
      api.trainOpponents().then(({ opponents }) => opponents).catch(() => []),
      api.listOpponents().catch(() => []),
    ]).then(([opponents, custom]) => {
      setOppList(opponents);
      setCustomList(custom);
      setOpps((prev) => {
        const next: Record<string, OppChoice> = {};
        for (const o of opponents) {
          const savedChoice = saved?.opponents?.[o.id] ?? prev[o.id];
          next[o.id] = savedChoice ?? {
            on: !o.held_out,
            weight: o.default_weight > 0 ? o.default_weight : 1 / opponents.length,
          };
        }
        // Custom opponents: default UNCHECKED with a 0 weight, but keep any
        // saved/restored choice so an opted-in custom mix survives a reload.
        for (const c of custom) {
          const savedChoice = saved?.opponents?.[c.id] ?? prev[c.id];
          next[c.id] = savedChoice ?? { on: false, weight: 0 };
        }
        return next;
      });
    });
    // saved is stable (module-eval ref); run once.
    // eslint-disable-next-line react-hooks/exhaustive-deps
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

  // Force every editable HP field to the recommended values.
  const applyRecommend = useCallback((r: RecommendResult) => {
    const h = r.hyperparams;
    setAlgo(r.algo);
    setTotalSteps(r.total_steps);
    setEvalEvery(r.eval_every);
    setStartMult(r.start_mult);
    setLr(h.lr);
    setGamma(h.gamma);
    setNStep(h.n_step);
    setTau(h.tau);
    setEntCoef(h.ent_coef);
    setClip(h.clip);
    setNetArch(r.net_arch);
  }, []);

  // Pull recommended hyperparameters whenever source spec or mode changes. On
  // the very first run with a restored save we keep the user's edits and only
  // record `rec` (for ghost/reset). Afterwards, changing source/mode re-applies
  // recommended values — except start_mult on finetune is authoritative (3.0).
  useEffect(() => {
    api
      .recommendTrain(sourceSpec, mode)
      .then((r) => {
        setRec(r);
        if (recAppliedRef.current) {
          // First load after a restore: don't clobber the saved edits, but for
          // finetune force start_mult to the backend's authoritative 3.0 and
          // lock net_arch to the recommended pairing.
          recAppliedRef.current = false;
          if (mode === 'finetune') {
            setStartMult(r.start_mult);
            setNetArch(r.net_arch);
          }
          return;
        }
        applyRecommend(r);
      })
      .catch(() => {});
  }, [sourceSpec, mode, applyRecommend]);

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

  // net_arch is locked to the base model on finetune (must match it).
  const netArchLocked = mode === 'finetune';

  // The algo-specific hyperparams that go in the POST body. Omit ones that
  // don't apply to the chosen algorithm so the backend ignores them cleanly.
  const hyperparams = useMemo<TrainHyperparamOverrides>(() => {
    const hp: TrainHyperparamOverrides = { lr, gamma, net_arch: netArch };
    if (algo === 'dqn') {
      hp.n_step = nStep;
      hp.tau = tau;
    } else {
      hp.ent_coef = entCoef;
      hp.clip = clip;
    }
    return hp;
  }, [algo, lr, gamma, netArch, nStep, tau, entCoef, clip]);

  // Raw included weights (the user's exact values), sent as-is. The UI requires
  // their sum to equal 1.00 — no auto-normalize.
  const incWeights = useMemo(() => includedWeights(opps), [opps]);
  const includedCount = useMemo(
    () => Object.values(opps).filter((c) => c.on).length,
    [opps],
  );
  const noOpponents = includedCount === 0;
  const weightSum = useMemo(
    () => Object.values(incWeights).reduce((s, w) => s + w, 0),
    [incWeights],
  );
  const weightsValid = !noOpponents && Math.abs(weightSum - 1) < 0.001;

  // Snapshot the full setup for persistence.
  const setupSnapshot = useCallback(
    (): TrainSetup => ({
      sourceKind,
      robotId,
      algo,
      mode,
      baseModelId,
      totalSteps,
      evalEvery,
      startMult,
      hyperparams: {
        lr,
        gamma,
        net_arch: netArch,
        n_step: nStep,
        tau,
        ent_coef: entCoef,
        clip,
      },
      smoke,
      adaptiveOpponents,
      opponents: opps,
    }),
    [
      sourceKind, robotId, algo, mode, baseModelId, totalSteps, evalEvery,
      startMult, lr, gamma, netArch, nStep, tau, entCoef, clip, smoke,
      adaptiveOpponents, opps,
    ],
  );

  // Persist on change (debounced ~400ms). Restored on next mount.
  useEffect(() => {
    const snap = setupSnapshot();
    const t = window.setTimeout(() => saveTrainSetup(snap), 400);
    return () => window.clearTimeout(t);
  }, [setupSnapshot]);

  function resetSetup() {
    clearTrainSetup();
    setSourceKind('current');
    setRobotId('');
    setMode('scratch');
    setBaseModelId('');
    setSmoke(false);
    if (rec) applyRecommend(rec);
    // Re-seed opponents to backend defaults.
    setOpps(() => {
      const next: Record<string, OppChoice> = {};
      for (const o of oppList) {
        next[o.id] = {
          on: !o.held_out,
          weight: o.default_weight > 0 ? o.default_weight : 1 / oppList.length,
        };
      }
      return next;
    });
  }

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
    saveTrainSetup(setupSnapshot()); // persist on submit too
    const body: StartTrainBody = {
      algo,
      mode,
      total_steps: totalSteps || undefined,
      eval_every: evalEvery || undefined,
      start_mult: startMult,
      hyperparams,
      opponent_weights: Object.keys(incWeights).length ? incWeights : undefined,
      adaptive_opponents: adaptiveOpponents || undefined,
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

          {/* Hyperparameters — editable, pre-filled from recommend */}
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <span className="micro text-fg-2" style={{ fontSize: 10 }}>
                HYPERPARAMETERS
              </span>
              {rec && (
                <button
                  type="button"
                  className="micro"
                  onClick={() => applyRecommend(rec)}
                  style={{
                    fontSize: 9,
                    color: 'var(--fg-2)',
                    background: 'transparent',
                    border: 'none',
                    cursor: 'pointer',
                  }}
                  title="Reset all hyperparameters to the recommended values"
                >
                  ⟲ RESET TO RECOMMENDED
                </button>
              )}
            </div>

            {/* Basic: budget + the two most-tuned knobs */}
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
            <SliderField
              label="START MULT"
              info="curriculum"
              value={startMult}
              onChange={setStartMult}
              min={0.5}
              max={5}
              step={0.05}
              format={(v) => v.toFixed(2)}
            />
            <HpField label="LEARNING RATE" value={lr} onChange={setLr} step={0.00005} />

            {/* Advanced (collapsible) */}
            <button
              type="button"
              className="micro flex items-center gap-1.5"
              onClick={() => setAdvanced((a) => !a)}
              style={{
                fontSize: 9,
                color: 'var(--cyan)',
                background: 'transparent',
                border: 'none',
                cursor: 'pointer',
                alignSelf: 'flex-start',
              }}
            >
              {advanced ? '▾' : '▸'} ADVANCED
            </button>
            {advanced && (
              <div className="flex flex-col gap-3 rounded border p-3"
                style={{ borderColor: 'var(--line)', background: 'var(--bg-2)' }}>
                <SliderField
                  label="GAMMA"
                  value={gamma}
                  onChange={setGamma}
                  min={0.9}
                  max={0.999}
                  step={0.001}
                  format={(v) => v.toFixed(3)}
                />
                {algo === 'dqn' ? (
                  <div className="grid grid-cols-2 gap-3">
                    <HpField label="N-STEP" value={nStep} onChange={(v) => setNStep(Math.round(v))} step={1} />
                    <HpField label="TAU (POLYAK)" value={tau} onChange={setTau} step={0.001} />
                  </div>
                ) : (
                  <div className="grid grid-cols-2 gap-3">
                    <HpField label="ENT COEF" value={entCoef} onChange={setEntCoef} step={0.005} />
                    <HpField label="CLIP" value={clip} onChange={setClip} step={0.01} />
                  </div>
                )}
                <IntPairField
                  label="NET ARCH"
                  value={netArch}
                  onChange={setNetArch}
                  disabled={netArchLocked}
                  note={netArchLocked ? 'matches the base model' : 'two hidden layers'}
                />
              </div>
            )}
          </div>

          {/* Opponent mix editor */}
          <OpponentEditor
            list={oppList}
            customList={customList}
            opps={opps}
            setOpps={setOpps}
            weightSum={weightSum}
            weightsValid={weightsValid}
          />

          {/* Adaptive opponent weighting */}
          <label className="flex cursor-pointer items-start gap-2">
            <input
              type="checkbox"
              checked={adaptiveOpponents}
              onChange={(e) => setAdaptiveOpponents(e.target.checked)}
              style={{ accentColor: 'var(--accent)', marginTop: 2 }}
            />
            <span className="num" style={{ fontSize: 11, color: 'var(--fg-1)' }}>
              ADAPTIVE OPPONENT WEIGHTS — re-weight the mix from each eval&rsquo;s
              per-opponent win-rates, auto-focusing on what the model is losing.
              The weights above seed the start; the built-in zoo keeps a reserved
              share, per-opponent weights are capped, and shifts are smoothed.
            </span>
          </label>

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

          {noOpponents && (
            <div className="num" style={{ fontSize: 11, color: 'var(--warn)' }}>
              Select at least one opponent for the training mix.
            </div>
          )}
          {!noOpponents && !weightsValid && (
            <div className="num" style={{ fontSize: 11, color: 'var(--warn)' }}>
              Opponent weights must sum to 1.00 (currently {weightSum.toFixed(2)}).
              Please adjust them so the sum is 1.
            </div>
          )}
          {error && (
            <div className="num" style={{ fontSize: 11, color: 'var(--loss)' }}>
              {error}
            </div>
          )}

          <div className="flex items-center gap-3">
            <button
              className="btn btn-primary"
              disabled={
                starting || robotMissing || baseMissing || noOpponents || !weightsValid
              }
              onClick={start}
              style={{ height: 38, fontSize: 14, flex: 1 }}
            >
              {starting ? 'LAUNCHING…' : 'TRAIN'}
            </button>
            <button
              type="button"
              className="btn btn-ghost"
              onClick={resetSetup}
              title="Clear the saved setup and restore recommended defaults"
              style={{ height: 38, fontSize: 11 }}
            >
              RESET SETUP
            </button>
          </div>
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

// ---------------------------------------------------------------------------
// Opponent-weights editor
// ---------------------------------------------------------------------------
// Each opponent: a checkbox (include in the mix) + a raw weight input. The
// displayed normalized value (and Σ) come from `normWeights`, recomputed on
// every include/weight change so the included set always sums to 1.00.
function OpponentEditor({
  list,
  customList,
  opps,
  setOpps,
  weightSum,
  weightsValid,
}: {
  list: TrainOpponent[];
  customList: CustomOpponentSummary[];
  opps: Record<string, OppChoice>;
  setOpps: Dispatch<SetStateAction<Record<string, OppChoice>>>;
  weightSum: number;
  weightsValid: boolean;
}) {
  const includedCount = Object.values(opps).filter((c) => c.on).length;

  function toggle(id: string) {
    setOpps((prev) => {
      const cur = prev[id] ?? { on: false, weight: 0 };
      // Guard: never let the last included opponent be unchecked.
      if (cur.on && includedCount <= 1) return prev;
      // When opting a custom opponent in for the first time, give it a small
      // starting weight so it isn't a no-op zero (the user still tunes Σ to 1).
      const turningOn = !cur.on;
      const weight = turningOn && cur.weight <= 0 ? 0.1 : cur.weight;
      return { ...prev, [id]: { on: turningOn, weight } };
    });
  }

  function setWeight(id: string, weight: number) {
    setOpps((prev) => {
      const cur = prev[id] ?? { on: true, weight: 0 };
      return { ...prev, [id]: { ...cur, weight: Math.max(0, weight) } };
    });
  }

  const row = (
    id: string,
    label: ReactNode,
    fallbackWeight: number,
  ) => {
    const choice = opps[id] ?? { on: false, weight: fallbackWeight };
    return (
      <div
        key={id}
        className="flex items-center gap-3 px-3 py-2"
        style={{ borderColor: 'var(--line)', opacity: choice.on ? 1 : 0.55 }}
      >
        <input
          type="checkbox"
          checked={choice.on}
          onChange={() => toggle(id)}
          style={{ accentColor: 'var(--accent)' }}
        />
        <span className="num text-fg-1" style={{ fontSize: 12, minWidth: 84, flex: 1 }}>
          {label}
        </span>
        <input
          type="number"
          className="ctl num"
          value={choice.weight}
          min={0}
          step={0.05}
          disabled={!choice.on}
          onChange={(e) => {
            const v = parseFloat(e.target.value);
            if (Number.isFinite(v)) setWeight(id, v);
          }}
          style={{ width: 80, height: 28, fontSize: 11, textAlign: 'right' }}
        />
      </div>
    );
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="micro text-fg-2" style={{ fontSize: 10 }}>
          OPPONENT MIX
        </span>
        <span
          className="num"
          style={{ fontSize: 10, color: weightsValid ? 'var(--win)' : 'var(--warn)' }}
        >
          Σ = {weightSum.toFixed(2)} · {includedCount} active
        </span>
      </div>
      <span className="num text-fg-2" style={{ fontSize: 10 }}>
        Check opponents to include and set each weight — they must sum to 1.00.
      </span>
      <div
        className="flex flex-col divide-y rounded border"
        style={{ borderColor: 'var(--line)' }}
      >
        {list.map((o) =>
          row(
            o.id,
            <>
              {o.id}
              {o.held_out && (
                <span className="micro" style={{ fontSize: 8, color: 'var(--cyan)' }}>
                  {' '}· EVAL-ONLY
                </span>
              )}
            </>,
            o.default_weight,
          ),
        )}
      </div>

      {/* Custom (user-authored DSL) opponents — default unchecked. */}
      {customList.length > 0 && (
        <>
          <span className="micro text-fg-2" style={{ fontSize: 9, marginTop: 4 }}>
            CUSTOM · DSL behavior on the standard enemy chassis
          </span>
          <div
            className="flex flex-col divide-y rounded border"
            style={{ borderColor: 'var(--line)' }}
          >
            {customList.map((c) =>
              row(
                c.id,
                <>
                  {c.name}
                  <span className="micro" style={{ fontSize: 8, color: 'var(--accent)' }}>
                    {' '}· CUSTOM
                  </span>
                </>,
                0,
              ),
            )}
          </div>
        </>
      )}
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
  const logs = useMemo(
    () =>
      (status?.events.filter((e) => e.t === 'log') ??
        []) as unknown as LogEvent[],
    [status],
  );
  const hasEntropy = useMemo(
    () => logs.some((l) => typeof l.entropy === 'number'),
    [logs],
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

      {/* Policy entropy (PPO) — its own series across the log cadence. */}
      {hasEntropy && (
        <Reveal index={4}>
          <Panel title="Policy Entropy · per log" live ticks bodyClassName="p-4">
            <EntropyChart logs={logs} />
            <div className="mt-2 flex gap-4">
              <span className="micro num" style={{ fontSize: 9, color: 'var(--cyan)' }}>
                ── ENTROPY (nats · 0 collapsed → ~2.2 uniform)
              </span>
            </div>
          </Panel>
        </Reveal>
      )}

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
