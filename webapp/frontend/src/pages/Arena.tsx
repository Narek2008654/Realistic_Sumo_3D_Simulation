// ARENA — stage a head-to-head battle (side A model vs another model OR a zoo
// opponent), then read win/self-out/push stats and scrub the recorded battle in
// the DOHYO-CAM replay.
//
// Two modes in one page:
//  • SETUP (idle): pick side A (a trained model, default-picked to a strong
//    ppo_robust_best-style model), pick side B (toggle: another model OR a zoo
//    opponent), set rounds / mult / seed, and optionally tweak side A's hardware
//    (chassis mass + drivetrain torque/omega, seeded from the default spec). B's
//    hardware is fixed to the default chassis in v1 — see the inline note.
//  • RESULT (after FIGHT): a STATS panel (instrument bars + numbers) for
//    a_wins / b_wins / draws / timeouts + self-outs, the recorded battle in the
//    TrajectoryPlayer (agent A = forge-orange, B = cyan), any backend `notes`,
//    and a REMATCH / NEW SETUP control.
//
// Battles run REAL physics on the backend and are synchronous — keep rounds
// small. The FIGHT button is disabled while a battle is in flight.

import { useEffect, useMemo, useState } from 'react';
import { api, ApiError } from '../api';
import { Panel, Reveal, StatusPill } from '../components/ui';
import { SliderField } from '../components/fields';
import { TrajectoryPlayer } from '../components/TrajectoryPlayer';
import type {
  BattleBody,
  BattleResult,
  BattleStats,
  HardwareSpec,
  ModelCard,
  TrainOpponent,
} from '../types';

type SideBKind = 'opponent' | 'model';

// A model is "battle-ready" if we can default-pick a strong, on-the-canonical
// 21-obs / 9-act contract (the ppo_robust / stageA family). Falls back to the
// first model if none match.
function defaultAModelId(models: ModelCard[]): string {
  if (models.length === 0) return '';
  const robust = models.find((m) => m.id === 'ppo_robust_best');
  if (robust) return robust.id;
  const canonical = models.find((m) => m.obs_dim === 21 && m.action_dim === 9);
  return (canonical ?? models[0]).id;
}

// ---------------------------------------------------------------------------
// Side-A hardware tweak — a small, best-effort subset of HardwareSpec fields
// threaded into the agent side as `a_spec`. Seeded from /api/hardware/default;
// "ON" must be toggled to actually send an override (otherwise A uses the
// model's trained chassis exactly).
// ---------------------------------------------------------------------------
function HardwareTweak({
  base,
  spec,
  setSpec,
  enabled,
  setEnabled,
}: {
  base: HardwareSpec | null;
  spec: HardwareSpec | null;
  setSpec: (s: HardwareSpec) => void;
  enabled: boolean;
  setEnabled: (b: boolean) => void;
}) {
  if (!base || !spec) {
    return (
      <span className="num text-fg-2" style={{ fontSize: 11 }}>
        Loading default chassis…
      </span>
    );
  }

  const c = spec.chassis;
  const d = spec.drivetrain;

  const setChassis = (key: 'mass_kg', v: number) =>
    setSpec({ ...spec, chassis: { ...c, [key]: v } });
  const setDrive = (key: 'max_torque_nm' | 'max_omega_rad_s', v: number) =>
    setSpec({ ...spec, drivetrain: { ...d, [key]: v } });

  return (
    <div className="flex flex-col gap-3">
      <label className="flex cursor-pointer items-center gap-2">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(e) => setEnabled(e.target.checked)}
          style={{ accentColor: 'var(--accent)' }}
        />
        <span className="num" style={{ fontSize: 11, color: 'var(--fg-1)' }}>
          OVERRIDE SIDE-A HARDWARE
        </span>
      </label>
      <span className="num text-fg-2" style={{ fontSize: 10 }}>
        Re-chassis the agent for this battle. Off → A fights on its trained spec.
      </span>
      <div
        className="flex flex-col gap-3 rounded border p-3"
        style={{
          borderColor: 'var(--line)',
          background: 'var(--bg-2)',
          opacity: enabled ? 1 : 0.5,
          pointerEvents: enabled ? 'auto' : 'none',
        }}
      >
        <SliderField
          label="MASS"
          unit="kg"
          value={c.mass_kg}
          min={0.1}
          max={1.5}
          step={0.01}
          format={(v) => v.toFixed(2)}
          onChange={(v) => setChassis('mass_kg', v)}
        />
        <SliderField
          label="MAX TORQUE"
          unit="N·m"
          value={d.max_torque_nm}
          min={0.02}
          max={0.5}
          step={0.005}
          format={(v) => v.toFixed(3)}
          onChange={(v) => setDrive('max_torque_nm', v)}
        />
        <SliderField
          label="MAX OMEGA"
          unit="rad/s"
          value={d.max_omega_rad_s}
          min={5}
          max={80}
          step={0.5}
          format={(v) => v.toFixed(1)}
          onChange={(v) => setDrive('max_omega_rad_s', v)}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Stat bar — a labelled count rendered as an instrument bar (fraction of rounds)
// ---------------------------------------------------------------------------
function StatBar({
  label,
  count,
  rounds,
  color,
}: {
  label: string;
  count: number;
  rounds: number;
  color: string;
}) {
  const frac = rounds > 0 ? count / rounds : 0;
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between">
        <span className="micro text-fg-2" style={{ fontSize: 10 }}>
          {label}
        </span>
        <span className="num" style={{ fontSize: 14, color }}>
          {count}
          <span className="text-fg-2" style={{ fontSize: 10 }}>
            {' '}
            / {rounds}
          </span>
        </span>
      </div>
      <div
        className="h-1.5 w-full overflow-hidden rounded"
        style={{ background: 'var(--bg-3)' }}
      >
        <div
          className="h-full rounded"
          style={{
            width: `${Math.max(0, Math.min(1, frac)) * 100}%`,
            background: color,
            boxShadow: `0 0 8px ${color}`,
            transition: 'width .4s ease',
          }}
        />
      </div>
    </div>
  );
}

function StatsPanel({
  stats,
  aLabel,
  bLabel,
}: {
  stats: BattleStats;
  aLabel: string;
  bLabel: string;
}) {
  const { rounds } = stats;
  const verdict =
    stats.a_wins > stats.b_wins
      ? { label: `${aLabel} WINS`, color: 'var(--accent)' }
      : stats.b_wins > stats.a_wins
        ? { label: `${bLabel} WINS`, color: 'var(--cyan)' }
        : { label: 'DEAD HEAT', color: 'var(--warn)' };

  return (
    <Panel title="Battle Result · aggregate" live ticks bodyClassName="flex flex-col gap-4 p-5">
      <div className="flex items-end justify-between">
        <div className="flex flex-col">
          <span className="micro text-fg-2" style={{ fontSize: 10 }}>
            VERDICT · {rounds} ROUND{rounds === 1 ? '' : 'S'}
          </span>
          <span
            className="font-display uppercase"
            style={{ fontSize: 24, color: verdict.color, letterSpacing: '.04em' }}
          >
            {verdict.label}
          </span>
        </div>
        <div className="num text-fg-2" style={{ fontSize: 12, textAlign: 'right' }}>
          <span style={{ color: 'var(--accent)' }}>A · {aLabel}</span>
          <br />
          <span style={{ color: 'var(--cyan)' }}>B · {bLabel}</span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-x-5 gap-y-3">
        <StatBar label="A WINS" count={stats.a_wins} rounds={rounds} color="var(--accent)" />
        <StatBar label="B WINS" count={stats.b_wins} rounds={rounds} color="var(--cyan)" />
        <StatBar label="DRAWS" count={stats.draws} rounds={rounds} color="var(--warn)" />
        <StatBar label="TIMEOUTS" count={stats.timeouts} rounds={rounds} color="var(--idle)" />
        <StatBar label="A SELF-OUT" count={stats.a_self_out} rounds={rounds} color="var(--loss)" />
        <StatBar label="B SELF-OUT" count={stats.b_self_out} rounds={rounds} color="var(--loss)" />
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------
export default function Arena() {
  // Pickers
  const [models, setModels] = useState<ModelCard[]>([]);
  const [opponents, setOpponents] = useState<TrainOpponent[]>([]);
  const [aModelId, setAModelId] = useState('');
  const [bKind, setBKind] = useState<SideBKind>('opponent');
  const [bOpponentId, setBOpponentId] = useState('dodger');
  const [bModelId, setBModelId] = useState('');

  // Controls
  const [rounds, setRounds] = useState(5);
  const [mult, setMult] = useState(3.0);
  const [seed, setSeed] = useState(4242);

  // Side-A hardware tweak
  const [defaultSpec, setDefaultSpec] = useState<HardwareSpec | null>(null);
  const [aSpec, setASpec] = useState<HardwareSpec | null>(null);
  const [tweakA, setTweakA] = useState(false);

  // Battle lifecycle
  const [fighting, setFighting] = useState(false);
  const [result, setResult] = useState<BattleResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Load models, opponents, default spec once.
  useEffect(() => {
    api
      .models()
      .then((ms) => {
        setModels(ms);
        setAModelId((cur) => cur || defaultAModelId(ms));
        // default B model = a different model from A if available
        setBModelId((cur) => cur || (ms.length > 1 ? ms.find((m) => m.id !== defaultAModelId(ms))?.id ?? '' : ''));
      })
      .catch(() => {});
    api
      .trainOpponents()
      .then(({ opponents: opps }) => {
        setOpponents(opps);
        setBOpponentId((cur) => (opps.some((o) => o.id === cur) ? cur : opps[0]?.id ?? ''));
      })
      .catch(() => {});
    api
      .hardwareDefault()
      .then((s) => {
        setDefaultSpec(s);
        setASpec(s);
      })
      .catch(() => {});
  }, []);

  const aModel = useMemo(() => models.find((m) => m.id === aModelId) ?? null, [models, aModelId]);
  // B-model picker excludes the A pick (a model can't battle itself meaningfully
  // on the same chassis — it would be a mirror match).
  const bModelOptions = useMemo(
    () => models.filter((m) => m.id !== aModelId),
    [models, aModelId],
  );

  const aLabel = aModelId || '—';
  const bLabel = bKind === 'opponent' ? bOpponentId : bModelId || '—';

  const canFight =
    !fighting &&
    !!aModelId &&
    (bKind === 'opponent' ? !!bOpponentId : !!bModelId && bModelId !== aModelId);

  async function fight() {
    if (!canFight) return;
    setFighting(true);
    setError(null);
    const body: BattleBody = {
      a_model_id: aModelId,
      rounds,
      mult,
      seed,
    };
    if (bKind === 'opponent') body.b_opponent_id = bOpponentId;
    else body.b_model_id = bModelId;
    // Send side-A's spec when hardware was overridden OR the ring size differs
    // from the default (the env reads the dohyo radius from side A's spec).
    const ringChanged =
      !!aSpec &&
      !!defaultSpec &&
      Math.abs(aSpec.dohyo.radius_m - defaultSpec.dohyo.radius_m) > 1e-6;
    if ((tweakA || ringChanged) && aSpec) body.a_spec = aSpec;

    try {
      const res = await api.runBattle(body);
      setResult(res);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Battle failed to run.');
    } finally {
      setFighting(false);
    }
  }

  function newSetup() {
    setResult(null);
    setError(null);
  }

  return (
    <div className="grid grid-cols-1 gap-5 xl:grid-cols-[420px_minmax(0,1fr)]">
      {/* ---- SETUP column ---- */}
      <Reveal index={0}>
        <Panel title="Battle Setup" live ticks bodyClassName="flex flex-col gap-5 p-5">
          {/* Side A */}
          <div className="flex flex-col gap-2">
            <span className="micro" style={{ fontSize: 10, color: 'var(--accent)' }}>
              ◤ SIDE A · AGENT
            </span>
            <select
              className="ctl num"
              value={aModelId}
              onChange={(e) => setAModelId(e.target.value)}
              style={{ fontSize: 12 }}
            >
              {models.length === 0 && <option value="">— no models —</option>}
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.id} · {m.algo} · {m.obs_dim}obs/{m.action_dim}act
                </option>
              ))}
            </select>
            {aModel && (
              <span className="num text-fg-2" style={{ fontSize: 10 }}>
                net {aModel.net_arch[0]}×{aModel.net_arch[1]} · {aModel.param_count.toLocaleString()} params
              </span>
            )}
          </div>

          {/* Side B */}
          <div className="flex flex-col gap-2">
            <span className="micro" style={{ fontSize: 10, color: 'var(--cyan)' }}>
              ◥ SIDE B · CHALLENGER
            </span>
            <Seg
              value={bKind}
              onChange={setBKind}
              options={[
                { value: 'opponent', label: 'Zoo Opponent' },
                { value: 'model', label: 'Model' },
              ]}
            />
            {bKind === 'opponent' ? (
              <select
                className="ctl num"
                value={bOpponentId}
                onChange={(e) => setBOpponentId(e.target.value)}
                style={{ fontSize: 12 }}
              >
                {opponents.length === 0 && <option value="">— no opponents —</option>}
                {opponents.map((o) => (
                  <option key={o.id} value={o.id}>
                    {o.id}
                    {o.held_out ? ' · eval-only' : ''}
                  </option>
                ))}
              </select>
            ) : bModelOptions.length === 0 ? (
              <span className="num" style={{ fontSize: 11, color: 'var(--warn)' }}>
                Need a second model to run model-vs-model. Train another, or pick a
                zoo opponent.
              </span>
            ) : (
              <select
                className="ctl num"
                value={bModelId}
                onChange={(e) => setBModelId(e.target.value)}
                style={{ fontSize: 12 }}
              >
                <option value="">— select a model —</option>
                {bModelOptions.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.id} · {m.algo} · {m.obs_dim}obs/{m.action_dim}act
                  </option>
                ))}
              </select>
            )}
            {bKind === 'model' && (
              <span className="num text-fg-2" style={{ fontSize: 10 }}>
                Model-vs-model runs both policies on the SAME (default) chassis to
                isolate the brains.
              </span>
            )}
          </div>

          {/* Match controls */}
          <div className="flex flex-col gap-3">
            <span className="micro text-fg-2" style={{ fontSize: 10 }}>
              MATCH PARAMETERS
            </span>
            <SliderField
              label="ROUNDS"
              value={rounds}
              onChange={(v) => setRounds(Math.round(v))}
              min={1}
              max={20}
              step={1}
              format={(v) => v.toFixed(0)}
            />
            <SliderField
              label="OPPONENT TORQUE MULT"
              value={mult}
              onChange={setMult}
              min={1}
              max={5}
              step={0.1}
              format={(v) => `${v.toFixed(1)}×`}
            />
            <SliderField
              label="RING SIZE · DOHYO RADIUS"
              unit="m"
              value={aSpec?.dohyo.radius_m ?? 0.35}
              onChange={(v) =>
                setASpec((prev) =>
                  prev ? { ...prev, dohyo: { ...prev.dohyo, radius_m: v } } : prev,
                )
              }
              min={0.2}
              max={0.6}
              step={0.005}
              format={(v) => `${v.toFixed(3)} m`}
            />
            <label className="flex flex-col gap-1">
              <span className="micro text-fg-2" style={{ fontSize: 10 }}>
                SEED
              </span>
              <input
                type="number"
                className="ctl num"
                value={seed}
                step={1}
                onChange={(e) => {
                  const v = parseInt(e.target.value, 10);
                  if (Number.isFinite(v)) setSeed(v);
                }}
                style={{ height: 32, fontSize: 12 }}
              />
            </label>
          </div>

          {/* Side-A hardware tweak */}
          <div className="flex flex-col gap-2">
            <span className="micro text-fg-2" style={{ fontSize: 10 }}>
              SIDE-A HARDWARE TWEAK · OPTIONAL
            </span>
            <HardwareTweak
              base={defaultSpec}
              spec={aSpec}
              setSpec={setASpec}
              enabled={tweakA}
              setEnabled={setTweakA}
            />
            <span className="num" style={{ fontSize: 10, color: 'var(--warn)' }}>
              ⚠ Side B always fights on the default chassis — per-opponent hardware
              isn't supported yet. Only side A can be re-chassied.
            </span>
          </div>

          {error && (
            <div className="num" style={{ fontSize: 11, color: 'var(--loss)' }}>
              {error}
            </div>
          )}

          <button
            className="btn btn-primary"
            disabled={!canFight}
            onClick={fight}
            style={{ height: 40, fontSize: 15 }}
          >
            {fighting ? 'FIGHTING…' : '⚔ FIGHT'}
          </button>
          {fighting && (
            <span className="micro animate-pulse" style={{ fontSize: 9, color: 'var(--cyan)' }}>
              RUNNING {rounds} ROUND{rounds === 1 ? '' : 'S'} OF REAL PHYSICS — THIS CAN TAKE A MOMENT…
            </span>
          )}
        </Panel>
      </Reveal>

      {/* ---- RESULT / REPLAY column ---- */}
      <div className="flex flex-col gap-5">
        {result ? (
          <>
            <Reveal index={1}>
              <StatsPanel stats={result.stats} aLabel={aLabel} bLabel={bLabel} />
            </Reveal>
            {result.notes && (
              <Reveal index={2}>
                <div
                  className="num rounded border px-4 py-3"
                  style={{
                    fontSize: 11,
                    color: 'var(--warn)',
                    borderColor: 'var(--line)',
                    background: 'var(--bg-2)',
                  }}
                >
                  ⚠ {result.notes}
                </div>
              </Reveal>
            )}
            <Reveal index={3} className="min-h-[420px]">
              <TrajectoryPlayer
                traj={result.trajectory}
                agentSpec={tweakA ? aSpec : null}
                label={`${aLabel} vs ${bLabel}`}
              />
            </Reveal>
            <Reveal index={4}>
              <div className="flex items-center gap-3">
                <button
                  className="btn btn-primary"
                  disabled={!canFight}
                  onClick={fight}
                  style={{ flex: 1, height: 38 }}
                >
                  {fighting ? 'FIGHTING…' : '↻ REMATCH'}
                </button>
                <button
                  className="btn btn-secondary"
                  onClick={newSetup}
                  style={{ height: 38 }}
                >
                  NEW SETUP
                </button>
              </div>
            </Reveal>
          </>
        ) : (
          <Reveal index={1} className="min-h-[420px]">
            <Panel title="Dohyo Cam" live ticks className="h-full">
              <div className="flex h-full min-h-[360px] flex-col items-center justify-center gap-3 py-10">
                <StatusPill status="idle" label="AWAITING FIGHT" />
                <p className="num max-w-sm text-center text-fg-2" style={{ fontSize: 12, lineHeight: 1.5 }}>
                  Configure side A and side B, then{' '}
                  <span style={{ color: 'var(--accent)' }}>FIGHT</span>. The
                  aggregate stats and a recorded round play back here in the
                  dohyo cam — agent A in forge-orange, B in cyan.
                </p>
              </div>
            </Panel>
          </Reveal>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Local segmented toggle (mirrors Train's Seg primitive).
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
