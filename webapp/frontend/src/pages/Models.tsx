// Model Registry: instrument-style table + per-model cards from /api/models.
import { useEffect, useState } from 'react';
import { api, ApiError } from '../api';
import { Panel, Reveal, StatusDot } from '../components/ui';
import type { EvalMode, ModelCard } from '../types';

function metricNumber(
  metrics: ModelCard['metrics'],
  keys: string[],
): number | null {
  if (!metrics) return null;
  for (const k of keys) {
    const v = metrics[k];
    if (typeof v === 'number') return v;
  }
  return null;
}

/** Win-rate may be 0..1 or 0..100; normalise to a 0..1 fraction. */
function winRateFraction(card: ModelCard): number | null {
  const wr = metricNumber(card.metrics, [
    'win_rate',
    'winrate',
    'wr',
    'win_pct',
  ]);
  if (wr == null) return null;
  return wr > 1 ? wr / 100 : wr;
}

function WinRateBar({
  frac,
  trackWidth = 80,
}: {
  frac: number | null;
  trackWidth?: number;
}) {
  if (frac == null) {
    return <span className="num text-fg-2">—</span>;
  }
  const pct = Math.max(0, Math.min(1, frac));
  return (
    <div className="flex items-center gap-2">
      <div
        className="h-1.5 overflow-hidden rounded"
        style={{ background: 'var(--bg-3)', width: trackWidth }}
      >
        <div
          className="h-full rounded"
          style={{
            width: `${pct * 100}%`,
            background: 'var(--accent)',
            boxShadow: '0 0 8px var(--accent-glow)',
          }}
        />
      </div>
      <span className="num" style={{ fontSize: 11, color: 'var(--cyan)' }}>
        {(pct * 100).toFixed(0)}%
      </span>
    </div>
  );
}

function formatTs(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/** Normalise a self-out value that may be a 0..1 rate or a 0..100 percent. */
function selfOutFraction(card: ModelCard): number | null {
  const v = metricNumber(card.metrics, ['self_out_rate', 'self_out%']);
  if (v == null) return null;
  return v > 1 ? v / 100 : v;
}

/** Mode tag: forge/warn for a quick probe, cyan for the full gauntlet. */
function ModeTag({ mode }: { mode: EvalMode }) {
  const isFull = mode === 'full';
  const color = isFull ? 'var(--cyan)' : 'var(--warn)';
  return (
    <span
      className="micro"
      title={
        isFull
          ? 'Full gauntlet: whole trained zoo + held-out (incl. novamax)'
          : 'Quick probe: 3 easy opponents — not the full picture'
      }
      style={{
        fontSize: 9,
        letterSpacing: '.1em',
        padding: '2px 6px',
        borderRadius: 'var(--radius)',
        border: `1px solid ${color}`,
        color,
        background: 'var(--bg-2)',
        whiteSpace: 'nowrap',
      }}
    >
      {isFull ? 'FULL GAUNTLET' : 'QUICK PROBE'}
    </span>
  );
}

/** One of the two unevaluated launch buttons (QUICK / FULL). */
function EvalButton({
  label,
  hint,
  running,
  disabled,
  onClick,
}: {
  label: string;
  hint: string;
  running: boolean;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={hint}
      className="micro flex-1"
      style={{
        fontSize: 10,
        letterSpacing: '.06em',
        padding: '4px 8px',
        borderRadius: 'var(--radius)',
        border: '1px solid var(--line-2)',
        background: 'var(--bg-2)',
        color: disabled ? 'var(--fg-2)' : 'var(--accent)',
        cursor: disabled ? 'default' : 'pointer',
      }}
    >
      {running ? `${label}…` : label}
    </button>
  );
}

function PerOpponentBreakdown({ card }: { card: ModelCard }) {
  const per = card.metrics?.per_opponent;
  if (!per || typeof per !== 'object') return null;
  const rows = Object.entries(per)
    .map(([name, m]) => ({ name, wr: typeof m?.wr === 'number' ? m.wr : null }))
    .sort((a, b) => (b.wr ?? -1) - (a.wr ?? -1));
  if (rows.length === 0) return null;
  return (
    <div className="mt-3">
      <div className="micro text-fg-2" style={{ fontSize: 9 }}>
        PER OPPONENT
      </div>
      <div className="mt-1.5 flex flex-col gap-1">
        {rows.map((r) => (
          <div key={r.name} className="flex items-center justify-between gap-2">
            <span
              className="num text-fg-1 truncate"
              style={{ fontSize: 10 }}
              title={r.name}
            >
              {r.name}
            </span>
            <WinRateBar frac={r.wr} trackWidth={56} />
          </div>
        ))}
      </div>
    </div>
  );
}

function ModelCardView({ card: initial }: { card: ModelCard }) {
  const [card, setCard] = useState<ModelCard>(initial);
  // Track which mode is currently running ('quick' | 'full' | null).
  const [running, setRunning] = useState<EvalMode | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const wr = winRateFraction(card);
  const selfOut = selfOutFraction(card);
  const evaluated = card.metrics != null;
  const m = card.metrics;
  const mode = (m?.mode === 'quick' || m?.mode === 'full' ? m.mode : null) as
    | EvalMode
    | null;
  const opponents = Array.isArray(m?.opponents) ? m!.opponents : null;
  const nOpps =
    opponents?.length ??
    (m?.per_opponent ? Object.keys(m.per_opponent).length : null);
  const mult = metricNumber(m, ['mult']);
  const nEpisodes = metricNumber(m, ['n_episodes']);

  async function runEval(mode: EvalMode) {
    if (running) return;
    setRunning(mode);
    setErr(null);
    try {
      setCard(await api.evaluate(card.id, mode));
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : 'eval failed');
    } finally {
      setRunning(null);
    }
  }

  // Build the "vs N opponents @ M× , n=K" conditions line from what we have.
  const conditions: string[] = [];
  if (nOpps != null) conditions.push(`vs ${nOpps} opponents`);
  if (mult != null) conditions.push(`@ ${mult}×`);
  if (nEpisodes != null) conditions.push(`n=${nEpisodes}`);

  return (
    <Panel
      title={card.id}
      live
      ticks
      right={
        <span className="micro" style={{ color: 'var(--accent)', fontSize: 10 }}>
          {card.algo}
        </span>
      }
    >
      <div className="grid grid-cols-3 gap-2">
        <Stat label="OBS / ACT" value={`${card.obs_dim} / ${card.action_dim}`} />
        <Stat
          label="NET ARCH"
          value={`${card.net_arch[0]}×${card.net_arch[1]}`}
        />
        <Stat label="PARAMS" value={card.param_count.toLocaleString()} />
      </div>

      {evaluated ? (
        <>
          <div className="mt-3 flex items-center justify-between">
            <span className="micro text-fg-2" style={{ fontSize: 9 }}>
              WIN RATE
            </span>
            {mode && <ModeTag mode={mode} />}
          </div>
          <div className="mt-1.5 flex items-center justify-end">
            <WinRateBar frac={wr} />
          </div>

          {conditions.length > 0 && (
            <div
              className="num mt-2 text-fg-2"
              style={{ fontSize: 10 }}
              title="Evaluation conditions"
            >
              {conditions.join(' ')}
            </div>
          )}

          {selfOut != null && (
            <div className="mt-1.5 flex items-center justify-between">
              <span className="micro text-fg-2" style={{ fontSize: 9 }}>
                SELF-OUT
              </span>
              <span
                className="num"
                style={{ fontSize: 11, color: 'var(--loss)' }}
              >
                {(selfOut * 100).toFixed(0)}%
              </span>
            </div>
          )}

          <PerOpponentBreakdown card={card} />

          <div className="mt-3 flex items-center justify-end gap-2">
            <span className="micro text-fg-2" style={{ fontSize: 9 }}>
              RE-EVAL
            </span>
            <button
              onClick={() => runEval('quick')}
              disabled={running != null}
              title="Re-run the quick 3-opponent probe"
              className="micro"
              style={{
                fontSize: 10,
                letterSpacing: '.06em',
                padding: '3px 8px',
                borderRadius: 'var(--radius)',
                border: '1px solid var(--line-2)',
                background: 'var(--bg-2)',
                color: running ? 'var(--fg-2)' : 'var(--accent)',
                cursor: running ? 'default' : 'pointer',
              }}
            >
              {running === 'quick' ? 'QUICK…' : 'QUICK'}
            </button>
            <button
              onClick={() => runEval('full')}
              disabled={running != null}
              title="Re-run the full gauntlet (whole zoo + held-out, slow)"
              className="micro"
              style={{
                fontSize: 10,
                letterSpacing: '.06em',
                padding: '3px 8px',
                borderRadius: 'var(--radius)',
                border: '1px solid var(--cyan-dim)',
                background: 'var(--bg-2)',
                color: running ? 'var(--fg-2)' : 'var(--cyan)',
                cursor: running ? 'default' : 'pointer',
              }}
            >
              {running === 'full' ? 'FULL…' : 'FULL'}
            </button>
          </div>
        </>
      ) : (
        <div className="mt-3">
          <div className="micro text-fg-2" style={{ fontSize: 9 }}>
            NOT EVALUATED
          </div>
          <div className="mt-1.5 flex gap-2">
            <EvalButton
              label="QUICK"
              hint="3 easy opponents (dodger/spinner/rammer), fast"
              running={running === 'quick'}
              disabled={running != null}
              onClick={() => runEval('quick')}
            />
            <EvalButton
              label="FULL"
              hint="Full gauntlet incl. novamax + held-out, slow"
              running={running === 'full'}
              disabled={running != null}
              onClick={() => runEval('full')}
            />
          </div>
          <div className="num mt-1.5 text-fg-2" style={{ fontSize: 9 }}>
            QUICK = 3 easy opponents, fast · FULL = whole zoo + held-out, slow
          </div>
        </div>
      )}

      {err && (
        <div className="num mt-1.5" style={{ fontSize: 10, color: 'var(--loss)' }}>
          {err}
        </div>
      )}
      <div
        className="num mt-3 truncate text-fg-2"
        style={{ fontSize: 10 }}
        title={card.obs_signature_hash ?? ''}
      >
        sig {card.obs_signature_hash ?? '—'} · {formatTs(card.created_at)}
      </div>
    </Panel>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div
      className="flex flex-col rounded border px-2 py-1.5"
      style={{ borderColor: 'var(--line)', background: 'var(--bg-2)' }}
    >
      <span className="micro text-fg-2" style={{ fontSize: 9 }}>
        {label}
      </span>
      <span className="num" style={{ fontSize: 13 }}>
        {value}
      </span>
    </div>
  );
}

export default function Models() {
  const [cards, setCards] = useState<ModelCard[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .models()
      .then(setCards)
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  }, []);

  if (error) {
    return (
      <Panel title="Model Registry" live>
        <div className="flex items-center gap-2">
          <StatusDot status="down" />
          <span className="num" style={{ color: 'var(--loss)' }}>
            {error}
          </span>
        </div>
      </Panel>
    );
  }

  if (!cards) {
    return (
      <Panel title="Model Registry" live>
        <span className="micro animate-pulse" style={{ color: 'var(--cyan)' }}>
          LOADING REGISTRY…
        </span>
      </Panel>
    );
  }

  if (cards.length === 0) {
    return (
      <Panel title="Model Registry" live ticks>
        <p className="num text-fg-1">
          No checkpoints found in <span style={{ color: 'var(--cyan)' }}>checkpoints/</span>.
        </p>
      </Panel>
    );
  }

  return (
    <div className="flex flex-col gap-5">
      <Reveal index={0}>
        <Panel title={`Registry · ${cards.length} models`} live bodyClassName="p-0">
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr>
                  {['ID', 'ALGO', 'OBS', 'ACT', 'NET', 'PARAMS', 'WIN RATE', 'SIGNATURE'].map(
                    (h) => (
                      <th
                        key={h}
                        className="micro px-3 py-2 text-left text-fg-2"
                        style={{
                          fontSize: 10,
                          borderBottom: '1px solid var(--line)',
                        }}
                      >
                        {h}
                      </th>
                    ),
                  )}
                </tr>
              </thead>
              <tbody>
                {cards.map((card, i) => (
                  <tr
                    key={card.id}
                    style={{
                      background: i % 2 ? 'var(--bg-2)' : 'transparent',
                    }}
                  >
                    <td className="num px-3 py-2" style={{ fontSize: 12 }}>
                      {card.id}
                    </td>
                    <td
                      className="num px-3 py-2"
                      style={{ fontSize: 12, color: 'var(--accent)' }}
                    >
                      {card.algo}
                    </td>
                    <td className="num px-3 py-2" style={{ fontSize: 12 }}>
                      {card.obs_dim}
                    </td>
                    <td className="num px-3 py-2" style={{ fontSize: 12 }}>
                      {card.action_dim}
                    </td>
                    <td className="num px-3 py-2 text-fg-1" style={{ fontSize: 12 }}>
                      {card.net_arch[0]}×{card.net_arch[1]}
                    </td>
                    <td className="num px-3 py-2 text-fg-1" style={{ fontSize: 12 }}>
                      {card.param_count.toLocaleString()}
                    </td>
                    <td className="px-3 py-2">
                      <WinRateBar frac={winRateFraction(card)} />
                    </td>
                    <td
                      className="num px-3 py-2 text-fg-2"
                      style={{ fontSize: 11 }}
                    >
                      {card.obs_signature_hash ?? '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      </Reveal>

      <div className="grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-3">
        {cards.map((card, i) => (
          <Reveal key={card.id} index={i + 1}>
            <ModelCardView card={card} />
          </Reveal>
        ))}
      </div>
    </div>
  );
}
