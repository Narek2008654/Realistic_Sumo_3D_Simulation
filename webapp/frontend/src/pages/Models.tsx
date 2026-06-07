// Model Registry: instrument-style table + per-model cards from /api/models.
import { useEffect, useState } from 'react';
import { api, ApiError } from '../api';
import { Panel, Reveal, StatusDot } from '../components/ui';
import type { ModelCard } from '../types';

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

function WinRateBar({ frac }: { frac: number | null }) {
  if (frac == null) {
    return <span className="num text-fg-2">—</span>;
  }
  const pct = Math.max(0, Math.min(1, frac));
  return (
    <div className="flex items-center gap-2">
      <div
        className="h-1.5 w-20 overflow-hidden rounded"
        style={{ background: 'var(--bg-3)' }}
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

function ModelCardView({ card: initial }: { card: ModelCard }) {
  const [card, setCard] = useState<ModelCard>(initial);
  const [evaluating, setEvaluating] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const wr = winRateFraction(card);
  const selfOut = metricNumber(card.metrics, ['self_out_rate', 'self_out%']);
  const evaluated = card.metrics != null;

  async function runEval() {
    setEvaluating(true);
    setErr(null);
    try {
      setCard(await api.evaluate(card.id));
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : 'eval failed');
    } finally {
      setEvaluating(false);
    }
  }

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
      <div className="mt-3 flex items-center justify-between">
        <span className="micro text-fg-2" style={{ fontSize: 9 }}>
          WIN RATE
        </span>
        {evaluated ? (
          <WinRateBar frac={wr} />
        ) : (
          <button
            onClick={runEval}
            disabled={evaluating}
            title="Runs a PyBullet eval on the backend (slow)"
            className="micro"
            style={{
              fontSize: 10,
              letterSpacing: '.06em',
              padding: '3px 8px',
              borderRadius: 'var(--radius)',
              border: '1px solid var(--line-2)',
              background: 'var(--bg-2)',
              color: evaluating ? 'var(--fg-2)' : 'var(--accent)',
              cursor: evaluating ? 'default' : 'pointer',
            }}
          >
            {evaluating ? 'EVALUATING…' : 'NOT EVALUATED · RUN'}
          </button>
        )}
      </div>
      {evaluated && selfOut != null && (
        <div className="mt-1.5 flex items-center justify-between">
          <span className="micro text-fg-2" style={{ fontSize: 9 }}>
            SELF-OUT
          </span>
          <span className="num" style={{ fontSize: 11, color: 'var(--loss)' }}>
            {(selfOut > 1 ? selfOut : selfOut * 100).toFixed(0)}%
          </span>
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
