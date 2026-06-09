// Model Registry: instrument-style table + per-model cards from /api/models,
// plus a TRAINING RUNS panel that promotes finished runs into the registry.
import { useCallback, useEffect, useState } from 'react';
import type { CSSProperties } from 'react';
import { api, ApiError } from '../api';
import { Panel, Reveal, StatusDot, StatusPill } from '../components/ui';
import type { EvalMode, ModelCard, TrainRun } from '../types';

/** Lower-kebab slug, [a-z0-9-] only — mirrors the backend's slug() so the
 *  prefilled name previews as the id the checkpoint will actually get. */
function slugify(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

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

function ModelCardView({
  card: initial,
  onDeleted,
}: {
  card: ModelCard;
  onDeleted: (id: string) => void;
}) {
  const [card, setCard] = useState<ModelCard>(initial);
  // Track which mode is currently running ('quick' | 'full' | null).
  const [running, setRunning] = useState<EvalMode | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [confirmDel, setConfirmDel] = useState(false);
  const [deleting, setDeleting] = useState(false);

  async function remove() {
    setDeleting(true);
    setErr(null);
    try {
      await api.deleteModel(card.id);
      onDeleted(card.id);
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : 'delete failed');
      setDeleting(false);
      setConfirmDel(false);
    }
  }

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

      <div className="mt-2 flex items-center justify-end">
        {card.protected ? (
          <span
            className="micro"
            style={{ fontSize: 9, color: 'var(--cyan)' }}
            title="Deployed/canonical model — remove with git if you truly must"
          >
            🔒 PROTECTED
          </span>
        ) : confirmDel ? (
          <span className="flex items-center gap-2">
            <span className="micro text-fg-2" style={{ fontSize: 9 }}>
              REMOVE THIS MODEL?
            </span>
            <button
              onClick={remove}
              disabled={deleting}
              className="micro"
              style={{
                fontSize: 9,
                padding: '2px 7px',
                borderRadius: 'var(--radius)',
                border: '1px solid var(--loss)',
                background: 'var(--bg-2)',
                color: 'var(--loss)',
                cursor: deleting ? 'default' : 'pointer',
              }}
            >
              {deleting ? 'REMOVING…' : 'CONFIRM'}
            </button>
            <button
              onClick={() => setConfirmDel(false)}
              disabled={deleting}
              className="micro"
              style={{ fontSize: 9, color: 'var(--fg-2)', cursor: 'pointer' }}
            >
              CANCEL
            </button>
          </span>
        ) : (
          <button
            onClick={() => setConfirmDel(true)}
            className="micro"
            title="Delete this checkpoint from the registry"
            style={{
              fontSize: 9,
              padding: '2px 7px',
              borderRadius: 'var(--radius)',
              border: '1px solid var(--line-2)',
              background: 'transparent',
              color: 'var(--fg-2)',
              cursor: 'pointer',
            }}
          >
            REMOVE
          </button>
        )}
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

/** Format a 0..1 rate as a 0-padded percent, or an em dash when null. */
function pct(frac: number | null): string {
  if (frac == null) return '—';
  return `${Math.round(Math.max(0, Math.min(1, frac)) * 100)}%`;
}

const RUN_BTN: CSSProperties = {
  fontSize: 10,
  letterSpacing: '.06em',
  padding: '3px 8px',
  borderRadius: 'var(--radius)',
  border: '1px solid var(--line-2)',
  background: 'var(--bg-2)',
  cursor: 'pointer',
};

/**
 * One training-run row with an inline promote control. A run is promotable when
 * it has at least one of best.pt / final.pt; we default `which` to whichever it
 * has (preferring best). `onPromoted` triggers a registry refetch in the parent
 * so the freshly-minted card appears in both the table and the card grid.
 */
function RunRow({
  run,
  onPromoted,
}: {
  run: TrainRun;
  onPromoted: () => void;
}) {
  const short = run.job_id.slice(0, 8);
  const defaultName = `${run.algo ?? 'model'}-${short}`;
  const canBest = run.has_best;
  const canFinal = run.has_final;
  const promotable = canBest || canFinal;

  const [open, setOpen] = useState(false);
  const [which, setWhich] = useState<'best' | 'final'>(
    canBest ? 'best' : 'final',
  );
  const [name, setName] = useState(defaultName);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [okId, setOkId] = useState<string | null>(null);

  const slug = slugify(name);

  async function promote() {
    if (busy || !slug) return;
    setBusy(true);
    setErr(null);
    setOkId(null);
    try {
      const card = await api.promoteRun({ job_id: run.job_id, which, name });
      setOkId(card.id);
      setOpen(false);
      onPromoted(); // refetch the registry so the new card shows up
    } catch (e) {
      if (e instanceof ApiError) {
        if (e.status === 404)
          setErr('Run or checkpoint not found (may have been deleted).');
        else if (e.status === 409)
          setErr(`Name "${slug}" is taken or protected — pick another.`);
        else if (e.status === 422)
          setErr(e.message || 'Invalid name or checkpoint selection.');
        else setErr(e.message);
      } else {
        setErr('Promote failed.');
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      className="flex flex-col gap-2 rounded border px-3 py-2.5"
      style={{ borderColor: 'var(--line)', background: 'var(--bg-2)' }}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2.5">
          <span className="num text-fg-0" style={{ fontSize: 12 }} title={run.job_id}>
            {short}
          </span>
          <span className="micro" style={{ fontSize: 9, color: 'var(--accent)' }}>
            {run.algo ?? '—'}
          </span>
          {run.mode && (
            <span className="micro text-fg-2" style={{ fontSize: 9 }}>
              {run.mode}
            </span>
          )}
          {run.running && (
            <StatusPill status="accent" label="RUNNING" pulse />
          )}
        </div>
        <div className="flex items-center gap-3">
          <span className="num text-fg-2" style={{ fontSize: 10 }}>
            {run.best_step != null ? `step ${run.best_step.toLocaleString()}` : 'no checkpoint'}
          </span>
          <WinRateBar frac={run.best_wr} trackWidth={64} />
          <span
            className="num"
            style={{ fontSize: 10, color: run.best_self_out != null ? 'var(--loss)' : 'var(--fg-2)' }}
            title="Self-out rate at best checkpoint"
          >
            SO {pct(run.best_self_out)}
          </span>
          {promotable ? (
            <button
              onClick={() => setOpen((v) => !v)}
              className="micro"
              style={{ ...RUN_BTN, color: 'var(--accent)' }}
            >
              {open ? 'CANCEL' : okId ? 'PROMOTE AGAIN' : 'PROMOTE'}
            </button>
          ) : (
            <span
              className="micro text-fg-2"
              style={{ fontSize: 9 }}
              title="No best.pt / final.pt yet — let the run produce a checkpoint."
            >
              NO WEIGHTS
            </span>
          )}
        </div>
      </div>

      {okId && !open && (
        <div className="num" style={{ fontSize: 10, color: 'var(--win)' }}>
          ✓ promoted to <span style={{ color: 'var(--cyan)' }}>{okId}</span>
        </div>
      )}

      {open && (
        <div
          className="flex flex-col gap-2 rounded border px-3 py-2.5"
          style={{ borderColor: 'var(--line)', background: 'var(--bg-1)' }}
        >
          <div className="flex flex-wrap items-end gap-3">
            <label className="flex flex-col gap-1">
              <span className="micro text-fg-2" style={{ fontSize: 9 }}>
                CHECKPOINT
              </span>
              <div className="flex gap-1.5">
                <button
                  onClick={() => setWhich('best')}
                  disabled={!canBest}
                  className="micro"
                  style={{
                    ...RUN_BTN,
                    borderColor: which === 'best' ? 'var(--accent)' : 'var(--line-2)',
                    color: !canBest ? 'var(--fg-2)' : which === 'best' ? 'var(--accent)' : 'var(--fg-1)',
                    cursor: canBest ? 'pointer' : 'default',
                  }}
                  title={canBest ? 'best.pt — highest win-rate snapshot' : 'no best.pt for this run'}
                >
                  BEST
                </button>
                <button
                  onClick={() => setWhich('final')}
                  disabled={!canFinal}
                  className="micro"
                  style={{
                    ...RUN_BTN,
                    borderColor: which === 'final' ? 'var(--accent)' : 'var(--line-2)',
                    color: !canFinal ? 'var(--fg-2)' : which === 'final' ? 'var(--accent)' : 'var(--fg-1)',
                    cursor: canFinal ? 'pointer' : 'default',
                  }}
                  title={canFinal ? 'final.pt — last snapshot at end of run' : 'no final.pt for this run'}
                >
                  FINAL
                </button>
              </div>
            </label>

            <label className="flex flex-1 flex-col gap-1" style={{ minWidth: 160 }}>
              <span className="micro text-fg-2" style={{ fontSize: 9 }}>
                MODEL NAME
              </span>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') promote();
                }}
                placeholder={defaultName}
                className="num"
                style={{
                  fontSize: 12,
                  color: 'var(--cyan)',
                  background: 'var(--bg-2)',
                  border: '1px solid var(--line)',
                  borderRadius: 'var(--radius)',
                  padding: '4px 7px',
                }}
              />
            </label>

            <button
              onClick={promote}
              disabled={busy || !slug}
              className="micro"
              style={{
                ...RUN_BTN,
                border: '1px solid var(--accent)',
                color: busy || !slug ? 'var(--fg-2)' : 'var(--accent)',
                cursor: busy || !slug ? 'default' : 'pointer',
              }}
            >
              {busy ? 'PROMOTING…' : 'PROMOTE →'}
            </button>
          </div>

          <div className="num text-fg-2" style={{ fontSize: 9 }}>
            saves as checkpoints/
            <span style={{ color: slug ? 'var(--cyan)' : 'var(--loss)' }}>
              {slug || 'enter a name'}
            </span>
            .pt
          </div>

          {err && (
            <div className="num" style={{ fontSize: 10, color: 'var(--loss)' }}>
              {err}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** TRAINING RUNS panel: lists job dirs from /api/models/runs and lets each one
 *  with weights be promoted into the registry. */
function TrainingRuns({ onPromoted }: { onPromoted: () => void }) {
  const [runs, setRuns] = useState<TrainRun[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(() => {
    api
      .listRuns()
      .then(setRuns)
      .catch((e) => setErr(e instanceof ApiError ? e.message : String(e)));
  }, []);

  useEffect(load, [load]);

  // After a promote, refetch both the registry (parent) and the runs list (so a
  // newly-appeared final.pt etc. is reflected).
  const handlePromoted = useCallback(() => {
    onPromoted();
    load();
  }, [onPromoted, load]);

  return (
    <Panel
      title="Training Runs"
      live
      right={
        <button
          onClick={load}
          className="micro"
          style={{ fontSize: 9, color: 'var(--cyan)', cursor: 'pointer', background: 'transparent', border: 'none' }}
          title="Refresh the run list"
        >
          ↻ REFRESH
        </button>
      }
    >
      {err ? (
        <div className="flex items-center gap-2">
          <StatusDot status="down" />
          <span className="num" style={{ fontSize: 11, color: 'var(--loss)' }}>
            {err}
          </span>
        </div>
      ) : !runs ? (
        <span className="micro animate-pulse" style={{ color: 'var(--cyan)' }}>
          LOADING RUNS…
        </span>
      ) : runs.length === 0 ? (
        <p className="num text-fg-1" style={{ fontSize: 12 }}>
          No training runs found. Start one on the{' '}
          <span style={{ color: 'var(--accent)' }}>Train</span> page.
        </p>
      ) : (
        <div className="flex flex-col gap-2">
          {runs.map((run) => (
            <RunRow key={run.job_id} run={run} onPromoted={handlePromoted} />
          ))}
        </div>
      )}
    </Panel>
  );
}

export default function Models() {
  const [cards, setCards] = useState<ModelCard[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Shared loader so a promote can refetch the whole registry — the new card
  // then appears in both the table and the card grid below.
  const loadModels = useCallback(() => {
    api
      .models()
      .then(setCards)
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  }, []);

  useEffect(loadModels, [loadModels]);

  // The TRAINING RUNS panel renders in every state (incl. error/empty) so a run
  // can be promoted even before any model exists in the registry.
  const runsPanel = (
    <Reveal index={0}>
      <TrainingRuns onPromoted={loadModels} />
    </Reveal>
  );

  if (error) {
    return (
      <div className="flex flex-col gap-5">
        {runsPanel}
        <Reveal index={1}>
          <Panel title="Model Registry" live>
            <div className="flex items-center gap-2">
              <StatusDot status="down" />
              <span className="num" style={{ color: 'var(--loss)' }}>
                {error}
              </span>
            </div>
          </Panel>
        </Reveal>
      </div>
    );
  }

  if (!cards) {
    return (
      <div className="flex flex-col gap-5">
        {runsPanel}
        <Reveal index={1}>
          <Panel title="Model Registry" live>
            <span className="micro animate-pulse" style={{ color: 'var(--cyan)' }}>
              LOADING REGISTRY…
            </span>
          </Panel>
        </Reveal>
      </div>
    );
  }

  if (cards.length === 0) {
    return (
      <div className="flex flex-col gap-5">
        {runsPanel}
        <Reveal index={1}>
          <Panel title="Model Registry" live ticks>
            <p className="num text-fg-1">
              No checkpoints found in{' '}
              <span style={{ color: 'var(--cyan)' }}>checkpoints/</span>. Promote a
              training run above to add one.
            </p>
          </Panel>
        </Reveal>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5">
      {runsPanel}
      <Reveal index={1}>
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
          <Reveal key={card.id} index={i + 2}>
            <ModelCardView
              card={card}
              onDeleted={(id) =>
                setCards((prev) => prev?.filter((c) => c.id !== id) ?? prev)
              }
            />
          </Reveal>
        ))}
      </div>
    </div>
  );
}
