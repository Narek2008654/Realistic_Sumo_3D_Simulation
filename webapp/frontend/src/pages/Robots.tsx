// Saved Robots: lists designs from /api/robots as instrument cards. Per robot:
// Load into Builder (fetch the full spec, stash it in the builder store, then
// navigate to /hardware), Download URDF, and Delete (with confirm).
import { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, ApiError } from '../api';
import { Panel, Reveal, StatusDot } from '../components/ui';
import { useBuilderStore } from '../store/builder';
import type { RobotSummary } from '../types';

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

function triggerDownload(filename: string, text: string) {
  const blob = new Blob([text], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function RobotCard({
  robot,
  onDeleted,
}: {
  robot: RobotSummary;
  onDeleted: (id: string) => void;
}) {
  const navigate = useNavigate();
  const loadSpec = useBuilderStore((s) => s.loadSpec);
  const [busy, setBusy] = useState<'load' | 'urdf' | 'delete' | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function doLoad() {
    setBusy('load');
    setErr(null);
    try {
      const rec = await api.getRobot(robot.id);
      loadSpec(rec.hardware_spec, rec.name);
      navigate('/hardware');
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : String(e));
      setBusy(null);
    }
  }

  async function doUrdf() {
    setBusy('urdf');
    setErr(null);
    try {
      const urdf = await api.getRobotUrdf(robot.id);
      triggerDownload(`${robot.id}.urdf`, urdf);
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  async function doDelete() {
    setBusy('delete');
    setErr(null);
    try {
      await api.deleteRobot(robot.id);
      onDeleted(robot.id);
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : String(e));
      setBusy(null);
      setConfirming(false);
    }
  }

  return (
    <Panel
      title={robot.name}
      live
      ticks
      right={
        <span className="num text-fg-2" style={{ fontSize: 10 }}>
          {formatTs(robot.created_at)}
        </span>
      }
    >
      <div className="grid grid-cols-3 gap-2">
        <Stat label="OBS DIM" value={String(robot.obs_dim)} tone="cyan" />
        <Stat label="ACT DIM" value={String(robot.action_dim)} tone="accent" />
        <Stat label="SIGNATURE" value={robot.obs_signature_hash ?? '—'} />
      </div>

      <div className="num mt-2 truncate text-fg-2" style={{ fontSize: 10 }}>
        id {robot.id}
      </div>

      {err && (
        <div className="num mt-2" style={{ fontSize: 10, color: 'var(--loss)' }}>
          {err}
        </div>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <button
          className="btn btn-primary"
          style={{ height: 28 }}
          onClick={doLoad}
          disabled={busy !== null}
        >
          {busy === 'load' ? 'LOADING…' : 'LOAD INTO BUILDER'}
        </button>
        <button
          className="btn btn-secondary"
          style={{ height: 28 }}
          onClick={doUrdf}
          disabled={busy !== null}
        >
          {busy === 'urdf' ? 'FETCHING…' : 'DOWNLOAD URDF'}
        </button>
        {confirming ? (
          <span className="flex items-center gap-1.5">
            <button
              className="btn btn-ghost"
              style={{ height: 28, color: 'var(--loss)' }}
              onClick={doDelete}
              disabled={busy !== null}
            >
              {busy === 'delete' ? 'DELETING…' : 'CONFIRM DELETE'}
            </button>
            <button
              className="btn btn-ghost"
              style={{ height: 28 }}
              onClick={() => setConfirming(false)}
              disabled={busy !== null}
            >
              CANCEL
            </button>
          </span>
        ) : (
          <button
            className="btn btn-ghost"
            style={{ height: 28, color: 'var(--fg-2)' }}
            onClick={() => setConfirming(true)}
            disabled={busy !== null}
          >
            DELETE
          </button>
        )}
      </div>
    </Panel>
  );
}

function Stat({
  label,
  value,
  tone = 'fg',
}: {
  label: string;
  value: string;
  tone?: 'fg' | 'cyan' | 'accent';
}) {
  const color =
    tone === 'cyan'
      ? 'var(--cyan)'
      : tone === 'accent'
        ? 'var(--accent)'
        : 'var(--fg-0)';
  return (
    <div
      className="flex flex-col rounded border px-2 py-1.5"
      style={{ borderColor: 'var(--line)', background: 'var(--bg-2)' }}
    >
      <span className="micro text-fg-2" style={{ fontSize: 9 }}>
        {label}
      </span>
      <span className="num truncate" style={{ fontSize: 13, color }}>
        {value}
      </span>
    </div>
  );
}

export default function Robots() {
  const [robots, setRobots] = useState<RobotSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  const reload = useCallback(() => {
    setError(null);
    api
      .listRobots()
      .then(setRobots)
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  }, []);

  useEffect(() => {
    reload();
  }, [reload]);

  function onDeleted(id: string) {
    setRobots((prev) => (prev ? prev.filter((r) => r.id !== id) : prev));
  }

  if (error) {
    return (
      <Panel title="Saved Robots" live>
        <div className="flex items-center gap-2">
          <StatusDot status="down" />
          <span className="num" style={{ color: 'var(--loss)' }}>
            {error}
          </span>
        </div>
        <p className="num mt-2 text-fg-2" style={{ fontSize: 12 }}>
          Is the backend running on 127.0.0.1:8000?
        </p>
      </Panel>
    );
  }

  if (!robots) {
    return (
      <Panel title="Saved Robots" live>
        <span className="micro animate-pulse" style={{ color: 'var(--cyan)' }}>
          LOADING ROBOTS…
        </span>
      </Panel>
    );
  }

  if (robots.length === 0) {
    return (
      <Reveal>
        <Panel title="Saved Robots" live ticks>
          <p className="num text-fg-1">
            No saved robots yet. Design one in the{' '}
            <span style={{ color: 'var(--cyan)' }}>Hardware Builder</span> and
            hit SAVE ROBOT.
          </p>
          <button
            className="btn btn-primary mt-3"
            style={{ height: 30 }}
            onClick={() => navigate('/hardware')}
          >
            OPEN BUILDER
          </button>
        </Panel>
      </Reveal>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-3">
      {robots.map((r, i) => (
        <Reveal key={r.id} index={i}>
          <RobotCard robot={r} onDeleted={onDeleted} />
        </Reveal>
      ))}
    </div>
  );
}
