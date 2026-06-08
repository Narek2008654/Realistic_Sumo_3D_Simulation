// Hardware Builder: seed a HardwareSpec (from the backend default, or a spec
// handed off by the Robots page) then either run the guided INTERVIEW (wizard
// that auto-builds the robot from plain-language answers) or jump to CALIBRATE
// (the full detailed form). A persistent right-pane 3D preview — with sensor
// overlays, an underside view, and a center-of-mass marker — is shown in both
// modes. Debounced validate + Save-to-robots happen in Calibrate.
import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, ApiError } from '../api';
import { Readout } from '../components/fields';
import { Interview, type SpecUpdaters } from '../components/Interview';
import { RobotPreview } from '../components/RobotPreview';
import { HardwareForm } from '../components/HardwareForm';
import { Info } from '../components/Info';
import { Panel, Reveal, StatusDot } from '../components/ui';
import { useBuilderStore } from '../store/builder';
import type {
  DistanceSensor,
  Geometry,
  HardwareSpec,
  PreviewView,
  ValidateResult,
} from '../types';

const DEBOUNCE_MS = 350;

type Mode = 'interview' | 'calibrate';

export default function Hardware() {
  const [spec, setSpec] = useState<HardwareSpec | null>(null);
  const [seedError, setSeedError] = useState<string | null>(null);
  const [seedName, setSeedName] = useState<string | null>(null);

  // Mode: the interview wizard (default for a fresh build) or the detailed
  // calibrate form. Loading a robot from /robots jumps straight to calibrate.
  const [mode, setMode] = useState<Mode>('interview');
  // Controlled preview camera so the interview can flip to the underside for
  // the line-sensor step; calibrate leaves the preview's own toggle in charge.
  const [previewView, setPreviewView] = useState<PreviewView>('TOP');

  const [geom, setGeom] = useState<Geometry | null>(null);
  const [geomError, setGeomError] = useState<string | null>(null);
  const [geomLoading, setGeomLoading] = useState(false);

  const [validation, setValidation] = useState<ValidateResult | null>(null);
  const [validating, setValidating] = useState(false);

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const consume = useBuilderStore((s) => s.consume);
  const navigate = useNavigate();

  // Seed: prefer a spec handed off from the Robots page; else the backend
  // default. Consume runs once on mount (StrictMode-safe via the ref guard).
  const seededRef = useRef(false);
  useEffect(() => {
    if (seededRef.current) return;
    seededRef.current = true;
    const handoff = consume();
    if (handoff) {
      // A robot loaded from /robots is already built — go straight to calibrate.
      setSpec(handoff.spec);
      setSeedName(handoff.name);
      setMode('calibrate');
      return;
    }
    api
      .hardwareDefault()
      .then(setSpec)
      .catch((e) =>
        setSeedError(e instanceof ApiError ? e.message : String(e)),
      );
  }, [consume]);

  // Debounced validate + geometry whenever the spec changes.
  const refresh = useCallback((s: HardwareSpec) => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      setValidating(true);
      setGeomLoading(true);
      const [v, g] = await Promise.allSettled([api.validate(s), api.geometry(s)]);
      if (v.status === 'fulfilled') setValidation(v.value);
      else
        setValidation({
          obs_dim: 0,
          action_dim: 0,
          obs_signature_hash: '',
          urdf_valid: false,
          errors: [
            v.reason instanceof ApiError ? v.reason.message : String(v.reason),
          ],
          finetune_candidates: [],
        });
      if (g.status === 'fulfilled') {
        setGeom(g.value);
        setGeomError(null);
      } else {
        setGeomError(
          g.reason instanceof ApiError ? g.reason.message : String(g.reason),
        );
      }
      setValidating(false);
      setGeomLoading(false);
    }, DEBOUNCE_MS);
  }, []);

  useEffect(() => {
    if (spec) refresh(spec);
  }, [spec, refresh]);

  // Immutable spec updaters --------------------------------------------------
  function patch(mut: (draft: HardwareSpec) => HardwareSpec) {
    setSpec((prev) => (prev ? mut(structuredClone(prev)) : prev));
  }
  const setChassis = (k: keyof HardwareSpec['chassis'], v: number) =>
    patch((d) => {
      (d.chassis as unknown as Record<string, unknown>)[k] = v;
      return d;
    });
  const setCom = (axis: 0 | 1 | 2, v: number) =>
    patch((d) => {
      d.chassis.com_xyz[axis] = v;
      return d;
    });
  const setDrive = (k: keyof HardwareSpec['drivetrain'], v: number) =>
    patch((d) => {
      (d.drivetrain as unknown as Record<string, unknown>)[k] = v;
      return d;
    });
  const setDohyo = (k: keyof HardwareSpec['dohyo'], v: number) =>
    patch((d) => {
      (d.dohyo as unknown as Record<string, unknown>)[k] = v;
      return d;
    });
  const setWedgePresent = (present: boolean) =>
    patch((d) => {
      d.chassis.wedge_present = present;
      return d;
    });

  function setSensor(i: number, mut: (s: DistanceSensor) => void) {
    patch((d) => {
      mut(d.distance_sensors[i]);
      return d;
    });
  }
  function setLine(i: number, axis: 0 | 1, v: number) {
    patch((d) => {
      d.line_sensors[i].mount_xy[axis] = v;
      return d;
    });
  }

  // Set the sensor counts directly (interview CountPicker). Grows by appending
  // sensible defaults, shrinks by trimming from the end — preserving existing
  // sensor configs the user already tuned.
  function setSensorCount(n: number) {
    patch((d) => {
      const cur = d.distance_sensors.length;
      if (n > cur) {
        for (let k = cur; k < n; k++) {
          d.distance_sensors.push({
            id: `tof_${k}`,
            mount_xyz: [0.045, 0.0, 0.007],
            angle_rad: 0,
            range_m: 0.8,
            noise_sigma: 0.016,
          });
        }
      } else if (n < cur) {
        d.distance_sensors.length = Math.max(0, n);
      }
      return d;
    });
  }
  function setLineCount(n: number) {
    patch((d) => {
      const cur = d.line_sensors.length;
      if (n > cur) {
        for (let k = cur; k < n; k++) {
          d.line_sensors.push({ id: `line_${k}`, mount_xy: [-0.024, 0.04] });
        }
      } else if (n < cur) {
        d.line_sensors.length = Math.max(0, n);
      }
      return d;
    });
  }

  // One bundle of updaters shared by the interview wizard. The calibrate form
  // uses the same primitives directly.
  const updaters: SpecUpdaters = {
    setChassis,
    setCom,
    setDrive,
    setDohyo,
    setWedgePresent,
    setSensor,
    setSensorCount,
    setLine,
    setLineCount,
  };

  // Save flow ----------------------------------------------------------------
  const [saving, setSaving] = useState(false);
  const [saveOpen, setSaveOpen] = useState(false);
  const [saveName, setSaveName] = useState('');
  const [saveMsg, setSaveMsg] = useState<{
    kind: 'ok' | 'err';
    text: string;
  } | null>(null);

  async function doSave() {
    if (!spec || !saveName.trim()) return;
    setSaving(true);
    setSaveMsg(null);
    try {
      const rec = await api.saveRobot(saveName.trim(), spec);
      setSaveMsg({ kind: 'ok', text: `SAVED · ${rec.id}` });
      setSaveOpen(false);
      setSaveName('');
    } catch (e) {
      setSaveMsg({
        kind: 'err',
        text: e instanceof ApiError ? e.message : String(e),
      });
    } finally {
      setSaving(false);
    }
  }

  if (seedError) {
    return (
      <Reveal>
        <Panel title="Hardware Spec" live>
          <div className="flex items-center gap-2">
            <StatusDot status="down" />
            <span className="num" style={{ color: 'var(--loss)' }}>
              Could not load default spec: {seedError}
            </span>
          </div>
          <p className="num mt-2 text-fg-2" style={{ fontSize: 12 }}>
            Is the backend running on 127.0.0.1:8000?
          </p>
        </Panel>
      </Reveal>
    );
  }

  if (!spec) {
    return (
      <Panel title="Hardware Spec" live>
        <span className="micro animate-pulse" style={{ color: 'var(--cyan)' }}>
          LOADING SPEC…
        </span>
      </Panel>
    );
  }

  // INTERVIEW MODE — guided wizard on the left, persistent preview on the right.
  if (mode === 'interview') {
    return (
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
        <Reveal index={0}>
          <Interview
            spec={spec}
            updaters={updaters}
            onViewRequest={setPreviewView}
            onFinish={() => setMode('calibrate')}
            onSkip={() => setMode('calibrate')}
          />
        </Reveal>
        <Reveal index={1} className="xl:sticky xl:top-5 xl:self-start">
          <RobotPreview
            geom={geom}
            spec={spec}
            loading={geomLoading}
            error={geomError}
            view={previewView}
            onViewChange={setPreviewView}
          />
        </Reveal>
      </div>
    );
  }

  // CALIBRATE MODE — full detailed form + validation + save bar.
  return (
    <div className="flex flex-col gap-5">
      {/* Save bar */}
      <Reveal index={0}>
        <Panel title="Builder" live bodyClassName="p-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <span className="micro text-fg-2" style={{ fontSize: 10 }}>
                EDITING
              </span>
              <span className="num" style={{ fontSize: 13, color: 'var(--fg-0)' }}>
                {seedName ?? spec.name}
              </span>
              {validation && (
                <span className="num text-fg-2" style={{ fontSize: 10 }}>
                  sig {validation.obs_signature_hash || '—'}
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              {saveMsg && (
                <span
                  className="num"
                  style={{
                    fontSize: 11,
                    color:
                      saveMsg.kind === 'ok' ? 'var(--win)' : 'var(--loss)',
                  }}
                >
                  {saveMsg.text}
                </span>
              )}
              {saveOpen ? (
                <span className="flex items-center gap-2">
                  <input
                    autoFocus
                    value={saveName}
                    onChange={(e) => setSaveName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') doSave();
                      if (e.key === 'Escape') setSaveOpen(false);
                    }}
                    placeholder="robot name"
                    className="num"
                    style={{
                      fontSize: 12,
                      color: 'var(--fg-0)',
                      background: 'var(--bg-2)',
                      border: '1px solid var(--line)',
                      borderRadius: 'var(--radius)',
                      padding: '5px 8px',
                      width: 180,
                    }}
                  />
                  <button
                    className="btn btn-primary"
                    style={{ height: 30 }}
                    onClick={doSave}
                    disabled={saving || !saveName.trim()}
                  >
                    {saving ? 'SAVING…' : 'CONFIRM'}
                  </button>
                  <button
                    className="btn btn-ghost"
                    style={{ height: 30 }}
                    onClick={() => setSaveOpen(false)}
                    disabled={saving}
                  >
                    CANCEL
                  </button>
                </span>
              ) : (
                <button
                  className="btn btn-primary"
                  style={{ height: 30 }}
                  onClick={() => {
                    setSaveName(seedName ?? spec.name);
                    setSaveMsg(null);
                    setSaveOpen(true);
                  }}
                >
                  SAVE ROBOT
                </button>
              )}
              <button
                className="btn btn-ghost"
                style={{ height: 30 }}
                onClick={() => {
                  setPreviewView('TOP');
                  setMode('interview');
                }}
              >
                ← INTERVIEW
              </button>
              <button
                className="btn btn-secondary"
                style={{ height: 30 }}
                onClick={() => navigate('/robots')}
              >
                ROBOTS
              </button>
            </div>
          </div>
        </Panel>
      </Reveal>

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
        {/* Left column: editable spec — shared full hardware editor */}
        <div className="flex flex-col gap-5">
          <HardwareForm
            spec={spec}
            setSpec={(updater) =>
              setSpec((prev) => (prev ? updater(prev) : prev))
            }
            startIndex={1}
          />
        </div>

        {/* Right column: preview + validation */}
        <div className="flex flex-col gap-5">
          <Reveal index={1}>
            <RobotPreview
              geom={geom}
              spec={spec}
              loading={geomLoading}
              error={geomError}
              view={previewView}
              onViewChange={setPreviewView}
            />
          </Reveal>

          <Reveal index={2}>
            <Panel
              title="Observation Contract"
              live={!!validation?.urdf_valid}
              right={
                <span className="flex items-center gap-2">
                  <StatusDot
                    status={
                      validating
                        ? 'warn'
                        : validation?.urdf_valid
                          ? 'ok'
                          : 'down'
                    }
                    pulse={validating}
                  />
                  <span className="micro" style={{ fontSize: 10 }}>
                    {validating ? 'VALIDATING' : 'VALIDATE'}
                  </span>
                </span>
              }
            >
              {validation && (
                <>
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                    <Readout
                      label="OBS DIM"
                      info="obs_dim"
                      value={String(validation.obs_dim)}
                      tone="cyan"
                    />
                    <Readout
                      label="ACTION DIM"
                      info="action_dim"
                      value={String(validation.action_dim)}
                      tone="accent"
                    />
                    <Readout
                      label="URDF"
                      value={validation.urdf_valid ? 'VALID' : 'INVALID'}
                      tone={validation.urdf_valid ? 'win' : 'loss'}
                    />
                    <Readout
                      label="SIGNATURE"
                      info="signature"
                      value={validation.obs_signature_hash || '—'}
                    />
                  </div>

                  {validation.errors.length > 0 && (
                    <div
                      className="mt-3 rounded border p-2"
                      style={{
                        borderColor: 'var(--loss)',
                        background: 'rgba(255,84,112,.08)',
                      }}
                    >
                      <span className="micro" style={{ color: 'var(--loss)' }}>
                        ERRORS
                      </span>
                      <ul className="num mt-1" style={{ fontSize: 11 }}>
                        {validation.errors.map((e, i) => (
                          <li key={i} className="text-fg-1">
                            • {e}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  <div className="mt-4">
                    <span className="micro inline-flex items-center gap-1.5 text-fg-2">
                      FINETUNE CANDIDATES ·{' '}
                      {validation.finetune_candidates.length}
                      <Info topic="finetune" />
                    </span>
                    {validation.finetune_candidates.length === 0 ? (
                      <p
                        className="num mt-1 text-fg-2"
                        style={{ fontSize: 11 }}
                      >
                        No committed checkpoint matches this obs/action
                        contract.
                      </p>
                    ) : (
                      <div className="mt-2 flex flex-col gap-1.5">
                        {validation.finetune_candidates.map((m) => (
                          <div
                            key={m.id}
                            className="flex items-center justify-between rounded border px-2.5 py-1.5"
                            style={{
                              borderColor: 'var(--line)',
                              background: 'var(--bg-2)',
                            }}
                          >
                            <span
                              className="num"
                              style={{ fontSize: 12, color: 'var(--fg-0)' }}
                            >
                              {m.id}
                            </span>
                            <span className="flex items-center gap-3">
                              <span
                                className="micro"
                                style={{
                                  color: 'var(--accent)',
                                  fontSize: 10,
                                }}
                              >
                                {m.algo}
                              </span>
                              <span
                                className="num text-fg-2"
                                style={{ fontSize: 10 }}
                              >
                                {m.obs_dim}/{m.action_dim}
                              </span>
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </>
              )}
            </Panel>
          </Reveal>
        </div>
      </div>
    </div>
  );
}
