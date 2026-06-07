// Hardware Builder: seed a HardwareSpec from the backend, edit chassis /
// drivetrain / distance sensors, live 3D preview + debounced validate.
import { useCallback, useEffect, useRef, useState } from 'react';
import { api, ApiError } from '../api';
import { Readout, SliderField } from '../components/fields';
import { RobotPreview } from '../components/RobotPreview';
import { Panel, Reveal, StatusDot } from '../components/ui';
import type {
  DistanceSensor,
  Geometry,
  HardwareSpec,
  ValidateResult,
} from '../types';

const DEBOUNCE_MS = 350;

function deg(rad: number): string {
  return `${((rad * 180) / Math.PI).toFixed(0)}°`;
}

export default function Hardware() {
  const [spec, setSpec] = useState<HardwareSpec | null>(null);
  const [seedError, setSeedError] = useState<string | null>(null);

  const [geom, setGeom] = useState<Geometry | null>(null);
  const [geomError, setGeomError] = useState<string | null>(null);
  const [geomLoading, setGeomLoading] = useState(false);

  const [validation, setValidation] = useState<ValidateResult | null>(null);
  const [validating, setValidating] = useState(false);

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Seed from backend on mount.
  useEffect(() => {
    api
      .hardwareDefault()
      .then(setSpec)
      .catch((e) =>
        setSeedError(e instanceof ApiError ? e.message : String(e)),
      );
  }, []);

  // Debounced validate + geometry whenever the spec changes.
  const refresh = useCallback((s: HardwareSpec) => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      setValidating(true);
      setGeomLoading(true);
      const [v, g] = await Promise.allSettled([
        api.validate(s),
        api.geometry(s),
      ]);
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
  const setDrive = (k: keyof HardwareSpec['drivetrain'], v: number) =>
    patch((d) => {
      (d.drivetrain as unknown as Record<string, unknown>)[k] = v;
      return d;
    });

  function setSensor(i: number, mut: (s: DistanceSensor) => void) {
    patch((d) => {
      mut(d.distance_sensors[i]);
      return d;
    });
  }
  function addSensor() {
    patch((d) => {
      const n = d.distance_sensors.length;
      d.distance_sensors.push({
        id: `tof_${n}`,
        mount_xyz: [0.045, 0.0, 0.007],
        angle_rad: 0,
        range_m: 0.8,
        noise_sigma: 0.016,
      });
      return d;
    });
  }
  function removeSensor(i: number) {
    patch((d) => {
      d.distance_sensors.splice(i, 1);
      return d;
    });
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

  const c = spec.chassis;
  const dt = spec.drivetrain;

  return (
    <div className="grid grid-cols-1 gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
      {/* Left column: editable spec */}
      <div className="flex flex-col gap-5">
        <Reveal index={0}>
          <Panel title="Chassis" live ticks>
            <div className="grid grid-cols-2 gap-x-4 gap-y-3">
              <SliderField
                label="Mass"
                unit="kg"
                value={c.mass_kg}
                min={0.1}
                max={1.5}
                step={0.01}
                format={(v) => v.toFixed(2)}
                onChange={(v) => setChassis('mass_kg', v)}
              />
              <SliderField
                label="Length"
                unit="m"
                value={c.length_m}
                min={0.04}
                max={0.15}
                step={0.001}
                onChange={(v) => setChassis('length_m', v)}
              />
              <SliderField
                label="Width"
                unit="m"
                value={c.width_m}
                min={0.04}
                max={0.15}
                step={0.001}
                onChange={(v) => setChassis('width_m', v)}
              />
              <SliderField
                label="Height"
                unit="m"
                value={c.height_m}
                min={0.02}
                max={0.12}
                step={0.001}
                onChange={(v) => setChassis('height_m', v)}
              />
            </div>
          </Panel>
        </Reveal>

        <Reveal index={1}>
          <Panel title="Drivetrain" live ticks>
            <div className="grid grid-cols-2 gap-x-4 gap-y-3">
              <SliderField
                label="Wheel Radius"
                unit="m"
                value={dt.wheel_radius_m}
                min={0.005}
                max={0.03}
                step={0.0005}
                format={(v) => v.toFixed(4)}
                onChange={(v) => setDrive('wheel_radius_m', v)}
              />
              <SliderField
                label="Track Width"
                unit="m"
                value={dt.track_width_m}
                min={0.04}
                max={0.14}
                step={0.001}
                format={(v) => v.toFixed(4)}
                onChange={(v) => setDrive('track_width_m', v)}
              />
              <SliderField
                label="Max Torque"
                unit="N·m"
                value={dt.max_torque_nm}
                min={0.02}
                max={0.5}
                step={0.005}
                format={(v) => v.toFixed(3)}
                onChange={(v) => setDrive('max_torque_nm', v)}
              />
              <SliderField
                label="Max Omega"
                unit="rad/s"
                value={dt.max_omega_rad_s}
                min={5}
                max={80}
                step={0.5}
                format={(v) => v.toFixed(1)}
                onChange={(v) => setDrive('max_omega_rad_s', v)}
              />
            </div>
          </Panel>
        </Reveal>

        <Reveal index={2}>
          <Panel
            title={`Distance Sensors · ${spec.distance_sensors.length}`}
            live
            ticks
            right={
              <button
                className="btn btn-secondary"
                style={{ height: 26 }}
                onClick={addSensor}
              >
                + Add ToF
              </button>
            }
          >
            <div className="flex flex-col gap-3">
              {spec.distance_sensors.map((s, i) => (
                <div
                  key={i}
                  className="rounded border p-3"
                  style={{
                    borderColor: 'var(--line)',
                    background: 'var(--bg-2)',
                  }}
                >
                  <div className="mb-2 flex items-center justify-between">
                    <span
                      className="micro num"
                      style={{ color: 'var(--accent)' }}
                    >
                      {s.id} · {deg(s.angle_rad)}
                    </span>
                    <button
                      className="btn btn-ghost"
                      style={{ height: 24, padding: '0 6px' }}
                      onClick={() => removeSensor(i)}
                      disabled={spec.distance_sensors.length <= 1}
                      title={
                        spec.distance_sensors.length <= 1
                          ? 'At least one sensor required'
                          : 'Remove sensor'
                      }
                    >
                      ✕
                    </button>
                  </div>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-3">
                    <SliderField
                      label="Angle"
                      unit="rad"
                      value={s.angle_rad}
                      min={-Math.PI / 2}
                      max={Math.PI / 2}
                      step={0.0175}
                      format={(v) => `${v.toFixed(2)} (${deg(v)})`}
                      onChange={(v) =>
                        setSensor(i, (sn) => {
                          sn.angle_rad = v;
                        })
                      }
                    />
                    <SliderField
                      label="Range"
                      unit="m"
                      value={s.range_m}
                      min={0.2}
                      max={2.0}
                      step={0.01}
                      format={(v) => v.toFixed(2)}
                      onChange={(v) =>
                        setSensor(i, (sn) => {
                          sn.range_m = v;
                        })
                      }
                    />
                  </div>
                </div>
              ))}
            </div>
          </Panel>
        </Reveal>

        <Reveal index={3}>
          <Panel title="Line Sensors" live>
            <div className="grid grid-cols-2 gap-3">
              {spec.line_sensors.map((l, i) => (
                <div
                  key={i}
                  className="rounded border p-2"
                  style={{
                    borderColor: 'var(--line)',
                    background: 'var(--bg-2)',
                  }}
                >
                  <span className="micro num" style={{ color: 'var(--cyan)' }}>
                    {l.id}
                  </span>
                  <div className="num mt-1 text-fg-2" style={{ fontSize: 11 }}>
                    x {l.mount_xy[0].toFixed(4)} · y {l.mount_xy[1].toFixed(4)}
                  </div>
                </div>
              ))}
            </div>
          </Panel>
        </Reveal>
      </div>

      {/* Right column: preview + validation */}
      <div className="flex flex-col gap-5">
        <Reveal index={1}>
          <RobotPreview
            geom={geom}
            loading={geomLoading}
            error={geomError}
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
                    value={String(validation.obs_dim)}
                    tone="cyan"
                  />
                  <Readout
                    label="ACTION DIM"
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
                  <span className="micro text-fg-2">
                    FINETUNE CANDIDATES ·{' '}
                    {validation.finetune_candidates.length}
                  </span>
                  {validation.finetune_candidates.length === 0 ? (
                    <p
                      className="num mt-1 text-fg-2"
                      style={{ fontSize: 11 }}
                    >
                      No committed checkpoint matches this obs/action contract.
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
  );
}
