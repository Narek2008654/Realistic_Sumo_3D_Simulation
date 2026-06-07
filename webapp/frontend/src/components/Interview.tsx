// Guided hardware INTERVIEW (wizard). One concern per step, plain-language
// wording, sensible defaults pre-filled from the spec. Each step shows only a
// few dual slider+number inputs and drives the shared 3D preview (view +
// highlight) in the parent. After the final Review step it calls onFinish to
// hand off to the detailed CALIBRATE form.
//
// The interview is a thin editing surface over the SAME HardwareSpec the
// calibrate form edits — it uses the parent's immutable updaters so whatever
// the user builds here flows straight into calibrate + save with no conversion.

import { AnimatePresence, motion } from 'framer-motion';
import { useEffect, useState } from 'react';
import { SliderField } from './fields';
import { Info } from './Info';
import { CornerTicks } from './ui';
import type { DistanceSensor, HardwareSpec, PreviewView } from '../types';

const deg = (rad: number) => `${((rad * 180) / Math.PI).toFixed(0)}°`;

// Updater bundle handed down from the Hardware page so the interview and the
// calibrate form mutate the spec through identical code paths.
export interface SpecUpdaters {
  setChassis: (k: keyof HardwareSpec['chassis'], v: number) => void;
  setCom: (axis: 0 | 1 | 2, v: number) => void;
  setDrive: (k: keyof HardwareSpec['drivetrain'], v: number) => void;
  setDohyo: (k: keyof HardwareSpec['dohyo'], v: number) => void;
  setWedgePresent: (present: boolean) => void;
  setSensor: (i: number, mut: (s: DistanceSensor) => void) => void;
  setSensorCount: (n: number) => void;
  setLine: (i: number, axis: 0 | 1, v: number) => void;
  setLineCount: (n: number) => void;
}

interface StepDef {
  key: string;
  title: string; // uppercase micro label
  question: string; // plain-language headline
  blurb: string; // one-line helper
  info: string; // glossary key for the step's plain-language intro
  view: PreviewView; // which camera the preview should show
  // Render the step's inputs. Returns the focused control cluster.
  render: (spec: HardwareSpec, u: SpecUpdaters) => React.ReactNode;
}

// A small stepper of named "counter" choices (how many sensors).
function CountPicker({
  label,
  value,
  min,
  max,
  onChange,
  info,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (n: number) => void;
  info?: string;
}) {
  const opts = [];
  for (let n = min; n <= max; n++) opts.push(n);
  return (
    <div className="flex flex-col gap-2">
      <span className="micro inline-flex items-center gap-1.5 text-fg-2" style={{ fontSize: 10 }}>
        {label}
        {info && <Info topic={info} />}
      </span>
      <div className="flex flex-wrap gap-2">
        {opts.map((n) => {
          const active = n === value;
          return (
            <button
              key={n}
              onClick={() => onChange(n)}
              className="num"
              style={{
                width: 40,
                height: 36,
                fontSize: 15,
                color: active ? 'var(--bg-0)' : 'var(--fg-1)',
                background: active ? 'var(--accent)' : 'var(--bg-2)',
                border: `1px solid ${active ? 'var(--accent)' : 'var(--line)'}`,
                borderRadius: 'var(--radius)',
                cursor: 'pointer',
              }}
            >
              {n}
            </button>
          );
        })}
      </div>
    </div>
  );
}

// Per-sensor sub-block reused inside the sensor steps.
function ToFEditor({
  spec,
  u,
  i,
}: {
  spec: HardwareSpec;
  u: SpecUpdaters;
  i: number;
}) {
  const s = spec.distance_sensors[i];
  if (!s) return null;
  return (
    <div
      className="rounded border p-3"
      style={{ borderColor: 'var(--line)', background: 'var(--bg-2)' }}
    >
      <span className="micro num mb-2 inline-block" style={{ color: 'var(--accent)' }}>
        {s.id} · {deg(s.angle_rad)}
      </span>
      <div className="grid grid-cols-2 gap-x-4 gap-y-3">
        <SliderField
          label="Mount x (fwd)"
          unit="m"
          info="tof_mount"
          value={s.mount_xyz[0]}
          min={-0.08}
          max={0.08}
          step={0.0005}
          format={(v) => v.toFixed(4)}
          onChange={(v) => u.setSensor(i, (sn) => { sn.mount_xyz[0] = v; })}
        />
        <SliderField
          label="Mount y (left)"
          unit="m"
          info="tof_mount"
          value={s.mount_xyz[1]}
          min={-0.06}
          max={0.06}
          step={0.0005}
          format={(v) => v.toFixed(4)}
          onChange={(v) => u.setSensor(i, (sn) => { sn.mount_xyz[1] = v; })}
        />
        <SliderField
          label="Mount z (up)"
          unit="m"
          info="tof_mount"
          value={s.mount_xyz[2]}
          min={-0.02}
          max={0.06}
          step={0.0005}
          format={(v) => v.toFixed(4)}
          onChange={(v) => u.setSensor(i, (sn) => { sn.mount_xyz[2] = v; })}
        />
        <SliderField
          label="Range"
          unit="m"
          info="tof_range"
          value={s.range_m}
          min={0.2}
          max={2.0}
          step={0.01}
          format={(v) => v.toFixed(2)}
          onChange={(v) => u.setSensor(i, (sn) => { sn.range_m = v; })}
        />
        <div className="col-span-2">
          <SliderField
            label="Facing angle (yaw)"
            unit="rad"
            info="tof_angle"
            value={s.angle_rad}
            min={-Math.PI}
            max={Math.PI}
            step={0.0175}
            format={(v) => `${v.toFixed(2)} (${deg(v)})`}
            onChange={(v) => u.setSensor(i, (sn) => { sn.angle_rad = v; })}
          />
        </div>
      </div>
    </div>
  );
}

function LineEditor({
  spec,
  u,
  i,
}: {
  spec: HardwareSpec;
  u: SpecUpdaters;
  i: number;
}) {
  const l = spec.line_sensors[i];
  if (!l) return null;
  return (
    <div
      className="rounded border p-3"
      style={{ borderColor: 'var(--line)', background: 'var(--bg-2)' }}
    >
      <span className="micro num mb-2 inline-block" style={{ color: 'var(--cyan)' }}>
        {l.id}
      </span>
      <div className="grid grid-cols-2 gap-x-4 gap-y-3">
        <SliderField
          label="Mount x (fwd)"
          unit="m"
          info="line_mount"
          value={l.mount_xy[0]}
          min={-0.08}
          max={0.08}
          step={0.0005}
          format={(v) => v.toFixed(4)}
          onChange={(v) => u.setLine(i, 0, v)}
        />
        <SliderField
          label="Mount y (left)"
          unit="m"
          info="line_mount"
          value={l.mount_xy[1]}
          min={-0.08}
          max={0.08}
          step={0.0005}
          format={(v) => v.toFixed(4)}
          onChange={(v) => u.setLine(i, 1, v)}
        />
      </div>
    </div>
  );
}

// ---- Step definitions ------------------------------------------------------
const STEPS: StepDef[] = [
  {
    key: 'body',
    title: 'Step 1 · Robot body',
    question: 'How big is the robot body?',
    blurb: 'The main chassis box — not counting the wedge up front.',
    info: 'step_body',
    view: 'TOP',
    render: (spec, u) => {
      const c = spec.chassis;
      return (
        <div className="grid grid-cols-1 gap-x-4 gap-y-4 sm:grid-cols-2">
          <SliderField
            label="Weight"
            unit="kg"
            info="mass"
            value={c.mass_kg}
            min={0.1}
            max={1.5}
            step={0.01}
            format={(v) => v.toFixed(2)}
            onChange={(v) => u.setChassis('mass_kg', v)}
          />
          <SliderField
            label="Body length (no wedge)"
            unit="m"
            info="body_length"
            value={c.length_m}
            min={0.04}
            max={0.15}
            step={0.001}
            onChange={(v) => u.setChassis('length_m', v)}
          />
          <SliderField
            label="Width"
            unit="m"
            info="width"
            value={c.width_m}
            min={0.04}
            max={0.15}
            step={0.001}
            onChange={(v) => u.setChassis('width_m', v)}
          />
          <SliderField
            label="Height"
            unit="m"
            info="height"
            value={c.height_m}
            min={0.02}
            max={0.12}
            step={0.001}
            onChange={(v) => u.setChassis('height_m', v)}
          />
        </div>
      );
    },
  },
  {
    key: 'drivetrain',
    title: 'Step 2 · Drivetrain',
    question: 'How does it drive?',
    blurb: 'Wheels and motors. Bigger wheels + faster spin = more speed.',
    info: 'step_drivetrain',
    view: 'TOP',
    render: (spec, u) => {
      const dt = spec.drivetrain;
      return (
        <div className="grid grid-cols-1 gap-x-4 gap-y-4 sm:grid-cols-2">
          <SliderField
            label="Wheel radius"
            unit="m"
            info="wheel_radius"
            value={dt.wheel_radius_m}
            min={0.005}
            max={0.03}
            step={0.0005}
            format={(v) => v.toFixed(4)}
            onChange={(v) => u.setDrive('wheel_radius_m', v)}
          />
          <SliderField
            label="Track width (wheel spacing)"
            unit="m"
            info="track_width"
            value={dt.track_width_m}
            min={0.04}
            max={0.14}
            step={0.001}
            format={(v) => v.toFixed(4)}
            onChange={(v) => u.setDrive('track_width_m', v)}
          />
          <SliderField
            label="Wheel front/back offset"
            unit="m"
            info="wheel_x_offset"
            value={dt.wheel_x_offset_m}
            min={-0.06}
            max={0.06}
            step={0.001}
            format={(v) => v.toFixed(4)}
            onChange={(v) => u.setDrive('wheel_x_offset_m', v)}
          />
          <SliderField
            label="Pushing power (max torque)"
            unit="N·m"
            info="max_torque"
            value={dt.max_torque_nm}
            min={0.02}
            max={0.5}
            step={0.005}
            format={(v) => v.toFixed(3)}
            onChange={(v) => u.setDrive('max_torque_nm', v)}
          />
          <SliderField
            label="How fast the wheels spin"
            unit="rad/s"
            info="max_omega"
            value={dt.max_omega_rad_s}
            min={5}
            max={80}
            step={0.5}
            format={(v) => v.toFixed(1)}
            onChange={(v) => u.setDrive('max_omega_rad_s', v)}
          />
        </div>
      );
    },
  },
  {
    key: 'wedge',
    title: 'Step 3 · Wedge / plow',
    question: 'Does it have a wedge up front?',
    blurb: 'A sloped plow scoops opponents up. Toggle it off for a flat front.',
    info: 'step_wedge',
    view: 'TOP',
    render: (spec, u) => {
      const c = spec.chassis;
      return (
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-3">
            <span className="micro inline-flex items-center gap-1.5 text-fg-2" style={{ fontSize: 10 }}>
              WEDGE
              <Info topic="wedge" />
            </span>
            <div
              className="flex overflow-hidden rounded border"
              style={{ borderColor: 'var(--line)' }}
            >
              {([true, false] as const).map((on) => {
                const active = c.wedge_present === on;
                return (
                  <button
                    key={String(on)}
                    onClick={() => u.setWedgePresent(on)}
                    className="micro"
                    style={{
                      fontSize: 10,
                      padding: '6px 14px',
                      color: active ? 'var(--bg-0)' : 'var(--fg-1)',
                      background: active ? 'var(--accent)' : 'var(--bg-2)',
                      border: 'none',
                      cursor: 'pointer',
                    }}
                  >
                    {on ? 'YES' : 'NO'}
                  </button>
                );
              })}
            </div>
          </div>
          <div
            className="grid grid-cols-1 gap-x-4 gap-y-4 sm:grid-cols-3"
            style={{ opacity: c.wedge_present ? 1 : 0.4, pointerEvents: c.wedge_present ? 'auto' : 'none' }}
          >
            <SliderField
              label="Wedge length"
              unit="m"
              info="wedge_length"
              value={c.wedge_length_m}
              min={0.0}
              max={0.08}
              step={0.0005}
              format={(v) => v.toFixed(4)}
              onChange={(v) => u.setChassis('wedge_length_m', v)}
            />
            <SliderField
              label="Low edge (front tip)"
              unit="m"
              info="wedge_low_height"
              value={c.wedge_low_height_m}
              min={0.0}
              max={0.04}
              step={0.0005}
              format={(v) => v.toFixed(4)}
              onChange={(v) => u.setChassis('wedge_low_height_m', v)}
            />
            <SliderField
              label="High edge (back of wedge)"
              unit="m"
              info="wedge_high_height"
              value={c.wedge_high_height_m}
              min={0.0}
              max={0.06}
              step={0.0005}
              format={(v) => v.toFixed(4)}
              onChange={(v) => u.setChassis('wedge_high_height_m', v)}
            />
          </div>
        </div>
      );
    },
  },
  {
    key: 'com',
    title: 'Step 4 · Center of mass',
    question: 'Where does it balance?',
    blurb: 'Drag these to move the magenta ⊕ marker. Low + forward = harder to flip.',
    info: 'step_com',
    view: 'TOP',
    render: (spec, u) => {
      const c = spec.chassis;
      return (
        <div className="grid grid-cols-1 gap-x-4 gap-y-4 sm:grid-cols-3">
          <SliderField
            label="Forward / back (x)"
            unit="m"
            info="com_x"
            value={c.com_xyz[0]}
            min={-0.05}
            max={0.05}
            step={0.0005}
            format={(v) => v.toFixed(4)}
            onChange={(v) => u.setCom(0, v)}
          />
          <SliderField
            label="Left / right (y)"
            unit="m"
            info="com_y"
            value={c.com_xyz[1]}
            min={-0.05}
            max={0.05}
            step={0.0005}
            format={(v) => v.toFixed(4)}
            onChange={(v) => u.setCom(1, v)}
          />
          <SliderField
            label="Up / down (z)"
            unit="m"
            info="com_z"
            value={c.com_xyz[2]}
            min={-0.02}
            max={0.06}
            step={0.0005}
            format={(v) => v.toFixed(4)}
            onChange={(v) => u.setCom(2, v)}
          />
        </div>
      );
    },
  },
  {
    key: 'tof',
    title: 'Step 5 · Forward sensors',
    question: 'How does it see the opponent?',
    blurb: 'Distance (ToF) sensors. Each shows a marker + a ray you can aim.',
    info: 'step_tof',
    view: 'TOP',
    render: (spec, u) => (
      <div className="flex flex-col gap-4">
        <CountPicker
          label="How many distance sensors?"
          info="tof_sensor"
          value={spec.distance_sensors.length}
          min={1}
          max={6}
          onChange={u.setSensorCount}
        />
        <div className="flex flex-col gap-3">
          {spec.distance_sensors.map((_, i) => (
            <ToFEditor key={i} spec={spec} u={u} i={i} />
          ))}
        </div>
      </div>
    ),
  },
  {
    key: 'line',
    title: 'Step 6 · Edge sensors',
    question: 'How does it avoid the edge?',
    blurb: 'Line sensors on the underside detect the ring border. View flipped underneath.',
    info: 'step_line',
    view: 'UNDERSIDE',
    render: (spec, u) => (
      <div className="flex flex-col gap-4">
        <CountPicker
          label="How many line sensors?"
          info="line_sensor"
          value={spec.line_sensors.length}
          min={0}
          max={6}
          onChange={u.setLineCount}
        />
        {spec.line_sensors.length === 0 ? (
          <p className="num text-fg-2" style={{ fontSize: 11 }}>
            No line sensors — the robot can't detect the border ring. Add at
            least one to avoid driving itself out.
          </p>
        ) : (
          <div className="flex flex-col gap-3">
            {spec.line_sensors.map((_, i) => (
              <LineEditor key={i} spec={spec} u={u} i={i} />
            ))}
          </div>
        )}
      </div>
    ),
  },
  {
    key: 'dohyo',
    title: 'Step 7 · Dohyo ring',
    question: 'How big is the ring?',
    blurb: 'The arena it fights in — radius and the painted border width.',
    info: 'step_dohyo',
    view: 'TOP',
    render: (spec, u) => (
      <div className="grid grid-cols-1 gap-x-4 gap-y-4 sm:grid-cols-2">
        <SliderField
          label="Ring radius"
          unit="m"
          info="dohyo_radius"
          value={spec.dohyo.radius_m}
          min={0.1}
          max={0.8}
          step={0.005}
          format={(v) => v.toFixed(3)}
          onChange={(v) => u.setDohyo('radius_m', v)}
        />
        <SliderField
          label="Border width"
          unit="m"
          info="dohyo_border"
          value={spec.dohyo.border_width_m}
          min={0.005}
          max={0.1}
          step={0.001}
          format={(v) => v.toFixed(3)}
          onChange={(v) => u.setDohyo('border_width_m', v)}
        />
      </div>
    ),
  },
];

function ReviewRow({ label, value }: { label: string; value: string }) {
  return (
    <div
      className="flex items-center justify-between rounded border px-3 py-2"
      style={{ borderColor: 'var(--line)', background: 'var(--bg-2)' }}
    >
      <span className="micro text-fg-2" style={{ fontSize: 10 }}>
        {label}
      </span>
      <span className="num" style={{ fontSize: 12, color: 'var(--fg-0)' }}>
        {value}
      </span>
    </div>
  );
}

export function Interview({
  spec,
  updaters,
  onViewRequest,
  onFinish,
  onSkip,
}: {
  spec: HardwareSpec;
  updaters: SpecUpdaters;
  onViewRequest: (v: PreviewView) => void;
  onFinish: () => void;
  onSkip: () => void;
}) {
  const total = STEPS.length + 1; // + review
  const [idx, setIdx] = useState(0);
  const isReview = idx === STEPS.length;

  // Drive the shared preview camera to match the active step.
  useEffect(() => {
    onViewRequest(isReview ? 'TOP' : STEPS[idx].view);
  }, [idx, isReview, onViewRequest]);

  const c = spec.chassis;

  return (
    <div className="panel panel-live relative flex h-full flex-col">
      <CornerTicks />

      {/* progress header */}
      <div className="panel-head flex-col items-stretch gap-2">
        <div className="flex items-center justify-between">
          <span className="micro text-fg-1">HARDWARE INTERVIEW</span>
          <button className="btn btn-ghost" style={{ height: 24, fontSize: 10 }} onClick={onSkip}>
            SKIP · ADVANCED →
          </button>
        </div>
        <div className="flex items-center gap-1.5">
          {Array.from({ length: total }).map((_, i) => (
            <div
              key={i}
              className="h-1 flex-1 rounded"
              style={{
                background:
                  i < idx
                    ? 'var(--accent)'
                    : i === idx
                      ? 'var(--accent)'
                      : 'var(--bg-3)',
                opacity: i === idx ? 1 : i < idx ? 0.55 : 1,
              }}
            />
          ))}
        </div>
        <span className="num text-fg-2" style={{ fontSize: 10 }}>
          {idx + 1} / {total}
        </span>
      </div>

      <div className="flex flex-1 flex-col p-4">
        <AnimatePresence mode="wait">
          <motion.div
            key={isReview ? 'review' : STEPS[idx].key}
            initial={{ opacity: 0, x: 18 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -18 }}
            transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
            className="flex flex-1 flex-col"
          >
            {isReview ? (
              <>
                <span className="micro" style={{ color: 'var(--cyan)', fontSize: 10 }}>
                  Step {total} · Review
                </span>
                <h3
                  className="mt-1"
                  style={{ fontSize: 20, color: 'var(--fg-0)', letterSpacing: '.02em' }}
                >
                  Ready to build.
                </h3>
                <p className="num mt-1 text-fg-1" style={{ fontSize: 12 }}>
                  Here's the robot you described. Build it and fine-tune anything
                  in the detailed Calibrate view.
                </p>
                <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2">
                  <ReviewRow label="Weight" value={`${c.mass_kg.toFixed(2)} kg`} />
                  <ReviewRow
                    label="Body L×W×H"
                    value={`${c.length_m.toFixed(3)} × ${c.width_m.toFixed(3)} × ${c.height_m.toFixed(3)} m`}
                  />
                  <ReviewRow
                    label="Wedge"
                    value={c.wedge_present ? `yes · ${c.wedge_length_m.toFixed(3)} m` : 'none'}
                  />
                  <ReviewRow
                    label="CoM (x,y,z)"
                    value={`${c.com_xyz[0].toFixed(3)}, ${c.com_xyz[1].toFixed(3)}, ${c.com_xyz[2].toFixed(3)}`}
                  />
                  <ReviewRow
                    label="Wheel radius"
                    value={`${spec.drivetrain.wheel_radius_m.toFixed(4)} m`}
                  />
                  <ReviewRow
                    label="Max spin"
                    value={`${spec.drivetrain.max_omega_rad_s.toFixed(1)} rad/s`}
                  />
                  <ReviewRow
                    label="Distance sensors"
                    value={String(spec.distance_sensors.length)}
                  />
                  <ReviewRow
                    label="Line sensors"
                    value={String(spec.line_sensors.length)}
                  />
                  <ReviewRow
                    label="Dohyo radius"
                    value={`${spec.dohyo.radius_m.toFixed(3)} m`}
                  />
                  <ReviewRow
                    label="Border width"
                    value={`${spec.dohyo.border_width_m.toFixed(3)} m`}
                  />
                </div>
              </>
            ) : (
              <>
                <span className="micro" style={{ color: 'var(--cyan)', fontSize: 10 }}>
                  {STEPS[idx].title}
                </span>
                <h3
                  className="mt-1 inline-flex items-center gap-2"
                  style={{ fontSize: 20, color: 'var(--fg-0)', letterSpacing: '.02em' }}
                >
                  {STEPS[idx].question}
                  <Info topic={STEPS[idx].info} color="var(--cyan)" />
                </h3>
                <p className="num mt-1 text-fg-1" style={{ fontSize: 12 }}>
                  {STEPS[idx].blurb}
                </p>
                <div className="mt-5">{STEPS[idx].render(spec, updaters)}</div>
              </>
            )}
          </motion.div>
        </AnimatePresence>

        {/* nav footer */}
        <div className="mt-6 flex items-center justify-between border-t pt-4" style={{ borderColor: 'var(--line)' }}>
          <button
            className="btn btn-secondary"
            style={{ height: 32 }}
            onClick={() => setIdx((i) => Math.max(0, i - 1))}
            disabled={idx === 0}
          >
            ← BACK
          </button>
          {isReview ? (
            <button className="btn btn-primary" style={{ height: 32 }} onClick={onFinish}>
              BUILD · CONTINUE TO CALIBRATE →
            </button>
          ) : (
            <button
              className="btn btn-primary"
              style={{ height: 32 }}
              onClick={() => setIdx((i) => Math.min(STEPS.length, i + 1))}
            >
              NEXT →
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
