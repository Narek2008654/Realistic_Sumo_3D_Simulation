// Shared, full-feature hardware editor — the CHASSIS / CENTER OF MASS / WEDGE /
// DRIVETRAIN / DOHYO / DISTANCE SENSORS / LINE SENSORS field sections, with dual
// slider+number inputs, live positionBounds clamping, and ⓘ help. Extracted from
// the Hardware page's CALIBRATE form so BOTH the robot Hardware builder and the
// Opponents authoring page edit hardware through one identical control set (DRY).
//
// Owns nothing: it's a controlled component driven by {spec, setSpec}. The parent
// keeps the spec in state and is responsible for seeding it (e.g. from
// /api/hardware/default) and for any validate/preview side-effects.

import { Reveal } from './ui';
import { Panel } from './ui';
import { Info } from './Info';
import { SliderField, clamp, positionBounds } from './fields';
import type { DistanceSensor, HardwareSpec } from '../types';

function deg(rad: number): string {
  return `${((rad * 180) / Math.PI).toFixed(0)}°`;
}

export function HardwareForm({
  spec,
  setSpec,
  startIndex = 0,
}: {
  spec: HardwareSpec;
  setSpec: (updater: (prev: HardwareSpec) => HardwareSpec) => void;
  // Reveal stagger offset so the sections animate in after any panels the
  // parent renders above them.
  startIndex?: number;
}) {
  // Immutable spec updaters --------------------------------------------------
  function patch(mut: (draft: HardwareSpec) => HardwareSpec) {
    setSpec((prev) => mut(structuredClone(prev)));
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
  const toggleWedge = () =>
    patch((d) => {
      d.chassis.wedge_present = !d.chassis.wedge_present;
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

  function setLine(i: number, axis: 0 | 1, v: number) {
    patch((d) => {
      d.line_sensors[i].mount_xy[axis] = v;
      return d;
    });
  }
  function addLine() {
    patch((d) => {
      const n = d.line_sensors.length;
      d.line_sensors.push({ id: `line_${n}`, mount_xy: [-0.024, 0.04] });
      return d;
    });
  }
  function removeLine(i: number) {
    patch((d) => {
      d.line_sensors.splice(i, 1);
      return d;
    });
  }

  const c = spec.chassis;
  const dt = spec.drivetrain;
  // Live position bounds derived from the chassis dims — keep CoM/sensor mounts
  // inside the robot body. Recomputed every render so editing the body resizes
  // these ranges immediately.
  const pb = positionBounds(c);

  return (
    <>
      <Reveal index={startIndex}>
        <Panel title="Chassis" live ticks>
          <div className="grid grid-cols-2 gap-x-4 gap-y-3">
            <SliderField
              label="Mass"
              unit="kg"
              info="mass"
              value={c.mass_kg}
              min={0.1}
              max={1.5}
              step={0.01}
              format={(v) => v.toFixed(2)}
              onChange={(v) => setChassis('mass_kg', v)}
            />
            <SliderField
              label="Body length (no wedge)"
              unit="m"
              info="body_length"
              value={c.length_m}
              min={0.04}
              max={0.15}
              step={0.001}
              onChange={(v) => setChassis('length_m', v)}
            />
            <SliderField
              label="Width"
              unit="m"
              info="width"
              value={c.width_m}
              min={0.04}
              max={0.15}
              step={0.001}
              onChange={(v) => setChassis('width_m', v)}
            />
            <SliderField
              label="Height"
              unit="m"
              info="height"
              value={c.height_m}
              min={0.02}
              max={0.12}
              step={0.001}
              onChange={(v) => setChassis('height_m', v)}
            />
            <SliderField
              label="Chassis friction"
              info="chassis_friction"
              value={c.chassis_friction}
              min={0}
              max={2}
              step={0.01}
              format={(v) => v.toFixed(2)}
              onChange={(v) => setChassis('chassis_friction', v)}
            />
            <SliderField
              label="Wheel friction"
              info="wheel_friction"
              value={c.wheel_friction}
              min={0}
              max={4}
              step={0.05}
              format={(v) => v.toFixed(2)}
              onChange={(v) => setChassis('wheel_friction', v)}
            />
          </div>

          <div
            className="mt-3 rounded border p-3"
            style={{ borderColor: 'var(--line)', background: 'var(--bg-2)' }}
          >
            <span className="micro inline-flex items-center gap-1.5 text-fg-2" style={{ fontSize: 9 }}>
              CENTER OF MASS · m
              <Info topic="com" />
            </span>
            <div className="mt-2 grid grid-cols-3 gap-x-4 gap-y-3">
              <SliderField
                label="CoM x"
                info="com_x"
                value={c.com_xyz[0]}
                min={pb.comX.min}
                max={pb.comX.max}
                step={0.0005}
                format={(v) => v.toFixed(4)}
                onChange={(v) => setCom(0, clamp(v, pb.comX.min, pb.comX.max))}
              />
              <SliderField
                label="CoM y"
                info="com_y"
                value={c.com_xyz[1]}
                min={pb.comY.min}
                max={pb.comY.max}
                step={0.0005}
                format={(v) => v.toFixed(4)}
                onChange={(v) => setCom(1, clamp(v, pb.comY.min, pb.comY.max))}
              />
              <SliderField
                label="CoM z"
                info="com_z"
                value={c.com_xyz[2]}
                min={pb.comZ.min}
                max={pb.comZ.max}
                step={0.0005}
                format={(v) => v.toFixed(4)}
                onChange={(v) => setCom(2, clamp(v, pb.comZ.min, pb.comZ.max))}
              />
            </div>
          </div>
        </Panel>
      </Reveal>

      <Reveal index={startIndex + 1}>
        <Panel
          title="Wedge · sharp part"
          live
          ticks
          right={
            <span className="flex items-center gap-2">
              <Info topic="wedge" />
              <button
                className="micro"
                onClick={toggleWedge}
                style={{
                  fontSize: 10,
                  letterSpacing: '.06em',
                  padding: '3px 9px',
                  borderRadius: 'var(--radius)',
                  border: '1px solid var(--line-2)',
                  background: c.wedge_present ? 'var(--accent)' : 'var(--bg-2)',
                  color: c.wedge_present ? 'var(--bg-0)' : 'var(--fg-1)',
                  cursor: 'pointer',
                }}
              >
                {c.wedge_present ? 'PRESENT' : 'ABSENT'}
              </button>
            </span>
          }
        >
          <div
            className="grid grid-cols-1 gap-x-4 gap-y-3 sm:grid-cols-3"
            style={{ opacity: c.wedge_present ? 1 : 0.45 }}
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
              onChange={(v) => setChassis('wedge_length_m', v)}
            />
            <SliderField
              label="Wedge low edge height"
              unit="m"
              info="wedge_low_height"
              value={c.wedge_low_height_m}
              min={0.0}
              max={0.04}
              step={0.0005}
              format={(v) => v.toFixed(4)}
              onChange={(v) => setChassis('wedge_low_height_m', v)}
            />
            <SliderField
              label="Wedge high edge height"
              unit="m"
              info="wedge_high_height"
              value={c.wedge_high_height_m}
              min={0.0}
              max={0.06}
              step={0.0005}
              format={(v) => v.toFixed(4)}
              onChange={(v) => setChassis('wedge_high_height_m', v)}
            />
          </div>
        </Panel>
      </Reveal>

      <Reveal index={startIndex + 2}>
        <Panel title="Drivetrain" live ticks>
          <div className="grid grid-cols-2 gap-x-4 gap-y-3">
            <SliderField
              label="Wheel Radius"
              unit="m"
              info="wheel_radius"
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
              info="track_width"
              value={dt.track_width_m}
              min={0.04}
              max={0.14}
              step={0.001}
              format={(v) => v.toFixed(4)}
              onChange={(v) => setDrive('track_width_m', v)}
            />
            <SliderField
              label="Wheel X Offset"
              unit="m"
              info="wheel_x_offset"
              value={dt.wheel_x_offset_m}
              min={-0.06}
              max={0.06}
              step={0.001}
              format={(v) => v.toFixed(4)}
              onChange={(v) => setDrive('wheel_x_offset_m', v)}
            />
            <SliderField
              label="Max Torque"
              unit="N·m"
              info="max_torque"
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
              info="max_omega"
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

      <Reveal index={startIndex + 3}>
        <Panel title="Dohyo · ring" live ticks>
          <div className="grid grid-cols-2 gap-x-4 gap-y-3">
            <SliderField
              label="Radius"
              unit="m"
              info="dohyo_radius"
              value={spec.dohyo.radius_m}
              min={0.1}
              max={0.8}
              step={0.005}
              format={(v) => v.toFixed(3)}
              onChange={(v) => setDohyo('radius_m', v)}
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
              onChange={(v) => setDohyo('border_width_m', v)}
            />
          </div>
        </Panel>
      </Reveal>

      <Reveal index={startIndex + 4}>
        <Panel
          title={`Distance Sensors · ${spec.distance_sensors.length}`}
          live
          ticks
          right={
            <span className="flex items-center gap-2">
              <Info topic="tof_sensor" />
              <button
                className="btn btn-secondary"
                style={{ height: 26 }}
                onClick={addSensor}
              >
                + Add ToF
              </button>
            </span>
          }
        >
          <div className="flex flex-col gap-3">
            {spec.distance_sensors.map((s, i) => (
              <div
                key={i}
                className="rounded border p-3"
                style={{ borderColor: 'var(--line)', background: 'var(--bg-2)' }}
              >
                <div className="mb-2 flex items-center justify-between">
                  <span className="micro num" style={{ color: 'var(--accent)' }}>
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
                    label="Mount x (fwd)"
                    unit="m"
                    info="tof_mount"
                    value={s.mount_xyz[0]}
                    min={pb.tofX.min}
                    max={pb.tofX.max}
                    step={0.0005}
                    format={(v) => v.toFixed(4)}
                    onChange={(v) =>
                      setSensor(i, (sn) => {
                        sn.mount_xyz[0] = clamp(v, pb.tofX.min, pb.tofX.max);
                      })
                    }
                  />
                  <SliderField
                    label="Mount y (left)"
                    unit="m"
                    info="tof_mount"
                    value={s.mount_xyz[1]}
                    min={pb.tofY.min}
                    max={pb.tofY.max}
                    step={0.0005}
                    format={(v) => v.toFixed(4)}
                    onChange={(v) =>
                      setSensor(i, (sn) => {
                        sn.mount_xyz[1] = clamp(v, pb.tofY.min, pb.tofY.max);
                      })
                    }
                  />
                  <SliderField
                    label="Mount z (up)"
                    unit="m"
                    info="tof_mount"
                    value={s.mount_xyz[2]}
                    min={pb.tofZ.min}
                    max={pb.tofZ.max}
                    step={0.0005}
                    format={(v) => v.toFixed(4)}
                    onChange={(v) =>
                      setSensor(i, (sn) => {
                        sn.mount_xyz[2] = clamp(v, pb.tofZ.min, pb.tofZ.max);
                      })
                    }
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
                    onChange={(v) =>
                      setSensor(i, (sn) => {
                        sn.range_m = v;
                      })
                    }
                  />
                  <div className="col-span-2">
                    <SliderField
                      label="Facing angle"
                      unit="°"
                      info="tof_angle"
                      value={(s.angle_rad * 180) / Math.PI}
                      min={-180}
                      max={180}
                      step={1}
                      format={(v) => `${v.toFixed(0)}°`}
                      onChange={(v) =>
                        setSensor(i, (sn) => {
                          sn.angle_rad = (v * Math.PI) / 180;
                        })
                      }
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Panel>
      </Reveal>

      <Reveal index={startIndex + 5}>
        <Panel
          title={`Line Sensors · ${spec.line_sensors.length}`}
          live
          ticks
          right={
            <span className="flex items-center gap-2">
              <Info topic="line_sensor" />
              <button
                className="btn btn-secondary"
                style={{ height: 26 }}
                onClick={addLine}
              >
                + Add Line
              </button>
            </span>
          }
        >
          <div className="flex flex-col gap-3">
            {spec.line_sensors.map((l, i) => (
              <div
                key={i}
                className="rounded border p-3"
                style={{ borderColor: 'var(--line)', background: 'var(--bg-2)' }}
              >
                <div className="mb-2 flex items-center justify-between">
                  <span className="micro num" style={{ color: 'var(--cyan)' }}>
                    {l.id}
                  </span>
                  <button
                    className="btn btn-ghost"
                    style={{ height: 24, padding: '0 6px' }}
                    onClick={() => removeLine(i)}
                    title="Remove line sensor"
                  >
                    ✕
                  </button>
                </div>
                <div className="grid grid-cols-2 gap-x-4 gap-y-3">
                  <SliderField
                    label="Mount x (fwd)"
                    unit="m"
                    info="line_mount"
                    value={l.mount_xy[0]}
                    min={pb.lineX.min}
                    max={pb.lineX.max}
                    step={0.0005}
                    format={(v) => v.toFixed(4)}
                    onChange={(v) => setLine(i, 0, clamp(v, pb.lineX.min, pb.lineX.max))}
                  />
                  <SliderField
                    label="Mount y (left)"
                    unit="m"
                    info="line_mount"
                    value={l.mount_xy[1]}
                    min={pb.lineY.min}
                    max={pb.lineY.max}
                    step={0.0005}
                    format={(v) => v.toFixed(4)}
                    onChange={(v) => setLine(i, 1, clamp(v, pb.lineY.min, pb.lineY.max))}
                  />
                </div>
              </div>
            ))}
            {spec.line_sensors.length === 0 && (
              <p className="num text-fg-2" style={{ fontSize: 11 }}>
                No line sensors. Add one to detect the dohyo border ring.
              </p>
            )}
          </div>
        </Panel>
      </Reveal>
    </>
  );
}
