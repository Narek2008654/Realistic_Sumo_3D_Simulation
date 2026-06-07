// HUD-framed 3D preview of a HardwareSpec, rendered from the backend's
// `/api/hardware/geometry` primitives. Builds the URDF kinematic tree from the
// joint list, composes each joint's origin to place every visual link, and
// draws box/cylinder primitives. Dohyo disc floor (sized from spec.dohyo),
// grid, vignette, corner brackets. Agent body is forge-orange (--accent);
// wheels/details stay neutral.
//
// Sensor overlays come from the SPEC (not the geometry): each distance (ToF)
// sensor draws a small marker at its mount + a forge-orange direction ray of
// length range_m along angle_rad in the robot XY plane; each line sensor draws
// a cyan marker on the chassis underside. A TOP / UNDERSIDE toggle flips the
// camera under the dohyo so the line-sensor footprints over the border ring
// are visible.

import { Canvas } from '@react-three/fiber';
import {
  ContactShadows,
  Environment,
  Grid,
  OrbitControls,
} from '@react-three/drei';
import { useMemo, useState } from 'react';
import * as THREE from 'three';
import type {
  DistanceSensor,
  Geometry,
  GeomLink,
  HardwareSpec,
  LineSensor,
  Vec3,
} from '../types';
import { CornerTicks } from './ui';

const ACCENT = '#ff7a18';
const CYAN = '#2ad4ff';

// URDF/PyBullet are Z-up; three.js is Y-up. We rotate the whole rig -90deg
// about X so the robot stands on the floor and reads naturally. Sensor overlays
// live INSIDE the same rotated group, so spec body-frame coords (x fwd / y left
// / z up) line up with the chassis automatically.
const Z_UP_TO_Y_UP = new THREE.Euler(-Math.PI / 2, 0, 0);

/** Local transform of a joint/visual origin (xyz + rpy euler, URDF order). */
function originMatrix(xyz: Vec3, rpy: Vec3): THREE.Matrix4 {
  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion().setFromEuler(
    new THREE.Euler(rpy[0], rpy[1], rpy[2], 'XYZ'),
  );
  m.compose(
    new THREE.Vector3(xyz[0], xyz[1], xyz[2]),
    q,
    new THREE.Vector3(1, 1, 1),
  );
  return m;
}

interface PlacedLink {
  link: GeomLink;
  frame: THREE.Matrix4;
}

/**
 * Resolve each link's frame by walking joints from the base link. Links not
 * reachable via a joint (e.g. the base itself) get identity.
 */
function placeLinks(geom: Geometry): PlacedLink[] {
  const childToJoint = new Map(
    geom.joints.filter((j) => j.child).map((j) => [j.child as string, j]),
  );

  const cache = new Map<string, THREE.Matrix4>();
  function frameOf(linkName: string): THREE.Matrix4 {
    const hit = cache.get(linkName);
    if (hit) return hit;
    const joint = childToJoint.get(linkName);
    let frame: THREE.Matrix4;
    if (!joint || !joint.parent) {
      frame = new THREE.Matrix4();
    } else {
      const parent = frameOf(joint.parent);
      frame = parent
        .clone()
        .multiply(originMatrix(joint.origin_xyz, joint.origin_rpy));
    }
    cache.set(linkName, frame);
    return frame;
  }

  return geom.links.map((link) => ({ link, frame: frameOf(link.name) }));
}

function isChassis(name: string): boolean {
  const n = name.toLowerCase();
  return n.includes('base') || n.includes('chassis') || n.includes('wedge');
}

function LinkMesh({ placed }: { placed: PlacedLink }) {
  const { link, frame } = placed;

  const { position, quaternion } = useMemo(() => {
    const world = frame
      .clone()
      .multiply(originMatrix(link.origin_xyz, link.origin_rpy));
    const pos = new THREE.Vector3();
    const quat = new THREE.Quaternion();
    const scl = new THREE.Vector3();
    world.decompose(pos, quat, scl);
    return { position: pos, quaternion: quat };
  }, [frame, link.origin_xyz, link.origin_rpy]);

  const color = isChassis(link.name)
    ? ACCENT
    : new THREE.Color(link.rgba[0], link.rgba[1], link.rgba[2]).getStyle();
  const emissive = isChassis(link.name) ? ACCENT : '#0a0d11';

  const geometryNode = useMemo(() => {
    if (link.shape === 'box' && link.size) {
      return <boxGeometry args={link.size} />;
    }
    if (link.shape === 'cylinder') {
      return (
        <cylinderGeometry
          args={[
            link.radius ?? 0.01,
            link.radius ?? 0.01,
            link.length ?? 0.01,
            28,
          ]}
        />
      );
    }
    if (link.shape === 'sphere') {
      return <sphereGeometry args={[link.radius ?? 0.01, 24, 16]} />;
    }
    return null;
  }, [link]);

  if (!geometryNode) return null;

  const cylFix = link.shape === 'cylinder';

  return (
    <group position={position} quaternion={quaternion}>
      <mesh rotation={cylFix ? [Math.PI / 2, 0, 0] : undefined} castShadow>
        {geometryNode}
        <meshStandardMaterial
          color={color}
          emissive={emissive}
          emissiveIntensity={isChassis(link.name) ? 0.35 : 0.05}
          metalness={0.35}
          roughness={0.55}
        />
      </mesh>
    </group>
  );
}

/** One ToF sensor: a marker at its mount + a direction ray of length range_m.
 * The ray is a thin cylinder (robust across r3f/three) lying in the body XY
 * plane along angle_rad, faintly transparent forge-orange. */
function DistanceSensorViz({ sensor }: { sensor: DistanceSensor }) {
  const [x, y, z] = sensor.mount_xyz;
  const range = Math.max(0.0001, sensor.range_m);
  const a = sensor.angle_rad; // yaw about +Z, positive toward +Y

  // Cylinder default axis is +Y. We want it along the in-plane direction
  // (cos a, sin a, 0), placed at the segment midpoint. Rotate +Y onto dir.
  const dir = useMemo(
    () => new THREE.Vector3(Math.cos(a), Math.sin(a), 0).normalize(),
    [a],
  );
  const quat = useMemo(() => {
    const q = new THREE.Quaternion();
    q.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    return q;
  }, [dir]);
  const mid = useMemo(
    () =>
      new THREE.Vector3(
        x + (dir.x * range) / 2,
        y + (dir.y * range) / 2,
        z,
      ),
    [x, y, z, dir, range],
  );
  const tip = useMemo(
    () => new THREE.Vector3(x + dir.x * range, y + dir.y * range, z),
    [x, y, z, dir, range],
  );

  return (
    <group>
      {/* mount marker */}
      <mesh position={[x, y, z]}>
        <sphereGeometry args={[0.004, 12, 8]} />
        <meshBasicMaterial color={ACCENT} />
      </mesh>
      {/* direction ray (faint forge-orange cylinder) */}
      <mesh position={mid} quaternion={quat}>
        <cylinderGeometry args={[0.0009, 0.0009, range, 8]} />
        <meshBasicMaterial color={ACCENT} transparent opacity={0.5} />
      </mesh>
      {/* arrow head at the ray tip */}
      <mesh position={tip} quaternion={quat}>
        <coneGeometry args={[0.004, 0.008, 10]} />
        <meshBasicMaterial color={ACCENT} transparent opacity={0.85} />
      </mesh>
    </group>
  );
}

/** One line sensor: a cyan marker on the chassis underside. */
function LineSensorViz({
  sensor,
  bottomZ,
}: {
  sensor: LineSensor;
  bottomZ: number;
}) {
  const [x, y] = sensor.mount_xy;
  return (
    <mesh position={[x, y, bottomZ]} rotation={[Math.PI / 2, 0, 0]}>
      <cylinderGeometry args={[0.005, 0.005, 0.0015, 16]} />
      <meshBasicMaterial color={CYAN} />
    </mesh>
  );
}

/** Line-sensor footprints projected onto the dohyo surface (underside view). */
function LineFootprint({ sensor }: { sensor: LineSensor }) {
  const [x, y] = sensor.mount_xy;
  return (
    <mesh position={[x, y, 0.001]} rotation={[0, 0, 0]}>
      <ringGeometry args={[0.006, 0.009, 24]} />
      <meshBasicMaterial color={CYAN} transparent opacity={0.85} />
    </mesh>
  );
}

function Dohyo({ radius, border }: { radius: number; border: number }) {
  const inner = Math.max(0.001, radius - border);
  return (
    <group>
      {/* black inner playing disc */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.0005, 0]} receiveShadow>
        <circleGeometry args={[inner, 64]} />
        <meshStandardMaterial color="#11171d" roughness={0.9} metalness={0.1} />
      </mesh>
      {/* white-ish border ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.0003, 0]}>
        <ringGeometry args={[inner, radius, 64]} />
        <meshStandardMaterial color="#3a4754" roughness={0.85} metalness={0.1} />
      </mesh>
      {/* cyan perimeter glow ring (dohyo edge) */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.0006, 0]}>
        <ringGeometry args={[radius - 0.004, radius, 64]} />
        <meshBasicMaterial color={CYAN} transparent opacity={0.5} />
      </mesh>
    </group>
  );
}

function Scene({
  geom,
  spec,
  underside,
}: {
  geom: Geometry;
  spec: HardwareSpec | null;
  underside: boolean;
}) {
  const placed = useMemo(() => placeLinks(geom), [geom]);
  const radius = spec?.dohyo.radius_m ?? 0.35;
  const border = spec?.dohyo.border_width_m ?? 0.025;
  // Underside marker height: just under the chassis (slightly below z=0 floor
  // contact). Use a small negative so markers sit on the bottom face plane.
  const bottomZ = 0.001;

  return (
    <>
      <ambientLight intensity={0.55} />
      <directionalLight
        position={[0.4, 0.7, 0.5]}
        intensity={1.8}
        castShadow
        shadow-mapSize={[1024, 1024]}
      />
      <directionalLight
        position={[-0.5, 0.3, -0.4]}
        intensity={0.5}
        color={CYAN}
      />
      {/* fill light from below for the underside view */}
      {underside && (
        <directionalLight position={[0, -0.6, 0.2]} intensity={0.9} color={CYAN} />
      )}

      <Grid
        position={[0, -0.001, 0]}
        args={[2, 2]}
        cellSize={0.05}
        cellThickness={0.5}
        cellColor="#1b2630"
        sectionSize={0.2}
        sectionThickness={1}
        sectionColor="#26323d"
        fadeDistance={1.4}
        fadeStrength={1.5}
        infiniteGrid
      />
      <Dohyo radius={radius} border={border} />

      {/* Z-up (URDF) -> Y-up (three) rotation for the whole rig + overlays. */}
      <group rotation={Z_UP_TO_Y_UP}>
        {placed.map((p) => (
          <LinkMesh key={p.link.name} placed={p} />
        ))}

        {spec?.distance_sensors.map((s, i) => (
          <DistanceSensorViz key={`tof-${s.id}-${i}`} sensor={s} />
        ))}

        {spec?.line_sensors.map((l, i) =>
          underside ? (
            <LineFootprint key={`lf-${l.id}-${i}`} sensor={l} />
          ) : (
            <LineSensorViz
              key={`ls-${l.id}-${i}`}
              sensor={l}
              bottomZ={bottomZ}
            />
          ),
        )}
      </group>

      <ContactShadows
        position={[0, 0, 0]}
        opacity={0.55}
        scale={0.9}
        blur={2.2}
        far={0.4}
      />
      <Environment preset="city" />
      <OrbitControls
        enablePan={false}
        minDistance={0.12}
        maxDistance={Math.max(0.7, radius * 2.4)}
        // Underside: allow the camera to swing below the floor to look up.
        minPolarAngle={underside ? Math.PI / 1.95 : 0}
        maxPolarAngle={underside ? Math.PI : Math.PI / 2.05}
        target={[0, underside ? -0.01 : 0.02, 0]}
      />
    </>
  );
}

export function RobotPreview({
  geom,
  spec,
  loading,
  error,
}: {
  geom: Geometry | null;
  spec?: HardwareSpec | null;
  loading?: boolean;
  error?: string | null;
}) {
  const [underside, setUnderside] = useState(false);

  // Camera differs per view: top-down-ish orbit vs. low-angle from beneath.
  const camera = underside
    ? { position: [0.16, -0.16, 0.18] as [number, number, number], fov: 44 }
    : { position: [0.18, 0.14, 0.22] as [number, number, number], fov: 42 };

  return (
    <div className="panel panel-live relative h-full min-h-[420px] overflow-hidden">
      <CornerTicks color="var(--accent-dim)" />

      {/* HUD top strip */}
      <div className="pointer-events-none absolute left-0 right-0 top-0 z-10 flex items-center justify-between px-4 py-2.5">
        <span className="micro num" style={{ color: 'var(--cyan)' }}>
          DOHYO CAM
        </span>
        <span className="micro num text-fg-2" style={{ fontSize: 10 }}>
          {geom ? `${geom.links.length} LINKS` : '— —'}
        </span>
      </div>

      {/* TOP / UNDERSIDE toggle */}
      <div className="absolute right-3 top-9 z-20 flex overflow-hidden rounded border"
        style={{ borderColor: 'var(--line)', background: 'var(--bg-1)' }}
      >
        {(['TOP', 'UNDERSIDE'] as const).map((mode) => {
          const active = (mode === 'UNDERSIDE') === underside;
          return (
            <button
              key={mode}
              onClick={() => setUnderside(mode === 'UNDERSIDE')}
              className="micro"
              style={{
                fontSize: 9,
                letterSpacing: '.08em',
                padding: '4px 8px',
                color: active ? 'var(--bg-0)' : 'var(--fg-1)',
                background: active ? 'var(--accent)' : 'transparent',
                border: 'none',
                cursor: 'pointer',
              }}
            >
              {mode}
            </button>
          );
        })}
      </div>

      {/* sensor legend (underside emphasises line footprints) */}
      <div className="pointer-events-none absolute bottom-2.5 left-4 z-10 flex gap-3">
        <span className="micro num" style={{ fontSize: 9, color: 'var(--accent)' }}>
          ◆ ToF · ray = range
        </span>
        <span className="micro num" style={{ fontSize: 9, color: 'var(--cyan)' }}>
          ● LINE {underside ? 'footprint' : 'sensor'}
        </span>
      </div>

      {/* scanline overlay */}
      <div
        className="pointer-events-none absolute inset-0 z-0"
        style={{
          backgroundImage:
            'repeating-linear-gradient(0deg, rgba(0,0,0,.18) 0px, rgba(0,0,0,.18) 1px, transparent 1px, transparent 3px)',
          opacity: 0.25,
        }}
      />

      {error && (
        <div className="absolute inset-0 z-20 flex items-center justify-center p-6 text-center">
          <div>
            <div className="micro mb-1" style={{ color: 'var(--loss)' }}>
              GEOMETRY ERROR
            </div>
            <div className="num text-fg-1" style={{ fontSize: 12 }}>
              {error}
            </div>
          </div>
        </div>
      )}

      {loading && !geom && (
        <div className="absolute inset-0 z-20 flex items-center justify-center">
          <span className="micro animate-pulse" style={{ color: 'var(--cyan)' }}>
            POWERING ON…
          </span>
        </div>
      )}

      <Canvas
        key={underside ? 'under' : 'top'}
        shadows
        dpr={[1, 2]}
        camera={camera}
        gl={{ antialias: true }}
        style={{
          background: 'radial-gradient(circle at 50% 42%, #0c1118, #080b0f 75%)',
        }}
      >
        {geom && <Scene geom={geom} spec={spec ?? null} underside={underside} />}
      </Canvas>
    </div>
  );
}
