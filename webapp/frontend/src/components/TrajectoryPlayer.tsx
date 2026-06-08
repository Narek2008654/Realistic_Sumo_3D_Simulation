// HUD-framed 3D replay of a training-checkpoint trajectory ("DOHYO CAM").
//
// Reuses RobotPreview's stage language: dark radial-gradient backdrop, faint
// grid floor, cyan-rimmed dohyo disc, corner brackets, scanline overlay. Given
// a Trajectory JSON (kinematics only — see webapp/shared/eval_and_record.py) it
// animates the agent + enemy through `frames`, applying each frame's world
// position + quaternion under the SAME Z-up (PyBullet/URDF) -> Y-up (three.js)
// basis used by the builder preview, so the poses read upright and consistent.
//
// Each robot is rendered from its FULL `/api/hardware/geometry` primitives
// (chassis box + front WEDGE + wheels) via the shared RobotMeshes builder — the
// same one the Hardware preview uses — tinted by side: agent A = forge-orange
// (--accent), enemy B = cyan (--cyan). A console-style transport bar drives
// play/pause, frame scrub, and 0.5/1/2x speed; the last frame shows the outcome
// chip (WIN / SELF-OUT / PUSH / TIMEOUT).

import { Canvas, useFrame } from '@react-three/fiber';
import { ContactShadows, Environment, Grid, OrbitControls } from '@react-three/drei';
import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { api } from '../api';
import type { Geometry, HardwareSpec, Trajectory, TrajectoryFrame } from '../types';
import { CornerTicks } from './ui';
import { RobotMeshes, Z_UP_TO_Y_UP } from './RobotMeshes';

const ACCENT = '#ff7a18';
const CYAN = '#2ad4ff';

// Fallback chassis block (metres) shown only while geometry is still loading —
// the final render is always the full geometry once it resolves.
const PLACEHOLDER_BOX: [number, number, number] = [0.1, 0.1, 0.05];

/** One robot, moved per frame by the trajectory pose (raw world Z-up coords —
 *  it lives inside the Z-up→Y-up group). Renders the full geometry (chassis +
 *  wedge + wheels) tinted by side once `geom` resolves; until then a light
 *  placeholder box keeps the stage populated. Meshes are built once per geom
 *  (RobotMeshes memoises the kinematic walk), not per frame. */
function Robot({
  pose,
  geom,
  color,
}: {
  pose: { p: THREE.Vector3; q: THREE.Quaternion };
  geom: Geometry | null;
  color: string;
}) {
  const ref = useRef<THREE.Group>(null);
  useFrame(() => {
    const g = ref.current;
    if (!g) return;
    g.position.copy(pose.p);
    g.quaternion.copy(pose.q);
  });
  return (
    <group ref={ref}>
      {geom ? (
        <RobotMeshes geom={geom} tint={color} />
      ) : (
        <mesh castShadow>
          <boxGeometry args={PLACEHOLDER_BOX} />
          <meshStandardMaterial
            color={color}
            emissive={color}
            emissiveIntensity={0.32}
            metalness={0.35}
            roughness={0.5}
          />
        </mesh>
      )}
    </group>
  );
}

function Dohyo({ radius }: { radius: number }) {
  const border = Math.min(0.025, radius * 0.07);
  const inner = Math.max(0.001, radius - border);
  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.0005, 0]} receiveShadow>
        <circleGeometry args={[inner, 64]} />
        <meshStandardMaterial color="#11171d" roughness={0.9} metalness={0.1} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.0003, 0]}>
        <ringGeometry args={[inner, radius, 64]} />
        <meshStandardMaterial color="#3a4754" roughness={0.85} metalness={0.1} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.0006, 0]}>
        <ringGeometry args={[radius - 0.004, radius, 64]} />
        <meshBasicMaterial color={CYAN} transparent opacity={0.5} />
      </mesh>
    </group>
  );
}

/** Resolve a frame's two poses into three.js vectors/quaternions (raw world
 *  Z-up coords — they live inside the Z-up->Y-up group). */
function framePoses(frame: TrajectoryFrame) {
  const mk = (f: { p: number[]; q: number[] }) => ({
    p: new THREE.Vector3(f.p[0], f.p[1], f.p[2]),
    q: new THREE.Quaternion(f.q[0], f.q[1], f.q[2], f.q[3]),
  });
  return { agent: mk(frame.agent), enemy: mk(frame.enemy) };
}

function Scene({
  traj,
  agentGeom,
  enemyGeom,
  frameIndex,
}: {
  traj: Trajectory;
  agentGeom: Geometry | null;
  enemyGeom: Geometry | null;
  frameIndex: number;
}) {
  const radius = traj.dohyo_radius || 0.35;
  const i = Math.max(0, Math.min(traj.frames.length - 1, Math.round(frameIndex)));
  const poses = useMemo(() => framePoses(traj.frames[i]), [traj, i]);

  return (
    <>
      <ambientLight intensity={0.55} />
      <directionalLight
        position={[0.4, 0.7, 0.5]}
        intensity={1.8}
        castShadow
        shadow-mapSize={[1024, 1024]}
      />
      <directionalLight position={[-0.5, 0.3, -0.4]} intensity={0.5} color={CYAN} />

      <Grid
        position={[0, -0.001, 0]}
        args={[2, 2]}
        cellSize={0.05}
        cellThickness={0.5}
        cellColor="#1b2630"
        sectionSize={0.2}
        sectionThickness={1}
        sectionColor="#26323d"
        fadeDistance={1.8}
        fadeStrength={1.5}
        infiniteGrid
      />
      <Dohyo radius={radius} />

      <group rotation={Z_UP_TO_Y_UP}>
        <Robot pose={poses.agent} geom={agentGeom} color={ACCENT} />
        <Robot pose={poses.enemy} geom={enemyGeom} color={CYAN} />
      </group>

      <ContactShadows position={[0, 0, 0]} opacity={0.5} scale={radius * 3} blur={2.4} far={0.5} />
      <Environment preset="city" />
      <OrbitControls
        enablePan={false}
        minDistance={radius * 1.1}
        maxDistance={radius * 6}
        maxPolarAngle={Math.PI / 2.05}
        target={[0, 0.02, 0]}
      />
    </>
  );
}

const OUTCOME = {
  win: { label: 'WIN', color: 'var(--win)' },
  self_out: { label: 'SELF-OUT', color: 'var(--loss)' },
  push_loss: { label: 'PUSH', color: 'var(--loss)' },
  mutual_out: { label: 'MUTUAL OUT', color: 'var(--warn)' },
  timeout: { label: 'TIMEOUT', color: 'var(--idle)' },
  unknown: { label: '—', color: 'var(--idle)' },
} as const;

function outcomeChip(reason: string) {
  return OUTCOME[reason as keyof typeof OUTCOME] ?? OUTCOME.unknown;
}

const SPEEDS = [0.5, 1, 2] as const;

// Module-level geometry cache, keyed by the spec's serialised identity. The
// same default spec (and repeated battles on one chassis) resolve to one fetch
// for the whole session, so geometry is never refetched per render or per frame.
const geomCache = new Map<string, Promise<Geometry>>();

function fetchGeometry(spec: HardwareSpec): Promise<Geometry> {
  const key = JSON.stringify(spec);
  let p = geomCache.get(key);
  if (!p) {
    p = api.geometry(spec);
    geomCache.set(key, p);
  }
  return p;
}

/** Resolve a side's geometry: use the explicit spec when given, else the cached
 *  default spec. Returns null until it loads (the caller shows a placeholder).
 *  Cached across renders, so the fetch happens once per distinct spec. */
function useSideGeometry(spec: HardwareSpec | null | undefined): Geometry | null {
  const [defaultSpec, setDefaultSpec] = useState<HardwareSpec | null>(null);
  const [geom, setGeom] = useState<Geometry | null>(null);

  // Fetch the default spec once (shared fallback for both sides).
  useEffect(() => {
    let alive = true;
    api
      .hardwareDefault()
      .then((s) => alive && setDefaultSpec(s))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, []);

  const effective = spec ?? defaultSpec;
  const key = effective ? JSON.stringify(effective) : null;

  useEffect(() => {
    if (!effective) return;
    let alive = true;
    setGeom(null);
    fetchGeometry(effective)
      .then((g) => alive && setGeom(g))
      .catch(() => {});
    return () => {
      alive = false;
    };
    // key captures the spec identity; effective is derived from it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  return geom;
}

export function TrajectoryPlayer({
  traj,
  agentSpec,
  opponentSpec,
  label,
}: {
  traj: Trajectory | null;
  agentSpec?: HardwareSpec | null;
  opponentSpec?: HardwareSpec | null;
  label?: string;
}) {
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>(1);

  const nFrames = traj?.frames.length ?? 0;
  const lastFrame = Math.max(0, nFrames - 1);
  const dt = traj?.dt || 0.05;
  const agentGeom = useSideGeometry(agentSpec);
  const enemyGeom = useSideGeometry(opponentSpec);

  // Reset transport when a new trajectory loads (powering-on micro-moment).
  useEffect(() => {
    setFrame(0);
    setPlaying(true);
  }, [traj]);

  // Advance the playhead on a wall-clock timer scaled by sim dt + speed.
  useEffect(() => {
    if (!playing || nFrames === 0) return;
    const intervalMs = Math.max(16, (dt * 1000) / speed);
    const id = window.setInterval(() => {
      setFrame((f) => {
        if (f >= lastFrame) {
          window.clearInterval(id);
          setPlaying(false);
          return lastFrame;
        }
        return f + 1;
      });
    }, intervalMs);
    return () => window.clearInterval(id);
  }, [playing, speed, dt, nFrames, lastFrame]);

  const atEnd = frame >= lastFrame && nFrames > 0;
  const chip = traj ? outcomeChip(traj.outcome.reason) : null;
  const showChip = atEnd && chip;

  function togglePlay() {
    if (atEnd) {
      setFrame(0);
      setPlaying(true);
    } else {
      setPlaying((p) => !p);
    }
  }

  return (
    <div className="panel panel-live relative h-full min-h-[420px] overflow-hidden">
      <CornerTicks color="var(--accent-dim)" />

      {/* HUD top strip */}
      <div className="pointer-events-none absolute left-0 right-0 top-0 z-10 flex items-center justify-between px-4 py-2.5">
        <span className="micro num" style={{ color: 'var(--cyan)' }}>
          DOHYO CAM{label ? ` · ${label}` : ''}
        </span>
        <span className="micro num text-fg-2" style={{ fontSize: 10 }}>
          {nFrames ? `FRAME ${frame} / ${lastFrame}` : '— —'}
        </span>
      </div>

      {/* outcome chip (top-right, last frame) */}
      {showChip && (
        <div className="pointer-events-none absolute right-3 top-9 z-20">
          <span
            className="inline-flex items-center gap-2 rounded border px-2 py-1"
            style={{ borderColor: 'var(--line)', background: 'var(--bg-1)' }}
          >
            <span
              className="inline-flex h-2 w-2 rounded-full"
              style={{ background: chip.color, boxShadow: `0 0 8px ${chip.color}` }}
            />
            <span className="micro" style={{ color: chip.color, letterSpacing: '.1em' }}>
              {chip.label}
            </span>
          </span>
        </div>
      )}

      {/* scanline overlay */}
      <div
        className="pointer-events-none absolute inset-0 z-0"
        style={{
          backgroundImage:
            'repeating-linear-gradient(0deg, rgba(0,0,0,.18) 0px, rgba(0,0,0,.18) 1px, transparent 1px, transparent 3px)',
          opacity: 0.25,
        }}
      />

      {/* legend */}
      <div className="pointer-events-none absolute bottom-14 left-4 z-30 flex gap-3">
        <span className="micro num" style={{ fontSize: 9, color: 'var(--accent)' }}>
          ■ AGENT
        </span>
        <span className="micro num" style={{ fontSize: 9, color: 'var(--cyan)' }}>
          ■ ENEMY
        </span>
      </div>

      {!traj && (
        <div className="absolute inset-0 z-20 flex items-center justify-center">
          <span className="micro text-fg-2">AWAITING CHECKPOINT…</span>
        </div>
      )}

      <Canvas
        shadows
        dpr={[1, 2]}
        camera={{ position: [0.55, 0.5, 0.55], fov: 42 }}
        gl={{ antialias: true }}
        style={{
          background: 'radial-gradient(circle at 50% 42%, #0c1118, #080b0f 75%)',
        }}
      >
        {traj && (
          <Scene
            traj={traj}
            agentGeom={agentGeom}
            enemyGeom={enemyGeom}
            frameIndex={frame}
          />
        )}
      </Canvas>

      {/* Transport bar */}
      {traj && (
        <div
          className="absolute bottom-0 left-0 right-0 z-30 flex items-center gap-3 border-t px-3 py-2"
          style={{ borderColor: 'var(--line)', background: 'rgba(8,11,15,.82)' }}
        >
          <button
            type="button"
            onClick={togglePlay}
            className="micro"
            style={{
              width: 30,
              height: 24,
              color: 'var(--bg-0)',
              background: 'var(--accent)',
              border: 'none',
              borderRadius: 'var(--radius)',
              cursor: 'pointer',
              fontSize: 11,
            }}
            title={atEnd ? 'Replay' : playing ? 'Pause' : 'Play'}
          >
            {atEnd ? '↺' : playing ? '❚❚' : '▶'}
          </button>

          <input
            type="range"
            min={0}
            max={lastFrame}
            step={1}
            value={frame}
            onChange={(e) => {
              setPlaying(false);
              setFrame(parseInt(e.target.value, 10));
            }}
            className="h-1.5 flex-1 cursor-pointer appearance-none rounded"
            style={{ accentColor: 'var(--cyan)', background: 'var(--bg-3)' }}
          />

          <span className="num text-fg-2" style={{ fontSize: 10, minWidth: 70, textAlign: 'right' }}>
            {(frame * dt).toFixed(2)}s
          </span>

          <div className="flex overflow-hidden rounded border" style={{ borderColor: 'var(--line)' }}>
            {SPEEDS.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setSpeed(s)}
                className="num"
                style={{
                  fontSize: 10,
                  padding: '3px 7px',
                  border: 'none',
                  cursor: 'pointer',
                  color: speed === s ? 'var(--bg-0)' : 'var(--fg-1)',
                  background: speed === s ? 'var(--cyan)' : 'transparent',
                }}
              >
                {s}×
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
