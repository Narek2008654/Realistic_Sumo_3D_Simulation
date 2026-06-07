// HUD-framed 3D preview of a HardwareSpec, rendered from the backend's
// `/api/hardware/geometry` primitives. Builds the URDF kinematic tree from the
// joint list, composes each joint's origin to place every visual link, and
// draws box/cylinder primitives. Dohyo disc floor, grid, vignette, corner
// brackets. Agent body is forge-orange (--accent); wheels/details stay neutral.

import { Canvas } from '@react-three/fiber';
import {
  ContactShadows,
  Environment,
  Grid,
  OrbitControls,
} from '@react-three/drei';
import { useMemo } from 'react';
import * as THREE from 'three';
import type { Geometry, GeomLink, Vec3 } from '../types';
import { CornerTicks } from './ui';

const ACCENT = '#ff7a18';

// URDF/PyBullet are Z-up; three.js is Y-up. We rotate the whole rig -90deg
// about X so the robot stands on the floor and reads naturally.
const Z_UP_TO_Y_UP = new THREE.Euler(-Math.PI / 2, 0, 0);

/** Local transform of a joint/visual origin (xyz + rpy euler, URDF order). */
function originMatrix(xyz: Vec3, rpy: Vec3): THREE.Matrix4 {
  const m = new THREE.Matrix4();
  // URDF rpy is fixed-axis XYZ (roll, pitch, yaw). three.js 'XYZ' euler with
  // intrinsic order matches URDF's extrinsic when applied as below.
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
  // world matrix of the link frame (before the visual-origin offset)
  frame: THREE.Matrix4;
}

/**
 * Resolve each link's frame by walking joints from the base link. Links not
 * reachable via a joint (e.g. the base itself) get identity.
 */
function placeLinks(geom: Geometry): PlacedLink[] {
  const childToJoint = new Map(
    geom.joints
      .filter((j) => j.child)
      .map((j) => [j.child as string, j]),
  );

  // Memoised frame resolver (parent frame ∘ joint origin).
  const cache = new Map<string, THREE.Matrix4>();
  function frameOf(linkName: string): THREE.Matrix4 {
    const hit = cache.get(linkName);
    if (hit) return hit;
    const joint = childToJoint.get(linkName);
    let frame: THREE.Matrix4;
    if (!joint || !joint.parent) {
      frame = new THREE.Matrix4(); // base / root link
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

  // Compose the link frame with the visual origin, then express the resulting
  // matrix as decomposed pos/quat/scale for the <group>.
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
      // URDF cylinder axis is +Z; three.js cylinder axis is +Y -> rotate.
      return (
        <cylinderGeometry
          args={[link.radius ?? 0.01, link.radius ?? 0.01, link.length ?? 0.01, 28]}
        />
      );
    }
    if (link.shape === 'sphere') {
      return <sphereGeometry args={[link.radius ?? 0.01, 24, 16]} />;
    }
    return null; // mesh primitives are not loaded in the lite preview
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

function Dohyo() {
  return (
    <group>
      {/* dohyo disc floor */}
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, -0.0005, 0]}
        receiveShadow
      >
        <circleGeometry args={[0.38, 64]} />
        <meshStandardMaterial color="#11171d" roughness={0.9} metalness={0.1} />
      </mesh>
      {/* perimeter ring (dohyo edge) */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.0006, 0]}>
        <ringGeometry args={[0.36, 0.38, 64]} />
        <meshBasicMaterial color="#2ad4ff" transparent opacity={0.5} />
      </mesh>
    </group>
  );
}

function Scene({ geom }: { geom: Geometry }) {
  const placed = useMemo(() => placeLinks(geom), [geom]);
  return (
    <>
      <ambientLight intensity={0.55} />
      <directionalLight
        position={[0.4, 0.7, 0.5]}
        intensity={1.8}
        castShadow
        shadow-mapSize={[1024, 1024]}
      />
      <directionalLight position={[-0.5, 0.3, -0.4]} intensity={0.5} color="#2ad4ff" />

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
      <Dohyo />

      {/* Z-up (URDF) -> Y-up (three) rotation for the whole rig. */}
      <group rotation={Z_UP_TO_Y_UP}>
        {placed.map((p) => (
          <LinkMesh key={p.link.name} placed={p} />
        ))}
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
        maxDistance={0.7}
        maxPolarAngle={Math.PI / 2.05}
        target={[0, 0.02, 0]}
      />
    </>
  );
}

export function RobotPreview({
  geom,
  loading,
  error,
}: {
  geom: Geometry | null;
  loading?: boolean;
  error?: string | null;
}) {
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

      {/* scanline overlay */}
      <div
        className="pointer-events-none absolute inset-0 z-10"
        style={{
          backgroundImage:
            'repeating-linear-gradient(0deg, rgba(0,0,0,.18) 0px, rgba(0,0,0,.18) 1px, transparent 1px, transparent 3px)',
          opacity: 0.25,
        }}
      />

      {error && (
        <div className="absolute inset-0 z-20 flex items-center justify-center p-6 text-center">
          <div>
            <div
              className="micro mb-1"
              style={{ color: 'var(--loss)' }}
            >
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
        shadows
        dpr={[1, 2]}
        camera={{ position: [0.18, 0.14, 0.22], fov: 42 }}
        gl={{ antialias: true }}
        style={{ background: 'radial-gradient(circle at 50% 42%, #0c1118, #080b0f 75%)' }}
      >
        {geom && <Scene geom={geom} />}
      </Canvas>
    </div>
  );
}
