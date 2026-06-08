// Shared robot-geometry → three.js meshes builder.
//
// Both RobotPreview (Hardware page) and TrajectoryPlayer (Arena / Opponents
// replays) need to draw a robot from the backend's `/api/hardware/geometry`
// primitives: the chassis box + the front WEDGE (the sharp part) + the wheels.
// This module owns the URDF kinematic-tree walk and the per-link primitive
// meshes so both call sites stay DRY and identical in geometry.
//
// `RobotMeshes` renders the link meshes WITHOUT the Z-up→Y-up rig rotation —
// the caller wraps it in a `<group rotation={Z_UP_TO_Y_UP}>` (RobotPreview pins
// it at the origin; TrajectoryPlayer additionally moves a per-robot group by the
// trajectory pose). Pass `tint` to override every link's colour with one side
// colour (Arena/Opponents tint A=orange / B=cyan instead of using each link's
// own rgba); omit it to keep RobotPreview's original per-link colouring.

import { useMemo } from 'react';
import * as THREE from 'three';
import type { Geometry, GeomLink, Vec3 } from '../types';

const ACCENT = '#ff7a18';

// URDF/PyBullet are Z-up; three.js is Y-up. Rotate the whole rig -90deg about X
// so the robot stands on the floor. Both call sites use this exact basis so the
// replay poses and the static preview line up identically with the floor.
export const Z_UP_TO_Y_UP = new THREE.Euler(-Math.PI / 2, 0, 0);

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

export function isChassis(name: string): boolean {
  const n = name.toLowerCase();
  return n.includes('base') || n.includes('chassis') || n.includes('wedge');
}

function LinkMesh({ placed, tint }: { placed: PlacedLink; tint?: string }) {
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

  const chassis = isChassis(link.name);
  // When tinted (replay sides), every link wears the side colour, with chassis
  // parts glowing a touch more so the wedge reads. Untinted (preview): keep the
  // original behaviour — chassis = accent, other links = their own rgba.
  const color = tint
    ? tint
    : chassis
      ? ACCENT
      : new THREE.Color(link.rgba[0], link.rgba[1], link.rgba[2]).getStyle();
  const emissive = tint ? tint : chassis ? ACCENT : '#0a0d11';
  const emissiveIntensity = tint ? (chassis ? 0.32 : 0.12) : chassis ? 0.35 : 0.05;

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
          emissiveIntensity={emissiveIntensity}
          metalness={0.35}
          roughness={0.55}
        />
      </mesh>
    </group>
  );
}

/**
 * All of a robot's link meshes (chassis + wedge + wheels), placed by the URDF
 * kinematic tree. Render INSIDE a `<group rotation={Z_UP_TO_Y_UP}>`. Geometry is
 * built once per `geom` (not per frame), so it's cheap to nest in an animated
 * group. `tint` overrides every link colour with a single side colour.
 */
export function RobotMeshes({
  geom,
  tint,
}: {
  geom: Geometry;
  tint?: string;
}) {
  const placed = useMemo(() => placeLinks(geom), [geom]);
  return (
    <>
      {placed.map((p) => (
        <LinkMesh key={p.link.name} placed={p} tint={tint} />
      ))}
    </>
  );
}
