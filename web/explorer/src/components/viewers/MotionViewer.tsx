import { useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, GizmoHelper, GizmoViewport, Grid, Line } from '@react-three/drei';
import * as THREE from 'three';
import { HAND_CONNECTIONS, BODY_CONNECTIONS, END_MARKERS_4, END_MARKERS_2 } from '@/data/connections';
import { BRAND, SURFACE } from '@/theme';
import type { FingeringData, MotionData, MotionFrame } from '@/data/types';

const SCALE = 0.001; // motion CSV is in mm; convert to scene meters

export type CameraPreset = 'top' | 'front' | 'side' | '3d' | 'fingering';

// Fingertip marker naming: [L|R] + [T|I|M|R|L] + F  (thumb / index / middle / ring / pinky=little)
const FINGER_LETTER = ['T', 'I', 'M', 'R', 'L'] as const;

function fingertipMarker(hand: 'L' | 'R', finger: number): string | null {
  const letter = FINGER_LETTER[finger - 1];
  if (!letter) return null;
  return `${hand}${letter}F`;
}

// finger 1..5 color (thumb=red, index=orange, middle=green, ring=blue, pinky=tertiary purple)
const FINGER_COLORS = ['#E53935', '#FB8C00', '#43A047', '#1E88E5', BRAND.TERTIARY];

function getActiveFingers(
  fingering: FingeringData | null | undefined,
  t: number,
): { L: number | null; R: number | null } {
  if (!fingering) return { L: null, R: null };
  let L: number | null = null;
  let R: number | null = null;
  for (const ev of fingering.events) {
    if (ev.hand == null || ev.finger == null) continue;
    if (ev.onset <= t && t < ev.offset) {
      if (ev.hand === 'L' && L === null) L = ev.finger;
      else if (ev.hand === 'R' && R === null) R = ev.finger;
    }
  }
  return { L, R };
}

interface SkeletonProps {
  frame: MotionFrame | null;
  showHand: boolean;
  showBody: boolean;
  activeFingerL: number | null;
  activeFingerR: number | null;
}

function Skeleton({ frame, showHand, showBody, activeFingerL, activeFingerR }: SkeletonProps) {
  const instRef = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  const endMarkers = frame?.joints?.LEnd1 ? END_MARKERS_4 : END_MARKERS_2;

  const handJointSet = useMemo(() => {
    const s = new Set<string>();
    for (const [a, b] of HAND_CONNECTIONS) {
      s.add(a);
      s.add(b);
    }
    return s;
  }, []);
  const bodyJointSet = useMemo(() => {
    const s = new Set<string>();
    for (const [a, b] of BODY_CONNECTIONS) {
      s.add(a);
      s.add(b);
    }
    return s;
  }, []);

  const { positions, handLines, bodyLines, endPositions, headEllipsoid } = useMemo(() => {
    type Ellipsoid = {
      position: [number, number, number];
      quaternion: [number, number, number, number];
      scale: [number, number, number];
    };
    if (!frame) {
      return {
        positions: [] as [number, number, number][],
        handLines: [] as [number, number, number][][],
        bodyLines: [] as [number, number, number][][],
        endPositions: [] as [number, number, number][],
        headEllipsoid: null as Ellipsoid | null,
      };
    }

    const pos: [number, number, number][] = [];
    const endPos: [number, number, number][] = [];

    for (const [name, j] of Object.entries(frame.joints)) {
      if (!Number.isFinite(j.x) || !Number.isFinite(j.y) || !Number.isFinite(j.z)) continue;
      const isHand = handJointSet.has(name);
      const isBody = bodyJointSet.has(name);
      const isEnd = endMarkers.includes(name);
      if (!isHand && !isBody && !isEnd) continue;
      if (isHand && !showHand) continue;
      if (!isHand && isBody && !showBody) continue;
      const p: [number, number, number] = [j.x * SCALE, j.z * SCALE, -j.y * SCALE];
      pos.push(p);
      if (isEnd) endPos.push(p);
    }

    const handLns: [number, number, number][][] = [];
    const bodyLns: [number, number, number][][] = [];
    const add = (
      pairs: [string, string][],
      into: [number, number, number][][],
    ) => {
      for (const [a, b] of pairs) {
        const ja = frame.joints[a];
        const jb = frame.joints[b];
        if (!ja || !jb) continue;
        if (
          !Number.isFinite(ja.x) ||
          !Number.isFinite(ja.y) ||
          !Number.isFinite(ja.z) ||
          !Number.isFinite(jb.x) ||
          !Number.isFinite(jb.y) ||
          !Number.isFinite(jb.z)
        ) {
          continue;
        }
        into.push([
          [ja.x * SCALE, ja.z * SCALE, -ja.y * SCALE],
          [jb.x * SCALE, jb.z * SCALE, -jb.y * SCALE],
        ]);
      }
    };
    if (showHand) add(HAND_CONNECTIONS, handLns);
    if (showBody) add(BODY_CONNECTIONS, bodyLns);

    // Visual3D-style head: oriented ellipsoid built from the head segment
    // (head_a = ear-line midpoint, head_b = apex) plus head_front (HEAD_FORHEAD)
    // for the anterior axis. Falls back to an axis-aligned ellipsoid if
    // head_front is missing / invalid.
    let head: Ellipsoid | null = null;
    const isFinite3 = (j?: { x: number; y: number; z: number }) =>
      !!j && Number.isFinite(j.x) && Number.isFinite(j.y) && Number.isFinite(j.z);
    const ja = frame.joints['head_a'];
    const jb = frame.joints['head_b'];
    const jf = frame.joints['head_front'];
    if (isFinite3(ja) && isFinite3(jb)) {
      const a = new THREE.Vector3(ja!.x * SCALE, ja!.z * SCALE, -ja!.y * SCALE);
      const b = new THREE.Vector3(jb!.x * SCALE, jb!.z * SCALE, -jb!.y * SCALE);
      const center = a.clone().add(b).multiplyScalar(0.5);
      const yAxis = b.clone().sub(a);
      const length = yAxis.length();
      if (length > 1e-4) {
        yAxis.normalize();

        let xAxis: THREE.Vector3 | null = null;
        if (isFinite3(jf)) {
          const f = new THREE.Vector3(jf!.x * SCALE, jf!.z * SCALE, -jf!.y * SCALE);
          const front = f.sub(center);
          // Project front onto the plane perpendicular to the cranial axis
          // so xAxis stays orthogonal to yAxis.
          front.sub(yAxis.clone().multiplyScalar(front.dot(yAxis)));
          if (front.lengthSq() > 1e-8) xAxis = front.normalize();
        }
        if (!xAxis) {
          // No anterior reference — pick any vector not parallel to yAxis.
          xAxis = Math.abs(yAxis.dot(new THREE.Vector3(1, 0, 0))) < 0.95
            ? new THREE.Vector3(1, 0, 0)
            : new THREE.Vector3(0, 0, 1);
          xAxis.sub(yAxis.clone().multiplyScalar(xAxis.dot(yAxis))).normalize();
        }
        const zAxis = new THREE.Vector3().crossVectors(xAxis, yAxis).normalize();
        // Re-orthogonalize x to guarantee a clean basis.
        xAxis = new THREE.Vector3().crossVectors(yAxis, zAxis).normalize();

        const basis = new THREE.Matrix4().makeBasis(xAxis, yAxis, zAxis);
        const q = new THREE.Quaternion().setFromRotationMatrix(basis);
        head = {
          position: [center.x, center.y, center.z],
          quaternion: [q.x, q.y, q.z, q.w],
          // X = anterior (face-back), Y = cranial (along segment), Z = lateral.
          // Adult head proportions roughly: AP > cranial > lateral.
          scale: [length * 0.95, length * 1.05, length * 0.75],
        };
      }
    }

    return {
      positions: pos,
      handLines: handLns,
      bodyLines: bodyLns,
      endPositions: endPos,
      headEllipsoid: head,
    };
  }, [frame, handJointSet, bodyJointSet, endMarkers, showHand, showBody]);

  useFrame(() => {
    if (!instRef.current) return;
    for (let i = 0; i < positions.length; i++) {
      dummy.position.set(...positions[i]);
      dummy.updateMatrix();
      instRef.current.setMatrixAt(i, dummy.matrix);
    }
    instRef.current.instanceMatrix.needsUpdate = true;
    instRef.current.count = positions.length;
  });

  // Active fingertip highlights — position + color per hand.
  const fingerHighlights = useMemo(() => {
    if (!frame || !showHand) return [] as Array<{ key: string; pos: [number, number, number]; color: string }>;
    const out: Array<{ key: string; pos: [number, number, number]; color: string }> = [];
    const addOne = (hand: 'L' | 'R', finger: number | null) => {
      if (finger == null) return;
      const name = fingertipMarker(hand, finger);
      if (!name) return;
      const j = frame.joints[name];
      if (!j || !Number.isFinite(j.x) || !Number.isFinite(j.y) || !Number.isFinite(j.z)) return;
      out.push({
        key: `${hand}-${finger}`,
        pos: [j.x * SCALE, j.z * SCALE, -j.y * SCALE],
        color: FINGER_COLORS[finger - 1] ?? BRAND.TERTIARY,
      });
    };
    addOne('L', activeFingerL);
    addOne('R', activeFingerR);
    return out;
  }, [frame, activeFingerL, activeFingerR, showHand]);

  if (!frame) return null;

  return (
    <group>
      <instancedMesh ref={instRef} args={[undefined, undefined, 300]}>
        <sphereGeometry args={[0.005, 10, 10]} />
        <meshBasicMaterial color={BRAND.PRIMARY} />
      </instancedMesh>

      {handLines.map((pts, i) => (
        <Line key={`h-${i}`} points={pts} color={BRAND.PRIMARY} lineWidth={1.8} />
      ))}
      {bodyLines.map((pts, i) => (
        <Line key={`b-${i}`} points={pts} color={BRAND.SECONDARY} lineWidth={2.6} />
      ))}

      {showBody && headEllipsoid && (
        <mesh
          position={headEllipsoid.position}
          quaternion={headEllipsoid.quaternion}
          scale={headEllipsoid.scale}
        >
          <sphereGeometry args={[1, 28, 24]} />
          <meshBasicMaterial color={BRAND.SECONDARY} transparent opacity={0.4} />
        </mesh>
      )}

      {endPositions.map((pos, i) => (
        <mesh key={`end-${i}`} position={pos}>
          <sphereGeometry args={[0.011, 16, 16]} />
          <meshBasicMaterial color={BRAND.TERTIARY} />
        </mesh>
      ))}

      {/* Active fingertip overlays (outer glow ring + solid sphere) */}
      {fingerHighlights.map((h) => (
        <group key={h.key} position={h.pos}>
          <mesh>
            <sphereGeometry args={[0.018, 20, 20]} />
            <meshBasicMaterial color={h.color} transparent opacity={0.3} />
          </mesh>
          <mesh>
            <sphereGeometry args={[0.011, 20, 20]} />
            <meshBasicMaterial color={h.color} />
          </mesh>
        </group>
      ))}
    </group>
  );
}

function useKeyboardTexture(): THREE.Texture | null {
  const [tex, setTex] = useState<THREE.Texture | null>(null);
  useEffect(() => {
    let cancelled = false;
    const loader = new THREE.TextureLoader();
    loader.load('/keyboard.png', (t) => {
      if (cancelled) return;
      (t as any).colorSpace = (THREE as any).SRGBColorSpace ?? (t as any).colorSpace;
      t.anisotropy = 4;
      setTex(t);
    });
    return () => {
      cancelled = true;
    };
  }, []);
  return tex;
}

interface KeyboardProps {
  motion: MotionData | null;
}

// Renders keyboard.png as a textured plane aligned to LEnd → REnd. Position
// computed once from the first frame with valid end markers (the piano is
// stationary, so per-frame recomputation would only add jitter).
function Keyboard({ motion }: KeyboardProps) {
  const texture = useKeyboardTexture();

  const geom = useMemo(() => {
    if (!motion) return null;
    const ok = (p: { x: number; y: number; z: number } | undefined) =>
      !!p && Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z);
    const toScene = (p: { x: number; y: number; z: number }): [number, number, number] => [
      p.x * SCALE, p.z * SCALE, -p.y * SCALE,
    ];
    const avg = (a: [number, number, number], b: [number, number, number]): [number, number, number] => [
      (a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2,
    ];
    for (const f of motion.frames) {
      const L = f.joints['LEnd'];
      const R = f.joints['REnd'];
      const L1 = f.joints['LEnd1'];
      const R1 = f.joints['REnd1'];
      if (!ok(L) || !ok(R)) continue;

      const pL = toScene(L!);
      const pR = toScene(R!);
      const has4 = ok(L1) && ok(R1);
      const pL1 = has4 ? toScene(L1!) : null;
      const pR1 = has4 ? toScene(R1!) : null;

      // L / R centers — pair midpoints when the second marker exists, else the
      // single marker. The line cL → cR sits on the image's horizontal midline.
      const cL = pL1 ? avg(pL, pL1) : pL;
      const cR = pR1 ? avg(pR, pR1) : pR;
      const center: [number, number, number] = avg(cL, cR);

      // Width axis in full 3D (includes Y diff so roll is captured).
      const u = new THREE.Vector3(cR[0] - cL[0], cR[1] - cL[1], cR[2] - cL[2]);
      const width = u.length();
      if (width < 1e-4) continue;
      u.divideScalar(width);

      // Plane normal: 4-marker → from raw depth cross-product (captures pitch);
      // 2-marker → project world-up onto the plane perpendicular to u.
      let n = new THREE.Vector3();
      if (pL1 && pR1) {
        const rd = new THREE.Vector3(
          ((pL1[0] - pL[0]) + (pR1[0] - pR[0])) / 2,
          ((pL1[1] - pL[1]) + (pR1[1] - pR[1])) / 2,
          ((pL1[2] - pL[2]) + (pR1[2] - pR[2])) / 2,
        );
        n.crossVectors(u, rd);
        if (n.lengthSq() < 1e-10) {
          // width and depth nearly parallel → fall back to world-up.
          const up = new THREE.Vector3(0, 1, 0);
          n.copy(up).addScaledVector(u, -up.dot(u));
        }
      } else {
        const up = new THREE.Vector3(0, 1, 0);
        n.copy(up).addScaledVector(u, -up.dot(u));
      }
      if (n.lengthSq() < 1e-10) continue;
      n.normalize();
      // Keep the plane's outward normal pointing up (y>0) for stable texture
      // side regardless of marker ordering.
      if (n.y < 0) n.negate();

      // Local +Y of the plane in world — orthogonal to u, lies in the plane.
      const v = new THREE.Vector3().crossVectors(n, u);

      const mat = new THREE.Matrix4().makeBasis(u, v, n);
      const q = new THREE.Quaternion().setFromRotationMatrix(mat);

      return {
        position: center,
        quaternion: [q.x, q.y, q.z, q.w] as [number, number, number, number],
        width,
      };
    }
    return null;
  }, [motion]);

  const aspect = texture?.image
    ? (texture.image.naturalWidth ?? texture.image.width) /
      (texture.image.naturalHeight ?? texture.image.height)
    : 8.17; // real 88-key keyboard width/depth ≈ 1225/150

  if (!texture || !geom) return null;
  const depth = geom.width / aspect;

  return (
    <mesh
      position={[geom.position[0], geom.position[1] - 0.02, geom.position[2]]}
      quaternion={geom.quaternion}
    >
      <planeGeometry args={[geom.width, depth]} />
      <meshBasicMaterial
        map={texture}
        transparent
        opacity={0.95}
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </mesh>
  );
}

interface SceneProps {
  frame: MotionFrame | null;
  motion: MotionData | null;
  cameraPreset: CameraPreset;
  showHand: boolean;
  showBody: boolean;
  activeFingerL: number | null;
  activeFingerR: number | null;
}

// Camera framing now centers on the upper-body/hand area (y ≈ .3m above origin)
// — body skeleton vertices sit between 0 and 0.7m so this keeps both hand
// (origin-centered) and body in view.
const CAMERA_POSITIONS: Record<CameraPreset, [number, number, number]> = {
  top: [0, 1.4468, -0.000578],
  front: [0, 0.4630, 1.2731],
  side: [2.2, 0.8, 0],
  '3d': [1.4575, 1.4575, 1.4575],
  fingering: [0.4858, 0.4858, 0.4858],
};
const CAMERA_TARGET: [number, number, number] = [0, 0, 0];

function Scene({ frame, motion, cameraPreset, showHand, showBody, activeFingerL, activeFingerR }: SceneProps) {
  const controlsRef = useRef<any>(null);

  useEffect(() => {
    if (controlsRef.current) {
      const pos = CAMERA_POSITIONS[cameraPreset] ?? CAMERA_POSITIONS['3d'];
      controlsRef.current.object.position.set(...pos);
      controlsRef.current.target.set(...CAMERA_TARGET);
      controlsRef.current.update();
    }
  }, [cameraPreset]);

  return (
    <>
      <ambientLight intensity={0.6} />
      <directionalLight position={[2, 4, 3]} intensity={0.4} />
      {/* Three.js Color does not accept 8-digit #RRGGBBAA. Use light tints of the
          primary navy on a white background instead of alpha. */}
      <Grid
        args={[4, 4]}
        cellSize={0.1}
        cellColor="#D6D6E5"
        sectionSize={0.5}
        sectionColor="#B8B8D0"
        fadeDistance={4}
        position={[0, -0.01, 0]}
      />
      <axesHelper args={[0.25]} />
      <Keyboard motion={motion} />
      <Skeleton
        frame={frame}
        showHand={showHand}
        showBody={showBody}
        activeFingerL={activeFingerL}
        activeFingerR={activeFingerR}
      />
      <OrbitControls ref={controlsRef} makeDefault />
      <GizmoHelper alignment="top-right" margin={[60, 60]}>
        <GizmoViewport />
      </GizmoHelper>
    </>
  );
}

export interface MotionViewerProps {
  /** Loaded motion data (hand + body merged). null = empty scene (grid + gizmo only). */
  motion: MotionData | null;
  /** Frame index to render (clamped to frameCount - 1). */
  currentFrame: number;
  /** Camera preset. Default: '3d'. */
  cameraPreset?: CameraPreset;
  /** Show hand bones. Default: true. */
  showHand?: boolean;
  /** Show body bones. Default: true. */
  showBody?: boolean;
  /** Fingering data (events + frames). Used to highlight active fingertip per hand. */
  fingering?: FingeringData | null;
  /** Current playback time in seconds — used to look up active fingers from `fingering`. */
  currentTimeSec?: number;
}

export default function MotionViewer({
  motion,
  currentFrame,
  cameraPreset = '3d',
  showHand = true,
  showBody = true,
  fingering,
  currentTimeSec = 0,
}: MotionViewerProps) {
  const frame: MotionFrame | null = motion
    ? motion.frames[Math.min(currentFrame, motion.frameCount - 1)] ?? null
    : null;

  const { L: activeFingerL, R: activeFingerR } = useMemo(
    () => getActiveFingers(fingering ?? null, currentTimeSec),
    [fingering, currentTimeSec],
  );

  return (
    <Canvas
      camera={{ position: [1.8, 1.8, 1.8], fov: 30, near: 0.01, far: 100 }}
      style={{ width: '100%', height: '100%', background: SURFACE.paper }}
      gl={{ antialias: true }}
    >
      <Scene
        frame={frame}
        motion={motion}
        cameraPreset={cameraPreset}
        showHand={showHand}
        showBody={showBody}
        activeFingerL={activeFingerL}
        activeFingerR={activeFingerR}
      />
    </Canvas>
  );
}
