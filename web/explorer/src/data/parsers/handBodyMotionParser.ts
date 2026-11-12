import type { Joint, MotionData, MotionFrame } from '@/data/types';

/**
 * Merges the hand and body CSVs into a single `MotionData`. Both CSVs are assumed to share the same timebase (120Hz) and
 * the same frame count — both are spatially aligned outputs of pipeline Stage3.
 *
 * CSV format (same as the widget version):
 *   Header: `<marker>_x, <marker>_y, <marker>_z, ...` (repeated)
 *   Row: one per frame, float, may be blank/NaN
 *
 * marker names are [L/R][Finger T|I|M|R|P][Position M/P/D/F] + body (ACROL, HEADT, etc.).
 */

interface MotionInput {
  handCsv?: string;
  bodyCsv?: string;
  /** Tried if `bodyCsv` misses or returns empty (e.g. no body imputation for Pro). */
  bodyCsvFallback?: string;
  trialId: string;
  stream: 'raw_flagged' | 'imputed';
  /** Default 30 Hz — s07_publish decimates source 120Hz mocap to 30Hz for the
      browser. Override only if loading raw 120Hz CSVs directly. */
  fps?: number;
}

function parseCsvJoints(csvText: string): {
  markers: string[];
  rows: Array<Record<string, Joint>>;
} {
  const lines = csvText.split(/\r?\n/).filter((l) => l.length > 0);
  if (lines.length < 2) return { markers: [], rows: [] };

  const header = lines[0].split(',').map((h) => h.trim());
  type MarkerCol = { marker: string; ix: number; iy: number; iz: number };
  const markerMap = new Map<string, { ix?: number; iy?: number; iz?: number }>();

  header.forEach((h, i) => {
    const m = h.match(/^(.+)_(x|y|z)$/i);
    if (!m) return;
    const [, name, axis] = m;
    const axisLower = axis.toLowerCase() as 'x' | 'y' | 'z';
    const entry = markerMap.get(name) ?? {};
    const key = `i${axisLower}` as 'ix' | 'iy' | 'iz';
    entry[key] = i;
    markerMap.set(name, entry);
  });

  const markers: MarkerCol[] = [];
  for (const [name, ix] of markerMap.entries()) {
    if (ix.ix !== undefined && ix.iy !== undefined && ix.iz !== undefined) {
      markers.push({ marker: name, ix: ix.ix, iy: ix.iy, iz: ix.iz });
    }
  }

  const rows: Array<Record<string, Joint>> = [];
  for (let r = 1; r < lines.length; r++) {
    const cells = lines[r].split(',');
    const joints: Record<string, Joint> = {};
    for (const m of markers) {
      const x = parseFloat(cells[m.ix]);
      const y = parseFloat(cells[m.iy]);
      const z = parseFloat(cells[m.iz]);
      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
        joints[m.marker] = { x, y, z };
      }
    }
    rows.push(joints);
  }

  return { markers: markers.map((m) => m.marker), rows };
}

export async function loadMotion({
  handCsv,
  bodyCsv,
  bodyCsvFallback,
  trialId,
  stream,
  fps = 30,
}: MotionInput): Promise<MotionData> {
  const [handText, bodyTextPrimary] = await Promise.all([
    handCsv ? fetchTextSafe(handCsv) : Promise.resolve(''),
    bodyCsv ? fetchTextSafe(bodyCsv) : Promise.resolve(''),
  ]);
  // Retry body fetch against fallback when primary returned empty (missing
  // file or HTML 404 stub).
  const bodyText =
    bodyTextPrimary ||
    (bodyCsvFallback ? await fetchTextSafe(bodyCsvFallback) : '');

  const hand = handText ? parseCsvJoints(handText) : { markers: [], rows: [] };
  const body = bodyText ? parseCsvJoints(bodyText) : { markers: [], rows: [] };

  const frameCount = Math.max(hand.rows.length, body.rows.length);
  const frames: MotionFrame[] = [];
  for (let i = 0; i < frameCount; i++) {
    const joints: Record<string, Joint> = {
      ...(hand.rows[i] ?? {}),
      ...(body.rows[i] ?? {}),
    };
    frames.push({ frame: i, joints });
  }

  const jointNames = Array.from(new Set([...hand.markers, ...body.markers]));
  const kind: MotionData['kind'] =
    hand.rows.length && body.rows.length
      ? 'hand+body'
      : hand.rows.length
        ? 'hand'
        : 'body';

  return {
    trialId,
    filename: `${handCsv ?? ''}|${bodyCsv ?? ''}`,
    stream,
    kind,
    frames,
    jointNames,
    frameCount,
    fps,
  };
}

async function fetchTextSafe(url: string): Promise<string> {
  try {
    const res = await fetch(url);
    if (!res.ok) return '';
    const text = await res.text();
    // Vite dev server returns index.html for any 404. Reject HTML here so
    // callers don't end up parsing `<!doctype html>` as CSV.
    const head = text.slice(0, 10).trimStart().toLowerCase();
    if (head.startsWith('<!doctype') || head.startsWith('<html')) return '';
    return text;
  } catch {
    return '';
  }
}
