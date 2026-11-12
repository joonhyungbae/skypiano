import type { FingeringData, FingeringEvent, HandLabel } from '@/data/types';

/**
 * Fingering: parses a single file, fingering_events.csv (per-note [onset, offset, pitch, hand,
 * finger, ...]) only. Header names are matched
 * loosely (case/name variations) for compatibility with pipeline output.
 */

function splitCsv(text: string): string[][] {
  return text
    .split(/\r?\n/)
    .filter((l) => l.length > 0)
    .map((l) => l.split(','));
}

function findCol(header: string[], candidates: string[]): number {
  const lower = header.map((h) => h.trim().toLowerCase());
  for (const c of candidates) {
    const idx = lower.indexOf(c.toLowerCase());
    if (idx >= 0) return idx;
  }
  return -1;
}

function parseEventsCsv(text: string): FingeringEvent[] {
  const rows = splitCsv(text);
  if (rows.length < 2) return [];
  const header = rows[0];

  const onsetIdx = findCol(header, ['onset', 'onset_sec', 'start', 'start_sec']);
  const offsetIdx = findCol(header, ['offset', 'offset_sec', 'end', 'end_sec']);
  const pitchIdx = findCol(header, ['pitch', 'midi', 'note']);
  const handIdx = findCol(header, ['hand']);
  const fingerIdx = findCol(header, ['finger', 'finger_index']);
  const fingerNameIdx = findCol(header, ['finger_name', 'fingername']);
  const distIdx = findCol(header, ['distance_mm', 'distance', 'dist_mm']);

  const out: FingeringEvent[] = [];
  for (let r = 1; r < rows.length; r++) {
    const cells = rows[r];
    const onset = parseFloat(cells[onsetIdx]);
    const offset = parseFloat(cells[offsetIdx]);
    const pitch = parseInt(cells[pitchIdx], 10);
    if (!Number.isFinite(onset) || !Number.isFinite(offset) || !Number.isFinite(pitch)) continue;

    const handRaw = handIdx >= 0 ? (cells[handIdx] ?? '').trim().toUpperCase() : '';
    const hand: HandLabel | null = handRaw === 'L' || handRaw === 'R' ? handRaw : null;
    const fingerRaw = fingerIdx >= 0 ? parseInt(cells[fingerIdx], 10) : NaN;
    const finger = Number.isFinite(fingerRaw) && fingerRaw >= 1 && fingerRaw <= 5 ? fingerRaw : null;
    const fingerName =
      fingerNameIdx >= 0 && cells[fingerNameIdx] ? cells[fingerNameIdx].trim() || null : null;
    const distance = distIdx >= 0 ? parseFloat(cells[distIdx]) : NaN;

    out.push({
      onset,
      offset,
      pitch,
      hand,
      finger,
      fingerName,
      distanceMm: Number.isFinite(distance) ? distance : null,
    });
  }
  return out;
}

export async function loadFingering({
  eventsPath,
  trialId,
}: {
  eventsPath?: string;
  trialId: string;
}): Promise<FingeringData> {
  const eventsText = eventsPath ? await fetchTextSafe(eventsPath) : '';
  const events = eventsText ? parseEventsCsv(eventsText) : [];
  return { trialId, events };
}

async function fetchTextSafe(url: string): Promise<string> {
  try {
    const res = await fetch(url);
    if (!res.ok) return '';
    return await res.text();
  } catch {
    return '';
  }
}
