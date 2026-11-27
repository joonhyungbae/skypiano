import { withBase } from '@/utils/assetUrl';
import type { SampleTrialEntry } from '@/data/types';
import type { PlayerPaths } from '@/views/SampleView';

/**
 * The full release, served next to the explorer by `web/explorer/serve_dataset.py`
 * (a downloaded copy of the Hugging Face dataset, or a Stage 12 release manifest).
 *
 *   {DATASET}/index.json                    metadata tables as JSON
 *   {DATASET}/data/P07/graded/<piece>/r1/   one performance: audio.wav, midi.mid, hand.csv, body.csv, ...
 *   {DATASET}/data/P07/takes/<take>/        one whole recorded take
 *
 * Set VITE_DATASET_URL to serve the release from somewhere else.
 */
export const DATASET_URL: string = (
  import.meta.env.VITE_DATASET_URL || withBase('/dataset')
).replace(/\/$/, '');

/** Every 4th frame of the 120 Hz motion CSVs, so a long piece stays light in the browser. */
const MOTION_STEP = 4;
const MOTION_FPS = 120 / MOTION_STEP;

export type Row = Record<string, string>;

export interface DatasetIndex {
  performances: Row[];
  takes: Row[];
  participants: Row[];
  repertoire: Row[];
  pieces: Row[];
}

let cached: Promise<DatasetIndex | null> | null = null;

/** The metadata tables, or null when no dataset is being served (e.g. on the public site). */
export function loadDatasetIndex(): Promise<DatasetIndex | null> {
  cached ??= (async () => {
    try {
      const res = await fetch(`${DATASET_URL}/index.json`);
      if (!res.ok || !(res.headers.get('content-type') ?? '').includes('json')) return null;
      const j = (await res.json()) as Partial<DatasetIndex>;
      return {
        performances: j.performances ?? [],
        takes: j.takes ?? [],
        participants: j.participants ?? [],
        repertoire: j.repertoire ?? [],
        pieces: j.pieces ?? [],
      };
    } catch {
      return null;
    }
  })();
  return cached;
}

export const TASK_NAMES: Record<string, string> = {
  technique1: 'Technique 1',
  technique2: 'Technique 2',
  technique_extra: 'Technique (extra)',
  graded: 'Graded',
  free: 'Free',
  take: 'Whole take',
};

/** Release task -> the explorer's task code (chip colour only). */
const TASK_CODE: Record<string, SampleTrialEntry['task']> = {
  technique1: 'T0',
  technique2: 'T1',
  technique_extra: 'T0',
  graded: 'T2',
  free: 'T9',
  take: 'T9',
};

export const participantLabel = (id: string) => `P${String(id).padStart(2, '0')}`;

export function pieceTitle(index: DatasetIndex, pieceId: string): string {
  return index.repertoire.find((r) => r.piece_id === pieceId)?.title || pieceId;
}

/** A performance or take row -> the player's entry and asset paths. */
export function toPlayer(
  index: DatasetIndex,
  row: Row,
  kind: 'performance' | 'take',
): { sample: SampleTrialEntry; paths: PlayerPaths; taskLabel: string; title: string } {
  const mods = new Set((row.modalities ?? '').split(/\s+/).filter(Boolean));
  const base = `${DATASET_URL}/${row.path.replace(/^\/+|\/+$/g, '')}`;
  const task = kind === 'take' ? 'take' : row.task;
  const views: SampleTrialEntry['videoViews'] = [];
  if (mods.has('top')) views.push('top');
  if (mods.has('right')) views.push('right');
  if (mods.has('right_low')) views.push('low_right');
  if (mods.has('left')) views.push('left');
  const title =
    kind === 'take'
      ? `${participantLabel(row.participant_id)} · take ${row.source_take ?? row.path.split('/').pop()}`
      : `${pieceTitle(index, row.piece_id)} · ${participantLabel(row.participant_id)} · round ${row.round}`;
  const sample: SampleTrialEntry = {
    trialId: row.path,
    session: row.session ?? '',
    subject: participantLabel(row.participant_id),
    level: row.group === 'professional' ? 'Professional' : 'Amateur',
    task: TASK_CODE[task] ?? 'T9',
    pieceLabel: title,
    basePath: base,
    videoViews: views,
    modalities: {
      audio: mods.has('audio'),
      midi: mods.has('midi'),
      motionHand: mods.has('hand'),
      motionBody: mods.has('body'),
      kinematics: false,
      fingering: mods.has('fingering') || mods.has('fingering_imputed'),
      video: views.length > 0,
      score: false,
    },
  };
  const motion = (f: string) => `${base}/${f}?every=${MOTION_STEP}`;
  const paths: PlayerPaths = {
    audio: `${base}/audio.wav`,
    midi: `${base}/midi.mid`,
    handRawFlagged: motion('hand.csv'),
    // without a gap-filled hand the Imputed stream shows the raw hand
    handImputed: motion(mods.has('hand_imputed') ? 'hand_imputed.csv' : 'hand.csv'),
    bodySegments: `${base}/body_segments.csv?src=${mods.has('body_imputed') ? 'body_imputed' : 'body'}&every=${MOTION_STEP}`,
    bodySegmentsRaw: `${base}/body_segments.csv?src=body&every=${MOTION_STEP}`,
    bodyRawFlagged: motion('body.csv'),
    kinematics: '',
    video: {
      // VideoViewer skips empty sources
      top: mods.has('top') ? `${base}/top.mp4` : '',
      right: mods.has('right') ? `${base}/right.mp4` : '',
      ...(mods.has('right_low') ? { low_right: `${base}/right_low.mp4` } : {}),
      ...(mods.has('left') ? { left: `${base}/left.mp4` } : {}),
    },
    fingeringEvents: `${base}/${mods.has('fingering_imputed') ? 'fingering_imputed.csv' : 'fingering.csv'}`,
    score: '',
    alignment: '',
    motionFps: MOTION_FPS,
  };
  return { sample, paths, taskLabel: TASK_NAMES[task] ?? task, title };
}
