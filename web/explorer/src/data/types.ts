export interface RawDataItem {
  name: string;
  recordingName: string;
  pieceName: string;
  level: string;
  player: string;
  duration: string;
  timeCodeLabel: boolean | null;
  preProcessing: boolean | null;
  reviewed: boolean | null;
  annotator: string;
  date: string;
  memo: string;
}

export interface DatasetManifestEntry {
  date: string;
  label: string;
  status: 'ready' | 'pending';
  basePath: string;
  dataFile: string;
  adapter: string;
  modalities: {
    video: boolean;
    audio: boolean;
    motion: boolean;
    midi: boolean;
    score: boolean;
  };
}

export interface DatasetManifest {
  version: number;
  generatedAt: string;
  datasets: DatasetManifestEntry[];
}

export interface MediaAssetSet {
  thumbnail?: string;
  video?: {
    top?: string;
    left?: string;
    right?: string;
  };
  audio?: string;
  motion?: string;
  midi?: string;
  score?: string;
}

export interface NormalizedDataItem extends RawDataItem {
  dateKey: string;
  assets: MediaAssetSet;
}

export interface DateAdapter {
  id: string;
  normalize(item: RawDataItem, dateKey: string, basePath: string): NormalizedDataItem;
}

// ── Motion (hand + body) ─────────────────────────────────────────────────────

export interface Joint {
  x: number;
  y: number;
  z: number;
}

export interface MotionFrame {
  frame: number;
  joints: Record<string, Joint>;
}

export interface MotionData {
  trialId: string;
  filename: string;
  stream: 'raw_flagged' | 'imputed';
  kind: 'hand' | 'body' | 'hand+body';
  frames: MotionFrame[];
  jointNames: string[];
  frameCount: number;
  fps: number;
}

// ── Fingering ────────────────────────────────────────────────────────────────

export type HandLabel = 'L' | 'R';

export interface FingeringEvent {
  /** MIDI note onset in seconds */
  onset: number;
  /** MIDI note offset in seconds */
  offset: number;
  /** MIDI pitch 0-127 */
  pitch: number;
  /** Hand label or null when unassigned */
  hand: HandLabel | null;
  /** 1-5 (thumb..pinky) or null */
  finger: number | null;
  /** Convenience finger name (T/I/M/R/P) */
  fingerName?: string | null;
  /** Distance (mm) from fingertip to key at onset — quality indicator */
  distanceMm?: number | null;
}

export interface FingeringData {
  trialId: string;
  events: FingeringEvent[];
}

// ── Kinematics (Visual3D) ────────────────────────────────────────────────────

export interface KinematicsSample {
  /** Time in seconds */
  time: number;
  /** Channel name → value (radians or degrees, as shipped) */
  channels: Record<string, number>;
}

export interface KinematicsData {
  trialId: string;
  channelNames: string[];
  samples: KinematicsSample[];
  fps: number;
}

// ── Sample trial manifest extension ──────────────────────────────────────────

/** Minimal trial descriptor that drives Home grid + SampleView. */
export interface SampleTrialEntry {
  trialId: string;
  session: string;
  subject: string;
  level: 'Professional' | 'Amateur';
  task: 'T0' | 'T1' | 'T2' | 'T9';
  pieceLabel: string;
  /** Graded difficulty of the piece itself (Set1 pool level). Independent of
      performer skill — used on T2 Graded cards to show curriculum tier. */
  pieceLevel?: 'Beginner' | 'Intermediate' | 'Advanced' | 'Professional';
  basePath: string;
  videoViews: Array<'top' | 'right' | 'low_right' | 'left'>;
  modalities: {
    audio: boolean;
    midi: boolean;
    motionHand: boolean;
    motionBody: boolean;
    kinematics: boolean;
    fingering: boolean;
    video: boolean;
    /** Optional — only set for pieces with published scores. */
    score?: boolean;
  };
  /**
   * Optional free-form blurb surfaced as a hover tooltip on the card.
   * Useful at submission-time to explain why a specific sample was chosen;
   * optional at scale.
   */
  narrativeBlurb?: string;
  /** Paired counterpart (e.g. pro↔amateur on the same exercise). */
  comparePartnerId?: string;
  /** Approximate duration shown on the card (e.g. "02:34"). */
  durationLabel?: string;
}
