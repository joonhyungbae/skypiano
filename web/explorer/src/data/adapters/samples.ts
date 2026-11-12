import type { SampleTrialEntry } from '@/data/types';

/**
 * Adapter for the paper's `submission sample set`. The file layout is flat:
 *   {basePath}/{trialId}/audio.m4a
 *   {basePath}/{trialId}/midi.mid
 *   {basePath}/{trialId}/motion_hand_raw_flagged.csv
 *   {basePath}/{trialId}/motion_hand_imputed.csv     (220819 excluded)
 *   {basePath}/{trialId}/motion_body_raw_flagged.csv
 *   {basePath}/{trialId}/body_segments.csv
 *   {basePath}/{trialId}/kinematics.csv
 *   {basePath}/{trialId}/video_top.mp4
 *   {basePath}/{trialId}/video_right.mp4
 *   {basePath}/{trialId}/video_low_right.mp4         (240828 only)
 *   {basePath}/{trialId}/video_left.mp4
 *   {basePath}/{trialId}/fingering_events.csv        (220819 excluded)
 *   {basePath}/{trialId}/score.mxl                   (T2 only)
 *   {basePath}/{trialId}/alignment.json              (T2 only)
 */
export interface SampleAssetPaths {
  audio: string;
  midi: string;
  handRawFlagged: string;
  handImputed: string;
  bodyRawFlagged: string;
  /** Visual3D-style 13-segment body (a/b endpoint columns), derived from
      bodyRawFlagged by pipeline/10_body_visual3D. This is what the web
      MotionViewer renders for the body layer. */
  bodySegments: string;
  kinematics: string;
  video: {
    top: string;
    right: string;
    low_right?: string;
    left?: string;
  };
  fingeringEvents: string;
  score: string;
  /** Stage 12 score↔performance alignment JSON. Optional: present only for
      trials with both score and performance MIDI processed. */
  alignment: string;
}

export function buildSamplePaths(entry: SampleTrialEntry): SampleAssetPaths {
  const b = `${entry.basePath.replace(/\/$/, '')}/${entry.trialId}`;
  const video: SampleAssetPaths['video'] = {
    top: `${b}/video_top.mp4`,
    right: `${b}/video_right.mp4`,
  };
  if (entry.videoViews.includes('low_right')) {
    video.low_right = `${b}/video_low_right.mp4`;
  }
  if (entry.videoViews.includes('left')) {
    video.left = `${b}/video_left.mp4`;
  }
  return {
    audio: `${b}/audio.m4a`,
    midi: `${b}/midi.mid`,
    handRawFlagged: `${b}/motion_hand_raw_flagged.csv`,
    handImputed: `${b}/motion_hand_imputed.csv`,
    bodyRawFlagged: `${b}/motion_body_raw_flagged.csv`,
    bodySegments: `${b}/body_segments.csv`,
    kinematics: `${b}/kinematics.csv`,
    video,
    fingeringEvents: `${b}/fingering_events.csv`,
    score: `${b}/score.mxl`,
    alignment: `${b}/alignment.json`,
  };
}
