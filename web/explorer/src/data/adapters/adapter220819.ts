import type { DateAdapter, NormalizedDataItem, RawDataItem } from '@/data/types';
import { withBase } from '@/utils/assetUrl';

const normalizePieceFile = (pieceName: string, ext: string): string => `${pieceName}${ext}`;

const parseConditionPlayer = (recording: string): { condition: string; player: string } | null => {
  const m = recording.match(/_C(\d+)_N(\d+)_/);
  if (!m) return null;
  return { condition: `C${m[1]}`, player: `N${m[2]}` };
};

const buildThumbnailPath = (dateKey: string, recording: string, piece: string): string => {
  return withBase(`/Data/Thumbnails/${dateKey}/${recording}/${piece}.webp`);
};

export const adapter220819: DateAdapter = {
  id: '220819',
  normalize(item: RawDataItem, dateKey: string, basePath: string): NormalizedDataItem {
    const recording = item.recordingName;
    const piece = item.pieceName;

    const parsed = parseConditionPlayer(recording);
    const directMotionPath = parsed
      ? `${basePath}/${parsed.condition}/${parsed.player}/Motion/Hands/sliced/${normalizePieceFile(recording, '.csv')}`
      : '';

    const pieceBasedMotionPath = `${basePath}/Motion/Hands/Sliced/${recording}/${normalizePieceFile(piece, '.csv')}`;

    return {
      ...item,
      dateKey,
      assets: {
        thumbnail: buildThumbnailPath(dateKey, recording, piece),
        video: {
          top: `${basePath}/Video/01/${recording}/${normalizePieceFile(piece, '.mp4')}`,
          left: `${basePath}/Video/02/${recording}/${normalizePieceFile(piece, '.mp4')}`,
          right: `${basePath}/Video/03/${recording}/${normalizePieceFile(piece, '.mp4')}`
        },
        audio: `${basePath}/Audio/Sliced_mp3/${recording}/${normalizePieceFile(piece, '.mp3')}`,
        motion: directMotionPath || pieceBasedMotionPath,
        midi: `${basePath}/MIDI/Sliced/${recording}/${normalizePieceFile(piece, '.mid')}`,
        score: `${basePath}/Score/2-${normalizePieceFile(piece, '.mxl')}`
      }
    };
  }
};
