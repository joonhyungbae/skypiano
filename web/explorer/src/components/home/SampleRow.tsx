import { Tooltip } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { styled } from '@mui/material/styles';
import { BRAND, SURFACE } from '@/theme';
import type { SampleTrialEntry } from '@/data/types';

export const TASK_COLORS: Record<SampleTrialEntry['task'], string> = {
  T0: BRAND.SECONDARY,
  T1: BRAND.PRIMARY,
  T2: BRAND.TERTIARY,
  T9: BRAND.PRIMARY,
};

// Letter-pill modality presence; tooltip uses paper Section 3 wording.
type ModalityKey =
  | 'audio'
  | 'midi'
  | 'motionHand'
  | 'motionBody'
  | 'kinematics'
  | 'fingering'
  | 'video'
  | 'score';

const MODALITY_DEF: ReadonlyArray<{
  key: ModalityKey;
  letter: string;
  label: string;
}> = [
  { key: 'audio', letter: 'A', label: 'Audio · 48kHz/24bit' },
  { key: 'midi', letter: 'M', label: 'MIDI · Disklavier direct capture' },
  { key: 'motionHand', letter: 'H', label: 'Hand mocap · OptiTrack 120Hz' },
  { key: 'motionBody', letter: 'B', label: 'Body mocap · Qualisys 120Hz' },
  { key: 'kinematics', letter: 'K', label: 'Visual3D-derived joint angles' },
  { key: 'fingering', letter: 'F', label: 'Pseudo fingering · per-note L/R+1-5' },
  { key: 'video', letter: 'V', label: 'Multi-view video · EgoBlur anonymized' },
  { key: 'score', letter: 'S', label: 'Engraved score (MusicXML)' },
];

// Shared 8-column grid template — also used by the header in HomeView.
// Task | Level | Grade | Piece | Subject | Session | Dur | Modalities
export const ROW_GRID =
  '52px 64px 96px minmax(180px, 1fr) 88px 110px 60px 200px';

const Row = styled('div', {
  shouldForwardProp: (prop) => prop !== 'accentColor',
})<{ accentColor: string }>(({ accentColor }) => ({
  display: 'grid',
  gridTemplateColumns: ROW_GRID,
  alignItems: 'center',
  columnGap: 12,
  padding: '8px 14px 8px 11px',
  borderLeft: `3px solid ${accentColor}`,
  borderBottom: `1px solid ${SURFACE.border}`,
  background: SURFACE.paper,
  cursor: 'pointer',
  transition: 'background 0.12s ease',
  '&:hover': {
    background: `${BRAND.PRIMARY}06`,
  },
  '& .chip': {
    fontSize: 10,
    fontWeight: 700,
    padding: '3px 7px',
    borderRadius: 4,
    letterSpacing: '0.03em',
    textTransform: 'uppercase',
    border: '1px solid',
    display: 'inline-flex',
    alignItems: 'center',
    flexShrink: 0,
  },
  '& .chip.level-pro': {
    background: `${BRAND.PRIMARY}10`,
    color: BRAND.PRIMARY,
    borderColor: `${BRAND.PRIMARY}33`,
  },
  '& .chip.level-amateur': {
    background: `${BRAND.SECONDARY}10`,
    color: BRAND.SECONDARY,
    borderColor: `${BRAND.SECONDARY}33`,
  },
  '& .chip.piece-beginner': {
    background: '#E8F5E910',
    color: '#2E7D32',
    borderColor: '#2E7D3240',
  },
  '& .chip.piece-intermediate': {
    background: '#E3F2FD10',
    color: '#1565C0',
    borderColor: '#1565C040',
  },
  '& .chip.piece-advanced': {
    background: `${BRAND.TERTIARY}10`,
    color: BRAND.TERTIARY,
    borderColor: `${BRAND.TERTIARY}40`,
  },
  '& .chip.piece-professional': {
    background: '#FFF3E010',
    color: '#E65100',
    borderColor: '#E6510040',
  },
  '& .task-cell': {
    display: 'flex',
    alignItems: 'center',
    gap: 4,
  },
  '& .piece': {
    fontSize: 13,
    fontWeight: 600,
    color: SURFACE.textPrimary,
    margin: 0,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    minWidth: 0,
  },
  '& .text-cell': {
    fontSize: 12,
    color: SURFACE.textSecondary,
    fontWeight: 500,
    fontVariantNumeric: 'tabular-nums',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  '& .text-cell.subject': {
    fontFamily: '"SFMono-Regular", "Menlo", "Consolas", monospace',
    fontSize: 11,
    color: SURFACE.textPrimary,
    fontWeight: 600,
  },
  '& .modalities': {
    display: 'flex',
    gap: 3,
  },
  '& .mod': {
    width: 18,
    height: 18,
    borderRadius: 4,
    fontSize: 10,
    fontWeight: 700,
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontFamily: '"SFMono-Regular", "Menlo", "Consolas", monospace',
    border: '1px solid',
    flexShrink: 0,
  },
  '& .mod.on': {
    background: `${BRAND.PRIMARY}12`,
    color: BRAND.PRIMARY,
    borderColor: `${BRAND.PRIMARY}33`,
  },
  '& .mod.off': {
    background: 'transparent',
    color: SURFACE.textMuted,
    borderColor: SURFACE.border,
    opacity: 0.45,
  },
}));

const formatSessionDate = (s: string): string => {
  if (/^\d{6}$/.test(s)) {
    return `20${s.slice(0, 2)}.${s.slice(2, 4)}.${s.slice(4, 6)}`;
  }
  return s;
};

interface Props {
  sample: SampleTrialEntry;
}

export default function SampleRow({ sample }: Props) {
  const navigate = useNavigate();
  const taskColor = TASK_COLORS[sample.task];
  const levelClass = sample.level === 'Professional' ? 'level-pro' : 'level-amateur';
  const pieceLevelClass = sample.pieceLevel
    ? `piece-${sample.pieceLevel.toLowerCase()}`
    : null;

  const viewCount = sample.videoViews.length;

  const rowContent = (
    <Row
      role="row"
      accentColor={taskColor}
      onClick={() => navigate(`/sample/${sample.trialId}`)}
    >
      {/* Task */}
      <div className="task-cell" role="cell">
        <span
          className="chip"
          style={{
            background: `${taskColor}10`,
            color: taskColor,
            borderColor: `${taskColor}33`,
          }}
        >
          {sample.task}
        </span>
      </div>

      {/* Level (performer skill) */}
      <div className="task-cell" role="cell">
        <span className={`chip ${levelClass}`}>
          {sample.level === 'Professional' ? 'Pro' : 'Am'}
        </span>
      </div>

      {/* Grade (piece curriculum tier — T2 only) */}
      <div className="task-cell" role="cell">
        {pieceLevelClass && sample.pieceLevel ? (
          <span className={`chip ${pieceLevelClass}`}>{sample.pieceLevel}</span>
        ) : (
          <span className="text-cell" style={{ opacity: 0.35 }}>—</span>
        )}
      </div>

      {/* Piece */}
      <h3 className="piece" role="cell" title={sample.pieceLabel}>
        {sample.pieceLabel}
      </h3>

      {/* Subject */}
      <div className="text-cell subject" role="cell">
        {sample.subject}
      </div>

      {/* Session */}
      <div className="text-cell" role="cell">
        {formatSessionDate(sample.session)}
      </div>

      {/* Duration */}
      <div className="text-cell" role="cell">
        {sample.durationLabel ?? ''}
      </div>

      {/* Modalities */}
      <div className="modalities" role="cell">
        {MODALITY_DEF.map(({ key, letter, label }) => {
          const present =
            key === 'video'
              ? sample.modalities.video && viewCount > 0
              : key === 'score'
                ? sample.modalities.score === true
                : Boolean(
                    (sample.modalities as Record<string, boolean | undefined>)[key],
                  );
          const tooltip =
            key === 'video' && present
              ? `${label} · ${viewCount}-view`
              : `${label}${present ? '' : ' — not available'}`;
          return (
            <Tooltip
              key={key}
              title={tooltip}
              placement="bottom"
              arrow
              disableInteractive
              enterDelay={300}
            >
              <span className={`mod ${present ? 'on' : 'off'}`}>{letter}</span>
            </Tooltip>
          );
        })}
      </div>
    </Row>
  );

  if (!sample.narrativeBlurb) return rowContent;

  return (
    <Tooltip
      title={sample.narrativeBlurb}
      placement="top"
      arrow
      disableInteractive
      enterDelay={600}
    >
      {rowContent}
    </Tooltip>
  );
}
