import { useState, type ReactNode } from 'react';
import { styled } from '@mui/material/styles';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { BRAND, GRADIENT, SURFACE } from '@/theme';

const Root = styled('header')({
  display: 'flex',
  flexDirection: 'column',
  gap: 6,
});

const TitleRow = styled('div')({
  display: 'flex',
  alignItems: 'baseline',
  gap: 12,
  flexWrap: 'wrap',
  '& .title': {
    fontSize: 22,
    fontWeight: 700,
    letterSpacing: '-0.01em',
    color: SURFACE.textPrimary,
    margin: 0,
  },
  '& .title .accent': {
    background: GRADIENT.brand,
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
  },
  '& .subtitle': {
    fontSize: 13,
    color: SURFACE.textSecondary,
    fontWeight: 500,
  },
  '& .subtitle .count': {
    color: BRAND.PRIMARY,
    fontWeight: 700,
  },
  '& .about-toggle': {
    marginLeft: 'auto',
    display: 'inline-flex',
    alignItems: 'center',
    gap: 4,
    fontSize: 12,
    fontWeight: 600,
    color: SURFACE.textSecondary,
    background: 'transparent',
    border: 'none',
    padding: '4px 6px',
    borderRadius: 6,
    cursor: 'pointer',
    '&:hover': { color: BRAND.PRIMARY, background: `${BRAND.PRIMARY}08` },
    '& svg': {
      fontSize: 16,
      transition: 'transform 0.2s ease',
    },
    '&.open svg': { transform: 'rotate(180deg)' },
  },
});

const ReleaseNotice = styled('div')({
  display: 'inline-flex',
  alignSelf: 'flex-start',
  alignItems: 'center',
  gap: 8,
  padding: '6px 12px',
  borderRadius: 999,
  border: `1px solid ${BRAND.PRIMARY}33`,
  background: `${BRAND.PRIMARY}08`,
  fontSize: 12,
  lineHeight: 1.4,
  color: SURFACE.textPrimary,
  '& .label': {
    fontWeight: 700,
    color: BRAND.PRIMARY,
    letterSpacing: '0.02em',
    textTransform: 'uppercase',
    fontSize: 11,
  },
  '& .sep': {
    width: 1,
    height: 12,
    background: `${BRAND.PRIMARY}33`,
  },
  '& .text': {
    fontWeight: 500,
    color: SURFACE.textSecondary,
  },
  '& .text strong': {
    color: BRAND.PRIMARY,
    fontWeight: 700,
  },
});

const AboutPanel = styled('div')({
  marginTop: 10,
  padding: '16px 18px',
  borderRadius: 10,
  border: `1px solid ${SURFACE.border}`,
  background: `${BRAND.PRIMARY}04`,
  display: 'flex',
  flexDirection: 'column',
  gap: 10,
  fontSize: 13,
  lineHeight: 1.55,
  color: SURFACE.textSecondary,
  '& .row': {
    display: 'flex',
    gap: 24,
    flexWrap: 'wrap',
  },
  '& .row .stat': {
    display: 'inline-flex',
    alignItems: 'baseline',
    gap: 5,
  },
  '& .row .stat .num': {
    fontSize: 15,
    fontWeight: 700,
    color: SURFACE.textPrimary,
  },
  '& p': { margin: 0 },
});

interface Props {
  trialCount: number;
  mode: 'sample-preview' | 'full-release';
  leading?: ReactNode;
}

export default function HomeHeader({ trialCount, mode, leading }: Props) {
  const [open, setOpen] = useState(false);

  const label =
    mode === 'sample-preview'
      ? `${trialCount} sample trials preview`
      : `${trialCount} trials`;

  return (
    <Root>
      <TitleRow>
        {leading}
        <h1 className="title">
          <span className="accent">SKY Piano</span>
        </h1>
        <span className="subtitle">
          Multimodal piano performance dataset ·{' '}
          <span className="count">{label}</span>
        </span>
        <button
          className={`about-toggle${open ? ' open' : ''}`}
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
        >
          About
          <ExpandMoreIcon />
        </button>
      </TitleRow>
      {mode === 'sample-preview' && (
        <ReleaseNotice>
          <span className="label">ISMIR 2026 Submission</span>
          <span className="sep" />
          <span className="text">
            Sample preview for peer review. Full dataset will be released on{' '}
            <strong>Zenodo</strong> upon acceptance.
          </span>
        </ReleaseNotice>
      )}
      {open && (
        <AboutPanel>
          <div className="row">
            <span className="stat">
              <span className="num">19</span> pianists (7 pro + 12 amateur)
            </span>
            <span className="stat">
              <span className="num">11 h</span> synchronized performance
            </span>
            <span className="stat">
              <span className="num">4</span> recording sessions
            </span>
            <span className="stat">
              <span className="num">2</span> mocap systems
            </span>
          </div>
          <p>
            {mode === 'sample-preview'
              ? 'This preview exposes a representative subset for ISMIR 2026 peer review. Full dataset will be released on Zenodo upon acceptance.'
              : 'Full multimodal release — every trial is scrubbable across video, hand/body mocap, audio, MIDI, fingering, and Visual3D kinematics.'}
          </p>
        </AboutPanel>
      )}
    </Root>
  );
}
