import { useEffect, useState, type ReactNode } from 'react';
import { styled } from '@mui/material/styles';
import { BRAND, GRADIENT, SURFACE } from '@/theme';

// Minimum viewport width for the multi-pane synchronized viewer layout
// (Video + Motion + Piano-roll + Kinematics + Audio waveform). Below this
// the dashboard collapses into unreadable slivers, so we gate instead of
// attempting a responsive mobile layout.
const MIN_DESKTOP_WIDTH = 1024;

const Overlay = styled('div')({
  position: 'fixed',
  inset: 0,
  background: SURFACE.bg,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  padding: 24,
  zIndex: 9999,
});

const Card = styled('div')({
  maxWidth: 520,
  width: '100%',
  background: SURFACE.paper,
  border: `1px solid ${SURFACE.border}`,
  borderRadius: 16,
  boxShadow: SURFACE.shadow,
  padding: '32px 28px',
  textAlign: 'center',
});

const BrandStrip = styled('div')({
  height: 4,
  borderRadius: 2,
  background: GRADIENT.brand,
  margin: '0 auto 20px',
  width: 56,
});

const Title = styled('h1')({
  margin: 0,
  fontSize: 22,
  fontWeight: 700,
  letterSpacing: '-0.01em',
  color: BRAND.PRIMARY,
});

const Body = styled('p')({
  margin: '14px 0 0',
  fontSize: 14,
  lineHeight: 1.6,
  color: SURFACE.textSecondary,
});

const Meta = styled('div')({
  marginTop: 20,
  padding: '10px 14px',
  borderRadius: 8,
  background: SURFACE.bg,
  border: `1px solid ${SURFACE.border}`,
  fontSize: 12,
  color: SURFACE.textMuted,
  fontVariantNumeric: 'tabular-nums',
});

interface Props {
  children: ReactNode;
}

export default function DesktopOnlyGate({ children }: Props) {
  const [width, setWidth] = useState<number>(() =>
    typeof window !== 'undefined' ? window.innerWidth : MIN_DESKTOP_WIDTH,
  );

  useEffect(() => {
    const onResize = () => setWidth(window.innerWidth);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  if (width >= MIN_DESKTOP_WIDTH) return <>{children}</>;

  return (
    <Overlay>
      <Card>
        <BrandStrip />
        <Title>Desktop Only</Title>
        <Body>
          SKY Piano Explorer is a dataset viewer built to explore audio, MIDI, motion, video and kinematics side by side. Please open it in a desktop browser at least&nbsp;
          <strong>{MIN_DESKTOP_WIDTH}px</strong> wide.
        </Body>
        <Meta>
          current viewport&nbsp;·&nbsp;{width}px &lt; {MIN_DESKTOP_WIDTH}px
        </Meta>
      </Card>
    </Overlay>
  );
}
