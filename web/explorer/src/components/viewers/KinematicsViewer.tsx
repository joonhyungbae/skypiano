import { useEffect, useMemo, useRef, useState } from 'react';
import { styled } from '@mui/material/styles';
import { loadKinematics } from '@/data/parsers/kinematicsParser';
import type { KinematicsData } from '@/data/types';
import { BRAND, SURFACE } from '@/theme';

const Root = styled('div')({
  position: 'relative',
  width: '100%',
  height: '100%',
  background: SURFACE.paper,
  overflow: 'hidden',
  display: 'flex',
  flexDirection: 'column',
});

const Controls = styled('div')({
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  padding: '8px 12px',
  borderBottom: `1px solid ${SURFACE.border}`,
  fontSize: 11,
  fontWeight: 600,
  color: SURFACE.textMuted,
  '& select': {
    padding: '4px 22px 4px 8px',
    border: `1px solid ${SURFACE.border}`,
    borderRadius: 4,
    background: SURFACE.paper,
    fontSize: 11,
    color: SURFACE.textPrimary,
    appearance: 'none',
    backgroundImage:
      "url(\"data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%235A5A6E' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M6 8l4 4 4-4'/%3e%3c/svg%3e\")",
    backgroundPosition: 'right 4px center',
    backgroundRepeat: 'no-repeat',
    backgroundSize: 12,
    cursor: 'pointer',
    maxWidth: 240,
    '&:focus': { outline: 'none', borderColor: BRAND.PRIMARY },
  },
});

const CanvasWrap = styled('div')({ flex: 1, position: 'relative', minHeight: 0 });
const Canvas = styled('canvas')({ display: 'block', width: '100%', height: '100%' });

const Overlay = styled('div')({
  position: 'absolute',
  inset: 0,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: SURFACE.textMuted,
  fontSize: 13,
  fontStyle: 'italic',
  pointerEvents: 'none',
});

export interface KinematicsViewerProps {
  path: string;
  trialId: string;
  currentTimeSec?: number;
  durationSec?: number;
}

// Friendly labels + unit suffixes for the pseudo-kinematics columns produced
// by pipeline/09_visual3d/steps/s05_compute_kinematics.py
const CHANNEL_META: Record<string, { label: string; unit: string }> = {
  trunk_lean_deg: { label: 'Trunk lean (sagittal)', unit: '°' },
  fingertip_vel_L: { label: 'Fingertip vel · Left (max)', unit: 'mm/s' },
  fingertip_vel_R: { label: 'Fingertip vel · Right (max)', unit: 'mm/s' },
  hand_span_L: { label: 'Hand span · Left (thumb–pinky)', unit: 'mm' },
  hand_span_R: { label: 'Hand span · Right (thumb–pinky)', unit: 'mm' },
};
const channelLabel = (k: string) => CHANNEL_META[k]?.label ?? k;
const channelUnit = (k: string) => CHANNEL_META[k]?.unit ?? '';

export default function KinematicsViewer({
  path,
  trialId,
  currentTimeSec = 0,
  durationSec,
}: KinematicsViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [data, setData] = useState<KinematicsData | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'empty' | 'ready'>('idle');
  const [channel, setChannel] = useState<string>('');
  const [, forceTick] = useState(0);

  useEffect(() => {
    if (!path) {
      setData(null);
      setStatus('empty');
      return;
    }
    let cancelled = false;
    setStatus('loading');
    loadKinematics({ path, trialId }).then((d) => {
      if (cancelled) return;
      if (d.samples.length === 0) {
        setData(null);
        setStatus('empty');
      } else {
        setData(d);
        setStatus('ready');
        // Pick the first channel that actually has finite values — for Sample 5
        // (no body) this skips the empty trunk_lean_deg column.
        const populated = d.channelNames.find((c) =>
          d.samples.some((s) => Number.isFinite(s.channels[c]))
        );
        setChannel(populated ?? d.channelNames[0] ?? '');
      }
    });
    return () => {
      cancelled = true;
    };
  }, [path, trialId]);

  const series = useMemo(() => {
    if (!data || !channel) return null;
    const samples = data.samples;
    const values: number[] = [];
    const times: number[] = [];
    let minV = Infinity;
    let maxV = -Infinity;
    for (const s of samples) {
      const v = s.channels[channel];
      if (!Number.isFinite(v)) continue;
      values.push(v);
      times.push(s.time);
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
    if (values.length === 0) return null;
    return { values, times, minV, maxV };
  }, [data, channel]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;
    const dpr = window.devicePixelRatio || 1;
    const w = parent.clientWidth;
    const h = parent.clientHeight;
    canvas.width = Math.max(1, Math.floor(w * dpr));
    canvas.height = Math.max(1, Math.floor(h * dpr));
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    if (!series) return;

    // Use the full canvas width so this track's playhead aligns horizontally
    // with Audio + MIDI tracks (which also use the full TrackBody width).
    // Reserve a small vertical inset so the line at min/max isn't clipped by
    // the canvas edge or hidden behind the bottom-row label chips.
    const insetTop = 16;
    const insetBottom = 22;
    const plotW = w;
    const plotH = h - insetTop - insetBottom;

    const totalDur = durationSec ?? series.times[series.times.length - 1] ?? 0;
    if (totalDur <= 0 || plotW <= 0 || plotH <= 0) return;

    const range = series.maxV - series.minV || 1;

    // Line first (so labels overlay it)
    ctx.strokeStyle = BRAND.PRIMARY;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    for (let i = 0; i < series.values.length; i++) {
      const x = (series.times[i] / totalDur) * plotW;
      const y = insetTop + plotH - ((series.values[i] - series.minV) / range) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Inline labels with semi-opaque chips so they don't fight the line.
    const unit = channelUnit(channel);
    const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1));
    const chip = (text: string, x: number, y: number, align: CanvasTextAlign) => {
      ctx.font = '10px system-ui, sans-serif';
      const tw = ctx.measureText(text).width;
      const padX = 4, padY = 2, h2 = 14;
      const left = align === 'right' ? x - tw - padX * 2 : x;
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.fillRect(left, y - h2 + padY, tw + padX * 2, h2);
      ctx.fillStyle = SURFACE.textMuted;
      ctx.textAlign = align;
      ctx.textBaseline = 'middle';
      ctx.fillText(text, x, y - h2 / 2 + padY);
    };
    chip(`${fmt(series.maxV)}${unit}`, 4, insetTop, 'left');
    chip(`${fmt(series.minV)}${unit}`, 4, insetTop + plotH + 12, 'left');
    chip(`${totalDur.toFixed(1)}s`, plotW - 4, insetTop + plotH + 12, 'right');

    // Playhead — same coordinate system as Audio/MIDI tracks.
    if (currentTimeSec >= 0 && totalDur > 0) {
      const px = Math.min(plotW, Math.max(0, (currentTimeSec / totalDur) * plotW));
      ctx.strokeStyle = BRAND.TERTIARY;
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(px, insetTop);
      ctx.lineTo(px, insetTop + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [series, currentTimeSec, durationSec]);

  useEffect(() => {
    const onResize = () => forceTick((t) => t + 1);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  return (
    <Root>
      <Controls>
        <span>Channel</span>
        <select value={channel} onChange={(e) => setChannel(e.target.value)}>
          {data?.channelNames.map((c) => (
            <option key={c} value={c}>
              {channelLabel(c)}{channelUnit(c) ? ` (${channelUnit(c)})` : ''}
            </option>
          ))}
          {(!data || data.channelNames.length === 0) && <option value="">—</option>}
        </select>
      </Controls>
      <CanvasWrap>
        <Canvas ref={canvasRef} />
        {status === 'loading' && <Overlay>Loading kinematics…</Overlay>}
        {status === 'empty' && <Overlay>Kinematics CSV not yet available</Overlay>}
      </CanvasWrap>
    </Root>
  );
}
