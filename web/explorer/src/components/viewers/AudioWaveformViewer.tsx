import { useEffect, useRef, useState } from 'react';
import { styled } from '@mui/material/styles';
import { BRAND, SURFACE } from '@/theme';

const Root = styled('div')({
  position: 'relative',
  width: '100%',
  height: '100%',
  background: SURFACE.paper,
  overflow: 'hidden',
});

const Canvas = styled('canvas')({
  display: 'block',
  width: '100%',
  height: '100%',
});

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

export interface AudioWaveformViewerProps {
  /** Audio file URL. Empty → empty state. */
  src: string;
  /** Total audio duration (seconds). Passed when known externally. */
  durationSec?: number;
  /** Current playhead in seconds (for playhead line). */
  currentTimeSec?: number;
  /** Bin count for waveform sampling. Default 1500. */
  bins?: number;
}

interface DecodeResult {
  min: Float32Array;
  max: Float32Array;
  duration: number;
  sampleRate: number;
}

async function decodeAndBin(src: string, bins: number): Promise<DecodeResult> {
  const res = await fetch(src);
  if (!res.ok) throw new Error(`fetch ${src} failed`);
  const buf = await res.arrayBuffer();
  const Ctx: typeof AudioContext =
    (window as any).AudioContext || (window as any).webkitAudioContext;
  const offline = new OfflineAudioContext(1, 1, 44100);
  const audioBuf = await offline.decodeAudioData(buf.slice(0));
  const channel = audioBuf.getChannelData(0);
  const binSize = Math.max(1, Math.floor(channel.length / bins));
  const min = new Float32Array(bins);
  const max = new Float32Array(bins);
  let peak = 0;
  for (let b = 0; b < bins; b++) {
    let lo = 1;
    let hi = -1;
    const start = b * binSize;
    const end = Math.min(channel.length, start + binSize);
    for (let i = start; i < end; i++) {
      const v = channel[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    min[b] = lo;
    max[b] = hi;
    const mag = Math.max(Math.abs(lo), Math.abs(hi));
    if (mag > peak) peak = mag;
  }
  // Peak-normalize so every recording fills the vertical space regardless of
  // recording gain. Floor at a small epsilon so (near-)silent files don't blow up.
  const norm = 1 / Math.max(peak, 1e-4);
  for (let b = 0; b < bins; b++) {
    min[b] *= norm;
    max[b] *= norm;
  }
  // prevent unused-var warning for Ctx
  void Ctx;
  return { min, max, duration: audioBuf.duration, sampleRate: audioBuf.sampleRate };
}

export default function AudioWaveformViewer({
  src,
  durationSec,
  currentTimeSec = 0,
  bins = 1500,
}: AudioWaveformViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [decoded, setDecoded] = useState<DecodeResult | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'empty' | 'ready' | 'error'>('idle');

  useEffect(() => {
    if (!src) {
      setStatus('empty');
      setDecoded(null);
      return;
    }
    let cancelled = false;
    setStatus('loading');
    decodeAndBin(src, bins)
      .then((d) => {
        if (cancelled) return;
        setDecoded(d);
        setStatus('ready');
      })
      .catch(() => {
        if (cancelled) return;
        setDecoded(null);
        setStatus('error');
      });
    return () => {
      cancelled = true;
    };
  }, [src, bins]);

  // Draw waveform + playhead whenever data/time/size change.
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

    if (!decoded) return;

    const mid = h / 2;
    const { min, max } = decoded;
    const n = min.length;
    const xStep = w / n;
    ctx.fillStyle = BRAND.PRIMARY;
    ctx.globalAlpha = 0.85;
    for (let i = 0; i < n; i++) {
      const x = i * xStep;
      const ymin = mid - max[i] * mid * 0.92;
      const ymax = mid - min[i] * mid * 0.92;
      const barH = Math.max(1, ymax - ymin);
      ctx.fillRect(x, ymin, Math.max(1, xStep), barH);
    }
    ctx.globalAlpha = 1;

    // center line
    ctx.strokeStyle = SURFACE.border;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(w, mid);
    ctx.stroke();

    // playhead
    const totalDur = durationSec ?? decoded.duration;
    if (totalDur > 0 && currentTimeSec >= 0) {
      const px = Math.min(w, Math.max(0, (currentTimeSec / totalDur) * w));
      ctx.strokeStyle = BRAND.TERTIARY;
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(px, 0);
      ctx.lineTo(px, h);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [decoded, currentTimeSec, durationSec]);

  // Redraw on resize.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;
    const ro = new ResizeObserver(() => {
      // Trigger the draw effect by nudging state — but we can simply redraw inline.
      const ev = new Event('resize');
      window.dispatchEvent(ev);
    });
    ro.observe(parent);
    return () => ro.disconnect();
  }, []);

  // Force a re-render on window resize so the draw effect runs.
  const [, forceTick] = useState(0);
  useEffect(() => {
    const onResize = () => forceTick((t) => t + 1);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  return (
    <Root>
      <Canvas ref={canvasRef} />
      {status === 'loading' && <Overlay>Decoding audio…</Overlay>}
      {status === 'empty' && <Overlay>No audio source</Overlay>}
      {status === 'error' && <Overlay>Audio file not yet available</Overlay>}
    </Root>
  );
}
