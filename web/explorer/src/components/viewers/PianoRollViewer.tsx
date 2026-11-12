import { useEffect, useRef, useState } from 'react';
import { styled } from '@mui/material/styles';
import { parseMidiFile } from '@/utils/midiParser';
import type { MidiData, MidiNote } from '@/utils/midiParser';
import { BRAND, SURFACE } from '@/theme';

const Root = styled('div')({
  position: 'relative',
  width: '100%',
  height: '100%',
  background: SURFACE.paper,
  overflow: 'hidden',
});

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

export interface PianoRollViewerProps {
  /** URL to .mid. Empty → empty state. */
  src: string;
  /** Current playhead in seconds. */
  currentTimeSec?: number;
  /** Override total duration (defaults to MIDI's own duration). */
  durationSec?: number;
  /** MIDI pitch range. Default 21..108 (A0..C8 = standard piano). */
  minNote?: number;
  maxNote?: number;
}

// Semitone classes that are black keys: C# D# F# G# A# → 1,3,6,8,10
const BLACK_KEYS = new Set([1, 3, 6, 8, 10]);

export default function PianoRollViewer({
  src,
  currentTimeSec = 0,
  durationSec,
  minNote = 21,
  maxNote = 108,
}: PianoRollViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [midi, setMidi] = useState<MidiData | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'empty' | 'ready' | 'error'>('idle');
  const [, forceTick] = useState(0);

  useEffect(() => {
    if (!src) {
      setMidi(null);
      setStatus('empty');
      return;
    }
    let cancelled = false;
    setStatus('loading');
    (async () => {
      try {
        const res = await fetch(src);
        if (!res.ok) throw new Error('fetch failed');
        const buf = await res.arrayBuffer();
        // Vite dev server returns index.html (`<!doctype …`) for any missing
        // static file. Detect that before handing to the MIDI parser so the
        // console isn't spammed with "Expected 'MHdr', got '<!do'" errors.
        const head = new Uint8Array(buf, 0, Math.min(4, buf.byteLength));
        if (head.length < 4 || head[0] !== 0x4d || head[1] !== 0x54 || head[2] !== 0x68 || head[3] !== 0x64) {
          throw new Error('not a MIDI file (missing MThd header)');
        }
        const data = await parseMidiFile(buf);
        if (cancelled) return;
        setMidi(data);
        setStatus('ready');
      } catch {
        if (cancelled) return;
        setMidi(null);
        setStatus('empty');
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [src]);

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

    // Grid + background
    ctx.fillStyle = SURFACE.paper;
    ctx.fillRect(0, 0, w, h);

    // Alternating row shading for black keys
    const noteRange = maxNote - minNote + 1;
    const rowH = h / noteRange;
    for (let n = minNote; n <= maxNote; n++) {
      if (BLACK_KEYS.has(n % 12)) {
        const y = h - (n - minNote + 1) * rowH;
        ctx.fillStyle = `${BRAND.PRIMARY}06`;
        ctx.fillRect(0, y, w, rowH);
      }
    }

    if (!midi) return;

    const totalDur = durationSec ?? midi.totalDuration;
    if (totalDur <= 0) return;

    // Draw notes.
    for (const note of midi.notes as MidiNote[]) {
      if (note.note < minNote || note.note > maxNote) continue;
      const x = (note.startTime / totalDur) * w;
      const widthPx = Math.max(1.5, (note.duration / totalDur) * w);
      const y = h - (note.note - minNote + 1) * rowH;
      const isBlack = BLACK_KEYS.has(note.note % 12);
      const baseColor = isBlack ? BRAND.TERTIARY : BRAND.PRIMARY;
      const alpha = 0.4 + (note.velocity / 127) * 0.55;
      ctx.fillStyle = `${baseColor}${Math.round(alpha * 255).toString(16).padStart(2, '0')}`;
      ctx.fillRect(x, y, widthPx, Math.max(1, rowH - 0.5));
    }

    // Playhead
    if (currentTimeSec >= 0 && totalDur > 0) {
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
  }, [midi, currentTimeSec, durationSec, minNote, maxNote]);

  useEffect(() => {
    const onResize = () => forceTick((t) => t + 1);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  return (
    <Root>
      <Canvas ref={canvasRef} />
      {status === 'loading' && <Overlay>Decoding MIDI…</Overlay>}
      {status === 'empty' && <Overlay>No MIDI source</Overlay>}
      {status === 'error' && <Overlay>MIDI file not yet available</Overlay>}
    </Root>
  );
}
