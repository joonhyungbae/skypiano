import { useEffect, useRef, useState } from 'react';
import { styled } from '@mui/material/styles';
import { OpenSheetMusicDisplay } from 'opensheetmusicdisplay';
import { BRAND, SURFACE } from '@/theme';

const Root = styled('div')({
  position: 'relative',
  width: '100%',
  height: '100%',
  background: SURFACE.paper,
  overflowX: 'hidden',
  overflowY: 'auto',
});

const OsmdHost = styled('div')({
  width: '100%',
  padding: 12,
  boxSizing: 'border-box',
  // No SVG scaling override: OSMD's autoResize already renders at container
  // width. Forcing { width: 100% } on the SVG was scaling its inner coords
  // away from the cursor element's absolute pixel position, causing the
  // cursor to drift downward as it crossed system breaks.
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
  textAlign: 'center',
  padding: 16,
  pointerEvents: 'none',
});

interface AlignmentMatch {
  score_id: string;
  score_pitch: number;
  score_onset_quarter: number; // partitura quarter-note offset
  midi_pitch: number;
  midi_time_sec: number;
  midi_duration_sec: number;
}

interface AlignmentFile {
  version: number;
  matches: AlignmentMatch[];
}

/** Sorted (midi_time_sec → score beat in whole-notes) timeline that drives
    the OSMD cursor. Polyphonic chords share one beat; deduplicated below. */
interface BeatStep {
  midiTime: number;
  beat: number; // whole-notes (OSMD's CurrentSourceTimestamp unit)
}

export interface ScoreViewerProps {
  /** URL to MusicXML or .mxl file. Empty → empty state. */
  src: string;
  /** URL to Stage 12 alignment.json. Optional — when present, OSMD cursor
      tracks playback. */
  alignmentSrc?: string;
  /** Master playback time, in seconds. Drives cursor when alignment loaded. */
  currentTimeSec?: number;
  /** Whether playback is active. Cursor only shown while a piece is being
      played; otherwise the score reads as a static page. */
  isPlaying?: boolean;
}

export default function ScoreViewer({
  src,
  alignmentSrc,
  currentTimeSec = 0,
  isPlaying = false,
}: ScoreViewerProps) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const hostRef = useRef<HTMLDivElement | null>(null);
  const osmdRef = useRef<OpenSheetMusicDisplay | null>(null);
  const timelineRef = useRef<BeatStep[] | null>(null);
  const stepIdxRef = useRef<number>(-1);
  const [status, setStatus] = useState<'idle' | 'loading' | 'empty' | 'ready' | 'error'>('idle');
  const [errorMsg, setErrorMsg] = useState<string>('');

  // ── Load + render score ─────────────────────────────────────────────────
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    if (!src) {
      setStatus('empty');
      osmdRef.current?.clear();
      return;
    }

    let cancelled = false;
    setStatus('loading');
    setErrorMsg('');

    if (osmdRef.current) {
      try { osmdRef.current.clear(); } catch { /* ignore */ }
      osmdRef.current = null;
    }
    host.innerHTML = '';

    const osmd = new OpenSheetMusicDisplay(host, {
      autoResize: true,
      backend: 'svg',
      drawTitle: false,
      drawSubtitle: false,
      drawComposer: false,
      drawPartNames: false,
      drawCredits: false,
      // Auto-scroll the parent (Root has overflow:auto) to keep the cursor in
      // view during playback.
      followCursor: true,
      // tertiary purple highlight color per design system.
      cursorsOptions: [{ type: 0, color: BRAND.TERTIARY, alpha: 0.2, follow: true }],
    });
    osmdRef.current = osmd;
    // Respect the source MusicXML's <print new-system/new-page> breaks so the
    // on-screen layout matches the PDF the XML was extracted from (typically
    // 4–6 measures per system). Any forced measures-per-line value would
    // override these hints, so we leave RenderXMeasuresPerLineAkaSystem at its
    // default (0 = auto).
    osmd.EngravingRules.NewSystemAtXMLNewSystemAttribute = true;
    osmd.EngravingRules.NewPageAtXMLNewPageAttribute = true;
    osmd.FollowCursor = true;

    osmd
      .load(src)
      .then(() => {
        if (cancelled) return;
        osmd.render();
        // Zoom-to-fit: if the container is narrower than the XML-declared
        // systems need, OSMD silently splits them (e.g. a 5-measure PDF line
        // becomes 3+2 on screen). Compare expected system count from
        // <print new-system="yes"> markers vs what actually rendered, and
        // shrink zoom until they match.
        const expected = 1 + osmd.Sheet.SourceMeasures.reduce(
          (n, m) => n + (m.printNewSystemXml ? 1 : 0),
          0,
        );
        const countRendered = () =>
          osmd.GraphicSheet.MusicPages.reduce((n, p) => n + p.MusicSystems.length, 0);
        let actual = countRendered();
        const MIN_ZOOM = 0.5;
        for (let i = 0; i < 8 && actual > expected && osmd.Zoom > MIN_ZOOM; i += 1) {
          const ratio = Math.max(expected / actual, 0.85);
          osmd.Zoom = Math.max(MIN_ZOOM, osmd.Zoom * ratio * 0.98);
          osmd.render();
          actual = countRendered();
        }
        setStatus('ready');
        // Force-hide cursor by default; effects below will show it once
        // alignment is loaded and playback is active.
        osmd.cursor?.hide();
      })
      .catch((err) => {
        if (cancelled) return;
        const msg = err instanceof Error ? err.message : String(err);
        // eslint-disable-next-line no-console
        console.error('[ScoreViewer]', src, err);
        setErrorMsg(msg);
        setStatus('error');
      });

    return () => {
      cancelled = true;
    };
  }, [src]);

  // ── Load alignment.json ────────────────────────────────────────────────
  useEffect(() => {
    timelineRef.current = null;
    stepIdxRef.current = -1;
    if (!alignmentSrc) return;
    let cancelled = false;
    fetch(alignmentSrc)
      .then((r) => (r.ok ? r.json() : null))
      .then((data: AlignmentFile | null) => {
        if (cancelled || !data?.matches?.length) return;
        // partitura onset_quarter is in quarter notes; OSMD timestamps in
        // whole notes — divide by 4. Dedup polyphonic events at the same
        // midi_time_sec by keeping the earliest beat.
        const beatMap = new Map<number, number>();
        for (const m of data.matches) {
          const t = m.midi_time_sec;
          const b = m.score_onset_quarter / 4;
          const prev = beatMap.get(t);
          if (prev === undefined || b < prev) beatMap.set(t, b);
        }
        const timeline = Array.from(beatMap.entries())
          .map(([midiTime, beat]) => ({ midiTime, beat }))
          .sort((a, b) => a.midiTime - b.midiTime);
        timelineRef.current = timeline;
        stepIdxRef.current = -1;
      })
      .catch(() => { /* alignment optional */ });
    return () => { cancelled = true; };
  }, [alignmentSrc]);

  // ── Drive cursor from currentTimeSec ───────────────────────────────────
  useEffect(() => {
    const osmd = osmdRef.current;
    const timeline = timelineRef.current;
    if (!osmd || status !== 'ready' || !timeline?.length) return;
    // Stop (currentTimeSec == 0 and not playing): rewind cursor + scroll to the
    // top of the score. Pause (currentTimeSec > 0 and not playing) keeps the
    // cursor visible at its current position so the user can see where they
    // left off.
    if (!isPlaying && currentTimeSec <= 0) {
      osmd.cursor?.hide();
      try { osmd.cursor?.reset(); } catch { /* ignore */ }
      stepIdxRef.current = -1;
      rootRef.current?.scrollTo({ top: 0, left: 0, behavior: 'auto' });
      return;
    }

    // Find the latest timeline entry whose midiTime ≤ currentTimeSec.
    // Linear scan from current position (cheap because we're monotonically
    // advancing during playback; binary search only on big jumps).
    let target = stepIdxRef.current;
    if (target < 0 || timeline[target].midiTime > currentTimeSec) {
      // backwards seek or initial — binary search
      let lo = 0;
      let hi = timeline.length - 1;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (timeline[mid].midiTime <= currentTimeSec) lo = mid + 1;
        else hi = mid - 1;
      }
      target = hi; // largest idx with midiTime ≤ currentTimeSec, or -1
    } else {
      while (target + 1 < timeline.length && timeline[target + 1].midiTime <= currentTimeSec) {
        target += 1;
      }
    }
    if (target < 0) {
      osmd.cursor?.hide();
      return;
    }
    if (target === stepIdxRef.current) return;
    stepIdxRef.current = target;

    osmd.cursor.show();
    const targetBeat = timeline[target].beat;
    // Walk OSMD's iterator to the target whole-note beat.
    let it = osmd.cursor.iterator;
    const cur = () => it?.CurrentSourceTimestamp?.RealValue ?? 0;
    const guard = timeline.length + 16;
    let steps = 0;
    if (cur() > targetBeat) {
      osmd.cursor.reset();
      it = osmd.cursor.iterator;
    }
    while (cur() < targetBeat && !it.EndReached && steps < guard) {
      osmd.cursor.next();
      it = osmd.cursor.iterator;
      steps += 1;
    }
    osmd.cursor.update?.();
  }, [currentTimeSec, isPlaying, status]);

  useEffect(() => {
    return () => {
      osmdRef.current?.clear();
      osmdRef.current = null;
    };
  }, []);

  return (
    <Root ref={rootRef}>
      <OsmdHost ref={hostRef} />
      {status === 'loading' && <Overlay>Rendering score…</Overlay>}
      {status === 'empty' && <Overlay>Score not available for this trial</Overlay>}
      {status === 'error' && (
        <Overlay>
          Score render failed
          {errorMsg ? <div style={{ marginTop: 6, fontSize: 11, opacity: 0.8 }}>{errorMsg}</div> : null}
        </Overlay>
      )}
    </Root>
  );
}
