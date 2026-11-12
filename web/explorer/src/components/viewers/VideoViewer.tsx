import { forwardRef, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { styled } from '@mui/material/styles';

export type VideoView = 'top' | 'right' | 'low_right' | 'left';

export const VIDEO_VIEW_LABELS: Record<VideoView, string> = {
  top: 'Top',
  right: 'Right',
  low_right: 'Low-right',
  left: 'Left',
};

const ALL_VIEWS: readonly VideoView[] = ['top', 'right', 'low_right', 'left'];

const Root = styled('div')({
  display: 'flex',
  flexDirection: 'column',
  width: '100%',
  height: '100%',
  background: '#0A0A10',
  position: 'relative',
});

const VideoEl = styled('video')({
  position: 'absolute',
  inset: 0,
  width: '100%',
  height: '100%',
  objectFit: 'contain',
  background: '#0A0A10',
  display: 'block',
});

const EmptyOverlay = styled('div')({
  position: 'absolute',
  inset: 0,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: 'rgba(255, 255, 255, 0.6)',
  fontSize: 13,
  fontStyle: 'italic',
  letterSpacing: '0.02em',
  pointerEvents: 'none',
});

export interface VideoViewerProps {
  /** All available view sources. Missing keys simply aren't rendered. */
  sources: Partial<Record<VideoView, string>>;
  /** Currently visible view. Must be a key present in `sources`. */
  activeView: VideoView;
  /** Playback rate (optional). Default 1. Applied to every mounted video. */
  playbackRate?: number;
  /** If true, hides native browser controls (use when external controls exist). */
  hideNativeControls?: boolean;
  /** Per-view visual rotation in degrees. Top camera is mounted upside-down
      so typical callers pass `{ top: 180 }`. */
  rotateDeg?: Partial<Record<VideoView, number>>;
}

// All views are mounted simultaneously with preload="auto" so the browser
// buffers every angle up-front. Switching views becomes: pause old → seek new
// to old.currentTime → play new if old was playing → toggle visibility.
// This avoids the src-swap stall (metadata fetch + progressive download)
// that caused the original stutter on Top/Right/Low-R toggle.
const VideoViewer = forwardRef<HTMLVideoElement, VideoViewerProps>(function VideoViewer(
  { sources, activeView, playbackRate = 1, hideNativeControls, rotateDeg },
  forwardedRef,
) {
  const videoRefs = useRef<Map<VideoView, HTMLVideoElement>>(new Map());
  const prevActiveRef = useRef<VideoView>(activeView);
  const [failedViews, setFailedViews] = useState<Set<VideoView>>(new Set());

  const assignForwarded = (el: HTMLVideoElement | null) => {
    if (typeof forwardedRef === 'function') forwardedRef(el);
    else if (forwardedRef) forwardedRef.current = el;
  };

  // Transfer playback state from previous active view to new active view.
  // useLayoutEffect so the seek/play happens before paint — the user sees the
  // new angle already showing the correct frame, not a blank or t=0 flash.
  useLayoutEffect(() => {
    const prev = prevActiveRef.current;
    const activeEl = videoRefs.current.get(activeView) ?? null;
    assignForwarded(activeEl);
    if (prev === activeView || !activeEl) {
      prevActiveRef.current = activeView;
      return;
    }
    const oldEl = videoRefs.current.get(prev);
    if (oldEl) {
      const wasPlaying = !oldEl.paused && !oldEl.ended;
      const t = Number.isFinite(oldEl.currentTime) ? oldEl.currentTime : 0;
      try { activeEl.currentTime = t; } catch { /* seek may throw if not ready */ }
      oldEl.pause();
      if (wasPlaying) void activeEl.play().catch(() => {});
    }
    prevActiveRef.current = activeView;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeView]);

  // Keep playbackRate applied across every mounted video so a later switch
  // inherits the right speed.
  useEffect(() => {
    for (const el of videoRefs.current.values()) {
      el.playbackRate = playbackRate;
    }
  }, [playbackRate, sources]);

  // Non-active videos are muted at the element level (audio comes from a
  // separate <audio> element in the parent). The active is also muted because
  // `hideNativeControls` mode treats the video purely as the visual track.
  const setVideoRef = (view: VideoView) => (node: HTMLVideoElement | null) => {
    if (node) {
      videoRefs.current.set(view, node);
      if (view === activeView) assignForwarded(node);
    } else {
      videoRefs.current.delete(view);
    }
  };

  const entries = ALL_VIEWS.flatMap((v) => {
    const src = sources[v];
    return src ? [[v, src] as const] : [];
  });

  return (
    <Root>
      {entries.length === 0 && <EmptyOverlay>No video source</EmptyOverlay>}
      {entries.map(([view, src]) => {
        const isActive = view === activeView;
        const rot = rotateDeg?.[view] ?? 0;
        const transform = rot ? `rotate(${rot}deg)` : undefined;
        return (
          <VideoEl
            key={view}
            ref={setVideoRef(view)}
            src={src}
            controls={isActive && !hideNativeControls}
            onError={() =>
              setFailedViews((s) => {
                if (s.has(view)) return s;
                const n = new Set(s);
                n.add(view);
                return n;
              })
            }
            preload="auto"
            crossOrigin="anonymous"
            muted
            playsInline
            style={{
              transform,
              visibility: isActive ? 'visible' : 'hidden',
              zIndex: isActive ? 1 : 0,
              pointerEvents: isActive ? 'auto' : 'none',
            }}
          />
        );
      })}
      {failedViews.has(activeView) && (
        <EmptyOverlay>Video file not yet available</EmptyOverlay>
      )}
    </Root>
  );
});

export default VideoViewer;
