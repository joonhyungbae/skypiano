import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { styled } from '@mui/material/styles';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import StopIcon from '@mui/icons-material/Stop';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import VolumeOffIcon from '@mui/icons-material/VolumeOff';
import { loadSampleTrials } from '@/data/dataRepository';
import type { FingeringData, MotionData, SampleTrialEntry } from '@/data/types';
import { loadMotion } from '@/data/parsers/handBodyMotionParser';
import { loadFingering } from '@/data/parsers/fingeringParser';
import { buildSamplePaths } from '@/data/adapters/samples';
import { formatTime } from '@/utils/midiParser';
import TopBar from '@/components/common/TopBar';
import MotionViewer, { type CameraPreset } from '@/components/viewers/MotionViewer';
import VideoViewer, { type VideoView } from '@/components/viewers/VideoViewer';
import AudioWaveformViewer from '@/components/viewers/AudioWaveformViewer';
import PianoRollViewer from '@/components/viewers/PianoRollViewer';
import KinematicsViewer from '@/components/viewers/KinematicsViewer';
import ScoreViewer from '@/components/viewers/ScoreViewer';
import { BRAND, SURFACE } from '@/theme';

const Page = styled('div')({
  flex: '1 1 0',
  minHeight: 0,
  display: 'flex',
  flexDirection: 'column',
  padding: '16px 20px 20px',
  boxSizing: 'border-box',
});

const Stage = styled('div')({
  display: 'flex',
  gap: 14,
  flex: 1,
  minHeight: 0,
});

const ScoreColumn = styled('div')({
  flexShrink: 0,
  height: '100%',
  aspectRatio: '5 / 6',
  minHeight: 0,
  background: SURFACE.paper,
  border: `1px solid ${SURFACE.border}`,
  borderRadius: 12,
  boxShadow: SURFACE.shadow,
  overflow: 'hidden',
});

const MainColumn = styled('div')({
  flex: 1,
  minWidth: 0,
  minHeight: 0,
  display: 'flex',
  flexDirection: 'column',
  gap: 14,
});

const TitleContent = styled('div')({
  display: 'flex',
  alignItems: 'center',
  gap: 10,
  flex: 1,
  minWidth: 0,
  '& .chip': {
    fontSize: 10,
    fontWeight: 700,
    padding: '3px 7px',
    borderRadius: 4,
    textTransform: 'uppercase',
    border: '1px solid',
    letterSpacing: '0.03em',
    flexShrink: 0,
  },
  '& .piece': {
    fontSize: 15,
    fontWeight: 700,
    color: SURFACE.textPrimary,
    margin: 0,
    letterSpacing: '-0.01em',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    minWidth: 0,
    flexShrink: 1,
  },
  '& .pair-link': {
    marginLeft: 'auto',
    display: 'inline-flex',
    alignItems: 'center',
    gap: 4,
    fontSize: 11,
    fontWeight: 600,
    color: BRAND.PRIMARY,
    textDecoration: 'none',
    padding: '4px 10px',
    borderRadius: 999,
    background: `${BRAND.PRIMARY}0D`,
    whiteSpace: 'nowrap',
    flexShrink: 0,
    '&:hover': { background: `${BRAND.PRIMARY}1A` },
    '& svg': { fontSize: 14 },
  },
});

const ViewerGrid = styled('div')({
  display: 'grid',
  gridTemplateColumns: '1fr 1fr',
  gap: 14,
  minHeight: 0,
  '@media (max-width: 1100px)': { gridTemplateColumns: '1fr' },
});

// ── Multi-track container (Audio / MIDI / Kinematics share a time axis) ─────

const TrackGroup = styled('div')({
  display: 'flex',
  flexDirection: 'column',
  flex: '1 1 0',
  minHeight: 0,
  borderRadius: 12,
  background: SURFACE.paper,
  border: `1px solid ${SURFACE.border}`,
  boxShadow: SURFACE.shadow,
  overflow: 'hidden',
});

const Track = styled('div')({
  display: 'flex',
  alignItems: 'stretch',
  minHeight: 0,
  '& + &': {
    borderTop: `1px solid ${SURFACE.border}`,
  },
});

const TrackLabel = styled('div')({
  width: 110,
  flexShrink: 0,
  padding: '10px 14px',
  display: 'flex',
  alignItems: 'center',
  borderRight: `1px solid ${SURFACE.border}`,
  background: `${BRAND.PRIMARY}05`,
  '& .title': {
    fontSize: 11,
    fontWeight: 700,
    color: SURFACE.textPrimary,
    letterSpacing: '0.06em',
    textTransform: 'uppercase',
    lineHeight: 1.2,
  },
});

const TrackBody = styled('div')({
  flex: 1,
  minWidth: 0,
  position: 'relative',
});

const ViewerPanel = styled('div')({
  display: 'flex',
  flexDirection: 'column',
  borderRadius: 12,
  background: SURFACE.paper,
  border: `1px solid ${SURFACE.border}`,
  boxShadow: SURFACE.shadow,
  overflow: 'hidden',
  '& .panel-header': {
    padding: '8px 12px',
    borderBottom: `1px solid ${SURFACE.border}`,
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    flexWrap: 'nowrap',
    minHeight: 36,
    overflow: 'hidden',
  },
  '& .panel-header .label': {
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: '0.06em',
    textTransform: 'uppercase',
    color: BRAND.PRIMARY,
  },
  '& .panel-header .controls': {
    marginLeft: 'auto',
    display: 'inline-flex',
    alignItems: 'center',
    gap: 8,
    flexWrap: 'nowrap',
    overflowX: 'auto',
  },
  '& .panel-header .ctrl-group': {
    display: 'inline-flex',
    gap: 4,
    flexWrap: 'nowrap',
  },
  '& .panel-header .ctrl-divider': {
    width: 1,
    alignSelf: 'stretch',
    background: SURFACE.border,
    margin: '2px 0',
    flexShrink: 0,
  },
  '& .panel-body': {
    width: '100%',
    aspectRatio: '16 / 9',
    position: 'relative',
  },
  '& .placeholder-body': {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 13,
    color: SURFACE.textMuted,
    fontStyle: 'italic',
    padding: 40,
  },
});

const CameraBtn = styled('button')<{ $active?: boolean }>(({ $active }) => ({
  fontSize: 10,
  fontWeight: 700,
  padding: '4px 8px',
  borderRadius: 4,
  letterSpacing: '0.04em',
  textTransform: 'uppercase',
  border: `1px solid ${$active ? BRAND.PRIMARY : SURFACE.border}`,
  background: $active ? `${BRAND.PRIMARY}15` : 'transparent',
  color: $active ? BRAND.PRIMARY : SURFACE.textMuted,
  cursor: 'pointer',
  transition: 'all 0.15s ease',
  '&:hover': { borderColor: BRAND.PRIMARY, color: BRAND.PRIMARY },
}));

const LayerToggle = styled('button')<{ $active?: boolean }>(({ $active }) => ({
  fontSize: 10,
  fontWeight: 700,
  padding: '4px 8px',
  borderRadius: 4,
  letterSpacing: '0.04em',
  textTransform: 'uppercase',
  border: '1px solid',
  cursor: 'pointer',
  transition: 'all 0.15s ease',
  background: $active ? `${BRAND.PRIMARY}15` : 'transparent',
  color: $active ? BRAND.PRIMARY : SURFACE.textMuted,
  borderColor: $active ? BRAND.PRIMARY : SURFACE.border,
}));

const TASK_LABEL: Record<SampleTrialEntry['task'], string> = {
  T0: 'T0 · Technique 1',
  T1: 'T1 · Technique 2',
  T2: 'T2 · Graded',
  T9: 'T9 · Free',
};

const TASK_COLORS: Record<SampleTrialEntry['task'], string> = {
  T0: BRAND.SECONDARY,
  T1: BRAND.PRIMARY,
  T2: BRAND.TERTIARY,
  T9: BRAND.PRIMARY,
};

const CAMERAS: CameraPreset[] = ['3d', 'top', 'front', 'side', 'fingering'];

const InfoText = styled('div')({
  color: SURFACE.textMuted,
  padding: 32,
  fontSize: 14,
});

const EmptyOverlay = styled('div')({
  position: 'absolute',
  top: 12,
  left: 12,
  padding: '6px 12px',
  fontSize: 11,
  color: SURFACE.textMuted,
  background: 'rgba(255,255,255,0.85)',
  border: `1px solid ${SURFACE.border}`,
  borderRadius: 4,
  fontStyle: 'italic',
});

const ComingSoon = styled('div')({
  padding: 16,
  borderRadius: 10,
  background: SURFACE.paper,
  border: `1px solid ${SURFACE.border}`,
  boxShadow: SURFACE.shadow,
  fontSize: 12,
  color: SURFACE.textMuted,
  fontStyle: 'italic',
});

// ── Playback bar ────────────────────────────────────────────────────────────

const PlaybackBar = styled('div')({
  display: 'flex',
  alignItems: 'center',
  gap: 12,
  padding: '10px 20px',
  background: SURFACE.paper,
  borderTop: `1px solid ${SURFACE.border}`,
  boxShadow: '0 -6px 18px rgba(15, 15, 112, 0.05)',
  flexShrink: 0,
  zIndex: 10,
});

const PlayBtn = styled('button')({
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  width: 36,
  height: 36,
  borderRadius: '50%',
  border: 'none',
  background: BRAND.PRIMARY,
  color: '#FFFFFF',
  cursor: 'pointer',
  transition: 'all 0.15s ease',
  boxShadow: `0 3px 10px ${BRAND.PRIMARY}33`,
  '&:hover': { background: BRAND.SECONDARY, transform: 'scale(1.04)' },
  '& svg': { fontSize: 22 },
  flexShrink: 0,
});

const IconBtn = styled('button')({
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  width: 30,
  height: 30,
  borderRadius: 6,
  border: 'none',
  background: 'transparent',
  color: SURFACE.textSecondary,
  cursor: 'pointer',
  transition: 'all 0.15s ease',
  '&:hover': { color: BRAND.PRIMARY, background: `${BRAND.PRIMARY}0D` },
  '& svg': { fontSize: 20 },
  flexShrink: 0,
});

const TimeLabel = styled('span')({
  fontFamily: 'monospace',
  fontSize: 12,
  fontWeight: 600,
  color: SURFACE.textSecondary,
  letterSpacing: '0.03em',
  minWidth: 42,
  textAlign: 'center',
  flexShrink: 0,
});

const ScrubInput = styled('input')({
  flex: 1,
  minWidth: 120,
  height: 4,
  appearance: 'none',
  WebkitAppearance: 'none',
  background: SURFACE.border,
  borderRadius: 999,
  outline: 'none',
  cursor: 'pointer',
  '&::-webkit-slider-thumb': {
    appearance: 'none',
    WebkitAppearance: 'none',
    width: 14,
    height: 14,
    borderRadius: '50%',
    background: BRAND.PRIMARY,
    border: `2px solid ${SURFACE.paper}`,
    boxShadow: `0 2px 6px ${BRAND.PRIMARY}55`,
    cursor: 'pointer',
  },
  '&::-moz-range-thumb': {
    width: 14,
    height: 14,
    borderRadius: '50%',
    background: BRAND.PRIMARY,
    border: `2px solid ${SURFACE.paper}`,
    boxShadow: `0 2px 6px ${BRAND.PRIMARY}55`,
    cursor: 'pointer',
  },
  '&:disabled': { opacity: 0.4, cursor: 'not-allowed' },
});

const RateSelect = styled('select')({
  padding: '4px 22px 4px 10px',
  border: `1px solid ${SURFACE.border}`,
  borderRadius: 6,
  background: SURFACE.paper,
  backgroundImage:
    "url(\"data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%235A5A6E' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M6 8l4 4 4-4'/%3e%3c/svg%3e\")",
  backgroundPosition: 'right 4px center',
  backgroundRepeat: 'no-repeat',
  backgroundSize: 12,
  fontSize: 12,
  fontWeight: 600,
  color: SURFACE.textPrimary,
  appearance: 'none',
  cursor: 'pointer',
  flexShrink: 0,
  '&:focus': { outline: 'none', borderColor: BRAND.PRIMARY, boxShadow: `0 0 0 3px ${BRAND.PRIMARY}22` },
});

const VolumeBlock = styled('div')({
  display: 'flex',
  alignItems: 'center',
  gap: 6,
  flexShrink: 0,
});

const VolumeInput = styled('input')({
  width: 80,
  height: 4,
  appearance: 'none',
  WebkitAppearance: 'none',
  background: SURFACE.border,
  borderRadius: 999,
  outline: 'none',
  cursor: 'pointer',
  '&::-webkit-slider-thumb': {
    appearance: 'none',
    WebkitAppearance: 'none',
    width: 12,
    height: 12,
    borderRadius: '50%',
    background: BRAND.PRIMARY,
    border: `2px solid ${SURFACE.paper}`,
    cursor: 'pointer',
  },
  '&::-moz-range-thumb': {
    width: 12,
    height: 12,
    borderRadius: '50%',
    background: BRAND.PRIMARY,
    border: `2px solid ${SURFACE.paper}`,
    cursor: 'pointer',
  },
});

export default function SampleView() {
  const { trialId } = useParams<{ trialId: string }>();
  const [samples, setSamples] = useState<SampleTrialEntry[]>([]);
  const [motion, setMotion] = useState<MotionData | null>(null);
  const [fingering, setFingering] = useState<FingeringData | null>(null);
  const [loading, setLoading] = useState(true);
  const [motionStatus, setMotionStatus] = useState<'idle' | 'loading' | 'empty' | 'loaded'>('idle');
  const [cameraPreset, setCameraPreset] = useState<CameraPreset>('3d');
  const [showHand, setShowHand] = useState(true);
  const [showBody, setShowBody] = useState(true);
  const [stream, setStream] = useState<'raw_flagged' | 'imputed'>('imputed');
  const [videoView, setVideoView] = useState<VideoView>('top');

  // ── Shared playback state (video is the master clock) ────────────────────
  // VideoViewer mounts every available angle up-front (preload="auto") and
  // only the active one is visible/playing. View switches are handled inside
  // VideoViewer: it transfers currentTime + playing-state from old → new,
  // so no parent-side resume bookkeeping is needed. Audio comes from a
  // separate <audio> element so blurred videos (ffmpeg -an) still have sound.
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [currentTimeSec, setCurrentTimeSec] = useState(0);
  const [durationSec, setDurationSec] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [volume, setVolume] = useState(1);
  const [muted, setMuted] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const list = await loadSampleTrials();
      if (!cancelled) {
        setSamples(list);
        setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const sample = useMemo(
    () => samples.find((s) => s.trialId === trialId) ?? null,
    [samples, trialId],
  );
  const byId = useMemo(() => {
    const m = new Map<string, SampleTrialEntry>();
    for (const s of samples) m.set(s.trialId, s);
    return m;
  }, [samples]);
  const partner = sample?.comparePartnerId ? byId.get(sample.comparePartnerId) ?? null : null;

  const paths = useMemo(() => (sample ? buildSamplePaths(sample) : null), [sample]);

  // Clamp video view to what this trial offers.
  useEffect(() => {
    if (!sample) return;
    if (!sample.videoViews.includes(videoView)) {
      setVideoView(sample.videoViews[0] ?? 'top');
    }
  }, [sample, videoView]);

  // Reset playback state on trial change.
  useEffect(() => {
    setCurrentTimeSec(0);
    setDurationSec(0);
    setIsPlaying(false);
  }, [trialId]);

  // Keep tab title in sync with the loaded piece so the tab stays informative.
  useEffect(() => {
    if (sample) {
      document.title = `SKY Piano · ${sample.pieceLabel}`;
    }
  }, [sample]);

  // Attach native listeners so <video> events drive React state. Browsers
  // throttle `timeupdate` to ~250ms which makes playheads (audio/midi/kinematics)
  // and motion frame index visibly jump. Sample currentTime via requestAnimationFrame
  // (~60Hz) while playing for smooth scrubbing across all viewers.
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    // During a view switch VideoViewer pauses the outgoing <video>; that pause
    // event can fire on the stale element after videoRef has already moved on.
    // Guard every handler on videoRef.current === v so we only react to events
    // from what is *currently* the active element.
    const isActive = () => videoRef.current === v;
    const onDur = () => {
      if (!isActive()) return;
      setDurationSec(Number.isFinite(v.duration) ? v.duration : 0);
    };
    const onPlay = () => { if (isActive()) setIsPlaying(true); };
    const onPause = () => {
      if (!isActive()) return;
      setIsPlaying(false);
      setCurrentTimeSec(v.currentTime);
    };
    const onEnded = () => { if (isActive()) setIsPlaying(false); };
    const onSeeked = () => { if (isActive()) setCurrentTimeSec(v.currentTime); };
    v.addEventListener('durationchange', onDur);
    v.addEventListener('loadedmetadata', onDur);
    v.addEventListener('play', onPlay);
    v.addEventListener('pause', onPause);
    v.addEventListener('ended', onEnded);
    v.addEventListener('seeked', onSeeked);
    return () => {
      v.removeEventListener('durationchange', onDur);
      v.removeEventListener('loadedmetadata', onDur);
      v.removeEventListener('play', onPlay);
      v.removeEventListener('pause', onPause);
      v.removeEventListener('ended', onEnded);
      v.removeEventListener('seeked', onSeeked);
    };
  }, [sample, videoView]);

  // High-frequency playhead pump — only runs while video is playing.
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !isPlaying) return;
    let raf = 0;
    const tick = () => {
      setCurrentTimeSec(v.currentTime);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [isPlaying, sample, videoView]);

  const togglePlay = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) void v.play();
    else v.pause();
  };

  const stopPlayback = () => {
    const v = videoRef.current;
    const a = audioRef.current;
    if (v) { v.pause(); v.currentTime = 0; }
    if (a) { a.pause(); a.currentTime = 0; }
    setCurrentTimeSec(0);
    setIsPlaying(false);
  };

  const seek = (t: number) => {
    const v = videoRef.current;
    if (!v || !Number.isFinite(t)) return;
    v.currentTime = Math.max(0, Math.min(durationSec || t, t));
  };

  // Mirror video → audio (play/pause/seek/rate). Audio.wav is the sound source;
  // the <video> element is muted (VideoViewer mounts every view with muted).
  // Same staleness guard as above: ignore events from the now-inactive element.
  useEffect(() => {
    const v = videoRef.current;
    const a = audioRef.current;
    if (!v || !a) return;
    const isActive = () => videoRef.current === v;
    const sync = () => {
      if (Math.abs(a.currentTime - v.currentTime) > 0.15) a.currentTime = v.currentTime;
    };
    const onPlay = () => { if (!isActive()) return; sync(); void a.play().catch(() => {}); };
    const onPause = () => { if (!isActive()) return; a.pause(); sync(); };
    const onSeek = () => { if (isActive()) sync(); };
    const onRate = () => { if (isActive()) a.playbackRate = v.playbackRate; };
    v.addEventListener('play', onPlay);
    v.addEventListener('pause', onPause);
    v.addEventListener('seeked', onSeek);
    v.addEventListener('ratechange', onRate);
    return () => {
      v.removeEventListener('play', onPlay);
      v.removeEventListener('pause', onPause);
      v.removeEventListener('seeked', onSeek);
      v.removeEventListener('ratechange', onRate);
    };
  }, [sample, videoView]);

  // Apply volume / mute to the audio element (video stays muted).
  useEffect(() => {
    const a = audioRef.current;
    if (!a) return;
    a.volume = Math.max(0, Math.min(1, volume));
    a.muted = muted;
  }, [volume, muted]);

  // Load fingering when sample changes.
  useEffect(() => {
    if (!sample || !sample.modalities.fingering) {
      setFingering(null);
      return;
    }
    let cancelled = false;
    const p = buildSamplePaths(sample);
    loadFingering({
      trialId: sample.trialId,
      eventsPath: p.fingeringEvents,
    }).then((d) => {
      if (cancelled) return;
      setFingering(d.events.length === 0 ? null : d);
    });
    return () => {
      cancelled = true;
    };
  }, [sample]);

  useEffect(() => {
    if (!sample) return;
    setMotionStatus('loading');
    let cancelled = false;
    const p = buildSamplePaths(sample);
    const handPath = stream === 'imputed' ? p.handImputed : p.handRawFlagged;
    // Body is rendered as Visual3D-style 13 segments (a/b endpoint pairs)
    // derived from the body marker CSV by pipeline/10_body_visual3D. The
    // marker-based body skeleton is not surfaced in the web viewer.
    loadMotion({
      trialId: sample.trialId,
      stream,
      handCsv: sample.modalities.motionHand ? handPath : undefined,
      bodyCsv: sample.modalities.motionBody ? p.bodySegments : undefined,
    }).then((data) => {
      if (cancelled) return;
      if (data.frameCount === 0) {
        setMotion(null);
        setMotionStatus('empty');
      } else {
        setMotion(data);
        setMotionStatus('loaded');
      }
    });
    return () => {
      cancelled = true;
    };
  }, [sample, stream]);

  if (loading) {
    return (
      <>
        <TopBar />
        <Page><InfoText>Loading…</InfoText></Page>
      </>
    );
  }

  if (!sample) {
    return (
      <>
        <TopBar />
        <Page>
          <InfoText>
            Sample <code>{trialId}</code> was not found in the manifest. Use the menu to pick another.
          </InfoText>
        </Page>
      </>
    );
  }

  const taskColor = TASK_COLORS[sample.task];

  return (
    <>
      <TopBar>
        <TitleContent>
          <span
            className="chip"
            style={{ background: `${taskColor}10`, color: taskColor, borderColor: `${taskColor}33` }}
          >
            {TASK_LABEL[sample.task]}
          </span>
          <span
            className="chip"
            style={{
              background: sample.level === 'Professional' ? `${BRAND.PRIMARY}10` : `${BRAND.SECONDARY}10`,
              color: sample.level === 'Professional' ? BRAND.PRIMARY : BRAND.SECONDARY,
              borderColor:
                sample.level === 'Professional' ? `${BRAND.PRIMARY}33` : `${BRAND.SECONDARY}33`,
            }}
          >
            {sample.level}
          </span>
          <h1 className="piece">{sample.pieceLabel}</h1>
          {partner && (
            <Link className="pair-link" to={`/sample/${partner.trialId}`}>
              <CompareArrowsIcon />
              {partner.level === 'Professional' ? `Pro · ${partner.subject}` : `Amateur · ${partner.subject}`}
            </Link>
          )}
        </TitleContent>
      </TopBar>
      <Page>
      <Stage>
        <ScoreColumn>
          <ScoreViewer
            src={sample.modalities.score && paths ? paths.score : ''}
            alignmentSrc={sample.modalities.score && paths ? paths.alignment : undefined}
            currentTimeSec={currentTimeSec}
            isPlaying={isPlaying}
          />
        </ScoreColumn>
        <MainColumn>
      <ViewerGrid>
        <ViewerPanel>
          <div className="panel-header">
            <span className="label">Motion</span>
            <div className="controls">
              <div className="ctrl-group">
                <LayerToggle type="button" $active={showHand} onClick={() => setShowHand((v) => !v)}>Hand</LayerToggle>
                <LayerToggle type="button" $active={showBody} onClick={() => setShowBody((v) => !v)}>Body</LayerToggle>
                <LayerToggle
                  type="button"
                  $active={stream === 'imputed'}
                  onClick={() => setStream((s) => (s === 'imputed' ? 'raw_flagged' : 'imputed'))}
                >
                  {stream === 'imputed' ? 'Imputed' : 'Raw'}
                </LayerToggle>
              </div>
              <div className="ctrl-divider" aria-hidden="true" />
              <div className="ctrl-group">
                {CAMERAS.map((c) => (
                  <CameraBtn key={c} type="button" $active={cameraPreset === c} onClick={() => setCameraPreset(c)}>
                    {c}
                  </CameraBtn>
                ))}
              </div>
            </div>
          </div>
          <div className="panel-body">
            {motionStatus === 'loading' && (
              <div className="placeholder-body">Loading motion…</div>
            )}
            {(motionStatus === 'loaded' || motionStatus === 'empty') && (
              <MotionViewer
                motion={motion}
                currentFrame={motion ? Math.floor(currentTimeSec * (motion.fps || 30)) : 0}
                currentTimeSec={currentTimeSec}
                cameraPreset={cameraPreset}
                showHand={showHand}
                showBody={showBody}
                fingering={fingering}
              />
            )}
            {motionStatus === 'empty' && (
              <EmptyOverlay>Motion CSV not yet available — grid + gizmo only.</EmptyOverlay>
            )}
          </div>
        </ViewerPanel>

        <ViewerPanel>
          <div className="panel-header">
            <span className="label">Video</span>
            <div className="controls">
              {(['top', 'right', 'low_right', 'left'] as const).map((v) => {
                const enabled = sample.videoViews.includes(v);
                if (!enabled) return null;
                const label = v === 'low_right' ? 'Low-R' : v === 'left' ? 'Left' : v === 'right' ? 'Right' : 'Top';
                return (
                  <LayerToggle
                    key={v}
                    type="button"
                    $active={videoView === v}
                    onClick={() => setVideoView(v)}
                  >
                    {label}
                  </LayerToggle>
                );
              })}
            </div>
          </div>
          <div className="panel-body">
            <VideoViewer
              ref={videoRef}
              sources={sample.modalities.video && paths ? paths.video : {}}
              activeView={videoView}
              playbackRate={playbackRate}
              hideNativeControls
              rotateDeg={{ top: 180 }}
            />
            {/* Sound source — separate <audio> so blurred videos (audio stripped
                by ffmpeg -an) still produce sound. Synced to videoRef above. */}
            {sample.modalities.audio && paths && (
              <audio
                ref={audioRef}
                src={paths.audio}
                preload="auto"
                style={{ display: 'none' }}
              />
            )}
          </div>
        </ViewerPanel>
      </ViewerGrid>

      <TrackGroup>
        <Track style={{ flex: '9 1 0' }}>
          <TrackLabel>
            <span className="title">Audio</span>
          </TrackLabel>
          <TrackBody>
            <AudioWaveformViewer
              src={sample.modalities.audio && paths ? paths.audio : ''}
              currentTimeSec={currentTimeSec}
              durationSec={durationSec}
            />
          </TrackBody>
        </Track>
        <Track style={{ flex: '17 1 0' }}>
          <TrackLabel>
            <span className="title">MIDI</span>
          </TrackLabel>
          <TrackBody>
            <PianoRollViewer
              src={sample.modalities.midi && paths ? paths.midi : ''}
              currentTimeSec={currentTimeSec}
              durationSec={durationSec || undefined}
            />
          </TrackBody>
        </Track>
        <Track style={{ flex: '17 1 0' }}>
          <TrackLabel>
            <span className="title">Kinematics</span>
          </TrackLabel>
          <TrackBody>
            <KinematicsViewer
              trialId={sample.trialId}
              path={sample.modalities.kinematics && paths ? paths.kinematics : ''}
              currentTimeSec={currentTimeSec}
              durationSec={durationSec || undefined}
            />
          </TrackBody>
        </Track>
      </TrackGroup>
        </MainColumn>
      </Stage>
      </Page>

      <PlaybackBar>
        <PlayBtn type="button" onClick={togglePlay} aria-label={isPlaying ? 'Pause' : 'Play'}>
          {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
        </PlayBtn>
        <PlayBtn
          type="button"
          onClick={stopPlayback}
          aria-label="Stop"
          disabled={!isPlaying && currentTimeSec === 0}
        >
          <StopIcon />
        </PlayBtn>
        <TimeLabel>{formatTime(currentTimeSec)}</TimeLabel>
        <ScrubInput
          type="range"
          min={0}
          max={durationSec || 0}
          step={0.01}
          value={Math.min(currentTimeSec, durationSec || 0)}
          onChange={(e) => seek(parseFloat(e.target.value))}
          disabled={durationSec <= 0}
        />
        <TimeLabel>{formatTime(durationSec)}</TimeLabel>
        <RateSelect
          value={playbackRate}
          onChange={(e) => {
            const r = parseFloat(e.target.value);
            setPlaybackRate(r);
          }}
        >
          {[0.5, 0.75, 1, 1.25, 1.5, 2].map((r) => (
            <option key={r} value={r}>
              {r}×
            </option>
          ))}
        </RateSelect>
        <VolumeBlock>
          <IconBtn
            type="button"
            onClick={() => setMuted((m) => !m)}
            aria-label={muted ? 'Unmute' : 'Mute'}
          >
            {muted || volume === 0 ? <VolumeOffIcon /> : <VolumeUpIcon />}
          </IconBtn>
          <VolumeInput
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={muted ? 0 : volume}
            onChange={(e) => {
              const v = parseFloat(e.target.value);
              setVolume(v);
              if (v > 0 && muted) setMuted(false);
            }}
          />
        </VolumeBlock>
      </PlaybackBar>
    </>
  );
}
