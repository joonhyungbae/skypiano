import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { Link, useMatch, useNavigate } from 'react-router-dom';
import { styled } from '@mui/material/styles';
import MenuIcon from '@mui/icons-material/Menu';
import CloseIcon from '@mui/icons-material/Close';
import SearchIcon from '@mui/icons-material/Search';
import { loadSampleTrials } from '@/data/dataRepository';
import type { SampleTrialEntry } from '@/data/types';
import PianoIcon from '../svgs/PianoIcon';
import { BRAND, GRADIENT, SURFACE } from '@/theme';

const TASK_ORDER: Record<SampleTrialEntry['task'], number> = { T0: 0, T1: 1, T2: 2, T9: 3 };
const LEVEL_ORDER: Record<SampleTrialEntry['level'], number> = { Professional: 0, Amateur: 1 };

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

const InlineToggle = styled('button')({
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  width: 32,
  height: 32,
  borderRadius: 8,
  border: `1px solid ${SURFACE.border}`,
  background: SURFACE.paper,
  color: SURFACE.textSecondary,
  cursor: 'pointer',
  transition: 'all 0.15s ease',
  padding: 0,
  flexShrink: 0,
  '&:hover': {
    color: BRAND.PRIMARY,
    borderColor: SURFACE.borderStrong,
    background: `${BRAND.PRIMARY}08`,
  },
  '& svg': { fontSize: 18 },
});

const Backdrop = styled('div')<{ $visible: boolean }>(({ $visible }) => ({
  position: 'fixed',
  inset: 0,
  background: 'rgba(15, 15, 112, 0.12)',
  backdropFilter: 'blur(2px)',
  opacity: $visible ? 1 : 0,
  pointerEvents: $visible ? 'auto' : 'none',
  transition: 'opacity 0.18s ease',
  zIndex: 2147483644,
}));

const Drawer = styled('aside')<{ $visible: boolean }>(({ $visible }) => ({
  position: 'fixed',
  top: 0,
  right: 0,
  bottom: 0,
  width: 'min(400px, 92vw)',
  background: SURFACE.paper,
  borderLeft: `1px solid ${SURFACE.border}`,
  boxShadow: '-16px 0 40px rgba(15, 15, 112, 0.18)',
  zIndex: 2147483645,
  transform: $visible ? 'translateX(0)' : 'translateX(100%)',
  transition: 'transform 0.22s ease',
  display: 'flex',
  flexDirection: 'column',
}));

const DrawerHeader = styled('header')({
  display: 'flex',
  flexDirection: 'column',
  gap: 10,
  padding: '18px 18px 12px',
  borderBottom: `1px solid ${SURFACE.border}`,
  '& .brand-row': {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    textDecoration: 'none',
    color: 'inherit',
    padding: '2px 4px',
    borderRadius: 8,
    transition: 'background 0.15s ease',
    '&:hover': { background: `${BRAND.PRIMARY}08` },
  },
  '& .brand-logo': {
    width: 30,
    height: 30,
    borderRadius: 7,
    boxShadow: '0 3px 10px rgba(15, 15, 112, 0.2)',
    flexShrink: 0,
  },
  '& .brand-text': {
    display: 'flex',
    flexDirection: 'column',
    lineHeight: 1.15,
  },
  '& .brand-name': {
    fontSize: 15,
    fontWeight: 700,
    color: SURFACE.textPrimary,
    letterSpacing: '-0.01em',
    background: GRADIENT.brand,
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
  },
  '& .brand-sub': {
    fontSize: 10,
    fontWeight: 600,
    color: SURFACE.textMuted,
    letterSpacing: '0.08em',
    textTransform: 'uppercase',
  },
  '& .close': {
    position: 'absolute',
    top: 14,
    right: 14,
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 30,
    height: 30,
    border: 'none',
    background: 'transparent',
    color: SURFACE.textMuted,
    borderRadius: 6,
    cursor: 'pointer',
    transition: 'all 0.15s ease',
    '&:hover': { background: `${BRAND.PRIMARY}0D`, color: BRAND.PRIMARY },
    '& svg': { fontSize: 18 },
  },
  '& .trial-title-row': {
    display: 'flex',
    alignItems: 'baseline',
    gap: 8,
  },
  '& .trial-title': {
    fontSize: 13,
    fontWeight: 700,
    color: SURFACE.textSecondary,
    letterSpacing: '0.04em',
    textTransform: 'uppercase',
    margin: 0,
  },
  '& .count': {
    fontSize: 11,
    fontWeight: 700,
    padding: '1px 7px',
    borderRadius: 999,
    background: `${BRAND.PRIMARY}12`,
    color: BRAND.PRIMARY,
    letterSpacing: '0.02em',
  },
});

const SearchBar = styled('label')({
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  padding: '10px 14px',
  borderBottom: `1px solid ${SURFACE.border}`,
  '& svg': { fontSize: 16, color: SURFACE.textMuted },
  '& input': {
    flex: 1,
    border: 'none',
    outline: 'none',
    background: 'transparent',
    fontSize: 13,
    color: SURFACE.textPrimary,
    minWidth: 0,
    padding: '2px 0',
    '&::placeholder': { color: SURFACE.textMuted },
  },
});

const List = styled('ul')({
  listStyle: 'none',
  margin: 0,
  padding: '6px 8px 12px',
  overflowY: 'auto',
  flex: 1,
});

const GroupHeader = styled('li')({
  padding: '14px 10px 6px',
  fontSize: 9,
  fontWeight: 700,
  letterSpacing: '0.1em',
  textTransform: 'uppercase',
  color: SURFACE.textMuted,
  pointerEvents: 'none',
});

const Row = styled('li')({
  display: 'flex',
  alignItems: 'flex-start',
  gap: 10,
  padding: '10px 12px',
  borderRadius: 8,
  cursor: 'pointer',
  transition: 'background 0.12s ease',
  '&:hover': { background: `${BRAND.PRIMARY}0A` },
  '&.active': {
    background: `${BRAND.PRIMARY}12`,
    outline: `1px solid ${BRAND.PRIMARY}33`,
  },
  '& .row-chip': {
    fontSize: 10,
    fontWeight: 700,
    padding: '2px 6px',
    borderRadius: 3,
    letterSpacing: '0.04em',
    textTransform: 'uppercase',
    border: '1px solid',
    flexShrink: 0,
    marginTop: 2,
  },
  '& .row-main': {
    flex: 1,
    minWidth: 0,
    display: 'flex',
    flexDirection: 'column',
    gap: 2,
    lineHeight: 1.3,
  },
  '& .row-piece': {
    fontSize: 13,
    fontWeight: 600,
    color: SURFACE.textPrimary,
  },
  '& .row-meta': {
    fontSize: 11,
    color: SURFACE.textMuted,
    fontFamily: 'monospace',
  },
  '& .row-level': {
    fontSize: 10,
    fontWeight: 700,
    color: SURFACE.textSecondary,
    textAlign: 'right',
    lineHeight: 1.3,
    flexShrink: 0,
    marginTop: 2,
  },
});

const Empty = styled('li')({
  padding: '24px 14px',
  fontSize: 12,
  color: SURFACE.textMuted,
  fontStyle: 'italic',
  textAlign: 'center',
  pointerEvents: 'none',
});

// ── Context + public API ────────────────────────────────────────────────────

interface TrialsMenuCtx {
  open: boolean;
  setOpen: (v: boolean) => void;
}

const TrialsMenuContext = createContext<TrialsMenuCtx | null>(null);

function useTrialsMenuCtx(): TrialsMenuCtx {
  const ctx = useContext(TrialsMenuContext);
  if (!ctx) throw new Error('useTrialsMenu must be used within <TrialsMenuProvider>');
  return ctx;
}

/** Inline button that opens the trials drawer. Mount wherever you like. */
export function TrialsMenuToggle(props: { className?: string; 'aria-label'?: string }) {
  const { setOpen } = useTrialsMenuCtx();
  return (
    <InlineToggle
      type="button"
      onClick={() => setOpen(true)}
      aria-haspopup="dialog"
      aria-label={props['aria-label'] ?? 'Open trials menu'}
      title="Trials"
      className={props.className}
    >
      <MenuIcon />
    </InlineToggle>
  );
}

/** Wrap the app once — provides drawer state and renders the backdrop+drawer. */
export function TrialsMenuProvider({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <TrialsMenuContext.Provider value={{ open, setOpen }}>
      {children}
      <TrialsDrawer open={open} onClose={() => setOpen(false)} />
    </TrialsMenuContext.Provider>
  );
}

// ── Drawer (backdrop + slide panel) ─────────────────────────────────────────

function TrialsDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const match = useMatch('/sample/:trialId');
  const activeTrialId = match?.params.trialId ?? null;
  const navigate = useNavigate();

  const [samples, setSamples] = useState<SampleTrialEntry[]>([]);
  const [query, setQuery] = useState('');
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    loadSampleTrials().then(setSamples);
  }, []);

  useEffect(() => {
    if (!open) return;
    setQuery('');
    setTimeout(() => inputRef.current?.focus(), 80);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  // Close on route change.
  useEffect(() => {
    onClose();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTrialId]);

  const sorted = useMemo(() => {
    return [...samples].sort((a, b) => {
      const t = TASK_ORDER[a.task] - TASK_ORDER[b.task];
      if (t !== 0) return t;
      const l = LEVEL_ORDER[a.level] - LEVEL_ORDER[b.level];
      if (l !== 0) return l;
      return a.subject.localeCompare(b.subject);
    });
  }, [samples]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return sorted;
    return sorted.filter((s) => {
      const hay = `${s.pieceLabel} ${s.subject} ${s.session} ${s.trialId}`.toLowerCase();
      return hay.includes(q);
    });
  }, [sorted, query]);

  const grouped = useMemo(() => {
    const map = new Map<SampleTrialEntry['task'], SampleTrialEntry[]>();
    for (const s of filtered) {
      const arr = map.get(s.task) ?? [];
      arr.push(s);
      map.set(s.task, arr);
    }
    return Array.from(map.entries()).sort((a, b) => TASK_ORDER[a[0]] - TASK_ORDER[b[0]]);
  }, [filtered]);

  const select = (trialId: string) => {
    navigate(`/sample/${trialId}`);
    onClose();
  };

  return (
    <>
      <Backdrop $visible={open} onClick={onClose} />
      <Drawer $visible={open} role="dialog" aria-label="Trials">
        <DrawerHeader>
          <button className="close" type="button" onClick={onClose} aria-label="Close">
            <CloseIcon />
          </button>
          <Link to="/" className="brand-row" onClick={onClose}>
            <span className="brand-logo">
              <PianoIcon />
            </span>
            <span className="brand-text">
              <span className="brand-name">SKY Piano</span>
              <span className="brand-sub">Dashboard</span>
            </span>
          </Link>
          <div className="trial-title-row">
            <span className="trial-title">Trials</span>
            {samples.length > 0 && <span className="count">{samples.length}</span>}
          </div>
        </DrawerHeader>
        <SearchBar>
          <SearchIcon />
          <input
            ref={inputRef}
            type="search"
            placeholder="Search piece or subject…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </SearchBar>
        <List>
          {grouped.length === 0 && <Empty>No trials match.</Empty>}
          {grouped.map(([task, rows]) => (
            <div key={task}>
              <GroupHeader>{TASK_LABEL[task]}</GroupHeader>
              {rows.map((s) => {
                const taskColor = TASK_COLORS[s.task];
                const isActive = s.trialId === activeTrialId;
                return (
                  <Row
                    key={s.trialId}
                    className={isActive ? 'active' : ''}
                    onClick={() => select(s.trialId)}
                  >
                    <span
                      className="row-chip"
                      style={{
                        background: `${taskColor}12`,
                        color: taskColor,
                        borderColor: `${taskColor}33`,
                      }}
                    >
                      {s.task}
                    </span>
                    <div className="row-main">
                      <span className="row-piece">{s.pieceLabel}</span>
                      <span className="row-meta">
                        {s.subject} · {s.session}
                        {s.durationLabel ? ` · ${s.durationLabel}` : ''}
                      </span>
                    </div>
                    <span className="row-level">
                      {s.level === 'Professional' ? 'Pro' : 'Amateur'}
                    </span>
                  </Row>
                );
              })}
            </div>
          ))}
        </List>
      </Drawer>
    </>
  );
}
