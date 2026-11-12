import { useEffect, useMemo, useState } from 'react';
import { styled } from '@mui/material/styles';
import { Link } from 'react-router-dom';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import ArrowDropUpIcon from '@mui/icons-material/ArrowDropUp';
import { loadSampleTrials } from '@/data/dataRepository';
import type { SampleTrialEntry } from '@/data/types';
import HomeHeader from '@/components/home/HomeHeader';
import SampleFilters, {
  DEFAULT_FILTER,
  type FilterState,
  type PieceLevelFilter,
} from '@/components/home/SampleFilters';
import SampleRow, { ROW_GRID } from '@/components/home/SampleRow';
import TopBar from '@/components/common/TopBar';
import PianoIcon from '@/components/svgs/PianoIcon';
import { BRAND, GRADIENT, SURFACE } from '@/theme';

const Page = styled('div')({
  flex: '1 1 0',
  minHeight: 0,
  display: 'flex',
  flexDirection: 'column',
  gap: 16,
  padding: '20px 32px 40px',
  boxSizing: 'border-box',
  overflowY: 'auto',
});

const Table = styled('div')({
  display: 'flex',
  flexDirection: 'column',
  borderRadius: 10,
  border: `1px solid ${SURFACE.border}`,
  background: SURFACE.paper,
  overflow: 'hidden',
  boxShadow: SURFACE.shadow,
});

const HeaderRow = styled('div')({
  display: 'grid',
  gridTemplateColumns: ROW_GRID,
  alignItems: 'center',
  columnGap: 12,
  padding: '10px 14px 10px 14px',
  background: `${BRAND.PRIMARY}06`,
  borderBottom: `1px solid ${SURFACE.borderStrong}`,
  position: 'sticky',
  top: 0,
  zIndex: 1,
  fontSize: 11,
  fontWeight: 700,
  color: SURFACE.textSecondary,
  letterSpacing: '0.04em',
  textTransform: 'uppercase',
  userSelect: 'none',
});

const HeaderCell = styled('button', {
  shouldForwardProp: (prop) => prop !== 'sortable' && prop !== 'active',
})<{ sortable: boolean; active: boolean }>(({ sortable, active }) => ({
  background: 'transparent',
  border: 'none',
  padding: 0,
  margin: 0,
  textAlign: 'left',
  font: 'inherit',
  letterSpacing: 'inherit',
  textTransform: 'inherit',
  color: active ? BRAND.PRIMARY : 'inherit',
  cursor: sortable ? 'pointer' : 'default',
  display: 'inline-flex',
  alignItems: 'center',
  gap: 2,
  minWidth: 0,
  '& svg': {
    fontSize: 16,
    marginLeft: -2,
    opacity: active ? 1 : 0,
  },
  '&:hover': sortable
    ? { color: BRAND.PRIMARY }
    : undefined,
  '&:focus-visible': {
    outline: `2px solid ${BRAND.PRIMARY}`,
    outlineOffset: 2,
  },
}));

const Body = styled('div')({
  display: 'flex',
  flexDirection: 'column',
  '& > *:last-child': {
    borderBottom: 'none',
  },
});

const EmptyState = styled('div')({
  padding: '48px 0',
  textAlign: 'center',
  color: SURFACE.textMuted,
  fontSize: 14,
});

const BrandSlot = styled(Link)({
  display: 'flex',
  alignItems: 'center',
  gap: 10,
  textDecoration: 'none',
  padding: '4px 6px',
  borderRadius: 8,
  transition: 'background 0.15s ease',
  '&:hover': { background: 'rgba(15, 15, 112, 0.04)' },
  '& .mark': { width: 26, height: 26, borderRadius: 6, flexShrink: 0 },
  '& .wordmark': {
    fontSize: 14,
    fontWeight: 700,
    letterSpacing: '-0.01em',
    background: GRADIENT.brand,
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
    lineHeight: 1,
  },
  '& .sub': {
    fontSize: 10,
    fontWeight: 600,
    color: SURFACE.textMuted,
    letterSpacing: '0.08em',
    textTransform: 'uppercase',
    paddingLeft: 8,
    marginLeft: 8,
    borderLeft: `1px solid ${SURFACE.border}`,
    lineHeight: 1,
  },
});

// ── Sort ────────────────────────────────────────────────────────────────────
type SortKey = 'task' | 'level' | 'grade' | 'piece' | 'subject' | 'session' | 'duration';
type SortDir = 'asc' | 'desc';
interface SortState {
  key: SortKey;
  dir: SortDir;
}
const DEFAULT_SORT: SortState = { key: 'task', dir: 'asc' };

const TASK_ORDER: Record<SampleTrialEntry['task'], number> = { T0: 0, T1: 1, T2: 2, T9: 3 };
const LEVEL_ORDER: Record<SampleTrialEntry['level'], number> = { Professional: 0, Amateur: 1 };
const PIECE_LEVEL_ORDER: Record<NonNullable<SampleTrialEntry['pieceLevel']>, number> = {
  Beginner: 0,
  Intermediate: 1,
  Advanced: 2,
  Professional: 3,
};

// "~M:SS" or "M:SS" → seconds. Falls back to Infinity so missing values sort last.
function durationToSeconds(label?: string): number {
  if (!label) return Number.POSITIVE_INFINITY;
  const m = label.match(/(\d+):(\d{2})/);
  if (!m) return Number.POSITIVE_INFINITY;
  return parseInt(m[1]!, 10) * 60 + parseInt(m[2]!, 10);
}

function compareSamples(a: SampleTrialEntry, b: SampleTrialEntry, key: SortKey): number {
  switch (key) {
    case 'task': {
      const t = TASK_ORDER[a.task] - TASK_ORDER[b.task];
      if (t !== 0) return t;
      // T2 graded → piece-difficulty order
      if (a.task === 'T2' && b.task === 'T2') {
        const pa = a.pieceLevel ? PIECE_LEVEL_ORDER[a.pieceLevel] : 99;
        const pb = b.pieceLevel ? PIECE_LEVEL_ORDER[b.pieceLevel] : 99;
        if (pa !== pb) return pa - pb;
      }
      const l = LEVEL_ORDER[a.level] - LEVEL_ORDER[b.level];
      if (l !== 0) return l;
      return a.subject.localeCompare(b.subject);
    }
    case 'level':
      return LEVEL_ORDER[a.level] - LEVEL_ORDER[b.level];
    case 'grade': {
      // Trials without pieceLevel sort last in both directions (treated as 99).
      const ga = a.pieceLevel ? PIECE_LEVEL_ORDER[a.pieceLevel] : 99;
      const gb = b.pieceLevel ? PIECE_LEVEL_ORDER[b.pieceLevel] : 99;
      return ga - gb;
    }
    case 'piece':
      return a.pieceLabel.localeCompare(b.pieceLabel);
    case 'subject':
      return a.subject.localeCompare(b.subject);
    case 'session':
      return a.session.localeCompare(b.session);
    case 'duration':
      return durationToSeconds(a.durationLabel) - durationToSeconds(b.durationLabel);
  }
}

function sortSamples(list: SampleTrialEntry[], sort: SortState): SampleTrialEntry[] {
  const sign = sort.dir === 'asc' ? 1 : -1;
  return [...list].sort((a, b) => {
    const primary = compareSamples(a, b, sort.key) * sign;
    if (primary !== 0) return primary;
    // Stable tiebreak by trialId so reorders are deterministic.
    return a.trialId.localeCompare(b.trialId);
  });
}

const TASK_KEYWORD_HINTS: Record<SampleTrialEntry['task'], string> = {
  T0: 't0 technique 1 warmup exercise',
  T1: 't1 technique 2 scale',
  T2: 't2 graded piece sonata',
  T9: 't9 free improvisation',
};

function matchesQuery(sample: SampleTrialEntry, q: string): boolean {
  if (!q) return true;
  const hay = [
    sample.pieceLabel,
    sample.subject,
    sample.trialId,
    sample.session,
    sample.level,
    TASK_KEYWORD_HINTS[sample.task] ?? sample.task,
    sample.pieceLevel ?? '',
  ]
    .join(' ')
    .toLowerCase();
  return hay.includes(q);
}

// Column header config — undefined sortKey = non-sortable column.
const COLUMNS: ReadonlyArray<{ label: string; sortKey?: SortKey }> = [
  { label: 'Task', sortKey: 'task' },
  { label: 'Level', sortKey: 'level' },
  { label: 'Grade', sortKey: 'grade' },
  { label: 'Piece', sortKey: 'piece' },
  { label: 'Subject', sortKey: 'subject' },
  { label: 'Session', sortKey: 'session' },
  { label: 'Dur', sortKey: 'duration' },
  { label: 'Modalities' },
];

export default function HomeView() {
  const [samples, setSamples] = useState<SampleTrialEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<FilterState>(DEFAULT_FILTER);
  const [sort, setSort] = useState<SortState>(DEFAULT_SORT);

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

  const availableSessions = useMemo(
    () => Array.from(new Set(samples.map((s) => s.session))).sort(),
    [samples],
  );
  const availableSubjects = useMemo(
    () => Array.from(new Set(samples.map((s) => s.subject))).sort(),
    [samples],
  );
  const availablePieceLevels = useMemo<PieceLevelFilter[]>(
    () =>
      Array.from(
        new Set(
          samples
            .map((s) => s.pieceLevel)
            .filter((p): p is NonNullable<SampleTrialEntry['pieceLevel']> => Boolean(p)),
        ),
      ),
    [samples],
  );

  const filtered = useMemo(() => {
    const q = filter.query.trim().toLowerCase();
    return sortSamples(
      samples.filter((s) => {
        if (filter.task !== 'All' && s.task !== filter.task) return false;
        if (filter.level !== 'All' && s.level !== filter.level) return false;
        if (filter.session !== 'All' && s.session !== filter.session) return false;
        if (filter.subject !== 'All' && s.subject !== filter.subject) return false;
        if (filter.pieceLevel !== 'All' && s.pieceLevel !== filter.pieceLevel) return false;
        if (filter.compareOnly && !s.comparePartnerId) return false;
        if (!matchesQuery(s, q)) return false;
        return true;
      }),
      sort,
    );
  }, [samples, filter, sort]);

  const onSortClick = (key?: SortKey) => {
    if (!key) return;
    setSort((prev) =>
      prev.key === key
        ? { key, dir: prev.dir === 'asc' ? 'desc' : 'asc' }
        : { key, dir: 'asc' },
    );
  };

  return (
    <>
    <TopBar>
      <BrandSlot to="/">
        <span className="mark"><PianoIcon /></span>
        <span className="wordmark">SKY Piano</span>
        <span className="sub">Dashboard</span>
      </BrandSlot>
    </TopBar>
    <Page>
      <HomeHeader trialCount={samples.length} mode="sample-preview" />
      <SampleFilters
        value={filter}
        onChange={setFilter}
        resultCount={filtered.length}
        totalCount={samples.length}
        availableSessions={availableSessions}
        availableSubjects={availableSubjects}
        availablePieceLevels={availablePieceLevels}
      />
      {loading ? (
        <EmptyState>Loading…</EmptyState>
      ) : filtered.length === 0 ? (
        <EmptyState>No trials match the current filters.</EmptyState>
      ) : (
        <Table role="table">
          <HeaderRow role="row">
            {COLUMNS.map((col) => {
              const active = !!col.sortKey && sort.key === col.sortKey;
              return (
                <HeaderCell
                  key={col.label}
                  role="columnheader"
                  type="button"
                  sortable={!!col.sortKey}
                  active={active}
                  onClick={() => onSortClick(col.sortKey)}
                  aria-sort={
                    active ? (sort.dir === 'asc' ? 'ascending' : 'descending') : 'none'
                  }
                >
                  <span>{col.label}</span>
                  {active && (sort.dir === 'asc' ? <ArrowDropUpIcon /> : <ArrowDropDownIcon />)}
                </HeaderCell>
              );
            })}
          </HeaderRow>
          <Body role="rowgroup">
            {filtered.map((s) => (
              <SampleRow key={s.trialId} sample={s} />
            ))}
          </Body>
        </Table>
      )}
    </Page>
    </>
  );
}
