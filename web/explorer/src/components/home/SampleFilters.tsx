import { styled } from '@mui/material/styles';
import SearchIcon from '@mui/icons-material/Search';
import { BRAND, SURFACE } from '@/theme';
import type { SampleTrialEntry } from '@/data/types';

export type PieceLevelFilter = 'All' | NonNullable<SampleTrialEntry['pieceLevel']>;

export interface FilterState {
  task: 'All' | SampleTrialEntry['task'];
  level: 'All' | SampleTrialEntry['level'];
  session: 'All' | string;
  subject: 'All' | string;
  pieceLevel: PieceLevelFilter;
  compareOnly: boolean;
  query: string;
}

export const DEFAULT_FILTER: FilterState = {
  task: 'All',
  level: 'All',
  session: 'All',
  subject: 'All',
  pieceLevel: 'All',
  compareOnly: false,
  query: '',
};

const Bar = styled('div')({
  display: 'flex',
  alignItems: 'center',
  gap: 10,
  flexWrap: 'wrap',
  padding: '10px 14px',
  borderRadius: 10,
  background: SURFACE.paper,
  border: `1px solid ${SURFACE.border}`,
});

const Group = styled('label')({
  display: 'inline-flex',
  alignItems: 'center',
  gap: 6,
  fontSize: 12,
  fontWeight: 600,
  color: SURFACE.textMuted,
  '& select': {
    padding: '6px 24px 6px 10px',
    border: `1px solid ${SURFACE.border}`,
    borderRadius: 6,
    backgroundColor: SURFACE.paper,
    backgroundImage:
      "url(\"data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%235A5A6E' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M6 8l4 4 4-4'/%3e%3c/svg%3e\")",
    backgroundPosition: 'right 6px center',
    backgroundRepeat: 'no-repeat',
    backgroundSize: 12,
    fontSize: 12,
    fontWeight: 600,
    color: SURFACE.textPrimary,
    appearance: 'none',
    cursor: 'pointer',
    minWidth: 88,
    '&:focus': { outline: 'none', borderColor: BRAND.PRIMARY, boxShadow: `0 0 0 3px ${BRAND.PRIMARY}22` },
  },
});

const Search = styled('label')({
  display: 'inline-flex',
  alignItems: 'center',
  gap: 6,
  flex: '1 1 200px',
  minWidth: 180,
  maxWidth: 360,
  padding: '4px 10px',
  border: `1px solid ${SURFACE.border}`,
  borderRadius: 6,
  background: SURFACE.paper,
  '& svg': { fontSize: 16, color: SURFACE.textMuted },
  '& input': {
    flex: 1,
    border: 'none',
    outline: 'none',
    background: 'transparent',
    fontSize: 13,
    color: SURFACE.textPrimary,
    padding: '4px 0',
    minWidth: 0,
    '&::placeholder': { color: SURFACE.textMuted },
  },
  '&:focus-within': {
    borderColor: BRAND.PRIMARY,
    boxShadow: `0 0 0 3px ${BRAND.PRIMARY}22`,
  },
});

const Toggle = styled('button', {
  shouldForwardProp: (prop) => prop !== 'active',
})<{ active: boolean }>(({ active }) => ({
  display: 'inline-flex',
  alignItems: 'center',
  gap: 4,
  fontSize: 12,
  fontWeight: 600,
  padding: '6px 10px',
  borderRadius: 999,
  border: `1px solid ${active ? BRAND.PRIMARY : SURFACE.border}`,
  background: active ? `${BRAND.PRIMARY}10` : SURFACE.paper,
  color: active ? BRAND.PRIMARY : SURFACE.textSecondary,
  cursor: 'pointer',
  transition: 'all 0.15s ease',
  '&:hover': {
    borderColor: BRAND.PRIMARY,
    color: BRAND.PRIMARY,
    background: `${BRAND.PRIMARY}08`,
  },
  '&:focus': {
    outline: 'none',
    boxShadow: `0 0 0 3px ${BRAND.PRIMARY}22`,
  },
}));

const ResetButton = styled('button')({
  fontSize: 11,
  fontWeight: 600,
  color: SURFACE.textMuted,
  background: 'transparent',
  border: 'none',
  padding: '4px 8px',
  borderRadius: 6,
  cursor: 'pointer',
  '&:hover': { color: BRAND.PRIMARY, background: `${BRAND.PRIMARY}08` },
});

const Count = styled('span')({
  fontSize: 12,
  color: SURFACE.textMuted,
  fontWeight: 600,
  marginLeft: 'auto',
  fontVariantNumeric: 'tabular-nums',
});

const TASK_OPTIONS: FilterState['task'][] = ['All', 'T0', 'T1', 'T2', 'T9'];
const LEVEL_OPTIONS: FilterState['level'][] = ['All', 'Professional', 'Amateur'];
const PIECE_LEVEL_OPTIONS: PieceLevelFilter[] = [
  'All',
  'Beginner',
  'Intermediate',
  'Advanced',
  'Professional',
];

const TASK_LABEL: Record<FilterState['task'], string> = {
  All: 'All',
  T0: 'T0 · Technique 1',
  T1: 'T1 · Technique 2',
  T2: 'T2 · Graded',
  T9: 'T9 · Free',
};

const formatSessionLabel = (s: string): string => {
  // 'YYMMDD' → '20YY.MM.DD'
  if (/^\d{6}$/.test(s)) {
    return `20${s.slice(0, 2)}.${s.slice(2, 4)}.${s.slice(4, 6)}`;
  }
  return s;
};

interface Props {
  value: FilterState;
  onChange: (next: FilterState) => void;
  resultCount: number;
  totalCount: number;
  availableSessions: string[];
  availableSubjects: string[];
  availablePieceLevels: PieceLevelFilter[];
}

export default function SampleFilters({
  value,
  onChange,
  resultCount,
  totalCount,
  availableSessions,
  availableSubjects,
  availablePieceLevels,
}: Props) {
  const patch = <K extends keyof FilterState>(key: K, v: FilterState[K]) =>
    onChange({ ...value, [key]: v });

  const isDirty =
    value.task !== 'All' ||
    value.level !== 'All' ||
    value.session !== 'All' ||
    value.subject !== 'All' ||
    value.pieceLevel !== 'All' ||
    value.compareOnly ||
    value.query !== '';

  const pieceLevelOptions: PieceLevelFilter[] = PIECE_LEVEL_OPTIONS.filter(
    (p) => p === 'All' || availablePieceLevels.includes(p),
  );

  return (
    <Bar>
      <Group>
        Task
        <select value={value.task} onChange={(e) => patch('task', e.target.value as FilterState['task'])}>
          {TASK_OPTIONS.map((t) => (
            <option key={t} value={t}>
              {TASK_LABEL[t]}
            </option>
          ))}
        </select>
      </Group>
      <Group>
        Level
        <select
          value={value.level}
          onChange={(e) => patch('level', e.target.value as FilterState['level'])}
        >
          {LEVEL_OPTIONS.map((l) => (
            <option key={l} value={l}>
              {l}
            </option>
          ))}
        </select>
      </Group>
      <Group>
        Session
        <select value={value.session} onChange={(e) => patch('session', e.target.value)}>
          <option value="All">All</option>
          {availableSessions.map((s) => (
            <option key={s} value={s}>
              {formatSessionLabel(s)}
            </option>
          ))}
        </select>
      </Group>
      <Group>
        Subject
        <select value={value.subject} onChange={(e) => patch('subject', e.target.value)}>
          <option value="All">All</option>
          {availableSubjects.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </Group>
      {pieceLevelOptions.length > 1 && (
        <Group>
          Piece
          <select
            value={value.pieceLevel}
            onChange={(e) => patch('pieceLevel', e.target.value as PieceLevelFilter)}
          >
            {pieceLevelOptions.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </Group>
      )}
      <Toggle
        type="button"
        active={value.compareOnly}
        onClick={() => patch('compareOnly', !value.compareOnly)}
        title="Show only trials paired across pro / amateur"
      >
        Paired
      </Toggle>
      <Search>
        <SearchIcon />
        <input
          type="search"
          value={value.query}
          placeholder="Search piece, subject, trial ID…"
          onChange={(e) => patch('query', e.target.value)}
        />
      </Search>
      {isDirty && (
        <ResetButton type="button" onClick={() => onChange(DEFAULT_FILTER)}>
          Reset
        </ResetButton>
      )}
      <Count>
        {resultCount === totalCount
          ? `${totalCount} trial${totalCount === 1 ? '' : 's'}`
          : `${resultCount} of ${totalCount}`}
      </Count>
    </Bar>
  );
}
