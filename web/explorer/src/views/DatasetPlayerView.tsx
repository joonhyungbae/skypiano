import { useEffect, useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { styled } from '@mui/material/styles';
import TopBar from '@/components/common/TopBar';
import { PerformancePlayer } from '@/views/SampleView';
import { SURFACE, BRAND } from '@/theme';
import { DATASET_URL, loadDatasetIndex, toPlayer, type DatasetIndex } from '@/data/dataset';

/** One performance or take of the served release in the synchronised player. */

const Info = styled('div')({ padding: 40, color: SURFACE.textSecondary });

const Extra = styled('span')({
  display: 'inline-flex',
  gap: 8,
  alignItems: 'center',
  marginLeft: 6,
  fontSize: 12,
  '& a': {
    color: BRAND.PRIMARY,
    fontWeight: 600,
    textDecoration: 'none',
    padding: '3px 8px',
    border: `1px solid ${SURFACE.border}`,
    borderRadius: 6,
    whiteSpace: 'nowrap',
  },
  '& a.on': { background: BRAND.PRIMARY, color: '#fff' },
});

export default function DatasetPlayerView() {
  const [params] = useSearchParams();
  const path = params.get('path') ?? '';
  const kind = params.get('kind') === 'take' ? 'take' : 'performance';
  const [index, setIndex] = useState<DatasetIndex | null | undefined>(undefined);

  useEffect(() => {
    void loadDatasetIndex().then(setIndex);
  }, []);

  const row = useMemo(() => {
    if (!index) return null;
    return (kind === 'take' ? index.takes : index.performances).find((r) => r.path === path) ?? null;
  }, [index, kind, path]);

  const player = useMemo(() => (index && row ? toPlayer(index, row, kind) : null), [index, row, kind]);

  // other rounds of the same piece by the same participant
  const rounds = useMemo(() => {
    if (!index || !row || kind !== 'performance') return [];
    return index.performances
      .filter((r) => r.participant_id === row.participant_id && r.piece_id === row.piece_id && r.task === row.task)
      .sort((a, b) => Number(a.round) - Number(b.round));
  }, [index, row, kind]);

  useEffect(() => {
    if (player) document.title = `SKY Piano · ${player.title}`;
  }, [player]);

  if (index === undefined) {
    return (<><TopBar /><Info>Loading…</Info></>);
  }
  if (!index || !row || !player) {
    return (
      <>
        <TopBar />
        <Info>
          <code>{path}</code> is not in the served dataset. <Link to="/data">Back to the dataset</Link>
        </Info>
      </>
    );
  }

  const files = `${DATASET_URL}/${row.path}`;
  return (
    <PerformancePlayer
      key={row.path}
      sample={player.sample}
      paths={player.paths}
      taskLabel={player.taskLabel}
      hideUnavailable
      titleExtra={
        <Extra>
          <Link to="/data">All data</Link>
          {rounds.length > 1 &&
            rounds.map((r) => (
              <Link
                key={r.path}
                className={r.path === row.path ? 'on' : ''}
                to={`/data/view?kind=performance&path=${encodeURIComponent(r.path)}`}
              >
                r{r.round}
              </Link>
            ))}
          {kind === 'performance' && row.source_take && (
            <span style={{ color: SURFACE.textMuted }} title={files}>
              {row.source_take}
            </span>
          )}
        </Extra>
      }
    />
  );
}
