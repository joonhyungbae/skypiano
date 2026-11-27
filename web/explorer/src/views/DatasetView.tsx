import { useEffect, useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { styled } from '@mui/material/styles';
import TopBar from '@/components/common/TopBar';
import { BRAND, SURFACE } from '@/theme';
import {
  loadDatasetIndex,
  participantLabel,
  pieceTitle,
  TASK_NAMES,
  type DatasetIndex,
  type Row,
} from '@/data/dataset';

/**
 * The whole release as one filterable table: every performance (one round of
 * one piece by one participant) or every whole recorded take. Filters live in
 * the URL, so a filtered list can be bookmarked and Back returns to it.
 */

const PAGE_SIZE = 100;

const Page = styled('div')({
  flex: '1 1 0',
  minHeight: 0,
  overflow: 'auto',
  padding: '20px 24px 40px',
  background: SURFACE.bg,
});

const Brand = styled('div')({
  display: 'flex',
  alignItems: 'baseline',
  gap: 10,
  '& .name': { fontWeight: 800, color: BRAND.PRIMARY, fontSize: 16 },
  '& .sub': { color: SURFACE.textSecondary, fontSize: 13 },
});

const Filters = styled('div')({
  display: 'flex',
  flexWrap: 'wrap',
  gap: 10,
  alignItems: 'center',
  marginBottom: 14,
  '& input, & select': {
    height: 34,
    padding: '0 10px',
    border: `1px solid ${SURFACE.border}`,
    borderRadius: 8,
    background: SURFACE.paper,
    color: SURFACE.textPrimary,
    fontSize: 13,
  },
  '& input': { width: 260 },
  '& .count': { marginLeft: 'auto', color: SURFACE.textSecondary, fontSize: 13 },
});

const Tabs = styled('div')({
  display: 'inline-flex',
  border: `1px solid ${SURFACE.border}`,
  borderRadius: 8,
  overflow: 'hidden',
  '& button': {
    height: 34,
    padding: '0 14px',
    border: 'none',
    background: SURFACE.paper,
    color: SURFACE.textSecondary,
    fontSize: 13,
    fontWeight: 600,
    cursor: 'pointer',
  },
  '& button.on': { background: BRAND.PRIMARY, color: '#fff' },
});

const Table = styled('table')({
  width: '100%',
  borderCollapse: 'collapse',
  background: SURFACE.paper,
  border: `1px solid ${SURFACE.border}`,
  borderRadius: 10,
  fontSize: 13,
  '& th': {
    position: 'sticky',
    top: -20,
    background: SURFACE.paper,
    textAlign: 'left',
    padding: '8px 10px',
    color: SURFACE.textSecondary,
    fontWeight: 600,
    borderBottom: `1px solid ${SURFACE.border}`,
    whiteSpace: 'nowrap',
  },
  '& td': { padding: '7px 10px', borderBottom: `1px solid ${SURFACE.border}`, verticalAlign: 'top' },
  '& tr:hover td': { background: `${BRAND.PRIMARY}06` },
  '& a': { color: BRAND.PRIMARY, fontWeight: 600, textDecoration: 'none' },
  '& a:hover': { textDecoration: 'underline' },
  '& .num': { textAlign: 'right', fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap' },
  '& .muted': { color: SURFACE.textMuted },
});

const Tag = styled('span')({
  display: 'inline-block',
  margin: '0 4px 2px 0',
  padding: '1px 6px',
  borderRadius: 5,
  fontSize: 11,
  fontWeight: 600,
  background: `${BRAND.SECONDARY}10`,
  color: BRAND.SECONDARY,
});

const Pager = styled('div')({
  display: 'flex',
  justifyContent: 'center',
  gap: 8,
  marginTop: 14,
  '& button': {
    height: 32,
    padding: '0 12px',
    border: `1px solid ${SURFACE.border}`,
    borderRadius: 8,
    background: SURFACE.paper,
    cursor: 'pointer',
  },
  '& button:disabled': { opacity: 0.4, cursor: 'default' },
  '& span': { alignSelf: 'center', color: SURFACE.textSecondary, fontSize: 13 },
});

const Note = styled('div')({
  maxWidth: 720,
  margin: '40px auto',
  padding: 24,
  background: SURFACE.paper,
  border: `1px solid ${SURFACE.border}`,
  borderRadius: 12,
  color: SURFACE.textPrimary,
  lineHeight: 1.6,
  '& code, & pre': { background: SURFACE.bg, borderRadius: 6, padding: '1px 6px', fontSize: 13 },
  '& pre': { padding: 12, overflowX: 'auto' },
});

/** Grouped modality tags shown per row (masks are implied by their file). */
function modalityTags(mods: string): string[] {
  const m = new Set(mods.split(/\s+/).filter(Boolean));
  const tags: string[] = [];
  const video = ['top', 'right', 'right_low', 'left'].filter((v) => m.has(v)).length;
  if (video) tags.push(`video ×${video}`);
  if (m.has('audio')) tags.push('audio');
  if (m.has('midi')) tags.push('MIDI');
  if (m.has('hand')) tags.push(m.has('hand_imputed') ? 'hand + filled' : 'hand');
  if (m.has('body')) tags.push(m.has('body_imputed') ? 'body + filled' : 'body');
  if (m.has('fingering')) tags.push('fingering');
  return tags;
}

const seconds = (a?: string, b?: string) => {
  const d = parseFloat(b ?? '') - parseFloat(a ?? '');
  return Number.isFinite(d) && d > 0 ? d : null;
};
const fmt = (s: number | null) =>
  s === null ? '' : s >= 60 ? `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}` : `${s.toFixed(1)} s`;

export default function DatasetView() {
  const [index, setIndex] = useState<DatasetIndex | null | undefined>(undefined);
  const [params, setParams] = useSearchParams();
  const kind = params.get('kind') === 'takes' ? 'takes' : 'performances';
  const q = params.get('q') ?? '';
  const task = params.get('task') ?? '';
  const pid = params.get('p') ?? '';
  const group = params.get('group') ?? '';
  const need = params.get('has') ?? '';
  const page = Math.max(0, parseInt(params.get('page') ?? '0', 10) || 0);

  useEffect(() => {
    void loadDatasetIndex().then(setIndex);
  }, []);

  const set = (k: string, v: string) => {
    const next = new URLSearchParams(params);
    if (v) next.set(k, v);
    else next.delete(k);
    if (k !== 'page') next.delete('page');
    setParams(next, { replace: true });
  };

  const rows = useMemo(() => {
    if (!index) return [] as Row[];
    const src = kind === 'takes' ? index.takes : index.performances;
    const words = q.toLowerCase().split(/\s+/).filter(Boolean);
    return src.filter((r) => {
      if (task && kind === 'performances' && r.task !== task) return false;
      if (pid && r.participant_id !== pid) return false;
      if (group && r.group !== group) return false;
      if (need && !(r.modalities ?? '').split(/\s+/).includes(need)) return false;
      if (words.length) {
        const hay = `${r.path} ${r.piece_id ?? ''} ${kind === 'performances' ? pieceTitle(index, r.piece_id) : ''} ${r.session ?? ''} ${r.source_take ?? ''}`.toLowerCase();
        if (!words.every((w) => hay.includes(w))) return false;
      }
      return true;
    });
  }, [index, kind, q, task, pid, group, need]);

  if (index === undefined) {
    return (
      <>
        <TopBar><Brand><span className="name">SKY Piano Dataset</span></Brand></TopBar>
        <Page><Note>Loading the dataset index…</Note></Page>
      </>
    );
  }

  if (index === null) {
    return (
      <>
        <TopBar><Brand><span className="name">SKY Piano Dataset</span></Brand></TopBar>
        <Page>
          <Note>
            <h3 style={{ marginTop: 0 }}>No dataset is being served</h3>
            <p>
              The full-dataset browser reads a local copy of the release. Download the dataset from
              Hugging Face, build the explorer once, and start the local server from the repository root:
            </p>
            <pre>{`cd web/explorer && npm install && npm run build:dataset && cd ../..
python web/explorer/serve_dataset.py --data path/to/SKY-Piano`}</pre>
            <p>Then open the address it prints. The public site only carries the curated samples.</p>
          </Note>
        </Page>
      </>
    );
  }

  const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const shown = rows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const tasks = Array.from(new Set(index.performances.map((r) => r.task))).sort();
  const pids = index.participants.map((r) => r.participant_id);
  const linkTo = (r: Row) =>
    `/data/view?kind=${kind === 'takes' ? 'take' : 'performance'}&path=${encodeURIComponent(r.path)}`;

  return (
    <>
      <TopBar>
        <Brand>
          <span className="name">SKY Piano Dataset</span>
          <span className="sub">
            {index.performances.length} performances · {index.takes.length} takes · {index.participants.length} participants
          </span>
        </Brand>
      </TopBar>
      <Page>
        <Filters>
          <Tabs>
            <button className={kind === 'performances' ? 'on' : ''} onClick={() => set('kind', '')}>Performances</button>
            <button className={kind === 'takes' ? 'on' : ''} onClick={() => set('kind', 'takes')}>Whole takes</button>
          </Tabs>
          <input placeholder="Search piece, path, session…" value={q} onChange={(e) => set('q', e.target.value)} />
          {kind === 'performances' && (
            <select value={task} onChange={(e) => set('task', e.target.value)}>
              <option value="">All tasks</option>
              {tasks.map((t) => <option key={t} value={t}>{TASK_NAMES[t] ?? t}</option>)}
            </select>
          )}
          <select value={pid} onChange={(e) => set('p', e.target.value)}>
            <option value="">All participants</option>
            {pids.map((p) => <option key={p} value={p}>{participantLabel(p)}</option>)}
          </select>
          <select value={group} onChange={(e) => set('group', e.target.value)}>
            <option value="">Both groups</option>
            <option value="professional">Professional</option>
            <option value="amateur">Amateur</option>
          </select>
          <select value={need} onChange={(e) => set('has', e.target.value)}>
            <option value="">Any modality</option>
            <option value="top">with video</option>
            <option value="body">with body</option>
            <option value="body_imputed">with filled body</option>
            <option value="hand_imputed">with filled hand</option>
            <option value="audio">with audio</option>
          </select>
          <span className="count">{rows.length} shown</span>
        </Filters>

        <Table>
          <thead>
            {kind === 'performances' ? (
              <tr>
                <th>Piece</th><th>Task</th><th>Participant</th><th>Round</th><th>Session</th>
                <th className="num">Length</th><th>Data</th>
              </tr>
            ) : (
              <tr>
                <th>Take</th><th>Participant</th><th>Session</th>
                <th className="num">Audio</th><th className="num">Motion</th><th className="num">Notes</th><th>Data</th>
              </tr>
            )}
          </thead>
          <tbody>
            {shown.map((r) =>
              kind === 'performances' ? (
                <tr key={r.path}>
                  <td><Link to={linkTo(r)}>{pieceTitle(index, r.piece_id)}</Link></td>
                  <td>{TASK_NAMES[r.task] ?? r.task}</td>
                  <td>{participantLabel(r.participant_id)} <span className="muted">{r.group}</span></td>
                  <td className="num">{r.round}</td>
                  <td>{r.session}</td>
                  <td className="num">{fmt(seconds(r.take_start_sec, r.take_end_sec))}</td>
                  <td>{modalityTags(r.modalities ?? '').map((t) => <Tag key={t}>{t}</Tag>)}</td>
                </tr>
              ) : (
                <tr key={r.path}>
                  <td><Link to={linkTo(r)}>{r.source_take || r.path.split('/').pop()}</Link></td>
                  <td>{participantLabel(r.participant_id)} <span className="muted">{r.group}</span></td>
                  <td>{r.session}</td>
                  <td className="num">{fmt(parseFloat(r.audio_sec) || null)}</td>
                  <td className="num">{fmt(parseFloat(r.motion_sec) || null)}</td>
                  <td className="num">{r.midi_notes}</td>
                  <td>{modalityTags(r.modalities ?? '').map((t) => <Tag key={t}>{t}</Tag>)}</td>
                </tr>
              ),
            )}
          </tbody>
        </Table>
        {pages > 1 && (
          <Pager>
            <button disabled={page === 0} onClick={() => set('page', String(page - 1))}>Previous</button>
            <span>{page + 1} / {pages}</span>
            <button disabled={page >= pages - 1} onClick={() => set('page', String(page + 1))}>Next</button>
          </Pager>
        )}
      </Page>
    </>
  );
}
