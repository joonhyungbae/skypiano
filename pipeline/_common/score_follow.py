"""
Pieces placed in a trial by their notes, in the order they were played.

Every performer played the same pieces in a known order (the session log, the
set's fixed order). A piece is placed by a reference performance of it: one
performance whose boundaries are confirmed (by review or by content), or,
where no confirmed performance exists, the most typical of the performances
cut from the logged spans.

Notes are reduced to onset frames: notes starting within FRAME_S of each other
form one frame, a unit vector over the 88 keys (a note also gives half weight
to the frames next to it, so a chord split across two frames still matches).
Time between frames is dropped, so the comparison does not depend on tempo.

A reference is matched against the whole trial by subsequence dynamic time
warping (steps (1,1), (1,2), (2,1): every reference frame is matched and the
tempo ratio stays within 1/2..2; the path of least summed cost 1 - cosine,
reported as its mean over the path).
Every stretch of the trial where the reference matches at mean cost
<= MAX_COST (a local minimum over end frames) is a candidate. The pieces are then chosen in log order, one
candidate per piece or none, without overlap, minimising the summed cost plus
LOG_W per second between a candidate's start and the logged start (a piece
played twice is taken at the attempt the log marks).

Repetitive pieces (an octave repeated, a scale run once more) are played more
or fewer times than the reference: a placed piece is extended while the trial
continues with one of the reference's first (last) EDGE_FRAMES frames at cost
< EDGE_COST,
each gap no longer than GAP_X times the piece's longest inter-onset gap (at most
GAP_CAP_X median gaps), and
never into the neighbouring placed piece; a single frame that does not fit (a
chord with a note missed) is passed over when the next one fits; a chord made
of some of the edge chord's notes only fits too (a fast repeat drops notes). Before that,
a first or last frame that repeats its neighbour's chord after a pause of
more than TRIM_X median gaps (and TRIM_MIN_S, unless the score's edge chord is
not repeated) is dropped -- the matching takes a repeated chord at no cost,
e.g. a scale's closing C followed later by the C another piece starts with.
Playing after the longest pause inside a piece is dropped when the piece
matches as well without it (another piece's start, a passage played again).
"""
from __future__ import annotations

import numpy as np
from numba import njit

FRAME_S = 0.05
MAX_COST = 0.45
LOG_W = 0.03
EDGE_COST = 0.3
EDGE_FRAMES = 1
GAP_X = 2.0
TRIM_X = 4.0
GAP_CAP_X = 3.0
TRIM_MIN_S = 1.0
SAME_CHORD = 0.95
TAIL_JOIN_S = 0.25
RESTART_GAP_S, RESTART_GAP_X, RESTART_TOL = 1.5, 8.0, 0.05
FOREIGN_COST = 0.35
REF_COUNT_TOL = 0.1
MEDOID_MAX = 8
LOCAL_MIN = 3
MAX_CANDIDATES = 40
SKIP_COST = MAX_COST + 60 * LOG_W


def onset_frames(notes: list[tuple[float, float, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """notes (start, end, pitch) -> (frame features, frame start time, frame last note-off).
    A frame is a chord: the notes starting within FRAME_S of its first note (a chord is
    never split by a fixed time grid); a note also counts half in a frame starting within
    2 * FRAME_S of its own, so a chord rolled over two frames still matches."""
    ns = sorted(n for n in notes if 21 <= n[2] <= 108)
    if not ns:
        return np.zeros((0, 88)), np.zeros(0), np.zeros(0)
    groups: list[list] = []
    for s, e, p in ns:
        if groups and s - groups[-1][0][0] <= FRAME_S:
            groups[-1].append((s, e, p))
        else:
            groups.append([(s, e, p)])
    F = np.zeros((len(groups), 88))
    ts = np.array([g[0][0] for g in groups])
    te = np.array([max(e for _, e, _ in g) for g in groups])
    for i, g in enumerate(groups):
        for _, _, p in g:
            F[i, p - 21] += 1
            for k in (i - 1, i + 1):
                if 0 <= k < len(groups) and abs(ts[k] - ts[i]) <= 2 * FRAME_S:
                    F[k, p - 21] += 0.5
    F /= np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-9)
    return F, ts, te


@njit(cache=True)
def _subdtw(C):
    n, m = C.shape
    D = np.full((n, m), np.inf)
    L = np.zeros((n, m))
    S = np.zeros((n, m), np.int64)
    for j in range(m):
        D[0, j] = C[0, j]
        L[0, j] = 1
        S[0, j] = j
    for i in range(1, n):
        for j in range(1, m):
            best = np.inf
            d = 0.0
            bl = 0.0
            bs = 0
            if L[i - 1, j - 1] > 0:
                v = D[i - 1, j - 1] + C[i, j]
                if v < best:
                    best, d, bl, bs = v, v, L[i - 1, j - 1] + 1, S[i - 1, j - 1]
            if j > 1 and L[i - 1, j - 2] > 0:
                v = D[i - 1, j - 2] + C[i, j - 1] + C[i, j]
                if v < best:
                    best, d, bl, bs = v, v, L[i - 1, j - 2] + 2, S[i - 1, j - 2]
            if i > 1 and L[i - 2, j - 1] > 0:
                v = D[i - 2, j - 1] + C[i - 1, j] + C[i, j]
                if v < best:
                    best, d, bl, bs = v, v, L[i - 2, j - 1] + 2, S[i - 2, j - 1]
            if best < np.inf:
                D[i, j] = d
                L[i, j] = bl
                S[i, j] = bs
    cost = np.full(m, np.inf)
    for j in range(m):
        if L[n - 1, j] > 0:
            cost[j] = D[n - 1, j] / L[n - 1, j]
    return cost, S[n - 1]


@njit(cache=True)
def _globaldtw(C):
    n, m = C.shape
    D = np.full((n, m), np.inf)
    L = np.zeros((n, m))
    D[0, 0] = C[0, 0]
    L[0, 0] = 1
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            best = np.inf
            bl = 0.0
            if i > 0 and j > 0 and L[i - 1, j - 1] > 0:
                v = D[i - 1, j - 1] + C[i, j]
                if v < best:
                    best, bl = v, L[i - 1, j - 1] + 1
            if i > 0 and j > 1 and L[i - 1, j - 2] > 0:
                v = D[i - 1, j - 2] + C[i, j - 1] + C[i, j]
                if v < best:
                    best, bl = v, L[i - 1, j - 2] + 2
            if i > 1 and j > 0 and L[i - 2, j - 1] > 0:
                v = D[i - 2, j - 1] + C[i - 1, j] + C[i, j]
                if v < best:
                    best, bl = v, L[i - 2, j - 1] + 2
            if best < np.inf:
                D[i, j] = best
                L[i, j] = bl
    return D[n - 1, m - 1] / L[n - 1, m - 1] if L[n - 1, m - 1] > 0 else np.inf


def whole_cost(ref: np.ndarray, tgt: np.ndarray) -> float:
    """Mean cost of matching the whole reference to the whole target, first frame to first and
    last to last (same steps as the search); inf where the lengths differ more than 2:1."""
    if len(ref) == 0 or len(tgt) == 0:
        return float("inf")
    return float(_globaldtw(1.0 - ref @ tgt.T))


def match_cost(ref: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per target end frame: mean cost of the best match of the whole reference, and its first frame."""
    if len(ref) == 0 or len(tgt) == 0:
        return np.full(len(tgt), np.inf), np.zeros(len(tgt), np.int64)
    return _subdtw(1.0 - ref @ tgt.T)


def occurrences(ref: np.ndarray, tgt: np.ndarray, max_cost: float = MAX_COST) -> list[tuple[float, int, int]]:
    """Stretches of tgt that match ref: (mean cost, first frame, last frame), best first. One per
    local minimum of the cost over end frames (within LOCAL_MIN frames), at most MAX_CANDIDATES;
    they may overlap (the choice in play order keeps one per piece)."""
    cost, start = match_cost(ref, tgt)
    out = []
    # best first; of equal cost the earlier end, so a stretch ending on a chord the next
    # piece begins with is not lost to one that takes that chord too
    for j in np.lexsort((np.arange(len(cost)), cost))[:4 * MAX_CANDIDATES]:
        if cost[j] > max_cost:
            break
        if cost[j] <= cost[max(0, j - LOCAL_MIN):j + LOCAL_MIN + 1].min() and                 not any(a == start[j] for _, a, _ in out):
            out.append((float(cost[j]), int(start[j]), int(j)))
    return out[:MAX_CANDIDATES]


def reference(performances: list[np.ndarray]) -> np.ndarray | None:
    """The most typical of several performances of one piece (frame features): among those whose
    frame count is within REF_COUNT_TOL of the median, the one with the least summed match cost."""
    perf = [f for f in performances if len(f) >= 2]
    if not perf:
        return None
    med = float(np.median([len(f) for f in perf]))
    pool = [f for f in perf if abs(len(f) - med) <= REF_COUNT_TOL * med][:MEDOID_MAX] or perf[:MEDOID_MAX]
    if len(pool) == 1:
        return pool[0]
    total = [sum(float(match_cost(a, b)[0].min()) for b in pool if b is not a) for a in pool]
    return pool[int(np.argmin(total))]


def place_in_order(notes: list[tuple[float, float, int]], refs: list[np.ndarray | None],
                   log_starts: list[float | None], breaks: list[float] = ()) -> list[tuple[float, float, float] | None]:
    """Place pieces (in play order) in a trial: per piece (start, end, mean cost) or None.
    The end is the last frame's note-off, or the next note's onset if a key is held past it.
    Several trials may be placed together (notes on one clock): no piece spans a time in `breaks`."""
    T, ts, te = onset_frames(notes)
    brk = np.asarray(sorted(breaks), float)
    seg = np.searchsorted(brk, ts)          # frame -> trial index
    cands = []
    for ref, t0 in zip(refs, log_starts):
        occ = occurrences(ref, T) if ref is not None and len(T) else []
        occ = [(c, a, j) for c, a, j in occ if seg[a] == seg[j]]
        cands.append([(c + (LOG_W * abs(ts[a] - t0) if t0 is not None else 0.0), c, a, j) for c, a, j in occ])
    states = [(0.0, -1, [])]        # (total, last frame used, picks)
    for cs in cands:
        nxt = []
        for tot, last, picks in states:
            nxt.append((tot + SKIP_COST, last, picks + [None]))
            for v, c, a, j in cs:
                if a >= last:              # a piece may begin on the chord the previous one ends on
                    nxt.append((tot + v, j, picks + [(c, a, j)]))
        keep: dict[int, tuple] = {}
        for s in nxt:
            if s[1] not in keep or s[0] < keep[s[1]][0]:
                keep[s[1]] = s
        states = sorted(keep.values(), key=lambda s: s[0])[:300]
    picks = min(states, key=lambda s: s[0])[2]
    # a chord shared by two neighbours (a scale ending on the C an arpeggio starts on) is the later one's
    prev = None
    for k, pk in enumerate(picks):
        if pk is None:
            continue
        if prev is not None and picks[prev][2] >= pk[1]:
            c0, a0, j0 = picks[prev]
            picks[prev] = (c0, a0, max(a0, pk[1] - 1))
        prev = k

    def bounds(k: int, a: int) -> tuple[int, int]:
        lo = max(max((q[2] for q in picks[:k] if q), default=-1) + 1, int(np.searchsorted(seg, seg[a])))
        hi = min(min((q[1] for q in picks[k + 1:] if q), default=len(ts)) - 1,
                 int(np.searchsorted(seg, seg[a], "right")) - 1)
        return lo, hi

    def pace(a: int, j: int) -> tuple[float, float]:
        ioi = np.diff(ts[a:j + 1])
        med = float(np.median(ioi)) if len(ioi) else 1.0
        # runs on over gaps up to GAP_X times the longest inside the piece, that longest
        # counted at most GAP_CAP_X median gaps (one long pause must not open the edges)
        return med, GAP_X * min(float(np.max(ioi)) if len(ioi) else 1.0, GAP_CAP_X * med)

    # pass 1, every piece: drop what does not belong at its edges, so that the neighbours'
    # bounds in pass 2 are the pieces' own
    for k, pk in enumerate(picks):
        if pk is None:
            continue
        c, a, j = pk
        R = refs[k]
        med, _ = pace(a, j)
        # a first / last frame repeating its neighbour's chord after a long pause is not the
        # piece's (a scale's closing C, and the C another piece starts on seconds later)
        # (after any pause longer than TRIM_X median gaps where the score's edge does not repeat)
        same = lambda u, v: float(T[u] @ T[v]) >= SAME_CHORD      # noqa: E731
        far = lambda e: TRIM_X * med if len(R) < 2 or float(R[e] @ R[e + (1 if e == 0 else -1)]) < SAME_CHORD             else max(TRIM_X * med, TRIM_MIN_S)                     # noqa: E731
        if j > a + 1 and ts[j] - ts[j - 1] > far(-1) and same(j, j - 1):
            j -= 1
        if j > a + 1 and ts[a + 1] - ts[a] > far(0) and same(a, a + 1):
            a += 1
        # after a long pause inside the piece: a start broken off and played again (the playing
        # from there on matches as well alone), or something else before it (not this piece)
        gaps = np.diff(ts[a:j + 1])
        if len(gaps):
            g = int(np.argmax(gaps))
            if gaps[g] > max(RESTART_GAP_S, RESTART_GAP_X * med) and j - (a + g + 1) >= 2:
                lead = T[a:a + g + 1]
                lead_cost = float(match_cost(lead, R)[0].min()) if len(R) >= len(lead) / 2 else np.inf
                restart = whole_cost(R, T[a + g + 1:j + 1]) <= whole_cost(R, T[a:j + 1]) + RESTART_TOL
                if restart or lead_cost > FOREIGN_COST:
                    a = a + g + 1
        # and the same at the end: playing after a long pause that the piece is as good without
        gaps = np.diff(ts[a:j + 1])
        if len(gaps):
            g = int(np.argmax(gaps))
            if gaps[g] > max(RESTART_GAP_S, RESTART_GAP_X * med) and g >= 2                     and whole_cost(R, T[a:a + g + 1]) <= whole_cost(R, T[a:j + 1]) + RESTART_TOL:
                j = a + g
        picks[k] = (c, a, j)

    # pass 2: run on over the piece's repeated edge chords, then end on the last release
    out = []
    for k, pk in enumerate(picks):
        if pk is None:
            out.append(None)
            continue
        c, a, j = pk
        R = refs[k]
        lo, hi = bounds(k, a)
        _, gap = pace(a, j)

        def fits(q: int, edge: np.ndarray) -> bool:
            # close to an edge chord, or some of its notes only (a fast repeat drops notes)
            if float(np.min(1.0 - edge @ T[q])) < EDGE_COST:
                return True
            own = T[q] >= 0.6 * T[q].max()
            return any(bool(np.all(e[own] > 0)) for e in edge)

        def step(i: int, d: int, lim: int, edge: np.ndarray) -> int:
            """Next frame the piece runs on to (one frame that does not fit may be passed over), or i."""
            for n in (1, 2):
                q = i + d * n
                if (q > lim if d > 0 else q < lim) or abs(ts[q] - ts[q - d]) > gap:
                    return i
                if fits(q, edge):
                    return q
            return i

        while (na := step(a, -1, lo, R[:EDGE_FRAMES])) != a:
            a = na
        while (nj := step(j, 1, hi, R[-EDGE_FRAMES:])) != j:
            j = nj
        picks[k] = (c, a, j)
        # the piece ends when its last chord, with any note struck within TAIL_JOIN_S after it
        # (a late note of that chord), is released -- or when the next note is struck
        e = j
        while e + 1 <= hi and seg[e + 1] == seg[j] and ts[e + 1] - ts[j] <= TAIL_JOIN_S:
            e += 1
        end = float(np.max(te[j:e + 1]))
        if e + 1 < len(ts) and seg[e + 1] == seg[j]:         # up to the next note struck
            end = min(end, float(ts[e + 1]))
        out.append((round(float(ts[a]), 3), round(float(end), 3), round(c, 3)))
    return out


PAD_CLEAR_S = 0.02


def silent_bounds(onsets: list[float], start: float, end: float) -> tuple[float, float]:
    """How far a piece's padding may reach: up to (but not into) the nearest note onset
    before its start and after its end -- a window's padding holds no other notes."""
    import bisect
    i = bisect.bisect_left(onsets, start - 1e-3)
    k = bisect.bisect_right(onsets, end)
    lo = onsets[i - 1] + PAD_CLEAR_S if i > 0 else float("-inf")
    hi = onsets[k] - PAD_CLEAR_S if k < len(onsets) else float("inf")
    return lo, hi


SUSTAIN_CC = 64


def sustain_spans(midi_path) -> list[tuple[float, float]]:
    """(down, up) times of the sustain pedal in a MIDI file (value >= 64 = down)."""
    import pretty_midi
    events = sorted((c.time, c.value >= 64) for i in pretty_midi.PrettyMIDI(str(midi_path)).instruments
                    for c in i.control_changes if c.number == SUSTAIN_CC)
    spans, down = [], None
    for t, is_down in events:
        if is_down and down is None:
            down = t
        elif not is_down and down is not None:
            spans.append((down, t))
            down = None
    if down is not None:
        spans.append((down, float("inf")))
    return spans


def sounding_end(end: float, spans: list[tuple[float, float]], onsets: list[float], limit: float) -> float:
    """When a piece whose last note is released at `end` stops sounding: at the pedal's
    release if the sustain pedal holds it, but never at or past the next note struck,
    nor past `limit` (the end of the recording)."""
    up = next((b for a, b in spans if a <= end < b), end)
    return max(end, min(up, silent_bounds(onsets, end, end)[1], limit))
