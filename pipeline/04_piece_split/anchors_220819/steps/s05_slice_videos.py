"""
Per-piece video for 220819.

220819 has no shared timecode, so each camera is put on the trial's audio
clock by its own audio track (video_t = audio_t + offset_sec). Audio alone is
ambiguous: a session holds several takes of the same pieces, so a trial's
audio also matches the other takes. The match is therefore constrained by
time:

  * trial wall-clock start: Qualisys TIME_STAMP of the trial's body take
    plus its body offset;
  * takes never overlap. Trial names do not give the recording order (a
    retake such as P1_t3 can come after P2), so the order is taken from
    wall-clock times, which the continuous views supply for every trial.

Two kinds of sources:

  continuous (Top, Right): the whole session, aligned first. A view may be
    split over several files with gaps between them (C0_N1: two Top files,
    two Right files plus a 12 s test clip, which is skipped), so every file
    gets its own video time of wall-clock 0 (base), voted by the wall-timed
    trials that match best in that file. A file holding none of them takes
    its base from trials timed by the Face recording clock: the Face
    recording that clearly best contains the trial (it must cover most of
    the trial; short clips match inside any long trial) gives its creation
    time plus the offset. A trial without a wall time gets the one its candidates
    agree on across the continuous views (offset - base), not overlapping
    other trials; then every trial takes its candidate at wall + base, or is
    `not_in_source` when that falls outside the recording (a camera stopped
    early).
  per take (Face GoPro GX, Left GoPro GH): one recording per take, long ones
    split into chapters (GX01xxxx, GX02xxxx = chapters of recording xxxx).
    Chapters are joined. Each recording gets a wall-clock start: its creation
    time (Face), or that of the Face recording of the same length in the
    same order (Left GoPros carry none). A trial takes the recording that
    starts within WALL_TOL_S of it, offset searched near the wall-clock
    estimate. A recording may start after or stop before the trial audio.
    The 2025 re-edited `220819_*.mp4` compilations in Left are not used
    (their take order is not the recording order).

Coarse offsets come from 50 Hz log-envelope NCC, refined at 1 kHz within
+-80 ms. Every piece window of `locate_manifest.json` is then cut from each
aligned view and re-encoded (frame-accurate start):

    Data/Pipeline/04_piece_split/220819/videos/{piece_stem}_{view}.mp4

Alignment per (trial, view): `anchors_220819/video_alignment.json`.

Usage (from the repository root):
  python pipeline/04_piece_split/anchors_220819/steps/s05_slice_videos.py --sessions C0_N2 C1_N1 --align-only
  python pipeline/04_piece_split/anchors_220819/steps/s05_slice_videos.py --sessions C0_N2 C1_N1
  python pipeline/04_piece_split/anchors_220819/steps/s05_slice_videos.py --sessions C0_N2 --views Top --force
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import wave
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.signal import correlate

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # pipeline/
from _common import video_xcorr as vx  # noqa: E402

DATASET = "220819"
RAW = Path("Data/Raw") / DATASET
NORM = Path("Data/Pipeline/01_ingest/Normalized") / DATASET
PIECES = Path("Data/Pipeline/04_piece_split")
ANCHORS = PIECES / "anchors_220819"
OUT = PIECES / DATASET / "videos"
ALIGN_JSON = ANCHORS / "video_alignment.json"

VIEWS = {"Top": "continuous", "Right": "continuous", "Face": "per_take", "Left": "per_take"}
FINE_SR = 1000          # Hz, fine envelope rate
FINE_SEARCH_S = 0.08    # fine search around the coarse offset
TOP_K = 6               # candidate peaks per trial in a continuous source
PEAK_SEP_S = 20.0
WALL_TOL_S = 20.0       # allowed disagreement with the wall-clock timing
MIN_NCC = 0.3
STRONG_NCC = 0.6      # required when a recording is chosen by order alone
TAKE_MARGIN = 0.1     # a trial's best per-take recording must beat the next by this
TAKE_COVER = 0.8      # a per-take recording covers at least this share of its trial
TAKE_EDGE_S = 30.0    # start/end slack searched around a per-take recording
MIN_SOURCE_S = 60.0   # continuous-view files shorter than this are test clips
WORKERS = 3


# ── time of each trial ──

def tsv_timestamp(path: Path) -> datetime | None:
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("TIME_STAMP"):
                return datetime.strptime(line.split("\t")[1].strip(), "%Y-%m-%d, %H:%M:%S.%f")
    return None


def trial_times(sess: str) -> tuple[list[str], dict[str, float], dict[str, float]]:
    """Trials in recording order, wall-clock start (s since midnight) where known, durations."""
    c, n = sess.split("_")
    mp = json.loads((NORM / sess / "_mapping.json").read_text(encoding="utf-8"))
    al = {e["stage1_stem"]: e for e in json.loads((ANCHORS / "alignment.json").read_text(encoding="utf-8"))["entries"]}
    order, wall, dur = [], {}, {}
    for t in mp["trials"]:
        stem = t["trial_stem"]
        wav = NORM / sess / "audio" / f"{stem}.wav"
        if not wav.exists():
            continue
        order.append(stem)
        with wave.open(str(wav)) as w:
            dur[stem] = w.getnframes() / w.getframerate()
        body = t.get("raw_sources", {}).get("motion_body")
        shift = (al.get(stem, {}).get("body") or {}).get("align_shift_sec")
        if body and shift is not None:
            ts = tsv_timestamp(RAW / c / n / "Motion" / "Body" / body)
            if ts:
                # body_t = audio_t + shift, body_t = 0 at TIME_STAMP
                wall[stem] = ts.hour * 3600 + ts.minute * 60 + ts.second + ts.microsecond / 1e6 + shift
    return order, wall, dur


# ── NCC ──

def _ncc_valid(sig: np.ndarray, tpl: np.ndarray) -> np.ndarray:
    n = len(tpl)
    t0 = tpl - tpl.mean()
    xc = correlate(sig, t0, mode="valid")
    cs, cs2 = np.concatenate(([0], np.cumsum(sig))), np.concatenate(([0], np.cumsum(sig * sig)))
    k = np.arange(len(xc))
    var = (cs2[k + n] - cs2[k]) - (cs[k + n] - cs[k]) ** 2 / n
    return xc / (np.sqrt((t0 * t0).sum()) * np.sqrt(np.maximum(var, 1e-9)))


def ncc_candidates(trial_wav: Path, src_wav: Path, k: int) -> list[tuple[float, float]]:
    """Top-k (offset_sec, ncc), video_t = audio_t + offset. When the source is
    shorter than the trial (camera started late / stopped early), the source
    is searched inside the trial instead (offset <= 0)."""
    tx, tsr = vx._read_wav_mono(trial_wav)
    sx, ssr = vx._read_wav_mono(src_wav)
    t_env, s_env = vx._envelope(tx, tsr), vx._envelope(sx, ssr)
    if len(s_env) < 10 * vx.ENV_SR:
        return []
    if len(t_env) < len(s_env):
        return peaks(_ncc_valid(s_env, t_env), k)
    return [(-o, c) for o, c in peaks(_ncc_valid(t_env, s_env), k)]


def peaks(ncc: np.ndarray, k: int) -> list[tuple[float, float]]:
    """Top-k (offset_sec, ncc) peaks at least PEAK_SEP_S apart."""
    sep = int(PEAK_SEP_S * vx.ENV_SR)
    order = np.argsort(ncc)[::-1]
    out: list[int] = []
    for i in order:
        if all(abs(i - j) >= sep for j in out):
            out.append(int(i))
            if len(out) == k:
                break
    return [(i / vx.ENV_SR, float(ncc[i])) for i in out]


def lag_ncc(trial_wav: Path, src_wav: Path, center: float, half: float, sr: int = vx.ENV_SR,
            min_overlap_s: float = 20.0) -> list[tuple[float, float]]:
    """Pearson correlation over the overlap for every offset in center +- half
    (video_t = audio_t + offset); works whatever the two lengths are.
    Returns [(offset_sec, ncc)] sorted by ncc, best first."""
    tx, tsr = vx._read_wav_mono(trial_wav)
    sx, ssr = vx._read_wav_mono(src_wav)
    t, v = vx._envelope(tx, tsr, sr), vx._envelope(sx, ssr, sr)
    out = []
    for k in range(int(round((center - half) * sr)), int(round((center + half) * sr)) + 1):
        a0, a1 = max(0, -k), min(len(t), len(v) - k)        # trial samples overlapping the source
        if a1 - a0 < min_overlap_s * sr:
            continue
        x, y = t[a0:a1], v[a0 + k:a1 + k]
        sx_, sy_ = x.std(), y.std()
        if sx_ == 0 or sy_ == 0:
            continue
        out.append((k / sr, float(((x - x.mean()) * (y - y.mean())).mean() / (sx_ * sy_))))
    return sorted(out, key=lambda z: -z[1])


def fine_offset(trial_wav: Path, src_wav: Path, coarse: float) -> tuple[float, float]:
    """Refine `coarse` at 1 kHz within +-FINE_SEARCH_S; returns (offset_sec, ncc)."""
    tx, tsr = vx._read_wav_mono(trial_wav)
    sx, ssr = vx._read_wav_mono(src_wav)
    t_env = vx._envelope(tx, tsr, FINE_SR)
    s_env = vx._envelope(sx, ssr, FINE_SR)
    pad = int(FINE_SEARCH_S * FINE_SR)
    o = int(round(coarse * FINE_SR))
    # overlap of the two signals at the coarse offset, as a template + search span
    a0, a1 = max(0, -o), min(len(t_env), len(s_env) - o)     # trial samples that overlap
    if a1 - a0 < 5 * FINE_SR:
        return coarse, float("nan")
    tpl = t_env[a0:a1]
    b0 = a0 + o - pad
    if b0 < 0 or b0 + len(tpl) + 2 * pad > len(s_env):
        return coarse, float("nan")
    ncc = _ncc_valid(s_env[b0:b0 + len(tpl) + 2 * pad], tpl)
    j = int(np.argmax(ncc))
    return round((b0 + j - a0) / FINE_SR, 4), round(float(ncc[j]), 4)


# ── sources ──

def recordings(sess: str, view: str) -> list[dict]:
    """Source recordings of a view: {files, audio, start_wall (s since midnight) | None}."""
    c, n = sess.split("_")
    d = RAW / c / n / "Video" / view
    files = sorted(p for p in d.glob("*") if p.suffix.lower() == ".mp4" and "_4-3" not in p.stem)
    if VIEWS[view] == "continuous":
        return [{"files": [f], "audio": vx.extract_audio_mono(f), "start_wall": None} for f in files]
    files = [f for f in files if not f.stem.startswith(f"{DATASET}_")]   # re-edited compilations
    groups: dict[str, list[Path]] = {}
    for f in files:                         # GoPro: GX + chapter (2 digits) + recording (4 digits)
        groups.setdefault(f.stem[4:], []).append(f)
    out = []
    for rec in sorted(groups):
        chs = sorted(groups[rec], key=lambda p: p.stem[2:4])
        audio = joined_audio(chs)
        out.append({"files": chs, "audio": audio, "start_wall": creation_wall(chs[0])})
    return out


def joined_audio(chapters: list[Path]) -> Path:
    if len(chapters) == 1:
        return vx.extract_audio_mono(chapters[0])
    parts = [vx.extract_audio_mono(c) for c in chapters]
    out = parts[0].with_name(f"join_{'_'.join(c.stem for c in chapters)}.wav")
    if not out.exists():
        data, sr = [], None
        for p in parts:
            x, sr = vx._read_wav_mono(p)
            data.append(x)
        x = np.concatenate(data)
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    return out


def creation_wall(f: Path) -> float | None:
    """GoPro creation_time (camera clock = local time) in s since midnight."""
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time",
                        "-of", "csv=p=0", str(f)], capture_output=True, text=True)
    t = r.stdout.strip()
    if not t:
        return None
    d = datetime.fromisoformat(t.replace("Z", ""))
    return d.hour * 3600 + d.minute * 60 + d.second + d.microsecond / 1e6


# ── assignment ──

def view_base(cands: dict, wall: dict) -> float | None:
    """Video time of wall-clock 0: the value most wall-timed trials' candidates agree on."""
    votes = [(o - wall[s], c, s) for s in cands if s in wall for o, c in cands[s]]
    base, best = None, -1.0
    for b, _, _ in votes:
        sc: dict[str, float] = {}
        for b2, c2, s2 in votes:
            if abs(b2 - b) <= WALL_TOL_S:
                sc[s2] = max(sc.get(s2, 0.0), c2)
        if sum(sc.values()) > best:
            base, best = b, sum(sc.values())
    return base


def file_bases(cands_per_file: list[dict], wall: dict) -> list[float | None]:
    """Base of each file of a split continuous view. A trial is in at most one
    file, so a wall-timed trial votes only in the file where it matches best."""
    best = lambda cs: max((c for _, c in cs), default=-1.0)
    home = {s: max(range(len(cands_per_file)), key=lambda i: best(cands_per_file[i].get(s, [])))
            for s in wall}
    return [view_base({s: cs for s, cs in cands.items() if home.get(s) == i}, wall)
            for i, cands in enumerate(cands_per_file)]


def place_continuous(rec, cands, base, wall, dur, wavs) -> dict:
    src_len = vx._read_wav_mono(rec["audio"])[0].size / vx.CLIP_SR
    out = {}
    for s, cs in cands.items():
        if s not in wall or base is None:
            out[s] = {"status": "unresolved"}
            continue
        exp = wall[s] + base
        ok = [(o, c) for o, c in cs if abs(o - exp) <= WALL_TOL_S and c >= MIN_NCC]
        if not ok:
            inside = 0 <= exp and exp + dur[s] <= src_len
            out[s] = ({"status": "unresolved", "expected_offset_sec": round(exp, 1)} if inside else
                      {"status": "not_in_source", "expected_offset_sec": round(exp, 1), "source_sec": round(src_len, 1)})
            continue
        o, c = max(ok, key=lambda x: x[1])
        fo, fn = fine_offset(wavs[s], rec["audio"], o)
        out[s] = {"files": [str(f.relative_to(RAW)).replace("\\", "/") for f in rec["files"]],
                  "offset_sec": fo, "coarse_offset_sec": round(o, 3), "ncc": round(c, 4), "fine_ncc": fn,
                  "runner_up_ncc": round(max([x[1] for x in cs if abs(x[0] - o) > 1] or [0]), 4),
                  "status": "ok"}
    return out


def infer_walls(order, wall, dur, views) -> dict[str, float]:
    """Wall time of trials without one: candidates of all continuous views,
    mapped to wall time (offset - base), clustered; the cluster with the
    largest summed NCC that overlaps no trial with a known wall time wins."""
    out = dict(wall)
    for s in order:
        if s in out:
            continue
        pts = [(o - base, c) for cands, base in views if base is not None for o, c in cands[s] if c >= MIN_NCC]
        best, score = None, 0.0
        for w, _ in pts:
            if any(w < out[q] + dur[q] - 1 and out[q] < w + dur[s] - 1 for q in wall):
                continue
            sc = sum(c for w2, c in pts if abs(w2 - w) <= 2.0)
            if sc > score:
                best, score = w, sc
        if best is not None:
            out[s] = best
    return out


def take_walls(recs, todo, known, dur, wavs) -> dict[str, float]:
    """Wall time of trials that still have none, from a per-take view's
    recording clock. A per-take camera records the whole take, so only
    recordings covering at least TAKE_COVER of the trial are considered
    (short clips match somewhere inside any long trial), searched over every
    offset where the two overlap that much, plus TAKE_EDGE_S of slack. The
    recording that matches clearly best (NCC >= STRONG_NCC, the next one at
    least TAKE_MARGIN lower) gives wall = recording start + offset. A time
    that overlaps a trial with a known wall time is not used."""
    out: dict[str, float] = {}
    for s in todo:
        scores = []
        for r, rec in enumerate(recs):
            L = rec_seconds(rec)
            if rec["start_wall"] is None or L < TAKE_COVER * dur[s]:
                continue
            need = TAKE_COVER * min(L, dur[s])
            cs = lag_ncc(wavs[s], rec["audio"], (L - dur[s]) / 2,
                         abs(L - dur[s]) / 2 + TAKE_EDGE_S, min_overlap_s=need)
            if cs:
                scores.append((cs[0][1], cs[0][0], r))
        scores.sort(reverse=True)
        if not scores or scores[0][0] < STRONG_NCC:
            continue
        if len(scores) > 1 and scores[0][0] - scores[1][0] < TAKE_MARGIN:
            continue
        c, o, r = scores[0]
        w = recs[r]["start_wall"] + o           # audio_t = 0 at video_t = offset
        taken = {**known, **out}
        if any(w < taken[q] + dur[q] - 1 and taken[q] < w + dur[s] - 1 for q in taken):
            continue
        out[s] = w
    return out


def merge_sources(results: list[dict]) -> dict:
    """One result per trial over the files of a continuous view: a match
    (highest NCC) over an in-source miss over not-in-source."""
    rank = {"ok": 0, "unresolved": 1, "not_in_source": 2}
    out: dict[str, dict] = {}
    for r in results:
        for s, v in r.items():
            cur = out.get(s)
            if cur is None or (rank[v["status"]], -v.get("ncc", 0)) < (rank[cur["status"]], -cur.get("ncc", 0)):
                out[s] = v
    return out


def rec_seconds(rec) -> float:
    return vx._read_wav_mono(rec["audio"])[0].size / vx.CLIP_SR


def borrow_start_walls(recs: list[dict], ref: list[dict]) -> None:
    """Recordings without a creation time take that of the reference view's
    recording of about the same length (within 3 s), matched in order."""
    j = 0
    ref_len = [rec_seconds(r) for r in ref]
    for rec in recs:
        if rec["start_wall"] is not None:
            continue
        L = rec_seconds(rec)
        for k in range(j, len(ref)):
            if abs(ref_len[k] - L) <= 3.0 and ref[k]["start_wall"] is not None:
                rec["start_wall"], rec["wall_from"] = ref[k]["start_wall"], ref[k]["files"][0].name
                j = k + 1
                break


def align_per_take(recs, order, wall, dur, wavs) -> dict:
    """Each trial -> the recording starting within WALL_TOL_S of it; offset near the wall estimate."""
    out = {}
    used: set[int] = set()
    for s in order:
        if s not in wall:
            out[s] = {"status": "unresolved", "reason": "no wall-clock time"}
            continue
        best = None
        for r, rec in enumerate(recs):
            if r in used or rec["start_wall"] is None:
                continue
            # video_t = audio_t + offset; the recording starts (video_t = 0) at its
            # wall time, the trial audio (audio_t = 0) at the trial's.
            exp = wall[s] - rec["start_wall"]
            if not (-dur[s] - WALL_TOL_S < exp < rec_seconds(rec) + WALL_TOL_S):
                continue                               # no time overlap with this recording
            cs = lag_ncc(wavs[s], rec["audio"], exp, WALL_TOL_S, min_overlap_s=min(20.0, 0.8 * dur[s]))
            if cs and (best is None or cs[0][1] > best[2]):
                best = (r, cs[0][0], cs[0][1])
        if best is None or best[2] < MIN_NCC:
            out[s] = {"status": "unresolved", "best_ncc": None if best is None else round(best[2], 4)}
            continue
        r, o, c = best
        used.add(r)
        f = lag_ncc(wavs[s], recs[r]["audio"], o, FINE_SEARCH_S, sr=FINE_SR, min_overlap_s=min(20.0, 0.8 * dur[s]))
        fo, fn = (round(f[0][0], 4), round(f[0][1], 4)) if f else (o, float("nan"))
        out[s] = {"files": [str(f.relative_to(RAW)).replace("\\", "/") for f in recs[r]["files"]],
                  "offset_sec": fo, "coarse_offset_sec": round(o, 3), "ncc": round(float(c), 4), "fine_ncc": fn,
                  "rec": r, "recording_wall_from": recs[r].get("wall_from", "creation_time"), "status": "ok"}
    # Second pass for trials left over: a recording without a wall time
    # (length differs from every reference recording, e.g. a missing chapter)
    # that lies, in recording number, between the recordings of the trial's
    # chronological neighbours; searched over every overlap.
    for i, s in enumerate(order):
        if out[s]["status"] == "ok":
            continue
        lo = max([out[q]["rec"] for q in order[:i] if out[q]["status"] == "ok"], default=-1)
        hi = min([out[q]["rec"] for q in order[i + 1:] if out[q]["status"] == "ok"], default=len(recs))
        best = None
        for r in range(lo + 1, hi):
            if r in used or recs[r]["start_wall"] is not None:
                continue
            L = rec_seconds(recs[r])
            cs = lag_ncc(wavs[s], recs[r]["audio"], (L - dur[s]) / 2, (L + dur[s]) / 2)
            if cs and (best is None or cs[0][1] > best[2]):
                best = (r, cs[0][0], cs[0][1])
        if best is None or best[2] < STRONG_NCC:
            continue
        r, o, c = best
        used.add(r)
        f = lag_ncc(wavs[s], recs[r]["audio"], o, FINE_SEARCH_S, sr=FINE_SR)
        fo, fn = (round(f[0][0], 4), round(f[0][1], 4)) if f else (o, float("nan"))
        out[s] = {"files": [str(f.relative_to(RAW)).replace("\\", "/") for f in recs[r]["files"]],
                  "offset_sec": fo, "coarse_offset_sec": round(o, 3), "ncc": round(float(c), 4), "fine_ncc": fn,
                  "rec": r, "recording_wall_from": "order", "status": "ok"}
    return out


def align_session(sess: str) -> dict:
    order, wall, dur = trial_times(sess)
    wavs = {s: NORM / sess / "audio" / f"{s}.wav" for s in order}
    print(f"  {sess}: {len(order)} trials, wall-clock from body for {len(wall)}")
    res: dict[str, dict] = {s: {} for s in order}

    def show(view, r):
        for s in sorted(r, key=lambda s: walls.get(s, 1e9)):
            v = r[s]
            res[s][view] = v
            src = v.get("files", ["-"])[0].rsplit("/", 1)[-1]
            print(f"    {s[-8:]} {view:5s} {v['status']:13s} {src:16s} off={v.get('offset_sec', v.get('expected_offset_sec', '-'))!s:>9} "
                  f"ncc={v.get('ncc', v.get('best_ncc', '-'))} 2nd={v.get('runner_up_ncc', '-')}")

    # Continuous views may be split over several files (camera file-size
    # limit, restarts) with gaps between them, so every file gets its own
    # base (video time of wall-clock 0) from the trials it holds.
    cont = {}
    for view in [v for v, k in VIEWS.items() if k == "continuous"]:
        recs = [r for r in recordings(sess, view) if rec_seconds(r) >= MIN_SOURCE_S]
        cands = [{s: ncc_candidates(wavs[s], r["audio"], TOP_K) for s in order} for r in recs]
        for i, (rec, cs, base) in enumerate(zip(recs, cands, file_bases(cands, wall))):
            cont[(view, i)] = (rec, cs, base)
    walls = infer_walls(order, wall, dur, [(c, b) for _, c, b in cont.values()])
    takes = {v: recordings(sess, v) for v, k in VIEWS.items() if k == "per_take"}
    ref = takes.get("Face") or []
    clock = dict(wall)                          # wall times from a clock, not from video matches
    for s, w in take_walls(ref, [s for s in order if s not in walls], walls, dur, wavs).items():
        walls[s] = clock[s] = w
        print(f"    {s[-8:]} wall-clock from the Face recording clock: {w:.2f}")
    # Files none of the body-timed trials is in take their base from
    # clock-timed trials only: trials placed through another file's video
    # match would vote with chance matches of short trials.
    for view in dict.fromkeys(v for v, _ in cont):
        keys = [k for k in cont if k[0] == view]
        bases = file_bases([cont[k][1] for k in keys], clock)
        for k, b in zip(keys, bases):
            if cont[k][2] is None:
                cont[k] = (cont[k][0], cont[k][1], b)
    for s in order:
        if s not in wall:
            print(f"    {s[-8:]} wall-clock inferred from video: {walls.get(s)}")
    for view in dict.fromkeys(v for v, _ in cont):
        show(view, merge_sources([place_continuous(rec, cands, base, walls, dur, wavs)
                                  for (v, _), (rec, cands, base) in cont.items() if v == view]))

    for view, recs in takes.items():
        if view != "Face":
            borrow_start_walls(recs, ref)
        if recs:
            show(view, align_per_take(recs, sorted(order, key=lambda s: walls.get(s, 1e9)), walls, dur, wavs))
    for s in order:
        res[s]["_wall_clock_start_s"] = round(walls[s], 2) if s in walls else None
    return res


def video_seconds(f: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(f)],
                       capture_output=True, text=True)
    return float(r.stdout.strip() or 0)


ENCODE = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
          "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2"]


def _encode(src: Path, start: float, dur: float, dst: Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{max(0.0, start):.3f}", "-i", str(src),
                    "-t", f"{dur:.3f}", "-map", "0:v:0", "-map", "0:a:0?", *ENCODE,
                    "-movflags", "+faststart", str(dst)], check=True, timeout=1800)


def cut(srcs: list[Path], lens: list[float], start: float, dur: float, dst: Path) -> bool:
    """Cut [start, start+dur] of a recording. GoPro chapters are cut separately
    (seeking into one file is fast; a joined stream would be decoded from its
    start) and the parts are joined without re-encoding."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    parts, c0 = [], 0.0
    for f, L in zip(srcs, lens):
        a, b = max(start, c0), min(start + dur, c0 + L)
        if b - a > 0.02:
            parts.append((f, a - c0, b - a))
        c0 += L
    tmp = dst.with_suffix(".part.mp4")
    try:
        if len(parts) == 1:
            _encode(*parts[0], tmp)
        else:
            pieces = []
            for i, (f, a, d) in enumerate(parts):
                q = dst.with_suffix(f".part{i}.mp4")
                _encode(f, a, d, q)
                pieces.append(q)
            lst = dst.with_suffix(".concat.txt")
            lst.write_text("".join(f"file '{q.resolve().as_posix()}'\n" for q in pieces), encoding="utf-8")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                            "-c", "copy", "-movflags", "+faststart", str(tmp)], check=True, timeout=600)
            for q in pieces:
                q.unlink(missing_ok=True)
            lst.unlink(missing_ok=True)
        tmp.replace(dst)                       # only finished files get the final name
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  failed {dst.name}: {e}")
        tmp.unlink(missing_ok=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--align-only", action="store_true")
    ap.add_argument("--realign", action="store_true", help="recompute offsets even if stored")
    ap.add_argument("--force", action="store_true", help="re-cut existing piece videos")
    ap.add_argument("--views", nargs="+", default=list(VIEWS), help="cut only these views (default: all)")
    a = ap.parse_args()

    store = json.loads(ALIGN_JSON.read_text(encoding="utf-8")) if ALIGN_JSON.exists() else {}
    for sess in a.sessions:
        trials = [t["trial_stem"] for t in json.loads((NORM / sess / "_mapping.json").read_text(encoding="utf-8"))["trials"]]
        if a.realign or not all(t in store for t in trials):
            store.update(align_session(sess))
            ALIGN_JSON.write_text(json.dumps(store, indent=1), encoding="utf-8")
    if a.align_only:
        return

    loc = json.loads((ANCHORS / "locate_manifest.json").read_text(encoding="utf-8"))
    jobs, outside = [], []
    lengths: dict[Path, float] = {}
    for e in loc["entries"]:
        if e.get("session") not in a.sessions or not e.get("ok") or not e.get("out_stem"):
            continue
        for view, al in store.get(e["stage1_stem"], {}).items():
            if view.startswith("_") or view not in a.views or al["status"] != "ok":
                continue
            dst = OUT / f"{e['out_stem']}_{view}.mp4"
            files = tuple(RAW / f for f in al["files"])
            for f in files:
                if f not in lengths:
                    lengths[f] = video_seconds(f)
            total = sum(lengths[f] for f in files)
            start, dur = al["offset_sec"] + e["t_start_sec"], e["t_end_sec"] - e["t_start_sec"]
            if start < -0.5 or start + dur > total + 0.5:
                outside.append(f"{e['out_stem']}_{view}")     # the camera was not recording then
                continue
            if dst.exists() and not a.force:
                continue
            jobs.append((list(files), [lengths[f] for f in files], start, dur, dst))
    if outside:
        print(f"{len(outside)} piece windows outside their camera recording (not cut): {outside[:12]}")
        (ANCHORS / "video_not_in_source.json").write_text(json.dumps(sorted(outside), indent=1), encoding="utf-8")
    print(f"cutting {len(jobs)} piece videos")
    with ThreadPoolExecutor(WORKERS) as ex:
        res = list(ex.map(lambda j: (j[4].name, cut(*j)), jobs))
    bad = [n for n, ok in res if not ok]
    print(f"done: {len(res) - len(bad)} ok, {len(bad)} failed {bad[:10]}")


if __name__ == "__main__":
    main()
