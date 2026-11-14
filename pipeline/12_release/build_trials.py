"""
Stage 12: release-ready trial (take) files -- every take of a released session,
whole, on one clock.

For every trial of a released session (datasets 220819, 221112, 240523, 240828;
the sessions of paper_builders/participant_map.py, never C0_N0) this step writes

    Data/Pipeline/12_release/trials/{ds}/{trial_stem}/audio.wav, midi.mid, hand.csv, body.csv

all on the trial clock: time 0 = the first sample of the trial audio, motion
rows = frames at 120 fps, row k at k / 120 s. It is the clock the Stage 04
piece cuts use, so a piece window [start_sec, end_sec] of the trial cuts the
piece files out of these files:

    audio   samples from int(int(start_sec * 1000) * rate / 1000) on (the
            millisecond cut of Stage 04)
    MIDI    the notes sounding in the window, shifted by -start_sec and
            clipped to it
    motion  rows [floor(start_sec * 120), floor(end_sec * 120)) -- or, for a
            piece whose motion Stage 04 cut at another frame, the rows listed
            for it in _trials.json (pieces.{piece}.hand / .body start_frame,
            end_frame; residual_frames is the difference; pieces.csv of the
            release carries them as hand/body_start_frame). In 220819 the cut
            rounds instead of flooring, so a piece there may sit one frame off.

Sources (the files the Stage 04 cuts were made from):
    220819  audio, MIDI: Stage 01 trial files; hand, body: Stage 02 Aligned
            {trial}.csv, motion time = audio time + the trial's hand
            align_shift_sec (anchors_220819/alignment.json)
    221112  audio, MIDI: Stage 01 trial files; hand, body: Stage 02 Aligned
            {trial}.csv (trial time = motion time)
    240523, 240828
            audio: Stage 04 s01 trial cut {trial}.wav; MIDI: Stage 03
            Aligned {raw}.mid; hand: Stage 02 Aligned {trial}.csv from the row
            whose TimeCode is the trial start (Stage 03 timecode_start_sec);
            body: Stage 02 Aligned {raw}.csv (frame 0 = trial time 0)

Motion offset per trial: the frame of each Stage 04 piece's motion in the
source is found by its values (the piece's fullest row is looked up row for row
in the source, exact to 1e-6), so the offset is the one the cut files carry,
whatever step cut them (s02, s03, s06, s08, s12, s14). A piece whose motion is
all empty is checked where its window puts it. The trial's offset is
the one most of its pieces use; a piece cut at another frame keeps its own rows
and its residual offset is recorded. A trial without pieces takes the offset
its source states (TimeCode start, align_shift_sec, 0). Motion before trial
time 0 is left out; a source that starts after it gets empty leading rows. The
TimeCode column is dropped, as in the piece files.

Speech muting: the audio is muted wherever the piano is not sounding by the
trial MIDI. Kept are [onset - 0.10 s, sounding end + 0.5 s] of every note,
where the sounding end is the note's release held on by the sustain pedal
(_common/score_follow.sounding_end: the pedal's release, never past the next
note struck nor the end of the recording); kept spans are merged, everything
else is muted, with a 20 ms linear fade inside the muted side of each edge (so
the kept samples are unchanged). The rule is exact, read from the MIDI. A trial
with audio but no MIDI cannot be muted by it: its audio is written as recorded,
with audio_muted = false and a note in _trials.json (takes.csv of the release).

MIDI: the notes and control changes of the source, written at the source's
resolution and tempo (120 BPM), so every note time is kept to the tick; other
meta events (track names, text) are not written.

Takes without audio and MIDI (221112 C1_N6 and C1_N10, the takes registered as
`audio_midi` in _common/known_absences.json) get hand.csv / body.csv only, on
the motion clock.

Outputs:
    Data/Pipeline/12_release/trials/{ds}/{trial}/...    the files above
    Data/Pipeline/12_release/trials/_trials.json         per trial: sources,
        offsets, durations, notes, muted share, and per Stage 04 piece its window
        and motion rows
    Data/Pipeline/12_release/trials/_verify.json         per piece: the check below

Verification (always run): every Stage 04 piece of a trial is cut out of the new
trial files at its window and compared with the piece files: the same MIDI notes
(pitch; onset and offset within 1 ms, or half a tick of the piece file where its
resolution is coarser), the same motion rows (count within one frame, values
within 1e-6), the same audio samples wherever the trial audio is not muted.
240523 is released per take: its trials are compared with the take files the
manifest lists as well.

Usage (from the repository root):
  python pipeline/12_release/build_trials.py [--dataset 240828 ...] [--trial STEM ...] [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import wave
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper_builders"))
sys.path.insert(0, str(ROOT / "pipeline"))
from participant_map import EXCLUDED_FROM_RELEASE, participant_id  # noqa: E402
from _common.known_absences import is_audio_midi_absent  # noqa: E402
from _common.score_follow import sounding_end, sustain_spans  # noqa: E402

PIPE = Path("Data/Pipeline")
NORM = PIPE / "01_ingest" / "Normalized"
HAND = PIPE / "02_spatial" / "hand" / "Aligned"
BODY = PIPE / "02_spatial" / "body" / "Aligned"
SYNC_MIDI = PIPE / "03_sync" / "midi_motion" / "Aligned"
PIECES = PIPE / "04_piece_split"
ANCHORS = PIECES / "anchors_220819"
QA = PIPE / "05_alignment_qa"
OUT = PIPE / "12_release" / "trials"

DATASETS = ["220819", "221112", "240523", "240828"]
FPS = 120
PRE_ONSET_S = 0.10
POST_SOUND_S = 0.5
FADE_S = 0.020
MIDI_TOL_S = 0.001
VALUE_TOL = 1e-6


# ---------------------------------------------------------------- sources

def released(ds: str, session: str) -> bool:
    return (ds, session) not in EXCLUDED_FROM_RELEASE and participant_id(ds, session) is not None


def frame(t: float) -> int:
    """Rows of a cut start: floor(t * 120), as the Stage 04 cuts compute it."""
    return int(t * FPS)


def trials_of(ds: str) -> list[dict]:
    """Every trial of the released sessions with its source files and Stage 04 pieces."""
    pieces: dict[str, list[dict]] = {}
    if ds == "220819":
        loc = json.loads((ANCHORS / "locate_manifest.json").read_text(encoding="utf-8"))
        for e in loc["entries"]:
            if e.get("ok") and e.get("out_stem"):
                pieces.setdefault(e["stage1_stem"], []).append(
                    {"piece": e["out_stem"], "start_sec": e["t_start_sec"], "end_sec": e["t_end_sec"]})
        align = {e["stage1_stem"]: e for e in
                 json.loads((ANCHORS / "alignment.json").read_text(encoding="utf-8"))["entries"]}
    else:
        for t in json.loads((PIECES / ds / "_pieces.json").read_text(encoding="utf-8")):
            pieces[t["trial_stem"]] = [{"piece": f"{t['trial_stem']}_p{p['piece_idx']}",
                                        "start_sec": p["start_sec"], "end_sec": p["end_sec"]}
                                       for p in t.get("pieces", [])]
    out = []
    for mp in sorted((NORM / ds).glob("*/_mapping.json")):
        session = mp.parent.name
        if not released(ds, session):
            continue
        for t in json.loads(mp.read_text(encoding="utf-8"))["trials"]:
            stem, raw = t["trial_stem"], t.get("raw_stem")
            sess = mp.parent
            rec = {"dataset": ds, "session": session, "trial": stem, "raw_stem": raw,
                   "pieces": sorted(pieces.get(stem, []), key=lambda p: p["start_sec"])}
            if ds in ("220819", "221112"):
                rec["audio"], rec["midi"] = sess / "audio" / f"{stem}.wav", sess / "midi" / f"{stem}.mid"
                rec["hand"], rec["body"] = HAND / ds / f"{stem}.csv", BODY / ds / f"{stem}.csv"
            else:
                rec["audio"], rec["midi"] = PIECES / ds / f"{stem}.wav", SYNC_MIDI / ds / f"{raw}.mid"
                rec["hand"], rec["body"] = HAND / ds / f"{stem}.csv", BODY / ds / f"{raw}.csv"
                if not rec["body"].exists():
                    rec["body"] = BODY / ds / f"{stem}.csv"
            rec["clock_start"] = None
            if ds in ("240523", "240828"):
                sj = SYNC_MIDI / ds / f"{raw}.json"
                if sj.exists():
                    rec["clock_start"] = json.loads(sj.read_text(encoding="utf-8")).get("timecode_start_sec")
            rec["shift_sec"] = None
            if ds == "220819":
                rec["shift_sec"] = ((align.get(stem) or {}).get("hand") or {}).get("align_shift_sec")
            rec["audio_midi_absent"] = is_audio_midi_absent(ds, stem)
            out.append(rec)
    return out


# ---------------------------------------------------------------- motion

def timecode_to_sec(tc: str) -> float:
    """Motive SMPTE "HH:MM:SS:FF.SS" (30 fps frames, 4 subframes) -> seconds (as Stage 04 s02)."""
    hh, mm, ss, ff = tc.split(":")
    fr, _, sub = ff.partition(".")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(fr) / 30 + int(sub or 0) / 120


def read_motion(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def numeric(df: pd.DataFrame) -> np.ndarray:
    return df.drop(columns=[c for c in ("TimeCode",) if c in df.columns]).to_numpy(dtype=np.float64)


def timecode_offset(df: pd.DataFrame, clock_start: float | None) -> int:
    """Stage 04 s02: frame 0 is the row whose TimeCode is closest to the trial start."""
    if clock_start is None or "TimeCode" not in df.columns:
        return 0
    tc = df["TimeCode"].dropna()
    if not len(tc):
        return 0
    secs = tc.map(timecode_to_sec).to_numpy()
    return int(tc.index[np.argmin(np.abs(secs - clock_start))])


def rows_equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per row: the same empty cells and every value within VALUE_TOL."""
    fa, fb = np.isfinite(a), np.isfinite(b)
    same_mask = np.all(fa == fb, axis=1)
    diff = np.where(fa & fb, np.abs(np.nan_to_num(a) - np.nan_to_num(b)), 0.0)
    return same_mask & np.all(diff <= VALUE_TOL, axis=1)


def locate(src: np.ndarray, piece: np.ndarray, guess: int) -> int | None:
    """Source row of the piece's row 0: its fullest row looked up in the source, value for value."""
    if piece.shape[1] != src.shape[1] or not len(piece):
        return None
    cnt = np.isfinite(piece).sum(axis=1)
    head = cnt[:1200]
    if cnt.max() == 0:
        return None
    i = int(np.argmax(head)) if head.max() > 0 else int(np.argmax(cnt))
    row = piece[i:i + 1]

    def ok(k: int) -> bool:
        return 0 <= k < len(src) and bool(rows_equal(src[k:k + 1], row)[0])

    for d in sorted(range(-3, 4), key=abs):
        if ok(guess + i + d):
            return guess + d
    j = int(np.nonzero(np.isfinite(row[0]))[0][0])
    idx = np.nonzero(np.abs(src[:, j] - row[0, j]) <= VALUE_TOL)[0]
    cands = [int(k) - i for k in idx if ok(int(k))]
    return min(cands, key=lambda c: abs(c - guess)) if cands else None


def write_motion(src: Path, shift: int, dst: Path) -> int:
    """The source rows from `shift` on (empty leading rows for a negative shift), TimeCode
    dropped; the values are copied as text, unchanged. Returns the number of rows."""
    with open(src, encoding="utf-8", newline="") as f:
        header = f.readline()
        lines = f.readlines()
    eol = "\r\n" if header.endswith("\r\n") else "\n"
    if header.startswith("TimeCode,"):
        header = header.split(",", 1)[1]
        lines = [ln.split(",", 1)[1] if "," in ln else eol for ln in lines]
    ncol = header.count(",") + 1
    body = lines[shift:] if shift >= 0 else [("," * (ncol - 1)) + eol] * (-shift) + lines
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="") as f:
        f.write(header)
        f.writelines(body)
    return len(body)


# ---------------------------------------------------------------- audio

def read_wav(path: Path) -> tuple[wave._wave_params, np.ndarray]:
    """Samples as int32 (n, channels) at their own width."""
    with wave.open(str(path)) as w:
        p = w.getparams()
        raw = w.readframes(p.nframes)
    if p.sampwidth == 2:
        x = np.frombuffer(raw, "<i2").astype(np.int32)
    elif p.sampwidth == 4:
        x = np.frombuffer(raw, "<i4").astype(np.int32)
    elif p.sampwidth == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        x = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(x >= 1 << 23, x - (1 << 24), x)
    else:
        raise ValueError(f"{path}: {p.sampwidth}-byte samples")
    return p, x.reshape(-1, p.nchannels)


def at_width(x: np.ndarray, width: int, target: int) -> np.ndarray:
    """Samples cut down to `target` bytes: a 24-bit file and the 32-bit copy Stage 04 wrote
    of it (pydub widens 24-bit samples, filling the new low byte) compare equal."""
    return x.astype(np.int64) >> (8 * (width - target))


def write_wav(path: Path, p, x: np.ndarray) -> None:
    flat = x.reshape(-1)
    if p.sampwidth == 2:
        raw = flat.astype("<i2").tobytes()
    elif p.sampwidth == 4:
        raw = flat.astype("<i4").tobytes()
    else:
        v = flat.astype(np.int32) & 0xFFFFFF
        raw = np.stack([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF], axis=1).astype(np.uint8).tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(p.nchannels)
        w.setsampwidth(p.sampwidth)
        w.setframerate(p.framerate)
        w.writeframes(raw)


def kept_spans(midi: Path, dur: float) -> list[tuple[float, float]]:
    """[onset - 0.10 s, sounding end + 0.5 s] of every note, merged, inside [0, dur]."""
    pm = pretty_midi.PrettyMIDI(str(midi))
    notes = sorted((n.start, n.end) for i in pm.instruments for n in i.notes)
    if not notes:
        return []
    onsets = [s for s, _ in notes]
    pedal = sustain_spans(midi)
    spans = sorted((max(0.0, s - PRE_ONSET_S), min(dur, sounding_end(e, pedal, onsets, dur) + POST_SOUND_S))
                   for s, e in notes)
    merged = [list(spans[0])]
    for a, b in spans[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b > a]


def gain_curve(n: int, rate: int, spans: list[tuple[float, float]]) -> np.ndarray:
    """1 inside the kept spans, 0 elsewhere, a linear FADE_S ramp on the muted side of each edge."""
    keep = np.zeros(n, bool)
    for a, b in spans:
        keep[int(round(a * rate)):int(round(b * rate))] = True
    if not keep.any():
        return np.zeros(n, np.float64)
    idx = np.arange(n)
    last = np.where(keep, idx, -1)
    last = np.maximum.accumulate(last)
    nxt = np.where(keep, idx, n + 10 ** 9)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    dist = np.minimum(np.where(last >= 0, idx - last, 10 ** 12), nxt - idx).astype(np.float64)
    g = np.clip(1.0 - dist / (FADE_S * rate), 0.0, 1.0)
    g[keep] = 1.0
    return g


# ---------------------------------------------------------------- MIDI

def write_midi(src: Path, dst: Path) -> int:
    """Notes and control changes of the source at its resolution (120 BPM); returns the note count."""
    pm = pretty_midi.PrettyMIDI(str(src))
    tempos = pm.get_tempo_changes()[1]
    if len(tempos) and any(abs(t - 120.0) > 1e-6 for t in tempos):
        raise ValueError(f"{src}: tempo {tempos} (expected 120 BPM)")
    out = pretty_midi.PrettyMIDI(resolution=pm.resolution, initial_tempo=120.0)
    n = 0
    for inst in pm.instruments:
        ni = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum,
                                    name=inst.name if inst.name.isascii() else "")
        ni.notes = [pretty_midi.Note(velocity=x.velocity, pitch=x.pitch, start=x.start, end=x.end)
                    for x in inst.notes if x.end > x.start]
        ni.control_changes = [pretty_midi.ControlChange(number=c.number, value=c.value, time=c.time)
                              for c in inst.control_changes]
        n += len(ni.notes)
        out.instruments.append(ni)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.write(str(dst))
    return n


def notes(path: Path) -> list[tuple[int, float, float]]:
    pm = pretty_midi.PrettyMIDI(str(path))
    return sorted((n.pitch, n.start, n.end) for i in pm.instruments for n in i.notes)


def check_midi(trial_notes, piece: Path, a: float, b: float) -> dict:
    """The trial's notes in [a, b] (shifted, clipped) against the piece file's notes, one for one.
    A note shorter than half a tick of the piece file is written zero-length there: on reading it
    is lost, or read as lasting to the next release of its key; both are accepted for such a note."""
    pm = pretty_midi.PrettyMIDI(str(piece))
    got = sorted((n.pitch, n.start, n.end) for i in pm.instruments for n in i.notes)
    tol = max(MIDI_TOL_S, 0.5 * 60.0 / (120.0 * pm.resolution) + 1e-9)
    tps = 2.0 * pm.resolution
    want = sorted((p, max(0.0, s - a), min(b - a, e - a)) for p, s, e in trial_notes if s < b and e > a)
    used = [False] * len(want)
    err, artifacts = 0.0, 0
    by_pitch: dict[int, list[int]] = {}
    for k, w in enumerate(want):
        by_pitch.setdefault(w[0], []).append(k)
    for pg, sg, eg in got:
        best = None
        for k in by_pitch.get(pg, []):
            if not used[k]:
                d = max(abs(want[k][1] - sg), abs(want[k][2] - eg))
                if best is None or d < best[0]:
                    best = (d, k)
        if best is None:
            return {"ok": False, "why": f"pitch {pg} at {sg:.3f} s not in the trial"}
        if best[0] > tol:
            # a zero-length note in the piece file is read back up to the next release of its key
            zt = [k for k in by_pitch[pg] if not used[k] and abs(want[k][1] - sg) <= tol
                  and round(want[k][2] * tps) <= round(want[k][1] * tps)]
            if zt and any(abs(want[k][2] - eg) <= tol for k in by_pitch[pg]):
                used[zt[0]] = True
                artifacts += 1
                continue
        used[best[1]] = True
        err = max(err, best[0])
    lost = [want[k] for k in range(len(want)) if not used[k]]
    if any(round(w[2] * tps) > round(w[1] * tps) for w in lost):
        return {"ok": False, "why": f"{len(lost)} trial notes not in the piece"}
    return {"ok": bool(err <= tol), "max_err_ms": round(err * 1000, 3), "tol_ms": round(tol * 1000, 3),
            "within_1ms": bool(err <= MIDI_TOL_S), "n": len(got), "zero_tick_lost": len(lost),
            "zero_tick_read_long": artifacts,
            **({} if err <= tol else {"why": f"note times differ by {err * 1000:.3f} ms"})}


# ---------------------------------------------------------------- per trial

def build_trial(rec: dict) -> dict:
    ds, stem = rec["dataset"], rec["trial"]
    dst = OUT / ds / stem
    dst.mkdir(parents=True, exist_ok=True)
    info = {k: rec[k] for k in ("dataset", "session", "trial", "raw_stem")}
    info["participant_id"] = participant_id(ds, rec["session"])
    info["files"], info["sources"], info["notes"] = [], {}, []
    checks: dict[str, dict] = {p["piece"]: {"start_sec": p["start_sec"], "end_sec": p["end_sec"]} for p in rec["pieces"]}

    # MIDI and audio
    has_audio = Path(rec["audio"]).exists() and not rec["audio_midi_absent"]
    has_midi = Path(rec["midi"]).exists() and not rec["audio_midi_absent"]
    trial_notes, gain, wav_p, trial_x, raw_x = None, None, None, None, None
    if has_midi:
        info["n_notes"] = write_midi(Path(rec["midi"]), dst / "midi.mid")
        info["files"].append("midi.mid")
        info["sources"]["midi"] = Path(rec["midi"]).as_posix()
        trial_notes = notes(dst / "midi.mid")
    else:
        (dst / "midi.mid").unlink(missing_ok=True)
    if has_audio and has_midi:
        wav_p, trial_x = read_wav(Path(rec["audio"]))
        dur = wav_p.nframes / wav_p.framerate
        spans = kept_spans(dst / "midi.mid", dur)
        gain = gain_curve(wav_p.nframes, wav_p.framerate, spans)
        muted = trial_x.copy()
        cut = gain < 1.0
        muted[cut] = np.round(trial_x[cut].astype(np.float64) * gain[cut, None]).astype(np.int32)
        write_wav(dst / "audio.wav", wav_p, muted)
        info["files"].append("audio.wav")
        info["sources"]["audio"] = Path(rec["audio"]).as_posix()
        info["audio_sec"] = round(dur, 3)
        info["audio_rate"] = wav_p.framerate
        info["muted_sec"] = round(float(cut.sum()) / wav_p.framerate, 3)
        info["kept_spans"] = len(spans)
        info["audio_muted"] = True
        raw_x, trial_x = trial_x, muted
    elif has_audio:                              # no MIDI to mute by: the audio as recorded
        wav_p, x = read_wav(Path(rec["audio"]))
        write_wav(dst / "audio.wav", wav_p, x)
        info["files"].append("audio.wav")
        info["sources"]["audio"] = Path(rec["audio"]).as_posix()
        info["audio_sec"] = round(wav_p.nframes / wav_p.framerate, 3)
        info["audio_rate"] = wav_p.framerate
        info["muted_sec"] = 0.0
        info["audio_muted"] = False
        info["audio_note"] = "no MIDI was recorded with this take, so speech is not muted"
    else:
        (dst / "audio.wav").unlink(missing_ok=True)

    # motion
    for kind in ("hand", "body"):
        src = Path(rec[kind])
        if not src.exists():
            (dst / f"{kind}.csv").unlink(missing_ok=True)
            continue
        df = read_motion(src)
        arr = numeric(df)
        if ds == "220819":
            sh = rec["shift_sec"] or 0.0
            expected = int(round(sh * FPS))
            guess = lambda a: int(round((a + sh) * FPS))          # noqa: E731  (anchors s03_slice)
        else:
            expected = timecode_offset(df, rec["clock_start"]) if kind == "hand" else 0
            guess = lambda a: expected + frame(a)                 # noqa: E731
        located = {}
        for p in rec["pieces"]:
            pf = PIECES / ds / f"{p['piece']}_{kind}.csv"
            if not pf.exists():
                continue
            parr = numeric(pd.read_csv(pf, low_memory=False))
            if not np.isfinite(parr).any():         # an empty piece: nothing to look up, check it where it was cut
                located[p["piece"]] = (guess(p["start_sec"]), parr, False)
                continue
            located[p["piece"]] = (locate(arr, parr, guess(p["start_sec"])), parr, True)
        offs = Counter(k - frame(checks[ps]["start_sec"]) for ps, (k, _, by_value) in located.items()
                       if k is not None and by_value)
        if offs:
            top = max(offs.values())
            shift = min((o for o, c in offs.items() if c == top), key=lambda o: abs(o - expected))
        else:
            shift = expected
        n = write_motion(src, shift, dst / f"{kind}.csv")
        if n != len(arr) - shift:
            raise RuntimeError(f"{src}: text rows and parsed rows differ")
        info["files"].append(f"{kind}.csv")
        info["sources"][kind] = src.as_posix()
        info[f"{kind}_source_offset_frames"] = shift
        info[f"{kind}_expected_offset_frames"] = expected
        info[f"{kind}_frames"] = n
        trial_arr = arr[shift:] if shift >= 0 else np.vstack([np.full((-shift, arr.shape[1]), np.nan), arr])
        for ps, (k, parr, by_value) in located.items():
            c = checks[ps]
            if k is None:
                c[kind] = {"ok": False, "why": "piece rows not found in the source"}
                continue
            s = k - shift
            # rows outside the trial motion compare as empty (a cut padded before or after the recording)
            seg = np.full(parr.shape, np.nan)
            lo, hi = max(0, s), min(len(trial_arr), s + len(parr))
            if hi > lo:
                seg[lo - s:hi - s] = trial_arr[lo:hi]
            # the window's rows, as far as the trial motion runs
            rule_n = max(0, min(frame(c["end_sec"]) - frame(c["start_sec"]), len(trial_arr) - s))
            eq = rows_equal(seg, parr)
            res = {"start_frame": s, "end_frame": s + len(parr), "residual_frames": s - frame(c["start_sec"]),
                   "rows": len(parr), "rule_rows": rule_n, "rows_off_rule": len(parr) - rule_n,
                   "ok": bool(eq.all()) and abs(len(parr) - rule_n) <= 1,
                   **({} if by_value else {"empty_piece": True}),
                   **({"rows_outside_trial": int(len(parr) - (hi - lo))} if hi - lo < len(parr) else {})}
            if not eq.all():
                res["why"] = f"{int((~eq).sum())} rows differ"
            elif abs(len(parr) - rule_n) > 1:
                res["why"] = f"{len(parr)} rows, the window holds {rule_n}"
            c[kind] = res

    # audio and MIDI checks of the pieces
    if trial_x is not None or trial_notes is not None:
        for p in rec["pieces"]:
            c = checks[p["piece"]]
            a, b = p["start_sec"], p["end_sec"]
            pm_path = PIECES / ds / f"{p['piece']}.mid"
            if trial_notes is not None and pm_path.exists():
                c["midi"] = check_midi(trial_notes, pm_path, a, b)
            pw = PIECES / ds / f"{p['piece']}.wav"
            if trial_x is not None and pw.exists():
                c["audio"] = check_audio(trial_x, wav_p, gain, raw_x, pw, a)
    info["pieces"] = {ps: {k: v for k, v in c.items()} for ps, c in checks.items()}
    return info


def check_audio(trial_x, p, gain, raw_x, piece: Path, a: float) -> dict:
    """Piece samples against the trial audio at the millisecond cut, where the trial is not muted."""
    pp, px = read_wav(piece)
    if pp.framerate != p.framerate or pp.nchannels != p.nchannels:
        return {"ok": False, "why": f"format {pp.framerate}/{pp.nchannels} vs {p.framerate}/{p.nchannels}"}
    ms = int(a * 1000)
    lead = 0
    if a < 0:                                    # Stage 04 pads a window before the recording with silence
        lead = int(int(round(-a * 1000)) * p.framerate / 1000.0)
        o = 0
    else:
        o = int(ms * p.framerate / 1000.0)
    w = min(pp.sampwidth, p.sampwidth)
    pv = at_width(px, pp.sampwidth, w)[lead:]
    n = min(len(pv), len(trial_x) - o)
    if n <= 0:
        return {"ok": False, "why": "window outside the trial audio"}
    tv = at_width(trial_x[o:o + n], p.sampwidth, w)
    rv = at_width(raw_x[o:o + n], p.sampwidth, w)
    keep = gain[o:o + n] >= 1.0
    same_raw = bool(np.array_equal(pv[:n], rv))
    if not same_raw:                             # look for the cut a few samples off
        for d in (-2, -1, 1, 2):
            if 0 <= o + d and o + d + n <= len(raw_x) and np.array_equal(pv[:n], at_width(raw_x[o + d:o + d + n], p.sampwidth, w)):
                return {"ok": False, "why": f"cut {d} samples off the millisecond rule"}
    ok_kept = bool(np.array_equal(pv[:n][keep], tv[keep]))
    return {"ok": ok_kept and same_raw, "samples": int(n), "kept_share": round(float(keep.mean()), 4),
            "len_diff_samples": int(len(pv) - n), **({} if same_raw else {"why": "samples differ from the trial audio"})}


def check_take_240523(info: dict) -> dict:
    """240523 is released per take: compare the trial files with the take files the manifest lists."""
    ds, stem = info["dataset"], info["trial"]
    dst = OUT / ds / stem
    out = {}
    raw = info["sources"]
    if "midi.mid" in info["files"]:
        a, b = notes(dst / "midi.mid"), notes(Path(raw["midi"]))
        out["midi"] = {"ok": len(a) == len(b) and all(x[0] == y[0] and abs(x[1] - y[1]) <= 1e-6 and abs(x[2] - y[2]) <= 1e-6
                                                      for x, y in zip(a, b))}
    for kind in ("hand", "body"):
        if f"{kind}.csv" in info["files"]:
            t = numeric(pd.read_csv(dst / f"{kind}.csv", low_memory=False))
            s = numeric(pd.read_csv(Path(raw[kind]), low_memory=False))
            n = min(len(t), len(s))
            out[kind] = {"ok": len(t) == len(s) and bool(rows_equal(t[:n], s[:n]).all()),
                         "offset_frames": info[f"{kind}_source_offset_frames"]}
    return out


def run_one(rec: dict) -> tuple[str, dict | None, str | None]:
    try:
        t0 = time.time()
        info = build_trial(rec)
        if rec["dataset"] == "240523":
            info["take_check"] = check_take_240523(info)
        info["build_sec"] = round(time.time() - t0, 1)
        return rec["trial"], info, None
    except Exception:
        return rec["trial"], None, traceback.format_exc()


def _plain(o):
    return o.item() if hasattr(o, "item") else str(o)


def summarize(index: dict) -> dict:
    """Pass / fail per dataset: every Stage 04 piece and the released (Stage 05 confirmed) ones."""
    out = {}
    for ds in DATASETS:
        status = json.loads((QA / ds / "_status.json").read_text(encoding="utf-8")) if (QA / ds / "_status.json").exists() else {}
        confirmed = {p["piece"] for p in status.get("pieces", []) if p.get("confirmed")
                     and p.get("boundary_confirmed_by") != "not_a_piece"}
        s = {"trials": 0, "audio_sec": 0.0, "muted_sec": 0.0, "pieces": Counter(), "confirmed": Counter(),
             "residual_pieces": {"hand": 0, "body": 0}, "residual_examples": [], "failures": [],
             "midi_over_1ms": 0, "midi_zero_tick_notes": 0, "notes": 0, "motion_sec": 0.0, "audio_trials": 0,
             "motion_only_trials": 0, "unmuted_audio": []}
        for stem, t in index.items():
            if t["dataset"] != ds:
                continue
            s["trials"] += 1
            s["audio_sec"] += t.get("audio_sec", 0.0)
            s["notes"] += t.get("n_notes", 0)
            s["motion_sec"] += max(t.get("hand_frames", 0), t.get("body_frames", 0)) / FPS
            s["audio_trials"] += "audio.wav" in t["files"]
            s["motion_only_trials"] += "midi.mid" not in t["files"] and "audio.wav" not in t["files"] and bool(t["files"])
            if t.get("audio_muted") is False:
                s["unmuted_audio"].append(stem)
            s["muted_sec"] += t.get("muted_sec", 0.0)
            for ps, c in t.get("pieces", {}).items():
                mods = [m for m in ("audio", "midi", "hand", "body") if m in c]
                ok = all(c[m].get("ok") for m in mods)
                for kind in ("hand", "body"):
                    if kind in c and c[kind].get("residual_frames"):
                        s["residual_pieces"][kind] += 1
                        if len(s["residual_examples"]) < 30:
                            s["residual_examples"].append(f"{ps}:{kind}:{c[kind]['residual_frames']}")
                if "midi" in c:
                    s["midi_over_1ms"] += not c["midi"].get("within_1ms", True)
                    s["midi_zero_tick_notes"] += c["midi"].get("zero_tick_lost", 0) + c["midi"].get("zero_tick_read_long", 0)
                key = "pass" if ok else "fail"
                s["pieces"][key] += 1
                if ps in confirmed:
                    s["confirmed"][key] += 1
                if not ok:
                    s["failures"].append({"piece": ps, "confirmed": ps in confirmed,
                                          **{m: c[m] for m in mods if not c[m].get("ok")}})
        s["muted_share"] = round(s["muted_sec"] / s["audio_sec"], 4) if s["audio_sec"] else None
        s["pieces"], s["confirmed"] = dict(s["pieces"]), dict(s["confirmed"])
        s["confirmed_total"] = len(confirmed)
        out[ds] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", default=DATASETS)
    ap.add_argument("--trial", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    recs = [r for ds in a.dataset for r in trials_of(ds)]
    if a.trial:
        recs = [r for r in recs if r["trial"] in a.trial]
    idx_path = OUT / "_trials.json"
    index = json.loads(idx_path.read_text(encoding="utf-8")) if idx_path.exists() else {}
    print(f"{len(recs)} trials", flush=True)
    errors = {}
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_one, r) for r in recs]
        for f in as_completed(futs):
            stem, info, err = f.result()
            done += 1
            if err:
                errors[stem] = err
                print(f"[{done}/{len(recs)}] {stem}: ERROR\n{err}", flush=True)
                continue
            index[stem] = info
            bad = [ps for ps, c in info["pieces"].items()
                   if not all(v.get("ok") for m, v in c.items() if m in ("audio", "midi", "hand", "body"))]
            print(f"[{done}/{len(recs)}] {stem}: {', '.join(info['files']) or 'no files'}; "
                  f"{len(info['pieces'])} pieces, {len(bad)} not reproduced ({info['build_sec']} s)", flush=True)
            if done % 10 == 0:
                idx_path.write_text(json.dumps(index, indent=1, default=_plain), encoding="utf-8")
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path.write_text(json.dumps(dict(sorted(index.items())), indent=1, default=_plain), encoding="utf-8")
    summary = summarize(index)
    summary["errors"] = errors
    (OUT / "_verify.json").write_text(json.dumps(summary, indent=1, default=_plain), encoding="utf-8")
    for ds, s in summary.items():
        if ds == "errors":
            continue
        print(f"{ds}: {s['trials']} trials, audio {s['audio_sec'] / 3600:.2f} h, muted {s['muted_share']}, "
              f"pieces {s['pieces']}, confirmed {s['confirmed']} of {s['confirmed_total']}, "
              f"residual offsets {s['residual_pieces']}")
    if errors:
        print(f"{len(errors)} trials failed: {sorted(errors)}")


if __name__ == "__main__":
    main()
