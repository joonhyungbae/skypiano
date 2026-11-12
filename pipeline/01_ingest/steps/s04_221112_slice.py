"""
Stage 01 step 04: 221112 -- place every motion take in its session recording.

221112 has, per subject folder Raw/221112/C1/N{n}/:
  - one continuous session recording (Audio/*.wav, MIDI/*.mid);
  - one hand (OptiTrack) and one body (Qualisys) file per take;
and per session a recording log (Data/Metadata/221112/C1/N{n}/221112_C1_N{n}.csv)
with every trial's start signal and every piece's start / end in session time.

The folder numbers of the three do not agree. They are paired by content:

  1. log <-> session MIDI: a log's piece start times fall on note onsets of
     exactly one session MIDI (after one constant shift: the note-taker's
     reaction time, ~0.3-0.7 s). Score = share of piece starts with an onset
     within LOG_ONSET_TOL_S, best shift within +-LOG_SHIFT_MAX_S.
  2. take position: a take is searched at note level (summed downward
     fingertip speed vs the onset train, 60 Hz) within SIGNAL_TOL_S of the
     trial start signals of a log. Takes follow the log's trial order (hand
     files are numbered in capture order), so a take is only paired with the
     log trial of its rank or a neighbouring one (a take missing or added);
     the pairing is one to one (Hungarian on the correlation).
  3. motion folder <-> session MIDI: the file (with its log) whose placement
     of the folder's takes correlates best on average, measured above the
     folder's background (its median over all files: wrong files still give
     chance correlations of ~0.1, and capture quality differs per folder).

Both pairings are one-to-one (Hungarian). A motion folder with no session
recording keeps its motion only. The session id is the motion folder, as in
every later stage.

  4. key position, for the paired folder and recording: the onset
     correlation of short repetitive takes peaks at several positions of
     similar height. Among all its peaks (local maxima above 0) in each
     take's search window the position is chosen where the most notes have a
     fingertip over the played key (motion_midi_xcorr.key_hit; keys along the
     capture's LEND -> REND markers), scored hit + r, and takes are paired
     with log trials again on that score within KEY_PAIR_TOL_S of the log
     signal; a paired take is then placed at its best peak within
     KEY_SIGNAL_TOL_S. The search runs on past the recording's last note (a
     session's last take is captured beyond it) and up to MAX_LEAD_PAD_S
     before its start (a first take can start before the recording).

A take whose audio and MIDI the known-absences registry lists as not
recorded (kind `audio_midi`) is not placed and keeps its motion only.

Trial audio / MIDI are the session recording cut at [position, position +
motion duration], so trial time = motion time. The position is corrected for
the key-press lead (motion_midi_xcorr.PRESS_LEAD_S).

Canonical stem: 221112_C1_N{n}_T1_P{piece}_t{take} from the hand file name
Piano02_{ss}_{piece}_{take}[_suffix]; a second file with the same piece/take
(a re-export with a numeric suffix) is kept when it is a different take (its
own position) and numbered after the folder's last take of that piece.

Results: Normalized/221112/C1_N{n}/_mapping.json per session, and
Normalized/221112/_pairing.json with the full score matrices.
"""
from __future__ import annotations

import csv
import functools
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pretty_midi
from pydub import AudioSegment
from scipy.optimize import linear_sum_assignment

_P = Path(__file__).resolve()
while _P.name != "pipeline":
    _P = _P.parent
sys.path.insert(0, str(_P))
from _common.normalized_io import NORMALIZED_ROOT, RAW_ROOT, session_dir, write_mapping  # noqa: E402
from _common.slicing import slice_audio, slice_midi  # noqa: E402
from _common import motion_midi_xcorr as mx  # noqa: E402
from _common.known_absences import is_audio_midi_absent  # noqa: E402

print = functools.partial(print, flush=True)

DATASET = "221112"
TASK = 1
FOLDERS = list(range(2, 12))           # C1/N2 .. C1/N11
LOG_ROOT = Path("Data/Metadata/221112/C1")
# Row labels of the (Korean) session log: start, end, remarks.
LOG_START, LOG_END, LOG_NOTE = "\uc2dc\uc791", "\ub05d", "\ud2b9\uc774\uc0ac\ud56d"
FS = mx.NOTE_FS                         # Hz, note-level grid
LOG_ONSET_TOL_S = 0.3
LOG_SHIFT_MAX_S = 5.0
LOG_MIN_SCORE = 0.9                     # share of log piece starts on an onset
FOLDER_MIN_LIFT = 0.04                  # mean placement r above the folder's median over files
RANK_SLACK = 1                          # a take pairs with log trials within this many ranks
SIGNAL_TOL_S = 30.0
KEY_PAIR_TOL_S = 90.0                   # step 4: takes pair with log trials on peaks this near the signal
KEY_SIGNAL_TOL_S = 150.0                # step 4: a paired take is placed this far from its signal
MAX_LEAD_PAD_S = 5.0                    # a take may start this long before the recording (silence padded)

HAND_RE = re.compile(r"^Piano02_(\d{2})_(\d{2})_(\d{2})(?:_(\d+))?$")


# ── inputs ──

def motion_duration(csv_path: Path) -> float:
    with open(csv_path, "r", encoding="latin1", errors="ignore") as f:
        parts = f.readline().rstrip("\n").split(",")
    kv = {parts[i].strip(): parts[i + 1].strip() for i in range(0, len(parts) - 1, 2)}
    return int(kv["Total Exported Frames"]) / float(kv["Export Frame Rate"])


def hand_takes(n: int) -> list[dict]:
    out = []
    for f in sorted((RAW_ROOT / DATASET / "C1" / f"N{n}" / "Motion" / "Hand").glob("*.csv")):
        m = HAND_RE.match(f.stem)
        if m:
            out.append({"stem": f.stem, "path": f, "piece": int(m.group(2)), "take": int(m.group(3)),
                        "suffix": m.group(4), "duration": motion_duration(f)})
    return out


def session_files(n: int) -> tuple[Path | None, Path | None]:
    base = RAW_ROOT / DATASET / "C1" / f"N{n}"
    w, m = base / "Audio" / f"221112_C1_N{n}.wav", base / "MIDI" / f"221112_C1_N{n}.mid"
    return (w if w.exists() else None), (m if m.exists() else None)


def _num(x: str) -> float | None:
    return float(x) if re.fullmatch(r"\d+(\.\d+)?", (x or "").strip()) else None


def log_trials(n: int) -> list[dict]:
    """Trials of a session log: set/trial/repeat, start signal, pieces (session time)."""
    rows = list(csv.reader(open(LOG_ROOT / f"N{n}" / f"221112_C1_N{n}.csv", encoding="utf-8")))
    out = []
    for i, r in enumerate(rows):
        if len(r) > 5 and r[4].strip() == "ID" and r[1].strip().isdigit() and i + 2 < len(rows):
            s, e = rows[i + 1], rows[i + 2]
            if s[4].strip() != LOG_START:
                continue
            has_end = e[4].strip() == LOG_END
            note = rows[i + 3] if i + 3 < len(rows) and rows[i + 3][4].strip() == LOG_NOTE else None
            pieces = []
            for k in range(6, len(r)):
                a = _num(s[k]) if k < len(s) else None
                b = _num(e[k]) if has_end and k < len(e) else None
                if r[k].strip() and a is not None:
                    pieces.append({"id": "".join(c for c in r[k] if c.isprintable()).strip(), "start": a, "end": b,
                                   "note": (note[k].strip() if note and k < len(note) else "") or None})
            out.append({"set": int(r[1]), "trial": int(r[2]), "repeat": int(r[3]),
                        "subject": r[0].strip(), "signal": _num(s[5]), "pieces": pieces})
    return out


# ── signals ──

def session_notes(midi: Path) -> tuple[np.ndarray, np.ndarray]:
    """(onset times, key centres) of every note, by onset."""
    ns = sorted((x.start, x.pitch) for i in pretty_midi.PrettyMIDI(str(midi)).instruments for x in i.notes)
    return np.array([t for t, _ in ns]), np.array([mx.key_center(p) for _, p in ns])


def take_lateral(path: Path) -> tuple[float, np.ndarray] | None:
    """(fps, fingertip position along LEND -> REND) of a raw OptiTrack take."""
    le, re_ = mx.optitrack_marker_median(path, "LEND"), mx.optitrack_marker_median(path, "REND")
    if le is None or re_ is None or not (np.isfinite(le).all() and np.isfinite(re_).all()):
        return None
    fps, xyz = mx._read_optitrack_fingertips(path)
    return fps, mx.keyboard_lateral(xyz, le, re_)


def onset_times(midi: Path) -> np.ndarray:
    return np.array(sorted(x.start for i in pretty_midi.PrettyMIDI(str(midi)).instruments for x in i.notes))


def onset_train(onsets: np.ndarray) -> np.ndarray:
    return mx.onset_train(onsets)


def press_train(path: Path) -> np.ndarray:
    fps, xyz = mx._read_optitrack_fingertips(path)          # OptiTrack: Y up
    return mx.press_train(xyz, fps, up_axis=1)


slide_ncc = mx.slide_ncc


# ── pairing ──

def log_score(starts: np.ndarray, onsets: np.ndarray) -> tuple[float, float]:
    """(share of log starts with an onset within tolerance, shift) over shifts."""
    best = (0.0, 0.0)
    for sh in np.arange(-LOG_SHIFT_MAX_S, LOG_SHIFT_MAX_S + 1e-9, 0.05):
        st = starts + sh
        i = np.clip(np.searchsorted(onsets, st - LOG_ONSET_TOL_S), 0, len(onsets) - 1)
        f = float(np.mean(np.abs(onsets[i] - st) <= LOG_ONSET_TOL_S))
        if f > best[0]:
            best = (f, float(sh))
    return best


def hungarian(score: np.ndarray, floor: float) -> list[tuple[int, int]]:
    r, c = linear_sum_assignment(-np.where(score >= floor, score, -1e3))
    return [(i, j) for i, j in zip(r, c) if score[i, j] >= floor]


def place_takes(takes, trains, trials, onsets, shift):
    """Pair takes with log trials; position = session time of motion t = 0."""
    signals = [t for t in trials if t["signal"] is not None]
    otr = onset_train(onsets)
    S = np.full((len(takes), len(signals)), -1.0)
    P = np.zeros_like(S)
    rank = sorted(range(len(takes)), key=lambda i: (takes[i]["piece"], takes[i]["take"], takes[i]["suffix"] or ""))
    for i, tk in enumerate(takes):
        for j, tr in enumerate(signals):
            if abs(rank.index(i) - j) > RANK_SLACK:
                continue
            c = tr["signal"] + shift
            a = max(0, int((c - SIGNAL_TOL_S) * FS))
            b = min(len(otr), int((c + SIGNAL_TOL_S) * FS) + len(trains[i]))
            r = slide_ncc(otr[a:b], trains[i])
            k = int(np.argmax(r))
            S[i, j], P[i, j] = r[k], (a + k) / FS
    pairs = hungarian(S, -0.5)
    return {i: (signals[j], float(P[i, j]), float(S[i, j])) for i, j in pairs}, S


def place_takes_by_key(takes, trains, lats, trials, notes, shift):
    """As place_takes, scored hit + r over the onset-correlation peaks (step 4)."""
    signals = [t for t in trials if t["signal"] is not None]
    onsets, keys = notes
    # the capture of a session's last take runs on after its last note, and a
    # first take can start before the recording does: silence on both sides
    lead = int(MAX_LEAD_PAD_S * FS)
    otr = np.concatenate([np.zeros(lead), onset_train(onsets), np.zeros(max(len(t) for t in trains))])
    S = np.full((len(takes), len(signals)), -1.0)
    F = np.full_like(S, -1.0)
    P = np.zeros_like(S)
    rank = sorted(range(len(takes)), key=lambda i: (takes[i]["piece"], takes[i]["take"], takes[i]["suffix"] or ""))
    for i, tk in enumerate(takes):
        for j, tr in enumerate(signals):
            if abs(rank.index(i) - j) > RANK_SLACK:
                continue
            c = tr["signal"] + shift
            a = max(0, int((c - KEY_SIGNAL_TOL_S) * FS) + lead)
            b = min(len(otr), int((c + KEY_SIGNAL_TOL_S) * FS) + lead + len(trains[i]))
            r = slide_ncc(otr[a:b], trains[i])
            peaks = np.flatnonzero((r[1:-1] > 0) & (r[1:-1] >= r[:-2]) & (r[1:-1] >= r[2:])) + 1
            scored = []
            for k in peaks:
                pos = (a + k - lead) / FS + mx.PRESS_LEAD_S
                hit = mx.key_hit(lats[i][1], lats[i][0], onsets, keys, -pos) if lats[i] else 0.0
                scored.append((hit + max(float(r[k]), 0.0), (a + k - lead) / FS, abs((a + k - lead) / FS - c)))
            near = [x for x in scored if x[2] <= KEY_PAIR_TOL_S]
            S[i, j] = max(near)[0] if near else -1.0
            F[i, j], P[i, j] = max(scored)[:2] if scored else (-1.0, 0.0)
    # takes are paired with log trials on the score near the signal (a wide
    # window would let a take claim a neighbouring trial's span); the paired
    # take is then placed at its best position anywhere in the wide window
    pairs = hungarian(S, -0.5)
    return {i: (signals[j], float(P[i, j]), float(F[i, j])) for i, j in pairs}, S


def run(dataset: str | None = None, force: bool = False):
    if dataset and dataset != DATASET:
        return
    print("Stage 01 s04 -- 221112 content pairing and take placement")
    files = [n for n in FOLDERS if session_files(n)[1]]
    onsets = {f: onset_times(session_files(f)[1]) for f in files}
    logs = {n: log_trials(n) for n in FOLDERS if (LOG_ROOT / f"N{n}" / f"221112_C1_N{n}.csv").exists()}

    # 1. log <-> session MIDI
    L = np.zeros((len(logs), len(files)))
    Lsh = np.zeros_like(L)
    for i, n in enumerate(logs):
        starts = np.array([p["start"] for t in logs[n] for p in t["pieces"]])
        for j, f in enumerate(files):
            L[i, j], Lsh[i, j] = log_score(starts, onsets[f])
    log_of_file = {files[j]: (list(logs)[i], float(Lsh[i, j]), float(L[i, j]))
                   for i, j in hungarian(L, LOG_MIN_SCORE)}
    for f, (n, sh, sc) in log_of_file.items():
        print(f"  log N{n} <-> session file N{f}: {sc:.2f} of piece starts on an onset (shift {sh:+.2f}s)")

    # 2-3. takes placed in every (file, log) pair; folder <-> file by mean correlation
    takes = {n: hand_takes(n) for n in FOLDERS}
    trains = {n: [press_train(t["path"]) for t in takes[n]] for n in FOLDERS}
    pfiles = list(log_of_file)
    V = np.zeros((len(FOLDERS), len(pfiles)))
    placements = {}
    for i, n in enumerate(FOLDERS):
        for j, f in enumerate(pfiles):
            lg = log_of_file[f]
            placed, S = place_takes(takes[n], trains[n], logs[lg[0]], onsets[f], lg[1])
            placements[(n, f)] = (placed, S)
            V[i, j] = np.mean([r for _, _, r in placed.values()]) if placed else -1
        print(f"  motion N{n}: mean placement r " + " ".join(f"N{f}:{V[i, j]:.2f}" for j, f in enumerate(pfiles)))
    lift = V - np.median(V, axis=1, keepdims=True)
    file_of_folder = {FOLDERS[i]: pfiles[j] for i, j in hungarian(lift, FOLDER_MIN_LIFT)}
    for n in FOLDERS:
        f = file_of_folder.get(n)
        i, j = FOLDERS.index(n), pfiles.index(f) if f else 0
        print(f"  motion N{n} <-> session file N{f} (mean r {V[i, j]:.2f}, {lift[i, j]:+.2f} over background)"
              if f else f"  motion N{n}: no session recording")
    files = pfiles

    pairing = {"log_vs_file": {f"log N{n}": {f"N{f}": [round(float(L[i, j]), 3), round(float(Lsh[i, j]), 2)]
                                             for j, f in enumerate(files)} for i, n in enumerate(logs)},
               "folder_vs_file_mean_r": {f"N{n}": {f"N{f}": round(float(V[i, j]), 3) for j, f in enumerate(files)}
                                         for i, n in enumerate(FOLDERS)},
               "folder_to_file": {f"N{n}": f"N{f}" for n, f in file_of_folder.items()},
               "file_to_log": {f"N{f}": f"N{v[0]}" for f, v in log_of_file.items()}, "sessions": {}}

    # 3. per motion folder: place takes, cut trials
    for n in FOLDERS:
        f = file_of_folder.get(n)
        lg = log_of_file.get(f) if f else None
        placed, S = ({}, None)
        if f:
            lats = [take_lateral(t["path"]) for t in takes[n]]
            placed, S = place_takes_by_key(takes[n], trains[n], lats, logs[lg[0]], session_notes(session_files(f)[1]),
                                           lg[1])
        write_session(n, takes[n], f, lg, logs.get(lg[0]) if lg else None, placed, force)
        pairing["sessions"][f"C1_N{n}"] = {
            "session_file": f"N{f}" if f else None, "log": f"N{lg[0]}" if lg else None,
            "take_vs_log_trial_r": S.round(3).tolist() if S is not None else None}
    (NORMALIZED_ROOT / DATASET / "_pairing.json").write_text(json.dumps(pairing, indent=1), encoding="utf-8")


def write_session(n, takes, f, lg, trials, placed, force):
    out = session_dir(DATASET, f"C1_N{n}", modalities=["motion_hand", "motion_body", "audio", "midi"])
    wav, mid = session_files(f) if f else (None, None)
    audio = AudioSegment.from_wav(str(wav)) if wav else None
    body_dir = RAW_ROOT / DATASET / "C1" / f"N{n}" / "Motion" / "Body"
    last_take: dict[int, int] = {}
    for tk in takes:
        last_take[tk["piece"]] = max(last_take.get(tk["piece"], 0), tk["take"])
    records, seen = [], set()
    for i, tk in enumerate(sorted(takes, key=lambda t: (t["piece"], t["take"], t["suffix"] or ""))):
        k = takes.index(tk)
        key = (tk["piece"], tk["take"])
        t_no = tk["take"]
        if key in seen:                                      # a second file of the same piece/take
            last_take[tk["piece"]] += 1
            t_no = last_take[tk["piece"]]
        first_of_key = key not in seen
        seen.add(key)
        stem = f"{DATASET}_C1_N{n}_T{TASK}_P{tk['piece']}_t{t_no}"
        rs = {"motion_hand": tk["path"].name}
        dst = out / "motion_hand" / f"{stem}.csv"
        if force or not dst.exists():
            shutil.copy2(tk["path"], dst)
        # body takes carry no suffix: the first hand file of a piece/take owns it
        base = f"2022Piano02_{tk['stem'][8:16]}"
        body = next((body_dir / f"{base}{e}" for e in (".tsv", ".csv") if (body_dir / f"{base}{e}").exists()), None)
        if body and first_of_key:
            dst = out / "motion_body" / f"{stem}{body.suffix}"
            if force or not dst.exists():
                shutil.copy2(body, dst)
            rs["motion_body"] = body.name
        rec = {"trial_stem": stem, "raw_stem": tk["stem"], "task": TASK, "canonical_p": tk["piece"],
               "canonical_t": t_no, "motion_duration_sec": round(tk["duration"], 3)}
        if k in placed and is_audio_midi_absent(DATASET, stem):
            print(f"    {stem[-9:]} {tk['stem']:22s} audio/MIDI known absent (registry) -> motion only")
        elif k in placed:
            tr, pos, r = placed[k]
            pos += mx.PRESS_LEAD_S                           # key-press activity leads the onset
            a, b = pos, pos + tk["duration"]
            if a >= -MAX_LEAD_PAD_S and audio is not None and a < len(audio) / 1000.0:   # a take may outlast the recording
                slice_audio(audio, a, b, out / "audio" / f"{stem}.wav")
                slice_midi(mid, a, b, out / "midi" / f"{stem}.mid")
                rs["audio"] = f"{wav.name} [{a:.3f}-{b:.3f}]"
                rs["midi"] = f"{mid.name} [{a:.3f}-{b:.3f}]"
            log_shift = lg[1]
            rec["placement"] = {
                "session_file": wav.name if wav else None, "position_sec": round(a, 3), "score": round(r, 3),
                "log": f"N{lg[0]}", "log_trial": f"{tr['set']}-{tr['trial']}_t{tr['repeat']}",
                "log_signal_sec": tr["signal"], "position_minus_signal_sec": round(a - (tr["signal"] + log_shift), 2),
                "log_shift_sec": round(log_shift, 2)}
            # pieces of the log that lie in this take, on the trial (= motion) clock
            in_take = [p for t in trials for p in t["pieces"]
                       if p["end"] is not None and a <= p["start"] + log_shift and p["end"] + log_shift <= b]
            rec["log_pieces"] = [{"id": p["id"], "start": round(p["start"] + log_shift - a, 3),
                                  "end": round(p["end"] + log_shift - a, 3), "note": p["note"]}
                                 for p in sorted(in_take, key=lambda p: p["start"])]
        if "audio" not in rs:                                # not placed: no trial audio / MIDI left behind
            for d, ext in (("audio", ".wav"), ("midi", ".mid")):
                (out / d / f"{stem}{ext}").unlink(missing_ok=True)
        rec["status"] = ("complete" if {"audio", "motion_body"} <= rs.keys() else
                         "body_missing" if "audio" in rs else "motion_only")
        rec["modalities"] = [m for m in ("audio", "midi", "motion_hand", "motion_body") if m in rs]
        rec["raw_sources"] = rs
        records.append(rec)
        print(f"    {stem[-9:]} {tk['stem']:22s} " + (
            f"at {rec['placement']['position_sec']:8.2f}s score={rec['placement']['score']:.2f} log {rec['placement']['log_trial']}"
            f" (signal {rec['placement']['position_minus_signal_sec']:+.1f}s) pieces {len(rec['log_pieces'])}"
            if "placement" in rec else "motion only"))
    write_mapping(DATASET, f"C1_N{n}", records, method="221112_content_placement", extra={
        "session_file": {"audio": wav.name if wav else None, "midi": mid.name if mid else None,
                         "folder": f"N{f}" if f else None},
        "log": f"N{lg[0]}" if lg else None,
        "capabilities": {"has_timecode": False, "has_video": True, "video_is_per_trial": False,
                         "has_body_motion": body_dir.exists(), "body_in_hand_csv": False,
                         "matching_method": "content (log onsets, note-level key presses)"},
        "notes": ["Session id = motion folder. The session recording and the log are the ones whose "
                  "content matches (Normalized/221112/_pairing.json); folder numbers do not agree.",
                  "Trial audio/MIDI = session recording cut at the take's position; trial time = motion time."],
    })


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    run(force=ap.parse_args().force)
