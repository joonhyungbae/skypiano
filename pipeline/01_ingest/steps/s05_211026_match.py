"""
Stage 01 step 05: 211026 -- pair per-take audio with the session MIDI takes.

211026 has, in Raw/211026/C0/N0/:
  Audio/take_NNN_{guid}.wav   one file per take (two files for some numbers)
  MIDI/takeN.mid              regions exported from one Logic session: note
                              times are on the session timeline, not the take's
  <song list>.txt             "N: composer, piece" for MIDI take N (in order)
No usable motion (one 16 s spreadsheet excerpt of take 1 only) and no video.

Pairing is checked by content: the MIDI onset train (relative to its first
note, NOTE_FS) is correlated with the audio onset strength at every offset;
only MIDI takes covering at least COVER of the audio are considered (a short
take matches somewhere inside any long one), and audio files and MIDI takes
are paired one to one (Hungarian). A pair is kept when it agrees with the take
number of both file names (`Take20-bis` counts as a take of 20); the offset is
refined at REFINE_FS within +-REFINE_S. Audio left without a MIDI take is
listed under `unpaired_audio` and not used.

The MIDI is moved onto its audio's clock (every event shifted by one offset).
Canonical stem: 211026_C0_N0_T9_P{piece}_t{take}, piece = rank of the piece
in the song list, take = order of the piece's takes on the session timeline.

Output: Data/Pipeline/01_ingest/Normalized/211026/C0_N0/{audio,midi}/ + _mapping.json
"""
from __future__ import annotations

import functools
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pretty_midi
from scipy.optimize import linear_sum_assignment

_P = Path(__file__).resolve()
while _P.name != "pipeline":
    _P = _P.parent
sys.path.insert(0, str(_P))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # pipeline/01_ingest/ (src)
from _common.normalized_io import RAW_ROOT, session_dir, write_mapping  # noqa: E402
from _common import motion_midi_xcorr as mx  # noqa: E402
from src.audio_midi_onsets import onset_strength, onset_train_at, shift_midi  # noqa: E402

print = functools.partial(print, flush=True)

DATASET, SESSION = "211026", "C0_N0"
BASE = RAW_ROOT / DATASET / "C0" / "N0"
FS = mx.NOTE_FS
COVER = 0.6
PAD_S = 10
REFINE_FS = 500
REFINE_S = 0.1


def audio_take(p: Path) -> int:
    return int(re.search(r"take_0*(\d+)_", p.name).group(1))


def midi_take(p: Path) -> int:
    return int(re.search(r"(\d+)", p.stem).group(1))


def song_list() -> list[str]:
    """Entries in order (the file numbers them 1..N; one line holds two entries)."""
    txt = next(BASE.glob("*.txt")).read_text(encoding="utf-8", errors="replace")
    return [m.group(2).strip() for m in re.finditer(r"(\d+):\s*(.*?)(?=\d+:|\n|$)", txt) if m.group(2).strip()]


def run(dataset: str | None = None, force: bool = False):
    if dataset and dataset != DATASET:
        return
    print(f"Stage 01 s05 -- {DATASET} audio <-> session MIDI pairing")
    auds = sorted((BASE / "Audio").glob("*.wav"), key=lambda p: (audio_take(p), p.name))
    mids = sorted((BASE / "MIDI").glob("*.mid"),
                  key=lambda p: min(n.start for i in pretty_midi.PrettyMIDI(str(p)).instruments for n in i.notes))
    onsets = [np.array(sorted(n.start for i in pretty_midi.PrettyMIDI(str(p)).instruments for n in i.notes)) for p in mids]
    A = [onset_strength(a, FS) for a in auds]
    S = np.full((len(auds), len(mids)), -1.0)
    O = np.zeros_like(S)
    for i, a in enumerate(A):
        padded = np.concatenate([np.zeros(PAD_S * FS), a, np.zeros(PAD_S * FS)])
        for j, ons in enumerate(onsets):
            m = mx.onset_train(ons - ons[0])
            if len(m) < COVER * len(a) or len(m) >= len(padded):
                continue
            r = mx.slide_ncc(padded, m)
            k = int(np.argmax(r))
            S[i, j], O[i, j] = r[k], k / FS - PAD_S            # first MIDI note at audio time O
    rows, cols = linear_sum_assignment(-S)
    names = song_list()
    piece_rank = {n: k for k, n in enumerate(dict.fromkeys(names), 1)}
    takes_seen: dict[int, int] = {}
    out = session_dir(DATASET, SESSION, modalities=["audio", "midi"])
    trials, used = [], set()
    pair = dict(zip(cols, rows))                                 # midi index -> audio index
    for j, mp in enumerate(mids):                               # session order
        i = pair.get(j)
        if i is None or S[i, j] < 0 or audio_take(auds[i]) != midi_take(mp):
            print(f"  {mp.name}: no audio agrees ({'none' if i is None else auds[i].name})")
            continue
        used.add(i)
        # refine the offset at REFINE_FS
        a = onset_strength(auds[i], REFINE_FS)
        m = onset_train_at(onsets[j] - onsets[j][0], REFINE_FS)
        c = int(round((O[i, j] + PAD_S) * REFINE_FS))
        h = int(REFINE_S * REFINE_FS)
        padded = np.concatenate([np.zeros(PAD_S * REFINE_FS), a, np.zeros(PAD_S * REFINE_FS)])
        seg = padded[max(0, c - h):c + h + len(m)]
        r = mx.slide_ncc(seg, m)
        k = int(np.argmax(r))
        offset = (max(0, c - h) + k) / REFINE_FS - PAD_S
        name = names[midi_take(mp) - 1]
        p = piece_rank[name]
        takes_seen[p] = takes_seen.get(p, 0) + 1
        stem = f"{DATASET}_{SESSION}_T9_P{p}_t{takes_seen[p]}"
        dst_a = out / "audio" / f"{stem}.wav"
        if force or not dst_a.exists():
            shutil.copy2(auds[i], dst_a)
        shift_midi(mp, out / "midi" / f"{stem}.mid", offset - onsets[j][0])
        others = np.delete(S[i], j)
        trials.append({
            "trial_stem": stem, "raw_stem": mp.stem, "task": 9, "status": "audio_midi",
            "modalities": ["audio", "midi"], "piece_name": name,
            "raw_sources": {"audio": auds[i].name, "midi": mp.name},
            "audio_midi_match": {"ncc": round(float(S[i, j]), 3), "margin": round(float(S[i, j] - others.max()), 3),
                                 "first_note_in_audio_sec": round(offset, 3),
                                 "midi_session_time_of_first_note_sec": round(float(onsets[j][0]), 3)},
        })
        print(f"  {stem}: {auds[i].name[:12]} + {mp.name}  r={S[i, j]:.2f}  first note at {offset:.3f}s  ({name})")
    unpaired = [a.name for k, a in enumerate(auds) if k not in used]
    write_mapping(DATASET, SESSION, trials, method="211026_content_pairing", extra={
        "unpaired_audio": unpaired,
        "capabilities": {"has_timecode": False, "has_video": False, "has_body_motion": False,
                         "has_hand_motion": False, "matching_method": "audio onset strength vs MIDI onsets"},
        "notes": ["MIDI takes are session-timeline exports; each is shifted onto its audio's clock.",
                  "No motion: only a 16 s spreadsheet excerpt of take 1 exists; not used."],
    })
    print(f"  {len(trials)} trials; unpaired audio: {unpaired}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    run(force=ap.parse_args().force)
