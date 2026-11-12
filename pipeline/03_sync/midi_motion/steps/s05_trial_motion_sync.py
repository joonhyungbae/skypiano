"""
MIDI <-> hand-motion offset per trial, for cohorts recorded per trial without
a shared clock (211215: separate audio/MIDI and motion files per trial; the
MIDI starts at the first note, the motion capture starts and stops on its own).

Two signals are scored at every offset (motion_t = midi_t + shift_sec):

  r    note-level key-press correlation (motion_midi_xcorr): the summed
       downward fingertip speed of the Stage 02 hand file (Z up, 120 fps)
       against the MIDI onset train at NOTE_FS; the key-press lead
       (PRESS_LEAD_S) is taken out.
  hit  key position: the share of notes for which a fingertip lies over the
       played key at the note's onset. Keys are laid out along the keyboard
       end markers (LEnd -> REnd, 23.5 mm white keys); the position of A0 on
       that axis is the most common fingertip-minus-key offset at this shift.

Short trials of repetitive material give onset correlations of similar
height at many offsets, some placing the motion over silence; the key
position separates them. Offsets are only searched where the motion holds at
least COVER_FRAC of the notes it can hold at best (the capture covers the
playing), and the offset maximises hit + max(r, 0).

  shift_sec  chosen offset
  ncc, hit   r and hit there
  margin     score minus the best score more than 1 s away
  status     `clear` (hit >= CLEAR_HIT and margin >= CLEAR_MARGIN) or `check`;
             it only orders the review by eye (the sync review page).

Output: Data/Pipeline/03_sync/midi_motion/TrialSync/{dataset}.json

Usage (from the repository root):
  python pipeline/03_sync/midi_motion/steps/s05_trial_motion_sync.py --dataset 211215
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # pipeline/
from _common import motion_midi_xcorr as mx  # noqa: E402

PIPE = Path("Data/Pipeline")
NORM = PIPE / "01_ingest" / "Normalized"
HAND = PIPE / "02_spatial" / "hand" / "Aligned"
OUT = PIPE / "03_sync" / "midi_motion" / "TrialSync"
MOTION_FPS = 120.0
PAD_S = 60
COVER_FRAC = 0.9
MIN_NOTES = mx.KEY_MIN_NOTES
CLEAR_HIT = 0.85
CLEAR_MARGIN = 0.3


def lateral(hand: Path, tips: np.ndarray) -> np.ndarray:
    """Fingertip position along the keyboard axis (LEnd -> REnd), mm."""
    df = pd.read_csv(hand, usecols=[f"{m}_{a}" for m in ("LEnd", "REnd") for a in "xyz"])
    le = np.nanmedian(df[["LEnd_x", "LEnd_y", "LEnd_z"]].to_numpy(float), axis=0)
    re_ = np.nanmedian(df[["REnd_x", "REnd_y", "REnd_z"]].to_numpy(float), axis=0)
    return mx.keyboard_lateral(tips, le, re_)


def run(ds: str) -> None:
    fs = mx.NOTE_FS
    out = {}
    for mp in sorted((NORM / ds).glob("*/_mapping.json")):
        for t in json.loads(mp.read_text(encoding="utf-8"))["trials"]:
            stem = t["trial_stem"]
            mid, hand = mp.parent / "midi" / f"{stem}.mid", HAND / ds / f"{stem}.csv"
            if not mid.exists() or not hand.exists():
                continue
            notes = sorted((n.start, n.pitch) for i in pretty_midi.PrettyMIDI(str(mid)).instruments for n in i.notes)
            if len(notes) < MIN_NOTES:
                continue
            onsets = np.array([s for s, _ in notes])
            keys = np.array([mx.key_center(p) for _, p in notes])
            tips = mx.fingertips_from_aligned_csv(hand)
            lat = lateral(hand, tips)
            p = mx.press_train(tips, MOTION_FPS, up_axis=2)
            long = np.concatenate([np.zeros(PAD_S * fs), mx.onset_train(onsets), np.zeros(PAD_S * fs + len(p))])
            r = mx.slide_ncc(long, p)
            # motion t = 0 sits at MIDI time k/fs - PAD_S (key presses lead the sound)
            shifts = PAD_S - np.arange(len(r)) / fs - mx.PRESS_LEAD_S
            dur = len(tips) / MOTION_FPS
            cover = (np.searchsorted(onsets, -shifts + dur, "right") - np.searchsorted(onsets, -shifts, "left")) \
                / len(onsets)
            feasible = cover >= COVER_FRAC * cover.max()
            hit = np.zeros(len(r))
            for k in np.flatnonzero(feasible):
                hit[k] = mx.key_hit(lat, MOTION_FPS, onsets, keys, shifts[k])
            score = np.where(feasible, hit + np.clip(r, 0, None), -1.0)
            k = int(np.argmax(score))
            rest = score.copy()
            rest[max(0, k - fs):k + fs] = -1
            margin = float(score[k] - max(rest.max(), 0.0))
            out[stem] = {"shift_sec": round(float(shifts[k]), 3), "ncc": round(float(r[k]), 3),
                         "hit": round(float(hit[k]), 3), "margin": round(margin, 3),
                         "status": "clear" if hit[k] >= CLEAR_HIT and margin >= CLEAR_MARGIN else "check"}
            print(f"  {stem}: shift {shifts[k]:+7.2f}s  hit {hit[k]:.2f}  r={r[k]:.2f}  margin {margin:.2f}  "
                  f"{out[stem]['status']}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{ds}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    n = sum(v["status"] == "clear" for v in out.values())
    print(f"{len(out)} trials ({n} clear) -> {OUT / f'{ds}.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    run(ap.parse_args().dataset)
