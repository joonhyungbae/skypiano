"""
Stage 04 step 07: cut the session notes' "discard" intervals out of pieces.

The recording session's labeling notes (240828: `[yamaha] 240828 trial
labeling.docx`, parsed into each trial's `exclusion_intervals` in the Stage
01 mapping, session time) mark stretches to drop, e.g. "24:43-25:35 marker
fixing". Whole-take discards are already left out at Stage 01. This step
handles a discard interval INSIDE a piece window:

  * the piece keeps its part before the interval;
  * the part after it becomes a new piece if it has notes and is not
    already covered by another piece (a later piece of the trial);
  * pieces whose edge merely touches an interval are NOT changed (the notes
    are written to the second, so an overlap of a few seconds may be the
    note-taker's rounding or a false start); they are listed for review.

Changed pieces are re-cut (audio, MIDI, hand, body) the same way as s02; their
previous files and the previous `_pieces.json` go to Data/Pipeline/_backups.
Per-piece videos of changed pieces are moved aside too (re-cut with the video
stage). Applied changes are recorded in `_discard_applied.json`.

Usage (from the repository root):
  python pipeline/04_piece_split/steps/s07_apply_discard_intervals.py --dataset 240828 [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import pretty_midi
from pydub import AudioSegment

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _common.slicing import slice_audio, slice_midi  # noqa: E402

PIPE = Path("Data/Pipeline")
NORM = PIPE / "01_ingest" / "Normalized"
PIECES = PIPE / "04_piece_split"
HAND = PIPE / "02_spatial" / "hand" / "Aligned"
BODY = PIPE / "02_spatial" / "body" / "Aligned"
SYNC_MIDI = PIPE / "03_sync" / "midi_motion" / "Aligned"
MOTION_FPS = 120.0
INSIDE_MARGIN_S = 1.0     # the interval must lie this far inside the piece on both sides
MIN_NEW_NOTES = 5


def cut_motion(src: Path, start: float, end: float, dst: Path) -> None:
    df = pd.read_csv(src)
    if "TimeCode" in df.columns:
        df = df.drop(columns=["TimeCode"])
    df.iloc[int(start * MOTION_FPS):int(end * MOTION_FPS)].reset_index(drop=True).to_csv(dst, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ds = a.dataset
    trials = {}
    for mp in (NORM / ds).glob("*/_mapping.json"):
        for t in json.loads(mp.read_text(encoding="utf-8"))["trials"]:
            trials[t["trial_stem"]] = t
    pj = PIECES / ds / "_pieces.json"
    pieces = json.loads(pj.read_text(encoding="utf-8"))
    changes, edge = [], []
    for tr in pieces:
        t = trials.get(tr["trial_stem"], {})
        s0 = t.get("session_start_sec") or 0.0
        ivs = [(x - s0, y - s0) for x, y in t.get("exclusion_intervals") or []]
        for p in list(tr["pieces"]):
            for x, y in ivs:
                ov = min(p["end_sec"], y) - max(p["start_sec"], x)
                if ov <= 0.05:
                    continue
                if x >= p["start_sec"] + INSIDE_MARGIN_S and y <= p["end_sec"] - INSIDE_MARGIN_S:
                    changes.append((tr, p, x, y))
                else:
                    edge.append({"piece": f"{tr['trial_stem']}_p{p['piece_idx']}", "overlap_s": round(ov, 2),
                                 "interval_trial_s": [round(x, 2), round(y, 2)]})
    print(f"{ds}: {len(changes)} pieces with a discard interval inside; {len(edge)} edge overlaps left for review")
    for e in edge:
        print("   edge:", e)
    if a.dry_run or not changes:
        return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    bak = PIPE / "_backups" / f"{stamp}_discard_intervals_{ds}"
    bak.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pj, bak / "_pieces.json")
    applied = []
    for tr, p, x, y in changes:
        stem, raw = tr["trial_stem"], tr.get("raw_stem")
        others = [q for q in tr["pieces"] if q is not p]
        old = (p["start_sec"], p["end_sec"])
        midi_src = SYNC_MIDI / ds / f"{raw}.mid"
        onsets = sorted(n.start for i in pretty_midi.PrettyMIDI(str(midi_src)).instruments for n in i.notes)
        after = (y, old[1])
        covered = any(q["start_sec"] < after[1] and q["end_sec"] > after[0] for q in others)
        n_after = sum(1 for o in onsets if after[0] <= o < after[1])
        p["end_sec"] = round(x, 3)
        p["n_notes"] = sum(1 for o in onsets if p["start_sec"] <= o < p["end_sec"])
        new = None
        if n_after >= MIN_NEW_NOTES and not covered:
            new = {"piece_idx": max(q["piece_idx"] for q in tr["pieces"]) + 1,
                   "start_sec": round(after[0], 3), "end_sec": round(after[1], 3), "n_notes": n_after,
                   "excluded": False, "identified": {"resolved_piece_id": None, "resolution_source": "split_by_session_notes"}}
            tr["pieces"].append(new)
        tr["pieces"].sort(key=lambda q: q["start_sec"])
        tr["n_pieces"] = len(tr["pieces"])
        # re-cut the changed / new pieces
        audio = AudioSegment.from_wav(str(PIECES / ds / f"{stem}.wav"))
        for q in [p] + ([new] if new else []):
            ps = f"{stem}_p{q['piece_idx']}"
            for f in sorted((PIECES / ds).glob(f"{ps}[._]*")) + sorted((PIECES / ds).glob(f"videos*/{ps}_*")):
                dst = bak / f.relative_to(PIECES / ds)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(dst))
            slice_audio(audio, q["start_sec"], q["end_sec"], PIECES / ds / f"{ps}.wav")
            slice_midi(midi_src, q["start_sec"], q["end_sec"], PIECES / ds / f"{ps}.mid")
            for kind, src in [("hand", HAND / ds / f"{stem}.csv"), ("body", BODY / ds / f"{raw}.csv")]:
                if src.exists():
                    cut_motion(src, q["start_sec"], q["end_sec"], PIECES / ds / f"{ps}_{kind}.csv")
        applied.append({"piece": f"{stem}_p{p['piece_idx']}", "old_window": old,
                        "new_window": [p["start_sec"], p["end_sec"]],
                        "discard_trial_s": [round(x, 2), round(y, 2)],
                        "after_part": ("new piece " + f"{stem}_p{new['piece_idx']}") if new else
                                      ("covered by another piece" if covered else f"dropped ({n_after} notes)")})
        print(f"   {applied[-1]}")
    pj.write_text(json.dumps(pieces, indent=2, ensure_ascii=False), encoding="utf-8")
    out = PIECES / ds / "_discard_applied.json"
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"applied": [], "edge_overlaps": []}
    prev["applied"] += applied
    prev["edge_overlaps"] = edge
    out.write_text(json.dumps(prev, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"   backup: {bak}")


if __name__ == "__main__":
    main()
