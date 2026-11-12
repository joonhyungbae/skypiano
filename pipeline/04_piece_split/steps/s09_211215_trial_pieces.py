"""
Stage 04 step 09: pieces = trials (211215, 211026).

211215 and 211026 were recorded one piece per trial (named in Stage 01
`piece_name`; 211215: the piece info sheet, 211026: the song list). Each trial becomes one piece over its whole
audio; hand motion is cut on the MIDI clock with the trial's offset from
03_sync s05_trial_motion_sync (motion_t = midi_t + shift_sec) where the trial
has motion (211026 has none). Body markers of 211215 are part of the hand
capture and are not exported separately.

Outputs, in the layout of the other datasets:
    Data/Pipeline/04_piece_split/211215/{trial}_p1.wav / .mid / _hand.csv
    Data/Pipeline/04_piece_split/211215/_pieces.json

Usage (from the repository root):
  python pipeline/04_piece_split/steps/s09_211215_trial_pieces.py [--dataset 211026]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import wave
from pathlib import Path

import pandas as pd
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PIPE = Path("Data/Pipeline")
FPS = 120.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="211215", choices=["211215", "211026"])
    DS = ap.parse_args().dataset
    NORM = PIPE / "01_ingest" / "Normalized" / DS
    HAND = PIPE / "02_spatial" / "hand" / "Aligned" / DS
    SYNC = PIPE / "03_sync" / "midi_motion" / "TrialSync" / f"{DS}.json"
    OUT = PIPE / "04_piece_split" / DS
    OUT.mkdir(parents=True, exist_ok=True)
    sync = json.loads(SYNC.read_text(encoding="utf-8")) if SYNC.exists() else {}
    index = []
    for mp in sorted(NORM.glob("*/_mapping.json")):
        for t in json.loads(mp.read_text(encoding="utf-8"))["trials"]:
            stem = t["trial_stem"]
            wav, mid = mp.parent / "audio" / f"{stem}.wav", mp.parent / "midi" / f"{stem}.mid"
            if not wav.exists() or not mid.exists():
                continue
            with wave.open(str(wav)) as w:
                dur = w.getnframes() / w.getframerate()
            ps = f"{stem}_p1"
            shutil.copy2(wav, OUT / f"{ps}.wav")
            shutil.copy2(mid, OUT / f"{ps}.mid")
            shift = (sync.get(stem) or {}).get("shift_sec")
            hand = HAND / f"{stem}.csv"
            if shift is not None and hand.exists():
                df = pd.read_csv(hand)
                a, b = int(round(shift * FPS)), int(round((dur + shift) * FPS))
                part = df.iloc[max(0, a):max(0, b)].reset_index(drop=True)
                if a < 0:                               # motion starts after the audio: pad with empty frames
                    part = part.reindex(range(a, len(part))).reset_index(drop=True)
                part.to_csv(OUT / f"{ps}_hand.csv", index=False)
            onsets = [n.start for i in pretty_midi.PrettyMIDI(str(mid)).instruments for n in i.notes]
            index.append({"trial_stem": stem, "raw_stem": t.get("raw_stem"), "task": 9, "pieces": [{
                "piece_idx": 1, "start_sec": 0.0, "end_sec": round(dur, 3), "n_notes": len(onsets),
                "log_note": t.get("recording_memo"),
                "identified": {"resolved_piece_id": t.get("piece_name"), "resolution_source": "session_log"}}]})
    (OUT / "_pieces.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(index)} pieces -> {OUT}")


if __name__ == "__main__":
    main()
