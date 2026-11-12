"""
Run MIDI fingerprint matching across all 240828 session MIDIs and save results.

Output: Data/Metadata/240828/fingerprint_matches.json — raw hits per session.
This is intermediate; the next step folds these into trial_piece_map.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
from src.midi_fingerprint import build_db, match  # noqa: E402

QUERY_MIDIS = {
    "sub04": "Data/Raw/240828/C0/N4/MIDI/240828sub04.mid",
    "sub05": "Data/Raw/240828/C0/N5/MIDI/240828sub05.mid",
    "sub06": "Data/Raw/240828/C0/N6/MIDI/240828sub06.mid",
    "sub07": "Data/Raw/240828/C0/N7/MIDI/240828sub07.mid",
    "sub08": "Data/Raw/240828/C0/N8/MIDI/240828sub08.mid",
}

OUT_PATH = Path("Data/Metadata/240828/fingerprint_matches.json")
MIN_RATIO = 0.30


def main():
    db, meta = build_db()
    print()
    out: dict = {
        "_comment": "Raw fingerprint matches: each anchor (piece+take) → estimated query_start_sec in 240828 session MIDI. Use as input for trial_piece_map.json fill-in.",
        "min_ratio": MIN_RATIO,
        "anchor_root": "Data/Anchors",
        "sessions": {},
    }
    for sub, mid in QUERY_MIDIS.items():
        if not Path(mid).is_file():
            print(f"[{sub}] MIDI not found: {mid}")
            continue
        hits = match(mid, db, meta, min_ratio=MIN_RATIO)
        out["sessions"][sub] = {
            "query_midi": mid,
            "match_count": len(hits),
            "matches": [
                {
                    "query_start_sec": round(h.query_start_sec, 3),
                    "piece_id": h.piece_id,
                    "anchor_take": h.take_id,
                    "anchor_duration_sec": round(h.anchor_duration_sec, 3),
                    "n_votes": h.n_votes,
                    "hit_ratio": round(h.hit_ratio, 3),
                }
                for h in hits
            ],
        }
        print(f"[{sub}] {len(hits)} candidate matches")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
