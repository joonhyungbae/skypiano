"""
Run segment-level MIDI fingerprint matching on all 240828 sessions.

Replaces the whole-session matcher (run_240828_fingerprint.py) which had poor
SNR on short T1 pieces. Here we segment by silence first, then identify each
segment against the anchor library.

Output: Data/Metadata/240828/fingerprint_matches.json
  Top-level structure:
    {
      "gap_threshold": 3.0,
      "anchor_root": "Data/Anchors",
      "confidence_thresholds": {...},
      "sessions": {
        "sub04": {
          "query_midi": "...",
          "segment_count": N,
          "segments": [ {index, start, end, dur, confidence, top, candidates}, ... ]
        }, ...
      }
    }

Confidence tiers:
  high   score ≥ 0.40 and coherent_votes ≥ 5 and anchor_fps ≥ 10
  medium score ≥ 0.20 and coherent_votes ≥ 3
  low    coherent_votes ≥ 2
  none   no candidate above low threshold
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # pipeline/04_piece_split/
from src.midi_fingerprint import build_db  # noqa: E402
from midi_segment_match import match_session, SegmentMatch  # noqa: E402

QUERY_MIDIS = {
    "sub04": "Data/Raw/240828/C0/N4/MIDI/240828sub04.mid",
    "sub05": "Data/Raw/240828/C0/N5/MIDI/240828sub05.mid",
    "sub06": "Data/Raw/240828/C0/N6/MIDI/240828sub06.mid",
    "sub07": "Data/Raw/240828/C0/N7/MIDI/240828sub07.mid",
    "sub08": "Data/Raw/240828/C0/N8/MIDI/240828sub08.mid",
}

OUT_PATH = Path("Data/Metadata/240828/fingerprint_matches.json")
GAP_THRESHOLD = 3.0
TOP_K = 5


def classify(cand: dict | None) -> str:
    if cand is None:
        return "none"
    score = cand["score"]
    coherent = cand["coherent"]
    afps = cand["anchor_fps"]
    if score >= 0.40 and coherent >= 5 and afps >= 10:
        return "high"
    if score >= 0.20 and coherent >= 3:
        return "medium"
    if coherent >= 2:
        return "low"
    return "none"


def serialize(sm: SegmentMatch) -> dict:
    top = sm.candidates[0] if sm.candidates else None
    return {
        "seg_index": sm.seg_index,
        "seg_start": round(sm.seg_start, 2),
        "seg_end": round(sm.seg_end, 2),
        "seg_duration": round(sm.seg_duration, 2),
        "seg_n_events": sm.seg_n_events,
        "seg_n_fingerprints": sm.seg_n_fingerprints,
        "confidence": classify(top),
        "top_piece": top["piece_id"] if top else None,
        "top_score": top["score"] if top else None,
        "candidates": sm.candidates,
    }


def summarize(segments: list[dict]) -> dict:
    tier_count: dict[str, int] = {"high": 0, "medium": 0, "low": 0, "none": 0}
    for s in segments:
        tier_count[s["confidence"]] += 1
    return tier_count


def main():
    db, meta = build_db()
    out = {
        "_comment": "Segment-level MIDI fingerprint matches. Each segment = a silence-bounded phrase (piece or take).",
        "gap_threshold_sec": GAP_THRESHOLD,
        "anchor_root": "Data/Anchors",
        "confidence_thresholds": {
            "high": "score >= 0.40 AND coherent_votes >= 5 AND anchor_fps >= 10",
            "medium": "score >= 0.20 AND coherent_votes >= 3",
            "low": "coherent_votes >= 2",
            "none": "below all thresholds",
        },
        "sessions": {},
    }
    print()
    for sub, mid in QUERY_MIDIS.items():
        if not Path(mid).is_file():
            print(f"[{sub}] MIDI not found: {mid}")
            continue
        results = match_session(mid, db, meta, gap_threshold=GAP_THRESHOLD, top_k=TOP_K)
        segs = [serialize(sm) for sm in results]
        tiers = summarize(segs)
        out["sessions"][sub] = {
            "query_midi": mid,
            "segment_count": len(segs),
            "tier_counts": tiers,
            "segments": segs,
        }
        print(f"[{sub}] {len(segs)} segs | "
              f"high={tiers['high']:>3}  med={tiers['medium']:>3}  "
              f"low={tiers['low']:>3}  none={tiers['none']:>3}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
