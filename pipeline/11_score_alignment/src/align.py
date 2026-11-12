"""Score↔performance alignment via Parangonar.

`align_one(score_path, midi_path)` returns a dict ready for json.dump.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import partitura as pt
from parangonar import AutomaticNoteMatcher


def align_one(
    score_path: str | Path,
    midi_path: str | Path,
    matcher_name: str = "AutomaticNoteMatcher",
) -> dict[str, Any]:
    score = pt.load_score(str(score_path))
    score_na = score.note_array()

    performance = pt.load_performance(str(midi_path))
    perf_na = performance.note_array()

    if matcher_name != "AutomaticNoteMatcher":
        # Other matchers exist (TheGlueNoteMatcher, DualDTWNoteMatcher, ...);
        # add wiring here when we want to compare.
        raise NotImplementedError(matcher_name)
    matcher = AutomaticNoteMatcher()
    alignment = matcher(score_na, perf_na, verbose_time=False)

    perf_idx = {row["id"]: row for row in perf_na}
    score_idx = {row["id"]: row for row in score_na}

    matches: list[dict[str, Any]] = []
    n_match = n_ins = n_del = 0
    for entry in alignment:
        label = entry.get("label")
        if label == "match":
            n_match += 1
            sid = entry["score_id"]
            pid = entry["performance_id"]
            s = score_idx.get(sid)
            p = perf_idx.get(pid)
            if s is None or p is None:
                continue
            matches.append(
                {
                    "score_id": str(sid),
                    "score_pitch": int(s["pitch"]),
                    # onset_quarter is always in quarter notes; onset_beat is
                    # in time-signature beat units which differ per meter.
                    # Web client converts quarters → whole notes by /4 to match
                    # OSMD's iterator timestamps.
                    "score_onset_quarter": float(s["onset_quarter"]),
                    "midi_pitch": int(p["pitch"]),
                    "midi_time_sec": float(p["onset_sec"]),
                    "midi_duration_sec": float(p["duration_sec"]),
                }
            )
        elif label == "insertion":
            n_ins += 1
        elif label == "deletion":
            n_del += 1

    matches.sort(key=lambda m: m["midi_time_sec"])

    return {
        "version": 1,
        "matcher": matcher_name,
        "score": Path(score_path).name,
        "midi": Path(midi_path).name,
        "stats": {
            "score_notes": int(len(score_na)),
            "perf_notes": int(len(perf_na)),
            "matches": n_match,
            "score_match_rate": n_match / max(1, len(score_na)),
            "perf_match_rate": n_match / max(1, len(perf_na)),
            "insertions": n_ins,
            "deletions": n_del,
        },
        "matches": matches,
    }
