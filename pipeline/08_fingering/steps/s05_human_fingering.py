"""
Stage 08 Step 05: human fingering labels per released performance.

The paper's audit labeled every note of three 240828 graded performances by
hand in the annotation tool (web/annotate). The labels come as one flat file,

    Data/Pipeline/08_fingering/Human/fingering.json
        {"<sample stem>#<note index>": finger_int, ...}
        finger_int 1..5 = L1..L5, 6..10 = R1..R5 (thumb .. little finger)

keyed by the note index of the tool's note list (Data/web_annotate/data/
notes.json, which also names each sample's source piece stem). That list
orders the notes of a chord differently from the release, so a label is
matched to its release note by (onset, pitch); notes that share both (a
doubled MIDI note) are matched in order. The release note list is the one
Step 04 wrote (Notes/raw/{dataset}/{stem}.csv), which follows midi.mid.

Output, one CSV per labeled performance (keyed by the source piece stem, as
pipeline/12_release/build_manifest.py keys its sources):
    Data/Pipeline/08_fingering/Notes/human/{dataset}/{stem}.csv
        note_idx  onset  offset  pitch   as in fingering.csv
        hand      L | R
        finger    1 (thumb) .. 5 (little finger)

Every note must be matched and every label used, or the step stops.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

LABELS = Path("Data/Pipeline/08_fingering/Human/fingering.json")
NOTES = Path("Data/web_annotate/data/notes.json")
RELEASE_NOTES = Path("Data/Pipeline/08_fingering/Notes/raw")
OUT = Path("Data/Pipeline/08_fingering/Notes/human")


def key(onset: float, pitch: int) -> tuple[float, int]:
    return round(float(onset), 4), int(pitch)


def main() -> None:
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    notes = json.loads(NOTES.read_text(encoding="utf-8"))["notes"]
    by_stem: dict[str, list[dict]] = defaultdict(list)
    for n in notes:
        by_stem[n["trial_alias"]].append(n)
    used = 0
    for stem, tool_notes in by_stem.items():
        ds = stem.split("_")[0]
        src = RELEASE_NOTES / ds / f"{stem}.csv"
        rows = list(csv.DictReader(src.open(encoding="utf-8")))
        if len(rows) != len(tool_notes):
            sys.exit(f"{stem}: {len(tool_notes)} labeled notes, {len(rows)} in {src}")
        slots: dict[tuple, list[dict]] = defaultdict(list)
        for r in rows:
            slots[key(r["onset"], r["pitch"])].append(r)
        out = {}
        for n in sorted(tool_notes, key=lambda n: n["note_idx"]):
            label = labels.get(f"{n['trial']}#{n['note_idx']}")
            if label is None or not 1 <= int(label) <= 10:
                sys.exit(f"{n['trial']}#{n['note_idx']}: no finger label ({label!r})")
            free = slots.get(key(n["onset_sec"], n["pitch"]))
            if not free:
                sys.exit(f"{n['trial']}#{n['note_idx']}: no release note at {n['onset_sec']} s, pitch {n['pitch']}")
            r = free.pop(0)
            v = int(label)
            out[int(r["note_idx"])] = {"note_idx": r["note_idx"], "onset": r["onset"], "offset": r["offset"],
                                       "pitch": r["pitch"], "hand": "L" if v <= 5 else "R",
                                       "finger": v if v <= 5 else v - 5}
            used += 1
        dst = OUT / ds / f"{stem}.csv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["note_idx", "onset", "offset", "pitch", "hand", "finger"])
            w.writeheader()
            w.writerows(out[i] for i in sorted(out))
        print(f"{stem}: {len(out)} notes -> {dst}")
    if used != len(labels):
        sys.exit(f"{len(labels) - used} labels in {LABELS} matched no note")


if __name__ == "__main__":
    main()
