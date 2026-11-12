"""
Stage 04 step 15: 220819 technique drill names in the words of the score.

220819 pieces are named by their anchor id from the session log ("Scale_Slow",
"Octave Repeat_Slow", "Octave Repeat 2", ...). The anchor id stays the key of
the anchor library; this step adds `piece_name` to every located entry of
Data/Pipeline/04_piece_split/anchors_220819/locate_manifest.json: the id in
the words of the score (_common/technique_names.score_name), an octave repeat
played five bars long marked ", x5" (counted from the cut piece's MIDI).
Stage 05 and the review page show `piece_name`.

Rerun after the anchors pipeline has rewritten the manifest.

Usage (from the repository root):
  python pipeline/04_piece_split/steps/s15_220819_score_names.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _common.technique_names import chord_count, score_name, with_repeats  # noqa: E402

PIECES = Path("Data/Pipeline/04_piece_split")
MANIFEST = PIECES / "anchors_220819" / "locate_manifest.json"


def main():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    changed = Counter()
    for e in m.get("entries", []):
        if not e.get("piece_id"):
            continue
        name = score_name(e["piece_id"])
        midi = PIECES / "220819" / f"{e['out_stem']}.mid" if e.get("ok") and e.get("out_stem") else None
        if midi is not None and midi.exists():
            onsets = [n.start for i in pretty_midi.PrettyMIDI(str(midi)).instruments for n in i.notes]
            name = with_repeats(name, chord_count(onsets))
        e["piece_name"] = name
        if name != e["piece_id"]:
            changed[(e["piece_id"], name)] += 1
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    for (a, b), n in sorted(changed.items()):
        print(f"  {n:3d}  {a} -> {b}")


if __name__ == "__main__":
    main()
