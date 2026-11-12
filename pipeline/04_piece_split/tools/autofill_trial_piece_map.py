"""
Auto-fill Data/Metadata/240828/trial_piece_map.json from Stage 04 Step 03's
piece identification results.

For each 240828 trial:
  - Read all pieces from `_pieces.json` (non-excluded)
  - Collect high/medium confidence piece_id's in piece-sequence order
  - Write into trial_piece_map.json with per-piece confidence
  - Set1/Free trials (no anchors) remain TBD — left untouched

Preserves existing Free mappings that were manually curated.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PIECES_MANIFEST = Path("Data/Pipeline/04_piece_split/240828/_pieces.json")
TRIAL_MAP = Path("Data/Metadata/240828/trial_piece_map.json")

TASK_INT_TO_NAME = {0: "Technique1", 1: "Technique2", 2: "Set1", 9: "Free"}
AUTO_FILL_TASKS = {"Technique1", "Technique2"}  # only these have anchors

RAW_STEM_RE = re.compile(r"^240828sub(\d{2})(\d{2})$")  # sub04xx → sub="04", trial="01"..


def trial_key(raw_stem: str) -> tuple[str, str] | None:
    m = RAW_STEM_RE.match(raw_stem)
    if not m:
        return None
    sub = f"sub{m.group(1)}"
    trial = f"{m.group(1)}{m.group(2)}"   # keeps full 4-digit docx trial id
    return sub, trial


def main():
    manifest = json.loads(PIECES_MANIFEST.read_text(encoding="utf-8"))
    tmap = json.loads(TRIAL_MAP.read_text(encoding="utf-8"))

    stats = {"filled": 0, "ambiguous": 0, "empty": 0, "skipped_task": 0}

    for entry in manifest:
        key = trial_key(entry["raw_stem"])
        if key is None:
            continue
        sub, trial = key
        if sub not in tmap or trial not in tmap[sub]:
            continue

        task_name = TASK_INT_TO_NAME.get(entry["task"])
        if task_name not in AUTO_FILL_TASKS:
            stats["skipped_task"] += 1
            continue

        # Already manually filled?
        current = tmap[sub][trial].get("pieces")
        note = tmap[sub][trial].get("_note", "")
        if current is not None and "TBD" not in str(note) and "auto" not in str(note):
            continue  # don't overwrite human edits

        # Collect high/medium ID'd pieces in piece_idx order
        accepted: list[dict] = []
        for p in sorted(entry["pieces"], key=lambda x: x["piece_idx"]):
            ident = p.get("identified") or {}
            conf = ident.get("confidence")
            pid = ident.get("piece_id")
            if conf in ("high", "medium") and pid:
                accepted.append({
                    "piece_id": pid,
                    "piece_idx": p["piece_idx"],
                    "confidence": conf,
                    "score": ident.get("candidates", [{}])[0].get("score"),
                })

        if accepted:
            tmap[sub][trial]["pieces"] = accepted
            tmap[sub][trial]["_note"] = f"auto-filled ({len(accepted)}/{len(entry['pieces'])} pieces identified)"
            stats["filled"] += 1
        else:
            tmap[sub][trial]["pieces"] = []
            tmap[sub][trial]["_note"] = f"auto-attempted (0/{len(entry['pieces'])} pieces identified - anchor misses)"
            stats["empty"] += 1

    tmap["_comment"] = (
        "trial → piece mapping for 240828. T1/T2 auto-filled from "
        "pipeline/04_piece_split s03 piece identification (high+medium confidence). "
        "Set1 + Free TBD (no anchor library). Free entries preserved from earlier manual fill."
    )
    tmap["_autofill_stats"] = stats

    with open(TRIAL_MAP, "w", encoding="utf-8") as f:
        json.dump(tmap, f, ensure_ascii=False, indent=2)

    print(f"trial_piece_map.json updated:")
    print(f"  filled:        {stats['filled']}")
    print(f"  empty (0 ids): {stats['empty']}")
    print(f"  skipped task:  {stats['skipped_task']}  (Set1/Free - no anchors)")
    print(f"  total T1/T2 attempted: {stats['filled'] + stats['empty']}")


if __name__ == "__main__":
    main()
