"""
Apply HIGH-confidence auto-label suggestions to set1_cluster_labels.json.

HIGH confidence = score gap between top-1 and top-2 suggestions >= 0.20.
Apply pool_id as the label automatically; MED and LOW clusters kept for
manual decision.

Propagates HIGH labels through autofill_set1_clusters.py → trial_piece_map.json.
"""
from __future__ import annotations

import json
from pathlib import Path

LABELS = Path("Data/Metadata/240828/set1_cluster_labels.json")


def main():
    d = json.loads(LABELS.read_text(encoding="utf-8"))
    sug = d.get("suggestions", {})
    labels = d.setdefault("labels", {})

    applied = []
    for cid, candidates in sug.items():
        if not candidates or candidates[0].get("score") is None:
            continue
        scores = [c["score"] for c in candidates if c.get("score") is not None]
        if len(scores) < 2:
            continue
        gap = scores[0] - scores[1]
        if gap >= 0.20 and labels.get(cid) is None:
            labels[cid] = candidates[0]["pool_id"]
            applied.append((cid, candidates[0]["pool_id"], candidates[0]["title"], gap))

    # Sort by applied gap descending
    applied.sort(key=lambda x: -x[3])
    print(f"Auto-applied {len(applied)} HIGH-confidence labels:")
    for cid, pid, title, gap in applied:
        print(f"  {cid}  =>  {pid:<26}  gap=+{gap:.2f}  {title}")

    d["_autoapply_note"] = (
        f"{len(applied)} HIGH-confidence (score gap >= 0.20) labels auto-applied. "
        "Curator should review MED/LOW confidence clusters manually via suggestions field."
    )
    with open(LABELS, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
