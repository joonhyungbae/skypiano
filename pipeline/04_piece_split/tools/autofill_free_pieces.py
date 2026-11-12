"""
Auto-label Free pieces using Piece Selection - Free.csv + temporal ordering.

User hint: Free pieces are played at the END of each session (after Set1).

Algorithm:
  1. Parse Piece Selection - Free.csv → per-subject Free piece list
  2. For each subject, find docx Free trials (task=9, status=ok) in trial_labels.json
     — they are the LAST N trials of the session.
  3. Match in docx-trial order to CSV order:
     - N=1 CSV entry, N=1 trial → all pieces in that trial = that CSV piece
       (e.g. sub06 Scriabin "both movements" covers multiple split pieces)
     - N=2 CSV entries, N=2 trials → trial[0] = CSV[0], trial[1] = CSV[1]
  4. Write labels into set1_cluster_labels.json for the corresponding clusters.

This bypasses Set1 pool matching for Free clusters (which was inappropriate —
Free pieces are subject-specific, not from the shared Set1 pool).
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

FREE_CSV = Path("Data/Metadata/Piece Selection - Free.csv")
TRIAL_LABELS = Path("Data/Metadata/240828/trial_labels.json")
TRIAL_MAP = Path("Data/Metadata/240828/trial_piece_map.json")
CLUSTER_LABELS = Path("Data/Metadata/240828/set1_cluster_labels.json")


def parse_free_csv() -> dict[str, list[str]]:
    """Return sub → list of Free piece titles in CSV order."""
    per_sub: dict[str, list[str]] = {}
    current_sub = None
    with open(FREE_CSV, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)   # header
        for row in reader:
            if not row or not row[0] and not row[1]:
                continue
            sub, title = row[0].strip(), row[1].strip()
            if sub:
                current_sub = sub
            if current_sub and title:
                per_sub.setdefault(current_sub, []).append(title)
    return per_sub


def main():
    free_per_sub = parse_free_csv()
    print("Free.csv per-subject:")
    for sub, titles in free_per_sub.items():
        print(f"  {sub}: {len(titles)} piece(s)")
        for t in titles:
            print(f"    - {t}")

    trial_labels = json.loads(TRIAL_LABELS.read_text(encoding="utf-8"))["240828"]
    tmap = json.loads(TRIAL_MAP.read_text(encoding="utf-8"))
    labels_data = json.loads(CLUSTER_LABELS.read_text(encoding="utf-8"))
    labels = labels_data.setdefault("labels", {})

    # For each sub: match Free trials (task=9, ok) to CSV entries in order
    applied = []
    for sub, trials in trial_labels.items():
        free_ok = [
            tid for tid, v in sorted(trials.items())
            if v["task"] == 9 and v.get("status") == "ok"
        ]
        if not free_ok:
            continue
        titles = free_per_sub.get(sub, [])
        if not titles:
            continue

        # Mapping rule
        if len(titles) == 1:
            # e.g. sub06: 1 piece across all Free trials (possibly multi-movement)
            trial_to_title = {tid: titles[0] for tid in free_ok}
        elif len(titles) == len(free_ok):
            # 1:1 mapping in CSV order
            trial_to_title = dict(zip(free_ok, titles))
        else:
            print(f"  [{sub}] mismatch: {len(titles)} titles vs {len(free_ok)} Free trials — skip")
            continue

        # For each Free trial, find its cluster IDs in trial_piece_map, apply label
        for tid, title in trial_to_title.items():
            if sub not in tmap or tid not in tmap[sub]:
                continue
            trial_entry = tmap[sub][tid]
            if trial_entry.get("task") != "Free":
                continue
            # pieces may be list[str] (pre-populated from CSV) or list[dict] (from cluster autofill)
            pieces = trial_entry.get("pieces") or []
            for piece in pieces:
                if isinstance(piece, dict):
                    cid = piece.get("cluster_id")
                    if cid and labels.get(cid) != title:
                        labels[cid] = title
                        applied.append((sub, tid, cid, title))
                    # stamp label into piece record too
                    piece["label"] = title
            # Also stamp trial-level label directly (metadata)
            trial_entry["free_piece_title"] = title

    print(f"\nApplied {len(applied)} Free cluster labels:")
    for sub, tid, cid, title in applied:
        print(f"  {sub}/{tid}  cluster={cid}  =>  {title}")

    # Save
    with open(TRIAL_MAP, "w", encoding="utf-8") as f:
        json.dump(tmap, f, ensure_ascii=False, indent=2)
    with open(CLUSTER_LABELS, "w", encoding="utf-8") as f:
        json.dump(labels_data, f, ensure_ascii=False, indent=2)

    print(f"\nUpdated: {TRIAL_MAP}")
    print(f"Updated: {CLUSTER_LABELS}")


if __name__ == "__main__":
    main()
