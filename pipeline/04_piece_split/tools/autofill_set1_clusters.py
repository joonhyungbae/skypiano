"""
Auto-fill Set1 & Free entries in trial_piece_map.json with cluster_id placeholders.

Each Set1/Free piece gets assigned a stable cluster_id (from set1_clusters.json).
Curator only needs to map cluster_id → piece title once, in a separate label file:
  Data/Metadata/240828/set1_cluster_labels.json
  { "cl_<hash>": "Chopin Waltz Op.64 No.2", ... }

Trial-map entries will look like:
  "0424": {
    "task": "Set1",
    "pieces": [
      { "piece_idx": 1, "cluster_id": "cl_abc12345", "duration_sec": 54.2 }
    ],
    "_note": "auto-filled from cluster cl_abc12345 (size=5 tight)"
  }
"""
from __future__ import annotations

import json
from pathlib import Path

CLUSTERS_PATH = Path("Data/Metadata/240828/set1_ensemble_v2_clusters.json")
TRIAL_MAP = Path("Data/Metadata/240828/trial_piece_map.json")
LABELS_PATH = Path("Data/Metadata/240828/set1_cluster_labels.json")


def main():
    clusters = json.loads(CLUSTERS_PATH.read_text(encoding="utf-8"))["clusters"]
    tmap = json.loads(TRIAL_MAP.read_text(encoding="utf-8"))

    # Load existing labels (if any) so curator work carries over
    existing_labels: dict[str, str | None] = {}
    if LABELS_PATH.is_file():
        existing_labels = json.loads(LABELS_PATH.read_text(encoding="utf-8")).get("labels", {})

    # Build: piece_key → (cluster_id, cluster_size, confidence)
    piece_to_cluster: dict[str, tuple[str, int, str]] = {}
    for c in clusters:
        cid = c["stable_cluster_id"]
        # ensemble clusters don't have explicit confidence; derive from size + intra-edge count
        conf = "confirmed" if c["size"] >= 2 and len(c.get("intra_edges", [])) > 0 else "singleton"
        for m in c["members"]:
            piece_to_cluster[m["piece_key"]] = (cid, c["size"], conf)

    # Group per sub/trial to aggregate multi-piece trials
    # piece_key format: sub04_0424_p1
    stats = {"set1_filled": 0, "free_filled": 0, "missing": 0}
    for pk, (cid, size, conf) in piece_to_cluster.items():
        parts = pk.split("_")
        sub = parts[0]            # "sub04"
        trial = parts[1]          # "0424"
        piece_idx = int(parts[2].lstrip("p"))
        if sub not in tmap or trial not in tmap[sub]:
            stats["missing"] += 1
            continue
        trial_entry = tmap[sub][trial]
        task = trial_entry.get("task")

        # Find duration from cluster member info (we have it in clusters json)
        dur = None
        for c in clusters:
            if c["stable_cluster_id"] == cid:
                for m in c["members"]:
                    if m["piece_key"] == pk:
                        dur = m["duration_sec"]
                        break
                break

        piece_entry = {
            "piece_idx": piece_idx,
            "cluster_id": cid,
            "cluster_size": size,
            "cluster_confidence": conf,
            "duration_sec": dur,
            "label": existing_labels.get(cid),   # may be None → TBD
        }

        # Initialize trial pieces list if needed (preserving non-Set1/Free pieces)
        if task in ("Set1", "Free"):
            if trial_entry.get("pieces") is None or "cluster_id" not in str(trial_entry.get("pieces", "")):
                trial_entry["pieces"] = []
            # Avoid duplicates: remove existing entry for same piece_idx
            trial_entry["pieces"] = [p for p in trial_entry["pieces"] if p.get("piece_idx") != piece_idx]
            trial_entry["pieces"].append(piece_entry)
            # Sort by piece_idx
            trial_entry["pieces"].sort(key=lambda p: p.get("piece_idx", 0))
            trial_entry["_note"] = f"auto-clustered ({len(trial_entry['pieces'])} pieces)"
            if task == "Set1":
                stats["set1_filled"] += 1
            else:
                stats["free_filled"] += 1

    # Write trial_map
    tmap["_comment"] = (
        "trial → piece mapping for 240828. T1/T2 auto-filled from Stage 04 s03 fingerprint. "
        "Set1/Free auto-filled with cluster_id from set1_clusters.json — resolve cluster_id → "
        "piece title in set1_cluster_labels.json."
    )
    with open(TRIAL_MAP, "w", encoding="utf-8") as f:
        json.dump(tmap, f, ensure_ascii=False, indent=2)

    # Write label stub file (if missing) with one entry per unique cluster, sorted by size desc
    label_stub = {
        "_comment": (
            "Cluster ID → piece title mapping. Fill `label` field once per cluster_id; "
            "autofill_set1_clusters.py will propagate to all trial entries on next run."
        ),
        "labels": {},
    }
    # Order by cluster size desc so most-shared pieces come first
    sorted_clusters = sorted(clusters, key=lambda c: (-c["size"], c["stable_cluster_id"]))
    for c in sorted_clusters:
        cid = c["stable_cluster_id"]
        label_stub["labels"][cid] = existing_labels.get(cid)
        # Also include a preview of the cluster for curator context
    label_stub["cluster_context"] = {
        c["stable_cluster_id"]: {
            "size": c["size"],
            "n_subjects": c["n_subjects"],
            "tasks": c["tasks"],
            "duration_median_sec": c["signature"]["duration_median"],
            "duration_range_sec": [c["signature"]["duration_min"], c["signature"]["duration_max"]],
            "notes_median": c["signature"].get("notes_median", None),
            "confidence": c.get("confidence", "confirmed" if c["size"] >= 2 else "singleton"),
            "sample_members": [m["piece_key"] for m in c["members"][:3]],
        }
        for c in sorted_clusters
    }

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(label_stub, f, ensure_ascii=False, indent=2)

    print(f"trial_piece_map.json:")
    print(f"  Set1 pieces auto-filled: {stats['set1_filled']}")
    print(f"  Free pieces auto-filled: {stats['free_filled']}")
    print(f"  missing trial entries:   {stats['missing']}")
    print(f"\nset1_cluster_labels.json:")
    labeled = sum(1 for v in label_stub["labels"].values() if v)
    print(f"  {len(label_stub['labels'])} unique clusters, {labeled} already labeled")
    print(f"  -> curator fills 'labels.{{cluster_id}}' field to resolve titles")


if __name__ == "__main__":
    main()
