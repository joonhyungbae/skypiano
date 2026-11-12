"""
Match Set1 clusters to pool entries via profile distance.

For each cluster, compute compatibility score with each of 17 pool entries:
  - Key match (exact or enharmonic): +3 points, mode match: +1
  - Tempo in range:        1.0 if within, else gaussian decay
  - Duration compatible:   1.0 if cluster range overlaps pool range, decay otherwise
                           (allow cluster to be SHORTER than pool — excerpts)
  - Density in range:      1.0 if within, gaussian decay

Top-3 candidates per cluster written to set1_cluster_labels.json as
`suggestions`. Curator confirms/overrides `label` field.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "_common"))

from set1_pool_reference import SET1_POOL, key_match  # noqa: E402

PROFILES = Path("Data/Metadata/240828/set1_cluster_profiles.json")
LABELS = Path("Data/Metadata/240828/set1_cluster_labels.json")


def score_cluster_vs_pool(cluster: dict, pool_entry: dict) -> tuple[float, dict]:
    """Return (total_score, breakdown)."""
    # Key score
    key_ok = key_match(cluster["key"], pool_entry["key"])
    mode_ok = cluster["mode"] == pool_entry["mode"]
    key_score = (3.0 if key_ok else 0.0) + (1.0 if mode_ok else 0.0)
    key_score = key_score / 4.0   # normalize to 0..1

    # Tempo score
    lo, hi = pool_entry["tempo_range_bpm"]
    t = cluster["tempo_bpm_median"]
    # Account for half/double-time detection errors (librosa common issue)
    tempo_variants = [t, t / 2.0, t * 2.0]
    tempo_score = 0.0
    for tv in tempo_variants:
        if lo <= tv <= hi:
            tempo_score = 1.0
            break
        center = (lo + hi) / 2
        sigma = (hi - lo) / 2
        tempo_score = max(tempo_score, float(np.exp(-((tv - center) ** 2) / (2 * sigma ** 2))))

    # Duration score: cluster duration should be ≤ pool upper bound (excerpts allowed)
    lo, hi = pool_entry["duration_range_sec"]
    d_max = cluster["duration_range"][1]
    d_med = cluster["duration_median"]
    if d_max <= hi * 1.1:   # cluster fits within pool duration
        duration_score = 1.0 if d_med >= lo * 0.3 else 0.5
    else:
        duration_score = 0.3   # cluster is longer than pool full — unlikely

    # Density score
    lo, hi = pool_entry["note_density_range"]
    dd = cluster["note_density_median"]
    if lo <= dd <= hi:
        density_score = 1.0
    else:
        center = (lo + hi) / 2
        sigma = (hi - lo) / 2
        density_score = float(np.exp(-((dd - center) ** 2) / (2 * sigma ** 2)))

    total = key_score * 0.4 + tempo_score * 0.2 + duration_score * 0.2 + density_score * 0.2
    breakdown = {
        "key": round(key_score, 2),
        "tempo": round(tempo_score, 2),
        "duration": round(duration_score, 2),
        "density": round(density_score, 2),
        "total": round(total, 3),
    }
    return total, breakdown


def main():
    profiles = json.loads(PROFILES.read_text(encoding="utf-8"))["clusters"]

    print(f"{'cluster_id':<13}  {'task':>4}  {'key':>8}  {'size':>4}  top-3 candidates (score)")
    print("-" * 120)

    suggestions: dict[str, list[dict]] = {}
    for c in profiles:
        task = c["tasks"][0] if c.get("tasks") else "Set1"
        # Skip Set1 pool matching for Free clusters — Free pool is subject-specific
        if task == "Free":
            suggestions[c["cluster_id"]] = [{
                "pool_id": "free_piece",
                "title": "(Free task — see Piece Selection - Free.csv for this subject)",
                "level": "N/A",
                "score": None,
                "breakdown": None,
            }]
            print(f"{c['cluster_id']:<13}  {'Free':>4}  {c['key']:>4} {c['mode'][:3]:>3}  "
                  f"{c['size']:>4}  -> skip pool match (Free)")
            continue

        scores = []
        for p in SET1_POOL:
            total, br = score_cluster_vs_pool(c, p)
            scores.append({
                "pool_id": p["pool_id"],
                "title": p["title"],
                "level": p["level"],
                "score": round(total, 3),
                "breakdown": br,
            })
        scores.sort(key=lambda s: -s["score"])
        suggestions[c["cluster_id"]] = scores[:3]

        # Confidence indicator: score gap top vs 2nd-best
        gap = scores[0]["score"] - scores[1]["score"]
        conf = "HIGH" if gap >= 0.20 else ("MED" if gap >= 0.10 else "LOW")

        key_str = f"{c['key']} {c['mode'][:3]}"
        top3 = " | ".join(
            f"{s['title'][:28]:<28} ({s['score']:.2f})" for s in scores[:3]
        )
        print(f"{c['cluster_id']:<13}  {'Set1':>4}  {key_str:>8}  {c['size']:>4}  [{conf:>4}] {top3}")

    # Merge into labels file
    labels_data = json.loads(LABELS.read_text(encoding="utf-8")) if LABELS.is_file() else {"labels": {}}
    for cid, sug in suggestions.items():
        labels_data.setdefault("suggestions", {})[cid] = sug

    labels_data["_comment"] = (
        "Cluster label mapping. `labels.{cluster_id}` is manual, "
        "`suggestions.{cluster_id}` is auto top-3 from Set1 pool matcher. "
        "Use suggestions as starting point and set `labels.{cluster_id}` to chosen pool_id or free-text."
    )
    with open(LABELS, "w", encoding="utf-8") as f:
        json.dump(labels_data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved suggestions → {LABELS}")


if __name__ == "__main__":
    main()
