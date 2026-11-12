"""
Cluster Set1/Free pieces using chroma-DTW pair similarities.

Compared to fingerprint clustering:
  - Chroma similarity has higher baseline noise floor for classical piano
    (common chords, scales in 12-dim space). Median pair is ~0.73.
  - Very short pieces (<20s) match everything — filter separately.
  - Threshold picked at ~2σ above median ≈ 0.86 based on observed p95=0.891.

Same Louvain refinement + dual-criterion as fingerprint clustering.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import networkx as nx

IN_PATH = Path("Data/Metadata/240828/set1_chroma_scores.json")
OUT_PATH = Path("Data/Metadata/240828/set1_chroma_clusters.json")

# Pair thresholds: chroma sim has higher floor than fingerprint, so demand higher sim.
# Above p95 (= 0.891) is a reasonable "signal" tier.
MIN_SIM = 0.90
MIN_DURATION = 20.0     # short pieces (<20s) excluded from graph (match everything)
LOUVAIN_MIN_COMPONENT = 5
LOUVAIN_SEED = 42


def main():
    data = json.loads(IN_PATH.read_text(encoding="utf-8"))
    pieces = {p["piece_key"]: p for p in data["pieces"]}
    pairs = data["pair_similarities"]

    # Filter short pieces from graph (keep as forced singletons)
    long_keys = {k for k, p in pieces.items() if p["duration_sec"] >= MIN_DURATION}
    short_keys = set(pieces) - long_keys
    print(f"Pieces: {len(pieces)} total, {len(long_keys)} long (>{MIN_DURATION}s), {len(short_keys)} short")

    # Build graph
    g = nx.Graph()
    for k in long_keys:
        g.add_node(k)
    edges_used = 0
    for p in pairs:
        a, b = p["a"], p["b"]
        if a not in long_keys or b not in long_keys:
            continue
        sim = p["similarity"]
        if sim >= MIN_SIM:
            g.add_edge(a, b, similarity=sim, weight=sim)
            edges_used += 1
    print(f"Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges (sim >= {MIN_SIM})")

    # Connected components + Louvain refine
    clusters: list[list[str]] = []
    for comp in nx.connected_components(g):
        comp_nodes = list(comp)
        if len(comp_nodes) < LOUVAIN_MIN_COMPONENT:
            clusters.append(sorted(comp_nodes))
            continue
        subg = g.subgraph(comp_nodes)
        try:
            communities = nx.community.louvain_communities(
                subg, weight="weight", seed=LOUVAIN_SEED
            )
        except Exception:
            clusters.append(sorted(comp_nodes))
            continue
        if len(communities) > 1 and all(len(c) >= 2 for c in communities):
            for c in communities:
                clusters.append(sorted(c))
        else:
            clusters.append(sorted(comp_nodes))

    # Add short pieces as singletons
    for k in short_keys:
        clusters.append([k])

    clusters.sort(key=lambda c: -len(c))

    # Format
    out_clusters = []
    for ci, members in enumerate(clusters):
        member_info = [pieces[k] for k in members]
        durs = sorted(m["duration_sec"] for m in member_info)
        # Collect intra-cluster edges
        intra = []
        if len(members) > 1 and g.has_node(members[0]):
            sub = g.subgraph(members)
            for u, v, d in sub.edges(data=True):
                intra.append({"a": u, "b": v, "similarity": round(d["similarity"], 3)})
        cid = "ch_" + hashlib.md5("|".join(sorted(members)).encode()).hexdigest()[:8]
        out_clusters.append({
            "cluster_idx": ci + 1,
            "stable_cluster_id": cid,
            "size": len(members),
            "n_subjects": len({m["sub"] for m in member_info}),
            "tasks": sorted({m["task"] for m in member_info}),
            "signature": {
                "duration_median": durs[len(durs) // 2],
                "duration_min": durs[0],
                "duration_max": durs[-1],
            },
            "members": member_info,
            "intra_edges": intra,
            "short_excluded": len(members) == 1 and members[0] in short_keys,
        })

    # Print table
    print(f"\n{'idx':>3}  {'size':>4}  {'subs':>4}  {'task':>4}  {'dur_med':>8}  "
          f"{'dur_range':>12}  first member")
    for c in out_clusters[:25]:
        tasks = "/".join(c["tasks"])
        sig = c["signature"]
        first = c["members"][0]["piece_key"]
        flag = " [short]" if c["short_excluded"] else ""
        print(f"  {c['cluster_idx']:>3}  {c['size']:>4}  {c['n_subjects']:>4}  "
              f"{tasks:>4}  {sig['duration_median']:>7.1f}s  "
              f"{sig['duration_min']:>5.0f}-{sig['duration_max']:>5.0f}s  {first}{flag}")

    report = {
        "_comment": "Chroma-DTW cross-sub clustering. Short pieces (<20s) forced to singletons.",
        "parameters": {
            "MIN_SIM": MIN_SIM,
            "MIN_DURATION": MIN_DURATION,
            "LOUVAIN_MIN_COMPONENT": LOUVAIN_MIN_COMPONENT,
        },
        "n_pieces": len(pieces),
        "n_clusters": len(out_clusters),
        "n_short_excluded": len(short_keys),
        "clusters": out_clusters,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
