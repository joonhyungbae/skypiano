"""
Extended ensemble clustering v2: 6 signals instead of 3.

Signals:
  1. fingerprint       (n-gram hash, symbolic local)
  2. chroma_dtw_basic  (CQT chroma + subsequence DTW)
  3. structural        (MIDI feature vector cosine)
  4. cens_dtw          (CENS chroma + subsequence DTW)  [NEW]
  5. twodftm           (2D FFT magnitude of chroma)     [NEW]
  6. mfcc              (MFCC mean/std cosine)           [NEW]

Voting rule: weighted by per-signal discriminative power (range of
distribution). Signals with compressed distributions (mfcc, 2DFTM) get lower
weight — they carry less information per unit pair.

Each signal has its own percentile threshold; edge accepted if the WEIGHTED
vote count exceeds a threshold or any single signal is extremely strong.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np

FP_CLUSTERS = Path("Data/Metadata/240828/set1_clusters.json")
CHROMA_SCORES = Path("Data/Metadata/240828/set1_chroma_scores.json")
STRUCT_SCORES = Path("Data/Metadata/240828/set1_structural_scores.json")
ADV_SCORES = Path("Data/Metadata/240828/set1_advanced_scores.json")
OUT_PATH = Path("Data/Metadata/240828/set1_ensemble_v2_clusters.json")

# Per-signal weights (sum arbitrary, relative). Signals with wider dynamic
# range of similarity → higher weight (more informative).
SIGNAL_WEIGHTS = {
    "fp_score":  1.0,    # symbolic local pattern (strong when it hits)
    "chroma":    1.0,    # audio harmonic, tempo-invariant
    "struct":    1.0,    # MIDI feature cosine
    "cens_dtw":  1.0,    # CENS chroma DTW (best audio signal for classical)
    # Dropped: twodftm (compressed, most pairs cluster at high values in
    # short-excerpt classical piano), mfcc (piano timbre is near-constant)
}

VOTE_PCTL = 0.95
VERY_STRONG_PCTL = 0.99
REQUIRED_WEIGHTED_VOTES = 2.0
MIN_DURATION = 20.0
LOUVAIN_SEED = 42


def load_fp_pairs():
    d = json.loads(FP_CLUSTERS.read_text(encoding="utf-8"))
    pairs = {}
    for c in d["clusters"]:
        for e in c.get("intra_edges", []):
            k = tuple(sorted([e["a"], e["b"]]))
            pairs[k] = e["score"]
    for e in d.get("near_miss_edges", []):
        k = tuple(sorted([e["a"], e["b"]]))
        if k not in pairs:
            pairs[k] = e["score"]
    return pairs


def rank_percentile(values):
    n = len(values)
    if n == 0:
        return {}
    sorted_idx = sorted(range(n), key=lambda i: values[i])
    return {idx: rank / (n - 1) for rank, idx in enumerate(sorted_idx)}


def main():
    fp = load_fp_pairs()
    chroma = json.loads(CHROMA_SCORES.read_text(encoding="utf-8"))
    struct = json.loads(STRUCT_SCORES.read_text(encoding="utf-8"))
    adv = json.loads(ADV_SCORES.read_text(encoding="utf-8"))

    pieces = {p["piece_key"]: p for p in struct["pieces"]}
    chroma_map = {tuple(sorted([p["a"], p["b"]])): p["similarity"]
                  for p in chroma["pair_similarities"]}
    struct_map = {tuple(sorted([p["a"], p["b"]])): p["similarity"]
                  for p in struct["pair_similarities"]}
    cens_map = {tuple(sorted([p["a"], p["b"]])): p["cens_dtw"]
                for p in adv["pair_similarities"]}
    twodftm_map = {tuple(sorted([p["a"], p["b"]])): p["twodftm"]
                   for p in adv["pair_similarities"]}
    mfcc_map = {tuple(sorted([p["a"], p["b"]])): p["mfcc"]
                for p in adv["pair_similarities"]}

    all_keys = (set(fp) | set(chroma_map) | set(struct_map)
                | set(cens_map) | set(twodftm_map) | set(mfcc_map))

    rows = []
    for k in all_keys:
        a, b = k
        if a not in pieces or b not in pieces:
            continue
        rows.append({
            "a": a, "b": b,
            "fp_score": fp.get(k, 0.0),
            "chroma": chroma_map.get(k, 0.0),
            "struct": struct_map.get(k, 0.0),
            "cens_dtw": cens_map.get(k, 0.0),
            "twodftm": twodftm_map.get(k, 0.0),
            "mfcc": mfcc_map.get(k, 0.0),
        })

    # Rank-percentiles per signal
    signals = list(SIGNAL_WEIGHTS.keys())
    ranks = {s: rank_percentile([r[s] for r in rows]) for s in signals}
    for i, r in enumerate(rows):
        for s in signals:
            r[f"{s}_pctl"] = round(ranks[s][i], 3)
        # Weighted ensemble score
        r["ensemble_pctl"] = round(
            sum(r[f"{s}_pctl"] * SIGNAL_WEIGHTS[s] for s in signals)
            / sum(SIGNAL_WEIGHTS.values()), 3
        )

    # Filter short pieces
    long_keys = {k for k, p in pieces.items() if p["duration_sec"] >= MIN_DURATION}

    # Build graph with weighted voting
    g = nx.Graph()
    for k in long_keys:
        g.add_node(k)

    for r in rows:
        if r["a"] not in long_keys or r["b"] not in long_keys:
            continue
        # Weighted vote count
        weighted_votes = sum(
            SIGNAL_WEIGHTS[s] for s in signals if r[f"{s}_pctl"] >= VOTE_PCTL
        )
        very_strong = any(r[f"{s}_pctl"] >= VERY_STRONG_PCTL for s in signals)
        accept = weighted_votes >= REQUIRED_WEIGHTED_VOTES or very_strong
        if accept:
            g.add_edge(r["a"], r["b"],
                       weight=r["ensemble_pctl"],
                       weighted_votes=round(weighted_votes, 2),
                       ensemble_pctl=r["ensemble_pctl"],
                       **{f"{s}_pctl": r[f"{s}_pctl"] for s in signals})

    print(f"Ensemble v2 graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    print(f"  Edge criterion: weighted_votes >= {REQUIRED_WEIGHTED_VOTES} "
          f"(per-signal pctl >= {VOTE_PCTL}) OR any signal pctl >= {VERY_STRONG_PCTL}")

    # Connected components + Louvain
    clusters = []
    for comp in nx.connected_components(g):
        nodes = list(comp)
        if len(nodes) < 5:
            clusters.append(sorted(nodes))
            continue
        subg = g.subgraph(nodes)
        try:
            communities = nx.community.louvain_communities(
                subg, weight="weight", seed=LOUVAIN_SEED
            )
        except Exception:
            clusters.append(sorted(nodes))
            continue
        if len(communities) > 1 and all(len(c) >= 2 for c in communities):
            for c in communities:
                clusters.append(sorted(c))
        else:
            clusters.append(sorted(nodes))

    in_cluster = set().union(*clusters) if clusters else set()
    for k in pieces:
        if k not in in_cluster:
            clusters.append([k])
    clusters.sort(key=lambda c: -len(c))

    # Format
    out_clusters = []
    for ci, members in enumerate(clusters):
        member_info = [pieces[k] for k in members]
        durs = sorted(m["duration_sec"] for m in member_info)
        cid = "v2_" + hashlib.md5("|".join(sorted(members)).encode()).hexdigest()[:8]
        intra = []
        if len(members) > 1 and g.has_node(members[0]):
            sub = g.subgraph(members)
            for u, v, d in sub.edges(data=True):
                intra.append({
                    "a": u, "b": v,
                    "weighted_votes": d["weighted_votes"],
                    "ensemble_pctl": d["ensemble_pctl"],
                    **{s: d[f"{s}_pctl"] for s in signals},
                })
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
        })

    # Summary
    print(f"\n{'idx':>3}  {'size':>4}  {'subs':>4}  {'task':>4}  {'dur_med':>8}  {'dur_range':>12}  first member")
    for c in out_clusters[:30]:
        tasks = "/".join(c["tasks"])
        sig = c["signature"]
        first = c["members"][0]["piece_key"]
        print(f"  {c['cluster_idx']:>3}  {c['size']:>4}  {c['n_subjects']:>4}  "
              f"{tasks:>4}  {sig['duration_median']:>7.1f}s  "
              f"{sig['duration_min']:>5.0f}-{sig['duration_max']:>5.0f}s  {first}")

    n_multi = sum(1 for c in out_clusters if c["size"] >= 2)
    n_sing = sum(1 for c in out_clusters if c["size"] == 1)
    print(f"\n{n_multi} multi-member + {n_sing} singleton = {len(out_clusters)} total")

    report = {
        "_comment": "Ensemble v2: 6-signal weighted voting with Louvain refinement.",
        "signal_weights": SIGNAL_WEIGHTS,
        "parameters": {
            "VOTE_PCTL": VOTE_PCTL,
            "VERY_STRONG_PCTL": VERY_STRONG_PCTL,
            "REQUIRED_WEIGHTED_VOTES": REQUIRED_WEIGHTED_VOTES,
            "MIN_DURATION": MIN_DURATION,
        },
        "n_pieces": len(pieces),
        "n_clusters": len(out_clusters),
        "clusters": out_clusters,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
