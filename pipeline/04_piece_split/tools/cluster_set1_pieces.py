"""
Cluster Set1/Free pieces across 240828 subjects via MIDI fingerprint similarity
+ Louvain community detection to break chain-merges apart.

Pipeline:
  1. Fingerprint every Set1/Free piece MIDI.
  2. All-pairs similarity: for each pair, compute
        coherent = max count in any 0.5s offset bin of shared hashes
        score    = coherent / min(fps_a, fps_b)
  3. Build weighted graph with edges accepted via dual criterion:
        coherent >= HIGH_COHERENT
        OR (coherent >= MIN_COHERENT AND score >= MIN_SIM)
  4. Connected components → candidate clusters.
  5. For each candidate cluster, run Louvain modularity; if it splits into
     multiple well-separated communities, use those as sub-clusters. This
     corrects chain-merges (A–B–C where A and C are actually different pieces).
  6. Output cluster report with intra-cluster edge summary.

Clusters preserve:
  - `stable_cluster_id` (deterministic across runs via sorted member keys)
  - `confidence` (tight = all members coh>=HIGH; loose = some weak edges)
  - `signature` (median duration, median notes) for downstream pool matching.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from src.midi_fingerprint import extract_events, fingerprint  # noqa: E402


PIECES_MANIFEST = Path("Data/Pipeline/04_piece_split/240828/_pieces.json")
PIECES_DIR = Path("Data/Pipeline/04_piece_split/240828")
OUT_PATH = Path("Data/Metadata/240828/set1_clusters.json")

TASK_SET = {2: "Set1", 9: "Free"}

# Edge criterion (dual): accept if coherent >= HIGH_COHERENT, OR (coh>=MIN_COHERENT AND score>=MIN_SIM).
HIGH_COHERENT = 15
MIN_SIM = 0.10
MIN_COHERENT = 5
OFFSET_BIN_SEC = 0.5

# Louvain split is applied only to components above this size (small comps are kept as-is).
LOUVAIN_MIN_COMPONENT = 5
LOUVAIN_SEED = 42


@dataclass
class PieceRecord:
    piece_key: str
    sub: str
    trial: str
    piece_idx: int
    task: str
    n_notes: int
    duration: float
    mid_path: Path
    fps: list[tuple[int, float]]


def offset_mode_count(offsets: list[float]) -> int:
    if not offsets:
        return 0
    buckets: dict[int, int] = defaultdict(int)
    for o in offsets:
        buckets[int(o / OFFSET_BIN_SEC)] += 1
    return max(buckets.values())


def pairwise_score(a: PieceRecord, b: PieceRecord) -> tuple[int, float]:
    a_map: dict[int, list[float]] = defaultdict(list)
    for h, t in a.fps:
        a_map[h].append(t)
    offsets: list[float] = []
    for h, b_t in b.fps:
        if h in a_map:
            for a_t in a_map[h]:
                offsets.append(b_t - a_t)
    coherent = offset_mode_count(offsets)
    denom = min(len(a.fps), len(b.fps)) or 1
    return coherent, coherent / denom


def collect_pieces() -> list[PieceRecord]:
    manifest = json.loads(PIECES_MANIFEST.read_text(encoding="utf-8"))
    records: list[PieceRecord] = []
    for t in manifest:
        task = TASK_SET.get(t["task"])
        if task is None:
            continue
        sub = t["raw_stem"][-4:-2]
        trial = t["raw_stem"][-4:]
        for p in t["pieces"]:
            mid = PIECES_DIR / f"{t['trial_stem']}_p{p['piece_idx']}.mid"
            if not mid.is_file():
                continue
            events = extract_events(mid)
            fps = fingerprint(events)
            records.append(PieceRecord(
                piece_key=f"sub{sub}_{trial}_p{p['piece_idx']}",
                sub=f"sub{sub}",
                trial=trial,
                piece_idx=p["piece_idx"],
                task=task,
                n_notes=p["n_notes"],
                duration=round(p["end_sec"] - p["start_sec"], 2),
                mid_path=mid,
                fps=fps,
            ))
    return records


def build_weighted_graph(records: list[PieceRecord]) -> tuple[nx.Graph, list[tuple[int, int, int, float]]]:
    """Build similarity graph. Returns (graph, near_miss_edges).

    Edge weight = normalized score; also stores coherent as attribute.
    """
    n = len(records)
    g = nx.Graph()
    for i in range(n):
        g.add_node(i)
    near_miss: list[tuple[int, int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            coh, sc = pairwise_score(records[i], records[j])
            if coh >= HIGH_COHERENT or (coh >= MIN_COHERENT and sc >= MIN_SIM):
                g.add_edge(i, j, coherent=coh, score=round(sc, 3), weight=sc)
            elif coh >= 3 and sc >= 0.03:
                near_miss.append((i, j, coh, round(sc, 3)))
    return g, near_miss


def refine_with_louvain(g: nx.Graph) -> list[list[int]]:
    """Connected components, with Louvain split for large (potentially chain-merged) components."""
    refined: list[list[int]] = []
    for comp in nx.connected_components(g):
        comp_nodes = list(comp)
        if len(comp_nodes) < LOUVAIN_MIN_COMPONENT:
            refined.append(sorted(comp_nodes))
            continue
        subg = g.subgraph(comp_nodes)
        try:
            communities = nx.community.louvain_communities(
                subg, weight="weight", seed=LOUVAIN_SEED
            )
        except Exception:
            refined.append(sorted(comp_nodes))
            continue
        # Only accept the split if it yields >1 non-trivial community
        if len(communities) > 1 and all(len(c) >= 2 for c in communities):
            for c in communities:
                refined.append(sorted(c))
        else:
            refined.append(sorted(comp_nodes))
    # Sort clusters by size descending
    refined.sort(key=lambda c: -len(c))
    return refined


def cluster_signature(members: list[PieceRecord]) -> dict:
    durs = sorted(m.duration for m in members)
    notes = sorted(m.n_notes for m in members)
    return {
        "duration_median": durs[len(durs) // 2],
        "duration_min": durs[0],
        "duration_max": durs[-1],
        "notes_median": notes[len(notes) // 2],
        "n_subjects": len({m.sub for m in members}),
    }


def cluster_stable_id(member_keys: list[str]) -> str:
    h = hashlib.md5("|".join(sorted(member_keys)).encode()).hexdigest()[:8]
    return f"cl_{h}"


def cluster_confidence(g: nx.Graph, nodes: list[int]) -> str:
    if len(nodes) == 1:
        return "singleton"
    sub = g.subgraph(nodes)
    edges = list(sub.edges(data=True))
    if not edges:
        return "isolated"
    min_coh = min(e[2]["coherent"] for e in edges)
    min_sc = min(e[2]["score"] for e in edges)
    if min_coh >= HIGH_COHERENT:
        return "tight"
    if min_coh >= MIN_COHERENT and min_sc >= MIN_SIM:
        return "medium"
    return "loose"


def main():
    print("Collecting Set1/Free piece MIDIs...")
    records = collect_pieces()
    print(f"  {len(records)} pieces ({sum(1 for r in records if r.task == 'Set1')} Set1, "
          f"{sum(1 for r in records if r.task == 'Free')} Free)")

    print("\nAll-pairs similarity + weighted graph...")
    g, near_miss = build_weighted_graph(records)
    print(f"  nodes={g.number_of_nodes()}  edges={g.number_of_edges()}")
    print(f"  near-miss (review candidates): {len(near_miss)}")

    print("\nConnected components + Louvain refinement on big components...")
    clusters_idx = refine_with_louvain(g)
    print(f"  {len(clusters_idx)} clusters after refinement")

    # Format output
    clusters = []
    for ci, nodes in enumerate(clusters_idx):
        members = [records[i] for i in nodes]
        sig = cluster_signature(members)
        conf = cluster_confidence(g, nodes)
        clusters.append({
            "cluster_idx": ci + 1,
            "stable_cluster_id": cluster_stable_id([m.piece_key for m in members]),
            "size": len(members),
            "n_subjects": sig["n_subjects"],
            "tasks": sorted({m.task for m in members}),
            "confidence": conf,
            "signature": sig,
            "label": None,
            "members": [
                {
                    "piece_key": m.piece_key,
                    "sub": m.sub,
                    "trial": m.trial,
                    "piece_idx": m.piece_idx,
                    "task": m.task,
                    "n_notes": m.n_notes,
                    "duration_sec": m.duration,
                }
                for m in members
            ],
            "intra_edges": [
                {
                    "a": records[u].piece_key,
                    "b": records[v].piece_key,
                    "coherent": d["coherent"],
                    "score": d["score"],
                }
                for u, v, d in g.subgraph(nodes).edges(data=True)
            ],
        })

    # Console table
    print(f"\n{'idx':>3}  {'size':>4}  {'subs':>4}  {'conf':>8}  "
          f"{'task':>4}  {'dur_med':>8}  {'dur_range':>12}  first member")
    for c in clusters[:30]:
        tasks = "/".join(c["tasks"])
        sig = c["signature"]
        first_member = c["members"][0]["piece_key"]
        print(f"  {c['cluster_idx']:>3}  {c['size']:>4}  {c['n_subjects']:>4}  "
              f"{c['confidence']:>8}  {tasks:>4}  "
              f"{sig['duration_median']:>7.1f}s  "
              f"{sig['duration_min']:>5.0f}-{sig['duration_max']:>5.0f}s  "
              f"{first_member}")

    report = {
        "_comment": "Set1/Free cross-sub clustering with Louvain refinement. label=null → awaiting manual label.",
        "parameters": {
            "HIGH_COHERENT": HIGH_COHERENT,
            "MIN_SIM": MIN_SIM,
            "MIN_COHERENT": MIN_COHERENT,
            "LOUVAIN_MIN_COMPONENT": LOUVAIN_MIN_COMPONENT,
        },
        "n_pieces": len(records),
        "n_clusters": len(clusters),
        "confidence_legend": {
            "tight": "all intra-edges coherent >= HIGH_COHERENT (very likely same piece)",
            "medium": "intra-edges meet (MIN_COHERENT, MIN_SIM) threshold",
            "loose": "contains weak intra-edges post-Louvain",
            "singleton": "single piece — no cross-sub match",
            "isolated": "subgraph has no edges (shouldn't happen)",
        },
        "clusters": clusters,
        "near_miss_edges": [
            {
                "a": records[i].piece_key,
                "b": records[j].piece_key,
                "coherent": coh,
                "score": sc,
            }
            for i, j, coh, sc in sorted(near_miss, key=lambda x: -x[3])[:50]
        ],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
