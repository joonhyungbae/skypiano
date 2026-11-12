"""
Group Technique pieces that no anchor names by matching them against each
other, so the reviewer can name a whole group at once.

Many 240828 Technique2 exercises are short scales/arpeggios that the anchor
library does not cover; s04 leaves them `no_match`. The same exercises were
played by every subject, so a piece's closest matches among the other
subjects' pieces usually are the same exercise. Every Technique piece of the
dataset is fingerprinted (`midi_fingerprint`), each pair scored the way s03
scores a piece against an anchor take (largest offset-coherent hash count /
the smaller fingerprint count), and pairs from different subjects with
score >= MIN_SCORE and >= MIN_COHERENT coherent hashes become edges. Louvain
communities of that graph are the groups.

A group is written when it holds at least one unnamed piece. Its suggestion
is the most common name among its named members (a manual label counts over
an s04 name); it is only a hint for the reviewer, never a label.

Output: Data/Metadata/{ds}/technique_self_clusters.json

Usage (from the repository root):
  python pipeline/04_piece_split/tools/technique_self_clusters.py --dataset 240828 [--task 1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.midi_fingerprint import extract_events, fingerprint  # noqa: E402

PIECES = Path("Data/Pipeline/04_piece_split")
META = Path("Data/Metadata")
OFFSET_BIN_SEC = 0.5
MIN_SCORE = 0.3
MIN_COHERENT = 4
TASK_NAMES = {0: "Technique1", 1: "Technique2"}


def subject_of(trial_stem: str) -> str:
    return "_".join(trial_stem.split("_")[1:3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="240828")
    ap.add_argument("--task", type=int, nargs="*", default=[0, 1])
    a = ap.parse_args()
    ds = a.dataset
    trials = json.loads((PIECES / ds / "_pieces_resolved.json").read_text(encoding="utf-8"))
    manual = json.loads((PIECES / ds / "_pieces_manual.json").read_text(encoding="utf-8")) \
        if (PIECES / ds / "_pieces_manual.json").exists() else {}

    out_clusters = []
    for task in a.task:
        items = []
        for t in trials:
            if t["task"] != task:
                continue
            for p in t["pieces"]:
                mid = PIECES / ds / f"{t['trial_stem']}_p{p['piece_idx']}.mid"
                if not mid.exists():
                    continue
                ident = p.get("identified") or {}
                m = (manual.get(t["trial_stem"]) or {}).get(str(p["piece_idx"])) or {}
                auto = ident.get("resolved_piece_id") if ident.get("resolution_source") != "no_match" else None
                items.append({"trial_stem": t["trial_stem"], "piece_idx": p["piece_idx"],
                              "subject": subject_of(t["trial_stem"]),
                              "duration_sec": round(p["end_sec"] - p["start_sec"], 2),
                              "manual": m.get("piece_id"), "auto": auto,
                              "fp": fingerprint(extract_events(mid))})
        index = defaultdict(list)
        for j, it in enumerate(items):
            for h, t0 in it["fp"]:
                index[h].append((j, t0))

        g = nx.Graph()
        g.add_nodes_from(range(len(items)))
        for i, it in enumerate(items):
            offs = defaultdict(list)
            for h, t0 in it["fp"]:
                for j, t1 in index.get(h, []):
                    if j > i and items[j]["subject"] != it["subject"]:
                        offs[j].append(t0 - t1)
            for j, o in offs.items():
                coherent = Counter(int(x // OFFSET_BIN_SEC) for x in o).most_common(1)[0][1]
                score = coherent / (min(len(it["fp"]), len(items[j]["fp"])) or 1)
                if score >= MIN_SCORE and coherent >= MIN_COHERENT:
                    g.add_edge(i, j, weight=score)

        comms = nx.community.louvain_communities(g, weight="weight", seed=0)
        n_unnamed = 0
        for comm in sorted(comms, key=lambda c: -len(c)):
            if len(comm) < 2:
                continue
            members = [items[j] for j in sorted(comm)]
            unnamed = [m for m in members if not m["manual"] and not m["auto"]]
            if not unnamed:
                continue
            n_unnamed += len(unnamed)
            names = Counter()
            for m in members:
                if m["manual"]:
                    names[m["manual"]] += 2
                elif m["auto"]:
                    names[m["auto"]] += 1
            key = ",".join(sorted(f"{m['trial_stem']}_p{m['piece_idx']}" for m in members))
            out_clusters.append({
                "cluster_id": f"self_T{task}_{hashlib.sha1(key.encode()).hexdigest()[:8]}",
                "task": task, "task_name": TASK_NAMES[task],
                "suggested_label": names.most_common(1)[0][0] if names else None,
                "label_votes": dict(names.most_common(3)),
                "n_subjects": len({m["subject"] for m in members}),
                "members": [{k: m[k] for k in ("trial_stem", "piece_idx", "subject", "duration_sec",
                                               "manual", "auto")} for m in members],
            })
        n_total = sum(1 for it in items if not it["manual"] and not it["auto"])
        print(f"{ds} T{task}: {len(items)} pieces, {n_total} unnamed; "
              f"{n_unnamed} unnamed pieces in {sum(c['task'] == task for c in out_clusters)} groups")

    out = META / ds / "technique_self_clusters.json"
    out.write_text(json.dumps({"dataset": ds, "min_score": MIN_SCORE, "min_coherent": MIN_COHERENT,
                               "clusters": out_clusters}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
