"""
Sequence-based (anchor-relative) clustering for Set1 pieces.

Phase-1 fingerprint clustering already groups pieces by audio similarity. Bulk-
review then attaches a manual `piece_id` to multi-subject clusters. This script
adds a third, complementary signal: **the recording-order pattern** within each
subject's Set1 session.

Idea
----
240828 subjects all played Set1 in (mostly) the same order. So if we know
sub04's piece-7 is `clementi_op36_no1`, and sub04's piece-8 is unlabeled but
shows up in another subject's piece-8 slot one trial after their own clementi,
those two pieces are very likely the SAME piece. We turn this into clusters
keyed by `(nearest labeled anchor, signed positional offset)`.

Output (`Data/Metadata/{ds}/set1_sequence_clusters.json`):
  * `subject_sequences` — each subject's Set1 pieces in time order, with their
    anchor coordinate.
  * `clusters` — pieces grouped by (anchor_label, offset). Offset = 0 clusters
    correspond to the labeled anchors themselves (cross-validation against
    bulk-review). Offset != 0 clusters are NEW candidate groups that bulk
    review hasn't seen yet.

How `suggested_label` is derived
--------------------------------
  * offset == 0  →  the anchor_label itself (cross-validated against bulk).
  * offset != 0  →  null (left for pool matching / human review).

Sliver handling
---------------
Pieces with `piece_idx > 1` and duration < `P2_SLIVER_MAX_SEC` are treated as
segmentation artifacts (e.g. silence-tail of the previous take). They inherit
the position of the previous non-sliver piece in the same trial — so the
position counter doesn't drift, but the sliver still shows up as a member of
the same coordinate cluster as its parent.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PIPELINE_ROOT = Path("Data/Pipeline/04_piece_split")
META_ROOT = Path("Data/Metadata")

# Anchor extraction parameters.
MIN_ANCHOR_SUBJECTS = 2          # piece label must appear in this many subjects to be an anchor
P2_SLIVER_MAX_SEC = 20.0         # piece_idx>1 below this duration is treated as sliver

_RAW_STEM_RE = re.compile(r"sub(\d{2})(\d{2})$")


def subject_of(raw_stem: str) -> str | None:
    m = _RAW_STEM_RE.search(raw_stem or "")
    return f"sub{m.group(1)}" if m else None


def piece_key_of(raw_stem: str, piece_idx: int) -> str | None:
    """Cluster-style piece key, matches `cluster_set1_pieces.py`."""
    m = _RAW_STEM_RE.search(raw_stem or "")
    if not m:
        return None
    return f"sub{m.group(1)}_{m.group(1)}{m.group(2)}_p{piece_idx}"


# ── Inputs ─────────────────────────────────────────────────────────────────

def load_pieces(ds: str) -> list[dict]:
    fp = PIPELINE_ROOT / ds / "_pieces.json"
    return json.loads(fp.read_text(encoding="utf-8"))


def load_cluster_labels(ds: str) -> dict[str, str]:
    """piece_key → manual label, from bulk-review cluster labels."""
    cl_path = META_ROOT / ds / "set1_ensemble_v2_clusters.json"
    lb_path = META_ROOT / ds / "set1_cluster_labels.json"
    if not cl_path.exists() or not lb_path.exists():
        return {}
    clusters = json.loads(cl_path.read_text(encoding="utf-8"))
    labels_doc = json.loads(lb_path.read_text(encoding="utf-8"))
    label_of = labels_doc.get("labels") or {}
    out: dict[str, str] = {}
    for c in clusters.get("clusters", []):
        cid = c.get("stable_cluster_id") or c.get("cluster_id")
        lab = label_of.get(cid)
        if not lab:
            continue
        for m in c.get("members", []):
            if m.get("task") != "Set1":
                continue
            out[m["piece_key"]] = lab
    return out


def load_phase1_assignment(ds: str) -> dict[str, str]:
    """piece_key → phase1 cluster_id (regardless of label)."""
    cl_path = META_ROOT / ds / "set1_ensemble_v2_clusters.json"
    if not cl_path.exists():
        return {}
    clusters = json.loads(cl_path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for c in clusters.get("clusters", []):
        cid = c.get("stable_cluster_id") or c.get("cluster_id")
        for m in c.get("members", []):
            if m.get("task") != "Set1":
                continue
            out[m["piece_key"]] = cid
    return out


def load_pieces_manual(ds: str, trial_stem_to_raw: dict[str, str]) -> dict[str, str]:
    """piece_key → manual label, from per-piece bulk-apply records."""
    fp = PIPELINE_ROOT / ds / "_pieces_manual.json"
    if not fp.exists():
        return {}
    doc = json.loads(fp.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for trial_stem, by_idx in doc.items():
        raw = trial_stem_to_raw.get(trial_stem)
        if not raw:
            continue
        for idx_str, entry in (by_idx or {}).items():
            try:
                idx = int(idx_str)
            except (TypeError, ValueError):
                continue
            label = (entry or {}).get("piece_id")
            if not label:
                continue
            pk = piece_key_of(raw, idx)
            if pk:
                out[pk] = label
    return out


# ── Sequence construction ──────────────────────────────────────────────────

def build_subject_sequences(trials: list[dict]) -> dict[str, list[dict]]:
    """Return {subject: [piece_record, ...]} sorted by (trial_stem, piece_idx).

    Each piece_record carries `position` (1-based, slivers inherit parent's
    position) and `is_sliver`.
    """
    by_sub: dict[str, list[dict]] = defaultdict(list)
    for t in trials:
        if t.get("task") != 2:
            continue
        sub = subject_of(t.get("raw_stem", ""))
        if not sub:
            continue
        for p in t.get("pieces") or []:
            duration = float(p.get("end_sec", 0)) - float(p.get("start_sec", 0))
            pk = piece_key_of(t.get("raw_stem", ""), p.get("piece_idx", 0))
            if not pk:
                continue
            by_sub[sub].append({
                "piece_key": pk,
                "trial_stem": t.get("trial_stem"),
                "raw_stem": t.get("raw_stem"),
                "trial_id": (t.get("raw_stem") or "")[-4:],
                "piece_idx": int(p.get("piece_idx", 0)),
                "duration_sec": round(duration, 3),
                "n_notes": int(p.get("n_notes", 0)),
            })

    for sub, items in by_sub.items():
        items.sort(key=lambda r: (r["trial_stem"] or "", r["piece_idx"]))
        pos = 0
        last_non_sliver_pos = 0
        for it in items:
            is_sliver = it["piece_idx"] > 1 and it["duration_sec"] < P2_SLIVER_MAX_SEC
            it["is_sliver"] = is_sliver
            if is_sliver:
                it["position"] = last_non_sliver_pos
            else:
                pos += 1
                it["position"] = pos
                last_non_sliver_pos = pos
    return dict(by_sub)


# ── Anchor extraction ──────────────────────────────────────────────────────

def select_anchors(
    by_sub: dict[str, list[dict]],
    label_map: dict[str, str],
    min_subjects: int,
) -> dict[str, str]:
    """piece_key → anchor_label (only for labels reaching `min_subjects` coverage)."""
    label_to_subs: dict[str, set[str]] = defaultdict(set)
    for sub, items in by_sub.items():
        for it in items:
            lab = label_map.get(it["piece_key"])
            if lab:
                label_to_subs[lab].add(sub)
    qualified = {lab for lab, subs in label_to_subs.items() if len(subs) >= min_subjects}
    out: dict[str, str] = {}
    for sub, items in by_sub.items():
        for it in items:
            lab = label_map.get(it["piece_key"])
            if lab and lab in qualified and not it["is_sliver"]:
                out[it["piece_key"]] = lab
    return out


# ── Anchor-relative coordinate ─────────────────────────────────────────────

def attach_coordinates(by_sub: dict[str, list[dict]], anchor_map: dict[str, str]) -> None:
    """Annotate each piece with `nearest_anchor`, `anchor_position`, `signed_offset`.

    Tie-break: smaller |offset| wins; if a tie, prefer the EARLIER anchor (lower
    position) so the resulting offset is positive when possible.
    """
    for sub, items in by_sub.items():
        anchors_in_sub: list[tuple[int, str]] = []
        for it in items:
            lab = anchor_map.get(it["piece_key"])
            if lab and not it["is_sliver"]:
                anchors_in_sub.append((it["position"], lab))
        anchors_in_sub.sort()

        for it in items:
            if not anchors_in_sub:
                it["nearest_anchor"] = None
                it["anchor_position"] = None
                it["signed_offset"] = None
                continue
            best: tuple[tuple[int, int], str, int, int] | None = None
            for ap, alabel in anchors_in_sub:
                offset = it["position"] - ap
                # Sort key: smaller |offset| wins; on tie pick the EARLIER
                # anchor (smaller ap) so the resulting offset is positive
                # whenever possible — this keeps "X+1" consistent across
                # subjects (e.g. post-clementi piece always reads `clementi+1`,
                # not `schubert-1` from the same cluster).
                key = (abs(offset), ap)
                if best is None or key < best[0]:
                    best = (key, alabel, offset, ap)
            assert best is not None
            it["nearest_anchor"] = best[1]
            it["signed_offset"] = best[2]
            it["anchor_position"] = best[3]


# ── Clusters ───────────────────────────────────────────────────────────────

def build_clusters(
    by_sub: dict[str, list[dict]],
    phase1_assignment: dict[str, str],
    label_map: dict[str, str],
) -> list[dict]:
    """Group pieces by (nearest_anchor, signed_offset)."""
    groups: dict[tuple[str, int], list[tuple[str, dict]]] = defaultdict(list)
    for sub, items in by_sub.items():
        for it in items:
            anchor = it.get("nearest_anchor")
            if anchor is None:
                continue
            key = (anchor, int(it["signed_offset"]))
            groups[key].append((sub, it))

    def _signed(o: int) -> str:
        return f"+{o}" if o >= 0 else str(o)

    clusters: list[dict] = []
    for (anchor, offset), members in sorted(
        groups.items(),
        key=lambda kv: (-len({s for s, _ in kv[1]}), abs(kv[0][1]), kv[0][0], kv[0][1]),
    ):
        durs = sorted(m["duration_sec"] for _, m in members)
        subjects = sorted({s for s, _ in members})
        phase1_ids = sorted({phase1_assignment.get(m["piece_key"]) for _, m in members if phase1_assignment.get(m["piece_key"])})
        # Cross-validation: how many members carry the same label as the anchor.
        n_label_match = sum(1 for _, m in members if label_map.get(m["piece_key"]) == anchor)
        n_label_other = sum(1 for _, m in members if label_map.get(m["piece_key"]) and label_map.get(m["piece_key"]) != anchor)
        suggested_label = anchor if offset == 0 else None
        clusters.append({
            "stable_cluster_id": f"seq:T2:{anchor}{_signed(offset)}",
            "anchor_label": anchor,
            "offset": offset,
            "size": len(members),
            "n_subjects": len(subjects),
            "subjects": subjects,
            "tasks": ["Set1"],
            "signature": {
                "duration_median": durs[len(durs) // 2],
                "duration_min": durs[0],
                "duration_max": durs[-1],
            },
            "members": [
                {
                    "piece_key": m["piece_key"],
                    "sub": s,
                    "trial": m["trial_id"],
                    "trial_stem": m["trial_stem"],
                    "piece_idx": m["piece_idx"],
                    "task": "Set1",
                    "duration_sec": m["duration_sec"],
                    "subject_position": m["position"],
                    "anchor_position": m["anchor_position"],
                    "signed_offset": m["signed_offset"],
                    "is_sliver": m["is_sliver"],
                    "phase1_cluster_id": phase1_assignment.get(m["piece_key"]),
                    "phase1_label": label_map.get(m["piece_key"]),
                }
                for s, m in members
            ],
            "phase1_cluster_ids": phase1_ids,
            "suggested_label": suggested_label,
            "n_label_match": n_label_match,
            "n_label_other": n_label_other,
            "confidence": _coord_confidence(members, offset, n_label_match, n_label_other),
        })

    for i, c in enumerate(clusters, 1):
        c["cluster_idx"] = i
    return clusters


def _coord_confidence(members: list[tuple[str, dict]], offset: int, n_match: int, n_other: int) -> str:
    """Heuristic label for cluster trustworthiness in the sequence track.

    `cross_validated`  — offset 0 cluster, all members carry a matching label.
    `confirmed`        — offset 0 cluster, label backed by the anchor itself.
    `proposed`         — offset != 0, multi-subject, no contradictions.
    `singleton`        — single member.
    `conflict`         — members carry different bulk-review labels.
    """
    if n_other > 0 and n_match > 0:
        return "conflict"
    if len(members) == 1:
        return "singleton"
    if offset == 0 and n_match == len(members):
        return "cross_validated"
    if offset == 0:
        return "confirmed"
    return "proposed"


# ── Output ─────────────────────────────────────────────────────────────────

def build_output(ds: str, by_sub: dict[str, list[dict]], anchor_map: dict[str, str], clusters: list[dict]) -> dict:
    anchor_labels = sorted(set(anchor_map.values()))
    return {
        "_comment": (
            "Anchor-relative sequence clustering for Set1 pieces. Each cluster "
            "key is (anchor_label, signed_offset) where offset is in units of "
            "non-sliver subject positions. offset=0 clusters mirror the labeled "
            "bulk-review groups (cross-validation). offset != 0 clusters are "
            "new sequence-only candidates."
        ),
        "parameters": {
            "MIN_ANCHOR_SUBJECTS": MIN_ANCHOR_SUBJECTS,
            "P2_SLIVER_MAX_SEC": P2_SLIVER_MAX_SEC,
        },
        "n_anchor_labels": len(anchor_labels),
        "anchor_labels": anchor_labels,
        "n_pieces": sum(len(v) for v in by_sub.values()),
        "n_clusters": len(clusters),
        "clusters": clusters,
        "subject_sequences": {
            sub: [
                {
                    "piece_key": it["piece_key"],
                    "trial_stem": it["trial_stem"],
                    "trial": it["trial_id"],
                    "piece_idx": it["piece_idx"],
                    "duration_sec": it["duration_sec"],
                    "is_sliver": it["is_sliver"],
                    "position": it["position"],
                    "nearest_anchor": it.get("nearest_anchor"),
                    "anchor_position": it.get("anchor_position"),
                    "signed_offset": it.get("signed_offset"),
                    "anchor_label_self": anchor_map.get(it["piece_key"]),
                }
                for it in items
            ]
            for sub, items in sorted(by_sub.items())
        },
    }


def process_dataset(ds: str) -> Path:
    trials = load_pieces(ds)
    trial_stem_to_raw = {t["trial_stem"]: t.get("raw_stem", "") for t in trials}
    cluster_labels = load_cluster_labels(ds)
    manual_labels = load_pieces_manual(ds, trial_stem_to_raw)
    label_map: dict[str, str] = {**cluster_labels, **manual_labels}  # manual wins on conflict
    phase1_assignment = load_phase1_assignment(ds)

    by_sub = build_subject_sequences(trials)
    anchor_map = select_anchors(by_sub, label_map, MIN_ANCHOR_SUBJECTS)
    attach_coordinates(by_sub, anchor_map)
    clusters = build_clusters(by_sub, phase1_assignment, label_map)
    out = build_output(ds, by_sub, anchor_map, clusters)

    out_path = META_ROOT / ds / "set1_sequence_clusters.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # Console summary
    print(f"[seq] {ds}: {out['n_pieces']} pieces, {out['n_clusters']} clusters, "
          f"{out['n_anchor_labels']} anchor labels")
    by_conf: dict[str, int] = defaultdict(int)
    for c in clusters:
        by_conf[c["confidence"]] += 1
    for k in ("cross_validated", "confirmed", "proposed", "conflict", "singleton"):
        if by_conf.get(k):
            print(f"    {k:<18s} {by_conf[k]:>4d}")
    print(f"    wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=["240828"],
                    help="Dataset directories under Data/Metadata to process.")
    args = ap.parse_args()
    for ds in args.datasets:
        process_dataset(ds)


if __name__ == "__main__":
    main()
