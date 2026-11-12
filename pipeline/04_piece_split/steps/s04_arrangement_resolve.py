"""
Stage 04 Step 04: Arrangement-aware piece ID resolution.

Three layers of signal combined per piece:

  1. Cluster label (T2 Set1) — from `set1_cluster_labels.json`.
  2. Fingerprint top-K (from s03) — rewards a single dominant match.
  3. Cumulative positional prior — ALL pieces by a single subject for a single
     task are concatenated in chronological order (trial stem sorted by the
     natural P#/t# numbers, then piece_idx). This concatenated query is scored
     against the canonical piece sequence (built from 220819/221112 arrangement
     CSVs) with a Gaussian position weight, so candidate i is more likely to be
     canonical[round(i * M/N)].

The goal is to rescue Technique 1 and Technique 2 piece IDs whose fingerprint
top-1 is only medium-confidence: if a nearby top-K candidate matches the
expected canonical position, we upgrade to `arrangement_positional` with that
candidate's piece_id.

Priority order for `resolved_piece_id` / `resolution_source`:

  1. cluster                 — T2 with a confirmed cluster label.
  2. fingerprint_high        — top-1 fingerprint already `confidence=high`.
  3. arrangement_positional  — cumulative prior promotes a top-K candidate.
  4. arrangement_verified    — exact template_n_pieces match at same position.
  5. fingerprint_best        — top-1 fingerprint fallback.
  6. no_match                — no signal at all.

Set1 entries lacking anchor coverage fall back to canonical-position priors and
are emitted with `confidence='assumed_from_position'` so downstream tooling and
paper stats can separate position-prior labels from evidence-backed ones.

Output: `_pieces_resolved.json` sidecar alongside `_pieces.json`.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

OUT_ROOT = Path("Data/Pipeline/04_piece_split")
META_ROOT = Path("Data/Metadata")
CANONICAL_PATH = META_ROOT / "canonical_order.json"

TASK_INT_TO_NAME = {0: "Technique1", 1: "Technique2", 2: "Set1", 9: "Free"}

# How spread the positional prior is (in units of canonical-sequence indices).
# A σ of 3.0 means ±3 canonical slots still count as close.
POS_PRIOR_SIGMA = 3.0
# Fingerprint score floor — below this a candidate is treated as noise.
MIN_FP_SCORE = 0.10


# ── Canonical sequence construction ────────────────────────────────────────

def build_canonical_sequence(task_data: dict) -> list[str]:
    """Concatenate top templates (ordered by count) into one ordered piece list.

    For T0 this produces the natural 30-piece Fink sequence in three batches
    (9 + 10 + 11 pieces) matching how 220819/221112 subjects recorded them.
    """
    seen: set[str] = set()
    seq: list[str] = []
    for t in sorted(task_data.get("templates", []), key=lambda x: -x["count"]):
        for p in t["sequence"]:
            if p not in seen:
                seen.add(p)
                seq.append(p)
    return seq


# ── Subject-task concatenation ─────────────────────────────────────────────

_TRIAL_ORDER_RE = re.compile(r"_P(\d+)_t(\d+)$")
_SUBJECT_RE = re.compile(r"(C\d+_N\d+)_T\d+_")


def trial_sort_key(trial_stem: str) -> tuple[int, int]:
    m = _TRIAL_ORDER_RE.search(trial_stem)
    if not m:
        return (10**9, 10**9)
    return (int(m.group(1)), int(m.group(2)))


def subject_of(trial_stem: str) -> str | None:
    m = _SUBJECT_RE.search(trial_stem)
    return m.group(1) if m else None


# ── Positional prior over canonical ────────────────────────────────────────

def positional_score(
    candidates: list[dict],
    canonical: list[str],
    query_pos: int,
    query_len: int,
) -> tuple[str, float] | None:
    """Pick the candidate that best balances fingerprint score × positional prior.

    Returns (piece_id, combined_score) or None if no usable candidate.
    """
    if not candidates or not canonical:
        return None
    canon_n = len(canonical)
    # Expected canonical index if the subject covers canonical pieces uniformly.
    expected = ((query_pos + 0.5) / max(1, query_len)) * canon_n
    canon_index = {pid: i for i, pid in enumerate(canonical)}

    best: tuple[str, float] | None = None
    for c in candidates:
        pid = c.get("piece_id")
        if not pid or pid not in canon_index:
            continue
        score = float(c.get("score", 0.0))
        if score < MIN_FP_SCORE:
            continue
        dist = abs(canon_index[pid] - expected)
        pos_weight = math.exp(-(dist * dist) / (2 * POS_PRIOR_SIGMA * POS_PRIOR_SIGMA))
        combined = score * (0.3 + 0.7 * pos_weight)
        if best is None or combined > best[1]:
            best = (pid, combined)
    return best


# ── Cluster labels (T2) ────────────────────────────────────────────────────

def load_cluster_labels(ds: str) -> dict[str, tuple[str, str]]:
    cl_path = META_ROOT / ds / "set1_ensemble_v2_clusters.json"
    lb_path = META_ROOT / ds / "set1_cluster_labels.json"
    if not cl_path.exists() or not lb_path.exists():
        return {}
    clusters = json.loads(cl_path.read_text(encoding="utf-8"))
    labels_doc = json.loads(lb_path.read_text(encoding="utf-8"))
    label_of = labels_doc.get("labels") or {}
    out: dict[str, tuple[str, str]] = {}
    for c in clusters.get("clusters", []):
        cid = c.get("stable_cluster_id") or c.get("cluster_id")
        lab = label_of.get(cid)
        if not lab:
            continue
        for m in c.get("members", []):
            out[m["piece_key"]] = (cid, lab)
    return out


def cluster_piece_key(raw_stem: str, piece_idx: int) -> str | None:
    m = re.search(r"sub(\d{2})(\d{2})$", raw_stem or "")
    if not m:
        return None
    return f"sub{m.group(1)}_{m.group(1)}{m.group(2)}_p{piece_idx}"


# ── Main resolve ───────────────────────────────────────────────────────────

def pick_best_template(task_canonical: dict, n_pieces: int) -> list[str] | None:
    templates = task_canonical.get("templates", [])
    exact = [t for t in templates if t["n_pieces"] == n_pieces]
    return list(exact[0]["sequence"]) if exact else None


def process_dataset(ds: str, canonical: dict) -> None:
    src = OUT_ROOT / ds / "_pieces.json"
    if not src.exists():
        print(f"[s04] {ds}: no _pieces.json")
        return
    trials: list[dict] = json.loads(src.read_text(encoding="utf-8"))
    cluster_map = load_cluster_labels(ds)

    # ── 1. Assemble subject-task queries (concat of pieces in recording order) ──
    # Key: (subject, task_int) → list of (trial_ref, piece_ref) in chronological order.
    groups: dict[tuple[str, int], list[tuple[dict, dict]]] = defaultdict(list)
    for t in trials:
        subj = subject_of(t.get("trial_stem", ""))
        task = t.get("task")
        if subj is None or task is None:
            continue
        groups[(subj, task)].append(t)

    # Sort trials within each group by natural P#/t# order, then flatten pieces.
    ordered_queries: dict[tuple[str, int], list[tuple[dict, dict]]] = {}
    for key, ts in groups.items():
        ts_sorted = sorted(ts, key=lambda t: trial_sort_key(t.get("trial_stem", "")))
        flat: list[tuple[dict, dict]] = []
        for t in ts_sorted:
            for p in t.get("pieces") or []:
                flat.append((t, p))
        ordered_queries[key] = flat

    # ── 2. Resolve each piece ──
    tally: dict[str, int] = defaultdict(int)
    for (subj, task_int), query in ordered_queries.items():
        task_name = TASK_INT_TO_NAME.get(task_int) or ""
        task_canon = canonical.get("tasks", {}).get(task_name) or {}
        canon_seq = build_canonical_sequence(task_canon)
        query_len = len(query)

        for qi, (trial, p) in enumerate(query):
            ident = p.get("identified") or {}
            candidates = ident.get("candidates") or []
            cand_ids = {c.get("piece_id") for c in candidates if c.get("piece_id")}
            top1 = candidates[0] if candidates else None
            conf = ident.get("confidence")

            resolved: str | None = None
            source = "no_match"
            arrangement_consistent: bool | None = None

            # 1. T2 cluster
            ckey = cluster_piece_key(trial.get("raw_stem", ""), p.get("piece_idx", qi + 1))
            if ckey and ckey in cluster_map:
                cid, lab = cluster_map[ckey]
                ident["cluster_id"] = cid
                ident["cluster_label"] = lab
                resolved = lab
                source = "cluster"
            # 2. Fingerprint high
            elif conf == "high" and top1:
                resolved = top1.get("piece_id")
                source = "fingerprint_high"
            # 3. Arrangement positional (cumulative prior)
            elif canon_seq:
                pp = positional_score(candidates, canon_seq, qi, query_len)
                if pp:
                    pid, _ = pp
                    # Only overwrite fingerprint top-1 if position prior picked a
                    # different candidate (and that candidate is in canonical).
                    if pid != (top1.get("piece_id") if top1 else None):
                        resolved = pid
                        source = "arrangement_positional"
                    else:
                        resolved = top1.get("piece_id") if top1 else None
                        source = "arrangement_verified"
                        arrangement_consistent = True

            # 4. Exact-template verified (legacy check for completeness)
            if resolved is None:
                template = pick_best_template(task_canon, len(trial.get("pieces") or []))
                idx_in_trial = None
                if template is not None:
                    for idx, pp2 in enumerate(trial.get("pieces") or []):
                        if pp2 is p:
                            idx_in_trial = idx
                            break
                if template and idx_in_trial is not None and idx_in_trial < len(template) and template[idx_in_trial] in cand_ids:
                    resolved = template[idx_in_trial]
                    source = "arrangement_verified"
                    arrangement_consistent = True
                    ident["expected_piece_at_position"] = template[idx_in_trial]
                    ident["template_n_pieces"] = len(template)

            # 4b. Canonical-order direct mapping (hardcoded Set1 case)
            # When a hardcoded canonical is flagged and the subject's cumulative
            # query length matches the canonical length exactly, trust the
            # position directly — no fingerprint candidates required. This
            # rescues Set1 pieces with no anchor library and no cluster label.
            if resolved is None and canon_seq and task_canon.get("hardcoded_canonical"):
                if query_len == len(canon_seq) and 0 <= qi < len(canon_seq):
                    resolved = canon_seq[qi]
                    source = "canonical_direct"
                    ident["expected_piece_at_position"] = canon_seq[qi]
                    ident["template_n_pieces"] = len(canon_seq)
                    # No fingerprint evidence backed this label — flag it so
                    # downstream consumers can filter position-prior assignments
                    # apart from evidence-backed ones (cluster / fingerprint /
                    # arrangement_verified).
                    ident["confidence"] = "assumed_from_position"

            # 5. Fingerprint best fallback
            if resolved is None and top1:
                resolved = top1.get("piece_id")
                source = "fingerprint_best"

            ident["resolved_piece_id"] = resolved
            ident["resolution_source"] = source
            if arrangement_consistent is not None:
                ident["arrangement_consistent"] = arrangement_consistent
            p["identified"] = ident
            tally[source] += 1

    out_path = OUT_ROOT / ds / "_pieces_resolved.json"
    out_path.write_text(json.dumps(trials, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[s04] {ds}: wrote {out_path}")
    total = sum(tally.values())
    for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
        pct = (v / total * 100) if total else 0
        print(f"    {k:<24s} {v:>4d}  ({pct:5.1f}%)")
    print(f"    {'TOTAL':<24s} {total:>4d}")


def main() -> None:
    if not CANONICAL_PATH.exists():
        print(f"[s04] missing {CANONICAL_PATH}")
        print("[s04] run: python pipeline/04_piece_split/tools/canonical_order.py first")
        sys.exit(1)
    canonical = json.loads(CANONICAL_PATH.read_text(encoding="utf-8"))
    targets = sys.argv[1:] or [p.name for p in OUT_ROOT.iterdir() if p.is_dir()]
    for ds in targets:
        if ds.startswith("_"):
            continue
        process_dataset(ds, canonical)


def run(dataset: str | None = None, force: bool = False) -> None:
    """Runner entry-point. `force` accepted for runner uniformity."""
    if not CANONICAL_PATH.exists():
        print(f"[s04] missing {CANONICAL_PATH}; skipping arrangement resolve.")
        return
    canonical = json.loads(CANONICAL_PATH.read_text(encoding="utf-8"))
    if dataset:
        targets = [dataset]
    else:
        targets = [p.name for p in OUT_ROOT.iterdir() if p.is_dir()]
    for ds in targets:
        if ds.startswith("_"):
            continue
        process_dataset(ds, canonical)


if __name__ == "__main__":
    main()
