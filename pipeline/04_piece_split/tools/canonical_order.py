"""
Build a canonical piece-order summary per task from hand-labeled arrangement
CSVs (220819, 221112 only — 240828 has none).

Input:  Data/Metadata/{date}/{C}/{N}/{date}_{C}_{N}_arrangement.csv
Output: Data/Metadata/canonical_order.json

The output supports downstream arrangement-aware piece identification:
  {
    "tasks": {
      "Technique1": {
        "pieces": ["4","6","7","10 - Forte", ...],     # all piece_ids observed
        "templates": [
          {"n_pieces": 6, "sequence": ["4","6","7","10 - Forte","10 - Piano","13"], "count": 5},
          {"n_pieces": 10, "sequence": [...], "count": 3},
          ...
        ],
        "position_pmf": [  # per 0-based position, how often each piece appears
          {"4": 0.38, "6": 0.21, "7": 0.16, ...},
          {"6": 0.32, "7": 0.25, "10 - Forte": 0.18, ...},
          ...
        ]
      },
      "Technique2": {...}
    }
  }

Key insight: `templates` gives the exact sequence observed in arrangement
CSVs when a trial contains N pieces. If a 240828 trial's detected piece
count matches one of these templates, we can map trial piece k → template[k].
`position_pmf` gives a softer prior useful when templates don't match.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

META_ROOT = Path("Data/Metadata")
OUTPUT = META_ROOT / "canonical_order.json"

# Task name mapping — arrangement CSV "Task" column → canonical task key
TASK_NORMALIZE = {
    "Technique1": "Technique1",
    "Technique2": "Technique2",
    "Set1": "Set1",
    "Free": "Free",
}

# Ad-hoc / experimental entries that slipped into some session arrangement CSVs
# but are not part of the real Fink or Technique 2 piece library. Excluded from
# the canonical sequence so they don't pollute positional priors downstream.
PIECE_ID_EXCLUDE = {
    "Technique1": {"New (\uc67c\uc190\ube60\ub974\uac8c)"},
    "Technique2": set(),
    "Set1": set(),
    "Free": set(),
}

# Hardcoded canonical orders for tasks that don't have arrangement CSVs at all
# (240828 has no Set1 arrangement metadata, but all subjects played in the
# fixed order published in Data/Score/set1.pdf). Injected as a synthetic
# template with a high "count" so the resolver treats it as authoritative.
HARDCODED_TEMPLATES: dict[str, list[str]] = {
    "Set1": [
        "chopin_prelude_op28_no4",          #  1
        "chopin_waltz_op64_no2",            #  2
        "chopin_waltz_op18_eb",             #  3
        "chopin_minute_waltz",              #  4
        "badarzewska_maidens_prayer",       #  5  Bądarzewska Op.4 (was mislabeled as Chopin/Liszt's Maiden's Wish)
        "beethoven_moonlight_1",            #  6
        "mozart_k545_1",                    #  7
        "silvery_waves",                    #  8  (Korean title: "Silvery Waves")
        "schumann_traumerei",               #  9
        "schubert_impromptu_op90_no4",      # 10
        "bach_prelude_c_wtc1",              # 11
        "mozart_twinkle_var",               # 12
        "bach_minuet_g",                    # 13
        "chopin_raindrop",                  # 14
        "chopin_nocturne_op9_no2",          # 15
        "clementi_op36_no1",                # 16
    ],
}


def parse_arrangement(path: Path) -> list[dict]:
    """Return list of rows: {id, task, start, end, renamed}. Skip header + blanks."""
    out: list[dict] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return out
    header = [h.strip() for h in rows[0]]
    # Map column names (case-insensitive) to indexes
    idx: dict[str, int] = {}
    for i, h in enumerate(header):
        idx[h.lower()] = i
    def get(row: list[str], name: str) -> str:
        i = idx.get(name.lower())
        return row[i].strip() if i is not None and i < len(row) else ""

    for row in rows[1:]:
        if not row or not any(row):
            continue
        pid = get(row, "id")
        task = get(row, "task")
        if not pid or not task:
            continue
        norm = TASK_NORMALIZE.get(task)
        if norm is None:
            continue
        if pid in PIECE_ID_EXCLUDE.get(norm, set()):
            continue
        try:
            start = float(get(row, "start")) if get(row, "start") else None
            end = float(get(row, "end")) if get(row, "end") else None
        except ValueError:
            start = end = None
        out.append({
            "id": pid,
            "task": norm,
            "start": start,
            "end": end,
            "renamed": get(row, "renamed"),
        })
    return out


def group_by_trial(rows: list[dict]) -> dict[str, list[dict]]:
    """Group rows by `renamed` (trial stem). Preserve arrangement-CSV order."""
    by_trial: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["renamed"]:
            by_trial[r["renamed"]].append(r)
    return by_trial


def build_canonical(rows: list[dict]) -> dict[str, Any]:
    """Build per-task templates + position PMFs from all arrangement rows."""
    # All rows, grouped by (task, trial_stem), preserving order within trial.
    by_task: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if not r["renamed"]:
            continue
        by_task[r["task"]][r["renamed"]].append(r["id"])

    out: dict[str, Any] = {"tasks": {}}
    for task, trials in by_task.items():
        pieces_seen = Counter()
        templates = Counter()  # tuple of piece_ids → count
        position_counts: list[Counter] = []
        for stem, seq in trials.items():
            templates[tuple(seq)] += 1
            for pid in seq:
                pieces_seen[pid] += 1
            # Extend position_counts if needed
            while len(position_counts) < len(seq):
                position_counts.append(Counter())
            for i, pid in enumerate(seq):
                position_counts[i][pid] += 1

        # Templates sorted by frequency
        tmpls = [
            {"n_pieces": len(seq), "sequence": list(seq), "count": cnt}
            for seq, cnt in templates.most_common()
        ]
        # Position PMF (normalized per position)
        pmf = []
        for pos_counter in position_counts:
            total = sum(pos_counter.values())
            pmf.append({p: round(c / total, 4) for p, c in pos_counter.most_common()})

        out["tasks"][task] = {
            "pieces": sorted(pieces_seen.keys()),
            "n_trials_observed": len(trials),
            "templates": tmpls,
            "position_pmf": pmf,
        }

    # ── Inject hardcoded canonical orders for tasks with no arrangement CSVs ──
    for task, seq in HARDCODED_TEMPLATES.items():
        task_out = out["tasks"].setdefault(task, {
            "pieces": [],
            "n_trials_observed": 0,
            "templates": [],
            "position_pmf": [],
        })
        # Insert the hardcoded template as the highest-ranked entry (count=999)
        # so s04 picks it when matching by n_pieces. Don't overwrite an existing
        # exact-length template from real data — just insert ahead of it.
        synthetic = {"n_pieces": len(seq), "sequence": list(seq), "count": 999}
        task_out["templates"] = [synthetic] + [
            t for t in task_out["templates"] if tuple(t["sequence"]) != tuple(seq)
        ]
        # Back-fill pieces list + position_pmf if previously empty.
        if not task_out["pieces"]:
            task_out["pieces"] = sorted(set(seq))
        if not task_out["position_pmf"]:
            task_out["position_pmf"] = [{pid: 1.0} for pid in seq]
        task_out["hardcoded_canonical"] = True

    return out


def main() -> None:
    arr_paths = sorted(META_ROOT.glob("*/C*/N*/*_arrangement.csv"))
    print(f"[canonical_order] scanning {len(arr_paths)} arrangement CSV files")
    all_rows: list[dict] = []
    for p in arr_paths:
        rows = parse_arrangement(p)
        print(f"  {p.relative_to(META_ROOT)}: {len(rows)} rows")
        all_rows.extend(rows)
    print(f"[canonical_order] total rows: {len(all_rows)}")
    result = build_canonical(all_rows)
    result["_meta"] = {
        "source_files": [str(p.relative_to(META_ROOT)).replace("\\", "/") for p in arr_paths],
        "n_source_files": len(arr_paths),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[canonical_order] wrote {OUTPUT}")
    # Summary
    for task, info in result["tasks"].items():
        print(f"  {task}: {len(info['pieces'])} distinct pieces, "
              f"{info['n_trials_observed']} trials, "
              f"{len(info['templates'])} distinct templates")


if __name__ == "__main__":
    main()
