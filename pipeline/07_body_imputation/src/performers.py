"""
Performer mapping for Stage 07 (full-body imputation).

Mirrors Stage 06's mapping but points at *_body.csv instead of *_hand.csv.
Rebuilds from each dataset's `_pieces.json` manifest — shared with Stage 06.
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _common.known_absences import all_entries


PIECE_SPLIT_ROOT = Path("Data/Pipeline/04_piece_split")

# Optional dataset filter via env var (comma-separated dataset ids like "240828,240809")
_DATASETS_FILTER = os.environ.get("DATASETS", "").strip()
DATASETS = (
    {d.strip() for d in _DATASETS_FILTER.split(",") if d.strip()}
    if _DATASETS_FILTER else None
)


def _parse_performer_from_trial_stem(trial_stem):
    """
    Extract (C, N) from a trial_stem like {yymmdd}_C?_N?_T{task}_P{piece}_t{trial}.
    """
    parts = trial_stem.split("_")
    if len(parts) < 3:
        return None
    c, n = parts[1], parts[2]
    if not (c.startswith("C") and n.startswith("N")):
        return None
    return (c, n)


def build_performer_map():
    """
    Returns: dict of (C, N) -> list of (dataset, piece_prefix).
    Only pieces whose *_body.csv exists are included.
    """
    performers = defaultdict(list)

    for ds_dir in sorted(PIECE_SPLIT_ROOT.iterdir()):
        if not ds_dir.is_dir():
            continue
        dataset = ds_dir.name
        if not (dataset.isdigit() and len(dataset) == 6):
            continue
        if DATASETS is not None and dataset not in DATASETS:
            continue

        manifest_path = ds_dir / "_pieces.json"
        if not manifest_path.is_file():
            continue

        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        for entry in manifest:
            trial_stem = entry["trial_stem"]
            cn = _parse_performer_from_trial_stem(trial_stem)
            if cn is None:
                continue
            for p in entry.get("pieces", []):
                if p.get("excluded"):
                    continue
                idx = p["piece_idx"]
                piece_prefix = f"{trial_stem}_p{idx}"
                body_csv = ds_dir / f"{piece_prefix}_body.csv"
                if not body_csv.exists():
                    continue
                performers[cn].append((dataset, piece_prefix))

    return dict(performers)


def performer_id(c, n):
    return f"{c}_{n}"


def get_piece_body_path(dataset, piece_prefix):
    return PIECE_SPLIT_ROOT / dataset / f"{piece_prefix}_body.csv"


def build_known_body_absent_trials():
    """Trials whose body stream is known-absent at raw capture time."""
    rows = []
    for entry in all_entries("body_data"):
        dataset = entry.get("scope", {}).get("ds")
        if DATASETS is not None and dataset not in DATASETS:
            continue
        cohort = entry.get("scope", {}).get("cohort")
        for trial_stem in entry.get("selectors", {}).get("trial_stems", []):
            rows.append({
                "dataset": dataset,
                "cohort": cohort,
                "trial_stem": trial_stem,
                "id": entry.get("id"),
                "reason": entry.get("reason", ""),
            })
    return rows


def log_known_body_absences(printer=print):
    """Emit a concise note about source-missing body trials excluded from Stage 07."""
    if not KNOWN_BODY_ABSENT_TRIALS:
        return
    printer(f"  known body-data absences: {len(KNOWN_BODY_ABSENT_TRIALS)} trial(s)")
    for row in KNOWN_BODY_ABSENT_TRIALS:
        printer(f"    - {row['dataset']} {row['trial_stem']} [{row['id']}]")


PERFORMER_MAP = build_performer_map()
KNOWN_BODY_ABSENT_TRIALS = build_known_body_absent_trials()


if __name__ == "__main__":
    total = 0
    for (c, n), pieces in sorted(PERFORMER_MAP.items()):
        datasets = defaultdict(int)
        for ds, _ in pieces:
            datasets[ds] += 1
        ds_str = ", ".join(f"{ds}:{cnt}" for ds, cnt in sorted(datasets.items()))
        pid = performer_id(c, n)
        print(f"  {pid:8s}: {len(pieces):4d} pieces  [{ds_str}]")
        total += len(pieces)
    print(f"\n  Total: {len(PERFORMER_MAP)} performers, {total} pieces")
    if KNOWN_BODY_ABSENT_TRIALS:
        print("")
        log_known_body_absences()
