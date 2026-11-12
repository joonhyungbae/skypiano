"""
Stage 08 Step 02: Export release-grade fingering annotations.

Splits the annotate-tool's working file (`human_verdicts.json`) into a
release artifact that contains ONLY entries an annotator explicitly pressed
a finger button for. Bulk-accepted model preloads stay in their original
files (`algorithm.json` from Stage 08 Step 01, `imputation.json` from Stage 06 hand imputation)
and are NOT republished as "human" data.

Background — why this step exists:
The annotate tool, for ergonomics, populates verdicts with model picks at
load time so the annotator can scroll an "everything-already-labeled" piano
roll. Each preloaded entry carries `source: 'algo' | 'imputed'` so it can be
distinguished from a real human press (which has `timestamp` and no `source`).
This step performs that distinction once, in code, so downstream consumers
(corpus stats, paper figures, public release) cannot accidentally count a
preload as a human annotation.

Definition of "human annotation" used here (single source of truth):
    An entry in human_verdicts.json is a human annotation iff
        entry.get("timestamp")  is truthy   AND
        entry.get("source")     is absent / falsy

Anything else in human_verdicts.json is a model preload that the annotator
left untouched, and must NOT be reported as human-verified.

Inputs (per annotation pool):
  - Data/web_annotate/data/human_verdicts.json   (annotate tool output)
  - Data/web_annotate/data/algorithm.json        (Stage 08 Step 01 model picks)
  - Data/web_annotate/data/imputation.json       (Stage 06 imputer picks)

Output (alongside the inputs):
  - Data/web_annotate/data/human_annotations.json
        Filtered to {note_id: entry} where entry has `timestamp` and NO
        `source` field. Each entry retains its original recorded fields plus
        a `note_id` echo for self-contained downstream use.
  - Data/web_annotate/data/_export_summary.json
        Counts by provenance, plus human-vs-model agreement breakdown.

Downstream rule:
Any script that prints a number representing "human-verified" / "human-
annotated" / "human-corrected" data for the paper or public corpus must
read from `human_annotations.json` only. Do not re-derive the same
filter elsewhere — the definition lives here.
"""
from __future__ import annotations

import functools
import json
import sys
from collections import Counter
from pathlib import Path

print = functools.partial(print, flush=True)

DATA_DIR = Path("Data/web_annotate/data")
VERDICTS_PATH = DATA_DIR / "human_verdicts.json"
ALGORITHM_PATH = DATA_DIR / "algorithm.json"
IMPUTATION_PATH = DATA_DIR / "imputation.json"
OUT_HUMAN = DATA_DIR / "human_annotations.json"
OUT_SUMMARY = DATA_DIR / "_export_summary.json"


def _load_json(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"missing input: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _is_human_pressed(entry: dict) -> bool:
    """True iff the entry was created by an explicit human button press.

    The annotate tool stamps `timestamp` on every press in setVerdictAt().
    Bulk-accepted preloads carry `source: 'algo' | 'imputed'` and never
    receive a timestamp. The two conditions are independent, so we require
    both: timestamp present AND source absent.
    """
    return bool(entry.get("timestamp")) and not entry.get("source")


def main() -> None:
    verdicts = _load_json(VERDICTS_PATH)
    algorithm = _load_json(ALGORITHM_PATH)
    imputation = _load_json(IMPUTATION_PATH)

    provenance = Counter()
    human_only: dict[str, dict] = {}
    for note_id, entry in verdicts.items():
        if _is_human_pressed(entry):
            provenance["human_pressed"] += 1
            human_only[note_id] = {"note_id": note_id, **entry}
        else:
            src = entry.get("source") or "unknown"
            provenance[f"preloaded_{src}"] += 1

    # Sanity: every note in human_only must also appear in the model files,
    # so analysts can join human pick ↔ algorithm pick ↔ imputed pick.
    missing_algo = [k for k in human_only if k not in algorithm]
    missing_imp = [k for k in human_only if k not in imputation]

    OUT_HUMAN.write_text(
        json.dumps(human_only, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )

    # Override / agreement breakdown for human-pressed entries against the
    # model picks they replaced.
    overrode_algo = 0
    matched_algo = 0
    overrode_imputed = 0
    matched_imputed = 0
    for note_id, entry in human_only.items():
        h = entry["finger_int"]
        if h < 1 or h > 10:
            continue  # 0 = unsure, -1 = no_press: not directly comparable
        a = algorithm.get(note_id, {}).get("int")
        if a is not None:
            if a == h:
                matched_algo += 1
            else:
                overrode_algo += 1
        i = imputation.get(note_id, {}).get("imputed_int")
        if i is not None:
            if i == h:
                matched_imputed += 1
            else:
                overrode_imputed += 1

    summary = {
        "verdicts_total": len(verdicts),
        "provenance": dict(provenance),
        "human_annotations_emitted": len(human_only),
        "human_vs_algorithm": {
            "matched": matched_algo,
            "overrode": overrode_algo,
        },
        "human_vs_imputed": {
            "matched": matched_imputed,
            "overrode": overrode_imputed,
        },
        "missing_algorithm_entries": missing_algo,
        "missing_imputation_entries": missing_imp,
    }
    OUT_SUMMARY.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"verdicts read:           {len(verdicts):>6} from {VERDICTS_PATH}")
    for k, v in sorted(provenance.items()):
        print(f"  {k:<24} {v:>6}")
    print(f"human_annotations.json:  {len(human_only):>6} → {OUT_HUMAN}")
    print(f"summary:                 {OUT_SUMMARY}")
    if missing_algo or missing_imp:
        print(
            f"WARNING: {len(missing_algo)} human entries missing algorithm join, "
            f"{len(missing_imp)} missing imputation join — see summary file."
        )


if __name__ == "__main__":
    main()
