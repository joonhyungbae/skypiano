"""
Stage 04 Step 03: Piece-ID verification via task-aware MIDI fingerprint

Each piece file produced by s02 carries an inherited task label (T0/T1/T2/T9)
from the docx labeling. That label is trusted at the TRIAL level, but within a
trial the individual piece IDs are NOT verified. This step:

  1. For each piece MIDI (`*_pN.mid`), determine its task from the filename
     (T0=Technique1, T1=Technique2, T2=Set1, T9=Free).
  2. Fingerprint the piece MIDI against a task-restricted anchor subset
     (e.g. T0 pieces are compared only to Technique1 anchors in Data/Anchors).
  3. Record the top-K candidate (piece_id, score) into the `_pieces.json`
     manifest under each piece's `identified` field.
  4. Free (T9) has no anchor library — its pieces get reason
     "no_anchors_for_task". A piece already named by manual review keeps
     its `identified` field.

Output: overwrites existing `_pieces.json` with identified fields populated.

Why this belongs in Stage 04 (piece_split):
  Trial-level alignment (Stage 03) is cheap-and-exact via SMPTE. Piece boundary
  detection (Stage 04 step 02) uses silence gaps. Piece content verification
  requires comparison to reference recordings — naturally the last step of the
  piece pipeline. Downstream stages (fingering, imputation) can use this
  identified piece_id to look up per-piece references.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))  # pipeline/04_piece_split/

from src.midi_fingerprint import (  # noqa: E402
    AnchorMeta, AnchorRef,
    build_db, events_from_notes, extract_events, fingerprint,
)

OUT_ROOT = Path("Data/Pipeline/04_piece_split")
ANCHOR_ROOT = Path("Data/Anchors")

DATASETS = ["240828"]

# Stem convention: 240828_C0_N4_T0_P1_t1_p1.mid
STEM_RE = re.compile(r"^(?P<date>\d{6})_C\d+_N\d+_T(?P<task>\d+)_P\d+_t\d+_p\d+$")

# task int → anchor subdir name (under Data/Anchors/, optionally one year deep).
# build_db walks both Data/Anchors/<task>/ and Data/Anchors/<year>/<task>/
# transparently, so we just name the task dir here.
TASK_TO_ANCHOR_DIR = {
    0: "Technique1",
    1: "Technique2",
    2: "Set1",
    # 9 (Free) still has no curated anchor library.
}

TOP_K = 3
MIN_MATCHED = 3          # need ≥3 hash hits to score anything
OFFSET_BIN_SEC = 0.5


def _offset_mode(offsets: list[float]) -> tuple[int, float]:
    if not offsets:
        return 0, 0.0
    buckets: dict[int, int] = defaultdict(int)
    for o in offsets:
        buckets[int(o / OFFSET_BIN_SEC)] += 1
    best_bin, best_cnt = max(buckets.items(), key=lambda kv: kv[1])
    return best_cnt, (best_bin + 0.5) * OFFSET_BIN_SEC


def identify(piece_mid: Path,
             db: dict[int, list[AnchorRef]],
             meta: dict[tuple[str, str], AnchorMeta],
             top_k: int = TOP_K) -> dict:
    """Score each anchor take against the piece; return top-K candidates."""
    events = extract_events(piece_mid)
    fps = fingerprint(events)

    if not fps:
        return {"piece_id": None, "confidence": "empty", "n_fingerprints": 0, "candidates": []}

    hits: dict[tuple[str, str], list[float]] = defaultdict(list)
    for h, q_t in fps:
        if h not in db:
            continue
        for ref in db[h]:
            hits[(ref.piece_id, ref.take_id)].append(q_t - ref.anchor_time)

    scored: list[dict] = []
    for (pid, tid), offs in hits.items():
        m = meta.get((pid, tid))
        if m is None or m.n_fingerprints == 0:
            continue
        if len(offs) < MIN_MATCHED:
            continue
        mode_cnt, mode_off = _offset_mode(offs)
        denom = min(len(fps), m.n_fingerprints) or 1
        scored.append({
            "piece_id": pid,
            "take_id": tid,
            "score": round(mode_cnt / denom, 3),
            "matched": len(offs),
            "coherent": mode_cnt,
            "anchor_fps": m.n_fingerprints,
            "offset": round(mode_off, 2),
        })
    scored.sort(key=lambda d: -d["score"])

    # Collapse per piece_id — keep best take per piece
    per_piece: dict[str, dict] = {}
    for s in scored:
        if s["piece_id"] not in per_piece:
            per_piece[s["piece_id"]] = s
    candidates = list(per_piece.values())[:top_k]

    top = candidates[0] if candidates else None
    confidence = classify_confidence(top)
    return {
        "piece_id": top["piece_id"] if top and confidence != "none" else None,
        "confidence": confidence,
        "n_fingerprints": len(fps),
        "candidates": candidates,
    }


def classify_confidence(top: dict | None) -> str:
    if top is None:
        return "none"
    if top["score"] >= 0.40 and top["coherent"] >= 5 and top["anchor_fps"] >= 10:
        return "high"
    if top["score"] >= 0.20 and top["coherent"] >= 3:
        return "medium"
    if top["coherent"] >= 2:
        return "low"
    return "none"


def task_dbs(anchor_root: Path) -> dict[int, tuple]:
    """Build per-task anchor DBs once."""
    cache: dict[int, tuple] = {}
    for task_int, subdir in TASK_TO_ANCHOR_DIR.items():
        cache[task_int] = build_db(anchor_root, task_filter={subdir})
    return cache


def process_dataset(dataset: str, dbs: dict[int, tuple]) -> None:
    ds_dir = OUT_ROOT / dataset
    manifest = ds_dir / "_pieces.json"
    if not manifest.is_file():
        print(f"  [skip {dataset}] no manifest")
        return

    with open(manifest, encoding="utf-8") as f:
        trials = json.load(f)

    n_pieces = 0
    n_identified = {"manual": 0, "high": 0, "medium": 0, "low": 0, "none": 0, "no_anchors": 0, "empty": 0}
    for trial in trials:
        trial_stem = trial["trial_stem"]
        task = trial.get("task")
        db_meta = dbs.get(task)

        for p in trial["pieces"]:
            n_pieces += 1
            if (p.get("identified") or {}).get("resolution_source") == "manual_review":
                n_identified["manual"] += 1
                continue
            piece_mid = ds_dir / f"{trial_stem}_p{p['piece_idx']}.mid"
            if not piece_mid.is_file():
                p["identified"] = {"piece_id": None, "confidence": "missing", "candidates": []}
                n_identified["none"] += 1
                continue
            if db_meta is None:
                p["identified"] = {
                    "piece_id": None,
                    "confidence": "no_anchors_for_task",
                    "candidates": [],
                }
                n_identified["no_anchors"] += 1
                continue
            db, meta = db_meta
            res = identify(piece_mid, db, meta)
            p["identified"] = res
            conf = res["confidence"]
            n_identified[conf if conf in n_identified else "none"] += 1

    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(trials, f, ensure_ascii=False, indent=2)

    print(f"  [{dataset}] {n_pieces} pieces | "
          + "  ".join(f"{k}={v}" for k, v in n_identified.items() if v))


def main():
    dbs = task_dbs(ANCHOR_ROOT)
    print()
    for ds in DATASETS:
        process_dataset(ds, dbs)


def run(dataset: str | None = None, force: bool = False) -> None:
    """Runner entry-point. `dataset`/`force` accepted for runner uniformity."""
    dbs = task_dbs(ANCHOR_ROOT)
    targets = [dataset] if dataset else DATASETS
    for ds in targets:
        process_dataset(ds, dbs)


if __name__ == "__main__":
    main()
