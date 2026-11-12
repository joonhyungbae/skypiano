"""Stage 11 entry point: align score ↔ performance MIDI per trial.

Usage:
    python run_pipeline.py --samples-dir <dir>     # batch over web sample folders
    python run_pipeline.py --trial <path>          # single trial folder

A trial folder must contain `score.mxl` (or `score.musicxml`) and `midi.mid`.
Writes `alignment.json` next to those files. Skips trials missing either input.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from src.align import align_one  # noqa: E402


SCORE_NAMES = ("score.mxl", "score.musicxml", "score.xml")
MIDI_NAMES = ("midi.mid", "midi.midi")


def find_score(trial: Path) -> Path | None:
    for n in SCORE_NAMES:
        p = trial / n
        if p.exists():
            return p
    return None


def find_midi(trial: Path) -> Path | None:
    for n in MIDI_NAMES:
        p = trial / n
        if p.exists():
            return p
    return None


def process_trial(trial: Path, force: bool = False) -> dict | None:
    score = find_score(trial)
    midi = find_midi(trial)
    if score is None or midi is None:
        print(f"[skip] {trial.name}: score={score is not None} midi={midi is not None}")
        return None
    out = trial / "alignment.json"
    if out.exists() and not force:
        print(f"[skip] {trial.name}: alignment.json exists (use --force to overwrite)")
        return None

    t0 = time.time()
    try:
        result = align_one(score, midi)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {trial.name}: {type(e).__name__}: {e}")
        return None
    dt = time.time() - t0

    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    s = result["stats"]
    print(
        f"[ok]   {trial.name}: {s['matches']}/{s['score_notes']} score "
        f"({s['score_match_rate']:.1%}), {s['matches']}/{s['perf_notes']} perf "
        f"({s['perf_match_rate']:.1%})  [{dt:.1f}s]"
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--samples-dir", type=Path, help="Parent dir of trial folders")
    g.add_argument("--trial", type=Path, help="Single trial folder")
    ap.add_argument("--force", action="store_true", help="Overwrite existing alignment.json")
    args = ap.parse_args()

    trials: list[Path]
    if args.trial:
        trials = [args.trial]
    else:
        trials = sorted(p for p in args.samples_dir.iterdir() if p.is_dir())

    n_ok = sum(1 for t in trials if process_trial(t, force=args.force) is not None)
    print(f"\n{n_ok}/{len(trials)} trials aligned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
