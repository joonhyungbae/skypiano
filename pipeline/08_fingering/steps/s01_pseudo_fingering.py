"""
Stage 08 Step 01: pseudo fingering labels (mocap-adapted)

Uses the geometry-based scoring pipeline in `fingering_mocap.FingeringMocap`
— adapted from prior fingering work — to produce per-note
fingering labels with explicit ambiguity flags (labeled / multi_candidate /
no_candidate). Replaces the earlier heuristic `PianoFingeringAnalyzer`.

Inputs (per piece):
  - Stage 06 Final imputed hand motion: Data/Pipeline/06_hand_imputation/Final/{subj}/{stem}.csv
  - Stage 04 piece-split MIDI:          Data/Pipeline/04_piece_split/{dataset}/{stem}.mid

Output (per piece):
  - Data/Pipeline/08_fingering/Fingering/{dataset}/{stem}_fingering.json
"""
import functools
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fingering_mocap import FingeringMocap  # noqa: E402

print = functools.partial(print, flush=True)

MOTION_ROOT = Path("Data/Pipeline/06_hand_imputation/Final")
MIDI_ROOT = Path("Data/Pipeline/04_piece_split")
OUT_ROOT = Path("Data/Pipeline/08_fingering/Fingering")

DATASETS = ["211215", "220819", "221112", "240523", "240809", "240828"]


def _discover_trials(dataset: str) -> list[dict]:
    """For one dataset, find every piece-level motion+MIDI pair available."""
    midi_dir = MIDI_ROOT / dataset
    if not midi_dir.exists():
        return []

    trials = []
    for midi_path in sorted(midi_dir.glob("*.mid")):
        stem = midi_path.stem  # e.g. 240828_C0_N4_T0_P1_t1_p1
        # Motion CSVs sit under {subject} subdirectories (e.g. C0_N4/)
        # Pull the subject token from the stem: {date}_{class}_{idnum}_...
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        subject_dir = f"{parts[1]}_{parts[2]}"  # C0_N4
        motion_path = MOTION_ROOT / subject_dir / f"{stem}.csv"
        if not motion_path.exists():
            continue
        trials.append({
            "stem": stem,
            "motion_path": str(motion_path),
            "midi_path": str(midi_path),
        })
    return trials


def run():
    t_start = time.time()
    print("Stage 08 s01: Pseudo Fingering (mocap-based, adapted from prior fingering work)")
    print(f"Motion source: {MOTION_ROOT}")
    print(f"MIDI source:   {MIDI_ROOT}")
    print(f"Output:        {OUT_ROOT}")
    print()

    total_files = 0
    total_notes = 0
    total_labeled = 0
    total_multi = 0
    total_none = 0
    failed: list[tuple[str, str, str]] = []

    for dataset in DATASETS:
        trials = _discover_trials(dataset)
        if not trials:
            continue

        out_dir = OUT_ROOT / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"{dataset}: {len(trials)} trials")

        for trial in tqdm(trials, desc=f"  {dataset}", leave=False):
            stem = trial["stem"]
            out_path = out_dir / f"{stem}_fingering.json"
            if out_path.exists():
                total_files += 1
                continue

            try:
                fm = FingeringMocap(trial["motion_path"], trial["midi_path"], fps=120)
                fm.load()
                result = fm.run()

                output = {
                    "stem": stem,
                    "dataset": dataset,
                    **result,
                }
                with open(out_path, "w") as f:
                    json.dump(output, f, indent=2)

                total_files += 1
                total_notes += result["n_notes"]
                total_labeled += result["n_labeled"]
                total_multi += result["n_multi_candidate"]
                total_none += result["n_no_candidate"]

            except Exception as e:
                failed.append((dataset, stem, str(e)))
                tqdm.write(f"    FAIL {dataset}/{stem}: {e}")

    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    print(f"Files written: {total_files}")
    print(f"Total notes:   {total_notes}")
    if total_notes:
        def pct(n):
            return 100 * n / total_notes
        print(f"  labeled:         {total_labeled:6d} ({pct(total_labeled):.1f}%)")
        print(f"  multi-candidate: {total_multi:6d} ({pct(total_multi):.1f}%)")
        print(f"  no-candidate:    {total_none:6d} ({pct(total_none):.1f}%)")
    if failed:
        print(f"Failures: {len(failed)}")
        for ds, stem, err in failed[:10]:
            print(f"  {ds}/{stem}: {err}")
    print(f"Elapsed: {elapsed:.1f} s")


if __name__ == "__main__":
    run()
