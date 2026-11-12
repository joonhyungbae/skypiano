"""
Stage 02 Spatial: bring every trial's motion into one metric, keyboard-centred
coordinate frame.

  hand/  OptiTrack hand markers (from Stage 01 Normalized)
         s01_unify_columns     canonical {joint}_{xyz} columns, 100->120 fps,
                               wrist joint centre (styloid midpoint) for 2022
         s02_restore_markers   restore missing keyboard end markers
         s03_center            keyboard midpoint -> origin
         s04_level_240809      level the reference dataset, save reference
         s05_cross_year_align  rotate every dataset onto the reference frame
                               (rigid; no scaling)
  body/  Qualisys body markers registered onto the hand frame
         (wrist-speed time offset + weighted Horn alignment)

Outputs: Data/Pipeline/02_spatial/{hand,body}/...

The body aligner runs as a separate process: hand/ and body/ each ship a
package named `src`, which cannot both be imported in one interpreter.

Usage (from the repository root):
  python pipeline/02_spatial/run_pipeline.py                 # all datasets
  python pipeline/02_spatial/run_pipeline.py --dataset 220819
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))        # pipeline/ (for _common)
sys.path.insert(0, str(HERE / "hand"))      # hand/src, hand/steps

from steps import s01_unify_columns as s01, s02_restore_markers as s02, s03_center as s03  # noqa: E402
from steps import s04_level_240809 as s04, s05_cross_year_align as s05                    # noqa: E402
from src.markers import DATASET_CONFIGS                                                    # noqa: E402


def run_hand(dataset: str | None):
    if dataset is None:
        for step in (s01, s02, s03, s04, s05):
            step.run()
        return
    cfg = DATASET_CONFIGS[dataset]
    s01.process_dataset(dataset, cfg)
    s02.auto_restore_ends(dataset)
    s02.fill_ends_from_neighbours(dataset)
    s02.process_dataset(dataset, cfg)
    s03.process_dataset(dataset)
    if dataset == "240809":
        s04.run()                       # the reference dataset is levelled, not cross-aligned
        return
    ref = s05.load_reference_params()
    if dataset in s05.FOUR_PT_DATASETS:
        s05.align_4pt_dataset(dataset, ref)
    elif dataset in s05.TWO_PT_DATASETS:
        s05.align_2pt_dataset(dataset, ref)


def run_body(dataset: str | None):
    cmd = [sys.executable, str(HERE / "body" / "run_pipeline.py")]
    if dataset:
        cmd += ["--dataset", dataset]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--hand-only", action="store_true")
    ap.add_argument("--body-only", action="store_true")
    a = ap.parse_args()
    print("=" * 60)
    print("  Stage 02 Spatial")
    print("=" * 60)
    if not a.body_only:
        run_hand(a.dataset)
    if not a.hand_only:
        run_body(a.dataset)


if __name__ == "__main__":
    main()
