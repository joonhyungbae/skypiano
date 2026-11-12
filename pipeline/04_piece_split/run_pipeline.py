"""
Stage 04 Piece split: cut every aligned trial into per-piece files (audio,
MIDI, hand, body, video) and identify each piece.

24-year datasets (240523 / 240809 / 240828), looped by this runner:
  s01_timecode_slice      trial-level audio from the session WAV (SMPTE)
  s02_piece_split         piece boundaries from MIDI gaps; slice all modalities
  s03_piece_identify      task-aware MIDI fingerprint vs the anchor library
  s04_arrangement_resolve combine fingerprints, clusters and position priors
220819 (session-log piece windows), run with the `anchors` step:
  anchors_220819/         locate logged pieces, final motion offset, slice

Standalone tool (not looped):
  s05_slice_piece_videos  per-piece video slicer

Usage (from the repository root):
  python pipeline/04_piece_split/run_pipeline.py              # s01..s04
  python pipeline/04_piece_split/run_pipeline.py s01 240828   # one step, one dataset
  python pipeline/04_piece_split/run_pipeline.py anchors      # 220819 slicing
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    args = sys.argv[1:]
    step = args[0] if args and not args[0].startswith("-") else None
    dataset = args[1] if len(args) > 1 and not args[1].startswith("-") else None
    force = "--force" in args

    if step is None or step == "s01":
        from steps.s01_timecode_slice import run as run_s01
        run_s01(dataset=dataset, force=force)

    if step is None or step == "s02":
        from steps.s02_piece_split import run as run_s02
        run_s02(dataset=dataset, force=force)

    if step is None or step == "s03":
        # step 3: task-aware piece-id fingerprint vs anchor library
        from steps.s03_piece_identify import run as run_s03
        run_s03(dataset=dataset, force=force)

    if step == "anchors":
        import subprocess
        cmd = [sys.executable, str(Path(__file__).resolve().parent / "anchors_220819" / "run_pipeline.py")]
        if force:
            cmd.append("--force")
        subprocess.run(cmd, check=True)
        return

    if step is None or step == "s04":
        # step 4: arrangement-aware resolution combining clusters + position
        from steps.s04_arrangement_resolve import run as run_s04
        run_s04(dataset=dataset, force=force)


if __name__ == "__main__":
    main()
