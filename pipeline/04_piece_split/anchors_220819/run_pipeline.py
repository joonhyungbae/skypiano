"""
Stage 04 Piece split / anchors_220819: per-piece slicing for 220819.

Piece windows come from the recording session log (via the anchor library),
not from MIDI gaps.

Pipeline:
  s01_locate         - find each anchor's window in Stage 01 audio (audio-time)
  s02_align_motion   - per-recording motion<->audio offset (hand + body)
  s03_slice          - cut all 4 modalities at windows, applying motion offsets
  s04_emit_index     - extend Anchors/index.json with paths + alignment metadata

Outputs land in Data/Pipeline/04_piece_split/220819/ in the same per-piece
layout as the other datasets, so downstream stages read them unchanged;
Anchors/index.json additionally records the piece-ID labels.
"""
import sys
from pathlib import Path

_P = Path(__file__).resolve()
while _P.name != "pipeline":
    _P = _P.parent
sys.path.insert(0, str(_P))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from steps.s01_locate import run as s01_run          # noqa: E402
from steps.s02_align_motion import run as s02_run    # noqa: E402
from steps.s03_slice import run as s03_run           # noqa: E402
from steps.s04_emit_index import run as s04_run      # noqa: E402


def main(dataset: str | None = None, force: bool = False) -> None:
    print("=" * 60)
    print("  Stage 04 Piece split / anchors_220819")
    print("=" * 60)
    s01_run(dataset=dataset, force=force)
    s02_run(dataset=dataset, force=force)
    s03_run(dataset=dataset, force=force)
    s04_run(dataset=dataset, force=force)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    main(dataset=args.dataset, force=args.force)
