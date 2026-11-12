"""
Stage 01 Ingest: pair each performance's raw audio / MIDI / motion / video,
cut sessions into trials, and write the canonical per-trial layout with
_mapping.json under Data/Pipeline/01_ingest/Normalized/.

One step per cohort strategy:
  s01_timecode_slice    240523 / 240809 / 240828  SMPTE-timecode slicing
  s02_211215_pair_match 211215                    filename pairing
  s03_220819_match      220819                    signal-xcorr take matching
  s04_221112_slice      221112                    content pairing + take placement
  s05_211026_match      211026                    audio <-> session MIDI pairing (no motion)
"""
import sys
from pathlib import Path

_P = Path(__file__).resolve()
while _P.name != 'pipeline':
    _P = _P.parent
sys.path.insert(0, str(_P))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from steps.s01_timecode_slice import run as s01_run
from steps.s02_211215_pair_match import run as s02_run
from steps.s03_220819_match import run as s03_run
from steps.s04_221112_slice import run as s04_run
from steps.s05_211026_match import run as s05_run


def main(dataset: str | None = None, force: bool = False):
    print("=" * 60)
    print("  Stage 01 Ingest")
    print("=" * 60)
    for run in (s01_run, s02_run, s03_run, s04_run, s05_run):
        run(dataset=dataset, force=force)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    main(dataset=args.dataset, force=args.force)
