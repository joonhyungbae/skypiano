"""
Pass 2 wrapper: MIDI <-> Motion xcorr alignment (post-body-alignment motion).

Thin wrapper around _common.midi_xcorr.run_pass. The algorithm lives in the
common module; this file just declares the dataset roots used by the second
alignment pass, which refines the pass-1 MIDI offsets using body-aligned
motion.

Inputs:
  hand CSVs Data/Pipeline/02_spatial/hand/Aligned (spatially aligned hand)
  MIDI      Data/Pipeline/03_sync/audio_midi/Aligned
  body      Data/Pipeline/02_spatial/body/Aligned (optional)

Output:
  Data/Pipeline/03_sync/midi_motion/Aligned/{dataset}/{stem}.{mid,json}
"""
import sys
from pathlib import Path

_P = Path(__file__).resolve()
while _P.name != 'pipeline': _P = _P.parent
sys.path.insert(0, str(_P))

from _common.midi_xcorr import run_pass

HAND_ROOT = Path("Data/Pipeline/02_spatial/hand/Aligned")
BODY_ROOT = Path("Data/Pipeline/02_spatial/body/Aligned")
MIDI_ROOT = Path("Data/Pipeline/03_sync/audio_midi/Aligned")
OUT_ROOT = Path("Data/Pipeline/03_sync/midi_motion/Aligned")
IMG_ROOT = Path("Data/Pipeline/03_sync/midi_motion/img")

DATASETS = ["211215", "220819", "221112", "240523", "240809", "240828"]


def run():
    run_pass(hand_root=HAND_ROOT, midi_root=MIDI_ROOT, body_root=BODY_ROOT,
             out_root=OUT_ROOT, img_root=IMG_ROOT,
             label="Stage 03 midi_motion s01", datasets=DATASETS)


if __name__ == "__main__":
    run()
