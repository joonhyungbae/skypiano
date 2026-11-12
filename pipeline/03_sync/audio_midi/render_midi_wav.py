"""
Stage 03 audio_midi: MIDI -> WAV pre-rendering

Renders every Stage 03 audio_midi MIDI file with the SalamanderGrandPiano soundfont so that
web_review can play Audio and MIDI side by side for comparison.

Targets:
  - Sliced MIDI  (Data/Pipeline/03_sync/audio_midi/Sliced/{dataset}/{session}/midi/*.mid)
  - Aligned MIDI (Data/Pipeline/03_sync/audio_midi/Aligned/{dataset}/*.mid)

Outputs:
  - Data/Pipeline/03_sync/audio_midi/Rendered/{dataset}/{session}/{stem}.wav   (Sliced)
  - Data/Pipeline/03_sync/audio_midi/Rendered/{dataset}/_aligned/{stem}.wav     (Aligned)

Usage:
  python render_midi_wav.py [--force]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
from tqdm import tqdm

SAMPLE_RATE = 22050

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]   # 03_sync/audio_midi -> pipeline -> repo root
SF2_PATH = _REPO / (
    "pipeline/03_sync/audio_midi/soundfonts/SalamanderGrandPiano-SF2-V3+20200602/"
    "SalamanderGrandPiano-V3+20200602.sf2"
)

SLICED_ROOT = Path("Data/Pipeline/03_sync/audio_midi/Sliced")
ALIGNED_ROOT = Path("Data/Pipeline/03_sync/audio_midi/Aligned")
RENDER_ROOT = Path("Data/Pipeline/03_sync/audio_midi/Rendered")


def render_midi(midi_path: Path, wav_path: Path):
    """Render a MIDI file with the soundfont and save as WAV"""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    audio = pm.fluidsynth(fs=SAMPLE_RATE, sf2_path=str(SF2_PATH)).astype(np.float32)

    # Normalize to prevent clipping
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.9

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav_path), audio, SAMPLE_RATE)


def collect_jobs(force: bool) -> list[tuple[Path, Path]]:
    """Collect (midi_path, wav_path) pairs to render"""
    jobs = []

    # Sliced MIDI
    if SLICED_ROOT.exists():
        for mid_file in SLICED_ROOT.rglob("midi/*.mid"):
            # .../Sliced/{dataset}/{session}/midi/{stem}.mid
            # -> .../Rendered/{dataset}/{session}/{stem}.wav
            parts = mid_file.relative_to(SLICED_ROOT).parts
            dataset, session = parts[0], parts[1]
            stem = mid_file.stem
            wav_path = RENDER_ROOT / dataset / session / f"{stem}.wav"
            if force or not wav_path.exists():
                jobs.append((mid_file, wav_path))

    # Aligned MIDI
    if ALIGNED_ROOT.exists():
        for mid_file in ALIGNED_ROOT.rglob("*.mid"):
            parts = mid_file.relative_to(ALIGNED_ROOT).parts
            dataset = parts[0]
            stem = mid_file.stem
            wav_path = RENDER_ROOT / dataset / "_aligned" / f"{stem}.wav"
            if force or not wav_path.exists():
                jobs.append((mid_file, wav_path))

    return jobs


def main():
    parser = argparse.ArgumentParser(description="Stage 03 audio_midi MIDI → WAV renderer")
    parser.add_argument("--force", action="store_true", help="Re-render even if WAV exists")
    args = parser.parse_args()

    if not SF2_PATH.exists():
        print(f"SoundFont not found: {SF2_PATH}")
        sys.exit(1)

    jobs = collect_jobs(args.force)
    print(f"SoundFont: {SF2_PATH.name}")
    print(f"Output:    {RENDER_ROOT}")
    print(f"To render: {len(jobs)} files")

    if not jobs:
        print("All files already rendered.")
        return

    success, fail = 0, 0
    for midi_path, wav_path in tqdm(jobs, desc="Rendering"):
        try:
            render_midi(midi_path, wav_path)
            success += 1
        except Exception as e:
            tqdm.write(f"FAIL {midi_path.name}: {e}")
            fail += 1

    print(f"\nDone: success={success} fail={fail}")


if __name__ == "__main__":
    main()
