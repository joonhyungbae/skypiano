"""
Stage 03 Sync: put audio, MIDI, video and motion on one timeline per trial.

  midi_motion/  (run in this order)
    s00_audio_video_sync      220819: per-trial offsets into the session videos
    s01_align_midi_motion     22-year: MIDI <-> spatially aligned motion xcorr
    s02_refine_pertrial_sync  per-trial video offsets (Left, RightFull)
    s03_timecode_align        24-year: cut trial MIDI from the session MIDI
                              by SMPTE timecode (no cross-correlation needed)
  audio_midi/   standalone tools, not run by this runner
    steps/s03_align.py              MAESTRO-style CQT DTW audio<->MIDI warp
    steps/s03_realign_211215.py     211215 sync-note + DTW re-alignment
    steps/s01_align_midi_motion.py  legacy first-pass MIDI <-> motion xcorr
    render_midi_wav.py              render MIDI to WAV for listening checks

Audio and MIDI were recorded together by the DAW, so per-trial audio and MIDI
share a clock; this stage aligns motion (and video) to that clock.

Outputs: Data/Pipeline/03_sync/{midi_motion,audio_midi}/...
220819's final per-recording motion offset is computed in
04_piece_split/anchors_220819 (s02_align_motion) right before slicing.
"""
import subprocess
import sys
from pathlib import Path

STEPS = ["s00_audio_video_sync", "s01_align_midi_motion",
         "s02_refine_pertrial_sync", "s03_timecode_align"]


def main():
    steps_dir = Path(__file__).resolve().parent / "midi_motion" / "steps"
    print("=" * 60)
    print("  Stage 03 Sync")
    print("=" * 60)
    for step in STEPS:
        subprocess.run([sys.executable, str(steps_dir / f"{step}.py")], check=True)


if __name__ == "__main__":
    main()
