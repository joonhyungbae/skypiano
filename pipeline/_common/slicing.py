"""Audio / MIDI slicing helpers used by multiple segment-stage scripts.

Extracted from the former stage1_preprocess/steps/s01_timecode.py during the
pipeline consolidation. Two functions that cut session-level audio (WAV) and
MIDI files down to a [start_sec, end_sec] trial window.
"""
from pathlib import Path

import pretty_midi
from pydub import AudioSegment


def slice_audio(audio: AudioSegment, start_sec: float, end_sec: float, out_path: Path) -> None:
    # A window starting before the recording is padded with silence, so the
    # slice keeps the window's clock (an AudioSegment index < 0 counts from the end).
    lead = AudioSegment.silent(duration=int(round(-start_sec * 1000)), frame_rate=audio.frame_rate)         if start_sec < 0 else AudioSegment.empty()
    trimmed = lead.set_channels(audio.channels).set_sample_width(audio.sample_width) +         audio[max(0, int(start_sec * 1000)): int(end_sec * 1000)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trimmed.export(str(out_path), format="wav")


def slice_midi(midi_path: Path, start_sec: float, end_sec: float, out_path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    new_pm = pretty_midi.PrettyMIDI()
    duration = end_sec - start_sec
    for inst in pm.instruments:
        new_inst = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        for note in inst.notes:
            s, e = max(0.0, note.start - start_sec), min(duration, note.end - start_sec)
            if note.start < end_sec and note.end > start_sec and e > s:   # zero-length notes cannot be written
                new_inst.notes.append(pretty_midi.Note(velocity=note.velocity, pitch=note.pitch, start=s, end=e))
        for cc in inst.control_changes:
            if start_sec <= cc.time < end_sec:
                new_inst.control_changes.append(pretty_midi.ControlChange(
                    number=cc.number, value=cc.value, time=cc.time - start_sec,
                ))
        new_pm.instruments.append(new_inst)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_pm.write(str(out_path))
