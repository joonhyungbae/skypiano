"""
SMPTE-timecode helpers for 24-year sessions (imported by s05_slice_piece_videos:
process_motion_csv, extract_video_timecode, build_deck_video_map).

Run directly, it re-slices whole sessions by timecode as a standalone check;
the production trial slicing is 01_ingest/steps/s01_timecode_slice.

Uses the TimeCode in the OptiTrack Hand CSV to slice continuously recorded
Audio/MIDI into per-trial segments.

Target datasets: 240523, 240809, 240828
"""
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi
from pydub import AudioSegment
from tqdm import tqdm

###############################################################################
# Dataset configuration
###############################################################################
RAW_ROOT = Path("Data/Raw")
OUT_ROOT = Path("Data/Pipeline/04_piece_split/_timecode_check")

# (base_dir, session_prefix_in_motion, audio_stem, video_dirs)
# motion prefix: portion of CSV filename before the trial number (e.g. "240809PJH01")
# audio_stem:    Audio/MIDI filename without extension (e.g. "240809PJH1")
DATASETS = {
    "240523": {
        "sessions": [
            {
                "base": RAW_ROOT / "240523/C1/N0",
                "motion_prefix": "240523sub02",
                "audio_stem": "240523sub02",
                "video_dirs": ["Top", "Left", "Right"],
                "video_naming": "same",  # video filename = motion filename
            },
        ],
    },
    "240809": {
        "sessions": [
            {
                "base": RAW_ROOT / "240809/C0/N0",
                "motion_prefix": "240809PJH01",
                "audio_stem": "240809PJH1",
                "video_dirs": ["Top", "Left", "Right", "RightLow"],
                "video_naming": "deck",
            },
            {
                "base": RAW_ROOT / "240809/C0/N0",
                "motion_prefix": "240809PJH02",
                "audio_stem": "240809PJH2",
                "video_dirs": ["Top", "Left", "Right", "RightLow"],
                "video_naming": "deck",
            },
        ],
    },
    "240828": {
        "sessions": [
            {
                "base": RAW_ROOT / f"240828/C0/N{n}",
                "motion_prefix": f"240828sub{n:02d}",
                "audio_stem": f"240828sub{n:02d}",
                "video_dirs": ["Top", "Left", "Right", "RightLow"],
                "video_naming": "deck",
            }
            for n in [4, 5, 6, 7, 8]
        ],
    },
}

TIMECODE_FPS = 30  # OptiTrack SMPTE timecode fps


###############################################################################
# Timecode parsing
###############################################################################
def convert_timecode_to_seconds(timecode: str) -> float:
    """Convert an HH:MM:SS:FF.xx timecode string to seconds."""
    hh, mm, ss, ff = timecode.split(":")
    ff = ff.split(".")[0]  # drop the sub-frame fractional part
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ff) / TIMECODE_FPS


###############################################################################
# Motion data processing
###############################################################################
def process_motion_csv(file_path: Path):
    """
    Read an OptiTrack Hand CSV and return
    (first_timecode_sec, last_timecode_sec, dataframe).
    """
    data = pd.read_csv(file_path, encoding="latin1", header=2, low_memory=False)
    data = data.iloc[3:, 2:]  # skip ID/Position/header rows, skip Frame/Time columns

    # Clean up column names
    cols = data.columns.tolist()
    cols = [
        c.replace("Markerset 001:", "").replace("Unnamed: 2", "TimeCode")
        for c in cols
    ]
    data.columns = cols

    # Extract TimeCode for rows that contain valid data
    valid_mask = data.iloc[:, 1:].notna().any(axis=1)
    valid_timecodes = data.loc[valid_mask, "TimeCode"]
    valid_seconds = valid_timecodes.apply(convert_timecode_to_seconds)

    first_sec = valid_seconds.iloc[0]
    last_sec = valid_seconds.iloc[-1]

    return first_sec, last_sec, data


###############################################################################
# Audio trimming
###############################################################################
def load_audio(audio_path: Path) -> AudioSegment:
    """Load a WAV file once."""
    return AudioSegment.from_wav(str(audio_path))


def trim_audio(audio: AudioSegment, start_sec: float, end_sec: float, out_path: Path):
    """Trim a pre-loaded Audio to the [start_sec, end_sec] range and save it."""
    trimmed = audio[int(start_sec * 1000) : int(end_sec * 1000)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trimmed.export(str(out_path), format="wav")


###############################################################################
# MIDI trimming
###############################################################################
def trim_midi(midi_path: Path, start_sec: float, end_sec: float, out_path: Path):
    """Trim a MIDI file to the [start_sec, end_sec] range and save it (shifted to start at time 0)."""
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    new_midi = pretty_midi.PrettyMIDI()

    for inst in midi.instruments:
        new_inst = pretty_midi.Instrument(
            program=inst.program, is_drum=inst.is_drum, name=inst.name
        )
        for note in inst.notes:
            if note.start < end_sec and note.end > start_sec:
                new_inst.notes.append(
                    pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=max(0.0, note.start - start_sec),
                        end=min(end_sec - start_sec, note.end - start_sec),
                    )
                )
        for cc in inst.control_changes:
            if start_sec <= cc.time < end_sec:
                new_inst.control_changes.append(
                    pretty_midi.ControlChange(
                        number=cc.number,
                        value=cc.value,
                        time=cc.time - start_sec,
                    )
                )
        new_midi.instruments.append(new_inst)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_midi.write(str(out_path))


###############################################################################
# Video timecode extraction (ffmpeg)
###############################################################################
def extract_video_timecode(video_path: Path) -> float | None:
    """Use ffmpeg to read the timecode from video metadata and return it in seconds."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", str(video_path), "-f", "ffmetadata", "-"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stderr.split("\n"):
            if "timecode" in line.lower() and ": " in line:
                tc = line.split(": ", 1)[1].strip()
                if re.match(r"\d{2}:\d{2}:\d{2}:\d{2}", tc):
                    hh, mm, ss, ff = map(int, tc.split(":"))
                    return hh * 3600 + mm * 60 + ss + ff / 60
    except Exception:
        pass
    return None


###############################################################################
# Video trimming (ffmpeg)
###############################################################################
def trim_video_ffmpeg(
    video_path: Path,
    motion_start_sec: float,
    motion_end_sec: float,
    video_start_sec: float,
    out_path: Path,
):
    """Use ffmpeg to trim the video by timecode and save it."""
    offset = motion_start_sec - video_start_sec
    duration = motion_end_sec - motion_start_sec
    if offset < 0:
        print(f"    [WARN] video starts later than motion: {video_path.name}")
        duration += offset
        offset = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{offset:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-c", "copy",
        str(out_path),
    ]
    subprocess.run(cmd, capture_output=True, timeout=300)


###############################################################################
# Deck video matching (240809, 240828)
###############################################################################
def build_deck_video_map(video_dir: Path) -> list[tuple[int, Path]]:
    """
    Return deck video files as a list of (take_number, path).
    Filename format: {deck}-Deck_{YYMMDDTTTT}_{NNNN}.mp4
    """
    result = []
    if not video_dir.exists():
        return result
    for f in sorted(video_dir.iterdir()):
        if f.suffix.lower() != ".mp4":
            continue
        m = re.search(r"_(\d{4})\.mp4$", f.name)
        if m:
            result.append((int(m.group(1)), f))
    return sorted(result, key=lambda x: x[0])


def get_take_number_from_csv(csv_path: Path) -> int | None:
    """Read Capture Start Frame from the CSV header and return it as a take identifier."""
    line = csv_path.read_text(encoding="latin1").split("\n")[0]
    parts = line.split(",")
    # Find Capture Start Frame
    for i, p in enumerate(parts):
        if p.strip() == "Capture Start Frame":
            return int(parts[i + 1].strip())
    return None


###############################################################################
# Main processing
###############################################################################
def process_session(dataset_id: str, session: dict, delay: float = 0.0):
    """Run timecode alignment for a single session (one Audio/MIDI plus its trials)."""
    base = session["base"]
    motion_prefix = session["motion_prefix"]
    audio_stem = session["audio_stem"]
    video_naming = session["video_naming"]
    video_dir_names = session["video_dirs"]

    motion_dir = base / "Motion" / "Hand"
    audio_path = base / "Audio" / f"{audio_stem}.wav"
    midi_path = base / "MIDI" / f"{audio_stem}.mid"

    out_base = OUT_ROOT / dataset_id / audio_stem

    if not motion_dir.exists():
        print(f"  [SKIP] Motion directory not found: {motion_dir}")
        return

    # Collect motion CSV files (sorted by trial number)
    csv_files = sorted([
        f for f in motion_dir.iterdir()
        if f.suffix.lower() == ".csv" and f.stem.startswith(motion_prefix)
    ])

    if not csv_files:
        print(f"  [SKIP] no motion files (prefix={motion_prefix})")
        return

    print(f"\n  session: {audio_stem} ({len(csv_files)} trials)")
    print(f"  Audio: {audio_path.name} (exists={audio_path.exists()})")
    print(f"  MIDI:  {midi_path.name} (exists={midi_path.exists()})")

    # Load Audio just once
    audio_data = None
    if audio_path.exists():
        print(f"  loading Audio...")
        audio_data = load_audio(audio_path)
        print(f"  Audio loaded ({len(audio_data)/1000:.1f}s)")

    for csv_path in tqdm(csv_files, desc=f"  {audio_stem}"):
        trial_stem = csv_path.stem  # e.g. 240809PJH0103
        trial_num = trial_stem[len(motion_prefix):]  # e.g. 03

        # 1) Extract timecode
        try:
            first_sec, last_sec, motion_data = process_motion_csv(csv_path)
        except Exception as e:
            print(f"    [ERR] motion parsing failed {trial_stem}: {e}")
            continue

        start_sec = first_sec + delay
        end_sec = last_sec + delay

        # 2) Trim Audio
        if audio_data is not None:
            out_audio = out_base / "audio" / f"{trial_stem}.wav"
            if not out_audio.exists():
                try:
                    trim_audio(audio_data, start_sec, end_sec, out_audio)
                except Exception as e:
                    print(f"    [ERR] Audio trimming failed: {e}")

        # 3) Trim MIDI
        if midi_path.exists():
            out_midi = out_base / "midi" / f"{trial_stem}.mid"
            if not out_midi.exists():
                try:
                    trim_midi(midi_path, start_sec, end_sec, out_midi)
                except Exception as e:
                    print(f"    [ERR] MIDI trimming failed: {e}")

        # 4) Save Motion (valid range only)
        out_motion = out_base / "motion" / f"{trial_stem}.csv"
        if not out_motion.exists():
            out_motion.parent.mkdir(parents=True, exist_ok=True)
            motion_data.to_csv(str(out_motion), index=False)

        # 5) Trim Video
        if video_naming == "same":
            # 240523: video filename = motion filename
            for vdir_name in video_dir_names:
                video_path = base / "Video" / vdir_name / f"{trial_stem}.mp4"
                if video_path.exists():
                    out_video = out_base / "video" / vdir_name / f"{trial_stem}.mp4"
                    if not out_video.exists():
                        tc_sec = extract_video_timecode(video_path)
                        if tc_sec is not None:
                            trim_video_ffmpeg(
                                video_path, first_sec, last_sec, tc_sec, out_video,
                            )
                        else:
                            print(f"    [WARN] no timecode: {video_path.name}")

        # deck videos are handled separately below


def match_and_trim_deck_videos(dataset_id: str, session: dict):
    """
    Match deck videos against motion timecodes and trim them.
    Match by comparing the video's embedded timecode to the motion timecode range.
    """
    if session["video_naming"] != "deck":
        return

    base = session["base"]
    motion_prefix = session["motion_prefix"]
    audio_stem = session["audio_stem"]
    video_dir_names = session["video_dirs"]
    out_base = OUT_ROOT / dataset_id / audio_stem

    motion_dir = base / "Motion" / "Hand"
    csv_files = sorted([
        f for f in motion_dir.iterdir()
        if f.suffix.lower() == ".csv" and f.stem.startswith(motion_prefix)
    ])

    if not csv_files:
        return

    # Collect each trial's timecode range
    trials = []
    for csv_path in csv_files:
        try:
            first_sec, last_sec, _ = process_motion_csv(csv_path)
            trials.append({
                "stem": csv_path.stem,
                "start": first_sec,
                "end": last_sec,
            })
        except Exception:
            continue

    if not trials:
        return

    # Process each video directory
    for vdir_name in video_dir_names:
        vdir = base / "Video" / vdir_name
        if not vdir.exists():
            continue

        videos = build_deck_video_map(vdir)
        if not videos:
            continue

        # Extract timecode per video
        video_infos = []
        for take_num, vpath in videos:
            tc_sec = extract_video_timecode(vpath)
            if tc_sec is not None:
                video_infos.append({"path": vpath, "take": take_num, "start": tc_sec})

        if not video_infos:
            print(f"    [WARN] {vdir_name}: timecode extraction failed")
            continue

        # Matching: check whether the motion trial's start timecode falls inside a video's range
        for trial in trials:
            best_video = None
            for vi in video_infos:
                # the video must start no later than the motion
                if vi["start"] <= trial["start"]:
                    if best_video is None or vi["start"] > best_video["start"]:
                        best_video = vi

            if best_video is not None:
                out_video = (
                    out_base / "video" / vdir_name / f"{trial['stem']}.mp4"
                )
                if not out_video.exists():
                    trim_video_ffmpeg(
                        best_video["path"],
                        trial["start"],
                        trial["end"],
                        best_video["start"],
                        out_video,
                    )


def run():
    print("=" * 70)
    print("Timecode session re-slice (standalone check)")
    print(f"Output: {OUT_ROOT}")
    print("=" * 70)

    for dataset_id, config in DATASETS.items():
        print(f"\n{'─' * 60}")
        print(f"  dataset: {dataset_id}")
        print(f"{'─' * 60}")

        for session in config["sessions"]:
            process_session(dataset_id, session)
            match_and_trim_deck_videos(dataset_id, session)

    print(f"\n{'=' * 70}")
    print("done")


if __name__ == "__main__":
    run()
