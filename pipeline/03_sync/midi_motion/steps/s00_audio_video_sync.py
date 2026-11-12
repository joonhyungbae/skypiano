"""
Stage 03 midi_motion Step 00: Audio-Video synchronization (220819)

Runs BEFORE the MIDI<->motion xcorr alignment (s01) — establishes per-trial
offsets within the continuous session videos so subsequent steps can address
the right video frames. 220819 is the only dataset with this layout.

Computes the offset of each trial within the video by cross-correlating the
per-trial WAV against the audio embedded in the video.

Video types:
  - Session-continuous (Right, Top): a single file covering the whole session
  - per-trial (Left, RightFull): recorded per trial, attempt 1:1 matching

Output: Data/Pipeline/03_sync/midi_motion/Aligned/220819/{stem}.json
"""
import functools
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

print = functools.partial(print, flush=True)

RAW_ROOT = Path("Data/Raw/220819/C0/N0")
AUDIO_DIR = RAW_ROOT / "Audio"
VIDEO_DIR = RAW_ROOT / "Video"
OUT_ROOT = Path("Data/Pipeline/03_sync/midi_motion/Aligned/220819")

SAMPLE_RATE = 48000
DOWNSAMPLE = 8  # work at 6kHz for speed
EFFECTIVE_SR = SAMPLE_RATE // DOWNSAMPLE


def load_wav_mono(path: Path, sr=SAMPLE_RATE) -> np.ndarray:
    """Load a WAV as mono float32"""
    rate, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if rate != sr:
        # Simple resample by ratio
        ratio = sr / rate
        n = int(len(data) * ratio)
        indices = np.linspace(0, len(data) - 1, n)
        data = np.interp(indices, np.arange(len(data)), data)
    return data


def extract_video_audio(video_path: Path, tmp_dir: Path) -> Path:
    """Extract the audio track from a video as WAV"""
    out_path = tmp_dir / f"{video_path.stem}.wav"
    if out_path.exists():
        return out_path
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
        str(out_path)
    ], capture_output=True, timeout=600)
    return out_path


def find_offset_xcorr(trial_audio: np.ndarray, video_audio: np.ndarray) -> tuple[float, float]:
    """
    Use cross-correlation to find where the trial audio sits inside the video audio.
    Returns (offset_sec, confidence)
    """
    # Downsample for speed
    t = trial_audio[::DOWNSAMPLE].astype(np.float64)
    v = video_audio[::DOWNSAMPLE].astype(np.float64)

    if len(t) < 100 or len(v) < 100:
        return 0.0, 0.0

    # Normalize
    t = (t - t.mean()) / max(t.std(), 1e-9)
    v = (v - v.mean()) / max(v.std(), 1e-9)

    # Cross-correlation via FFT
    corr = fftconvolve(v, t[::-1], mode='full')
    # Normalize by overlap length
    n_t, n_v = len(t), len(v)
    offsets = np.arange(len(corr)) - (n_t - 1)
    overlap = np.minimum(
        np.minimum(np.arange(len(corr)) + 1, n_t),
        np.minimum(len(corr) - np.arange(len(corr)), n_v)
    ).astype(np.float64)
    overlap = np.maximum(overlap, 1)
    corr = corr / overlap

    # Only consider offsets where trial fits within video
    valid_mask = (offsets >= 0) & (offsets + n_t <= n_v)
    if not valid_mask.any():
        # Allow partial overlap (at least 50%)
        valid_mask = (offsets >= -n_t // 2) & (offsets <= n_v)

    corr_valid = corr.copy()
    corr_valid[~valid_mask] = -np.inf

    best_idx = np.argmax(corr_valid)
    best_offset_samples = offsets[best_idx] * DOWNSAMPLE
    best_offset_sec = best_offset_samples / SAMPLE_RATE

    # Confidence: peak prominence
    peak_val = corr_valid[best_idx]
    valid_vals = corr_valid[valid_mask]
    median_val = np.median(valid_vals)
    std_val = np.std(valid_vals)
    confidence = min(max((peak_val - median_val) / max(std_val, 1e-9) / 10, 0), 1)

    return best_offset_sec, confidence


def get_video_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, timeout=30
    )
    try:
        return float(r.stdout.strip())
    except:
        return 0.0


def run():
    print("Stage 03 midi_motion s00: Audio-Video Synchronization (220819)")
    print(f"Output: {OUT_ROOT}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Collect trial audio files
    trial_wavs = sorted(AUDIO_DIR.glob("*.wav"))
    print(f"\nTrials: {len(trial_wavs)}")

    # Collect video files by directory
    video_dirs = {}
    for vdir in sorted(VIDEO_DIR.iterdir()):
        if not vdir.is_dir():
            continue
        valid_videos = []
        for vf in sorted(vdir.glob("*.mp4")):
            dur = get_video_duration(vf)
            if dur > 5:  # Skip corrupted/tiny files
                valid_videos.append({"path": vf, "duration": dur})
        if valid_videos:
            video_dirs[vdir.name] = valid_videos
            total_dur = sum(v["duration"] for v in valid_videos)
            print(f"  {vdir.name}: {len(valid_videos)} videos, {total_dur:.0f}s total")

    # Create temp dir for extracted audio
    tmp_dir = Path("Data/Pipeline/03_sync/midi_motion/_tmp_audio")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Process each video directory
    for vdir_name, videos in video_dirs.items():
        print(f"\n{'='*60}")
        print(f"Processing: {vdir_name} ({len(videos)} videos)")
        print(f"{'='*60}")

        is_continuous = len(videos) == 1 and videos[0]["duration"] > 3600

        if is_continuous:
            # Single continuous video — extract audio once, match all trials
            video = videos[0]
            print(f"  Extracting audio from {video['path'].name} ({video['duration']:.0f}s)...")
            audio_path = extract_video_audio(video["path"], tmp_dir)
            video_audio = load_wav_mono(audio_path)
            print(f"  Audio loaded: {len(video_audio)/SAMPLE_RATE:.0f}s")

            for wav_path in trial_wavs:
                stem = wav_path.stem
                out_json = OUT_ROOT / f"{stem}.json"

                # Load existing result if present, check if this vdir already done
                existing = {}
                if out_json.exists():
                    with open(out_json) as f:
                        existing = json.load(f)
                    if vdir_name in existing.get("video_sync", {}):
                        print(f"  {stem}/{vdir_name} - skip")
                        continue

                trial_audio = load_wav_mono(wav_path)
                offset_sec, confidence = find_offset_xcorr(trial_audio, video_audio)
                trial_dur = len(trial_audio) / SAMPLE_RATE

                result_entry = {
                    "video_file": video["path"].name,
                    "video_dir": vdir_name,
                    "offset_sec": round(offset_sec, 4),
                    "trial_duration_sec": round(trial_dur, 2),
                    "confidence": round(confidence, 3),
                    "video_duration_sec": round(video["duration"], 2),
                }

                # Merge into existing
                if "video_sync" not in existing:
                    existing = {"stem": stem, "dataset": "220819", "video_sync": {}}
                existing["video_sync"][vdir_name] = result_entry
                with open(out_json, "w") as f:
                    json.dump(existing, f, indent=2)

                quality = "good" if confidence >= 0.5 else ("fair" if confidence >= 0.2 else "poor")
                print(f"  {stem}/{vdir_name}: offset={offset_sec:.2f}s conf={confidence:.3f} ({quality})")

        else:
            # Per-trial videos — match each video to its trial
            print(f"  Extracting audio from {len(videos)} videos...")
            video_audios = []
            for v in videos:
                audio_path = extract_video_audio(v["path"], tmp_dir)
                try:
                    va = load_wav_mono(audio_path)
                    video_audios.append({"info": v, "audio": va})
                except:
                    print(f"    SKIP {v['path'].name} (audio extraction failed)")

            for wav_path in trial_wavs:
                stem = wav_path.stem
                out_json = OUT_ROOT / f"{stem}.json"

                existing = {}
                if out_json.exists():
                    with open(out_json) as f:
                        existing = json.load(f)
                    if vdir_name in existing.get("video_sync", {}):
                        print(f"  {stem}/{vdir_name} - skip")
                        continue

                trial_audio = load_wav_mono(wav_path)
                trial_dur = len(trial_audio) / SAMPLE_RATE

                # Find best matching video (require >= 80% overlap)
                best_offset, best_conf, best_video = 0.0, 0.0, None
                for va in video_audios:
                    offset, conf = find_offset_xcorr(trial_audio, va["audio"])
                    vdur = va["info"]["duration"]
                    overlap = max(0, min(offset + trial_dur, vdur) - max(offset, 0))
                    overlap_ratio = overlap / max(trial_dur, 1e-9)
                    # Penalize low overlap: effective confidence = conf * overlap_ratio
                    eff_conf = conf * min(overlap_ratio / 0.8, 1.0) if overlap_ratio >= 0.5 else 0.0
                    if eff_conf > best_conf:
                        best_offset, best_conf, best_video = offset, eff_conf, va["info"]

                if best_video is None:
                    print(f"  {stem}/{vdir_name}: NO MATCH")
                    continue

                result_entry = {
                    "video_file": best_video["path"].name,
                    "video_dir": vdir_name,
                    "offset_sec": round(best_offset, 4),
                    "trial_duration_sec": round(trial_dur, 2),
                    "confidence": round(best_conf, 3),
                    "video_duration_sec": round(best_video["duration"], 2),
                }

                if "video_sync" not in existing:
                    existing = {"stem": stem, "dataset": "220819", "video_sync": {}}
                existing["video_sync"][vdir_name] = result_entry
                with open(out_json, "w") as f:
                    json.dump(existing, f, indent=2)

                quality = "good" if best_conf >= 0.5 else ("fair" if best_conf >= 0.2 else "poor")
                print(f"  {stem}/{vdir_name}: {best_video['path'].name} offset={best_offset:.2f}s conf={best_conf:.3f} ({quality})")

    # Cleanup
    print(f"\n{'='*60}")
    print("Summary:")
    results = list(OUT_ROOT.glob("*.json"))
    print(f"  {len(results)} trial sync results")
    for jf in sorted(results)[:5]:
        with open(jf) as f:
            r = json.load(f)
        syncs = r.get("video_sync", {})
        dirs = ", ".join(f"{k}={v['confidence']:.2f}" for k, v in syncs.items())
        print(f"  {r['stem']}: {dirs}")
    if len(results) > 5:
        print(f"  ... ({len(results) - 5} more)")


if __name__ == "__main__":
    run()
