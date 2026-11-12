"""
Stage 03 midi_motion Step 02: Refine per-trial video synchronization (Left, RightFull)

Using the Right (session-continuous) audio as the reference axis:
1. Determine which session segment each per-trial video corresponds to
2. Match against each trial's already-known session position (Right offset)
3. Compute the refined offset
"""
import functools
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

print = functools.partial(print, flush=True)

RAW_ROOT = Path("Data/Raw/220819/C0/N0")
VIDEO_DIR = RAW_ROOT / "Video"
OUT_ROOT = Path("Data/Pipeline/03_sync/midi_motion/Aligned/220819")
TMP_DIR = Path("Data/Pipeline/03_sync/midi_motion/_tmp_audio")

SAMPLE_RATE = 48000
DOWNSAMPLE = 8
EFFECTIVE_SR = SAMPLE_RATE // DOWNSAMPLE


def load_wav_mono(path: Path) -> np.ndarray:
    rate, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32)


def extract_video_audio(video_path: Path) -> Path:
    out_path = TMP_DIR / f"{video_path.stem}.wav"
    if out_path.exists():
        return out_path
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
        str(out_path)
    ], capture_output=True, timeout=600)
    return out_path


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


def find_offset(short_audio: np.ndarray, long_audio: np.ndarray) -> tuple[float, float]:
    """Find where short sits inside long. Returns (offset_sec, confidence)"""
    s = short_audio[::DOWNSAMPLE].astype(np.float64)
    l = long_audio[::DOWNSAMPLE].astype(np.float64)

    if len(s) < 100 or len(l) < 100:
        return 0.0, 0.0

    s = (s - s.mean()) / max(s.std(), 1e-9)
    l = (l - l.mean()) / max(l.std(), 1e-9)

    corr = fftconvolve(l, s[::-1], mode='full')
    n_s, n_l = len(s), len(l)
    offsets = np.arange(len(corr)) - (n_s - 1)

    overlap = np.minimum(
        np.minimum(np.arange(len(corr)) + 1, n_s),
        np.minimum(len(corr) - np.arange(len(corr)), n_l)
    ).astype(np.float64)
    overlap = np.maximum(overlap, 1)
    corr = corr / overlap

    # Only valid where short fits fully inside long
    valid = (offsets >= 0) & (offsets + n_s <= n_l)
    if not valid.any():
        valid = (offsets >= -n_s // 4) & (offsets <= n_l)

    corr_v = corr.copy()
    corr_v[~valid] = -np.inf

    best_idx = np.argmax(corr_v)
    best_offset_sec = offsets[best_idx] * DOWNSAMPLE / SAMPLE_RATE

    peak = corr_v[best_idx]
    vals = corr_v[valid]
    conf = min(max((peak - np.median(vals)) / max(np.std(vals), 1e-9) / 10, 0), 1)

    return best_offset_sec, conf


def run():
    print("Stage 03 midi_motion s02: Refine per-trial video sync using Right as reference")
    print(f"Output: {OUT_ROOT}")
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Right session audio (reference)
    right_videos = sorted(VIDEO_DIR.glob("Right/*.mp4"))
    right_video = None
    for v in right_videos:
        dur = get_video_duration(v)
        if dur > 3600:
            right_video = v
            break

    if not right_video:
        print("ERROR: No Right session video found")
        return

    print(f"Reference: {right_video.name} ({get_video_duration(right_video):.0f}s)")
    right_audio_path = extract_video_audio(right_video)
    right_audio = load_wav_mono(right_audio_path)
    right_dur = len(right_audio) / SAMPLE_RATE
    print(f"Reference audio: {right_dur:.0f}s")

    # 2. Load trial positions from existing Right sync
    trial_positions = {}  # stem -> session_offset_sec
    for jf in sorted(OUT_ROOT.glob("*.json")):
        with open(jf) as f:
            r = json.load(f)
        right_sync = r.get("video_sync", {}).get("Right", {})
        if right_sync and right_sync["confidence"] >= 0.9:
            trial_positions[r["stem"]] = {
                "session_offset": right_sync["offset_sec"],
                "duration": right_sync["trial_duration_sec"],
            }
    print(f"Trial positions from Right: {len(trial_positions)}")

    # 3. Process each per-trial video directory
    for vdir_name in ["Left", "RightFull"]:
        vdir = VIDEO_DIR / vdir_name
        if not vdir.exists():
            continue

        videos = []
        for vf in sorted(vdir.glob("*.mp4")):
            dur = get_video_duration(vf)
            if dur > 5:
                videos.append({"path": vf, "duration": dur})

        print(f"\n{'='*60}")
        print(f"{vdir_name}: {len(videos)} videos")
        print(f"{'='*60}")

        # 3a. Find where each video sits in the session (cross-correlate with Right)
        print("  Locating videos in session timeline...")
        video_positions = []
        for v in videos:
            audio_path = extract_video_audio(v["path"])
            try:
                va = load_wav_mono(audio_path)
            except:
                print(f"    SKIP {v['path'].name}")
                continue

            offset, conf = find_offset(va, right_audio)
            video_positions.append({
                **v,
                "session_offset": offset,
                "session_end": offset + v["duration"],
                "locate_conf": conf,
            })
            print(f"    {v['path'].name}: session=[{offset:.1f}, {offset + v['duration']:.1f}]s  conf={conf:.3f}")

        # 3b. Match trials to videos
        print(f"\n  Matching {len(trial_positions)} trials to {len(video_positions)} videos...")
        for stem, tpos in sorted(trial_positions.items()):
            t_start = tpos["session_offset"]
            t_end = t_start + tpos["duration"]

            # Find video with best overlap
            best_video = None
            best_overlap = 0
            for vp in video_positions:
                overlap_start = max(t_start, vp["session_offset"])
                overlap_end = min(t_end, vp["session_end"])
                overlap = max(0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_video = vp

            if best_video is None or best_overlap < tpos["duration"] * 0.3:
                print(f"    {stem}: NO MATCH (best overlap={best_overlap:.1f}s / {tpos['duration']:.1f}s)")
                continue

            # Offset within video = trial_session_start - video_session_start
            offset_in_video = t_start - best_video["session_offset"]
            overlap_ratio = best_overlap / tpos["duration"]

            # Combined confidence
            confidence = best_video["locate_conf"] * min(overlap_ratio / 0.8, 1.0)

            result_entry = {
                "video_file": best_video["path"].name,
                "video_dir": vdir_name,
                "offset_sec": round(offset_in_video, 4),
                "trial_duration_sec": round(tpos["duration"], 2),
                "confidence": round(confidence, 3),
                "video_duration_sec": round(best_video["duration"], 2),
                "session_offset_sec": round(t_start, 2),
                "overlap_ratio": round(overlap_ratio, 3),
                "method": "reference_based",
            }

            # Update JSON
            out_json = OUT_ROOT / f"{stem}.json"
            existing = {}
            if out_json.exists():
                with open(out_json) as f:
                    existing = json.load(f)
            if "video_sync" not in existing:
                existing = {"stem": stem, "dataset": "220819", "video_sync": {}}
            existing["video_sync"][vdir_name] = result_entry
            with open(out_json, "w") as f:
                json.dump(existing, f, indent=2)

            quality = "good" if confidence >= 0.5 else ("fair" if confidence >= 0.2 else "poor")
            print(f"    {stem}: {best_video['path'].name}  offset={offset_in_video:.2f}s  overlap={overlap_ratio:.0%}  conf={confidence:.3f} ({quality})")

    # Summary
    print(f"\n{'='*60}")
    print("Final summary:")
    for vdir in ['Right', 'Top', 'Left', 'RightFull']:
        confs = []
        for jf in sorted(OUT_ROOT.glob("*.json")):
            with open(jf) as f:
                r = json.load(f)
            sync = r.get("video_sync", {}).get(vdir, {})
            if sync:
                confs.append(sync["confidence"])
        if confs:
            good = sum(1 for c in confs if c >= 0.5)
            print(f"  {vdir:12s}: {len(confs)} synced, {good} good, avg={sum(confs)/len(confs):.3f}, min={min(confs):.3f}")


if __name__ == "__main__":
    run()
