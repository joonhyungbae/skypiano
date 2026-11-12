# Standalone diagnostic; not part of the run_pipeline.py loop. Invoked manually.
"""
Stage 03 audio_midi: 211215-specific Audio-MIDI re-alignment (hybrid strategy)

Strategy:
  1. Sync-note detection: find the first onset (sync stroke) in both audio and MIDI to get a coarse offset
  2. Trim both sides at the sync onset -> aligns the DTW starting point
  3. Peak-normalize + tight DTW for fine alignment
  4. Apply the trim offset to the warping result to recover the original time axis

Usage:
  cd pipeline/03_sync/audio_midi/steps
  python s03_realign_211215.py [--force]
"""
import argparse
import functools
import sys
from pathlib import Path

import numpy as np
import librosa
import pretty_midi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from s01_check_pairs import DATASET_DIRS, check_pairs_for_node
from s03_align import (
    SAMPLE_RATE, HOP_LENGTH, CQT_FMIN, CQT_N_BINS, CQT_BINS_PER_OCTAVE,
    EPSILON, MIN_DB, SF2_PATH, OUT_ROOT, IMG_ROOT,
    sakoe_chiba_dtw, estimate_penalty, apply_warping, save_diagnostic,
    _align_chunked, MAX_FRAMES,
)

print = functools.partial(print, flush=True)

# -- 211215-specific settings --
DTW_BAND_SEC = 5.0            # after sync-note alignment a tight band is sufficient
SILENCE_THRESHOLD_DB = -40.0  # onset detection threshold
ONSET_MARGIN_SEC = 0.3        # leading margin before the sync note


def compute_cqt_normalized(samples, sr, hop_length):
    """Peak-normalize then CQT -- features come through even on quiet audio"""
    peak = np.abs(samples).max()
    if peak > 0:
        samples = samples / peak

    cqt_mag = np.abs(
        librosa.cqt(samples, sr=sr, hop_length=hop_length,
                     fmin=CQT_FMIN, n_bins=CQT_N_BINS,
                     bins_per_octave=CQT_BINS_PER_OCTAVE)
    ).astype(np.float32)
    max_val = max(cqt_mag.max(), EPSILON)
    cqt_db = 20.0 * np.log10(cqt_mag / max_val + EPSILON)
    cqt_db = np.clip(cqt_db, a_min=MIN_DB, a_max=None)
    cqt_amp = 10.0 ** (cqt_db / 20.0)
    normed = cqt_amp.copy()
    col_mins = normed.min(axis=0)
    kernel = np.ones(5) / 5.0
    avg_mins = np.convolve(col_mins, kernel, mode='same').astype(np.float32)
    avg_mins = np.maximum(avg_mins, EPSILON)
    normed /= avg_mins[np.newaxis, :]
    return normed.astype(np.float32)


def detect_audio_onset(samples, sr):
    """Detect the first-note (sync-note) onset in audio, in seconds"""
    rms = librosa.feature.rms(y=samples, frame_length=2048, hop_length=512)[0]
    rms_db = 20 * np.log10(rms + EPSILON)
    above = np.where(rms_db > SILENCE_THRESHOLD_DB)[0]
    if len(above) == 0:
        return 0.0
    onset_frame = above[0]
    return max(0.0, onset_frame * 512 / sr)


def detect_midi_onset(pm):
    """Return the first note onset (in seconds) from MIDI"""
    first = None
    for inst in pm.instruments:
        for note in inst.notes:
            if first is None or note.start < first:
                first = note.start
    return first if first is not None else 0.0


def align_pair_211215(audio_path, midi_path, out_midi_path, img_dir, stem, sf2_path):
    """
    Hybrid alignment:
      Phase 1 -- coarse alignment via sync-note onset (trim both sides)
      Phase 2 -- fine alignment via peak-normalized DTW
    """
    audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE)
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    midi_audio = pm.fluidsynth(fs=SAMPLE_RATE, sf2_path=str(sf2_path)).astype(np.float32)

    # -- Phase 1: Sync-note onset detection --
    audio_onset = detect_audio_onset(audio, SAMPLE_RATE)
    midi_onset = detect_midi_onset(pm)

    # Trim both sides relative to the sync note (start slightly before the onset)
    audio_trim = max(0.0, audio_onset - ONSET_MARGIN_SEC)
    midi_trim = max(0.0, midi_onset - ONSET_MARGIN_SEC)

    audio_trim_samples = int(audio_trim * SAMPLE_RATE)
    midi_trim_samples = int(midi_trim * SAMPLE_RATE)

    audio_trimmed = audio[audio_trim_samples:]
    midi_trimmed = midi_audio[midi_trim_samples:]

    print(f"  sync: audio_onset={audio_onset:.2f}s  midi_onset={midi_onset:.2f}s  "
          f"offset={audio_onset - midi_onset:.2f}s")

    # -- Phase 2: DTW fine alignment --
    max_len = max(len(audio_trimmed), len(midi_trimmed))
    audio_trimmed = np.pad(audio_trimmed, (0, max(0, max_len - len(audio_trimmed))))
    midi_trimmed = np.pad(midi_trimmed, (0, max(0, max_len - len(midi_trimmed))))

    cqt_aud = compute_cqt_normalized(audio_trimmed, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    cqt_mid = compute_cqt_normalized(midi_trimmed, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)

    use_chunked = max(cqt_mid.shape[1], cqt_aud.shape[1]) > MAX_FRAMES
    if use_chunked:
        print(f"  chunked mode ({cqt_mid.shape[1]}x{cqt_aud.shape[1]} frames)")
        pi, pj, cost = _align_chunked(cqt_mid, cqt_aud, SAMPLE_RATE, HOP_LENGTH)
    else:
        dist = cdist(cqt_mid.T, cqt_aud.T, metric='cosine').astype(np.float32)
        penalty = estimate_penalty(dist)
        pi, pj, cost = sakoe_chiba_dtw(
            dist, band_radius_sec=DTW_BAND_SEC,
            sr=SAMPLE_RATE, hop_length=HOP_LENGTH, penalty=penalty,
        )
        del dist
    del cqt_aud, cqt_mid

    # -- Warping: map trimmed coordinates back to the original time axis --
    sec_pf = HOP_LENGTH / SAMPLE_RATE
    # pi = MIDI trimmed frame index -> original MIDI time
    warp_from = pi.astype(np.float64) * sec_pf + midi_trim
    # pj = audio trimmed frame index -> original audio time
    warp_to = pj.astype(np.float64) * sec_pf + audio_trim

    # Reload MIDI and apply warping (applied to the original pm)
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    skipped = apply_warping(pm, warp_from, warp_to)
    out_midi_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_midi_path))

    # -- Diagnostic --
    try:
        midi_aligned_audio = pm.fluidsynth(fs=SAMPLE_RATE, sf2_path=str(sf2_path))
        midi_aligned_audio = np.pad(midi_aligned_audio, (0, max(0, len(audio) - len(midi_aligned_audio))))
        save_diagnostic(audio, midi_aligned_audio, pi, pj, pm,
                        SAMPLE_RATE, HOP_LENGTH, stem, "211215", img_dir)
    except Exception as e:
        print(f"  img error: {e}")

    # -- Stats (deviation measured on trimmed coordinates) --
    deviation = (pj.astype(np.float64) - pi.astype(np.float64)) * sec_pf
    max_dev = np.abs(deviation).max()
    mean_dev = np.abs(deviation).mean()
    print(f"  cost={cost:.1f}  max_dev={max_dev:.3f}s  mean_dev={mean_dev:.3f}s  skipped={skipped}")
    return cost, max_dev, mean_dev


def run():
    parser = argparse.ArgumentParser(description="211215 Audio-MIDI re-alignment (hybrid)")
    parser.add_argument("--force", action="store_true", help="Re-align even if output exists")
    args = parser.parse_args()

    if not SF2_PATH.exists():
        print(f"SoundFont not found: {SF2_PATH}")
        sys.exit(1)

    dataset_id = "211215"
    config = DATASET_DIRS[dataset_id]

    pairs = []
    for audio_dir, midi_dir in config["nodes"]:
        results = check_pairs_for_node(audio_dir, midi_dir)
        for r in results:
            if r["status"] == "both":
                pairs.append({
                    "stem": r["stem"],
                    "audio_path": Path(r["audio_path"]),
                    "midi_path": Path(r["midi_paths"][0]),
                })

    print(f"211215 hybrid re-alignment (sync note + DTW)")
    print(f"SoundFont: {SF2_PATH.name}")
    print(f"Output: {OUT_ROOT / dataset_id}")
    print(f"Pairs: {len(pairs)}")
    print(f"DTW band: {DTW_BAND_SEC}s (after sync alignment)")
    print("=" * 70)

    success, fail = 0, 0
    for i, p in enumerate(pairs, 1):
        out_path = OUT_ROOT / dataset_id / f"{p['stem']}.mid"
        img_dir = IMG_ROOT / dataset_id

        if out_path.exists() and not args.force:
            print(f"[{i}/{len(pairs)}] {p['stem']} - skip (use --force)")
            success += 1
            continue

        print(f"[{i}/{len(pairs)}] {p['stem']}")
        try:
            align_pair_211215(
                p["audio_path"], p["midi_path"], out_path, img_dir,
                p["stem"], SF2_PATH,
            )
            success += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            fail += 1

    print(f"\nDone: success={success} fail={fail}")


if __name__ == "__main__":
    run()
