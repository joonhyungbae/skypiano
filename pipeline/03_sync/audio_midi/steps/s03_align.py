# Standalone diagnostic; not part of the run_pipeline.py loop. Invoked manually.
"""
Stage 03 audio_midi Step 03: Full Audio-MIDI alignment (Maestro-style DTW)

Alternative DTW-based alignment pipeline. The production runner uses
`s01_align_midi_motion` + `s03_timecode_align` instead; this script is kept
as a standalone reference / fallback path.
"""
import functools
import sys
from pathlib import Path

import numpy as np
import librosa
import pretty_midi
import soundfile as sf_io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit
from scipy.spatial.distance import cdist

from s01_check_pairs import DATASET_DIRS, check_pairs_for_node

print = functools.partial(print, flush=True)

SAMPLE_RATE = 22050
HOP_LENGTH = 512
CQT_BINS_PER_OCTAVE = 12
CQT_N_BINS = 48
CQT_FMIN = librosa.midi_to_hz(36)
MIN_DB = -80.0
EPSILON = 1e-10
BAND_RADIUS_SEC = 5.0

SF2_PATH = (
    Path(__file__).resolve().parent.parent
    / "soundfonts"
    / "SalamanderGrandPiano-SF2-V3+20200602"
    / "SalamanderGrandPiano-V3+20200602.sf2"
)
OUT_ROOT = Path("Data/Pipeline/03_sync/audio_midi/Aligned")
IMG_ROOT = Path("Data/Pipeline/03_sync/audio_midi/img")


def compute_cqt(samples, sr, hop_length):
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


@njit(cache=True)
def _dtw_numba(dist, band_frames, penalty):
    n, m = dist.shape
    INF = np.float32(1e30)
    D = np.full((n + 1, m + 1), INF, dtype=np.float32)
    D[0, 0] = np.float32(0.0)
    ptr = np.zeros((n + 1, m + 1), dtype=np.int8)
    for i in range(1, n + 1):
        j_lo = max(1, i - band_frames)
        j_hi = min(m, i + band_frames)
        for j in range(j_lo, j_hi + 1):
            cost = dist[i - 1, j - 1]
            d_diag = D[i - 1, j - 1] + cost
            d_up = D[i - 1, j] + penalty
            d_left = D[i, j - 1] + penalty
            if d_diag <= d_up and d_diag <= d_left:
                D[i, j] = d_diag; ptr[i, j] = 0
            elif d_up <= d_left:
                D[i, j] = d_up; ptr[i, j] = 1
            else:
                D[i, j] = d_left; ptr[i, j] = 2
    path_i = np.empty(n + m, dtype=np.int32)
    path_j = np.empty(n + m, dtype=np.int32)
    k = 0; i, j = n, m
    while i > 0 and j > 0:
        path_i[k] = i - 1; path_j[k] = j - 1; k += 1
        p = ptr[i, j]
        if p == 0: i -= 1; j -= 1
        elif p == 1: i -= 1
        else: j -= 1
    pi = path_i[:k][::-1].copy()
    pj = path_j[:k][::-1].copy()
    return D[n, m], pi, pj


def sakoe_chiba_dtw(dist_matrix, band_radius_sec, sr, hop_length, penalty):
    sec_per_frame = hop_length / sr
    band_frames = int(round(band_radius_sec / sec_per_frame))
    total_cost, pi, pj = _dtw_numba(dist_matrix, band_frames, penalty)
    return pi, pj, total_cost


def estimate_penalty(dist_matrix, sample_size=100000):
    n, m = dist_matrix.shape
    if n * m <= sample_size:
        return float(dist_matrix.mean())
    idx_i = np.random.randint(0, n, size=sample_size)
    idx_j = np.random.randint(0, m, size=sample_size)
    return float(dist_matrix[idx_i, idx_j].mean())


def apply_warping(pm, warp_from_sec, warp_to_sec):
    sec_per_frame = HOP_LENGTH / SAMPLE_RATE
    skipped = 0
    for inst in pm.instruments:
        for note in inst.notes:
            s_new = float(np.interp(note.start, warp_from_sec, warp_to_sec))
            e_new = float(np.interp(note.end, warp_from_sec, warp_to_sec))
            if e_new < s_new: s_new, e_new = e_new, s_new
            if (e_new - s_new) < sec_per_frame: skipped += 1
            note.start = s_new; note.end = e_new
    return skipped


def save_diagnostic(audio, midi_audio, pi, pj, pm_aligned,
                    sr, hop_length, stem, dataset, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    sec_pf = hop_length / sr
    warp_from = pi.astype(np.float64) * sec_pf
    warp_to = pj.astype(np.float64) * sec_pf
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    deviation = warp_to - warp_from
    axes[0].plot(warp_from, deviation, linewidth=0.5)
    axes[0].set_ylabel("Warp deviation (s)")
    axes[0].set_title(f"{dataset}/{stem}")
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.5)
    all_starts = [n.start for inst in pm_aligned.instruments for n in inst.notes]
    if all_starts:
        onset = min(all_starts)
        view = 0.05
        s_start = max(0, int((onset - view) * sr))
        s_end = min(len(audio), int((onset + view) * sr))
        if s_end > s_start:
            t = np.linspace(onset - view, onset + view, s_end - s_start)
            axes[1].plot(t[:len(audio[s_start:s_end])], audio[s_start:s_end], color='blue', alpha=0.7, label='Audio')
            if s_end <= len(midi_audio):
                axes[1].plot(t[:len(midi_audio[s_start:s_end])], midi_audio[s_start:s_end], color='red', alpha=0.7, label='Aligned MIDI')
            axes[1].legend(fontsize=8)
            axes[1].set_title("First note onset")
    hop_env = sr // 10
    env_a = np.array([np.abs(audio[i:i+hop_env]).max() for i in range(0, len(audio)-hop_env, hop_env)])
    env_m = np.array([np.abs(midi_audio[i:i+hop_env]).max() for i in range(0, min(len(midi_audio), len(audio))-hop_env, hop_env)])
    t_env = np.arange(len(env_a)) * (hop_env / sr)
    axes[2].plot(t_env, env_a, color='blue', alpha=0.7, label='Audio')
    axes[2].plot(t_env[:len(env_m)], env_m, color='red', alpha=0.7, label='MIDI')
    axes[2].legend(fontsize=8)
    axes[2].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}.png", dpi=100)
    plt.close(fig)


MAX_FRAMES = 50000
CHUNK_SEC = 300.0
OVERLAP_SEC = 30.0


def _align_chunked(cqt_mid, cqt_aud, sr, hop_length):
    sec_pf = hop_length / sr
    chunk_frames = int(CHUNK_SEC / sec_pf)
    overlap_frames = int(OVERLAP_SEC / sec_pf)
    n_frames_mid = cqt_mid.shape[1]
    n_frames_aud = cqt_aud.shape[1]
    all_pi, all_pj = [], []
    total_cost = 0.0
    mid_offset, aud_offset = 0, 0
    chunk_idx = 0
    while mid_offset < n_frames_mid and aud_offset < n_frames_aud:
        mid_end = min(mid_offset + chunk_frames, n_frames_mid)
        aud_end = min(aud_offset + chunk_frames, n_frames_aud)
        cqt_m_chunk = cqt_mid[:, mid_offset:mid_end]
        cqt_a_chunk = cqt_aud[:, aud_offset:aud_end]
        dist = cdist(cqt_m_chunk.T, cqt_a_chunk.T, metric='cosine').astype(np.float32)
        penalty = estimate_penalty(dist)
        pi, pj, cost = sakoe_chiba_dtw(dist, band_radius_sec=BAND_RADIUS_SEC, sr=sr, hop_length=hop_length, penalty=penalty)
        del dist
        pi_global = pi + mid_offset
        pj_global = pj + aud_offset
        if chunk_idx == 0:
            all_pi.append(pi_global); all_pj.append(pj_global)
        else:
            mid_overlap_boundary = mid_offset + overlap_frames // 2
            start_idx = 0
            for k in range(len(pi_global)):
                if pi_global[k] >= mid_overlap_boundary:
                    start_idx = k; break
            all_pi.append(pi_global[start_idx:]); all_pj.append(pj_global[start_idx:])
        total_cost += cost
        if mid_end >= n_frames_mid and aud_end >= n_frames_aud:
            break
        mid_offset = int(pi_global[-1]) - overlap_frames // 2
        aud_offset = int(pj_global[-1]) - overlap_frames // 2
        if mid_offset <= (int(pi_global[0]) if len(pi_global) > 0 else 0):
            break
        chunk_idx += 1
        print(f"    chunk {chunk_idx}: mid_offset={mid_offset}, aud_offset={aud_offset}")
    pi_all = np.concatenate(all_pi)
    pj_all = np.concatenate(all_pj)
    mask = np.ones(len(pi_all), dtype=bool)
    for k in range(1, len(pi_all)):
        if pi_all[k] < pi_all[k-1] or pj_all[k] < pj_all[k-1]:
            mask[k] = False
    return pi_all[mask], pj_all[mask], total_cost


def align_pair(audio_path, midi_path, out_midi_path, img_dir, stem, dataset, sf2_path):
    audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE)
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    midi_audio = pm.fluidsynth(fs=SAMPLE_RATE, sf2_path=str(sf2_path)).astype(np.float32)
    max_len = max(len(audio), len(midi_audio))
    audio = np.pad(audio, (0, max_len - len(audio)))
    midi_audio = np.pad(midi_audio, (0, max_len - len(midi_audio)))
    cqt_aud = compute_cqt(audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    cqt_mid = compute_cqt(midi_audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    use_chunked = max(cqt_mid.shape[1], cqt_aud.shape[1]) > MAX_FRAMES
    if use_chunked:
        print(f"  chunked mode ({cqt_mid.shape[1]}x{cqt_aud.shape[1]} frames)")
        pi, pj, cost = _align_chunked(cqt_mid, cqt_aud, SAMPLE_RATE, HOP_LENGTH)
    else:
        dist = cdist(cqt_mid.T, cqt_aud.T, metric='cosine').astype(np.float32)
        penalty = estimate_penalty(dist)
        pi, pj, cost = sakoe_chiba_dtw(dist, band_radius_sec=BAND_RADIUS_SEC, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, penalty=penalty)
        del dist
    del cqt_aud, cqt_mid
    sec_pf = HOP_LENGTH / SAMPLE_RATE
    warp_from = pi.astype(np.float64) * sec_pf
    warp_to = pj.astype(np.float64) * sec_pf
    skipped = apply_warping(pm, warp_from, warp_to)
    out_midi_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_midi_path))
    try:
        midi_aligned_audio = pm.fluidsynth(fs=SAMPLE_RATE, sf2_path=str(sf2_path))
        midi_aligned_audio = np.pad(midi_aligned_audio, (0, max(0, len(audio) - len(midi_aligned_audio))))
        save_diagnostic(audio, midi_aligned_audio, pi, pj, pm, SAMPLE_RATE, HOP_LENGTH, stem, dataset, img_dir)
    except Exception as e:
        print(f"  img error: {e}")
    deviation = warp_to - warp_from
    max_dev = np.abs(deviation).max()
    mean_dev = np.abs(deviation).mean()
    print(f"  cost={cost:.1f}  max_dev={max_dev:.3f}s  mean_dev={mean_dev:.3f}s  skipped={skipped}")
    return cost, max_dev, mean_dev


def run():
    if not SF2_PATH.exists():
        print(f"SoundFont not found: {SF2_PATH}")
        sys.exit(1)
    print(f"SoundFont: {SF2_PATH.name}")
    print(f"Output: {OUT_ROOT}")
    pairs = []
    for dataset_id, config in DATASET_DIRS.items():
        for audio_dir, midi_dir in config["nodes"]:
            results = check_pairs_for_node(audio_dir, midi_dir)
            for r in results:
                if r["status"] == "both":
                    pairs.append({
                        "dataset": dataset_id, "stem": r["stem"],
                        "audio_path": Path(r["audio_path"]),
                        "midi_path": Path(r["midi_paths"][0]),
                    })
    print(f"\n{len(pairs)} pairs")
    print("=" * 70)
    success, fail = 0, 0
    for i, p in enumerate(pairs, 1):
        out_path = OUT_ROOT / p["dataset"] / f"{p['stem']}.mid"
        img_dir = IMG_ROOT / p["dataset"]
        if out_path.exists():
            print(f"[{i}/{len(pairs)}] {p['dataset']}/{p['stem']} - skip")
            success += 1; continue
        print(f"[{i}/{len(pairs)}] {p['dataset']}/{p['stem']}")
        try:
            align_pair(p["audio_path"], p["midi_path"], out_path, img_dir,
                       p["stem"], p["dataset"], SF2_PATH)
            success += 1
        except Exception as e:
            print(f"  FAIL: {e}"); fail += 1
    print(f"\nDone: success={success} fail={fail}")


if __name__ == "__main__":
    run()
