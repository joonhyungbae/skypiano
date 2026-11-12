"""
Shared MIDI <-> motion temporal alignment (coarse-to-fine xcorr).

Factored out of the two near-identical pass scripts:
  pipeline/03_sync/audio_midi/steps/s01_align_midi_motion.py  (Stage 03 audio_midi pass)
  pipeline/03_sync/midi_motion/steps/s01_align_midi_motion.py (Stage 03 midi_motion pass)

The only differences between the two passes are their input/output roots -
the audio_midi pass reads pre-body-alignment motion, the midi_motion pass
reads post-body-alignment motion and the audio_midi MIDI output. The algorithm is identical.

Usage from a wrapper:

    from _common.midi_xcorr import run_pass
    run_pass(hand_root=..., midi_root=..., body_root=...,
             out_root=..., img_root=..., label="pass 1", datasets=[...])
"""
import functools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi

print = functools.partial(print, flush=True)

MOTION_FPS = 120.0
MAX_SHIFT_SEC = 25.0

FINGERTIPS = [
    f"{side}{finger}F"
    for side in ["L", "R"]
    for finger in ["I", "M", "R", "L", "T"]
]
WRISTS = ["LWrist", "RWrist"]


def smooth(arr, w=2):
    arr = np.asarray(arr, dtype=np.float64)
    if w < 1 or len(arr) < 2 * w + 1:
        return arr
    kernel = np.ones(2 * w + 1) / (2 * w + 1)
    return np.convolve(arr, kernel, mode='same')


def normalize(arr):
    arr = np.asarray(arr, dtype=np.float64)
    std = arr.std()
    if std < 1e-12:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def decimate(arr, factor):
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    out = []
    for i in range(0, n, factor):
        out.append(arr[i:i+factor].mean())
    return np.array(out, dtype=np.float64)


def dot_score(sig_a, sig_b, offset):
    sig_a = np.asarray(sig_a); sig_b = np.asarray(sig_b)
    n_a, n_b = len(sig_a), len(sig_b)
    b_lo = max(0, offset)
    b_hi = min(n_b, n_a + offset)
    if b_hi <= b_lo:
        return 0.0
    a_lo = b_lo - offset
    a_hi = b_hi - offset
    return float(np.dot(sig_a[a_lo:a_hi], sig_b[b_lo:b_hi])) / (b_hi - b_lo)


def find_top_k_peaks(scores, k, min_gap):
    scores = np.asarray(scores)
    indexed = np.argsort(scores)[::-1]
    peaks = []
    for idx in indexed:
        if len(peaks) >= k:
            break
        if all(abs(int(idx) - p) >= min_gap for p in peaks):
            peaks.append(int(idx))
    return peaks


def extract_fingertip_z_speed(df):
    n = len(df)
    speed = np.zeros(n, dtype=np.float64)
    for tip in FINGERTIPS:
        col = f"{tip}_z"
        if col not in df.columns: continue
        z = df[col].values.astype(np.float64)
        dz = np.zeros(n)
        valid = ~np.isnan(z)
        for i in range(1, n):
            if valid[i] and valid[i-1]:
                d = z[i-1] - z[i]
                if d > 0: dz[i] = d
        speed += dz
    return speed


def extract_fingertip_speed(df):
    n = len(df)
    speed = np.zeros(n, dtype=np.float64)
    for tip in FINGERTIPS:
        cols = [f"{tip}_{a}" for a in "xyz"]
        if not all(c in df.columns for c in cols): continue
        xyz = df[cols].values.astype(np.float64)
        for i in range(1, n):
            if not (np.any(np.isnan(xyz[i])) or np.any(np.isnan(xyz[i-1]))):
                speed[i] += np.linalg.norm(xyz[i] - xyz[i-1])
    return speed


def extract_fingertip_accel(df):
    spd = extract_fingertip_speed(df)
    accel = np.zeros_like(spd)
    accel[1:] = np.abs(np.diff(spd))
    return accel


def extract_wrist_speed(df):
    n = len(df)
    speed = np.zeros(n, dtype=np.float64)
    cnt = 0
    for wrist in WRISTS:
        cols = [f"{wrist}_{a}" for a in "xyz"]
        if not all(c in df.columns for c in cols): continue
        xyz = df[cols].values.astype(np.float64)
        for i in range(1, n):
            if not (np.any(np.isnan(xyz[i])) or np.any(np.isnan(xyz[i-1]))):
                speed[i] += np.linalg.norm(xyz[i] - xyz[i-1])
        cnt += 1
    if cnt > 0: speed /= cnt
    return speed


def detect_keystroke_onsets(df, fps, min_peak_gap_sec=0.03, threshold_factor=1.5):
    n = len(df)
    raw_onsets = []
    for tip in FINGERTIPS:
        col = f"{tip}_z"
        if col not in df.columns: continue
        z = df[col].values.astype(np.float64)
        vy = np.zeros(n)
        valid = ~np.isnan(z)
        for i in range(1, n):
            if valid[i] and valid[i-1]:
                vy[i] = z[i] - z[i-1]
        detrend_w = max(1, int(fps * 0.25))
        vy_trend = smooth(vy, w=detrend_w)
        vy_det = vy - vy_trend
        vy_s = smooth(vy_det, w=1)
        abs_vy = np.abs(vy_s[vy_s != 0])
        if len(abs_vy) == 0: continue
        median_abs = np.median(abs_vy)
        p75_abs = np.percentile(abs_vy, 75)
        threshold = max(median_abs * threshold_factor, p75_abs * 0.5)
        for i in range(1, n - 1):
            if vy_s[i] < -threshold and vy_s[i] < vy_s[i-1] and vy_s[i] < vy_s[i+1]:
                raw_onsets.append((i / fps, abs(vy_s[i])))
    raw_onsets.sort(key=lambda x: x[0])
    merged = []
    for t, s in raw_onsets:
        if merged and (t - merged[-1][0]) < min_peak_gap_sec:
            prev_t, prev_s = merged[-1]
            merged[-1] = (prev_t, prev_s + s)
        else:
            merged.append((t, s))
    return merged


def midi_density_curve(pm, fps, n_frames, sigma_sec=0.04):
    sigma_frames = sigma_sec * fps
    radius = int(math.ceil(sigma_frames * 3))
    curve = np.zeros(n_frames, dtype=np.float64)
    for inst in pm.instruments:
        for note in inst.notes:
            center = note.start * fps
            weight = note.velocity / 127.0
            lo = max(0, int(center) - radius)
            hi = min(n_frames - 1, int(center) + radius)
            f_arr = np.arange(lo, hi + 1)
            d = (f_arr - center) / max(sigma_frames, 1e-9)
            curve[lo:hi+1] += weight * np.exp(-0.5 * d * d)
    return curve


def motion_onset_density(onsets, fps, n_frames, sigma_sec=0.06):
    sigma_frames = sigma_sec * fps
    radius = int(math.ceil(sigma_frames * 3))
    curve = np.zeros(n_frames, dtype=np.float64)
    max_s = max((s for _, s in onsets), default=1.0)
    for t_sec, strength in onsets:
        center = t_sec * fps
        weight = strength / max(max_s, 1e-9)
        lo = max(0, int(center) - radius)
        hi = min(n_frames - 1, int(center) + radius)
        f_arr = np.arange(lo, hi + 1)
        d = (f_arr - center) / max(sigma_frames, 1e-9)
        curve[lo:hi+1] += weight * np.exp(-0.5 * d * d)
    return curve


def coarse_to_fine_correlate(midi_density, motion_keystroke_onsets,
                              motion_fingertip_speed, motion_fingertip_accel,
                              motion_wrist_speed, n_motion_frames, motion_fps,
                              max_shift_sec=MAX_SHIFT_SEC):
    max_shift = min(int(max_shift_sec * motion_fps), int(n_motion_frames / 2), 3000)

    mot_density_coarse = normalize(smooth(
        motion_onset_density(motion_keystroke_onsets, motion_fps, n_motion_frames, 0.15), w=5))
    mot_density_fine = normalize(smooth(
        motion_onset_density(motion_keystroke_onsets, motion_fps, n_motion_frames, 0.06), w=3))
    midi_coarse = normalize(smooth(midi_density, w=int(0.15 * motion_fps)))
    midi_fine = normalize(smooth(midi_density, w=3))
    n_ft_speed = normalize(smooth(motion_fingertip_speed))
    n_ft_accel = normalize(smooth(motion_fingertip_accel))
    n_wr_speed = normalize(smooth(motion_wrist_speed))

    decim_factor = max(1, int(motion_fps / 10))
    midi_c_dec = decimate(midi_coarse, decim_factor)
    motion_c_dec = decimate(mot_density_coarse, decim_factor)
    coarse_max_shift = max_shift // decim_factor

    coarse_scores = np.array([dot_score(midi_c_dec, motion_c_dec, off)
                              for off in range(-coarse_max_shift, coarse_max_shift + 1)])

    min_gap_coarse = max(1, int(0.5 * motion_fps / decim_factor))
    top_indices = find_top_k_peaks(coarse_scores, k=5, min_gap=min_gap_coarse)
    candidates = [(-coarse_max_shift + idx) * decim_factor for idx in top_indices]

    fine_radius = int(0.5 * motion_fps)
    best_offset, best_score, best_fine = 0, float("-inf"), []

    for cand in candidates:
        lo = max(-max_shift, cand - fine_radius)
        hi = min(max_shift, cand + fine_radius)
        if hi <= lo: continue
        local_scores = []
        for off in range(lo, hi + 1):
            # Score weight pattern: motion-density carries the strongest signal (note onset
            # anchors); fingertip-speed and accel are secondary; wrist is a soft prior.
            # Weights sum to 8.0 so the result stays in [0, 1] for typical inputs.
            s = (4.0 * dot_score(midi_fine, mot_density_fine, off)
                 + 2.0 * dot_score(midi_fine, n_ft_speed, off)
                 + 1.5 * dot_score(midi_fine, n_ft_accel, off)
                 + 0.5 * dot_score(midi_fine, n_wr_speed, off)) / 8.0
            local_scores.append(s)
        idx = int(np.argmax(local_scores))
        if local_scores[idx] > best_score:
            best_score = local_scores[idx]
            best_offset = lo + idx
            best_fine = local_scores

    return best_offset, best_score, best_fine


def compute_confidence(midi_onsets_sec, motion_onsets, offset_sec,
                        best_score, fine_scores, tolerance_sec=0.12):
    if fine_scores:
        arr = np.array(fine_scores)
        rng = arr.max() - arr.min()
        peak_dist = min(max((best_score - np.median(arr)) / max(rng, 1e-9), 0), 1)
    else:
        peak_dist = 0.0
    motion_times = [t + offset_sec for t, _ in motion_onsets]
    if motion_times:
        overlap_lo = min(motion_times) - tolerance_sec
        overlap_hi = max(motion_times) + tolerance_sec
        midi_in_range = [mt for mt in midi_onsets_sec if overlap_lo <= mt <= overlap_hi]
    else:
        midi_in_range = midi_onsets_sec
    matched = sum(1 for mt in midi_in_range if any(abs(mt - mot) < tolerance_sec for mot in motion_times))
    total = max(min(len(midi_in_range), len(motion_times)), 1)
    match_rate = matched / total
    confidence = 0.3 * peak_dist + 0.7 * match_rate
    quality = "good" if confidence >= 0.7 else ("fair" if confidence >= 0.4 else "poor")
    return confidence, quality


def save_diagnostic(midi_density, mot_density, ft_z_speed,
                    offset_frames, fps, stem, dataset, out_dir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    sec_pf = 1.0 / fps
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    t_midi = np.arange(len(midi_density)) * sec_pf
    t_mot = np.arange(len(mot_density)) * sec_pf
    offset_sec = offset_frames * sec_pf
    axes[0].plot(t_midi, normalize(midi_density), color='blue', alpha=0.7, label='MIDI density', linewidth=0.5)
    axes[0].plot(t_mot + offset_sec, normalize(mot_density), color='red', alpha=0.7, label='Motion onset (shifted)', linewidth=0.5)
    axes[0].set_title(f"{dataset}/{stem} — offset={offset_sec:.3f}s")
    axes[0].legend(fontsize=8)
    t_ftz = np.arange(len(ft_z_speed)) * sec_pf
    axes[1].plot(t_midi, normalize(midi_density), color='blue', alpha=0.7, label='MIDI density', linewidth=0.5)
    axes[1].plot(t_ftz + offset_sec, normalize(smooth(ft_z_speed)), color='green', alpha=0.7, label='Fingertip Z-speed (shifted)', linewidth=0.5)
    axes[1].legend(fontsize=8); axes[1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}.png", dpi=100); plt.close(fig)


def align_trial(hand_csv, midi_path, stem, dataset, out_root, img_root, body_csv=None):
    df = pd.read_csv(hand_csv)
    n_frames = len(df)
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    midi_duration = pm.get_end_time()
    n_midi_frames = int(midi_duration * MOTION_FPS) + 1
    motion_duration = n_frames / MOTION_FPS

    ft_cols = [c for c in df.columns if c.endswith("_z") and any(
        c.startswith(f"{s}{f}F") for s in ["L","R"] for f in ["I","M","R","L","T"])]
    if not ft_cols or df[ft_cols].notna().sum().sum() == 0:
        out_dir = out_root / dataset; out_dir.mkdir(parents=True, exist_ok=True)
        result = {"stem": stem, "dataset": dataset, "offset_frames": 0,
                  "offset_sec": 0.0, "confidence": 0.0, "quality": "no_motion_data",
                  "best_score": 0.0, "n_midi_onsets": len([n for inst in pm.instruments for n in inst.notes]),
                  "n_motion_onsets": 0, "hand_csv": str(hand_csv), "midi_path": str(midi_path),
                  "body_csv": str(body_csv) if body_csv else None}
        with open(out_dir / f"{stem}.json", "w") as f:
            json.dump(result, f, indent=2)
        return result

    ft_z_speed = extract_fingertip_z_speed(df)
    ft_accel = extract_fingertip_accel(df)
    wr_speed = extract_wrist_speed(df)
    keystroke_onsets = detect_keystroke_onsets(df, MOTION_FPS)
    target_len = max(n_frames, n_midi_frames)
    midi_dens = midi_density_curve(pm, MOTION_FPS, target_len)
    offset_frames, best_score, fine_scores = coarse_to_fine_correlate(
        midi_dens, keystroke_onsets, ft_z_speed, ft_accel, wr_speed, n_frames, MOTION_FPS)
    offset_sec = offset_frames / MOTION_FPS

    motion_lo = offset_sec
    motion_hi = offset_sec + motion_duration
    midi_onsets_sec = [n.start for inst in pm.instruments for n in inst.notes]
    midi_in_overlap = [t for t in midi_onsets_sec if motion_lo <= t <= motion_hi]
    if len(midi_in_overlap) == 0 and len(midi_onsets_sec) > 0 and len(keystroke_onsets) > 0:
        motion_onset_times = [t for t, _ in keystroke_onsets]
        best_fb_off, best_fb_matched = 0.0, 0
        min_off = -midi_duration
        max_off = midi_duration
        for off_i in range(int(min_off * 100), int(max_off * 100) + 1):
            off = off_i / 100.0
            shifted = [t + off for t in motion_onset_times]
            matched = sum(1 for mt in midi_onsets_sec if any(abs(mt - s) < 0.12 for s in shifted))
            if matched > best_fb_matched:
                best_fb_matched = matched
                best_fb_off = off
        if best_fb_matched > 0:
            for off_i in range(int((best_fb_off - 0.1) * 1000), int((best_fb_off + 0.1) * 1000) + 1):
                off = off_i / 1000.0
                shifted = [t + off for t in motion_onset_times]
                matched = sum(1 for mt in midi_onsets_sec if any(abs(mt - s) < 0.12 for s in shifted))
                if matched > best_fb_matched:
                    best_fb_matched = matched
                    best_fb_off = off
            offset_sec = best_fb_off
            offset_frames = int(offset_sec * MOTION_FPS)
            print(f"  fallback: offset={offset_sec:.3f}s matched={best_fb_matched}")
    confidence, quality = compute_confidence(midi_onsets_sec, keystroke_onsets, offset_sec, best_score, fine_scores)
    out_dir = out_root / dataset; out_dir.mkdir(parents=True, exist_ok=True)
    pm_shifted = pretty_midi.PrettyMIDI(str(midi_path))
    for inst in pm_shifted.instruments:
        for note in inst.notes:
            note.start = max(0.0, note.start + offset_sec)
            note.end = max(note.start + 0.001, note.end + offset_sec)
    pm_shifted.write(str(out_dir / f"{stem}.mid"))
    result = {"stem": stem, "dataset": dataset, "offset_frames": int(offset_frames),
              "offset_sec": round(offset_sec, 4), "confidence": round(confidence, 3),
              "quality": quality, "best_score": round(best_score, 4),
              "n_midi_onsets": len(midi_onsets_sec), "n_motion_onsets": len(keystroke_onsets),
              "hand_csv": str(hand_csv), "midi_path": str(midi_path),
              "body_csv": str(body_csv) if body_csv else None}
    with open(out_dir / f"{stem}.json", "w") as f:
        json.dump(result, f, indent=2)
    mot_dens = motion_onset_density(keystroke_onsets, MOTION_FPS, n_frames)
    try:
        save_diagnostic(midi_dens, mot_dens, ft_z_speed, offset_frames, MOTION_FPS, stem, dataset, img_root / dataset)
    except Exception as e:
        print(f"  img err: {e}")
    return result


def run_pass(hand_root: Path, midi_root: Path, body_root: Path,
             out_root: Path, img_root: Path, label: str,
             datasets: list[str]) -> None:
    print(f"{label}: MIDI <-> Motion alignment")
    print(f"Output: {out_root}")
    trials = []
    for dataset in datasets:
        hand_dir = hand_root / dataset
        midi_dir = midi_root / dataset
        if not hand_dir.exists() or not midi_dir.exists(): continue
        hand_files = {f.stem: f for f in hand_dir.glob("*.csv")}
        midi_files = {f.stem: f for f in midi_dir.glob("*.mid")}
        body_dir = body_root / dataset
        body_files = {f.stem: f for f in body_dir.glob("*.csv")} if body_dir.exists() else {}
        for stem in sorted(set(hand_files.keys()) & set(midi_files.keys())):
            trials.append({"dataset": dataset, "stem": stem, "hand_csv": hand_files[stem],
                           "midi_path": midi_files[stem], "body_csv": body_files.get(stem)})
    print(f"\n{len(trials)} trials")
    print("=" * 70)
    success, fail = 0, 0
    qc = {"good": 0, "fair": 0, "poor": 0, "no_motion_data": 0}
    for i, t in enumerate(trials, 1):
        out_json = out_root / t["dataset"] / f"{t['stem']}.json"
        if out_json.exists():
            with open(out_json) as f: existing = json.load(f)
            q = existing.get("quality", "poor")
            qc.setdefault(q, 0)
            qc[q] += 1
            print(f"[{i}/{len(trials)}] {t['dataset']}/{t['stem']} - skip"); success += 1; continue
        print(f"[{i}/{len(trials)}] {t['dataset']}/{t['stem']}")
        try:
            r = align_trial(t["hand_csv"], t["midi_path"], t["stem"], t["dataset"],
                            out_root, img_root, t["body_csv"])
            q = r["quality"]
            qc.setdefault(q, 0)
            qc[q] += 1
            print(f"  offset={r['offset_sec']:.3f}s  conf={r['confidence']:.3f} ({r['quality']})")
            success += 1
        except Exception as e:
            print(f"  FAIL: {e}"); fail += 1
    print(f"\nDone: success={success} fail={fail}")
    print(f"Quality: good={qc['good']} fair={qc['fair']} poor={qc['poor']} no_motion={qc.get('no_motion_data', 0)}")
