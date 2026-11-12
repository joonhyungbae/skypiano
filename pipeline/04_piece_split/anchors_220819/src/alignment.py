"""
Motion <-> MIDI temporal alignment.

220819 has independent clocks per capture system:
  - audio + MIDI: co-recorded (effectively shared clock)
  - hand motion (OptiTrack Motive): own clock
  - body motion (Qualisys): own clock

Each recording needs per-modality offsets so that slices computed in audio-time
can be translated to motion-frame indices.

Definitions
-----------
align_shift_sec: offset to add to an audio-time timestamp to get the
    corresponding motion-time timestamp. I.e.
        motion_t = audio_t + align_shift_sec
        motion_frame_idx = int(round((audio_t + align_shift_sec) * MOTION_FPS))
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate, find_peaks

MOTION_FPS = 120.0      # verified via row_count / reported_duration (OptiTrack Motive + Qualisys both at 120 Hz for 220819)
MAX_SHIFT_SEC = 300.0   # body motion may be session-continuous; hand typically ±20 s
VELOCITY_SMOOTH_W = 3
PEAK_PROMINENCE = 0.5   # normalized (z-scored) velocity units

# Marker sets for onset detection. Using fingertips (piano-key strikes) for
# hand since they have the sharpest velocity spikes; a broader set of body
# markers for the envelope-based body alignment path.
HAND_ONSET_MARKERS = ["LIF", "LMF", "LRF", "LLF", "LTF", "RIF", "RMF", "RRF", "RLF", "RTF"]
BODY_ENVELOPE_MARKERS = ["HEADT", "HEADA", "HEADR", "HEADL", "C7",
                        "ACROR", "ACROL", "TRUNKAR", "TRUNKAL"]

# Envelope alignment parameters (used for body). Lower sample rate + heavy
# smoothing, then xcorr the continuous envelopes. Body motion is dominated by
# low-frequency musical-phrasing oscillation, not discrete strikes.
ENVELOPE_SR = 10.0        # Hz - resampled grid for envelope xcorr
ENVELOPE_SIGMA_SEC = 1.5  # gaussian smoothing width in seconds


def _read_motion_csv(csv_path: Path) -> tuple[list[str], np.ndarray]:
    """Return (column_names, data_array_rows_x_cols). First row is header."""
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        data_rows = []
        for row in reader:
            if not row:
                continue
            try:
                data_rows.append([float(x) if x else np.nan for x in row])
            except ValueError:
                data_rows.append([np.nan] * len(header))
    return header, np.array(data_rows, dtype=np.float64)


def _marker_xyz_idxs(header: list[str], marker: str) -> tuple[int, int, int] | None:
    try:
        return (
            header.index(f"{marker}_x"),
            header.index(f"{marker}_y"),
            header.index(f"{marker}_z"),
        )
    except ValueError:
        return None


def _smooth(x: np.ndarray, w: int = VELOCITY_SMOOTH_W) -> np.ndarray:
    if w < 1 or len(x) < 2 * w + 1:
        return x
    kernel = np.ones(2 * w + 1) / (2 * w + 1)
    return np.convolve(x, kernel, mode="same")


def _marker_velocity(data: np.ndarray, xyz: tuple[int, int, int]) -> np.ndarray:
    x = data[:, xyz[0]]
    y = data[:, xyz[1]]
    z = data[:, xyz[2]]
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    dz = np.diff(z, prepend=z[0])
    v = np.sqrt(dx * dx + dy * dy + dz * dz)
    # NaN -> 0 (missing marker frames)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return _smooth(v, VELOCITY_SMOOTH_W)


def motion_onset_pulses(csv_path: Path, markers: list[str]) -> np.ndarray:
    """Compute a fused velocity-peak pulse train from the given markers.

    For each marker: velocity magnitude -> peak detection -> pulses at peaks
    weighted by peak prominence. Sum across markers.
    """
    header, data = _read_motion_csv(csv_path)
    if len(data) == 0:
        return np.zeros(0, dtype=np.float32)

    pulses = np.zeros(len(data), dtype=np.float32)
    n_markers_used = 0
    for m in markers:
        idx = _marker_xyz_idxs(header, m)
        if idx is None:
            continue
        v = _marker_velocity(data, idx)
        std = v.std()
        if std < 1e-9:
            continue
        v_z = (v - v.mean()) / std
        peaks, props = find_peaks(v_z, prominence=PEAK_PROMINENCE)
        for p, prom in zip(peaks, props["prominences"]):
            pulses[p] += float(prom)
        n_markers_used += 1

    return pulses


def motion_activity_envelope(csv_path: Path, markers: list[str],
                              native_fps: float = MOTION_FPS,
                              target_sr: float = ENVELOPE_SR,
                              sigma_sec: float = ENVELOPE_SIGMA_SEC) -> np.ndarray:
    """Continuous activity envelope: sum of marker velocity magnitudes,
    gaussian-smoothed, downsampled to target_sr."""
    header, data = _read_motion_csv(csv_path)
    if len(data) == 0:
        return np.zeros(0, dtype=np.float32)
    total_v = np.zeros(len(data), dtype=np.float64)
    for m in markers:
        idx = _marker_xyz_idxs(header, m)
        if idx is None:
            continue
        total_v += _marker_velocity(data, idx)
    # smooth at native rate then downsample
    sigma_frames = sigma_sec * native_fps
    smoothed = gaussian_filter1d(total_v, sigma=sigma_frames, mode="reflect")
    factor = native_fps / target_sr
    target_len = int(round(len(smoothed) / factor))
    src_idx = np.linspace(0, len(smoothed) - 1, target_len)
    downsampled = np.interp(src_idx, np.arange(len(smoothed)), smoothed)
    return downsampled.astype(np.float32)


def midi_activity_envelope(midi_pulses: np.ndarray, midi_fps: float,
                           target_sr: float = ENVELOPE_SR,
                           sigma_sec: float = ENVELOPE_SIGMA_SEC) -> np.ndarray:
    """Continuous activity envelope from MIDI pulses: gaussian-smoothed note
    density. Matches motion_activity_envelope's scale."""
    sigma_frames = sigma_sec * midi_fps
    smoothed = gaussian_filter1d(midi_pulses.astype(np.float64), sigma=sigma_frames, mode="reflect")
    factor = midi_fps / target_sr
    target_len = int(round(len(smoothed) / factor))
    src_idx = np.linspace(0, len(smoothed) - 1, target_len)
    downsampled = np.interp(src_idx, np.arange(len(smoothed)), smoothed)
    return downsampled.astype(np.float32)


def align_envelopes(midi_env: np.ndarray, motion_env: np.ndarray,
                    sr: float = ENVELOPE_SR,
                    max_shift_sec: float = MAX_SHIFT_SEC) -> dict:
    """Cross-correlate continuous envelopes, return align_shift_sec."""
    if len(midi_env) == 0 or len(motion_env) == 0:
        return {"ok": False, "reason": "empty_envelope"}
    m = midi_env - midi_env.mean()
    n = motion_env - motion_env.mean()
    m_norm = m / (np.linalg.norm(m) + 1e-9)
    n_norm = n / (np.linalg.norm(n) + 1e-9)
    corr = correlate(n_norm, m_norm, mode="full", method="fft")
    lags = np.arange(len(corr)) - (len(midi_env) - 1)
    max_lag = int(round(max_shift_sec * sr))
    mask = np.abs(lags) <= max_lag
    if not mask.any():
        return {"ok": False, "reason": "no_valid_lag_range"}
    corr_m = corr[mask]
    lags_m = lags[mask]
    peak_i = int(np.argmax(corr_m))
    peak_val = float(corr_m[peak_i])
    peak_lag = int(lags_m[peak_i])
    # mask a ±3 sec window around peak for secondary-peak search
    mask_radius = int(round(3.0 * sr))
    masked = corr_m.copy()
    lo = max(0, peak_i - mask_radius)
    hi = min(len(masked), peak_i + mask_radius + 1)
    masked[lo:hi] = -np.inf
    secondary = float(np.max(masked)) if np.any(np.isfinite(masked)) else 0.0
    ratio = peak_val / secondary if secondary > 1e-9 else float("inf")
    return {
        "ok": True,
        "align_shift_sec": round(peak_lag / sr, 3),
        "peak_corr": round(peak_val, 4),
        "secondary_ratio": None if ratio == float("inf") else round(ratio, 2),
    }


def align_midi_to_motion(midi_pulses: np.ndarray, motion_pulses: np.ndarray,
                         midi_fps: float, motion_fps: float = MOTION_FPS,
                         max_shift_sec: float = MAX_SHIFT_SEC) -> dict:
    """
    Cross-correlate pulse trains (possibly at different sample rates).

    Resamples motion_pulses to midi_fps grid via nearest-frame mapping so that
    a single shift in frames == shift in seconds / midi_fps.

    Returns:
        align_shift_sec: motion_t - audio_t
        peak_corr: raw xcorr peak value
        secondary_ratio: peak / second-peak (confidence; higher is better)
        n_midi_onsets, n_motion_onsets
    """
    # Resample motion pulses to midi grid: if midi_fps=100 and motion_fps=100,
    # no change. For other rates, interpolate.
    if abs(midi_fps - motion_fps) > 1e-3:
        # simple linear resample preserving energy approximately
        target_len = int(round(len(motion_pulses) * midi_fps / motion_fps))
        src_idx = np.linspace(0, len(motion_pulses) - 1, target_len)
        motion_rs = np.interp(src_idx, np.arange(len(motion_pulses)), motion_pulses)
    else:
        motion_rs = motion_pulses

    m = midi_pulses - midi_pulses.mean()
    n = motion_rs - motion_rs.mean()
    m_norm = m / (np.linalg.norm(m) + 1e-9)
    n_norm = n / (np.linalg.norm(n) + 1e-9)
    corr = correlate(n_norm, m_norm, mode="full", method="fft")
    lags = np.arange(len(corr)) - (len(midi_pulses) - 1)

    max_lag = int(round(max_shift_sec * midi_fps))
    mask = np.abs(lags) <= max_lag
    if not mask.any():
        return {"ok": False, "reason": "no_valid_lag_range"}
    corr_m = corr[mask]
    lags_m = lags[mask]
    peak_i = int(np.argmax(corr_m))
    peak_val = float(corr_m[peak_i])
    peak_lag = int(lags_m[peak_i])

    mask_radius = int(round(2.0 * midi_fps))
    masked = corr_m.copy()
    lo = max(0, peak_i - mask_radius)
    hi = min(len(masked), peak_i + mask_radius + 1)
    masked[lo:hi] = -np.inf
    secondary = float(np.max(masked)) if np.any(np.isfinite(masked)) else 0.0
    ratio = peak_val / secondary if secondary > 1e-9 else float("inf")

    return {
        "ok": True,
        "align_shift_sec": round(peak_lag / midi_fps, 3),
        "peak_corr": round(peak_val, 4),
        "secondary_ratio": None if ratio == float("inf") else round(ratio, 2),
        "n_midi_onsets": int((midi_pulses > 0).sum()),
        "n_motion_onsets": int((motion_rs > 0).sum()),
    }
