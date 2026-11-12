"""
Locate helpers: find short anchor slice within long Stage 01 recording.

Primary method: MIDI onset-pulse cross-correlation (pitch-weighted).
Fallback:       audio onset-strength envelope cross-correlation.

Both produce (t_start_sec, peak_ncc, secondary_ratio). Caller chooses.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pretty_midi
from scipy.signal import fftconvolve

GRID_SR = 100  # frames-per-second for both onset-pulse and onset-envelope grids
AUDIO_LOAD_SR = 22050

# On agreement check: distance below which audio+MIDI locates are considered
# to converge (same event).
AGREE_TOL_SEC = 1.0


def midi_to_onset_pulses(mid_path: Path, sr: int = GRID_SR) -> np.ndarray:
    """Rasterize MIDI note onsets to a pitch-weighted pulse train at sr Hz."""
    pm = pretty_midi.PrettyMIDI(str(mid_path))
    dur = pm.get_end_time()
    n = int(np.ceil(dur * sr)) + 1
    pulses = np.zeros(n, dtype=np.float32)
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            idx = int(round(note.start * sr))
            if 0 <= idx < n:
                pulses[idx] += note.pitch / 127.0
    return pulses


def audio_to_onset_env(wav_path: Path, sr: int = AUDIO_LOAD_SR, hop_sr: int = GRID_SR) -> np.ndarray:
    """Load audio and compute onset-strength envelope at hop_sr Hz."""
    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    hop_length = max(1, sr // hop_sr)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    return env.astype(np.float32)


def _ncc_locate(short: np.ndarray, long: np.ndarray) -> tuple[int, float, float]:
    """Normalized cross-correlation with sliding L2 normalization.
    Returns (peak_idx_frames, peak_ncc, secondary_ratio)."""
    s = short - short.mean()
    l = long - long.mean()
    corr = fftconvolve(l, s[::-1], mode="valid")
    norm = np.sqrt(np.sum(s * s)) * np.sqrt(
        np.convolve(l * l, np.ones(len(s)), mode="valid")
    )
    norm = np.where(norm > 1e-9, norm, 1e-9)
    ncc = corr / norm

    peak_idx = int(np.argmax(ncc))
    peak_val = float(ncc[peak_idx])

    mask_radius = 3 * GRID_SR
    masked = ncc.copy()
    lo = max(0, peak_idx - mask_radius)
    hi = min(len(masked), peak_idx + mask_radius + 1)
    masked[lo:hi] = -np.inf
    secondary = float(np.max(masked)) if np.any(np.isfinite(masked)) else 0.0
    ratio = peak_val / secondary if secondary > 0 else float("inf")
    return peak_idx, peak_val, ratio


def locate_midi(anchor_mid: Path, source_mid: Path) -> dict:
    """Locate anchor MIDI inside source MIDI. Returns dict with method=midi."""
    short = midi_to_onset_pulses(anchor_mid)
    if short.sum() == 0:
        return {"method": "midi", "ok": False, "reason": "anchor_midi_empty"}
    long = midi_to_onset_pulses(source_mid)
    if len(short) >= len(long):
        return {"method": "midi", "ok": False, "reason": "anchor_longer_than_source"}
    idx, peak, ratio = _ncc_locate(short, long)
    return {
        "method": "midi",
        "ok": True,
        "t_start_sec": round(idx / GRID_SR, 3),
        "t_end_sec": round((idx + len(short)) / GRID_SR, 3),
        "anchor_dur_sec": round(len(short) / GRID_SR, 3),
        "source_dur_sec": round(len(long) / GRID_SR, 3),
        "peak_ncc": round(peak, 4),
        "secondary_ratio": None if ratio == float("inf") else round(ratio, 2),
    }


def locate_audio(anchor_wav: Path, source_wav: Path) -> dict:
    """Locate anchor audio inside source audio via onset-envelope xcorr."""
    short = audio_to_onset_env(anchor_wav)
    long = audio_to_onset_env(source_wav)
    if len(short) >= len(long):
        return {"method": "audio", "ok": False, "reason": "anchor_longer_than_source"}
    idx, peak, ratio = _ncc_locate(short, long)
    return {
        "method": "audio",
        "ok": True,
        "t_start_sec": round(idx / GRID_SR, 3),
        "t_end_sec": round((idx + len(short)) / GRID_SR, 3),
        "anchor_dur_sec": round(len(short) / GRID_SR, 3),
        "source_dur_sec": round(len(long) / GRID_SR, 3),
        "peak_ncc": round(peak, 4),
        "secondary_ratio": None if ratio == float("inf") else round(ratio, 2),
    }


# MIDI-pulse ratio threshold above which we accept without audio cross-check.
MIDI_STRONG_RATIO = 1.5

# Audio-env ratio and ncc thresholds for fallback acceptance.
AUDIO_OK_RATIO = 1.3
AUDIO_OK_NCC = 0.7


def combined_locate(anchor_mid: Path, source_mid: Path,
                    anchor_wav: Path, source_wav: Path) -> dict:
    """
    Tiered locator:
      1) MIDI xcorr; if ratio >= MIDI_STRONG_RATIO → accept (method='midi').
      2) Else audio xcorr; if (ratio >= AUDIO_OK_RATIO AND ncc >= AUDIO_OK_NCC)
         AND agrees with MIDI within AGREE_TOL_SEC → accept (method='midi+audio').
      3) Else accept best-ratio result with low_confidence=true.

    Returns dict with: method, t_start_sec, t_end_sec, confidence ∈
    {'strong', 'medium', 'low'}, plus raw midi/audio sub-results.
    """
    m = locate_midi(anchor_mid, source_mid)
    a = locate_audio(anchor_wav, source_wav)

    # Empty-MIDI fallback: trust audio if reasonably strong.
    if not m.get("ok"):
        if a.get("ok") and (a["secondary_ratio"] or 0) >= AUDIO_OK_RATIO and a["peak_ncc"] >= AUDIO_OK_NCC:
            return {
                "method": "audio",
                "t_start_sec": a["t_start_sec"],
                "t_end_sec": a["t_end_sec"],
                "anchor_dur_sec": a["anchor_dur_sec"],
                "confidence": "medium",
                "reason": m.get("reason"),
                "midi": m,
                "audio": a,
            }
        return {
            "method": "audio" if a.get("ok") else "none",
            "t_start_sec": a["t_start_sec"] if a.get("ok") else None,
            "t_end_sec": a["t_end_sec"] if a.get("ok") else None,
            "anchor_dur_sec": a.get("anchor_dur_sec"),
            "confidence": "low",
            "reason": m.get("reason"),
            "midi": m,
            "audio": a,
        }

    # MIDI strong → accept; agreement with audio (if ok) promotes to strong.
    agrees = (
        a.get("ok")
        and abs(a["t_start_sec"] - m["t_start_sec"]) <= AGREE_TOL_SEC
    )
    midi_ratio = m["secondary_ratio"] or 99.0

    if midi_ratio >= MIDI_STRONG_RATIO:
        confidence = "strong" if agrees else "medium"
        return {
            "method": "midi+audio" if agrees else "midi",
            "t_start_sec": m["t_start_sec"],
            "t_end_sec": m["t_end_sec"],
            "anchor_dur_sec": m["anchor_dur_sec"],
            "confidence": confidence,
            "midi": m,
            "audio": a,
        }

    # MIDI weak but audio may help
    if agrees:
        return {
            "method": "midi+audio",
            "t_start_sec": m["t_start_sec"],
            "t_end_sec": m["t_end_sec"],
            "anchor_dur_sec": m["anchor_dur_sec"],
            "confidence": "medium",
            "midi": m,
            "audio": a,
        }

    # No agreement and MIDI weak: take MIDI anyway but flag low.
    return {
        "method": "midi",
        "t_start_sec": m["t_start_sec"],
        "t_end_sec": m["t_end_sec"],
        "anchor_dur_sec": m["anchor_dur_sec"],
        "confidence": "low",
        "midi": m,
        "audio": a,
    }
