"""
Audio cross-correlation helpers for Stage 01 video rough-cut.

Given a per-trial WAV and a session-level video (or audio), find the offset
within the source where the trial starts, by cross-correlating low-rate
envelopes. Stream-copy cut with padding is handled in the caller.

This module is intentionally minimal — only does envelope-xcorr and caching.
"""
from __future__ import annotations
import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np
from scipy.signal import correlate, resample_poly
import wave

# Cache directory for extracted video-audio mono PCM. Defaults to a path under
# the in-repo Data/ tree but can be redirected by external callers via the
# VIDEO_XCORR_CACHE environment variable.
DEFAULT_CACHE_ROOT = Path("Data/Pipeline/01_ingest/_cache/video_audio")
CACHE_ROOT = Path(os.environ.get("VIDEO_XCORR_CACHE", DEFAULT_CACHE_ROOT))
ENV_SR = 50          # Hz, envelope sample rate after rectify+downsample
CLIP_SR = 16000      # Hz, intermediate mono PCM sample rate


def _cache_key(src: Path) -> Path:
    h = hashlib.sha1(str(src.resolve()).encode("utf-8")).hexdigest()[:16]
    return CACHE_ROOT / f"{src.stem}_{h}.wav"


def extract_audio_mono(src: Path) -> Path:
    """Extract a 16kHz mono PCM WAV from any video/audio source. Cached."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    out = _cache_key(src)
    if out.exists():
        return out
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(CLIP_SR), "-f", "wav", str(out),
    ]
    subprocess.run(cmd, check=True, timeout=600)
    return out


def _read_wav_mono(p: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(p), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
        ch = w.getnchannels()
        sw = w.getsampwidth()
    if sw == 3:
        # 24-bit PCM: place each little-endian sample in the top three bytes of an int32
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        x = (b[:, 0].astype(np.int32) << 8 | b[:, 1].astype(np.int32) << 16
             | b[:, 2].astype(np.int32) << 24).astype(np.float32) / 256.0
    else:
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
        x = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sw > 1:
        x /= float(1 << (8 * sw - 1))
    return x, sr


def _envelope(x: np.ndarray, sr: int, out_sr: int = ENV_SR) -> np.ndarray:
    """Rectify + downsample + log-compress to a loudness envelope."""
    rect = np.abs(x)
    g = np.gcd(sr, out_sr)
    up = out_sr // g
    down = sr // g
    env = resample_poly(rect, up, down).astype(np.float32)
    # The polyphase filter rings slightly below zero next to sharp onsets; at
    # high output rates that can push log1p below its domain (NaN).
    env = np.log1p(np.maximum(env, 0.0) * 32.0)  # compress dynamic range so xcorr isn't peak-dominated
    return env


def find_offset(trial_wav: Path, source_audio: Path) -> dict:
    """
    Normalized cross-correlation between log-envelopes.

    Returns {offset_sec, ncc, confidence, method}, where `ncc` is the
    Pearson correlation at the best offset (-1..1) and `confidence` is
    peak_ncc divided by the std of the nonpeak ncc distribution.
    """
    tx, tsr = _read_wav_mono(trial_wav)
    sx, ssr = _read_wav_mono(source_audio)
    t_env = _envelope(tx, tsr)
    s_env = _envelope(sx, ssr)
    if len(t_env) >= len(s_env):
        return {"offset_sec": 0.0, "ncc": 0.0, "confidence": 0.0,
                "method": "log_env_ncc", "error": "trial_longer_than_source"}

    # Zero-mean the template (required for NCC)
    t0 = t_env - t_env.mean()
    t_norm = float(np.sqrt((t0 * t0).sum())) or 1e-9

    # Raw correlation of template(zero-mean) with source
    xc = correlate(s_env, t0, mode="valid")

    # Per-offset local norm of the source window
    n = len(t_env)
    s_sq_cum = np.concatenate(([0.0], np.cumsum(s_env * s_env)))
    s_sum_cum = np.concatenate(([0.0], np.cumsum(s_env)))
    k = np.arange(len(xc))
    win_sq = s_sq_cum[k + n] - s_sq_cum[k]
    win_sum = s_sum_cum[k + n] - s_sum_cum[k]
    win_var = win_sq - (win_sum * win_sum) / n
    win_norm = np.sqrt(np.maximum(win_var, 1e-9))

    ncc = xc / (t_norm * win_norm)
    peak_idx = int(np.argmax(ncc))
    peak = float(ncc[peak_idx])

    # Confidence: peak height vs the spread of the rest of the distribution
    mask = np.ones_like(ncc, dtype=bool)
    guard = max(int(0.5 * ENV_SR), 1)
    lo = max(0, peak_idx - guard)
    hi = min(len(ncc), peak_idx + guard + 1)
    mask[lo:hi] = False
    bg = ncc[mask]
    bg_std = float(bg.std()) or 1e-9
    confidence = (peak - float(bg.mean())) / bg_std

    return {
        "offset_sec": round(peak_idx / ENV_SR, 3),
        "ncc": round(peak, 4),
        "confidence": round(confidence, 3),
        "method": "log_env_ncc",
    }


def rough_cut(video_src: Path, offset_sec: float, duration_sec: float,
              out_path: Path, pad_sec: float = 2.0) -> bool:
    """Stream-copy rough cut. Returns True on success."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start = max(0.0, offset_sec - pad_sec)
    total = duration_sec + 2 * pad_sec
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-i", str(video_src),
        "-t", f"{total:.3f}",
        "-c", "copy",
        "-map", "0:v:0", "-map", "0:a?",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=300)
        return True
    except subprocess.CalledProcessError:
        return False
