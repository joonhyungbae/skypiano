"""Place a MIDI take on its audio recording's clock by note onsets.

Used by Stage 01 where audio and MIDI of a take were recorded separately and
start at different moments (211026: session-timeline MIDI regions; 211215:
MIDI starting at the first note, audio with its own lead-in).

The MIDI onset train (relative to its first note) is correlated with the
audio onset strength at every offset at which the whole train lies inside
the audio padded by PAD_S on each side (NOTE_FS), then refined at REFINE_FS
within +-REFINE_S of the best one.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi

from _common import motion_midi_xcorr as mx

FS = mx.NOTE_FS
PAD_S = 10
REFINE_FS = 500
REFINE_S = 0.1


def onset_strength(wav: Path, fs: int) -> np.ndarray:
    import librosa
    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    o = librosa.onset.onset_strength(y=y, sr=sr, hop_length=int(round(sr / fs)))
    return (o - o.mean()) / (o.std() + 1e-9)


def onset_train_at(onsets: np.ndarray, fs: int) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d
    x = np.zeros(int(onsets.max() * fs) + 2)
    np.add.at(x, (onsets * fs).astype(int), 1)
    return gaussian_filter1d(x, mx.NOTE_SMOOTH * fs / FS)


def shift_midi(src: Path, dst: Path, delta: float) -> None:
    """Write `src` with every event moved by `delta` seconds (events before 0 dropped)."""
    pm = pretty_midi.PrettyMIDI(str(src))
    out = pretty_midi.PrettyMIDI()
    for inst in pm.instruments:
        ni = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        ni.notes = [pretty_midi.Note(n.velocity, n.pitch, n.start + delta, n.end + delta)
                    for n in inst.notes if n.end + delta > 0]
        for n in ni.notes:
            n.start = max(0.0, n.start)
        ni.control_changes = [pretty_midi.ControlChange(c.number, c.value, c.time + delta)
                              for c in inst.control_changes if c.time + delta >= 0]
        ni.pitch_bends = [pretty_midi.PitchBend(b.pitch, b.time + delta) for b in inst.pitch_bends if b.time + delta >= 0]
        out.instruments.append(ni)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.write(str(dst))


def coarse_offset(audio_fs: np.ndarray, onsets: np.ndarray) -> tuple[float, float, float]:
    """(audio time of the first note, ncc, margin) at NOTE_FS; `audio_fs` from onset_strength(wav, FS).
    margin = ncc minus the best correlation more than 1 s away."""
    padded = np.concatenate([np.zeros(PAD_S * FS), audio_fs, np.zeros(PAD_S * FS)])
    m = mx.onset_train(onsets - onsets[0])
    if len(m) >= len(padded):
        return float("nan"), -1.0, 0.0
    r = mx.slide_ncc(padded, m)
    k = int(np.argmax(r))
    rest = r.copy()
    rest[max(0, k - FS):k + FS] = -1
    return k / FS - PAD_S, float(r[k]), float(r[k] - rest.max())


def refine_offset(wav: Path, onsets: np.ndarray, coarse: float, within: float = REFINE_S) -> float:
    """Audio time of the first note at REFINE_FS within +-`within` s of `coarse`."""
    a = onset_strength(wav, REFINE_FS)
    m = onset_train_at(onsets - onsets[0], REFINE_FS)
    c = int(round((coarse + PAD_S) * REFINE_FS))
    h = int(within * REFINE_FS)
    padded = np.concatenate([np.zeros(PAD_S * REFINE_FS), a, np.zeros(PAD_S * REFINE_FS)])
    seg = padded[max(0, c - h):c + h + len(m)]
    k = int(np.argmax(mx.slide_ncc(seg, m)))
    return (max(0, c - h) + k) / REFINE_FS - PAD_S


# ── Pitch-wise placement (for dense or repetitive material) ──
# Onset strength alone gives near-equal peaks for repetitive passages. Here
# each of the 88 keys is matched separately: the audio's per-key energy rise
# (CQT, one bin per key, half-wave rectified log difference) against the
# MIDI's per-key onsets, summed over keys and divided by the audio energy
# under the window. CQT frames are centred, so the result runs a few tens of
# ms early; refine_offset settles the fine timing.
CQT_SR = 22050
CQT_HOP = 256


def _key_flux(wav: Path) -> np.ndarray:
    import librosa
    y, _ = librosa.load(str(wav), sr=CQT_SR, mono=True)
    c = np.abs(librosa.cqt(y, sr=CQT_SR, hop_length=CQT_HOP, fmin=librosa.note_to_hz("A0"),
                           n_bins=88, bins_per_octave=12))
    lg = np.log1p(100 * c / (c.max() + 1e-9))
    return np.concatenate([np.zeros((88, 1)), np.maximum(0, np.diff(lg, axis=1))], axis=1)


def pitch_offset(wav: Path, onsets_pitches: list[tuple[float, int]], pad_s: float = 20) -> tuple[float, float]:
    """(audio time of the first note, margin) -- margin = 1 - second best / best, > 1 s away."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import fftconvolve
    fs = CQT_SR / CQT_HOP
    first = min(t for t, _ in onsets_pitches)
    roll = np.zeros((88, int((max(t for t, _ in onsets_pitches) - first) * fs) + 4))
    for t, p in onsets_pitches:
        if 21 <= p <= 108:
            roll[p - 21, int((t - first) * fs)] += 1
    roll = gaussian_filter1d(roll, 1.0, axis=1)
    pad = int(pad_s * fs)
    a = _key_flux(wav)
    a = np.concatenate([np.zeros((88, pad)), a, np.zeros((88, pad))], axis=1)
    if a.shape[1] <= roll.shape[1]:
        return float("nan"), 0.0
    tot = sum(fftconvolve(a[k], roll[k][::-1], mode="valid") for k in range(88) if roll[k].any())
    energy = fftconvolve((a ** 2).sum(0), np.ones(roll.shape[1]), mode="valid")
    norm = np.sqrt(np.clip(energy, 0, None))
    s = tot / (norm + 1e-3 * norm.max())          # windows over the silent padding stay near 0
    k = int(np.argmax(s))
    rest = s.copy()
    rest[max(0, k - int(fs)):k + int(fs)] = -np.inf
    return k / fs - pad_s, float(1 - rest.max() / s[k]) if s[k] > 0 else 0.0


ATTACK_FS = 1000
ATTACK_HP_HZ = 2000          # hammer attacks stand out above the sustained partials
ATTACK_GAP_S = 0.15          # only notes this far from the previous onset
ATTACK_WIN_S = (-0.08, 0.12)
ATTACK_MIN_NOTES = 5


def attack_refine(wav: Path, onsets: np.ndarray, coarse: float) -> tuple[float, int]:
    """Fine placement at note level: (audio time of the first note, notes used).

    With the MIDI placed at `coarse`, each isolated note's attack is the
    steepest rise of the high-passed short-time energy (1 ms) within
    ATTACK_WIN_S of it; the median lag over the notes is added to `coarse`.
    """
    import librosa
    from scipy.signal import butter, sosfilt
    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    y = sosfilt(butter(4, ATTACK_HP_HZ, "hp", fs=sr, output="sos"), y)
    hop = sr // ATTACK_FS
    e = np.sqrt(np.convolve(y ** 2, np.ones(hop * 3) / (hop * 3), "same")[::hop] + 1e-12)
    rise = np.maximum(np.diff(np.log(e + 1e-6), prepend=0), 0)
    rel = np.sort(onsets) - np.min(onsets)
    rel = rel[np.r_[True, np.diff(rel) > ATTACK_GAP_S]]
    lo_w, hi_w = int(ATTACK_WIN_S[0] * ATTACK_FS), int(ATTACK_WIN_S[1] * ATTACK_FS)
    lags = []
    for t in rel:
        c = int(round((t + coarse) * ATTACK_FS))
        if c + lo_w < 0 or c + hi_w >= len(rise):
            continue
        lags.append((int(np.argmax(rise[c + lo_w:c + hi_w])) + lo_w) / ATTACK_FS)
    if len(lags) < ATTACK_MIN_NOTES:
        return coarse, len(lags)
    return coarse + float(np.median(lags)), len(lags)
