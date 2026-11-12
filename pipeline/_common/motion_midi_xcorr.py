"""
Signal-based matching of raw motion takes to MIDI trials.

Duration alone cannot tell apart takes of similar length, and raw motion
filenames do not always follow the trial order. These helpers score a
(motion, trial) pair by cross-correlating activity envelopes:

  hand vs MIDI : summed downward fingertip speed (key-press activity;
                 distal-phalanx markers where a capture has no tip markers)
                 against the note-onset train.
  body vs hand : summed speed of the body wrist/hand markers against the
                 summed fingertip speed of the matched hand take.

Both envelopes are smoothed (SMOOTH_SEC) so the dominant structure is the
play / pause pattern between pieces, then resampled to GRID_HZ. The score is
the maximum normalized cross-correlation over lags whose overlap satisfies
the caller's overlap rule.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pretty_midi

GRID_HZ = 20
SMOOTH_SEC = 1.0

FINGERTIP_RE = re.compile(r"[LR][ITMRL][FD]")
# A finger's tip marker (..F) is used when it holds data in at least this share
# of frames; otherwise its distal-phalanx marker (..D), which moves down with
# the key press as well (221112 captures carry no tip markers).
TIP_MIN_COVER = 0.5
BODY_WRIST_MARKERS = ["RHAND", "LHAND", "RMDW", "LMDW", "RLAW", "LLAW", "RMEW", "LMEW"]


def _smooth(x: np.ndarray) -> np.ndarray:
    k = np.hanning(max(3, int(SMOOTH_SEC * GRID_HZ) * 2 + 1))
    return np.convolve(x, k / k.sum(), mode="same")


def _to_grid(act: np.ndarray, fps: float) -> np.ndarray:
    t = np.arange(len(act)) / fps
    if len(t) < 2:
        return np.zeros(1)
    return _smooth(np.interp(np.arange(0, t[-1], 1 / GRID_HZ), t, act))


def _read_optitrack_fingertips(path: Path) -> tuple[float, np.ndarray]:
    """Return (fps, fingertip positions [frames, markers, xyz]) from an OptiTrack CSV."""
    with open(path, encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f))
    fps = float(rows[0][rows[0].index("Export Frame Rate") + 1])
    names = next(r for r in rows[:8] if len(r) > 1 and r[1] == "Name")
    axes = next(r for r in rows[:10] if r and r[0] == "Frame")
    start = rows.index(axes) + 1
    cols: dict[str, dict[str, int]] = {}
    for i, (n, a) in enumerate(zip(names, axes)):
        short = n.split(":")[-1]
        if FINGERTIP_RE.fullmatch(short) and a in ("X", "Y", "Z"):
            cols.setdefault(short, {}).setdefault(a, i)
    names_ = [m for m in cols if len(cols[m]) == 3]
    idx = [[cols[m][a] for a in ("X", "Y", "Z")] for m in names_]
    body = rows[start:]
    arr = np.full((len(body), len(idx), 3), np.nan)
    for r_i, r in enumerate(body):
        for m_i, trip in enumerate(idx):
            for a_i, ci in enumerate(trip):
                if ci < len(r) and r[ci] != "":
                    arr[r_i, m_i, a_i] = float(r[ci])
    return fps, arr[:, _pick_fingers(names_, np.isfinite(arr[:, :, 1]).mean(axis=0)), :]


def _pick_fingers(names: list[str], cover) -> list[int]:
    """Per finger, the tip marker if it holds enough data, else the distal one."""
    pos = {n: i for i, n in enumerate(names)}
    out = []
    for finger in sorted({n[:2] for n in names}):
        tip, dist = pos.get(finger + "F"), pos.get(finger + "D")
        if tip is not None and cover[tip] >= TIP_MIN_COVER:
            out.append(tip)
        elif dist is not None:
            out.append(dist)
    return out


def hand_press_envelope(path: Path) -> np.ndarray:
    """Key-press activity: summed downward (-Y, OptiTrack is Y-up) fingertip speed."""
    fps, xyz = _read_optitrack_fingertips(path)
    vy = np.diff(xyz[:, :, 1], axis=0) * fps
    return _to_grid(np.nan_to_num(np.clip(-vy, 0, None)).sum(axis=1), fps)


def hand_speed_envelope(path: Path) -> np.ndarray:
    """Overall fingertip activity: summed 3D fingertip speed."""
    fps, xyz = _read_optitrack_fingertips(path)
    v = np.linalg.norm(np.diff(xyz, axis=0), axis=2) * fps
    return _to_grid(np.nan_to_num(v).sum(axis=1), fps)


def body_speed_envelope(path: Path) -> np.ndarray:
    """Summed 3D speed of the wrist/hand markers of a Qualisys TSV."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    header = {l.split("\t")[0]: l.split("\t")[1:] for l in lines[:12] if "\t" in l}
    fps = float(header["FREQUENCY"][0])
    i_cols = next(i for i, l in enumerate(lines) if l.startswith("Frame\t"))
    cols = lines[i_cols].split("\t")
    idx = [cols.index(f"{m} {a}") for m in BODY_WRIST_MARKERS if f"{m} X" in cols for a in "XYZ"]
    rows = [l.split("\t") for l in lines[i_cols + 1:] if l.strip()]
    arr = np.array([[float(r[i]) if i < len(r) and r[i] not in ("", "0.000") else np.nan
                     for i in idx] for r in rows])
    xyz = arr.reshape(len(arr), -1, 3)
    v = np.linalg.norm(np.diff(xyz, axis=0), axis=2) * fps
    return _to_grid(np.nan_to_num(v).sum(axis=1), fps)


def midi_onset_envelope(path: Path) -> np.ndarray:
    pm = pretty_midi.PrettyMIDI(str(path))
    x = np.zeros(int(pm.get_end_time() * GRID_HZ) + 1)
    for inst in pm.instruments:
        for n in inst.notes:
            x[min(int(n.start * GRID_HZ), len(x) - 1)] += 1
    return _smooth(x)


def best_ncc(a: np.ndarray, b: np.ndarray, min_overlap: float, relative_to: str = "longer") -> tuple[float, float]:
    """
    Max normalized cross-correlation of `a` vs `b` over lags.

    A lag is admissible when the overlap is at least `min_overlap` times the
    length of the longer signal (relative_to="longer") or of `b`
    (relative_to="b"). Returns (score, lag_sec of b relative to a); score is
    -inf when no lag is admissible.
    """
    if len(a) < 2 or len(b) < 2 or a.std() == 0 or b.std() == 0:
        return float("-inf"), 0.0
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    ref = max(len(a), len(b)) if relative_to == "longer" else len(b)
    full = np.correlate(a, b, mode="full")
    lags = np.arange(-(len(b) - 1), len(a))
    overlap = np.minimum(len(a), lags + len(b)) - np.maximum(0, lags)
    score = np.where(overlap >= min_overlap * ref, full / np.maximum(overlap, 1), -np.inf)
    i = int(np.argmax(score))
    return float(score[i]), float(lags[i] / GRID_HZ)


# ── Fine MIDI <-> motion lag from key-press activity ──

# The downward fingertip speed peaks shortly before the key sounds. Measured on
# SMPTE-synchronised 240828 (trials with a clear key-press signal).
PRESS_LEAD_S = -0.07


def fingertips_from_aligned_csv(path) -> np.ndarray:
    """Fingertip positions [frames, tips, xyz] from a 02_spatial hand CSV (Z up)."""
    import pandas as pd
    head = pd.read_csv(path, nrows=0).columns
    names = [f"{s}{f}{k}" for s in "LR" for f in "ITMRL" for k in "FD" if f"{s}{f}{k}_z" in head]
    df = pd.read_csv(path, usecols=[f"{t}_{a}" for t in names for a in "xyz"])
    xyz = np.stack([df[[f"{t}_x", f"{t}_y", f"{t}_z"]].to_numpy(float) for t in names], axis=1)
    return xyz[:, _pick_fingers(names, np.isfinite(xyz[:, :, 2]).mean(axis=0)), :]


def press_onset_curve(tips_xyz: np.ndarray, fps: float, note_onsets_s, search_s: float = 0.6):
    """
    NCC of key-press activity (summed downward fingertip speed) against MIDI
    note onsets over lags in [-search_s, +search_s]. A positive lag means the
    motion is late relative to the MIDI. Returns (lags_s, curve) or None.
    """
    from scipy.ndimage import gaussian_filter1d
    if len(tips_xyz) < 3:
        return None
    vz = np.diff(tips_xyz[:, :, 2], axis=0) * fps
    act = gaussian_filter1d(np.nan_to_num(np.clip(-vz, 0, None)).sum(axis=1), 3)
    m = np.zeros(len(act))
    for t in note_onsets_s:
        f = int(t * fps)
        if 0 <= f < len(m):
            m[f] += 1
    m = gaussian_filter1d(m, 3)
    if m.std() == 0 or act.std() == 0:
        return None
    a = (act - act.mean()) / act.std()
    b = (m - m.mean()) / m.std()
    L = int(search_s * fps)
    lags = np.arange(-L, L + 1)
    curve = np.array([np.dot(a[max(0, k):len(a) + min(0, k)], b[max(0, -k):len(b) - max(0, k)]) / (len(a) - abs(k))
                      for k in lags])
    return lags / fps, curve


# ── Note-level key-press <-> onset correlation (whole takes) ──

NOTE_FS = 60          # Hz
NOTE_SMOOTH = 2       # samples (gaussian sigma)


def onset_train(onsets) -> np.ndarray:
    """MIDI note onsets (s) as a lightly smoothed pulse train at NOTE_FS."""
    from scipy.ndimage import gaussian_filter1d
    onsets = np.asarray(onsets, float)
    x = np.zeros(int(onsets.max() * NOTE_FS) + 2 if len(onsets) else 1)
    np.add.at(x, (onsets * NOTE_FS).astype(int), 1)
    return gaussian_filter1d(x, NOTE_SMOOTH)


def press_train(tips_xyz: np.ndarray, fps: float, up_axis: int) -> np.ndarray:
    """Summed downward fingertip speed at NOTE_FS (up_axis: 1 = Y up, 2 = Z up)."""
    from scipy.ndimage import gaussian_filter1d
    v = np.nan_to_num(np.clip(-np.diff(tips_xyz[:, :, up_axis], axis=0) * fps, 0, None)).sum(axis=1)
    t = np.arange(len(v)) / fps
    return gaussian_filter1d(np.interp(np.arange(0, t[-1], 1 / NOTE_FS), t, v), NOTE_SMOOTH)


def slide_ncc(long: np.ndarray, short: np.ndarray) -> np.ndarray:
    """Pearson r of `short` at every full-overlap position of `long`."""
    from scipy.signal import fftconvolve
    n = len(short)
    if len(long) <= n or short.std() == 0:
        return np.full(1, -1.0)
    s = (short - short.mean()) / short.std()
    num = fftconvolve(long, s[::-1], mode="valid")
    c1 = np.concatenate(([0], np.cumsum(long)))
    c2 = np.concatenate(([0], np.cumsum(long ** 2)))
    k = np.arange(len(num))
    var = (c2[k + n] - c2[k]) / n - ((c1[k + n] - c1[k]) / n) ** 2
    return num / (n * np.sqrt(np.maximum(var, 1e-12)))


# ── Key position: is a fingertip over the played key at its onset? ──
# Keys are laid out along the keyboard end markers (left end -> right end) at
# 23.5 mm per white key; the position of A0 on that axis is taken where the
# fingertip-minus-key offsets gather at the offset being scored. Unlike the
# onset correlation this uses the pitch, so repetitive passages that line up
# in time at many offsets still line up on the keys at only one.

WHITE_KEY_MM = 23.5
_WHITE = {0, 2, 4, 5, 7, 9, 11}
KEY_MIN_NOTES = 8


def key_center(pitch: int) -> float:
    """Key centre along the keyboard axis, mm from the centre of A0."""
    wi = sum(1 for p in range(21, pitch) if p % 12 in _WHITE)
    return WHITE_KEY_MM * (wi if pitch % 12 in _WHITE else wi - 0.5)


def keyboard_lateral(tips: np.ndarray, left_end: np.ndarray, right_end: np.ndarray) -> np.ndarray:
    """Fingertip position along the keyboard axis (left end -> right end), mm."""
    u = (right_end - left_end) / np.linalg.norm(right_end - left_end)
    return (tips - left_end) @ u


def key_hit(lat: np.ndarray, fps: float, onsets: np.ndarray, keys: np.ndarray, shift: float) -> float:
    """Share of notes with a fingertip within half a white key of the played key
    (motion_t = midi_t + shift; `lat` from keyboard_lateral, `keys` from key_center)."""
    frames = np.round((onsets + shift) * fps).astype(int)
    m = (frames >= 0) & (frames < len(lat))
    if m.sum() < KEY_MIN_NOTES:
        return 0.0
    res = lat[frames[m]] - keys[m][:, None]
    vals = res[np.isfinite(res)]
    if len(vals) < KEY_MIN_NOTES:
        return 0.0
    h, e = np.histogram(vals, bins=np.arange(vals.min(), vals.max() + 4, 4.0))
    a0 = e[np.argmax(h)] + 2
    return float(np.mean(np.any(np.abs(res - a0) <= WHITE_KEY_MM / 2, axis=1)))


def optitrack_marker_median(path: Path, marker: str) -> np.ndarray | None:
    """Median position of one marker (matched case-insensitively) in a raw OptiTrack CSV."""
    with open(path, encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f))
    names = next(r for r in rows[:8] if len(r) > 1 and r[1] == "Name")
    axes = next(r for r in rows[:10] if r and r[0] == "Frame")
    idx = [i for i, (n, a) in enumerate(zip(names, axes)) if n.split(":")[-1].upper() == marker.upper()
           and a in ("X", "Y", "Z")][:3]
    if len(idx) < 3:
        return None
    body = rows[rows.index(axes) + 1:]
    v = np.array([[float(r[i]) if i < len(r) and r[i] != "" else np.nan for i in idx] for r in body])
    return np.nanmedian(v, axis=0)
