"""
Stage 02 body alignment: temporal sync + spatial alignment (body → hand frame).

Adapted from an internal precompute-alignment helper for
the stage2 pipeline output.
"""
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# 6 bilateral landmark pairs: (hand_joint, body_marker)
LANDMARK_PAIRS = [
    ('LSh',    'ACROL'),   # shoulder
    ('RSh',    'ACROR'),
    ('LElbow', 'LELBL'),   # elbow
    ('RElbow', 'RELBL'),
    ('LWrist', 'LLAW'),    # wrist
    ('RWrist', 'RLAW'),
]

# Per-landmark base weights.
# Shoulders carry the highest weight: shoulder residuals are the most reliable
# indicator of overall alignment quality, while distal joints (elbows, wrists)
# can drift due to soft-tissue artifact and are weighted accordingly.
LANDMARK_WEIGHTS = [2.0, 2.0, 1.5, 1.5, 1.0, 1.0]


# ── Temporal sync ──────────────────────────────────────

def _speed_from_df(df: pd.DataFrame, joints: list) -> np.ndarray:
    """Compute per-frame speed as avg displacement of given joints."""
    coords = []
    for j in joints:
        cols = [f'{j}_x', f'{j}_y', f'{j}_z']
        if all(c in df.columns for c in cols):
            coords.append(df[cols].values)
    if not coords:
        return np.zeros(len(df))

    avg = np.nanmean(coords, axis=0)  # (N, 3)
    diff = np.diff(avg, axis=0)
    speed = np.sqrt(np.nansum(diff ** 2, axis=1))
    return np.concatenate([[0], speed])


def _smooth(arr: np.ndarray, w: int = 5) -> np.ndarray:
    kernel = np.ones(2 * w + 1) / (2 * w + 1)
    return np.convolve(arr, kernel, mode='same')


def _normalize(arr: np.ndarray) -> np.ndarray:
    m = np.nanmean(arr)
    s = np.nanstd(arr)
    if s < 1e-12:
        return np.zeros_like(arr)
    return (arr - m) / s


def find_body_offset(
    hand_df: pd.DataFrame, hand_fps: float,
    body_df: pd.DataFrame, body_fps: float,
) -> int:
    """Find frame offset via wrist speed cross-correlation (vectorized).

    Returns body_offset: hand_frame - body_offset = temporal sync point.
    """
    h_speed = _smooth(_speed_from_df(hand_df, ['LWrist', 'RWrist']))
    b_speed = _smooth(_speed_from_df(body_df, ['LLAW', 'RLAW']))

    h_norm = _normalize(h_speed)
    b_norm = _normalize(b_speed)

    # Resample body speed to hand FPS for direct cross-correlation
    n_body = len(b_norm)
    n_hand = len(h_norm)
    body_time = np.arange(n_body) / max(body_fps, 1)
    hand_time = np.arange(int(body_time[-1] * hand_fps) + 1) / max(hand_fps, 1)
    b_resampled = np.interp(hand_time, body_time, b_norm)

    # Use numpy correlate (full mode) — much faster than Python loop
    corr = np.correlate(b_resampled, h_norm, mode='full')
    # corr[k] = sum(b_resampled[k+i] * h_norm[i]) for valid i
    # offset = k - (len(h_norm) - 1)
    best_k = np.argmax(corr)
    best_offset = best_k - (n_hand - 1)

    # Negate: our convention is hand_frame - body_offset maps to body
    return -best_offset


def find_body_offset_candidates(
    hand_df: pd.DataFrame, hand_fps: float,
    body_df: pd.DataFrame, body_fps: float,
    top_k: int = 5,
    min_gap_frames: int = 240,
    grid_step: int = 300,
) -> list:
    """Return a candidate list of body_offset values (in hand-frame units) for
    the caller to score by spatial residual.

    Combines two sources:
      1. Top-K cross-correlation peaks (with non-maximum suppression).
      2. A coarse uniform grid covering all plausible body_offset values where
         body overlaps hand by at least 30% of the body's length.

    Cross-correlation alone is unreliable when wrist activity is repetitive or
    body is much shorter than hand — its strongest peak can be off by hundreds
    of frames from the spatially-correct alignment. Adding a coarse grid as
    fallback ensures the spatially-optimal region is always probed.
    """
    h_speed = _smooth(_speed_from_df(hand_df, ['LWrist', 'RWrist']))
    b_speed = _smooth(_speed_from_df(body_df, ['LLAW', 'RLAW']))

    h_norm = _normalize(h_speed)
    b_norm = _normalize(b_speed)

    n_body = len(b_norm)
    n_hand = len(h_norm)
    body_time = np.arange(n_body) / max(body_fps, 1)
    hand_time = np.arange(int(body_time[-1] * hand_fps) + 1) / max(hand_fps, 1)
    b_resampled = np.interp(hand_time, body_time, b_norm)

    corr = np.correlate(b_resampled, h_norm, mode='full')

    # 1) Top-K cross-correlation peaks with NMS
    xcorr_peaks: list = []
    masked = corr.copy()
    for _ in range(top_k):
        if not np.any(np.isfinite(masked)):
            break
        k = int(np.argmax(masked))
        if not np.isfinite(masked[k]) or masked[k] == -np.inf:
            break
        offset = k - (n_hand - 1)
        xcorr_peaks.append(-offset)  # convention negation
        lo = max(0, k - min_gap_frames)
        hi = min(len(masked), k + min_gap_frames + 1)
        masked[lo:hi] = -np.inf

    # 2) Coarse grid over plausible range
    n_body_in_hand = len(b_resampled)
    min_overlap = int(0.3 * n_body_in_hand)
    # body_offset ∈ [-(n_body_in_hand - min_overlap), n_hand - min_overlap]
    grid_lo = -(n_body_in_hand - min_overlap)
    grid_hi = n_hand - min_overlap
    grid = list(range(grid_lo, grid_hi + 1, grid_step))

    # Merge and dedupe (keep order: xcorr peaks first since they're more likely)
    seen = set()
    out: list = []
    for c in xcorr_peaks + grid:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# ── Spatial alignment (Horn's method) ──────────────────

def _map_hand_to_body_frame(hand_frame: int, body_offset: int, hand_fps: float, body_fps: float) -> int:
    t = (hand_frame - body_offset) / max(hand_fps, 1)
    return round(t * body_fps)


def _extract_point(df: pd.DataFrame, frame: int, joint: str) -> Optional[np.ndarray]:
    cols = [f'{joint}_x', f'{joint}_y', f'{joint}_z']
    if not all(c in df.columns for c in cols):
        return None
    vals = df.iloc[frame][cols].values.astype(float)
    if np.any(np.isnan(vals)) or np.all(vals == 0):
        return None
    return vals


def _extract_landmark_arrays(df: pd.DataFrame, landmarks: list) -> list:
    """Pre-extract each landmark's xyz columns into a dense numpy array.

    Returns a list parallel to `landmarks`, each entry is either an (n, 3)
    ndarray or None if the columns don't exist. NaN rows are kept so callers
    can index by frame.
    """
    out = []
    for name in landmarks:
        cols = [f'{name}_x', f'{name}_y', f'{name}_z']
        if all(c in df.columns for c in cols):
            out.append(df[cols].to_numpy(dtype=np.float64, copy=False))
        else:
            out.append(None)
    return out


def fast_alignment_residual(
    hand_arrs: list, body_arrs: list, n_hand: int, n_body: int,
    hand_fps: float, body_fps: float, body_offset: int,
    n_samples: int = 12, window_size: int = 80,
) -> float:
    """Quick-screening variant of multi_window_alignment + residuals.

    Unlike the simple "pool all windows" approach (which hides temporal errors
    because the mean landmark position is insensitive to offset), this uses
    PER-WINDOW means as separate point observations. With `n_samples` windows
    and up to 6 landmarks, Horn's method fits (n_samples × 6) points. A wrong
    temporal offset shows up as per-window mismatches that inflate the
    residual, while a correct offset gives a tight fit across all windows.

    Returns weighted mean residual in mm, or inf if the fit failed. This is
    ~15-20x faster than the full multi_window_alignment because it uses
    pre-extracted numpy arrays and a smaller window size.
    """
    if n_hand == 0:
        return float('inf')

    src_pts = []
    tgt_pts = []
    weights = []

    half = window_size // 2
    for s in range(n_samples):
        ref = int(n_hand * (s + 0.5) / n_samples)
        h_lo = max(0, ref - half)
        h_hi = min(n_hand, ref + half + 1)

        for pi, (harr, barr) in enumerate(zip(hand_arrs, body_arrs)):
            if harr is None or barr is None:
                continue
            h_sum = np.zeros(3)
            b_sum = np.zeros(3)
            cnt = 0
            for f in range(h_lo, h_hi):
                hp = harr[f]
                if np.isnan(hp).any():
                    continue
                bf = round((f - body_offset) / max(hand_fps, 1) * body_fps)
                if bf < 0 or bf >= n_body:
                    continue
                bp = barr[bf]
                if np.isnan(bp).any():
                    continue
                h_sum += hp
                b_sum += bp
                cnt += 1
            if cnt > 0:
                src_pts.append(b_sum / cnt)
                tgt_pts.append(h_sum / cnt)
                weights.append(LANDMARK_WEIGHTS[pi])

    if len(src_pts) < 3:
        return float('inf')

    src_arr = np.array(src_pts)
    tgt_arr = np.array(tgt_pts)
    w_arr = np.array(weights)

    result = horn_alignment(src_arr, tgt_arr, w_arr)
    if result is None:
        return float('inf')
    R, t = result

    aligned = (R @ src_arr.T).T + t
    diffs = np.linalg.norm(aligned - tgt_arr, axis=1)
    return float((diffs * w_arr).sum() / w_arr.sum())


def multi_window_alignment(
    hand_df: pd.DataFrame, hand_fps: float,
    body_df: pd.DataFrame, body_fps: float,
    body_offset: int,
    n_samples: int = 20,
    window_size: int = 120,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Compute rotation + translation aligning body → hand coordinate frame.

    Samples multiple windows, averages landmarks, then applies weighted Horn.

    Returns:
        (R, t) where R is (3,3) rotation matrix and t is (3,) translation.
        body_in_hand_frame = R @ body_point + t
    """
    n_hand = len(hand_df)
    if n_hand == 0:
        return None

    # Accumulate per-landmark averages across windows
    all_hand = [[] for _ in LANDMARK_PAIRS]
    all_body = [[] for _ in LANDMARK_PAIRS]

    for s in range(n_samples):
        ref = int(n_hand * (s + 0.5) / n_samples)
        half = window_size // 2
        h_start = max(0, ref - half)
        h_end = min(n_hand - 1, ref + half)

        for pi, (h_name, b_name) in enumerate(LANDMARK_PAIRS):
            h_acc = np.zeros(3)
            b_acc = np.zeros(3)
            hc = bc = 0

            for f in range(h_start, h_end + 1):
                hp = _extract_point(hand_df, f, h_name)
                bf = _map_hand_to_body_frame(f, body_offset, hand_fps, body_fps)
                bp = _extract_point(body_df, bf, b_name) if 0 <= bf < len(body_df) else None

                if hp is not None:
                    h_acc += hp; hc += 1
                if bp is not None:
                    b_acc += bp; bc += 1

            if hc > 0 and bc > 0:
                all_hand[pi].append(h_acc / hc)
                all_body[pi].append(b_acc / bc)

    # Average across all windows per landmark
    src_pts = []  # body
    tgt_pts = []  # hand
    weights = []

    for pi in range(len(LANDMARK_PAIRS)):
        if not all_body[pi]:
            continue
        src_pts.append(np.mean(all_body[pi], axis=0))
        tgt_pts.append(np.mean(all_hand[pi], axis=0))
        weights.append(LANDMARK_WEIGHTS[pi] * len(all_body[pi]))

    if len(src_pts) < 3:
        return None

    return horn_alignment(
        np.array(src_pts),
        np.array(tgt_pts),
        np.array(weights),
    )


def horn_alignment(
    source: np.ndarray,  # (N, 3) body points
    target: np.ndarray,  # (N, 3) hand points
    weights: np.ndarray,  # (N,)
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Weighted Horn's method for rigid body alignment.

    Returns (R, t) such that target ≈ R @ source + t.
    """
    if len(source) < 3:
        return None

    w = weights / weights.sum()
    sc = (w[:, None] * source).sum(axis=0)
    tc = (w[:, None] * target).sum(axis=0)

    sp = source - sc
    tp = target - tc

    # Cross-covariance matrix
    H = np.zeros((3, 3))
    for i in range(len(sp)):
        H += weights[i] * np.outer(sp[i], tp[i])

    # SVD solution
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])  # correct reflection
    R = Vt.T @ D @ U.T

    t = tc - R @ sc
    return R, t


def per_landmark_residuals(
    hand_df: pd.DataFrame, hand_fps: float,
    body_df_aligned: pd.DataFrame, body_fps: float,
    body_offset: int,
    n_samples: int = 30,
) -> Dict[str, float]:
    """Mean |hand - body_aligned| per landmark pair, sampled across the trial.

    Call AFTER apply_body_transform so body_df_aligned is in hand frame.
    Returns {pair_name: mean_distance_mm, ..., '_max': overall max}.
    """
    n_hand = len(hand_df)
    if n_hand == 0:
        return {}

    result: Dict[str, list] = {f'{h}-{b}': [] for h, b in LANDMARK_PAIRS}

    for s in range(n_samples):
        hf = int(n_hand * (s + 0.5) / n_samples)
        bf = _map_hand_to_body_frame(hf, body_offset, hand_fps, body_fps)
        if not (0 <= bf < len(body_df_aligned)):
            continue
        for h_name, b_name in LANDMARK_PAIRS:
            hp = _extract_point(hand_df, hf, h_name)
            bp = _extract_point(body_df_aligned, bf, b_name)
            if hp is None or bp is None:
                continue
            result[f'{h_name}-{b_name}'].append(float(np.linalg.norm(hp - bp)))

    out: Dict[str, float] = {}
    for k, vs in result.items():
        out[k] = float(np.mean(vs)) if vs else float('nan')
    finite = [v for v in out.values() if np.isfinite(v)]
    out['_max'] = max(finite) if finite else float('nan')
    out['_mean'] = float(np.mean(finite)) if finite else float('nan')
    return out


def apply_body_transform(body_df: pd.DataFrame, R: np.ndarray, t: np.ndarray) -> pd.DataFrame:
    """Apply rigid transform to all body marker columns."""
    df = body_df.copy()
    markers = set()
    for col in df.columns:
        if col.endswith('_x'):
            markers.add(col[:-2])

    for m in markers:
        cols = [f'{m}_x', f'{m}_y', f'{m}_z']
        if not all(c in df.columns for c in cols):
            continue
        coords = df[cols].values  # (N, 3)
        valid = ~np.isnan(coords).any(axis=1)
        if valid.any():
            transformed = (R @ coords[valid].T).T + t
            df.loc[df.index[valid], cols] = transformed

    return df
