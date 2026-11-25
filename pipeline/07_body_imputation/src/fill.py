"""
Stage 07: training-free gap filling of body markers.

Every method fills each NaN run of a marker that has an observed frame on both sides; runs touching the
start or end of the piece are left as NaN. Coordinates are mm, one row per frame.

  linear  straight line between the two boundary frames.
  spline  cubic spline through up to SPLINE_CTX observed frames on each side.
  rigid   rigid-body fill: markers on the same body segment keep their distances, so the segment's
          rotation + translation between a boundary frame and the gap frame (Kabsch on >= 3 donor markers
          observed in both) carries the target marker from the boundary frame into the gap. The forward
          (from the frame before) and backward (from the frame after) estimates are blended linearly.
          Frames with fewer than 3 donors fall back to `fallback`.
  pca     marker intercorrelation (after Gloersen & Federolf 2016): a PCA of the target and its nearest
          markers over the piece's frames where all of them are observed; in a gap frame the target is the
          PCA reconstruction that best matches the observed neighbours (ridge least squares). The error at
          the two boundary frames is removed by a linear correction.
  rule    per gap, the method that was most accurate for its body part and length in the held-out comparison
          (s06; chosen on performers C0_N4/N6/N8, checked on C0_N5/N7): spline for gaps shorter than
          SHORT_GAP_SEC and for the head; spline for the trunk from LONG_GAP_SEC on; rigid with a pca
          fallback for the arm; rigid for the rest. The comparison measured gaps of 0.1 / 0.5 / 1 / 2 s;
          SHORT_GAP_SEC and LONG_GAP_SEC are the geometric means of neighbouring lengths.

`fill(df, method, max_gap=None)` returns a copy of df with the runs filled; with max_gap (frames), longer runs
are left as NaN.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from src.utils import BODY_MARKERS

SPLINE_CTX = 24
PCA_NEIGHBOURS = 12
PCA_VAR = 0.995
PCA_MAX_FRAMES = 20000
RIDGE = 1e-3
FPS = 120
SHORT_GAP_SEC = (0.1 * 0.5) ** 0.5
LONG_GAP_SEC = (0.5 * 1.0) ** 0.5

# Rigid segments (sets of markers that move together). A marker may sit on two segments (e.g. a knee
# marker on thigh and shank); per gap frame the segment with the most usable donors is used.
_SIDE = {
    "upper_arm": ["{s}UPARM1", "{s}UPARM2", "{s}ELBL", "ACRO{s}"],
    "forearm": ["{s}LOWARM1", "{s}LOWARM2", "{s}ELBL", "{s}LAW"],
    "thigh": ["{s}THIGH1", "{s}THIGH2", "{s}KNEEL", "HIP{s}"],
    "shank": ["{s}SHANK1", "{s}SHANK2", "{s}KNEEL", "{s}ANKLEL", "{s}ANKLEM"],
    "foot": ["{s}HEEL", "{s}ANKLEL", "{s}ANKLEM", "{s}TOEL", "{s}TOEM", "{s}TOEF"],
}
SEGMENTS = [
    ["HEADT", "HEADA", "HEADR", "HEADL"],
    ["ACROR", "ACROL", "TRUNKAR", "TRUNKAL", "TRUNKPR", "TRUNKPL"],
    ["ASISR", "ASISL", "PSISR", "PSISL", "HIPR", "HIPL"],
] + [[m.format(s=s) for m in seg] for s in "RL" for seg in _SIDE.values()]


def runs(missing):
    """(start, end_exclusive) of NaN runs with an observed frame on both sides."""
    out, n, i = [], len(missing), 0
    while i < n:
        if missing[i]:
            j = i
            while j < n and missing[j]:
                j += 1
            if i > 0 and j < n:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def xyz(df, m):
    return df[[f"{m}_x", f"{m}_y", f"{m}_z"]].to_numpy(dtype=float)


def _linear(P, s, e):
    t = (np.arange(1, e - s + 1) / (e - s + 1))[:, None]
    return P[s - 1] + (P[e] - P[s - 1]) * t


def _spline(P, s, e):
    obs = ~np.isnan(P[:, 0])
    left = np.flatnonzero(obs[:s])[-SPLINE_CTX:]
    right = s + np.flatnonzero(obs[s:])[:SPLINE_CTX] if e < len(P) else np.array([], int)
    idx = np.concatenate([left, right])
    if len(idx) < 4:
        return _linear(P, s, e)
    return CubicSpline(idx, P[idx], axis=0)(np.arange(s, e))


def _kabsch(A, B):
    """R, t with R @ a + t ~= b for rows of A, B (n x 3)."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - R @ ca


def _rigid(df, m, P, s, e, fallback):
    segs = [[d for d in seg if d != m and f"{d}_x" in df.columns] for seg in SEGMENTS if m in seg]
    donors = {d: xyz(df, d) for seg in segs for d in seg}
    out = fallback.copy()
    for k, t in enumerate(range(s, e)):
        est, w = [], []
        for ref, wt in ((s - 1, 1 - (k + 1) / (e - s + 1)), (e, (k + 1) / (e - s + 1))):
            best = None
            for seg in segs:
                ok = [d for d in seg if not np.isnan(donors[d][t, 0]) and not np.isnan(donors[d][ref, 0])]
                if len(ok) >= 3 and (best is None or len(ok) > len(best)):
                    best = ok
            if best:
                R, tr = _kabsch(np.array([donors[d][ref] for d in best]), np.array([donors[d][t] for d in best]))
                est.append(R @ P[ref] + tr); w.append(wt)
        if est:
            w = np.array(w) / max(sum(w), 1e-9) if len(est) > 1 else np.ones(1)
            out[k] = (np.array(est) * w[:, None]).sum(0)
    return out


def _neighbours(df, m, k=PCA_NEIGHBOURS):
    P = xyz(df, m)
    d = {}
    for o in BODY_MARKERS + ["RLAW", "LLAW"]:
        if o == m or f"{o}_x" not in df.columns:
            continue
        diff = np.linalg.norm(xyz(df, o) - P, axis=1)
        if np.isfinite(diff).sum() > 0:
            d[o] = np.nanmedian(diff)
    return sorted(d, key=d.get)[:k]


def _pca(df, m, P, s, e, fallback, cache):
    if m not in cache:
        nb = _neighbours(df, m)
        X = np.concatenate([P] + [xyz(df, o) for o in nb], axis=1)
        full = ~np.isnan(X).any(1)
        if full.sum() < 50:
            cache[m] = None
        else:
            Xf = X[full]
            if len(Xf) > PCA_MAX_FRAMES:
                Xf = Xf[np.linspace(0, len(Xf) - 1, PCA_MAX_FRAMES).astype(int)]
            mu = Xf.mean(0)
            _, S, Vt = np.linalg.svd(Xf - mu, full_matrices=False)
            r = int(np.searchsorted(np.cumsum(S ** 2) / (S ** 2).sum(), PCA_VAR)) + 1
            cache[m] = (X, mu, Vt[:r].T)
    if cache[m] is None:
        return fallback
    X, mu, V = cache[m]
    out = fallback.copy()

    def recon(t):
        obs = ~np.isnan(X[t]); obs[:3] = False
        if obs.sum() < 6:
            return None
        A = V[obs]
        z = np.linalg.solve(A.T @ A + RIDGE * np.eye(A.shape[1]), A.T @ (X[t, obs] - mu[obs]))
        return mu[:3] + V[:3] @ z

    ends = [recon(s - 1), recon(e)]
    if ends[0] is None or ends[1] is None:
        return fallback
    off0, off1 = P[s - 1] - ends[0], P[e] - ends[1]
    for k, t in enumerate(range(s, e)):
        r = recon(t)
        if r is not None:
            a = (k + 1) / (e - s + 1)
            out[k] = r + (1 - a) * off0 + a * off1
    return out


def rule_method(m: str, n: int, fps: float = FPS) -> str:
    """The method `rule` uses for a gap of n frames of marker m."""
    sec = n / fps
    if sec < SHORT_GAP_SEC or m.startswith("HEAD"):
        return "spline"
    if m.startswith(("ACRO", "TRUNK")):
        return "spline" if sec >= LONG_GAP_SEC else "rigid"
    if any(k in m for k in ("UPARM", "LOWARM", "ELBL")):
        return "rigid_pca"
    return "rigid"


def fill(df: pd.DataFrame, method: str, markers=None, max_gap: int | None = None) -> pd.DataFrame:
    """Fill the interior NaN runs of each body marker with `method` (linear|spline|rigid|pca|rigid_pca|rule)."""
    out = df.copy()
    cache = {}
    for m in markers or BODY_MARKERS:
        if f"{m}_x" not in df.columns:
            continue
        P = xyz(df, m)
        for s, e in runs(np.isnan(P[:, 0])):
            if max_gap is not None and e - s > max_gap:
                continue
            meth = rule_method(m, e - s) if method == "rule" else method
            v = _one(df, m, P, s, e, meth, cache)
            out.loc[s:e - 1, [f"{m}_x", f"{m}_y", f"{m}_z"]] = v
    return out


def _one(df, m, P, s, e, method, cache):
    if method == "linear":
        return _linear(P, s, e)
    if method == "spline":
        return _spline(P, s, e)
    if method == "pca":
        return _pca(df, m, P, s, e, _linear(P, s, e), cache)
    if method == "rigid":
        return _rigid(df, m, P, s, e, _linear(P, s, e))
    if method == "rigid_pca":
        return _rigid(df, m, P, s, e, _pca(df, m, P, s, e, _linear(P, s, e), cache))
    raise ValueError(method)
