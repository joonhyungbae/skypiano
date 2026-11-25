"""
Stage 07 Step 00: rule-based outlier removal for body markers, before imputation.

Optical capture sometimes gives a body marker's label to another point: a static reflection near the
pedals (it then sits still at the same place, often for a whole piece), a marker of the other leg, or a
one-frame ghost. Those values are set to NaN here, so the imputation model never learns from them and
fills them like any other gap. Rules, applied in order; each counts only cells not flagged by an earlier one:

  0. Frozen: a marker at exactly the same position for >= FROZEN_RUN consecutive frames. A real marker always
     shows measurement noise; an exact repeat is a static reflection or a held value. Removed before the
     rigid-pair baselines are measured, so a ghost that holds a label for most of a piece cannot become them.
  1. Rigid pairs (per performer): markers whose distance hardly changes while playing (head, pelvis,
     hip-knee-ankle, heel-ankle-toe, acromion-elbow, elbow-wrist). Baseline = per-performer median over all
     pieces; a frame deviates when |d - median| > max(TOL_MM, 5 * 1.4826 * MAD). Blame per deviating pair:
     the end in more deviating pairs that frame; both ends on a tie.
  2. Segment membership: a cluster marker (thigh, shank, upper arm, forearm) must lie on its segment:
     d(m, A) + d(m, B) - d(A, B) <= SEG_MM, A/B = the segment's end markers (both valid this frame).
     A foot marker (heel, toes, medial ankle) must be within FOOT_MM of the lateral ankle.
  3. Short excursions: a run of <= SPIKE_RUN frames that starts and ends with an inter-frame jump of more
     than JUMP_MM (3.6 m/s at 120 fps) is removed.

Input:  Data/Pipeline/04_piece_split/{dataset}/{piece_prefix}_body.csv
Output: Data/Pipeline/07_body_imputation/Clean/{dataset}/{piece_prefix}_body.csv   (outliers -> NaN)
        Data/Pipeline/07_body_imputation/Clean/{dataset}/{piece_prefix}_outlier.csv (1 = removed here)
        Data/Pipeline/07_body_imputation/Clean/_baselines.json, _summary.csv (cells removed per rule)
"""
import functools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.performers import PERFORMER_MAP, performer_id, get_piece_body_path, CLEAN_ROOT

print = functools.partial(print, flush=True)

TOL_MM, SEG_MM, FOOT_MM = 25.0, 150.0, 300.0
JUMP_MM, SPIKE_RUN, FROZEN_RUN = 30.0, 24, 12

HEAD = ["HEADT", "HEADA", "HEADR", "HEADL"]
RIGID = [(a, b) for i, a in enumerate(HEAD) for b in HEAD[i + 1:]]
RIGID += [("ASISR", "ASISL"), ("PSISR", "PSISL"), ("ASISR", "PSISR"), ("ASISL", "PSISL"),
          ("ASISR", "PSISL"), ("ASISL", "PSISR")]
SEGMENT, FOOT = {}, {}
for s in "RL":
    RIGID += [(f"HIP{s}", f"ASIS{s}"), (f"HIP{s}", f"{s}KNEEL"), (f"{s}KNEEL", f"{s}ANKLEL"),
              (f"{s}ANKLEL", f"{s}HEEL"), (f"{s}HEEL", f"{s}TOEL"),
              (f"ACRO{s}", f"{s}ELBL"), (f"{s}ELBL", f"{s}LAW")]
    for k in "12":
        SEGMENT[f"{s}THIGH{k}"] = (f"HIP{s}", f"{s}KNEEL")
        SEGMENT[f"{s}SHANK{k}"] = (f"{s}KNEEL", f"{s}ANKLEL")
        SEGMENT[f"{s}UPARM{k}"] = (f"ACRO{s}", f"{s}ELBL")
        SEGMENT[f"{s}LOWARM{k}"] = (f"{s}ELBL", f"{s}LAW")
    for m in ("HEEL", "ANKLEM", "TOEL", "TOEM", "TOEF"):
        FOOT[f"{s}{m}"] = f"{s}ANKLEL"


def markers(df):
    return {c[:-2]: df[[c, c[:-2] + "_y", c[:-2] + "_z"]].to_numpy(float) for c in df.columns if c.endswith("_x")}


def frozen(p):
    """Frames inside a run of >= FROZEN_RUN identical, observed positions."""
    same = np.all(p[1:] == p[:-1], axis=1)            # NaN never compares equal
    out = np.zeros(len(p), bool)
    k = np.flatnonzero(np.diff(np.r_[0, same.astype(int), 0]))
    for s, e in zip(k[::2], k[1::2]):                  # same[s:e] True -> frames s..e identical
        if e - s + 1 >= FROZEN_RUN:
            out[s:e + 1] = True
    return out


def load(ds, pp):
    """Piece markers with frozen runs already NaN, and the frozen masks."""
    df = pd.read_csv(get_piece_body_path(ds, pp))
    P = markers(df)
    fz = {m: frozen(p) for m, p in P.items()}
    for m, f in fz.items():
        P[m][f] = np.nan
    return df, P, fz


def dist(a, b):
    return np.linalg.norm(a - b, axis=1)


def baselines(pieces):
    """Per-pair median and tolerance over all of a performer's pieces."""
    acc = {p: [] for p in RIGID}
    for ds, pp in pieces:
        P = load(ds, pp)[1]
        for a, b in RIGID:
            if a in P and b in P:
                d = dist(P[a], P[b])
                acc[(a, b)].append(d[~np.isnan(d)][::4])
    out = {}
    for p, v in acc.items():
        v = np.concatenate(v) if v else np.empty(0)
        if len(v) < 1000:
            continue
        med = float(np.median(v)); mad = float(np.median(np.abs(v - med)))
        out[p] = (med, max(TOL_MM, 5 * 1.4826 * mad))
    return out


def clean(P, base, fz):
    """Return {marker: bool array (n,)} of removed frames (frozen included), and per-rule counts.
    P has the frozen frames already NaN."""
    n = len(next(iter(P.values())))
    bad = {m: fz[m].copy() for m in P}
    counts = {"frozen": int(sum(f.sum() for f in fz.values())), "rigid": 0, "segment": 0, "foot": 0, "spike": 0}

    def flag(m, mask, rule):
        new = mask & ~bad[m] & ~np.isnan(P[m][:, 0])
        bad[m] |= new; counts[rule] += int(new.sum())

    # 1. rigid pairs
    dev = {}
    for (a, b), (med, tol) in base.items():
        if a in P and b in P:
            with np.errstate(invalid="ignore"):
                dev[(a, b)] = np.abs(dist(P[a], P[b]) - med) > tol
    score = {m: np.zeros(n, int) for m in P}
    for (a, b), d in dev.items():
        score[a] += d; score[b] += d
    for (a, b), d in dev.items():
        flag(a, d & (score[a] >= score[b]), "rigid")
        flag(b, d & (score[b] >= score[a]), "rigid")

    def valid(m):
        return ~np.isnan(P[m][:, 0]) & ~bad[m]

    # 2. segment membership and foot radius
    with np.errstate(invalid="ignore"):
        for m, (a, b) in SEGMENT.items():
            if m in P and a in P and b in P:
                ok = valid(a) & valid(b)
                ex = dist(P[m], P[a]) + dist(P[m], P[b]) - dist(P[a], P[b])
                flag(m, ok & (ex > SEG_MM), "segment")
        for m, a in FOOT.items():
            if m in P and a in P:
                flag(m, valid(a) & (dist(P[m], P[a]) > FOOT_MM), "foot")

    # 3. short excursions between two jumps
    for m, p in P.items():
        idx = np.flatnonzero(valid(m))
        if len(idx) < 3:
            continue
        jump = np.flatnonzero(dist(p[idx[1:]], p[idx[:-1]]) > JUMP_MM)   # jump between idx[j] and idx[j+1]
        for j0, j1 in zip(jump[:-1], jump[1:]):
            s, e = idx[j0 + 1], idx[j1]                                   # frames s..e sit between the jumps
            if e - s + 1 <= SPIKE_RUN:
                r = np.zeros(n, bool); r[s:e + 1] = True
                flag(m, r, "spike")
    return bad, counts


def run(only=None, skip=None, force=False):
    print("Stage 07 s00: remove body-marker outliers")
    print(f"  FROZEN_RUN={FROZEN_RUN} TOL={TOL_MM} SEG={SEG_MM} FOOT={FOOT_MM} JUMP={JUMP_MM} SPIKE_RUN={SPIKE_RUN}")
    CLEAN_ROOT.mkdir(parents=True, exist_ok=True)
    bl_path, rows = CLEAN_ROOT / "_baselines.json", []
    all_bl = json.loads(bl_path.read_text()) if bl_path.exists() else {}
    for (c, n), pieces in sorted(PERFORMER_MAP.items()):
        pid = performer_id(c, n)
        if (only and pid not in only) or (skip and pid in skip):
            continue
        base = baselines(pieces)
        all_bl[pid] = {f"{a}-{b}": [round(m, 2), round(t, 2)] for (a, b), (m, t) in base.items()}
        for ds, pp in tqdm(pieces, desc=f"  {pid}"):
            out = CLEAN_ROOT / ds / f"{pp}_body.csv"
            df, P, fz = load(ds, pp)
            bad, counts = clean(P, base, fz)
            obs = {m: int((~np.isnan(P[m][:, 0])).sum() + fz[m].sum()) for m in P}
            rows.append({"pid": pid, "dataset": ds, "piece": pp, "frames": len(df),
                         "observed_cells": sum(obs.values()), **counts,
                         "markers_hit": " ".join(f"{m}:{int(b.sum())}" for m, b in bad.items() if b.sum())})
            if out.exists() and not force:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            mask = pd.DataFrame(0, index=df.index, columns=df.columns, dtype=np.int8)
            for m, b in bad.items():
                cols = [m + "_x", m + "_y", m + "_z"]
                df.loc[b, cols] = np.nan
                mask.loc[b, cols] = 1
            df.to_csv(out, index=False)
            mask.to_csv(CLEAN_ROOT / ds / f"{pp}_outlier.csv", index=False)
    bl_path.write_text(json.dumps(all_bl, indent=1))
    s = pd.DataFrame(rows)
    s.to_csv(CLEAN_ROOT / "_summary.csv", index=False)
    if len(s):
        g = s.groupby("pid")[["observed_cells", "frozen", "rigid", "segment", "foot", "spike"]].sum()
        g["removed_%"] = (g[["frozen", "rigid", "segment", "foot", "spike"]].sum(axis=1) / g["observed_cells"] * 100).round(2)
        print(g.to_string())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--skip", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run(only=a.only, skip=a.skip, force=a.force)
