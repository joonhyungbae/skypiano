"""
Stage 02 body, step 2: refine the session body->hand transform, accounting for
where each system's landmark markers sit.

The registration pairs hand-system markers (shoulder, elbow, wrist) with body-
system markers (acromion, lateral elbow, wrist styloid) that are not on the
same spot: on 2024 data the hand wrist marker is ~40 mm from the body styloid
marker. Treating them as one point bends the fitted transform. The offset of a
hand marker from its body marker is fixed in the frame of the body segment
both are attached to (forearm, upper arm, trunk), which turns with the pose.
Pooling the frames of a session, this step solves for one rigid correction
(dR, dt) together with the six fixed offsets d_i:

    h_i(f) ~ dR (b_i(f) + Q_i(f) d_i) + dt

h_i hand landmark, b_i body landmark, Q_i the body segment's orientation built
from markers every dataset has (forearm: LAW, elbow, forearm cluster; upper
arm: acromion, elbow, upper-arm cluster; trunk: acromions, posterior trunk).
The offsets are only used to estimate dR, dt; the body data is moved by the
rigid correction alone, so the body markers keep their measured geometry.

The fit is checked on held-out trials (fit on every other trial, residuals on
the rest), both transforms with their offsets fitted on the fit half; the
correction is applied only if the refined transform's held-out median
residual is lower than the current one's. Results go to Aligned/{dataset}/_transforms.json under "refine".

Usage (from the repository root):
  python pipeline/02_spatial/body/steps/s02_refine_session_transform.py --dataset 240828
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HAND = Path("Data/Pipeline/02_spatial/hand/Aligned")
BODY = Path("Data/Pipeline/02_spatial/body/Aligned")
NORM = Path("Data/Pipeline/01_ingest/Normalized")
STYLOID_MID = {"220819", "221112"}      # hand wrist = LAW/MEW midpoint (02 hand `wrist_center`)
STEP = 12                               # 10 Hz sampling of the 120 fps frames
LAMBDA = 50.0                           # shrinkage of the offsets (frames-equivalent)
ITERS = 30


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _frame(a, b):
    """Orthonormal frames [N,3,3] (columns e1, e2, e3) from two direction fields."""
    e1 = _unit(a)
    e2 = _unit(b - (b * e1).sum(-1, keepdims=True) * e1)
    return np.stack([e1, e2, np.cross(e1, e2)], axis=-1)


def _pairs(ds: str, stem: str):
    h = pd.read_csv(HAND / ds / f"{stem}.csv")
    b = pd.read_csv(BODY / ds / f"{stem}.csv")
    n = min(len(h), len(b))
    g = lambda df, m: df[[f"{m}_{k}" for k in "xyz"]].to_numpy(float)[:n:STEP]
    out = []
    for s in "LR":
        law, elb, acr = g(b, f"{s}LAW"), g(b, f"{s}ELBL"), g(b, f"ACRO{s}")
        fore = _frame(law - elb, g(b, f"{s}LOWARM1") - g(b, f"{s}LOWARM2"))
        upper = _frame(acr - elb, g(b, f"{s}UPARM1") - g(b, f"{s}UPARM2"))
        acl, acrr = g(b, "ACROL"), g(b, "ACROR")
        trunk = _frame(acrr - acl, (acl + acrr) / 2 - (g(b, "TRUNKPL") + g(b, "TRUNKPR")) / 2)
        bw = (law + g(b, f"{s}MEW")) / 2 if ds in STYLOID_MID else law
        out += [(f"{s}Wrist", g(h, f"{s}Wrist"), bw, fore),
                (f"{s}Elbow", g(h, f"{s}Elbow"), elb, upper),
                (f"{s}Sh", g(h, f"{s}Sh"), acr, trunk)]
    return out


def _stack(trials):
    acc = {}
    for t in trials:
        for name, H, B, Q in t:
            ok = np.isfinite(H).all(1) & np.isfinite(B).all(1) & np.isfinite(Q).all((1, 2))
            a = acc.setdefault(name, [[], [], []])
            a[0].append(H[ok]); a[1].append(B[ok]); a[2].append(Q[ok])
    return {k: tuple(np.concatenate(x) for x in v) for k, v in acc.items() if sum(len(x) for x in v[0]) > 50}


def _horn(A, B):
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def offsets_for(S, R, t):
    """Best fixed offsets for a given transform."""
    d = {}
    for k, (H, B, Q) in S.items():
        r = (R.T @ (H - t).T).T - B
        d[k] = np.einsum("nji,nj->ni", Q, r).sum(0) / (len(r) + LAMBDA)
    return d


def fit(S):
    """Alternate offsets (given R, t) and R, t (Horn on offset-corrected body points,
    every landmark weighted equally)."""
    R, t = np.eye(3), np.zeros(3)
    d = {k: np.zeros(3) for k in S}
    for _ in range(ITERS):
        A, Bm = [], []
        for k, (H, B, Q) in S.items():
            idx = np.linspace(0, len(H) - 1, min(len(H), 3000)).astype(int)
            A.append((B + np.einsum("nij,j->ni", Q, d[k]))[idx]); Bm.append(H[idx])
        R, t = _horn(np.concatenate(A), np.concatenate(Bm))
        d = offsets_for(S, R, t)
    return R, t, d


def residuals(S, R, t, d=None):
    out = {}
    for k, (H, B, Q) in S.items():
        C = B + (np.einsum("nij,j->ni", Q, d[k]) if d else 0)
        out[k] = round(float(np.median(np.linalg.norm((R @ C.T).T + t - H, axis=1))), 1)
    return out


def _copies(ds: str, stem: str, record: dict) -> list[Path]:
    """The trial's body file and the alias copies the body aligner wrote for it
    (listed in _transforms.json; e.g. raw 2022Piano* / 240828subNNNN names)."""
    names = record.get("trials", {}).get(stem, {}).get("aliases") or [stem]
    if stem not in names:
        names = [stem] + names
    return [BODY / ds / f"{n}.csv" for n in names if (BODY / ds / f"{n}.csv").exists()]


def apply_rigid(path: Path, R, t):
    df = pd.read_csv(path)
    for c in [c[:-2] for c in df.columns if c.endswith("_x")]:
        cols = [f"{c}_x", f"{c}_y", f"{c}_z"]
        if all(x in df.columns for x in cols):
            df[cols] = (R @ df[cols].to_numpy(float).T).T + t
    df.to_csv(path, index=False)


def run(ds: str) -> None:
    rec_path = BODY / ds / "_transforms.json"
    record = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.exists() else {}
    record.setdefault("refine", {})
    sessions: dict[str, list[str]] = {}
    for mp in sorted((NORM / ds).glob("*/_mapping.json")):
        for t in json.loads(mp.read_text(encoding="utf-8"))["trials"]:
            if (BODY / ds / f"{t['trial_stem']}.csv").exists() and (HAND / ds / f"{t['trial_stem']}.csv").exists():
                sessions.setdefault(mp.parent.name, []).append(t)
    for sess, trials in sorted(sessions.items()):
        if record["refine"].get(sess, {}).get("applied"):
            print(f"  [{sess}] already refined; skip")
            continue
        data = []
        for t in trials:
            try:
                data.append((t, _pairs(ds, t["trial_stem"])))
            except KeyError as e:
                print(f"  [{sess}] {t['trial_stem']}: missing marker {e}")
        if len(data) < 4:
            print(f"  [{sess}] too few trials ({len(data)})")
            continue
        SA, SB = _stack([x for _, x in data[0::2]]), _stack([x for _, x in data[1::2]])
        I, Z = np.eye(3), np.zeros(3)
        # Both sides with offsets fitted on the fit half, so the comparison is
        # between transforms: the current one (identity) vs the refined one.
        ra, ta, da = fit(SA)
        d0 = offsets_for(SA, I, Z)
        before, after = residuals(SB, I, Z, d0), residuals(SB, ra, ta, da)
        raw = residuals(SB, I, Z, {k: Z for k in SB})          # as written now, offsets ignored
        R, t, d = fit(_stack([x for _, x in data]))          # final fit on every trial
        med_b, med_a = float(np.median(list(before.values()))), float(np.median(list(after.values())))
        ok = med_a < med_b
        ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        print(f"  [{sess}] held-out median residual {med_b:.1f} -> {med_a:.1f} mm; "
              f"correction {ang:.2f} deg / {np.linalg.norm(t):.1f} mm -> {'applied' if ok else 'not applied'}")
        if ok:
            for tr, _ in data:
                for f in _copies(ds, tr["trial_stem"], record):
                    apply_rigid(f, R, t)
        record["refine"][sess] = {
            "applied": ok, "rotation_deg": round(ang, 3), "translation_mm": round(float(np.linalg.norm(t)), 2),
            "R": R.round(8).tolist(), "t": t.round(4).tolist(),
            "held_out_residual_mm": {"current_transform": before, "refined_transform": after,
                                     "current_without_offsets": raw},
            "offsets_in_segment_frame_mm": {k: v.round(1).tolist() for k, v in d.items()},
        }
        rec_path.write_text(json.dumps(record, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    run(ap.parse_args().dataset)


if __name__ == "__main__":
    main()
