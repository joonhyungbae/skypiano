"""
Stage 07 Step 09: released body motion as Visual3D joints (Stage 09 s06), unfilled and gap-filled.

Per piece (or 240523 whole take):
  {pp}_v3d.csv               joints of the released body markers (Stage 04 piece split / take), not cleaned
                             and not filled: a joint is empty in a frame where any of its markers is missing.
  {pp}_v3d_suspect_mask.csv  1 = that joint value uses a marker that s00 removed as an outlier.
  {pp}_v3d_imputed.csv       joints of the cleaned markers (s00) after gap filling: method `rule` of
                             src/fill.py with the hand-capture arm markers (wrist, elbow, shoulder of the
                             piece's hand CSV) as extra Kabsch donors of the arm segments (weight HAND_W),
                             gaps up to 5 s for lower-body and 1 s for upper-body markers (MAX_GAP_SEC); body, wrist (RLAW/LLAW) and hand (RHAND/LHAND)
                             markers are filled. Longer gaps and gaps at the piece ends stay empty.
  {pp}_v3d_imputed_mask.csv  1 = joint value present only because of the filling (empty in the joints of
                             the cleaned, unfilled markers).
All four have the same rows as the released body CSV and the columns of Stage 09 s06 (mm).

Held-out joint error of this fill (s08, 240828, 59 pieces; gaps 0.1 / 0.5 / 1 / 2 s): 0.10 / 1.41 / 3.67 /
9.6 mm, against 0.49 / 3.86 / 8.01 / 15.1 mm for linear interpolation of the markers.

Input:  Data/Pipeline/07_body_imputation/Clean/{ds}/{pp}_body.csv, {pp}_outlier.csv   (s00)
        the released body CSV and the hand CSV of the piece (Stage 04 / 12_release trials), or with
        --raw-root: {raw_root}/{ds}/{pp}_body.csv and {pp}_hand.csv
Output: Data/Pipeline/07_body_imputation/ReleaseV3D/{ds}/
"""
import functools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "09_visual3d" / "steps"))
import src.fill as F
from src.utils import BODY_MARKERS
from s06_joints import joints

print = functools.partial(print, flush=True)

CLEAN = Path("Data/Pipeline/07_body_imputation/Clean")
OUT = Path("Data/Pipeline/07_body_imputation/ReleaseV3D")
# Longest gap filled, per body part: the longest held-out gap length (s08: 0.5 / 1 / 2 / 3 / 5 s) at which
# every joint built from that part's markers kept a mean error of about 10 mm or less. Lower body: pelvis,
# hip, knee, ankle and foot stay at 2-11 mm up to 5 s (not tested beyond). Upper body: head, neck, shoulder,
# elbow and wrist are 2-13 mm at 1 s but 8-32 mm at 2 s.
LOWER = ("ASIS", "PSIS", "HIP", "THIGH", "KNEE", "SHANK", "ANKLE", "HEEL", "TOE")
MAX_GAP_SEC = {"lower": 5.0, "upper": 1.0}
HAND_W = 0.1
EXTRA = ["RLAW", "LLAW", "RHAND", "LHAND"]


def sources(ds, pp, raw_root):
    if raw_root:
        return Path(raw_root) / ds / f"{pp}_body.csv", Path(raw_root) / ds / f"{pp}_hand.csv"
    from src.performers import get_piece_body_path
    b = get_piece_body_path(ds, pp)
    return b, (b.with_name("hand.csv") if b.name == "body.csv" else b.with_name(f"{pp}_hand.csv"))


def one(job):
    ds, pp, raw_root = job
    raw_p, hand_p = sources(ds, pp, raw_root)
    raw = pd.read_csv(raw_p)
    clean = pd.read_csv(CLEAN / ds / f"{pp}_body.csv")
    outl = pd.read_csv(CLEAN / ds / f"{pp}_outlier.csv")
    n = len(raw)
    assert len(clean) == n == len(outl), (pp, n, len(clean), len(outl))
    markers = [m for m in BODY_MARKERS + EXTRA if f"{m}_x" in clean.columns]
    work = clean
    if hand_p.exists():
        h = pd.read_csv(hand_p)
        cols = [f"{m}_{a}" for m in sorted(F.HAND_DONORS) for a in "xyz" if f"{m}_x" in h.columns]
        h = h[cols].iloc[:n].reset_index(drop=True).reindex(range(n))
        work = pd.concat([clean, h], axis=1)
    F.HAND_W = HAND_W
    lower = [m for m in markers if any(k in m for k in LOWER)]
    upper = [m for m in markers if m not in lower]
    filled = F.fill(work, "rule", lower, max_gap=int(round(MAX_GAP_SEC["lower"] * F.FPS)))
    filled = F.fill(filled, "rule", upper, max_gap=int(round(MAX_GAP_SEC["upper"] * F.FPS)))

    j_raw, j_clean, j_fill = joints(raw), joints(clean), joints(filled)
    # suspect: joints whose markers include an outlier sample (outlier mask has the body CSV's columns)
    sus = joints(raw.where(outl.reindex(columns=raw.columns).fillna(0).to_numpy() == 0))
    suspect = (j_raw.notna() & sus.isna()).astype(np.int8)
    imp_mask = (j_clean.isna() & j_fill.notna()).astype(np.int8)
    d = OUT / ds
    d.mkdir(parents=True, exist_ok=True)
    j_raw.to_csv(d / f"{pp}_v3d.csv", index=False, float_format="%.3f")
    suspect.to_csv(d / f"{pp}_v3d_suspect_mask.csv", index=False)
    j_fill.to_csv(d / f"{pp}_v3d_imputed.csv", index=False, float_format="%.3f")
    imp_mask.to_csv(d / f"{pp}_v3d_imputed_mask.csv", index=False)
    cells = j_raw.size
    return ds, int(j_raw.isna().to_numpy().sum()), int(suspect.to_numpy().sum()), int(imp_mask.to_numpy().sum()), \
        int(j_fill.isna().to_numpy().sum()), cells, hand_p.exists()


def takes(datasets):
    """Whole takes of the piece-level sessions (Data/Pipeline/12_release/trials/{ds}/{trial}/body.csv) are
    released unfilled: only {trial}_v3d.csv (their pieces carry the filled joints)."""
    root = Path("Data/Pipeline/12_release/trials")
    n = 0
    for ds in datasets:
        for b in sorted((root / ds).glob("*/body.csv")) if (root / ds).is_dir() else []:
            out = OUT / ds / f"{b.parent.name}_v3d.csv"
            if (OUT / ds / f"{b.parent.name}_v3d_imputed.csv").exists():
                continue                      # a take that s00 cleaned (240523) keeps its filled set
            out.parent.mkdir(parents=True, exist_ok=True)
            joints(pd.read_csv(b)).to_csv(out, index=False, float_format="%.3f")
            n += 1
    print(f"Stage 07 s09 --takes: {n} whole takes, unfilled joints")


def run(datasets, raw_root, workers, force):
    jobs = []
    for ds in datasets:
        for f in sorted((CLEAN / ds).glob("*_body.csv")):
            pp = f.name[:-len("_body.csv")]
            if force or not (OUT / ds / f"{pp}_v3d_imputed.csv").exists():
                jobs.append((ds, pp, raw_root))
    print(f"Stage 07 s09: {len(jobs)} pieces, max gap {MAX_GAP_SEC} s, hand donors weight {HAND_W}, workers {workers}")
    tot = {}
    with ProcessPoolExecutor(workers) as ex:
        for k, (ds, miss, sus, imp, left, cells, hand) in enumerate(ex.map(one, jobs), 1):
            t = tot.setdefault(ds, np.zeros(7))
            t += [1, miss, sus, imp, left, cells, hand]
            if k % 50 == 0:
                print(f"  {k}/{len(jobs)}")
    for ds, (n, miss, sus, imp, left, cells, hand) in sorted(tot.items()):
        print(f"  {ds}: {int(n)} pieces ({int(hand)} with hand donors); unfilled joints empty {miss / cells * 100:.2f}%, "
              f"suspect {sus / cells * 100:.2f}%, filled {imp / cells * 100:.2f}%, still empty {left / cells * 100:.2f}%")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["220819", "221112", "240523", "240828"])
    ap.add_argument("--raw-root", default=None, help="flat {ds}/{pp}_body.csv + _hand.csv instead of Stage 04 / trials")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--takes", action="store_true", help="only the unfilled joints of the whole takes")
    a = ap.parse_args()
    if a.takes:
        takes(a.datasets)
    else:
        run(a.datasets, a.raw_root, a.workers, a.force)
