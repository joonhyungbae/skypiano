"""
Stage 07 Step 05: held-out accuracy of the body imputation.

Per performer, every 10th piece (sorted by name) is held out; a model with the s02 configuration is
trained on the other pieces only. In each held-out piece (cleaned by s00), observed stretches of every
marker are hidden on purpose (gaps of 0.1 / 0.5 / 1 / 2 s, one gap length per pass, up to 3 gaps per
marker, never overlapping a real gap), imputed the way s03 does it, and compared with the hidden truth.
Linear interpolation over the same gaps is the baseline.

Output: Data/Pipeline/07_body_imputation/Eval/_errors.csv  (one row per hidden gap)
        Data/Pipeline/07_body_imputation/Eval/_summary.csv (mean 3D error in mm per performer x gap length)
        Data/Pipeline/07_body_imputation/Eval/Models/{pid}/saits_body.pypots
"""
import functools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import body_joint_cols
from src.performers import PERFORMER_MAP, performer_id, get_clean_body_path
from steps.s01_slice_data import compute_features, slice_trial, WINDOW, STEP

print = functools.partial(print, flush=True)

EVAL_ROOT = Path("Data/Pipeline/07_body_imputation/Eval")
GAPS = {"0.1s": 12, "0.5s": 60, "1s": 120, "2s": 240}
PER_MARKER = 3
MARKERS = [c[:-2] for c in body_joint_cols[::3]]


def saits(n_features, n_samples, device, path=None, epochs=1000, patience=30):
    from pypots.imputation import SAITS
    return SAITS(n_steps=WINDOW, n_features=n_features, n_layers=2, d_model=256, n_heads=4, d_k=64, d_v=64,
                 d_ffn=128, dropout=0.1, epochs=epochs, patience=min(patience, epochs), batch_size=min(32, n_samples),
                 saving_path=path, device=device)


def impute(model, df):
    """s03's procedure: non-overlapping 3 s windows over the whole piece; positions back in mm."""
    n = len(df)
    comb = compute_features(df, body_joint_cols)
    pad = (WINDOW - n % WINDOW) % WINDOW
    if pad:
        comb = np.concatenate([comb, np.full((pad, comb.shape[1]), np.nan, np.float32)])
    out = model.impute({"X": comb.reshape(-1, WINDOW, comb.shape[1])}).reshape(-1, comb.shape[1])[:n]
    return out[:, :len(body_joint_cols)] * 100.0


def place_gaps(obs, length, rng):
    """Up to PER_MARKER non-overlapping gap starts inside fully observed stretches, with 1 observed frame
    on each side (so linear interpolation has both ends)."""
    ok = np.convolve(obs.astype(int), np.ones(length + 2, int), "valid") == length + 2   # start s-1 .. s+length
    starts, taken = [], np.zeros(len(obs), bool)
    for s in rng.permutation(np.flatnonzero(ok)):
        if len(starts) == PER_MARKER:
            break
        if not taken[s:s + length + 2].any():
            starts.append(s + 1); taken[s:s + length + 2] = True
    return starts


def run(device, only=None, epochs=1000):
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA not available; never on CPU.")
    rng = np.random.default_rng(0)
    rows = []
    for (c, n), pieces in sorted(PERFORMER_MAP.items()):
        pid = performer_id(c, n)
        if only and pid not in only:
            continue
        pieces = sorted(pieces, key=lambda x: x[1])
        test = pieces[::10]
        train = [p for p in pieces if p not in test]
        mpath = EVAL_ROOT / "Models" / pid / "saits_body.pypots"
        if mpath.exists():
            n_feat = len(body_joint_cols) * 4
            model = saits(n_feat, 32, device); model.load(str(mpath))
        else:
            sl = []
            for ds, pp in train:
                df = pd.read_csv(get_clean_body_path(ds, pp))
                if sum(col in df.columns for col in body_joint_cols) >= len(body_joint_cols) // 2:
                    sl += slice_trial(compute_features(df, body_joint_cols), WINDOW, STEP)
            X = np.array(sl, np.float32)
            print(f"  {pid}: train {X.shape} on {len(train)} pieces, test {len(test)} pieces")
            model = saits(X.shape[2], len(X), device, str(EVAL_ROOT / "Models" / pid / "saits_body"), epochs=epochs)
            model.fit({"X": X})
            mpath.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(mpath), overwrite=True)
        for ds, pp in test:
            df = pd.read_csv(get_clean_body_path(ds, pp))
            for gname, L in GAPS.items():
                hid, plan = df.copy(), []
                for m in MARKERS:
                    if m + "_x" not in df.columns:
                        continue
                    obs = df[m + "_x"].notna().to_numpy()
                    for s in place_gaps(obs, L, rng):
                        plan.append((m, s))
                        hid.loc[s:s + L - 1, [m + "_x", m + "_y", m + "_z"]] = np.nan
                if not plan:
                    continue
                pred = impute(model, hid)
                for m, s in plan:
                    k = body_joint_cols.index(m + "_x")
                    true = df[[m + "_x", m + "_y", m + "_z"]].to_numpy()[s:s + L]
                    p = pred[s:s + L, k:k + 3]
                    a, b = df[[m + "_x", m + "_y", m + "_z"]].to_numpy()[[s - 1, s + L]]
                    t = (np.arange(1, L + 1) / (L + 1))[:, None]
                    lin = a + (b - a) * t
                    rows.append({"pid": pid, "piece": pp, "gap": gname, "marker": m,
                                 "saits_mm": float(np.linalg.norm(p - true, axis=1).mean()),
                                 "linear_mm": float(np.linalg.norm(lin - true, axis=1).mean())})
            print(f"    {pp}: {sum(r['piece'] == pp for r in rows)} gaps")
        del model
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    e = pd.DataFrame(rows)
    e.to_csv(EVAL_ROOT / "_errors.csv", index=False)
    s = e.groupby(["pid", "gap"])[["saits_mm", "linear_mm"]].mean().round(1).unstack("gap")
    s.to_csv(EVAL_ROOT / "_summary.csv")
    print(s.to_string())
    print(e.groupby("gap")[["saits_mm", "linear_mm"]].mean().round(1).to_string())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--epochs", type=int, default=1000)
    a = ap.parse_args()
    run(a.device, a.only, a.epochs)
