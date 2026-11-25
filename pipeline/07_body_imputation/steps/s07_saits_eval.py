"""
Stage 07 Step 07: SAITS variants on the s06 held-out gaps (GPU only).

Reads the hidden inputs s06 wrote (Eval/hidden/{piece}_{gap}.csv.gz) and writes predictions in mm to
Eval/extra/{variant}/{piece}_{gap}.npz, which s06 then scores next to the training-free methods.

  saits       the s05 models (Eval/Models/{pid}), s05's procedure: raw/100 coordinates + residual +
              velocity + acceleration, non-overlapping 3 s windows.
  saits_bc    the same, with a boundary correction: each gap is widened by one frame on each side before
              imputation, and the error the model makes on those two (known) frames is removed linearly
              across the gap, so the fill meets the observed trajectory at both ends.
  saits2_bc   a model trained on normalised coordinates only (per window, the body centroid is removed;
              then each coordinate is standardised with the training mean / std), half-overlapping
              windows averaged, plus the boundary correction. Trained on the pieces s05 trained on
              (every performer piece except the held-out ones), saved to Eval/ModelsV2/{pid}.
"""
import functools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import body_joint_cols
from src.fill import runs

print = functools.partial(print, flush=True)

ROOT = Path("Data/Pipeline/07_body_imputation")
EVAL = ROOT / "Eval"
WINDOW, HALF = 360, 180
NF = len(body_joint_cols)


def widen(X):
    """Hide one more frame on each side of every interior gap; returns the widened array and the runs."""
    W, spans = X.copy(), []
    for j in range(0, NF, 3):
        for s, e in runs(np.isnan(X[:, j])):
            spans.append((j, s, e))
            W[s - 1, j:j + 3] = np.nan
            W[e, j:j + 3] = np.nan
    return W, spans


def correct(X, P, spans):
    """Remove the model's error at the two boundary frames linearly across each gap; observed frames
    are copied back from X."""
    out = np.where(np.isnan(X), P, X)
    for j, s, e in spans:
        if s - 1 < 0 or e >= len(X):
            continue
        o0, o1 = X[s - 1, j:j + 3] - P[s - 1, j:j + 3], X[e, j:j + 3] - P[e, j:j + 3]
        if np.isnan(o0).any() or np.isnan(o1).any():
            continue
        a = (np.arange(1, e - s + 1) / (e - s + 1))[:, None]
        out[s:e, j:j + 3] = P[s:e, j:j + 3] + (1 - a) * o0 + a * o1
    return out


# ---------- s05 models ----------
def compute_features(df, joint_cols):
    """Copy of s01's features (raw/100, residual, velocity, acceleration); s01 needs the piece manifests."""
    f = np.full((len(df), len(joint_cols)), np.nan, dtype=np.float32)
    for i, col in enumerate(joint_cols):
        if col in df.columns:
            f[:, i] = df[col].values.astype(np.float32) / 100.0
    v = np.zeros_like(f); v[1:] = f[1:] - f[:-1]
    a = np.zeros_like(f); a[1:] = v[1:] - v[:-1]
    return np.concatenate([f, f - np.nanmean(f, axis=0, keepdims=True), v, a], axis=1)


def model_v1(device, pid):
    from pypots.imputation import SAITS
    m = SAITS(n_steps=WINDOW, n_features=NF * 4, n_layers=2, d_model=256, n_heads=4, d_k=64, d_v=64, d_ffn=128,
              dropout=0.1, epochs=1, batch_size=32, device=device)
    m.load(str(EVAL / "Models" / pid / "saits_body.pypots"))
    return m


def predict_v1(model, X):
    """X: frames x NF in mm (NaN = missing). Returns the model's values (observed frames included)."""
    df = pd.DataFrame(X, columns=body_joint_cols)
    n = len(df)
    comb = compute_features(df, body_joint_cols)
    pad = (WINDOW - n % WINDOW) % WINDOW
    if pad:
        comb = np.concatenate([comb, np.full((pad, comb.shape[1]), np.nan, np.float32)])
    out = model.predict({"X": comb.reshape(-1, WINDOW, comb.shape[1])})["imputation"]
    return out.reshape(-1, comb.shape[1])[:n, :NF] * 100.0


# ---------- normalised model ----------
def centroid(win):
    c = np.nanmean(win.reshape(len(win), -1, 3), axis=(0, 1))
    return np.where(np.isnan(c), 0.0, c)


def windows(X, stride):
    n = len(X)
    starts = list(range(0, max(n - WINDOW, 0) + 1, stride))
    if starts[-1] + WINDOW < n:
        starts.append(n - WINDOW)
    return starts


def norm_windows(X, stats, stride):
    mu, sd = stats
    pad = max(0, WINDOW - len(X))
    if pad:
        X = np.concatenate([X, np.full((pad, NF), np.nan)])
    st = windows(X, stride)
    W, C = [], []
    for s in st:
        w = X[s:s + WINDOW]
        c = centroid(w)
        W.append((w - np.tile(c, NF // 3) - mu) / sd); C.append(c)
    return np.array(W, np.float32), np.array(C), st, pad


def train_v2(device, pid, train_pieces, epochs, patience):
    from pypots.imputation import SAITS
    mdir = EVAL / "ModelsV2" / pid
    if (mdir / "saits2.pypots").exists():
        st = np.load(mdir / "stats.npz")
        m = SAITS(n_steps=WINDOW, n_features=NF, n_layers=2, d_model=256, n_heads=4, d_k=64, d_v=64, d_ffn=128,
                  dropout=0.1, epochs=1, batch_size=32, device=device)
        m.load(str(mdir / "saits2.pypots"))
        return m, (st["mu"], st["sd"])
    arrs = [pd.read_csv(p).reindex(columns=body_joint_cols).to_numpy(float) for p in train_pieces]
    cent = []
    for X in arrs:
        for s in windows(X, 120) if len(X) >= WINDOW else []:
            w = X[s:s + WINDOW]
            cent.append(w - np.tile(centroid(w), NF // 3))
    allc = np.concatenate(cent)
    mu, sd = np.nanmean(allc, 0), np.nanstd(allc, 0)
    mu, sd = np.where(np.isnan(mu), 0, mu), np.where(np.isnan(sd) | (sd < 1), 1, sd)
    D = np.concatenate([norm_windows(X, (mu, sd), 120)[0] for X in arrs if len(X) >= WINDOW])
    print(f"  {pid}: train {D.shape} from {len(arrs)} pieces")
    mdir.mkdir(parents=True, exist_ok=True)
    m = SAITS(n_steps=WINDOW, n_features=NF, n_layers=2, d_model=256, n_heads=4, d_k=64, d_v=64, d_ffn=128,
              dropout=0.1, epochs=epochs, patience=patience, batch_size=32, saving_path=str(mdir / "ckpt"),
              device=device)
    m.fit({"X": D})
    m.save(str(mdir / "saits2.pypots"), overwrite=True)
    np.savez(mdir / "stats.npz", mu=mu, sd=sd)
    return m, (mu, sd)


def predict_v2(model, stats, X):
    mu, sd = stats
    n = len(X)
    W, C, st, pad = norm_windows(X, stats, HALF)
    P = model.predict({"X": W})["imputation"]
    acc, cnt = np.zeros((n + pad, NF)), np.zeros((n + pad, 1))
    for k, s in enumerate(st):
        acc[s:s + WINDOW] += P[k] * sd + mu + np.tile(C[k], NF // 3)
        cnt[s:s + WINDOW] += 1
    return (acc / np.maximum(cnt, 1))[:n]


def run(device, variants, only, epochs, patience):
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA not available; never on CPU.")
    split = pd.read_csv(EVAL / "_errors.csv")[["pid", "piece"]].drop_duplicates()
    for pid, grp in split.groupby("pid"):
        if only and pid not in only:
            continue
        test = set(grp.piece)
        hidden = sorted((EVAL / "hidden").glob(f"240828_{pid}_*.csv.gz"))
        if "saits" in variants or "saits_bc" in variants:
            m1 = model_v1(device, pid)
            for h in hidden:
                X = pd.read_csv(h).reindex(columns=body_joint_cols).to_numpy(float)
                name = h.name[:-len(".csv.gz")]
                for v, bc in (("saits", False), ("saits_bc", True)):
                    if v not in variants:
                        continue
                    if bc:
                        Wd, spans = widen(X)
                        P = correct(X, predict_v1(m1, Wd), spans)
                    else:
                        P = np.where(np.isnan(X), predict_v1(m1, X), X)
                    (EVAL / "extra" / v).mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(EVAL / "extra" / v / f"{name}.npz", pred=P.astype(np.float32))
            print(f"  {pid}: s05 model, {len(hidden)} hidden inputs")
        if "saits2_bc" in variants:
            train = [p for p in sorted((ROOT / "Clean" / "240828").glob(f"240828_{pid}_*_body.csv"))
                     if p.name[:-len("_body.csv")] not in test]
            m2, stats = train_v2(device, pid, train, epochs, patience)
            for h in hidden:
                X = pd.read_csv(h).reindex(columns=body_joint_cols).to_numpy(float)
                Wd, spans = widen(X)
                P = correct(X, predict_v2(m2, stats, Wd), spans)
                (EVAL / "extra" / "saits2_bc").mkdir(parents=True, exist_ok=True)
                np.savez_compressed(EVAL / "extra" / "saits2_bc" / f"{h.name[:-len('.csv.gz')]}.npz",
                                    pred=P.astype(np.float32))
            print(f"  {pid}: normalised model, {len(hidden)} hidden inputs")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--variants", nargs="*", default=["saits", "saits_bc", "saits2_bc"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=20)
    a = ap.parse_args()
    run(a.device, a.variants, a.only, a.epochs, a.patience)
