"""
Additional audio similarity signals (librosa-based) for ensemble.

Four complementary signals:
  A. CENS (Chroma Energy Normalized Statistics) — upgrade of basic CQT-chroma,
     designed for cross-version classical music similarity (Müller & Ewert).
     More robust to tempo/expression than raw chroma.
  B. 2DFTM chroma — 2D Fourier transform magnitude of chroma SSM. Key- and
     tempo-invariant representation (Bertin-Mahieux 2012). Standard in cover
     song ID benchmarks.
  C. SSM (Self-Similarity Matrix) shape — each piece has a structural signature
     via its chroma-based SSM. Compare SSMs via their eigenvalue decomposition
     (global structure, not local content).
  D. MFCC mean-var — timbral similarity signal (complementary to harmonic).

Output: Data/Metadata/240828/set1_advanced_scores.json with per-method
pair similarities.
"""
from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np


PIECES_MANIFEST = Path("Data/Pipeline/04_piece_split/240828/_pieces.json")
PIECES_DIR = Path("Data/Pipeline/04_piece_split/240828")
OUT_PATH = Path("Data/Metadata/240828/set1_advanced_scores.json")

TASK_SET = {2: "Set1", 9: "Free"}

SR = 22050
HOP = 11025   # 0.5 Hz


def cens_feature(y: np.ndarray) -> np.ndarray:
    """CENS chroma: normalized smooth chroma. Returns (12, T)."""
    chroma = librosa.feature.chroma_cens(y=y, sr=SR, hop_length=HOP, n_chroma=12)
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return chroma / norms


def twodftm_feature(chroma: np.ndarray, window_frames: int = 64) -> np.ndarray:
    """2D Fourier Transform Magnitude of chroma. Returns (144,) flat vector (key-invariant).

    Take a sliding 64-frame window (~32s at 2Hz), 2D-FFT, average magnitude across windows.
    Uses only first 12×12 = 144 coefficients as summary (Bertin-Mahieux style).
    """
    if chroma.shape[1] < 4:
        return np.zeros(144, dtype=np.float32)
    accum = np.zeros((12, 12), dtype=np.float32)
    step = max(1, window_frames // 2)
    n_windows = 0
    for start in range(0, max(1, chroma.shape[1] - window_frames + 1), step):
        w = chroma[:, start:start + window_frames]
        if w.shape[1] < 4:
            continue
        # 2D FFT magnitude
        F = np.abs(np.fft.fft2(w))
        # Keep only first 12×12 (low-frequency)
        F_small = F[:12, :12]
        accum += F_small
        n_windows += 1
    if n_windows == 0:
        F = np.abs(np.fft.fft2(chroma))
        accum = F[:12, :min(12, F.shape[1])]
        if accum.shape != (12, 12):
            padded = np.zeros((12, 12), dtype=np.float32)
            padded[:accum.shape[0], :accum.shape[1]] = accum
            accum = padded
    else:
        accum /= n_windows
    v = accum.flatten()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def ssm_eigen_feature(chroma: np.ndarray, n_eigen: int = 32) -> np.ndarray:
    """Self-similarity matrix top-K eigenvalues as structural signature.

    Returns sorted top-N eigenvalues of chroma SSM, L2-normalized.
    """
    # Downsample to manageable size (max 128 frames)
    T = chroma.shape[1]
    if T > 128:
        idx = np.linspace(0, T - 1, 128).astype(int)
        c = chroma[:, idx]
    else:
        c = chroma
    ssm = c.T @ c    # cosine similarity since L2-normed
    # Symmetric matrix, use eigvalsh
    evals = np.linalg.eigvalsh(ssm)
    evals = np.sort(evals)[::-1]   # descending
    top = evals[:n_eigen]
    if len(top) < n_eigen:
        top = np.pad(top, (0, n_eigen - len(top)))
    norm = np.linalg.norm(top)
    return (top / norm).astype(np.float32) if norm > 0 else top.astype(np.float32)


def mfcc_feature(y: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
    """Timbral signature: mean and std of MFCCs concatenated (2*n_mfcc dim)."""
    mfcc = librosa.feature.mfcc(y=y, sr=SR, hop_length=HOP, n_mfcc=n_mfcc)
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    norm = np.linalg.norm(feat)
    return (feat / norm).astype(np.float32) if norm > 0 else feat.astype(np.float32)


def collect_features():
    manifest = json.loads(PIECES_MANIFEST.read_text(encoding="utf-8"))
    records = []
    print("Extracting CENS + 2DFTM + SSM + MFCC per piece...")
    import time
    t0 = time.time()
    n_done = 0
    for t in manifest:
        task = TASK_SET.get(t["task"])
        if task is None:
            continue
        sub = t["raw_stem"][-4:-2]
        trial = t["raw_stem"][-4:]
        for p in t["pieces"]:
            wav = PIECES_DIR / f"{t['trial_stem']}_p{p['piece_idx']}.wav"
            if not wav.is_file():
                continue
            try:
                y, _ = librosa.load(str(wav), sr=SR, mono=True)
                cens = cens_feature(y)
                twodftm = twodftm_feature(cens)
                ssm_e = ssm_eigen_feature(cens)
                mfcc = mfcc_feature(y)
            except Exception as e:
                print(f"  [skip {wav.name}] {e}")
                continue
            records.append({
                "piece_key": f"sub{sub}_{trial}_p{p['piece_idx']}",
                "sub": f"sub{sub}",
                "trial": trial,
                "piece_idx": p["piece_idx"],
                "task": task,
                "duration_sec": round(p["end_sec"] - p["start_sec"], 2),
                "cens": cens.astype(np.float32),
                "twodftm": twodftm,
                "ssm_eigen": ssm_e,
                "mfcc": mfcc,
            })
            n_done += 1
            if n_done % 10 == 0:
                print(f"  {n_done} pieces processed [{time.time()-t0:.1f}s]")
    print(f"  total {n_done} pieces in {time.time()-t0:.1f}s")
    return records


def pair_sim_from_vector(rec_a: dict, rec_b: dict, key: str) -> float:
    """Cosine sim between two fixed-dim vectors (L2-normed)."""
    return float(np.clip(np.dot(rec_a[key], rec_b[key]), -1.0, 1.0))


def cens_dtw_sim(rec_a: dict, rec_b: dict) -> float:
    """Subsequence DTW over CENS chroma frames."""
    x, y = rec_a["cens"], rec_b["cens"]
    if x.shape[1] <= y.shape[1]:
        a, b = x, y
    else:
        a, b = y, x
    La, Lb = a.shape[1], b.shape[1]
    if La < 2:
        return 0.0
    cost = 1.0 - a.T @ b
    D = np.full((La, Lb), np.inf, dtype=np.float32)
    D[0] = cost[0]
    for i in range(1, La):
        for j in range(Lb):
            best = D[i - 1, j]
            if j > 0:
                if D[i - 1, j - 1] < best: best = D[i - 1, j - 1]
                if D[i, j - 1] < best: best = D[i, j - 1]
            D[i, j] = cost[i, j] + best
    min_cost = float(D[-1].min())
    return 1.0 - min_cost / La


def main():
    records = collect_features()

    print("\nComputing all-pairs similarities (4 methods)...")
    import time
    t0 = time.time()
    n = len(records)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                "a": records[i]["piece_key"],
                "b": records[j]["piece_key"],
                "cens_dtw": round(cens_dtw_sim(records[i], records[j]), 4),
                "twodftm": round(pair_sim_from_vector(records[i], records[j], "twodftm"), 4),
                "ssm_eigen": round(pair_sim_from_vector(records[i], records[j], "ssm_eigen"), 4),
                "mfcc": round(pair_sim_from_vector(records[i], records[j], "mfcc"), 4),
            })
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{n} rows  [{time.time()-t0:.1f}s]")

    # Distributions
    print("\nSimilarity distributions:")
    for method in ["cens_dtw", "twodftm", "ssm_eigen", "mfcc"]:
        vals = sorted(p[method] for p in pairs)
        pct = lambda q: vals[int(len(vals) * q)]
        print(f"  {method:<10}  p99={pct(0.01):.3f}  p95={pct(0.05):.3f}  "
              f"p50={pct(0.5):.3f}  p5={pct(0.95):.3f}")

    # Save
    out = {
        "_comment": "Advanced audio similarity signals: CENS-DTW, 2DFTM, SSM eigenvalue, MFCC.",
        "pieces": [
            {k: v for k, v in r.items() if k not in {"cens", "twodftm", "ssm_eigen", "mfcc"}}
            for r in records
        ],
        "pair_similarities": pairs,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
