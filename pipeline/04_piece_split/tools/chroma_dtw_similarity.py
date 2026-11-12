"""
Phase 1 SoTA upgrade: Audio chroma + DTW cross-sub similarity for Set1/Free pieces.

Why chroma-DTW over hand-crafted fingerprint:
  - Classical piano performances of the same piece vary in tempo and expression.
    DTW explicitly aligns two time series at variable rates, unlike rigid
    offset-bin voting in our fingerprint.
  - Chroma (12-dim pitch-class profile) is tempo/octave invariant and well-
    established as the primary audio feature for cover song identification
    and classical-music matching (Sync-Toolbox, Müller 2015, Thomé 2013).
  - We already have per-piece WAVs (Stage 04) — zero additional data prep.

Algorithm:
  1. Load per-piece WAV from Data/Pipeline/04_piece_split/240828/*_pN.wav.
  2. Compute CQT-chroma (librosa.feature.chroma_cqt) at 2 Hz (hop=22050).
     CQT-chroma gives better note separation than STFT-chroma for piano.
  3. L2-normalize each frame.
  4. For each pair (A, B), compute subsequence-DTW distance:
        - Allows B to match any contiguous window of A (and vice versa).
        - Cost function = 1 - cos_sim between chroma frames.
     Score = 1 - (min_path_cost / path_len).
  5. Emit pairwise scores → reuse graph pipeline for clustering.

Performance: 50 pieces → 1225 pairs. Each DTW O(N*M) on short frame sequences
(60s @ 2Hz = 120 frames) → fast. Full matrix computes in < 2 min.

Output: Data/Metadata/240828/set1_chroma_scores.json
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))


PIECES_MANIFEST = Path("Data/Pipeline/04_piece_split/240828/_pieces.json")
PIECES_DIR = Path("Data/Pipeline/04_piece_split/240828")
OUT_PATH = Path("Data/Metadata/240828/set1_chroma_scores.json")

TASK_SET = {2: "Set1", 9: "Free"}

SR = 22050
HOP = 11025   # 0.5s per frame (2 Hz); plenty for structural matching
N_BINS = 84   # CQT bins (7 octaves × 12)


@dataclass
class PieceChroma:
    piece_key: str
    sub: str
    trial: str
    piece_idx: int
    task: str
    duration: float
    chroma: np.ndarray   # shape (12, T), L2-normalized


def compute_chroma(wav_path: Path) -> np.ndarray:
    """Load WAV and return L2-normalized CQT-chroma, shape (12, T)."""
    y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP, n_chroma=12, n_octaves=6)
    # L2-normalize each frame
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return chroma / norms


def subsequence_dtw_distance(x: np.ndarray, y: np.ndarray) -> tuple[float, int, int]:
    """Subsequence DTW: find best contiguous window of the LONGER sequence
    that matches the SHORTER in full. Returns (mean_cost, path_len, matched_start_frame).

    Cost = 1 - dot(frame_x, frame_y)  [cosine distance since chroma is L2-normed].
    Uses Sakoe-Chiba band of max(|len_x - len_y| + 5, 5) for efficiency.
    """
    # Always treat `a` as shorter, `b` as longer for subsequence semantics
    if x.shape[1] <= y.shape[1]:
        a, b = x, y
    else:
        a, b = y, x
    La, Lb = a.shape[1], b.shape[1]

    # Cost matrix: cost[i, j] = distance between a[:, i] and b[:, j]
    cost = 1.0 - a.T @ b   # (La, Lb)
    # Subsequence DTW: start row free (D[0, j] = cost[0, j]), end row minimum over j
    INF = np.inf
    D = np.full((La, Lb), INF, dtype=np.float32)
    D[0, :] = cost[0, :]
    for i in range(1, La):
        # Standard 3-step DTW
        prev = D[i - 1]
        # Shift and take min of (prev[j], prev[j-1], D[i, j-1])
        for j in range(Lb):
            best = prev[j]
            if j > 0:
                if prev[j - 1] < best:
                    best = prev[j - 1]
                if D[i, j - 1] < best:
                    best = D[i, j - 1]
            D[i, j] = cost[i, j] + best

    j_end = int(np.argmin(D[-1]))
    min_cost = float(D[-1, j_end])

    # Backtrack to get path length (approximate)
    i, j = La - 1, j_end
    path_len = 1
    while i > 0:
        if j == 0:
            i -= 1
        else:
            prev = [D[i - 1, j], D[i - 1, j - 1], D[i, j - 1]]
            k = int(np.argmin(prev))
            if k == 0:
                i -= 1
            elif k == 1:
                i -= 1
                j -= 1
            else:
                j -= 1
        path_len += 1

    return min_cost / path_len, path_len, j_end - path_len + 1


def collect_pieces() -> list[PieceChroma]:
    manifest = json.loads(PIECES_MANIFEST.read_text(encoding="utf-8"))
    records: list[PieceChroma] = []
    print("Computing chroma features...")
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
                chroma = compute_chroma(wav)
            except Exception as e:
                print(f"  [skip {wav.name}] {e}")
                continue
            records.append(PieceChroma(
                piece_key=f"sub{sub}_{trial}_p{p['piece_idx']}",
                sub=f"sub{sub}",
                trial=trial,
                piece_idx=p["piece_idx"],
                task=task,
                duration=round(p["end_sec"] - p["start_sec"], 2),
                chroma=chroma.astype(np.float32),
            ))
    print(f"  {len(records)} pieces, total chroma frames: {sum(r.chroma.shape[1] for r in records)}")
    return records


def main():
    records = collect_pieces()
    n = len(records)

    print(f"\nComputing all-pairs chroma-DTW similarity ({n*(n-1)//2} pairs)...")
    sims = []
    import time
    t0 = time.time()
    for i in range(n):
        for j in range(i + 1, n):
            mean_cost, plen, _ = subsequence_dtw_distance(records[i].chroma, records[j].chroma)
            # Convert cost (0 = perfect match, 1 = orthogonal) to similarity
            sim = 1.0 - mean_cost
            sims.append({
                "a": records[i].piece_key,
                "b": records[j].piece_key,
                "similarity": round(float(sim), 4),
                "path_length": plen,
            })
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n} rows  [{elapsed:.1f}s elapsed]")
    print(f"  done in {time.time() - t0:.1f}s")

    # Sort by similarity descending
    sims.sort(key=lambda s: -s["similarity"])

    # Report: distribution of similarity scores
    scores = [s["similarity"] for s in sims]
    pct = lambda p: scores[int(len(scores) * p)]
    print(f"\nSimilarity distribution:  p99={pct(0.01):.3f}  p95={pct(0.05):.3f}  "
          f"p50={pct(0.5):.3f}  p5={pct(0.95):.3f}  p1={pct(0.99):.3f}")

    # Top 20 pairs
    print("\nTop 20 similar pairs:")
    for s in sims[:20]:
        print(f"  {s['a']:<22} <-> {s['b']:<22}  sim={s['similarity']:.3f}")

    out = {
        "_comment": "Pairwise chroma-DTW similarity for 240828 Set1/Free pieces. Higher=more similar.",
        "sr": SR,
        "hop_length": HOP,
        "method": "CQT-chroma + subsequence DTW (cosine cost)",
        "n_pieces": n,
        "pieces": [
            {
                "piece_key": r.piece_key,
                "sub": r.sub,
                "trial": r.trial,
                "piece_idx": r.piece_idx,
                "task": r.task,
                "duration_sec": r.duration,
                "n_chroma_frames": int(r.chroma.shape[1]),
            }
            for r in records
        ],
        "pair_similarities": sims,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
