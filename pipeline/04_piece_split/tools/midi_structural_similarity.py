"""
Phase 2: Structural MIDI feature similarity for Set1/Free clustering.

Motivation: our hand-crafted n-gram fingerprint matches exact local patterns
(pitch intervals + IOI ratios), which fails when two performances differ in
phrasing but share global harmonic/rhythmic structure. This module extracts
higher-level features per MIDI:

  - Pitch-class histogram (12)      → tonality fingerprint
  - Pitch-class bigram transition (12x12 flattened, top-K kept) → harmonic flow
  - Pitch-range histogram (8, binned over MIDI 21-108) → register/voicing
  - IOI (inter-onset-interval) log histogram (8) → rhythmic character
  - Duration log histogram (8) → note-length character
  - Polyphony (chord size) histogram (8) → texture
  - Overall note density (notes per second, single scalar)

All histograms L1-normalized, concatenated, then L2-normalized → cosine sim.

No pretrained model needed; runs in milliseconds per piece. Complements the
fingerprint (local) + chroma (audio-harmonic) signals with symbolic-structural
similarity.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pretty_midi

PIECES_MANIFEST = Path("Data/Pipeline/04_piece_split/240828/_pieces.json")
PIECES_DIR = Path("Data/Pipeline/04_piece_split/240828")
OUT_PATH = Path("Data/Metadata/240828/set1_structural_scores.json")

TASK_SET = {2: "Set1", 9: "Free"}


def _log_hist(values: list[float], n_bins: int = 8, min_val: float = 0.01,
              max_val: float = 10.0) -> np.ndarray:
    """Log-spaced histogram (useful for durations, IOIs)."""
    if not values:
        return np.zeros(n_bins, dtype=np.float32)
    edges = np.logspace(math.log10(min_val), math.log10(max_val), n_bins + 1)
    hist, _ = np.histogram(np.clip(values, min_val, max_val), bins=edges)
    hist = hist.astype(np.float32)
    s = hist.sum()
    return hist / s if s > 0 else hist


def structural_features(midi_path: Path) -> np.ndarray:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes: list[pretty_midi.Note] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        notes.extend(inst.notes)
    if not notes:
        return np.zeros(12 + 32 + 8 + 8 + 8 + 8 + 1, dtype=np.float32)
    notes.sort(key=lambda n: n.start)
    duration = pm.get_end_time() or 1.0

    # Pitch-class histogram (12)
    pc_hist = np.zeros(12, dtype=np.float32)
    for n in notes:
        pc_hist[n.pitch % 12] += 1
    pc_hist = pc_hist / pc_hist.sum() if pc_hist.sum() > 0 else pc_hist

    # Pitch-class bigram: transitions in melodic sequence (monophonic proxy)
    # use top note per event (highest pitch per onset) for melody line
    from collections import defaultdict
    event_groups: dict[float, list[int]] = defaultdict(list)
    for n in notes:
        event_groups[round(n.start / 0.04) * 0.04].append(n.pitch)
    melody = [max(ps) % 12 for _, ps in sorted(event_groups.items())]
    bigram = np.zeros((12, 12), dtype=np.float32)
    for a, b in zip(melody[:-1], melody[1:]):
        bigram[a, b] += 1
    if bigram.sum() > 0:
        bigram /= bigram.sum()
    # Flatten top 32 entries (sparse representation)
    bigram_flat = bigram.flatten()
    top32_idx = np.argsort(-bigram_flat)[:32]
    bigram_top32 = np.zeros(32, dtype=np.float32)
    bigram_top32[:] = bigram_flat[top32_idx]

    # Pitch-range histogram (8 bins over MIDI 21-108)
    pr_edges = np.linspace(21, 108, 9)
    pr_hist, _ = np.histogram([n.pitch for n in notes], bins=pr_edges)
    pr_hist = pr_hist.astype(np.float32)
    pr_hist = pr_hist / pr_hist.sum() if pr_hist.sum() > 0 else pr_hist

    # IOI histogram (log-spaced)
    iois = [notes[i + 1].start - notes[i].start for i in range(len(notes) - 1)
            if notes[i + 1].start > notes[i].start]
    ioi_hist = _log_hist(iois, n_bins=8, min_val=0.02, max_val=4.0)

    # Duration histogram
    durs = [n.end - n.start for n in notes if n.end > n.start]
    dur_hist = _log_hist(durs, n_bins=8, min_val=0.05, max_val=8.0)

    # Polyphony histogram: chord size per onset
    chord_sizes = [len(ps) for _, ps in event_groups.items()]
    poly_hist, _ = np.histogram(chord_sizes, bins=[1, 2, 3, 4, 5, 6, 7, 9, 100])
    poly_hist = poly_hist.astype(np.float32)
    poly_hist = poly_hist / poly_hist.sum() if poly_hist.sum() > 0 else poly_hist

    # Note density (notes per second) — single scalar but weighted to be informative
    density = len(notes) / duration / 10.0   # ÷ 10 to keep in similar magnitude to histograms

    feat = np.concatenate([pc_hist, bigram_top32, pr_hist, ioi_hist, dur_hist, poly_hist, [density]])
    # L2-normalize
    norm = np.linalg.norm(feat)
    return feat / norm if norm > 0 else feat


def collect_features():
    manifest = json.loads(PIECES_MANIFEST.read_text(encoding="utf-8"))
    records = []
    for t in manifest:
        task = TASK_SET.get(t["task"])
        if task is None:
            continue
        sub = t["raw_stem"][-4:-2]
        trial = t["raw_stem"][-4:]
        for p in t["pieces"]:
            mid = PIECES_DIR / f"{t['trial_stem']}_p{p['piece_idx']}.mid"
            if not mid.is_file():
                continue
            feat = structural_features(mid)
            records.append({
                "piece_key": f"sub{sub}_{trial}_p{p['piece_idx']}",
                "sub": f"sub{sub}",
                "trial": trial,
                "piece_idx": p["piece_idx"],
                "task": task,
                "duration_sec": round(p["end_sec"] - p["start_sec"], 2),
                "feature": feat,
            })
    return records


def main():
    print("Extracting structural features...")
    records = collect_features()
    print(f"  {len(records)} pieces, feature dim = {records[0]['feature'].shape[0]}")

    # All-pairs cosine similarity
    F = np.stack([r["feature"] for r in records])    # (N, D)
    sim_matrix = F @ F.T    # cosine since already L2-normalized

    # Flatten into edge list
    pairs = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            pairs.append({
                "a": records[i]["piece_key"],
                "b": records[j]["piece_key"],
                "similarity": round(float(sim_matrix[i, j]), 4),
            })
    pairs.sort(key=lambda p: -p["similarity"])

    sims = [p["similarity"] for p in pairs]
    pct = lambda q: sims[int(len(sims) * q)]
    print(f"\nStructural similarity distribution:")
    print(f"  p99={pct(0.01):.3f}  p95={pct(0.05):.3f}  p50={pct(0.5):.3f}  p5={pct(0.95):.3f}")

    print("\nTop 20 structural pairs:")
    for p in pairs[:20]:
        print(f"  {p['a']:<22} <-> {p['b']:<22}  sim={p['similarity']:.3f}")

    out = {
        "_comment": "Pairwise structural-feature cosine similarity for 240828 Set1/Free.",
        "method": "MIR features (pitch-class hist + bigram top32 + range + IOI + duration + polyphony + density)",
        "feature_dim": int(records[0]["feature"].shape[0]),
        "n_pieces": len(records),
        "pieces": [
            {k: v for k, v in r.items() if k != "feature"}
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
