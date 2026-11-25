"""Stage 10 step 07: order the corner check by how face-like each corner tile looks.

Learns from the reviewer's corner marks so far (the checked clips of face_labels "corner_checked" /
"corner"): every tile of a corner strip (s06_corner_strips.py) is described by the brightness, saturation and
a skin-tone share of a 3 x 4 grid of cells (the face enters the corner from its bottom edge), and a random
forest gives each tile of the unchecked clips a face probability. The review dashboard sorts the clips by
their highest tile and outlines likely tiles; it never marks anything itself.

Output: Data/Pipeline/10_face_blur/{dataset}/face_scan/corner/_suggest.json
        {"trained_on": {"clips", "tiles", "marked"}, "clips": {clip: [p per tile]}}
Tile features are cached in corner/_features.npz.

Usage: python s07_corner_suggest.py --dataset 240828
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import face_labels as FL  # noqa: E402

GRID = (3, 4)
MIN_MARKED = 20


def tile_features(img: np.ndarray, n: int, cols: int, tw: int, th: int) -> np.ndarray:
    out = []
    for k in range(n):
        t = img[(k // cols) * th:(k // cols + 1) * th, (k % cols) * tw:(k % cols + 1) * tw]
        hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV).astype(float) / 255
        ycc = cv2.cvtColor(t, cv2.COLOR_BGR2YCrCb)
        skin = ((ycc[..., 1] > 135) & (ycc[..., 1] < 180) & (ycc[..., 2] > 85) & (ycc[..., 2] < 135)).astype(float)
        f = []
        for gy in range(GRID[0]):
            for gx in range(GRID[1]):
                sl = (slice(gy * th // GRID[0], (gy + 1) * th // GRID[0]), slice(gx * tw // GRID[1], (gx + 1) * tw // GRID[1]))
                f += [skin[sl].mean(), hsv[..., 2][sl].mean(), hsv[..., 1][sl].mean()]
        out.append(f)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pipeline-root", default="Data/Pipeline")
    ap.add_argument("--metadata-root", default="Data/Metadata")
    a = ap.parse_args()
    cdir = Path(a.pipeline_root) / "10_face_blur" / a.dataset / "face_scan" / "corner"
    idx = json.loads((cdir / "_index.json").read_text(encoding="utf-8"))
    tw, th = idx["tile"]
    cache_f = cdir / "_features.npz"
    cache = dict(np.load(cache_f)) if cache_f.exists() else {}
    feats = {}
    for clip, c in idx["clips"].items():
        if clip not in cache or len(cache[clip]) != c["tiles"]:
            cache[clip] = tile_features(cv2.imread(str(cdir / f"{clip}.jpg")), c["tiles"], idx["cols"], tw, th)
        feats[clip] = cache[clip]
    np.savez_compressed(cache_f, **cache)

    lab = FL.read_labels(a.dataset, Path(a.metadata_root))
    marks = {c: {round(m["t"], 2) for m in v} for c, v in lab["corner"].items()}
    train = [c for c in lab["corner_checked"] if c in feats]
    X = np.vstack([feats[c] for c in train]) if train else np.zeros((0, 36))
    y = np.array([round(k * idx["step"], 2) in marks.get(c, set()) for c in train for k in range(len(feats[c]))])
    if y.sum() < MIN_MARKED:
        sys.exit(f"only {int(y.sum())} marked tiles; need {MIN_MARKED}")
    model = RandomForestClassifier(300, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(X, y)
    out = {clip: [round(float(p), 3) for p in model.predict_proba(f)[:, 1]] for clip, f in feats.items()}
    res = {"trained_on": {"clips": len(train), "tiles": int(len(y)), "marked": int(y.sum())}, "clips": out}
    (cdir / "_suggest.json").write_text(json.dumps(res), encoding="utf-8")
    print(f"trained on {len(train)} clips ({int(y.sum())} marked tiles); {len(out)} clips scored")


if __name__ == "__main__":
    main()
