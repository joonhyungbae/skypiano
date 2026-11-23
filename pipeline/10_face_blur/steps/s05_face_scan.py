"""Stage 10 face scan: run the face detectors alone over clips and keep every hit, for a person to label.

For views where the face is seldom in frame (240828 Left looks down on the keyboard from above), masking every
detector hit masks mostly forearms and clothing, and a fixed zone masks the floor. The scan records the hits of
YOLOv8-face (every STRIDE-th frame, full size) and EgoBlur (every EGO_STRIDE-th frame, at EGO_SCALE) with their
scores; src/face_labels.py groups them into episodes, and the reviewer labels each episode in the review
dashboard (face, other_face, arm_hand, other). s04_strict_blur.py --face-labels then leaves a clip unmasked
when every episode is labelled and none is a face.

Output, per dataset, under Data/Pipeline/10_face_blur/{dataset}/face_scan/:
    {clip stem}.json                hits + scan settings
    crops/{clip stem}/f{frame}_{x1}_{y1}_{x2}_{y2}.jpg       context crop (1.5 box sizes around, box in red)
    crops/{clip stem}/f{frame}_{x1}_{y1}_{x2}_{y2}_full.jpg  the whole frame, 960 px wide
    _index.json                     per clip: frames, fps, hit count, episodes (rebuilt after every run;
                                    the review dashboard reads only this and the crops)

Crops are made for each episode's sample hits; --regroup only rebuilds the crops and the index from the stored
hits (after a change of the grouping), without running the detectors.

    python pipeline/10_face_blur/steps/s05_face_scan.py --dataset 240828 --views Left [--gpu 0] [--only LIST]
    python pipeline/10_face_blur/steps/s05_face_scan.py --dataset 240828 --regroup
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import face_labels as FL  # noqa: E402

STRIDE = 5               # YOLOv8-face on every 5th frame, full size
EGO_STRIDE = 30          # EgoBlur on every 30th frame ...
EGO_SCALE = 0.5          # ... at half size (0.7 s per full 1080p frame on a GTX 980)
MIN_SCORE = 0.3
SRC_ROOT = Path("Data/Pipeline/04_piece_split")
OUT_ROOT = Path("Data/Pipeline/10_face_blur")


def scan_clip(det, src: Path) -> dict:
    cap = cv2.VideoCapture(str(src))
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 30.0
    hits, i = [], 0
    try:
        while True:
            if i % STRIDE:
                if not cap.grab():
                    break
                i += 1
                continue
            ok, fr = cap.read()
            if not ok:
                break
            ego = []
            if i % EGO_STRIDE == 0:
                small = cv2.resize(fr, None, fx=EGO_SCALE, fy=EGO_SCALE)
                ego = [[v / EGO_SCALE for v in b[:4]] + [b[4]] for b in det.egoblur(small)]
            for name, boxes in (("egoblur", ego), ("yolo_face", det.yolo_face(fr))):
                hits += [{"frame": i, "det": name, "box": [round(v) for v in b[:4]], "score": round(float(b[4]), 3)}
                         for b in boxes if b[4] >= MIN_SCORE]
            i += 1
    finally:
        cap.release()
    return {"clip": src.name, "frames": n, "fps": fps, "stride": STRIDE, "ego_stride": EGO_STRIDE,
            "ego_scale": EGO_SCALE, "min_score": MIN_SCORE, "hits": hits}


def write_crops(scan: dict, src: Path, crop_dir: Path) -> None:
    """Context crop + whole frame for each episode's sample hits that has none yet."""
    want = [(h["frame"], tuple(h["box"])) for e in FL.episodes(scan) for h in e["samples"]]
    want = [(f, b) for f, b in want if not (crop_dir / f"f{f}_{'_'.join(map(str, b))}_full.jpg").exists()]
    if not want:
        return
    crop_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    try:
        for f, b in sorted(set(want)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, fr = cap.read()
            if not ok:
                continue
            h, w = fr.shape[:2]
            x1, y1, x2, y2 = b
            cv2.rectangle(fr, (x1, y1), (x2, y2), (0, 0, 255), 3)
            c = max(x2 - x1, y2 - y1) * 1.5
            crop = fr[max(0, int(y1 - c)):min(h, int(y2 + c)), max(0, int(x1 - c)):min(w, int(x2 + c))]
            stem = f"f{f}_{'_'.join(map(str, b))}"
            cv2.imwrite(str(crop_dir / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            cv2.imwrite(str(crop_dir / f"{stem}_full.jpg"), cv2.resize(fr, (960, int(960 * h / w))),
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
    finally:
        cap.release()


def write_index(out: Path) -> None:
    rows = {}
    for f in sorted(out.glob("*.json")):
        if f.name.startswith("_"):
            continue
        s = json.loads(f.read_text(encoding="utf-8"))
        rows[f.stem] = {"frames": s["frames"], "fps": s["fps"], "hits": len(s["hits"]), "episodes": FL.episodes(s)}
    (out / "_index.json").write_text(json.dumps({"clips": rows, "built": time.strftime("%Y-%m-%d %H:%M:%S")}),
                                     encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--views", default="Left", help="comma-separated camera views, e.g. Left,Right")
    ap.add_argument("--only", default=None, help="file listing clip names to scan (default: every clip of the views)")
    ap.add_argument("--src-dir", default=None, help="default: Data/Pipeline/04_piece_split/{dataset}/videos")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--regroup", action="store_true", help="rebuild crops and index from the stored hits only")
    a = ap.parse_args()
    src_dir = Path(a.src_dir) if a.src_dir else SRC_ROOT / a.dataset / "videos"
    out = OUT_ROOT / a.dataset / "face_scan"
    out.mkdir(parents=True, exist_ok=True)
    if a.regroup:
        for f in sorted(out.glob("*.json")):
            if not f.name.startswith("_"):
                write_crops(json.loads(f.read_text(encoding="utf-8")), src_dir / f"{f.stem}.mp4", out / "crops" / f.stem)
        write_index(out)
        return
    views = a.views.split(",")
    names = (Path(a.only).read_text(encoding="utf-8").split() if a.only else
             sorted(p.name for p in src_dir.glob(f"{a.dataset}_*.mp4") if any(p.stem.endswith(f"_{v}") for v in views)))
    names = [n for n in names if "_C0_N0_" not in n]          # C0_N0 is never released
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import s04_strict_blur as S
    S.MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "ego_blur_face.jit"
    det = S.Detectors(a.gpu, False, verify=True)
    for name in names:
        dst = out / (Path(name).stem + ".json")
        if dst.exists():
            continue
        t0 = time.time()
        scan = scan_clip(det, src_dir / name)
        write_crops(scan, src_dir / name, out / "crops" / Path(name).stem)
        scan["sec"] = round(time.time() - t0, 1)
        tmp = dst.with_suffix(".tmp")
        tmp.write_text(json.dumps(scan), encoding="utf-8")
        tmp.replace(dst)
        print(f"{name} hits={len(scan['hits'])} episodes={len(FL.episodes(scan))} {scan['sec']}s", flush=True)
    write_index(out)


if __name__ == "__main__":
    main()
