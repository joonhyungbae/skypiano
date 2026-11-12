"""
Stage 10 step 04: strict face blur for released video, with a verification pass.

The piece-video blur (s02) detects on every FRAME_STRIDE-th frame and carries
boxes over; when the player moves, the box lags or drifts and the face shows
for seconds. This step is for release:

  1. detect on every frame at DETECT_THRESHOLD (low: a missed face costs more
     than a blurred shoulder);
  2. pad every box by PAD of its width and height on each side;
  3. blur, on each frame, the union of the padded boxes of the frames within
     +-SPAN frames, and keep a box HOLD frames after its last detection, so
     short detection gaps and fast motion stay covered;
  4. pixelate each region to MOSAIC_CELLS cells across its width, then blur
     it (not recoverable by deblurring);
  5. verify: run the detector on every frame of the blurred clip at
     VERIFY_THRESHOLD; any detection fails the clip. A failed clip is blurred
     again with the padding doubled, up to RETRIES times; a clip that still
     fails is reported and must not be released.

Per clip, `{clip}.json` beside the output and one row of --stats-csv:
frames, frames_with_boxes (detected on the frame itself), frames_blurred,
pad, verify_detections, verify_max_score, passed.

--verify-only runs step 5 on existing clips (any blur) and changes nothing.

Usage (from the repository root; GPU index as seen by torch):
  python pipeline/10_face_blur/steps/s04_strict_blur.py --src-dir SRC --dst-dir DST --only LIST --stats-csv CSV
  python pipeline/10_face_blur/steps/s04_strict_blur.py --verify-only --src-dir DIR --only LIST --stats-csv CSV
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision  # noqa: F401  (registers torchvision::nms for the TorchScript model)

MODEL_PATH = Path("pipeline/10_face_blur/models/ego_blur_face.jit")
DETECT_THRESHOLD = 0.5
VERIFY_THRESHOLD = 0.5
NMS_IOU = 0.3
PAD = 0.4
SPAN = 8
HOLD = 30
MOSAIC_CELLS = 8
RETRIES = 2


def load_model(device: torch.device):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"EgoBlur model missing: {MODEL_PATH}")
    return torch.jit.load(str(MODEL_PATH), map_location="cpu").to(device)


@torch.inference_mode()
def detect(model, device, bgr: np.ndarray, thr: float) -> tuple[np.ndarray, np.ndarray]:
    t = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1).to(device)
    boxes, _, scores, _ = model(t)
    if boxes.numel() == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    keep = torchvision.ops.nms(boxes, scores, NMS_IOU)
    boxes, scores = boxes[keep], scores[keep]
    m = scores > thr
    return boxes[m].cpu().numpy().astype(np.float32), scores[m].cpu().numpy().astype(np.float32)


def frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                return
            yield fr
    finally:
        cap.release()


def video_info(path: Path) -> tuple[float, int, int]:
    cap = cv2.VideoCapture(str(path))
    info = (cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info


def pad_box(b: np.ndarray, pad: float, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = b
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    return (max(0, int(x1 - dx)), max(0, int(y1 - dy)), min(w, int(x2 + dx + 1)), min(h, int(y2 + dy + 1)))


def mosaic(bgr: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = bgr[y1:y2, x1:x2]
    cw = max(1, MOSAIC_CELLS)
    ch = max(1, round(cw * (y2 - y1) / (x2 - x1)))
    small = cv2.resize(roi, (cw, ch), interpolation=cv2.INTER_AREA)
    big = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    bgr[y1:y2, x1:x2] = cv2.GaussianBlur(big, (0, 0), sigmaX=max(3.0, (x2 - x1) / cw / 2))


def detections(model, device, src: Path, thr: float) -> list[tuple[np.ndarray, np.ndarray]]:
    return [detect(model, device, fr, thr) for fr in frames(src)]


def blur(model, device, src: Path, dst: Path, per_frame: list, pad: float) -> int:
    """Write src blurred to dst (via a .part name); return the number of frames with a blurred region."""
    fps, w, h = video_info(src)
    n = len(per_frame)
    last_seen = -10 ** 9
    tmp = dst.with_suffix(".part.mp4")
    dst.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-", "-an", "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "20", "-movflags", "+faststart",
                           str(tmp)], stdin=subprocess.PIPE)
    blurred, held = 0, []
    try:
        for i, fr in enumerate(frames(src)):
            near = [b for j in range(max(0, i - SPAN), min(n, i + SPAN + 1)) for b in per_frame[j][0]]
            if len(per_frame[i][0]):
                last_seen, held = i, list(per_frame[i][0])
            boxes = near + (held if i - last_seen <= HOLD else [])
            for b in boxes:
                mosaic(fr, pad_box(b, pad, w, h))
            blurred += bool(boxes)
            ff.stdin.write(fr.tobytes())
    finally:
        ff.stdin.close()
        ff.wait()
    if ff.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed on {src.name}")
    tmp.replace(dst)
    return blurred


def verify(model, device, clip: Path) -> tuple[int, float, int]:
    hits, best, n = 0, 0.0, 0
    for fr in frames(clip):
        _, s = detect(model, device, fr, VERIFY_THRESHOLD)
        n += 1
        if len(s):
            hits += 1
            best = max(best, float(s.max()))
    return hits, round(best, 4), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--dst-dir", default=None)
    ap.add_argument("--only", required=True, help="file listing clip names (one per line)")
    ap.add_argument("--stats-csv", required=True)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    device = torch.device(f"cuda:{a.gpu}")
    model = load_model(device)
    src_dir = Path(a.src_dir)
    dst_dir = Path(a.dst_dir) if a.dst_dir else None
    names = [n for n in Path(a.only).read_text(encoding="utf-8").split() if (src_dir / n).exists()]
    rows = []
    for k, name in enumerate(names, 1):
        t0 = time.time()
        if a.verify_only:
            hits, best, n = verify(model, device, src_dir / name)
            row = {"clip": name, "frames": n, "verify_detections": hits, "verify_max_score": best,
                   "passed": hits == 0}
        else:
            dst = dst_dir / name
            meta = dst.with_suffix(".json")
            if meta.exists() and not a.force:
                rows.append(json.loads(meta.read_text(encoding="utf-8")))
                continue
            per_frame = detections(model, device, src_dir / name, DETECT_THRESHOLD)
            pad = PAD
            for attempt in range(RETRIES + 1):
                blurred = blur(model, device, src_dir / name, dst, per_frame, pad)
                hits, best, n = verify(model, device, dst)
                if hits == 0:
                    break
                pad *= 2
            row = {"clip": name, "frames": len(per_frame),
                   "frames_with_boxes": sum(len(b) > 0 for b, _ in per_frame), "frames_blurred": blurred,
                   "pad": pad, "verify_detections": hits, "verify_max_score": best, "passed": hits == 0}
            if hits:
                dst.rename(dst.with_suffix(".failed.mp4"))
            meta.write_text(json.dumps(row), encoding="utf-8")
        row["elapsed_sec"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"[{k}/{len(names)}] {name} {row}", flush=True)
    out = Path(a.stats_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(f for r in rows for f in r))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} clips, {sum(not r['passed'] for r in rows)} failed -> {out}")


if __name__ == "__main__":
    main()
