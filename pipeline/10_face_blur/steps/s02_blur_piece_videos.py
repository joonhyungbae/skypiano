"""
Standalone CLI tool; not part of the run_pipeline.py loop.

Stage 6 Step 06: EgoBlur face anonymization for per-piece videos.

  * No PNG frame dump — pipes raw BGR24 directly into ffmpeg stdin for encoding.
  * Configurable FRAME_STRIDE (detection every N frames) with box TTL so
    cadence stays smooth. Default 5.
  * Multi-GPU fanout (--workers N): each worker owns a GPU, consumes videos
    from a queue.

  * --only FILE limits the run to the clip names listed in FILE (one per
    line); --stats-csv writes one row per clip: frames, frames with a face
    box (detected or carried over), boxes, so a release can state how many
    clips needed blurring. Each clip's numbers are also kept beside it
    ({clip}.json).

Usage:
    python s02_blur_piece_videos.py            # all Right+RightLow pieces
    python s02_blur_piece_videos.py --workers 2
    FRAME_STRIDE=3 python s02_blur_piece_videos.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision  # noqa: F401 — registers torchvision::nms for TorchScript load


STAGE6_ROOT = Path("Data/Pipeline/04_piece_split")
TARGET_DS = "240828"
# Explorer-served angles: blur every view since the web dashboard exposes
# Top/Left as well, and incidental face pixels can still appear there during
# posture shifts even when the primary framing is the keyboard.
BLUR_VIEWS = ["Top", "Left", "Right", "RightLow"]
MODEL_PATH = Path("pipeline/10_face_blur/models/ego_blur_face.jit")

# Tunables
FACE_THRESHOLD = float(os.environ.get("EGOBLUR_FACE_THRESHOLD", 0.9))
NMS_IOU        = float(os.environ.get("EGOBLUR_NMS_IOU", 0.3))
BLUR_SIGMA_K   = float(os.environ.get("EGOBLUR_BLUR_SIGMA_K", 0.35))
FRAME_STRIDE   = max(1, int(os.environ.get("EGOBLUR_FRAME_STRIDE", 5)))
STALE_TTL      = int(os.environ.get("EGOBLUR_STALE_TTL", 15))


def load_model(device: torch.device):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"EgoBlur model missing: {MODEL_PATH}")
    # Load on CPU, move to GPU — avoids mixed-device constants.
    m = torch.jit.load(str(MODEL_PATH), map_location="cpu")
    return m.to(device)


@torch.inference_mode()
def detect_faces(model, bgr: np.ndarray, device: torch.device) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).to(device)
    boxes, _, scores, _ = model(tensor)
    if boxes.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32)
    keep = torchvision.ops.nms(boxes, scores, NMS_IOU)
    boxes = boxes[keep]
    scores = scores[keep]
    mask = scores > FACE_THRESHOLD
    return boxes[mask].cpu().numpy().astype(np.float32)


def blur_boxes(bgr: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    for x1, y1, x2, y2 in boxes:
        x1i = max(0, int(x1)); y1i = max(0, int(y1))
        x2i = min(w, int(x2)); y2i = min(h, int(y2))
        if x2i - x1i < 4 or y2i - y1i < 4:
            continue
        roi = bgr[y1i:y2i, x1i:x2i]
        sigma = max(5.0, BLUR_SIGMA_K * (x2i - x1i))
        bgr[y1i:y2i, x1i:x2i] = cv2.GaussianBlur(roi, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return bgr


def process_video(model, device, src: Path, dst: Path) -> dict:
    """Blur one video, streaming frames directly to ffmpeg stdin."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return {"ok": False, "reason": "cv2 open failed"}

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    dst.parent.mkdir(parents=True, exist_ok=True)
    final, dst = dst, dst.with_suffix(".part.mp4")      # only a finished clip gets its final name

    ff = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", "-pix_fmt", "bgr24",
            "-r", f"{fps:.6f}",
            "-i", "-",
            "-an",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "fast", "-crf", "23",
            "-movflags", "+faststart",
            str(dst),
        ],
        stdin=subprocess.PIPE,
    )

    t0 = time.time()
    stats = {"ok": True, "frames": 0, "detected_frames": 0,
             "total_boxes": 0, "width": width, "height": height, "fps": fps}
    last_boxes = np.zeros((0, 4), dtype=np.float32)
    last_boxes_age = STALE_TTL + 1
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % FRAME_STRIDE == 0:
                boxes = detect_faces(model, frame, device)
                if len(boxes) > 0:
                    last_boxes = boxes
                    last_boxes_age = 0
                else:
                    last_boxes_age += 1
                    if last_boxes_age > STALE_TTL:
                        last_boxes = np.zeros((0, 4), dtype=np.float32)
            else:
                last_boxes_age += 1
                if last_boxes_age > STALE_TTL:
                    last_boxes = np.zeros((0, 4), dtype=np.float32)

            if len(last_boxes) > 0:
                frame = blur_boxes(frame, last_boxes.copy())
                stats["detected_frames"] += 1
                stats["total_boxes"] += len(last_boxes)

            ff.stdin.write(frame.tobytes())
            idx += 1
    finally:
        cap.release()
        if ff.stdin:
            ff.stdin.close()
        ff.wait()

    stats["frames"] = idx
    stats["elapsed_sec"] = time.time() - t0
    if ff.returncode == 0 and idx > 0:
        dst.replace(final)
    else:
        dst.unlink(missing_ok=True)
        stats.update(ok=False, reason=f"ffmpeg rc={ff.returncode}, frames={idx}")
    return stats


# ── Worker: one GPU, consumes (src, dst) from queue ──

def worker(gpu_id: int, q: Queue):
    device = torch.device(f"cuda:{gpu_id}")
    print(f"[worker {gpu_id}] loading model on {device}", flush=True)
    model = load_model(device)
    print(f"[worker {gpu_id}] ready", flush=True)

    while True:
        item = q.get()
        if item is None:
            break
        src, dst = Path(item[0]), Path(item[1])
        if dst.exists():
            print(f"[gpu{gpu_id}] SKIP {src.name} (exists)", flush=True)
            continue
        print(f"[gpu{gpu_id}] START {src.name}", flush=True)
        stats = process_video(model, device, src, dst)
        if stats.get("ok"):
            dst.with_suffix(".json").write_text(json.dumps({"clip": dst.name, **stats}), encoding="utf-8")
        if stats.get("ok"):
            fps_eff = stats["frames"] / max(1e-6, stats["elapsed_sec"])
            print(f"[gpu{gpu_id}] DONE  {src.name}  "
                  f"{stats['frames']}f/{stats['elapsed_sec']:.1f}s = {fps_eff:.1f}fps  "
                  f"det={stats['detected_frames']}  → {dst.stat().st_size/1024/1024:.1f}MB",
                  flush=True)
        else:
            print(f"[gpu{gpu_id}] FAIL  {src.name}: {stats.get('reason')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(2, torch.cuda.device_count()))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ds", default=TARGET_DS,
                    help="dataset code (default: 240828). Use to apply blur to a different release session.")
    ap.add_argument("--src-dir", default=None,
                    help="Override source dir (default: <STAGE6>/{ds}/videos). Useful when piece-split videos live alongside the trial files rather than under a videos/ subdir.")
    ap.add_argument("--dst-dir", default=None,
                    help="Override destination dir (default: <STAGE6>/{ds}/videos_blurred).")
    ap.add_argument("--views", nargs="*", default=BLUR_VIEWS,
                    help="View suffixes to blur (default: Top Left Right RightLow). Restrict for sessions whose camera set differs.")
    ap.add_argument("--only", default=None, help="file listing the clip names to blur (one per line)")
    ap.add_argument("--stats-csv", default=None, help="write per-clip face-detection counts here")
    args = ap.parse_args()
    only = set(Path(args.only).read_text(encoding="utf-8").split()) if args.only else None

    src_dir = Path(args.src_dir) if args.src_dir else STAGE6_ROOT / args.ds / "videos"
    dst_dir = Path(args.dst_dir) if args.dst_dir else STAGE6_ROOT / args.ds / "videos_blurred"
    dst_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, str]] = []
    for view in args.views:
        for src in sorted(src_dir.glob(f"*_{view}.mp4")):
            if only is not None and src.name not in only:
                continue
            dst = dst_dir / src.name
            if dst.exists() and not args.force:
                continue
            jobs.append((str(src), str(dst)))

    print(f"[main] {len(jobs)} videos to blur  FRAME_STRIDE={FRAME_STRIDE}  workers={args.workers}")
    if jobs:
        q: Queue = Queue()
        for j in jobs:
            q.put(j)
        for _ in range(args.workers):
            q.put(None)     # sentinel
        procs = [Process(target=worker, args=(i, q)) for i in range(args.workers)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        print("[main] all workers finished")
    else:
        print("[main] nothing to do")
    if args.stats_csv:
        names = sorted(only) if only is not None else sorted(p.name for v in args.views for p in dst_dir.glob(f"*_{v}.mp4"))
        rows = [json.loads((dst_dir / n).with_suffix(".json").read_text(encoding="utf-8"))
                for n in names if (dst_dir / n).with_suffix(".json").exists()]
        Path(args.stats_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.stats_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["clip", "ok", "frames", "detected_frames", "total_boxes",
                                              "width", "height", "fps", "elapsed_sec", "reason"], extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"[main] {len(rows)} clips, {sum(r.get('detected_frames', 0) > 0 for r in rows)} with a face -> {args.stats_csv}")


if __name__ == "__main__":
    main()
