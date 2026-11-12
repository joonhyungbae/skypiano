"""Stage 11 Step 01: EgoBlur face anonymization over Stage 4 sliced videos.

Input : Data/Pipeline/03_sync/audio_midi/Sliced/{ds}/{sess}/video/{angle}/{stem}.mp4
Output: Data/Pipeline/10_face_blur/{ds}/{sess}/{angle}/{stem}.mp4

Per-frame face detection (torch.jit) + Gaussian blur over detected bboxes,
then H.264 re-encode via ffmpeg. Detection can be skipped every N frames to
save GPU time (`EGOBLUR_FRAME_STRIDE`); gaps are filled with the previous
frame's boxes (TTL 15 frames) so brief detection misses don't leak pixels.

Some local workspaces still keep the Stage 4 trial-sliced tree under the
legacy path `Data/Pipeline/03_sync/audio_midi/slice_legacy/Sliced`. This step
accepts either layout and prefers the canonical root when both exist.
"""
from __future__ import annotations

import argparse
import functools
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from tqdm import tqdm

try:
    import torchvision  # noqa: F401 — used for torchvision.ops.nms
    _HAVE_TORCHVISION = True
except ImportError:
    _HAVE_TORCHVISION = False

print = functools.partial(print, flush=True)

# ── Paths ────────────────────────────────────────────────────────────
STAGE4_SLICED = Path("Data/Pipeline/03_sync/audio_midi/Sliced")
STAGE4_SLICED_LEGACY = Path("Data/Pipeline/03_sync/audio_midi/slice_legacy/Sliced")
OUT_ROOT      = Path("Data/Pipeline/10_face_blur")
MODEL_PATH    = Path(__file__).resolve().parent.parent / "models" / "ego_blur_face.jit"

# ── Release scope policy ─────────────────────────────────────────────
# Only datasets listed here are processed, and only the angles named for each.
# Rationale: we release only the camera angles that need anonymization;
# frontal/top angles and older years are not included in the EgoBlur release.
# 240809 is entirely excluded from the public release, so it is not blurred.
#
# Override with CLI flags (--only, --angle) for one-off runs.
DEFAULT_SCOPE: dict[str, list[str]] = {
    "240828": ["Top", "Left", "Right", "RightLow"],
}

# ── Tunables (env-overridable) ───────────────────────────────────────
FACE_THRESHOLD = float(os.environ.get("EGOBLUR_FACE_THRESHOLD", 0.9))
NMS_IOU        = float(os.environ.get("EGOBLUR_NMS_IOU", 0.3))
BLUR_SIGMA_K   = float(os.environ.get("EGOBLUR_BLUR_SIGMA_K", 0.35))
FRAME_STRIDE   = max(1, int(os.environ.get("EGOBLUR_FRAME_STRIDE", 1)))
DEVICE_OVERRIDE = os.environ.get("EGOBLUR_DEVICE", "").strip()
STALE_TTL      = 15     # frames — how long a prior detection survives when
                        # the current frame has no detection

# ── Model ────────────────────────────────────────────────────────────
def pick_device() -> torch.device:
    if DEVICE_OVERRIDE:
        return torch.device(DEVICE_OVERRIDE)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_face_model(device: torch.device):
    if not _HAVE_TORCHVISION:
        print("[error] torchvision is required for NMS. Install with:")
        print("        pip install torchvision")
        sys.exit(2)
    if not MODEL_PATH.exists():
        print(f"[error] model not found: {MODEL_PATH}")
        print("        Download ego_blur_face.jit and place it under models/.")
        sys.exit(2)
    model = torch.jit.load(str(MODEL_PATH), map_location=device).to(device).eval()
    return model


def pick_stage4_sliced_root() -> Path | None:
    if STAGE4_SLICED.exists():
        return STAGE4_SLICED
    if STAGE4_SLICED_LEGACY.exists():
        return STAGE4_SLICED_LEGACY
    return None


# ── Inference ────────────────────────────────────────────────────────
@torch.inference_mode()
def detect_faces(model, bgr: np.ndarray, device: torch.device) -> np.ndarray:
    """Return Nx4 float32 ndarray of [x1, y1, x2, y2] in pixel coords."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).to(device)
    out = model(tensor)
    # EgoBlur output: (boxes, _, scores, _) in pixel coords
    boxes, _, scores, _ = out
    if boxes.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32)
    keep = torchvision.ops.nms(boxes, scores, NMS_IOU)
    boxes = boxes[keep]
    scores = scores[keep]
    mask = scores > FACE_THRESHOLD
    return boxes[mask].cpu().numpy().astype(np.float32)


def blur_boxes(bgr: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Gaussian-blur each bbox in-place. sigma scales with bbox width."""
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


# ── Video I/O ────────────────────────────────────────────────────────
def encode_to_mp4(frames_dir: Path, out_path: Path, fps: float):
    """Encode a directory of PNG frames to H.264 MP4 via ffmpeg."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", f"{fps:.6f}",
        "-i", str(frames_dir / "frame_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-preset", "medium",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def process_video(model, device: torch.device, src: Path, dst: Path) -> dict:
    """Blur faces in one video; write to dst. Returns stats."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return {"ok": False, "reason": "cv2 open failed"}

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stats = {"ok": True, "frames": 0, "detected_frames": 0,
             "total_boxes": 0, "width": width, "height": height,
             "fps": fps, "t0": time.time()}

    tmp = Path(tempfile.mkdtemp(prefix="egoblur_"))
    try:
        last_boxes = np.zeros((0, 4), dtype=np.float32)
        last_boxes_age = STALE_TTL + 1
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Run detection every FRAME_STRIDE frames; otherwise reuse cached boxes.
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

            cv2.imwrite(str(tmp / f"frame_{idx:06d}.png"), frame)
            idx += 1
            stats["frames"] = idx

        cap.release()
        encode_to_mp4(tmp, dst, fps)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    stats["elapsed_sec"] = time.time() - stats["t0"]
    return stats


# ── Batch orchestration ──────────────────────────────────────────────
def iter_inputs(only_ds: list[str] | None,
                only_session: str | None,
                only_angle: str | None,
                ignore_scope: bool = False) -> Iterable[tuple[Path, Path]]:
    """Yield (src_video, dst_video) pairs for all matching sliced videos.

    Filtering order:
      1. --only / --session / --angle CLI flags narrow first.
      2. If those are unset (and ignore_scope is False) the DEFAULT_SCOPE
         release policy applies (240828 only; all 4 angles
         Top/Left/Right/RightLow).
    """
    stage4_root = pick_stage4_sliced_root()
    if stage4_root is None:
        print(f"[warn] input root missing: {STAGE4_SLICED}")
        print(f"[warn] legacy input root missing: {STAGE4_SLICED_LEGACY}")
        return

    datasets = sorted(p.name for p in stage4_root.iterdir() if p.is_dir())
    if only_ds:
        datasets = [d for d in datasets if d in only_ds]
    elif not ignore_scope:
        datasets = [d for d in datasets if d in DEFAULT_SCOPE]

    for ds in datasets:
        ds_dir = stage4_root / ds
        allowed_angles = None
        if only_angle:
            allowed_angles = {only_angle}
        elif not ignore_scope and ds in DEFAULT_SCOPE:
            allowed_angles = set(DEFAULT_SCOPE[ds])

        for sess_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            if only_session and sess_dir.name != only_session:
                continue
            video_root = sess_dir / "video"
            if not video_root.exists():
                continue
            for angle_dir in sorted(p for p in video_root.iterdir() if p.is_dir()):
                if allowed_angles is not None and angle_dir.name not in allowed_angles:
                    continue
                for mp4 in sorted(angle_dir.glob("*.mp4")):
                    dst = OUT_ROOT / ds / sess_dir.name / angle_dir.name / mp4.name
                    yield mp4, dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="dataset filter (e.g. 240809 240828)")
    ap.add_argument("--session", default=None,
                    help="session folder under dataset (e.g. 240809PJH1)")
    ap.add_argument("--angle", default=None,
                    help="camera angle subfolder (e.g. Top / Left / Right / RightLow)")
    ap.add_argument("--force", action="store_true",
                    help="re-encode even if output exists")
    ap.add_argument("--all", action="store_true",
                    help="ignore DEFAULT_SCOPE and process every dataset/angle found")
    args = ap.parse_args()

    device = pick_device()
    print(f"[stage11] device={device} model={MODEL_PATH.name} "
          f"threshold={FACE_THRESHOLD} stride={FRAME_STRIDE}")
    model = load_face_model(device)

    pairs = list(iter_inputs(args.only, args.session, args.angle,
                              ignore_scope=args.all))
    if not pairs:
        print("[stage11] no matching input videos.")
        return

    todo = []
    skipped = 0
    for src, dst in pairs:
        if not args.force and dst.exists() and dst.stat().st_size > 0:
            skipped += 1
            continue
        todo.append((src, dst))

    print(f"[stage11] {len(todo)} to process, {skipped} skipped (already present).")
    if not todo:
        return

    ok = fail = 0
    for src, dst in tqdm(todo, desc="stage11"):
        try:
            stats = process_video(model, device, src, dst)
            if stats.get("ok"):
                ok += 1
                print(f"  [ok] {src.name}  "
                      f"{stats['frames']}f  "
                      f"{stats['detected_frames']}/{stats['frames']} blurred  "
                      f"{stats['total_boxes']} boxes  "
                      f"{stats['elapsed_sec']:.1f}s")
            else:
                fail += 1
                print(f"  [fail] {src}: {stats.get('reason')}")
        except Exception as e:
            fail += 1
            print(f"  [error] {src}: {e}")

    print(f"[stage11] done. ok={ok} fail={fail} skipped={skipped}")


if __name__ == "__main__":
    main()
