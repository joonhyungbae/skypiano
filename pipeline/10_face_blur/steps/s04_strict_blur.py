"""
Stage 10 step 04: release face anonymization, with a coverage check.

Chosen by a benchmark on 800 labelled frames of the 240828 piece videos (all
four views): a face detector alone missed 3-15% of identifiable faces on the
side views and almost all of them on the overhead Left view (an upside-down
face at the bottom-right corner when the player leans); the union below
covered 99.5% at about 7% of the frame. Per frame:

  * EgoBlur face boxes with score >= FACE_THRESHOLD;
  * a head box from body pose (RTMO, rtmlib): around the nose / eye / ear
    keypoints with score >= KEYPOINT_THRESHOLD, sized by the shoulder width;
  * optionally RetinaFace-R50 boxes (--retinaface);
  * Left view: a fixed head zone at the bottom-right corner (LEFT_ZONE, the
    region of the labelled Left faces) while pose sees the player (--left-zone).

Every box is padded by PAD of its size on each side, and a frame is masked
with the union of the boxes of the frames within +-HOLD. The mask is either
an opaque fill of the padded ellipse (--mask solid, default) or a pixel
mosaic (--mask mosaic).

Coverage check, on the original frames against the mask: a face is
"confirmed" where two different detectors agree (score >= 0.5, IoU > 0.3) or
a face detection lies inside a pose head box; the verifier detectors are the
masking ones plus YOLOv8-face (VERIFY_EXTRA). A frame is flagged when a
confirmed face is less than COVER_MIN inside the mask. Per clip ({clip}.json
and one row of --stats-csv): frames, frames_masked, mask_area, confirmed,
flagged_frames, passed (no flagged frame). A flagged clip keeps its output
under {clip}.flagged.mp4 for review.

Detector weights: EgoBlur `pipeline/10_face_blur/models/ego_blur_face.jit`;
RTMO, RetinaFace and YOLOv8-face are fetched by rtmlib / batch-face /
huggingface_hub on first use. The detectors take one frame per call; run
several clips at once with --workers (each loads its own models).

Usage (from the repository root):
  python pipeline/10_face_blur/steps/s04_strict_blur.py --src-dir SRC --dst-dir DST --only LIST \
      --stats-csv CSV [--view Left] [--mask solid|mosaic] [--retinaface] [--workers 3]
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path("pipeline/10_face_blur/models/ego_blur_face.jit")
RTMO_URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/"
            "rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip")
YOLO_FACE = ("arnabdhar/YOLOv8-Face-Detection", "model.pt")
FACE_THRESHOLD = 0.5
KEYPOINT_THRESHOLD = 0.3
PAD = 0.4
HOLD = 15
NMS_IOU = 0.3
LEFT_ZONE = (1170, 620, 1920, 1080)       # 1920x1080 frame; scaled to other sizes
VERIFY_THRESHOLD = 0.5
VERIFY_IOU = 0.3
COVER_MIN = 0.9
VERIFY_EXTRA = ("yolo_face",)


class Detectors:
    def __init__(self, gpu: int, retinaface: bool, verify: bool):
        import torch
        import torchvision  # noqa: F401  (registers torchvision::nms for the TorchScript model)
        from rtmlib import RTMO
        self.torch, self.tv = torch, torchvision
        self.device = torch.device(f"cuda:{gpu}")
        self.ego = torch.jit.load(str(MODEL_PATH), map_location="cpu").to(self.device)
        self.pose = RTMO(RTMO_URL, backend="onnxruntime", device="cuda")
        self.retina = None
        if retinaface:
            from batch_face import RetinaFace
            self.retina = RetinaFace(gpu_id=gpu, network="resnet50")
        self.yolo = None
        if verify:
            from huggingface_hub import hf_hub_download
            from ultralytics import YOLO
            self.yolo = YOLO(hf_hub_download(*YOLO_FACE))
        self.gpu = gpu

    def egoblur(self, bgr):
        torch = self.torch
        with torch.inference_mode():
            t = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1).to(self.device)
            b, _, s, _ = self.ego(t)
            if b.numel() == 0:
                return []
            keep = self.tv.ops.nms(b, s, NMS_IOU)
            return [bb + [ss] for bb, ss in zip(b[keep].tolist(), s[keep].tolist())]

    def heads(self, bgr):
        """(head boxes with a score, whether a person is seen)."""
        kps, scores = self.pose(bgr)
        out = []
        for k, s in zip(kps, scores):
            head = [i for i in range(5) if s[i] >= KEYPOINT_THRESHOLD]       # nose, eyes, ears (COCO 0-4)
            if not head:
                continue
            pts = k[head]
            x1, y1 = pts.min(0)
            x2, y2 = pts.max(0)
            sh = np.linalg.norm(k[5] - k[6]) if min(s[5], s[6]) >= KEYPOINT_THRESHOLD else \
                max(x2 - x1, y2 - y1) * 2
            r = max(x2 - x1, y2 - y1, 0.35 * sh) * 0.9 + 1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            out.append([cx - r, cy - r * 1.2, cx + r, cy + r, float(np.mean(s[head]))])
        person = any(float(np.max(s)) >= KEYPOINT_THRESHOLD for s in scores)
        return out, person

    def retinaface(self, bgr):
        if self.retina is None:
            return []
        return [list(b) + [s] for b, _, s in self.retina(bgr, threshold=0.1, resize=1.0)]

    def yolo_face(self, bgr):
        r = self.yolo.predict(bgr, conf=0.1, imgsz=1280, verbose=False, device=self.gpu)[0]
        return [b + [s] for b, s in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]


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


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def padded(b, w: int, h: int, pad: float = PAD):
    dx, dy = (b[2] - b[0]) * pad, (b[3] - b[1]) * pad
    return (max(0, int(b[0] - dx)), max(0, int(b[1] - dy)), min(w, int(b[2] + dx + 1)), min(h, int(b[3] + dy + 1)))


def mask_of(boxes, w: int, h: int, shape: str) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in boxes:
        if shape == "solid":
            # the ellipse through the box corners covers the whole box
            c = ((x1 + x2) // 2, (y1 + y2) // 2)
            ax = (int((x2 - x1) * 0.7072) + 1, int((y2 - y1) * 0.7072) + 1)
            cv2.ellipse(m, c, ax, 0, 0, 360, 255, -1)
        else:
            m[y1:y2, x1:x2] = 255
    return m


def apply_mask(fr: np.ndarray, m: np.ndarray, shape: str, boxes) -> None:
    if shape == "solid":
        fr[m > 0] = (32, 32, 32)
        return
    for x1, y1, x2, y2 in boxes:
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = fr[y1:y2, x1:x2]
        small = cv2.resize(roi, (8, max(1, round(8 * (y2 - y1) / (x2 - x1)))), interpolation=cv2.INTER_AREA)
        big = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        fr[y1:y2, x1:x2] = cv2.GaussianBlur(big, (0, 0), sigmaX=max(3.0, (x2 - x1) / 16))


def process(det: Detectors, src: Path, dst: Path, view: str, shape: str, left_zone: bool) -> dict:
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    zone = tuple(int(v * s) for v, s in zip(LEFT_ZONE, (w / 1920, h / 1080, w / 1920, h / 1080)))
    # pass 1: detections on the original frames
    per = []
    for fr in frames(src):
        d = {"egoblur": det.egoblur(fr), "retinaface": det.retinaface(fr)}
        d["pose_head"], person = det.heads(fr)
        if det.yolo is not None:
            d["yolo_face"] = det.yolo_face(fr)
        mask_boxes = [b for b in d["egoblur"] + d["retinaface"] if b[4] >= FACE_THRESHOLD] + d["pose_head"]
        per.append({"d": d, "boxes": [padded(b, w, h) for b in mask_boxes],
                    "zone": left_zone and view == "Left" and person})
    n = len(per)
    # pass 2: write the masked clip; the coverage check runs on each frame's mask
    tmp = dst.with_suffix(".part.mp4")
    dst.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-", "-an", "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "20", "-movflags", "+faststart",
                           str(tmp)], stdin=subprocess.PIPE)
    masked = flagged = confirmed = 0
    area = 0.0
    flags = []
    try:
        for i, fr in enumerate(frames(src)):
            boxes = [b for j in range(max(0, i - HOLD), min(n, i + HOLD + 1)) for b in per[j]["boxes"]]
            if any(per[j]["zone"] for j in range(max(0, i - HOLD), min(n, i + HOLD + 1))):
                boxes.append(zone)
            m = mask_of(boxes, w, h, "solid" if shape == "solid" else "box")
            masked += bool(boxes)
            area += float((m > 0).mean())
            faces = confirmed_faces(per[i]["d"])
            confirmed += len(faces)
            for f in faces:
                x1, y1, x2, y2 = (max(0, int(f[0])), max(0, int(f[1])), min(w, int(f[2])), min(h, int(f[3])))
                if x2 > x1 and y2 > y1 and (m[y1:y2, x1:x2] > 0).mean() < COVER_MIN:
                    flagged += 1
                    flags.append(round(i / fps, 2))
                    break
            apply_mask(fr, m, shape, boxes)
            ff.stdin.write(fr.tobytes())
    finally:
        ff.stdin.close()
        ff.wait()
    if ff.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed on {src.name}")
    out = dst if flagged == 0 else dst.with_suffix(".flagged.mp4")
    tmp.replace(out)
    return {"clip": dst.name, "view": view, "mask": shape, "frames": n, "frames_masked": masked,
            "mask_area": round(area / max(1, n), 4), "confirmed": confirmed, "flagged_frames": flagged,
            "flag_times_sec": flags[:50], "passed": flagged == 0}


def confirmed_faces(d: dict) -> list:
    """Faces two different detectors agree on, or face detections inside a pose head box."""
    face_methods = [k for k in d if k != "pose_head"]
    boxes = [(k, b) for k in face_methods for b in d[k] if b[4] >= VERIFY_THRESHOLD]
    out = []
    for i, (k, b) in enumerate(boxes):
        agree = any(k2 != k and iou(b, b2) > VERIFY_IOU for k2, b2 in boxes)
        in_head = any(b[0] >= hb[0] and b[1] >= hb[1] and b[2] <= hb[2] and b[3] <= hb[3] for hb in d["pose_head"])
        if agree or in_head:
            out.append(b)
    return out


def worker(a, names: list[str], q) -> None:
    det = Detectors(a.gpu, a.retinaface, verify=True)
    src_dir, dst_dir = Path(a.src_dir), Path(a.dst_dir)
    for name in names:
        dst = dst_dir / name
        meta = dst.with_suffix(".json")
        if meta.exists() and not a.force:
            q.put(json.loads(meta.read_text(encoding="utf-8")))
            continue
        view = a.view or next((v for v in ("RightLow", "Right", "Left", "Top") if f"_{v}." in name), "")
        t0 = time.time()
        row = process(det, src_dir / name, dst, view, a.mask, a.left_zone)
        row["elapsed_sec"] = round(time.time() - t0, 1)
        meta.write_text(json.dumps(row), encoding="utf-8")
        print(json.dumps(row), flush=True)
        q.put(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--dst-dir", required=True)
    ap.add_argument("--only", required=True, help="file listing clip names (one per line)")
    ap.add_argument("--stats-csv", required=True)
    ap.add_argument("--view", default=None, help="camera view (default: from the clip name)")
    ap.add_argument("--mask", choices=["solid", "mosaic"], default="solid")
    ap.add_argument("--retinaface", action="store_true", help="add RetinaFace-R50 boxes to the mask")
    ap.add_argument("--left-zone", action=argparse.BooleanOptionalAction, default=True,
                    help="Left view: mask the bottom-right head zone while pose sees the player")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    names = [n for n in Path(a.only).read_text(encoding="utf-8").split() if (Path(a.src_dir) / n).exists()]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(a, names[i::a.workers], q)) for i in range(a.workers) if names[i::a.workers]]
    for p in procs:
        p.start()
    rows = [q.get() for _ in names]
    for p in procs:
        p.join()
    rows.sort(key=lambda r: names.index(r["clip"]))
    out = Path(a.stats_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} clips, {sum(not r['passed'] for r in rows)} flagged -> {out}")


if __name__ == "__main__":
    main()
