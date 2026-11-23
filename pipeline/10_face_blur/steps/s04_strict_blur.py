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
    region of the labelled Left faces) while pose sees the player (--left-zone),
    minus the player's hands on that frame (a box around each wrist, extended
    along the forearm; when pose loses the player, the hands of the nearest
    frame within +-HOLD), so the zone never hides a hand on the keyboard. The
    same cut applies to the pose head boxes (coarse, and large when the player
    leans in) and to the padding of the face boxes; the face boxes themselves,
    padded only by CORE_PAD and taken from the frames within +-FACE_STRIDE,
    are drawn after the cut and always win.

EgoBlur (the slowest detector by far) may run on every FACE_STRIDE-th frame
only (--face-stride, default 1): the benchmark's held union sampled every
5th frame (offsets 0, +-5, +-10, +-15) and covered 99.5% of faces, and pose
and the verifier still run on every frame.

Every box is padded by PAD of its size on each side, and a frame is masked
with the union of the boxes of the frames within +-HOLD. The mask is either
a pixel mosaic of MOSAIC_CELL px cells (--mask mosaic, default; --mosaic-cell) or an opaque fill
(--mask solid); face and head regions are the ellipse inscribed in the padded box, the zone a box.

Coverage check, on the original frames against the mask: a face is
"confirmed" where a face detection lies inside a pose head box of the frame, or
where two different detectors agree (score >= 0.5, IoU > 0.3) and the face
centre lies in a padded pose head box of the frames within +-HOLD; when pose
sees no head in that window, agreement alone confirms. (Two detectors also
agree on marker-taped shoulders; such boxes are nowhere near a head.) The
verifier detectors are the masking ones plus YOLOv8-face (VERIFY_EXTRA). A frame is flagged when a
confirmed face is less than COVER_MIN inside the mask. Per clip ({clip}.json
and one row of --stats-csv): frames, frames_masked, mask_area, confirmed,
flagged_frames, passed (no flagged frame). A flagged clip keeps its output
under {clip}.flagged.mp4 for review.

--heads-win: the pose head boxes are drawn after the hand cut-outs, so a hand raised to the face never opens
the head (side views, where a player touching the face showed a cheek through the cut).

--arm-guard: a face detection that lies on the player's upper arm or forearm (pose shoulder-elbow-wrist),
outside every pose head box of the frames within +-HOLD, is neither masked nor checked (a low side view sees
the raised upper arm large, and two detectors both take it for a face). While pose sees no head within
+-HOLD nothing is dropped. Recorded per clip (arm_guard, faces_dropped_on_arms).

--fixed-zone X1,Y1,X2,Y2 (per camera, recorded in the clip JSON): a box masked in every frame, with no
hand cut-outs (onlookers' hands would open it; place it clear of the keyboard), for a wide view where onlookers sit in the background (220819 Left).

--hand-priority (per clip, recorded in the clip JSON): a face box that overlaps a hand box while pose
sees no head within +-HOLD is dropped; for Top clips where EgoBlur takes a raised forearm for a face.

--head-shape: SAM 2 (ultralytics, --sam-weights) segments each pose head box, and only the segment (grown
by SEG_DILATE px, kept inside the box) is masked instead of the whole box; a segment covering less than
SEG_MIN of its box falls back to the box. Face detector boxes stay boxes.

--face-labels: the reviewer's labels of the face scan (s05_face_scan.py, src/face_labels.py) decide per clip.
A clip whose scan found nothing, or whose episodes are all labelled and none is a face, is copied unmasked
(label_gate no_hits / no_face). A clip with an unlabelled episode is skipped (label_gate unlabelled, no output).
In a clip with a face episode, each face episode's box is masked from one second before it to one second after
it, and a detector box matching an episode labelled not a face is neither masked nor checked. A clip with no
scan is masked as without the option (label_gate no_scan).

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
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import face_labels as FL  # noqa: E402

MODEL_PATH = Path("pipeline/10_face_blur/models/ego_blur_face.jit")
RTMO_URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/"
            "rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip")
YOLO_FACE = ("arnabdhar/YOLOv8-Face-Detection", "model.pt")
FACE_THRESHOLD = 0.5
KEYPOINT_THRESHOLD = 0.3
PAD = 0.4
CORE_PAD = 0.1                            # the part of a face box a hand cut never removes
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
        self.sam = None                                    # --head-shape, loaded on first use

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
        """(head boxes with a score, hand boxes, whether a person is seen, the player's arm segments).

        Hand boxes are the player's only (the person whose confident keypoints span the largest area): an
        onlooker's raised hand must not open the mask over the onlooker's face."""
        kps, scores = self.pose(bgr)
        out, hands, arms = [], [], []

        def extent(k, s):
            pts = k[s >= KEYPOINT_THRESHOLD]
            return float(np.prod(pts.max(0) - pts.min(0))) if len(pts) > 1 else 0.0
        player = max(range(len(kps)), key=lambda i: extent(kps[i], scores[i]), default=None)
        for k, s in ([(kps[player], scores[player])] if player is not None else []):
            for a, b in ((5, 7), (7, 9), (6, 8), (8, 10)):                  # shoulder-elbow, elbow-wrist
                if min(s[a], s[b]) >= KEYPOINT_THRESHOLD:
                    arms.append((k[a].astype(float), k[b].astype(float)))
            for wi, ei in ((9, 7), (10, 8)):                                 # wrists, elbows (COCO)
                if s[wi] < KEYPOINT_THRESHOLD:
                    continue
                w = k[wi]
                if s[ei] >= KEYPOINT_THRESHOLD:
                    c, r = w + 0.35 * (w - k[ei]), max(0.6 * float(np.linalg.norm(w - k[ei])), 80.0)
                else:
                    c, r = w, 140.0
                hands.append((int(c[0] - r), int(c[1] - r), int(c[0] + r), int(c[1] + r)))
                hands.append((int(w[0] - 40), int(w[1] - 40), int(w[0] + 40), int(w[1] + 40)))
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
        return out, hands, person, arms

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


def padded(b, w: int, h: int, pad: float | None = None):
    pad = PAD if pad is None else pad
    dx, dy = (b[2] - b[0]) * pad, (b[3] - b[1]) * pad
    return (max(0, int(b[0] - dx)), max(0, int(b[1] - dy)), min(w, int(b[2] + dx + 1)), min(h, int(b[3] + dy + 1)))


def mask_of(faces, cores, heads, zone, hands, w: int, h: int, shape: str, head_mask=None) -> np.ndarray:
    """Face boxes as the ellipse inscribed in the padded box (solid) or the box (mosaic); the zone as a box.

    The inscribed ellipse still holds the whole unpadded face box: with PAD = 0.4 on each side the face is
    1/1.8 of the padded box, inside the ellipse's inner rectangle of 0.707.
    With --head-shape, `head_mask` (the segmented heads) is drawn in place of the head boxes.
    """
    m = np.zeros((h, w), np.uint8)

    def draw(boxes):
        for x1, y1, x2, y2 in boxes:
            if shape == "solid":
                cv2.ellipse(m, ((x1 + x2) // 2, (y1 + y2) // 2), ((x2 - x1) // 2 + 1, (y2 - y1) // 2 + 1),
                            0, 0, 360, 255, -1)
            else:
                m[y1:y2, x1:x2] = 255

    if zone is not None:
        m[zone[1]:zone[3], zone[0]:zone[2]] = 255
    draw_heads = (lambda: np.maximum(m, head_mask, out=m)) if head_mask is not None else (lambda: draw(heads))
    draw_heads()
    draw(faces)
    for x1, y1, x2, y2 in hands:                             # no padding, head box or zone hides a hand
        m[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 0
    for x1, y1, x2, y2 in cores:                             # the faces themselves always stay masked
        m[y1:y2, x1:x2] = 255
    if HEADS_WIN:                                            # --heads-win: a hand never opens a head box
        draw_heads()
    return m


HEADS_WIN = False         # --heads-win
HEAD_SHAPE = False        # --head-shape
SAM_WEIGHTS = "sam2.1_b.pt"
SEG_DILATE = 15           # px grown around a head segment
SEG_MIN = 0.15            # a segment below this share of its box falls back to the whole box
LABEL_PAD = 0.3          # --face-labels: margin of a labelled face episode box


def merge_boxes(boxes) -> list:
    """Union of overlapping boxes (the held head boxes of +-HOLD frames are one head)."""
    out = [list(b[:4]) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                if min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
                    out[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del out[j]
                    changed = True
                    break
            if changed:
                break
    return out


def head_segments(det, fr: np.ndarray, boxes: list) -> np.ndarray:
    """--head-shape: SAM 2 segments each (merged) head box; the mask is the segment, grown by SEG_DILATE px and
    kept inside its box. A box whose segment covers less than SEG_MIN of it is masked whole."""
    h, w = fr.shape[:2]
    m = np.zeros((h, w), np.uint8)
    boxes = merge_boxes(boxes)
    if not boxes:
        return m
    if det.sam is None:
        from ultralytics import SAM
        det.sam = SAM(SAM_WEIGHTS)
    r = det.sam(fr, bboxes=[[float(v) for v in b] for b in boxes], verbose=False, device=det.gpu)[0]
    segs = r.masks.data.cpu().numpy() if r.masks is not None else []
    kernel = np.ones((2 * SEG_DILATE + 1, 2 * SEG_DILATE + 1), np.uint8)
    for k, (x1, y1, x2, y2) in enumerate(tuple(int(v) for v in b) for b in boxes):
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        seg = cv2.resize(segs[k].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) if k < len(segs) else None
        box = np.zeros((h, w), np.uint8)
        box[y1:y2, x1:x2] = 1
        if seg is None or (seg & box).sum() < SEG_MIN * box.sum():
            m[y1:y2, x1:x2] = 255
            continue
        m[(cv2.dilate(seg, kernel) & box) > 0] = 255
    return m
MOSAIC_CELL = 48          # pixels per mosaic cell (--mosaic-cell); 24 px still showed hair and face outline


def apply_mask(fr: np.ndarray, m: np.ndarray, shape: str) -> None:
    if not m.any():
        return
    if shape == "solid":
        fr[m > 0] = (32, 32, 32)
        return
    h, w = fr.shape[:2]
    small = cv2.resize(fr, (max(1, w // MOSAIC_CELL), max(1, h // MOSAIC_CELL)), interpolation=cv2.INTER_AREA)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    fr[m > 0] = big[m > 0]


def process(det: Detectors, src: Path, dst: Path, view: str, shape: str, left_zone: bool,
            face_stride: int = 1, hand_priority: bool = False, fixed_zone=None, arm_guard: bool = False,
            labels: dict | None = None) -> dict:
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    scale = (w / 1920, h / 1080, w / 1920, h / 1080)
    zone = tuple(int(v * s) for v, s in zip(LEFT_ZONE, scale))
    fixed = tuple(int(v * s) for v, s in zip(fixed_zone, scale)) if fixed_zone else None
    # pass 1: detections on the original frames
    per = []
    t0 = time.time()
    for fr in frames(src):
        if len(per) % 600 == 0:
            print(f"PROGRESS {dst.name} detect {len(per)} frames {time.time() - t0:.0f}s", flush=True)
        on = len(per) % face_stride == 0
        d = {"egoblur": det.egoblur(fr) if on else [], "retinaface": det.retinaface(fr) if on else []}
        d["pose_head"], hands, person, arms = det.heads(fr)
        if det.yolo is not None:
            d["yolo_face"] = det.yolo_face(fr)
        face_boxes = [b for b in d["egoblur"] + d["retinaface"] if b[4] >= FACE_THRESHOLD]
        per.append({"d": d, "faces": [padded(b, w, h) for b in face_boxes],
                    "cores": [padded(b, w, h, CORE_PAD) for b in face_boxes],
                    "heads": [padded(b, w, h) for b in d["pose_head"]], "hands": hands, "arms": arms,
                    "zone": left_zone and view == "Left" and person})
    n = len(per)
    dropped = 0
    if hand_priority:
        # a face box on a hand while pose sees no head within +-HOLD is a hand, not a face
        for i, f in enumerate(per):
            near = range(max(0, i - HOLD), min(n, i + HOLD + 1))
            if not f["faces"] or any(per[j]["heads"] for j in near):
                continue
            keep = [k for k, b in enumerate(f["cores"])
                    if not any(min(b[2], hb[2]) > max(b[0], hb[0]) and min(b[3], hb[3]) > max(b[1], hb[1]) for hb in f["hands"])]
            dropped += len(f["faces"]) - len(keep)
            f["faces"], f["cores"] = [f["faces"][k] for k in keep], [f["cores"][k] for k in keep]
    arm_dropped = 0
    if arm_guard:
        for i, f in enumerate(per):
            if not f["faces"]:
                continue
            keep = [k for k, b in enumerate(f["cores"]) if not on_arm(b, per, i)]
            arm_dropped += len(f["faces"]) - len(keep)
            f["faces"], f["cores"] = [f["faces"][k] for k in keep], [f["cores"][k] for k in keep]
    labelled_off = labelled_on = 0
    if labels is not None:
        g = FL.GAP_SEC * fps
        notface = lambda b, i: any(e["start"] - g <= i <= e["end"] + g and FL.near(b, e["box"])
                                   for e in labels["non_face_episodes"])
        for i, f in enumerate(per):
            keep = [k for k, b in enumerate(f["cores"]) if not notface(b, i)]
            labelled_off += len(f["faces"]) - len(keep)
            f["faces"], f["cores"] = [f["faces"][k] for k in keep], [f["cores"][k] for k in keep]
        for e in labels["face_episodes"]:
            for i in range(max(0, int(e["start"] - g)), min(n, int(e["end"] + g) + 1)):
                b = padded(e["box"], w, h, max(PAD, LABEL_PAD))   # the face moves within an episode
                per[i]["faces"].append(b)
                per[i]["cores"].append(padded(e["box"], w, h, CORE_PAD))
                labelled_on += 1
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
            if i % 600 == 0:
                print(f"PROGRESS {dst.name} write {i}/{n} frames {time.time() - t0:.0f}s", flush=True)
            near = range(max(0, i - HOLD), min(n, i + HOLD + 1))
            faces = [b for j in near for b in per[j]["faces"]]
            heads = [b for j in near for b in per[j]["heads"]]
            z = zone if any(per[j]["zone"] for j in near) else None
            # pose often loses the player for a few frames (upside-down, occluded): take the hands of the
            # nearest frame that has them
            hands = next((per[j]["hands"] for j in sorted(near, key=lambda j: abs(j - i)) if per[j]["hands"]), [])
            # face cores of the frames within +-face_stride: with a stride, most frames have no EgoBlur box of
            # their own, and a hand cut next to the face would otherwise eat into it
            cores = [b for j in range(max(0, i - face_stride), min(n, i + face_stride + 1)) for b in per[j]["cores"]]
            hm = head_segments(det, fr, heads) if HEAD_SHAPE else None
            m = mask_of(faces, cores, heads, z, hands, w, h, shape, hm)
            if fixed:                                    # onlookers' own hands must not open it: no cut-outs
                m[fixed[1]:fixed[3], fixed[0]:fixed[2]] = 255
            masked += bool(m.any())
            area += float((m > 0).mean())
            seen = confirmed_faces(per[i]["d"], heads)
            if arm_guard:
                seen = [b for b in seen if not on_arm(b, per, i)]
            if labels is not None:
                seen = [b for b in seen if not notface(b, i)]
            confirmed += len(seen)
            for f in seen:
                x1, y1, x2, y2 = (max(0, int(f[0])), max(0, int(f[1])), min(w, int(f[2])), min(h, int(f[3])))
                if x2 > x1 and y2 > y1 and (m[y1:y2, x1:x2] > 0).mean() < COVER_MIN:
                    flagged += 1
                    flags.append(round(i / fps, 2))
                    break
            apply_mask(fr, m, shape)
            ff.stdin.write(fr.tobytes())
    finally:
        ff.stdin.close()
        ff.wait()
    if ff.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed on {src.name}")
    out = dst if flagged == 0 else dst.with_suffix(".flagged.mp4")
    tmp.replace(out)
    return {"clip": dst.name, "view": view, "mask": shape, "mosaic_cell": MOSAIC_CELL if shape == "mosaic" else None, "frames": n, "frames_masked": masked,
            "mask_area": round(area / max(1, n), 4), "confirmed": confirmed, "flagged_frames": flagged,
            "flag_times_sec": flags[:50], "passed": flagged == 0,
            **({"fixed_zone": list(fixed_zone)} if fixed_zone else {}),
            **({"hand_priority": True, "faces_dropped_on_hands": dropped} if hand_priority else {}),
            **({"arm_guard": True, "faces_dropped_on_arms": arm_dropped} if arm_guard else {}),
            **({"head_shape": True} if HEAD_SHAPE else {}),
            **({"label_gate": "face", "face_boxes_dropped_by_labels": labelled_off,
                "face_episode_box_frames": labelled_on} if labels is not None else {})}


ARM_PX = 60.0            # --arm-guard: a box centre this close to an arm segment (or 0.25 x its length) is on it


def on_arm(b, per: list, i: int) -> bool:
    """--arm-guard: a face box is the player's arm when pose sees the head within +-HOLD, the box centre lies
    in none of those (unpadded) pose head boxes, and it lies on a shoulder-elbow or elbow-wrist segment of
    the nearest frame with arms. With no pose head within +-HOLD it is never dropped."""
    n = len(per)
    near = range(max(0, i - HOLD), min(n, i + HOLD + 1))
    heads = [hb for j in near for hb in per[j]["d"]["pose_head"]]
    if not heads:
        return False
    c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], float)
    if any(hb[0] <= c[0] <= hb[2] and hb[1] <= c[1] <= hb[3] for hb in heads):
        return False
    arms = next((per[j]["arms"] for j in sorted(near, key=lambda j: abs(j - i)) if per[j]["arms"]), [])
    for a, e in arms:
        ab = e - a
        L = float(np.linalg.norm(ab))
        t = float(np.clip(np.dot(c - a, ab) / max(L * L, 1e-6), 0, 1))
        if np.linalg.norm(c - (a + t * ab)) <= max(ARM_PX, 0.25 * L):
            return True
    return False


def confirmed_faces(d: dict, heads: list) -> list:
    """Face detections inside a pose head box, or faces two different detectors agree on whose centre
    lies in one of `heads` (the padded pose head boxes of the frames within +-HOLD; agreement alone
    when `heads` is empty)."""
    face_methods = [k for k in d if k != "pose_head"]
    boxes = [(k, b) for k in face_methods for b in d[k] if b[4] >= VERIFY_THRESHOLD]
    out = []
    for i, (k, b) in enumerate(boxes):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        agree = any(k2 != k and iou(b, b2) > VERIFY_IOU for k2, b2 in boxes) and (
            not heads or any(hb[0] <= cx <= hb[2] and hb[1] <= cy <= hb[3] for hb in heads))
        in_head = any(b[0] >= hb[0] and b[1] >= hb[1] and b[2] <= hb[2] and b[3] <= hb[3] for hb in d["pose_head"])
        if agree or in_head:
            out.append(b)
    return out


def worker(a, names: list[str], q) -> None:
    global MOSAIC_CELL, HEADS_WIN, PAD, CORE_PAD, HOLD, HEAD_SHAPE, SAM_WEIGHTS
    MOSAIC_CELL, HEADS_WIN, PAD, CORE_PAD, HOLD = a.mosaic_cell, a.heads_win, a.pad, a.core_pad, a.hold
    HEAD_SHAPE, SAM_WEIGHTS = a.head_shape, a.sam_weights
    det = Detectors(a.gpu, a.retinaface, verify=True)
    src_dir, dst_dir = Path(a.src_dir), Path(a.dst_dir)
    for name in names:
        dst = dst_dir / name
        meta = dst.with_suffix(".json")
        if meta.exists() and not a.force:
            q.put(json.loads(meta.read_text(encoding="utf-8")))
            continue
        view = a.view or next((v for v in ("RightLow", "Right", "Left", "Top") if f"_{v}." in name), "")
        print(f"START {name}", flush=True)
        t0 = time.time()
        fz = tuple(int(v) for v in a.fixed_zone.split(",")) if a.fixed_zone else None
        labels, gate = None, None
        if a.face_labels:
            ds = name.split("_")[0]
            scan_f = Path(a.face_scan_root) / ds / "face_scan" / f"{Path(name).stem}.json"
            if not scan_f.exists():
                gate = "no_scan"
            else:
                dec = FL.clip_decision(json.loads(scan_f.read_text(encoding="utf-8")),
                                       FL.read_labels(ds, Path(a.metadata_root)))
                gate = dec["status"]
                if gate == "unlabelled":
                    print(f"SKIP {name}: {dec['episodes'] - dec['labelled']} face-scan episodes not labelled", flush=True)
                    q.put({"clip": name, "view": view, "label_gate": gate, "passed": False, "skipped": True})
                    continue
                if gate in ("no_hits", "no_face"):
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src_dir / name, dst.with_suffix(".part.mp4"))
                    dst.with_suffix(".part.mp4").replace(dst)
                    row = {"clip": name, "view": view, "mask": None, "frames_masked": 0, "mask_area": 0.0,
                           "flagged_frames": 0, "passed": True, "label_gate": gate,
                           "face_scan_episodes": dec["episodes"]}
                    meta.write_text(json.dumps(row), encoding="utf-8")
                    print("DONE " + json.dumps(row), flush=True)
                    q.put(row)
                    continue
                labels = dec
        row = process(det, src_dir / name, dst, view, a.mask, a.left_zone, a.face_stride, a.hand_priority, fz,
                      a.arm_guard, labels)
        if gate == "no_scan":
            row["label_gate"] = gate
        row["face_stride"] = a.face_stride
        row.update(pad=a.pad, core_pad=a.core_pad, hold=a.hold)
        row["elapsed_sec"] = round(time.time() - t0, 1)
        meta.write_text(json.dumps(row), encoding="utf-8")
        print("DONE " + json.dumps(row), flush=True)
        q.put(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--dst-dir", required=True)
    ap.add_argument("--only", required=True, help="file listing clip names (one per line)")
    ap.add_argument("--stats-csv", required=True)
    ap.add_argument("--view", default=None, help="camera view (default: from the clip name)")
    ap.add_argument("--mask", choices=["solid", "mosaic"], default="mosaic")
    ap.add_argument("--mosaic-cell", type=int, default=MOSAIC_CELL, help="mosaic cell size in pixels")
    ap.add_argument("--face-stride", type=int, default=1, help="run EgoBlur / RetinaFace on every n-th frame")
    ap.add_argument("--retinaface", action="store_true", help="add RetinaFace-R50 boxes to the mask")
    ap.add_argument("--left-zone", action=argparse.BooleanOptionalAction, default=True,
                    help="Left view: mask the bottom-right head zone while pose sees the player")
    ap.add_argument("--heads-win", action="store_true",
                    help="pose head boxes are drawn after the hand cut-outs (a hand raised to the face cannot open it)")
    ap.add_argument("--fixed-zone", default=None, metavar="X1,Y1,X2,Y2",
                    help="a box (1920x1080 pixels) masked in every frame, no hand cut-outs, e.g. where onlookers sit")
    ap.add_argument("--hand-priority", action="store_true",
                    help="drop a face box that overlaps a hand while pose sees no head within +-HOLD (per-clip fix)")
    ap.add_argument("--pad", type=float, default=PAD, help="margin added on each side of face and pose head boxes")
    ap.add_argument("--core-pad", type=float, default=CORE_PAD, help="margin of the face core no hand cut removes")
    ap.add_argument("--hold", type=int, default=HOLD, help="boxes of the frames within +-HOLD are merged into a frame's mask")
    ap.add_argument("--arm-guard", action="store_true",
                    help="drop a face box on the player's arm while pose sees the head elsewhere (mask and check)")
    ap.add_argument("--head-shape", action="store_true",
                    help="mask the SAM 2 segment inside each pose head box instead of the whole box")
    ap.add_argument("--sam-weights", default=SAM_WEIGHTS, help="SAM 2 checkpoint for --head-shape")
    ap.add_argument("--face-labels", action="store_true",
                    help="gate each clip by the reviewer's face-scan labels (see the docstring)")
    ap.add_argument("--face-scan-root", default="Data/Pipeline/10_face_blur", help="{root}/{dataset}/face_scan/")
    ap.add_argument("--metadata-root", default="Data/Metadata", help="{root}/{dataset}/_face_labels.json")
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
    import queue
    rows = []
    while len(rows) < len(names):
        try:
            rows.append(q.get(timeout=60))
        except queue.Empty:
            if not any(p.is_alive() for p in procs):         # a worker died: do not wait forever
                raise SystemExit(f"workers exited with {len(rows)}/{len(names)} clips done; see the log above")
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
