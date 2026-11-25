"""Face-detection episodes and the reviewer's labels for them (Stage 10).

The face scan (steps/s05_face_scan.py) runs face detectors on a clip and records every hit. Hits of one clip
that overlap in the image (or lie within one box size of each other) and follow each other within GAP_SEC are
one *episode*: one thing the detectors kept seeing. The reviewer labels episodes in the review dashboard.

Labels live in Data/Metadata/{dataset}/_face_labels.json:

    {"dataset": "240828", "classes": [...],
     "clips": {"<clip stem>": [{"start": 120, "end": 150, "box": [x1, y1, x2, y2], "label": "face",
                                "at": "2026-11-23 17:00:00"}, ...]}}

A label is stored with the frame range and box of the episode it was given to, not with an episode number, so
it survives a re-scan: a new episode takes the label of a stored one when their frame ranges overlap and
their boxes are near (`label_of`); a label never reaches an episode it does not overlap in time, so an episode
nobody saw stays unlabelled. Where two stored labels match, a face label wins.

Classes: face (the player), other_face (anyone else), arm_hand, other (clothing, objects). FACE_CLASSES are
the ones that must be masked.

Corner check: in a view where every labelled face lies in one corner (`face_zone`, the box around all face
labels of that view), a reviewer also looks at that corner of the clips the labels would release unmasked,
every CORNER_STEP_SEC, and marks the times a face shows there ("corner": {"<clip stem>": [{"t": 3.5, ...}]}).
A marked clip counts as a face clip.

Masking a face clip with the zone (s04 --face-labels): only `face_zone` is masked, from ZONE_SEC before to
ZONE_SEC after each face episode and each corner mark (`zone_windows`).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

GAP_SEC = 1.0
ZONE_SEC = 2.0
CORNER_STEP_SEC = 0.5
ZONE_EDGE_PX = 64          # a zone side this close to the frame edge is moved onto it
CLASSES = ("face", "other_face", "arm_hand", "other")
FACE_CLASSES = ("face", "other_face")
METADATA_ROOT = Path("Data/Metadata")


def labels_path(dataset: str, root: Path = METADATA_ROOT) -> Path:
    return root / dataset / "_face_labels.json"


def read_labels(dataset: str, root: Path = METADATA_ROOT) -> dict:
    f = labels_path(dataset, root)
    d = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    d.setdefault("dataset", dataset)
    d["classes"] = list(CLASSES)
    d.setdefault("clips", {})
    d.setdefault("corner", {})
    d.setdefault("corner_checked", {})      # clip -> time the reviewer checked its corner strip
    return d


def write_labels(d: dict, root: Path = METADATA_ROOT) -> None:
    f = labels_path(d["dataset"], root)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
    os.replace(tmp, f)


def near(a, b) -> bool:
    """Boxes overlap, or their centres lie within the larger box size of each other on both axes."""
    if min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
        return True
    s = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
    return abs((a[0] + a[2]) - (b[0] + b[2])) / 2 < s and abs((a[1] + a[3]) - (b[1] + b[3])) / 2 < s


def episodes(scan: dict) -> list[dict]:
    """Group a scan's hits into episodes, in order of their first frame.

    Each: id (clip#k, display only), start, end (frames), box (of the strongest hit), n, max_score, dets, and
    samples: the strongest hit, then the first / middle / last ones (at most 4 hits)."""
    clip, fps = scan["clip"].rsplit(".", 1)[0], scan["fps"]
    eps: list[dict] = []
    for h in sorted(scan["hits"], key=lambda h: h["frame"]):
        for e in eps:
            if h["frame"] - e["last"] <= GAP_SEC * fps and near(h["box"], e["cur"]):
                e["hits"].append(h)
                e["last"], e["cur"] = h["frame"], h["box"]
                break
        else:
            eps.append({"hits": [h], "last": h["frame"], "cur": h["box"]})
    out = []
    for k, e in enumerate(eps):
        hs = e["hits"]
        best = max(hs, key=lambda h: h["score"])
        rest = [hs[i] for i in sorted({0, len(hs) // 2, len(hs) - 1}) if hs[i] is not best]
        out.append({"id": f"{clip}#{k}", "clip": clip, "fps": fps, "start": hs[0]["frame"], "end": hs[-1]["frame"],
                    "box": best["box"], "n": len(hs), "max_score": best["score"],
                    "dets": sorted({h["det"] for h in hs}), "samples": [best] + rest[:3]})
    return out


def same(s: dict, ep: dict) -> bool:
    """A stored label belongs to an episode: frame ranges overlap and boxes are near."""
    return s["start"] <= ep["end"] and ep["start"] <= s["end"] and near(s["box"], ep["box"])


def label_of(ep: dict, stored: list[dict], fps: float | None = None) -> str | None:
    """The stored label of an episode (see `same`); None if none."""
    hit = [s["label"] for s in stored if same(s, ep)]
    if not hit:
        return None
    return next((c for c in FACE_CLASSES if c in hit), hit[0])


def set_label(d: dict, ep: dict, label: str | None) -> None:
    """Store (or with None remove) the label of an episode; replaces the stored labels it matches."""
    stored = d["clips"].setdefault(ep["clip"], [])
    keep = [s for s in stored if not same(s, ep)]
    if label is not None:
        if label not in CLASSES:
            raise ValueError(f"unknown class {label}")
        keep.append({"start": ep["start"], "end": ep["end"], "box": list(ep["box"]), "label": label,
                     "at": time.strftime("%Y-%m-%d %H:%M:%S")})
    d["clips"][ep["clip"]] = sorted(keep, key=lambda s: s["start"])


def view_of(clip: str) -> str:
    return next((v for v in ("RightLow", "Right", "Left", "Top") if clip.endswith("_" + v)), "")


def face_zone(d: dict, view: str, w: int = 1920, h: int = 1080) -> list[int] | None:
    """The box around every face-labelled box of a view (pixels of a w x h frame); None without face labels."""
    b = [s["box"] for clip, ss in d["clips"].items() if view_of(clip) == view for s in ss
         if s["label"] in FACE_CLASSES]
    if not b:
        return None
    x1, y1 = min(v[0] for v in b), min(v[1] for v in b)
    x2, y2 = max(v[2] for v in b), max(v[3] for v in b)
    snap = lambda v, lo, hi: lo if v - lo < ZONE_EDGE_PX else hi if hi - v < ZONE_EDGE_PX else v
    return [int(snap(x1, 0, w)), int(snap(y1, 0, h)), int(snap(x2, 0, w)), int(snap(y2, 0, h))]


def zone_windows(dec: dict, fps: float) -> list[tuple[int, int]]:
    """Frame ranges to mask the zone in: each face episode and corner mark, widened by ZONE_SEC."""
    g = ZONE_SEC * fps
    spans = [(e["start"], e["end"]) for e in dec["face_episodes"]] + [(t * fps, t * fps) for t in dec["corner_times"]]
    return [(max(0, int(a - g)), int(b + g)) for a, b in spans]


def clip_decision(scan: dict, d: dict) -> dict:
    """What the labels say about a scanned clip.

    status: no_hits (the detectors saw nothing), no_face (every episode labelled, none a face), face (some
    episode is a face, or a corner mark), unlabelled (some episode has no label yet). Plus the counts, the
    face episodes and the corner marks (seconds)."""
    eps = episodes(scan)
    clip = scan["clip"].rsplit(".", 1)[0]
    stored = d["clips"].get(clip, [])
    corner = sorted(m["t"] for m in d.get("corner", {}).get(clip, []))
    labs = [label_of(e, stored, scan["fps"]) for e in eps]
    faces = [e for e, lab in zip(eps, labs) if lab in FACE_CLASSES]
    if faces or corner:
        status = "face"
    elif not eps:
        status = "no_hits"
    elif any(lab is None for lab in labs):
        status = "unlabelled"
    else:
        status = "no_face"
    return {"status": status, "episodes": len(eps), "labelled": sum(lab is not None for lab in labs),
            "corner_times": corner,
            "face_episodes": [{"start": e["start"], "end": e["end"], "box": e["box"]} for e in faces],
            "non_face_episodes": [{"start": e["start"], "end": e["end"], "box": e["box"]}
                                  for e, lab in zip(eps, labs) if lab is not None and lab not in FACE_CLASSES]}
