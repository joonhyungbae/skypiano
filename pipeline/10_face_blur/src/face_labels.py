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
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

GAP_SEC = 1.0
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


def clip_decision(scan: dict, d: dict) -> dict:
    """What the labels say about a scanned clip.

    status: no_hits (the detectors saw nothing), no_face (every episode labelled, none a face), face (some
    episode is a face), unlabelled (some episode has no label yet). Plus the counts and the face episodes."""
    eps = episodes(scan)
    stored = d["clips"].get(scan["clip"].rsplit(".", 1)[0], [])
    labs = [label_of(e, stored, scan["fps"]) for e in eps]
    faces = [e for e, lab in zip(eps, labs) if lab in FACE_CLASSES]
    if not eps:
        status = "no_hits"
    elif faces:
        status = "face"
    elif any(lab is None for lab in labs):
        status = "unlabelled"
    else:
        status = "no_face"
    return {"status": status, "episodes": len(eps), "labelled": sum(lab is not None for lab in labs),
            "face_episodes": [{"start": e["start"], "end": e["end"], "box": e["box"]} for e in faces],
            "non_face_episodes": [{"start": e["start"], "end": e["end"], "box": e["box"]}
                                  for e, lab in zip(eps, labs) if lab is not None and lab not in FACE_CLASSES]}
