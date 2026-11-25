"""Stage 10 step 06: corner strips for the reviewer's corner check.

In a view where every labelled face lies in one corner (face_labels.face_zone), the clips the labels would
release unmasked (status no_hits / no_face) are checked by eye: this step cuts that corner out of each clip
every CORNER_STEP_SEC and tiles the crops into one image per clip, so the reviewer can mark the times a face
shows (the review dashboard, Faces page, corner check). Marked clips are masked with --label-zone.

Output: Data/Pipeline/10_face_blur/{dataset}/face_scan/corner/{clip}.jpg (COLS tiles per row, tile k = time
k * CORNER_STEP_SEC) and corner/_index.json ({"zone", "step", "cols", "tile": [w, h], "clips": {clip:
{"tiles", "status"}}}).

Usage: python s06_corner_strips.py --dataset 240828 --view Left [--workers 4]
"""
import argparse
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import face_labels as FL  # noqa: E402

COLS = 10
TILE_W = 260


def strip(src: Path, dst: Path, zone, tile_h: int) -> int:
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(src)], capture_output=True, text=True).stdout.strip() or 0)
    n = max(1, math.ceil(dur / FL.CORNER_STEP_SEC))
    x1, y1, x2, y2 = zone
    vf = (f"fps={1 / FL.CORNER_STEP_SEC},crop={x2 - x1}:{y2 - y1}:{x1}:{y1},scale={TILE_W}:{tile_h},"
          f"tile={COLS}x{math.ceil(n / COLS)}")
    tmp = dst.with_suffix(".part.jpg")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", vf, "-frames:v", "1", "-q:v", "4",
                    str(tmp)], check=True)
    tmp.replace(dst)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--view", default="Left")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pipeline-root", default="Data/Pipeline")
    ap.add_argument("--metadata-root", default="Data/Metadata")
    a = ap.parse_args()
    lab = FL.read_labels(a.dataset, Path(a.metadata_root))
    zone = FL.face_zone(lab, a.view)
    if not zone:
        sys.exit(f"no face labels in {a.dataset} {a.view}")
    scan_dir = Path(a.pipeline_root) / "10_face_blur" / a.dataset / "face_scan"
    out = scan_dir / "corner"
    out.mkdir(parents=True, exist_ok=True)
    src_dir = Path(a.pipeline_root) / "04_piece_split" / a.dataset / "videos"
    todo = []
    for f in sorted(scan_dir.glob(f"*_{a.view}.json")):
        scan = json.loads(f.read_text(encoding="utf-8"))
        st = FL.clip_decision(scan, lab)["status"]
        if st in ("no_hits", "no_face") or (st == "face" and lab["corner"].get(f.stem)):
            todo.append((f.stem, st))
    tile_h = round(TILE_W * (zone[3] - zone[1]) / (zone[2] - zone[0]))
    idx_f = out / "_index.json"
    idx = json.loads(idx_f.read_text(encoding="utf-8")) if idx_f.exists() else {}
    clips = idx.get("clips", {}) if idx.get("zone") == zone else {}

    def one(item):
        stem, st = item
        jpg = out / f"{stem}.jpg"
        n = clips.get(stem, {}).get("tiles") if jpg.exists() else None
        return stem, st, n or strip(src_dir / f"{stem}.mp4", jpg, zone, tile_h)

    with ThreadPoolExecutor(a.workers) as ex:
        for k, (stem, st, n) in enumerate(ex.map(one, todo), 1):
            clips[stem] = {"tiles": n, "status": st}
            if k % 50 == 0:
                print(f"{k}/{len(todo)}", flush=True)
    idx = {"dataset": a.dataset, "view": a.view, "zone": zone, "step": FL.CORNER_STEP_SEC, "cols": COLS,
           "tile": [TILE_W, tile_h], "clips": clips}
    idx_f.write_text(json.dumps(idx, indent=1), encoding="utf-8")
    print(f"{len(todo)} clips -> {out}")


if __name__ == "__main__":
    main()
