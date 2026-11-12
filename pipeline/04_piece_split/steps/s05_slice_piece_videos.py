"""
Standalone CLI tool; not part of the run_pipeline.py loop.

Stage 6 Step 05: Per-piece video slicing for per-piece release samples.

Uses SMPTE timecode alignment (helpers in align_timecode.py next to this file):
for each target sample piece, the motion CSV supplies the trial's SMPTE range,
the piece's trial-local offset comes from `_pieces.json`, and a Deck video is
matched by its embedded SMPTE timecode.

Targets are read from `Data/Metadata/{ds}/web_sample_selection.json` (output of
`tools/select_web_samples.py`). The trial→raw_motion mapping is recovered from
the per-session `_mapping.json` written by Stage 1.

Output: Data/Pipeline/04_piece_split/{ds}/videos/{stem}_{view}.mp4
        (view ∈ Top, Right, RightLow, Left)

--views limits the cut to some views; --force re-cuts existing clips (after a
piece window changed). A view's timecoded file list is read once per session.

Usage (from the repository root):
  python pipeline/04_piece_split/steps/s05_slice_piece_videos.py --ds 240828 --selection SEL.json --views Top --force
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Reuse existing timecode helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))  # align_timecode.py sits beside this file
from align_timecode import (  # type: ignore
    process_motion_csv,
    extract_video_timecode,
    build_deck_video_map,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _common.known_absences import is_view_absent, absence_reason


RAW_ROOT = Path("Data/Raw")
STAGE6_ROOT = Path("Data/Pipeline/04_piece_split")
INGEST_ROOT = Path("Data/Pipeline/01_ingest/Normalized")
METADATA_ROOT = Path("Data/Metadata")


def lookup_raw_motion(ds: str, trial_stem: str) -> tuple[tuple[str, str], str]:
    """Return ((cohort, node), raw_motion_stem) by reading Stage 1 _mapping.json.

    e.g. trial_stem="240828_C0_N5_T2_P7_t1" → (("C0","N5"), "240828sub0528").
    """
    m = re.match(rf"{ds}_(C\d+)_(N\d+)_", trial_stem)
    if not m:
        raise ValueError(f"unparseable trial_stem: {trial_stem}")
    cohort, node = m.group(1), m.group(2)
    mapping = json.loads(
        (INGEST_ROOT / ds / f"{cohort}_{node}" / "_mapping.json").read_text(encoding="utf-8")
    )
    for trow in mapping.get("trials", []):
        if trow.get("trial_stem") == trial_stem:
            raw_stem = trow.get("raw_stem")
            if not raw_stem:
                raise RuntimeError(f"_mapping.json: trial {trial_stem} has no raw_stem")
            return (cohort, node), raw_stem
    raise RuntimeError(f"trial {trial_stem} not found in {ds}/{cohort}_{node}/_mapping.json")


def load_selection_samples(selection_path: Path, ds: str) -> list[dict]:
    """Build SAMPLES from a web_sample_selection.json file."""
    sel = json.loads(selection_path.read_text(encoding="utf-8"))
    out = []
    for s in sel.get("samples", []):
        sess_path, raw_motion = lookup_raw_motion(ds, s["trial_stem"])
        out.append(dict(
            ds=ds, session_path=sess_path,
            trial_stem=s["trial_stem"], raw_motion=raw_motion,
            piece_idx=int(s["piece_idx"]),
        ))
    return out

VIEW_DIRS = ["Top", "Right", "RightLow", "Left"]
VIEWS = list(VIEW_DIRS)          # views to cut (--views)
FORCE = False                    # re-cut existing clips (--force)
_VIDEO_TC: dict[Path, list[tuple[Path, float, float]]] = {}


def timecoded_videos(vdir: Path) -> list[tuple[Path, float, float]]:
    """(path, tc_start, tc_end) of a view's files with readable timecode, sorted; read once per folder."""
    if vdir not in _VIDEO_TC:
        vids = []
        for _take, vpath in build_deck_video_map(vdir):
            tc = extract_video_timecode(vpath)
            d = probe_duration(vpath) if tc is not None else 0.0
            if tc is not None and d > 0:
                vids.append((vpath, tc, tc + d))
        _VIDEO_TC[vdir] = sorted(vids, key=lambda x: x[1])
    return _VIDEO_TC[vdir]


def load_piece_offset(ds: str, trial_stem: str, piece_idx: int) -> tuple[float, float]:
    """Return (start_sec, end_sec) trial-local for the target piece."""
    import json
    pieces = json.load(open(STAGE6_ROOT / ds / "_pieces.json", encoding="utf-8"))
    trow = next((t for t in pieces if t.get("trial_stem") == trial_stem), None)
    if not trow:
        raise RuntimeError(f"trial {trial_stem} not in _pieces.json")
    p = next((pp for pp in (trow.get("pieces") or []) if pp.get("piece_idx") == piece_idx), None)
    if not p:
        raise RuntimeError(f"piece p{piece_idx} not in trial {trial_stem}")
    return float(p["start_sec"]), float(p["end_sec"])


def find_matching_video(videos_with_tc: list[tuple[Path, float, float]],
                        piece_smpte_start: float,
                        piece_smpte_end: float) -> tuple[Path, float] | None:
    """Return (video_path, offset_in_video) for the video that covers the piece.

    videos_with_tc: list of (path, tc_start_sec, tc_end_sec) sorted by tc_start.
    Requires the piece to fit entirely within one file (typical for piece-level
    segments which are short).
    """
    for path, tc_start, tc_end in videos_with_tc:
        if tc_start <= piece_smpte_start and piece_smpte_end <= tc_end:
            return path, piece_smpte_start - tc_start
    return None


def probe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "default=nokey=1:noprint_wrappers=1",
         "-show_entries", "format=duration", str(path)],
        capture_output=True, text=True, timeout=10,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def ffmpeg_extract(src: Path, offset: float, duration: float, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    # Seek before -i for speed, then -c copy for fast + lossless slice.
    # Piece boundaries are not keyframe-aligned; use -avoid_negative_ts to
    # preserve timing, and re-encode with H.264 for web compatibility.
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{offset:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(out),
    ]
    tmp = out.with_suffix(".part.mp4")
    cmd[-1] = str(tmp)
    # 4K sources decode slowly: allow ~20 s of encoding per second of clip
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=max(300.0, 20.0 * duration))
    if r.returncode == 0:
        tmp.replace(out)
        return True
    tmp.unlink(missing_ok=True)
    return False


def process_sample(entry: dict) -> None:
    ds = entry["ds"]
    sess_path = entry["session_path"]     # e.g. ("C0", "N4")
    trial_stem = entry["trial_stem"]
    raw_motion = entry["raw_motion"]
    piece_idx = entry["piece_idx"]

    base = RAW_ROOT / ds / sess_path[0] / sess_path[1]
    motion_csv = base / "Motion" / "Hand" / f"{raw_motion}.csv"
    if not motion_csv.exists():
        print(f"  [SKIP] motion CSV missing: {motion_csv}")
        return

    # 1. trial SMPTE range from motion CSV
    trial_smpte_start, trial_smpte_end, _ = process_motion_csv(motion_csv)

    # 2. piece trial-local offset
    p_start, p_end = load_piece_offset(ds, trial_stem, piece_idx)
    piece_smpte_start = trial_smpte_start + p_start
    piece_smpte_end = trial_smpte_start + p_end
    piece_dur = p_end - p_start

    print(f"\n== {trial_stem}_p{piece_idx} ==")
    print(f"  trial SMPTE {trial_smpte_start:.2f}-{trial_smpte_end:.2f}s")
    print(f"  piece trial-local {p_start:.2f}-{p_end:.2f}s  dur={piece_dur:.2f}s")
    print(f"  piece SMPTE       {piece_smpte_start:.2f}-{piece_smpte_end:.2f}s")

    out_stem = f"{trial_stem}_p{piece_idx}"

    for view in VIEWS:
        if is_view_absent(ds, trial_stem, view):
            print(f"  [{view}] expected-absent: {absence_reason(ds, trial_stem, view)}")
            continue
        vdir = base / "Video" / view
        if not vdir.exists():
            print(f"  [{view}] dir missing")
            continue
        match = find_matching_video(timecoded_videos(vdir), piece_smpte_start, piece_smpte_end)
        if not match:
            print(f"  [{view}] NO match - piece SMPTE not inside any file's TC range")
            continue
        src, offset = match
        out = STAGE6_ROOT / ds / "videos" / f"{out_stem}_{view}.mp4"
        if out.exists() and not FORCE:
            print(f"  [{view}] exists: {out.name}")
            continue
        print(f"  [{view}] {src.name}  offset={offset:.3f}s  → {out.name}")
        ok = ffmpeg_extract(src, offset, piece_dur, out)
        print(f"    {'OK' if ok else 'FAILED'}  ({out.stat().st_size/1024/1024:.1f}MB)" if ok else "    ffmpeg failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="240828",
                    help="dataset code (default: 240828)")
    ap.add_argument("--selection", default=None,
                    help="path to web_sample_selection.json "
                         "(default: Data/Metadata/<ds>/web_sample_selection.json)")
    ap.add_argument("--views", nargs="+", default=VIEW_DIRS, help="cut only these views (default: all)")
    ap.add_argument("--force", action="store_true", help="re-cut existing clips")
    ap.add_argument("--workers", type=int, default=1, help="pieces cut in parallel")
    args = ap.parse_args()
    global VIEWS, FORCE
    VIEWS, FORCE = args.views, args.force

    sel_path = Path(args.selection) if args.selection \
        else METADATA_ROOT / args.ds / "web_sample_selection.json"
    samples = load_selection_samples(sel_path, args.ds)

    print(f"Loaded {len(samples)} sample targets from {sel_path}")

    def run(entry):
        try:
            process_sample(entry)
        except Exception as e:
            print(f"  [ERROR] {entry.get('trial_stem')}: {e}")

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(run, samples))


if __name__ == "__main__":
    main()
