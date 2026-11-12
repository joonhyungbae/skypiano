"""
Stage 04 step 11: 240523 per-piece video.

240523 video was exported per trial (Raw/240523/C1/N0/Video/{Top,Left,Right}/
240523sub02NN.mp4) without the deck timecode the 240828 slicer reads, and the
file numbers do not always follow the trials (a recording is missing and the
next one carries its number). Each recording is therefore located in the
session recording (50 Hz log envelopes, NCC >= MIN_NCC) and serves every
trial whose session window it covers by at least MIN_COVER; the offset
(video_t = trial_t + offset) follows from the two positions and is refined at
1 kHz against the 04 trial audio.

With --takes, each trial is cut whole instead ({trial_stem}_{view}.mp4): the
release ships 240523 by take. --reuse-alignment cuts from the stored
_video_alignment.json without locating the recordings again.

Outputs:
    Data/Pipeline/04_piece_split/240523/videos/{piece_stem}_{view}.mp4
    Data/Pipeline/04_piece_split/240523/videos/{trial_stem}_{view}.mp4   (--takes)
    Data/Pipeline/04_piece_split/240523/_video_alignment.json

Usage (from the repository root):
  python pipeline/04_piece_split/steps/s11_240523_videos.py [--align-only] [--force]
  python pipeline/04_piece_split/steps/s11_240523_videos.py --takes --views Top --reuse-alignment
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("s10_221112_videos", HERE / "s10_221112_videos.py")
s10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s10)                     # envelope / best_offset / cutting helpers
sv, vx = s10.sv, s10.vx

DS = "240523"
RAW = Path("Data/Raw") / DS / "C1" / "N0" / "Video"
NORM = Path("Data/Pipeline/01_ingest/Normalized") / DS
PIECES = Path("Data/Pipeline/04_piece_split") / DS
OUT = PIECES / "videos"
ALIGN_JSON = PIECES / "_video_alignment.json"
VIEWS = ["Top", "Left", "Right"]
MIN_NCC = 0.5
MIN_COVER = 0.9                 # share of the trial window a recording must cover
SESSION_WAV = Path("Data/Raw") / DS / "C1" / "N0" / "Audio" / "240523sub02.wav"


def trial_seconds(wav: Path) -> float:
    import wave
    with wave.open(str(wav)) as w:
        return w.getnframes() / w.getframerate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--align-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--views", nargs="+", default=VIEWS, help="cut only these views (default: all)")
    ap.add_argument("--takes", action="store_true", help="cut whole trials instead of pieces")
    ap.add_argument("--reuse-alignment", action="store_true", help="use the stored alignment")
    a = ap.parse_args()
    # trial windows on the session clock (Stage 01 cut the trial audio there)
    window = {}
    for mp in NORM.glob("*/_mapping.json"):
        for t in json.loads(mp.read_text(encoding="utf-8"))["trials"]:
            m = re.search(r"\[([\d.]+)-([\d.]+)\]", (t.get("raw_sources") or {}).get("audio") or "")
            if m and (PIECES / f"{t['trial_stem']}.wav").exists():
                window[t["trial_stem"]] = (float(m.group(1)), float(m.group(2)))
    store = {}
    if a.reuse_alignment and ALIGN_JSON.exists():
        store = json.loads(ALIGN_JSON.read_text(encoding="utf-8"))
    session_env = None if store else s10.envelope(SESSION_WAV)
    for view in ([] if store else VIEWS):
        # each recording is located in the session recording; it serves the
        # trials whose session window it covers (file numbers are not trusted)
        cover: dict[str, tuple[float, Path, Path, float]] = {}
        for src in sorted((RAW / view).glob("*.mp4")):
            audio = vx.extract_audio_mono(src)
            env = s10.envelope(audio)
            start, c = s10.best_offset(env, session_env)          # session_t = video_t + start
            if c < MIN_NCC:
                print(f"  {view:5s} {src.name}: not found in the session (ncc {c:.2f})")
                continue
            end = start + len(env) / vx.ENV_SR
            for st, (a0, a1) in window.items():
                share = (min(end, a1) - max(start, a0)) / (a1 - a0)
                if share >= MIN_COVER and (st not in cover or c > cover[st][0]):
                    cover[st] = (c, src, audio, a0 - start)
        for st in sorted(window):
            if st not in cover:
                store.setdefault(st, {})[view] = {"status": "no_recording"}
                continue
            c, src, audio, o = cover[st]
            f = sv.lag_ncc(PIECES / f"{st}.wav", audio, float(o), sv.FINE_SEARCH_S, sr=sv.FINE_SR, min_overlap_s=5.0)
            rec = {"file": str(src.relative_to(Path("Data/Raw"))).replace("\\", "/"), "ncc": round(float(c), 4),
                   "status": "ok", "coarse_offset_sec": round(float(o), 3),
                   "offset_sec": round(f[0][0], 4) if f else round(float(o), 3)}
            store.setdefault(st, {})[view] = rec
            print(f"  {st} {view:5s} ok {src.name} ncc={c:.2f} off={rec['offset_sec']}")
    if not a.reuse_alignment:
        ALIGN_JSON.write_text(json.dumps(store, indent=1), encoding="utf-8")
    if a.align_only:
        return
    jobs, lengths = [], {}
    for t in json.loads((PIECES / "_pieces.json").read_text(encoding="utf-8")):
        for view, al in store.get(t["trial_stem"], {}).items():
            if view not in a.views or al["status"] != "ok":
                continue
            src = Path("Data/Raw") / al["file"]
            lengths.setdefault(src, sv.video_seconds(src))
            if a.takes:
                start, dur = al["offset_sec"], trial_seconds(PIECES / f"{t['trial_stem']}.wav")
                dst = OUT / f"{t['trial_stem']}_{view}.mp4"
                if start > -0.5 and start + dur <= lengths[src] + 0.5 and (a.force or not dst.exists()):
                    jobs.append(([src], [lengths[src]], max(0.0, start), dur, dst))
                continue
            for p in t["pieces"]:
                start, dur = al["offset_sec"] + p["start_sec"], p["end_sec"] - p["start_sec"]
                dst = OUT / f"{t['trial_stem']}_p{p['piece_idx']}_{view}.mp4"
                if start < -0.5 or start + dur > lengths[src] + 0.5 or (dst.exists() and not a.force):
                    continue
                jobs.append(([src], [lengths[src]], start, dur, dst))
    print(f"cutting {len(jobs)} piece videos")
    with ThreadPoolExecutor(sv.WORKERS) as ex:
        res = list(ex.map(lambda j: (j[4].name, sv.cut(*j)), jobs))
    bad = [n for n, ok in res if not ok]
    print(f"done: {len(res) - len(bad)} ok, {len(bad)} failed {bad[:10]}")


if __name__ == "__main__":
    main()
