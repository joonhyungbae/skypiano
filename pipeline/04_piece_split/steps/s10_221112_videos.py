"""
Stage 04 step 10: 221112 per-piece video.

221112 video is a pool of per-take recordings for all subjects
(Raw/221112/C1/Video_1080p/Top: GoPro, re-encoded without creation times;
Right: a few Atomos takes). Every trial with audio (Stage 01) is matched to
the recording whose audio contains it:

  score[trial, recording] = best NCC of the 50 Hz log envelopes over every
  offset where one contains the other (video_t = trial_t + offset), for
  recordings covering at least COVER of the trial (a per-take camera records
  the whole take; a short clip matches somewhere inside any long trial);
  recordings and trials are paired one to one (Hungarian on the score). A
  pair is used when its NCC >= MIN_NCC and it beats the trial's next best
  recording by MARGIN; the offset is refined at 1 kHz.

Every piece of `_pieces.json` is then cut from each matched view:

    Data/Pipeline/04_piece_split/221112/videos/{piece_stem}_{view}.mp4
    Data/Pipeline/04_piece_split/221112/_video_alignment.json

Usage (from the repository root):
  python pipeline/04_piece_split/steps/s10_221112_videos.py [--align-only] [--force]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]                     # pipeline/
sys.path.insert(0, str(ROOT))
from _common import video_xcorr as vx  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "slice_videos_220819", ROOT / "04_piece_split" / "anchors_220819" / "steps" / "s05_slice_videos.py")
sv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sv)                                    # generic NCC / cutting helpers

DS = "221112"
RAW = Path("Data/Raw") / DS / "C1" / "Video_1080p"
NORM = Path("Data/Pipeline/01_ingest/Normalized") / DS
PIECES = Path("Data/Pipeline/04_piece_split") / DS
OUT = PIECES / "videos"
ALIGN_JSON = PIECES / "_video_alignment.json"
VIEWS = ["Top", "Right"]
MIN_SOURCE_S = 10.0
MIN_NCC = 0.5
MARGIN = 0.1
COVER = 0.8


def envelope(wav: Path) -> np.ndarray:
    x, sr = vx._read_wav_mono(wav)
    return vx._envelope(x, sr)


def _ncc(sig: np.ndarray, tpl: np.ndarray) -> np.ndarray:
    """NCC over full-overlap positions; positions where the source window is
    (near) constant -- silence -- have no defined correlation and are dropped."""
    r = sv._ncc_valid(sig, tpl)
    return np.where(np.abs(r) <= 1.0 + 1e-6, r, -1.0)


def best_offset(t_env: np.ndarray, s_env: np.ndarray) -> tuple[float, float]:
    """(offset, ncc): video_t = trial_t + offset, one containing the other."""
    if len(t_env) <= len(s_env):
        (o, c), = sv.peaks(_ncc(s_env, t_env), 1)
        return o, c
    (o, c), = sv.peaks(_ncc(t_env, s_env), 1)
    return -o, c


def align_view(view: str, trials: dict[str, Path]) -> dict:
    files = sorted(p for p in (RAW / view).glob("*") if p.suffix.lower() in (".mp4", ".mov"))
    recs = [(f, vx.extract_audio_mono(f)) for f in files]
    recs = [(f, a) for f, a in recs if sv.vx._read_wav_mono(a)[0].size / vx.CLIP_SR >= MIN_SOURCE_S]
    s_env = [envelope(a) for _, a in recs]
    stems = list(trials)
    t_env = [envelope(trials[s]) for s in stems]
    S = np.full((len(stems), len(recs)), -1.0)
    O = np.zeros_like(S)
    for i, te in enumerate(t_env):
        for j, se in enumerate(s_env):
            if len(se) >= COVER * len(te):
                O[i, j], S[i, j] = best_offset(te, se)
    rows, cols = linear_sum_assignment(-S)
    out = {}
    for i, j in zip(rows, cols):
        others = np.delete(S[i], j)
        margin = float(S[i, j] - (others.max() if len(others) else -1))
        rec = {"file": str(recs[j][0].relative_to(Path("Data/Raw"))).replace("\\", "/"),
               "ncc": round(float(S[i, j]), 4), "margin": round(margin, 4)}
        if S[i, j] >= MIN_NCC and margin >= MARGIN:
            f = sv.lag_ncc(trials[stems[i]], recs[j][1], float(O[i, j]), sv.FINE_SEARCH_S, sr=sv.FINE_SR,
                           min_overlap_s=5.0)
            rec.update({"status": "ok", "coarse_offset_sec": round(float(O[i, j]), 3),
                        "offset_sec": round(f[0][0], 4) if f else round(float(O[i, j]), 3)})
        else:
            rec["status"] = "unresolved"
        out[stems[i]] = rec
    for i, s in enumerate(stems):
        out.setdefault(s, {"status": "unresolved", "reason": "no recording left"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--align-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    trials = {w.stem: w for w in sorted(NORM.glob("C1_N*/audio/*.wav"))}
    store = {}
    for view in VIEWS:
        res = align_view(view, trials)
        for s, r in res.items():
            store.setdefault(s, {})[view] = r
        ok = sum(r["status"] == "ok" for r in res.values())
        print(f"{view}: {ok}/{len(res)} trials matched")
    ALIGN_JSON.write_text(json.dumps(store, indent=1), encoding="utf-8")
    if a.align_only:
        return
    jobs, lengths = [], {}
    for t in json.loads((PIECES / "_pieces.json").read_text(encoding="utf-8")):
        for view, al in store.get(t["trial_stem"], {}).items():
            if al["status"] != "ok":
                continue
            src = Path("Data/Raw") / al["file"]
            lengths.setdefault(src, sv.video_seconds(src))
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
