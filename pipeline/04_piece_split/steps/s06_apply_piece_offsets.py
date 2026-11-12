"""
Stage 04 step 06: re-cut the motion of pieces with a confirmed motion offset.

Some trials carry a shift inside them (e.g. the capture paused between two
pieces of a take without timecode), so one trial offset does not fit every
piece. Per-piece offsets are confirmed by eye in the review viewer's sync
check and stored in `Data/Metadata/{ds}/_alignment_review.json` under
`piece_offsets` ({piece_stem: {offset_sec, ...}}), on top of the trial's
offset:

    motion_t = midi_t + trial_shift + offset_sec

A trial-level nudge saved with a sync verdict (`sync` -> shift_adjust_sec)
applies the same way to every piece of the trial that has no offset of its
own (a saved piece offset already holds the whole nudge).

This step re-cuts `{piece_stem}_hand.csv` and `_body.csv` with that offset,
with the same frame rule as the regular cut (rows = frames at 120 fps).
Audio and MIDI are not touched. The applied offsets are recorded in
`Data/Pipeline/04_piece_split/{ds}/_piece_motion_offsets.json`.

`--trials` re-cuts every piece of the given trials (with their confirmed
offsets, 0 otherwise), e.g. after their Stage 02 motion was regenerated.

Usage (from the repository root):
  python pipeline/04_piece_split/steps/s06_apply_piece_offsets.py --dataset 240828
  python pipeline/04_piece_split/steps/s06_apply_piece_offsets.py --dataset 240828 --trials 240828_C0_N4_T1_P14_t1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PIPE = Path("Data/Pipeline")
META = Path("Data/Metadata")
NORM = PIPE / "01_ingest" / "Normalized"
HAND = PIPE / "02_spatial" / "hand" / "Aligned"
BODY = PIPE / "02_spatial" / "body" / "Aligned"
PIECES = PIPE / "04_piece_split"
ANCHORS = PIECES / "anchors_220819"
TRIAL_SYNC = PIPE / "03_sync" / "midi_motion" / "TrialSync"   # per-trial offsets (211215)
MOTION_FPS = 120.0


def piece_index(ds: str) -> dict[str, dict]:
    """piece_stem -> {trial, raw, start, end, hand_shift, body_shift} on the trial clock."""
    out = {}
    if ds == "220819":
        al = {e["stage1_stem"]: e for e in json.loads((ANCHORS / "alignment.json").read_text(encoding="utf-8"))["entries"]}
        loc = json.loads((ANCHORS / "locate_manifest.json").read_text(encoding="utf-8"))
        for e in loc["entries"]:
            if e.get("ok") and e.get("out_stem"):
                a = al.get(e["stage1_stem"], {})
                out[e["out_stem"]] = {"trial": e["stage1_stem"], "raw": e["stage1_stem"],
                                      "start": e["t_start_sec"], "end": e["t_end_sec"],
                                      "hand_shift": (a.get("hand") or {}).get("align_shift_sec"),
                                      "body_shift": (a.get("body") or {}).get("align_shift_sec")}
        return out
    trial_sync = {}
    if (TRIAL_SYNC / f"{ds}.json").exists():
        trial_sync = json.loads((TRIAL_SYNC / f"{ds}.json").read_text(encoding="utf-8"))
    raw = {}
    for mp in (NORM / ds).glob("*/_mapping.json"):
        for t in json.loads(mp.read_text(encoding="utf-8"))["trials"]:
            raw[t["trial_stem"]] = t.get("raw_stem")
    for t in json.loads((PIECES / ds / "_pieces.json").read_text(encoding="utf-8")):
        for p in t.get("pieces", []):
            out[f"{t['trial_stem']}_p{p['piece_idx']}"] = {
                "trial": t["trial_stem"], "raw": raw.get(t["trial_stem"]) or t.get("raw_stem"),
                "start": p["start_sec"], "end": p["end_sec"],
                "hand_shift": (trial_sync.get(t["trial_stem"]) or {}).get("shift_sec", 0.0) if trial_sync else 0.0,
                "body_shift": None if trial_sync else 0.0}
    return out


def cut(src: Path, start: float, end: float, dst: Path) -> int:
    df = pd.read_csv(src)
    if "TimeCode" in df.columns:
        df = df.drop(columns=["TimeCode"])
    f0, f1 = int(start * MOTION_FPS), int(end * MOTION_FPS)
    part = df.iloc[max(0, f0):max(0, f1)].reset_index(drop=True)
    if f0 < 0:          # the shifted window starts before the recording: pad with empty frames
        pad = pd.DataFrame(index=range(-f0), columns=df.columns, dtype=float)
        part = pd.concat([pad, part], ignore_index=True)
    part.to_csv(dst, index=False)
    return len(part)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--trials", nargs="*", default=None, help="re-cut every piece of these trials")
    a = ap.parse_args()
    ds = a.dataset
    rev_path = META / ds / "_alignment_review.json"
    rev = json.loads(rev_path.read_text(encoding="utf-8")) if rev_path.exists() else {}
    confirmed = rev.get("piece_offsets", {})
    nudge = {k: float(v["shift_adjust_sec"]) for k, v in rev.get("sync", {}).items()
             if v.get("verdict") and abs(v.get("shift_adjust_sec") or 0.0) > 1e-9}
    idx = piece_index(ds)
    offsets = dict(confirmed)
    for ps, info in idx.items():
        if info["trial"] in nudge and ps not in offsets:
            offsets[ps] = {"offset_sec": nudge[info["trial"]], "at": rev["sync"][info["trial"]].get("at"),
                           "from": "trial nudge"}
    # a piece cut with an offset that is no longer confirmed goes back to 0
    applied_path = PIECES / ds / "_piece_motion_offsets.json"
    before = json.loads(applied_path.read_text(encoding="utf-8")) if applied_path.exists() else {}
    for ps, v in before.items():
        if ps not in offsets and abs(v.get("offset_sec") or 0.0) > 1e-9:
            offsets[ps] = {"offset_sec": 0.0, "at": None, "from": "offset withdrawn"}
    for ps, info in idx.items():
        if a.trials and info["trial"] in a.trials and ps not in offsets:
            offsets[ps] = {"offset_sec": 0.0, "at": None}
    if not offsets:
        print(f"{ds}: nothing to re-cut")
        return
    applied = {}
    for ps, v in sorted(offsets.items()):
        info = idx.get(ps)
        if info is None:
            print(f"  {ps}: unknown piece, skipped")
            continue
        off = float(v["offset_sec"])
        done = {}
        for kind, root, name, shift in [("hand", HAND, info["trial"], info["hand_shift"]),
                                        ("body", BODY, info["raw"], info["body_shift"])]:
            src = root / ds / f"{name}.csv"
            if not src.exists():
                src = root / ds / f"{info['trial']}.csv"
            dst = PIECES / ds / f"{ps}_{kind}.csv"
            if shift is None or not src.exists() or not dst.exists():
                continue
            n = cut(src, info["start"] + shift + off, info["end"] + shift + off, dst)
            done[kind] = n
        applied[ps] = {"offset_sec": off, "re_cut": done, "confirmed_at": v.get("at")}
        print(f"  {ps}: offset {off:+.3f}s re-cut {done}")
    out = PIECES / ds / "_piece_motion_offsets.json"
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev.update(applied)
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
