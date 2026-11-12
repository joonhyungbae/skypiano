"""
Stage 04 step 12: keep piece windows inside their trial recording and apart
from each other.

The 2024 piece splitter pads a window past the last note; at the end of a
trial the padding can run past the end of the trial audio, so `_pieces.json`
lists a window longer than the cut files (they stop where the recording does).
This step sets such an `end_sec` to the trial audio's length (and a negative
`start_sec` to 0); audio and MIDI already end there.

A note held across the silence gap between two pieces (a key left down)
carried a window's end past the next piece's start. Such an `end_sec` is set
to the next piece's `start_sec`; the piece's audio, MIDI and videos are cut
back to the new length (previous files go to Data/Pipeline/_backups).

Piece motion running past its window (the motion capture outlasted the audio,
or the window was shortened) is cut back to the window's frames (120 fps).
Changes are listed in `_window_clamped.json`.

Usage (from the repository root):
  python pipeline/04_piece_split/steps/s12_clamp_piece_windows.py --dataset 240828 [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _common.slicing import slice_midi  # noqa: E402

PIPE = Path("Data/Pipeline")
PIECES = PIPE / "04_piece_split"
SYNC_MIDI = PIPE / "03_sync" / "midi_motion" / "Aligned"
FPS = 120.0


def trim_wav(path: Path, dur: float) -> None:
    with wave.open(str(path)) as w:
        params = w.getparams()
        frames = w.readframes(int(round(dur * w.getframerate())))
    with wave.open(str(path), "wb") as w:
        w.setparams(params)
        w.writeframes(frames)


def trim_video(path: Path, dur: float) -> None:
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-t", f"{dur:.3f}",
                    "-c", "copy", str(tmp)], check=True)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["240523", "240828"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ds = a.dataset
    pj = PIECES / ds / "_pieces.json"
    data = json.loads(pj.read_text(encoding="utf-8"))
    changed, overlaps = [], []
    for t in data:
        wav = PIECES / ds / f"{t['trial_stem']}.wav"
        if not wav.exists():
            continue
        with wave.open(str(wav)) as w:
            dur = round(w.getnframes() / w.getframerate(), 3)
        for p in t["pieces"]:
            old = (p["start_sec"], p["end_sec"])
            p["start_sec"] = max(0.0, p["start_sec"])
            p["end_sec"] = min(p["end_sec"], dur)
            if (p["start_sec"], p["end_sec"]) != old:
                changed.append({"piece": f"{t['trial_stem']}_p{p['piece_idx']}", "old": old,
                                "new": [p["start_sec"], p["end_sec"]], "trial_audio_sec": dur})
        ps = sorted(t["pieces"], key=lambda q: q["start_sec"])
        for p, nxt in zip(ps, ps[1:]):
            if p["end_sec"] - nxt["start_sec"] > 1e-3:
                old = (p["start_sec"], p["end_sec"])
                p["end_sec"] = nxt["start_sec"]
                overlaps.append((t, p, {"piece": f"{t['trial_stem']}_p{p['piece_idx']}", "old": old,
                                        "new": [p["start_sec"], p["end_sec"]],
                                        "next_piece": f"{t['trial_stem']}_p{nxt['piece_idx']}"}))
    print(f"{ds}: {len(changed)} windows clamped to the trial audio, "
          f"{len(overlaps)} windows ended at the next piece's start")
    for _, _, c in overlaps:
        print(f"   {c['piece']}: {c['old']} -> {c['new']}")
    if a.dry_run:
        return

    if overlaps:
        bak = PIPE / "_backups" / f"{time.strftime('%Y%m%d_%H%M%S')}_piece_overlaps_{ds}"
        bak.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pj, bak / "_pieces.json")
        for t, p, _ in overlaps:
            ps_stem = f"{t['trial_stem']}_p{p['piece_idx']}"
            new_dur = p["end_sec"] - p["start_sec"]
            files = sorted((PIECES / ds).glob(f"{ps_stem}[._]*")) + \
                sorted((PIECES / ds).glob(f"videos/{ps_stem}_*.mp4"))
            for f in files:
                dst = bak / f.relative_to(PIECES / ds)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
            wav = PIECES / ds / f"{ps_stem}.wav"
            if wav.exists():
                trim_wav(wav, new_dur)
            midi_src = SYNC_MIDI / ds / f"{t['raw_stem']}.mid"
            if midi_src.exists():
                slice_midi(midi_src, p["start_sec"], p["end_sec"], PIECES / ds / f"{ps_stem}.mid")
            for v in sorted((PIECES / ds / "videos").glob(f"{ps_stem}_*.mp4")):
                trim_video(v, new_dur)

    pj.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    trimmed = []
    for t in data:
        for p in t["pieces"]:
            n = round((p["end_sec"] - p["start_sec"]) * FPS)
            for kind in ("hand", "body"):
                f = PIECES / ds / f"{t['trial_stem']}_p{p['piece_idx']}_{kind}.csv"
                if f.exists():
                    df = pd.read_csv(f)
                    if len(df) > n:
                        df.iloc[:n].to_csv(f, index=False)
                        trimmed.append(f"{f.name}: {len(df)} -> {n} frames")
    prev_path = PIECES / ds / "_window_clamped.json"
    prev = json.loads(prev_path.read_text(encoding="utf-8")) if prev_path.exists() else []
    prev = prev if isinstance(prev, dict) else {"windows": prev, "motion_trimmed": []}
    prev["windows"] += changed
    prev.setdefault("overlaps", [])
    prev["overlaps"] += [c for _, _, c in overlaps]
    prev["motion_trimmed"] += trimmed
    prev_path.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"{ds}: {len(trimmed)} motion files trimmed")


if __name__ == "__main__":
    main()
