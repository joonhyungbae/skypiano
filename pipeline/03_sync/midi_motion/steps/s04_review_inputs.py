"""
Inputs for the sync review viewer, per trial.

The viewer plays a trial's audio and MIDI with its motion on one clock and
lets the reviewer judge the sync by eye. This step gathers what it needs:

  shift_sec      motion_t = midi_t + shift_sec (the trial's sync offset)
  midi_path,     the trial MIDI and audio on that clock
  audio_path
  played_sec,    first/last note onset, and the motion span on the MIDI clock
  motion_sec
  pieces         piece windows on the trial clock (for per-piece offsets)
  keyboard       an approximate key layout to draw under the hands: white
                 keys 23.5 mm apart along LEnd -> REnd, A0 where the
                 fingertips gather at white-key onsets. The piano does not
                 move within a session and Stage 02 puts its end markers in
                 one place, so A0 is the median over the session's trials
                 (one short trial alone can settle on a finger that is not
                 playing). Drawing only; it is not used to judge anything.

Clocks per cohort:
  220819       trial audio/MIDI from Stage 01; motion offset from the anchors
               aligner of Stage 04; pieces from its locate manifest.
  2024 (SMPTE) trial MIDI cut by s03_timecode_align at the timecode of the
               Stage 02 hand file's first frame, trial audio cut at the same
               times by 04 s01: offset 0; pieces from 04 `_pieces.json`.
  221112       trial audio/MIDI cut by Stage 01 at the take's position in the
               session recording (trial time = motion time): offset 0;
               pieces from 04 `_pieces.json`.
  211215       trial audio/MIDI from Stage 01; motion offset from
               s05_trial_motion_sync (TrialSync/211215.json); pieces from 04
               `_pieces.json` (one per trial).

Outputs Data/Pipeline/03_sync/midi_motion/ReviewInputs/{ds}/{trial}.json.

Usage (from the repository root):
  python pipeline/03_sync/midi_motion/steps/s04_review_inputs.py --dataset 220819 --sessions C0_N2 C1_N1
  python pipeline/03_sync/midi_motion/steps/s04_review_inputs.py --dataset 240828
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # pipeline/
from _common.motion_midi_xcorr import fingertips_from_aligned_csv  # noqa: E402

PIPE = Path("Data/Pipeline")
NORM = PIPE / "01_ingest" / "Normalized"
HAND = PIPE / "02_spatial" / "hand" / "Aligned"
OUT = PIPE / "03_sync" / "midi_motion" / "ReviewInputs"
SYNC_MIDI = PIPE / "03_sync" / "midi_motion" / "Aligned"
PIECE_DIR = PIPE / "04_piece_split"
# 220819's trial offsets and piece windows come from the anchors aligner of Stage 04.
ANCHORS = PIECE_DIR / "anchors_220819"
TIMECODE_DATASETS = {"240523", "240809", "240828"}
MOTION_CLOCK_DATASETS = {"221112"}      # Stage 01 cuts the trial MIDI on the motion clock
TRIAL_SYNC_DATASETS = {"211215"}        # offsets from s05_trial_motion_sync
TRIAL_SYNC = PIPE / "03_sync" / "midi_motion" / "TrialSync"

FPS = 120.0
WHITE_KEY_MM = 23.5
WHITE = {0, 2, 4, 5, 7, 9, 11}
MCPS = [f"{s}{f}M" for s in "LR" for f in "IMRL"]


def trial_inputs(ds: str, sess_dir: Path, t: dict) -> tuple[Path, Path, float | None]:
    """(midi, audio, shift) of one trial on its sync clock."""
    stem = t["trial_stem"]
    if ds in TIMECODE_DATASETS:
        return SYNC_MIDI / ds / f"{t.get('raw_stem')}.mid", PIECE_DIR / ds / f"{stem}.wav", 0.0
    if ds in TRIAL_SYNC_DATASETS:
        sync = json.loads((TRIAL_SYNC / f"{ds}.json").read_text(encoding="utf-8"))
        return sess_dir / "midi" / f"{stem}.mid", sess_dir / "audio" / f"{stem}.wav",             (sync.get(stem) or {}).get("shift_sec")
    if ds in MOTION_CLOCK_DATASETS:
        return sess_dir / "midi" / f"{stem}.mid", sess_dir / "audio" / f"{stem}.wav", 0.0
    if ds != "220819":
        raise SystemExit(f"{ds}: no sync source wired")
    al = json.loads((ANCHORS / "alignment.json").read_text(encoding="utf-8"))
    shift = next((e["hand"].get("align_shift_sec") for e in al["entries"] if e["stage1_stem"] == stem), None)
    return sess_dir / "midi" / f"{stem}.mid", sess_dir / "audio" / f"{stem}.wav", shift


def signal_strength(ds: str, t: dict) -> dict:
    """How clearly the trial's sync source settled its offset (to order the
    review; it does not judge the sync). `weak` trials should be looked at."""
    stem = t["trial_stem"]
    if ds in TIMECODE_DATASETS:
        return {"level": "clear", "basis": "timecode"}
    if ds in TRIAL_SYNC_DATASETS:
        s = json.loads((TRIAL_SYNC / f"{ds}.json").read_text(encoding="utf-8")).get(stem) or {}
        return {"level": "clear" if s.get("status") == "clear" else "weak", "basis": "key-press correlation",
                "ncc": s.get("ncc"), "margin": s.get("margin")}
    if ds in MOTION_CLOCK_DATASETS:
        pl = t.get("placement") or {}
        ok = (pl.get("score") or 0) >= 1.0 and abs(pl.get("position_minus_signal_sec") or 99) <= 8
        return {"level": "clear" if ok else "weak", "basis": "take placement in the session recording",
                "score": pl.get("score"), "from_log_signal_sec": pl.get("position_minus_signal_sec")}
    al = json.loads((ANCHORS / "alignment.json").read_text(encoding="utf-8"))
    h = next((e["hand"] for e in al["entries"] if e["stage1_stem"] == stem), {})
    fine = (h.get("fine") or {}).get("ncc")
    ok = h.get("confidence") == "strong" or (fine or 0) >= 0.3
    return {"level": "clear" if ok else "weak", "basis": "onset cross-correlation",
            "confidence": h.get("confidence"), "fine_ncc": fine}


def piece_windows(ds: str) -> dict[str, list[dict]]:
    """trial -> [{piece, start, end}] on the trial (MIDI) clock."""
    out: dict[str, list] = {}
    if ds == "220819":
        loc = json.loads((ANCHORS / "locate_manifest.json").read_text(encoding="utf-8"))
        for e in loc["entries"]:
            if e.get("ok") and e.get("out_stem"):
                out.setdefault(e["stage1_stem"], []).append(
                    {"piece": f"p{e['p_slot']}", "start": e["t_start_sec"], "end": e["t_end_sec"]})
    else:
        pj = PIECE_DIR / ds / "_pieces.json"
        if pj.exists():
            for t in json.loads(pj.read_text(encoding="utf-8")):
                out[t["trial_stem"]] = [{"piece": f"p{p['piece_idx']}", "start": p["start_sec"], "end": p["end_sec"]}
                                        for p in t.get("pieces", [])]
    for v in out.values():
        v.sort(key=lambda x: x["start"])
    return out


def white_index(pitch: int) -> int:
    return sum(1 for p in range(21, pitch) if p % 12 in WHITE)


def key_center(pitch: int, a: float) -> float:
    wi = white_index(pitch)
    return a + WHITE_KEY_MM * (wi if pitch % 12 in WHITE else wi - 0.5)


def layout(path: Path, tips: np.ndarray, notes: list, shift: float, a: float | None = None) -> dict | None:
    """Approximate key layout for drawing (see module docstring); `a` fixes A0."""
    head = pd.read_csv(path, nrows=0).columns
    cols = [c for c in head if c.split("_")[0] in ("LEnd", "REnd") or c.split("_")[0] in MCPS]
    df = pd.read_csv(path, usecols=cols)
    get = lambda m: np.nanmedian(df[[f"{m}_x", f"{m}_y", f"{m}_z"]].to_numpy(float), axis=0)
    le, re_ = get("LEnd"), get("REnd")
    mcp = np.nanmean(np.stack([df[[f"{m}_x", f"{m}_y", f"{m}_z"]].to_numpy(float) for m in MCPS
                               if f"{m}_x" in df], axis=1), axis=1)
    u = (re_ - le) / np.linalg.norm(re_ - le)
    v = np.cross([0.0, 0.0, 1.0], u); v /= np.linalg.norm(v)
    if np.nanmean((mcp - np.nanmean(tips, axis=1)) @ v) < 0:      # v points toward the player
        v = -v
    U, Z, V = (tips - le) @ u, tips[..., 2], (tips - le) @ v
    idx = np.array([int(round((t + shift) * FPS)) for t, _ in notes])
    ok = (idx >= 0) & (idx < len(tips))
    res = [U[idx[k]] - WHITE_KEY_MM * white_index(p) for k, (_, p) in enumerate(notes) if ok[k] and p % 12 in WHITE]
    vals = np.concatenate(res) if res else np.array([])
    vals = vals[np.isfinite(vals)]
    if a is None and len(vals) < 50:
        # Too few onsets to place the keys: give the keyboard frame only (end
        # markers, key height) so the viewer can frame the hands; no keys drawn.
        span = float(np.linalg.norm(re_ - le))
        return {"lend": le.round(2).tolist(), "u": u.round(5).tolist(), "v": v.round(5).tolist(),
                "a0_center_mm": round((span - 51 * WHITE_KEY_MM) / 2, 2), "white_key_mm": WHITE_KEY_MM,
                "key_top_z": round(float(np.nanpercentile(Z, 5)), 2),
                "tip_v_mm": round(float(np.nanmedian(V)), 2), "approximate": True, "frame_only": True}
    if a is None:
        h, e = np.histogram(vals, bins=np.arange(vals.min(), vals.max() + 2, 2.0))
        a = float(e[np.argmax(h)] + 1.0)
        a = float(np.median(vals[np.abs(vals - a) <= WHITE_KEY_MM / 2]))
    zs, vs = [], []
    for k, (_, p) in enumerate(notes):
        if ok[k] and p % 12 in WHITE:
            d = np.abs(U[idx[k]] - key_center(p, a))
            j = np.nanargmin(np.where(np.isfinite(d), d, np.inf))
            if d[j] <= WHITE_KEY_MM / 2 and np.isfinite(Z[idx[k], j]):
                zs.append(Z[idx[k], j]); vs.append(V[idx[k], j])
    return {"lend": le.round(2).tolist(), "u": u.round(5).tolist(), "v": v.round(5).tolist(),
            "a0_center_mm": round(a, 2), "white_key_mm": WHITE_KEY_MM,
            "key_top_z": round(float(np.median(zs)) if zs else float(np.nanpercentile(Z, 5)), 2),
            "tip_v_mm": round(float(np.median(vs)) if vs else 0.0, 2), "approximate": True}


def trial_record(hand_csv: Path, midi: Path, shift: float, pieces: list, a: float | None = None) -> dict:
    tips = fingertips_from_aligned_csv(hand_csv)
    notes = sorted((n.start, n.pitch) for i in pretty_midi.PrettyMIDI(str(midi)).instruments for n in i.notes)
    if not notes:
        return {"status": "na", "reason": "no MIDI notes"}
    kb = layout(hand_csv, tips, notes, shift, a)
    if kb is None:
        return {"status": "na", "reason": "too few onsets with motion"}
    return {"status": "ok", "shift_sec": shift, "fps": FPS,
            "played_sec": [float(notes[0][0]), float(notes[-1][0])],
            "motion_sec": [-shift, len(tips) / FPS - shift],
            "pieces": pieces, "keyboard": kb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sessions", nargs="*", default=None)
    a = ap.parse_args()
    out_dir = OUT / a.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = piece_windows(a.dataset)
    for mp in sorted((NORM / a.dataset).glob("*/_mapping.json")):
        sess = mp.parent.name
        if a.sessions and sess not in a.sessions:
            continue
        trials = json.loads(mp.read_text(encoding="utf-8"))["trials"]
        inputs = {t["trial_stem"]: (HAND / a.dataset / f"{t['trial_stem']}.csv",) + trial_inputs(a.dataset, mp.parent, t)
                  for t in trials}
        # A0 per trial, then the session median for every trial
        a0s = []
        for stem, (hand, midi, audio, shift) in inputs.items():
            if hand.exists() and midi.exists() and shift is not None:
                r = trial_record(hand, midi, shift, [])
                if r["status"] == "ok" and not r["keyboard"].get("frame_only"):
                    a0s.append(r["keyboard"]["a0_center_mm"])
        session_a0 = float(np.median(a0s)) if a0s else None
        print(f"  {sess}: A0 {session_a0} mm (median of {len(a0s)} trials)")
        for t in trials:
            stem = t["trial_stem"]
            hand, midi, audio, shift = inputs[stem]
            if not hand.exists() or not midi.exists() or shift is None:
                r = {"status": "na", "reason": "no hand motion or no sync offset"}
            else:
                r = trial_record(hand, midi, shift, windows.get(stem, []), session_a0)
            if r["status"] == "ok":
                r["signal"] = signal_strength(a.dataset, t)
            r.update({"dataset": a.dataset, "session": sess, "trial_stem": stem,
                      "midi_path": str(midi).replace("\\", "/"), "audio_path": str(audio).replace("\\", "/")})
            (out_dir / f"{stem}.json").write_text(json.dumps(r), encoding="utf-8")
            print(f"  {stem:28s} {r['status']}{'' if r['status'] == 'ok' else ' (' + str(r.get('reason')) + ')'}")


if __name__ == "__main__":
    main()
