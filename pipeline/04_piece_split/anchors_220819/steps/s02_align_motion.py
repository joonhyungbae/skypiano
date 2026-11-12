"""
Stage 04_piece_split/anchors_220819 / Step 02: Per-recording motion<->audio alignment.

For each Stage 01 recording that has motion data:
  1. Load Stage 01 MIDI onset pulse train (100 Hz; assumed audio-time since
     audio+MIDI are co-recorded).
  2. Load hand motion CSV (02_spatial/hand/Aligned/220819/<trial_stem>.csv)
     and compute fingertip velocity-peak pulse train.
  3. xcorr -> hand_align_shift_sec (motion_t - audio_t).
  4. Body motion (02_spatial/body/Aligned/220819/<trial_stem>.csv) takes the
     hand's offset: Stage 02 puts the body file on the hand file's clock
     (same frame count; the wrist markers of both systems coincide at lag 0).
  5. Emit Data/Pipeline/04_piece_split/anchors_220819/alignment.json.

Applied downstream by s03_slice when cutting motion.
"""
from __future__ import annotations

import functools
import json
import sys
import time

import numpy as np
import pretty_midi
from collections import defaultdict
from pathlib import Path

_P = Path(__file__).resolve()
while _P.name != "pipeline":
    _P = _P.parent
sys.path.insert(0, str(_P))

from _common.normalized_io import NORMALIZED_ROOT  # noqa: E402

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))
from src.alignment import (  # noqa: E402
    HAND_ONSET_MARKERS, MOTION_FPS,
    align_midi_to_motion, motion_onset_pulses,
)
from src.locate import GRID_SR, midi_to_onset_pulses  # noqa: E402

print = functools.partial(print, flush=True)

DATASET = "220819"
OUT_ROOT = Path("Data/Pipeline/04_piece_split/anchors_220819")
ALIGNMENT_PATH = OUT_ROOT / "alignment.json"

HAND_ALIGNED_ROOT = Path("Data/Pipeline/02_spatial/hand/Aligned") / DATASET
BODY_ALIGNED_ROOT = Path("Data/Pipeline/02_spatial/body/Aligned") / DATASET
MIDI_MOTION_JSON_ROOT = Path("Data/Pipeline/03_sync/midi_motion/Aligned") / DATASET


def _classify_confidence(align: dict) -> str:
    if not align.get("ok"):
        return "failed"
    ratio = align.get("secondary_ratio") or 99.0
    if ratio >= 3.0:
        return "strong"
    if ratio >= 1.5:
        return "medium"
    return "low"


def run(dataset: str | None = None, force: bool = False) -> None:
    if dataset is not None and dataset != DATASET:
        print(f"[s02_align_motion] skip - only {DATASET} supported")
        return
    if ALIGNMENT_PATH.is_file() and not force:
        print(f"[s02_align_motion] alignment exists; pass --force to recompute: {ALIGNMENT_PATH}")
        return

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Walk all 220819 Stage 01 recordings, pair up by canonical trial_stem.
    recordings: list[dict] = []
    for sess_dir in sorted((NORMALIZED_ROOT / DATASET).iterdir()):
        mpath = sess_dir / "_mapping.json"
        if not mpath.is_file():
            continue
        data = json.loads(mpath.read_text(encoding="utf-8"))
        for trial in data["trials"]:
            stem = trial["trial_stem"]
            rs = trial.get("raw_sources", {})
            src_mid = sess_dir / "midi" / f"{stem}.mid"
            # Motion is read under the canonical trial stem: the Stage 02 hand
            # pipeline writes canonical names from Stage 01 Normalized, and the
            # body aligner writes every alias. Raw-named (2022Piano*) files in
            # those folders are left over from the pre-Normalized pipeline.
            hand_src = HAND_ALIGNED_ROOT / f"{stem}.csv" if rs.get("motion_hand") else None
            body_src = BODY_ALIGNED_ROOT / f"{stem}.csv" if rs.get("motion_body") else None
            if not src_mid.is_file():
                continue
            recordings.append({
                "session": sess_dir.name,
                "stage1_stem": stem,
                "midi_path": src_mid,
                "hand_path": hand_src if (hand_src and hand_src.is_file()) else None,
                "body_path": body_src if (body_src and body_src.is_file()) else None,
            })

    print(f"[s02_align_motion] {len(recordings)} recordings to align")

    results: list[dict] = []
    t_all = time.time()
    conf_counts: dict = defaultdict(lambda: defaultdict(int))

    for i, rec in enumerate(recordings, 1):
        midi_pulses = midi_to_onset_pulses(rec["midi_path"], sr=GRID_SR)
        entry = {
            "session": rec["session"],
            "stage1_stem": rec["stage1_stem"],
            "midi_duration_sec": round(len(midi_pulses) / GRID_SR, 3),
            "n_midi_onsets": int((midi_pulses > 0).sum()),
        }

        # Hand alignment
        if rec["hand_path"]:
            pulses = motion_onset_pulses(rec["hand_path"], HAND_ONSET_MARKERS)
            align = align_midi_to_motion(midi_pulses, pulses,
                                         midi_fps=GRID_SR, motion_fps=MOTION_FPS)
            conf = _classify_confidence(align)
            entry["hand"] = {
                "align_shift_sec": align.get("align_shift_sec"),
                "peak_corr": align.get("peak_corr"),
                "secondary_ratio": align.get("secondary_ratio"),
                "n_motion_onsets": align.get("n_motion_onsets"),
                "source_fps": MOTION_FPS,
                "confidence": conf,
                "reason": align.get("reason"),
            }
            conf_counts["hand"][conf] += 1
        else:
            entry["hand"] = {"confidence": "missing", "align_shift_sec": None}

        # Body: Stage 02 registers the body file onto the hand file in space
        # and time (same frame count, wrist markers coincide at lag 0), so the
        # body shares the hand's offset; it is filled in after the hand's is final.
        entry["body"] = ({"method": "hand_clock", "confidence": "pending", "align_shift_sec": None}
                         if rec["body_path"] else {"confidence": "missing", "align_shift_sec": None})

        results.append(entry)
        if i % 10 == 0 or i == len(recordings):
            print(f"  [{i}/{len(recordings)}] {time.time()-t_all:.0f}s")

    # Stage 03 midi_motion JSON override for low-confidence hand alignments.
    # Stage 03 midi_motion uses coarse-to-fine multi-signal xcorr (fingertip z-speed + accel +
    # keystroke onsets + wrist speed weighted sum), which is more robust than the
    # local pulse-train xcorr for noisy C0_N0 P4 trials. Sign convention is compatible
    # (both return motion_t - audio_t).
    s5_override_count = 0
    if MIDI_MOTION_JSON_ROOT.is_dir():
        # Build raw_midi_stem -> (session, trial_stem) via _mapping.json
        raw_to_trial: dict[str, tuple[str, str]] = {}
        # Stems the trial's hand take is known by (canonical + raw). A Stage 03
        # offset computed on a different hand file (e.g. before a Stage 01
        # re-match) must not be inherited.
        hand_ids: dict[tuple[str, str], set] = {}
        for sess in (NORMALIZED_ROOT / DATASET).iterdir():
            mpath = sess / "_mapping.json"
            if not mpath.is_file():
                continue
            mapping = json.loads(mpath.read_text(encoding="utf-8"))
            for trial in mapping["trials"]:
                rm = trial.get("raw_sources", {}).get("midi")
                if rm:
                    raw_to_trial[Path(rm).stem] = (sess.name, trial["trial_stem"])
                    rh = trial.get("raw_sources", {}).get("motion_hand")
                    hand_ids[(sess.name, trial["trial_stem"])] = {
                        trial["trial_stem"], Path(rh).stem if rh else None}

        result_idx = {(e["session"], e["stage1_stem"]): e for e in results}
        for json_path in MIDI_MOTION_JSON_ROOT.glob("*.json"):
            s5 = json.loads(json_path.read_text(encoding="utf-8"))
            s5_quality = s5.get("quality", "")
            if s5_quality not in ("good", "fair"):
                continue
            key = raw_to_trial.get(json_path.stem)
            if key is None or key not in result_idx:
                continue
            if Path(s5.get("hand_csv", "")).stem not in hand_ids.get(key, set()):
                continue
            entry = result_idx[key]
            h = entry["hand"]
            if h.get("confidence") == "low":
                h["align_shift_sec_original"] = h.get("align_shift_sec")
                h["align_shift_sec"] = round(s5["offset_sec"], 3)
                h["confidence"] = "stage5_inherited"
                h["fallback_source"] = f"stage5:{s5_quality}"
                h["stage5_confidence"] = s5.get("confidence")
                conf_counts["hand"]["low"] -= 1
                conf_counts["hand"]["stage5_inherited"] += 1
                s5_override_count += 1

    fine_count = _refine_hand_shifts(results, recordings)

    # Body = hand clock (see above).
    for e in results:
        b = e.get("body", {})
        if b.get("method") == "hand_clock":
            h = e.get("hand", {})
            b["align_shift_sec"] = h.get("align_shift_sec")
            b["confidence"] = h.get("confidence") if h.get("align_shift_sec") is not None else "missing"
            conf_counts["body"][b["confidence"]] += 1

    manifest = {
        "dataset": DATASET,
        "n_recordings": len(results),
        "hand_fine_refined_count": fine_count,
        "confidence_counts": {k: dict(v) for k, v in conf_counts.items()},
        "hand_stage5_override_count": s5_override_count,
        "entries": results,
    }
    ALIGNMENT_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[s02_align_motion] done ({time.time()-t_all:.0f}s)")
    for mod, cc in conf_counts.items():
        print(f"  {mod}: {dict(cc)}")
    print(f"  alignment: {ALIGNMENT_PATH}")


FINE_SEARCH_S = 0.6       # fine search range around the coarse shift
FINE_MIN_NCC = 0.30       # refine only when the summed key-press curve is this clear


def _refine_hand_shifts(results: list[dict], recordings: list[dict]) -> int:
    """
    Fine pass: inside the final piece windows (s01), cross-correlate key-press
    activity with the MIDI onsets, sum the curves over the recording's pieces
    and move the shift so the press lead equals PRESS_LEAD_S. The coarse
    shift is kept as `align_shift_sec_coarse`. Body slicing reuses the hand
    shift, so it is refined too.
    """
    from _common import motion_midi_xcorr as mx
    locate_path = OUT_ROOT / "locate_manifest.json"
    if not locate_path.is_file():
        return 0
    windows: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for e in json.loads(locate_path.read_text(encoding="utf-8")).get("entries", []):
        if e.get("ok") and e.get("t_start_sec") is not None:
            windows[e["stage1_stem"]].append((e["t_start_sec"], e["t_end_sec"]))
    by_key = {(r["session"], r["stage1_stem"]): r for r in recordings}
    n = 0
    for entry in results:
        h = entry.get("hand", {})
        shift = h.get("align_shift_sec")
        rec = by_key.get((entry["session"], entry["stage1_stem"]))
        wins = windows.get(entry["stage1_stem"])
        if shift is None or rec is None or not rec.get("hand_path") or not wins:
            continue
        tips = mx.fingertips_from_aligned_csv(rec["hand_path"])
        onsets = sorted(nt.start for i in pretty_midi.PrettyMIDI(str(rec["midi_path"])).instruments for nt in i.notes)
        curves, lags = [], None
        for t0, t1 in wins:
            seg = tips[max(0, int((t0 + shift) * MOTION_FPS)):max(0, int((t1 + shift) * MOTION_FPS))]
            notes = [t - t0 for t in onsets if t0 <= t < t1]
            r = mx.press_onset_curve(seg, MOTION_FPS, notes, FINE_SEARCH_S)
            if r is not None:
                lags, c = r
                curves.append(c)
        if not curves:
            continue
        total = np.mean(curves, axis=0)
        k = int(np.argmax(total))
        lag, ncc = float(lags[k]), float(total[k])
        h["fine"] = {"lag_s": round(lag, 3), "ncc": round(ncc, 3), "n_pieces": len(curves)}
        if ncc >= FINE_MIN_NCC:
            delta = lag - mx.PRESS_LEAD_S
            h["align_shift_sec_coarse"] = shift
            h["align_shift_sec"] = round(shift + delta, 3)
            h["fine"]["applied_delta_s"] = round(delta, 3)
            n += 1
    return n


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run(dataset=args.dataset, force=args.force)
