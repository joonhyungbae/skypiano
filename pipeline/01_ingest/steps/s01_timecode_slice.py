"""
Stage 01 step 01: Timecode-based normalization (240523 / 240809 / 240828)

These three datasets have OptiTrack-embedded timecodes in their per-trial motion
CSVs and a single continuous session audio + MIDI per node. We use the motion
CSV's first/last timecode to slice per-trial audio + MIDI from the session
recording, and copy the motion CSV as-is (Stage 02 reads it later).

Output:
  Data/Pipeline/01_ingest/Normalized/{dataset}/{session}/
      motion/{trial_stem}.csv
      audio/{trial_stem}.wav
      midi/{trial_stem}.mid
      _mapping.json
"""
from __future__ import annotations
import functools
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi
from pydub import AudioSegment
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common.normalized_io import RAW_ROOT, session_dir, write_mapping  # noqa: E402

print = functools.partial(print, flush=True)

TIMECODE_FPS = 30  # OptiTrack SMPTE timecode fps

# ──────────────────────────────────────────────────────────────────────
# Session config: motion_prefix matches motion CSVs by stem prefix,
# audio_stem identifies the corresponding session audio + MIDI files.
# ──────────────────────────────────────────────────────────────────────
import re

from src.docx_labels import load_labels  # noqa: E402

# 240523: free-choice pieces under Task 9, one per motion counter NN.
# 240809: single-cohort dataset with two raw recording blocks (opaque labels
#         PJH1/PJH2 on disk), all Task 2. Block P1 has only 2 usable trials
#         (03,04); block P2 has 29 (with exclusions).
# 240828: mixed Tasks 0/1/2/9 per group label — handled by s07_240828_match.

SESSIONS = {
    "240523": [
        {
            "base": RAW_ROOT / "240523/C1/N0",
            "canonical_session": "C1_N0",
            "motion_prefix": "240523sub02",
            "audio_stems": ["240523sub02"],
            "task": 9,
            "canonical_class": "C1",
            "canonical_n": "N0",
            "stem_regex": re.compile(r"^240523sub02(\d+)$"),
            "has_body": True,
            "use_labels": False,
        },
    ],
    "240809": [
        {
            "base": RAW_ROOT / "240809/C0/N0",
            "canonical_session": "C0_N0",
            "motion_prefix": "240809PJH",
            # Each raw block has its own session audio/MIDI; chosen by prefix.
            "audio_stem_map": {
                "01": "240809PJH1",
                "02": "240809PJH2",
            },
            "task": 2,
            "canonical_class": "C0",
            "canonical_n": "N0",
            "stem_regex": re.compile(r"^240809PJH(\d{2})(\d{2})$"),  # block, nn
            # groups[0] = "01" or "02" -> used as audio_stem_map key (after strip)
            "has_body": True,
            "use_labels": True,
            "labels_dataset_key": "240809",
            "labels_subject_key": "S01",
        },
    ],
    "240828": [
        {
            "base": RAW_ROOT / f"240828/C0/N{n}",
            "canonical_session": f"C0_N{n}",
            "motion_prefix": f"240828sub{n:02d}",
            "audio_stems": [f"240828sub{n:02d}"],
            "task": -1,  # determined per-trial from labels
            "canonical_class": "C0",
            "canonical_n": f"N{n}",
            "stem_regex": re.compile(rf"^240828sub(\d{{2}})(\d{{2}})$"),
            "has_body": True,
            "use_labels": True,
            "labels_dataset_key": "240828",
            "labels_subject_key": f"sub{n:02d}",
        }
        for n in [4, 5, 6, 7, 8]
    ],
}


def parse_timecode(tc: str) -> float:
    """HH:MM:SS:FF[.xx] → seconds"""
    hh, mm, ss, ff = tc.split(":")
    ff = ff.split(".")[0]
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ff) / TIMECODE_FPS


def _read_motion_header(csv_path: Path) -> dict:
    """Parse first-line OptiTrack export header into a dict of fields."""
    with open(csv_path, "r", encoding="latin1") as f:
        line = f.readline()
    parts = line.rstrip("\n").split(",")
    hdr: dict = {}
    for i in range(0, len(parts) - 1, 2):
        hdr[parts[i].strip()] = parts[i + 1].strip()
    return hdr


def get_motion_timecodes(csv_path: Path) -> tuple[float, float]:
    """Read first/last SMPTE timecodes from a raw OptiTrack CSV (frame-based column)."""
    data = pd.read_csv(csv_path, encoding="latin1", header=2, low_memory=False)
    data = data.iloc[3:, 2:]
    cols = data.columns.tolist()
    cols = [c.replace("Markerset 001:", "").replace("Unnamed: 2", "TimeCode") for c in cols]
    data.columns = cols
    if "TimeCode" not in data.columns:
        raise KeyError("no TimeCode column")
    valid_mask = data.iloc[:, 1:].notna().any(axis=1)
    valid_tc = data.loc[valid_mask, "TimeCode"]
    sec = valid_tc.apply(parse_timecode).to_numpy(float)
    # A capture can lose its timecode before it stops (the column drops to
    # 00:00:00:00): keep the run up to the first backward jump and count the
    # frames after it at the export rate.
    back = np.flatnonzero(np.diff(sec) < 0)
    if len(back):
        rate = float(_read_motion_header(csv_path)["Export Frame Rate"])
        i = int(back[0])
        return float(sec[0]), float(sec[i] + (len(sec) - 1 - i) / rate)
    return float(sec[0]), float(sec[-1])


def get_motion_times_via_header(csv_path: Path, anchor_csf: int, anchor_sec: float,
                                anchor_rate: float) -> tuple[float, float]:
    """Fallback for CSVs without a TimeCode column: compute absolute start/end via
    the header's `Capture Start Frame` and the given anchor (another trial's
    known SMPTE_sec, Capture_Start_Frame, Frame_Rate)."""
    hdr = _read_motion_header(csv_path)
    csf = int(hdr["Capture Start Frame"])
    total = int(hdr["Total Exported Frames"])
    rate = float(hdr["Export Frame Rate"])
    start_sec = anchor_sec + (csf - anchor_csf) / anchor_rate
    end_sec = start_sec + total / rate
    return start_sec, end_sec


def get_trial_times(csv_path: Path, anchor: dict | None = None) -> tuple[float, float, bool]:
    """Return (start_sec, end_sec, used_timecode_column)."""
    try:
        s, e = get_motion_timecodes(csv_path)
        return s, e, True
    except KeyError:
        if anchor is None:
            raise
        s, e = get_motion_times_via_header(
            csv_path,
            anchor_csf=anchor["csf"],
            anchor_sec=anchor["sec"],
            anchor_rate=anchor["rate"],
        )
        return s, e, False


# slice_audio / slice_midi extracted to _common/slicing.py during consolidation
from _common.slicing import slice_audio, slice_midi  # noqa: F401


def process_session(dataset: str, sess: dict, force: bool):
    base = sess["base"]
    canonical_session = sess["canonical_session"]
    hand_dir = base / "Motion" / "Hand"
    body_dir = base / "Motion" / "Body"
    has_body = sess.get("has_body", False) and body_dir.exists()

    if not hand_dir.exists():
        print(f"  [SKIP] no hand motion dir: {hand_dir}")
        return

    csv_files = sorted([
        f for f in hand_dir.iterdir()
        if f.suffix.lower() == ".csv" and f.stem.startswith(sess["motion_prefix"])
    ])
    if not csv_files:
        print(f"  [SKIP] no motion files for prefix {sess['motion_prefix']}")
        return

    print(f"\n  Session {dataset}/{canonical_session}: {len(csv_files)} trials")

    # Load label data if applicable
    labels_for_subject: dict = {}
    if sess.get("use_labels"):
        all_labels = load_labels()
        labels_for_subject = (
            all_labels.get(sess["labels_dataset_key"], {}).get(sess["labels_subject_key"], {})
        )
        print(f"    Loaded {len(labels_for_subject)} label entries")

    # Determine audio stems and cache AudioSegment per stem
    if "audio_stem_map" in sess:
        audio_stem_map = sess["audio_stem_map"]
    else:
        audio_stem_map = {"_": sess["audio_stems"][0]}

    audio_cache: dict[str, AudioSegment] = {}
    midi_path_cache: dict[str, Path] = {}
    for key, stem in audio_stem_map.items():
        a = base / "Audio" / f"{stem}.wav"
        m = base / "MIDI" / f"{stem}.mid"
        if a.exists():
            print(f"    Loading audio {a.name}...")
            audio_cache[key] = AudioSegment.from_wav(str(a))
            print(f"    Audio {len(audio_cache[key]) / 1000:.1f}s loaded")
        if m.exists():
            midi_path_cache[key] = m

    mods = ["motion_hand", "audio", "midi"]
    if has_body:
        mods.insert(1, "motion_body")
    out = session_dir(dataset, canonical_session, modalities=mods)
    trials = []
    default_task = sess["task"]
    c_tok = sess["canonical_class"]
    n_tok = sess["canonical_n"]
    stem_re = sess["stem_regex"]

    # Pre-scan for an anchor (first CSV with a TimeCode column) so we can
    # fall back to Capture Start Frame math for trials that were exported
    # without SMPTE.
    anchor: dict | None = None
    for csv_path in csv_files:
        try:
            s, _ = get_motion_timecodes(csv_path)
            hdr = _read_motion_header(csv_path)
            anchor = {
                "csf": int(hdr["Capture Start Frame"]),
                "sec": s,
                "rate": float(hdr["Export Frame Rate"]),
                "source": csv_path.name,
            }
            break
        except Exception:
            continue
    if anchor:
        print(f"    TimeCode anchor: {anchor['source']} (CSF {anchor['csf']}, {anchor['sec']:.2f}s, {anchor['rate']} fps)")

    # Dense Piece counter per task (so canonical P is 1..N within each Task)
    piece_counter: dict[int, int] = {}

    for csv_path in tqdm(csv_files, desc=f"  {canonical_session}"):
        raw_stem = csv_path.stem
        mm = stem_re.match(raw_stem)
        if not mm:
            print(f"    SKIP {raw_stem}: stem regex mismatch")
            continue
        groups = mm.groups()
        nn_key = "".join(groups)  # full key joining all regex groups
        # Choose audio/midi source by first capture group if map is per-block
        if "audio_stem_map" in sess:
            block_key = groups[0]
            audio_data = audio_cache.get(block_key)
            midi_path = midi_path_cache.get(block_key)
            audio_stem_label = audio_stem_map.get(block_key, "")
        else:
            audio_data = audio_cache.get("_")
            midi_path = midi_path_cache.get("_")
            audio_stem_label = audio_stem_map.get("_", "")

        # Labels + task determination
        label_entry = labels_for_subject.get(nn_key, {})
        task = label_entry.get("task", default_task)
        status_from_label = label_entry.get("status", "unknown")
        exclusions = label_entry.get("exclusions", [])
        notes = label_entry.get("notes", "")
        if status_from_label == "discarded":
            trials.append({
                "trial_stem": f"{dataset}_{c_tok}_{n_tok}_T{task}_P0_t0",
                "raw_stem": raw_stem,
                "status": "excluded",
                "task": task,
                "label_notes": notes,
            })
            continue

        piece_counter[task] = piece_counter.get(task, 0) + 1
        piece_no = piece_counter[task]
        canonical_stem = f"{dataset}_{c_tok}_{n_tok}_T{task}_P{piece_no}_t1"

        try:
            first_sec, last_sec, used_tc = get_trial_times(csv_path, anchor=anchor)
        except Exception as e:
            print(f"    ERR {raw_stem}: {e}")
            continue
        duration = last_sec - first_sec
        if duration <= 0:
            print(f"    SKIP {raw_stem}: invalid timecode range")
            continue

        out_hand = out / "motion_hand" / f"{canonical_stem}.csv"
        if force or not out_hand.exists():
            shutil.copy2(csv_path, out_hand)
        raw_sources: dict = {"motion_hand": csv_path.name}

        if has_body:
            for ext in (".tsv", ".csv"):
                candidate = body_dir / f"{raw_stem}{ext}"
                if candidate.exists():
                    out_body = out / "motion_body" / f"{canonical_stem}{ext}"
                    if force or not out_body.exists():
                        shutil.copy2(candidate, out_body)
                    raw_sources["motion_body"] = candidate.name
                    break

        out_audio = out / "audio" / f"{canonical_stem}.wav"
        if audio_data is not None and (force or not out_audio.exists()):
            try:
                slice_audio(audio_data, first_sec, last_sec, out_audio)
                raw_sources["audio"] = f"{audio_stem_label}.wav [{first_sec:.3f}-{last_sec:.3f}]"
            except Exception as e:
                print(f"    ERR audio {canonical_stem}: {e}")

        out_midi = out / "midi" / f"{canonical_stem}.mid"
        if midi_path is not None and midi_path.exists() and (force or not out_midi.exists()):
            try:
                slice_midi(midi_path, first_sec, last_sec, out_midi)
                raw_sources["midi"] = f"{audio_stem_label}.mid [{first_sec:.3f}-{last_sec:.3f}]"
            except Exception as e:
                print(f"    ERR midi {canonical_stem}: {e}")

        trials.append({
            "trial_stem": canonical_stem,
            "raw_stem": raw_stem,
            "task": task,
            "canonical_p": piece_no,
            "canonical_t": 1,
            "status": "complete" if audio_data is not None and midi_path is not None else "partial",
            "label_status": status_from_label,
            "label_notes": notes,
            "exclusion_intervals": exclusions,
            "session_start_sec": round(first_sec, 4),
            "session_end_sec": round(last_sec, 4),
            "duration_sec": round(duration, 3),
            "raw_sources": raw_sources,
        })

    write_mapping(dataset, canonical_session, trials, method="timecode", extra={
        "audio_stem_map": audio_stem_map,
        "capabilities": {
            "has_timecode": True,
            "has_video": True,
            "has_body_motion": has_body,
            "body_in_hand_csv": False,
            "needs_xcorr_matching": False,
            "needs_dtw_audio_midi": False,
            "matching_method": "smpte_timecode_slice",
        },
        "uses_trial_labels": sess.get("use_labels", False),
        "task_summary": dict(piece_counter),
    })
    print(f"    -> {out} ({len(trials)} trials)")


def run(dataset: str | None = None, force: bool = False):
    print("Stage 01 s01 -- timecode-based normalization")
    targets = [dataset] if dataset else list(SESSIONS.keys())
    for ds in targets:
        if ds not in SESSIONS:
            continue
        for sess in SESSIONS[ds]:
            process_session(ds, sess, force=force)


if __name__ == "__main__":
    run()
