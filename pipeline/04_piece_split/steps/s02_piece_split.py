"""
Stage 04 Step 02: Per-piece splitting of all modalities

Each trial may contain multiple pieces (especially task 0/1). This step
detects piece boundaries from MIDI note gaps and splits ALL modalities
into individual pieces:
  - Audio (WAV)
  - MIDI
  - Motion hand (CSV, 120fps from Stage 02 hand)
  - Motion body+hand (CSV from Stage 02 body, if available)
  - Video (MP4, 4 camera angles from Raw, if available)

Input:
  - Data/Pipeline/04_piece_split/{dataset}/{trial_stem}.wav   (from s01)
  - Data/Pipeline/03_sync/midi_motion/Aligned/{dataset}/{raw_stem}.mid
  - Data/Pipeline/02_spatial/hand/Aligned/{dataset}/{trial_stem}.csv   (hand motion, 120fps)
  - Data/Pipeline/02_spatial/body/Aligned/{dataset}/{raw_stem}.csv     (body+hand, if exists)
  - Data/Raw/{dataset}/.../Video/{angle}/{raw_stem}.mp4  (if exists)
  - Stage 01 _mapping.json (task, exclusions, session_start)

Output per piece:
  Data/Pipeline/04_piece_split/{dataset}/{trial_stem}_p{N}.wav
  Data/Pipeline/04_piece_split/{dataset}/{trial_stem}_p{N}.mid
  Data/Pipeline/04_piece_split/{dataset}/{trial_stem}_p{N}_hand.csv
  Data/Pipeline/04_piece_split/{dataset}/{trial_stem}_p{N}_body.csv   (if body available)
  Data/Pipeline/04_piece_split/{dataset}/{trial_stem}_p{N}_{angle}.mp4 (if video available)
  Data/Pipeline/04_piece_split/{dataset}/_pieces.json   (manifest)
"""
from __future__ import annotations

import functools
import json
import os
import subprocess
import shutil
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi
from pydub import AudioSegment
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _common.known_absences import body_absence_entry

print = functools.partial(print, flush=True)

RAW_ROOT = Path("Data/Raw")
STAGE0_ROOT = Path("Data/Pipeline/01_ingest/Normalized")
STAGE2_ROOT = Path("Data/Pipeline/02_spatial/hand/Aligned")  # hand motion (trial_stem names)
STAGE3_ROOT = Path("Data/Pipeline/02_spatial/body/Aligned")  # body+hand motion (raw_stem names)
STAGE5_ROOT = Path("Data/Pipeline/03_sync/midi_motion/Aligned")
OUT_ROOT = Path("Data/Pipeline/04_piece_split")

DATASETS = ["240809", "240828", "240523"]
MOTION_FPS = 120.0
VIDEO_ANGLES = ["Left", "Right", "RightLow", "Top"]

# Minimum gap (seconds) to consider a piece boundary
MIN_GAP_SEC = 4.0  # silence gap defining a piece boundary; chosen empirically over the 240828 corpus
# Padding around each piece (seconds before first note, after last note)
PAD_BEFORE = 0.5  # leading pad to retain attack/pre-roll context for the first note
PAD_AFTER = 1.0   # trailing pad to retain decay/release tail past the last note
# Noise filter: piece dropped if BOTH conditions hold
MIN_PIECE_NOTES = 10  # below this is typically a stray test/tuning fragment, not a real take
MIN_PIECE_DUR = 2.0   # below this duration cannot constitute a meaningful piece take
# Exclusion handling: piece dropped entirely if overlap ratio >= this
EXCLUSION_DROP_RATIO = 0.8  # >=80% overlap with an exclusion span means the piece is essentially excluded
# Review: piece flagged as borderline if midpoint within this of an exclusion edge
BORDERLINE_DIST = 2.0


@dataclass
class Piece:
    piece_idx: int
    start_sec: float
    end_sec: float
    n_notes: int
    first_onset: float
    last_offset: float


def build_body_metadata(dataset: str, trial_stem: str, has_body: bool) -> dict:
    """Annotate trial-level body availability for downstream manifests/logs."""
    meta = {"has_body": has_body, "body_status": "present" if has_body else "missing"}
    if has_body:
        return meta
    entry = body_absence_entry(dataset, trial_stem)
    if entry is None:
        return meta
    meta["body_status"] = "expected_absent"
    meta["body_absence_id"] = entry.get("id")
    meta["body_absence_reason"] = entry.get("reason", "")
    return meta


# ── Piece boundary detection with exclusion-aware splitting ──

def _gap_split(notes: list[tuple[float, float]], gap: float) -> list[dict]:
    """Split a sorted note list at gaps ≥ `gap`. Returns list of {first_onset,last_offset,n_notes,notes}."""
    if not notes:
        return []
    onsets = [n[0] for n in notes]
    offsets = [n[1] for n in notes]
    splits = [i for i in range(1, len(onsets)) if onsets[i] - onsets[i - 1] >= gap]
    boundaries = [0] + splits + [len(onsets)]
    out = []
    for si, ei in zip(boundaries[:-1], boundaries[1:]):
        first = onsets[si]
        last = max(offsets[si:ei])
        out.append({
            "first_onset": round(first, 3),
            "last_offset": round(last, 3),
            "n_notes": ei - si,
            "notes": notes[si:ei],
        })
    return out


def _clamp_pad(first: float, last: float, exclusions_rel: list[tuple[float, float]]) -> tuple[float, float]:
    """Apply PAD_BEFORE/PAD_AFTER, but don't let pads cross into an exclusion zone."""
    start = max(0.0, first - PAD_BEFORE)
    end = last + PAD_AFTER
    for ex_s, ex_e in exclusions_rel:
        if ex_e <= first and ex_e > start:  # exclusion ends before piece → clamp start
            start = ex_e
        if ex_s >= last and ex_s < end:  # exclusion starts after piece → clamp end
            end = ex_s
    return round(start, 3), round(end, 3)


def detect_pieces(
    midi_path: Path,
    exclusions_rel: list[tuple[float, float]] | None = None,
    gap_threshold: float = MIN_GAP_SEC,
) -> tuple[list[Piece], dict]:
    """
    Detect pieces from MIDI, applying exclusion at the note level so that
    exclusions inside a piece split it into sub-pieces rather than dropping
    the whole piece.

    Returns (pieces, review) where `review` collects diagnostics:
      - original_pieces: pieces detected before exclusion/noise filtering
      - fully_excluded: pieces whose exclusion-overlap ratio ≥ EXCLUSION_DROP_RATIO
      - split_by_exclusion: pieces split into 2+ sub-pieces by interior exclusions
      - noise_filtered: sub-pieces dropped as noise (n_notes<MIN AND dur<MIN)
      - borderline: kept pieces with midpoint within BORDERLINE_DIST of an excl edge
    """
    exclusions_rel = exclusions_rel or []
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = sorted((n.start, n.end) for inst in pm.instruments for n in inst.notes)
    review: dict = {
        "original_pieces": [],
        "fully_excluded": [],
        "split_by_exclusion": [],
        "noise_filtered": [],
        "borderline": [],
    }
    if not notes:
        return [], review

    # Phase 1: detect original pieces (pre-exclusion)
    originals = _gap_split(notes, gap_threshold)
    review["original_pieces"] = [
        {k: v for k, v in o.items() if k != "notes"} for o in originals
    ]

    # Phase 2: apply exclusions per piece (note-level filter + re-split)
    kept_raw: list[dict] = []
    for orig in originals:
        orig_range = (orig["first_onset"], orig["last_offset"])
        piece_dur = max(orig_range[1] - orig_range[0], 1e-6)
        intersections = []
        for ex_s, ex_e in exclusions_rel:
            s = max(ex_s, orig_range[0])
            e = min(ex_e, orig_range[1])
            if e > s:
                intersections.append((s, e))

        if not intersections:
            kept_raw.append(orig)
            continue

        overlap_ratio = sum(e - s for s, e in intersections) / piece_dur
        if overlap_ratio >= EXCLUSION_DROP_RATIO:
            review["fully_excluded"].append({
                "first_onset": orig["first_onset"],
                "last_offset": orig["last_offset"],
                "n_notes": orig["n_notes"],
                "overlap_ratio": round(overlap_ratio, 3),
                "exclusion_intersections": [(round(s, 3), round(e, 3)) for s, e in intersections],
            })
            continue

        # Partial overlap → drop notes whose onset lies inside any exclusion, re-split
        kept_notes = [
            n for n in orig["notes"]
            if not any(s <= n[0] <= e for s, e in intersections)
        ]
        sub_pieces = _gap_split(kept_notes, gap_threshold)
        if len(sub_pieces) != 1 or sub_pieces[0]["n_notes"] != orig["n_notes"]:
            review["split_by_exclusion"].append({
                "original": {k: v for k, v in orig.items() if k != "notes"},
                "intersections": [(round(s, 3), round(e, 3)) for s, e in intersections],
                "sub_pieces": [
                    {k: v for k, v in sp.items() if k != "notes"} for sp in sub_pieces
                ],
            })
        kept_raw.extend(sub_pieces)

    # Phase 3: noise filter
    survivors: list[dict] = []
    for p in kept_raw:
        dur = p["last_offset"] - p["first_onset"]
        if p["n_notes"] < MIN_PIECE_NOTES and dur < MIN_PIECE_DUR:
            review["noise_filtered"].append({
                "first_onset": p["first_onset"],
                "last_offset": p["last_offset"],
                "n_notes": p["n_notes"],
                "dur": round(dur, 3),
            })
        else:
            survivors.append(p)

    # Phase 4: build Piece objects (pad-clamped) and borderline detection
    pieces: list[Piece] = []
    for idx, p in enumerate(survivors):
        start_sec, end_sec = _clamp_pad(p["first_onset"], p["last_offset"], exclusions_rel)
        pieces.append(Piece(
            piece_idx=idx + 1,
            start_sec=start_sec,
            end_sec=end_sec,
            n_notes=p["n_notes"],
            first_onset=p["first_onset"],
            last_offset=p["last_offset"],
        ))
        mid = (p["first_onset"] + p["last_offset"]) / 2
        for ex_s, ex_e in exclusions_rel:
            d = min(abs(mid - ex_s), abs(mid - ex_e))
            if d < BORDERLINE_DIST:
                review["borderline"].append({
                    "piece_idx": idx + 1,
                    "midpoint": round(mid, 3),
                    "exclusion": (round(ex_s, 3), round(ex_e, 3)),
                    "dist_to_edge": round(d, 3),
                })
                break

    # A note held across the silence gap (e.g. a key left down) carries
    # last_offset past the next piece's first onset; windows stop where the
    # next piece starts so no two pieces share a stretch of the trial.
    for a, b in zip(pieces, pieces[1:]):
        a.end_sec = min(a.end_sec, b.start_sec)

    return pieces, review


# ── Slicing functions ──

def slice_midi(midi_path: Path, start: float, end: float) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    new_pm = pretty_midi.PrettyMIDI()
    dur = end - start
    for inst in pm.instruments:
        ni = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        for n in inst.notes:
            if n.start < end and n.end > start:
                ni.notes.append(pretty_midi.Note(
                    velocity=n.velocity, pitch=n.pitch,
                    start=max(0.0, n.start - start), end=min(dur, n.end - start)))
        for cc in inst.control_changes:
            if start <= cc.time < end:
                ni.control_changes.append(
                    pretty_midi.ControlChange(number=cc.number, value=cc.value, time=cc.time - start))
        new_pm.instruments.append(ni)
    return new_pm


def timecode_to_sec(tc: str) -> float:
    """Motive SMPTE "HH:MM:SS:FF.SS" (30 fps frames, 4 subframes) -> seconds."""
    hh, mm, ss, ff = tc.split(":")
    frame, _, sub = ff.partition(".")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(frame) / 30 + int(sub or 0) / 120


def slice_motion_csv(csv_path: Path, start: float, end: float, fps: float, out_path: Path,
                     clock_start: float | None = None):
    """Slice a motion CSV by time range (seconds). Assumes rows = frames at given fps.

    With `clock_start` (session seconds of trial time 0) and a TimeCode column,
    frame 0 is the row whose timecode is closest to it, so a motion file that
    starts earlier than the trial MIDI/audio clock is cut at the same instants.
    """
    df = pd.read_csv(csv_path)
    offset = 0
    if clock_start is not None and "TimeCode" in df.columns:
        tc = df["TimeCode"].dropna()
        if len(tc):
            secs = tc.map(timecode_to_sec).to_numpy()
            offset = int(tc.index[np.argmin(np.abs(secs - clock_start))])
    # Drop TimeCode column if present (not meaningful after slicing)
    if "TimeCode" in df.columns:
        df = df.drop(columns=["TimeCode"])
    start_frame = offset + int(start * fps)
    end_frame = offset + int(end * fps)
    sliced = df.iloc[start_frame:end_frame].reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sliced.to_csv(out_path, index=False)


def find_trial_videos(dataset: str, raw_stem: str) -> dict[str, Path]:
    """Find video files for a trial across camera angles. Returns angle → path."""
    result = {}
    # Search pattern: Raw/{dataset}/.../Video/{angle}/{file matching raw_stem}
    ds_dir = RAW_ROOT / dataset
    for root, dirs, files in os.walk(ds_dir):
        if "Video" not in root:
            continue
        for f in files:
            if not f.lower().endswith(".mp4"):
                continue
            # Match by raw_stem in filename
            if raw_stem in f:
                # Determine angle from parent dir
                angle = Path(root).name
                if angle in VIDEO_ANGLES:
                    result[angle] = Path(root) / f
    return result


def slice_video(video_path: Path, start: float, end: float, out_path: Path):
    """Slice video using ffmpeg. Requires ffmpeg in PATH."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end - start
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-c", "copy",  # fast copy without re-encoding
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


# ── Data loaders ──

def load_trial_info(dataset: str) -> dict[str, dict]:
    result = {}
    ds_dir = STAGE0_ROOT / dataset
    if not ds_dir.is_dir():
        return result
    for sess in ds_dir.iterdir():
        mp = sess / "_mapping.json"
        if not mp.is_file():
            continue
        with open(mp, encoding="utf-8") as f:
            mapping = json.load(f)
        for trial in mapping.get("trials", []):
            rs = trial.get("raw_stem")
            if rs:
                result[rs] = trial
    return result


def load_stage5_alignments(dataset: str) -> dict[str, dict]:
    result = {}
    ds_dir = STAGE5_ROOT / dataset
    if not ds_dir.is_dir():
        return result
    for f in ds_dir.iterdir():
        if f.suffix == ".json" and not f.name.startswith("_"):
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            result[data.get("stem", f.stem)] = data
    return result


# ── Check ffmpeg availability ──

def has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


# ── Main processing ──

def process_dataset(dataset: str, force: bool = False, skip_video: bool = False):
    print(f"\n{'='*60}")
    print(f"{dataset}: Piece-level splitting (audio + MIDI + motion + video)")
    print(f"{'='*60}")

    trial_info = load_trial_info(dataset)
    alignments = load_stage5_alignments(dataset)
    out_dir = OUT_ROOT / dataset

    if not out_dir.is_dir():
        print(f"  No s01 output — run s01 first")
        return

    do_video = not skip_video and has_ffmpeg()
    if not do_video and not skip_video:
        print("  [WARN] ffmpeg not found — skipping video slicing")

    raw_to_trial = {info.get("raw_stem"): info.get("trial_stem")
                    for info in trial_info.values()
                    if info.get("raw_stem") and info.get("trial_stem")}

    manifest = []
    review_entries = []
    stats = {
        "pieces": 0, "fully_excluded": 0, "split_by_exclusion": 0,
        "noise_filtered": 0, "borderline": 0,
        "motion_hand": 0, "motion_body": 0, "video": 0,
        "body_expected_absent_trials": 0, "body_missing_trials": 0,
    }

    midi_files = sorted(
        STAGE5_ROOT / dataset / f
        for f in os.listdir(STAGE5_ROOT / dataset)
        if f.endswith(".mid")
    )

    for midi_path in tqdm(midi_files, desc=f"  {dataset}"):
        raw_stem = midi_path.stem
        trial_stem = raw_to_trial.get(raw_stem, raw_stem)
        info = trial_info.get(raw_stem, {})
        alignment = alignments.get(raw_stem, {})
        task = info.get("task", -1)

        # Check trial-level WAV exists (from s01)
        trial_wav = out_dir / f"{trial_stem}.wav"
        if not trial_wav.exists():
            continue

        # Exclusion zones: session-absolute → trial-relative
        session_start = alignment.get("timecode_start_sec")
        if session_start is None:
            session_start = info.get("session_start_sec", 0)
        exclusions_rel = [
            (ex[0] - session_start, ex[1] - session_start)
            for ex in info.get("exclusion_intervals", [])
        ]

        # Detect pieces (with exclusion-aware splitting + noise filter)
        pieces, review = detect_pieces(midi_path, exclusions_rel=exclusions_rel)

        # Always record a review entry (even when no issues) so consumers can index by trial
        if (review["fully_excluded"] or review["split_by_exclusion"]
                or review["noise_filtered"] or review["borderline"]):
            review_entries.append({
                "trial_stem": trial_stem,
                "raw_stem": raw_stem,
                "task": task,
                "exclusions_rel": [(round(s, 3), round(e, 3)) for s, e in exclusions_rel],
                **review,
            })
            stats["fully_excluded"] += len(review["fully_excluded"])
            stats["split_by_exclusion"] += len(review["split_by_exclusion"])
            stats["noise_filtered"] += len(review["noise_filtered"])
            stats["borderline"] += len(review["borderline"])

        # Locate motion CSVs
        hand_csv = STAGE2_ROOT / dataset / f"{trial_stem}.csv"
        body_csv = STAGE3_ROOT / dataset / f"{raw_stem}.csv"
        has_hand = hand_csv.exists()
        has_body = body_csv.exists()
        body_meta = build_body_metadata(dataset, trial_stem, has_body)
        if body_meta["body_status"] == "expected_absent":
            stats["body_expected_absent_trials"] += 1
            print(f"  [body expected-absent] {trial_stem}: {body_meta['body_absence_reason']}")
        elif body_meta["body_status"] == "missing":
            stats["body_missing_trials"] += 1
            print(f"  [WARN] body CSV missing for {trial_stem}: {body_csv}")

        if not pieces:
            manifest.append({
                "trial_stem": trial_stem,
                "raw_stem": raw_stem,
                "task": task,
                "pieces": [],
                "n_pieces": 0,
                **body_meta,
            })
            continue

        # Locate videos
        videos = find_trial_videos(dataset, raw_stem) if do_video else {}

        # Load audio once per trial
        audio = AudioSegment.from_wav(str(trial_wav))

        entry = {
            "trial_stem": trial_stem,
            "raw_stem": raw_stem,
            "task": task,
            "pieces": [],
            **body_meta,
        }

        for p in pieces:
            prefix = f"{trial_stem}_p{p.piece_idx}"

            # Audio
            out_wav = out_dir / f"{prefix}.wav"
            if not out_wav.exists() or force:
                trimmed = audio[int(p.start_sec * 1000):int(p.end_sec * 1000)]
                trimmed.export(str(out_wav), format="wav")

            # MIDI
            out_mid = out_dir / f"{prefix}.mid"
            if not out_mid.exists() or force:
                pm = slice_midi(midi_path, p.start_sec, p.end_sec)
                pm.write(str(out_mid))

            # Motion hand
            if has_hand:
                out_hand = out_dir / f"{prefix}_hand.csv"
                if not out_hand.exists() or force:
                    slice_motion_csv(hand_csv, p.start_sec, p.end_sec, MOTION_FPS, out_hand,
                                     clock_start=alignment.get("timecode_start_sec"))
                stats["motion_hand"] += 1

            # Motion body+hand
            if has_body:
                out_body = out_dir / f"{prefix}_body.csv"
                if not out_body.exists() or force:
                    slice_motion_csv(body_csv, p.start_sec, p.end_sec, MOTION_FPS, out_body)
                stats["motion_body"] += 1

            # Video (each angle)
            for angle, vpath in videos.items():
                out_vid = out_dir / f"{prefix}_{angle}.mp4"
                if not out_vid.exists() or force:
                    try:
                        slice_video(vpath, p.start_sec, p.end_sec, out_vid)
                        stats["video"] += 1
                    except Exception as e:
                        print(f"    video ERR {prefix}_{angle}: {e}")

            entry["pieces"].append(asdict(p))
            stats["pieces"] += 1

        entry["n_pieces"] = len(pieces)
        manifest.append(entry)

    # Write manifest
    manifest_path = out_dir / "_pieces.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Write review log
    review_path = out_dir / "_review.json"
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": dataset,
            "thresholds": {
                "MIN_GAP_SEC": MIN_GAP_SEC,
                "MIN_PIECE_NOTES": MIN_PIECE_NOTES,
                "MIN_PIECE_DUR": MIN_PIECE_DUR,
                "EXCLUSION_DROP_RATIO": EXCLUSION_DROP_RATIO,
                "BORDERLINE_DIST": BORDERLINE_DIST,
            },
            "summary": {
                "trials": len(manifest),
                "pieces": stats["pieces"],
                "fully_excluded": stats["fully_excluded"],
                "split_by_exclusion": stats["split_by_exclusion"],
                "noise_filtered": stats["noise_filtered"],
                "borderline": stats["borderline"],
                "body_expected_absent_trials": stats["body_expected_absent_trials"],
                "body_missing_trials": stats["body_missing_trials"],
            },
            "entries": review_entries,
        }, f, indent=2, ensure_ascii=False)

    print(f"  {len(manifest)} trials → {stats['pieces']} pieces")
    print(f"  Review: {stats['fully_excluded']} fully_excluded, "
          f"{stats['split_by_exclusion']} split_by_exclusion, "
          f"{stats['noise_filtered']} noise_filtered, "
          f"{stats['borderline']} borderline")
    print(f"  Motion: {stats['motion_hand']} hand, {stats['motion_body']} body")
    print(f"  Body trials: {stats['body_expected_absent_trials']} expected-absent, "
          f"{stats['body_missing_trials']} unexpected-missing")
    print(f"  Video: {stats['video']} clips")


def run(dataset: str | None = None, force: bool = False, skip_video: bool = False):
    print("Stage 04 s02: Piece-level splitting (all modalities)")
    print(f"Output: {OUT_ROOT}")

    targets = [dataset] if dataset else DATASETS
    for ds in targets:
        if ds not in DATASETS:
            print(f"  [{ds}] Not supported — skipped")
            continue
        process_dataset(ds, force=force, skip_video=skip_video)

    # Summary
    print(f"\n{'='*60}")
    for ds in targets:
        ds_dir = OUT_ROOT / ds
        if not ds_dir.is_dir():
            continue
        wavs = len([f for f in ds_dir.iterdir() if f.suffix == ".wav" and "_p" in f.name])
        mids = len([f for f in ds_dir.iterdir() if f.suffix == ".mid" and "_p" in f.name])
        csvs = len([f for f in ds_dir.iterdir() if f.suffix == ".csv" and "_hand" in f.name])
        vids = len([f for f in ds_dir.iterdir() if f.suffix == ".mp4"])
        print(f"  {ds}: {wavs} WAV, {mids} MIDI, {csvs} motion, {vids} video")


if __name__ == "__main__":
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else None
    force = "--force" in sys.argv
    skip_video = "--no-video" in sys.argv
    run(dataset=ds, force=force, skip_video=skip_video)
