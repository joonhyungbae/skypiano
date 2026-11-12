"""
Stage 04_piece_split/anchors_220819 / Step 03: Slice all modalities at located windows.

Reads:
  - Data/Pipeline/04_piece_split/anchors_220819/locate_manifest.json  (from s01 - audio-time windows)
  - Data/Pipeline/04_piece_split/anchors_220819/alignment.json        (from s02 - motion offsets)

For each ok entry, writes per-song slices to Stage 04 layout:

  Data/Pipeline/04_piece_split/220819/
    {stage1_stem}_p{N}.wav
    {stage1_stem}_p{N}.mid
    {stage1_stem}_p{N}_hand.csv
    {stage1_stem}_p{N}_body.csv

Motion time translation:
  motion_t = audio_t + align_shift_sec   (per-recording, per-modality)
  motion_frame_idx = round(motion_t * MOTION_FPS)

Motion sources (220819):
  - hand markers (LIM_*, LIF_*, LWrist, ...):
        02_spatial/hand/Aligned/220819/<stage1_stem>.csv
  - body markers (HEADT_*, ACROR_*, C7_*, ...):
        02_spatial/body/Aligned/220819/<stage1_stem>.csv
"""
from __future__ import annotations

import csv
import functools
import json
import sys
from pathlib import Path

_P = Path(__file__).resolve()
while _P.name != "pipeline":
    _P = _P.parent
sys.path.insert(0, str(_P))

from _common.normalized_io import NORMALIZED_ROOT  # noqa: E402

import pretty_midi  # noqa: E402
from pydub import AudioSegment  # noqa: E402

print = functools.partial(print, flush=True)

DATASET = "220819"
OUT_ROOT = Path("Data/Pipeline/04_piece_split/anchors_220819")
LOCATE_PATH = OUT_ROOT / "locate_manifest.json"
ALIGNMENT_PATH = OUT_ROOT / "alignment.json"
PIECE_SPLIT_ROOT = Path("Data/Pipeline/04_piece_split") / DATASET
HAND_ALIGNED_ROOT = Path("Data/Pipeline/02_spatial/hand/Aligned") / DATASET
BODY_ALIGNED_ROOT = Path("Data/Pipeline/02_spatial/body/Aligned") / DATASET

MOTION_FPS = 120  # native 120 Hz for 220819 OptiTrack + Qualisys (verified via row_count/duration)

# Refuse to slice motion when alignment confidence is below this level.
# Entries with 'low' / 'failed' alignment get audio+MIDI slices only.
# 'inferred' = session-median fallback for body (Qualisys is session-level).
# 'stage5_inherited' = offset lifted from Stage 03 midi_motion coarse-to-fine xcorr output.
REQUIRE_ALIGNMENT_CONFIDENCE = {"strong", "medium", "inferred", "stage5_inherited"}


def _load_mapping_cache() -> dict[str, dict]:
    cache: dict[str, dict] = {}
    for mjson in (NORMALIZED_ROOT / DATASET).glob("*/_mapping.json"):
        cache[mjson.parent.name] = json.loads(mjson.read_text(encoding="utf-8"))
    return cache


def _raw_stems_for_trial(mapping: dict, trial_stem: str) -> dict:
    for trial in mapping.get("trials", []):
        if trial["trial_stem"] == trial_stem:
            rs = trial.get("raw_sources", {})
            out = {}
            if rs.get("motion_hand"):
                out["hand"] = Path(rs["motion_hand"]).stem
            if rs.get("motion_body"):
                out["body"] = Path(rs["motion_body"]).stem
            return out
    return {}


def _alignment_index(alignment: dict) -> dict:
    """(session, stage1_stem) -> entry."""
    return {(e["session"], e["stage1_stem"]): e for e in alignment["entries"]}


def slice_audio(src: Path, dst: Path, t_start: float, t_end: float) -> None:
    seg = AudioSegment.from_file(src)[int(t_start * 1000):int(t_end * 1000)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    seg.export(dst, format="wav")


def slice_midi(src: Path, dst: Path, t_start: float, t_end: float) -> None:
    pm = pretty_midi.PrettyMIDI(str(src))
    out = pretty_midi.PrettyMIDI()
    for inst in pm.instruments:
        new_inst = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        for note in inst.notes:
            if note.end <= t_start or note.start >= t_end:
                continue
            new_inst.notes.append(pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=max(0.0, note.start - t_start),
                end=min(t_end - t_start, note.end - t_start),
            ))
        for cc in inst.control_changes:
            if t_start <= cc.time < t_end:
                new_inst.control_changes.append(
                    pretty_midi.ControlChange(number=cc.number, value=cc.value, time=cc.time - t_start)
                )
        if new_inst.notes or new_inst.control_changes:
            out.instruments.append(new_inst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.write(str(dst))


def slice_motion(src: Path, dst: Path, motion_t_start: float, motion_t_end: float, fps: int) -> int:
    """Slice motion CSV by motion-time window (already offset-corrected). A
    window starting before the recording gets empty leading rows, so row 0 is
    always the window start."""
    first = int(round(motion_t_start * fps))
    start_frame = max(0, first)
    end_frame = int(round(motion_t_end * fps))
    dst.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with src.open("r", encoding="utf-8", newline="") as fi, dst.open("w", encoding="utf-8", newline="") as fo:
        reader = csv.reader(fi)
        writer = csv.writer(fo)
        header = next(reader, None)
        if header:
            writer.writerow(header)
            for _ in range(max(0, -first)):
                writer.writerow([""] * len(header))
                n_written += 1
        for i, row in enumerate(reader):
            if i < start_frame:
                continue
            if i >= end_frame:
                break
            writer.writerow(row)
            n_written += 1
    return n_written


def run(dataset: str | None = None, force: bool = False, limit: int | None = None) -> None:
    if dataset is not None and dataset != DATASET:
        print(f"[s03_slice] skip - only {DATASET} supported")
        return
    if not LOCATE_PATH.is_file():
        print(f"[s03_slice] locate_manifest missing; run s01_locate first")
        return
    if not ALIGNMENT_PATH.is_file():
        print(f"[s03_slice] alignment missing; run s02_align_motion first")
        return

    locate = json.loads(LOCATE_PATH.read_text(encoding="utf-8"))
    alignment = json.loads(ALIGNMENT_PATH.read_text(encoding="utf-8"))
    align_idx = _alignment_index(alignment)

    entries = [e for e in locate["entries"] if e.get("ok")]
    if limit:
        entries = entries[:limit]
    print(f"[s03_slice] {len(entries)} entries to slice (limit={limit})")

    mapping_cache = _load_mapping_cache()
    PIECE_SPLIT_ROOT.mkdir(parents=True, exist_ok=True)

    stats = {"audio": 0, "midi": 0, "hand": 0, "body": 0, "skipped_existing": 0,
             "hand_no_source": 0, "hand_bad_align": 0,
             "body_no_source": 0, "body_bad_align": 0}

    for e in entries:
        out_stem = e["out_stem"]
        t_start = e["t_start_sec"]
        t_end = e["t_end_sec"]
        session = e["session"]
        stage1_stem = e["stage1_stem"]

        session_dir = NORMALIZED_ROOT / DATASET / session
        src_wav = session_dir / "audio" / f"{stage1_stem}.wav"
        src_mid = session_dir / "midi" / f"{stage1_stem}.mid"

        mapping = mapping_cache.get(session, {})
        raw_stems = _raw_stems_for_trial(mapping, stage1_stem)
        align = align_idx.get((session, stage1_stem), {})

        # audio
        dst_wav = PIECE_SPLIT_ROOT / f"{out_stem}.wav"
        if force or not dst_wav.is_file():
            slice_audio(src_wav, dst_wav, t_start, t_end)
            stats["audio"] += 1
        else:
            stats["skipped_existing"] += 1

        # midi
        dst_mid = PIECE_SPLIT_ROOT / f"{out_stem}.mid"
        if force or not dst_mid.is_file():
            slice_midi(src_mid, dst_mid, t_start, t_end)
            stats["midi"] += 1

        # hand motion - gated by alignment confidence
        if "hand" not in raw_stems:
            stats["hand_no_source"] += 1
        else:
            hand_align = align.get("hand", {})
            if hand_align.get("confidence") in REQUIRE_ALIGNMENT_CONFIDENCE:
                shift = hand_align["align_shift_sec"]
                src_hand = HAND_ALIGNED_ROOT / f"{stage1_stem}.csv"
                if src_hand.is_file():
                    dst_hand = PIECE_SPLIT_ROOT / f"{out_stem}_hand.csv"
                    if force or not dst_hand.is_file():
                        n = slice_motion(src_hand, dst_hand, t_start + shift, t_end + shift, MOTION_FPS)
                        if n > 0:
                            stats["hand"] += 1
                        else:
                            dst_hand.unlink(missing_ok=True)  # empty slice; drop
                            stats["hand_bad_align"] += 1
                else:
                    stats["hand_no_source"] += 1
            else:
                stats["hand_bad_align"] += 1

        # body motion - shares hand clock (Stage 02 body/Aligned has been
        # body->hand wrist-xcorr-aligned already), so reuse hand's shift.
        if "body" not in raw_stems:
            stats["body_no_source"] += 1
        else:
            hand_align = align.get("hand", {})
            if hand_align.get("confidence") in REQUIRE_ALIGNMENT_CONFIDENCE:
                shift = hand_align["align_shift_sec"]
                src_body = BODY_ALIGNED_ROOT / f"{stage1_stem}.csv"
                if src_body.is_file():
                    dst_body = PIECE_SPLIT_ROOT / f"{out_stem}_body.csv"
                    if force or not dst_body.is_file():
                        n = slice_motion(src_body, dst_body, t_start + shift, t_end + shift, MOTION_FPS)
                        if n > 0:
                            stats["body"] += 1
                        else:
                            dst_body.unlink(missing_ok=True)
                            stats["body_bad_align"] += 1
                else:
                    stats["body_no_source"] += 1
            else:
                stats["body_bad_align"] += 1

    print(f"[s03_slice] done. stats: {stats}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N entries (for piloting)")
    args = ap.parse_args()
    run(dataset=args.dataset, force=args.force, limit=args.limit)
