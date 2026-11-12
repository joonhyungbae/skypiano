"""
Stage 04 Step 01: Timecode-based per-trial audio slicing (240809, 240828, 240523)

These datasets have session-level WAV recordings and SMPTE timecodes that map
each trial to a time range within the session. Stage 03 midi_motion provides the verified
alignment times (timecode_start_sec / timecode_end_sec).

Input:
  - Stage 03 midi_motion alignment JSON per trial (verified timecode or xcorr times)
  - Raw session WAV files
  - Stage 01 _mapping.json for raw_stem → trial_stem mapping

Output:
  Data/Pipeline/04_piece_split/{dataset}/{trial_stem}.wav
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path

from pydub import AudioSegment
from tqdm import tqdm

print = functools.partial(print, flush=True)

RAW_ROOT = Path("Data/Raw")
STAGE0_ROOT = Path("Data/Pipeline/01_ingest/Normalized")
STAGE5_ROOT = Path("Data/Pipeline/03_sync/midi_motion/Aligned")
OUT_ROOT = Path("Data/Pipeline/04_piece_split")

DATASETS = ["240809", "240828", "240523"]


def load_stage5_alignments(dataset: str) -> dict[str, dict]:
    """Load all Stage 03 midi_motion alignment JSONs. Keyed by raw_stem."""
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


def build_raw_to_trial_map(dataset: str) -> dict[str, str]:
    """Build raw_stem → trial_stem from Stage 01 _mapping.json."""
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
            rs, ts = trial.get("raw_stem"), trial.get("trial_stem")
            if rs and ts:
                result[rs] = ts
    return result


def find_session_wav(dataset: str, midi_stem: str) -> Path | None:
    """Find session WAV matching the MIDI stem in Raw directory."""
    for root, _dirs, files in os.walk(RAW_ROOT / dataset):
        for f in files:
            if f.lower().endswith(".wav") and Path(f).stem == midi_stem:
                return Path(root) / f
    return None


def process_dataset(dataset: str, force: bool = False):
    """Slice session audio into per-trial WAVs using Stage 03 verified times."""
    print(f"\n  [{dataset}] Loading Stage 03 alignments...")
    alignments = load_stage5_alignments(dataset)
    raw_to_trial = build_raw_to_trial_map(dataset)

    if not alignments:
        print(f"  [{dataset}] No Stage 03 alignments — skipped")
        return

    # Group by session MIDI stem
    by_session: dict[str, list[tuple[str, dict]]] = {}
    for raw_stem, data in alignments.items():
        sess = data.get("session_midi", "")
        by_session.setdefault(sess, []).append((raw_stem, data))

    out_dir = OUT_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    total_success, total_skip, total_fail = 0, 0, 0

    for sess_stem, trials in sorted(by_session.items()):
        wav_path = find_session_wav(dataset, sess_stem)
        if not wav_path:
            print(f"  [{dataset}] SKIP session {sess_stem}: WAV not found")
            total_fail += len(trials)
            continue

        print(f"  [{dataset}] {sess_stem}: loading {wav_path.name}...")
        audio = AudioSegment.from_wav(str(wav_path))
        audio_duration_sec = len(audio) / 1000.0
        print(f"  [{dataset}] {sess_stem}: {audio_duration_sec:.1f}s, {len(trials)} trials")

        for raw_stem, data in tqdm(sorted(trials), desc=f"    {sess_stem}"):
            trial_stem = raw_to_trial.get(raw_stem, raw_stem)
            out_path = out_dir / f"{trial_stem}.wav"

            if out_path.exists() and not force:
                total_skip += 1
                continue

            start = data["timecode_start_sec"]
            end = data["timecode_end_sec"]

            if end > audio_duration_sec + 1.0:
                print(f"    WARN {raw_stem}: end {end:.1f}s > audio {audio_duration_sec:.1f}s")

            try:
                start_ms = int(start * 1000)
                end_ms = int(min(end, audio_duration_sec) * 1000)
                trimmed = audio[start_ms:end_ms]
                trimmed.export(str(out_path), format="wav")
                total_success += 1
            except Exception as e:
                print(f"    ERR {raw_stem}: {e}")
                total_fail += 1

    print(f"  [{dataset}] Done: {total_success} sliced, {total_skip} skipped, {total_fail} failed")
    return total_success, total_skip, total_fail


def run(dataset: str | None = None, force: bool = False):
    print("Stage 04 s01: Timecode-based audio slicing (24-year datasets)")
    print(f"Output: {OUT_ROOT}")

    targets = [dataset] if dataset else DATASETS
    grand_success, grand_skip, grand_fail = 0, 0, 0

    for ds in targets:
        if ds not in DATASETS:
            print(f"  [{ds}] Not a timecode dataset — skipped")
            continue
        result = process_dataset(ds, force=force)
        if result:
            grand_success += result[0]
            grand_skip += result[1]
            grand_fail += result[2]

    print(f"\n{'='*60}")
    print(f"s01 complete: {grand_success} sliced, {grand_skip} skipped, {grand_fail} failed")
    for ds in targets:
        ds_dir = OUT_ROOT / ds
        if ds_dir.is_dir():
            n = len([f for f in ds_dir.iterdir() if f.suffix == ".wav"])
            print(f"  {ds}: {n} WAV files")


if __name__ == "__main__":
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else None
    force = "--force" in sys.argv
    run(dataset=ds, force=force)
