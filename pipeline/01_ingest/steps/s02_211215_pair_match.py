"""
Stage 01 step 02: Filename pairing (211215 cohort)

211215 already has trial-level audio + MIDI + motion files with consistent
stems (e.g. 211215_C0_N0_T3_P1_t1.{csv,wav,mid}). Naming is perfect, so we
pair by filename and copy raw files into the normalized layout.

Status breakdown (Raw as source of truth):
  - 109 trials total (T3_P1..P50, 9 pieces have 3 takes, others 2)
  - 90 trials complete (motion + audio + midi)
  - 19 trials motion-missing (audio + midi only, no motion capture)

The mapping JSON groups trials by completeness status so downstream stages
can filter easily:
  - "complete" list: full multimodal trials (Stage 03 motion-MIDI sync OK)
  - "motion_missing" list: audio-only / midi-only trials that Stage 03 audio-MIDI can
    still process, but Stage 03 motion-MIDI and Stage 02 body registration should
    skip them

Capabilities (for downstream decision-making):
  - has_timecode: false     (no OptiTrack SMPTE in this dataset)
  - has_video: false        (no camera recordings)
  - has_body_motion: true   (but integrated into the Hand CSV, not separate)
  - body_in_hand_csv: true  (Stage 02 body registration should skip)

The raw Vicon CSV header reads "# piano:LIM,piano:LIM.1,piano:LIM.2,...";
motion is written with plain {marker}_x/_y/_z columns (the .1/.2 columns are
y/z, checked against the earlier converted export). Piece names and the
recording memos come from Meta/<piece info>.xlsx: P{k} is its Take{k} and
t{n} its Trial tn.

Audio and MIDI of a trial were recorded separately: the MIDI starts at the
first note, the audio 2-14 s earlier. Each MIDI is moved onto its audio's
clock (one shift, src/audio_midi_onsets: placed key by key, then by the
median note attack lag) so that trial time = audio time; the match is recorded under
`audio_midi_match`.

Output:
  Data/Pipeline/01_ingest/Normalized/211215/C0_N0/
      motion/{stem}.csv     (90 files)
      audio/{stem}.wav      (109 files)
      midi/{stem}.mid       (109 files)
      _mapping.json         (enriched schema)
"""
from __future__ import annotations
import functools
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # pipeline/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pipeline/01_ingest/ (src)
from _common.normalized_io import RAW_ROOT, session_dir, write_mapping  # noqa: E402

print = functools.partial(print, flush=True)

DATASET = "211215"
SESSION = "C0_N0"

MOTION_DIR = RAW_ROOT / "211215/C0/N0/Motion/Hand"
AUDIO_DIR = RAW_ROOT / "211215/C0/N0/Audio"
MIDI_DIR = RAW_ROOT / "211215/C0/N0/MIDI"
META_DIR = RAW_ROOT / "211215/C0/N0/Meta"
AUDIO_PAD_S = 1.0


def write_motion(src: Path, dst: Path) -> None:
    """Raw Vicon CSV -> {marker}_x/_y/_z columns."""
    import pandas as pd
    df = pd.read_csv(src)
    names = []
    for c in df.columns:
        base, _, k = c.lstrip("# ").split(":", 1)[-1].partition(".")
        names.append(f"{base}_{'xyz'[int(k) if k else 0]}")
    df.columns = names
    df.to_csv(dst, index=False)


def audio_clock_midi(wav: Path, mid: Path, dst: Path) -> dict | None:
    """Write `mid` shifted onto the clock of `wav`; None when it has too few notes."""
    import numpy as np
    import pretty_midi
    from src import audio_midi_onsets as am
    notes = sorted((n.start, n.pitch) for i in pretty_midi.PrettyMIDI(str(mid)).instruments for n in i.notes)
    if len(notes) < 8:
        return None
    onsets = np.array([t for t, _ in notes])
    # the audio recording runs from before the first note to after the last
    coarse, margin = am.pitch_offset(wav, notes, pad_s=AUDIO_PAD_S)
    if not np.isfinite(coarse):
        return None
    first, n_attacks = am.attack_refine(wav, onsets, coarse)
    am.shift_midi(mid, dst, first - onsets[0])
    return {"first_note_in_audio_sec": round(first, 3), "pitch_match_sec": round(coarse, 3),
            "margin": round(margin, 3), "attack_notes": n_attacks,
            "midi_time_of_first_note_sec": round(float(onsets[0]), 3)}


def piece_info() -> dict[tuple[int, int], dict]:
    """(P, t) -> {name, memo} from the piece info sheet (first .xlsx in Meta/)."""
    import pandas as pd
    xs = sorted(META_DIR.glob("*.xlsx"))
    if not xs:
        return {}
    x = pd.read_excel(xs[0])
    x.columns = [str(c).strip() for c in x.columns]
    out = {}
    for _, r in x.iterrows():
        k, t = re.search(r"\d+", str(r["Take"])), re.search(r"\d+", str(r["Trial"]))
        if k and t:
            memo = r.get("memo")
            out[(int(k.group()), int(t.group()))] = {
                "name": str(r["piece name"]).strip(),
                "memo": None if pd.isna(memo) else str(memo).strip()}
    return out


def run(dataset: str | None = None, force: bool = False):
    if dataset and dataset != DATASET:
        return
    print(f"Stage 01 s02 -- filename pair for {DATASET}/{SESSION}")

    motion = {f.stem: f for f in MOTION_DIR.iterdir() if f.suffix.lower() == ".csv"} \
        if MOTION_DIR.exists() else {}
    audio = {f.stem: f for f in AUDIO_DIR.iterdir() if f.suffix.lower() == ".wav"} \
        if AUDIO_DIR.exists() else {}
    midi = {f.stem: f for f in MIDI_DIR.iterdir() if f.suffix.lower() == ".mid"} \
        if MIDI_DIR.exists() else {}

    all_stems = sorted(set(motion) | set(audio) | set(midi))
    print(f"  motion={len(motion)} audio={len(audio)} midi={len(midi)}  union={len(all_stems)}")

    # 211215 raw carries 'T3' but the actual Task is 9.
    # Canonical schema rewrites T3 -> T9.
    def to_canonical(raw_stem: str) -> str:
        return raw_stem.replace("_T3_", "_T9_")

    out = session_dir(DATASET, SESSION)
    info = piece_info()
    trials = []
    # Grouping lists for quick downstream filtering
    by_status: dict[str, list[str]] = {
        "complete": [],
        "motion_missing": [],
        "audio_missing": [],
        "midi_missing": [],
        "motion_only": [],
        "incomplete_other": [],
    }

    for stem in all_stems:
        m = motion.get(stem)
        a = audio.get(stem)
        d = midi.get(stem)

        modalities = []
        if m: modalities.append("motion")
        if a: modalities.append("audio")
        if d: modalities.append("midi")

        canonical_stem = to_canonical(stem)

        has_all = bool(m and a and d)
        if has_all:
            status = "complete"
        elif a and d and not m:
            status = "motion_missing"
        elif m and d and not a:
            status = "audio_missing"
        elif m and a and not d:
            status = "midi_missing"
        elif m and not a and not d:
            status = "motion_only"
        else:
            status = "incomplete_other"
        by_status[status].append(canonical_stem)

        if m:
            out_m = out / "motion" / f"{canonical_stem}.csv"
            if force or not out_m.exists():
                write_motion(m, out_m)
        if a:
            out_a = out / "audio" / f"{canonical_stem}.wav"
            if force or not out_a.exists():
                shutil.copy2(a, out_a)
        match = None
        if d:
            out_d = out / "midi" / f"{canonical_stem}.mid"
            match = audio_clock_midi(a, d, out_d) if a else None
            if match is None:
                shutil.copy2(d, out_d)

        pt = re.search(r"_P(\d+)_t(\d+)$", stem)
        meta = info.get((int(pt.group(1)), int(pt.group(2)))) if pt else None
        trials.append({
            "trial_stem": canonical_stem,
            "raw_stem": stem,
            "piece_name": meta["name"] if meta else None,
            "recording_memo": meta["memo"] if meta else None,
            "status": status,
            "modalities": modalities,
            "raw_sources": {
                "motion": m.name if m else None,
                "audio": a.name if a else None,
                "midi": d.name if d else None,
            },
            "audio_midi_match": match,
        })

    # Remove empty buckets
    by_status = {k: v for k, v in by_status.items() if v}

    summary = {k: len(v) for k, v in by_status.items()}
    print(f"  status summary: {summary}")
    print(f"  -> {out}")

    write_mapping(DATASET, SESSION, trials, method="filename_match", extra={
        "summary": summary,
        "trials_by_status": by_status,
        "capabilities": {
            "has_timecode": False,
            "has_video": False,
            "has_body_motion": True,
            "body_in_hand_csv": True,
            "needs_xcorr_matching": False,
            "needs_dtw_audio_midi": False,
        },
        "notes": [
            "Raw Vicon CSV uses 'piano:' prefix + dot notation (.1/.2); motion is "
            "written with {marker}_x/_y/_z columns. Hand and body markers are in "
            "the same Motion/Hand CSV.",
            "No OptiTrack SMPTE timecode -- cannot use timecode-based alignment.",
            "No camera recordings -- Stage 03 video sync should skip.",
            "MIDI is moved onto the audio clock (audio_midi_match per trial).",
            "Stage 02 body registration should skip (body already integrated).",
            "Motion is missing for 19 trials (mostly P5-P17 range). Audio/MIDI "
            "is complete and can still be processed by Stage 03 audio-MIDI DTW.",
        ],
    })


if __name__ == "__main__":
    run()
