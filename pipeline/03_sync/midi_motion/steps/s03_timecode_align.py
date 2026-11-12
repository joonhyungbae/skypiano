"""
Stage 03 midi_motion Step 03: Timecode-based MIDI extraction (240523, 240809, 240828)

These datasets carry timecodes embedded in the OptiTrack CSVs, so the exact
segment can be extracted from the session MIDI directly, without cross-correlation.
"""
import functools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi

print = functools.partial(print, flush=True)

RAW_ROOT = Path("Data/Raw")
NORMALIZED_ROOT = Path("Data/Pipeline/01_ingest/Normalized")  # Stage 01 mapping files
HAND_ALIGNED_ROOT = Path("Data/Pipeline/02_spatial/hand/Aligned")
OUT_ROOT = Path("Data/Pipeline/03_sync/midi_motion/Aligned")

TIMECODE_FPS = 30  # OptiTrack SMPTE timecode fps
MOTION_FPS = 120.0


def build_raw_to_trial_map(dataset: str) -> dict[str, str]:
    """Build raw_stem → trial_stem map from Stage 01 _mapping.json files."""
    result = {}
    ds_dir = NORMALIZED_ROOT / dataset
    if not ds_dir.is_dir():
        return result
    for sess in ds_dir.iterdir():
        mpath = sess / "_mapping.json"
        if not mpath.is_file():
            continue
        with open(mpath, encoding="utf-8") as f:
            mapping = json.load(f)
        for trial in mapping.get("trials", []):
            rs, ts = trial.get("raw_stem"), trial.get("trial_stem")
            if rs and ts:
                result[rs] = ts
    return result


def align_by_xcorr(
    hand_csv_path: Path, midi_path: Path, search_start: float, search_end: float,
) -> tuple[float, float, float]:
    """Fallback alignment via MIDI-onset ↔ motion-velocity cross-correlation.

    Sweeps candidate MIDI offsets in [search_start, search_end] and returns
    (best_offset, best_offset + motion_duration, score).
    """
    mdf = pd.read_csv(hand_csv_path)
    motion_fps = 120.0
    motion_duration = len(mdf) / motion_fps

    # Wrist velocity as onset proxy
    vel_cols = [c for c in mdf.columns if "Wrist" in c]
    vel = np.zeros(len(mdf))
    for col in vel_cols:
        v = mdf[col].diff().fillna(0).values
        vel += v ** 2
    vel = np.sqrt(vel)

    bin_size = 0.1
    n_bins = int(motion_duration / bin_size) + 1
    motion_signal = np.zeros(n_bins)
    for i in range(len(vel)):
        b = min(int((i / motion_fps) / bin_size), n_bins - 1)
        motion_signal[b] = max(motion_signal[b], vel[i])
    std = motion_signal.std()
    if std < 1e-8:
        raise ValueError("flat motion signal — cannot xcorr")
    motion_signal = (motion_signal - motion_signal.mean()) / std

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    midi_onsets = np.array(sorted(n.start for inst in pm.instruments for n in inst.notes))

    # Coarse sweep (0.5s step)
    best_score, best_offset = -np.inf, search_start
    for offset in np.arange(search_start, search_end, 0.5):
        midi_signal = np.zeros(n_bins)
        for onset in midi_onsets:
            t = onset - offset
            if 0 <= t < motion_duration:
                b = min(int(t / bin_size), n_bins - 1)
                midi_signal[b] += 1
        if midi_signal.sum() == 0:
            continue
        midi_signal = (midi_signal - midi_signal.mean()) / (midi_signal.std() + 1e-8)
        score = np.dot(motion_signal, midi_signal) / len(motion_signal)
        if score > best_score:
            best_score, best_offset = score, offset

    # Fine sweep (0.05s step, ±2s around best)
    coarse = best_offset
    for offset in np.arange(coarse - 2, coarse + 2, 0.05):
        midi_signal = np.zeros(n_bins)
        for onset in midi_onsets:
            t = onset - offset
            if 0 <= t < motion_duration:
                b = min(int(t / bin_size), n_bins - 1)
                midi_signal[b] += 1
        if midi_signal.sum() == 0:
            continue
        midi_signal = (midi_signal - midi_signal.mean()) / (midi_signal.std() + 1e-8)
        score = np.dot(motion_signal, midi_signal) / len(motion_signal)
        if score > best_score:
            best_score, best_offset = score, offset

    return best_offset, best_offset + motion_duration, best_score

SESSIONS = [
    {"dataset": "240523", "base": RAW_ROOT / "240523/C1/N0",
     "motion_prefix": "240523sub02", "midi_stem": "240523sub02"},
    {"dataset": "240809", "base": RAW_ROOT / "240809/C0/N0",
     "motion_prefix": "240809PJH01", "midi_stem": "240809PJH1"},
    {"dataset": "240809", "base": RAW_ROOT / "240809/C0/N0",
     "motion_prefix": "240809PJH02", "midi_stem": "240809PJH2"},
    *[{"dataset": "240828", "base": RAW_ROOT / f"240828/C0/N{n}",
       "motion_prefix": f"240828sub{n:02d}", "midi_stem": f"240828sub{n:02d}"}
      for n in [4, 5, 6, 7, 8]],
]


def parse_timecode(tc: str) -> float:
    """HH:MM:SS:FF.xx → seconds"""
    hh, mm, ss, ff = tc.split(":")
    ff = ff.split(".")[0]
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ff) / TIMECODE_FPS


def _span_with_dropout(sec) -> tuple[float, float]:
    """(first, last) of a per-frame timecode series. A capture can lose its
    timecode before it stops (the column drops to 00:00:00:00): the run up to
    the first backward jump is kept and the frames after it are counted at the
    run's own frame rate."""
    sec = np.asarray(sec, float)
    back = np.flatnonzero(np.diff(sec) < 0)
    if not len(back) or back[0] == 0:
        return float(sec[0]), float(sec[-1])
    i = int(back[0])
    rate = i / (sec[i] - sec[0])
    return float(sec[0]), float(sec[i] + (len(sec) - 1 - i) / rate)


def get_trial_timecodes_from_raw(csv_path: Path) -> tuple[float, float]:
    """Extract the first/last valid timecodes from a raw OptiTrack CSV (legacy fallback)"""
    data = pd.read_csv(csv_path, encoding="latin1", header=2, low_memory=False)
    data = data.iloc[3:, 2:]
    cols = data.columns.tolist()
    cols = [c.replace("Markerset 001:", "").replace("Unnamed: 2", "TimeCode") for c in cols]
    data.columns = cols

    if "TimeCode" not in data.columns:
        raise KeyError("no TimeCode column")

    valid_mask = data.iloc[:, 1:].notna().any(axis=1)
    valid_tc = data.loc[valid_mask, "TimeCode"]
    return _span_with_dropout(valid_tc.apply(parse_timecode))


def get_trial_timecodes_from_hand_csv(hand_csv_path: Path) -> tuple[float, float]:
    """
    Extract the first/last valid TimeCode from a Stage 02 hand CSV in seconds.
    Only works when Stage 02 preserves the TimeCode column (240809/240828/240523).

    Stage 02 may trim some frames from raw, so MIDI must be extracted using the
    actual timecode of Stage 02 frame 0 — not the raw timecode — for accurate sync.
    """
    df = pd.read_csv(hand_csv_path, low_memory=False)
    if "TimeCode" not in df.columns:
        raise KeyError(f"no TimeCode column in {hand_csv_path.name}")
    tc = df["TimeCode"].dropna()
    if len(tc) == 0:
        raise ValueError(f"all TimeCode values are NaN in {hand_csv_path.name}")
    return _span_with_dropout(tc.astype(str).apply(parse_timecode))


def extract_trial_midi(midi_path: Path, start_sec: float, end_sec: float) -> pretty_midi.PrettyMIDI:
    """Extract the [start, end] segment from the session MIDI and shift it to a 0-based timeline"""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    new_pm = pretty_midi.PrettyMIDI()
    duration = end_sec - start_sec

    for inst in pm.instruments:
        new_inst = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        for note in inst.notes:
            if note.start < end_sec and note.end > start_sec:
                new_inst.notes.append(pretty_midi.Note(
                    velocity=note.velocity, pitch=note.pitch,
                    start=max(0.0, note.start - start_sec),
                    end=min(duration, note.end - start_sec)))
        for cc in inst.control_changes:
            if start_sec <= cc.time < end_sec:
                new_inst.control_changes.append(pretty_midi.ControlChange(
                    number=cc.number, value=cc.value, time=cc.time - start_sec))
        new_pm.instruments.append(new_inst)
    return new_pm


def run():
    print("Stage 03 midi_motion s03: Timecode-based MIDI extraction")
    print(f"Output: {OUT_ROOT}")

    success, fail = 0, 0
    qc = {"good": 0, "no_motion_data": 0}

    # Pre-build raw_stem → trial_stem maps per dataset
    stem_maps: dict[str, dict[str, str]] = {}
    for sess in SESSIONS:
        ds = sess["dataset"]
        if ds not in stem_maps:
            stem_maps[ds] = build_raw_to_trial_map(ds)

    for sess in SESSIONS:
        dataset = sess["dataset"]
        motion_dir = sess["base"] / "Motion" / "Hand"
        midi_path = sess["base"] / "MIDI" / f"{sess['midi_stem']}.mid"

        if not motion_dir.exists() or not midi_path.exists():
            print(f"\n[SKIP] {dataset}/{sess['midi_stem']}: missing data")
            continue

        csv_files = sorted([
            f for f in motion_dir.iterdir()
            if f.suffix.lower() == ".csv" and f.stem.startswith(sess["motion_prefix"])
        ])

        pm_session = pretty_midi.PrettyMIDI(str(midi_path))
        session_duration = pm_session.get_end_time()
        print(f"\nSession: {dataset}/{sess['midi_stem']} ({session_duration:.1f}s, {len(csv_files)} trials)")

        raw_to_trial = stem_maps.get(dataset, {})

        # Pre-scan: collect known timecodes for each trial to bound xcorr search
        known_tc: dict[str, tuple[float, float]] = {}  # stem → (start, end)
        for cp in csv_files:
            try:
                s, e = get_trial_timecodes_from_raw(cp)
                known_tc[cp.stem] = (s, e)
            except (KeyError, Exception):
                pass

        for csv_path in csv_files:
            stem = csv_path.stem  # raw_stem
            out_dir = OUT_ROOT / dataset
            out_json = out_dir / f"{stem}.json"

            if out_json.exists():
                with open(out_json) as f:
                    existing = json.load(f)
                q = existing.get("quality", "good")
                qc[q] = qc.get(q, 0) + 1
                print(f"  {stem} - skip")
                success += 1
                continue

            # Resolve Stage 02 hand CSV: try trial_stem first, then raw_stem
            trial_stem = raw_to_trial.get(stem, stem)
            hand_csv_path = HAND_ALIGNED_ROOT / dataset / f"{trial_stem}.csv"
            if not hand_csv_path.exists():
                # Fallback: raw_stem name
                hand_csv_path = HAND_ALIGNED_ROOT / dataset / f"{stem}.csv"
            if not hand_csv_path.exists():
                print(f"  {stem} - no Stage 02 hand CSV (tried {trial_stem}, {stem})")
                fail += 1
                continue

            try:
                # Prefer Stage 02 hand CSV timecodes (reflects Stage 02's actual frame 0..N-1).
                # Fall back to raw CSV timecodes.
                # Last resort: MIDI-motion cross-correlation bounded by neighbor TCs.
                try:
                    first_sec, last_sec = get_trial_timecodes_from_hand_csv(hand_csv_path)
                    tc_source = "stage2"
                except (KeyError, ValueError):
                    try:
                        first_sec, last_sec = get_trial_timecodes_from_raw(csv_path)
                        tc_source = "raw"
                    except KeyError:
                        # Bound search to range between previous and next known TCs
                        idx = [cp.stem for cp in csv_files].index(stem)
                        prev_end = 0.0
                        for j in range(idx - 1, -1, -1):
                            tc = known_tc.get(csv_files[j].stem)
                            if tc:
                                prev_end = tc[1]
                                break
                        next_start = session_duration
                        for j in range(idx + 1, len(csv_files)):
                            tc = known_tc.get(csv_files[j].stem)
                            if tc:
                                next_start = tc[0]
                                break
                        first_sec, last_sec, xcorr_score = align_by_xcorr(
                            hand_csv_path, midi_path,
                            search_start=max(0, prev_end - 10),
                            search_end=min(session_duration, next_start + 10),
                        )
                        tc_source = f"xcorr(score={xcorr_score:.4f})"
                trial_duration = last_sec - first_sec
                if trial_duration <= 0:
                    print(f"  {stem}  SKIP (invalid timecode range)")
                    fail += 1
                    continue

                pm_trial = extract_trial_midi(midi_path, first_sec, last_sec)
                n_notes = sum(len(inst.notes) for inst in pm_trial.instruments)
                n_onsets = len([n.start for inst in pm_trial.instruments for n in inst.notes])

                # Check Stage 02 hand frame count for motion_onsets info
                hand_header_df = pd.read_csv(hand_csv_path, nrows=0)
                hand_frames = sum(1 for _ in open(hand_csv_path)) - 1  # minus header

                out_dir.mkdir(parents=True, exist_ok=True)
                pm_trial.write(str(out_dir / f"{stem}.mid"))

                quality = "good" if n_notes > 0 else "no_motion_data"

                result = {
                    "stem": stem, "dataset": dataset,
                    "session_midi": sess["midi_stem"],
                    "timecode_start_sec": round(first_sec, 4),
                    "timecode_end_sec": round(last_sec, 4),
                    "trial_duration_sec": round(trial_duration, 2),
                    "confidence": 1.0,
                    "quality": quality,
                    "alignment_method": "timecode",
                    "timecode_source": tc_source,
                    "n_midi_notes": n_notes,
                    "n_midi_onsets": n_onsets,
                    "n_motion_frames": hand_frames,
                    "hand_csv": str(hand_csv_path),
                }
                with open(out_json, "w") as f:
                    json.dump(result, f, indent=2)

                qc[quality] = qc.get(quality, 0) + 1
                print(f"  {stem}  tc[{tc_source}]=[{first_sec:.1f}-{last_sec:.1f}]s  notes={n_notes}  ({quality})")
                success += 1

            except Exception as e:
                print(f"  {stem}  FAIL: {e}")
                fail += 1

    print(f"\n{'=' * 70}")
    print(f"Done: success={success} fail={fail}")
    print(f"Quality: good={qc.get('good', 0)} no_motion={qc.get('no_motion_data', 0)}")


if __name__ == "__main__":
    run()
