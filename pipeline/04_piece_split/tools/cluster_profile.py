"""
Extract per-cluster musical profile for Set1 pool matching.

For each cluster, compute:
  - Key signature (Krumhansl-Schmuckler algorithm on pitch-class distribution)
  - Tempo estimate (librosa.beat.tempo)
  - Duration statistics
  - Note density
  - Pitch range (lowest, highest MIDI note; mean)
  - Mode (major/minor)

Output: set1_cluster_profiles.json — signature per cluster for pool matching.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import pretty_midi


CLUSTERS_PATH = Path("Data/Metadata/240828/set1_ensemble_v2_clusters.json")
PIECES_DIR = Path("Data/Pipeline/04_piece_split/240828")
OUT_PATH = Path("Data/Metadata/240828/set1_cluster_profiles.json")

# Krumhansl-Schmuckler key profiles (12-dim for each of 24 keys)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def estimate_key(pitch_class_hist: np.ndarray) -> tuple[str, str, float]:
    """Return (key_name, mode, correlation_strength)."""
    best = ("C", "major", -1.0)
    for i in range(12):
        rotated = np.roll(pitch_class_hist, -i)
        for mode_name, profile in [("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)]:
            c = np.corrcoef(rotated, profile)[0, 1]
            if c > best[2]:
                best = (KEYS[i], mode_name, c)
    return best


def midi_profile(midi_path: Path) -> dict | None:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes: list[pretty_midi.Note] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        notes.extend(inst.notes)
    if not notes:
        return None
    duration = pm.get_end_time() or 1.0
    # pitch class histogram
    pc = np.zeros(12, dtype=np.float32)
    for n in notes:
        pc[n.pitch % 12] += (n.end - n.start)   # duration-weighted
    pc /= pc.sum() if pc.sum() > 0 else 1.0
    key, mode, corr = estimate_key(pc)
    pitches = [n.pitch for n in notes]
    return {
        "key": key,
        "mode": mode,
        "key_corr": round(float(corr), 3),
        "pitch_class_hist": [round(float(x), 3) for x in pc],
        "pitch_low": int(min(pitches)),
        "pitch_high": int(max(pitches)),
        "pitch_mean": float(np.mean(pitches)),
        "pitch_std": float(np.std(pitches)),
        "note_density": round(len(notes) / duration, 2),
    }


def audio_tempo(wav_path: Path) -> float:
    try:
        y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        t = float(np.atleast_1d(tempo)[0])
        return round(t, 1)
    except Exception:
        return 0.0


def main():
    data = json.loads(CLUSTERS_PATH.read_text(encoding="utf-8"))
    clusters = data["clusters"]

    print("Profiling clusters...")
    # Load the pieces manifest once and build piece_key -> {mid, wav} lookups.
    pieces_manifest = json.loads(
        Path("Data/Pipeline/04_piece_split/240828/_pieces.json").read_text(encoding="utf-8")
    )
    # Build: piece_key (sub04_0424_p1) → mid path
    key_to_mid: dict[str, Path] = {}
    key_to_wav: dict[str, Path] = {}
    for t in pieces_manifest:
        sub = t["raw_stem"][-4:-2]
        trial = t["raw_stem"][-4:]
        for p in t["pieces"]:
            pk = f"sub{sub}_{trial}_p{p['piece_idx']}"
            key_to_mid[pk] = PIECES_DIR / f"{t['trial_stem']}_p{p['piece_idx']}.mid"
            key_to_wav[pk] = PIECES_DIR / f"{t['trial_stem']}_p{p['piece_idx']}.wav"

    out = []
    for c in clusters:
        member_data = []
        for m in c["members"]:
            pk = m["piece_key"]
            mid = key_to_mid.get(pk)
            wav = key_to_wav.get(pk)
            prof = midi_profile(mid) if mid and mid.is_file() else None
            tempo = audio_tempo(wav) if wav and wav.is_file() else 0.0
            if prof:
                prof["tempo_bpm"] = tempo
                member_data.append({"piece_key": pk, **prof})

        if not member_data:
            continue

        # Aggregate cluster-level stats
        keys = Counter((m["key"], m["mode"]) for m in member_data)
        top_key, top_count = keys.most_common(1)[0]

        def med(field):
            vals = sorted(m[field] for m in member_data)
            return vals[len(vals) // 2]

        def mean(field):
            return round(float(np.mean([m[field] for m in member_data])), 2)

        cluster_prof = {
            "cluster_id": c["stable_cluster_id"],
            "size": c["size"],
            "n_subjects": c["n_subjects"],
            "tasks": c["tasks"],
            "mode": top_key[1],
            "key": top_key[0],
            "key_consensus": f"{top_count}/{len(member_data)}",
            "tempo_bpm_median": med("tempo_bpm"),
            "note_density_median": med("note_density"),
            "pitch_mean": mean("pitch_mean"),
            "pitch_std": mean("pitch_std"),
            "pitch_low_min": min(m["pitch_low"] for m in member_data),
            "pitch_high_max": max(m["pitch_high"] for m in member_data),
            "duration_median": c["signature"]["duration_median"],
            "duration_range": [c["signature"]["duration_min"], c["signature"]["duration_max"]],
            "members": member_data,
        }
        out.append(cluster_prof)

    # Print summary
    print(f"\n{'cluster_id':<13}  {'size':>4}  {'task':>4}  {'key':>8}  {'tempo':>6}  "
          f"{'dur_med':>8}  {'density':>8}")
    for c in out:
        tasks = "/".join(c["tasks"])
        key_str = f"{c['key']} {c['mode'][:3]}"
        print(f"{c['cluster_id']:<13}  {c['size']:>4}  {tasks:>4}  "
              f"{key_str:>8}  {c['tempo_bpm_median']:>6.1f}  "
              f"{c['duration_median']:>7.1f}s  {c['note_density_median']:>7.2f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"n_clusters": len(out), "clusters": out}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
