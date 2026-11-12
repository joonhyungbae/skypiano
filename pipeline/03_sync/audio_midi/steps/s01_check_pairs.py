# Standalone diagnostic; not part of the run_pipeline.py loop. Invoked manually.
"""
Stage 03 audio_midi Step 01: Full Audio-MIDI pair audit

Scans each dataset's Raw directory, matches Audio and MIDI files, and reports
the pair status (both / audio_only / midi_only).

Imported by the standalone DTW-based alignment scripts (`s03_align.py`,
`s03_realign_211215.py`) for their `DATASET_DIRS` config.
"""
import os
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# Per-dataset Audio/MIDI directory configuration
# ─────────────────────────────────────────────────────────────
RAW_ROOT = Path("Data/Raw")

DATASET_DIRS = {
    "211215": {
        "nodes": [
            (RAW_ROOT / "211215/C0/N0/Audio", RAW_ROOT / "211215/C0/N0/MIDI"),
        ],
    },
    "220819": {
        "nodes": [
            (RAW_ROOT / "220819/C0/N0/Audio", RAW_ROOT / "220819/C0/N0/MIDI"),
            (RAW_ROOT / "220819/C0/N1/Audio", RAW_ROOT / "220819/C0/N1/MIDI"),
            (RAW_ROOT / "220819/C0/N2/Audio", RAW_ROOT / "220819/C0/N2/MIDI"),
            (RAW_ROOT / "220819/C1/N0/Audio", RAW_ROOT / "220819/C1/N0/MIDI"),
            (RAW_ROOT / "220819/C1/N1/Audio", RAW_ROOT / "220819/C1/N1/MIDI"),
        ],
    },
    "221112": {
        "nodes": [
            (RAW_ROOT / f"221112/C1/N{n}/Audio", RAW_ROOT / f"221112/C1/N{n}/MIDI")
            for n in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        ],
    },
    "240523": {
        "nodes": [
            (RAW_ROOT / "240523/C1/N0/Audio", RAW_ROOT / "240523/C1/N0/MIDI"),
        ],
    },
    "240809": {
        "nodes": [
            (RAW_ROOT / "240809/C0/N0/Audio", RAW_ROOT / "240809/C0/N0/MIDI"),
        ],
    },
    "240828": {
        "nodes": [
            (RAW_ROOT / f"240828/C0/N{n}/Audio", RAW_ROOT / f"240828/C0/N{n}/MIDI")
            for n in [4, 5, 6, 7, 8]
        ],
    },
}

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".aiff"}
MIDI_EXTS = {".mid", ".midi"}
SKIP_FILES = {"desktop.ini", "thumbs.db"}


def list_files(directory: Path, extensions: set) -> dict[str, Path]:
    result = {}
    if not directory.exists():
        return result
    for f in directory.iterdir():
        if f.name.lower() in SKIP_FILES:
            continue
        if f.suffix.lower() in extensions:
            result[f.stem] = f
    return result


def check_pairs_for_node(audio_dir: Path, midi_dir: Path) -> list[dict]:
    audio_files = list_files(audio_dir, AUDIO_EXTS)
    midi_files = list_files(midi_dir, MIDI_EXTS)

    midi_base_map = {}
    for stem, path in midi_files.items():
        base = stem
        for suffix in ["_r1", "_r2", "_r3"]:
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break
        if base not in midi_base_map:
            midi_base_map[base] = []
        midi_base_map[base].append(path)

    all_stems = set(audio_files.keys()) | set(midi_base_map.keys())
    results = []

    for stem in sorted(all_stems):
        has_audio = stem in audio_files
        has_midi = stem in midi_base_map

        if has_audio and has_midi:
            status = "both"
        elif has_audio:
            status = "audio_only"
        else:
            status = "midi_only"

        results.append({
            "stem": stem,
            "status": status,
            "audio_path": str(audio_files[stem]) if has_audio else None,
            "midi_paths": [str(p) for p in midi_base_map[stem]] if has_midi else None,
        })

    return results


def run():
    summary = defaultdict(lambda: {"both": 0, "audio_only": 0, "midi_only": 0})
    all_results = {}

    for dataset_id, config in DATASET_DIRS.items():
        dataset_results = []
        for audio_dir, midi_dir in config["nodes"]:
            pairs = check_pairs_for_node(audio_dir, midi_dir)
            for p in pairs:
                p["audio_dir"] = str(audio_dir)
                p["midi_dir"] = str(midi_dir)
            dataset_results.extend(pairs)

        all_results[dataset_id] = dataset_results
        for r in dataset_results:
            summary[dataset_id][r["status"]] += 1

    print("=" * 80)
    print("Stage 03 audio_midi Step 01: Audio-MIDI pair check")
    print("=" * 80)

    for dataset_id in sorted(all_results.keys()):
        results = all_results[dataset_id]
        s = summary[dataset_id]
        print(f"\n  {dataset_id}  |  both: {s['both']}  |  audio_only: {s['audio_only']}  |  midi_only: {s['midi_only']}")
        problems = [r for r in results if r["status"] != "both"]
        if not problems:
            print("  -> All paired.")
        else:
            for r in problems:
                tag = "AUDIO_ONLY" if r["status"] == "audio_only" else "MIDI_ONLY"
                print(f"  [{tag}] {r['stem']}")

    total_both = sum(s["both"] for s in summary.values())
    total_audio = sum(s["audio_only"] for s in summary.values())
    total_midi = sum(s["midi_only"] for s in summary.values())
    print(f"\nTOTAL  both={total_both}  audio_only={total_audio}  midi_only={total_midi}")

    return all_results, summary


if __name__ == "__main__":
    run()
