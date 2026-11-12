"""
Build anchor library from Data/Processed/220819 piece slices.

Output:
  Data/Anchors/
    index.json                          piece → list of takes (audio+midi paired)
    Technique1/{piece_id}/{source_take}.{wav,mid}
    Technique2/{piece_id}/{source_take}.{wav,mid}

Anchors are used by the symbolic-fingerprint matcher to locate piece onsets
within longer untagged session recordings (e.g. 240828 sessions).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

PROCESSED_ROOT = Path("Data/Processed/220819")
ANCHOR_ROOT = Path("Data/Anchors")

# Map piece IDs to Task category. Keep in sync with task_taxonomy.json /
# arrangement_parser.py T1_PIECE_META and T2_PIECE_MAP.
TECHNIQUE1_PIECE_IDS = {
    "4", "6", "7", "8", "10 - Forte", "10 - Piano", "13", "14", "15",
    "16", "17", "18", "19",
    "27", "28", "29", "30", "34", "36",
    "40", "41 - Forte", "41 - Piano", "43", "44", "45",
    "46", "47", "48", "50", "51",
}

TECHNIQUE2_DRILL_IDS = {
    "Scale_Slow", "Arpeggio_Slow", "Octave Repeat_Slow", "Octave Scale_Slow",
    "Broken Octave Arpeggio_Slow",
    "Scale_Fast", "Arpeggio_Fast", "Repetition_Fast", "Octave Scale_Fast",
    "Broken Octave Arpeggio_Fast",
}
TECHNIQUE2_APPLIED_IDS = {
    "Octave Repeat 1", "Octave Repeat 2", "Octave Repeat 3",
    "Beethoven 1", "Beethoven 2",
}


def classify_piece(pid: str) -> tuple[str, str]:
    """Return (task, subtype) — subtype is 'piece' for T1, 'drill'/'applied' for T2."""
    if pid in TECHNIQUE1_PIECE_IDS:
        return "Technique1", "piece"
    if pid in TECHNIQUE2_DRILL_IDS:
        return "Technique2", "drill"
    if pid in TECHNIQUE2_APPLIED_IDS:
        return "Technique2", "applied"
    return "Unknown", "unknown"


def collect_takes() -> dict:
    """Walk Processed/220819 and collect per-piece audio+midi pairs."""
    pieces: dict[str, dict] = {}

    for cls_n in ["C0/N2", "C1/N1"]:
        base = PROCESSED_ROOT / cls_n
        if not base.is_dir():
            continue
        for mod, ext in [("Audio", ".wav"), ("MIDI", ".mid")]:
            mod_root = base / mod
            if not mod_root.is_dir():
                continue
            for trial_dir in sorted(mod_root.iterdir()):
                if not trial_dir.is_dir():
                    continue
                for f in sorted(trial_dir.iterdir()):
                    if f.suffix.lower() != ext.lower():
                        continue
                    pid = f.stem
                    src = cls_n.replace("/", "_")
                    take_key = f"{trial_dir.name}"
                    take = pieces.setdefault(pid, {}).setdefault(take_key, {
                        "source_session": src,
                        "source_trial": trial_dir.name,
                    })
                    if mod == "Audio":
                        take["audio_src"] = str(f)
                    else:
                        take["midi_src"] = str(f)
    return pieces


def _safe_name(s: str) -> str:
    """Return a filesystem-safe variant of an arbitrary piece id."""
    return s.replace("/", "_").replace("\\", "_")


def build_library(dry_run: bool = False) -> dict:
    """Copy all anchor takes into ANCHOR_ROOT/{Task}/{piece}/{take}.{wav,mid} layout.
    Returns the index dict written to index.json."""
    pieces = collect_takes()

    index = {
        "_comment": "Anchor library: piece slices from Processed/220819 used as templates for fingerprint matching.",
        "source": str(PROCESSED_ROOT),
        "tasks": {"Technique1": {}, "Technique2": {}, "Unknown": {}},
        "stats": {},
    }

    for pid in sorted(pieces):
        task, subtype = classify_piece(pid)
        safe = _safe_name(pid)
        out_dir = ANCHOR_ROOT / task / safe
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)

        take_records = []
        for take_key in sorted(pieces[pid]):
            take = pieces[pid][take_key]
            audio_src = take.get("audio_src")
            midi_src = take.get("midi_src")
            take_id = take["source_trial"]  # e.g. 220819_C0_N2_T1_t1
            audio_dst = out_dir / f"{take_id}.wav"
            midi_dst = out_dir / f"{take_id}.mid"

            if not dry_run:
                if audio_src:
                    shutil.copy2(audio_src, audio_dst)
                if midi_src:
                    shutil.copy2(midi_src, midi_dst)

            take_records.append({
                "take_id": take_id,
                "source_session": take["source_session"],
                "audio_present": audio_src is not None,
                "midi_present": midi_src is not None,
                "audio_kb": round(Path(audio_src).stat().st_size / 1024, 1) if audio_src else None,
                "midi_kb": round(Path(midi_src).stat().st_size / 1024, 1) if midi_src else None,
            })

        index["tasks"][task][pid] = {
            "subtype": subtype,
            "take_count": len(take_records),
            "takes": take_records,
        }

    # Stats
    for task in ("Technique1", "Technique2", "Unknown"):
        n_pieces = len(index["tasks"][task])
        n_takes = sum(p["take_count"] for p in index["tasks"][task].values())
        index["stats"][task] = {"pieces": n_pieces, "takes": n_takes}

    if not dry_run:
        ANCHOR_ROOT.mkdir(parents=True, exist_ok=True)
        with open(ANCHOR_ROOT / "index.json", "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    return index


if __name__ == "__main__":
    import sys
    dry = "--dry" in sys.argv
    idx = build_library(dry_run=dry)
    print("Anchor library {}built at {}".format("(dry-run) " if dry else "", ANCHOR_ROOT))
    print(json.dumps(idx["stats"], indent=2))
