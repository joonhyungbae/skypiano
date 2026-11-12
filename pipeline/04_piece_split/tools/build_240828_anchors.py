"""
Build a 240828-specific anchor library from reviewer-confirmed labels.

Reads `_pieces_manual.json` (Phase 1/2 cluster review output) and creates a
parallel anchor structure at Data/Anchors/2024/, mirroring the existing
2022 layout (Technique1/Technique2/Set1/Free → piece_id → takes).

Files are hard-linked (no extra disk usage) from Data/Pipeline/04_piece_split/240828/.
Re-runnable: idempotent. Removes stale links before re-creating.

Output:
  Data/Anchors/2024/{task_dir}/{piece_id}/{take_id}.{wav,mid}
  Data/Anchors/2024/index.json   (same schema as the 2022 index)
"""
from __future__ import annotations

import functools
import json
import os
import re
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

MANUAL_LABELS = Path("Data/Pipeline/04_piece_split/240828/_pieces_manual.json")
PIECES_MANIFEST = Path("Data/Pipeline/04_piece_split/240828/_pieces.json")
PIECE_ROOT = Path("Data/Pipeline/04_piece_split/240828")
ANCHOR_OUT = Path("Data/Anchors/2024")

# Stage 04 task tag → anchor task directory name.
TASK_TO_DIR = {0: "Technique1", 1: "Technique2", 2: "Set1", 9: "Free"}

TRIAL_TASK_RE = re.compile(r"_T(\d+)_")
TRIAL_SESSION_RE = re.compile(r"^(240828)_(C\d+_N\d+)_")


def safe_filename(name: str) -> str:
    """piece_id may contain spaces or punctuation — keep as-is for parity with
    the existing 2022 anchor structure (which uses 'Octave Repeat 2' etc.).
    Just guard against path separators."""
    return name.replace("/", "_").replace("\\", "_").strip()


def load_pieces_lookup():
    """Map (trial_stem, piece_idx) → {start_sec, end_sec, n_notes} from
    _pieces.json so we can populate t_start_sec / t_end_sec in the index."""
    if not PIECES_MANIFEST.is_file():
        return {}
    data = json.loads(PIECES_MANIFEST.read_text(encoding="utf-8"))
    out = {}
    for tr in data:
        ts = tr.get("trial_stem")
        if not ts:
            continue
        for p in tr.get("pieces", []):
            out[(ts, p.get("piece_idx"))] = p
    return out


def link_or_copy(src: Path, dst: Path):
    """Try hard-link first; fall back to copy on failure (e.g., across vols)."""
    if dst.exists():
        try:
            dst.unlink()
        except Exception:
            pass
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        import shutil
        shutil.copy2(src, dst)
        return "copy"


def main():
    if not MANUAL_LABELS.is_file():
        print(f"[ERR] {MANUAL_LABELS} not found")
        return
    manual = json.loads(MANUAL_LABELS.read_text(encoding="utf-8"))
    piece_meta = load_pieces_lookup()
    n_total_manual = sum(len(v) for v in manual.values())
    print(f"Reading {n_total_manual} manual labels across {len(manual)} trials")

    # Wipe ONLY trial-derived hardlinks/copies (file basenames matching the
    # 240828_C0_N#_T#_P#_t#_p# stem pattern). Preserve YouTube reference
    # anchors (yt_<videoid>.{wav,mid,json}) and any other external references
    # that were dropped in by tools/download_set1_anchors.py — those are not
    # produced by this script so they shouldn't be its to delete.
    if ANCHOR_OUT.is_dir():
        print(f"Cleaning trial-derived takes under {ANCHOR_OUT} (preserving yt_* anchors)...")
        n_removed = 0
        n_kept = 0
        for sub_path in ANCHOR_OUT.rglob("*"):
            if not sub_path.is_file():
                continue
            name = sub_path.name
            # Trial-derived anchor names start with the 240828 session stem;
            # YouTube anchors start with 'yt_'. Anything else (manual drops,
            # additional sidecars) is preserved.
            if re.match(r"^240828_C\d+_N\d+_T\d+_", name):
                try:
                    sub_path.unlink()
                    n_removed += 1
                except Exception:
                    pass
            else:
                n_kept += 1
        # Drop now-empty piece directories (so stale labels don't linger).
        for sub_path in sorted(ANCHOR_OUT.rglob("*"), reverse=True):
            if sub_path.is_dir() and not any(sub_path.iterdir()):
                try:
                    sub_path.rmdir()
                except Exception:
                    pass
        print(f"  removed {n_removed} trial-derived files; kept {n_kept} non-trial files")
    ANCHOR_OUT.mkdir(parents=True, exist_ok=True)

    # Index structure mirroring 2022/index.json
    index = {
        "_comment": "240828 reviewer-confirmed anchors. Built by "
                    "pipeline/04_piece_split/tools/build_240828_anchors.py from "
                    "_pieces_manual.json. Re-run when new labels are confirmed.",
        "source": "Data/Pipeline/04_piece_split/240828 (hard-linked)",
        "tasks": {},
    }

    n_takes = 0
    n_skipped_no_files = 0
    by_link_method = {"hardlink": 0, "copy": 0}

    for trial_stem, by_idx in manual.items():
        tm = TRIAL_TASK_RE.search(trial_stem)
        sm = TRIAL_SESSION_RE.match(trial_stem)
        if not tm or not sm:
            continue
        task = int(tm.group(1))
        session = sm.group(2)        # e.g. "C0_N4"
        task_dir_name = TASK_TO_DIR.get(task)
        if task_dir_name is None:
            continue

        for idx_str, info in by_idx.items():
            piece_id = (info or {}).get("piece_id")
            if not piece_id:
                continue
            try:
                idx = int(idx_str)
            except ValueError:
                continue

            # Source files
            src_wav = PIECE_ROOT / f"{trial_stem}_p{idx}.wav"
            src_mid = PIECE_ROOT / f"{trial_stem}_p{idx}.mid"
            if not src_wav.is_file() and not src_mid.is_file():
                n_skipped_no_files += 1
                continue

            # Take id: source piece file stem (uniquely identifies the take)
            take_id = f"{trial_stem}_p{idx}"

            piece_dir = ANCHOR_OUT / task_dir_name / safe_filename(piece_id)
            piece_dir.mkdir(parents=True, exist_ok=True)

            audio_kb = midi_kb = None
            if src_wav.is_file():
                method = link_or_copy(src_wav, piece_dir / f"{take_id}.wav")
                by_link_method[method] = by_link_method.get(method, 0) + 1
                audio_kb = round(src_wav.stat().st_size / 1024, 1)
            if src_mid.is_file():
                method = link_or_copy(src_mid, piece_dir / f"{take_id}.mid")
                by_link_method[method] = by_link_method.get(method, 0) + 1
                midi_kb = round(src_mid.stat().st_size / 1024, 1)

            pmeta = piece_meta.get((trial_stem, idx)) or {}
            t_start = pmeta.get("start_sec")
            t_end = pmeta.get("end_sec")

            # Append to index
            tasks = index["tasks"].setdefault(task_dir_name, {})
            piece_entry = tasks.setdefault(safe_filename(piece_id), {
                "subtype": "piece",
                "take_count": 0,
                "takes": [],
            })
            piece_entry["takes"].append({
                "take_id": take_id,
                "source_session": session,
                "audio_present": audio_kb is not None,
                "midi_present": midi_kb is not None,
                "audio_kb": audio_kb,
                "midi_kb": midi_kb,
                "resolved_stem": trial_stem,
                "out_stem": take_id,
                "t_start_sec": t_start,
                "t_end_sec": t_end,
                "reviewer": (info or {}).get("reviewer"),
                "timestamp": (info or {}).get("timestamp"),
            })
            piece_entry["take_count"] = len(piece_entry["takes"])
            n_takes += 1

    # Sort takes within each piece by source_session for stable ordering
    for task_name, task_pieces in index["tasks"].items():
        for piece_id, piece_entry in task_pieces.items():
            piece_entry["takes"].sort(key=lambda t: t.get("source_session", ""))

    # Merge in any yt_<videoid>.wav anchors that were dropped in by
    # tools/download_set1_anchors.py (Set1 / Free YouTube references). They
    # share the anchor directory layout but are out-of-band relative to
    # `_pieces_manual.json`, so they need an explicit pass.
    import wave as _wave
    for piece_dir in ANCHOR_OUT.rglob("*"):
        if not piece_dir.is_dir():
            continue
        rel = piece_dir.relative_to(ANCHOR_OUT)
        parts = rel.parts
        if len(parts) != 2:
            continue
        task_name, piece_id = parts
        for wav_path in sorted(piece_dir.glob("yt_*.wav")):
            take_id = wav_path.stem
            mid_path = piece_dir / f"{take_id}.mid"
            sidecar = piece_dir / f"{take_id}.json"
            sidecar_data = {}
            if sidecar.is_file():
                try:
                    sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
                except Exception:
                    pass
            try:
                with _wave.open(str(wav_path), "rb") as w:
                    duration_sec = w.getnframes() / float(w.getframerate())
            except Exception:
                duration_sec = None
            yt_take = {
                "take_id": take_id,
                "source": "youtube",
                "youtube_url": sidecar_data.get("url"),
                "youtube_title": sidecar_data.get("title"),
                "youtube_channel": sidecar_data.get("channel"),
                "audio_present": True,
                "midi_present": mid_path.is_file(),
                "audio_kb": round(wav_path.stat().st_size / 1024, 1),
                "midi_kb": round(mid_path.stat().st_size / 1024, 1) if mid_path.is_file() else 0.0,
                "duration_sec": round(duration_sec, 2) if duration_sec else None,
            }
            task_block = index["tasks"].setdefault(task_name, {})
            piece_entry = task_block.setdefault(piece_id, {
                "subtype": "piece", "take_count": 0, "takes": [],
            })
            existing = {t.get("take_id") for t in piece_entry["takes"]}
            if take_id not in existing:
                piece_entry["takes"].append(yt_take)
                piece_entry["take_count"] = len(piece_entry["takes"])

    index_path = ANCHOR_OUT / "index.json"
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Summary
    print(f"\nBuilt {n_takes} takes at {ANCHOR_OUT}")
    print(f"  link method: {by_link_method}")
    if n_skipped_no_files:
        print(f"  [WARN] skipped {n_skipped_no_files} entries — source files missing")
    print(f"\nBreakdown by task:")
    for task_name in ("Technique1", "Technique2", "Set1", "Free"):
        tp = index["tasks"].get(task_name, {})
        total = sum(p["take_count"] for p in tp.values())
        print(f"  {task_name:12s}: {len(tp):3d} pieces, {total:3d} takes")
    print(f"\nIndex: {index_path}")


if __name__ == "__main__":
    main()
