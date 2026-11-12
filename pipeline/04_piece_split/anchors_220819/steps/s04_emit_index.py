"""
Stage 04_piece_split/anchors_220819 / Step 04: Emit extended Anchors/index.json.

Reads:
  - Data/Pipeline/04_piece_split/anchors_220819/locate_manifest.json  (from s01 - audio windows)
  - Data/Pipeline/04_piece_split/anchors_220819/alignment.json        (from s02 - motion offsets)
  - Data/Anchors/index.json                        (existing audio/MIDI catalog)
  - Stage 04 output files at Data/Pipeline/04_piece_split/220819/  (from s03)

Writes:
  - Data/Anchors/index.json                        (overwritten)

Each take entry is augmented with:
  resolved_stem, out_stem, t_start_sec, t_end_sec,
  locate: {method, confidence}
  alignment: {
    hand: {align_shift_sec, confidence},
    body: {align_shift_sec, confidence}
  }
  paths: {anchor_audio, anchor_midi, audio, midi, hand, body}
  coverage: {audio, midi, hand, body}

Existing Technique{1,2}/<piece>/<take>.{wav,mid} files in Data/Anchors/
are unchanged; index.json merely grows new fields.
"""
from __future__ import annotations

import functools
import json
import shutil
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

DATASET = "220819"
ANCHOR_ROOT = Path("Data/Anchors")
PIPELINE_ROOT = Path("Data/Pipeline")
LOCATE_PATH = PIPELINE_ROOT / "04_piece_split/anchors_220819" / "locate_manifest.json"
ALIGNMENT_PATH = PIPELINE_ROOT / "04_piece_split/anchors_220819" / "alignment.json"
PIECE_SPLIT_ROOT = PIPELINE_ROOT / "04_piece_split" / DATASET
INDEX_PATH = ANCHOR_ROOT / "index.json"
BACKUP_PATH = ANCHOR_ROOT / "index.json.bak"


def _rel_under(p: Path, root: Path) -> str:
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def run(dataset: str | None = None, force: bool = False) -> None:
    if dataset is not None and dataset != DATASET:
        print(f"[s04_emit_index] skip - only {DATASET} supported")
        return
    if not LOCATE_PATH.is_file():
        print(f"[s04_emit_index] locate_manifest missing: {LOCATE_PATH}")
        return
    if not ALIGNMENT_PATH.is_file():
        print(f"[s04_emit_index] alignment missing: {ALIGNMENT_PATH}")
        return

    locate = json.loads(LOCATE_PATH.read_text(encoding="utf-8"))
    alignment = json.loads(ALIGNMENT_PATH.read_text(encoding="utf-8"))
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))

    # Backup original index on first run
    if not BACKUP_PATH.is_file():
        shutil.copyfile(INDEX_PATH, BACKUP_PATH)
        print(f"[s04_emit_index] backup saved: {BACKUP_PATH}")

    locate_map = {(e["task"], e["piece_id"], e["take_id"]): e
                  for e in locate["entries"]}
    align_map = {(e["session"], e["stage1_stem"]): e for e in alignment["entries"]}

    for task, pieces in index["tasks"].items():
        if task == "Unknown":
            continue
        for piece_id, pinfo in pieces.items():
            for take in pinfo["takes"]:
                key = (task, piece_id, take["take_id"])
                loc = locate_map.get(key)
                if loc is None or not loc.get("ok"):
                    take["locate"] = {"ok": False, "reason": (loc or {}).get("reason", "not_processed")}
                    continue

                out_stem = loc["out_stem"]
                piece_audio = PIECE_SPLIT_ROOT / f"{out_stem}.wav"
                piece_midi = PIECE_SPLIT_ROOT / f"{out_stem}.mid"
                piece_hand = PIECE_SPLIT_ROOT / f"{out_stem}_hand.csv"
                piece_body = PIECE_SPLIT_ROOT / f"{out_stem}_body.csv"
                anchor_audio = ANCHOR_ROOT / task / piece_id / f"{take['take_id']}.wav"
                anchor_midi = ANCHOR_ROOT / task / piece_id / f"{take['take_id']}.mid"

                align_entry = align_map.get((loc["session"], loc["stage1_stem"]), {})
                hand_align = align_entry.get("hand", {})
                body_align = align_entry.get("body", {})

                take["resolved_stem"] = loc["stage1_stem"]
                take["out_stem"] = out_stem
                take["t_start_sec"] = loc["t_start_sec"]
                take["t_end_sec"] = loc["t_end_sec"]
                take["locate"] = {
                    "method": loc["method"],
                    "confidence": loc["confidence"],
                }
                take["alignment"] = {
                    "hand": {
                        "align_shift_sec": hand_align.get("align_shift_sec"),
                        "confidence": hand_align.get("confidence", "missing"),
                    },
                    "body": {
                        "align_shift_sec": body_align.get("align_shift_sec"),
                        "confidence": body_align.get("confidence", "missing"),
                    },
                }
                take["paths"] = {
                    "anchor_audio": _rel_under(anchor_audio, ANCHOR_ROOT),
                    "anchor_midi":  _rel_under(anchor_midi, ANCHOR_ROOT),
                    "audio": _rel_under(piece_audio, PIPELINE_ROOT) if piece_audio.is_file() else None,
                    "midi":  _rel_under(piece_midi, PIPELINE_ROOT) if piece_midi.is_file() else None,
                    "hand":  _rel_under(piece_hand, PIPELINE_ROOT) if piece_hand.is_file() else None,
                    "body":  _rel_under(piece_body, PIPELINE_ROOT) if piece_body.is_file() else None,
                }
                take["coverage"] = {
                    "audio": piece_audio.is_file(),
                    "midi":  piece_midi.is_file(),
                    "hand":  piece_hand.is_file(),
                    "body":  piece_body.is_file(),
                }

    INDEX_PATH.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")

    total = sum(
        1 for t, pieces in index["tasks"].items() if t != "Unknown"
        for p, pinfo in pieces.items() for _ in pinfo["takes"]
    )
    labeled = sum(
        1 for t, pieces in index["tasks"].items() if t != "Unknown"
        for p, pinfo in pieces.items() for tk in pinfo["takes"] if "out_stem" in tk
    )
    full_cov = sum(
        1 for t, pieces in index["tasks"].items() if t != "Unknown"
        for p, pinfo in pieces.items() for tk in pinfo["takes"]
        if tk.get("coverage", {}).get("audio") and tk["coverage"].get("midi")
        and tk["coverage"].get("hand") and tk["coverage"].get("body")
    )
    print(f"[s04_emit_index] updated {INDEX_PATH}")
    print(f"  total takes:        {total}")
    print(f"  labeled with paths: {labeled}")
    print(f"  full 4-modality coverage: {full_cov}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run(dataset=args.dataset, force=args.force)
