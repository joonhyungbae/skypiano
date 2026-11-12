"""
Stage 04 step 13: reference performances of each piece, from confirmed cuts.

A piece cut whose boundaries are confirmed (by review, or by an exact check in
Stage 05; a near match, played with slips, is not used) holds exactly that piece
as one performer played it. Per piece name,
the most typical confirmed cut (score_follow.reference) among those of all
release datasets (the same exercises were played in several sessions) is kept
as the piece's reference, written per dataset; the piece-placing steps match it against
every trial (score_follow.place_in_order).

Rerun after reviews or a Stage 05 run have confirmed more pieces; the
placing steps then use the new references.

Output:
    Data/Pipeline/04_piece_split/references/{dataset}/r{k}.mid   (notes from 0 s)
    Data/Pipeline/04_piece_split/references/{dataset}/_index.json
        {name: {"file", "piece", "confirmed_by", "n_confirmed"}}

Usage (from the repository root):
  python pipeline/04_piece_split/steps/s13_piece_references.py [dataset ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _common.score_follow import onset_frames, reference  # noqa: E402

PIPE = Path("Data/Pipeline")
SPLIT = PIPE / "04_piece_split"
QA = PIPE / "05_alignment_qa"
OUT = SPLIT / "references"
DATASETS = ["220819", "221112", "240523", "240828"]


def piece_name(name: str) -> str:
    """Log names as written, without control characters (a stray backspace in some log cells)."""
    return "".join(c for c in name if c.isprintable()).strip()


def notes_of(midi: Path) -> list[tuple[float, float, int]]:
    return [(n.start, n.end, n.pitch) for i in pretty_midi.PrettyMIDI(str(midi)).instruments for n in i.notes]


def build(ds: str) -> dict:
    if not (QA / ds / "_status.json").exists():
        return {}
    # the same exercises were played in several sessions: confirmed cuts of every release
    # dataset count, those of this dataset first
    groups: dict[str, list] = {}
    for d in [ds] + [x for x in DATASETS if x != ds]:
        status = QA / d / "_status.json"
        if not status.exists():
            continue
        for p in json.loads(status.read_text(encoding="utf-8"))["pieces"]:
            midi = SPLIT / d / f"{p['piece']}.mid"
            if p.get("boundary_confirmed_by") not in (None, "not_a_piece", "near_match") and p.get("name") and midi.exists():
                notes = notes_of(midi)
                if notes:
                    groups.setdefault(piece_name(p["name"]), []).append((p["piece"], p["boundary_confirmed_by"], notes))
    out_dir = OUT / ds
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("r*.mid"):
        old.unlink()
    index = {}
    for k, (name, items) in enumerate(sorted(groups.items()), 1):
        feats = [onset_frames(n)[0] for _, _, n in items]
        ref = reference(feats)
        piece, by, notes = items[next(i for i, f in enumerate(feats) if f is ref)]
        t0 = min(n[0] for n in notes)
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(0)
        inst.notes = [pretty_midi.Note(velocity=64, pitch=p, start=a - t0, end=b - t0) for a, b, p in notes]
        pm.instruments.append(inst)
        pm.write(str(out_dir / f"r{k}.mid"))
        index[name] = {"file": f"r{k}.mid", "piece": piece, "confirmed_by": by, "n_confirmed": len(items)}
    (out_dir / "_index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    return index


def load(ds: str) -> dict[str, list[tuple[float, float, int]]]:
    """name -> reference notes, for the placing steps."""
    f = OUT / ds / "_index.json"
    if not f.exists():
        return {}
    return {name: notes_of(OUT / ds / e["file"]) for name, e in json.loads(f.read_text(encoding="utf-8")).items()}


def main():
    for ds in sys.argv[1:] or DATASETS:
        index = build(ds)
        print(f"{ds}: {len(index)} references")


if __name__ == "__main__":
    main()
