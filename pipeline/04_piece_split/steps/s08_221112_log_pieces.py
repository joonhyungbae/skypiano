"""
Stage 04 step 08: 221112 pieces from the session log.

Stage 01 places every 221112 motion take in its session recording and lists
the log's pieces that lie in the take, on the trial clock (trial time = motion
time; `log_pieces` in the session `_mapping.json`). This step cuts them:

  window = [log start - PAD_BEFORE_S, log end + PAD_AFTER_S], the padding
  never running into a neighbouring piece's logged span or any other note. Pieces whose logged
  spans overlap are cut as logged (the review page lists the overlap). A log
  entry whose end is not after its start (a copied end time) is not cut; it
  keeps its number and is listed under `skipped`.

The log times are written by hand: a logged end often lies after the
music has stopped, so the window takes in the next piece's first notes.
Every subject played the same pieces in the logged order, so a piece is
placed by its notes (_common/score_follow.place_in_order): each trial's
logged pieces are matched in order against the whole trial, the attempt
nearest the logged start taken where a piece was played twice. A piece's
reference is its confirmed reference performance (s13_piece_references) --
for a SET 2 drill the score's notes (s14_240828_program_pieces.set2_score);
a piece with none yet uses the most typical of its logged spans, refined
once from the pieces so placed. Where a piece is placed, its core replaces
the logged span before padding; otherwise the logged span is kept, clipped
where it runs into a neighbour placed by its notes.
A window the reviewer set by hand (_common/window_overrides) is the piece's
core in place of the computed one (window_source "reviewer"), padded the same
way. A core ends when its last note stops sounding (the sustain pedal's
release, score_follow.sounding_end).
`window_source` (reviewer / aligned / log) and `align_cost` (mean match cost, 0 =
note for note) are recorded per piece. Piece names are the log's, without
control characters, the SET 2 drills in the words of the score
(_common/technique_names: slow -> moderate, "Broken Arpeggio" -> "Broken
Octave Arpeggio", the numbered extra octave repeats -> the fivefold repeat
at the fastest tempo; an octave repeat played five bars long gets ", x5").

Audio and MIDI come from the Stage 01 trial files, hand and body motion from
Stage 02 (frame 0 = trial time 0, 120 fps). Outputs, in the layout of the
other datasets:

    Data/Pipeline/04_piece_split/221112/{trial}_p{k}.wav / .mid / _hand.csv / _body.csv
    Data/Pipeline/04_piece_split/221112/_pieces.json

Usage (from the repository root):
  python pipeline/04_piece_split/steps/s08_221112_log_pieces.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pretty_midi
from pydub import AudioSegment

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _common.slicing import slice_audio, slice_midi  # noqa: E402
from _common.score_follow import onset_frames, place_in_order, reference, silent_bounds, sounding_end, sustain_spans  # noqa: E402
from _common.technique_names import chord_count, score_name, with_repeats  # noqa: E402
from _common import window_overrides  # noqa: E402

def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


s13 = _load("s13_piece_references")
s14 = _load("s14_240828_program_pieces")

DS = "221112"
PIPE = Path("Data/Pipeline")
NORM = PIPE / "01_ingest" / "Normalized" / DS
HAND = PIPE / "02_spatial" / "hand" / "Aligned" / DS
BODY = PIPE / "02_spatial" / "body" / "Aligned" / DS
OUT = PIPE / "04_piece_split" / DS
FPS = 120.0
PAD_BEFORE_S = 0.3
PAD_AFTER_S = 1.0
NEIGHBOUR_GAP_S = 0.05
REFINE_COST = 0.2
MIN_CLIPPED_S = 1.0


def with_bars(name: str, notes: list, a: float, b: float) -> str:
    """An octave repeat named by how many bars were played (technique_names.with_repeats)."""
    return with_repeats(name, chord_count([n[0] for n in notes if a <= n[0] < b]))


def place_pieces(trials: list[tuple[str, list[dict], list]]) -> dict[tuple[str, int], tuple[float, float, float]]:
    """(trial, piece k) -> (core start, core end, mean match cost) for pieces placed by their notes."""
    confirmed = {name: onset_frames(notes)[0] for name, notes in s13.load(DS).items()}
    # the SET 2 drills by the score's notes (s14), at either tempo; the octave repeat's 8 chords
    # grow over further bars by the edge extension of place_in_order
    score = {d: onset_frames(v)[0] for d, v in s14.set2_score().items()}
    for _, pieces, _ in trials:
        for p in pieces:
            if p["id"].split(" (")[0] in score:
                confirmed[p["id"]] = score[p["id"].split(" (")[0]]

    def own(spans):
        perf: dict[str, list] = {}
        for stem, pieces, notes in trials:
            for k, p in enumerate(pieces, 1):
                if p["id"] not in confirmed and p["end"] > p["start"]:
                    a, b = spans.get((stem, k), (p["start"], p["end"]))
                    perf.setdefault(p["id"], []).append(onset_frames([n for n in notes if a <= n[0] <= b])[0])
        return {name: reference(f) for name, f in perf.items()}

    def run(refs):
        out = {}
        for stem, pieces, notes in trials:
            placed = place_in_order(notes, [refs.get(p["id"]) if p["end"] > p["start"] else None for p in pieces],
                                    [p["start"] for p in pieces])
            out.update({(stem, k): pl for k, pl in enumerate(placed, 1) if pl})
        return out

    out = run({**own({}), **confirmed})
    good = {key: (a, b) for key, (a, b, c) in out.items() if c <= REFINE_COST}
    out = run({**own(good), **confirmed})
    print(f"  references: {len(confirmed)} confirmed, {len({p['id'] for _, ps, _ in trials for p in ps} - set(confirmed))} from logged spans")
    return out


def clip_logged(pieces: list[dict], placed: list[bool]) -> list[dict]:
    """A piece kept at its logged span stops where a neighbour placed by its notes begins
    (and starts where one ends): the placed span is the music, the log is written by hand."""
    out = [dict(p) for p in pieces]
    for i, p in enumerate(out):
        if placed[i] or p["end"] <= p["start"]:
            continue
        for j, q in enumerate(out):
            if j == i or not placed[j]:
                continue
            if q["start"] <= p["start"] < q["end"]:
                p["start"] = q["end"] + NEIGHBOUR_GAP_S
            if q["start"] < p["end"] <= q["end"] or (p["start"] < q["start"] < p["end"]):
                if q["start"] > p["start"]:
                    p["end"] = min(p["end"], q["start"] - NEIGHBOUR_GAP_S)
        if p["end"] - p["start"] < MIN_CLIPPED_S:      # nothing left: keep the log, the overlap is listed
            out[i] = dict(pieces[i])
    return out


def cut_motion(src: Path, start: float, end: float, dst: Path) -> bool:
    df = pd.read_csv(src)
    if "TimeCode" in df.columns:
        df = df.drop(columns=["TimeCode"])
    part = df.iloc[max(0, int(start * FPS)):int(end * FPS)].reset_index(drop=True)
    if part.empty:
        return False
    part.to_csv(dst, index=False)
    return True


def windows(pieces: list[dict], trial_len: float, onsets: list[float]) -> list[tuple[float, float]]:
    """Core plus padding; the padding never reaches a neighbour or any other note."""
    out = []
    for i, p in enumerate(pieces):
        lo = pieces[i - 1]["end"] + NEIGHBOUR_GAP_S if i > 0 else 0.0
        hi = pieces[i + 1]["start"] - NEIGHBOUR_GAP_S if i + 1 < len(pieces) else trial_len
        quiet_lo, quiet_hi = silent_bounds(onsets, p["start"], p["end"])
        lo, hi = max(lo, quiet_lo), min(hi, quiet_hi)
        a = max(min(p["start"], max(lo, p["start"] - PAD_BEFORE_S)), 0.0)
        b = min(max(p["end"], min(hi, p["end"] + PAD_AFTER_S)), trial_len)
        out.append((round(a, 3), round(b, 3)))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    trials = []
    for mp in sorted(NORM.glob("C1_N*/_mapping.json")):
        sess = mp.parent
        for t in json.loads(mp.read_text(encoding="utf-8"))["trials"]:
            stem, pieces = t["trial_stem"], t.get("log_pieces") or []
            wav, mid = sess / "audio" / f"{stem}.wav", sess / "midi" / f"{stem}.mid"
            if pieces and wav.exists():
                notes = [(n.start, n.end, n.pitch) for i in pretty_midi.PrettyMIDI(str(mid)).instruments for n in i.notes]
                pieces = [dict(p, id=score_name(s13.piece_name(p["id"]))) for p in pieces]
                trials.append((stem, pieces, notes, t, sess))
    cores = place_pieces([(stem, pieces, notes) for stem, pieces, notes, _, _ in trials])
    set_windows = window_overrides.load(DS)          # windows the reviewer set by hand
    print(f"  {len(cores)} of {sum(len(t[1]) for t in trials)} pieces placed by their notes")
    for stem, logged, notes, t, sess in trials:
        wav, mid = sess / "audio" / f"{stem}.wav", sess / "midi" / f"{stem}.mid"
        pieces = [dict(p, start=cores[(stem, k)][0], end=cores[(stem, k)][1]) if (stem, k) in cores else p
                  for k, p in enumerate(logged, 1)]
        pieces = clip_logged(pieces, [(stem, k) in cores for k in range(1, len(pieces) + 1)])
        audio = AudioSegment.from_wav(str(wav))
        onsets = sorted(n[0] for n in notes)
        rec = {"trial_stem": stem, "raw_stem": t.get("raw_stem"), "task": t.get("task"), "pieces": [], "skipped": []}
        set_cores, by_reviewer = window_overrides.apply([(p["start"], p["end"]) for p in pieces], set_windows.get(stem, []))
        pedal = sustain_spans(mid)
        pieces = [dict(p, start=a, end=round(sounding_end(b, pedal, onsets, len(audio) / 1000.0), 3)) for p, (a, b) in zip(pieces, set_cores)]
        wins = windows(pieces, len(audio) / 1000.0, onsets)
        for k, (p, (a, b)) in enumerate(zip(pieces, wins), 1):
            lg = logged[k - 1]
            if lg["end"] <= lg["start"]:
                rec["skipped"].append({"piece_idx": k, "log_window_sec": [p["start"], p["end"]], "id": p["id"],
                                       "reason": "log end not after start"})
                continue
            ps = f"{stem}_p{k}"
            slice_audio(audio, a, b, OUT / f"{ps}.wav")
            slice_midi(mid, a, b, OUT / f"{ps}.mid")
            for kind, src in (("hand", HAND / f"{stem}.csv"), ("body", BODY / f"{stem}.csv")):
                if src.exists():
                    cut_motion(src, a, b, OUT / f"{ps}_{kind}.csv")
            rec["pieces"].append({
                "piece_idx": k, "start_sec": a, "end_sec": b,
                "log_window_sec": [lg["start"], lg["end"]], "core_sec": [p["start"], p["end"]],
                "window_source": "reviewer" if k - 1 in by_reviewer else "aligned" if (stem, k) in cores else "log",
                "align_cost": cores[(stem, k)][2] if (stem, k) in cores else None,
                "n_notes": sum(1 for o in onsets if a <= o < b),
                "log_note": p.get("note"),
                "identified": {"resolved_piece_id": with_bars(p["id"], notes, a, b),
                               "resolution_source": "session_log"}})
        index.append(rec)
        print(f"  {stem}: {len(rec['pieces'])} pieces")
    (OUT / "_pieces.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{sum(len(r['pieces']) for r in index)} pieces in {len(index)} trials -> {OUT}")


if __name__ == "__main__":
    main()
