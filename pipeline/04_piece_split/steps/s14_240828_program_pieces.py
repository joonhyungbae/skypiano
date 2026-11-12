"""
Stage 04 step 14: 240828 technique trials (tasks 0 and 1) cut by their scores.

Task 0: every subject played the score Data/Score/technique1.pdf ("240809
session mandatory techniques") once per take (PROGRAM_T0): C major scale and
arpeggio and D-flat major scale and arpeggio from the lowest note "to high C
(D-flat)" at 120 BPM, the 3-octave C major scale and octave scale (twice),
the scale and arpeggio at 200 BPM "to high C ... to low C", the loudness
variation (the C major chord 4 times at each of ppp .. fff) and the
articulation variation (the chord held shorter and shorter). Its references
are written from the score (t0_score); the left hand plays an octave below
the right, octave scales in octaves in both hands.

In task 1 every 240828 subject played the technique score of the 221112
sessions (Data/Score/technique2_explaination.pdf), twice: the Fink
exercises in score order (PROGRAM_EXERCISES), then SET 2 -- one continuous
drill of scale, arpeggio, octave repeat (one bar), octave scale and broken
octave arpeggio (SET2) -- "twice through, at 120 BPM and at 160 BPM or
faster", then "the octave-repeat bar five times at the fastest tempo"
(OCTAVE_REPEAT_X5). Takes run in take order. The gap-based split of s02
merges exercises played without a pause and its names come from
fingerprints; this step re-cuts the task-0 and task-1 takes by the scores instead.

Per subject and task, the takes are put on one clock (take order) and the
program -- task 1: twice over; task 0: once per take, at most T0_ROUNDS --
is placed in order by its notes
(_common/score_follow.place_in_order; no piece across a take boundary).
References of the Fink exercises are the confirmed reference performances
of the 221112 pieces (s13_piece_references), the 240828 ones where they
exist. The SET 2 drills are written from the score (set2_score): C major,
eighth notes, the left hand an octave (octave drills: two octaves) below
the right; the octave-repeat bar is 8 C-octave chords, 40 for the fivefold
repeat. A SET 2 drill is named by the tempo it was
played at, not by its place: "(moderate)" when its median inter-onset
interval is at least SLOW_IOI_S (an eighth at 120 BPM is 0.25 s, at
160 BPM 0.19 s), "(fast)" otherwise -- the score's words. Playing between
placed pieces (a warm-up, a piece with no reference, a repeat) with at least
MIN_LEFT_NOTES notes and MIN_LEFT_S seconds is cut as an unnamed piece, split
at gaps of LEFT_GAP_S. Notes inside the session notes' discard intervals
(`exclusion_intervals` of the Stage 01 mapping) are left out of both: a
discarded false start is neither placed nor cut.

Windows: the piece's core -- its placed notes, or the window the reviewer set
by hand (_common/window_overrides), which takes the place of the core it
overlaps most -- ending when the last note stops sounding (the sustain pedal's
release, score_follow.sounding_end), padded by PAD_BEFORE_S / PAD_AFTER_S but
never into a neighbouring piece or any other note. Names carry resolution_source
"program_order" (a suggestion, confirmed by review). A manual label of an
old piece moves to the new piece it overlaps most.

Audio, MIDI, hand and body are cut as in s02. Videos are cut afterwards by
s05 with the selection this step writes. Old piece files of these takes are removed
first; _pieces.json, _pieces_resolved.json and _pieces_manual.json are backed
up to _backups/{time}_s14/.

Afterwards: s06 --trials (trial sync nudges on the motion), s05 videos,
Stage 05. Old pieces of the re-cut takes are removed first (files and
videos).

Outputs:
    Data/Pipeline/04_piece_split/240828/{trial}_p{k}.wav / .mid / _hand.csv / _body.csv
    Data/Pipeline/04_piece_split/240828/_pieces.json, _pieces_resolved.json (task-0/1 trials)
    Data/Pipeline/04_piece_split/240828/_program_pieces.json   (placement per piece)
    Data/Metadata/240828/program_video_selection.json          (for s05)
    Data/Metadata/240828/technique_program.json                (the program, for the review page)

Usage (from the repository root):
  python pipeline/04_piece_split/steps/s14_240828_program_pieces.py [--dry-run] [--trial STEM ...]

--trial re-cuts only those takes (e.g. after a window set by hand); the other
takes' pieces, placement and video selection are kept as they are.
"""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
import time
from pathlib import Path

import pretty_midi
from pydub import AudioSegment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from _common.score_follow import onset_frames, place_in_order, silent_bounds, sounding_end, sustain_spans  # noqa: E402
from _common import window_overrides  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses look their module up here
    spec.loader.exec_module(mod)
    return mod


s02 = _load("s02_piece_split")
s13 = _load("s13_piece_references")

DS = "240828"
OUT = s02.OUT_ROOT / DS
MIDI = s02.STAGE5_ROOT / DS
META = Path("Data/Metadata") / DS
PROGRAM_EXERCISES = ["4", "6", "7", "8", "10 - Forte", "10 - Piano", "13", "14", "15", "16", "17", "18", "19",
                     "27", "28", "29", "30", "34", "36", "40", "41 - Forte", "41 - Piano", "43", "44", "45", "46",
                     "47", "48", "50", "51", "Beethoven"]
SET2 = ["Scale", "Arpeggio", "Octave Repeat", "Octave Scale", "Broken Octave Arpeggio"]
OCTAVE_REPEAT_X5 = "Octave Repeat (fastest, x5)"
PROGRAM_DRILLS = SET2 * 2 + [OCTAVE_REPEAT_X5]          # placement slots; SET 2 names get their tempo after
PROGRAM = (PROGRAM_EXERCISES + PROGRAM_DRILLS) * 2
C_MAJOR_UP = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84]       # C4 .. C6
ARPEGGIO_UP = [60, 64, 67, 72, 76, 79, 84, 88, 91, 96]                        # C4 .. C7
OCTAVE_SCALE_UP = [36, 38, 40, 41, 43, 45, 47, 48]                            # lowest note of each chord
BROKEN_OCTAVE_ARPEGGIO = [36, 43, 40, 48, 43, 52, 48, 55, 52, 60, 55, 60,      # up, broken ...
                          52, 55, 48, 52, 43, 48, 40, 43, 36]                 # ... and down
OCTAVE_CHORD = (36, 48, 60, 72)                          # C2 C3 | C4 C5
SLOW_IOI_S = 0.21
X5_MIN_CHORDS = 20
TEMPO_TAGS = ("moderate", "fast")                     # the score: "moderate (120 BPM)" and "fast (160 BPM+)"
C_MAJOR, DB_MAJOR = {0, 2, 4, 5, 7, 9, 11}, {1, 3, 5, 6, 8, 10, 0}
C_TRIAD, DB_TRIAD = {0, 4, 7}, {1, 5, 8}
C_CHORD = (48, 52, 55, 60, 64, 67)                       # C3 E3 G3 | C4 E4 G4
PROGRAM_T0 = ["C Major Scale", "C Major Arpeggio", "Db Major Scale", "Db Major Arpeggio",
              "C Major Scale (3 octaves)", "C Major Octave Scale (3 octaves)",
              "C Major Scale (3 octaves)", "C Major Octave Scale (3 octaves)",
              "C Major Scale (fast)", "C Major Arpeggio (fast)",
              "Loudness Variation", "Articulation Variation"]
T0_ROUNDS = 2
CLOCK_GAP_S = 30.0
PAD_BEFORE_S = 0.3
PAD_AFTER_S = 1.0
NEIGHBOUR_GAP_S = 0.05
MIN_LEFT_NOTES = 10
MIN_LEFT_S = 2.0
LEFT_GAP_S = 4.0


def take_key(stem: str) -> tuple[int, int]:
    m = re.search(r"_P(\d+)_t(\d+)$", stem)
    return int(m.group(1)), int(m.group(2))


def notes_of(midi: Path) -> list[tuple[float, float, int]]:
    return [(n.start, n.end, n.pitch) for i in pretty_midi.PrettyMIDI(str(midi)).instruments for n in i.notes]


def up_down(up: list[int]) -> list[int]:
    return up + up[-2::-1]


def chords(seq: list[tuple[int, ...]]) -> list[tuple[float, float, int]]:
    """Eighth notes at 120 BPM (the tempo does not matter to the matching)."""
    return [(i * 0.25, i * 0.25 + 0.2, p) for i, ch in enumerate(seq) for p in ch]


def octave_repeat(n: int) -> list[tuple[float, float, int]]:
    return chords([OCTAVE_CHORD] * n)


def keys(lo: int, hi: int, pcs: set[int]) -> list[int]:
    return [p for p in range(lo, hi + 1) if p % 12 in pcs]


def t0_score() -> dict[str, list[tuple[float, float, int]]]:
    """technique1.pdf as notes, item by item (left hand an octave below the right)."""
    pairs = lambda ps: [(p, p + 12) for p in ps]                       # noqa: E731
    octaves = lambda ps: [(p, p + 12, p + 24, p + 36) for p in ps]     # noqa: E731
    return {
        "C Major Scale": chords(pairs(keys(24, 96, C_MAJOR))),           # C1 .. "to high C"
        "C Major Arpeggio": chords(pairs(keys(24, 96, C_TRIAD))),
        "Db Major Scale": chords(pairs(keys(25, 85, DB_MAJOR))),
        "Db Major Arpeggio": chords(pairs(keys(25, 85, DB_TRIAD))),
        "C Major Scale (3 octaves)": chords(pairs(up_down(keys(48, 84, C_MAJOR)))),
        "C Major Octave Scale (3 octaves)": chords(octaves(up_down(keys(36, 72, C_MAJOR)))),
        "C Major Scale (fast)": chords(pairs(up_down(keys(24, 96, C_MAJOR)))),
        "C Major Arpeggio (fast)": chords(pairs(up_down(keys(24, 96, C_TRIAD)))),
        "Loudness Variation": chords([C_CHORD] * 32),
        "Articulation Variation": chords([C_CHORD] * 12),
    }


def set2_score() -> dict[str, list[tuple[float, float, int]]]:
    """SET 2 of the technique score as notes, drill by drill."""
    octaves = lambda b: (b, b + 12, b + 24, b + 36)      # noqa: E731  both hands in octaves, two octaves apart
    return {
        "Scale": chords([(p - 12, p) for p in up_down(C_MAJOR_UP)]),
        "Arpeggio": chords([(p - 12, p) for p in up_down(ARPEGGIO_UP)]),
        "Octave Repeat": octave_repeat(8),
        "Octave Scale": chords([octaves(b) for b in up_down(OCTAVE_SCALE_UP)]),
        "Broken Octave Arpeggio": chords([octaves(b) for b in BROKEN_OCTAVE_ARPEGGIO]),
    }


def references() -> dict:
    lib = {n: onset_frames(v)[0] for n, v in s13.load("221112").items()}
    refs = {n: lib[n] for n in PROGRAM_EXERCISES if n in lib}
    refs.update({d: onset_frames(ns)[0] for d, ns in set2_score().items()})
    refs[OCTAVE_REPEAT_X5] = onset_frames(octave_repeat(40))[0]
    refs.update({n: onset_frames(v)[0] for n, v in s13.load(DS).items() if n in PROGRAM})
    return refs


def tempo_name(slot: str, notes: list, core: tuple[float, float]) -> str:
    """A SET 2 slot named by the tempo it was played at."""
    if slot not in SET2:
        return slot
    on = sorted({round(n[0], 2) for n in notes if core[0] <= n[0] <= core[1]})
    gaps = [b - a for a, b in zip(on, on[1:]) if b - a > 0.04]
    ioi = sorted(gaps)[len(gaps) // 2] if gaps else 0.0
    tempo = TEMPO_TAGS[0] if ioi >= SLOW_IOI_S else TEMPO_TAGS[1]
    x5 = ", x5" if slot == "Octave Repeat" and len(on) >= X5_MIN_CHORDS else ""     # the bar played five times
    return f"{slot} ({tempo}{x5})"


def leftovers(notes: list, cores: list[tuple[float, float]]) -> list[tuple[float, float, int]]:
    """Stretches of playing outside the placed cores: (first onset, last note-off, n notes)."""
    free = sorted(n for n in notes if not any(a <= n[0] <= b for a, b in cores))
    out, cur = [], []
    for n in free:
        inside_core = cur and any(cur[-1][0] < a < n[0] for a, _ in cores)
        if cur and (n[0] - max(x[1] for x in cur) > LEFT_GAP_S or inside_core):
            out.append(cur)
            cur = []
        cur.append(n)
    if cur:
        out.append(cur)
    spans = []
    for g in out:
        nxt = min((a for a, _ in cores if a > g[-1][0]), default=float("inf"))
        b = min(max(x[1] for x in g), nxt - NEIGHBOUR_GAP_S)    # a held key stops at the next piece
        if len(g) >= MIN_LEFT_NOTES and b - g[0][0] >= MIN_LEFT_S:
            spans.append((g[0][0], b, len(g)))
    return spans


def windows(cores: list[tuple[float, float]], trial_len: float, onsets: list[float]) -> list[tuple[float, float]]:
    """Core plus padding; the padding never reaches a neighbour or any other note."""
    out = []
    for i, (s, e) in enumerate(cores):
        lo = cores[i - 1][1] + NEIGHBOUR_GAP_S if i > 0 else 0.0
        hi = cores[i + 1][0] - NEIGHBOUR_GAP_S if i + 1 < len(cores) else trial_len
        quiet_lo, quiet_hi = silent_bounds(onsets, s, e)
        lo, hi = max(lo, quiet_lo), min(hi, quiet_hi)
        out.append((round(max(min(s, max(lo, s - PAD_BEFORE_S)), 0.0), 3),
                    round(min(max(e, min(hi, e + PAD_AFTER_S)), trial_len), 3)))
    return out


def plan(manifest: list[dict], task: int) -> dict[str, list[dict]]:
    """trial -> pieces [{core, name, cost}] in time order, for the takes of one task."""
    refs = references() if task == 1 else {n: onset_frames(v)[0] for n, v in t0_score().items()}
    info, align = s02.load_trial_info(DS), s02.load_stage5_alignments(DS)

    def discards(raw: str) -> list[tuple[float, float]]:
        t0 = align.get(raw, {}).get("timecode_start_sec")
        t0 = info.get(raw, {}).get("session_start_sec", 0) if t0 is None else t0
        return [(a - t0, b - t0) for a, b in info.get(raw, {}).get("exclusion_intervals", [])]

    subjects: dict[str, list[dict]] = {}
    for t in manifest:
        if t.get("task") == task and (MIDI / f"{t['raw_stem']}.mid").exists():
            subjects.setdefault(t["trial_stem"].split("_T")[0], []).append(t)
    out = {}
    for subj, takes in sorted(subjects.items()):
        takes.sort(key=lambda t: take_key(t["trial_stem"]))
        notes, spans, off = [], {}, 0.0
        for t in takes:
            drop = discards(t["raw_stem"])
            ns = [n for n in notes_of(MIDI / f"{t['raw_stem']}.mid") if not any(a <= n[0] < b for a, b in drop)]
            dur = max((b for _, b, _ in ns), default=0.0)
            spans[t["trial_stem"]] = (off, dur, ns)
            notes += [(a + off, b + off, p) for a, b, p in ns]
            off += dur + CLOCK_GAP_S
        breaks = [o + d + CLOCK_GAP_S / 2 for o, d, _ in list(spans.values())[:-1]]
        program = PROGRAM if task == 1 else PROGRAM_T0 * min(len(takes), T0_ROUNDS)
        placed = place_in_order(notes, [refs.get(n) for n in program], [None] * len(program), breaks)
        n_placed = 0
        for stem, (o, dur, own) in spans.items():
            pieces = [{"core": (round(p[0] - o, 3), round(p[1] - o, 3)), "cost": p[2],
                       "name": tempo_name(program[k], own, (p[0] - o, p[1] - o))}
                      for k, p in enumerate(placed) if p and o <= p[0] <= o + dur]
            n_placed += len(pieces)
            pieces += [{"core": (round(a, 3), round(b, 3)), "name": None, "cost": None}
                       for a, b, _ in leftovers(own, [p["core"] for p in pieces])]
            out[stem] = sorted(pieces, key=lambda p: p["core"][0])
        print(f"  {subj} task {task}: {n_placed} of {len(program)} program pieces placed in {len(takes)} takes")
    return out


def overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def main():
    dry = "--dry-run" in sys.argv
    manifest = json.loads((OUT / "_pieces.json").read_text(encoding="utf-8"))
    resolved = json.loads((OUT / "_pieces_resolved.json").read_text(encoding="utf-8"))
    manual = json.loads((OUT / "_pieces_manual.json").read_text(encoding="utf-8"))
    new = {**plan(manifest, 0), **plan(manifest, 1)}
    only = sys.argv[sys.argv.index("--trial") + 1:] if "--trial" in sys.argv else None
    if only:
        only = [a for a in only if not a.startswith("--")]
        missing = [a for a in only if a not in new]
        if missing:
            sys.exit(f"not a task-0/1 take: {missing}")
        new = {k: v for k, v in new.items() if k in only}
    if dry:
        for stem, ps in new.items():
            print(stem, [(p["name"], *p["core"], p["cost"]) for p in ps])
        return

    bak = OUT / "_backups" / f"{time.strftime('%Y-%m-%d_%H%M%S')}_s14"
    bak.mkdir(parents=True, exist_ok=True)
    for f in ("_pieces.json", "_pieces_resolved.json", "_pieces_manual.json"):
        shutil.copy2(OUT / f, bak / f)

    alignments = s02.load_stage5_alignments(DS)
    set_windows = window_overrides.load(DS)          # windows the reviewer set by hand
    by_stem = {t["trial_stem"]: t for t in manifest}
    placement, selection = {}, []
    for stem, plist in new.items():
        t = by_stem[stem]
        old = [((p["start_sec"], p["end_sec"]), p["piece_idx"]) for p in t["pieces"]]
        for f in list(OUT.glob(f"{stem}_p*")) + list((OUT / "videos").glob(f"{stem}_p*")):
            if re.match(rf"{re.escape(stem)}_p\d+[._]", f.name):
                f.unlink()
        wav, midi = OUT / f"{stem}.wav", MIDI / f"{t['raw_stem']}.mid"
        audio = AudioSegment.from_wav(str(wav))
        hand = s02.STAGE2_ROOT / DS / f"{stem}.csv"
        body = s02.STAGE3_ROOT / DS / f"{t['raw_stem']}.csv"
        clock = alignments.get(t["raw_stem"], {}).get("timecode_start_sec")
        ons = sorted(n[0] for n in notes_of(midi))
        pieces = []
        cores, by_reviewer = window_overrides.apply([tuple(p["core"]) for p in plist], set_windows.get(stem, []))
        pedal = sustain_spans(midi)
        cores = [(a, round(sounding_end(b, pedal, ons, len(audio) / 1000.0), 3)) for a, b in cores]
        wins = windows(cores, len(audio) / 1000.0, ons)
        for k, (p, (a, b), core) in enumerate(zip(plist, wins, cores), 1):
            pre = f"{stem}_p{k}"
            audio[int(a * 1000):int(b * 1000)].export(str(OUT / f"{pre}.wav"), format="wav")
            s02.slice_midi(midi, a, b).write(str(OUT / f"{pre}.mid"))
            if hand.exists():
                s02.slice_motion_csv(hand, a, b, s02.MOTION_FPS, OUT / f"{pre}_hand.csv", clock_start=clock)
            if body.exists():
                s02.slice_motion_csv(body, a, b, s02.MOTION_FPS, OUT / f"{pre}_body.csv")
            inside = [o for o in ons if a <= o < b]
            ident = {"resolved_piece_id": p["name"], "resolution_source": "program_order", "match_cost": p["cost"]} \
                if p["name"] else {"resolved_piece_id": None, "resolution_source": "unplaced"}
            pieces.append({"piece_idx": k, "start_sec": a, "end_sec": b, "n_notes": len(inside),
                           "first_onset": inside[0] if inside else None,
                           "last_offset": core[1], "identified": ident,
                           **({"window_source": "reviewer"} if k - 1 in by_reviewer else {})})
            placement[pre] = {"core_sec": list(core), "name": p["name"], "match_cost": p["cost"]}
            selection.append({"trial_stem": stem, "piece_idx": k})
        # manual labels follow the old piece's window to the new piece it overlaps most
        labels = manual.pop(stem, {})
        moved = {}
        for idx, lab in labels.items():
            w = next((w for w, i in old if str(i) == idx), None)
            best = max(pieces, key=lambda p: overlap(w, (p["start_sec"], p["end_sec"])), default=None) if w else None
            if best and overlap(w, (best["start_sec"], best["end_sec"])) > 0:
                moved[str(best["piece_idx"])] = lab
        if moved:
            manual[stem] = moved
        t["pieces"], t["n_pieces"] = pieces, len(pieces)
        for r in resolved:
            if r["trial_stem"] == stem:
                r["pieces"] = pieces
        print(f"  {stem}: {len(pieces)} pieces ({sum(1 for p in plist if p['name'])} named)")

    (OUT / "_pieces.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "_pieces_resolved.json").write_text(json.dumps(resolved, indent=2, ensure_ascii=False), encoding="utf-8")
    # labels the review page saved while this ran are kept: only the re-cut takes' labels are replaced
    on_disk = json.loads((OUT / "_pieces_manual.json").read_text(encoding="utf-8"))
    manual = {**{k: v for k, v in on_disk.items() if k not in new}, **{k: v for k, v in manual.items() if k in new}}
    (OUT / "_pieces_manual.json").write_text(json.dumps(manual, indent=2, ensure_ascii=False), encoding="utf-8")
    if only:
        # keep the other takes' placement and video selection
        prev = json.loads((OUT / "_program_pieces.json").read_text(encoding="utf-8"))
        placement = {**{k: v for k, v in prev.items() if k.rsplit("_p", 1)[0] not in new}, **placement}
        prev_sel = json.loads((META / "program_video_selection.json").read_text(encoding="utf-8"))["samples"]
        selection = [x for x in prev_sel if x["trial_stem"] not in new] + selection
    (OUT / "_program_pieces.json").write_text(json.dumps(placement, indent=2, ensure_ascii=False), encoding="utf-8")
    META.mkdir(parents=True, exist_ok=True)
    (META / "technique_program.json").write_text(json.dumps(
        {"programs": {"1": PROGRAM_EXERCISES + [f"{d} ({t})" for t in TEMPO_TAGS for d in SET2] + [OCTAVE_REPEAT_X5],
                      "0": PROGRAM_T0},
         "rounds": {"1": 2, "0": T0_ROUNDS}}, indent=2), encoding="utf-8")
    (META / "program_video_selection.json").write_text(json.dumps({"samples": selection}, indent=2), encoding="utf-8")
    print(f"{len(placement)} pieces in {len(new)} task-0/1 takes -> {OUT} (backup {bak.name})")


if __name__ == "__main__":
    main()
