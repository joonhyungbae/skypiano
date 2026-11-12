"""
Stage 04_piece_split/anchors_220819 / Step 01: Locate each anchor take inside its Stage 01 source.

For each (task, piece_id, take) in Data/Anchors/index.json:
  1. Resolve take_id -> Stage 01 piece stem via _mapping.json[raw_sources.audio].
  2. MIDI onset-pulse xcorr (primary, fast, cached per stage1 stem).
  3. If MIDI is empty or weak (ratio < MIDI_STRONG_RATIO), run audio xcorr
     fallback and use for cross-validation / tiebreak.
  4. Assign _p{N} slot number = rank of t_start among anchors sharing same
     Stage 01 stem.
  5. Emit Data/Pipeline/04_piece_split/anchors_220819/locate_manifest.json.

The manifest is consumed by s02_slice.py; no data files are produced here.
"""
from __future__ import annotations

import functools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_P = Path(__file__).resolve()
while _P.name != "pipeline":
    _P = _P.parent
sys.path.insert(0, str(_P))

from _common.normalized_io import NORMALIZED_ROOT  # noqa: E402
from _common.technique_names import DRILLS, score_name  # noqa: E402
from _common import window_overrides  # noqa: E402
from _common.score_follow import onset_frames, place_in_order, silent_bounds, sounding_end, sustain_spans  # noqa: E402

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))
from src.locate import (  # noqa: E402
    AGREE_TOL_SEC, AUDIO_OK_NCC, AUDIO_OK_RATIO, GRID_SR, MIDI_STRONG_RATIO,
    _ncc_locate, audio_to_onset_env, midi_to_onset_pulses,
)

print = functools.partial(print, flush=True)

DATASET = "220819"
ANCHOR_ROOT = Path("Data/Anchors")
OUT_ROOT = Path("Data/Pipeline/04_piece_split/anchors_220819")
MANIFEST_PATH = OUT_ROOT / "locate_manifest.json"
METADATA_ROOT = Path("Data/Metadata")

# Final piece windows: the session log (arrangement CSV) defines each piece's
# start/end; the MIDI/audio locate result only corrects the log's timing when
# it agrees with it. A locate result far from the log is a mismatch (repeated
# material elsewhere in the take), not a better estimate.
# The log times are written by hand (and the locate fails on some pieces), so
# the logged pieces of a take are placed by their notes, in log order
# (_common/score_follow.place_in_order, as for 221112): the SET 2 drills by the
# score (s14_240828_program_pieces.set2_score), the exercises by the confirmed
# 221112 / 220819 reference performances (s13_piece_references). A piece placed
# within PLACE_MAX_FROM_LOG_S of its logged start takes the placed span as its
# core (window_source "notes"); otherwise the logged window stays.
PLACE_MAX_FROM_LOG_S = 10.0
JUNCTION_SEARCH_S = 1.0     # the scale's first D octave is looked for this near its placed start
JUNCTION_MAX_GAP_S = 0.5    # repeat and scale run on: no longer pause before the scale's C
# Where the octave repeat runs straight into the octave scale (no silence),
# the boundary is set by the notes: the scale starts on the C octave before
# its first D octave (_split_octave_junctions).
LOG_AGREE_SEC = 1.0      # use the located shift only when within this of the log
PAD_BEFORE_SEC = 0.3     # lead-in kept before the logged start
PAD_AFTER_SEC = 1.0      # tail kept after the logged end (reverb, pedal release); no padding reaches another note
NEIGHBOUR_GAP_SEC = 0.05 # a window never runs into the next piece's start
OVERLAP_SEARCH_SEC = 0.5 # overlapping logged windows: search margin for the silence to cut at


def resolve_stage1_stem(take_id: str) -> tuple[Path, str] | None:
    """take_id -> (session_dir, trial_stem) via _mapping.json."""
    parts = take_id.split("_", 3)
    if len(parts) < 4 or parts[0] != DATASET:
        return None
    session_dir = NORMALIZED_ROOT / DATASET / f"{parts[1]}_{parts[2]}"
    mapping_path = session_dir / "_mapping.json"
    if not mapping_path.is_file():
        return None
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    raw_audio = f"{take_id}.wav"
    for trial in data.get("trials", []):
        if trial.get("raw_sources", {}).get("audio") == raw_audio:
            return session_dir, trial["trial_stem"]
    return None


def _midi_locate_with_cache(anchor_mid: Path, source_mid: Path, cache: dict) -> dict:
    key = str(source_mid)
    if key not in cache:
        cache[key] = midi_to_onset_pulses(source_mid)
    long = cache[key]
    short = midi_to_onset_pulses(anchor_mid)
    if short.sum() == 0:
        return {"ok": False, "reason": "anchor_midi_empty"}
    if len(short) >= len(long):
        return {"ok": False, "reason": "anchor_longer_than_source"}
    idx, peak, ratio = _ncc_locate(short, long)
    return {
        "ok": True,
        "t_start_sec": round(idx / GRID_SR, 3),
        "t_end_sec": round((idx + len(short)) / GRID_SR, 3),
        "anchor_dur_sec": round(len(short) / GRID_SR, 3),
        "peak_ncc": round(peak, 4),
        "secondary_ratio": None if ratio == float("inf") else round(ratio, 2),
    }


def _audio_locate_with_cache(anchor_wav: Path, source_wav: Path, cache: dict) -> dict:
    key = str(source_wav)
    if key not in cache:
        cache[key] = audio_to_onset_env(source_wav)
    long = cache[key]
    short = audio_to_onset_env(anchor_wav)
    if len(short) >= len(long):
        return {"ok": False, "reason": "anchor_longer_than_source"}
    idx, peak, ratio = _ncc_locate(short, long)
    return {
        "ok": True,
        "t_start_sec": round(idx / GRID_SR, 3),
        "t_end_sec": round((idx + len(short)) / GRID_SR, 3),
        "anchor_dur_sec": round(len(short) / GRID_SR, 3),
        "peak_ncc": round(peak, 4),
        "secondary_ratio": None if ratio == float("inf") else round(ratio, 2),
    }


def _classify(midi_res: dict, audio_res: dict | None) -> tuple[str, str, float | None, float | None]:
    """Return (method, confidence, t_start, t_end)."""
    if midi_res.get("ok"):
        m_ratio = midi_res.get("secondary_ratio") or 99.0
        if audio_res and audio_res.get("ok"):
            agrees = abs(audio_res["t_start_sec"] - midi_res["t_start_sec"]) <= AGREE_TOL_SEC
            if m_ratio >= MIDI_STRONG_RATIO and agrees:
                return "midi+audio", "strong", midi_res["t_start_sec"], midi_res["t_end_sec"]
            if m_ratio >= MIDI_STRONG_RATIO:
                return "midi", "medium", midi_res["t_start_sec"], midi_res["t_end_sec"]
            if agrees:
                return "midi+audio", "medium", midi_res["t_start_sec"], midi_res["t_end_sec"]
            return "midi", "low", midi_res["t_start_sec"], midi_res["t_end_sec"]
        # no audio result
        if m_ratio >= MIDI_STRONG_RATIO:
            return "midi", "medium", midi_res["t_start_sec"], midi_res["t_end_sec"]
        return "midi", "low", midi_res["t_start_sec"], midi_res["t_end_sec"]

    # MIDI not ok - audio must carry
    if audio_res and audio_res.get("ok"):
        a_ratio = audio_res.get("secondary_ratio") or 0.0
        a_ncc = audio_res["peak_ncc"]
        if a_ratio >= AUDIO_OK_RATIO and a_ncc >= AUDIO_OK_NCC:
            return "audio", "medium", audio_res["t_start_sec"], audio_res["t_end_sec"]
        return "audio", "low", audio_res["t_start_sec"], audio_res["t_end_sec"]
    return "none", "failed", None, None


def _logged_windows() -> dict:
    """(raw take id, piece id) -> (start, end) from the session-log arrangement CSVs."""
    import csv
    out = {}
    for f in METADATA_ROOT.glob(f"{DATASET}/*/*/{DATASET}_*_arrangement.csv"):
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    out[(r["Renamed"], r["ID"])] = (float(r["Start"]), float(r["End"]))
                except (KeyError, ValueError):
                    pass
    return out


# Session-log take names with no exported audio of that name. C1_N0: the log's
# second Technique2 take is "T2_t2", but only T2_t1 and T2_t3 were exported
# (the T2_t2 Logic project has no bounce); the log's T2_t2 windows end 0.9 s
# after the last note of T2_t3 and every one of them holds notes there.
#
# C1_N1: the Technique1 bounces are numbered in recording order (part 1, 2, 3 of
# round 1, then parts 1 and 2 of round 2) while the log names takes part-round
# ("T1-<part>_t<round>"). Each log take is in the bounce whose notes are its
# logged pieces, in order (checked against the confirmed references of all its
# pieces): T1-1_t2 is in T1-2_t2, T1-2_t1 in T1-1_t2, T1-2_t2 in T1-3_t1 and
# T1-3_t1 in T1-2_t1.
LOG_TAKE_ALIAS = {"220819_C1_N0_T2_t2": "220819_C1_N0_T2_t3",
                  "220819_C1_N1_T1-1_t2": "220819_C1_N1_T1-2_t2",
                  "220819_C1_N1_T1-2_t1": "220819_C1_N1_T1-1_t2",
                  "220819_C1_N1_T1-2_t2": "220819_C1_N1_T1-3_t1",
                  "220819_C1_N1_T1-3_t1": "220819_C1_N1_T1-2_t1"}


def _apply_take_alias(entries: list[dict]) -> None:
    """An entry of a log take in another bounce than its name is put on that bounce's trial."""
    for e in entries:
        alias = LOG_TAKE_ALIAS.get(e.get("take_id"))
        if alias and e.get("method") != "session_log":
            resolved = resolve_stage1_stem(alias)
            if resolved is not None:
                e["take_alias"] = alias
                e["stage1_stem"] = resolved[1]
                e["session"] = resolved[0].name


def _log_only_entries(entries: list[dict]) -> list[dict]:
    """
    Entries for session-log rows of sessions with no anchor takes (the anchor
    library covers C0_N2 and C1_N1). Their windows come from the log alone
    (window_source "log"); the take is resolved to its trial by audio name.
    """
    import csv
    have = {e.get("session") for e in entries if e.get("session")}
    have |= {"_".join(e["take_id"].split("_")[1:3]) for e in entries if e.get("take_id")}
    out = []
    for f in sorted(METADATA_ROOT.glob(f"{DATASET}/*/*/{DATASET}_*_arrangement.csv")):
        with open(f, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            take = r.get("Renamed") or ""
            session = "_".join(take.split("_")[1:3])
            if not take or session in have:
                continue
            resolved = resolve_stage1_stem(LOG_TAKE_ALIAS.get(take, take))
            e = {"task": r.get("Task"), "piece_id": r.get("ID"), "take_id": take,
                 "session": session, "midi": {}, "audio": None, "ok": False,
                 "method": "session_log", "confidence": "log"}
            if take in LOG_TAKE_ALIAS:
                e["take_alias"] = LOG_TAKE_ALIAS[take]
            if resolved is None:
                e["reason"] = "stage1_stem_unresolved"
            else:
                e["stage1_stem"] = resolved[1]
            out.append(e)
    return out


def _onsets(source_mid: Path, cache: dict) -> list[float]:
    key = str(source_mid)
    if key not in cache:
        import pretty_midi
        cache[key] = sorted(n.start for i in pretty_midi.PrettyMIDI(key).instruments for n in i.notes)             if source_mid.is_file() else []
    return cache[key]


def _notes_in(onsets: list[float], t0: float, t1: float) -> int:
    import bisect
    return bisect.bisect_left(onsets, t1) - bisect.bisect_left(onsets, t0)


def _split_overlaps(group: list[dict], onsets: list[float]) -> None:
    """
    Logged windows of back-to-back pieces can overlap (the log end is rough).
    Cut each overlap at the middle of the largest silence between note onsets
    within it (+-OVERLAP_SEARCH_SEC), which is where one piece stops.
    """
    for a, b in zip(group, group[1:]):
        if b["window_core"][0] >= a["window_core"][1]:
            continue
        lo, hi = b["window_core"][0] - OVERLAP_SEARCH_SEC, a["window_core"][1] + OVERLAP_SEARCH_SEC
        ons = [t for t in onsets if lo <= t <= hi]
        if len(ons) >= 2:
            gap, x = max((y - x, x) for x, y in zip(ons, ons[1:]))
            cut = round(x + gap / 2, 3)
        else:
            cut = round((b["window_core"][0] + a["window_core"][1]) / 2, 3)
        a["window_core"][1] = cut
        b["window_core"][0] = cut
        a["window_boundary_end"] = b["window_boundary_start"] = "midi_gap"


def _chords(source_mid: Path, t0: float, t1: float) -> list[tuple[float, set[int]]]:
    """(onset, pitches) of the chords in [t0, t1]: notes starting within 50 ms."""
    import pretty_midi
    if not source_mid.is_file():
        return []
    out: list[tuple[float, set[int]]] = []
    for s, q in sorted((n.start, n.pitch) for i in pretty_midi.PrettyMIDI(str(source_mid)).instruments
                       for n in i.notes if t0 <= n.start <= t1):
        if out and s - out[-1][0] < 0.05:
            out[-1][1].add(q)
        else:
            out.append((s, {q}))
    return out


_REFS: dict | None = None


def _references() -> dict:
    """score name -> reference frames (SET 2 from the score, exercises from confirmed cuts)."""
    global _REFS
    if _REFS is None:
        import importlib.util
        steps = Path(__file__).resolve().parents[2] / "steps"

        def load(name):
            spec = importlib.util.spec_from_file_location(name, steps / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod

        s13, s14 = load("s13_piece_references"), load("s14_240828_program_pieces")
        _REFS = {}
        for ds in ("221112", DATASET):
            _REFS.update({score_name(n): onset_frames(v)[0] for n, v in s13.load(ds).items()})
        _REFS.update({d: onset_frames(v)[0] for d, v in s14.set2_score().items()})
    return _REFS


def _reference_for(piece_id: str):
    refs, name = _references(), score_name(piece_id)
    base = name.split(" (")[0]
    if base in DRILLS:                               # a SET 2 drill at either tempo: the score's notes
        return refs[base]
    if name in refs:
        return refs[name]
    return refs.get(name.rstrip(" 0123456789"))      # "Beethoven 2" -> "Beethoven"


def _place_by_notes(group: list[dict], source_mid: Path) -> None:
    """Replace a logged core by the span of its notes where the piece is found near the log."""
    import pretty_midi
    if not source_mid.is_file():
        return
    notes = [(n.start, n.end, n.pitch) for i in pretty_midi.PrettyMIDI(str(source_mid)).instruments for n in i.notes]
    placed = place_in_order(notes, [_reference_for(e["piece_id"]) for e in group],
                            [e["window_core"][0] for e in group])
    for e, pl in zip(group, placed):
        if pl and abs(pl[0] - e["window_core"][0]) <= PLACE_MAX_FROM_LOG_S:
            e["window_core"] = [pl[0], pl[1]]
            e["window_source"] = "notes"
            e["match_cost"] = pl[2]


def is_d_octave(pitches: set[int]) -> bool:
    """A chord of D's only (at least two): the octave scale's second step."""
    return len(pitches) >= 2 and all(q % 12 == 2 for q in pitches)


def _trim_scale_starts(group: list[dict], source_mid: Path) -> None:
    """An octave repeat running on into a scale attempt that was broken off (and played
    again later) ends before that attempt's first C: a C octave followed by the D octave."""
    for e in group:
        if not score_name(e["piece_id"]).startswith("Octave Repeat"):
            continue
        ch = _chords(source_mid, e["window_core"][0], e["window_core"][1] + JUNCTION_SEARCH_S)
        inside = [k for k, (t, _) in enumerate(ch) if t <= e["window_core"][1]]
        if len(inside) < 2:
            continue
        k = inside[-1]
        if k + 1 < len(ch) and all(q % 12 == 0 for q in ch[k][1]) and is_d_octave(ch[k + 1][1]) and ch[k + 1][0] - ch[k][0] <= JUNCTION_MAX_GAP_S:
            e["window_core"][1] = round((ch[k - 1][0] + ch[k][0]) / 2, 3)


def _split_octave_junctions(group: list[dict], source_mid: Path) -> None:
    """
    The octave repeat (C octaves) runs straight into the octave scale, which
    starts on the same C octave (C D E ...): no silence marks the boundary.
    The scale begins on the last C-octave chord before the first D-octave
    chord; the cut is half way between that chord and the C before it.
    """
    for a, b in zip(group, group[1:]):
        if not (score_name(a["piece_id"]).startswith("Octave Repeat")
                and score_name(b["piece_id"]).startswith("Octave Scale")):
            continue
        # only where the scale (as placed) starts within the repeat's run of C octaves:
        # a scale played again after a pause keeps its own start
        ch = _chords(source_mid, a["window_core"][0], b["window_core"][0] + JUNCTION_SEARCH_S)
        is_c = [all(q % 12 == 0 for q in c) for _, c in ch]
        # the scale's D octave: a chord of D's only (a stray D among the repeats is not it)
        first_d = next((k for k in range(1, len(ch)) if is_c[k - 1] and is_d_octave(ch[k][1])
                        and ch[k][0] >= b["window_core"][0] - JUNCTION_SEARCH_S), None)
        if first_d is None or first_d < 2 or ch[first_d - 1][0] - ch[first_d - 2][0] > JUNCTION_MAX_GAP_S:
            continue
        cut = round((ch[first_d - 2][0] + ch[first_d - 1][0]) / 2, 3)
        a["window_core"][1] = cut
        b["window_core"][0] = cut
        a["window_boundary_end"] = b["window_boundary_start"] = "octave_scale_start"


def _apply_session_log_windows(entries: list[dict]) -> None:
    """Set t_start_sec / t_end_sec from the session log, corrected by locate when it agrees."""
    logged = _logged_windows()
    note_cache: dict = {}
    for e in entries:
        if "midi" not in e or "stage1_stem" not in e:
            continue
        e["located"] = {"t_start_sec": e.get("t_start_sec"), "t_end_sec": e.get("t_end_sec")}
        win = logged.get((e["take_id"], e["piece_id"]))
        if win is None:
            e["window_source"] = "locate"          # no log entry: keep the located window
            e["window_core"] = [e.get("t_start_sec"), e.get("t_end_sec")]
            continue
        log_s, log_e = win
        loc_s = e.get("t_start_sec")
        if loc_s is not None and abs(loc_s - log_s) <= LOG_AGREE_SEC:
            shift = loc_s - log_s
            e["window_source"] = "log+locate"
        else:
            shift = 0.0
            e["window_source"] = "log"             # locate disagreed or failed
        e["log_shift_sec"] = round(shift, 3)
        e["t_start_sec"] = round(log_s + shift, 3)
        e["t_end_sec"] = round(log_e + shift, 3)
        e["window_core"] = [e["t_start_sec"], e["t_end_sec"]]
        e["ok"] = True
        # A logged piece the recording does not contain (the take was cut
        # short): no notes in its window. Do not cut it from other material.
        src = NORMALIZED_ROOT / DATASET / e["session"] / "midi" / f"{e['stage1_stem']}.mid"
        onsets = _onsets(src, note_cache)
        if onsets and _notes_in(onsets, e["t_start_sec"], e["t_end_sec"]) == 0:
            e["ok"] = False
            e["reason"] = "logged_piece_not_in_recording"
    # Pad, without running into the neighbouring piece of the same recording:
    # starts first (bounded by the previous piece's core end), then ends
    # (bounded by the next piece's padded start).
    by_rec: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        if e.get("ok") and e.get("t_start_sec") is not None:
            by_rec[e["stage1_stem"]].append(e)
    for group in by_rec.values():
        group.sort(key=lambda x: x["t_start_sec"])
        g0 = group[0]
        src = NORMALIZED_ROOT / DATASET / g0["session"] / "midi" / f"{g0['stage1_stem']}.mid"
        _place_by_notes(group, src)
        _split_overlaps(group, _onsets(src, note_cache))
        _split_octave_junctions(group, src)
        _trim_scale_starts(group, src)
        # windows the reviewer set by hand are cores in place of the computed ones
        cores, by_reviewer = window_overrides.apply([tuple(g["window_core"]) for g in group],
                                                    window_overrides.load(DATASET).get(g0["stage1_stem"], []))
        ons = _onsets(src, note_cache)
        wav = NORMALIZED_ROOT / DATASET / g0["session"] / "audio" / f"{g0['stage1_stem']}.wav"
        rec_end = float("inf")
        if wav.is_file():                         # the padding never runs past the recording
            import wave
            with wave.open(str(wav)) as w:
                rec_end = w.getnframes() / w.getframerate()
        pedal = sustain_spans(src)                # a core ends when its last note stops sounding
        for i, g in enumerate(group):
            g["window_core"] = [cores[i][0], round(sounding_end(cores[i][1], pedal, ons, rec_end), 3)]
            g["t_start_sec"], g["t_end_sec"] = g["window_core"]
            if i in by_reviewer:
                g["window_source"] = "reviewer"
        core_s = [g["t_start_sec"] for g in group]
        core_e = [g["t_end_sec"] for g in group]
        quiet = [silent_bounds(ons, a, b) for a, b in zip(core_s, core_e)]   # padding holds no other note
        for i, e in enumerate(group):
            lo = max(core_e[i - 1] + NEIGHBOUR_GAP_SEC if i > 0 else 0.0, quiet[i][0])
            e["t_start_sec"] = round(max(lo, core_s[i] - PAD_BEFORE_SEC) if lo < core_s[i] else core_s[i], 3)
        for i, e in enumerate(group):
            hi = min(group[i + 1]["t_start_sec"] - NEIGHBOUR_GAP_SEC if i + 1 < len(group) else rec_end, quiet[i][1])
            e["t_end_sec"] = round(min(hi, core_e[i] + PAD_AFTER_SEC) if hi > core_e[i] else core_e[i], 3)


def recompute_windows() -> None:
    """Re-derive final windows (and _p slots) from an existing manifest's locate results."""
    m = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = [e for e in m["entries"] if e.get("method") != "session_log"]
    entries += _log_only_entries(entries)
    _apply_take_alias(entries)
    m["entries"] = entries
    for e in entries:
        if "located" in e:                          # restore raw locate before re-deriving
            e["t_start_sec"] = e["located"]["t_start_sec"]
            e["t_end_sec"] = e["located"]["t_end_sec"]
            e["ok"] = e["t_start_sec"] is not None
            for k in ("reason", "window_boundary_start", "window_boundary_end"):
                e.pop(k, None)
    _apply_session_log_windows(entries)
    by_stem: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in entries:
        e.pop("p_slot", None); e.pop("out_stem", None)
        if e.get("ok"):
            by_stem[(e["session"], e["stage1_stem"])].append(e)
    for group in by_stem.values():
        group.sort(key=lambda x: x["t_start_sec"])
        for idx, e in enumerate(group, 1):
            e["p_slot"] = idx
            e["out_stem"] = f"{e['stage1_stem']}_p{idx}"
    m["window_rule"] = {"LOG_AGREE_SEC": LOG_AGREE_SEC, "OVERLAP_SEARCH_SEC": OVERLAP_SEARCH_SEC,
                        "PAD_BEFORE_SEC": PAD_BEFORE_SEC,
                        "PAD_AFTER_SEC": PAD_AFTER_SEC, "NEIGHBOUR_GAP_SEC": NEIGHBOUR_GAP_SEC}
    MANIFEST_PATH.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    print("[s01_locate] windows:", dict(Counter(e.get("window_source") for e in entries if e.get("ok"))))
    dropped = [e["out_stem"] if "out_stem" in e else f"{e['stage1_stem']}:{e['piece_id']}" for e in entries if e.get("reason") == "logged_piece_not_in_recording"]
    print("[s01_locate] logged but not in recording:", dropped)


def run(dataset: str | None = None, force: bool = False) -> None:
    if dataset is not None and dataset != DATASET:
        print(f"[s01_locate] skip - only {DATASET} supported")
        return
    if MANIFEST_PATH.is_file() and not force:
        print(f"[s01_locate] manifest exists; pass --force to recompute: {MANIFEST_PATH}")
        return

    index = json.loads((ANCHOR_ROOT / "index.json").read_text(encoding="utf-8"))
    jobs: list[tuple[str, str, dict]] = []
    for task, pieces in index["tasks"].items():
        if task == "Unknown":
            continue
        for piece_id, pinfo in pieces.items():
            for take in pinfo["takes"]:
                jobs.append((task, piece_id, take))

    print(f"[s01_locate] {len(jobs)} anchor takes to locate")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    midi_cache: dict = {}
    audio_cache: dict = {}
    entries: list[dict] = []
    t_all = time.time()

    # Pass 1: MIDI xcorr for all
    for i, (task, piece_id, take) in enumerate(jobs, 1):
        take_id = take["take_id"]
        resolved = resolve_stage1_stem(take_id)
        if resolved is None:
            entries.append({"task": task, "piece_id": piece_id, "take_id": take_id,
                            "ok": False, "reason": "stage1_stem_unresolved"})
            continue
        session_dir, stage1_stem = resolved
        anchor_mid = ANCHOR_ROOT / task / piece_id / f"{take_id}.mid"
        source_mid = session_dir / "midi" / f"{stage1_stem}.mid"
        if not anchor_mid.is_file() or not source_mid.is_file():
            entries.append({"task": task, "piece_id": piece_id, "take_id": take_id,
                            "ok": False, "reason": "midi_input_missing"})
            continue

        m = _midi_locate_with_cache(anchor_mid, source_mid, midi_cache)
        entries.append({
            "task": task, "piece_id": piece_id, "take_id": take_id,
            "session": session_dir.name, "stage1_stem": stage1_stem,
            "midi": m, "audio": None, "ok": False,
        })
        if i % 40 == 0 or i == len(jobs):
            print(f"  [midi {i}/{len(jobs)}] {time.time()-t_all:.0f}s")

    # Pass 2: audio xcorr for entries where MIDI weak/empty (cross-validation or fallback)
    need_audio: list[dict] = []
    for e in entries:
        if "midi" not in e:
            continue
        m = e["midi"]
        if not m.get("ok"):
            need_audio.append(e)
            continue
        ratio = m.get("secondary_ratio") or 99.0
        if ratio < MIDI_STRONG_RATIO:
            need_audio.append(e)

    print(f"\n  audio fallback/tiebreak needed: {len(need_audio)} entries")
    t_audio = time.time()
    for i, e in enumerate(need_audio, 1):
        session_dir = NORMALIZED_ROOT / DATASET / e["session"]
        anchor_wav = ANCHOR_ROOT / e["task"] / e["piece_id"] / f"{e['take_id']}.wav"
        source_wav = session_dir / "audio" / f"{e['stage1_stem']}.wav"
        if not anchor_wav.is_file() or not source_wav.is_file():
            e["audio"] = {"ok": False, "reason": "audio_input_missing"}
            continue
        e["audio"] = _audio_locate_with_cache(anchor_wav, source_wav, audio_cache)
        if i % 10 == 0 or i == len(need_audio):
            print(f"  [audio {i}/{len(need_audio)}] {time.time()-t_audio:.0f}s")

    # Classify + assemble final fields
    for e in entries:
        if "midi" not in e:
            continue
        method, confidence, t_s, t_e = _classify(e["midi"], e.get("audio"))
        e["method"] = method
        e["confidence"] = confidence
        e["t_start_sec"] = t_s
        e["t_end_sec"] = t_e
        e["anchor_dur_sec"] = (e["midi"].get("anchor_dur_sec")
                               or (e["audio"] or {}).get("anchor_dur_sec"))
        e["ok"] = t_s is not None

    entries += _log_only_entries(entries)
    _apply_take_alias(entries)
    _apply_session_log_windows(entries)

    # Assign _p{N} slot by sorted rank of t_start within each Stage 01 stem
    by_stem: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in entries:
        if e["ok"]:
            by_stem[(e["session"], e["stage1_stem"])].append(e)
    for group in by_stem.values():
        group.sort(key=lambda x: x["t_start_sec"])
        for idx, e in enumerate(group, 1):
            e["p_slot"] = idx
            e["out_stem"] = f"{e['stage1_stem']}_p{idx}"

    ok = sum(1 for e in entries if e["ok"])
    conf_counts: dict[str, int] = defaultdict(int)
    for e in entries:
        if e["ok"]:
            conf_counts[e["confidence"]] += 1

    manifest = {
        "dataset": DATASET,
        "n_entries": len(entries),
        "n_ok": ok,
        "confidence_counts": dict(conf_counts),
        "entries": entries,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[s01_locate] done ({time.time()-t_all:.0f}s)")
    print(f"  ok: {ok}/{len(entries)}")
    for k in ("strong", "medium", "low", "failed"):
        if conf_counts.get(k):
            print(f"    confidence={k}: {conf_counts[k]}")
    print(f"  manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--windows-only", action="store_true",
                    help="re-derive final windows from the existing manifest's locate results")
    args = ap.parse_args()
    if args.windows_only:
        recompute_windows()
    else:
        run(dataset=args.dataset, force=args.force)
