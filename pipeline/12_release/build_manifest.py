"""
Stage 12: the release manifest -- which files of Data/ a release version holds,
and under which path.

Nothing is copied. The manifest maps each released path to its source file in
Data/Pipeline; the upload step sends those files as they are. A version is a
semantic version (0.1.0, 0.2.0, ..., 1.0.0).

What is released (Stage 05 is the gate; a dataset must have passed):

  pieces  every confirmed, named piece cut of a piece-level dataset
          (220819, 221112, 240828), one per participant x task x piece x round:
          the release keeps a representative performance, not retakes. Where
          several confirmed cuts fall in one group, the last one played is kept.
          Rounds: 240828 plays its technique programs twice (and its T0 program
          names some drills twice within a pass); a cut is placed on the next
          slot of the program that carries its name, a cut whose name has no
          later slot opens the next pass when the pass is past its middle, and
          is otherwise a retake of the slot it repeats. 220819 recorded every
          drill in planned repeated takes, and some drills twice within a take,
          so there every confirmed cut is its own round. Elsewhere every take of a piece is the same round.
  takes   the take-level dataset (240523, amateur free repertoire): every trial
          as a whole, as the paper states for free repertoire. It is released
          once, as its take folder (data/P{id}/takes/...; muted, on the trial
          clock), which performances.csv lists as the performance and
          pieces.csv marks best; with --no-takes (and before take folders) it
          was data/P{id}/free/take01, ... from the unmuted trial sources.
  full recordings
          every take (trial) of every released session, whole, in the take
          folders below: on the trial clock of build_trials.py (time 0 = the
          trial audio's start; Data/Pipeline/12_release/trials), speech muted
          where the piano is silent by the MIDI. 221112 participants 14 and 18
          have hand and body motion only (no audio/MIDI was recorded for them).
          pieces.csv marks every piece window in them, and which one is the
          representative performance of the piece-level folders.

Cuts the reviewer or Stage 05 called not a piece are left out, and so is every
session out of release scope (C0_N0; paper_builders/participant_map.py). In a
retake dataset (221112) only the confirmed take of a piece is released, never a
retake Stage 05 settled through it (`represented_by`). Session-log names that
are no program piece are left out as practice variants: one-hand practice of a
Fink exercise ("28(left hand)") and notes written in the log's own language
(any non-ASCII name); the subject's regular performance of the piece is
released instead.

Layout: data/P{participant id}/{task}/{piece}/r{n}/{modality file}, the
participant id (1-19) from paper_builders/participant_map.py, r1, r2, ... the
participant's performances of that piece in play order (free repertoire by
whole take: its take folder, see below). The task follows the paper's
categories, read from the piece rather than from the session's T-tag (240828
labels its Fink programme "technique2" and its warm-up drills "technique1"):
    technique1        the numbered Fink exercises (4 ... 51)
    technique2        the drills (scale, arpeggio, octave repeat, octave scale,
                      broken octave arpeggio; moderate and fast) and their
                      applied pieces (Beethoven 1, 2; 240828: Beethoven, unnumbered)
    technique_extra   240828's scale and arpeggio programme (C and D-flat major,
                      loudness and articulation variation), outside the paper's
                      26 task items
    graded            the Set 1 graded pieces
    free              free repertoire

Take folders: data/P{participant id}/takes/{session}_{take}/ -- the session date
and the take's own label in the recording (240828_T1_P13_t1 = 240828 task tag 1,
take P13, attempt t1; the T-tag is the recording's, not the paper's task) --
holding audio.wav, midi.mid, hand.csv, body.csv where they exist. A piece window
of pieces.csv cuts a take's files as build_trials.py states (audio from the
millisecond, MIDI notes in the window, motion rows hand/body_start_frame ..
_end_frame). Trial-level video is not released yet, except where a free-repertoire
take carries its own (top.mp4, with fingering and the Stage 06 files, which stay
keyed by the take's former path free/take{nn}). takes.csv says whether a take's
audio is muted (audio_muted; false, with audio_note, for a take with no MIDI).

Modalities per version (MODALITIES): audio (.wav), MIDI (.mid), hand and body
motion (.csv, 120 fps, frame 0 = time 0 of the audio and MIDI); optional ones
(OPTIONAL, listed only where the file exists, never counted missing): the Top
camera view as cut at the same window and passed through EgoBlur (Stage 10,
Data/Pipeline/10_face_blur/{ds}/videos_blurred/{stem}_Top.mp4), and per-note
fingering labeled from the raw hand motion (Stage 08 step 03,
Data/Pipeline/08_fingering/Notes/raw/{ds}/{stem}.csv -> fingering.csv).

Outputs, under Data/Release/v{version}/:
    manifest.csv       release_path, source_path, bytes
    performances.csv   one row per performance: path, participant, task, piece,
                       round, session, source take and window in it, first
                       onset and release (seconds from the file's start)
    participants.csv   participant id, group, sessions
    repertoire.csv     piece id, task, title
    takes.csv          one row per take folder: path, participant, session, source
                       take, modalities, audio / motion seconds, MIDI notes, muted
                       seconds, and the free-repertoire performance it is (240523)
    pieces.csv         one row per piece window Stage 04 cut in a released take
                       (retakes, practice variants and cuts that are no piece
                       included): take folder, window (take clock) and motion rows,
                       piece id, name, task, status (confirmed / retake / rejected /
                       practice_variant / not_a_piece / unnamed / unreviewed), best
                       (the performance released in the piece-level folders, with its
                       path), confirmed_by. Names not written in English are given as
                       "unlisted" (name_note says so); practice variants keep task
                       "practice".
    summary.json       counts per dataset, and what was left out and why; take
                       totals (count, hours of audio and of motion, MIDI notes)

Usage (from the repository root):
  python pipeline/12_release/build_manifest.py --version 0.1.0 [--dataset 240828 ...]
      [--optional-in top=240523,220819]   (an optional modality for some datasets only)
      [--imputed-model TAG]                (list hand_imputed*, fingering_imputed of that model)
      [--no-takes]                         (piece-level folders only, no take folders)

Take folders need Data/Pipeline/12_release/trials (build_trials.py) first.

Imputed hand motion is listed only with --imputed-model: the gap-filled hand
motion of that model (Stage 06 release fill, IMPUTED/<tag>/<release path>/
hand_imputed.csv and hand_imputed_mask.csv, 1 = filled) and the fingering
labeled from it (Stage 08 --hand imputed --tag TAG, FINGERING_IMPUTED/<tag>/{ds}/{stem}.csv).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import pretty_midi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper_builders"))
sys.path.insert(0, str(ROOT / "pipeline"))
from participant_map import EXCLUDED_FROM_RELEASE, participant_id, participant_role  # noqa: E402
from _common.score_follow import sounding_end, sustain_spans  # noqa: E402

PIPE = Path("Data/Pipeline")
QA = PIPE / "05_alignment_qa"
PIECES = PIPE / "04_piece_split"
HAND = PIPE / "02_spatial" / "hand" / "Aligned"
BODY = PIPE / "02_spatial" / "body" / "Aligned"
SYNC_MIDI = PIPE / "03_sync" / "midi_motion" / "Aligned"
META = Path("Data/Metadata")
OUT = Path("Data/Release")

PIECE_LEVEL = ["220819", "221112", "240828"]
ROUND_PER_CUT = {"220819"}            # planned repeated takes: every confirmed cut is released
TAKE_LEVEL = ["240523"]
MODALITIES = {"audio": "audio.wav", "midi": "midi.mid", "hand": "hand.csv", "body": "body.csv"}
PIECE_SOURCES = {"audio": ".wav", "midi": ".mid", "hand": "_hand.csv", "body": "_body.csv"}
BLURRED = PIPE / "10_face_blur"
FINGERING = PIPE / "08_fingering" / "Notes" / "raw"
FINGERING_IMPUTED = PIPE / "08_fingering" / "Notes" / "imputed"
IMPUTED = PIPE / "06_hand_imputation" / "Release"       # <model tag>/<release path>/hand_imputed*.csv
SUSPECT = PIPE / "06_hand_imputation" / "Suspect"       # <release path>/hand_suspect_mask.csv (flag_suspect_hand.py)
TRIALS = PIPE / "12_release" / "trials"                 # build_trials.py: whole takes on the trial clock
TAKE_FILES = ("audio.wav", "midi.mid", "hand.csv", "body.csv")
MODALITIES["top"] = "top.mp4"
MODALITIES["fingering"] = "fingering.csv"
MODALITIES["hand_imputed"] = "hand_imputed.csv"
MODALITIES["hand_imputed_mask"] = "hand_imputed_mask.csv"
MODALITIES["fingering_imputed"] = "fingering_imputed.csv"
MODALITIES["hand_suspect_mask"] = "hand_suspect_mask.csv"
OPTIONAL = {"top", "fingering", "hand_imputed", "hand_imputed_mask", "fingering_imputed", "hand_suspect_mask"}
IMPUTED_MODALITIES = ("hand_imputed", "hand_imputed_mask", "fingering_imputed")
SESSION_ORDER = ["220819", "221112", "240523", "240828"]
PRACTICE_VARIANT = re.compile(r"^\d+\s*\((left|right|both) hands?\)$")   # one-hand practice of a Fink exercise
FINK = re.compile(r"^(\d+)(?: - (Forte|Piano))?$")
DRILL = re.compile(r"^(Scale|Arpeggio|Octave Repeat|Octave Scale|Broken Octave Arpeggio) "
                   r"\((moderate|fast|fastest)(, x5)?\)$")
EXTRA_240828 = re.compile(r"^((C|Db) Major (Scale|Arpeggio|Octave Scale)( \(3 octaves\)| \(fast\))?"
                          r"|Loudness Variation|Articulation Variation)$")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def classify(ds: str, task: str, name: str | None) -> tuple[str, str]:
    """(task, piece id) in the paper's categories, from the piece itself."""
    name = name or ""
    if ds in TAKE_LEVEL:
        return "free", ""
    if task == "9":
        return "free", slug(re.sub(r",? both movements$", "", name))
    if task == "2" and ds == "240828":
        return "graded", name
    if m := FINK.match(name):
        return "technique1", "fink_" + slug(" ".join(x for x in m.groups() if x))
    if DRILL.match(name) or re.match(r"^Beethoven( \d)?$", name):   # 240828 names it without a number
        return "technique2", slug(name)
    if ds == "240828" and EXTRA_240828.match(name):
        return "technique_extra", slug(name)
    raise ValueError(f"{ds} T{task}: no task category for piece {name!r}")


def onset_release(midi: Path) -> tuple[float | None, float | None]:
    """First note onset, and when the last note stops sounding, in the file's time."""
    pm = pretty_midi.PrettyMIDI(str(midi))
    notes = [n for i in pm.instruments for n in i.notes]
    if not notes:
        return None, None
    ons = sorted(n.start for n in notes)
    end = max(n.end for n in notes)
    return round(ons[0], 3), round(sounding_end(end, sustain_spans(midi), ons, max(pm.get_end_time(), end)), 3)


def read_json(p: Path, default):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def parse(stem: str) -> dict:
    """240828_C0_N4_T1_P13_t1_p7 -> dataset, session, task, take order, piece index."""
    m = re.match(r"^(\d{6})_(C\d_N\d+)_T(\w+?)_P(\d+)_t(\d+)(?:_p(\d+))?$", stem)
    if not m:
        raise ValueError(stem)
    ds, sess, task, p, t, k = m.groups()
    return {"dataset": ds, "session": sess, "task": task, "trial": stem.rsplit("_p", 1)[0] if k else stem,
            "order": (int(p), int(t)), "piece_idx": int(k) if k else None}


def programs() -> dict[str, list[str]]:
    """240828 technique programs, by task (one pass)."""
    return {k: v for k, v in read_json(META / "240828" / "technique_program.json", {}).get("programs", {}).items()}


def rounds_240828(pieces: list[dict], program: list[str]) -> None:
    """Set p["slot"] = (pass, program index) on one participant's cuts of one task, in play order."""
    passno, ptr, last_slot = 1, 0, {}
    for p in pieces:
        name = p["name"]
        nxt = next((j for j in range(ptr, len(program)) if program[j] == name), None)
        if nxt is None and name in program and ptr > len(program) // 2:
            passno, ptr = passno + 1, 0            # the program starts again
            nxt = next((j for j in range(ptr, len(program)) if program[j] == name), None)
        if nxt is not None:
            p["slot"] = (passno, nxt)
            last_slot[name] = (passno, nxt)
            ptr = nxt + 1
        else:
            p["slot"] = last_slot.get(name, (passno, -1))   # a retake of the slot it repeats


def piece_rows(ds: str, left_out: dict) -> list[dict]:
    status = read_json(QA / ds / "_status.json", {})
    if not status.get("summary", {}).get("passed"):
        sys.exit(f"{ds}: Stage 05 has not passed; run pipeline/05_alignment_qa first")
    rows = []
    for p in status["pieces"]:
        info = parse(p["piece"])
        if (ds, info["session"]) in EXCLUDED_FROM_RELEASE:
            left_out["out_of_scope"] = left_out.get("out_of_scope", 0) + 1
            continue
        if p.get("boundary_confirmed_by") == "not_a_piece":
            left_out["not_a_piece"] = left_out.get("not_a_piece", 0) + 1
            continue
        if not p.get("name"):
            left_out["unnamed"] = left_out.get("unnamed", 0) + 1
            continue
        if p.get("represented_by") and not p.get("confirmed"):
            left_out["retake_of_confirmed_take"] = left_out.get("retake_of_confirmed_take", 0) + 1
            continue
        if PRACTICE_VARIANT.match(p["name"]) or not p["name"].isascii():
            left_out["practice_variant"] = left_out.get("practice_variant", 0) + 1
            continue
        rows.append({**info, "piece": p["piece"], "name": p["name"], "window_sec": p["window_sec"],
                     "confirmed_by": p.get("boundary_confirmed_by")})
    rows.sort(key=lambda r: (r["session"], r["task"], r["order"], r["window_sec"][0]))
    if ds == "240828":
        progs = programs()
        by = {}
        for r in rows:
            by.setdefault((r["session"], r["task"]), []).append(r)
        for (_, task), rs in by.items():
            if task in progs:
                rounds_240828(rs, progs[task])
    if ds in ROUND_PER_CUT:
        n: dict[tuple, int] = {}
        for r in rows:
            k = (r["session"], r["task"], r["name"])
            n[k] = n.get(k, 0) + 1
            r["slot"] = (n[k], r["piece"])
    # one per participant x task x piece x round: the last one played
    keep: dict[tuple, dict] = {}
    for r in rows:
        key = (r["session"], r["task"], r["name"], r.get("slot"))
        if key in keep:
            left_out["retake_not_representative"] = left_out.get("retake_not_representative", 0) + 1
        keep[key] = r
    out = list(keep.values())
    for r in out:
        r["round"] = r["slot"][0] if r.get("slot") else 1
    return out


def video_released(f: Path) -> bool:
    """A blurred clip ships only with its Stage 10 step 04 record: passed the
    coverage check, or flagged and passed on review ({clip}.json "review")."""
    j = read_json(f.with_suffix(".json"), None)
    return bool(j) and (j.get("passed") is True or (j.get("review") or {}).get("verdict") == "pass")


def files_for_piece(r: dict) -> dict[str, Path]:
    return {**{m: PIECES / r["dataset"] / f"{r['piece']}{suf}" for m, suf in PIECE_SOURCES.items()},
            "top": BLURRED / r["dataset"] / "videos_blurred" / f"{r['piece']}_Top.mp4",
            "fingering": FINGERING / r["dataset"] / f"{r['piece']}.csv"}


def take_rows(ds: str) -> list[dict]:
    status = read_json(QA / ds / "_status.json", {})
    if not status.get("summary", {}).get("passed"):
        sys.exit(f"{ds}: Stage 05 has not passed; run pipeline/05_alignment_qa first")
    rows = []
    for t in status["trials"]:
        if not t.get("confirmed"):
            continue
        info = parse(t["trial"])
        if (ds, info["session"]) in EXCLUDED_FROM_RELEASE:
            continue
        rows.append({**info, "piece": t["trial"], "name": None, "window_sec": None, "round": 1,
                     "confirmed_by": "sync_" + (t.get("confirmed_by") or "review")})
    return rows


def files_for_take(r: dict, raw_stems: dict[str, str]) -> dict[str, Path]:
    ds, stem = r["dataset"], r["piece"]
    return {"audio": PIECES / ds / f"{stem}.wav",
            "midi": SYNC_MIDI / ds / f"{raw_stems[stem]}.mid",
            "hand": HAND / ds / f"{stem}.csv",
            "body": BODY / ds / f"{stem}.csv",
            "top": BLURRED / ds / "videos_blurred" / f"{stem}_Top.mp4",
            "fingering": FINGERING / ds / f"{stem}.csv"}


def take_path(pid: int, ds: str, session: str, trial: str) -> str:
    """data/P{id}/takes/{session date}_{take label}: 240828_C0_N4_T1_P13_t1 -> data/P03/takes/240828_T1_P13_t1."""
    return f"data/P{pid:02d}/takes/{ds}_{trial[len(ds) + len(session) + 2:]}"


def take_rows_full(datasets: list[str], index: dict, performance_of: dict[str, str]) -> tuple[list, list, dict]:
    """Manifest rows and takes.csv rows of every released take, and the totals per dataset."""
    manifest, table, totals = [], [], {}
    sync = {}
    for ds in datasets:
        for t in read_json(QA / ds / "_status.json", {}).get("trials", []):
            sync[t["trial"]] = t.get("sync_verdict")
    for stem, t in sorted(index.items(), key=lambda kv: (SESSION_ORDER.index(kv[1]["dataset"]), kv[0])):
        ds = t["dataset"]
        if ds not in datasets or (ds, t["session"]) in EXCLUDED_FROM_RELEASE or not t["files"]:
            continue
        if sync.get(stem) == "no" and ("audio.wav" in t["files"] or "midi.mid" in t["files"]):
            continue                                   # Stage 05 rejected its sync (a motion-only take has none)
        pid = participant_id(ds, t["session"])
        if pid is None:
            sys.exit(f"no participant id (out of release scope?): {ds} {t['session']}")
        base = take_path(pid, ds, t["session"], stem)
        listed = performance_of.get(stem) == base       # a free-repertoire take: listed as its performance
        have = []
        for name in TAKE_FILES:
            f = TRIALS / ds / stem / name
            if name in t["files"] and f.exists():
                have.append(name.split(".")[0])
                if listed:
                    continue
                manifest.append({"release_path": f"{base}/{name}", "source_path": f.as_posix(),
                                 "bytes": f.stat().st_size})
        motion = max(t.get("hand_frames", 0), t.get("body_frames", 0)) / 120.0
        table.append({"path": base, "participant_id": pid, "group": participant_role(pid), "session": ds,
                      "session_subject": t["session"], "source_take": stem, "modalities": " ".join(have),
                      "audio_sec": t.get("audio_sec"), "motion_sec": round(motion, 3) if motion else None,
                      "midi_notes": t.get("n_notes"), "muted_sec": t.get("muted_sec"),
                      "audio_muted": t.get("audio_muted", "muted_sec" in t) if "audio" in have else None,
                      "audio_note": t.get("audio_note", "") if "audio" in have else "",
                      "performance_path": performance_of.get(stem)})
        v = totals.setdefault(ds, {"takes": 0, "audio_hours": 0.0, "motion_hours": 0.0, "midi_notes": 0,
                                   "motion_only_takes": 0, "muted_hours": 0.0})
        v["takes"] += 1
        v["audio_hours"] += (t.get("audio_sec") or 0.0) / 3600
        v["motion_hours"] += motion / 3600
        v["muted_hours"] += (t.get("muted_sec") or 0.0) / 3600
        v["midi_notes"] += t.get("n_notes") or 0
        v["motion_only_takes"] += "audio" not in have and "midi" not in have
    for v in totals.values():
        for k in ("audio_hours", "motion_hours", "muted_hours"):
            v[k] = round(v[k], 3)
    return manifest, table, totals


def piece_status(p: dict, kept: set[str]) -> str:
    """What became of a Stage 04 cut, in the order piece_rows decides it."""
    if p.get("boundary_confirmed_by") == "not_a_piece":
        return "not_a_piece"
    if p["piece"] in kept:
        return "confirmed"
    if p.get("boundary_verdict") == "no":
        return "rejected"
    if not p.get("name"):
        return "unnamed" if p.get("confirmed") else "unreviewed"
    if p.get("represented_by") and not p.get("confirmed"):
        return "retake"
    if PRACTICE_VARIANT.match(p["name"]) or not p["name"].isascii():
        return "practice_variant"
    return "retake" if p.get("confirmed") else "unreviewed"


def pieces_table(datasets: list[str], index: dict, best: dict[str, str],
                 performance_of: dict[str, str]) -> list[dict]:
    """pieces.csv: every piece window Stage 04 cut in a released take; a take released whole as a
    free-repertoire performance (240523) is one more row spanning the take, the best one."""
    rows = []
    sync_by = {t["trial"]: t.get("confirmed_by") for ds in datasets
               for t in read_json(QA / ds / "_status.json", {}).get("trials", [])}
    for stem, path in performance_of.items():
        t = index[stem]
        pid = participant_id(t["dataset"], t["session"])
        end = t.get("audio_sec") or round(max(t.get("hand_frames", 0), t.get("body_frames", 0)) / 120.0, 3)
        rows.append({"participant_id": pid, "take": path, "source_piece": stem, "start_sec": 0.0, "end_sec": end,
                     "hand_start_frame": 0 if t.get("hand_frames") else None, "hand_end_frame": t.get("hand_frames"),
                     "body_start_frame": 0 if t.get("body_frames") else None, "body_end_frame": t.get("body_frames"),
                     "motion_residual_frames": 0, "piece_id": "", "name": "", "name_note": "whole take",
                     "task": "free", "status": "confirmed", "best": True, "best_path": path,
                     "confirmed_by": "sync_" + (sync_by.get(stem) or "review"),
                     "reproduces_piece_files": None})
    for ds in datasets:
        for p in read_json(QA / ds / "_status.json", {}).get("pieces", []):
            info = parse(p["piece"])
            if (ds, info["session"]) in EXCLUDED_FROM_RELEASE:
                continue
            pid = participant_id(ds, info["session"])
            t = index.get(info["trial"])
            if pid is None or t is None:
                continue
            status = piece_status(p, set(best))
            name, note = (p.get("name") or ""), ""
            if status == "not_a_piece":
                name = ""
            practice = status == "practice_variant" or bool(name and (PRACTICE_VARIANT.match(name) or not name.isascii()))
            if name and not name.isascii():
                name, note = "unlisted", "the session log names it in another language"
            task, piece_id = "", ""
            if ds in TAKE_LEVEL:
                task = "free"
            elif practice:
                task = "practice"
            elif name:
                try:
                    task, piece_id = classify(ds, info["task"], name)
                except ValueError:
                    task = "unclassified"
            c = t.get("pieces", {}).get(p["piece"], {})
            hand, body = c.get("hand") or {}, c.get("body") or {}
            checked = [c[m].get("ok") for m in ("audio", "midi", "hand", "body") if m in c]
            rows.append({"participant_id": pid, "take": take_path(pid, ds, info["session"], info["trial"]),
                         "source_piece": p["piece"], "start_sec": p["window_sec"][0], "end_sec": p["window_sec"][1],
                         "hand_start_frame": hand.get("start_frame"), "hand_end_frame": hand.get("end_frame"),
                         "body_start_frame": body.get("start_frame"), "body_end_frame": body.get("end_frame"),
                         "motion_residual_frames": hand.get("residual_frames", body.get("residual_frames")),
                         "piece_id": piece_id, "name": name, "name_note": note, "task": task, "status": status,
                         "best": p["piece"] in best, "best_path": best.get(p["piece"], ""),
                         "confirmed_by": p.get("boundary_confirmed_by") or "",
                         "reproduces_piece_files": all(checked) if checked else None})
    rows.sort(key=lambda r: (r["participant_id"], r["take"], r["start_sec"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--dataset", nargs="*", default=PIECE_LEVEL + TAKE_LEVEL)
    ap.add_argument("--optional-in", action="append", default=[], metavar="MODALITY=DS,DS",
                    help="list an optional modality only for these datasets (empty: none), "
                         "e.g. top=240523,220819 while the rest is still being processed")
    ap.add_argument("--imputed-model", default=None,
                    help="model tag under Data/Pipeline/06_hand_imputation/Release/ whose fill is listed")
    ap.add_argument("--no-takes", action="store_true",
                    help="leave out the take folders, takes.csv and pieces.csv")
    args = ap.parse_args()
    only: dict[str, set[str]] = {}
    for spec in args.optional_in:
        mod, _, dss = spec.partition("=")
        if mod not in OPTIONAL:
            sys.exit(f"--optional-in: {mod!r} is not an optional modality ({sorted(OPTIONAL)})")
        only[mod] = {d for d in dss.split(",") if d}
    if not re.match(r"^\d+\.\d+\.\d+$", args.version):
        sys.exit("version must be MAJOR.MINOR.PATCH")

    perf, summary = [], {"version": args.version, "datasets": {}}
    for ds in sorted(args.dataset, key=SESSION_ORDER.index):
        left_out: dict[str, int] = {}
        if ds in TAKE_LEVEL:
            raw = {t["trial_stem"]: t["raw_stem"] for t in read_json(PIECES / ds / "_pieces.json", [])}
            rows, level = take_rows(ds), "take"
        else:
            rows, level = piece_rows(ds, left_out), "piece"
        for r in rows:
            pid = participant_id(ds, r["session"])
            if pid is None:
                sys.exit(f"no participant id (out of release scope?): {ds} {r['session']}")
            task, piece = classify(ds, r["task"], r["name"])
            perf.append({**r, "pid": pid, "cat": task, "piece_id": piece, "level": level,
                         "src": files_for_take(r, raw) if level == "take" else files_for_piece(r)})
        summary["datasets"][ds] = {"level": level, "released": len(rows), "left_out": left_out}

    # r1, r2, ...: a participant's performances of a piece in play order (free takes: take01, ...)
    perf.sort(key=lambda r: (r["pid"], r["cat"], r["piece_id"], SESSION_ORDER.index(r["dataset"]), r["order"],
                             r["window_sec"][0] if r["window_sec"] else 0.0))
    index = None
    if not args.no_takes:
        index = read_json(TRIALS / "_trials.json", None)
        if index is None:
            sys.exit(f"{TRIALS / '_trials.json'} missing: run pipeline/12_release/build_trials.py first (or --no-takes)")
    count: dict[tuple, int] = {}
    manifest, table, missing = [], [], {}
    best, performance_of = {}, {}            # piece stem / take stem -> the folder of its performance
    for r in perf:
        k = (r["pid"], r["cat"], r["piece_id"])
        count[k] = count.get(k, 0) + 1
        leaf = f"take{count[k]:02d}" if r["level"] == "take" else f"{r['piece_id']}/r{count[k]}"
        base = key = f"data/P{r['pid']:02d}/{r['cat']}/{leaf}"   # key: the path Stage 06 files are stored by
        if r["level"] == "take" and index is not None:
            # a free-repertoire take is released once, as its take folder (muted, on the trial clock)
            base = take_path(r["pid"], r["dataset"], r["session"], r["piece"])
            t = index.get(r["piece"], {})
            r["src"] = {**r["src"], **{m: TRIALS / r["dataset"] / r["piece"] / MODALITIES[m]
                                       for m in ("audio", "midi", "hand", "body")}}
            if not t:
                sys.exit(f"{r['piece']}: not in {TRIALS / '_trials.json'}; run build_trials.py")
        (performance_of if r["level"] == "take" else best)[r["piece"]] = base
        r["src"] = {**r["src"], "hand_suspect_mask": SUSPECT / key / "hand_suspect_mask.csv"}
        if args.imputed_model:
            fill = IMPUTED / args.imputed_model / key
            r["src"] = {**r["src"], "hand_imputed": fill / "hand_imputed.csv",
                        "hand_imputed_mask": fill / "hand_imputed_mask.csv",
                        "fingering_imputed": FINGERING_IMPUTED / args.imputed_model / r["dataset"] / f"{r['piece']}.csv"}
        have = []
        for m, f in r["src"].items():
            if m in only and r["dataset"] not in only[m]:
                continue
            if f.exists() and (m != "top" or video_released(f)):
                manifest.append({"release_path": f"{base}/{MODALITIES[m]}", "source_path": f.as_posix(),
                                 "bytes": f.stat().st_size})
                have.append(m)
            elif m not in OPTIONAL:
                missing.setdefault(r["dataset"], []).append(f"{r['piece']}:{m}")
        onset, release = onset_release(r["src"]["midi"]) if r["src"]["midi"].exists() else (None, None)
        table.append({"path": base, "participant_id": r["pid"], "group": participant_role(r["pid"]),
                      "task": r["cat"], "piece_id": r["piece_id"] or None,
                      "round": count[k] if r["level"] == "piece" else None,
                      "session": r["dataset"], "session_subject": r["session"], "source_take": r["trial"],
                      "take_start_sec": r["window_sec"][0] if r["window_sec"] else None,
                      "take_end_sec": r["window_sec"][1] if r["window_sec"] else None,
                      "onset_sec": onset, "release_sec": release,
                      "confirmed_by": r["confirmed_by"], "modalities": " ".join(have)})
    for ds, v in summary["datasets"].items():
        v["missing_files"] = len(missing.get(ds, []))
        v["missing_examples"] = missing.get(ds, [])[:20]
        print(f"{ds}: {v['released']} {v['level']}s, left out {v['left_out']}, missing files {v['missing_files']}")

    participants: dict[int, dict] = {}
    for r in table:
        p = participants.setdefault(r["participant_id"], {"participant_id": r["participant_id"],
                                                          "group": r["group"], "sessions": set()})
        p["sessions"].add(r["session"])
    titles: dict[tuple, str] = {}
    for r in perf:
        if r["piece_id"]:
            titles.setdefault((r["cat"], r["piece_id"]), r["name"])

    takes, pieces, take_totals = [], [], {}
    if not args.no_takes:
        dss = sorted(args.dataset, key=SESSION_ORDER.index)
        take_manifest, takes, take_totals = take_rows_full(dss, index, performance_of)
        manifest += take_manifest
        pieces = pieces_table(dss, index, best, performance_of)
        for r in takes:                                # participants released through their takes only
            p = participants.setdefault(r["participant_id"], {"participant_id": r["participant_id"],
                                                              "group": r["group"], "sessions": set()})
            p["sessions"].add(r["session"])

    out = OUT / f"v{args.version}"
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "manifest.csv": manifest,
        "performances.csv": table,
        "participants.csv": [{**p, "sessions": " ".join(sorted(p["sessions"], key=SESSION_ORDER.index))}
                             for _, p in sorted(participants.items())],
        "repertoire.csv": [{"piece_id": pc, "task": cat, "title": title} for (cat, pc), title in sorted(titles.items())],
    }
    if not args.no_takes:
        tables["takes.csv"] = takes
        tables["pieces.csv"] = pieces
    for name, rows in tables.items():
        if not rows:                               # e.g. no piece ids when only take-level data is listed
            (out / name).write_text("", encoding="utf-8")
            continue
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if args.no_takes:
        (out / "pieces.csv").unlink(missing_ok=True)
    else:
        summary["takes"] = {"count": len(takes),
                            "audio_hours": round(sum(v["audio_hours"] for v in take_totals.values()), 3),
                            "motion_hours": round(sum(v["motion_hours"] for v in take_totals.values()), 3),
                            "muted_hours": round(sum(v["muted_hours"] for v in take_totals.values()), 3),
                            "midi_notes": sum(v["midi_notes"] for v in take_totals.values()),
                            "participants": len({t["participant_id"] for t in takes}),
                            "datasets": take_totals}
        summary["piece_windows"] = {"count": len(pieces),
                                    "by_status": {k: sum(r["status"] == k for r in pieces)
                                                  for k in sorted({r["status"] for r in pieces})},
                                    "best": sum(bool(r["best"]) for r in pieces)}
        print(f"takes: {summary['takes']['count']}, {summary['takes']['audio_hours']:.2f} h audio, "
              f"{summary['takes']['motion_hours']:.2f} h motion, {summary['takes']['midi_notes']} notes; "
              f"{len(pieces)} piece windows {summary['piece_windows']['by_status']}")
    summary["performances"] = len(table)
    summary["files"] = len(manifest)
    summary["bytes"] = sum(r["bytes"] for r in manifest)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{len(table)} performances, {len(manifest)} files, {summary['bytes'] / 1e9:.1f} GB -> {out}")


if __name__ == "__main__":
    main()
