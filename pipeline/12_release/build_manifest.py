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
          as a whole, as the paper states for free repertoire.

Cuts the reviewer or Stage 05 called not a piece are left out, and so is every
session out of release scope (C0_N0; paper_builders/participant_map.py).

Layout: data/P{participant id}/{task}/{piece}/r{n}/{modality file}, the
participant id (1-19) from paper_builders/participant_map.py, r1, r2, ... the
participant's performances of that piece in play order (free repertoire by
whole take: data/P{id}/free/take01, ...). The task follows the paper's
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
    summary.json       counts per dataset, and what was left out and why

Usage (from the repository root):
  python pipeline/12_release/build_manifest.py --version 0.1.0 [--dataset 240828 ...]
      [--optional-in top=240523,220819]   (an optional modality for some datasets only)
      [--imputed-model TAG]                (list hand_imputed*, fingering_imputed of that model)

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
MODALITIES["top"] = "top.mp4"
MODALITIES["fingering"] = "fingering.csv"
MODALITIES["hand_imputed"] = "hand_imputed.csv"
MODALITIES["hand_imputed_mask"] = "hand_imputed_mask.csv"
MODALITIES["fingering_imputed"] = "fingering_imputed.csv"
MODALITIES["hand_suspect_mask"] = "hand_suspect_mask.csv"
OPTIONAL = {"top", "fingering", "hand_imputed", "hand_imputed_mask", "fingering_imputed", "hand_suspect_mask"}
IMPUTED_MODALITIES = ("hand_imputed", "hand_imputed_mask", "fingering_imputed")
SESSION_ORDER = ["220819", "221112", "240523", "240828"]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--dataset", nargs="*", default=PIECE_LEVEL + TAKE_LEVEL)
    ap.add_argument("--optional-in", action="append", default=[], metavar="MODALITY=DS,DS",
                    help="list an optional modality only for these datasets (empty: none), "
                         "e.g. top=240523,220819 while the rest is still being processed")
    ap.add_argument("--imputed-model", default=None,
                    help="model tag under Data/Pipeline/06_hand_imputation/Release/ whose fill is listed")
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
    count: dict[tuple, int] = {}
    manifest, table, missing = [], [], {}
    for r in perf:
        k = (r["pid"], r["cat"], r["piece_id"])
        count[k] = count.get(k, 0) + 1
        leaf = f"take{count[k]:02d}" if r["level"] == "take" else f"{r['piece_id']}/r{count[k]}"
        base = f"data/P{r['pid']:02d}/{r['cat']}/{leaf}"
        r["src"] = {**r["src"], "hand_suspect_mask": SUSPECT / base / "hand_suspect_mask.csv"}
        if args.imputed_model:
            fill = IMPUTED / args.imputed_model / base
            r["src"] = {**r["src"], "hand_imputed": fill / "hand_imputed.csv",
                        "hand_imputed_mask": fill / "hand_imputed_mask.csv",
                        "fingering_imputed": FINGERING_IMPUTED / args.imputed_model / r["dataset"] / f"{r['piece']}.csv"}
        have = []
        for m, f in r["src"].items():
            if m in only and r["dataset"] not in only[m]:
                continue
            if f.exists():
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

    out = OUT / f"v{args.version}"
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "manifest.csv": manifest,
        "performances.csv": table,
        "participants.csv": [{**p, "sessions": " ".join(sorted(p["sessions"], key=SESSION_ORDER.index))}
                             for _, p in sorted(participants.items())],
        "repertoire.csv": [{"piece_id": pc, "task": cat, "title": title} for (cat, pc), title in sorted(titles.items())],
    }
    for name, rows in tables.items():
        if not rows:                               # e.g. no piece ids when only take-level data is listed
            (out / name).write_text("", encoding="utf-8")
            continue
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    (out / "pieces.csv").unlink(missing_ok=True)
    summary["performances"] = len(table)
    summary["files"] = len(manifest)
    summary["bytes"] = sum(r["bytes"] for r in manifest)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{len(table)} performances, {len(manifest)} files, {summary['bytes'] / 1e9:.1f} GB -> {out}")


if __name__ == "__main__":
    main()
