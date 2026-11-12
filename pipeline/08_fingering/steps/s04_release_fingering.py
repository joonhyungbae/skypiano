"""
Stage 08 Step 04: per-note fingering for the performances of a release.

For every performance in Data/Release/v{version}/performances.csv, read its
MIDI and hand motion where the release manifest takes them from (nothing is
copied), and label every MIDI note with a hand and finger in three tiers:

  A  geometric scoring of the measured fingertips over the note's duration
     (fingering_mocap.FingeringMocap, Pass 1);
  B  z-onset refinement of what Tier A left multi- or no-candidate
     (Passes 2 and 3: the lowest fingertip at onset, with a clear z gap);
  C  the BiLSTM imputer (finger_imputer.FingerImputer) for the residual;
     on a multi-candidate note it chooses only among the geometric candidates.

Tier C fills every note, so each note carries a finger; the ambiguity flag
keeps what Tiers A/B could not settle visible.

The keyboard frame comes from the auxiliary end markers where a session has
them (2024) and from the primary ones otherwise (220819; key rectangles
shifted by PRIMARY_END_W_OFFSET_MM). A piece without any end-marker sample
takes its session's primary end-marker positions (session_ends).

The hand source is a switch, so the same code labels the raw (flagged) motion
now and the imputed motion later:
    --hand raw       the release file hand.csv
    --hand imputed   the release file hand_imputed.csv (HAND_FILES)
Both are resolved through the version's manifest.csv, never from a fixed
directory, unless --hand-root names a folder that holds them by release path
(<hand-root>/<release path>/hand_imputed.csv), for a hand source the manifest
does not list yet. --tag keeps the output of each imputation model apart
(Notes/imputed/<tag>/...).

Output, one CSV per performance (keyed by the source piece stem, as
pipeline/12_release/build_manifest.py keys its sources):
    Data/Pipeline/08_fingering/Notes/{hand}/{dataset}/{piece stem}.csv
        note_idx    0.. in the order below
        onset       seconds, time 0 = the start of that performance's midi.mid
        offset      seconds
        pitch       MIDI pitch
        velocity    MIDI velocity
        hand        L | R
        finger      1 (thumb) .. 5 (little)
        tier        A | B | C
        ambiguity   empty when Tier A or B committed; multi_candidate or
                    no_candidate when Tier C chose the label
        candidates  geometric candidates (e.g. R2|R3), lowest z or highest
                    score first; empty when there were none
        confidence  Tier A: duration score ratio of the chosen finger;
                    Tier B: empty (decided on z at onset);
                    Tier C: model probability of the chosen finger
Notes are every note of every instrument in midi.mid, sorted by onset, then
pitch. Coverage per session (share of notes per tier, flagged) goes to
    Data/Pipeline/08_fingering/Notes/{hand}/_coverage.csv

Usage (from the repository root):
  python pipeline/08_fingering/steps/s04_release_fingering.py --version 0.1.0 --hand raw
         [--dataset 240828 ...] [--limit N] [--overwrite] [--workers N]
         [--hand-root DIR] [--tag NAME]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from finger_imputer import MASK_TOKEN, FingerImputer, decode, encode  # noqa: E402
from fingering_mocap import FingeringMocap  # noqa: E402

RELEASE = Path("Data/Release")
OUT_ROOT = Path("Data/Pipeline/08_fingering/Notes")
CHECKPOINT = Path("Data/Pipeline/08_fingering/models/tier_c_bilstm.pt")
HAND_FILES = {"raw": "hand.csv", "imputed": "hand_imputed.csv"}
TAKE_LEVEL = {"240523"}          # released by whole take: keyed by the take stem

TIER_B_PATHS = {"z_discrim_multi", "z_rescue_single", "z_rescue_with_gap"}
COLUMNS = ["note_idx", "onset", "offset", "pitch", "velocity", "hand", "finger",
           "tier", "ambiguity", "candidates", "confidence"]


def piece_stem(perf: dict, midi_src: str) -> str:
    """The source stem build_manifest.py keys a performance by."""
    return perf["source_take"] if perf["session"] in TAKE_LEVEL else Path(midi_src).stem


def output_root(hand: str, tag: str | None) -> Path:
    return OUT_ROOT / hand / tag if tag else OUT_ROOT / hand


def output_path(root: Path, dataset: str, stem: str) -> Path:
    return root / dataset / f"{stem}.csv"


def velocities(midi_path: str) -> dict[tuple, list[int]]:
    """(onset, offset, pitch) -> velocities, to attach velocity to Tier A/B notes."""
    out: dict[tuple, list[int]] = {}
    for inst in pretty_midi.PrettyMIDI(midi_path).instruments:
        for n in inst.notes:
            out.setdefault((float(n.start), float(n.end), int(n.pitch)), []).append(int(n.velocity))
    return out


def session_ends(hand_paths: list[str]) -> tuple | None:
    """Primary end-marker (LEnd, REnd) positions of a session: the median over
    its pieces that have them. Used for a piece whose end markers are missing
    throughout (the piano does not move within a session)."""
    meds = []
    for p in hand_paths:
        d = pd.read_csv(p, usecols=[f"{m}_{a}" for m in ("LEnd", "REnd") for a in "xyz"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmedian(d.values, axis=0)
        if not np.isnan(m).any():
            meds.append(m)
        if len(meds) >= 5:
            break
    if not meds:
        return None
    m = np.median(meds, axis=0)
    return m[:3], m[3:]


def tiers_ab(hand_path: str, midi_path: str, fallback_ends: tuple | None = None) -> list[dict]:
    """Tiers A and B on one performance; notes sorted by onset, then pitch."""
    res = FingeringMocap(hand_path, midi_path, fps=120, fallback_ends=fallback_ends)
    res.load()
    notes = res.run()["fingering"]
    vel = velocities(midi_path)
    notes.sort(key=lambda n: (n["onset"], n["pitch"]))
    for n in notes:
        n["velocity"] = vel[(n["onset"], n["offset"], n["pitch"])].pop(0)
        n["cand_tokens"] = [encode(c["hand"], c["finger"]) for c in n["candidates"]]
        if n["status"] == "labeled":
            n["tier"] = "B" if n["decision_path"] in TIER_B_PATHS else "A"
    return notes


def label_text(token: int) -> str:
    hand, finger = decode(token)
    return f"{hand[0]}{finger}"


def finish(notes: list[dict], imputer: FingerImputer) -> list[dict]:
    """Tier C on what Tiers A/B left; returns the output rows."""
    seq = [{"onset": n["onset"], "offset": n["offset"], "pitch": n["pitch"],
            "token": encode(n["hand"], n["finger"]) if n["status"] == "labeled" else MASK_TOKEN,
            "candidates": n["cand_tokens"] if n["status"] == "multi_candidate" else []}
           for n in notes]
    filled = imputer.predict(seq)
    rows = []
    for i, (n, c) in enumerate(zip(notes, filled)):
        if c is None:
            token, tier, flag = encode(n["hand"], n["finger"]), n["tier"], ""
            conf = f"{n['score']:.4f}" if tier == "A" and n["score"] is not None else ""
        else:
            token, tier, flag, conf = c[0], "C", n["status"], f"{c[1]:.4f}"
        hand, finger = decode(token)
        rows.append({"note_idx": i, "onset": f"{n['onset']:.6f}", "offset": f"{n['offset']:.6f}",
                     "pitch": n["pitch"], "velocity": n["velocity"], "hand": hand[0], "finger": finger,
                     "tier": tier, "ambiguity": flag,
                     "candidates": "|".join(label_text(t) for t in n["cand_tokens"] if t),
                     "confidence": conf})
    return rows


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def coverage(jobs: list[dict]) -> list[dict]:
    """Share of notes per tier and flagged, per session (dataset x subject)."""
    agg: dict[tuple, dict] = {}
    for j in jobs:
        if not j["out"].exists():
            continue
        a = agg.setdefault((j["dataset"], j["subject"]), {"performances": 0, "notes": 0, "A": 0, "B": 0,
                                                          "C": 0, "multi_candidate": 0, "no_candidate": 0})
        a["performances"] += 1
        for r in read_rows(j["out"]):
            a["notes"] += 1
            a[r["tier"]] += 1
            if r["ambiguity"]:
                a[r["ambiguity"]] += 1
    out = []
    for (ds, subj), a in sorted(agg.items()):
        n = max(1, a["notes"])
        out.append({"dataset": ds, "session_subject": subj, "performances": a["performances"],
                    "notes": a["notes"], "tier_a": a["A"], "tier_b": a["B"], "tier_c": a["C"],
                    "multi_candidate": a["multi_candidate"], "no_candidate": a["no_candidate"],
                    "share_a": round(a["A"] / n, 4), "share_b": round(a["B"] / n, 4),
                    "share_c": round(a["C"] / n, 4),
                    "share_flagged": round((a["multi_candidate"] + a["no_candidate"]) / n, 4)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="release version whose performances are labeled")
    ap.add_argument("--hand", choices=sorted(HAND_FILES), required=True, help="hand motion source")
    ap.add_argument("--dataset", nargs="*", help="only these datasets")
    ap.add_argument("--limit", type=int, help="only the first N performances (testing)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="processes for Tiers A/B")
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT, help="Tier C weights")
    ap.add_argument("--hand-root", type=Path,
                    help="read <hand-root>/<release path>/<hand file> instead of the manifest's source")
    ap.add_argument("--tag", help="output subfolder, e.g. the imputation model (Notes/<hand>/<tag>/)")
    args = ap.parse_args()
    out_root = output_root(args.hand, args.tag)

    rel = RELEASE / f"v{args.version}"
    with open(rel / "manifest.csv", newline="", encoding="utf-8") as f:
        src = {r["release_path"]: r["source_path"] for r in csv.DictReader(f)}
    with open(rel / "performances.csv", newline="", encoding="utf-8") as f:
        perfs = list(csv.DictReader(f))

    jobs, skipped = [], {}
    for p in perfs:
        if args.dataset and p["session"] not in args.dataset:
            continue
        midi = src.get(f"{p['path']}/midi.mid")
        if args.hand_root:
            hand = args.hand_root / p["path"] / HAND_FILES[args.hand]
            hand = hand.as_posix() if hand.exists() else None
        else:
            hand = src.get(f"{p['path']}/{HAND_FILES[args.hand]}")
        if not midi or not hand:
            skipped["no_midi" if not midi else f"no_{HAND_FILES[args.hand]}"] = \
                skipped.get("no_midi" if not midi else f"no_{HAND_FILES[args.hand]}", 0) + 1
            continue
        stem = piece_stem(p, midi)
        jobs.append({"path": p["path"], "dataset": p["session"], "subject": p["session_subject"],
                     "midi": midi, "hand": hand, "out": output_path(out_root, p["session"], stem)})
    if args.limit:
        jobs = jobs[:args.limit]
    todo = [j for j in jobs if args.overwrite or not j["out"].exists()]
    print(f"v{args.version}, hand={args.hand}: {len(jobs)} performances, {len(todo)} to label"
          + (f", skipped {skipped}" if skipped else ""), flush=True)

    by_session: dict[tuple, list[str]] = {}
    for j in jobs:
        by_session.setdefault((j["dataset"], j["subject"]), []).append(j["hand"])
    ends = {k: session_ends(v) for k, v in by_session.items()
            if any((j["dataset"], j["subject"]) == k for j in todo)}

    imputer = FingerImputer(args.checkpoint) if todo else None
    failed, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(tiers_ab, j["hand"], j["midi"], ends.get((j["dataset"], j["subject"]))): j
                for j in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            j = futs[fut]
            try:
                write_csv(j["out"], finish(fut.result(), imputer), COLUMNS)
            except Exception as e:  # keep going; report at the end
                failed.append((j["path"], repr(e)))
            if k % 25 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)}  {time.time() - t0:.0f} s", flush=True)

    cov = coverage(jobs)
    if cov:
        write_csv(out_root / "_coverage.csv", cov, list(cov[0].keys()))
        tot = {k: sum(c[k] for c in cov) for k in ("performances", "notes", "tier_a", "tier_b", "tier_c",
                                                    "multi_candidate", "no_candidate")}
        n = max(1, tot["notes"])
        print(f"{tot['performances']} performances, {tot['notes']} notes: "
              f"A {tot['tier_a'] / n:.1%}, B {tot['tier_b'] / n:.1%}, C {tot['tier_c'] / n:.1%} "
              f"(multi {tot['multi_candidate']}, none {tot['no_candidate']})")
    if failed:
        print(f"{len(failed)} failed:")
        for path, err in failed:
            print(f"  {path}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
