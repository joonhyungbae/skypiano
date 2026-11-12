"""
Stage 05 Alignment QA: the record of the reviewer's confirmations -- the gate
between alignment (01-04) and post-processing (06-11).

Nothing here estimates whether an alignment is right; that is judged by eye
on the review pages. This stage gathers, per dataset:

  trials  sync source, the source's signal level (`weak` = look first), and
          the reviewer's sync verdict (Metadata/{ds}/_alignment_review.json
          "sync"; a verdict goes stale when the offset it judged changed);
  pieces  window, exact window findings (overlap with a neighbouring piece,
          overlap with a session-notes discard interval, a window ending
          < 0.2 s after its last note), name and where it comes from, and the
          reviewer's boundary verdict ("reviews").

A trial is confirmed by a "yes" sync verdict; a piece by confirmed
boundaries and a name from the session log or a manual label. Boundaries are
confirmed by a "yes" verdict, or -- unless the reviewer said "no" -- by
content (_common.piece_content_match: the cut holds exactly its piece as the
other performers of that piece played it), or by the session sample: the
review page asks three randomly chosen pieces of a session that raised no
doubt, and when all three are right it confirms the session's other such
pieces with a "yes" verdict noted "session_sample" (their names as manual
labels by "session_sample"), or by the score: a technique drill cut that
matches its written score note for note, first and last chord included
(_common.score_check; its name is confirmed with it). `boundary_confirmed_by`
records which (review / content_match / session_sample / score_match /
performance_match: a cut matching, note for note and first to last chord, a
confirmed performance of the same piece in any dataset; its name is confirmed
with it / not_a_piece: an unnamed cut of at most FEW_NOTES onset frames, or one
that matches inside the piece next to it -- a broken-off start or a retake --
or any cut the reviewer called not a piece; `not_a_piece` gives which / near_match: the same, but with playing slips allowed
-- mean cost <= 0.15, first and last chords sharing most notes; kept apart so it
can be reviewed as a group). A dataset passes
when every trial with hand motion and every piece is confirmed. A take-level
dataset (TAKE_LEVEL_DATASETS: free repertoire released as whole takes) passes
on its trials alone; its pieces are still listed but do not gate it. In a
retake dataset (RETAKE_DATASETS: a piece's further takes are retakes, and one
confirmed take of it is released) a piece another confirmed take of the same
subject and piece already holds is settled (`represented_by`).

Timecode sessions (2024): motion, audio and MIDI are cut at one SMPTE
timecode, so a sync error would be the session's (recorder setup), not a
trial's. Each session has sample trials -- its longest and its shortest of
at least SAMPLE_MIN_SPAN_S of playing -- and the other timecode trials are
confirmed when every sample is "yes" and no trial of the session is "no"
(a trial's own verdict still counts first). Trials placed without timecode
(`_timecode_note` in the Stage 01 mapping) are reviewed one by one.

Outputs Data/Pipeline/05_alignment_qa/{ds}/_status.json (+ _summary.json).

Usage (from the repository root):
  python pipeline/05_alignment_qa/run_pipeline.py [--dataset 240828 ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pipeline/
from _common.piece_content_match import content_matches  # noqa: E402
from _common.score_check import cut_frames, near_match, performance_match, score_for, score_match  # noqa: E402
from _common.score_follow import match_cost  # noqa: E402

PIPE = Path("Data/Pipeline")
NORM = PIPE / "01_ingest" / "Normalized"
PIECES = PIPE / "04_piece_split"
REVIEW_INPUTS = PIPE / "03_sync" / "midi_motion" / "ReviewInputs"
OUT = PIPE / "05_alignment_qa"
META = Path("Data/Metadata")
LOCATE_220819 = PIECES / "anchors_220819" / "locate_manifest.json"
DATASETS = ["220819", "221112", "240523", "240828"]   # release datasets (211026, 211215: C0_N0 only)
# Out of release scope. C0_N0 is one professional across 211026, 211215, 220819
# and 240809, and none of that person's sessions is released.
EXCLUDED_SESSIONS = {("211026", "C0_N0"), ("211215", "C0_N0"), ("220819", "C0_N0"), ("240809", "C0_N0")}
TIGHT_TAIL_S = 0.2
TIMECODE_DATASETS = {"240523", "240828"}
TAKE_LEVEL_DATASETS = {"240523"}     # the amateur free-repertoire session ships at the take level
RETAKE_DATASETS = {"221112"}         # a piece's further takes are retakes: one confirmed take releases it
SAMPLE_MIN_SPAN_S = 10.0


def read_json(p: Path, default):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def piece_windows(ds: str) -> dict[str, list[dict]]:
    """trial -> [{piece, piece_idx, start, end, name, name_source}]"""
    out: dict[str, list[dict]] = {}
    if ds == "220819":
        for e in read_json(LOCATE_220819, {}).get("entries", []):
            if e.get("ok") and e.get("out_stem"):
                out.setdefault(e["stage1_stem"], []).append({
                    "piece": f"p{e['p_slot']}", "piece_idx": e["p_slot"], "start": e["t_start_sec"],
                    "end": e["t_end_sec"], "name": e.get("piece_name") or e.get("piece_id"),
                    "name_source": "session_log"})
        return out
    # Windows come from _pieces.json; s04's arrangement resolution (names only)
    # from its sidecar, which may predate later window edits.
    resolved = {(t["trial_stem"], p["piece_idx"]): p.get("identified") or {}
                for t in read_json(PIECES / ds / "_pieces_resolved.json", []) for p in t.get("pieces", [])}
    for t in read_json(PIECES / ds / "_pieces.json", []):
        for p in t.get("pieces", []):
            ident = p.get("identified") or {}
            if not ident.get("resolved_piece_id"):
                ident = resolved.get((t["trial_stem"], p["piece_idx"])) or ident
            src = ident.get("resolution_source")
            out.setdefault(t["trial_stem"], []).append({
                "piece": f"p{p['piece_idx']}", "piece_idx": p["piece_idx"], "start": p["start_sec"],
                "end": p["end_sec"], "name": ident.get("resolved_piece_id"),
                "name_source": {"manual_review": "manual", "session_log": "session_log",
                                "program_order": "program_order"}.get(src,
                               "auto" if ident.get("resolved_piece_id") else None)})
    return out


def findings(ds: str, trial: dict, pieces: list[dict]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {p["piece"]: [] for p in pieces}
    ps = sorted(pieces, key=lambda p: p["start"])
    for a, b in zip(ps, ps[1:]):
        ov = a["end"] - b["start"]
        if ov > 0.05:
            out[a["piece"]].append(f"overlaps {b['piece']} by {ov:.2f}s")
            out[b["piece"]].append(f"overlaps {a['piece']} by {ov:.2f}s")
    s0 = trial.get("session_start_sec") or 0.0
    for x0, y0 in trial.get("exclusion_intervals") or []:
        for p in ps:
            ov = min(p["end"], y0 - s0) - max(p["start"], x0 - s0)
            if ov > 0.05:
                out[p["piece"]].append(f"overlaps a discard interval by {ov:.2f}s")
    for p in ps:
        mid = PIECES / ds / f"{trial['trial_stem']}_{p['piece']}.mid"
        if mid.exists():
            pm = pretty_midi.PrettyMIDI(str(mid))
            last = max((n.end for i in pm.instruments for n in i.notes), default=None)
            if last is not None and (p["end"] - p["start"]) - last < TIGHT_TAIL_S:
                out[p["piece"]].append("ends while the last note sounds")
    return out


def confirm_timecode_sessions(trials: list[dict]) -> None:
    """Pick each timecode session's sample trials and confirm the others from them (see module docstring)."""
    sessions: dict[str, list[dict]] = {}
    for t in trials:
        if t["sync_reviewable"] and t["sync_basis"] == "timecode":
            sessions.setdefault(t["session"], []).append(t)
    for ts in sessions.values():
        long_enough = sorted((t for t in ts if t["_span"] >= SAMPLE_MIN_SPAN_S), key=lambda t: t["_span"])
        samples = {id(t) for t in (long_enough[:1] + long_enough[-1:])}
        for t in ts:
            t["sample"] = id(t) in samples
        ok = bool(samples) and all(t["sync_verdict"] == "yes" for t in ts if t["sample"])             and not any(t["sync_verdict"] == "no" for t in ts)
        for t in ts:
            if t["sync_verdict"] is not None:
                t["confirmed_by"] = "own"
            elif ok:
                t["confirmed"], t["confirmed_by"] = True, "session_sample"
    for t in trials:
        t.pop("_span", None)


PERF_REFS_MAX = 12          # confirmed performances a cut is compared with, per piece name
FEW_NOTES = 5               # an unnamed cut of this many notes or fewer is not a piece
PART_COST = 0.15            # an unnamed cut matching inside a neighbour's piece this well is part of it
CONFIRMED_BY = ("review", "content_match", "score_match", "performance_match", "session_sample")


def confirmed_performances(ds_now: str, pieces_now: list[dict]) -> dict[str, list]:
    """name -> onset frames of confirmed cuts, from every dataset's last status (this one's
    from this run)."""
    pool: dict[str, list] = {}
    for d in DATASETS:
        rows = pieces_now if d == ds_now else read_json(OUT / d / "_status.json", {}).get("pieces", [])
        for p in rows:
            if p.get("name") and p.get("boundary_confirmed_by") in CONFIRMED_BY and len(pool.get(p["name"], [])) < PERF_REFS_MAX:
                midi = PIECES / d / f"{p['piece']}.mid"
                if midi.exists():
                    pool.setdefault(p["name"], []).append(cut_frames(midi))
    return pool


def run(ds: str) -> dict:
    review = read_json(META / ds / "_alignment_review.json", {})
    sync, verdicts = review.get("sync", {}), review.get("reviews", {})
    manual = read_json(PIECES / ds / "_pieces_manual.json", {})
    wins = piece_windows(ds)
    trials_out, pieces_out = [], []
    for mp in sorted((NORM / ds).glob("*/_mapping.json")):
        sess = mp.parent.name
        if (ds, sess) in EXCLUDED_SESSIONS:
            continue
        for t in read_json(mp, {})["trials"]:
            stem = t["trial_stem"]
            ri = read_json(REVIEW_INPUTS / ds / f"{stem}.json", {})
            sv = sync.get(stem) or {}
            stale = bool(sv and ri.get("shift_sec") is not None and sv.get("check_shift_sec") is not None
                         and abs(sv["check_shift_sec"] - ri["shift_sec"]) > 1e-6)
            reviewable = ri.get("status") == "ok"
            verdict = None if stale else sv.get("verdict")
            basis = (ri.get("signal") or {}).get("basis")
            if ds in TIMECODE_DATASETS and t.get("_timecode_note"):
                basis = "motion-MIDI correlation (no timecode)"
            span = (ri["played_sec"][1] - ri["played_sec"][0]) if ri.get("played_sec") else 0.0
            trials_out.append({
                "trial": stem, "session": sess, "sync_reviewable": reviewable,
                "signal": (ri.get("signal") or {}).get("level"), "sync_basis": basis, "sync_verdict": verdict,
                "stale": stale, "confirmed": (verdict == "yes") if reviewable else None,
                "_span": span})
            ps = wins.get(stem, [])
            found = findings(ds, t, ps)
            for p in ps:
                ps_stem = f"{stem}_{p['piece']}"
                m = (manual.get(stem) or {}).get(str(p["piece_idx"])) or {}
                # a name confirmed through the session sample is kept apart from one the reviewer gave
                msrc = "session_sample" if m.get("reviewer") == "session_sample" else "manual"
                name, src = (m["piece_id"], msrc) if m.get("piece_id") else (p["name"], p["name_source"])
                v = verdicts.get(ps_stem) or {}
                b = v.get("verdict")
                pieces_out.append({
                    "piece": ps_stem, "window_sec": [p["start"], p["end"]], "findings": found[p["piece"]],
                    "name": name, "name_source": src, "name_confirmed": src in ("manual", "session_log", "session_sample"),
                    "boundary_verdict": b, "_by_sample": b == "yes" and v.get("note") == "session_sample",
                    "_not_a_piece": b == "no" and v.get("note") == "not_a_piece"})
    matches = content_matches([(p["piece"], p["name"], PIECES / ds / f"{p['piece']}.mid") for p in pieces_out])
    scored = {p["piece"]: score_match(PIECES / ds / f"{p['piece']}.mid", p["name"]) for p in pieces_out}
    for p in pieces_out:
        cm = matches.get(p["piece"]) or {}
        # an octave repeat is one chord over and over: a cut missing part of the run still
        # matches the others' chords, so its content match does not confirm the boundaries
        if (p["name"] or "").startswith("Octave Repeat"):
            cm = {**cm, "match": False}
        p["content_match"] = cm.get("match")
        p["content_score"] = cm.get("score")
        sm = scored.get(p["piece"]) or {}
        p["score_match"] = sm.get("match")
        p["score_cost"] = sm.get("cost")
        verdict = p["boundary_verdict"]
        by_sample = p.pop("_by_sample")
        if p.pop("_not_a_piece"):
            # the reviewer said the cut is not a piece (a retake, a broken-off start): settled,
            # left out of the release as a piece
            p["boundary_confirmed_by"], p["not_a_piece"], p["confirmed"] = "not_a_piece", "review", True
            continue
        if verdict == "yes":
            by = "session_sample" if by_sample else "review"
        elif verdict == "no":
            by = None
        else:
            by = "content_match" if cm.get("match") else "score_match" if sm.get("match") else None
        p["boundary_confirmed_by"] = by
        if sm.get("match") and verdict != "no":
            p["name_confirmed"] = True          # the notes are that drill's score
        p["confirmed"] = by is not None and p["name_confirmed"]
    # cuts no exact check has settled yet: against confirmed performances of the same piece
    pool = confirmed_performances(ds, pieces_out)
    for p in pieces_out:
        if p["boundary_confirmed_by"] or p["boundary_verdict"] == "no":
            continue
        pm = performance_match(PIECES / ds / f"{p['piece']}.mid", p["name"], pool.get(p["name"], [])) or {}
        p["performance_cost"] = pm.get("cost")
        if pm.get("match"):
            p["boundary_confirmed_by"] = "performance_match"
            p["name_confirmed"] = True          # the notes are a confirmed performance of that piece
            p["confirmed"] = True
            continue
        nm = near_match(PIECES / ds / f"{p['piece']}.mid", p["name"], pool.get(p["name"], [])) or {}
        p["near_cost"] = nm.get("cost")
        if nm.get("match"):
            p["boundary_confirmed_by"] = "near_match"      # the right stretch, played with slips
            p["name_confirmed"] = True
            p["confirmed"] = True
    # unnamed cuts that are not pieces: a handful of notes, or a stretch of the piece next to
    # them played again (a broken-off start, a retake)
    by_trial: dict[str, list[dict]] = {}
    for p in pieces_out:
        by_trial.setdefault(p["piece"].rsplit("_p", 1)[0], []).append(p)
    for ps in by_trial.values():
        ps.sort(key=lambda q: q["window_sec"][0])
        for i, p in enumerate(ps):
            if p["name"] or p["boundary_confirmed_by"] or p["boundary_verdict"]:
                continue
            midi = PIECES / ds / f"{p['piece']}.mid"
            F = cut_frames(midi) if midi.exists() else None
            if F is None:
                continue
            reason = "few_notes" if len(F) <= FEW_NOTES else None
            if reason is None:
                for q in ps[max(0, i - 2):i + 3]:
                    R = score_for(q["name"]) if q["name"] else None
                    if R is None and q["name"] in pool:
                        R = pool[q["name"]][0]
                    if R is not None and len(R) >= len(F) / 2 and float(match_cost(F, R)[0].min()) <= PART_COST:
                        reason = f"part_of:{q['name']}"
                        break
            if reason:
                p["boundary_confirmed_by"] = "not_a_piece"
                p["not_a_piece"] = reason
                p["confirmed"] = True               # settled: left out of the release as a piece
    # a retake of a piece another confirmed take already holds needs no answer of its own
    if ds in RETAKE_DATASETS:
        held: dict[tuple, str] = {}
        for p in pieces_out:
            if p["confirmed"] and p["name"] and p["boundary_confirmed_by"] != "not_a_piece":
                held.setdefault((p["piece"].split("_")[2], p["name"]), p["piece"])
        for p in pieces_out:
            k = (p["piece"].split("_")[2], p["name"])
            if not p["confirmed"] and p["name"] and k in held:
                p["represented_by"] = held[k]
    confirm_timecode_sessions(trials_out)
    rv = [t for t in trials_out if t["sync_reviewable"]]
    summary = {
        "dataset": ds,
        "trials_reviewable": len(rv),
        "trials_confirmed": sum(t["confirmed"] for t in rv),
        "trials_weak_signal": sum(t["signal"] == "weak" for t in rv),
        "trials_rejected": sum(t["sync_verdict"] == "no" for t in rv),
        "trials_stale": sum(t["stale"] for t in rv),
        "trials_sample": sum(bool(t.get("sample")) for t in rv),
        "trials_confirmed_by_session": sum(t.get("confirmed_by") == "session_sample" for t in rv),
        "pieces": len(pieces_out),
        "pieces_confirmed": sum(p["confirmed"] for p in pieces_out),
        "pieces_boundary_by_content": sum(p["boundary_confirmed_by"] == "content_match" for p in pieces_out),
        "pieces_boundary_by_review": sum(p["boundary_confirmed_by"] == "review" for p in pieces_out),
        "pieces_boundary_by_session_sample": sum(p["boundary_confirmed_by"] == "session_sample" for p in pieces_out),
        "pieces_boundary_by_score": sum(p["boundary_confirmed_by"] == "score_match" for p in pieces_out),
        "pieces_boundary_by_performance": sum(p["boundary_confirmed_by"] == "performance_match" for p in pieces_out),
        "pieces_boundary_by_near_match": sum(p["boundary_confirmed_by"] == "near_match" for p in pieces_out),
        "pieces_not_a_piece": sum(p["boundary_confirmed_by"] == "not_a_piece" for p in pieces_out),
        "pieces_rejected": sum(p["boundary_verdict"] == "no" for p in pieces_out),
        "pieces_with_findings": sum(bool(p["findings"]) for p in pieces_out),
        "pieces_name_unconfirmed": sum(not p["name_confirmed"] for p in pieces_out),
        "pieces_represented_by_take": sum(bool(p.get("represented_by")) for p in pieces_out),
    }
    summary["take_level"] = ds in TAKE_LEVEL_DATASETS
    summary["passed"] = (summary["trials_confirmed"] == summary["trials_reviewable"]
                         and (summary["take_level"] or summary["pieces_confirmed"]
                              + summary["pieces_represented_by_take"] == summary["pieces"]))
    (OUT / ds).mkdir(parents=True, exist_ok=True)
    (OUT / ds / "_status.json").write_text(json.dumps({"summary": summary, "trials": trials_out,
                                                       "pieces": pieces_out}, indent=1, ensure_ascii=False),
                                           encoding="utf-8")
    (OUT / ds / "_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", default=DATASETS)
    a = ap.parse_args()
    print(f"{'dataset':8s} {'sync ok':>9s} {'weak':>5s} {'no':>4s} {'pieces ok':>11s} {'findings':>9s} {'no name':>8s}  passed")
    for ds in a.dataset:
        s = run(ds)
        print(f"{ds:8s} {s['trials_confirmed']:>4d}/{s['trials_reviewable']:<4d} {s['trials_weak_signal']:>5d} "
              f"{s['trials_rejected']:>4d} {s['pieces_confirmed']:>5d}/{s['pieces']:<5d} {s['pieces_with_findings']:>9d} "
              f"{s['pieces_name_unconfirmed']:>8d}  {s['passed']}")


if __name__ == "__main__":
    main()
