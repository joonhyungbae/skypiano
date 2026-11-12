"""
Stage 01 step 03: 220819 canonical trial matching (all 5 nodes)

Replaces the Processed/-dependent logic of s02/s03 for 220819. Operates
entirely from Raw.

Strategy:
  1. Per node, collect raw audio / midi / motion hand / motion body.
  2. Build a canonical trial skeleton from audio+midi:
       - piece ordinal: dense 1.. by appearance order of raw T# tokens
       - r1 / r2 MIDI entries become separate trials; chronological t#
         runs r1-pass then r2-pass within each piece.
  3. Assign motion files to canonical trials by signal cross-correlation
     (_common/motion_midi_xcorr.py), one-to-one via Hungarian:
       - hand: fingertip key-press envelope vs trial MIDI onsets
       - body: body wrist-speed envelope vs the matched hand take
               (vs MIDI onsets when the trial has no hand take)
     Trials without MIDI fall back to duration matching:
       - C0 sessions: per-piece Hungarian (motion piece_ord == canonical P)
       - C1 sessions: global Hungarian (piece ordinal mismatch)
  4. Copy raw files to canonical stems. Write enriched _mapping.json,
     carrying over per-trial `videos` from the previous mapping.

Canonical stem:  220819_C{c}_N{n}_T1_P{p}_t{t}
Canonical layout per session:
    Data/Pipeline/01_ingest/Normalized/220819/C{c}_N{n}/
        motion_hand/{stem}.csv
        motion_body/{stem}.tsv
        audio/{stem}.wav
        midi/{stem}.mid
        _mapping.json
"""
from __future__ import annotations
import functools
import re
import shutil
import sys
import wave
from pathlib import Path

import mido
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common.normalized_io import RAW_ROOT, read_mapping, session_dir, write_mapping  # noqa: E402
from _common import motion_midi_xcorr as mx  # noqa: E402
from _common.known_absences import body_absence_reason  # noqa: E402

print = functools.partial(print, flush=True)

DATASET = "220819"
NODES = ["C0/N2", "C0/N1", "C0/N0", "C1/N1", "C1/N0"]

MOTION_RE = re.compile(r"^2022Piano(\d{2})_(\d{2})_(\d{2})_(\d{2})(?:_\d{3})?$")
AUDIO_RE = re.compile(r"^220819_C(\d)_N(\d)_T([\w-]+)_t(\d+)(?:_(\d+))?$")
MIDI_RE = re.compile(r"^220819_C(\d)_N(\d)_T([\w-]+)_t(\d+)(?:_r(\d+))?$")

MOTION_BIAS = 18.0
TOLERANCE = 35.0

# Signal matching thresholds (validated on 220819 C0_N2 / C1_N1: correct
# pairs score 0.50-0.86, spurious best pairs 0.22).
HAND_MIN_NCC = 0.35
HAND_TIE_EPS = 0.02     # summed-score difference below which a swap of two takes is a tie
HAND_OVERLAP = 0.7      # overlap >= 70% of the longer of (MIDI, hand take)
BODY_MIN_NCC = 0.40
BODY_OVERLAP = 0.9      # overlap >= 90% of the body take
BODY_MIN_COVER = 0.4    # body take must span >= 40% of the reference signal

MODALITIES = ["motion_hand", "motion_body", "audio", "midi"]


# ─────────────────────────────────────────────────────────
# Duration helpers
# ─────────────────────────────────────────────────────────

def wav_dur(p: Path) -> float:
    try:
        with wave.open(str(p), "rb") as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return 0.0


def midi_dur(p: Path) -> float:
    """Musical duration = time of last note_on. Avoids trailing empty time."""
    try:
        mf = mido.MidiFile(str(p))
        tempo = 500000
        cur = 0.0
        last_note = 0.0
        for msg in mido.merge_tracks(mf.tracks):
            if msg.type == "set_tempo":
                tempo = msg.tempo
            cur += mido.tick2second(msg.time, mf.ticks_per_beat, tempo)
            if msg.type == "note_on" and msg.velocity > 0:
                last_note = cur
        return last_note if last_note > 0 else mf.length
    except Exception:
        return 0.0


def motion_dur(p: Path) -> float:
    """Hand CSV (OptiTrack) or Body TSV (Qualisys) duration from header."""
    total = rate = None
    try:
        sep = "\t" if p.suffix.lower() == ".tsv" else ","
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            head_lines = [f.readline() for _ in range(10)]
        for line in head_lines:
            parts = line.strip().split(sep)
            for i, tok in enumerate(parts):
                t = tok.strip()
                if t in ("Total Exported Frames", "Total Frames in Take", "NO_OF_FRAMES") and i + 1 < len(parts):
                    try: total = int(parts[i + 1])
                    except ValueError: pass
                if t in ("Export Frame Rate", "Capture Frame Rate", "FREQUENCY") and i + 1 < len(parts):
                    try: rate = float(parts[i + 1])
                    except ValueError: pass
        if total and rate:
            return total / rate
    except Exception:
        pass
    return 0.0


# ─────────────────────────────────────────────────────────
# Raw collection
# ─────────────────────────────────────────────────────────

def collect_audio(base: Path):
    out = []
    d = base / "Audio"
    if not d.exists():
        return out
    for f in sorted(d.iterdir()):
        if f.suffix.lower() != ".wav":
            continue
        m = AUDIO_RE.match(f.stem)
        if not m:
            continue
        out.append({
            "piece": m.group(3),
            "take": int(m.group(4)),
            "dup": m.group(5),
            "path": f,
            "dur": wav_dur(f),
        })
    return out


def collect_midi(base: Path):
    out = []
    d = base / "MIDI"
    if not d.exists():
        return out
    for f in sorted(d.iterdir()):
        if f.suffix.lower() != ".mid":
            continue
        m = MIDI_RE.match(f.stem)
        if not m:
            continue
        out.append({
            "piece": m.group(3),
            "take": int(m.group(4)),
            "r_pass": int(m.group(5)) if m.group(5) else 1,
            "path": f,
            "dur": midi_dur(f),
        })
    return out


def collect_motion(base: Path, kind: str):
    out = []
    d = base / ("Motion/Hand" if kind == "hand" else "Motion/Body")
    if not d.exists():
        return out
    exts = (".csv",) if kind == "hand" else (".csv", ".tsv")
    excluded = []
    for f in sorted(d.iterdir()):
        if f.suffix.lower() not in exts:
            continue
        if "static" in f.stem.lower():
            excluded.append(f.name)
            continue
        m = MOTION_RE.match(f.stem)
        if not m:
            continue
        out.append({
            "class_code": int(m.group(1)),
            "session_ord": int(m.group(2)),
            "piece_ord": int(m.group(3)),
            "take_ord": int(m.group(4)),
            "path": f,
            "dur": motion_dur(f),
        })
    return out, excluded


# ─────────────────────────────────────────────────────────
# Canonical skeleton + motion matching
# ─────────────────────────────────────────────────────────

def build_canonical(audio, midi):
    """Return (piece_order, trials)."""
    piece_order = []
    for a in audio:
        if a["piece"] not in piece_order:
            piece_order.append(a["piece"])
    for mi in midi:
        if mi["piece"] not in piece_order:
            piece_order.append(mi["piece"])

    trials = []
    for p_idx, piece in enumerate(piece_order, start=1):
        audio_by_take: dict[int, list] = {}
        for a in audio:
            if a["piece"] != piece:
                continue
            audio_by_take.setdefault(a["take"], []).append(a)
        m_r1 = {mi["take"]: mi for mi in midi if mi["piece"] == piece and mi["r_pass"] == 1}
        m_r2 = sorted(
            [mi for mi in midi if mi["piece"] == piece and mi["r_pass"] == 2],
            key=lambda x: x["take"],
        )

        t_counter = 0
        pass1_takes = sorted(set(audio_by_take) | set(m_r1))
        for tk in pass1_takes:
            a_entries = audio_by_take.get(tk, [])
            m1 = m_r1.get(tk)
            if a_entries:
                for dup_idx, a in enumerate(a_entries):
                    t_counter += 1
                    trials.append({
                        "canonical_p": p_idx,
                        "canonical_t": t_counter,
                        "raw_piece": piece,
                        "audio": a,
                        "midi": m1 if dup_idx == 0 else None,
                        "target_dur": a["dur"],
                        "r_note": None,
                        "dup_note": a["dup"],
                    })
            else:
                t_counter += 1
                trials.append({
                    "canonical_p": p_idx,
                    "canonical_t": t_counter,
                    "raw_piece": piece,
                    "audio": None,
                    "midi": m1,
                    "target_dur": m1["dur"] if m1 else 0.0,
                    "r_note": None,
                    "dup_note": None,
                })

        for mi in m_r2:
            t_counter += 1
            trials.append({
                "canonical_p": p_idx,
                "canonical_t": t_counter,
                "raw_piece": piece,
                "audio": None,
                "midi": mi,
                "target_dur": mi["dur"],
                "r_note": f"r2_from_t{mi['take']}",
                "dup_note": None,
            })

    return piece_order, trials


def hungarian(trials_sub, motion_sub):
    if not trials_sub or not motion_sub:
        return {}
    n_t = len(trials_sub)
    n_m = len(motion_sub)
    size = max(n_t, n_m)
    cost = np.full((size, size), 1e6, dtype=float)
    for i, tr in enumerate(trials_sub):
        for j, mo in enumerate(motion_sub):
            cost[i, j] = abs(tr["target_dur"] - mo["dur"] - MOTION_BIAS)
    row, col = linear_sum_assignment(cost)
    out = {}
    for r, c in zip(row, col):
        if r < n_t and c < n_m and cost[r, c] < TOLERANCE:
            tr = trials_sub[r]
            raw_diff = abs(tr["target_dur"] - motion_sub[c]["dur"])
            out[(tr["canonical_p"], tr["canonical_t"])] = (motion_sub[c], raw_diff)
    return out


def assign_motion(trials, motion_list, mode):
    if mode == "per_piece":
        out = {}
        by_p = {}
        for mo in motion_list:
            by_p.setdefault(mo["piece_ord"], []).append(mo)
        for p_ord, subs in by_p.items():
            piece_trials = [tr for tr in trials if tr["canonical_p"] == p_ord]
            out.update(hungarian(piece_trials, subs))
        return out
    else:
        return hungarian(trials, motion_list)


def _signal_hungarian(trial_keys, motions, score_fn, min_score):
    """One-to-one assignment maximizing score; pairs below min_score are rejected."""
    if not trial_keys or not motions:
        return {}
    scores = np.array([[score_fn(k, m) for m in motions] for k in trial_keys], dtype=float)
    cost = np.where(np.isfinite(scores) & (scores >= min_score), 1.0 - scores, 1e6)
    size = max(cost.shape)
    padded = np.full((size, size), 1e6)
    padded[:cost.shape[0], :cost.shape[1]] = cost
    row, col = linear_sum_assignment(padded)
    out = {}
    for r, c in zip(row, col):
        if r < len(trial_keys) and c < len(motions) and padded[r, c] < 1e6:
            finite = sorted(scores[r][np.isfinite(scores[r])], reverse=True)
            margin = finite[0] - finite[1] if len(finite) > 1 else None
            out[trial_keys[r]] = (motions[c], float(scores[r, c]), margin)
    return out


def assign_motion_xcorr(trials, hand, body, mode):
    """
    Return (hand_assign, body_assign): {(P, t): (motion, raw_diff, method, score, margin)}.
    """
    key = lambda tr: (tr["canonical_p"], tr["canonical_t"])
    by_key = {key(tr): tr for tr in trials}
    midi_env = {key(tr): mx.midi_onset_envelope(tr["midi"]["path"]) for tr in trials if tr["midi"]}
    press_env = {id(m): mx.hand_press_envelope(m["path"]) for m in hand}

    def hand_score(k, m):
        return mx.best_ncc(midi_env[k], press_env[id(m)], HAND_OVERLAP)[0]

    hand_assign = {}
    for k, (m, sc, mg) in _signal_hungarian(list(midi_env), hand, hand_score, HAND_MIN_NCC).items():
        hand_assign[k] = (m, abs(by_key[k]["target_dur"] - m["dur"]), "xcorr_midi", sc, mg)
    # Two takes of one piece played the same way score alike on either trial;
    # when swapping them changes the summed score by no more than HAND_TIE_EPS
    # the signal cannot tell, and the takes follow the capture order.
    keys = sorted(hand_assign)
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            m1, m2 = hand_assign[k1][0], hand_assign[k2][0]
            if k1[0] != k2[0] or m1["piece_ord"] != m2["piece_ord"] or m1["take_ord"] < m2["take_ord"]:
                continue
            s11, s22 = hand_assign[k1][3], hand_assign[k2][3]
            s12, s21 = hand_score(k1, m2), hand_score(k2, m1)
            if abs((s11 + s22) - (s12 + s21)) <= HAND_TIE_EPS:
                hand_assign[k1] = (m2, abs(by_key[k1]["target_dur"] - m2["dur"]), "xcorr_midi_tie_take_order", s12, None)
                hand_assign[k2] = (m1, abs(by_key[k2]["target_dur"] - m1["dur"]), "xcorr_midi_tie_take_order", s21, None)
    # Trials without MIDI: duration fallback over the takes left unassigned.
    used = {id(v[0]) for v in hand_assign.values()}
    rest_t = [tr for tr in trials if not tr["midi"]]
    rest_m = [m for m in hand if id(m) not in used]
    for k, (m, diff) in assign_motion(rest_t, rest_m, mode).items():
        hand_assign[k] = (m, diff, "duration", None, None)

    speed_env = {id(v[0]): mx.hand_speed_envelope(v[0]["path"]) for v in hand_assign.values()}
    bspeed = {id(b): mx.body_speed_envelope(b["path"]) for b in body}

    def body_score(k, b):
        env_b = bspeed[id(b)]
        if k in hand_assign:
            ref = speed_env[id(hand_assign[k][0])]
        elif k in midi_env:
            ref = midi_env[k]
        else:
            return float("-inf")
        if len(env_b) < BODY_MIN_COVER * len(ref):
            return float("-inf")
        return mx.best_ncc(ref, env_b, BODY_OVERLAP, relative_to="b")[0]

    body_keys = [k for k in by_key if k in hand_assign or k in midi_env]
    body_assign = {}
    for k, (b, sc, mg) in _signal_hungarian(body_keys, body, body_score, BODY_MIN_NCC).items():
        ref = "xcorr_hand" if k in hand_assign else "xcorr_midi"
        body_assign[k] = (b, abs(by_key[k]["target_dur"] - b["dur"]), ref, sc, mg)
    return hand_assign, body_assign


# ─────────────────────────────────────────────────────────
# Per-node processing
# ─────────────────────────────────────────────────────────

def process_node(node: str, force: bool):
    base = RAW_ROOT / DATASET / node
    print(f"\n--- {DATASET}/{node} ---")
    audio = collect_audio(base)
    midi = collect_midi(base)
    hand, hand_excluded = collect_motion(base, "hand")
    body, body_excluded = collect_motion(base, "body")
    print(f"  raw: audio={len(audio)} midi={len(midi)} "
          f"hand={len(hand)} body={len(body)}")

    piece_order, trials = build_canonical(audio, midi)
    mode = "global" if node.startswith("C1/") else "per_piece"
    hand_assign, body_assign = assign_motion_xcorr(trials, hand, body, mode)
    previous = read_mapping(DATASET, node.replace("/", "_")) or {}
    prev_videos = {t["trial_stem"]: t["videos"] for t in previous.get("trials", []) if t.get("videos")}

    # Output session folder (C0_N0 style)
    session_name = node.replace("/", "_")
    out = session_dir(DATASET, session_name, modalities=MODALITIES)

    c = node[1]
    n = node[4]

    trial_records = []
    status_counts: dict[str, int] = {}
    for tr in trials:
        stem = f"{DATASET}_C{c}_N{n}_T1_P{tr['canonical_p']}_t{tr['canonical_t']}"
        key = (tr["canonical_p"], tr["canonical_t"])
        h = hand_assign.get(key)
        b = body_assign.get(key)
        # A body take the registry records as not captured for this trial:
        # whatever file the matcher found belongs to another take.
        if b and body_absence_reason(DATASET, stem):
            print(f"  {stem}: body {b[0]['path'].name} dropped (known absence)")
            b = None

        modalities_present = []
        raw_sources = {}

        if tr["audio"]:
            modalities_present.append("audio")
            dst = out / "audio" / f"{stem}.wav"
            if force or not dst.exists():
                shutil.copy2(tr["audio"]["path"], dst)
            raw_sources["audio"] = tr["audio"]["path"].name
        if tr["midi"]:
            modalities_present.append("midi")
            dst = out / "midi" / f"{stem}.mid"
            if force or not dst.exists():
                shutil.copy2(tr["midi"]["path"], dst)
            raw_sources["midi"] = tr["midi"]["path"].name
        if h:
            modalities_present.append("motion_hand")
            src = h[0]["path"]
            dst = out / "motion_hand" / f"{stem}{src.suffix}"
            if force or not dst.exists():
                shutil.copy2(src, dst)
            raw_sources["motion_hand"] = src.name
        if b:
            modalities_present.append("motion_body")
            src = b[0]["path"]
            dst = out / "motion_body" / f"{stem}{src.suffix}"
            if force or not dst.exists():
                shutil.copy2(src, dst)
            raw_sources["motion_body"] = src.name

        # A re-match can leave a trial without a take it had before; drop the
        # stale canonical copy so downstream stages do not pick it up.
        for kind, assigned in (("motion_hand", h), ("motion_body", b)):
            if not assigned:
                for stale in (out / kind).glob(f"{stem}.*"):
                    stale.unlink()

        has_audio = tr["audio"] is not None
        has_midi = tr["midi"] is not None
        has_hand = h is not None
        has_body = b is not None

        if has_audio and has_midi and has_hand and has_body:
            status = "complete"
        elif not has_audio and has_midi and (has_hand or has_body):
            status = "audio_missing"
        elif has_audio and not has_midi:
            status = "midi_missing"
        elif not has_hand and not has_body:
            status = "motion_missing"
        elif not has_hand:
            status = "hand_missing"
        elif not has_body:
            status = "body_missing"
        else:
            status = "partial"
        status_counts[status] = status_counts.get(status, 0) + 1

        durations = {
            "audio": round(tr["audio"]["dur"], 3) if tr["audio"] else None,
            "midi":  round(tr["midi"]["dur"], 3) if tr["midi"] else None,
            "hand":  round(h[0]["dur"], 3) if h else None,
            "body":  round(b[0]["dur"], 3) if b else None,
        }

        trial_records.append({
            "trial_stem": stem,
            "canonical_p": tr["canonical_p"],
            "canonical_t": tr["canonical_t"],
            "raw_piece": tr["raw_piece"],
            "status": status,
            "modalities": modalities_present,
            "duration_sec": durations,
            "match_diff_sec": {
                "hand": round(h[1], 3) if h else None,
                "body": round(b[1], 3) if b else None,
            },
            "match": {
                kind: ({"method": a[2],
                        "score": round(a[3], 3) if a[3] is not None else None,
                        "margin": round(a[4], 3) if a[4] is not None else None} if a else None)
                for kind, a in (("hand", h), ("body", b))
            },
            "r_note": tr.get("r_note"),
            "dup_note": tr.get("dup_note"),
            "raw_sources": raw_sources,
            **({"videos": prev_videos[stem]} if stem in prev_videos else {}),
        })

    print(f"  canonical trials: {len(trial_records)}  status: {status_counts}")
    print(f"  -> {out}")

    write_mapping(
        DATASET,
        session_name,
        trial_records,
        method="220819_xcorr_match",
        extra={
            "summary": status_counts,
            "piece_order": [
                {"raw_token": p, "canonical_p": i + 1}
                for i, p in enumerate(piece_order)
            ],
            "motion_match_mode": mode,
            "capabilities": {
                "has_timecode": False,
                "has_video": True,
                "video_is_per_trial": False,
                "has_body_motion": True,
                "body_in_hand_csv": False,
                "needs_xcorr_matching": True,
                "needs_dtw_audio_midi": True,
                "matching_method": "xcorr_hungarian (duration fallback for MIDI-less trials)",
            },
            "excluded_static": {
                "hand": hand_excluded,
                "body": body_excluded,
            },
            "notes": [
                "Raw 'T#' token is actually the Piece, not the Task. Task is 1 for 220819.",
                "MIDI _r1/_r2 are separate trials: r1 co-recorded with audio, r2 MIDI-only re-play. "
                "Canonical t# is chronological within (Task, Piece).",
                "Motion takes are matched by activity cross-correlation (hand vs MIDI onsets, body vs "
                "matched hand); raw motion filenames and the session log are not trusted for pairing.",
                "Video is session-level (Top/Right) or auto-split (Face/Left/RightFull). Stage 01 "
                "does not slice videos -- that is Stage 04 (04_piece_split, s05_slice_piece_videos.py).",
            ],
        },
    )


def run(dataset: str | None = None, force: bool = False):
    if dataset and dataset != DATASET:
        return
    print(f"Stage 01 s03 -- 220819 canonical trial matching")
    for node in NODES:
        try:
            process_node(node, force=force)
        except Exception as e:
            print(f"  ERROR {node}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    run()
