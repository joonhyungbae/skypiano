#!/usr/bin/env python3
"""
Stage 02 Spatial / body: register Qualisys body markers onto the hand frame.

Runs after 02_spatial/hand, because the hand markers must already be in the
common keyboard frame before the raw body markers can be registered to them.

Inputs:
  hand CSVs   Data/Pipeline/02_spatial/hand/Aligned/{dataset}/
  body files  raw motion files for {dataset} (resolved via the Stage 01 mapping)

Output:
  body CSVs   Data/Pipeline/02_spatial/body/Aligned/{dataset}/

For each dataset with body data (240523, 240809, 240828, 221112, 220819):
  1. Load the 02_spatial/hand CSV + raw body file.
  2. Find temporal offset via wrist-speed cross-correlation.
  3. Compute spatial alignment (weighted Horn's method) per trial, then use
     ONE transform per session: both capture systems stay put for a session
     (the keyboard end markers vary < 1-2 mm between its trials), so the true
     body->hand transform is the same for all its trials. Per-trial fits agree
     within a few mm for most trials but some are off by 30-70 mm; the session
     transform is the chordal mean of the inlier trials' transforms (within
     SESSION_INLIER_MM of the median). Transforms and each trial's deviation
     are written to Aligned/{dataset}/_transforms.json.
  4. Apply transform to body data and trim to the hand temporal range.
  4b. steps/s02_refine_session_transform: refine each session transform for
     the fixed offsets between the two systems' landmark markers.
  5. Write the aligned body CSV under every hand-stem alias so canonical and
     raw naming both resolve.

Usage:
  python run_pipeline.py                   # All datasets
  python run_pipeline.py --dataset 240809  # Single dataset
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.body_io import load_body_file, get_body_files
from src.alignment import (
    find_body_offset, find_body_offset_candidates, multi_window_alignment,
    apply_body_transform, per_landmark_residuals, _map_hand_to_body_frame,
    _extract_landmark_arrays, fast_alignment_residual, LANDMARK_PAIRS,
)

# Threshold (mm): if the worst per-landmark residual exceeds this, flag the
# trial. Typical healthy sessions run ~10-20mm; numbers above ~30mm usually
# indicate a marker-selection or scale mismatch in Stage 02 hand.
RESIDUAL_WARN_MM = 30.0
SESSION_INLIER_MM = 10.0
# Points around the keyboard and the player (hand frame, mm) at which two
# transforms are compared.
PROBE_POINTS = np.array([[0, 0, 0], [600, 0, 0], [-600, 0, 0], [0, 400, 300]], float)


def _probe(R, t):
    """Where the probe points of the hand frame come from in body coordinates."""
    return (np.asarray(R).T @ (PROBE_POINTS - np.asarray(t)).T).T


def _session_transform(items):
    """Chordal mean of the inlier transforms; returns (R, t, inlier stems)."""
    pts = np.array([_probe(R, t) for _, R, t in items])
    med = np.median(pts, axis=0)
    dev = np.linalg.norm(pts - med, axis=2).max(axis=1)
    keep = [it for it, d in zip(items, dev) if d <= SESSION_INLIER_MM] or \
           [items[int(np.argmin(dev))]]
    M = np.mean([R for _, R, _ in keep], axis=0)
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    t = np.median([tv for _, _, tv in keep], axis=0)
    return R, t, [k[0] for k in keep]

# Datasets whose hand-side wrist is the styloid midpoint (`wrist_center` in
# 02_spatial/hand/src/markers.py). For these, the body wrist landmark
# used to estimate the alignment is the same midpoint of LAW/MEW, so the
# wrist pair compares like with like. The written body data is unchanged.
STYLOID_CENTER_WRIST_DATASETS = {'220819', '221112'}


def _body_alignment_reference(body_df: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    """Copy of body_df whose LLAW/RLAW columns hold the LAW/MEW midpoint."""
    if dataset_id not in STYLOID_CENTER_WRIST_DATASETS:
        return body_df
    ref = body_df.copy()
    for side in ('L', 'R'):
        for ax in ('x', 'y', 'z'):
            law, mew = f'{side}LAW_{ax}', f'{side}MEW_{ax}'
            if law in ref.columns and mew in ref.columns:
                ref[law] = (body_df[law] + body_df[mew]) / 2.0
    return ref

HAND_ALIGNED_ROOT = 'Data/Pipeline/02_spatial/hand/Aligned'
OUTPUT_ROOT = 'Data/Pipeline/02_spatial/body/Aligned'
NORMALIZED_ROOT = 'Data/Pipeline/01_ingest/Normalized'
DATASETS_WITH_BODY = ['240523', '240809', '240828', '221112', '220819']


def load_trial_aliases(dataset_id: str) -> dict:
    """
    Load Stage 01 mappings for a dataset and return a dict that maps every
    known stem alias to a canonical raw motion stem.

    For each Stage 01 trial we record:
      - canon stem (e.g. 220819_C0_N0_T1_t1)            → raw motion stem (2022Piano01_03_01_01)
      - raw motion stem itself                          → raw motion stem (identity)

    The body registration can then look up either name in this alias table when matching
    against body files (which use the raw motion stem).
    """
    aliases: dict[str, str] = {}
    ds_dir = os.path.join(NORMALIZED_ROOT, dataset_id)
    if not os.path.isdir(ds_dir):
        return aliases
    for sess in os.listdir(ds_dir):
        mp = os.path.join(ds_dir, sess, '_mapping.json')
        if not os.path.isfile(mp):
            continue
        try:
            with open(mp, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
        except Exception:
            continue
        for t in mapping.get('trials', []):
            canon = t.get('trial_stem')
            raw_stem = t.get('raw_stem')
            if not raw_stem:
                raw_motion = t.get('raw_sources', {}).get('motion_hand') or t.get('raw_motion')
                if raw_motion:
                    raw_stem = os.path.splitext(os.path.basename(raw_motion))[0]
            if not canon or not raw_stem:
                continue
            aliases[canon] = raw_stem
            aliases[raw_stem] = raw_stem
    return aliases


def get_hand_files(dataset_id: str) -> dict:
    """Return {trial_name: filepath} for Stage 02 spatially-aligned hand data."""
    hand_dir = os.path.join(HAND_ALIGNED_ROOT, dataset_id)
    result = {}
    if os.path.isdir(hand_dir):
        for f in sorted(os.listdir(hand_dir)):
            if f.endswith('.csv'):
                result[os.path.splitext(f)[0]] = os.path.join(hand_dir, f)
    return result


def align_dataset(dataset_id: str):
    print(f"\n{'='*60}")
    print(f"  Stage 02 body alignment for {dataset_id}")
    print(f"{'='*60}")

    hand_files = get_hand_files(dataset_id)
    body_files = get_body_files(dataset_id)

    if not hand_files:
        print(f"  No Stage 02 hand files for {dataset_id}")
        return
    if not body_files:
        print(f"  No body files for {dataset_id}")
        return

    # Match hand ↔ body by trial name.
    # Body file naming may differ from hand file naming. We use Stage 01 mapping
    # to translate canonical hand stems (e.g. 220819_C0_N0_T1_t1) into the
    # original raw motion stem (e.g. 2022Piano01_03_01_01) which body files use.
    body_lookup = dict(body_files)
    for k, v in list(body_files.items()):
        if k.startswith('2022'):
            body_lookup[k[4:]] = v  # also register without prefix

    aliases = load_trial_aliases(dataset_id)
    if aliases:
        print(f"  Stage 01 aliases loaded: {len(aliases)} entries")

    # Group hand files by their resolved body file. Multiple hand stems (e.g.
    # 220819_C0_N0_T1_t1 and 2022Piano01_03_01_01) often refer to the same
    # physical trial — they share one body file. We process each body once but
    # record every hand stem so the body output is written under each name.
    groups: dict[str, list[tuple[str, str]]] = {}
    for trial, hand_path in hand_files.items():
        body_path = body_lookup.get(trial) or body_lookup.get('2022' + trial)
        if body_path is None and trial in aliases:
            body_path = body_lookup.get(aliases[trial]) or body_lookup.get('2022' + aliases[trial])
        if body_path:
            groups.setdefault(body_path, []).append((trial, hand_path))

    matched = []  # (primary_trial, primary_hand_path, body_path, all_trial_aliases)
    for body_path, hands in groups.items():
        # Prefer canonical (220819_/221112_/etc) names over raw (2022Piano*) ones
        hands.sort(key=lambda x: (0 if not x[0].startswith('2022Piano') else 1, x[0]))
        primary_trial, primary_hand_path = hands[0]
        # Include both trial_stem and raw_stem as output aliases so downstream
        # (Stage 04 piece_split) can find the body CSV by either naming scheme.
        all_trials: list[str] = []
        for t, _ in hands:
            if t not in all_trials:
                all_trials.append(t)
            raw_alias = aliases.get(t)
            if raw_alias and raw_alias not in all_trials:
                all_trials.append(raw_alias)
        matched.append((primary_trial, primary_hand_path, body_path, all_trials))

    print(f"  Hand files: {len(hand_files)}, Body files: {len(body_files)}, "
          f"Trial groups: {len(matched)}, Total hand aliases: {sum(len(g[3]) for g in matched)}")

    if not matched:
        print(f"  No matching pairs found")
        return

    out_dir = os.path.join(OUTPUT_ROOT, dataset_id)
    os.makedirs(out_dir, exist_ok=True)

    results = []
    estimates = []   # pass 1: per-trial time offset and transform
    for trial, hand_path, body_path, all_trials in tqdm(matched, desc=f"  {dataset_id}"):
        try:
            # Load
            hand_df = pd.read_csv(hand_path)
            body_fps, body_df = load_body_file(body_path)
            hand_fps = 120.0  # Stage 02 hand output is always 120fps
            # Alignment is estimated on body_ref; the transform is applied to body_df.
            body_ref = _body_alignment_reference(body_df, dataset_id)

            # Step 1: Temporal sync — two-stage search.
            #   (a) Generate many candidates (xcorr peaks + coarse grid).
            #   (b) Fast-screen each with a lightweight per-candidate residual.
            #   (c) Run the full multi_window_alignment only on the top-3
            #       screening winners and pick the spatially best.
            candidates = find_body_offset_candidates(
                hand_df, hand_fps, body_ref, body_fps,
                top_k=5, grid_step=50,
            )
            if not candidates:
                print(f"    SKIP {trial}: no temporal candidates")
                results.append((trial, None, None, None, None, None))
                continue

            # Pre-extract landmark arrays once for the fast screening pass
            hand_names = [p[0] for p in LANDMARK_PAIRS]
            body_names = [p[1] for p in LANDMARK_PAIRS]
            hand_arrs = _extract_landmark_arrays(hand_df, hand_names)
            body_arrs = _extract_landmark_arrays(body_ref, body_names)
            n_hand_rows = len(hand_df)
            n_body_rows = len(body_ref)

            scored = []
            for cand in candidates:
                s = fast_alignment_residual(
                    hand_arrs, body_arrs, n_hand_rows, n_body_rows,
                    hand_fps, body_fps, cand,
                )
                scored.append((s, cand))
            scored.sort(key=lambda x: x[0])
            top_refined = [c for sc, c in scored[:5] if np.isfinite(sc)]
            # Guard: if fast scores are all inf, fall back to xcorr best
            if not top_refined:
                top_refined = [candidates[0]]

            best = None  # (mean_resid, body_offset, R, t, body_aligned, residuals)
            for cand_offset in top_refined:
                result = multi_window_alignment(
                    hand_df, hand_fps, body_ref, body_fps, cand_offset,
                )
                if result is None:
                    continue
                R_c, t_c = result
                b_aligned_c = apply_body_transform(body_df, R_c, t_c)
                res_c = per_landmark_residuals(
                    hand_df, hand_fps, apply_body_transform(body_ref, R_c, t_c), body_fps, cand_offset,
                )
                mean_c = res_c.get('_mean', float('inf'))
                if not np.isfinite(mean_c):
                    continue
                if best is None or mean_c < best[0]:
                    best = (mean_c, cand_offset, R_c, t_c, b_aligned_c, res_c)

            if best is None:
                print(f"    SKIP {trial}: insufficient landmarks (aliases: {all_trials})")
                results.append((trial, candidates[0], None, None, None, None))
                continue

            mean_resid, body_offset, R, t, body_aligned, residuals = best
            if body_offset != candidates[0]:
                print(f"    {trial}: picked offset {body_offset} (mean_res={mean_resid:.1f}mm) "
                      f"over xcorr top-1 {candidates[0]}")
            estimates.append({"trial": trial, "hand_path": hand_path, "body_path": body_path,
                              "aliases": all_trials, "offset": body_offset, "R": R, "t": t,
                              "trial_mean_resid": mean_resid})

        except Exception as e:
            print(f"    ERROR {trial}: {e!r}")
            results.append((trial, None, None, None, None, None))

    # Pass 2: one transform per session, applied with each trial's own offset.
    import re
    def session_of(trial):
        m = re.search(r'C\d+_N\d+', trial)
        return m.group(0) if m else '_all'
    by_sess: dict[str, list] = {}
    for e in estimates:
        by_sess.setdefault(session_of(e["trial"]), []).append(e)
    record = {"session_inlier_mm": SESSION_INLIER_MM, "sessions": {}, "trials": {}}
    for sess, es in sorted(by_sess.items()):
        R_s, t_s, inliers = _session_transform([(e["trial"], e["R"], e["t"]) for e in es])
        record["sessions"][sess] = {"R": R_s.round(8).tolist(), "t": t_s.round(4).tolist(),
                                    "n_trials": len(es), "inliers": len(inliers)}
        print(f"  [{sess}] session transform from {len(inliers)}/{len(es)} trials")
        for e in es:
            trial = e["trial"]
            hand_df = pd.read_csv(e["hand_path"])
            body_fps, body_df = load_body_file(e["body_path"])
            hand_fps = 120.0
            body_ref = _body_alignment_reference(body_df, dataset_id)
            body_aligned = apply_body_transform(body_df, R_s, t_s)
            residuals = per_landmark_residuals(
                hand_df, hand_fps, apply_body_transform(body_ref, R_s, t_s), body_fps, e["offset"])
            mean_resid = residuals.get('_mean', float('nan'))
            max_resid = residuals.get('_max', float('nan'))
            dev = float(np.linalg.norm(_probe(e["R"], e["t"]) - _probe(R_s, t_s), axis=1).max())

            # Resample body to the hand timeline (120fps)
            n_hand = len(hand_df)
            body_resampled_rows = []
            for hf in range(n_hand):
                bf = _map_hand_to_body_frame(hf, e["offset"], hand_fps, body_fps)
                if 0 <= bf < len(body_aligned):
                    body_resampled_rows.append(body_aligned.iloc[bf].values)
                else:
                    body_resampled_rows.append([np.nan] * len(body_aligned.columns))
            body_out = pd.DataFrame(body_resampled_rows, columns=body_aligned.columns)
            # Save under every hand-stem alias so any downstream lookup
            # (canonical or raw naming) finds the same body-aligned data.
            for alias in e["aliases"]:
                body_out.to_csv(os.path.join(out_dir, f"{alias}.csv"), index=False)

            record["trials"][trial] = {"session": sess, "offset": e["offset"], "aliases": e["aliases"],
                                       "trial_fit_vs_session_mm": round(dev, 1),
                                       "inlier": trial in inliers,
                                       "mean_resid_mm": round(float(mean_resid), 1) if np.isfinite(mean_resid) else None,
                                       "max_resid_mm": round(float(max_resid), 1) if np.isfinite(max_resid) else None}
            if np.isfinite(max_resid) and max_resid > RESIDUAL_WARN_MM:
                worst_pair = max(
                    ((k, v) for k, v in residuals.items() if not k.startswith('_')),
                    key=lambda kv: kv[1] if np.isfinite(kv[1]) else -1,
                )
                print(f"    WARN  {trial}: max residual {max_resid:.1f}mm "
                      f"({worst_pair[0]}={worst_pair[1]:.1f}mm)")
            angle = np.degrees(np.arccos(np.clip((np.trace(R_s) - 1) / 2, -1, 1)))
            results.append((trial, e["offset"], angle, float(np.linalg.norm(t_s)), mean_resid, max_resid))
    with open(os.path.join(out_dir, "_transforms.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)

    # Summary
    print(f"\n  {'Trial':<35} {'Offset':>7} {'Rot°':>7} {'Trans':>8} {'μResid':>8} {'MaxResid':>9}")
    print(f"  {'-'*75}")
    for trial, offset, angle, trans, mean_r, max_r in results:
        off_s = f"{offset:>7}" if offset is not None else "  FAIL"
        ang_s = f"{angle:>7.2f}" if angle is not None else "   SKIP"
        tra_s = f"{trans:>8.1f}" if trans is not None else "    SKIP"
        mr_s  = f"{mean_r:>8.1f}" if mean_r is not None and np.isfinite(mean_r) else "    SKIP"
        Mr_s  = f"{max_r:>9.1f}" if max_r is not None and np.isfinite(max_r) else "     SKIP"
        flag  = '  !!' if max_r is not None and np.isfinite(max_r) and max_r > RESIDUAL_WARN_MM else ''
        print(f"  {trial:<35} {off_s} {ang_s} {tra_s} {mr_s} {Mr_s}{flag}")

    print(f"\n  Output: {out_dir}")
    print(f"  {len([r for r in results if r[2] is not None])}/{len(matched)} files aligned")


def main():
    parser = argparse.ArgumentParser(
        description='Stage 02 body-to-hand coordinate frame alignment '
                    '(feeds the per-trial slicer under 02_spatial/body).')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Process single dataset (default: all)')
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else DATASETS_WITH_BODY

    print("=" * 60)
    print("  Stage 02 body: Body -> Hand Coordinate Frame Alignment")
    print("=" * 60)

    for ds in datasets:
        if ds not in DATASETS_WITH_BODY:
            print(f"  Skipping {ds} (no body data)")
            continue
        align_dataset(ds)
        # Step 2: refine each session's transform for the landmark offsets
        # (applied only where it lowers the held-out residual).
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "s02_refine_session_transform",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "steps", "s02_refine_session_transform.py"))
        refine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(refine)
        refine.run(ds)

    print("\n" + "=" * 60)
    print("  BODY ALIGNMENT COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
