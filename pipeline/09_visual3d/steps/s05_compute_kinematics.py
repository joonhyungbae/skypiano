"""
Standalone CLI tool; not run by run_stage.sh.

Stage 09 step s05: compute pseudo-kinematics from per-trial motion CSVs.

Outputs `kinematics.csv` next to each trial's motion CSVs with columns:
    time_sec, trunk_lean_deg, fingertip_vel_L, fingertip_vel_R, hand_span_L, hand_span_R

- trunk_lean_deg: sagittal lean of (ACRO midpoint) above (HIP midpoint), 0° = upright.
- fingertip_vel_*: max-over-fingers of |dz/dt| (mm/s) per hand, low-pass smoothed.
- hand_span_*: thumb tip <-> pinky tip Euclidean distance (mm).

Body markers are 120Hz in mm. Hand markers same. Frames are aligned (Stage 02).

Usage:
    python steps/s05_compute_kinematics.py <trial_dir> [<trial_dir> ...]

Each <trial_dir> must contain motion_hand_*.csv and motion_body_raw_flagged.csv
(or equivalents). Source CSVs are assumed pre-decimated to FPS Hz; if your
inputs are at a different rate, set the FPS constant below to match.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Iterable

sys.stdout.reconfigure(encoding="utf-8")

# Source motion CSVs are assumed already decimated to this rate by an
# upstream sampling step; this script does not re-decimate the input frames.
FPS = 30.0
DT = 1.0 / FPS
OUT_FPS = 30.0


def parse_csv(path: Path) -> tuple[list[str], list[list[float]]]:
    if not path.exists():
        return [], []
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], []
    header = rows[0]
    data: list[list[float]] = []
    for r in rows[1:]:
        if len(r) < len(header):
            r = r + [""] * (len(header) - len(r))
        out: list[float] = []
        for v in r:
            try:
                out.append(float(v))
            except ValueError:
                out.append(float("nan"))
        data.append(out)
    return header, data


def col_idx(header: list[str], name: str, axis: str) -> int | None:
    key = f"{name}_{axis}"
    try:
        return header.index(key)
    except ValueError:
        return None


def get_xyz(header: list[str], row: list[float], name: str) -> tuple[float, float, float] | None:
    ix = col_idx(header, name, "x")
    iy = col_idx(header, name, "y")
    iz = col_idx(header, name, "z")
    if ix is None or iy is None or iz is None:
        return None
    x, y, z = row[ix], row[iy], row[iz]
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return None
    return x, y, z


def midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)


def trunk_lean(body_header: list[str], body_row: list[float]) -> float | None:
    """Sagittal lean: angle between (HIP→ACRO vector) and global +Z (up). Degrees."""
    al = get_xyz(body_header, body_row, "ACROL")
    ar = get_xyz(body_header, body_row, "ACROR")
    hl = get_xyz(body_header, body_row, "HIPL")
    hr = get_xyz(body_header, body_row, "HIPR")
    if not all([al, ar, hl, hr]):
        return None
    shoulder = midpoint(al, ar)
    hip = midpoint(hl, hr)
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    dz = shoulder[2] - hip[2]
    horiz = math.hypot(dx, dy)
    return math.degrees(math.atan2(horiz, abs(dz)))


def fingertip_speeds(hand_header, hand_rows, hand: str) -> list[float | None]:
    """Per-frame max |fingertip vertical velocity| (mm/s) across 5 fingers of one hand."""
    fingers = [f"{hand}{f}F" for f in ("T", "I", "M", "R", "L")]
    z_idx = [col_idx(hand_header, n, "z") for n in fingers]
    speeds: list[float | None] = [None] * len(hand_rows)
    if all(i is None for i in z_idx):
        return speeds
    # Central differences for stability; fall back to forward/back at edges.
    for i in range(len(hand_rows)):
        best = 0.0
        seen = False
        for fi, ix in enumerate(z_idx):
            if ix is None:
                continue
            zc = hand_rows[i][ix]
            if not math.isfinite(zc):
                continue
            if 0 < i < len(hand_rows) - 1:
                zp = hand_rows[i - 1][ix]
                zn = hand_rows[i + 1][ix]
                if math.isfinite(zp) and math.isfinite(zn):
                    v = abs((zn - zp) / (2 * DT))
                else:
                    v = 0.0
            else:
                v = 0.0
            if v > best:
                best = v
                seen = True
        speeds[i] = best if seen else None
    return speeds


def hand_span(hand_header, hand_row, hand: str) -> float | None:
    t = get_xyz(hand_header, hand_row, f"{hand}TF")
    p = get_xyz(hand_header, hand_row, f"{hand}LF")
    if t is None or p is None:
        return None
    return math.hypot(t[0] - p[0], math.hypot(t[1] - p[1], t[2] - p[2]))


def smooth_ema(values: list[float | None], alpha: float = 0.15) -> list[float | None]:
    out: list[float | None] = []
    state: float | None = None
    for v in values:
        if v is None or not math.isfinite(v):
            out.append(state)
            continue
        state = v if state is None else alpha * v + (1 - alpha) * state
        out.append(state)
    return out


def downsample(rows: list[dict], step: int) -> Iterable[dict]:
    for i in range(0, len(rows), step):
        yield rows[i]


def process_trial(base: Path) -> None:
    trial_id = base.name
    if not base.exists():
        print(f"[skip] {trial_id} (no trial dir at {base})")
        return
    hand_path = base / "motion_hand_imputed.csv"
    if not hand_path.exists():
        hand_path = base / "motion_hand_raw_flagged.csv"
    body_path = base / "motion_body_raw_flagged.csv"

    hh, hand_rows = parse_csv(hand_path)
    bh, body_rows = parse_csv(body_path)
    if not hand_rows and not body_rows:
        print(f"[skip] {trial_id} (no motion CSVs)")
        return

    n = max(len(hand_rows), len(body_rows))

    # Per-frame metrics
    if body_rows:
        trunk = [trunk_lean(bh, body_rows[i]) if i < len(body_rows) else None for i in range(n)]
    else:
        trunk = [None] * n

    if hand_rows:
        v_l = fingertip_speeds(hh, hand_rows, "L")
        v_r = fingertip_speeds(hh, hand_rows, "R")
        span_l = [hand_span(hh, hand_rows[i], "L") if i < len(hand_rows) else None for i in range(n)]
        span_r = [hand_span(hh, hand_rows[i], "R") if i < len(hand_rows) else None for i in range(n)]
    else:
        v_l = v_r = span_l = span_r = [None] * n

    # Smooth velocity (Savitzky-like via EMA — short enough sigma for visual clarity)
    v_l = smooth_ema(v_l, alpha=0.18)
    v_r = smooth_ema(v_r, alpha=0.18)

    rows = []
    for i in range(n):
        rows.append({
            "time_sec": i * DT,
            "trunk_lean_deg": trunk[i],
            "fingertip_vel_L": v_l[i],
            "fingertip_vel_R": v_r[i],
            "hand_span_L": span_l[i],
            "hand_span_R": span_r[i],
        })

    step = max(1, int(round(FPS / OUT_FPS)))
    out_path = base / "kinematics.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_sec", "trunk_lean_deg", "fingertip_vel_L", "fingertip_vel_R",
                    "hand_span_L", "hand_span_R"])
        for r in downsample(rows, step):
            w.writerow([
                f"{r['time_sec']:.4f}",
                "" if r["trunk_lean_deg"] is None else f"{r['trunk_lean_deg']:.3f}",
                "" if r["fingertip_vel_L"] is None else f"{r['fingertip_vel_L']:.1f}",
                "" if r["fingertip_vel_R"] is None else f"{r['fingertip_vel_R']:.1f}",
                "" if r["hand_span_L"] is None else f"{r['hand_span_L']:.2f}",
                "" if r["hand_span_R"] is None else f"{r['hand_span_R']:.2f}",
            ])

    n_out = sum(1 for _ in downsample(rows, step))
    have_body = sum(1 for v in trunk if v is not None) > 0
    have_hand = sum(1 for v in v_l if v is not None) + sum(1 for v in v_r if v is not None) > 0
    print(f"[ok] {trial_id}: {n_out} samples @ {OUT_FPS:.0f}Hz  "
          f"body={'Y' if have_body else '.'}  hand={'Y' if have_hand else '.'}")


def main():
    if len(sys.argv) < 2:
        print("usage: s05_compute_kinematics.py <trial_dir> [<trial_dir> ...]")
        return
    for arg in sys.argv[1:]:
        base = Path(arg)
        try:
            process_trial(base)
        except Exception as e:
            print(f"[err] {base}: {e}")


if __name__ == "__main__":
    main()
