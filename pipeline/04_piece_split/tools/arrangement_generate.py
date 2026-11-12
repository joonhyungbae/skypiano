"""
Generate arrangement CSVs for all sessions from their raw metadata CSVs.

For each session registered in SESSION_CONFIGS:
  1. parse_raw(raw_csv)
  2. build_arrangement_rows with session's filename_map / renamed_map
  3. apply overrides (if _overrides.json exists)
  4. write to {date}_{C}_{N}_arrangement.csv  (or _arrangement_auto.csv with
     --suffix to avoid overwriting ground truth)
  5. if ground truth file exists, diff and report

Usage:
  python pipeline/04_piece_split/tools/arrangement_generate.py                # all sessions, auto suffix
  python pipeline/04_piece_split/tools/arrangement_generate.py 220819_C0_N1   # single session
  python pipeline/04_piece_split/tools/arrangement_generate.py --overwrite    # write to canonical name
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))  # tools/
from arrangement_parser import (  # noqa: E402
    parse_raw, build_arrangement_rows, write_arrangement,
    load_overrides, apply_overrides,
)

META_ROOT = Path("Data/Metadata")


# ────────────────────────────────────────────────────────────────
# Per-session configs
# ────────────────────────────────────────────────────────────────
# Key = "{date}_{C}_{N}" (matches raw CSV stem)
# Each config:
#   filename_map: (set, trial, repeat) → raw motion CSV basename (or "" if missing)
#   renamed_map:  (set, trial, repeat) → canonical stem
#   include_feature_cols: True for C0 (Professional) arrangement format
#
# Raw motion filenames follow:
#   C0: 2022Piano01_{subject_idx:02}_{set:02}_{repeat:02}.csv  (concat per set)
#   C1: 2022Piano02_{subject_idx:02}_{set:02}_{sub:02}.csv     (per-trial sub)
# where subject_idx is the recording order within (date, class), NOT the N index.

SESSION_CONFIGS: dict[str, dict] = {}


# ─── 220819 C0/N2 (Professional) — already validated ───
SESSION_CONFIGS["220819_C0_N2"] = {
    "filename_map": {
        (1, 1, 1): "2022Piano01_01_01_01.csv", (1, 2, 1): "2022Piano01_01_01_01.csv", (1, 3, 1): "2022Piano01_01_01_01.csv",
        (1, 1, 2): "2022Piano01_01_01_02.csv", (1, 2, 2): "2022Piano01_01_01_02.csv", (1, 3, 2): "2022Piano01_01_01_02.csv",
        (2, 1, 2): "2022Piano01_01_02_01.csv",
        (2, 2, 3): "2022Piano01_01_02_02.csv",
        (3, 1, 2): "2022Piano01_01_03_01.csv",
    },
    "renamed_map": {
        (1, 1, 1): "220819_C0_N2_T1_t1", (1, 2, 1): "220819_C0_N2_T1_t1", (1, 3, 1): "220819_C0_N2_T1_t1",
        (1, 1, 2): "220819_C0_N2_T1_t2", (1, 2, 2): "220819_C0_N2_T1_t2", (1, 3, 2): "220819_C0_N2_T1_t2",
        (2, 1, 2): "220819_C0_N2_T2_t1",
        (2, 2, 3): "220819_C0_N2_T2_t2",
        (3, 1, 2): "220819_C0_N2_T3_t1",
    },
    "include_feature_cols": True,
}


# ─── 220819 C1/N1 (Amateur) — already validated ───
SESSION_CONFIGS["220819_C1_N1"] = {
    "filename_map": {
        (1, 1, 1): "2022Piano02_01_01_01.csv",
        (1, 2, 1): "2022Piano02_01_01_02.csv",
        (1, 3, 1): "2022Piano02_01_01_03.csv",
        (1, 1, 2): "2022Piano02_01_01_04.csv",
        (1, 2, 2): "2022Piano02_01_01_05.csv",
        (2, 1, 2): "2022Piano02_01_02_01.csv",
        (2, 2, 2): "2022Piano02_01_02_02.csv",
        (3, 1, 2): "2022Piano02_01_03_01.csv",
        (3, 2, 2): "2022Piano02_01_03_02.csv",
    },
    "renamed_map": {
        (1, 1, 1): "220819_C1_N1_T1-1_t1",
        (1, 2, 1): "220819_C1_N1_T1-2_t1",
        (1, 3, 1): "220819_C1_N1_T1-3_t1",
        (1, 1, 2): "220819_C1_N1_T1-1_t2",
        (1, 2, 2): "220819_C1_N1_T1-2_t2",
        (2, 1, 2): "220819_C1_N1_T2_t1",
        (2, 2, 2): "220819_C1_N1_T2_t2",
        (3, 1, 2): "220819_C1_N1_T3_t1",
        (3, 2, 2): "220819_C1_N1_T3_t2",
    },
    "include_feature_cols": False,
}


# ─── 220819 C0/N1 (Professional, concat mode) ───
# Raw motion/Hand/ has only Set2 + Set3 files (Set1 hand data lost).
# Set1 blocks are SKIPPED (no motion → no arrangement entry).
SESSION_CONFIGS["220819_C0_N1"] = {
    "filename_map": {
        (2, 1, 1): "2022Piano01_02_02_01.csv",
        (2, 1, 2): "2022Piano01_02_02_02.csv",
        (3, 1, 1): "2022Piano01_02_03_01.csv",
        (3, 1, 2): "2022Piano01_02_03_02.csv",
    },
    "renamed_map": {
        (2, 1, 1): "220819_C0_N1_T2_t1",
        (2, 1, 2): "220819_C0_N1_T2_t2",
        (3, 1, 1): "220819_C0_N1_T3_t1",
        (3, 1, 2): "220819_C0_N1_T3_t2",
    },
    "include_feature_cols": True,
}


# ─── 220819 C1/N0 (Amateur) ───
# Raw motion/Hand/: _02_02_{01..03}, _02_03_{01..02} = 5 files.
# Set2 has 2 repeats in raw metadata (R1 + R3 after skipped R2), mapping to
# files _02_02_01 / _02_02_02 / _02_02_03 = 3 files — so Set2 raw "repeat 3"
# actually lands in _02_02_03 (by recording order).
SESSION_CONFIGS["220819_C1_N0"] = {
    "filename_map": {
        (2, 1, 1): "2022Piano02_02_02_01.csv",
        (2, 2, 3): "2022Piano02_02_02_03.csv",
        (3, 1, 2): "2022Piano02_02_03_01.csv",
        (3, 2, 2): "2022Piano02_02_03_02.csv",
    },
    "renamed_map": {
        (2, 1, 1): "220819_C1_N0_T2_t1",
        (2, 2, 3): "220819_C1_N0_T2_t2",
        (3, 1, 2): "220819_C1_N0_T3_t1",
        (3, 2, 2): "220819_C1_N0_T3_t2",
    },
    "include_feature_cols": False,
}


def _make_221112_config(n: str, subject_idx: str, *,
                        set1_files: list[str], set2_files: list[str]) -> dict:
    """Build a C1 221112 session config. 221112 uses SPLIT mode for Set1
    (one motion file per (Trial, Repeat)).

    set1_files: ordered [t1_r1, t2_r1, t3_r1, t1_r2, t2_r2, t3_r2] basenames.
                Items may be "" if the file is missing for that (trial,repeat).
    set2_files: ordered [t1_r1, t2_r3] basenames. Items may be "".
    """
    stem = f"221112_C1_N{n}"
    # Set3 has no motion files across 221112 sessions → omit entirely.
    filename_map, renamed_map = {}, {}

    # Set 1
    keys_t = [(1, 1, 1), (1, 2, 1), (1, 3, 1), (1, 1, 2), (1, 2, 2), (1, 3, 2)]
    r_names = [f"{stem}_T1-1_t1", f"{stem}_T1-2_t1", f"{stem}_T1-3_t1",
               f"{stem}_T1-1_t2", f"{stem}_T1-2_t2", f"{stem}_T1-3_t2"]
    for key, fname, rname in zip(keys_t, set1_files, r_names):
        if fname:
            filename_map[key] = fname
            renamed_map[key] = rname

    # Set 2
    keys_s2 = [(2, 1, 1), (2, 2, 3)]
    r_names2 = [f"{stem}_T2_t1", f"{stem}_T2_t2"]
    for key, fname, rname in zip(keys_s2, set2_files, r_names2):
        if fname:
            filename_map[key] = fname
            renamed_map[key] = rname

    return {
        "_subject_idx": subject_idx,  # informational
        "filename_map": filename_map,
        "renamed_map": renamed_map,
        "include_feature_cols": False,
        "time_convention": "session_global",  # 221112 raw uses session-continuous timestamps
    }


# 221112 sessions — explicit filename lists to handle per-session anomalies
# (OptiTrack sub-index suffixes, missing takes, extra duplicate takes).
SESSION_CONFIGS["221112_C1_N2"] = _make_221112_config(
    n="2", subject_idx="03",
    set1_files=["Piano02_03_01_01_001.csv", "Piano02_03_01_02.csv", "Piano02_03_01_03.csv",
                "Piano02_03_01_04.csv", "Piano02_03_01_05.csv", "Piano02_03_01_06.csv"],
    set2_files=["Piano02_03_02_01.csv", "Piano02_03_02_02.csv"],
)
SESSION_CONFIGS["221112_C1_N3"] = _make_221112_config(
    n="3", subject_idx="04",
    set1_files=["Piano02_04_01_01.csv", "Piano02_04_01_02.csv", "Piano02_04_01_03.csv",
                "Piano02_04_01_04.csv", "Piano02_04_01_05.csv", "Piano02_04_01_06.csv"],
    set2_files=["Piano02_04_02_01.csv", "Piano02_04_02_02.csv"],
)
SESSION_CONFIGS["221112_C1_N4"] = _make_221112_config(
    n="4", subject_idx="05",
    set1_files=["Piano02_05_01_01.csv", "Piano02_05_01_02.csv", "Piano02_05_01_03.csv",
                "Piano02_05_01_04.csv", "Piano02_05_01_05.csv", "Piano02_05_01_06.csv"],
    set2_files=["Piano02_05_02_01.csv", "Piano02_05_02_02.csv"],
)
# N5 has an extra duplicate take _01_03_2.csv (second try of T3_t1) — ignored
# here; the _03 file (first take) is used.
SESSION_CONFIGS["221112_C1_N5"] = _make_221112_config(
    n="5", subject_idx="06",
    set1_files=["Piano02_06_01_01.csv", "Piano02_06_01_02.csv", "Piano02_06_01_03.csv",
                "Piano02_06_01_04.csv", "Piano02_06_01_05.csv", "Piano02_06_01_06.csv"],
    set2_files=["Piano02_06_02_01.csv", "Piano02_06_02_02.csv"],
)
SESSION_CONFIGS["221112_C1_N6"] = _make_221112_config(
    n="6", subject_idx="07",
    set1_files=["Piano02_07_01_01_01.csv", "Piano02_07_01_02.csv", "Piano02_07_01_03.csv",
                "Piano02_07_01_04.csv", "Piano02_07_01_05.csv", "Piano02_07_01_06.csv"],
    set2_files=["Piano02_07_02_01.csv", "Piano02_07_02_02.csv"],  # N6 has 3rd file _02_03 which is likely an extra take, ignored
)
# N7 has no _01_02 file — subject's T1-2_t1 was missing/not captured. Skip it.
SESSION_CONFIGS["221112_C1_N7"] = _make_221112_config(
    n="7", subject_idx="08",
    set1_files=["Piano02_08_01_01.csv", "",                      "Piano02_08_01_03.csv",
                "Piano02_08_01_04.csv", "Piano02_08_01_05.csv", "Piano02_08_01_06.csv"],
    set2_files=["Piano02_08_02_01.csv", "Piano02_08_02_02.csv"],
)
SESSION_CONFIGS["221112_C1_N8"] = _make_221112_config(
    n="8", subject_idx="09",
    set1_files=["Piano02_09_01_01.csv", "Piano02_09_01_02.csv", "Piano02_09_01_03.csv",
                "Piano02_09_01_04.csv", "Piano02_09_01_05.csv", "Piano02_09_01_06.csv"],
    set2_files=["Piano02_09_02_01.csv", "Piano02_09_02_02.csv"],
)
SESSION_CONFIGS["221112_C1_N9"] = _make_221112_config(
    n="9", subject_idx="10",
    set1_files=["Piano02_10_01_01.csv", "Piano02_10_01_02.csv", "Piano02_10_01_03.csv",
                "Piano02_10_01_04.csv", "Piano02_10_01_05.csv", "Piano02_10_01_06.csv"],
    set2_files=["Piano02_10_02_01.csv", "Piano02_10_02_02.csv"],
)
SESSION_CONFIGS["221112_C1_N10"] = _make_221112_config(
    n="10", subject_idx="11",
    set1_files=["Piano02_11_01_01.csv", "Piano02_11_01_02.csv", "Piano02_11_01_03.csv",
                "Piano02_11_01_04.csv", "Piano02_11_01_05.csv", "Piano02_11_01_06.csv"],
    set2_files=["Piano02_11_02_01.csv", "Piano02_11_02_02.csv"],
)
SESSION_CONFIGS["221112_C1_N11"] = _make_221112_config(
    n="11", subject_idx="12",
    set1_files=["Piano02_12_01_01.csv", "Piano02_12_01_02.csv", "Piano02_12_01_03.csv",
                "Piano02_12_01_04.csv", "Piano02_12_01_05.csv", "Piano02_12_01_06.csv"],
    set2_files=["Piano02_12_02_01.csv", "Piano02_12_02_02.csv"],
)


# ────────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────────

def generate_one(session_id: str, overwrite: bool = False, verbose: bool = True) -> dict:
    """Generate arrangement for one session. Returns stats dict."""
    conf = SESSION_CONFIGS.get(session_id)
    if conf is None:
        return {"session": session_id, "status": "no_config"}

    # Locate raw CSV: e.g. Metadata/220819/C0/N1/220819_C0_N1.csv
    parts = session_id.split("_")   # [date, C#, N#]
    date, c, n = parts[0], parts[1], parts[2]
    raw_path = META_ROOT / date / c / n / f"{session_id}.csv"
    if not raw_path.exists():
        return {"session": session_id, "status": "no_raw", "path": str(raw_path)}

    blocks = parse_raw(raw_path)
    rows = build_arrangement_rows(
        blocks,
        filename_map=conf["filename_map"],
        renamed_map=conf["renamed_map"],
        include_feature_cols=conf["include_feature_cols"],
        time_convention=conf.get("time_convention", "file_local"),
    )

    overrides_path = raw_path.with_name(raw_path.stem + "_overrides.json")
    edits = load_overrides(overrides_path)
    n_override = apply_overrides(rows, edits) if edits else 0

    # Decide output path
    truth_path = raw_path.with_name(f"{session_id}_arrangement.csv")
    out_path = truth_path if overwrite or not truth_path.exists() else \
        raw_path.with_name(f"{session_id}_arrangement_auto.csv")

    write_arrangement(rows, out_path, include_feature_cols=conf["include_feature_cols"])

    stats = {
        "session": session_id,
        "status": "ok",
        "blocks": len(blocks),
        "rows": len(rows),
        "overrides": n_override,
        "out": out_path.name,
        "has_truth": truth_path.exists() and truth_path != out_path,
    }

    if verbose:
        print(f"[{session_id}] blocks={stats['blocks']} rows={stats['rows']} "
              f"overrides={stats['overrides']} out={stats['out']}")

    return stats


def main():
    args = sys.argv[1:]
    overwrite = "--overwrite" in args
    keys = [a for a in args if not a.startswith("--")]
    if not keys:
        keys = list(SESSION_CONFIGS)

    for k in keys:
        r = generate_one(k, overwrite=overwrite)
        if r.get("status") != "ok":
            print(f"[{k}] SKIP: {r.get('status')} {r.get('path', '')}")


if __name__ == "__main__":
    main()
