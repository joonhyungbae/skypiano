"""
Regression harness: run arrangement_parser on raw metadata CSVs whose ground
truth is already hand-written, diff the generated output against ground truth.

Usage (from the repository root):
  python pipeline/04_piece_split/tools/arrangement_verify.py            # run both sessions
  python pipeline/04_piece_split/tools/arrangement_verify.py c0n2       # one session
  python pipeline/04_piece_split/tools/arrangement_verify.py --write    # also write /tmp/*.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))  # tools/ on path
from arrangement_parser import (  # noqa: E402
    parse_raw, build_arrangement_rows, write_arrangement,
    load_overrides, apply_overrides,
)


META_ROOT = Path("Data/Metadata")


# ── Session configurations ──────────────────────────────────────
# filename_map: (set, trial, repeat) → raw motion CSV basename
# renamed_map:  (set, trial, repeat) → canonical stem
# These are hand-curated from the existing arrangement ground truth plus the
# raw motion dir listings (Data/Raw/220819/{C}/{N}/Motion/Hand/).

# C0/N2 (Professional, concat mode) — 3 trials per repeat collapse into one
# arrangement unit; motion dir has 5 files total.
C0_N2_CONFIG = {
    "filename_map": {
        (1, 1, 1): "2022Piano01_01_01_01.csv",
        (1, 2, 1): "2022Piano01_01_01_01.csv",
        (1, 3, 1): "2022Piano01_01_01_01.csv",
        (1, 1, 2): "2022Piano01_01_01_02.csv",
        (1, 2, 2): "2022Piano01_01_01_02.csv",
        (1, 3, 2): "2022Piano01_01_01_02.csv",
        # Repeat 3 (HYC_*03) exists in raw but is dropped from arrangement
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

# C1/N1 (Amateur, split mode) — each (Trial, Repeat) in Set 1 is its own file.
# Raw motion dir: _01_01_{01..06}.csv (6 files, arrangement uses 5; _06 was
# dropped because Set1 T3 R2 audio was lost.)
C1_N1_CONFIG = {
    "filename_map": {
        (1, 1, 1): "2022Piano02_01_01_01.csv",
        (1, 2, 1): "2022Piano02_01_01_02.csv",
        (1, 3, 1): "2022Piano02_01_01_03.csv",
        (1, 1, 2): "2022Piano02_01_01_04.csv",
        (1, 2, 2): "2022Piano02_01_01_05.csv",
        # (1, 3, 2) intentionally missing — audio was lost
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

SESSIONS = {
    "c0n2": {
        "raw":    META_ROOT / "220819/C0/N2/220819_C0_N2.csv",
        "truth":  META_ROOT / "220819/C0/N2/220819_C0_N2_arrangement.csv",
        "config": C0_N2_CONFIG,
    },
    "c1n1": {
        "raw":    META_ROOT / "220819/C1/N1/220819_C1_N1.csv",
        "truth":  META_ROOT / "220819/C1/N1/220819_C1_N1_arrangement.csv",
        "config": C1_N1_CONFIG,
    },
}


def _read_csv(path: Path) -> list[list[str]]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.reader(f))


def _normalize_num(s: str) -> str:
    """Normalize number formatting for comparison (e.g., '19.5' vs '19.50')."""
    s = s.strip()
    try:
        v = float(s)
        if v == int(v):
            return str(int(v))
        return f"{v:.6f}".rstrip("0").rstrip(".")
    except ValueError:
        return s


def _compare_rows(
    truth_rows: list[list[str]],
    gen_rows: list[list[str]],
    header_cmp: list[str],
    num_cols: set[int],
) -> tuple[int, int, int, list[str]]:
    """Return (matches, mismatches, missing, details)."""
    # Build comparable projection: use header position mapping
    t_header = [c.strip() for c in truth_rows[0]]
    g_header = [c.strip() for c in gen_rows[0]]

    def col_idx(header: list[str], name: str) -> int:
        for i, h in enumerate(header):
            if h == name or h.startswith(name.split()[0]):
                return i
        return -1

    t_idx = {name: col_idx(t_header, name) for name in header_cmp}
    g_idx = {name: col_idx(g_header, name) for name in header_cmp}

    def key_row(row: list[str], idx: dict[str, int]) -> tuple:
        renamed = row[idx["Renamed"]].strip() if idx["Renamed"] >= 0 else ""
        start = _normalize_num(row[idx["Start"]]) if idx["Start"] >= 0 else ""
        pid = row[idx["ID"]].strip() if idx["ID"] >= 0 else ""
        return (renamed, pid, start)

    t_map = {key_row(r, t_idx): r for r in truth_rows[1:] if any(r)}
    g_map = {key_row(r, g_idx): r for r in gen_rows[1:] if any(r)}

    matches = 0
    mismatches = 0
    missing = 0
    details: list[str] = []

    for k in sorted(set(t_map) | set(g_map)):
        t_row = t_map.get(k)
        g_row = g_map.get(k)
        if t_row is None:
            missing += 1
            details.append(f"[EXTRA   ] {k}: in generated but not in truth")
            continue
        if g_row is None:
            missing += 1
            details.append(f"[MISSING ] {k}: in truth but not in generated")
            continue
        diffs = []
        for name in header_cmp:
            ti, gi = t_idx[name], g_idx[name]
            if ti < 0 or gi < 0:
                continue
            tv = t_row[ti].strip() if ti < len(t_row) else ""
            gv = g_row[gi].strip() if gi < len(g_row) else ""
            if name in {"Start", "End"}:
                tv, gv = _normalize_num(tv), _normalize_num(gv)
            if tv != gv:
                diffs.append(f"{name}: truth={tv!r} gen={gv!r}")
        if diffs:
            mismatches += 1
            details.append(f"[MISMATCH] {k}: " + "; ".join(diffs))
        else:
            matches += 1
    return matches, mismatches, missing, details


def verify_session(key: str, write: bool = False) -> bool:
    cfg = SESSIONS[key]
    raw_path = cfg["raw"]
    truth_path = cfg["truth"]
    conf = cfg["config"]

    print(f"\n{'=' * 60}")
    print(f"  Verify: {key}  ({raw_path.name})")
    print(f"{'=' * 60}")

    blocks = parse_raw(raw_path)
    print(f"  parsed {len(blocks)} raw blocks")
    total_pieces = sum(len(b.pieces) for b in blocks)
    print(f"  total pieces across blocks: {total_pieces}")

    rows = build_arrangement_rows(
        blocks,
        filename_map=conf["filename_map"],
        renamed_map=conf["renamed_map"],
        include_feature_cols=conf["include_feature_cols"],
    )
    print(f"  generated {len(rows)} arrangement rows")

    # Apply curator overrides (if present)
    overrides_path = raw_path.with_name(raw_path.stem + "_overrides.json")
    edits = load_overrides(overrides_path)
    if edits:
        n_mod = apply_overrides(rows, edits)
        print(f"  applied {n_mod} override(s) from {overrides_path.name}")

    # Write to /tmp for inspection
    if write:
        out_path = Path(f"tmp_arrangement_{key}.csv")
        write_arrangement(rows, out_path, include_feature_cols=conf["include_feature_cols"])
        print(f"  wrote {out_path}")
    else:
        out_path = Path(f"Data/Metadata/_verify_tmp_{key}.csv")
        write_arrangement(rows, out_path, include_feature_cols=conf["include_feature_cols"])

    truth_rows = _read_csv(truth_path)
    gen_rows = _read_csv(out_path)

    print(f"  truth rows: {len(truth_rows) - 1}, gen rows: {len(gen_rows) - 1}")

    header_cmp = ["Group", "ID", "Code", "Start", "End", "Filename", "Renamed",
                  "Tempo", "Piece", "Memo"]
    if conf["include_feature_cols"]:
        header_cmp.extend(["Feature", "Feature_end"])
    matches, mismatches, missing, details = _compare_rows(truth_rows, gen_rows, header_cmp, {3, 4})
    print(f"  match: {matches}, mismatch: {mismatches}, missing/extra: {missing}")

    max_show = 30
    for d in details[:max_show]:
        print(f"    {d}")
    if len(details) > max_show:
        print(f"    … +{len(details) - max_show} more")

    if not write:
        out_path.unlink(missing_ok=True)

    return (mismatches == 0 and missing == 0)


def main():
    args = sys.argv[1:]
    write = "--write" in args
    keys = [a for a in args if not a.startswith("--")]
    if not keys:
        keys = list(SESSIONS)

    results = []
    for k in keys:
        ok = verify_session(k, write=write)
        results.append((k, ok))

    print("\n" + "=" * 60)
    print("Summary:")
    for k, ok in results:
        print(f"  {k}: {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
