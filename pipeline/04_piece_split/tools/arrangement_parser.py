"""
Parse session-level metadata CSVs (wide block format) and emit the long-form
arrangement CSV used downstream by Stage 04 (piece split) for trial/piece boundary lookup.

Raw format (one file per session, stored under Data/Metadata/{date}/{C}/{N}/):

  subject,Set,Trial,Repeat, , , , , ...
  <one row of session-level notes>
  [BLOCK per (Set,Trial,Repeat) — 7 rows]
    row 0: name,set#,trial#,repeat#,ID,start_signal,<piece_id>,<piece_id>,...
    row 1: stem_code, , , ,start,<session_sync_sec>,<start_sec>,<start_sec>,...
    row 2:          , , , ,end  ,                  ,<end_sec>  ,<end_sec>  ,...
    row 3:          , , , ,note ,                  ,<note>     ,<note>     ,...
    row 4:          , , , ,duration,               ,<duration> ,<duration> ,...
    row 5:          , , , ,start(relative),        ,<rel_start>,<rel_start>,...
    row 6:          , , , ,end(relative),          ,<rel_end>  ,<rel_end>  ,...

Long arrangement format (output):

  Group, ID, Code, Start, End, Filename, Renamed, Tempo, Piece, [Feature,
  Feature_end,] Memo

Two grouping modes determine how (Set, Trial, Repeat) blocks map to Renamed
stems, driven by how raw motion files are structured per session:

  concat (C0 style) — one motion file per (Set, Repeat); all Trials inside one
                       Repeat concatenate into one arrangement unit
                       Renamed = {date}_{C_N}_T{Set}_t{Repeat}
  split  (C1 style) — one motion file per (Trial, Repeat) within a Set; each
                       Trial becomes its own sub-task
                       Renamed = {date}_{C_N}_T{Set}-{Trial}_t{Repeat}
                       (Sets 2/3 collapse to T{Set}_t{Repeat} since only Trial 1 exists)
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Piece:
    id: str                   # raw piece label from header row (e.g. "4", "10 - Forte", "Scale_Slow")
    start_sec: float | None   # from start row
    end_sec: float | None     # from end row
    note: str = ""            # from note row
    column_label: str = ""    # raw header column text (may carry hand annotation like "27-left_only")


@dataclass
class RawBlock:
    subject: str
    set_no: int
    trial_no: int
    repeat_no: int
    stem_code: str | None = None
    session_signal_sec: float | None = None
    pieces: list[Piece] = field(default_factory=list)
    raw_row_index: int = 0    # starting line index in the source CSV


BLOCK_ROW_ORDER = ["\uc2dc\uc791", "\ub05d", "\ud2b9\uc774\uc0ac\ud56d", "\uae38\uc774", "\uc2dc\uc791(\uc0c1\ub300\uc2dc\uac04)", "\ub05d(\uc0c1\ub300\uc2dc\uac04)"]


_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")

# Take-marker patterns — memos that only describe "which take was used" rather
# than per-piece performance notes. Drop so arrangement memo column stays clean.
#
#   "2nd"                    bare ordinal
#   "2 out of 2, 2nd"          "Nth of M" with count
#   "last of several"          "last of many" (variant wording)
#   "1st of several"           same shape, explicit ordinal
#   "3rd/ 2nd back"          mixed-script fragments
#   "last" / "last one"      English bare take hint
#
_TAKE_RE_LIST = [
    re.compile(r"^\s*\d+(?:st|nd|rd|th)\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*\uac1c\s*\uc911(?:\s*\S+)?\s*$"),
    re.compile(r"^\s*\uc5ec\ub7ec\s*\uac1c?\s*\uc911\s*\S+.*$"),
    re.compile(r"^\s*\d+\s*(?:st|nd|rd|th)?\s*(?:from\s+back|from\s+last|back)\s*$", re.IGNORECASE),
    re.compile(r"^\s*last(?:\s+one)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$"),  # bare integer fragment (e.g. from "4th/4" split)
]


def _clean_cell(s: str) -> str:
    """Remove control chars (backspace etc. occasionally pasted from Excel) and trim."""
    if s is None:
        return ""
    return _CTRL_RE.sub("", s).strip()


def _is_take_marker(s: str) -> bool:
    return any(rx.match(s) for rx in _TAKE_RE_LIST)


def _clean_note(note: str) -> str:
    """Drop take markers from the note column; also drop 1-char fragments
    (curator mis-entered a multi-char memo one char per column — see 221112
    N10 'Set1 T2 R2' note where a 4-character memo was typed across 4 cells,
    one character per cell).
    """
    s = _clean_cell(note)
    if not s:
        return ""
    # Drop 1-char memos — almost always fragments of a multi-cell memo the
    # curator typed one char per column (see 221112 N10 Set1 T2 R2 where a
    # 4-character memo was split across 4 cells, one character per cell).
    if len(s) == 1:
        return ""
    parts = [p.strip() for p in re.split(r"[;/|]|,\s*", s) if p.strip()]
    kept = [p for p in parts if not _is_take_marker(p)]
    return " / ".join(kept)


def _normalize_piece_id(pid: str) -> str:
    """Normalize bonus-piece labels.

    Raw N10 has 'Beethoven ' (trailing space); N11 uses 'Beethoven' (no number)
    for T1-3_t1 and a localized spelling of "Beethoven" for T1-3_t2. Convention
    across 220819 sessions is 'Beethoven 1' / 'Beethoven 2'. Since 221112 only
    plays the first Beethoven as a Set1 bonus take, normalize bare labels to
    'Beethoven 1'.
    """
    s = pid.strip()
    if s in {"Beethoven", "\ubca0\ud1a0\ubca4"}:
        return "Beethoven 1"
    if s == "\ubca0\ud1a0\ubca4 1":
        return "Beethoven 1"
    if s == "\ubca0\ud1a0\ubca4 2":
        return "Beethoven 2"
    return s


def _strip_hand_suffix(piece_id: str) -> tuple[str, str | None]:
    """Split piece id and hand annotation, if any.

    Examples:
        "27"                                 -> ("27", None)
        "27-<left-only-localized>"            -> ("27", "left")
        "30 - <right-only-localized>"          -> ("30", "right")
        "28(left hand)"                      -> ("28", "left")
        "28(both hand)"                      -> ("28", "both")
    """
    s = _clean_cell(piece_id)

    # Parenthetical "(left hand)" / "(both hand)" / "(right hand)"
    m = re.match(r"^(.+?)\s*\((left|both|right)\s*hand\)\s*$", s, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).lower()

    # Hyphenated localized hand-only suffixes: "- left-only" / "- right-only" /
    # "left-only" / "right-only" / "left-fast" (matched as Unicode escapes in regex)
    m = re.match(r"^(.+?)\s*[-–]\s*(\uc67c\uc190\ub9cc|\uc624\ub978\uc190\ub9cc|\uc67c\uc190\ube60\ub974\uac8c)\s*$", s)
    if m:
        tag = m.group(2)
        hand = "left" if "\uc67c" in tag else "right"
        return m.group(1).strip(), hand

    return s, None


def _parse_float(val: str) -> float | None:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    # Excel error values
    if s.startswith("#") or s.upper() in {"N/A", "NA"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _is_block_header(row: list[str]) -> bool:
    """Header row signature: col A = subject name, col E = 'ID', cols B-D = ints."""
    if len(row) < 6:
        return False
    if row[4].strip() != "ID":
        return False
    if not row[0].strip():
        return False
    try:
        int(row[1]); int(row[2]); int(row[3])
    except (ValueError, IndexError):
        return False
    return True


def parse_raw(csv_path: Path) -> list[RawBlock]:
    """Parse a session metadata CSV → list of RawBlock (one per Set,Trial,Repeat block)."""
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)

    blocks: list[RawBlock] = []
    i = 0
    while i < len(rows):
        row = rows[i]
        if not _is_block_header(row):
            i += 1
            continue

        header = row
        # Expect the next 6 data rows in fixed order (tolerant of extra/blank rows between)
        data_rows: dict[str, list[str]] = {}
        j = i + 1
        while j < len(rows) and len(data_rows) < len(BLOCK_ROW_ORDER):
            r = rows[j]
            if len(r) < 5:
                j += 1
                continue
            key = r[4].strip()
            if key in BLOCK_ROW_ORDER and key not in data_rows:
                data_rows[key] = r
                j += 1
                continue
            # Non-matching row — stop parsing this block
            break

        if "\uc2dc\uc791" not in data_rows or "\ub05d" not in data_rows:
            i = j
            continue

        start_row = data_rows["\uc2dc\uc791"]
        end_row = data_rows["\ub05d"]
        note_row = data_rows.get("\ud2b9\uc774\uc0ac\ud56d", [""] * len(header))

        block = RawBlock(
            subject=row[0].strip(),
            set_no=int(row[1]),
            trial_no=int(row[2]),
            repeat_no=int(row[3]),
            stem_code=start_row[0].strip() or None,
            session_signal_sec=_parse_float(start_row[5]) if len(start_row) > 5 else None,
            raw_row_index=i,
        )

        # Pieces are in columns 6+ (index starts after the start-signal column at idx 5)
        # The number of piece columns = number of non-empty header cells from idx 6 onward
        max_col = max(len(header), len(start_row), len(end_row), len(note_row))
        for ci in range(6, max_col):
            pid_raw = _clean_cell(header[ci]) if ci < len(header) else ""
            if not pid_raw:
                continue
            # Skip mid-trial sync-signal annotations that live in piece-header
            # slots (e.g. "start-signal between 30-34", "start-signal between 15",
            # "14 (for motion)" — these are block-level metadata, not actual pieces).
            if "\uc2dc\uc791\uc2e0\ud638" in pid_raw or "motion\uc6a9" in pid_raw or "\uc0ac\uc6a9\uac00\ub2a5" in pid_raw:
                continue
            pid_clean, hand = _strip_hand_suffix(pid_raw)
            pid_clean = _normalize_piece_id(pid_clean)
            start_val = _clean_cell(start_row[ci]) if ci < len(start_row) else ""
            end_val = _clean_cell(end_row[ci]) if ci < len(end_row) else ""
            note_val_raw = note_row[ci] if ci < len(note_row) else ""
            note_val = _clean_note(note_val_raw)

            start_sec = _parse_float(start_val)
            end_sec = _parse_float(end_val)

            # Skip wholly empty piece cells (no times, no meaningful note)
            if start_sec is None and end_sec is None and not note_val:
                continue

            # Hand annotation stored separately (arrangement has no dedicated col,
            # but we keep it in note prefix so it survives when relevant).
            if hand is not None:
                note_val = (f"hand={hand}" + (f" | {note_val}" if note_val else "")).strip()

            block.pieces.append(Piece(
                id=pid_clean,
                start_sec=start_sec,
                end_sec=end_sec,
                note=note_val,
                column_label=pid_raw,
            ))

        blocks.append(block)
        i = j

    return blocks


# ────────────────────────────────────────────────────────────────
# Piece lookup — maps raw piece ID to (Group, Piece_num) metadata.
# Derived from the two existing arrangement files (C0/N2 + C1/N1) + generalized.
# ────────────────────────────────────────────────────────────────

# T1 technique-book pieces: raw ID → full metadata record
# Tempo / Feature / Feature_end values extracted from C0/N2 + C1/N1 arrangement
# ground truth (both agree on Tempo where populated). Feature columns appear
# only in C0 (Professional) arrangement exports.
T1_PIECE_META: dict[str, dict[str, str]] = {
    "4":          {"group": "Group 1", "piece_num": "13", "tempo": "",         "feature": "00:00", "feature_end": "00:14"},
    "6":          {"group": "Group 1", "piece_num": "14", "tempo": "82",       "feature": "00:00", "feature_end": "00:14"},
    "7":          {"group": "Group 1", "piece_num": "15", "tempo": "60~65",    "feature": "00:00", "feature_end": "00:17"},
    "8":          {"group": "Group 2", "piece_num": "16", "tempo": "95",       "feature": "00:01 - 00:03, 00:05 - (truncated at end)", "feature_end": ""},
    "10 - Forte": {"group": "Group 2", "piece_num": "18", "tempo": "",         "feature": "0:02",  "feature_end": "0:12"},
    "10 - Piano": {"group": "Group 2", "piece_num": "18", "tempo": "",         "feature": "",      "feature_end": ""},
    "13":         {"group": "Group 2", "piece_num": "19", "tempo": "140",      "feature": "",      "feature_end": ""},
    "14":         {"group": "Group 2", "piece_num": "20", "tempo": "200",      "feature": "",      "feature_end": ""},
    "15":         {"group": "Group 2", "piece_num": "21", "tempo": "118",      "feature": "",      "feature_end": ""},
    "16":         {"group": "Group 3", "piece_num": "22", "tempo": "100->115", "feature": "",      "feature_end": ""},
    "17":         {"group": "Group 3", "piece_num": "23", "tempo": "",         "feature": "",      "feature_end": ""},
    "18":         {"group": "Group 3", "piece_num": "24", "tempo": "",         "feature": "",      "feature_end": ""},
    "19":         {"group": "Group 3", "piece_num": "25", "tempo": "",         "feature": "",      "feature_end": ""},
    "27":         {"group": "Group 4", "piece_num": "29", "tempo": "100",      "feature": "",      "feature_end": ""},
    "28":         {"group": "Group 4", "piece_num": "30", "tempo": "130 (65)", "feature": "",      "feature_end": ""},
    "29":         {"group": "Group 4", "piece_num": "31", "tempo": "140",      "feature": "",      "feature_end": ""},
    "30":         {"group": "Group 4", "piece_num": "32", "tempo": "",         "feature": "",      "feature_end": ""},
    "34":         {"group": "Group 4", "piece_num": "36", "tempo": "",         "feature": "",      "feature_end": ""},
    "36":         {"group": "Group 4", "piece_num": "38", "tempo": "78",       "feature": "",      "feature_end": ""},
    "40":         {"group": "Group 5", "piece_num": "40", "tempo": "160",      "feature": "",      "feature_end": ""},
    "41 - Forte": {"group": "Group 5", "piece_num": "41", "tempo": "140 (70)", "feature": "",      "feature_end": ""},
    "41 - Piano": {"group": "Group 5", "piece_num": "41", "tempo": "",         "feature": "",      "feature_end": ""},
    "43":         {"group": "Group 5", "piece_num": "42", "tempo": "",         "feature": "",      "feature_end": ""},
    "44":         {"group": "Group 5", "piece_num": "43", "tempo": "",         "feature": "",      "feature_end": ""},
    "45":         {"group": "Group 5", "piece_num": "44", "tempo": "140",      "feature": "",      "feature_end": ""},
    "46":         {"group": "Group 6", "piece_num": "45", "tempo": "63",       "feature": "",      "feature_end": ""},
    "47":         {"group": "Group 6", "piece_num": "46", "tempo": "110",      "feature": "",      "feature_end": ""},
    "48":         {"group": "Group 6", "piece_num": "47", "tempo": "70",       "feature": "",      "feature_end": ""},
    "50":         {"group": "Group 6", "piece_num": "48", "tempo": "118",      "feature": "",      "feature_end": ""},
    "51":         {"group": "Group 6", "piece_num": "49", "tempo": "108",      "feature": "",      "feature_end": ""},
}

# T2 drill pieces: raw ID (header text) → (normalized label, Code/key)
# Arrangement uses underscore-joined labels like "Scale_Slow" but raw uses
# parenthesized " (slow)" style — normalize on output.
T2_PIECE_MAP: dict[str, tuple[str, str]] = {
    "Scale (slow)":               ("Scale_Slow",                   "C Major"),
    "Arpeggio (slow)":            ("Arpeggio_Slow",                "C Major"),
    "Octave Repeat (slow)":       ("Octave Repeat_Slow",           "C"),
    "Octave Scale (slow)":        ("Octave Scale_Slow",            "C"),
    "Broken Arpeggio (slow)":     ("Broken Octave Arpeggio_Slow",  "C Major"),
    "Scale (fast)":               ("Scale_Fast",                   "C Major"),
    "Arpeggio (fast)":            ("Arpeggio_Fast",                "C Major"),
    "Octave Repeat (fast)":       ("Repetition_Fast",              "C"),       # arrangement renamed
    "Octave Scale (fast)":        ("Octave Scale_Fast",            "C"),
    "Broken Arpeggio (fast)":     ("Broken Octave Arpeggio_Fast",  "C Major"),
}

# T2 default tempos (BPM) by normalized label — from arrangement ground truth
T2_TEMPO_DEFAULT: dict[str, str] = {
    "Scale_Slow": "100", "Arpeggio_Slow": "100",
    "Octave Repeat_Slow": "80", "Octave Scale_Slow": "80",
    "Broken Octave Arpeggio_Slow": "80",
    "Scale_Fast": "140", "Arpeggio_Fast": "140",
    "Repetition_Fast": "140", "Octave Scale_Fast": "140",
    "Broken Octave Arpeggio_Fast": "140",
}

# T3 applied pieces — arrangement uses raw ID as-is (Octave Repeat 1/2/3,
# Beethoven 1/2). No standard Tempo/Code.
T3_PIECE_IDS = {"Octave Repeat 1", "Octave Repeat 2", "Octave Repeat 3",
                "Beethoven 1", "Beethoven 2"}


@dataclass
class ArrangementRow:
    group: str = ""
    piece_label: str = ""    # ID column in arrangement
    code: str = ""
    task: str = ""           # "Technique1" (textbook 30 pieces) / "Technique2" (drills + applied 15 pieces)
    start_sec: float | None = None
    end_sec: float | None = None
    filename: str = ""
    renamed: str = ""
    tempo: str = ""
    piece_num: str = ""      # numeric piece book ID for Technique1, empty otherwise
    feature: str = ""
    feature_end: str = ""
    memo: str = ""


def _sort_key_t1(row: ArrangementRow) -> tuple:
    """T1 sort: by Group then Piece_num, with Forte before Piano on same piece."""
    group_num = int(re.match(r"Group\s*(\d+)", row.group).group(1)) if row.group else 99
    piece_num = int(row.piece_num) if row.piece_num else 999
    # Forte < Piano for same piece
    dyn = 0 if "Forte" in row.piece_label else (1 if "Piano" in row.piece_label else 0)
    return (group_num, piece_num, dyn)


def build_arrangement_rows(
    blocks: list[RawBlock],
    *,
    filename_map: dict[tuple[int, int, int], str],
    renamed_map: dict[tuple[int, int, int], str],
    include_feature_cols: bool = False,
    time_convention: str = "file_local",
) -> list[ArrangementRow]:
    """Convert raw blocks → arrangement rows, resolving piece lookups.

    filename_map: (set, trial, repeat) → raw motion CSV basename
    renamed_map:  (set, trial, repeat) → canonical stem (e.g. 220819_C0_N2_T1_t1)
    time_convention:
        "file_local"     (220819) — raw start/end already in motion-file-local time.
        "session_global" (221112) — raw start/end are session-continuous timestamps;
                                    subtract the block's start-signal to get file-local time.
    """
    out: list[ArrangementRow] = []

    # Group rows per Renamed so we can sort by piece order within each trial unit
    rows_per_renamed: dict[str, list[ArrangementRow]] = {}

    for block in blocks:
        key = (block.set_no, block.trial_no, block.repeat_no)
        fname = filename_map.get(key, "")
        renamed = renamed_map.get(key, "")
        if not renamed:
            continue

        # Rebase raw times to file-local if session uses global timestamps
        offset = 0.0
        if time_convention == "session_global" and block.session_signal_sec is not None:
            offset = block.session_signal_sec

        for pc in block.pieces:
            s = pc.start_sec - offset if pc.start_sec is not None else None
            e = pc.end_sec - offset if pc.end_sec is not None else None
            row = ArrangementRow(
                start_sec=s,
                end_sec=e,
                filename=fname,
                renamed=renamed,
                memo=pc.note,
            )

            # Semantic Task label (independent of raw Set value):
            #   Technique1 = textbook 30 pieces (Group 1-6, numeric IDs)
            #   Technique2 = drills (Set 2) + applied (Set 3) — 15 pieces total
            if block.set_no == 1 and pc.id in T1_PIECE_META:
                meta = T1_PIECE_META[pc.id]
                row.piece_label = pc.id
                row.group = meta["group"]
                row.piece_num = meta["piece_num"]
                row.tempo = meta["tempo"]
                row.feature = meta["feature"]
                row.feature_end = meta["feature_end"]
                row.task = "Technique1"
            elif block.set_no == 2 and pc.id in T2_PIECE_MAP:
                label, code = T2_PIECE_MAP[pc.id]
                row.piece_label = label
                row.code = code
                row.tempo = T2_TEMPO_DEFAULT.get(label, "")
                row.task = "Technique2"
            elif block.set_no == 3 or pc.id in T3_PIECE_IDS:
                row.piece_label = pc.id
                row.task = "Technique2"
            else:
                # Unknown piece — keep raw label; infer Task from context
                row.piece_label = pc.id
                if block.set_no == 1:
                    row.task = "Technique1"
                elif block.set_no in (2, 3):
                    row.task = "Technique2"

            rows_per_renamed.setdefault(renamed, []).append(row)

    # Sort pieces within each Renamed block in canonical order
    for renamed, rows in rows_per_renamed.items():
        if any(r.group for r in rows):  # T1-like
            rows.sort(key=_sort_key_t1)

    # Flatten by (set, trial, repeat) sort key so takes of the same logical
    # trial cluster together (e.g. for C1 split mode: T1-1_t1, T1-1_t2 before
    # T1-2_t1). For concat mode this is equivalent to raw-block order.
    renamed_min_key: dict[str, tuple] = {}
    for block in blocks:
        key = (block.set_no, block.trial_no, block.repeat_no)
        renamed = renamed_map.get(key, "")
        if not renamed:
            continue
        # For concat mode, multiple blocks share a Renamed — take the min key
        # (= earliest Set/Trial/Repeat) as the representative sort anchor.
        # Sort key: (set, trial, repeat) — groups takes of the same trial
        # together (T1-1_t1, T1-1_t2, T1-2_t1, …). For concat mode all blocks
        # under one Renamed share (set, repeat) anyway, so the result is also
        # in-order.
        cand = (block.set_no, block.trial_no, block.repeat_no)
        if renamed not in renamed_min_key or cand < renamed_min_key[renamed]:
            renamed_min_key[renamed] = cand

    for renamed, _ in sorted(renamed_min_key.items(), key=lambda x: x[1]):
        out.extend(rows_per_renamed.get(renamed, []))

    # Feature/Feature_end are ground-truth populated only on the FIRST occurrence
    # of each piece ID (curator convention: metadata applies once per piece,
    # subsequent takes leave the columns blank).
    first_seen: set[str] = set()
    for row in out:
        key = f"{row.group}|{row.piece_label}"
        if key in first_seen:
            row.feature = ""
            row.feature_end = ""
        else:
            first_seen.add(key)

    return out


# ────────────────────────────────────────────────────────────────
# Curator overrides — manual edits applied on top of raw conversion.
# ────────────────────────────────────────────────────────────────
# Overrides file format (JSON, co-located next to raw CSV):
#
#   {
#     "edits": [
#       {
#         "renamed": "220819_C0_N2_T1_t1",   # required
#         "piece_id": "30",                   # required, matches arrangement "ID"
#         "match":   {"start_sec": 402.5},   # optional disambiguator
#         "set":     {"end_sec": 409.6, "memo": "…"},  # fields to overwrite
#         "reason":  "fix typo in raw end_sec"  # free-form, not applied
#       },
#       ...
#     ]
#   }
#
# Applied AFTER build_arrangement_rows. If match is omitted and multiple rows
# share (renamed, piece_id), ALL matches are edited (useful for Memo blanking).

_OVERRIDE_FIELD_MAP = {
    "start_sec": "start_sec",
    "end_sec":   "end_sec",
    "memo":      "memo",
    "group":     "group",
    "code":      "code",
    "tempo":     "tempo",
    "piece_num": "piece_num",
    "feature":   "feature",
    "feature_end": "feature_end",
}


def load_overrides(path: Path) -> list[dict]:
    """Load overrides JSON; returns list of edit dicts, or [] if file missing."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("edits", [])


def apply_overrides(rows: list[ArrangementRow], edits: list[dict]) -> int:
    """Apply curator edits in-place. Returns number of rows modified."""
    modified = 0
    for edit in edits:
        renamed = edit.get("renamed", "")
        piece_id = edit.get("piece_id", "")
        match = edit.get("match", {})
        set_fields = edit.get("set", {})

        for row in rows:
            if row.renamed != renamed or row.piece_label != piece_id:
                continue
            # Apply optional match conditions (disambiguation)
            if "start_sec" in match:
                if row.start_sec is None or abs(row.start_sec - float(match["start_sec"])) > 0.01:
                    continue
            # Apply sets
            for k, v in set_fields.items():
                target = _OVERRIDE_FIELD_MAP.get(k)
                if target is None:
                    continue
                if k in {"start_sec", "end_sec"}:
                    setattr(row, target, None if v is None else float(v))
                else:
                    setattr(row, target, "" if v is None else str(v))
            modified += 1
    return modified


# ────────────────────────────────────────────────────────────────
# Output writer
# ────────────────────────────────────────────────────────────────

ARR_HEADER_WITH_FEATURE = [
    "Group", "ID", "Code", "Task", "Start", "End", "Filename", "Renamed",
    "Tempo (to be measured at reference performer's tempo)",
    "Piece", "Feature", "Feature_end", "Memo",
]
ARR_HEADER_WITHOUT_FEATURE = [
    "Group", "ID", "Code", "Task", "Start", "End", "Filename", "Renamed",
    "Tempo (to be measured at reference performer's tempo)",
    "Piece", "Memo",
]


def _fmt_num(v: float | None) -> str:
    if v is None:
        return ""
    # Preserve integer form if integral
    if v == int(v) and abs(v) < 1e12:
        return str(int(v))
    # Strip trailing zeros
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return s


def write_arrangement(
    rows: list[ArrangementRow],
    out_path: Path,
    include_feature_cols: bool = False,
) -> None:
    header = ARR_HEADER_WITH_FEATURE if include_feature_cols else ARR_HEADER_WITHOUT_FEATURE
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            base = [
                r.group, r.piece_label, r.code, r.task,
                _fmt_num(r.start_sec), _fmt_num(r.end_sec),
                r.filename, r.renamed,
                r.tempo, r.piece_num,
            ]
            if include_feature_cols:
                base.extend([r.feature, r.feature_end])
            base.append(r.memo)
            w.writerow(base)
