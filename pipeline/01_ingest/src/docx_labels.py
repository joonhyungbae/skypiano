"""
Parser for the 240828/240809 trial labeling docx.

The docx (authored manually) lists every motion trial per subject, grouped
under a task-group label (for 240828: "technique1" / "technique2" / "Set1" /
"Free"; for 240809 the raw blocks are written as opaque tokens "PJH1" /
"PJH2"). Each trial has usability notes in free-form localized text.

We normalize this into a structured JSON:

    {
      "240828": {
        "sub04": {
          "0401": {"task": 0, "status": "ok", "exclusions": [], "notes": "..."},
          ...
        },
        ...
      },
      "240809": {
        "S01": {
          "0103": {"task": 2, "status": "ok", ...},
          ...
        }
      }
    }

Exclusions are pairs of [start_sec, end_sec] describing sub-intervals of the
session recording that should be ignored (notes phrased as "HH:MM-HH:MM may be
discarded" in the source docx).
Status is one of:
  "ok"              -- usable, possibly with partial exclusions
  "discarded"       -- entire trial marked as "discard wholly / discard all"
  "unknown"         -- notes present but not parsed

Run this module directly to regenerate the JSON cache; other Stage 01 code
imports `load_labels()`.
"""
from __future__ import annotations
import json
import re
import zipfile
from pathlib import Path

_METADATA_DIR = Path("Data/Metadata/240828")
# Match the labeling docx by suffix so any local prefix is fine.
_docx_matches = sorted(_METADATA_DIR.glob("*240828 trial labeling.docx"))
DOCX_PATH = _docx_matches[0] if _docx_matches else _METADATA_DIR / "240828 trial labeling.docx"
OUT_JSON = _METADATA_DIR / "trial_labels.json"

GROUP_TO_TASK = {
    "technique1": 0,
    "technique2": 1,
    "set1": 2,
    "\uc790\uc720\uace1": 9,
}
GROUP_LABELS = set(GROUP_TO_TASK.keys())

# 240809 note: every trial in this cohort is Task 2 regardless of which raw block (P1/P2) it appears in.
TASK_240809 = 2


def _extract_paragraphs(docx_path: Path) -> list[str]:
    with zipfile.ZipFile(docx_path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    paras: list[str] = []
    cur: list[str] = []
    for m in re.finditer(r"<w:p[\s>]|<w:t[^>]*>([^<]*)</w:t>", xml):
        if m.group(0).startswith("<w:p"):
            if cur:
                paras.append("".join(cur))
                cur = []
        else:
            cur.append(m.group(1))
    if cur:
        paras.append("".join(cur))
    return [p.strip() for p in paras if p.strip()]


def _parse_timecode(tc: str) -> float:
    """HH:MM or MM:SS -> seconds."""
    parts = [int(x) for x in tc.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"bad timecode: {tc}")


_TIME_RE = re.compile(r"(\d{1,2}:\d{2}(?::\d{2})?)")
_START, _END = "\ucc98\uc74c", "\ub05d"       # "the start", "the end" in the notes
_TC = r"\d{1,2}:\d{2}(?::\d{2})?"
# "<start>" may carry the start time in brackets: "<start>(16:36)-17:15"
_RANGE_RE = re.compile(rf"({_TC}|{_START}(?:\s*\(\s*{_TC}\s*\))?)\s*[-~]\s*({_TC}|{_END})")
_EXCLUDE_LINE_RE = re.compile(r"\ub0a0\ub824\ub3c4 \ub428|\ubc84\ub824\ub3c4 \ub428|\ubc84\ub824\uc57c")


def _parse_trial_notes(lines: list[str], session_end_sec: float | None = None) -> tuple[str, list[list[float]], str]:
    """Return (status, exclusions, raw_notes_joined).

    lines: the lines that appeared AFTER the trial number but before the next.
    """
    joined = " | ".join(lines)
    if not lines:
        return "unknown", [], ""

    status = "ok"
    exclusions: list[list[float]] = []

    for ln in lines:
        low = ln.lower()
        if "\ubaa8\ub450 ok" in ln or low.endswith("ok") and "\ub0a0\ub824" not in ln:
            continue
        if "\ud1b5\uc9f8\ub85c" in ln or "\ub2e4 \ub0a0\ub824" in ln or "\ub2e4 \ubc84\ub824" in ln:
            status = "discarded"
            continue
        if _EXCLUDE_LINE_RE.search(ln):
            # Every range on the line: "HH:MM-HH:MM", "<start>-HH:MM" (from the
            # start), "HH:MM-<end>" (to the end); ranges are read one by one, so
            # "<start>-14:57, 15:50-16:16" gives [0, 14:57] and [15:50, 16:16].
            for a_tok, b_tok in _RANGE_RE.findall(ln):
                try:
                    if a_tok.startswith(_START):
                        given = _TIME_RE.search(a_tok)
                        a = _parse_timecode(given.group(1)) if given else 0.0
                    else:
                        a = _parse_timecode(a_tok)
                    if b_tok == _END:
                        if session_end_sec is None:
                            continue
                        b = session_end_sec
                    else:
                        b = _parse_timecode(b_tok)
                except ValueError:
                    continue
                if b > a:
                    exclusions.append([a, b])
    return status, exclusions, joined


def parse_240828(paras: list[str]) -> dict:
    """Extract per-trial labels for 240828 (sub04..sub08)."""
    out: dict[str, dict] = {}
    current_sub: str | None = None
    current_task: int | None = None
    # Find the 240828 section
    end_idx = None
    for i, p in enumerate(paras):
        if "240809" in p:
            end_idx = i
            break
    section = paras[:end_idx] if end_idx else paras

    i = 0
    while i < len(section):
        p = section[i].strip()
        low = p.lower()
        m_sub = re.match(r"^sub(\d+)", low)
        if m_sub:
            current_sub = f"sub{int(m_sub.group(1)):02d}"
            out.setdefault(current_sub, {})
            current_task = None
            i += 1
            continue
        if low in GROUP_LABELS or p.strip() in GROUP_LABELS:
            current_task = GROUP_TO_TASK[p.strip().lower() if low in GROUP_LABELS else p.strip()]
            i += 1
            continue
        # Trial number detection: the first token should be NNNN maybe with "(o)" etc.
        m_tr = re.match(r"^(\d{4})\b(.*)$", p)
        if m_tr and current_sub:
            trial_nn = m_tr.group(1)
            head_tail = m_tr.group(2).strip()
            note_lines: list[str] = []
            if head_tail:
                note_lines.append(head_tail)
            j = i + 1
            while j < len(section):
                nxt = section[j].strip()
                if re.match(r"^(\d{4})\b", nxt):
                    break
                if nxt.lower() in GROUP_LABELS or re.match(r"^sub\d+", nxt.lower()):
                    break
                note_lines.append(nxt)
                j += 1
            status, exclusions, notes = _parse_trial_notes(note_lines)
            out[current_sub][trial_nn] = {
                "task": current_task if current_task is not None else -1,
                "status": status,
                "exclusions": exclusions,
                "notes": notes,
            }
            i = j
            continue
        i += 1
    return out


def parse_240809(paras: list[str]) -> dict:
    """Extract per-trial labels for 240809 (single-cohort, raw blocks P1 / P2)."""
    out: dict[str, dict] = {"S01": {}}
    start = None
    for i, p in enumerate(paras):
        if "240809" in p:
            start = i
            break
    if start is None:
        return out
    section = paras[start:]

    current_block: str | None = None
    i = 0
    while i < len(section):
        p = section[i].strip()
        if p == "PJH1":
            current_block = "01"
            i += 1
            continue
        if p == "PJH2":
            current_block = "02"
            i += 1
            continue
        # Block-P1 stems: full per-trial form is PJH1NN where NN is the index.
        m_full = re.match(r"^PJH1(\d{2})\b(.*)$", p)
        if m_full and current_block == "01":
            nn = m_full.group(1)
            note_lines: list[str] = [m_full.group(2).strip()] if m_full.group(2).strip() else []
            j = i + 1
            while j < len(section):
                nxt = section[j].strip()
                if re.match(r"^(PJH1\d|20\d)\b", nxt) or nxt.startswith("PJH"):
                    break
                note_lines.append(nxt)
                j += 1
            status, exclusions, notes = _parse_trial_notes(note_lines)
            out["S01"][f"01{nn}"] = {
                "task": TASK_240809, "status": status,
                "exclusions": exclusions, "notes": notes,
            }
            i = j
            continue
        # Block-P2 stems: bare "2NN" form (no "PJH2" prefix on each trial).
        m_pjh2 = re.match(r"^2(\d{2})\b(.*)$", p)
        if m_pjh2 and current_block == "02":
            nn = m_pjh2.group(1)
            note_lines: list[str] = [m_pjh2.group(2).strip()] if m_pjh2.group(2).strip() else []
            j = i + 1
            while j < len(section):
                nxt = section[j].strip()
                if re.match(r"^2\d{2}\b", nxt) or nxt.startswith("PJH"):
                    break
                note_lines.append(nxt)
                j += 1
            status, exclusions, notes = _parse_trial_notes(note_lines)
            out["S01"][f"02{nn}"] = {
                "task": TASK_240809, "status": status,
                "exclusions": exclusions, "notes": notes,
            }
            i = j
            continue
        i += 1
    return out


def parse_all() -> dict:
    paras = _extract_paragraphs(DOCX_PATH)
    return {
        "240828": parse_240828(paras),
        "240809": parse_240809(paras),
    }


def load_labels() -> dict:
    if OUT_JSON.exists():
        return json.load(open(OUT_JSON, encoding="utf-8"))
    data = parse_all()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    data = parse_all()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"wrote {OUT_JSON}")
    # Summary
    for ds, subs in data.items():
        print(f"\n{ds}:")
        for sub, trials in subs.items():
            counts: dict[str, int] = {}
            task_counts: dict[int, int] = {}
            for t in trials.values():
                counts[t["status"]] = counts.get(t["status"], 0) + 1
                task_counts[t["task"]] = task_counts.get(t["task"], 0) + 1
            print(f"  {sub}: {len(trials)} trials  tasks={task_counts}  status={counts}")
