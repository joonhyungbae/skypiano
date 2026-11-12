"""Single-source-of-truth registry of known data absences.

Backed by `pipeline/_common/known_absences.json`. Pipeline code must consult
this registry before logging a missing artifact as a failure: any absence
recorded here is a permanent characteristic of the raw capture (camera off,
marker dislodge, sensor failure) and should be treated as expected.

Typical use:

    from pipeline._common.known_absences import is_view_absent, expected_views

    if is_view_absent(ds="240828", trial_stem=ts, view="Left"):
        log.info("[Left] expected-absent: %s", absence_reason(ds, ts, "Left"))
        continue

    coverage = f"{n}/{len(expected_views(ds, ts))}"
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

ALL_VIEWS = ("Top", "Left", "Right", "RightLow")

_REGISTRY_PATH = Path(__file__).with_name("known_absences.json")


@lru_cache(maxsize=1)
def _load() -> dict:
    return json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))


def _entries(kind: str) -> list[dict]:
    return [e for e in _load().get("absences", []) if e.get("kind") == kind]


def is_view_absent(ds: str, trial_stem: str, view: str) -> bool:
    """Return True if (ds, trial_stem, view) is a known camera absence."""
    for e in _entries("video_view"):
        if e["scope"].get("ds") != ds:
            continue
        sel = e["selectors"]
        if trial_stem in sel.get("trial_stems", []) and view in sel.get("views", []):
            return True
    return False


def absent_views(ds: str, trial_stem: str) -> set[str]:
    """All known-absent views for a trial."""
    out: set[str] = set()
    for e in _entries("video_view"):
        if e["scope"].get("ds") != ds:
            continue
        sel = e["selectors"]
        if trial_stem in sel.get("trial_stems", []):
            out.update(sel.get("views", []))
    return out


def expected_views(ds: str, trial_stem: str, all_views: Iterable[str] = ALL_VIEWS) -> tuple[str, ...]:
    """Views that should exist for this trial after subtracting known absences."""
    absent = absent_views(ds, trial_stem)
    return tuple(v for v in all_views if v not in absent)


def absence_reason(ds: str, trial_stem: str, view: str) -> str | None:
    for e in _entries("video_view"):
        if e["scope"].get("ds") != ds:
            continue
        sel = e["selectors"]
        if trial_stem in sel.get("trial_stems", []) and view in sel.get("views", []):
            return e.get("reason", "")
    return None


def body_absence_entry(ds: str, trial_stem: str) -> dict | None:
    """Full registry entry when a trial's body-motion stream is source-missing."""
    for e in _entries("body_data"):
        if e["scope"].get("ds") != ds:
            continue
        if trial_stem in e["selectors"].get("trial_stems", []):
            return e
    return None


def body_absence_reason(ds: str, trial_stem: str) -> str | None:
    """Human-readable reason for a source-missing body-motion trial."""
    entry = body_absence_entry(ds, trial_stem)
    if entry is None:
        return None
    return entry.get("reason", "")


def is_body_absent(ds: str, trial_stem: str) -> bool:
    """Body-motion stream missing at the source for a trial."""
    return body_absence_entry(ds, trial_stem) is not None


def hand_marker_status(ds: str, trial_stem: str | None = None,
                       cohort: str | None = None,
                       pool_id: str | None = None,
                       task: str | None = None) -> dict | None:
    """Return the matching hand_marker absence entry (or None).

    Match by trial_stem when given; else by (cohort, pool_id, task) tuple.
    The returned dict contains the full registry entry including 'recovered_in'
    if the absence was downstream-recovered.
    """
    for e in _entries("hand_marker"):
        if e["scope"].get("ds") != ds:
            continue
        sel = e["selectors"]
        if trial_stem is not None and trial_stem in sel.get("trial_stems", []):
            return e
        if (cohort is not None and e["scope"].get("cohort") == cohort
                and pool_id is not None and sel.get("pool_id") == pool_id
                and (task is None or sel.get("task") == task)):
            return e
    return None


def is_trial_skipped(ds: str, trial_stem: str, stage: str | None = None) -> bool:
    for e in _entries("trial_skip"):
        if e["scope"].get("ds") != ds:
            continue
        sel = e["selectors"]
        if trial_stem not in sel.get("trial_stems", []):
            continue
        if stage is None or stage in sel.get("stages", []):
            return True
    return False


def is_audio_midi_absent(ds: str, trial_stem: str) -> bool:
    """True if the take's audio and MIDI are known not to be in the recording."""
    return any(e["scope"].get("ds") == ds and trial_stem in e["selectors"].get("trial_stems", [])
               for e in _entries("audio_midi"))


def all_entries(kind: str | None = None) -> list[dict]:
    """Return all absence entries, optionally filtered by kind."""
    if kind is None:
        return list(_load().get("absences", []))
    return _entries(kind)


def restored_marker_config(ds: str) -> tuple[list[str], dict | None]:
    """Aggregate `marker_restored` entries for `ds` into the legacy
    (missing_files, missing_restore) shape that
    `pipeline/02_spatial/hand/src/markers.py` DATASET_CONFIGS exposes
    and that `s02_restore_markers.py` consumes.

    Returns:
        files: union of all `selectors.filenames` across matching entries
        restore: {target_cols, input_cols} from the entries; None if none
                 match. All matching entries for one ds must share the same
                 recovery target/input columns; mixed targets raise ValueError.
    """
    files: list[str] = []
    restore: dict | None = None
    for e in _entries("marker_restored"):
        if e["scope"].get("ds") != ds:
            continue
        files.extend(e["selectors"].get("filenames", []))
        rec = e.get("recovery") or {}
        cur = {
            "target_cols": list(rec.get("target_cols", [])),
            "input_cols": list(rec.get("input_cols", [])),
        }
        if not cur["target_cols"]:
            continue
        if restore is None:
            restore = cur
        elif restore != cur:
            raise ValueError(
                f"known_absences: ds={ds} has marker_restored entries with "
                f"conflicting recovery configs: {restore} vs {cur}"
            )
    return files, restore
