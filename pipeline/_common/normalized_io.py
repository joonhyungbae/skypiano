"""
Stage 01 output convention:

    Data/Pipeline/01_ingest/Normalized/{dataset}/{session}/
        motion/{trial_stem}.csv
        audio/{trial_stem}.wav
        midi/{trial_stem}.mid
        _mapping.json

Each session folder has a _mapping.json describing where each trial came from
(raw motion file, raw audio file, raw midi file, slicing offsets, method,
confidence).
"""
import json
import os
from pathlib import Path

NORMALIZED_ROOT = Path("Data/Pipeline/01_ingest/Normalized")
RAW_ROOT = Path("Data/Raw")
PROCESSED_ROOT = Path("Data/Processed")


def session_dir(dataset: str, session: str, modalities: list[str] | None = None) -> Path:
    """Create (if needed) and return the session output dir.

    modalities: list of subdirectories to ensure exist. Defaults to
    ``["motion", "audio", "midi"]`` for backwards compatibility with 211215.
    Datasets with split body/hand motion pass
    ``["motion_hand", "motion_body", "audio", "midi"]``.
    """
    p = NORMALIZED_ROOT / dataset / session
    if modalities is None:
        modalities = ["motion", "audio", "midi"]
    for m in modalities:
        (p / m).mkdir(parents=True, exist_ok=True)
    return p


def write_mapping(dataset: str, session: str, trials: list[dict], method: str, extra: dict | None = None):
    """Write _mapping.json with provenance info for a session."""
    sess_p = NORMALIZED_ROOT / dataset / session
    sess_p.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "session": session,
        "method": method,
        "trial_count": len(trials),
        "trials": trials,
    }
    if extra:
        payload.update(extra)
    with open(sess_p / "_mapping.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_mapping(dataset: str, session: str) -> dict | None:
    p = NORMALIZED_ROOT / dataset / session / "_mapping.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def all_session_mappings() -> dict[str, dict[str, dict]]:
    """Return {dataset: {session: mapping_dict}} across all normalized data."""
    result: dict[str, dict[str, dict]] = {}
    if not NORMALIZED_ROOT.exists():
        return result
    for ds_dir in NORMALIZED_ROOT.iterdir():
        if not ds_dir.is_dir():
            continue
        result[ds_dir.name] = {}
        for sess_dir in ds_dir.iterdir():
            if not sess_dir.is_dir():
                continue
            m = read_mapping(ds_dir.name, sess_dir.name)
            if m:
                result[ds_dir.name][sess_dir.name] = m
    return result
