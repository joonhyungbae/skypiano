"""
Per-performer mapping configuration (Stage 04 per-piece based)

C0 = professional pianist, C1 = amateur pianist
N{X} = unique identifier

The same C/N refers to the same performer regardless of dataset.
Imputation must be performed per performer (bone lengths differ).

Mapped at the Stage 04 per-piece granularity. Each piece has a prefix of the form {trial_stem}_p{N}.
"""
import json
import os
from pathlib import Path
from collections import defaultdict


PIECE_SPLIT_ROOT = Path("Data/Pipeline/04_piece_split")


def _parse_performer_from_trial_stem(dataset, trial_stem):
    """
    Extract (C, N) from trial_stem.
    Standard format: {yymmdd}_{C?}_{N?}_T{task}_P{piece}_t{trial}
    """
    parts = trial_stem.split("_")
    if len(parts) < 3:
        return None
    c, n = parts[1], parts[2]
    if not (c.startswith("C") and n.startswith("N")):
        return None
    return (c, n)


def build_performer_map():
    """
    Returns: dict of (C, N) -> list of (dataset, piece_prefix)
    where piece_prefix = "{trial_stem}_p{N}"
    """
    performers = defaultdict(list)

    for ds_dir in sorted(PIECE_SPLIT_ROOT.iterdir()):
        if not ds_dir.is_dir():
            continue
        dataset = ds_dir.name
        # Skip non-dataset subdirs (anchors_220819, references) — only dataset dirs (YYMMDD)
        if not (dataset.isdigit() and len(dataset) == 6):
            continue

        manifest_path = ds_dir / "_pieces.json"
        if not manifest_path.is_file():
            # Datasets whose pieces are cut outside 04_piece_split (220819:
            # 04_piece_split/anchors_220819 s03_slice) have no _pieces.json; use the piece files.
            for hand in sorted(ds_dir.glob("*_p*_hand.csv")):
                piece_prefix = hand.name[: -len("_hand.csv")]
                trial_stem = piece_prefix.rsplit("_p", 1)[0]
                cn = _parse_performer_from_trial_stem(dataset, trial_stem)
                if cn is not None:
                    performers[cn].append((dataset, piece_prefix))
            continue

        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        for entry in manifest:
            trial_stem = entry["trial_stem"]
            cn = _parse_performer_from_trial_stem(dataset, trial_stem)
            if cn is None:
                continue
            for p in entry.get("pieces", []):
                if p.get("excluded"):
                    continue
                idx = p["piece_idx"]
                piece_prefix = f"{trial_stem}_p{idx}"
                performers[cn].append((dataset, piece_prefix))

    return dict(performers)


def performer_id(c, n):
    return f"{c}_{n}"


def get_piece_hand_path(dataset, piece_prefix):
    """Stage 04 piece-level hand CSV path"""
    return PIECE_SPLIT_ROOT / dataset / f"{piece_prefix}_hand.csv"


def get_piece_body_path(dataset, piece_prefix):
    """Stage 04 piece-level body CSV path (None if it does not exist)"""
    p = PIECE_SPLIT_ROOT / dataset / f"{piece_prefix}_body.csv"
    return p if p.exists() else None


# Backward compat alias
get_trial_path = get_piece_hand_path


def _restrict(performers):
    """Honour PERFORMER_ONLY="C0_N2,C1_N1" so every step (including those
    without --only/--skip) processes just those performers."""
    only = os.environ.get("PERFORMER_ONLY", "").strip()
    if not only:
        return performers
    keep = {tuple(p.split("_", 1)) for p in only.split(",") if p.strip()}
    return {cn: v for cn, v in performers.items() if cn in keep}


# Prebuilt map for import
PERFORMER_MAP = _restrict(build_performer_map())


if __name__ == "__main__":
    total = 0
    for (c, n), pieces in sorted(PERFORMER_MAP.items()):
        datasets = defaultdict(int)
        for ds, _ in pieces:
            datasets[ds] += 1
        ds_str = ", ".join(f"{ds}:{cnt}" for ds, cnt in sorted(datasets.items()))
        pid = performer_id(c, n)
        print(f"  {pid:8s}: {len(pieces):4d} pieces  [{ds_str}]")
        total += len(pieces)
    print(f"\n  Total: {len(PERFORMER_MAP)} performers, {total} pieces")
