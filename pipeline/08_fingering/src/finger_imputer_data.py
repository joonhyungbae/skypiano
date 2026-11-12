"""
Training data for the Tier C finger imputer (steps/s03_train_tier_c.py).

Two corpora, both reduced to the token layout of finger_imputer.py
(0 = null, 1..5 = left thumb..little, 6..10 = right thumb..little):

  SKY-Piano  Tier A output of steps/s01_pseudo_fingering.py,
             Data/Pipeline/08_fingering/Fingering/240828/*_fingering.json.
             Only notes with status 'labeled' count as labels; multi- and
             no-candidate notes enter the sequence as null tokens.
  PIG        The PIG fingering dataset (Nakamura et al. 2020), obtained
             separately: one *_fingering.txt per annotator and piece, lines
             "idx onset offset pitch_name velocity ? channel finger" with
             positive fingers for the right hand and negative for the left.
             Expected under Data/PianoFingeringDataset_v1.2/FingeringFiles/
             unless --pig-dir points elsewhere.

Each performance becomes one TrialSeq sorted by onset. The trainer masks a
share of the labeled positions and learns to recover them from context.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from finger_imputer import NULL_TOKEN, encode

SKY_FINGERING_DIR = Path("Data/Pipeline/08_fingering/Fingering/240828")
PIG_FINGERING_DIR = Path("Data/PianoFingeringDataset_v1.2/FingeringFiles")


@dataclass
class TrialSeq:
    source: str              # 'sky' or 'pig'
    name: str                # file stem
    tokens: List[int]        # 0..10 per note
    has_label: List[bool]    # True where the token is a recorded finger
    pitches: List[int]       # MIDI pitch, 0 when unparseable
    iois: List[float]        # seconds since the previous onset (0 for the first note)
    durations: List[float]   # seconds


def encode_pig(finger_field: str) -> int:
    """PIG finger field ('4', '-3', '4_1' for a substitution, '') -> token.

    A substitution keeps its first finger; anything unparseable is null.
    """
    head = finger_field.split("_", 1)[0].strip() if finger_field else ""
    try:
        v = int(head)
    except ValueError:
        return NULL_TOKEN
    if v == 0 or abs(v) > 5:
        return NULL_TOKEN
    return encode("Right" if v > 0 else "Left", abs(v))


_NOTE_NAME_RE = re.compile(r"^([A-Ga-g])([#bBxX]*)(-?\d+)$")


def note_name_to_midi(name: str) -> int:
    """PIG pitch names such as 'C4', 'F#5', 'Bb3' -> MIDI number (0 if unparseable)."""
    m = _NOTE_NAME_RE.match(name.strip())
    if not m:
        return 0
    letter, accidental, octave = m.group(1).upper(), m.group(2), int(m.group(3))
    semis = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter] + (octave + 1) * 12
    for ch in accidental:
        if ch == "#":
            semis += 1
        elif ch.lower() == "b":
            semis -= 1
        elif ch.lower() == "x":
            semis += 2
    return max(0, min(127, semis))


def _sequence(source: str, name: str, rows) -> Optional[TrialSeq]:
    """rows: (onset, offset, pitch, token), any order."""
    rows = sorted(rows, key=lambda r: r[0])
    if not rows:
        return None
    t = TrialSeq(source, name, [], [], [], [], [])
    prev = rows[0][0]
    for onset, offset, pitch, tok in rows:
        t.tokens.append(tok)
        t.has_label.append(tok != NULL_TOKEN)
        t.pitches.append(pitch)
        t.iois.append(max(0.0, onset - prev))
        t.durations.append(max(0.0, offset - onset))
        prev = onset
    return t


def load_sky_trials(root: Path = SKY_FINGERING_DIR) -> List[TrialSeq]:
    out: List[TrialSeq] = []
    for path in sorted(root.glob("*_fingering.json")):
        try:
            notes = json.loads(path.read_text(encoding="utf-8")).get("fingering", [])
        except Exception as e:
            print(f"[skip] {path.name}: {e}")
            continue
        rows = []
        for n in notes:
            onset = float(n.get("onset", 0.0))
            tok = encode(n.get("hand"), n.get("finger")) if n.get("status") == "labeled" else NULL_TOKEN
            rows.append((onset, float(n.get("offset", onset)), int(n.get("pitch", 0)), tok))
        seq = _sequence("sky", path.stem.removesuffix("_fingering"), rows)
        if seq:
            out.append(seq)
    return out


def load_pig_trials(root: Path = PIG_FINGERING_DIR) -> List[TrialSeq]:
    out: List[TrialSeq] = []
    for path in sorted(root.glob("*_fingering.txt")):
        rows = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                parts = line.split("\t")
                if len(parts) < 8:
                    parts = line.split()
                if len(parts) < 8:
                    continue
                rows.append((float(parts[1]), float(parts[2]),
                             note_name_to_midi(parts[3]), encode_pig(parts[7])))
        except Exception as e:
            print(f"[skip] {path.name}: {e}")
            continue
        seq = _sequence("pig", path.stem.removesuffix("_fingering"), rows)
        if seq:
            out.append(seq)
    return out


class FingerSeqDataset(Dataset):
    """One trial per item; collate() pads and truncates."""

    def __init__(self, trials: List[TrialSeq]):
        self.trials = trials

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, idx: int) -> TrialSeq:
        return self.trials[idx]


def collate(batch: List[TrialSeq], max_len: int = 1500):
    """Pad to the longest trial in the batch.

    Returns tokens (B, T) long, has_label (B, T) bool, pitches (B, T) long and
    times (B, T, 2) float = log1p(IOI), log1p(duration). Padding is null
    token, pitch 0 and has_label False, so it is never a training target.
    """
    n = [min(len(t.tokens), max_len) for t in batch]
    B, L = len(batch), max(n)
    tokens = torch.full((B, L), NULL_TOKEN, dtype=torch.long)
    has_label = torch.zeros(B, L, dtype=torch.bool)
    pitches = torch.zeros(B, L, dtype=torch.long)
    times = torch.zeros(B, L, 2, dtype=torch.float)
    for i, (t, k) in enumerate(zip(batch, n)):
        tokens[i, :k] = torch.tensor(t.tokens[:k])
        has_label[i, :k] = torch.tensor(t.has_label[:k])
        pitches[i, :k] = torch.tensor(t.pitches[:k])
        times[i, :k, 0] = torch.tensor([math.log1p(x) for x in t.iois[:k]])
        times[i, :k, 1] = torch.tensor([math.log1p(x) for x in t.durations[:k]])
    return tokens, has_label, pitches, times
