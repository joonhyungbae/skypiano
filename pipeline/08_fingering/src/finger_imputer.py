"""
Tier C of the fingering pipeline: a BiLSTM that fills the notes Tiers A and B
left unresolved.

The model reads a performance as a note sequence (sorted by onset) and, per
note, sees the geometry-derived finger label (or a mask token where Tiers A/B
did not commit), the MIDI pitch, the inter-onset interval and the duration.
It was trained in a masked-fill fashion: labeled positions were masked at
random and the model learned to recover them from context (SKY-Piano
algorithm labels plus PIG as a symbolic prior; 84.3% recovery of held-out
algorithm labels on a 10% trial-level holdout).

Inference on an unresolved note:
  * multi_candidate: the softmax is restricted to the note's geometric
    candidate set, so the prediction is always a geometrically plausible
    finger;
  * no_candidate: the softmax runs over all ten fingers.

Token layout (shared with training): 0 = null, 1..5 = left thumb..little,
6..10 = right thumb..little, 11 = mask. The head scores classes 0..10;
class 0 is never predicted.

Feature layout (fixed by the trained weights): concat(finger embedding (64),
pitch embedding (32), log1p(IOI s), log1p(duration s)) -> linear 256 -> ReLU
-> 3-layer bidirectional LSTM (256 per direction) -> LayerNorm -> linear 11.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

NULL_TOKEN = 0
MASK_TOKEN = 11
N_TOKENS = 12
N_CLASSES = 11


def encode(hand: Optional[str], finger: Optional[int]) -> int:
    """('Left'|'Right', 1..5) -> token 1..10; anything else -> null."""
    if hand not in ("Left", "Right") or finger is None:
        return NULL_TOKEN
    return int(finger) if hand == "Left" else int(finger) + 5


def decode(token: int) -> tuple[str, int]:
    """Token 1..10 -> ('Left'|'Right', finger 1..5)."""
    return ("Left", token) if token <= 5 else ("Right", token - 5)


class FingerBiLSTM(nn.Module):
    """The Tier C network, shared by inference and steps/s03_train_tier_c.py.

    `times` is (B, T, 2): log1p of the inter-onset interval and of the
    duration, both in seconds. Dropout acts between LSTM layers in training
    only; padding positions use the null token and pitch 0.
    """

    def __init__(self, hidden: int = 256, layers: int = 3, proj: int = 256,
                 dropout: float = 0.0):
        super().__init__()
        self.finger_emb = nn.Embedding(N_TOKENS, 64, padding_idx=NULL_TOKEN)
        self.pitch_emb = nn.Embedding(128, 32, padding_idx=0)
        self.proj = nn.Linear(64 + 32 + 2, proj)
        self.lstm = nn.LSTM(proj, hidden, layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.norm = nn.LayerNorm(2 * hidden)
        self.head = nn.Linear(2 * hidden, N_CLASSES)

    def forward(self, tokens, pitches, times):
        x = torch.cat([self.finger_emb(tokens), self.pitch_emb(pitches), times], dim=-1)
        h, _ = self.lstm(torch.relu(self.proj(x)))
        return self.head(self.norm(h))


class FingerImputer:
    """Loads the Tier C weights once and fills unresolved notes of a sequence."""

    def __init__(self, checkpoint: Path, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        state = ck.get("model", ck)
        self.model = FingerBiLSTM()
        self.model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        self.model.eval().to(self.device)

    @torch.no_grad()
    def predict(self, notes: list[dict]) -> list[Optional[tuple[int, float]]]:
        """Fill the unresolved notes of one performance.

        Args:
            notes: sorted by onset; each has onset, offset, pitch, token
                (1..10 when Tiers A/B committed, else MASK_TOKEN) and
                candidates (tokens allowed for a multi-candidate note; empty
                means all ten fingers).

        Returns:
            per note, (token, probability) for masked notes, None otherwise.
        """
        if not notes:
            return []
        tokens, pitches, times, prev = [], [], [], notes[0]["onset"]
        for n in notes:
            tokens.append(n["token"])
            pitches.append(min(127, max(0, int(n["pitch"]))))
            times.append([math.log1p(max(0.0, n["onset"] - prev)),
                          math.log1p(max(0.0, n["offset"] - n["onset"]))])
            prev = n["onset"]
        logits = self.model(torch.tensor([tokens], device=self.device),
                            torch.tensor([pitches], device=self.device),
                            torch.tensor([times], dtype=torch.float, device=self.device))[0].cpu()
        out: list[Optional[tuple[int, float]]] = []
        for i, n in enumerate(notes):
            if n["token"] != MASK_TOKEN:
                out.append(None)
                continue
            allowed = torch.zeros(N_CLASSES, dtype=torch.bool)
            cands = [c for c in n.get("candidates") or [] if 1 <= c <= 10]
            allowed[cands if cands else list(range(1, 11))] = True
            prob = logits[i].masked_fill(~allowed, float("-inf")).softmax(-1)
            tok = int(prob.argmax())
            out.append((tok, float(prob[tok])))
        return out
