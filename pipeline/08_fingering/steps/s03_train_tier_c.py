"""
Stage 08 Step 03: train the Tier C finger imputer (src/finger_imputer.py).

The BiLSTM learns to fill fingers from context: in every batch a share of the
labeled notes is replaced by the mask token, and the loss is cross-entropy on
those notes only. Data comes from src/finger_imputer_data.py:

  --data sky       SKY-Piano Tier A labels only
  --data pig       PIG only
  --data mix       SKY-Piano + PIG in one pool (the released model)
  --data pig+sky   pre-train on PIG, then fine-tune on SKY-Piano

10% of the trials of each source are held out (fixed seed). Validation uses
the SKY-Piano holdout whenever SKY-Piano is loaded, and reports the share of
masked labels recovered; the best epoch is saved.

The released weights (Data/Pipeline/08_fingering/models/tier_c_bilstm.pt,
read by s04_release_fingering.py) were trained with --data mix and the
default network size; their best holdout recovery was 84.3% at epoch 97.

Usage (from the repository root, after s01_pseudo_fingering.py):
  python pipeline/08_fingering/steps/s03_train_tier_c.py --data mix
         [--pig-dir DIR] [--out PATH] [--overwrite] [--epochs N] [--no-dp]
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from finger_imputer import MASK_TOKEN, FingerBiLSTM  # noqa: E402
from finger_imputer_data import (PIG_FINGERING_DIR, SKY_FINGERING_DIR,  # noqa: E402
                                 FingerSeqDataset, collate, load_pig_trials,
                                 load_sky_trials)

DEFAULT_OUT = Path("Data/Pipeline/08_fingering/models/tier_c_bilstm.pt")


def split_trials(trials, val_ratio=0.10, seed=20260427):
    idx = list(range(len(trials)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(len(idx) * val_ratio))
    return [trials[i] for i in idx[n_val:]], [trials[i] for i in idx[:n_val]]


def apply_mask(tokens, has_label, mask_rate):
    """Replace a random mask_rate share of the labeled positions by MASK_TOKEN."""
    m_pos = has_label & (torch.rand(tokens.shape, device=tokens.device) < mask_rate)
    masked = tokens.clone()
    masked[m_pos] = MASK_TOKEN
    return masked, m_pos


def _batches(loader, device):
    for tokens, has_label, pitches, times in loader:
        yield tokens.to(device), has_label.to(device), pitches.to(device), times.to(device)


def evaluate(model, loader, device, mask_rate, n_passes=4):
    """Share of masked labels recovered, averaged over n_passes random maskings."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_passes):
            for tokens, has_label, pitches, times in _batches(loader, device):
                masked, m_pos = apply_mask(tokens, has_label, mask_rate)
                pred = model(masked, pitches, times).argmax(dim=-1)
                correct += (pred[m_pos] == tokens[m_pos]).sum().item()
                total += int(m_pos.sum().item())
    return correct / max(1, total), total


def train_one(model, train_loader, val_loader, *, device, lr, epochs, mask_rate, save_path):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc, best_epoch = 0.0, -1
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum, n_steps = 0.0, 0
        for tokens, has_label, pitches, times in _batches(train_loader, device):
            masked, m_pos = apply_mask(tokens, has_label, mask_rate)
            if not m_pos.any():
                continue
            logits = model(masked, pitches, times)
            loss = torch.nn.functional.cross_entropy(logits[m_pos], tokens[m_pos])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item())
            n_steps += 1
        val_acc, val_n = evaluate(model, val_loader, device, mask_rate)
        print(f"epoch {ep:>3}  train_loss {loss_sum / max(1, n_steps):.4f}  "
              f"val_acc {val_acc * 100:5.2f}% (n={val_n})  {time.time() - t0:.1f}s")
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, ep
            save_path.parent.mkdir(parents=True, exist_ok=True)
            state = getattr(model, "module", model).state_dict()
            torch.save({"model": state, "epoch": ep, "val_acc": val_acc}, save_path)
    print(f"best val_acc {best_acc * 100:.2f}% at epoch {best_epoch} -> {save_path}")
    return best_acc


def loader(trials, batch_size, shuffle):
    return DataLoader(FingerSeqDataset(trials), batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collate)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", choices=["sky", "pig", "mix", "pig+sky"], default="mix")
    ap.add_argument("--sky-dir", type=Path, default=SKY_FINGERING_DIR)
    ap.add_argument("--pig-dir", type=Path, default=PIG_FINGERING_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="checkpoint path; pig+sky also writes <out>_pig.pt for its first stage")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing checkpoint at --out (the released weights by default)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--ft-epochs", type=int, default=15, help="fine-tune epochs for pig+sky")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mask-rate", type=float, default=0.28)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--no-dp", action="store_true", help="disable DataParallel")
    args = ap.parse_args()

    if args.out.exists() and not args.overwrite:
        sys.exit(f"{args.out} exists; pass --overwrite or a different --out")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sky = load_sky_trials(args.sky_dir) if args.data != "pig" else []
    pig = load_pig_trials(args.pig_dir) if args.data != "sky" else []
    print(f"device {device}  trials SKY={len(sky)} PIG={len(pig)}")
    if args.data != "pig" and not sky:
        sys.exit(f"no SKY-Piano fingering under {args.sky_dir}; run s01_pseudo_fingering.py first")
    if args.data != "sky" and not pig:
        sys.exit(f"no PIG files under {args.pig_dir}; pass --pig-dir")

    sky_train, sky_val = split_trials(sky) if sky else ([], [])
    pig_train, pig_val = split_trials(pig) if pig else ([], [])

    model = FingerBiLSTM(hidden=args.hidden, layers=args.layers, proj=args.hidden,
                         dropout=args.dropout).to(device)
    print(f"model params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    if torch.cuda.device_count() > 1 and not args.no_dp:
        model = torch.nn.DataParallel(model)

    common = dict(device=device, mask_rate=args.mask_rate)
    if args.data == "pig+sky":
        print("stage 1: PIG pre-train")
        train_one(model, loader(pig_train, args.batch_size, True),
                  loader(pig_val, args.batch_size, False), lr=args.lr,
                  epochs=args.epochs, save_path=args.out.with_name(args.out.stem + "_pig.pt"),
                  **common)
        print("stage 2: SKY-Piano fine-tune")
        train_one(model, loader(sky_train, args.batch_size, True),
                  loader(sky_val, args.batch_size, False), lr=args.lr * 0.3,
                  epochs=args.ft_epochs, save_path=args.out, **common)
    else:
        train = sky_train + pig_train
        val = sky_val or pig_val
        train_one(model, loader(train, args.batch_size, True),
                  loader(val, args.batch_size, False), lr=args.lr,
                  epochs=args.epochs, save_path=args.out, **common)


if __name__ == "__main__":
    main()
