"""Behavior-clone from collected planner-v3b trajectories.

Usage:
    python training/ae/train_bc.py \
        --data training/ae/data/bc.npz \
        --out  training/ae/checkpoints/bc.pt \
        --epochs 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from model import PolicyNetwork, num_parameters


class BCDataset(Dataset):
    def __init__(self, npz_path: Path):
        data = np.load(npz_path)
        self.agent_views = torch.from_numpy(data["agent_views"]).float()
        self.base_views = torch.from_numpy(data["base_views"]).float()
        self.scalars = torch.from_numpy(data["scalars"]).float()
        self.action_masks = torch.from_numpy(data["action_masks"]).float()
        self.actions = torch.from_numpy(data["actions"]).long()

    def __len__(self) -> int:
        return self.actions.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.agent_views[idx],
            self.base_views[idx],
            self.scalars[idx],
            self.action_masks[idx],
            self.actions[idx],
        )


def _masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Add log(mask) so illegal actions become -inf under softmax/argmax."""
    return logits + torch.log(mask.clamp(min=1e-9))


def evaluate(model: PolicyNetwork, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for agent_v, base_v, scalars, mask, actions in loader:
            agent_v = agent_v.to(device, non_blocking=True)
            base_v = base_v.to(device, non_blocking=True)
            scalars = scalars.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            logits = _masked_logits(model(agent_v, base_v, scalars), mask)
            loss = nn.functional.cross_entropy(logits, actions, reduction="sum")
            total_loss += loss.item()
            total_correct += (logits.argmax(-1) == actions).sum().item()
            total += actions.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def train(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
              else "cpu")
    )
    print(f"device: {device}")

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Dataset not found at {data_path}. Run collect_bc.py first.")
    dataset = BCDataset(data_path)
    print(f"samples: {len(dataset):,}")

    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(0),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = PolicyNetwork().to(device)
    print(f"params: {num_parameters(model):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_n = 0
        epoch_start = time.time()
        for agent_v, base_v, scalars, mask, actions in train_loader:
            agent_v = agent_v.to(device, non_blocking=True)
            base_v = base_v.to(device, non_blocking=True)
            scalars = scalars.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)

            logits = _masked_logits(model(agent_v, base_v, scalars), mask)
            loss = nn.functional.cross_entropy(logits, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * actions.size(0)
            train_correct += (logits.argmax(-1) == actions).sum().item()
            train_n += actions.size(0)
        scheduler.step()

        train_loss /= max(train_n, 1)
        train_acc = train_correct / max(train_n, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - epoch_start
        print(
            f"epoch {epoch:>2}/{args.epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, out_path)
            print(f"  ✓ best so far → saved to {out_path}")

    print(f"\nBest val_acc: {best_val_acc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training/ae/data/bc.npz")
    parser.add_argument("--out", default="training/ae/checkpoints/bc.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
