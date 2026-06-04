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
    """Load a BC dataset from either a legacy ``.npz`` file or a memmap directory.

    Directory format (written by ``collect_bc.py --stream-dir``):
      <dir>/agent_views.npy   float16
      <dir>/base_views.npy    float16
      <dir>/beliefs.npy       float16  (optional, when with_belief=True)
      <dir>/scalars.npy       float16
      <dir>/action_masks.npy  uint8
      <dir>/actions.npy       int8
      <dir>/meta.json         {"n_frames": int, "with_belief": bool,
                               "n_samples": int, "fields": {...}}

    All arrays are memory-mapped read-only so RAM stays O(1) in dataset size.
    ``__getitem__`` casts to the same dtypes as the legacy npz path so the
    rest of the training code (``_unpack_batch``, model forward) works
    unchanged.
    """

    def __init__(self, data_path: Path):
        data_path = Path(data_path)
        if data_path.is_dir():
            self._init_from_dir(data_path)
        else:
            self._init_from_npz(data_path)

    def _init_from_npz(self, npz_path: Path) -> None:
        data = np.load(npz_path)
        self.agent_views = torch.from_numpy(data["agent_views"]).float()
        self.base_views = torch.from_numpy(data["base_views"]).float()
        self.scalars = torch.from_numpy(data["scalars"]).float()
        self.action_masks = torch.from_numpy(data["action_masks"]).float()
        self.actions = torch.from_numpy(data["actions"]).long()
        if "n_frames" in data.files:
            self.n_frames = int(data["n_frames"])
        else:
            self.n_frames = max(1, int(self.agent_views.shape[1] // 25))
        # Belief tensors (NEW): present when collect_bc was run with
        # --no-belief NOT set. Falls back to "no belief" for old datasets.
        if "beliefs" in data.files:
            self.beliefs = torch.from_numpy(data["beliefs"]).float()
            self.has_belief = True
        else:
            self.beliefs = None
            self.has_belief = False
        self._n_samples = self.actions.shape[0]
        self._use_mmap = False

    def _init_from_dir(self, dir_path: Path) -> None:
        import json as _json
        meta = _json.loads((dir_path / "meta.json").read_text())
        self.n_frames = int(meta["n_frames"])
        self.has_belief = bool(meta.get("with_belief", False))
        self._n_samples = int(meta["n_samples"])
        self._use_mmap = True

        # Memory-map each field lazily — load returns an ndarray view,
        # no data copied to RAM until __getitem__ accesses a row.
        self._mm_agent  = np.load(str(dir_path / "agent_views.npy"), mmap_mode="r")
        self._mm_base   = np.load(str(dir_path / "base_views.npy"),  mmap_mode="r")
        self._mm_scalar = np.load(str(dir_path / "scalars.npy"),     mmap_mode="r")
        self._mm_mask   = np.load(str(dir_path / "action_masks.npy"), mmap_mode="r")
        self._mm_action = np.load(str(dir_path / "actions.npy"),     mmap_mode="r")
        if self.has_belief:
            self._mm_belief = np.load(str(dir_path / "beliefs.npy"), mmap_mode="r")
        else:
            self._mm_belief = None

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int):
        if self._use_mmap:
            # np.asarray copies the mmap slice into a plain ndarray so the
            # returned tensor does not hold a mmap view alive.
            agent_v  = torch.from_numpy(np.asarray(self._mm_agent[idx],  dtype=np.float32))
            base_v   = torch.from_numpy(np.asarray(self._mm_base[idx],   dtype=np.float32))
            scalars  = torch.from_numpy(np.asarray(self._mm_scalar[idx], dtype=np.float32))
            mask     = torch.from_numpy(np.asarray(self._mm_mask[idx],   dtype=np.float32))
            action   = torch.tensor(int(self._mm_action[idx]), dtype=torch.long)
            if self.has_belief:
                belief = torch.from_numpy(np.asarray(self._mm_belief[idx], dtype=np.float32))
                return (agent_v, base_v, scalars, mask, belief, action)
            return (agent_v, base_v, scalars, mask, action)
        else:
            if self.has_belief:
                return (
                    self.agent_views[idx],
                    self.base_views[idx],
                    self.scalars[idx],
                    self.action_masks[idx],
                    self.beliefs[idx],
                    self.actions[idx],
                )
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


def _unpack_batch(batch, has_belief: bool, device):
    """Move a (with-or-without-belief) batch tuple to ``device``."""
    if has_belief:
        agent_v, base_v, scalars, mask, belief, actions = batch
        belief = belief.to(device, non_blocking=True)
    else:
        agent_v, base_v, scalars, mask, actions = batch
        belief = None
    return (
        agent_v.to(device, non_blocking=True),
        base_v.to(device, non_blocking=True),
        scalars.to(device, non_blocking=True),
        mask.to(device, non_blocking=True),
        belief,
        actions.to(device, non_blocking=True),
    )


def evaluate(model: PolicyNetwork, loader: DataLoader, has_belief: bool,
             device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            agent_v, base_v, scalars, mask, belief, actions = _unpack_batch(batch, has_belief, device)
            logits = _masked_logits(model(agent_v, base_v, scalars, belief_map=belief), mask)
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

    use_belief = dataset.has_belief and not args.no_belief
    model = PolicyNetwork(n_frames=dataset.n_frames, use_belief=use_belief).to(device)
    print(f"n_frames: {dataset.n_frames}; use_belief: {use_belief}; "
          f"params: {num_parameters(model):,}")

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
        for batch in train_loader:
            agent_v, base_v, scalars, mask, belief, actions = _unpack_batch(batch, use_belief, device)
            logits = _masked_logits(model(agent_v, base_v, scalars, belief_map=belief), mask)
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
        val_loss, val_acc = evaluate(model, val_loader, use_belief, device)
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
                "n_frames": dataset.n_frames,
                "use_belief": use_belief,
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
    parser.add_argument("--no-belief", action="store_true",
                        help="Force-disable belief input even if the dataset has it.")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
