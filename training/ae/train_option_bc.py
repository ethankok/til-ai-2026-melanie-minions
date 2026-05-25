"""Train an 8-way AE option selector from heuristic intent labels."""

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
REPO_ROOT = THIS_DIR.parents[1]
AE_SRC = REPO_ROOT / "ae" / "src"
for path in (THIS_DIR, AE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model import PolicyNetwork, num_parameters  # noqa: E402
from option_policy import NUM_OPTIONS, OPTION_NAMES  # noqa: E402


class OptionBCDataset(Dataset):
    def __init__(self, npz_path: Path):
        data = np.load(npz_path)
        self.agent_views = torch.from_numpy(data["agent_views"]).float()
        self.base_views = torch.from_numpy(data["base_views"]).float()
        self.scalars = torch.from_numpy(data["scalars"]).float()
        self.options = torch.from_numpy(data["options"]).long()
        self.n_frames = int(data["n_frames"]) if "n_frames" in data.files else max(1, self.agent_views.shape[1] // 25)
        if "beliefs" in data.files:
            self.beliefs = torch.from_numpy(data["beliefs"]).float()
            self.has_belief = True
        else:
            self.beliefs = None
            self.has_belief = False

    def __len__(self) -> int:
        return int(self.options.shape[0])

    def __getitem__(self, idx: int):
        if self.has_belief:
            return (
                self.agent_views[idx],
                self.base_views[idx],
                self.scalars[idx],
                self.beliefs[idx],
                self.options[idx],
            )
        return (
            self.agent_views[idx],
            self.base_views[idx],
            self.scalars[idx],
            self.options[idx],
        )


def _device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    )


def _unpack(batch, has_belief: bool, device: torch.device):
    if has_belief:
        agent_v, base_v, scalars, belief, options = batch
        belief = belief.to(device, non_blocking=True)
    else:
        agent_v, base_v, scalars, options = batch
        belief = None
    return (
        agent_v.to(device, non_blocking=True),
        base_v.to(device, non_blocking=True),
        scalars.to(device, non_blocking=True),
        belief,
        options.to(device, non_blocking=True),
    )


def _class_weights(dataset: OptionBCDataset, power: float, device: torch.device) -> torch.Tensor | None:
    if power <= 0.0:
        return None
    counts = torch.bincount(dataset.options, minlength=NUM_OPTIONS).float().clamp(min=1.0)
    weights = (counts.mean() / counts).pow(power)
    weights = weights / weights.mean()
    return weights.to(device)


def evaluate(
    model: PolicyNetwork,
    loader: DataLoader,
    has_belief: bool,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            agent_v, base_v, scalars, belief, options = _unpack(batch, has_belief, device)
            logits = model(agent_v, base_v, scalars, belief_map=belief)
            loss = criterion(logits, options)
            total_loss += float(loss.item()) * int(options.size(0))
            total_correct += int((logits.argmax(-1) == options).sum().item())
            total += int(options.size(0))
    return total_loss / max(total, 1), total_correct / max(total, 1)


def train(args: argparse.Namespace) -> None:
    device = _device()
    print(f"device: {device}")
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Dataset not found at {data_path}. Run collect_option_bc.py first.")
    dataset = OptionBCDataset(data_path)
    use_belief = dataset.has_belief and not args.no_belief
    print(f"samples: {len(dataset):,}; n_frames={dataset.n_frames}; use_belief={use_belief}")

    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.split_seed),
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

    model = PolicyNetwork(n_frames=dataset.n_frames, action_dim=NUM_OPTIONS, use_belief=use_belief).to(device)
    print(f"params: {num_parameters(model):,}; action_dim={NUM_OPTIONS}")
    weights = _class_weights(dataset, args.class_weight_power, device)
    if weights is not None:
        print("class_weights: " + " ".join(f"{name}={float(weights[i]):.2f}" for i, name in enumerate(OPTION_NAMES)))
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    counts = torch.bincount(dataset.options, minlength=NUM_OPTIONS)
    print("dataset_options: " + " ".join(f"{name}={int(counts[i])}" for i, name in enumerate(OPTION_NAMES)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_n = 0
        t0 = time.time()
        for batch in train_loader:
            agent_v, base_v, scalars, belief, options = _unpack(batch, use_belief, device)
            logits = model(agent_v, base_v, scalars, belief_map=belief)
            loss = criterion(logits, options)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            train_loss += float(loss.item()) * int(options.size(0))
            train_correct += int((logits.argmax(-1) == options).sum().item())
            train_n += int(options.size(0))
        scheduler.step()
        val_loss, val_acc = evaluate(model, val_loader, use_belief, device, criterion)
        train_loss /= max(train_n, 1)
        train_acc = train_correct / max(train_n, 1)
        print(
            f"epoch {epoch:>2}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"({time.time() - t0:.1f}s)"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "n_frames": dataset.n_frames,
                "use_belief": use_belief,
                "action_dim": NUM_OPTIONS,
                "option_names": OPTION_NAMES,
                "model_type": "ae_option_policy",
                "source_data": str(data_path),
                "class_weight_power": args.class_weight_power,
            }, out_path)
            print(f"  best so far -> saved {out_path}")
    print(f"\nBest val_acc: {best_acc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training/ae/data/option_bc.npz")
    parser.add_argument("--out", default="training/ae/checkpoints/option_policy.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--class-weight-power", type=float, default=0.5,
                        help="0 disables class weighting; 0.5 is sqrt inverse-frequency weighting")
    parser.add_argument("--no-belief", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
