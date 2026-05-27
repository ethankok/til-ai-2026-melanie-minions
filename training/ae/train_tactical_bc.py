"""Train a tactical option selector from outcome-weighted full-game data."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") is None:
    os.environ["PYTHONHASHSEED"] = "0"

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
from tactical_policy import NUM_TACTICAL_OPTIONS, TACTICAL_OPTION_NAMES  # noqa: E402


class TacticalDataset(Dataset):
    def __init__(
        self,
        npz_path: Path,
        min_weight: float = 0.0,
        positive_only: bool = False,
        positive_delta_floor: float = 0.0,
    ):
        data = np.load(npz_path)
        weights = data["weights"].astype(np.float32)
        keep = weights >= float(min_weight)
        if positive_only:
            advantages = data["advantages"].astype(np.float32)
            is_baseline = (
                data["is_baseline"].astype(bool)
                if "is_baseline" in data.files
                else np.zeros_like(advantages, dtype=bool)
            )
            keep = keep & (advantages > float(positive_delta_floor)) & (~is_baseline)
        if not np.any(keep):
            raise SystemExit(
                f"No samples remain after min_weight={min_weight} positive_only={positive_only} "
                f"positive_delta_floor={positive_delta_floor}"
            )
        self.agent_views = torch.from_numpy(data["agent_views"][keep]).float()
        self.base_views = torch.from_numpy(data["base_views"][keep]).float()
        self.scalars = torch.from_numpy(data["scalars"][keep]).float()
        self.options = torch.from_numpy(data["options"][keep]).long()
        if "prior_options" in data.files:
            self.prior_options = torch.from_numpy(data["prior_options"][keep]).long()
        else:
            self.prior_options = self.options.clone()
        self.weights = torch.from_numpy(weights[keep]).float()
        self.advantages = torch.from_numpy(data["advantages"][keep]).float()
        if "is_baseline" in data.files:
            self.is_baseline = torch.from_numpy(data["is_baseline"][keep].astype(np.int8)).bool()
        else:
            self.is_baseline = torch.zeros_like(self.options, dtype=torch.bool)
        self.n_frames = int(data["n_frames"]) if "n_frames" in data.files else max(1, self.agent_views.shape[1] // 25)
        if "beliefs" in data.files:
            self.beliefs = torch.from_numpy(data["beliefs"][keep]).float()
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
                self.weights[idx],
                self.advantages[idx],
            )
        return (
            self.agent_views[idx],
            self.base_views[idx],
            self.scalars[idx],
            self.options[idx],
            self.weights[idx],
            self.advantages[idx],
        )


def _device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    )


def _unpack(batch, has_belief: bool, device: torch.device):
    if has_belief:
        agent_v, base_v, scalars, belief, options, weights, advantages = batch
        belief = belief.to(device, non_blocking=True)
    else:
        agent_v, base_v, scalars, options, weights, advantages = batch
        belief = None
    return (
        agent_v.to(device, non_blocking=True),
        base_v.to(device, non_blocking=True),
        scalars.to(device, non_blocking=True),
        belief,
        options.to(device, non_blocking=True),
        weights.to(device, non_blocking=True),
        advantages.to(device, non_blocking=True),
    )


def _class_weights(dataset: TacticalDataset, power: float, device: torch.device) -> torch.Tensor | None:
    if power <= 0.0:
        return None
    counts = torch.bincount(dataset.options, minlength=NUM_TACTICAL_OPTIONS).float().clamp(min=1.0)
    weights = (counts.mean() / counts).pow(power)
    weights = weights / weights.mean()
    return weights.to(device)


def _positive_delta_metadata(dataset: TacticalDataset) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((NUM_TACTICAL_OPTIONS, NUM_TACTICAL_OPTIONS), dtype=np.int32)
    weights = np.zeros((NUM_TACTICAL_OPTIONS, NUM_TACTICAL_OPTIONS), dtype=np.float32)
    mask = (~dataset.is_baseline) & (dataset.advantages > 0.0) & (dataset.options != dataset.prior_options)
    if not bool(mask.any()):
        return counts, weights
    priors = dataset.prior_options[mask].cpu().numpy()
    options = dataset.options[mask].cpu().numpy()
    sample_weights = dataset.weights[mask].cpu().numpy()
    for prior, option, weight in zip(priors, options, sample_weights):
        counts[int(prior), int(option)] += 1
        weights[int(prior), int(option)] += float(weight)
    return counts, weights


def _harm_aware_metadata(data_path: Path) -> dict:
    """Pull the W1.1 harm-aware matrices from the collector npz.

    For npz files produced before W1.1 these keys don't exist; we return
    empty arrays so older datasets still train cleanly. The inference manager
    sees zero attempted counts and falls back to the legacy positive-only
    support check.
    """
    transition_shape = (NUM_TACTICAL_OPTIONS, NUM_TACTICAL_OPTIONS)
    bucket_shape = (NUM_TACTICAL_OPTIONS, NUM_TACTICAL_OPTIONS, 3)
    out = {
        "attempted_transition_counts": np.zeros(transition_shape, dtype=np.int64),
        "positive_transition_counts_full": np.zeros(transition_shape, dtype=np.int64),
        "negative_transition_counts": np.zeros(transition_shape, dtype=np.int64),
        "transition_net_delta_sum": np.zeros(transition_shape, dtype=np.float64),
        "transition_weight_sum": np.zeros(transition_shape, dtype=np.float64),
        "transition_weighted_delta_sum": np.zeros(transition_shape, dtype=np.float64),
        "bucket_attempted": np.zeros(bucket_shape, dtype=np.int64),
        "bucket_positive": np.zeros(bucket_shape, dtype=np.int64),
        "bucket_net_delta_sum": np.zeros(bucket_shape, dtype=np.float64),
    }
    with np.load(data_path, allow_pickle=True) as data:
        files = set(data.files)
        if "attempted_transition_counts" in files:
            out["attempted_transition_counts"] = np.asarray(data["attempted_transition_counts"], dtype=np.int64)
        if "positive_transition_counts" in files:
            arr = np.asarray(data["positive_transition_counts"])
            if arr.shape == transition_shape:
                out["positive_transition_counts_full"] = arr.astype(np.int64)
        if "negative_transition_counts" in files:
            out["negative_transition_counts"] = np.asarray(data["negative_transition_counts"], dtype=np.int64)
        if "transition_net_delta_sum" in files:
            out["transition_net_delta_sum"] = np.asarray(data["transition_net_delta_sum"], dtype=np.float64)
        if "transition_weight_sum" in files:
            out["transition_weight_sum"] = np.asarray(data["transition_weight_sum"], dtype=np.float64)
        if "transition_weighted_delta_sum" in files:
            out["transition_weighted_delta_sum"] = np.asarray(data["transition_weighted_delta_sum"], dtype=np.float64)
        if "bucket_attempted" in files:
            out["bucket_attempted"] = np.asarray(data["bucket_attempted"], dtype=np.int64)
        if "bucket_positive" in files:
            out["bucket_positive"] = np.asarray(data["bucket_positive"], dtype=np.int64)
        if "bucket_net_delta_sum" in files:
            out["bucket_net_delta_sum"] = np.asarray(data["bucket_net_delta_sum"], dtype=np.float64)
    return out


def _weighted_loss(logits: torch.Tensor, options: torch.Tensor, weights: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    losses = criterion(logits, options)
    return (losses * weights).sum() / weights.sum().clamp(min=1e-6)


def evaluate(
    model: PolicyNetwork,
    loader: DataLoader,
    has_belief: bool,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    total_correct = 0.0
    total = 0
    with torch.no_grad():
        for batch in loader:
            agent_v, base_v, scalars, belief, options, weights, _advantages = _unpack(batch, has_belief, device)
            logits = model(agent_v, base_v, scalars, belief_map=belief)
            losses = criterion(logits, options)
            total_loss += float((losses * weights).sum().item())
            total_weight += float(weights.sum().item())
            pred = logits.argmax(-1)
            total_correct += float(((pred == options).float() * weights).sum().item())
            total += int(options.size(0))
    return (
        total_loss / max(total_weight, 1e-6),
        total_correct / max(total_weight, 1e-6),
        total_weight / max(total, 1),
    )


def train(args: argparse.Namespace) -> None:
    device = _device()
    print(f"device: {device}")
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Dataset not found at {data_path}. Run collect_tactical_outcome.py first.")
    dataset = TacticalDataset(
        data_path,
        min_weight=args.min_sample_weight,
        positive_only=args.positive_only,
        positive_delta_floor=args.positive_delta_floor,
    )
    use_belief = dataset.has_belief and not args.no_belief
    print(
        f"samples: {len(dataset):,}; n_frames={dataset.n_frames}; use_belief={use_belief}; "
        f"weight_mean={float(dataset.weights.mean()):.3f}; "
        f"adv_mean={float(dataset.advantages.mean()):.4f}"
    )

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

    model = PolicyNetwork(n_frames=dataset.n_frames, action_dim=NUM_TACTICAL_OPTIONS, use_belief=use_belief).to(device)
    print(f"params: {num_parameters(model):,}; action_dim={NUM_TACTICAL_OPTIONS}")
    class_weights = _class_weights(dataset, args.class_weight_power, device)
    if class_weights is not None:
        print("class_weights: " + " ".join(
            f"{name}={float(class_weights[i]):.2f}" for i, name in enumerate(TACTICAL_OPTION_NAMES)
        ))
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    counts = torch.bincount(dataset.options, minlength=NUM_TACTICAL_OPTIONS)
    wcounts = torch.zeros(NUM_TACTICAL_OPTIONS)
    for option, weight in zip(dataset.options, dataset.weights):
        wcounts[int(option)] += float(weight)
    print("dataset_options:")
    for i, name in enumerate(TACTICAL_OPTION_NAMES):
        print(f"  {name:24s} raw={int(counts[i]):>7d} weighted={float(wcounts[i]):>9.1f}")
    positive_delta_counts, positive_delta_weights = _positive_delta_metadata(dataset)
    print("positive_delta_transitions:")
    any_delta = False
    for prior in range(NUM_TACTICAL_OPTIONS):
        for option in range(NUM_TACTICAL_OPTIONS):
            count = int(positive_delta_counts[prior, option])
            if count <= 0:
                continue
            any_delta = True
            print(
                f"  {TACTICAL_OPTION_NAMES[prior]:24s} -> {TACTICAL_OPTION_NAMES[option]:24s} "
                f"raw={count:>5d} weighted={float(positive_delta_weights[prior, option]):>8.1f}"
            )
    if not any_delta:
        print("  none")

    harm_aware = _harm_aware_metadata(data_path)
    attempted_total = int(np.sum(harm_aware["attempted_transition_counts"]))
    if attempted_total > 0:
        print("harm_aware_transitions (attempted/positive/positive_rate/mean_delta):")
        attempted_mat = harm_aware["attempted_transition_counts"]
        positive_mat = harm_aware["positive_transition_counts_full"]
        net_delta_mat = harm_aware["transition_net_delta_sum"]
        for prior in range(NUM_TACTICAL_OPTIONS):
            for option in range(NUM_TACTICAL_OPTIONS):
                attempted = int(attempted_mat[prior, option])
                if attempted <= 0:
                    continue
                positive = int(positive_mat[prior, option])
                pos_rate = positive / attempted if attempted else 0.0
                mean_delta = float(net_delta_mat[prior, option]) / max(attempted, 1)
                marker = "  " if prior == option else "* "  # * highlights actual deltas
                print(
                    f"  {marker}{TACTICAL_OPTION_NAMES[prior]:24s} -> {TACTICAL_OPTION_NAMES[option]:24s} "
                    f"att={attempted:>5d} pos={positive:>5d} pos_rate={pos_rate:.3f} mean_delta={mean_delta:+.4f}"
                )
    else:
        print("harm_aware_transitions: dataset has no attempted_transition_counts (pre-W1.1 npz)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_score = -float("inf")
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_weight = 0.0
        train_correct = 0.0
        t0 = time.time()
        for batch in train_loader:
            agent_v, base_v, scalars, belief, options, weights, _advantages = _unpack(batch, use_belief, device)
            logits = model(agent_v, base_v, scalars, belief_map=belief)
            loss = _weighted_loss(logits, options, weights, criterion)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                losses = criterion(logits, options)
                train_loss += float((losses * weights).sum().item())
                train_weight += float(weights.sum().item())
                train_correct += float(((logits.argmax(-1) == options).float() * weights).sum().item())
        scheduler.step()
        val_loss, val_acc, val_mean_weight = evaluate(model, val_loader, use_belief, device, criterion)
        train_loss /= max(train_weight, 1e-6)
        train_acc = train_correct / max(train_weight, 1e-6)
        score = val_acc - args.loss_penalty * val_loss
        print(
            f"epoch {epoch:>2}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_wacc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_wacc={val_acc:.4f} val_wmean={val_mean_weight:.3f} "
            f"({time.time() - t0:.1f}s)"
        )
        if score > best_score:
            best_score = score
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_weighted_acc": val_acc,
                "n_frames": dataset.n_frames,
                "use_belief": use_belief,
                "action_dim": NUM_TACTICAL_OPTIONS,
                "option_names": TACTICAL_OPTION_NAMES,
                "model_type": "ae_tactical_policy",
                "source_data": str(data_path),
                "class_weight_power": args.class_weight_power,
                "loss_penalty": args.loss_penalty,
                "positive_delta_transition_counts": positive_delta_counts,
                "positive_delta_transition_weights": positive_delta_weights,
                # W1.1 harm-aware metadata; consumed by
                # tactical_hybrid_manager._delta_is_supported at inference.
                # Empty for pre-W1.1 npz files; inference manager treats that
                # as "no harm-aware data, fall back to legacy gate".
                "attempted_transition_counts": harm_aware["attempted_transition_counts"],
                "positive_transition_counts_full": harm_aware["positive_transition_counts_full"],
                "negative_transition_counts": harm_aware["negative_transition_counts"],
                "transition_net_delta_sum": harm_aware["transition_net_delta_sum"],
                "transition_weight_sum": harm_aware["transition_weight_sum"],
                "transition_weighted_delta_sum": harm_aware["transition_weighted_delta_sum"],
                "bucket_attempted": harm_aware["bucket_attempted"],
                "bucket_positive": harm_aware["bucket_positive"],
                "bucket_net_delta_sum": harm_aware["bucket_net_delta_sum"],
                "weighted_delta_mean": float(dataset.advantages.mul(dataset.weights).sum() / dataset.weights.sum().clamp(min=1e-6)),
            }, out_path)
            print(f"  best so far -> saved {out_path}")
    print(f"\nBest val_loss: {best_val_loss:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training/ae/data/tactical_outcome.npz")
    parser.add_argument("--out", default="training/ae/checkpoints/tactical_policy.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--class-weight-power", type=float, default=0.0,
                        help="0 keeps terminal outcome weights primary; increase only if useful options vanish")
    parser.add_argument("--loss-penalty", type=float, default=0.25,
                        help="checkpoint score is val_wacc - loss_penalty * val_loss")
    parser.add_argument("--min-sample-weight", type=float, default=0.05)
    parser.add_argument("--positive-only", action="store_true",
                        help="filter dataset to non-baseline rows with advantages > positive-delta-floor "
                             "(warm-start macro-PPO at a 'where the heuristic is suboptimal' prior)")
    parser.add_argument("--positive-delta-floor", type=float, default=0.0,
                        help="threshold for --positive-only (kept if advantage > floor)")
    parser.add_argument("--no-belief", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
