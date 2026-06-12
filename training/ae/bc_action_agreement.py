"""BC fidelity gate: clone-vs-teacher action agreement + lift over modal baseline.

A STAY-heavy teacher scores high agreement by always-STAY, so raw agreement is
misleading — the signal is the LIFT above the modal-action baseline (curry's gate).
Mirrors train_bc.evaluate's forward; reuses its dataset/unpack/mask helpers.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from model import PolicyNetwork  # noqa: E402
from train_bc import BCDataset, _masked_logits, _unpack_batch  # noqa: E402


def agreement_and_lift(pred: np.ndarray, true: np.ndarray, n_actions: int = 6) -> tuple[float, float, float]:
    """Return (agreement, modal_baseline, lift). Pure; no model/IO."""
    pred = np.asarray(pred)
    true = np.asarray(true)
    n = max(len(true), 1)
    agreement = float((pred == true).sum()) / n
    modal = float(np.bincount(true, minlength=n_actions).max()) / n
    return agreement, modal, agreement - modal


def measure(clone_path: str, data_path: str, batch_size: int = 256) -> tuple[float, float, float]:
    """Load the clone + held-out demos, mirror train_bc.evaluate's forward, return (agreement, modal_baseline, lift). CPU-only (offline gate)."""
    device = torch.device("cpu")
    dataset = BCDataset(Path(data_path))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    ckpt = torch.load(clone_path, map_location=device)
    # train_bc.py saves {"model_state_dict": ...}; fall back to older key names.
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "model_state", "model"):
            if key in ckpt:
                state = ckpt[key]
                break
        else:
            state = ckpt  # raw state dict (no wrapper)
    else:
        state = ckpt
    use_belief = bool(dataset.has_belief)
    model = PolicyNetwork(n_frames=int(dataset.n_frames), use_belief=use_belief)
    model.load_state_dict(state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            agent_v, base_v, scalars, mask, belief, actions = _unpack_batch(
                batch, dataset.has_belief, device
            )
            logits = _masked_logits(model(agent_v, base_v, scalars, belief_map=belief), mask)
            preds.append(logits.argmax(-1).cpu().numpy())
            trues.append(actions.cpu().numpy())
    return agreement_and_lift(np.concatenate(preds), np.concatenate(trues))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clone", required=True)
    p.add_argument("--data", required=True, help="held-out demo .npz")
    p.add_argument("--min-lift", type=float, default=0.20)
    args = p.parse_args()
    agreement, modal, lift = measure(args.clone, args.data)
    print(f"agreement={agreement:.4f}  modal_baseline={modal:.4f}  lift={lift:+.4f}")
    if lift < args.min_lift:
        raise SystemExit(f"FIDELITY FAIL: lift {lift:+.4f} < --min-lift {args.min_lift}")
    print("FIDELITY PASS")


if __name__ == "__main__":
    raise SystemExit(main())
