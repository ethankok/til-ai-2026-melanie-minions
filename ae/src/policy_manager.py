"""Neural-network policy variant of :class:`AEManager`.

Drop-in replacement for the heuristic planner — exposes the same ``ae()``
method, returns an int action 0-5. The network is loaded once at the first
:class:`PolicyAEManager` construction and shared across resets.

Looks up its checkpoint at:

  1. ``$AE_POLICY_CHECKPOINT`` (env override) if set; otherwise
  2. ``<ae>/models/bc.pt``.

If no checkpoint is found, raises :class:`FileNotFoundError` so the server
can fall back to the heuristic ``AEManager``.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from encoder import encode_observation
from model import PolicyNetwork


def _candidate_checkpoints() -> list[Path]:
    """Paths to check, in priority order.

    - In the Docker container, ``COPY src .`` puts source files directly at
      ``/workspace/``, so weights live at ``/workspace/models/bc.pt`` and
      ``__file__/../models/bc.pt`` is the right path.
    - In local development, source lives at ``ae/src/`` and weights at
      ``ae/models/``, so ``__file__/../../models/bc.pt`` is the right path.

    Check both so the same code runs in both contexts.
    """

    here = Path(__file__).resolve().parent
    return [here / "models" / "bc.pt", here.parent / "models" / "bc.pt"]


_MODEL_CACHE: PolicyNetwork | None = None
_DEVICE_CACHE: torch.device | None = None


def _resolve_checkpoint_path() -> Path:
    override = os.environ.get("AE_POLICY_CHECKPOINT")
    if override:
        return Path(override)
    for candidate in _candidate_checkpoints():
        if candidate.exists():
            return candidate
    # Nothing found — return the first candidate for the not-found error message.
    return _candidate_checkpoints()[0]


def _load_model(checkpoint_path: Path) -> tuple[PolicyNetwork, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = PolicyNetwork().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(
        f"AE policy loaded from {checkpoint_path} "
        f"(epoch={ckpt.get('epoch')}, val_acc={ckpt.get('val_acc')}, device={device})"
    )
    return model, device


class PolicyAEManager:
    """Stateless policy wrapper. State (memory) lives entirely in the network."""

    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4
    PLACE_BOMB = 5

    def __init__(self):
        global _MODEL_CACHE, _DEVICE_CACHE
        if _MODEL_CACHE is None:
            ckpt_path = _resolve_checkpoint_path()
            if not ckpt_path.exists():
                searched = [str(p) for p in _candidate_checkpoints()]
                raise FileNotFoundError(
                    f"AE policy checkpoint not found. Searched: {searched}. "
                    "Copy a trained checkpoint to one of those paths, or set "
                    "AE_POLICY_CHECKPOINT to point at it explicitly."
                )
            _MODEL_CACHE, _DEVICE_CACHE = _load_model(ckpt_path)
        self.model = _MODEL_CACHE
        self.device = _DEVICE_CACHE

    def ae(self, observation: dict) -> int:
        encoded = encode_observation(observation)
        agent_view = torch.from_numpy(encoded["agent_view"]).to(self.device)
        base_view = torch.from_numpy(encoded["base_view"]).to(self.device)
        scalars = torch.from_numpy(encoded["scalars"]).to(self.device)
        mask = torch.from_numpy(encoded["action_mask"]).to(self.device)
        return self.model.select_action(
            agent_view, base_view, scalars, action_mask=mask, greedy=True
        )
