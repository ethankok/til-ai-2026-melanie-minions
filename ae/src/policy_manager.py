"""Neural-network policy variant of :class:`AEManager`.

Drop-in replacement for the heuristic planner — exposes the same ``ae()``
method, returns an int action 0-5. The network is loaded once at the first
:class:`PolicyAEManager` construction and shared across resets; each
manager instance owns its own :class:`FrameStacker` so the per-game frame
history is cleared on ``/reset`` (which constructs a fresh manager).

Looks up its checkpoint at:

  1. ``$AE_POLICY_CHECKPOINT`` (env override) if set; otherwise
  2. ``<src dir>/models/bc.pt`` (container layout); or
  3. ``<src dir>/../models/bc.pt`` (local dev layout).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from encoder import FrameStacker
from model import PolicyNetwork


def _candidate_checkpoints() -> list[Path]:
    """Paths to check, in priority order. See docstring for layout details."""

    here = Path(__file__).resolve().parent
    return [here / "models" / "bc.pt", here.parent / "models" / "bc.pt"]


_MODEL_CACHE: PolicyNetwork | None = None
_DEVICE_CACHE: torch.device | None = None
_N_FRAMES_CACHE: int | None = None


def _resolve_checkpoint_path() -> Path:
    override = os.environ.get("AE_POLICY_CHECKPOINT")
    if override:
        return Path(override)
    for candidate in _candidate_checkpoints():
        if candidate.exists():
            return candidate
    return _candidate_checkpoints()[0]


def _load_model(checkpoint_path: Path) -> tuple[PolicyNetwork, torch.device, int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    n_frames = int(ckpt.get("n_frames", 1))
    model = PolicyNetwork(n_frames=n_frames).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(
        f"AE policy loaded from {checkpoint_path} "
        f"(n_frames={n_frames}, epoch={ckpt.get('epoch')}, "
        f"val_acc={ckpt.get('val_acc')}, ppo_eval={ckpt.get('ppo_eval_score')}, "
        f"device={device})"
    )
    return model, device, n_frames


class PolicyAEManager:
    """Stateful policy wrapper. Holds a frame-stack across observations."""

    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4
    PLACE_BOMB = 5

    def __init__(self):
        global _MODEL_CACHE, _DEVICE_CACHE, _N_FRAMES_CACHE
        if _MODEL_CACHE is None:
            ckpt_path = _resolve_checkpoint_path()
            if not ckpt_path.exists():
                searched = [str(p) for p in _candidate_checkpoints()]
                raise FileNotFoundError(
                    f"AE policy checkpoint not found. Searched: {searched}. "
                    "Copy a trained checkpoint to one of those paths, or set "
                    "AE_POLICY_CHECKPOINT to point at it explicitly."
                )
            _MODEL_CACHE, _DEVICE_CACHE, _N_FRAMES_CACHE = _load_model(ckpt_path)
        self.model = _MODEL_CACHE
        self.device = _DEVICE_CACHE
        # Each new manager (created at /reset) gets a clean frame buffer.
        self.stacker = FrameStacker(_N_FRAMES_CACHE or 1)
        self._last_step: int | None = None

    def ae(self, observation: dict) -> int:
        # Defensive reset: if the env sends step=0 without going through
        # /reset, treat it as a fresh game and clear the frame stack.
        step = observation.get("step")
        try:
            step_int = int(step) if step is not None else None
        except Exception:
            step_int = None
        if step_int == 0 or (
            self._last_step is not None and step_int is not None and step_int < self._last_step
        ):
            self.stacker.reset()
        self._last_step = step_int

        stacked = self.stacker.observe(observation)
        agent_view = torch.from_numpy(stacked["agent_view"]).to(self.device)
        base_view = torch.from_numpy(stacked["base_view"]).to(self.device)
        scalars = torch.from_numpy(stacked["scalars"]).to(self.device)
        mask = torch.from_numpy(stacked["action_mask"]).to(self.device)
        return self.model.select_action(
            agent_view, base_view, scalars, action_mask=mask, greedy=True
        )
