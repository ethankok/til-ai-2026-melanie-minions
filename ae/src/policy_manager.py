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

Speed notes (vs prior version, ~13 ms/call on CPU):
- ``torch.set_num_threads(1)`` removes contention with uvloop's worker
  thread on a single-core container; a 4-thread torch on a 1-vCPU host was
  costing roughly 40% in interop overhead.
- ``torch.inference_mode()`` is consistently a touch cheaper than the
  legacy ``no_grad()`` on small CNNs.
- A warmup forward pass at construction time pays the cudnn/MKLDNN
  algorithm-pick cost off the critical path; the first real ``/ae`` call
  was previously paying ~30 ms of one-time setup that got attributed to
  inference latency.
- Pre-allocated input tensors avoid per-call ``torch.from_numpy().to()``
  allocations; we copy into reused buffers in place.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from encoder import (
    FrameStacker,
    SCALAR_DIM,
)
from model import (
    AGENT_VIEW_HW,
    BASE_VIEW_HW,
    PolicyNetwork,
    VIEW_CHANNELS,
)


# CPU thread pinning. uvicorn workers and torch otherwise contend on the
# same physical core in the AE container; pinning to 1 thread is the
# single biggest per-call latency win we measured locally.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # set_num_interop_threads must be called before any parallel work; if
    # something already grabbed the pool, swallow the error rather than crash.
    pass


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


def _warmup(model: PolicyNetwork, device: torch.device, n_frames: int) -> None:
    """One synthetic forward pass to pay JIT / cudnn init off the critical path."""

    agent_ch = VIEW_CHANNELS * n_frames
    base_ch = VIEW_CHANNELS * n_frames
    scalar = SCALAR_DIM * n_frames
    agent_view = torch.zeros(1, agent_ch, AGENT_VIEW_HW[0], AGENT_VIEW_HW[1], device=device)
    base_view = torch.zeros(1, base_ch, BASE_VIEW_HW[0], BASE_VIEW_HW[1], device=device)
    scalars = torch.zeros(1, scalar, device=device)
    with torch.inference_mode():
        # Two passes — first builds caches, second uses them. Cheap insurance.
        model(agent_view, base_view, scalars)
        model(agent_view, base_view, scalars)


def _load_model(checkpoint_path: Path) -> tuple[PolicyNetwork, torch.device, int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    n_frames = int(ckpt.get("n_frames", 1))
    model = PolicyNetwork(n_frames=n_frames).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    _warmup(model, device, n_frames)
    print(
        f"AE policy loaded from {checkpoint_path} "
        f"(n_frames={n_frames}, epoch={ckpt.get('epoch')}, "
        f"val_acc={ckpt.get('val_acc')}, ppo_eval={ckpt.get('ppo_eval_score')}, "
        f"device={device}, threads={torch.get_num_threads()})"
    )
    return model, device, n_frames


class PolicyAEManager:
    """Stateful policy wrapper. Holds a frame-stack across observations.

    Owns a set of pre-allocated input tensors that we copy into per-call,
    bypassing the per-step ``torch.from_numpy().to(device)`` allocations
    that dominated the old inference path.
    """

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
        n_frames = _N_FRAMES_CACHE or 1
        # Each new manager (created at /reset) gets a clean frame buffer.
        self.stacker = FrameStacker(n_frames)
        self._last_step: int | None = None

        # Pre-allocated, persistent input tensors. Copy stacked-numpy into
        # these in place rather than reallocating each tick.
        agent_ch = VIEW_CHANNELS * n_frames
        base_ch = VIEW_CHANNELS * n_frames
        scalar_dim = SCALAR_DIM * n_frames
        self._agent_buf = torch.zeros(
            1, agent_ch, AGENT_VIEW_HW[0], AGENT_VIEW_HW[1], device=self.device
        )
        self._base_buf = torch.zeros(
            1, base_ch, BASE_VIEW_HW[0], BASE_VIEW_HW[1], device=self.device
        )
        self._scalar_buf = torch.zeros(1, scalar_dim, device=self.device)
        self._mask_buf = torch.zeros(6, device=self.device)

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
        # In-place copy from numpy to preallocated tensors — avoids the
        # per-call allocator overhead of torch.from_numpy().to(device).
        self._agent_buf[0].copy_(torch.from_numpy(np.ascontiguousarray(stacked["agent_view"])))
        self._base_buf[0].copy_(torch.from_numpy(np.ascontiguousarray(stacked["base_view"])))
        self._scalar_buf[0].copy_(torch.from_numpy(np.ascontiguousarray(stacked["scalars"])))
        self._mask_buf.copy_(torch.from_numpy(np.ascontiguousarray(stacked["action_mask"])))

        with torch.inference_mode():
            logits = self.model(self._agent_buf, self._base_buf, self._scalar_buf).squeeze(0)
            logits = logits + torch.log(self._mask_buf.clamp(min=1e-9))
            return int(logits.argmax().item())

    def ae_logits(self, observation: dict) -> tuple[int, "torch.Tensor"]:
        """Same as ``ae`` but also returns the masked logits.

        Used by :class:`HybridAEManager` to gate on policy confidence
        without paying for a second forward pass.
        """

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
        self._agent_buf[0].copy_(torch.from_numpy(np.ascontiguousarray(stacked["agent_view"])))
        self._base_buf[0].copy_(torch.from_numpy(np.ascontiguousarray(stacked["base_view"])))
        self._scalar_buf[0].copy_(torch.from_numpy(np.ascontiguousarray(stacked["scalars"])))
        self._mask_buf.copy_(torch.from_numpy(np.ascontiguousarray(stacked["action_mask"])))

        with torch.inference_mode():
            logits = self.model(self._agent_buf, self._base_buf, self._scalar_buf).squeeze(0)
            masked = logits + torch.log(self._mask_buf.clamp(min=1e-9))
            return int(masked.argmax().item()), masked.detach().clone()
