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

Belief-map support:
- If the checkpoint has ``use_belief=True``, the manager owns a private
  :class:`AEManager` purely for belief tracking. On each call we update
  the belief AEManager from the observation, rasterize it into a
  (BELIEF_CHANNELS, 16, 16) tensor, and pass it through the network's
  belief branch.
- Old checkpoints (``use_belief`` absent or False) work unchanged — the
  belief AEManager is still created (cheap) but its tensor is never
  rasterized.

Speed notes:
- ``torch.set_num_threads(1)`` removes contention with uvloop's worker
  thread on a single-core container.
- ``torch.inference_mode()`` is consistently a touch cheaper than
  ``no_grad()`` on small CNNs.
- Warmup forward at construction pays JIT/cudnn init off the critical path.
- Pre-allocated input tensors avoid per-call ``torch.from_numpy().to()``
  allocations.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from ae_manager import AEManager
from encoder import (
    BELIEF_CHANNELS,
    FrameStacker,
    SCALAR_DIM,
    rasterize_belief,
)
from model import (
    AGENT_VIEW_HW,
    BASE_VIEW_HW,
    BELIEF_HW,
    PolicyNetwork,
    VIEW_CHANNELS,
    build_policy_network,
)


# CPU thread pinning. uvicorn workers and torch otherwise contend on the
# same physical core in the AE container; pinning to 1 thread is the
# single biggest per-call latency win we measured locally.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def _candidate_checkpoints() -> list[Path]:
    """Paths to check, in priority order. See docstring for layout details."""

    here = Path(__file__).resolve().parent
    return [here / "models" / "bc.pt", here.parent / "models" / "bc.pt"]


_MODEL_CACHE: PolicyNetwork | None = None
_DEVICE_CACHE: torch.device | None = None
_N_FRAMES_CACHE: int | None = None
_USE_BELIEF_CACHE: bool = False


def _resolve_checkpoint_path() -> Path:
    override = os.environ.get("AE_POLICY_CHECKPOINT")
    if override:
        return Path(override)
    for candidate in _candidate_checkpoints():
        if candidate.exists():
            return candidate
    return _candidate_checkpoints()[0]


def _warmup(model: PolicyNetwork, device: torch.device, n_frames: int,
            use_belief: bool) -> None:
    """One synthetic forward pass to pay JIT / cudnn init off the critical path."""

    agent_ch = VIEW_CHANNELS * n_frames
    base_ch = VIEW_CHANNELS * n_frames
    scalar = SCALAR_DIM * n_frames
    agent_view = torch.zeros(1, agent_ch, AGENT_VIEW_HW[0], AGENT_VIEW_HW[1], device=device)
    base_view = torch.zeros(1, base_ch, BASE_VIEW_HW[0], BASE_VIEW_HW[1], device=device)
    scalars = torch.zeros(1, scalar, device=device)
    belief = torch.zeros(1, BELIEF_CHANNELS, BELIEF_HW[0], BELIEF_HW[1], device=device) if use_belief else None
    with torch.inference_mode():
        model(agent_view, base_view, scalars, belief_map=belief)
        model(agent_view, base_view, scalars, belief_map=belief)


def _load_model(checkpoint_path: Path) -> tuple[PolicyNetwork, torch.device, int, bool]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    n_frames = int(ckpt.get("n_frames", 1))
    use_belief = bool(ckpt.get("use_belief", False))
    state_dict = ckpt["model_state_dict"]
    # build_policy_network auto-detects legacy-small vs default arch from
    # the first conv layer's channel count. Required because the
    # ppo-full-rl-elo-v1 / ppo-full-rl-v1 checkpoints are legacy-small.
    model = build_policy_network(
        n_frames=n_frames,
        use_belief=use_belief,
        state_dict=state_dict,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    _warmup(model, device, n_frames, use_belief)
    print(
        f"AE policy loaded from {checkpoint_path} "
        f"(arch={getattr(model, 'model_arch', 'default')}, "
        f"n_frames={n_frames}, use_belief={use_belief}, "
        f"epoch={ckpt.get('epoch')}, val_acc={ckpt.get('val_acc')}, "
        f"ppo_eval={ckpt.get('ppo_eval_score')}, "
        f"device={device}, threads={torch.get_num_threads()})"
    )
    return model, device, n_frames, use_belief


class PolicyAEManager:
    """Stateful policy wrapper. Holds a frame-stack across observations.

    Owns:
    - the loaded network (shared via module-level cache)
    - per-game frame history (cleared on /reset via factory recreation)
    - a private ``AEManager`` for belief tracking (only used if the loaded
      model has ``use_belief=True``; cheap if not)
    - pre-allocated input tensors that we ``copy_`` into per call
    """

    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4
    PLACE_BOMB = 5

    def __init__(self):
        global _MODEL_CACHE, _DEVICE_CACHE, _N_FRAMES_CACHE, _USE_BELIEF_CACHE
        if _MODEL_CACHE is None:
            ckpt_path = _resolve_checkpoint_path()
            if not ckpt_path.exists():
                searched = [str(p) for p in _candidate_checkpoints()]
                raise FileNotFoundError(
                    f"AE policy checkpoint not found. Searched: {searched}. "
                    "Copy a trained checkpoint to one of those paths, or set "
                    "AE_POLICY_CHECKPOINT to point at it explicitly."
                )
            (_MODEL_CACHE, _DEVICE_CACHE, _N_FRAMES_CACHE,
             _USE_BELIEF_CACHE) = _load_model(ckpt_path)
        self.model = _MODEL_CACHE
        self.device = _DEVICE_CACHE
        n_frames = _N_FRAMES_CACHE or 1
        self.use_belief = _USE_BELIEF_CACHE

        # Per-game state: cleared at /reset (which constructs a fresh manager).
        self.stacker = FrameStacker(n_frames)
        self.belief_manager = AEManager()  # cheap; no-op when use_belief=False
        self._last_step: int | None = None

        # Pre-allocated, persistent input tensors. Copy stacked numpy into
        # these in place rather than reallocating each tick.
        agent_ch = VIEW_CHANNELS * n_frames
        base_ch = VIEW_CHANNELS * n_frames
        scalar_dim = SCALAR_DIM * n_frames
        self._agent_buf = torch.zeros(1, agent_ch, AGENT_VIEW_HW[0], AGENT_VIEW_HW[1], device=self.device)
        self._base_buf = torch.zeros(1, base_ch, BASE_VIEW_HW[0], BASE_VIEW_HW[1], device=self.device)
        self._scalar_buf = torch.zeros(1, scalar_dim, device=self.device)
        self._mask_buf = torch.zeros(6, device=self.device)
        if self.use_belief:
            self._belief_buf = torch.zeros(
                1, BELIEF_CHANNELS, BELIEF_HW[0], BELIEF_HW[1], device=self.device
            )
        else:
            self._belief_buf = None

    # ------------------------------------------------------------------
    # Belief plumbing
    # ------------------------------------------------------------------
    def _maybe_reset(self, observation: dict) -> None:
        """Reset frame stack + belief if step rolled over (defensive)."""

        step = observation.get("step")
        try:
            step_int = int(step) if step is not None else None
        except Exception:
            step_int = None
        if step_int == 0 or (
            self._last_step is not None and step_int is not None and step_int < self._last_step
        ):
            self.stacker.reset()
            self.belief_manager = AEManager()
        self._last_step = step_int

    def _build_belief(self, observation: dict) -> np.ndarray | None:
        """Update the belief AEManager and rasterize. Returns None when
        the loaded model doesn't use belief (skip rasterization cost)."""

        if not self.use_belief:
            return None
        # Drive the belief manager's memory update without running its
        # full action selection (no need; we only want the belief state).
        # The cheap path is _update_memory + _age_bombs + scalar bookkeeping.
        bm = self.belief_manager
        step = bm._as_int(observation.get("step"), default=(bm.last_step or 0) + 1)
        if bm.last_step is None or step == 0 or step < bm.last_step:
            bm._reset_memory()
        bm._age_bombs(step)
        bm._blast_cache = {}
        bm.last_step = step
        location = bm._location(observation.get("location"))
        direction = bm._as_int(observation.get("direction"), default=0) % 4
        bm._update_memory(observation, step, location, direction)
        return rasterize_belief(bm, observation)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _copy_into_buffers(self, stacked: dict, belief: np.ndarray | None) -> None:
        self._agent_buf[0].copy_(torch.from_numpy(np.ascontiguousarray(stacked["agent_view"])))
        self._base_buf[0].copy_(torch.from_numpy(np.ascontiguousarray(stacked["base_view"])))
        self._scalar_buf[0].copy_(torch.from_numpy(np.ascontiguousarray(stacked["scalars"])))
        self._mask_buf.copy_(torch.from_numpy(np.ascontiguousarray(stacked["action_mask"])))
        if belief is not None and self._belief_buf is not None:
            self._belief_buf[0].copy_(torch.from_numpy(np.ascontiguousarray(belief)))

    def ae(self, observation: dict) -> int:
        self._maybe_reset(observation)
        belief = self._build_belief(observation)
        stacked = self.stacker.observe(observation, belief_map=belief)
        self._copy_into_buffers(stacked, belief)

        with torch.inference_mode():
            logits = self.model(
                self._agent_buf, self._base_buf, self._scalar_buf,
                belief_map=self._belief_buf if self.use_belief else None,
            ).squeeze(0)
            logits = logits + torch.log(self._mask_buf.clamp(min=1e-9))
            return int(logits.argmax().item())

    def ae_logits(self, observation: dict) -> tuple[int, "torch.Tensor"]:
        """Return (greedy action, masked logits) in one forward pass.

        Used by consultant wrappers (e.g. the deployed
        ``ConfidencePolicyHybridAEManager``) and training-side adapters that
        need the masked logits without paying for a second forward pass.
        """

        self._maybe_reset(observation)
        belief = self._build_belief(observation)
        stacked = self.stacker.observe(observation, belief_map=belief)
        self._copy_into_buffers(stacked, belief)

        with torch.inference_mode():
            logits = self.model(
                self._agent_buf, self._base_buf, self._scalar_buf,
                belief_map=self._belief_buf if self.use_belief else None,
            ).squeeze(0)
            masked = logits + torch.log(self._mask_buf.clamp(min=1e-9))
            return int(masked.argmax().item()), masked.detach().clone()
