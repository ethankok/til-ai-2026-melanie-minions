"""Observation encoder + frame stacker for AE policy networks.

`encode_observation` produces a single-step tensor dict. `FrameStacker` keeps
a rolling buffer of the last N encoded observations and emits the
channel-concatenated stack, which is what the v2 `PolicyNetwork` consumes.

Output layout (single frame):
    agent_view:  (25, 7, 5) channel-first viewcone
    base_view:   (25, 7, 7) channel-first viewcone
    scalars:     (SCALAR_DIM,)   flat features
    action_mask: (6,)            binary

After N-frame stacking:
    agent_view:  (25 * N, 7, 5)
    base_view:   (25 * N, 7, 7)
    scalars:     (SCALAR_DIM * N,)
    action_mask: (6,)   — always taken from the latest frame only
"""

from __future__ import annotations

import numpy as np


SCALAR_DIM = 17
GRID_SIZE = 16
MAX_HEALTH = 60.0
MAX_BASE_HEALTH = 100.0
MAX_RESOURCES = 10.0
MAX_BOMBS = 10.0
MAX_STEPS = 200.0


def _to_array(value, dtype=np.float32) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def _scalar(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        if hasattr(value, "item"):
            return float(value.item())
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                return float(default)
            return float(value[0])
        return float(value)
    except Exception:
        return float(default)


def _onehot(index: int, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    if 0 <= index < n:
        out[index] = 1.0
    return out


def encode_observation(obs: dict) -> dict:
    """Encode one raw observation dict into the policy input tensors."""

    agent_view = _to_array(obs.get("agent_viewcone"))
    if agent_view.ndim == 3 and agent_view.shape[-1] == 25:
        agent_view = np.transpose(agent_view, (2, 0, 1))
    base_view = _to_array(obs.get("base_viewcone"))
    if base_view.ndim == 3 and base_view.shape[-1] == 25:
        base_view = np.transpose(base_view, (2, 0, 1))

    direction = int(_scalar(obs.get("direction"), 0)) % 4
    frozen_ticks = max(0, min(3, int(_scalar(obs.get("frozen_ticks"), 0))))

    location = _to_array(obs.get("location", [0, 0]))
    if location.ndim > 1:
        location = location.flatten()
    base_location = _to_array(obs.get("base_location", [0, 0]))
    if base_location.ndim > 1:
        base_location = base_location.flatten()

    health = _scalar(obs.get("health"), MAX_HEALTH)
    base_health = _scalar(obs.get("base_health"), MAX_BASE_HEALTH)
    team_resources = _scalar(obs.get("team_resources"), 0.0)
    team_bombs = _scalar(obs.get("team_bombs"), 0.0)
    step = _scalar(obs.get("step"), 0.0)

    scalars = np.concatenate([
        _onehot(direction, 4),
        _onehot(frozen_ticks, 4),
        np.array([location[0] / GRID_SIZE, location[1] / GRID_SIZE], dtype=np.float32),
        np.array([base_location[0] / GRID_SIZE, base_location[1] / GRID_SIZE], dtype=np.float32),
        np.array([health / MAX_HEALTH, base_health / MAX_BASE_HEALTH], dtype=np.float32),
        np.array([min(team_resources / MAX_RESOURCES, 1.0),
                  min(team_bombs / MAX_BOMBS, 1.0)], dtype=np.float32),
        np.array([min(step / MAX_STEPS, 1.0)], dtype=np.float32),
    ]).astype(np.float32)
    assert scalars.shape[0] == SCALAR_DIM, f"got scalar shape {scalars.shape}, expected {SCALAR_DIM}"

    action_mask = _to_array(obs.get("action_mask", [1, 1, 1, 1, 1, 1]))
    action_mask = action_mask.flatten().astype(np.float32)
    if action_mask.shape[0] != 6:
        fixed = np.ones(6, dtype=np.float32)
        fixed[: min(6, action_mask.shape[0])] = action_mask[: min(6, action_mask.shape[0])]
        action_mask = fixed

    return {
        "agent_view": agent_view.astype(np.float32, copy=False),
        "base_view": base_view.astype(np.float32, copy=False),
        "scalars": scalars,
        "action_mask": action_mask,
    }


class FrameStacker:
    """Rolling buffer of the last N encoded observations.

    First-call behavior: pads the buffer with copies of the first observation
    so the policy always receives ``N`` frames, even on step 0/1/2 of a game.
    Designed for both training (one stacker per env instance) and deployment
    (one stacker per `PolicyAEManager`, cleared on `/reset` via factory).
    """

    def __init__(self, n_frames: int = 4):
        if n_frames < 1:
            raise ValueError("n_frames must be ≥ 1")
        self.n_frames = n_frames
        self.frames: list[dict] = []

    def reset(self) -> None:
        self.frames = []

    def observe(self, obs: dict) -> dict:
        """Encode `obs`, push into the buffer, return the stacked input."""
        encoded = encode_observation(obs)
        if not self.frames:
            # Pad: on the first observation of a game, fill the buffer with
            # copies of the same encoding so the network has a valid input.
            self.frames = [encoded] * self.n_frames
        else:
            self.frames.append(encoded)
            if len(self.frames) > self.n_frames:
                self.frames.pop(0)
        return self._stack()

    def _stack(self) -> dict:
        return {
            "agent_view": np.concatenate([f["agent_view"] for f in self.frames], axis=0),
            "base_view": np.concatenate([f["base_view"] for f in self.frames], axis=0),
            "scalars": np.concatenate([f["scalars"] for f in self.frames], axis=0),
            "action_mask": self.frames[-1]["action_mask"],
        }


def stacked_dim(n_frames: int) -> tuple[int, int]:
    """Return (view_channels, scalar_dim) for an N-frame stack."""
    return 25 * n_frames, SCALAR_DIM * n_frames
