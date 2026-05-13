"""Observation encoder for AE policy networks.

Converts a raw env observation dict into float32 numpy arrays suitable for the
policy network. Used by both BC dataset collection and live policy inference,
so a single source of truth for the observation -> tensor mapping.

Output layout:
    agent_view:  (25, 7, 5) channel-first viewcone
    base_view:   (25, 7, 7) channel-first viewcone
    scalars:     (SCALAR_DIM,) flat features
    action_mask: (6,) binary
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
        _onehot(direction, 4),                                              # 4
        _onehot(frozen_ticks, 4),                                           # 4
        np.array([location[0] / GRID_SIZE, location[1] / GRID_SIZE], dtype=np.float32),       # 2
        np.array([base_location[0] / GRID_SIZE, base_location[1] / GRID_SIZE], dtype=np.float32),  # 2
        np.array([health / MAX_HEALTH, base_health / MAX_BASE_HEALTH], dtype=np.float32),    # 2
        np.array([min(team_resources / MAX_RESOURCES, 1.0),
                  min(team_bombs / MAX_BOMBS, 1.0)], dtype=np.float32),    # 2
        np.array([min(step / MAX_STEPS, 1.0)], dtype=np.float32),          # 1
    ]).astype(np.float32)
    assert scalars.shape[0] == SCALAR_DIM, f"got scalar shape {scalars.shape}, expected {SCALAR_DIM}"

    action_mask = _to_array(obs.get("action_mask", [1, 1, 1, 1, 1, 1]))
    action_mask = action_mask.flatten().astype(np.float32)
    if action_mask.shape[0] != 6:
        # Pad or trim defensively; the env always returns 6.
        fixed = np.ones(6, dtype=np.float32)
        fixed[: min(6, action_mask.shape[0])] = action_mask[: min(6, action_mask.shape[0])]
        action_mask = fixed

    return {
        "agent_view": agent_view.astype(np.float32, copy=False),
        "base_view": base_view.astype(np.float32, copy=False),
        "scalars": scalars,
        "action_mask": action_mask,
    }
