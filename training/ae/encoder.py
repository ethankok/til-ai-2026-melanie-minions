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

Belief-map augmentation (NEW, optional):
    belief_map:  (BELIEF_CHANNELS, 16, 16)   single frame, NOT stacked
                 — the heuristic's belief state is already cumulative,
                 so frame-stacking it adds no information.
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

# Belief-map channel layout. Order matters — checkpoints are tied to it.
# Keep in lockstep with `rasterize_belief`; if reordered/added, retrain
# (don't try to load an old checkpoint).
BELIEF_CHANNELS = 11
BELIEF_VISITED = 0
BELIEF_WALL = 1
BELIEF_DESTRUCTIBLE = 2
BELIEF_MISSION = 3
BELIEF_RECON = 4
BELIEF_RESOURCE = 5
BELIEF_ENEMY_AGENT = 6
BELIEF_ENEMY_BASE = 7
BELIEF_BOMB_BLAST = 8
BELIEF_OWN_POSITION = 9
BELIEF_BASE_POSITION = 10

# Freshness decay windows (steps): item/enemy memory fades linearly to zero.
ITEM_FRESH_WINDOW = 20.0
ENEMY_FRESH_WINDOW = 5.0


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


def rasterize_belief(ae_manager, observation: dict) -> np.ndarray:
    """Render the heuristic's belief state into a (K, 16, 16) tensor.

    K = ``BELIEF_CHANNELS``. The caller is responsible for ensuring
    ``ae_manager``'s memory has been updated with the current
    observation (via ``ae_manager._update_memory(...)`` or the public
    ``ae_manager.ae(...)`` call).

    Channel semantics (see constants above):
      0 visited            1 if the cell has ever been seen
      1 wall               1 if any wall edge is known here
      2 destructible       1 if any destructible-wall edge is known
      3 mission_fresh      freshness ∈ (0, 1]; older = lower
      4 recon_fresh        freshness ∈ (0, 1]
      5 resource_fresh     freshness ∈ (0, 1]
      6 enemy_agent_fresh  freshness ∈ (0, 1] over a 5-step window
      7 enemy_base         1 if known
      8 bomb_blast         max(1 - timer/BOMB_TIMER, 0) per cell in any
                           known bomb's blast — higher = sooner detonation
      9 own_position       1 at the agent's current cell
     10 base_position      1 at our base's cell
    """

    grid = max(GRID_SIZE, ae_manager.grid_size)
    out = np.zeros((BELIEF_CHANNELS, grid, grid), dtype=np.float32)

    step = ae_manager.last_step if ae_manager.last_step is not None else 0

    for x, y in ae_manager.seen:
        if 0 <= x < grid and 0 <= y < grid:
            out[BELIEF_VISITED, x, y] = 1.0

    # Wall edges collapse the four directional edges into a single
    # "any wall here" channel; per-direction detail lives in agent_viewcone.
    for x, y, _d in ae_manager.walls:
        if 0 <= x < grid and 0 <= y < grid:
            out[BELIEF_WALL, x, y] = 1.0
    for x, y, _d in ae_manager.destructible:
        if 0 <= x < grid and 0 <= y < grid:
            out[BELIEF_DESTRUCTIBLE, x, y] = 1.0

    item_to_channel = {
        "mission": BELIEF_MISSION,
        "recon": BELIEF_RECON,
        "resource": BELIEF_RESOURCE,
    }
    for (x, y), (kind, last_seen) in ae_manager.last_seen_items.items():
        ch = item_to_channel.get(kind)
        if ch is None or not (0 <= x < grid and 0 <= y < grid):
            continue
        age = max(0, step - int(last_seen))
        fresh = max(0.0, 1.0 - age / ITEM_FRESH_WINDOW)
        out[ch, x, y] = max(out[ch, x, y], fresh)

    for (x, y), last_seen in ae_manager.enemy_agents.items():
        if not (0 <= x < grid and 0 <= y < grid):
            continue
        age = max(0, step - int(last_seen))
        fresh = max(0.0, 1.0 - age / ENEMY_FRESH_WINDOW)
        out[BELIEF_ENEMY_AGENT, x, y] = max(out[BELIEF_ENEMY_AGENT, x, y], fresh)

    for x, y in ae_manager.enemy_bases:
        if 0 <= x < grid and 0 <= y < grid:
            out[BELIEF_ENEMY_BASE, x, y] = 1.0

    # Bomb blast cells weighted by 1 - timer/BOMB_TIMER (1-tick bomb -> 1.0,
    # BOMB_TIMER-tick bomb -> 0): graded "imminent danger" signal.
    bomb_timer_max = float(getattr(ae_manager, "BOMB_TIMER", 3))
    for bomb_pos, data in ae_manager.known_bombs.items():
        timer = max(1, int(data.get("timer", bomb_timer_max)))
        urgency = max(0.0, 1.0 - (timer - 1) / bomb_timer_max)
        try:
            blast = ae_manager._blast_cells(bomb_pos)
        except Exception:
            blast = {bomb_pos}
        for x, y in blast:
            if 0 <= x < grid and 0 <= y < grid:
                out[BELIEF_BOMB_BLAST, x, y] = max(
                    out[BELIEF_BOMB_BLAST, x, y], urgency
                )

    location = observation.get("location")
    if location is not None:
        try:
            if hasattr(location, "tolist"):
                location = location.tolist()
            x, y = int(location[0]), int(location[1])
            if 0 <= x < grid and 0 <= y < grid:
                out[BELIEF_OWN_POSITION, x, y] = 1.0
        except Exception:
            pass

    base_location = observation.get("base_location")
    if base_location is not None:
        try:
            if hasattr(base_location, "tolist"):
                base_location = base_location.tolist()
            x, y = int(base_location[0]), int(base_location[1])
            if 0 <= x < grid and 0 <= y < grid:
                out[BELIEF_BASE_POSITION, x, y] = 1.0
        except Exception:
            pass

    if grid != GRID_SIZE:
        cropped = np.zeros((BELIEF_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        copy_dim = min(grid, GRID_SIZE)
        cropped[:, :copy_dim, :copy_dim] = out[:, :copy_dim, :copy_dim]
        return cropped
    return out


class FrameStacker:
    """Rolling buffer of the last N encoded observations.

    First-call behavior: pads the buffer with copies of the first observation
    so the policy always receives ``N`` frames, even on step 0/1/2 of a game.
    Designed for both training (one stacker per env instance) and deployment
    (one stacker per `PolicyAEManager`, cleared on `/reset` via factory).

    Belief maps (when supplied) are NOT stacked — only the latest is kept,
    since the belief state is already cumulative over the whole episode.
    """

    def __init__(self, n_frames: int = 4):
        if n_frames < 1:
            raise ValueError("n_frames must be ≥ 1")
        self.n_frames = n_frames
        self.frames: list[dict] = []
        self.latest_belief: np.ndarray | None = None

    def reset(self) -> None:
        self.frames = []
        self.latest_belief = None

    def observe(self, obs: dict, belief_map: np.ndarray | None = None) -> dict:
        """Encode `obs`, push into the buffer, return the stacked input.

        If ``belief_map`` is provided, the latest belief is included in the
        returned dict under key ``belief_map`` (single frame, not stacked).
        """
        encoded = encode_observation(obs)
        if not self.frames:
            self.frames = [encoded] * self.n_frames
        else:
            self.frames.append(encoded)
            if len(self.frames) > self.n_frames:
                self.frames.pop(0)
        if belief_map is not None:
            self.latest_belief = belief_map.astype(np.float32, copy=False)
        out = self._stack()
        if self.latest_belief is not None:
            out["belief_map"] = self.latest_belief
        return out

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