"""Option-selector hybrid AE manager.

The neural model chooses one of a small set of strategic options; planner
variants execute the chosen option into a legal Bomberman action. This keeps
the learned policy's action space aligned with the fixed Novice map strategy:
choose the right intent for the matchup, while the existing planner still owns
movement legality, bomb escape mechanics, and map memory.
"""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path

import numpy as np
import torch

from ae_manager import AEManager
from encoder import BELIEF_CHANNELS, FrameStacker, SCALAR_DIM, rasterize_belief
from model import AGENT_VIEW_HW, BASE_VIEW_HW, BELIEF_HW, PolicyNetwork, VIEW_CHANNELS, build_policy_network
from option_policy import (
    NUM_OPTIONS,
    OPTION_BASE_BOMB,
    OPTION_COLLECT_MISSION,
    OPTION_COLLECT_RESOURCE,
    OPTION_DEFEND_BASE,
    OPTION_ESCAPE,
    OPTION_EXPLORE,
    OPTION_HUNT_ENEMY,
    OPTION_NAMES,
    OPTION_RUSH_BASE,
    option_name,
)


torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _candidate_checkpoints() -> list[Path]:
    here = Path(__file__).resolve().parent
    return [
        here / "models" / "option_policy.pt",
        here.parent / "models" / "option_policy.pt",
    ]


def _resolve_checkpoint_path() -> Path:
    override = os.environ.get("AE_OPTION_POLICY_CHECKPOINT")
    if override:
        return Path(override)
    for candidate in _candidate_checkpoints():
        if candidate.exists():
            return candidate
    return _candidate_checkpoints()[0]


_MODEL_CACHE: PolicyNetwork | None = None
_DEVICE_CACHE: torch.device | None = None
_N_FRAMES_CACHE: int | None = None
_USE_BELIEF_CACHE: bool = False


def _checkpoint_action_dim(state_dict: dict[str, torch.Tensor], fallback: int) -> int:
    final = state_dict.get("head.4.weight")
    if final is not None and len(final.shape) >= 1:
        return int(final.shape[0])
    return int(fallback)


def _warmup(model: PolicyNetwork, device: torch.device, n_frames: int, use_belief: bool) -> None:
    agent_view = torch.zeros(1, VIEW_CHANNELS * n_frames, AGENT_VIEW_HW[0], AGENT_VIEW_HW[1], device=device)
    base_view = torch.zeros(1, VIEW_CHANNELS * n_frames, BASE_VIEW_HW[0], BASE_VIEW_HW[1], device=device)
    scalars = torch.zeros(1, SCALAR_DIM * n_frames, device=device)
    belief = torch.zeros(1, BELIEF_CHANNELS, BELIEF_HW[0], BELIEF_HW[1], device=device) if use_belief else None
    with torch.inference_mode():
        model(agent_view, base_view, scalars, belief_map=belief)
        model(agent_view, base_view, scalars, belief_map=belief)


def _load_model(checkpoint_path: Path) -> tuple[PolicyNetwork, torch.device, int, bool]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    action_dim = int(ckpt.get("action_dim", _checkpoint_action_dim(state_dict, NUM_OPTIONS)))
    if action_dim != NUM_OPTIONS:
        raise ValueError(
            f"option checkpoint has action_dim={action_dim}, expected {NUM_OPTIONS}. "
            "Do not load a raw 6-action AE policy as an option selector."
        )
    n_frames = int(ckpt.get("n_frames", 4))
    use_belief = bool(ckpt.get("use_belief", False))
    model = build_policy_network(
        n_frames=n_frames,
        action_dim=NUM_OPTIONS,
        use_belief=use_belief,
        state_dict=state_dict,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    _warmup(model, device, n_frames, use_belief)
    print(
        f"AE option policy loaded from {checkpoint_path} "
        f"(n_frames={n_frames}, use_belief={use_belief}, "
        f"epoch={ckpt.get('epoch')}, val_acc={ckpt.get('val_acc')}, "
        f"ppo_eval={ckpt.get('ppo_eval_score')}, device={device})",
        flush=True,
    )
    return model, device, n_frames, use_belief


class OptionPolicyAEManager:
    """Inference wrapper for an 8-way option policy checkpoint."""

    def __init__(self):
        global _MODEL_CACHE, _DEVICE_CACHE, _N_FRAMES_CACHE, _USE_BELIEF_CACHE
        if _MODEL_CACHE is None:
            ckpt_path = _resolve_checkpoint_path()
            if not ckpt_path.exists():
                searched = [str(p) for p in _candidate_checkpoints()]
                raise FileNotFoundError(
                    f"AE option-policy checkpoint not found. Searched: {searched}. "
                    "Set AE_OPTION_POLICY_CHECKPOINT or train/copy option_policy.pt."
                )
            (_MODEL_CACHE, _DEVICE_CACHE, _N_FRAMES_CACHE, _USE_BELIEF_CACHE) = _load_model(ckpt_path)
        self.model = _MODEL_CACHE
        self.device = _DEVICE_CACHE
        self.n_frames = _N_FRAMES_CACHE or 4
        self.use_belief = _USE_BELIEF_CACHE
        self.stacker = FrameStacker(self.n_frames)
        self.belief_manager = AEManager()
        self._last_step: int | None = None

        self._agent_buf = torch.zeros(
            1, VIEW_CHANNELS * self.n_frames, AGENT_VIEW_HW[0], AGENT_VIEW_HW[1], device=self.device
        )
        self._base_buf = torch.zeros(
            1, VIEW_CHANNELS * self.n_frames, BASE_VIEW_HW[0], BASE_VIEW_HW[1], device=self.device
        )
        self._scalar_buf = torch.zeros(1, SCALAR_DIM * self.n_frames, device=self.device)
        self._belief_buf = (
            torch.zeros(1, BELIEF_CHANNELS, BELIEF_HW[0], BELIEF_HW[1], device=self.device)
            if self.use_belief else None
        )

    def _maybe_reset(self, observation: dict) -> None:
        try:
            step = int(observation.get("step", 0))
        except Exception:
            step = None
        if step == 0 or (self._last_step is not None and step is not None and step < self._last_step):
            self.stacker.reset()
            self.belief_manager = AEManager()
        self._last_step = step

    def _build_belief(self, observation: dict) -> np.ndarray | None:
        if not self.use_belief:
            return None
        return _memory_only_belief(self.belief_manager, observation, use_belief=True)

    def option_logits(self, observation: dict) -> tuple[int, torch.Tensor]:
        self._maybe_reset(observation)
        belief = self._build_belief(observation)
        stacked = self.stacker.observe(observation, belief_map=belief)
        self._agent_buf[0].copy_(torch.from_numpy(np.ascontiguousarray(stacked["agent_view"])))
        self._base_buf[0].copy_(torch.from_numpy(np.ascontiguousarray(stacked["base_view"])))
        self._scalar_buf[0].copy_(torch.from_numpy(np.ascontiguousarray(stacked["scalars"])))
        if belief is not None and self._belief_buf is not None:
            self._belief_buf[0].copy_(torch.from_numpy(np.ascontiguousarray(belief)))

        with torch.inference_mode():
            logits = self.model(
                self._agent_buf,
                self._base_buf,
                self._scalar_buf,
                belief_map=self._belief_buf if self.use_belief else None,
            ).squeeze(0)
        return int(logits.argmax().item()), logits.detach().clone()


def _memory_only_belief(manager: AEManager, observation: dict, use_belief: bool) -> np.ndarray | None:
    """Advance an AEManager's map memory without running target selection."""

    step = manager._as_int(observation.get("step"), default=(manager.last_step or 0) + 1)
    if manager.last_step is None or step == 0 or step < manager.last_step:
        manager._reset_memory()
    manager._age_bombs(step)
    manager._blast_cache = {}
    manager.last_step = step
    manager.team_bombs = manager._as_int(observation.get("team_bombs"), default=0)
    manager.health = manager._as_int(observation.get("health"), default=60)
    manager.base_health = manager._as_int(observation.get("base_health"), default=100)
    location = manager._location(observation.get("location"))
    direction = manager._as_int(observation.get("direction"), default=0) % 4
    manager._update_memory(observation, step, location, direction)
    if location is not None:
        manager.visit_count[location] = manager.visit_count.get(location, 0) + 1
        manager.recent_locations.append(location)
        if len(manager.recent_locations) > 10:
            manager.recent_locations.pop(0)
    return rasterize_belief(manager, observation) if use_belief else None


def _configure_variant(name: str) -> AEManager:
    m = AEManager()
    m.playbook = None
    if name == "rush":
        m.ENEMY_BASE_VALUE = 185.0
        m.DIST_PENALTY = 0.65
        m.PATH_THREAT_PENALTY = 1.2
        m.CELL_THREAT_PENALTY = 3.0
        m.ITEM_VALUES = {"mission": 10.0, "resource": 4.0, "recon": 1.0}
    elif name == "bomb":
        m.ENEMY_BASE_VALUE = 220.0
        m.DIST_PENALTY = 0.50
        m.PATH_THREAT_PENALTY = 0.8
        m.CELL_THREAT_PENALTY = 2.5
        m.ITEM_VALUES = {"mission": 5.0, "resource": 2.0, "recon": 0.5}
    elif name == "defend":
        m.BASE_DEFENSE_HEALTH = 101.0
        m.BASE_DEFENSE_RADIUS = 9
        m.ENEMY_CHASE_VALUE = 35.0
        m.ENEMY_CHASE_RADIUS = 8
        m.ENEMY_BASE_VALUE = 55.0
        m.PATH_THREAT_PENALTY = 3.0
        m.CELL_THREAT_PENALTY = 8.0
    elif name == "mission":
        m.ENEMY_BASE_VALUE = 70.0
        m.DIST_PENALTY = 1.0
        m.ITEM_VALUES = {"mission": 85.0, "resource": 18.0, "recon": 8.0}
    elif name == "resource":
        m.ENEMY_BASE_VALUE = 55.0
        m.DIST_PENALTY = 1.0
        m.ITEM_VALUES = {"mission": 45.0, "resource": 42.0, "recon": 16.0}
    elif name == "hunt":
        m.ENEMY_BASE_VALUE = 75.0
        m.ENEMY_CHASE_VALUE = 50.0
        m.ENEMY_CHASE_RADIUS = 9
        m.PREDICTIVE_BOMB_RANGE = 2
        m.PATH_THREAT_PENALTY = 0.8
        m.CELL_THREAT_PENALTY = 2.5
    return m


class OptionExecutor:
    """Execute an option through planner variants with safety vetoes."""

    FORWARD = AEManager.FORWARD
    BACKWARD = AEManager.BACKWARD
    LEFT = AEManager.LEFT
    RIGHT = AEManager.RIGHT
    STAY = AEManager.STAY
    PLACE_BOMB = AEManager.PLACE_BOMB

    def __init__(self):
        self.heuristic = AEManager()
        self.option_counts: Counter[str] = Counter()
        self.veto_counts: Counter[str] = Counter()
        self.decision_counts: Counter[str] = Counter()
        self._managers = {
            OPTION_RUSH_BASE: _configure_variant("rush"),
            OPTION_BASE_BOMB: _configure_variant("bomb"),
            OPTION_DEFEND_BASE: _configure_variant("defend"),
            OPTION_COLLECT_MISSION: _configure_variant("mission"),
            OPTION_COLLECT_RESOURCE: _configure_variant("resource"),
            OPTION_HUNT_ENEMY: _configure_variant("hunt"),
        }

    def reset(self) -> None:
        self.__init__()

    def act(self, option: int, observation: dict) -> int:
        action, _info = self.act_with_info(option, observation)
        return action

    def act_with_info(
        self,
        option: int,
        observation: dict,
        heuristic_action: int | None = None,
        heuristic_already_run: bool = False,
    ) -> tuple[int, dict]:
        option = int(option)
        option_label = option_name(option)
        self.option_counts[option_label] += 1
        if heuristic_action is None or not heuristic_already_run:
            heuristic_action = int(self.heuristic.ae(observation))
        suspended_bomb = self._suspend_hypothetical_heuristic_bomb(observation, heuristic_action)

        self._sync_inactive_managers(observation, selected=option)

        forced_escape = self._forced_escape_action(observation)
        if forced_escape is not None:
            self.veto_counts["active_escape"] += 1
            self.decision_counts[f"escape_{option_label}"] += 1
            return forced_escape, {"option": option_label, "executed": "escape", "veto": "active_escape"}

        if self._must_take_heuristic(observation):
            self.veto_counts["forced_heuristic"] += 1
            self.decision_counts[f"forced_{option_label}"] += 1
            return heuristic_action, {"option": option_label, "executed": "heuristic", "veto": "forced"}

        if option == OPTION_ESCAPE:
            candidate = self._escape_action(observation, heuristic_action)
            self.decision_counts["option_escape"] += 1
            return candidate, {"option": option_label, "executed": "escape", "veto": ""}

        if option == OPTION_DEFEND_BASE:
            candidate = self._defend_action(observation, heuristic_action)
            veto_reason = self._veto_reason(observation, candidate, heuristic_action)
            if veto_reason:
                self.veto_counts[veto_reason] += 1
                self.decision_counts[f"veto_{option_label}_{veto_reason}"] += 1
                self._restore_hypothetical_heuristic_bomb(suspended_bomb)
                return heuristic_action, {"option": option_label, "executed": "heuristic", "veto": veto_reason}
            if candidate == self.PLACE_BOMB:
                self._adopt_local_bomb(observation)
            self.decision_counts["option_defend_base"] += 1
            return candidate, {"option": option_label, "executed": "defend_base", "veto": ""}

        manager = self._managers.get(option)
        if manager is None:
            self.decision_counts[f"heuristic_{option_label}"] += 1
            self._restore_hypothetical_heuristic_bomb(suspended_bomb)
            return heuristic_action, {"option": option_label, "executed": "heuristic", "veto": ""}

        candidate = int(manager.ae(observation))
        veto_reason = self._veto_reason(observation, candidate, heuristic_action)
        if veto_reason:
            self.veto_counts[veto_reason] += 1
            self.decision_counts[f"veto_{option_label}_{veto_reason}"] += 1
            self._restore_hypothetical_heuristic_bomb(suspended_bomb)
            return heuristic_action, {"option": option_label, "executed": "heuristic", "veto": veto_reason}

        if candidate == self.PLACE_BOMB:
            self._adopt_bomb_state(manager, observation)
        self.decision_counts[f"option_{option_label}"] += 1
        return candidate, {"option": option_label, "executed": option_label, "veto": ""}

    def _suspend_hypothetical_heuristic_bomb(self, observation: dict, heuristic_action: int) -> dict | None:
        if heuristic_action != self.PLACE_BOMB:
            return None
        location = self.heuristic._location(observation.get("location"))
        if location is None:
            return None
        step = self.heuristic._as_int(observation.get("step"), default=self.heuristic.last_step or 0)
        bomb_state = self.heuristic.known_bombs.get(location)
        if bomb_state is None or int(bomb_state.get("last_step", -1)) != step:
            return None
        state = {
            "location": location,
            "bomb_state": dict(bomb_state),
            "escape_target": self.heuristic.escape_target,
            "escape_until_step": self.heuristic.escape_until_step,
        }
        self.heuristic.known_bombs.pop(location, None)
        self.heuristic.escape_target = None
        self.heuristic.escape_until_step = None
        return state

    def _restore_hypothetical_heuristic_bomb(self, state: dict | None) -> None:
        if not state:
            return
        location = state["location"]
        self.heuristic.known_bombs[location] = dict(state["bomb_state"])
        self.heuristic.escape_target = state["escape_target"]
        self.heuristic.escape_until_step = state["escape_until_step"]

    def _sync_inactive_managers(self, observation: dict, selected: int) -> None:
        if getattr(self.heuristic, "is_fixed_novice_map", False):
            for manager in self._managers.values():
                self._copy_fixed_map_state(manager)
        for option, manager in self._managers.items():
            if option == selected:
                continue
            _memory_only_belief(manager, observation, use_belief=False)
            if getattr(self.heuristic, "is_fixed_novice_map", False):
                self._copy_fixed_map_state(manager)

    def _copy_fixed_map_state(self, manager: AEManager) -> None:
        if getattr(manager, "is_fixed_novice_map", False):
            return
        manager.is_fixed_novice_map = getattr(self.heuristic, "is_fixed_novice_map", False)
        manager.fixed_team_idx = getattr(self.heuristic, "fixed_team_idx", None)
        manager.seen = set(getattr(self.heuristic, "seen", set()))
        manager.walls = set(getattr(self.heuristic, "walls", set()))
        manager.destructible = set(getattr(self.heuristic, "destructible", set()))
        manager.enemy_bases = dict(getattr(self.heuristic, "enemy_bases", {}))
        manager.last_seen_items = dict(getattr(self.heuristic, "last_seen_items", {}))
        manager.base_location = getattr(self.heuristic, "base_location", None)
        manager.dijkstra_bomb_cost = getattr(self.heuristic, "dijkstra_bomb_cost", manager.dijkstra_bomb_cost)

    def _must_take_heuristic(self, observation: dict) -> bool:
        if self._frozen_ticks(observation) > 0:
            return True
        if self.heuristic._location(observation.get("location")) is None:
            return True
        return False

    def _forced_escape_action(self, observation: dict) -> int | None:
        location = self.heuristic._location(observation.get("location"))
        if location is None or self.heuristic.escape_target is None:
            return None
        step = self.heuristic._as_int(observation.get("step"), default=self.heuristic.last_step or 0)
        path = self.heuristic._active_escape_path(location, step)
        if not path or len(path) < 2:
            return None
        direction = self.heuristic._as_int(observation.get("direction"), default=0) % 4
        action = self.heuristic._action_for_path(location, direction, path)
        if action is not None and self.heuristic._legal(observation, action):
            return int(action)
        return None

    def _escape_action(self, observation: dict, fallback: int) -> int:
        forced = self._forced_escape_action(observation)
        if forced is not None:
            return forced
        location = self.heuristic._location(observation.get("location"))
        if location is None:
            return fallback
        direction = self.heuristic._as_int(observation.get("direction"), default=0) % 4
        legal = [
            action for action in (self.FORWARD, self.BACKWARD, self.LEFT, self.RIGHT, self.STAY)
            if self.heuristic._legal(observation, action)
        ]
        if not legal:
            return fallback
        danger = self.heuristic._danger_cells()
        threats = self.heuristic._enemy_threat_cells()
        best_action = legal[0]
        best_score = -float("inf")
        base = self.heuristic.base_location
        for action in legal:
            new_pos, _new_dir = self.heuristic._simulate_action(location, direction, action)
            score = 0.0
            if new_pos in danger:
                score -= 200.0
            if new_pos in threats:
                score -= 30.0
            score += 2.0 * sum(1 for neighbor in self.heuristic._raw_neighbors(new_pos) if neighbor not in danger)
            score -= 0.3 * self.heuristic.visit_count.get(new_pos, 0)
            if base is not None:
                score += 0.05 * self.heuristic._manhattan(new_pos, base)
            if action == self.STAY:
                score -= 2.0
            if score > best_score:
                best_score = score
                best_action = action
        return int(best_action)

    def _defend_action(self, observation: dict, fallback: int) -> int:
        location = self.heuristic._location(observation.get("location"))
        if location is None:
            return fallback
        direction = self.heuristic._as_int(observation.get("direction"), default=0) % 4
        base = self.heuristic.base_location or self.heuristic._location(observation.get("base_location"))
        if base is None:
            return fallback

        danger = self.heuristic._danger_cells()
        step = self.heuristic.last_step if self.heuristic.last_step is not None else 0
        fresh_enemies = [
            pos for pos, last_seen in self.heuristic.enemy_agents.items()
            if step - int(last_seen) <= max(8, self.heuristic.ENEMY_STALENESS)
            and self.heuristic._manhattan(pos, base) <= 10
        ]
        if self.heuristic._legal(observation, self.PLACE_BOMB):
            blast = self.heuristic._blast_cells(location)
            if any(pos in blast for pos in fresh_enemies) and self._bomb_has_escape(observation):
                return self.PLACE_BOMB

        target = self._defense_target(location, base, fresh_enemies, danger)
        if target is not None and target != location:
            if getattr(self.heuristic, "is_fixed_novice_map", False):
                _distance, parent = self.heuristic._dijkstra_distance_map(location, danger)
            else:
                _distance, parent = self.heuristic._bfs_distance_map(location, danger)
            path = self.heuristic._reconstruct_path(parent, location, target)
            action = self.heuristic._action_for_path(location, direction, path)
            if action is not None and self.heuristic._legal(observation, action):
                return int(action)
        return self._base_guard_action(observation, location, direction, base, danger)

    def _defense_target(
        self,
        location: tuple[int, int],
        base: tuple[int, int],
        fresh_enemies: list[tuple[int, int]],
        danger: set[tuple[int, int]],
    ) -> tuple[int, int] | None:
        if getattr(self.heuristic, "is_fixed_novice_map", False):
            distance, parent = self.heuristic._dijkstra_distance_map(location, danger)
        else:
            distance, parent = self.heuristic._bfs_distance_map(location, danger)
        candidates: list[tuple[float, tuple[int, int]]] = []
        for enemy in fresh_enemies:
            for cell in (enemy, *tuple(self.heuristic._raw_neighbors(enemy))):
                if cell in distance:
                    score = 140.0 - 2.0 * self.heuristic._manhattan(cell, enemy) - distance[cell]
                    score -= 0.3 * self.heuristic._manhattan(cell, base)
                    candidates.append((score, cell))
        guard_cells = [base, *list(self.heuristic._raw_neighbors(base))]
        for cell in guard_cells:
            if cell in distance:
                score = 70.0 - distance[cell] - 0.5 * self.heuristic._manhattan(cell, base)
                candidates.append((score, cell))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def _base_guard_action(
        self,
        observation: dict,
        location: tuple[int, int],
        direction: int,
        base: tuple[int, int],
        danger: set[tuple[int, int]],
    ) -> int:
        legal = [
            action for action in (self.FORWARD, self.BACKWARD, self.LEFT, self.RIGHT, self.STAY)
            if self.heuristic._legal(observation, action)
        ]
        if not legal:
            return self.STAY
        threats = self.heuristic._enemy_threat_cells()
        best_action = legal[0]
        best_score = -float("inf")
        for action in legal:
            new_pos, _new_dir = self.heuristic._simulate_action(location, direction, action)
            score = -self.heuristic._manhattan(new_pos, base)
            if new_pos in danger:
                score -= 200.0
            if new_pos in threats:
                score -= 15.0
            score -= 0.2 * self.heuristic.visit_count.get(new_pos, 0)
            if action == self.STAY:
                score -= 0.5
            if score > best_score:
                best_score = score
                best_action = action
        return int(best_action)

    def _veto_reason(self, observation: dict, action: int, heuristic_action: int) -> str:
        if not self._action_legal(observation, action):
            return "illegal"
        if action == self.PLACE_BOMB and not self._bomb_has_escape(observation):
            return "unsafe_bomb"
        if self._steps_into_danger(observation, action) and not self._steps_into_danger(observation, heuristic_action):
            return "danger"
        if action == self.STAY and self._frozen_ticks(observation) == 0 and heuristic_action != self.STAY:
            if self.heuristic._legal(observation, heuristic_action):
                return "idle_stay"
        return ""

    def _action_legal(self, observation: dict, action: int) -> bool:
        mask = observation.get("action_mask")
        if mask is None:
            return 0 <= action <= self.PLACE_BOMB
        try:
            return bool(mask[action])
        except Exception:
            return 0 <= action <= self.PLACE_BOMB

    def _bomb_has_escape(self, observation: dict) -> bool:
        location = self.heuristic._location(observation.get("location"))
        if location is None:
            return False
        if self.heuristic._as_int(observation.get("team_bombs"), default=0) <= 0:
            return False
        blast = self.heuristic._blast_cells(location)
        base = self.heuristic.base_location or self.heuristic._location(observation.get("base_location"))
        if base is not None and base in blast:
            return False
        return self.heuristic._safe_escape_within(location, blast, self.heuristic.BOMB_TIMER) is not None

    def _steps_into_danger(self, observation: dict, action: int) -> bool:
        location = self.heuristic._location(observation.get("location"))
        if location is None:
            return False
        direction = self.heuristic._as_int(observation.get("direction"), default=0) % 4
        new_pos, _new_direction = self.heuristic._simulate_action(location, direction, action)
        danger = self.heuristic._danger_cells()
        if new_pos == location:
            return location in danger
        return new_pos in danger

    def _frozen_ticks(self, observation: dict) -> int:
        return self.heuristic._as_int(observation.get("frozen_ticks"), default=0)

    def _adopt_bomb_state(self, source: AEManager, observation: dict) -> None:
        location = self.heuristic._location(observation.get("location"))
        if location is not None and location in source.known_bombs:
            bomb_state = dict(source.known_bombs[location])
            self.heuristic.known_bombs[location] = bomb_state
            for manager in self._managers.values():
                manager.known_bombs[location] = dict(bomb_state)
        if source.escape_target is not None:
            self.heuristic.escape_target = source.escape_target
            self.heuristic.escape_until_step = source.escape_until_step
            for manager in self._managers.values():
                manager.escape_target = source.escape_target
                manager.escape_until_step = source.escape_until_step

    def _adopt_local_bomb(self, observation: dict) -> None:
        location = self.heuristic._location(observation.get("location"))
        if location is None:
            return
        blast = self.heuristic._blast_cells(location)
        escape = self.heuristic._safe_escape_within(location, blast, self.heuristic.BOMB_TIMER)
        if escape is None:
            return
        bomb_state = {
            "timer": self.heuristic.BOMB_TIMER,
            "own": True,
            "last_step": self.heuristic.last_step or 0,
        }
        self.heuristic.known_bombs[location] = bomb_state
        self.heuristic.escape_target = escape
        self.heuristic.escape_until_step = (self.heuristic.last_step or 0) + self.heuristic.BOMB_TIMER
        for manager in self._managers.values():
            manager.known_bombs[location] = dict(bomb_state)
            manager.escape_target = escape
            manager.escape_until_step = self.heuristic.escape_until_step


class OptionHybridAEManager:
    """Neural option selector plus safe planner execution."""

    def __init__(self, policy: OptionPolicyAEManager | None = None):
        self.policy = policy or OptionPolicyAEManager()
        self.executor = OptionExecutor()
        self.heuristic = self.executor.heuristic
        self.decision_counts = self.executor.decision_counts
        self.option_counts = self.executor.option_counts
        self.veto_counts = self.executor.veto_counts
        self.policy_conf_threshold = _env_float("AE_OPTION_HYBRID_CONF", 0.0)

    def ae(self, observation: dict) -> int:
        heuristic_action = int(self.heuristic.ae(observation))
        try:
            option, logits = self.policy.option_logits(observation)
        except Exception:
            self.decision_counts["policy_error_fallback"] += 1
            return heuristic_action

        if self.policy_conf_threshold > 0.0:
            probs = torch.softmax(logits, dim=-1)
            if float(probs.max().item()) < self.policy_conf_threshold:
                self.decision_counts["low_conf_fallback"] += 1
                return heuristic_action

        action, _info = self.executor.act_with_info(
            option,
            observation,
            heuristic_action=heuristic_action,
            heuristic_already_run=True,
        )
        return int(action)
