"""Tactical-option hybrid AE manager.

This is the second-generation option path for semifinals work. The policy
chooses one of 12 tactical macros, then a planner-backed executor converts the
macro into a legal raw Bomberman action.
"""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path

import numpy as np
import torch

from ae_manager import AEManager
from encoder import BELIEF_CHANNELS, FrameStacker, SCALAR_DIM
from model import AGENT_VIEW_HW, BASE_VIEW_HW, BELIEF_HW, PolicyNetwork, VIEW_CHANNELS, build_policy_network
from option_hybrid_manager import (
    OPTION_BASE_BOMB,
    OPTION_COLLECT_MISSION,
    OPTION_COLLECT_RESOURCE,
    OPTION_ESCAPE,
    OPTION_HUNT_ENEMY,
    OPTION_RUSH_BASE,
    OptionExecutor,
    _memory_only_belief,
)
from tactical_policy import (
    NUM_TACTICAL_OPTIONS,
    TACTICAL_BOMB_BASE_THREAT,
    TACTICAL_BOMB_ENEMY_BASE,
    TACTICAL_COLLECT_MISSION_SAFE,
    TACTICAL_COLLECT_RESOURCE_SAFE,
    TACTICAL_COUNTER_RUSH,
    TACTICAL_DENY_ENEMY_MISSION,
    TACTICAL_ESCAPE_BOMB,
    TACTICAL_GUARD_BASE_LANE,
    TACTICAL_HUNT_VISIBLE_ENEMY,
    TACTICAL_INTERCEPT_BASE_THREAT,
    TACTICAL_OPTION_NAMES,
    TACTICAL_RUSH_ENEMY_BASE,
    TACTICAL_STALL_WHEN_WINNING,
    tactical_from_manager,
    tactical_option_name,
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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_option_set(name: str) -> set[int] | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    allowed: set[int] = set()
    by_name = {option_name: i for i, option_name in enumerate(TACTICAL_OPTION_NAMES)}
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if token.isdigit():
            idx = int(token)
        else:
            idx = by_name.get(token)
        if idx is not None and 0 <= int(idx) < NUM_TACTICAL_OPTIONS:
            allowed.add(int(idx))
    return allowed


def _env_transition_set(name: str) -> set[tuple[int, int]] | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    by_name = {option_name: i for i, option_name in enumerate(TACTICAL_OPTION_NAMES)}
    allowed: set[tuple[int, int]] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if "->" in token:
            left, right = token.split("->", 1)
        elif ":" in token:
            left, right = token.split(":", 1)
        else:
            continue
        left = left.strip()
        right = right.strip()
        prior = int(left) if left.isdigit() else by_name.get(left)
        option = int(right) if right.isdigit() else by_name.get(right)
        if (
            prior is not None
            and option is not None
            and 0 <= int(prior) < NUM_TACTICAL_OPTIONS
            and 0 <= int(option) < NUM_TACTICAL_OPTIONS
        ):
            allowed.add((int(prior), int(option)))
    return allowed


def _env_string_set(name: str, valid: set[str]) -> set[str] | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    out = {part.strip().lower() for part in raw.split(",") if part.strip()}
    return {part for part in out if part in valid}


def _candidate_checkpoints() -> list[Path]:
    here = Path(__file__).resolve().parent
    return [
        here / "models" / "tactical_policy.pt",
        here.parent / "models" / "tactical_policy.pt",
    ]


def _resolve_checkpoint_path() -> Path:
    override = os.environ.get("AE_TACTICAL_POLICY_CHECKPOINT")
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
_DELTA_TRANSITION_COUNTS_CACHE: np.ndarray | None = None
_HARM_AWARE_CACHE: dict[str, np.ndarray] | None = None


def _harm_aware_from_checkpoint(ckpt: dict) -> dict[str, np.ndarray]:
    """Pull W1.1 harm-aware matrices from a tactical checkpoint.

    Pre-W1.1 checkpoints lack these keys; we return zeros so the gate sees
    `attempted == 0` everywhere and skips the new harm-rate / net-delta
    checks. The legacy `positive_delta_transition_counts >= min_delta_support`
    check still applies in that fallback path.
    """
    shape_2d = (NUM_TACTICAL_OPTIONS, NUM_TACTICAL_OPTIONS)
    shape_3d = (NUM_TACTICAL_OPTIONS, NUM_TACTICAL_OPTIONS, 3)
    keys_2d_int = [
        "attempted_transition_counts",
        "positive_transition_counts_full",
        "negative_transition_counts",
    ]
    keys_2d_float = [
        "transition_net_delta_sum",
        "transition_weight_sum",
        "transition_weighted_delta_sum",
    ]
    keys_3d_int = ["bucket_attempted", "bucket_positive"]
    keys_3d_float = ["bucket_net_delta_sum"]
    out: dict[str, np.ndarray] = {}
    for key in keys_2d_int:
        arr = ckpt.get(key)
        out[key] = np.asarray(arr, dtype=np.int64) if arr is not None else np.zeros(shape_2d, dtype=np.int64)
        if out[key].shape != shape_2d:
            out[key] = np.zeros(shape_2d, dtype=np.int64)
    for key in keys_2d_float:
        arr = ckpt.get(key)
        out[key] = np.asarray(arr, dtype=np.float64) if arr is not None else np.zeros(shape_2d, dtype=np.float64)
        if out[key].shape != shape_2d:
            out[key] = np.zeros(shape_2d, dtype=np.float64)
    for key in keys_3d_int:
        arr = ckpt.get(key)
        out[key] = np.asarray(arr, dtype=np.int64) if arr is not None else np.zeros(shape_3d, dtype=np.int64)
        if out[key].shape != shape_3d:
            out[key] = np.zeros(shape_3d, dtype=np.int64)
    for key in keys_3d_float:
        arr = ckpt.get(key)
        out[key] = np.asarray(arr, dtype=np.float64) if arr is not None else np.zeros(shape_3d, dtype=np.float64)
        if out[key].shape != shape_3d:
            out[key] = np.zeros(shape_3d, dtype=np.float64)
    return out


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


def _load_model(
    checkpoint_path: Path,
) -> tuple[PolicyNetwork, torch.device, int, bool, np.ndarray, dict[str, np.ndarray]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    action_dim = int(ckpt.get("action_dim", _checkpoint_action_dim(state_dict, NUM_TACTICAL_OPTIONS)))
    if action_dim != NUM_TACTICAL_OPTIONS:
        raise ValueError(
            f"tactical checkpoint has action_dim={action_dim}, expected {NUM_TACTICAL_OPTIONS}."
        )
    n_frames = int(ckpt.get("n_frames", 4))
    use_belief = bool(ckpt.get("use_belief", False))
    model = build_policy_network(
        n_frames=n_frames,
        action_dim=NUM_TACTICAL_OPTIONS,
        use_belief=use_belief,
        state_dict=state_dict,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    _warmup(model, device, n_frames, use_belief)
    delta_transition_counts = np.asarray(
        ckpt.get(
            "positive_delta_transition_counts",
            np.zeros((NUM_TACTICAL_OPTIONS, NUM_TACTICAL_OPTIONS), dtype=np.int32),
        ),
        dtype=np.float32,
    )
    if delta_transition_counts.shape != (NUM_TACTICAL_OPTIONS, NUM_TACTICAL_OPTIONS):
        delta_transition_counts = np.zeros((NUM_TACTICAL_OPTIONS, NUM_TACTICAL_OPTIONS), dtype=np.float32)
    harm_aware = _harm_aware_from_checkpoint(ckpt)
    attempted_total = int(harm_aware["attempted_transition_counts"].sum())
    print(
        f"AE tactical policy loaded from {checkpoint_path} "
        f"(n_frames={n_frames}, use_belief={use_belief}, "
        f"epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss')}, "
        f"weighted_delta={ckpt.get('weighted_delta_mean')}, "
        f"harm_aware_attempted_total={attempted_total}, device={device})",
        flush=True,
    )
    return model, device, n_frames, use_belief, delta_transition_counts, harm_aware


class TacticalPolicyAEManager:
    """Inference wrapper for a 12-way tactical option checkpoint."""

    def __init__(self):
        global _MODEL_CACHE, _DEVICE_CACHE, _N_FRAMES_CACHE, _USE_BELIEF_CACHE
        global _DELTA_TRANSITION_COUNTS_CACHE, _HARM_AWARE_CACHE
        if _MODEL_CACHE is None:
            ckpt_path = _resolve_checkpoint_path()
            if not ckpt_path.exists():
                searched = [str(p) for p in _candidate_checkpoints()]
                raise FileNotFoundError(
                    f"AE tactical-policy checkpoint not found. Searched: {searched}. "
                    "Set AE_TACTICAL_POLICY_CHECKPOINT or train/copy tactical_policy.pt."
                )
            (
                _MODEL_CACHE,
                _DEVICE_CACHE,
                _N_FRAMES_CACHE,
                _USE_BELIEF_CACHE,
                _DELTA_TRANSITION_COUNTS_CACHE,
                _HARM_AWARE_CACHE,
            ) = _load_model(ckpt_path)
        self.model = _MODEL_CACHE
        self.device = _DEVICE_CACHE
        self.n_frames = _N_FRAMES_CACHE or 4
        self.use_belief = _USE_BELIEF_CACHE
        self.delta_transition_counts = _DELTA_TRANSITION_COUNTS_CACHE
        self.harm_aware = _HARM_AWARE_CACHE or {}
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

    def tactical_logits(self, observation: dict) -> tuple[int, torch.Tensor]:
        self._maybe_reset(observation)
        belief = _memory_only_belief(self.belief_manager, observation, use_belief=self.use_belief)
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


class TacticalExecutor(OptionExecutor):
    """Execute sharper tactical macros with the old planner as fallback."""

    _OPTION_MAP = {
        TACTICAL_ESCAPE_BOMB: OPTION_ESCAPE,
        TACTICAL_RUSH_ENEMY_BASE: OPTION_RUSH_BASE,
        TACTICAL_BOMB_ENEMY_BASE: OPTION_BASE_BOMB,
        TACTICAL_HUNT_VISIBLE_ENEMY: OPTION_HUNT_ENEMY,
        TACTICAL_COLLECT_MISSION_SAFE: OPTION_COLLECT_MISSION,
        TACTICAL_COLLECT_RESOURCE_SAFE: OPTION_COLLECT_RESOURCE,
    }

    def __init__(self):
        super().__init__()
        self.tactical_counts: Counter[str] = Counter()

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
        label = tactical_option_name(option)
        self.tactical_counts[label] += 1

        mapped = self._OPTION_MAP.get(option)
        if mapped is not None:
            action, info = super().act_with_info(
                mapped,
                observation,
                heuristic_action=heuristic_action,
                heuristic_already_run=heuristic_already_run,
            )
            info["tactical_option"] = label
            return action, info

        if heuristic_action is None or not heuristic_already_run:
            heuristic_action = int(self.heuristic.ae(observation))
        suspended_bomb = self._suspend_hypothetical_heuristic_bomb(observation, heuristic_action)
        self._sync_inactive_managers(observation, selected=-1)

        forced_escape = self._forced_escape_action(observation)
        if forced_escape is not None:
            self.veto_counts["active_escape"] += 1
            self.decision_counts[f"escape_{label}"] += 1
            return forced_escape, {"tactical_option": label, "executed": "escape", "veto": "active_escape"}

        if self._must_take_heuristic(observation):
            self.veto_counts["forced_heuristic"] += 1
            self.decision_counts[f"forced_{label}"] += 1
            return heuristic_action, {"tactical_option": label, "executed": "heuristic", "veto": "forced"}

        candidate = self._candidate_for_tactical(option, observation, heuristic_action)
        veto_reason = self._tactical_veto_reason(
            observation,
            candidate,
            heuristic_action,
            allow_idle=(option == TACTICAL_STALL_WHEN_WINNING),
        )
        if veto_reason:
            self.veto_counts[veto_reason] += 1
            self.decision_counts[f"veto_{label}_{veto_reason}"] += 1
            self._restore_hypothetical_heuristic_bomb(suspended_bomb)
            return heuristic_action, {"tactical_option": label, "executed": "heuristic", "veto": veto_reason}

        if candidate == self.PLACE_BOMB:
            self._adopt_local_bomb(observation)
        self.decision_counts[f"tactical_{label}"] += 1
        return candidate, {"tactical_option": label, "executed": label, "veto": ""}

    def _candidate_for_tactical(self, option: int, observation: dict, fallback: int) -> int:
        if option == TACTICAL_INTERCEPT_BASE_THREAT:
            return self._intercept_base_threat(observation, fallback)
        if option == TACTICAL_GUARD_BASE_LANE:
            return self._guard_base_lane(observation, fallback)
        if option == TACTICAL_BOMB_BASE_THREAT:
            return self._bomb_base_threat(observation, fallback)
        if option == TACTICAL_DENY_ENEMY_MISSION:
            return self._deny_enemy_mission(observation, fallback)
        if option == TACTICAL_COUNTER_RUSH:
            return self._counter_rush(observation, fallback)
        if option == TACTICAL_STALL_WHEN_WINNING:
            return self._stall_when_winning(observation, fallback)
        return fallback

    def _fresh_base_threats(self) -> list[tuple[int, int]]:
        base = self.heuristic.base_location
        if base is None:
            return []
        step = self.heuristic.last_step if self.heuristic.last_step is not None else 0
        radius = max(8, self.heuristic.BASE_DEFENSE_RADIUS + 2)
        threats = []
        for pos, last_seen in self.heuristic.enemy_agents.items():
            if step - int(last_seen) <= max(8, self.heuristic.ENEMY_STALENESS) and self.heuristic._manhattan(pos, base) <= radius:
                threats.append(pos)
        threats.sort(key=lambda p: self.heuristic._manhattan(p, base))
        return threats

    def _action_to_target(
        self,
        observation: dict,
        target: tuple[int, int],
        danger: set[tuple[int, int]] | None = None,
    ) -> int | None:
        location = self.heuristic._location(observation.get("location"))
        if location is None:
            return None
        direction = self.heuristic._as_int(observation.get("direction"), default=0) % 4
        danger = danger or self.heuristic._danger_cells()
        if getattr(self.heuristic, "is_fixed_novice_map", False):
            _distance, parent = self.heuristic._dijkstra_distance_map(location, danger)
        else:
            _distance, parent = self.heuristic._bfs_distance_map(location, danger)
        path = self.heuristic._reconstruct_path(parent, location, target)
        action = self.heuristic._action_for_path(location, direction, path)
        if action is not None and self.heuristic._legal(observation, action):
            return int(action)
        return None

    def _intercept_base_threat(self, observation: dict, fallback: int) -> int:
        danger = self.heuristic._danger_cells()
        location = self.heuristic._location(observation.get("location"))
        if location is None:
            return fallback
        targets: list[tuple[float, tuple[int, int]]] = []
        base = self.heuristic.base_location
        for enemy in self._fresh_base_threats():
            for cell in (enemy, *tuple(self.heuristic._raw_neighbors(enemy))):
                if cell in danger:
                    continue
                score = 120.0 - self.heuristic._manhattan(location, cell)
                if base is not None:
                    score -= 0.5 * self.heuristic._manhattan(cell, base)
                targets.append((score, cell))
        if not targets:
            return self._guard_base_lane(observation, fallback)
        for _score, target in sorted(targets, key=lambda item: -item[0]):
            action = self._action_to_target(observation, target, danger)
            if action is not None:
                return action
        return fallback

    def _guard_base_lane(self, observation: dict, fallback: int) -> int:
        base = self.heuristic.base_location or self.heuristic._location(observation.get("base_location"))
        location = self.heuristic._location(observation.get("location"))
        if base is None or location is None:
            return fallback
        danger = self.heuristic._danger_cells()
        candidates = [base, *list(self.heuristic._raw_neighbors(base))]
        threats = self._fresh_base_threats()
        if threats:
            enemy = threats[0]
            candidates.extend(self.heuristic._raw_neighbors(enemy))
        scored = []
        for cell in candidates:
            if cell in danger:
                continue
            score = 80.0 - self.heuristic._manhattan(cell, base) - 0.3 * self.heuristic._manhattan(location, cell)
            scored.append((score, cell))
        for _score, target in sorted(scored, key=lambda item: -item[0]):
            action = self._action_to_target(observation, target, danger)
            if action is not None:
                return action
        return self._base_guard_action(
            observation,
            location,
            self.heuristic._as_int(observation.get("direction"), default=0) % 4,
            base,
            danger,
        )

    def _bomb_base_threat(self, observation: dict, fallback: int) -> int:
        location = self.heuristic._location(observation.get("location"))
        if location is None:
            return fallback
        if self.heuristic._legal(observation, self.PLACE_BOMB):
            blast = self.heuristic._blast_cells(location)
            base = self.heuristic.base_location or self.heuristic._location(observation.get("base_location"))
            base_safe = base is None or base not in blast
            if base_safe and any(pos in blast for pos in self._fresh_base_threats()) and self._bomb_has_escape(observation):
                return self.PLACE_BOMB
        return self._intercept_base_threat(observation, fallback)

    def _deny_enemy_mission(self, observation: dict, fallback: int) -> int:
        location = self.heuristic._location(observation.get("location"))
        if location is None:
            return fallback
        danger = self.heuristic._danger_cells()
        enemies = list(self.heuristic.enemy_agents.keys())
        missions = [pos for pos, (kind, _step) in self.heuristic.last_seen_items.items() if kind == "mission"]
        if not missions:
            return fallback
        scored = []
        for mission in missions:
            enemy_near = min((self.heuristic._manhattan(mission, enemy) for enemy in enemies), default=8)
            score = 90.0 - 0.8 * enemy_near - self.heuristic._manhattan(location, mission)
            scored.append((score, mission))
        for _score, target in sorted(scored, key=lambda item: -item[0]):
            action = self._action_to_target(observation, target, danger)
            if action is not None:
                return action
        return fallback

    def _counter_rush(self, observation: dict, fallback: int) -> int:
        if self._fresh_base_threats():
            action = self._bomb_base_threat(observation, fallback)
            if action != fallback:
                return action
            return self._intercept_base_threat(observation, fallback)
        return fallback

    def _stall_when_winning(self, observation: dict, fallback: int) -> int:
        location = self.heuristic._location(observation.get("location"))
        if location is None:
            return fallback
        if self._fresh_base_threats():
            return self._guard_base_lane(observation, fallback)
        if (
            self.heuristic._legal(observation, self.STAY)
            and location not in self.heuristic._danger_cells()
            and location not in self.heuristic._enemy_threat_cells()
        ):
            return self.STAY
        return self._escape_action(observation, fallback)

    def _tactical_veto_reason(
        self,
        observation: dict,
        action: int,
        heuristic_action: int,
        allow_idle: bool = False,
    ) -> str:
        if not self._action_legal(observation, action):
            return "illegal"
        if action == self.PLACE_BOMB and not self._bomb_has_escape(observation):
            return "unsafe_bomb"
        if self._steps_into_danger(observation, action) and not self._steps_into_danger(observation, heuristic_action):
            return "danger"
        if not allow_idle and action == self.STAY and self._frozen_ticks(observation) == 0 and heuristic_action != self.STAY:
            if self.heuristic._legal(observation, heuristic_action):
                return "idle_stay"
        return ""


class TacticalHybridAEManager:
    """Neural tactical option selector plus safe planner execution."""

    def __init__(self, policy: TacticalPolicyAEManager | None = None):
        self.policy = policy or TacticalPolicyAEManager()
        self.executor = TacticalExecutor()
        self.heuristic = self.executor.heuristic
        self.decision_counts = self.executor.decision_counts
        self.option_counts = self.executor.tactical_counts
        self.veto_counts = self.executor.veto_counts
        self.profile = os.environ.get("AE_TACTICAL_PROFILE", "safe").strip().lower()
        bracket_profile = self.profile in {"bracket", "learned", "tactical"}
        self.policy_conf_threshold = _env_float("AE_TACTICAL_HYBRID_CONF", 0.0)
        self.delta_conf_threshold = _env_float("AE_TACTICAL_DELTA_CONF", 0.60 if bracket_profile else 0.80)
        self.allow_mapped_deltas = _env_bool("AE_TACTICAL_ALLOW_MAPPED_DELTAS", False)
        self.require_delta_support = _env_bool("AE_TACTICAL_REQUIRE_DELTA_SUPPORT", True)
        # Legacy: min count of positive (delta>0, option!=prior) samples for
        # this transition. Default preserves bracket-profile behaviour from
        # the 26 May 400-game checkpoint deploy.
        self.min_delta_support = int(_env_float("AE_TACTICAL_MIN_DELTA_SUPPORT", 3.0 if bracket_profile else 999999.0))
        # W1.1 harm-aware gates. Defaults are permissive (== "off") so that
        # an unset env yields exactly the pre-W1.1 behaviour. Enable per
        # deployment with explicit env overrides — see ae/NOTES.md for the
        # recommended starting point (e.g. positive_rate>=0.55, mean_delta>=0).
        self.min_positive_rate = _env_float("AE_TACTICAL_MIN_POSITIVE_RATE", 0.0)
        self.min_net_delta = _env_float("AE_TACTICAL_MIN_NET_DELTA", float("-inf"))
        self.min_attempted = int(_env_float("AE_TACTICAL_MIN_ATTEMPTED", 0.0))
        # When 1, additionally check bucket_attempted / bucket_positive for
        # the current distance bucket (near/mid/far). When 0 (default) only
        # the global per-transition matrices are consulted.
        self.use_distance_bucket_gate = _env_bool("AE_TACTICAL_USE_DISTANCE_BUCKET_GATE", False)
        self.allowed_delta_options = _env_option_set("AE_TACTICAL_ALLOWED_DELTA_OPTIONS")
        self.allowed_delta_transitions = _env_transition_set("AE_TACTICAL_ALLOWED_DELTA_TRANSITIONS")
        self.allowed_delta_distance_buckets = _env_string_set(
            "AE_TACTICAL_ALLOWED_DELTA_DISTANCE_BUCKETS",
            {"near", "mid", "far"},
        )

    _BUCKET_INDEX = {"near": 0, "mid": 1, "far": 2}

    def _delta_is_supported(
        self,
        prior_option: int,
        option: int,
        bucket: str | None,
    ) -> tuple[bool, str]:
        """Return (ok, reason) for the harm-aware support check.

        Defaults make this function pass-through (always ok) when the env
        thresholds are at their defaults. With harm-aware npz metadata
        present in the checkpoint, it requires:
          attempted >= min_attempted
          positive / attempted >= min_positive_rate
          mean_net_delta >= min_net_delta
        per (prior_option, option), and optionally per distance bucket.
        """
        harm = self.policy.harm_aware or {}
        attempted_mat = harm.get("attempted_transition_counts")
        if attempted_mat is None:
            return True, "no_harm_aware_data"
        attempted = int(attempted_mat[prior_option, option])
        if attempted < self.min_attempted:
            return False, "insufficient_attempted"
        if attempted > 0:
            positive_mat = harm.get("positive_transition_counts_full")
            net_delta_mat = harm.get("transition_net_delta_sum")
            positive = int(positive_mat[prior_option, option]) if positive_mat is not None else 0
            mean_delta = (float(net_delta_mat[prior_option, option]) / attempted) if net_delta_mat is not None else 0.0
            pos_rate = positive / attempted
            if pos_rate < self.min_positive_rate:
                return False, "low_positive_rate"
            if mean_delta < self.min_net_delta:
                return False, "low_mean_net_delta"
        if self.use_distance_bucket_gate and bucket is not None:
            bucket_idx = self._BUCKET_INDEX.get(bucket)
            if bucket_idx is not None:
                bucket_attempted_mat = harm.get("bucket_attempted")
                bucket_positive_mat = harm.get("bucket_positive")
                bucket_net_delta_mat = harm.get("bucket_net_delta_sum")
                b_att = int(bucket_attempted_mat[prior_option, option, bucket_idx]) if bucket_attempted_mat is not None else 0
                if b_att < self.min_attempted:
                    return False, "bucket_insufficient_attempted"
                if b_att > 0:
                    b_pos = int(bucket_positive_mat[prior_option, option, bucket_idx]) if bucket_positive_mat is not None else 0
                    b_mean = (float(bucket_net_delta_mat[prior_option, option, bucket_idx]) / b_att) if bucket_net_delta_mat is not None else 0.0
                    if (b_pos / b_att) < self.min_positive_rate:
                        return False, "bucket_low_positive_rate"
                    if b_mean < self.min_net_delta:
                        return False, "bucket_low_mean_net_delta"
        return True, "ok"

    def _distance_bucket(self, observation: dict) -> str | None:
        location = self.heuristic._location(observation.get("location"))
        base = self.heuristic.base_location or self.heuristic._location(observation.get("base_location"))
        if location is None or base is None:
            return None
        dist = self.heuristic._manhattan(location, base)
        return "near" if dist <= 4 else ("mid" if dist <= 8 else "far")

    def ae(self, observation: dict) -> int:
        heuristic_action = int(self.heuristic.ae(observation))
        prior_option = int(tactical_from_manager(self.heuristic, action=heuristic_action))
        try:
            option, logits = self.policy.tactical_logits(observation)
        except Exception:
            self.decision_counts["policy_error_fallback"] += 1
            return heuristic_action

        probs = None

        if self.policy_conf_threshold > 0.0:
            probs = torch.softmax(logits, dim=-1)
            if float(probs.max().item()) < self.policy_conf_threshold:
                self.decision_counts["low_conf_fallback"] += 1
                return heuristic_action

        if int(option) == prior_option:
            self.decision_counts[f"exact_heuristic_{tactical_option_name(prior_option)}"] += 1
            return heuristic_action

        if not self.allow_mapped_deltas and int(option) in TacticalExecutor._OPTION_MAP:
            self.decision_counts[f"mapped_delta_fallback_{tactical_option_name(int(option))}"] += 1
            return heuristic_action

        if self.allowed_delta_options is not None and int(option) not in self.allowed_delta_options:
            self.decision_counts[f"disallowed_delta_{tactical_option_name(int(option))}"] += 1
            return heuristic_action

        if (
            self.allowed_delta_transitions is not None
            and (prior_option, int(option)) not in self.allowed_delta_transitions
        ):
            self.decision_counts[
                f"disallowed_delta_{tactical_option_name(prior_option)}_to_{tactical_option_name(int(option))}"
            ] += 1
            return heuristic_action

        bucket = self._distance_bucket(observation)
        if self.allowed_delta_distance_buckets is not None and bucket not in self.allowed_delta_distance_buckets:
            self.decision_counts[f"disallowed_delta_distance_{bucket or 'unknown'}"] += 1
            return heuristic_action

        if self.require_delta_support:
            support = 0.0
            counts = self.policy.delta_transition_counts
            if counts is not None:
                support = float(counts[prior_option, int(option)])
            if support < self.min_delta_support:
                self.decision_counts[
                    f"unsupported_delta_{tactical_option_name(prior_option)}_to_{tactical_option_name(int(option))}"
                ] += 1
                return heuristic_action

        # W1.1 harm-aware gate. Returns ok=True when env thresholds are at
        # defaults (no opinion) or when the checkpoint has no harm-aware
        # metadata. With explicit thresholds + W1.1 npz data, this is what
        # blocks transitions that worked positively often in absolute counts
        # but had a low success rate or negative mean net delta in context.
        ok, reason = self._delta_is_supported(prior_option, int(option), bucket)
        if not ok:
            self.decision_counts[
                f"harm_aware_{reason}_{tactical_option_name(prior_option)}_to_{tactical_option_name(int(option))}"
            ] += 1
            return heuristic_action

        if self.delta_conf_threshold > 0.0:
            if probs is None:
                probs = torch.softmax(logits, dim=-1)
            if float(probs[int(option)].item()) < self.delta_conf_threshold:
                self.decision_counts[f"low_delta_conf_{tactical_option_name(int(option))}"] += 1
                return heuristic_action

        action, _info = self.executor.act_with_info(
            option,
            observation,
            heuristic_action=heuristic_action,
            heuristic_already_run=True,
        )
        self.decision_counts[
            f"tactical_delta_{tactical_option_name(prior_option)}_to_{tactical_option_name(int(option))}"
        ] += 1
        if bucket is not None:
            self.decision_counts[
                f"tactical_delta_dist_{bucket}_{tactical_option_name(prior_option)}_to_{tactical_option_name(int(option))}"
            ] += 1
        return int(action)
