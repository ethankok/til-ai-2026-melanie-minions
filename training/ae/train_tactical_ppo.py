"""PPO for the AE planner-first tactical macro policy.

This is the "proper hybrid" trainer: the actor chooses one of the 12 tactical
macros from ``tactical_policy.py``. ``TacticalExecutor`` turns that macro into
a legal raw action through planner code, so RL optimizes *when to deviate from
the heuristic*, not pathing, turn mechanics, action masks, or bomb escape.

Each rollout game is paired with a same-suite/same-seed heuristic baseline.
The PPO reward is normal shaped environment reward plus a terminal differential
bonus for beating that baseline. The checkpoint also stores harm-aware
transition matrices consumed by ``macro_hybrid_manager.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import random
import sys
import time
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") is None:
    os.environ["PYTHONHASHSEED"] = "0"
    os.execvp(sys.executable, [sys.executable, *sys.argv])

import numpy as np
import torch
from torch import optim
from torch.distributions import Categorical

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
AE_SRC = REPO_ROOT / "ae" / "src"
TIL_AE = REPO_ROOT / "til-26-ae"
for path in (THIS_DIR, AE_SRC, TIL_AE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ae_manager import AEManager  # noqa: E402
from encoder import FrameStacker  # noqa: E402
from macro_hybrid_manager import MacroHybridAEManager, _apply_baseline_profile  # noqa: E402
from model import PolicyNetwork, num_parameters  # noqa: E402
from option_hybrid_manager import _memory_only_belief  # noqa: E402
from tactical_hybrid_manager import TacticalExecutor  # noqa: E402
from tactical_policy import (  # noqa: E402
    NUM_TACTICAL_OPTIONS,
    TACTICAL_OPTION_NAMES,
    tactical_from_manager,
    tactical_option_name,
)
from train_option_ppo import (  # noqa: E402
    FURNISHED_WEIGHTS,
    ValueNetwork,
    _device,
    _legalize_action,
    _make_env,
    _make_opponents,
    _obs_to_python,
    _safe_float,
    _shaped_reward,
    _weighted_eval,
    ppo_update,
)


DEFAULT_TRAIN_SUITES = [
    "base_rush_exploit",
    "top_seed_proxy",
    "bracket_proxy",
    "strong_realistic",
    "cloudsuite",
    "pressure2",
    "defense_trap",
    "mixed",
]

DISTANCE_BUCKET_NAMES = ("near", "mid", "far")


@dataclass
class TacticalTransition:
    agent_view: np.ndarray
    base_view: np.ndarray
    scalars: np.ndarray
    belief: np.ndarray | None
    option: int
    logprob: float
    value: float
    prior_option: int
    distance_bucket: int
    health: float | None = None
    base_health: float | None = None
    reward: float = 0.0
    done: bool = False


class HarmStats:
    def __init__(self) -> None:
        shape = (NUM_TACTICAL_OPTIONS, NUM_TACTICAL_OPTIONS)
        bucket_shape = (NUM_TACTICAL_OPTIONS, NUM_TACTICAL_OPTIONS, len(DISTANCE_BUCKET_NAMES))
        self.attempted = np.zeros(shape, dtype=np.int64)
        self.positive = np.zeros(shape, dtype=np.int64)
        self.negative = np.zeros(shape, dtype=np.int64)
        self.net_delta = np.zeros(shape, dtype=np.float64)
        self.weight = np.zeros(shape, dtype=np.float64)
        self.weighted_delta = np.zeros(shape, dtype=np.float64)
        self.bucket_attempted = np.zeros(bucket_shape, dtype=np.int64)
        self.bucket_positive = np.zeros(bucket_shape, dtype=np.int64)
        self.bucket_net_delta = np.zeros(bucket_shape, dtype=np.float64)

    @classmethod
    def from_checkpoint(cls, ckpt: dict | None) -> "HarmStats":
        harm = cls()
        if not ckpt:
            return harm

        def load_array(key: str, target: np.ndarray) -> None:
            arr = ckpt.get(key)
            if arr is None:
                return
            arr_np = np.asarray(arr, dtype=target.dtype)
            if arr_np.shape == target.shape:
                target[:] = arr_np

        load_array("attempted_transition_counts", harm.attempted)
        load_array("positive_transition_counts_full", harm.positive)
        if not harm.positive.any() and harm.attempted.any():
            load_array("positive_delta_transition_counts", harm.positive)
        load_array("negative_transition_counts", harm.negative)
        load_array("transition_net_delta_sum", harm.net_delta)
        load_array("transition_weight_sum", harm.weight)
        load_array("transition_weighted_delta_sum", harm.weighted_delta)
        load_array("bucket_attempted", harm.bucket_attempted)
        load_array("bucket_positive", harm.bucket_positive)
        load_array("bucket_net_delta_sum", harm.bucket_net_delta)
        return harm

    def add_episode(self, transitions: list[TacticalTransition], delta: float, min_positive_delta: float) -> None:
        for tr in transitions:
            prior = int(tr.prior_option)
            option = int(tr.option)
            bucket = int(tr.distance_bucket)
            self.attempted[prior, option] += 1
            self.net_delta[prior, option] += float(delta)
            self.weight[prior, option] += 1.0
            self.weighted_delta[prior, option] += float(delta)
            if 0 <= bucket < len(DISTANCE_BUCKET_NAMES):
                self.bucket_attempted[prior, option, bucket] += 1
                self.bucket_net_delta[prior, option, bucket] += float(delta)
            if delta > min_positive_delta:
                self.positive[prior, option] += 1
                if 0 <= bucket < len(DISTANCE_BUCKET_NAMES):
                    self.bucket_positive[prior, option, bucket] += 1
            else:
                self.negative[prior, option] += 1

    def merge(self, other: "HarmStats") -> None:
        self.attempted += other.attempted
        self.positive += other.positive
        self.negative += other.negative
        self.net_delta += other.net_delta
        self.weight += other.weight
        self.weighted_delta += other.weighted_delta
        self.bucket_attempted += other.bucket_attempted
        self.bucket_positive += other.bucket_positive
        self.bucket_net_delta += other.bucket_net_delta

    def as_harm_aware(self) -> dict[str, np.ndarray]:
        return {
            "attempted_transition_counts": self.attempted.astype(np.int64),
            "positive_transition_counts_full": self.positive.astype(np.int64),
            "negative_transition_counts": self.negative.astype(np.int64),
            "transition_net_delta_sum": self.net_delta.astype(np.float64),
            "transition_weight_sum": self.weight.astype(np.float64),
            "transition_weighted_delta_sum": self.weighted_delta.astype(np.float64),
            "bucket_attempted": self.bucket_attempted.astype(np.int64),
            "bucket_positive": self.bucket_positive.astype(np.int64),
            "bucket_net_delta_sum": self.bucket_net_delta.astype(np.float64),
        }


class InMemoryTacticalPolicy:
    """TacticalPolicyAEManager-compatible adapter for the current PPO actor."""

    def __init__(
        self,
        actor: PolicyNetwork,
        device: torch.device,
        n_frames: int,
        use_belief: bool,
        harm: HarmStats,
    ) -> None:
        self.model = actor
        self.device = device
        self.n_frames = n_frames
        self.use_belief = use_belief
        self.delta_transition_counts = harm.positive.astype(np.float32)
        self.harm_aware = harm.as_harm_aware()
        self.stacker = FrameStacker(self.n_frames)
        self.belief_manager = AEManager()
        self._last_step: int | None = None

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
        _stacked, agent_v, base_v, scalars, belief_t = _stacked_tensors(
            self.stacker,
            observation,
            belief,
            self.device,
        )
        belief_b = belief_t.unsqueeze(0) if belief_t is not None else None
        with torch.inference_mode():
            logits = self.model(
                agent_v.unsqueeze(0),
                base_v.unsqueeze(0),
                scalars.unsqueeze(0),
                belief_map=belief_b,
            ).squeeze(0)
        return int(logits.argmax().item()), logits.detach().clone()


def _distance_bucket(executor: TacticalExecutor, obs_py: dict) -> int:
    location = executor.heuristic._location(obs_py.get("location"))
    base = executor.heuristic.base_location or executor.heuristic._location(obs_py.get("base_location"))
    if location is None or base is None:
        return 1
    dist = executor.heuristic._manhattan(location, base)
    if dist <= 4:
        return 0
    if dist <= 8:
        return 1
    return 2


def _stacked_tensors(stacker: FrameStacker, obs_py: dict, belief: np.ndarray | None, device: torch.device):
    stacked = stacker.observe(obs_py, belief_map=belief)
    agent_v = torch.from_numpy(np.ascontiguousarray(stacked["agent_view"])).float().to(device)
    base_v = torch.from_numpy(np.ascontiguousarray(stacked["base_view"])).float().to(device)
    scalars = torch.from_numpy(np.ascontiguousarray(stacked["scalars"])).float().to(device)
    belief_t = torch.from_numpy(np.ascontiguousarray(belief)).float().to(device) if belief is not None else None
    return stacked, agent_v, base_v, scalars, belief_t


def select_tactical(
    actor: PolicyNetwork,
    critic: ValueNetwork,
    stacker: FrameStacker,
    belief_manager: AEManager,
    obs_py: dict,
    device: torch.device,
    use_belief: bool,
    greedy: bool,
    temperature: float,
    epsilon: float,
) -> tuple[int, float, float, dict, np.ndarray | None]:
    belief = _memory_only_belief(belief_manager, obs_py, use_belief=use_belief)
    stacked, agent_v, base_v, scalars, belief_t = _stacked_tensors(stacker, obs_py, belief, device)
    belief_b = belief_t.unsqueeze(0) if belief_t is not None else None
    with torch.no_grad():
        logits = actor(
            agent_v.unsqueeze(0),
            base_v.unsqueeze(0),
            scalars.unsqueeze(0),
            belief_map=belief_b,
        ).squeeze(0)
        value = critic(
            agent_v.unsqueeze(0),
            base_v.unsqueeze(0),
            scalars.unsqueeze(0),
            belief_map=belief_b,
        ).squeeze(0)
        if greedy:
            option_t = logits.argmax()
            logprob = torch.log_softmax(logits, dim=-1)[option_t]
        else:
            dist = Categorical(logits=logits / max(float(temperature), 1e-3))
            option_t = dist.sample()
            if epsilon > 0.0 and random.random() < epsilon:
                option_t = torch.tensor(random.randrange(NUM_TACTICAL_OPTIONS), device=logits.device)
            logprob = dist.log_prob(option_t)
    return int(option_t.item()), float(logprob.item()), float(value.item()), stacked, belief


def run_heuristic_baseline(suite: str, seed: int, args: argparse.Namespace) -> float:
    env = _make_env(novice=not args.non_novice)
    env.reset(seed=seed)
    our_agent = env.possible_agents[0]
    other_ids = list(env.possible_agents[1:])
    opponents = _make_opponents(suite, seed)
    for op in opponents:
        if hasattr(op, "reset_for_game"):
            op.reset_for_game()
        if hasattr(op, "_reset_memory"):
            op._reset_memory()
    planner = AEManager()
    total = 0.0
    for agent in env.agent_iter():
        obs, reward, termination, truncation, _info = env.last()
        if agent == our_agent:
            total += float(reward)
        if termination or truncation:
            env.step(None)
            continue
        obs_py = _obs_to_python(obs)
        if agent == our_agent:
            if obs_py.get("step") == 0:
                planner = AEManager()
            action = int(planner.ae(obs_py))
        else:
            slot = other_ids.index(agent)
            action = int(opponents[slot](obs_py))
            action = _legalize_action(env, agent, obs_py, action)
        env.step(action)
    env.close()
    return total / 1000.0


def collect_rollouts(
    actor: PolicyNetwork,
    critic: ValueNetwork,
    args: argparse.Namespace,
    device: torch.device,
    update: int,
) -> tuple[list[TacticalTransition], float, HarmStats, dict[str, float]]:
    env = _make_env(novice=not args.non_novice)
    our_agent = env.possible_agents[0]
    other_ids = list(env.possible_agents[1:])
    transitions: list[TacticalTransition] = []
    harm = HarmStats()
    rollout_scores: list[float] = []
    deltas: list[float] = []
    use_belief = bool(getattr(actor, "use_belief", False))

    for game in range(args.games_per_update):
        seed = args.seed + update * 100_000 + game
        suite = args.opponent_suites[(update + game - 1) % len(args.opponent_suites)]
        baseline_score = run_heuristic_baseline(suite, seed, args) if args.same_seed_baseline else 0.0

        random.seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
        torch.manual_seed(seed)
        env.reset(seed=seed)
        opponents = _make_opponents(suite, seed)
        for op in opponents:
            if hasattr(op, "reset_for_game"):
                op.reset_for_game()
            if hasattr(op, "_reset_memory"):
                op._reset_memory()

        stacker = FrameStacker(args.n_frames)
        belief_manager = AEManager()
        executor = TacticalExecutor()
        pending_idx: int | None = None
        episode_indices: list[int] = []
        total = 0.0

        for agent in env.agent_iter():
            obs, reward, termination, truncation, _info = env.last()
            obs_py_for_reward = _obs_to_python(obs) if agent == our_agent else None
            if agent == our_agent and pending_idx is not None:
                transitions[pending_idx].reward += _shaped_reward(
                    float(reward),
                    transitions[pending_idx],
                    obs_py_for_reward or {},
                    bool(termination or truncation),
                    args,
                )
                total += float(reward)
                if termination or truncation:
                    transitions[pending_idx].done = True
                    pending_idx = None

            if termination or truncation:
                env.step(None)
                continue

            obs_py = _obs_to_python(obs)
            if agent == our_agent:
                if obs_py.get("step") == 0:
                    stacker.reset()
                    belief_manager = AEManager()
                    executor = TacticalExecutor()
                heuristic_action = int(executor.heuristic.ae(obs_py))
                prior = int(tactical_from_manager(executor.heuristic, action=heuristic_action))
                option, logprob, value, stacked, belief = select_tactical(
                    actor,
                    critic,
                    stacker,
                    belief_manager,
                    obs_py,
                    device,
                    use_belief,
                    greedy=False,
                    temperature=args.option_temperature,
                    epsilon=args.option_epsilon,
                )
                action, _info = executor.act_with_info(
                    option,
                    obs_py,
                    heuristic_action=heuristic_action,
                    heuristic_already_run=True,
                )
                transitions.append(TacticalTransition(
                    agent_view=stacked["agent_view"],
                    base_view=stacked["base_view"],
                    scalars=stacked["scalars"],
                    belief=belief,
                    option=option,
                    logprob=logprob,
                    value=value,
                    prior_option=prior,
                    distance_bucket=_distance_bucket(executor, obs_py),
                    health=_safe_float(obs_py.get("health")),
                    base_health=_safe_float(obs_py.get("base_health")),
                ))
                pending_idx = len(transitions) - 1
                episode_indices.append(pending_idx)
            else:
                slot = other_ids.index(agent)
                action = int(opponents[slot](obs_py))
                action = _legalize_action(env, agent, obs_py, action)
            env.step(action)

        if pending_idx is not None:
            transitions[pending_idx].done = True

        score = total / 1000.0
        delta = score - baseline_score
        rollout_scores.append(score)
        deltas.append(delta)
        episode_transitions = [transitions[i] for i in episode_indices]
        harm.add_episode(episode_transitions, delta, args.min_positive_delta)
        if episode_indices:
            transitions[episode_indices[-1]].reward += args.delta_reward_coef * delta

    env.close()
    diagnostics = {
        "rollout_score": float(np.mean(rollout_scores)) if rollout_scores else 0.0,
        "mean_delta_vs_baseline": float(np.mean(deltas)) if deltas else 0.0,
        "positive_delta_rate": float(np.mean([d > args.min_positive_delta for d in deltas])) if deltas else 0.0,
    }
    return transitions, diagnostics["rollout_score"], harm, diagnostics


def evaluate_ungated_policy(actor: PolicyNetwork, critic: ValueNetwork, args: argparse.Namespace, device: torch.device):
    env = _make_env(novice=not args.non_novice)
    our_agent = env.possible_agents[0]
    other_ids = list(env.possible_agents[1:])
    use_belief = bool(getattr(actor, "use_belief", False))
    suite_scores: dict[str, list[float]] = {suite: [] for suite in args.eval_suites}
    suite_deltas: dict[str, list[float]] = {suite: [] for suite in args.eval_suites}
    option_counts = np.zeros(NUM_TACTICAL_OPTIONS, dtype=np.int64)

    for game in range(args.eval_games):
        seed = args.eval_seed + game
        suite = args.eval_suites[game % len(args.eval_suites)]
        baseline_score = run_heuristic_baseline(suite, seed, args) if args.same_seed_baseline else 0.0
        env.reset(seed=seed)
        opponents = _make_opponents(suite, seed)
        for op in opponents:
            if hasattr(op, "reset_for_game"):
                op.reset_for_game()
            if hasattr(op, "_reset_memory"):
                op._reset_memory()
        stacker = FrameStacker(args.n_frames)
        belief_manager = AEManager()
        executor = TacticalExecutor()
        total = 0.0
        for agent in env.agent_iter():
            obs, reward, termination, truncation, _info = env.last()
            if agent == our_agent:
                total += float(reward)
            if termination or truncation:
                env.step(None)
                continue
            obs_py = _obs_to_python(obs)
            if agent == our_agent:
                if obs_py.get("step") == 0:
                    stacker.reset()
                    belief_manager = AEManager()
                    executor = TacticalExecutor()
                heuristic_action = int(executor.heuristic.ae(obs_py))
                option, _logprob, _value, _stacked, _belief = select_tactical(
                    actor,
                    critic,
                    stacker,
                    belief_manager,
                    obs_py,
                    device,
                    use_belief,
                    greedy=True,
                    temperature=1.0,
                    epsilon=0.0,
                )
                option_counts[option] += 1
                action, _info = executor.act_with_info(
                    option,
                    obs_py,
                    heuristic_action=heuristic_action,
                    heuristic_already_run=True,
                )
            else:
                slot = other_ids.index(agent)
                action = int(opponents[slot](obs_py))
                action = _legalize_action(env, agent, obs_py, action)
            env.step(action)
        score = total / 1000.0
        suite_scores[suite].append(score)
        suite_deltas[suite].append(score - baseline_score)
    env.close()
    parts = {suite: float(np.mean(values)) for suite, values in suite_scores.items() if values}
    deltas = {suite: float(np.mean(values)) for suite, values in suite_deltas.items() if values}
    score = _weighted_eval(parts, args)
    diagnostics = {
        "weighted_delta_vs_baseline": _weighted_eval(deltas, args),
        "worst_suite_score": min(parts.values()) if parts else 0.0,
    }
    for i, name in enumerate(TACTICAL_OPTION_NAMES):
        diagnostics[f"tactical_{name}"] = float(option_counts[i])
    return score, parts, deltas, diagnostics


def evaluate_macro_hybrid(
    actor: PolicyNetwork,
    args: argparse.Namespace,
    device: torch.device,
    harm: HarmStats,
) -> tuple[float, dict[str, float], dict[str, float], dict[str, float]]:
    """Evaluate the same gated wrapper used by AE_MODE=macro_hybrid."""

    env = _make_env(novice=not args.non_novice)
    our_agent = env.possible_agents[0]
    other_ids = list(env.possible_agents[1:])
    use_belief = bool(getattr(actor, "use_belief", False))
    suite_scores: dict[str, list[float]] = {suite: [] for suite in args.eval_suites}
    suite_deltas: dict[str, list[float]] = {suite: [] for suite in args.eval_suites}
    decision_counts: dict[str, float] = {}
    accept_total = 0.0
    fallback_total = 0.0

    for game in range(args.eval_games):
        seed = args.eval_seed + game
        suite = args.eval_suites[game % len(args.eval_suites)]
        baseline_score = run_heuristic_baseline(suite, seed, args) if args.same_seed_baseline else 0.0
        env.reset(seed=seed)
        opponents = _make_opponents(suite, seed)
        for op in opponents:
            if hasattr(op, "reset_for_game"):
                op.reset_for_game()
            if hasattr(op, "_reset_memory"):
                op._reset_memory()

        policy = InMemoryTacticalPolicy(actor, device, args.n_frames, use_belief, harm)
        manager = MacroHybridAEManager(policy=policy)
        total = 0.0
        for agent in env.agent_iter():
            obs, reward, termination, truncation, _info = env.last()
            if agent == our_agent:
                total += float(reward)
            if termination or truncation:
                env.step(None)
                continue
            obs_py = _obs_to_python(obs)
            if agent == our_agent:
                action = int(manager.ae(obs_py))
            else:
                slot = other_ids.index(agent)
                action = int(opponents[slot](obs_py))
                action = _legalize_action(env, agent, obs_py, action)
            env.step(action)

        for key, count in manager.decision_counts.items():
            decision_counts[key] = decision_counts.get(key, 0.0) + float(count)
            if key.startswith("macro_accept_") and not key.startswith("macro_accept_dist_"):
                accept_total += float(count)
            elif (
                key.startswith("macro_exact_heuristic_")
                or key.startswith("macro_no_accepted_delta_")
                or "fallback" in key
                or key.startswith("macro_reject_")
            ):
                fallback_total += float(count)
        score = total / 1000.0
        suite_scores[suite].append(score)
        suite_deltas[suite].append(score - baseline_score)

    env.close()
    parts = {suite: float(np.mean(values)) for suite, values in suite_scores.items() if values}
    deltas = {suite: float(np.mean(values)) for suite, values in suite_deltas.items() if values}
    score = _weighted_eval(parts, args)
    total_decisions = max(accept_total + fallback_total, 1.0)
    diagnostics = {
        "weighted_delta_vs_baseline": _weighted_eval(deltas, args),
        "worst_suite_score": min(parts.values()) if parts else 0.0,
        "macro_accept_total": accept_total,
        "macro_fallback_or_reject_total": fallback_total,
        "macro_accept_rate": accept_total / total_decisions,
        "harm_attempted_total": float(harm.attempted.sum()),
        "harm_positive_total": float(harm.positive.sum()),
    }
    for key, value in sorted(decision_counts.items(), key=lambda item: -item[1])[:20]:
        diagnostics[f"decision_{key}"] = value
    return score, parts, deltas, diagnostics


def load_actor(args: argparse.Namespace, device: torch.device) -> tuple[PolicyNetwork, bool, dict | None]:
    ckpt_path = Path(args.bc_checkpoint)
    ckpt = None
    n_frames = args.n_frames
    use_belief = not args.no_belief
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        n_frames = int(ckpt.get("n_frames", n_frames))
        use_belief = bool(ckpt.get("use_belief", use_belief)) and not args.no_belief
    actor = PolicyNetwork(n_frames=n_frames, action_dim=NUM_TACTICAL_OPTIONS, use_belief=use_belief).to(device)
    actor.use_belief = use_belief
    args.n_frames = n_frames
    if ckpt is not None:
        action_dim = int(ckpt.get("action_dim", NUM_TACTICAL_OPTIONS))
        if action_dim != NUM_TACTICAL_OPTIONS:
            raise ValueError(f"checkpoint action_dim={action_dim}, expected {NUM_TACTICAL_OPTIONS}")
        actor.load_state_dict(ckpt["model_state_dict"])
        print(f"warm-started tactical actor from {ckpt_path} (epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss')})")
    else:
        print(f"no tactical checkpoint at {ckpt_path}; training from scratch")
    return actor, use_belief, ckpt


def save_checkpoint(
    actor: PolicyNetwork,
    critic: ValueNetwork,
    args: argparse.Namespace,
    path: Path,
    update: int,
    eval_score: float,
    eval_parts: dict[str, float],
    eval_deltas: dict[str, float],
    eval_diagnostics: dict[str, float],
    harm: HarmStats,
    ungated_eval: dict[str, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "update": update,
        "ppo_eval_score": eval_score,
        "ppo_eval_parts": eval_parts,
        "ppo_eval_deltas_vs_baseline": eval_deltas,
        "ppo_eval_diagnostics": eval_diagnostics,
        "wrapper_eval_score": eval_score,
        "wrapper_eval_parts": eval_parts,
        "wrapper_eval_deltas_vs_baseline": eval_deltas,
        "wrapper_eval_diagnostics": eval_diagnostics,
        "ungated_policy_eval": ungated_eval or {},
        "n_frames": args.n_frames,
        "use_belief": bool(getattr(actor, "use_belief", False)),
        "action_dim": NUM_TACTICAL_OPTIONS,
        "option_names": TACTICAL_OPTION_NAMES,
        "model_type": "ae_tactical_policy",
        "training_mode": "planner_first_macro_ppo",
        "eval_mode": "macro_hybrid_wrapper",
        "harm_stats_source": "cumulative_training_rollouts",
        "macro_gate_settings": _macro_gate_settings(),
        "baseline_profile": args.baseline_profile,
        "opponent_suites": args.opponent_suites,
        "eval_suites": args.eval_suites,
        "positive_delta_transition_counts": harm.positive.astype(np.int64),
        "attempted_transition_counts": harm.attempted.astype(np.int64),
        "positive_transition_counts_full": harm.positive.astype(np.int64),
        "negative_transition_counts": harm.negative.astype(np.int64),
        "transition_net_delta_sum": harm.net_delta.astype(np.float64),
        "transition_weight_sum": harm.weight.astype(np.float64),
        "transition_weighted_delta_sum": harm.weighted_delta.astype(np.float64),
        "bucket_attempted": harm.bucket_attempted.astype(np.int64),
        "bucket_positive": harm.bucket_positive.astype(np.int64),
        "bucket_net_delta_sum": harm.bucket_net_delta.astype(np.float64),
    }, path)


def _required_suites_ok(parts: dict[str, float], args: argparse.Namespace) -> bool:
    if args.min_required_suite_score <= 0:
        return True
    return all(parts.get(suite, 0.0) >= args.min_required_suite_score for suite in args.required_suites)


def _macro_gate_settings() -> dict[str, str]:
    keys = [
        "AE_MACRO_BASELINE_PROFILE",
        "AE_MACRO_TOP_K",
        "AE_MACRO_DELTA_MARGIN",
        "AE_TACTICAL_REQUIRE_DELTA_SUPPORT",
        "AE_TACTICAL_MIN_DELTA_SUPPORT",
        "AE_TACTICAL_MIN_ATTEMPTED",
        "AE_TACTICAL_MIN_POSITIVE_RATE",
        "AE_TACTICAL_MIN_NET_DELTA",
        "AE_TACTICAL_ALLOWED_DELTA_OPTIONS",
    ]
    return {key: os.environ.get(key, "") for key in keys}


def train(args: argparse.Namespace) -> None:
    os.environ["AE_MACRO_BASELINE_PROFILE"] = args.baseline_profile
    _apply_baseline_profile()
    device = _device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed % (2 ** 32 - 1))
    torch.manual_seed(args.seed)
    actor, use_belief, actor_ckpt = load_actor(args, device)
    critic = ValueNetwork(n_frames=args.n_frames, use_belief=use_belief).to(device)
    if actor_ckpt is not None and actor_ckpt.get("critic_state_dict") is not None:
        try:
            critic.load_state_dict(actor_ckpt["critic_state_dict"])
            print("warm-started critic from tactical checkpoint")
        except Exception as exc:
            print(f"WARN: critic warm-start skipped ({exc})")
    optimizer = optim.AdamW(
        list(actor.parameters()) + list(critic.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    print(
        f"device={device}; actor_params={num_parameters(actor):,}; "
        f"critic_params={num_parameters(critic):,}; n_frames={args.n_frames}; "
        f"use_belief={use_belief}; baseline_profile={args.baseline_profile}"
    )
    out_path = Path(args.out)
    latest_path = Path(args.latest_out) if args.latest_out else None
    best_eval = -float("inf")
    cumulative_harm = HarmStats.from_checkpoint(actor_ckpt)
    if cumulative_harm.attempted.sum() > 0:
        print(
            "loaded cumulative harm/support stats from warm-start checkpoint "
            f"(attempted={int(cumulative_harm.attempted.sum())}, "
            f"positive={int(cumulative_harm.positive.sum())})"
        )
    t0 = time.time()
    for update in range(1, args.updates + 1):
        actor.train()
        critic.train()
        transitions, rollout_score, harm, rollout_diag = collect_rollouts(actor, critic, args, device, update)
        cumulative_harm.merge(harm)
        if not transitions:
            raise SystemExit("No transitions collected.")
        progress = (update - 1) / max(args.updates - 1, 1)
        entropy_coef = args.entropy_coef + (
            (args.entropy_final_coef - args.entropy_coef) * min(1.0, max(0.0, progress))
        )
        stats = ppo_update(actor, critic, transitions, optimizer, args, device, entropy_coef)
        eval_score = float("nan")
        eval_parts: dict[str, float] = {}
        eval_deltas: dict[str, float] = {}
        eval_diag: dict[str, float] = {}
        ungated_eval: dict[str, float] = {}
        should_eval = update % args.eval_every == 0 or (args.eval_first and update == 1)
        if should_eval:
            actor.eval()
            critic.eval()
            eval_score, eval_parts, eval_deltas, eval_diag = evaluate_macro_hybrid(
                actor,
                args,
                device,
                cumulative_harm,
            )
            if args.eval_ungated_policy:
                ungated_score, ungated_parts, _ungated_deltas, ungated_diag = evaluate_ungated_policy(
                    actor,
                    critic,
                    args,
                    device,
                )
                ungated_eval = {
                    "score": ungated_score,
                    "weighted_delta_vs_baseline": ungated_diag.get("weighted_delta_vs_baseline", 0.0),
                    "worst_suite_score": ungated_diag.get("worst_suite_score", 0.0),
                    **{f"part_{key}": value for key, value in ungated_parts.items()},
                }
            if latest_path is not None:
                save_checkpoint(
                    actor,
                    critic,
                    args,
                    latest_path,
                    update,
                    eval_score,
                    eval_parts,
                    eval_deltas,
                    eval_diag,
                    cumulative_harm,
                    ungated_eval,
                )
            required_ok = _required_suites_ok(eval_parts, args)
            delta_ok = eval_diag.get("weighted_delta_vs_baseline", 0.0) >= args.min_eval_delta
            if eval_score >= args.save_floor and required_ok and delta_ok and eval_score > best_eval:
                best_eval = eval_score
                save_checkpoint(
                    actor,
                    critic,
                    args,
                    out_path,
                    update,
                    eval_score,
                    eval_parts,
                    eval_deltas,
                    eval_diag,
                    cumulative_harm,
                    ungated_eval,
                )
                print(f"  best macro-hybrid wrapper eval {best_eval:.4f} -> saved {out_path}")
            else:
                print(
                    f"  gate not met: wrapper_eval={eval_score:.4f} "
                    f"wrapper_delta={eval_diag.get('weighted_delta_vs_baseline', 0.0):+.4f} "
                    f"save_floor={args.save_floor:.4f} min_delta={args.min_eval_delta:+.4f} "
                    f"required_ok={required_ok} best={best_eval:.4f}"
                )
        part_tag = " ".join(f"{k}={v:.4f}" for k, v in eval_parts.items())
        decision_tags = []
        if eval_diag:
            decision_tags = sorted(
                (
                    (key.removeprefix("decision_"), value)
                    for key, value in eval_diag.items()
                    if key.startswith("decision_")
                ),
                key=lambda item: -item[1],
            )[:4]
        ungated_tag = ""
        if ungated_eval:
            ungated_tag = (
                f" ungated={ungated_eval.get('score', 0.0):.4f}"
                f" ungated_delta={ungated_eval.get('weighted_delta_vs_baseline', 0.0):+.4f}"
            )
        print(
            f"update {update:>4}/{args.updates} samples={len(transitions):>5} "
            f"rollout={rollout_score:.4f} rollout_delta={rollout_diag['mean_delta_vs_baseline']:+.4f} "
            f"pos_rate={rollout_diag['positive_delta_rate']:.2f} wrapper_eval={eval_score:.4f} "
            f"wrapper_delta={eval_diag.get('weighted_delta_vs_baseline', 0.0):+.4f} "
            f"accept_rate={eval_diag.get('macro_accept_rate', 0.0):.3f} "
            f"harm_n={eval_diag.get('harm_attempted_total', float(cumulative_harm.attempted.sum())):.0f} "
            f"{part_tag}{ungated_tag} "
            + " ".join(f"dec_{name}={count:.0f}" for name, count in decision_tags)
            + f" pi_loss={stats['policy_loss']:.4f} v_loss={stats['value_loss']:.4f} "
              f"entropy={stats['entropy']:.3f} kl={stats['approx_kl']:.4f} "
              f"elapsed={(time.time() - t0) / 60:.1f}m"
        )
    print(f"\nBest saved eval score: {best_eval:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-checkpoint", default="training/ae/checkpoints/tactical_policy.pt")
    parser.add_argument("--out", default="training/ae/checkpoints/tactical_policy_ppo.pt")
    parser.add_argument("--latest-out", default="training/ae/checkpoints/tactical_policy_ppo_latest.pt")
    parser.add_argument("--baseline-profile", default="combo_c_bomb7",
                        choices=["combo_c_bomb7", "heuristic", "none"],
                        help="heuristic profile used by the planner fallback and same-seed baseline")
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--games-per-update", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.20)
    parser.add_argument("--entropy-coef", type=float, default=0.012)
    parser.add_argument("--entropy-final-coef", type=float, default=0.003)
    parser.add_argument("--value-coef", type=float, default=0.50)
    parser.add_argument("--max-grad-norm", type=float, default=0.50)
    parser.add_argument("--reward-scale", type=float, default=50.0)
    parser.add_argument("--return-clip", type=float, default=10.0)
    parser.add_argument("--health-delta-coef", type=float, default=0.15)
    parser.add_argument("--base-health-delta-coef", type=float, default=1.25)
    parser.add_argument("--base-survival-bonus", type=float, default=0.35)
    parser.add_argument("--base-destroyed-penalty", type=float, default=0.75)
    parser.add_argument("--delta-reward-coef", type=float, default=8.0)
    parser.add_argument("--min-positive-delta", type=float, default=0.0)
    parser.add_argument("--option-temperature", type=float, default=1.25)
    parser.add_argument("--option-epsilon", type=float, default=0.05)
    parser.add_argument("--n-frames", type=int, default=4)
    parser.add_argument("--no-belief", action="store_true")
    parser.add_argument("--non-novice", action="store_true")
    parser.add_argument("--same-seed-baseline", action="store_true", default=True)
    parser.add_argument("--no-same-seed-baseline", dest="same_seed_baseline", action="store_false")
    parser.add_argument("--opponent-suites", nargs="+", default=DEFAULT_TRAIN_SUITES)
    parser.add_argument("--eval-suites", nargs="+", default=[
        "cloudsuite", "pressure2", "strong_realistic", "base_rush_exploit",
        "bracket_proxy", "top_seed_proxy", "defense_trap", "mixed",
    ])
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=16)
    parser.add_argument("--eval-seed", type=int, default=137)
    parser.add_argument("--eval-first", action="store_true",
                        help="run wrapper eval after update 1; useful for smoke tests, slow for real PPO")
    parser.add_argument("--eval-ungated-policy", action="store_true",
                        help="also print the old standalone policy/executor eval as a diagnostic")
    parser.add_argument("--save-floor", type=float, default=0.285)
    parser.add_argument("--min-eval-delta", type=float, default=0.005)
    parser.add_argument("--use-furnished-weights", dest="use_furnished_weights", action="store_true")
    parser.add_argument("--uniform-eval-weights", dest="use_furnished_weights", action="store_false")
    parser.set_defaults(use_furnished_weights=True)
    parser.add_argument("--required-suites", nargs="+", default=["top_seed_proxy", "base_rush_exploit", "bracket_proxy"])
    parser.add_argument("--min-required-suite-score", type=float, default=0.16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    train(parser.parse_args())


if __name__ == "__main__":
    main()
