"""PPO fine-tuning for the AE option selector.

The actor samples one of the 8 strategic options. ``OptionExecutor`` converts
that option to a legal raw action through planner variants, so PPO optimizes
strategy selection rather than relearning pathing, turns, and bomb safety from
scratch.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
AE_SRC = REPO_ROOT / "ae" / "src"
for path in (THIS_DIR, AE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ae_manager import AEManager  # noqa: E402
from encoder import BELIEF_CHANNELS, FrameStacker, SCALAR_DIM  # noqa: E402
from model import AGENT_VIEW_HW, BASE_VIEW_HW, BELIEF_HW, PolicyNetwork, VIEW_CHANNELS, num_parameters  # noqa: E402
from opponents import make_opponent, resolve_opponent_spec  # noqa: E402
from option_hybrid_manager import OptionExecutor, _memory_only_belief  # noqa: E402
from option_policy import NUM_OPTIONS, OPTION_NAMES  # noqa: E402
from til_environment import bomberman_env  # noqa: E402
from til_environment.config import default_config  # noqa: E402


DEFAULT_TRAIN_SUITES = [
    "strong_realistic",
    "base_rush_exploit",
    "bracket_proxy",
    "top_seed_proxy",
    "cloudsuite",
    "pressure2",
]

FURNISHED_WEIGHTS = {
    "cloudsuite": 0.18,
    "pressure2": 0.18,
    "strong_realistic": 0.20,
    "base_rush_exploit": 0.18,
    "bracket_proxy": 0.14,
    "top_seed_proxy": 0.08,
    "defense_trap": 0.04,
    "mixed": 0.04,
}


@dataclass
class Transition:
    agent_view: np.ndarray
    base_view: np.ndarray
    scalars: np.ndarray
    belief: np.ndarray | None
    option: int
    logprob: float
    value: float
    health: float | None = None
    base_health: float | None = None
    reward: float = 0.0
    done: bool = False


class ValueNetwork(nn.Module):
    def __init__(self, n_frames: int = 4, use_belief: bool = False):
        super().__init__()
        in_ch = VIEW_CHANNELS * n_frames
        scalar_dim = SCALAR_DIM * n_frames
        self.use_belief = bool(use_belief)
        self.agent_conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.base_conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        head_in = 16 * AGENT_VIEW_HW[0] * AGENT_VIEW_HW[1] + 8 * BASE_VIEW_HW[0] * BASE_VIEW_HW[1] + scalar_dim
        if self.use_belief:
            self.belief_conv = nn.Sequential(
                nn.Conv2d(BELIEF_CHANNELS, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AvgPool2d(2),
                nn.Conv2d(16, 8, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            head_in += 8 * (BELIEF_HW[0] // 2) * (BELIEF_HW[1] // 2)
        self.head = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        agent_view: torch.Tensor,
        base_view: torch.Tensor,
        scalars: torch.Tensor,
        belief_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        a = self.agent_conv(agent_view).flatten(start_dim=1)
        b = self.base_conv(base_view).flatten(start_dim=1)
        parts = [a, b, scalars]
        if self.use_belief:
            if belief_map is None:
                raise ValueError("ValueNetwork(use_belief=True) requires belief_map")
            parts.append(self.belief_conv(belief_map).flatten(start_dim=1))
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)


def _obs_to_python(obs) -> dict:
    out = {}
    for key, value in obs.items():
        if hasattr(value, "tolist"):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        if hasattr(value, "item"):
            return float(value.item())
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                return None
            return float(value[0])
        return float(value)
    except Exception:
        return None


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    )


def _make_env(novice: bool):
    cfg = default_config()
    cfg.env.novice = novice
    return bomberman_env.basic_env(env_wrappers=[], cfg=cfg)


def _make_opponents(spec: str, seed: int) -> list:
    names = resolve_opponent_spec(spec)
    return [make_opponent(name, seed=seed + 1000 + i) for i, name in enumerate(names)]


def _legalize_action(env, agent: str, obs_py: dict, action: int) -> int:
    mask = obs_py.get("action_mask")
    if mask is None:
        return int(action)
    try:
        if int(mask[action]):
            return int(action)
    except Exception:
        pass
    for i, allowed in enumerate(mask):
        try:
            if int(allowed):
                return int(i)
        except Exception:
            pass
    return int(env.action_space(agent).sample())


def _stacked_tensors(
    stacker: FrameStacker,
    obs_py: dict,
    belief: np.ndarray | None,
    device: torch.device,
):
    stacked = stacker.observe(obs_py, belief_map=belief)
    agent_v = torch.from_numpy(stacked["agent_view"]).float().to(device)
    base_v = torch.from_numpy(stacked["base_view"]).float().to(device)
    scalars = torch.from_numpy(stacked["scalars"]).float().to(device)
    belief_t = torch.from_numpy(belief).float().to(device) if belief is not None else None
    return stacked, agent_v, base_v, scalars, belief_t


def select_option(
    actor: PolicyNetwork,
    critic: ValueNetwork,
    stacker: FrameStacker,
    belief_manager: AEManager,
    obs_py: dict,
    device: torch.device,
    use_belief: bool,
    greedy: bool = False,
    temperature: float = 1.0,
    epsilon: float = 0.0,
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
            option = int(logits.argmax().item())
            logprob = torch.log_softmax(logits, dim=-1)[option]
        else:
            temp = max(float(temperature), 1e-3)
            dist = Categorical(logits=logits / temp)
            sampled = dist.sample()
            if epsilon > 0.0 and random.random() < epsilon:
                sampled = torch.tensor(random.randrange(NUM_OPTIONS), device=logits.device)
            option = int(sampled.item())
            logprob = dist.log_prob(sampled)
    return option, float(logprob.item()), float(value.item()), stacked, belief


def _shaped_reward(
    raw_reward: float,
    previous: Transition,
    obs_py: dict,
    done: bool,
    args: argparse.Namespace,
) -> float:
    shaped = float(raw_reward) / args.reward_scale
    health = _safe_float(obs_py.get("health"))
    if health is not None and previous.health is not None:
        shaped += args.health_delta_coef * (health - previous.health) / 60.0
    base_health = _safe_float(obs_py.get("base_health"))
    if base_health is not None and previous.base_health is not None:
        shaped += args.base_health_delta_coef * (base_health - previous.base_health) / 100.0
    if done and base_health is not None:
        if base_health > 0.0:
            shaped += args.base_survival_bonus
        else:
            shaped -= args.base_destroyed_penalty
    return shaped


def collect_rollouts(
    actor: PolicyNetwork,
    critic: ValueNetwork,
    args: argparse.Namespace,
    device: torch.device,
    update: int,
) -> tuple[list[Transition], float]:
    env = _make_env(novice=not args.non_novice)
    our_agent = env.possible_agents[0]
    other_ids = list(env.possible_agents[1:])
    transitions: list[Transition] = []
    total_reward = 0.0
    use_belief = bool(getattr(actor, "use_belief", False))

    for game in range(args.games_per_update):
        seed = args.seed + update * 100_000 + game
        random.seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
        torch.manual_seed(seed)
        env.reset(seed=seed)
        suite = args.opponent_suites[(update + game - 1) % len(args.opponent_suites)]
        opponents = _make_opponents(suite, seed)
        for op in opponents:
            if hasattr(op, "reset_for_game"):
                op.reset_for_game()
            if hasattr(op, "_reset_memory"):
                op._reset_memory()

        stacker = FrameStacker(args.n_frames)
        belief_manager = AEManager()
        executor = OptionExecutor()
        pending_idx: int | None = None

        for agent in env.agent_iter():
            obs, reward, termination, truncation, _info = env.last()
            obs_py_for_reward = _obs_to_python(obs) if agent == our_agent else None
            if agent == our_agent and pending_idx is not None:
                shaped = _shaped_reward(
                    float(reward),
                    transitions[pending_idx],
                    obs_py_for_reward or {},
                    bool(termination or truncation),
                    args,
                )
                transitions[pending_idx].reward += shaped
                total_reward += float(reward)
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
                    executor = OptionExecutor()
                option, logprob, value, stacked, belief = select_option(
                    actor, critic, stacker, belief_manager, obs_py, device, use_belief,
                    greedy=False, temperature=args.option_temperature, epsilon=args.option_epsilon,
                )
                action = executor.act(option, obs_py)
                transitions.append(Transition(
                    agent_view=stacked["agent_view"],
                    base_view=stacked["base_view"],
                    scalars=stacked["scalars"],
                    belief=belief,
                    option=option,
                    logprob=logprob,
                    value=value,
                    health=_safe_float(obs_py.get("health")),
                    base_health=_safe_float(obs_py.get("base_health")),
                ))
                pending_idx = len(transitions) - 1
            else:
                slot = other_ids.index(agent)
                action = int(opponents[slot](obs_py))
                action = _legalize_action(env, agent, obs_py, action)
            env.step(action)

        if pending_idx is not None:
            transitions[pending_idx].done = True

    env.close()
    return transitions, total_reward / max(args.games_per_update, 1) / 1000.0


def _encode_transitions(transitions: list[Transition], device: torch.device, use_belief: bool):
    agent_views = torch.from_numpy(np.stack([t.agent_view for t in transitions])).float().to(device)
    base_views = torch.from_numpy(np.stack([t.base_view for t in transitions])).float().to(device)
    scalars = torch.from_numpy(np.stack([t.scalars for t in transitions])).float().to(device)
    options = torch.tensor([t.option for t in transitions], dtype=torch.long, device=device)
    old_logprobs = torch.tensor([t.logprob for t in transitions], dtype=torch.float32, device=device)
    values = torch.tensor([t.value for t in transitions], dtype=torch.float32, device=device)
    rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=device)
    dones = torch.tensor([float(t.done) for t in transitions], dtype=torch.float32, device=device)
    if use_belief:
        beliefs = torch.from_numpy(np.stack([t.belief for t in transitions])).float().to(device)
    else:
        beliefs = None
    return agent_views, base_views, scalars, beliefs, options, old_logprobs, values, rewards, dones


def _advantages(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, args: argparse.Namespace):
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            next_nonterminal = 1.0 - dones[t]
            next_value = 0.0
        else:
            next_nonterminal = 1.0 - dones[t]
            next_value = values[t + 1]
        delta = rewards[t] + args.gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    if args.return_clip > 0:
        returns = returns.clamp(-args.return_clip, args.return_clip)
    return advantages, returns


def ppo_update(
    actor: PolicyNetwork,
    critic: ValueNetwork,
    transitions: list[Transition],
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
    entropy_coef: float,
) -> dict[str, float]:
    use_belief = bool(getattr(actor, "use_belief", False))
    agent_v, base_v, scalars, beliefs, options, old_logprobs, old_values, rewards, dones = _encode_transitions(
        transitions, device, use_belief,
    )
    advantages, returns = _advantages(rewards, old_values, dones, args)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    n = options.shape[0]
    idx = torch.arange(n, device=device)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
    batches = 0
    for _epoch in range(args.ppo_epochs):
        perm = idx[torch.randperm(n, device=device)]
        for start in range(0, n, args.batch_size):
            mb = perm[start:start + args.batch_size]
            logits = actor(agent_v[mb], base_v[mb], scalars[mb], belief_map=beliefs[mb] if beliefs is not None else None)
            dist = Categorical(logits=logits)
            new_logprob = dist.log_prob(options[mb])
            entropy = dist.entropy().mean()
            new_value = critic(agent_v[mb], base_v[mb], scalars[mb], belief_map=beliefs[mb] if beliefs is not None else None)

            logratio = new_logprob - old_logprobs[mb]
            ratio = logratio.exp()
            pg_loss1 = -advantages[mb] * ratio
            pg_loss2 = -advantages[mb] * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
            policy_loss = torch.max(pg_loss1, pg_loss2).mean()
            value_loss = 0.5 * (returns[mb] - new_value).pow(2).mean()
            loss = policy_loss + args.value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - logratio).mean()
            stats["policy_loss"] += float(policy_loss.item())
            stats["value_loss"] += float(value_loss.item())
            stats["entropy"] += float(entropy.item())
            stats["approx_kl"] += float(approx_kl.item())
            batches += 1
    for key in stats:
        stats[key] /= max(batches, 1)
    return stats


def evaluate(
    actor: PolicyNetwork,
    critic: ValueNetwork,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, dict[str, float], dict[str, float]]:
    env = _make_env(novice=not args.non_novice)
    our_agent = env.possible_agents[0]
    other_ids = list(env.possible_agents[1:])
    use_belief = bool(getattr(actor, "use_belief", False))
    suite_scores: dict[str, list[float]] = {suite: [] for suite in args.eval_suites}
    suite_base: dict[str, list[float]] = {suite: [] for suite in args.eval_suites}
    option_counts = np.zeros(NUM_OPTIONS, dtype=np.int64)

    for game in range(args.eval_games):
        seed = args.eval_seed + game
        env.reset(seed=seed)
        suite = args.eval_suites[game % len(args.eval_suites)]
        opponents = _make_opponents(suite, seed + 50_000)
        for op in opponents:
            if hasattr(op, "reset_for_game"):
                op.reset_for_game()
            if hasattr(op, "_reset_memory"):
                op._reset_memory()
        stacker = FrameStacker(args.n_frames)
        belief_manager = AEManager()
        executor = OptionExecutor()
        total = 0.0
        final_base = 0.0
        for agent in env.agent_iter():
            obs, reward, termination, truncation, _info = env.last()
            if agent == our_agent:
                total += float(reward)
                obs_tmp = _obs_to_python(obs)
                base_health = _safe_float(obs_tmp.get("base_health"))
                if base_health is not None:
                    final_base = base_health
            if termination or truncation:
                env.step(None)
                continue
            obs_py = _obs_to_python(obs)
            if agent == our_agent:
                if obs_py.get("step") == 0:
                    stacker.reset()
                    belief_manager = AEManager()
                    executor = OptionExecutor()
                option, _logprob, _value, _stacked, _belief = select_option(
                    actor, critic, stacker, belief_manager, obs_py, device, use_belief, greedy=True,
                )
                if 0 <= option < NUM_OPTIONS:
                    option_counts[option] += 1
                action = executor.act(option, obs_py)
            else:
                slot = other_ids.index(agent)
                action = int(opponents[slot](obs_py))
                action = _legalize_action(env, agent, obs_py, action)
            env.step(action)
        suite_scores[suite].append(total / 1000.0)
        suite_base[suite].append(final_base)
    env.close()
    parts = {suite: float(np.mean(values)) for suite, values in suite_scores.items() if values}
    score = _weighted_eval(parts, args)
    diagnostics = {
        "mean_final_base_health": float(np.mean([v for values in suite_base.values() for v in values])) if suite_base else 0.0,
        "worst_suite_score": min(parts.values()) if parts else 0.0,
    }
    for i, name in enumerate(OPTION_NAMES):
        diagnostics[f"option_{name}"] = float(option_counts[i])
    return score, parts, diagnostics


def _weighted_eval(parts: dict[str, float], args: argparse.Namespace) -> float:
    if not parts:
        return 0.0
    if not args.use_furnished_weights:
        return float(np.mean(list(parts.values())))
    weighted = 0.0
    total_weight = 0.0
    for suite, score in parts.items():
        weight = FURNISHED_WEIGHTS.get(suite, 1.0)
        weighted += weight * score
        total_weight += weight
    return weighted / total_weight if total_weight else 0.0


def _required_suites_ok(parts: dict[str, float], args: argparse.Namespace) -> bool:
    if args.min_required_suite_score <= 0:
        return True
    for suite in args.required_suites:
        if suite in parts and parts[suite] < args.min_required_suite_score:
            return False
    return True


def load_actor(args: argparse.Namespace, device: torch.device) -> tuple[PolicyNetwork, bool, dict | None]:
    ckpt_path = Path(args.bc_checkpoint)
    ckpt = None
    n_frames = args.n_frames
    use_belief = not args.no_belief
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        n_frames = int(ckpt.get("n_frames", n_frames))
        use_belief = bool(ckpt.get("use_belief", use_belief)) and not args.no_belief
    actor = PolicyNetwork(n_frames=n_frames, action_dim=NUM_OPTIONS, use_belief=use_belief).to(device)
    actor.use_belief = use_belief
    args.n_frames = n_frames
    if ckpt is not None:
        actor.load_state_dict(ckpt["model_state_dict"])
        print(f"warm-started actor from {ckpt_path} (epoch={ckpt.get('epoch')}, val_acc={ckpt.get('val_acc')})")
    else:
        print(f"no BC checkpoint at {ckpt_path}; training option actor from scratch")
    return actor, use_belief, ckpt


def save_checkpoint(
    actor: PolicyNetwork,
    critic: ValueNetwork,
    args: argparse.Namespace,
    path: Path,
    update: int,
    eval_score: float,
    eval_parts: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "update": update,
        "ppo_eval_score": eval_score,
        "ppo_eval_parts": eval_parts,
        "n_frames": args.n_frames,
        "use_belief": bool(getattr(actor, "use_belief", False)),
        "action_dim": NUM_OPTIONS,
        "option_names": OPTION_NAMES,
        "model_type": "ae_option_policy",
        "opponent_suites": args.opponent_suites,
        "eval_suites": args.eval_suites,
    }, path)


def train(args: argparse.Namespace) -> None:
    device = _device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed % (2 ** 32 - 1))
    torch.manual_seed(args.seed)
    actor, use_belief, actor_ckpt = load_actor(args, device)
    critic = ValueNetwork(n_frames=args.n_frames, use_belief=use_belief).to(device)
    if actor_ckpt is not None and actor_ckpt.get("critic_state_dict") is not None:
        try:
            critic.load_state_dict(actor_ckpt["critic_state_dict"])
            print("warm-started critic from checkpoint")
        except Exception as exc:
            print(f"WARN: critic warm-start skipped ({exc})")
    optimizer = optim.AdamW(
        list(actor.parameters()) + list(critic.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    print(
        f"device={device}; actor_params={num_parameters(actor):,}; critic_params={num_parameters(critic):,}; "
        f"n_frames={args.n_frames}; use_belief={use_belief}; "
        f"train_suites={args.opponent_suites}; eval_suites={args.eval_suites}"
    )
    out_path = Path(args.out)
    latest_path = Path(args.latest_out) if args.latest_out else None
    best_eval = -float("inf")
    t0 = time.time()
    for update in range(1, args.updates + 1):
        actor.train()
        critic.train()
        transitions, rollout_score = collect_rollouts(actor, critic, args, device, update)
        if not transitions:
            raise SystemExit("No transitions collected.")
        progress = (update - 1) / max(args.updates - 1, 1)
        entropy_coef = args.entropy_coef + (
            (args.entropy_final_coef - args.entropy_coef) * min(1.0, max(0.0, progress))
        )
        stats = ppo_update(actor, critic, transitions, optimizer, args, device, entropy_coef)
        eval_score = float("nan")
        eval_parts: dict[str, float] = {}
        eval_diag: dict[str, float] = {}
        if update == 1 or update % args.eval_every == 0:
            actor.eval()
            critic.eval()
            eval_score, eval_parts, eval_diag = evaluate(actor, critic, args, device)
            if latest_path is not None:
                save_checkpoint(actor, critic, args, latest_path, update, eval_score, eval_parts)
            required_ok = _required_suites_ok(eval_parts, args)
            if eval_score >= args.save_floor and required_ok and eval_score > best_eval:
                best_eval = eval_score
                save_checkpoint(actor, critic, args, out_path, update, eval_score, eval_parts)
                print(f"  best option PPO eval {best_eval:.4f} -> saved {out_path}")
            else:
                print(
                    f"  gate not met: eval={eval_score:.4f} save_floor={args.save_floor:.4f} "
                    f"required_ok={required_ok} best={best_eval:.4f}"
                )
        part_tag = " ".join(f"{k}={v:.4f}" for k, v in eval_parts.items())
        diag_tag = ""
        if eval_diag:
            top_options = sorted(
                ((name, eval_diag.get(f"option_{name}", 0.0)) for name in OPTION_NAMES),
                key=lambda item: -item[1],
            )[:4]
            diag_tag = (
                f" base={eval_diag.get('mean_final_base_health', 0.0):.1f} "
                + " ".join(f"opt_{name}={count:.0f}" for name, count in top_options)
            )
        print(
            f"update {update:>4}/{args.updates} samples={len(transitions):>5} "
            f"rollout={rollout_score:.4f} eval={eval_score:.4f} {part_tag}{diag_tag} "
            f"pi_loss={stats['policy_loss']:.4f} v_loss={stats['value_loss']:.4f} "
            f"entropy={stats['entropy']:.3f} kl={stats['approx_kl']:.4f} "
            f"entcoef={entropy_coef:.3f} elapsed={(time.time() - t0) / 60:.1f}m"
        )
    print(f"\nBest saved eval score: {best_eval:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-checkpoint", default="training/ae/checkpoints/option_policy.pt")
    parser.add_argument("--out", default="training/ae/checkpoints/option_policy_ppo.pt")
    parser.add_argument("--latest-out", default="")
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--games-per-update", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.20)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--entropy-final-coef", type=float, default=0.003)
    parser.add_argument("--value-coef", type=float, default=0.50)
    parser.add_argument("--max-grad-norm", type=float, default=0.50)
    parser.add_argument("--reward-scale", type=float, default=50.0)
    parser.add_argument("--return-clip", type=float, default=10.0)
    parser.add_argument("--health-delta-coef", type=float, default=0.15)
    parser.add_argument("--base-health-delta-coef", type=float, default=1.25)
    parser.add_argument("--base-survival-bonus", type=float, default=0.35)
    parser.add_argument("--base-destroyed-penalty", type=float, default=0.75)
    parser.add_argument("--option-temperature", type=float, default=1.35)
    parser.add_argument("--option-epsilon", type=float, default=0.05)
    parser.add_argument("--n-frames", type=int, default=4)
    parser.add_argument("--no-belief", action="store_true")
    parser.add_argument("--non-novice", action="store_true")
    parser.add_argument("--opponent-suites", nargs="+", default=DEFAULT_TRAIN_SUITES)
    parser.add_argument("--eval-suites", nargs="+", default=[
        "cloudsuite", "pressure2", "strong_realistic", "base_rush_exploit",
        "bracket_proxy", "top_seed_proxy", "defense_trap", "mixed",
    ])
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=12)
    parser.add_argument("--eval-seed", type=int, default=137)
    parser.add_argument("--save-floor", type=float, default=0.285)
    parser.add_argument("--use-furnished-weights", dest="use_furnished_weights", action="store_true")
    parser.add_argument("--uniform-eval-weights", dest="use_furnished_weights", action="store_false")
    parser.set_defaults(use_furnished_weights=True)
    parser.add_argument("--required-suites", nargs="+", default=["top_seed_proxy", "base_rush_exploit"])
    parser.add_argument("--min-required-suite-score", type=float, default=0.16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    train(parser.parse_args())


if __name__ == "__main__":
    main()
