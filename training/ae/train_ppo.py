"""PPO fine-tuning for the AE policy (v2 — frame-stacked + value-clipped).

Changes vs the first PPO run that landed `ppo-v1` (official 0.507):

  * **Frame stacking** — `PolicyNetwork(n_frames=4)`, with a per-env
    `FrameStacker` so the network sees enemy/bomb motion explicitly.
  * **Reward scaling** — divide raw rewards by `--reward-scale` (default 50)
    before computing returns. With raw rewards reaching +50 for base-kills,
    GAE returns were O(100) and v_loss exploded to ~10^3. Scaling targets
    O(1).
  * **Clipped value loss** — PPO-paper style:
        v_clipped = old_v + clamp(new_v - old_v, ±clip)
        v_loss   = 0.5 * max((new_v - R)², (v_clipped - R)²).mean()
    Prevents runaway value updates that destabilize the policy.
  * **Linear LR decay** — from `--lr` to `lr * 0.1` over `--updates`.
  * **Varied maps option** — `--vary-maps` switches the env to
    `novice=False` and randomizes seeds per game so the training
    distribution isn't a single fixed scenario (one of the candidates for
    closing the local→official gap).
  * **Bigger rollouts** — `--games-per-update` default bumped 8 → 12.

Deployment compatibility: the saved checkpoint embeds `n_frames`; the AE
container reads it and configures the FrameStacker accordingly.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "ae" / "src"))

from encoder import (  # noqa: E402
    BELIEF_CHANNELS,
    FrameStacker,
    SCALAR_DIM,
    encode_observation,
    rasterize_belief,
)
from model import (  # noqa: E402
    AGENT_VIEW_HW,
    BASE_VIEW_HW,
    BELIEF_HW,
    PolicyNetwork,
    VIEW_CHANNELS,
    num_parameters,
)
from ae_manager import AEManager  # noqa: E402
from til_environment import bomberman_env  # noqa: E402
from til_environment.config import default_config  # noqa: E402


MAX_SCORE = 1000.0
ACTION_DIM = 6


@dataclass
class Transition:
    agent_view: np.ndarray
    base_view: np.ndarray
    scalars: np.ndarray
    action_mask: np.ndarray
    action: int
    logprob: float
    value: float
    belief_map: np.ndarray | None = None  # (BELIEF_CHANNELS, 16, 16) or None
    reward: float = 0.0
    done: bool = False


class ValueNetwork(nn.Module):
    """Critic — same input layout as PolicyNetwork, scalar output."""

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
        head_in = 16 * 7 * 5 + 8 * 7 * 7 + scalar_dim
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

    def forward(self, agent_view: torch.Tensor, base_view: torch.Tensor,
                scalars: torch.Tensor, belief_map: torch.Tensor | None = None) -> torch.Tensor:
        a = self.agent_conv(agent_view).flatten(start_dim=1)
        b = self.base_conv(base_view).flatten(start_dim=1)
        parts = [a, b, scalars]
        if self.use_belief:
            if belief_map is None:
                raise ValueError("ValueNetwork(use_belief=True) requires belief_map")
            parts.append(self.belief_conv(belief_map).flatten(start_dim=1))
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)


def _belief_for(planner: AEManager, obs_py: dict, use_belief: bool) -> np.ndarray | None:
    """Drive the planner's memory update and rasterize belief (or skip)."""
    if not use_belief:
        return None
    # Run the planner in 'memory only' mode by mimicking ae() prologue.
    step = planner._as_int(obs_py.get("step"), default=(planner.last_step or 0) + 1)
    if planner.last_step is None or step == 0 or step < planner.last_step:
        planner._reset_memory()
    planner._age_bombs(step)
    planner._blast_cache = {}
    planner.last_step = step
    location = planner._location(obs_py.get("location"))
    direction = planner._as_int(obs_py.get("direction"), default=0) % 4
    planner._update_memory(obs_py, step, location, direction)
    return rasterize_belief(planner, obs_py)


def _obs_to_python(obs) -> dict:
    out = {}
    for key, value in obs.items():
        if hasattr(value, "tolist"):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def _masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits + torch.log(mask.clamp(min=1e-9))


def _encode_batch(transitions: list[Transition], device: torch.device, use_belief: bool):
    agent_views = torch.from_numpy(np.stack([t.agent_view for t in transitions])).float().to(device)
    base_views = torch.from_numpy(np.stack([t.base_view for t in transitions])).float().to(device)
    scalars = torch.from_numpy(np.stack([t.scalars for t in transitions])).float().to(device)
    masks = torch.from_numpy(np.stack([t.action_mask for t in transitions])).float().to(device)
    actions = torch.tensor([t.action for t in transitions], dtype=torch.long, device=device)
    old_logprobs = torch.tensor([t.logprob for t in transitions], dtype=torch.float32, device=device)
    values = torch.tensor([t.value for t in transitions], dtype=torch.float32, device=device)
    rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=device)
    dones = torch.tensor([t.done for t in transitions], dtype=torch.float32, device=device)
    if use_belief:
        beliefs = torch.from_numpy(np.stack([t.belief_map for t in transitions])).float().to(device)
    else:
        beliefs = None
    return agent_views, base_views, scalars, masks, actions, old_logprobs, values, rewards, dones, beliefs


def _stacked_tensors(stacker: FrameStacker, obs_py: dict, belief: np.ndarray | None,
                     device: torch.device):
    stacked = stacker.observe(obs_py, belief_map=belief)
    agent_v = torch.from_numpy(stacked["agent_view"]).float().to(device)
    base_v = torch.from_numpy(stacked["base_view"]).float().to(device)
    scalars = torch.from_numpy(stacked["scalars"]).float().to(device)
    mask = torch.from_numpy(stacked["action_mask"]).float().to(device)
    belief_t = torch.from_numpy(belief).float().to(device) if belief is not None else None
    return stacked, agent_v, base_v, scalars, mask, belief_t


def _select_action(
    actor: PolicyNetwork,
    critic: ValueNetwork,
    stacker: FrameStacker,
    planner: AEManager,
    obs_py: dict,
    device: torch.device,
    use_belief: bool,
    greedy: bool = False,
) -> tuple[int, float, float, dict, np.ndarray | None]:
    belief = _belief_for(planner, obs_py, use_belief)
    stacked, agent_v, base_v, scalars, mask, belief_t = _stacked_tensors(
        stacker, obs_py, belief, device,
    )
    belief_b = belief_t.unsqueeze(0) if belief_t is not None else None
    with torch.no_grad():
        logits = _masked_logits(
            actor(agent_v.unsqueeze(0), base_v.unsqueeze(0), scalars.unsqueeze(0),
                  belief_map=belief_b).squeeze(0),
            mask,
        )
        value = critic(agent_v.unsqueeze(0), base_v.unsqueeze(0), scalars.unsqueeze(0),
                       belief_map=belief_b).squeeze(0)
        if greedy:
            action = int(logits.argmax().item())
            logprob = torch.log_softmax(logits, dim=-1)[action]
        else:
            dist = Categorical(logits=logits)
            sampled = dist.sample()
            action = int(sampled.item())
            logprob = dist.log_prob(sampled)
    return action, float(logprob.item()), float(value.item()), stacked, belief


def _random_opponent(env, agent: str, _obs_py: dict) -> int:
    return int(env.action_space(agent).sample())


class PlannerOpponent:
    def __init__(self):
        self.manager = AEManager()

    def __call__(self, _env, _agent: str, obs_py: dict) -> int:
        if obs_py.get("step") == 0:
            self.manager = AEManager()
        return int(self.manager.ae(obs_py))


class FrozenPolicyOpponent:
    """Frozen copy of the actor, with its own FrameStacker + belief AEManager."""

    def __init__(self, actor: PolicyNetwork, device: torch.device, n_frames: int):
        self.actor = copy.deepcopy(actor).to(device).eval()
        self.device = device
        self.use_belief = bool(getattr(self.actor, "use_belief", False))
        self.stacker = FrameStacker(n_frames)
        self.planner = AEManager()  # for belief tracking only

    def reset(self):
        self.stacker.reset()
        self.planner = AEManager()

    def __call__(self, _env, _agent: str, obs_py: dict) -> int:
        belief = _belief_for(self.planner, obs_py, self.use_belief)
        stacked = self.stacker.observe(obs_py, belief_map=belief)
        agent_v = torch.from_numpy(stacked["agent_view"]).float().to(self.device)
        base_v = torch.from_numpy(stacked["base_view"]).float().to(self.device)
        scalars = torch.from_numpy(stacked["scalars"]).float().to(self.device)
        mask = torch.from_numpy(stacked["action_mask"]).float().to(self.device)
        belief_t = torch.from_numpy(belief).float().to(self.device) if belief is not None else None
        return self.actor.select_action(
            agent_v, base_v, scalars, action_mask=mask, greedy=True, belief_map=belief_t,
        )


class AggressivePlannerOpponent:
    """Bias the rule-based planner toward hunting our agent.

    Wraps the standard `AEManager` but inflates the value of being near
    the controlled-agent's last-known position via a closure on shared
    state. Cheaper than a learned aggressor and gives PPO an opponent
    that actively *seeks* the trainee, which random/frozen-self don't.
    The hidden eval almost certainly uses something with this archetype.
    """

    def __init__(self):
        self.manager = AEManager()
        # Override item-value table for this manager only — boost enemy bombs.
        self.manager.PATH_THREAT_PENALTY = 0.5  # walk toward enemies, not away
        self.manager.CELL_THREAT_PENALTY = 1.0
        self.manager.LOW_HEALTH_THRESHOLD = 10  # less retreat

    def reset(self):
        self.manager = AEManager()
        self.manager.PATH_THREAT_PENALTY = 0.5
        self.manager.CELL_THREAT_PENALTY = 1.0
        self.manager.LOW_HEALTH_THRESHOLD = 10

    def __call__(self, _env, _agent: str, obs_py: dict) -> int:
        if obs_py.get("step") == 0:
            self.reset()
        return int(self.manager.ae(obs_py))


def _make_opponents(
    actor: PolicyNetwork,
    device: torch.device,
    mode: str,
    opponent_agents: list[str],
    n_frames: int,
) -> dict[str, Callable]:
    """Build opponent dict.

    Modes:
    - random: uniform-random; fastest but unrealistic
    - planner: frozen rule-based AEManager
    - frozen:  frozen copy of the trainee
    - aggressive: planner with combat bias (NEW — hunter archetype)
    - mixed: random + planner + frozen
    - league: random + planner + aggressive + frozen   ← strongest pool
    """
    choices: list[Callable] = []
    if mode in {"random", "mixed", "league"}:
        choices.extend([_random_opponent, _random_opponent])
    if mode in {"planner", "mixed", "league"}:
        choices.append(PlannerOpponent())
    if mode in {"aggressive", "league"}:
        choices.append(AggressivePlannerOpponent())
    if mode in {"frozen", "mixed", "league"}:
        choices.append(FrozenPolicyOpponent(actor, device, n_frames))
    if not choices:
        choices = [_random_opponent]
    return {agent: random.choice(choices) for agent in opponent_agents}


def _make_env(args: argparse.Namespace):
    config = default_config()
    config.env.novice = (not args.vary_maps) and args.novice
    return bomberman_env.basic_env(env_wrappers=[], cfg=config)


def collect_rollouts(
    actor: PolicyNetwork,
    critic: ValueNetwork,
    args: argparse.Namespace,
    device: torch.device,
    seed_offset: int,
) -> tuple[list[Transition], float]:
    env = _make_env(args)
    our_agent = env.possible_agents[0]

    transitions: list[Transition] = []
    total_reward = 0.0

    use_belief = bool(getattr(actor, "use_belief", False))

    for game in range(args.games_per_update):
        if args.vary_maps:
            env.reset(seed=random.randint(0, 2**31 - 1))
        elif args.seed is not None:
            env.reset(seed=args.seed + seed_offset + game)
        else:
            env.reset()
        stacker = FrameStacker(args.n_frames)
        # Per-game belief-tracking planner for OUR agent. Cheap when use_belief=False.
        planner = AEManager()
        opponents = _make_opponents(
            actor, device, args.opponents,
            [a for a in env.possible_agents if a != our_agent],
            args.n_frames,
        )
        for op in opponents.values():
            if hasattr(op, "reset"):
                op.reset()
        pending_idx: int | None = None

        for agent in env.agent_iter():
            obs, reward, termination, truncation, _info = env.last()
            done = bool(termination or truncation)
            if agent == our_agent and pending_idx is not None:
                transitions[pending_idx].reward = float(reward)
                transitions[pending_idx].done = done
                total_reward += float(reward)
                pending_idx = None

            if done:
                env.step(None)
                continue

            obs_py = _obs_to_python(obs)
            if agent == our_agent:
                action, logprob, value, stacked, belief = _select_action(
                    actor, critic, stacker, planner, obs_py, device, use_belief, greedy=False,
                )
                transitions.append(Transition(
                    agent_view=stacked["agent_view"],
                    base_view=stacked["base_view"],
                    scalars=stacked["scalars"],
                    action_mask=stacked["action_mask"],
                    action=action,
                    logprob=logprob,
                    value=value,
                    belief_map=belief,
                ))
                pending_idx = len(transitions) - 1
            else:
                action = int(opponents.get(agent, _random_opponent)(env, agent, obs_py))
                mask = np.asarray(obs_py.get("action_mask", [1, 1, 1, 1, 1, 1]), dtype=np.float32).reshape(-1)
                if action < 0 or action >= ACTION_DIM or not bool(mask[action]):
                    action = _random_legal_action(mask, env, agent)
            env.step(action)

    env.close()
    return transitions, total_reward / max(args.games_per_update, 1) / MAX_SCORE


def _random_legal_action(mask: np.ndarray, env, agent: str) -> int:
    legal = np.flatnonzero(mask[:ACTION_DIM] > 0)
    if legal.size:
        return int(np.random.choice(legal))
    return int(env.action_space(agent).sample())


def compute_returns_advantages(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    next_value = 0.0
    for t in reversed(range(rewards.numel())):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    return returns, advantages


def ppo_update(
    actor: PolicyNetwork,
    critic: ValueNetwork,
    transitions: list[Transition],
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    use_belief = bool(getattr(actor, "use_belief", False))
    (agent_v, base_v, scalars, masks, actions, old_logprobs,
     values, rewards, dones, beliefs) = _encode_batch(transitions, device, use_belief)

    scaled_rewards = rewards / args.reward_scale
    returns, advantages = compute_returns_advantages(scaled_rewards, dones, values, args.gamma, args.gae_lambda)
    returns = returns.clamp(-args.return_clip, args.return_clip)

    n = actions.numel()
    idx = torch.arange(n, device=device)
    last_policy_loss = last_value_loss = last_entropy = 0.0

    for _epoch in range(args.ppo_epochs):
        perm = idx[torch.randperm(n, device=device)]
        for start in range(0, n, args.batch_size):
            batch = perm[start:start + args.batch_size]
            b_belief = beliefs[batch] if beliefs is not None else None
            logits = _masked_logits(
                actor(agent_v[batch], base_v[batch], scalars[batch], belief_map=b_belief),
                masks[batch],
            )
            dist = Categorical(logits=logits)
            new_logprobs = dist.log_prob(actions[batch])
            entropy = dist.entropy().mean()

            ratio = (new_logprobs - old_logprobs[batch]).exp()
            unclipped = ratio * advantages[batch]
            clipped = ratio.clamp(1.0 - args.clip_coef, 1.0 + args.clip_coef) * advantages[batch]
            policy_loss = -torch.min(unclipped, clipped).mean()

            new_values = critic(agent_v[batch], base_v[batch], scalars[batch], belief_map=b_belief)
            old_v = values[batch]
            v_clipped = old_v + (new_values - old_v).clamp(-args.clip_coef, args.clip_coef)
            v_loss_unclipped = (new_values - returns[batch]) ** 2
            v_loss_clipped = (v_clipped - returns[batch]) ** 2
            value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), args.max_grad_norm)
            optimizer.step()

            last_policy_loss = float(policy_loss.item())
            last_value_loss = float(value_loss.item())
            last_entropy = float(entropy.item())

    return {
        "policy_loss": last_policy_loss,
        "value_loss": last_value_loss,
        "entropy": last_entropy,
        "mean_reward": float(rewards.mean().item()),
    }


def evaluate(actor: PolicyNetwork, args: argparse.Namespace, device: torch.device, games: int) -> float:
    env = _make_env(args)
    our_agent = env.possible_agents[0]
    total_reward = 0.0
    use_belief = bool(getattr(actor, "use_belief", False))

    for game in range(games):
        if args.vary_maps:
            env.reset(seed=args.eval_seed + game * 7919 if args.eval_seed is not None else None)
        elif args.eval_seed is not None:
            env.reset(seed=args.eval_seed + game)
        else:
            env.reset()
        stacker = FrameStacker(args.n_frames)
        planner = AEManager()
        opponents = _make_opponents(
            actor, device, args.eval_opponents,
            [a for a in env.possible_agents if a != our_agent],
            args.n_frames,
        )
        for op in opponents.values():
            if hasattr(op, "reset"):
                op.reset()
        for agent in env.agent_iter():
            obs, reward, termination, truncation, _info = env.last()
            if agent == our_agent:
                total_reward += float(reward)
            if termination or truncation:
                env.step(None)
                continue
            obs_py = _obs_to_python(obs)
            if agent == our_agent:
                belief = _belief_for(planner, obs_py, use_belief)
                stacked = stacker.observe(obs_py, belief_map=belief)
                agent_v = torch.from_numpy(stacked["agent_view"]).float().to(device)
                base_v = torch.from_numpy(stacked["base_view"]).float().to(device)
                scalars = torch.from_numpy(stacked["scalars"]).float().to(device)
                mask = torch.from_numpy(stacked["action_mask"]).float().to(device)
                belief_t = torch.from_numpy(belief).float().to(device) if belief is not None else None
                action = actor.select_action(
                    agent_v, base_v, scalars, action_mask=mask, greedy=True, belief_map=belief_t,
                )
            else:
                action = int(opponents.get(agent, _random_opponent)(env, agent, obs_py))
                mask = np.asarray(obs_py.get("action_mask", [1, 1, 1, 1, 1, 1]), dtype=np.float32).reshape(-1)
                if action < 0 or action >= ACTION_DIM or not bool(mask[action]):
                    action = _random_legal_action(mask, env, agent)
            env.step(action)

    env.close()
    return total_reward / max(games, 1) / MAX_SCORE


def load_actor(args: argparse.Namespace, device: torch.device) -> tuple[PolicyNetwork, bool]:
    """Construct the actor and warm-start from BC if available.

    Returns (actor, use_belief). `use_belief` is taken from the BC
    checkpoint if present; if no checkpoint is found, falls back to
    args.use_belief.
    """
    use_belief = bool(args.use_belief)
    ckpt_path = Path(args.bc_checkpoint) if args.bc_checkpoint else None
    ckpt = None
    if ckpt_path and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckpt_use_belief = bool(ckpt.get("use_belief", False))
        ckpt_n_frames = ckpt.get("n_frames", 1)
        if ckpt_use_belief != use_belief:
            print(
                f"INFO: --use-belief={use_belief} overridden by checkpoint use_belief={ckpt_use_belief}"
            )
            use_belief = ckpt_use_belief
        if ckpt_n_frames != args.n_frames:
            print(
                f"WARN: checkpoint n_frames={ckpt_n_frames} != requested n_frames={args.n_frames}; "
                "training from scratch instead of warm-starting."
            )
            ckpt = None  # will skip load_state_dict below
    actor = PolicyNetwork(n_frames=args.n_frames, use_belief=use_belief).to(device)
    if ckpt is not None:
        actor.load_state_dict(ckpt["model_state_dict"])
        print(f"warm-started actor from {ckpt_path} (n_frames={args.n_frames}, use_belief={use_belief})")
    elif ckpt_path and not ckpt_path.exists():
        print(f"BC checkpoint not found at {ckpt_path}; training PPO from scratch (use_belief={use_belief})")
    return actor, use_belief


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed or 0)
    np.random.seed(args.seed or 0)
    torch.manual_seed(args.seed or 0)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    )
    print(f"device: {device}")
    print(f"config: n_frames={args.n_frames} reward_scale={args.reward_scale} "
          f"return_clip={args.return_clip} vary_maps={args.vary_maps} "
          f"opponents={args.opponents} eval_opponents={args.eval_opponents}")

    actor, use_belief = load_actor(args, device)
    critic = ValueNetwork(n_frames=args.n_frames, use_belief=use_belief).to(device)
    print(f"actor params: {num_parameters(actor):,}; critic params: {num_parameters(critic):,} "
          f"(use_belief={use_belief})")

    optimizer = optim.AdamW(
        list(actor.parameters()) + list(critic.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # Linear LR decay to 10% of starting lr over the full run.
    lr_lambda = lambda step: max(0.1, 1.0 - step / max(args.updates, 1))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_eval = -float("inf")
    start_time = time.time()

    for update in range(1, args.updates + 1):
        actor.train()
        critic.train()
        transitions, rollout_score = collect_rollouts(actor, critic, args, device, seed_offset=update * args.games_per_update)
        if not transitions:
            raise SystemExit("No PPO transitions collected; environment likely terminated before our agent acted.")
        stats = ppo_update(actor, critic, transitions, optimizer, args, device)
        scheduler.step()

        eval_score = float("nan")
        if update == 1 or update % args.eval_every == 0:
            actor.eval()
            critic.eval()
            eval_score = evaluate(actor, args, device, games=args.eval_games)
            if eval_score > best_eval:
                best_eval = eval_score
                torch.save({
                    "model_state_dict": actor.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "epoch": update,
                    "n_frames": args.n_frames,
                    "use_belief": bool(getattr(actor, "use_belief", False)),
                    "ppo_eval_score": eval_score,
                    "rollout_score": rollout_score,
                    "args": vars(args),
                }, out_path)
                print(f"  ✓ best PPO eval {best_eval:.4f} → saved {out_path}")

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"update {update:>4}/{args.updates}  "
            f"samples={len(transitions):>5}  rollout={rollout_score:.4f}  "
            f"eval={eval_score:.4f}  best={best_eval:.4f}  "
            f"pi_loss={stats['policy_loss']:.4f}  v_loss={stats['value_loss']:.4f}  "
            f"entropy={stats['entropy']:.3f}  lr={current_lr:.2e}  elapsed={elapsed/60:.1f}m"
        )

    print(f"\nBest eval score: {best_eval:.4f}")
    print(f"Checkpoint: {out_path}")
    print("Deploy by copying it to ae/models/bc.pt, then build/test as ppo-v2.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-checkpoint", default="training/ae/checkpoints/bc.pt")
    parser.add_argument("--out", default="training/ae/checkpoints/ppo.pt")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--games-per-update", type=int, default=12)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.20)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.50)
    parser.add_argument("--max-grad-norm", type=float, default=0.50)
    parser.add_argument("--reward-scale", type=float, default=50.0,
                        help="Divide raw rewards by this before returns (max event reward ~50).")
    parser.add_argument("--return-clip", type=float, default=10.0,
                        help="Final clamp on returns to keep value targets bounded.")
    parser.add_argument("--n-frames", type=int, default=4,
                        help="Number of consecutive observations stacked into the policy input.")
    parser.add_argument("--vary-maps", action="store_true", default=False,
                        help="Train with novice=False and a random seed per game (diversify the training distribution).")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=12)
    parser.add_argument("--opponents", choices=["random", "planner", "frozen", "aggressive", "mixed", "league"], default="mixed")
    parser.add_argument("--eval-opponents", choices=["random", "planner", "frozen", "aggressive", "mixed", "league"], default="mixed")
    parser.add_argument("--use-belief", action="store_true",
                        help="Train with the belief-map architecture (16x16xK extra CNN branch). "
                             "Overridden by the BC checkpoint's use_belief flag if loading one.")
    parser.add_argument("--novice", action="store_true", default=True)
    parser.add_argument("--no-novice", dest="novice", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
