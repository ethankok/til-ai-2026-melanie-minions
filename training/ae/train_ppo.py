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
import os
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
sys.path.insert(1, str(REPO_ROOT / "ae" / "src"))

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
    build_policy_network,
    num_parameters,
)
from ae_manager import AEManager  # noqa: E402
from til_environment import bomberman_env  # noqa: E402
from til_environment.config import default_config  # noqa: E402


MAX_SCORE = 1000.0
ACTION_DIM = 6
OPPONENT_MODES = (
    "static",
    "random",
    "planner",
    "frozen",
    "aggressive",
    "mixed",
    "league",
    "selfplay",
    "scripted",
    "cloudsuite",
    "pressure2",
)

OPPONENT_MIX_PRESETS = {
    # Full fixed-Novice RL blend. The core mass is scripted+cloudsuite because
    # local random is known-misleading, while small planner/aggressive/league
    # slices keep the policy from overfitting one handcrafted proxy.
    "full-rl": "scripted:0.35,cloudsuite:0.35,pressure2:0.15,planner:0.05,aggressive:0.05,league:0.05",
}

CURRICULA = {
    # Warm up on easy legal-action pressure, then move into the opponent
    # families that have been most predictive of hidden-eval failure.
    "pressure": (
        (0.00, "scripted"),
        (0.35, "cloudsuite"),
        (0.70, "pressure2"),
        (0.88, "league"),
    ),
    # Broader Bomberman curriculum inspired by public Pommerman/TIL training
    # recipes: learn bomb usage safely, then progressively add moving and
    # adversarial opponents.
    "bomberman": (
        (0.00, "static"),
        (0.15, "scripted"),
        (0.55, "cloudsuite"),
        (0.82, "pressure2"),
    ),
}


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


class AdaptiveRewardShaper:
    """Small training-only reward shaping layer.

    Bomberman-style PPO has two recurring pathologies: timid policies stop
    bombing because early bombs are dangerous, while over-aggressive policies
    ignore base survival. This shaper keeps the raw environment reward as the
    dominant signal and adds bounded nudges for exploration, bombing, and
    health/base preservation. It is never used in the deployed manager.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.success_ema = 0.0
        self.visit_counts = np.zeros((16, 16), dtype=np.int32)
        self.last_health: float | None = None
        self.last_base_health: float | None = None
        self.episode_raw_reward = 0.0

    def start_game(self) -> None:
        self.visit_counts.fill(0)
        self.last_health = None
        self.last_base_health = None
        self.episode_raw_reward = 0.0

    @property
    def explore_weight(self) -> float:
        if self.args.explore_bonus <= 0.0:
            return 0.0
        anneal = 1.0 - math.tanh(self.args.explore_anneal_k * self.success_ema)
        anneal = max(self.args.explore_min_scale, anneal)
        return self.args.explore_bonus * anneal

    def shape(self, raw_reward: float, obs_py: dict | None, action: int, done: bool) -> float:
        shaped = float(raw_reward)
        self.episode_raw_reward += float(raw_reward)

        if obs_py is not None:
            shaped += self._explore_bonus(obs_py)
            shaped += self._health_delta_bonus(obs_py)

        if action == ACTION_DIM - 1 and self.args.bomb_action_bonus:
            shaped += self.args.bomb_action_bonus

        if done:
            success = 1.0 if self.episode_raw_reward >= self.args.explore_success_reward else 0.0
            self.success_ema = 0.95 * self.success_ema + 0.05 * success

        return shaped

    def _explore_bonus(self, obs_py: dict) -> float:
        weight = self.explore_weight
        if weight <= 0.0:
            return 0.0
        loc = obs_py.get("location")
        try:
            if hasattr(loc, "tolist"):
                loc = loc.tolist()
            x, y = int(loc[0]), int(loc[1])
        except Exception:
            return 0.0
        if not (0 <= x < self.visit_counts.shape[0] and 0 <= y < self.visit_counts.shape[1]):
            return 0.0
        visits = int(self.visit_counts[x, y])
        self.visit_counts[x, y] += 1
        return weight / (1.0 + visits)

    def _health_delta_bonus(self, obs_py: dict) -> float:
        bonus = 0.0

        health = _safe_float(obs_py.get("health"))
        if health is not None:
            if self.last_health is not None:
                bonus += self.args.health_delta_coef * (health - self.last_health)
            self.last_health = health

        base_health = _safe_float(obs_py.get("base_health"))
        if base_health is not None:
            if self.last_base_health is not None:
                bonus += self.args.base_health_delta_coef * (base_health - self.last_base_health)
            self.last_base_health = base_health

        return bonus


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


def _seed_eval_rngs(seed: int | None) -> None:
    """Pin all local RNGs for held-out evaluation games.

    The AE env and scripted opponents are mostly seedable through `env.reset`
    and their factories, but a few fallback paths still touch module-global RNGs.
    Seeding them during evaluation makes checkpoint selection repeatable without
    changing the stochastic PPO rollout sampler.
    """

    if seed is None:
        return
    seed = int(seed) % (2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _static_opponent(_env, _agent: str, obs_py: dict) -> int:
    mask = np.asarray(obs_py.get("action_mask", [1, 1, 1, 1, 1, 1]), dtype=np.float32).reshape(-1)
    if mask.size > 4 and bool(mask[4]):
        return 4
    legal = np.flatnonzero(mask[:ACTION_DIM] > 0)
    return int(legal[0]) if legal.size else 4


class PlannerOpponent:
    def __init__(self):
        self.manager = AEManager()

    def __call__(self, _env, _agent: str, obs_py: dict) -> int:
        if obs_py.get("step") == 0:
            self.manager = AEManager()
        return int(self.manager.ae(obs_py))


class SnapshotPool:
    """Bounded ring buffer of historical actor snapshots for self-play.

    The TIL workshop materials (notebook 05 — Multi-Agent Introduction)
    explicitly call out the failure mode our prior PPO runs hit:

        "It's easy to overfit to a weak fixed opponent and regress when
         the opponent improves."

    And prescribe the fix:

        "Self-play trains an agent by having it compete against a copy
         of itself. Periodically, the opponent is updated to a
         checkpoint of the current policy."

    Implementation notes:

    - We keep snapshots on CPU (each PolicyNetwork is ~149k–704k params
      with belief, so 5 snapshots fits in <5 MB regardless of training
      device). :class:`FrozenPolicyOpponent` deepcopies + .to(device)
      at opponent construction.
    - ``add`` does a deepcopy so further training updates on the live
      actor don't mutate the stored snapshot.
    - ``sample`` returns ``None`` when the pool is empty; callers should
      fall back to the live actor in that case (matches pre-self-play
      behavior for the first few snapshot intervals).
    """

    def __init__(self, max_size: int = 5):
        self.max_size = max(1, int(max_size))
        self._snapshots: list[PolicyNetwork] = []

    def add(self, actor: PolicyNetwork) -> None:
        # Detach to CPU so we don't pin GPU memory; clone the state.
        snap = copy.deepcopy(actor).cpu().eval()
        self._snapshots.append(snap)
        # Drop oldest when over cap. FIFO; this preserves the "old +
        # recent" diversity the workshop and league-training papers
        # recommend (the alternative — always dropping a uniformly chosen entry — degrades
        # to "always recent" in expectation).
        if len(self._snapshots) > self.max_size:
            self._snapshots.pop(0)

    def sample(self) -> PolicyNetwork | None:
        if not self._snapshots:
            return None
        return random.choice(self._snapshots)

    def __len__(self) -> int:
        return len(self._snapshots)


class LivePolicyAdapter:
    """Expose the in-training actor through the deployed policy interface.

    `HybridAEManager` expects an object with `ae_logits(observation)`. Using
    this adapter lets checkpoint selection score the live actor through the same
    hybrid veto wrapper we will deploy, instead of selecting on pure-policy
    behavior that may never be used in the Docker image.
    """

    def __init__(self, actor: PolicyNetwork, device: torch.device, n_frames: int):
        self.model = actor
        self.device = device
        self.n_frames = n_frames
        self.use_belief = bool(getattr(actor, "use_belief", False))
        self.stacker = FrameStacker(n_frames)
        self.belief_manager = AEManager()
        self._last_step: int | None = None

    def _maybe_reset(self, observation: dict) -> None:
        try:
            step = int(observation.get("step", 0))
        except Exception:
            step = None
        if step == 0 or (
            self._last_step is not None and step is not None and step < self._last_step
        ):
            self.stacker.reset()
            self.belief_manager = AEManager()
        self._last_step = step

    def ae_logits(self, observation: dict) -> tuple[int, torch.Tensor]:
        self._maybe_reset(observation)
        belief = _belief_for(self.belief_manager, observation, self.use_belief)
        _stacked, agent_v, base_v, scalars, mask, belief_t = _stacked_tensors(
            self.stacker, observation, belief, self.device,
        )
        belief_b = belief_t.unsqueeze(0) if belief_t is not None else None
        with torch.inference_mode():
            logits = self.model(
                agent_v.unsqueeze(0),
                base_v.unsqueeze(0),
                scalars.unsqueeze(0),
                belief_map=belief_b,
            ).squeeze(0)
            masked = _masked_logits(logits, mask)
            return int(masked.argmax().item()), masked.detach().clone()


class FrozenPolicyOpponent:
    """Frozen copy of an actor, with its own FrameStacker + belief AEManager.

    ``actor`` may be either the live trainee (legacy behavior — gives you
    "play your immediate shadow", not real self-play) or a historical
    snapshot drawn from a :class:`SnapshotPool` (proper self-play: the
    opponent is frozen at an *older* policy state). Either way we
    deepcopy at construction so the rollout sees a fixed opponent for
    its duration.
    """

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
    snapshot_pool: SnapshotPool | None = None,
    seed_base: int | None = None,
    elo_pop: "EloPopulation | None" = None,
    live_rating: "LiveRating | None" = None,
) -> tuple[dict[str, Callable], int | None]:
    """Build opponent dict.

    Modes:
    - static:      always STAY when legal; easy bomb-usage curriculum
    - random:      legacy uniform-random sanity mode; not used in active presets
    - planner:     frozen rule-based AEManager
    - frozen:      frozen copy of the trainee (live actor — your shadow)
    - aggressive:  planner with combat bias (hunter archetype)
    - mixed:       planner + frozen
    - league:      planner + aggressive + frozen-from-pool
                   ← strongest pool; recommended for the qualifier
    - selfplay:    pool-only — face historical snapshots of yourself,
                   no heuristic mix. Workshop's pure-self-play setup;
                   may be unstable on its own but useful as an A/B.
    - scripted:    Tier 2 #9 — greedy + bomber + defender + hunter + rusher
                   from training/ae/opponents.py. No self-play. Trains a
                   policy robust to any of the cloud-likely behavior types.
    - cloudsuite:  rusher + hunter_sticky + bomber_fast + defender + mixed. This is the
                   pressure-heavy proxy pool for tactical-helper experiments,
                   not a default full-policy replacement path.
    - pressure2:   rusher_fast + rusher_safe + hunter_sticky + bomber_fast
                   + base_bomber. Adversarial stress proxy, mostly for
                   rejecting brittle candidates.

    Snapshot pool behavior:
    - When ``snapshot_pool`` is provided AND non-empty, frozen opponents
      are drawn from the pool (real self-play: face yourself-from-N-updates-ago).
    - When the pool is empty (early training) or absent, frozen opponents
      fall back to the live actor (legacy behavior — equivalent to
      "play your shadow").
    """
    # The Elo-matched path: when an EloPopulation is provided and non-empty,
    # frozen opponents are sampled via Gaussian-weighted matchmaking on the
    # live policy's current Elo. The chosen snapshot_id is recorded so the
    # training loop can apply post-game Elo updates symmetrically.
    chosen_snapshot_id: int | None = None
    def _frozen_opponent_actor() -> PolicyNetwork:
        nonlocal chosen_snapshot_id
        if elo_pop is not None and live_rating is not None and len(elo_pop) > 0:
            picked = elo_pop.sample_matched(live_rating.rating)
            if picked is not None:
                snap, snap_id = picked
                chosen_snapshot_id = snap_id
                return snap
        if snapshot_pool is not None:
            snap = snapshot_pool.sample()
            if snap is not None:
                return snap
        return actor

    choices: list[Callable] = []
    if mode == "static":
        choices.append(_static_opponent)
    if mode == "random":
        choices.extend([_random_opponent, _random_opponent])
    if mode in {"planner", "mixed", "league"}:
        choices.append(PlannerOpponent())
    if mode in {"aggressive", "league"}:
        choices.append(AggressivePlannerOpponent())
    if mode in {"frozen", "mixed", "league", "selfplay"}:
        choices.append(FrozenPolicyOpponent(_frozen_opponent_actor(), device, n_frames))
    if mode in {"scripted", "cloudsuite", "pressure2"}:
        # Tier 2 #9: train against the same scripted library we use in
        # training/ae/simulate.py so the policy learns to be robust across
        # the strategy space cloud opponents likely occupy. ``cloudsuite``
        # swaps in the pressure-heavy rusher/hunter mix used by local
        # validation so tactical helpers can be trained against the same pool.
        #
        # Each enemy slot gets its OWN factory rather than sharing one
        # instance — AEManager-derived opponents track per-agent belief
        # state (visible enemies, walls, bombs); sharing the same instance
        # across multiple agents in the same game would interleave belief
        # updates between agents and corrupt their decisions.
        try:
            from opponents import make_opponent  # noqa: WPS433
        except Exception as exc:
            print(f"[train_ppo] scripted opponents unavailable ({exc}); using random", flush=True)
            choices = [_random_opponent]
        else:
            if mode == "cloudsuite":
                scripted_names = ["rusher", "hunter_sticky", "bomber_fast", "defender", "mixed"]
            elif mode == "pressure2":
                scripted_names = ["rusher_fast", "rusher_safe", "hunter_sticky", "bomber_fast", "base_bomber"]
            else:
                scripted_names = ["greedy", "bomber", "defender", "hunter", "rusher"]

            class _ScriptedAdapter:
                """Wrap one scripted opponent for the (env, agent, obs_py)
                interface. Holds a private AEManager-or-RandomOpponent and
                resets it whenever a new game starts (step == 0)."""

                def __init__(self, name: str, seed: int):
                    self._name = name
                    self._seed = seed
                    self._op = make_opponent(name, seed=seed)

                def reset(self):
                    # Re-instantiate the inner opponent so belief state is
                    # zeroed at game boundaries. Cheap; AEManager init is
                    # ~ms.
                    self._op = make_opponent(self._name, seed=self._seed)

                def __call__(self, _env, _agent: str, obs_py: dict) -> int:
                    if obs_py.get("step") == 0:
                        self.reset()
                    try:
                        return int(self._op(obs_py))
                    except Exception:
                        # Defensive fallback: if a scripted opponent crashes
                        # (e.g. the env handed it a malformed observation)
                        # we don't want to take down PPO training.
                        mask = obs_py.get("action_mask")
                        if mask is not None:
                            for i, m in enumerate(mask):
                                try:
                                    if int(m):
                                        return i
                                except Exception:
                                    pass
                        return 4  # STAY

            # Assign each opponent_agent its own dedicated adapter so they
            # all run in parallel without sharing state.
            if seed_base is None:
                seed_base = random.randint(0, 2**31 - 1)
            scripted_assignment: dict[str, Callable] = {}
            for i, agent in enumerate(opponent_agents):
                # Cycle through the 5 scripted types so a 5-enemy game has
                # one of each type. This matches how `simulate.py --opponents
                # library` already works.
                op_name = scripted_names[i % len(scripted_names)]
                scripted_assignment[agent] = _ScriptedAdapter(op_name, seed=seed_base + i)
            return scripted_assignment, chosen_snapshot_id
    if not choices:
        choices = [_random_opponent]
    return {agent: random.choice(choices) for agent in opponent_agents}, chosen_snapshot_id


def _make_env(args: argparse.Namespace):
    config = default_config()
    config.env.novice = (not args.vary_maps) and args.novice
    return bomberman_env.basic_env(env_wrappers=[], cfg=config)


def _parse_opponent_mix(args: argparse.Namespace) -> list[tuple[str, float]]:
    raw = (args.opponent_mix or "").strip()
    preset = getattr(args, "opponent_mix_preset", "none")
    if preset != "none":
        raw = OPPONENT_MIX_PRESETS[preset]
    if not raw:
        return []

    mix: list[tuple[str, float]] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise SystemExit(
                f"Invalid --opponent-mix entry {entry!r}; expected mode:weight"
            )
        mode, weight_s = entry.split(":", 1)
        mode = mode.strip()
        if mode not in OPPONENT_MODES:
            raise SystemExit(f"Unknown opponent mix mode {mode!r}; choices: {', '.join(OPPONENT_MODES)}")
        try:
            weight = float(weight_s)
        except ValueError as exc:
            raise SystemExit(f"Invalid opponent mix weight in {entry!r}") from exc
        if weight <= 0.0:
            raise SystemExit(f"Opponent mix weight must be positive in {entry!r}")
        mix.append((mode, weight))

    total = sum(weight for _mode, weight in mix)
    if total <= 0.0:
        return []
    return [(mode, weight / total) for mode, weight in mix]


def _opponent_plan_for_update(args: argparse.Namespace) -> list[str]:
    """Return a shuffled, stratified per-game opponent-mode plan.

    For small update batches like 16 games, pure random sampling can easily
    omit a low-probability but important family. Stratifying keeps the intended
    full-RL mix present in every update while still shuffling order.
    """

    mix = _parse_opponent_mix(args)
    if not mix:
        return []

    n = int(args.games_per_update)
    exact = [(mode, weight * n) for mode, weight in mix]
    counts = {mode: int(math.floor(value)) for mode, value in exact}
    remaining = n - sum(counts.values())
    by_fraction = sorted(
        exact,
        key=lambda item: item[1] - math.floor(item[1]),
        reverse=True,
    )
    for mode, _value in by_fraction[:remaining]:
        counts[mode] += 1

    plan: list[str] = []
    for mode, count in counts.items():
        plan.extend([mode] * count)
    # Guard against roundoff/edge cases.
    if len(plan) < n:
        plan.extend([mix[0][0]] * (n - len(plan)))
    elif len(plan) > n:
        plan = plan[:n]
    random.shuffle(plan)
    return plan


def _format_mode_counts(mode_counts: dict[str, int]) -> str:
    if not mode_counts:
        return ""
    ordered = sorted(mode_counts.items(), key=lambda item: (-item[1], item[0]))
    return ",".join(f"{mode}:{count}" for mode, count in ordered)


def collect_rollouts(
    actor: PolicyNetwork,
    critic: ValueNetwork,
    args: argparse.Namespace,
    device: torch.device,
    seed_offset: int,
    snapshot_pool: SnapshotPool | None = None,
    opponent_mode: str | None = None,
    elo_pop: "EloPopulation | None" = None,
    live_rating: "LiveRating | None" = None,
    elo_baseline: float = 0.55,
) -> tuple[list[Transition], float, dict[str, int]]:
    env = _make_env(args)
    our_agent = env.possible_agents[0]

    transitions: list[Transition] = []
    total_reward = 0.0
    reward_shaper = AdaptiveRewardShaper(args)
    mode = opponent_mode or args.opponents
    opponent_plan = _opponent_plan_for_update(args)
    mode_counts: dict[str, int] = {}

    use_belief = bool(getattr(actor, "use_belief", False))

    for game in range(args.games_per_update):
        game_mode = opponent_plan[game] if opponent_plan else mode
        mode_counts[game_mode] = mode_counts.get(game_mode, 0) + 1
        reward_shaper.start_game()
        reset_seed = None
        if args.vary_maps:
            reset_seed = random.randint(0, 2**31 - 1)
            env.reset(seed=reset_seed)
        elif args.seed is not None:
            reset_seed = args.seed + seed_offset + game
            env.reset(seed=reset_seed)
        else:
            env.reset()
        stacker = FrameStacker(args.n_frames)
        # Per-game belief-tracking planner for OUR agent. Cheap when use_belief=False.
        planner = AEManager()
        opponents, elo_snapshot_id = _make_opponents(
            actor, device, game_mode,
            [a for a in env.possible_agents if a != our_agent],
            args.n_frames,
            snapshot_pool=snapshot_pool,
            seed_base=None if reset_seed is None else reset_seed + 100_000,
            elo_pop=elo_pop,
            live_rating=live_rating,
        )
        # Snapshot whatever our cumulative reward is so we can extract
        # this game's score for Elo update.
        game_start_reward = total_reward
        for op in opponents.values():
            if hasattr(op, "reset"):
                op.reset()
        pending_idx: int | None = None

        for agent in env.agent_iter():
            obs, reward, termination, truncation, _info = env.last()
            done = bool(termination or truncation)
            if agent == our_agent and pending_idx is not None:
                try:
                    current_obs_py = _obs_to_python(obs)
                except Exception:
                    current_obs_py = None
                transitions[pending_idx].reward = reward_shaper.shape(
                    float(reward),
                    current_obs_py,
                    transitions[pending_idx].action,
                    done,
                )
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

        # Game finished. If this game used an Elo-pool snapshot opponent,
        # update its rating + the live rating based on our normalized score
        # margin vs ``elo_baseline``. ``elo_baseline`` is set to a reasonable
        # mean cloud score so that the sigmoid centers on "did we play above
        # or below average for this codebase".
        if elo_pop is not None and live_rating is not None and elo_snapshot_id is not None:
            game_reward = (total_reward - game_start_reward) / MAX_SCORE
            from elo_population import score_to_outcome  # local import to avoid hard dep
            outcome = score_to_outcome(game_reward, elo_baseline)
            elo_pop.update(elo_snapshot_id, live_rating, outcome)

    env.close()
    return transitions, total_reward / max(args.games_per_update, 1) / MAX_SCORE, mode_counts


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
    clip_coef: float,
    entropy_coef: float,
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
    approx_kls: list[float] = []
    clipfracs: list[float] = []
    batches = 0
    early_stop = False

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

            logratio = new_logprobs - old_logprobs[batch]
            ratio = logratio.exp()
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()
                approx_kls.append(float(approx_kl.item()))
                clipfracs.append(float(clipfrac.item()))
            unclipped = ratio * advantages[batch]
            clipped = ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef) * advantages[batch]
            policy_loss = -torch.min(unclipped, clipped).mean()

            new_values = critic(agent_v[batch], base_v[batch], scalars[batch], belief_map=b_belief)
            old_v = values[batch]
            v_clipped = old_v + (new_values - old_v).clamp(-clip_coef, clip_coef)
            v_loss_unclipped = (new_values - returns[batch]) ** 2
            v_loss_clipped = (v_clipped - returns[batch]) ** 2
            value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            loss = policy_loss + args.value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), args.max_grad_norm)
            optimizer.step()

            last_policy_loss = float(policy_loss.item())
            last_value_loss = float(value_loss.item())
            last_entropy = float(entropy.item())
            batches += 1

            if args.target_kl > 0.0 and float(approx_kl.item()) > args.target_kl:
                early_stop = True
                break
        if early_stop:
            break

    return {
        "policy_loss": last_policy_loss,
        "value_loss": last_value_loss,
        "entropy": last_entropy,
        "mean_reward": float(rewards.mean().item()),
        "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
        "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
        "batches": float(batches),
        "early_stop": float(early_stop),
    }


def evaluate(
    actor: PolicyNetwork,
    args: argparse.Namespace,
    device: torch.device,
    games: int,
    opponent_mode: str | None = None,
) -> float:
    env = _make_env(args)
    our_agent = env.possible_agents[0]
    total_reward = 0.0
    use_belief = bool(getattr(actor, "use_belief", False))
    mode = opponent_mode or args.eval_opponents
    use_hybrid = getattr(args, "selection_manager", "policy") == "hybrid"

    for game in range(games):
        reset_seed = None
        if args.vary_maps:
            reset_seed = args.eval_seed + game * 7919 if args.eval_seed is not None else None
            _seed_eval_rngs(None if reset_seed is None else reset_seed + 200_000)
            env.reset(seed=reset_seed)
        elif args.eval_seed is not None:
            reset_seed = args.eval_seed + game
            _seed_eval_rngs(reset_seed + 200_000)
            env.reset(seed=reset_seed)
        else:
            env.reset()
        stacker = FrameStacker(args.n_frames)
        planner = AEManager()
        hybrid_manager = None
        if use_hybrid:
            from hybrid_manager import HybridAEManager  # noqa: WPS433

            fixed_map_shortcut = getattr(args, "selection_fixed_map_shortcut", "on")
            previous_shortcut = os.environ.get("AE_HYBRID_FIXED_MAP_SHORTCUT")
            if fixed_map_shortcut == "off":
                os.environ["AE_HYBRID_FIXED_MAP_SHORTCUT"] = "0"
            try:
                hybrid_manager = HybridAEManager(
                    policy=LivePolicyAdapter(actor, device, args.n_frames)
                )
            finally:
                if previous_shortcut is None:
                    os.environ.pop("AE_HYBRID_FIXED_MAP_SHORTCUT", None)
                else:
                    os.environ["AE_HYBRID_FIXED_MAP_SHORTCUT"] = previous_shortcut
        opponents, _ = _make_opponents(
            actor, device, mode,
            [a for a in env.possible_agents if a != our_agent],
            args.n_frames,
            seed_base=None if reset_seed is None else reset_seed + 100_000,
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
                if hybrid_manager is not None:
                    action = int(hybrid_manager.ae(obs_py))
                else:
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


def load_actor(args: argparse.Namespace, device: torch.device) -> tuple[PolicyNetwork, bool, bool, dict | None]:
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
    state_dict = ckpt.get("model_state_dict") if ckpt is not None else None
    actor = build_policy_network(
        n_frames=args.n_frames,
        use_belief=use_belief,
        state_dict=state_dict,
    ).to(device)
    warm_started = ckpt is not None
    if warm_started:
        actor.load_state_dict(state_dict)
        print(
            f"warm-started actor from {ckpt_path} "
            f"(arch={getattr(actor, 'model_arch', 'default')}, "
            f"n_frames={args.n_frames}, use_belief={use_belief})"
        )
    elif ckpt_path and not ckpt_path.exists():
        print(f"BC checkpoint not found at {ckpt_path}; training PPO from scratch (use_belief={use_belief})")
    return actor, use_belief, warm_started, ckpt


def save_policy_checkpoint(
    actor: PolicyNetwork,
    critic: ValueNetwork,
    args: argparse.Namespace,
    out_path: Path,
    update: int,
    eval_score: float,
    eval_parts: dict[str, float],
    rollout_score: float,
    metadata: dict | None = None,
) -> None:
    payload = {
        "model_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "epoch": update,
        "n_frames": args.n_frames,
        "use_belief": bool(getattr(actor, "use_belief", False)),
        "model_arch": getattr(actor, "model_arch", "default"),
        "ppo_eval_score": eval_score,
        "ppo_eval_parts": eval_parts,
        "rollout_score": rollout_score,
        "selection_manager": getattr(args, "selection_manager", "policy"),
        "args": vars(args),
    }
    if metadata:
        payload.update(metadata)
    torch.save(payload, out_path)


def _linear(start: float, end: float, progress: float) -> float:
    progress = min(1.0, max(0.0, progress))
    return start + (end - start) * progress


def _current_opponent_mode(args: argparse.Namespace, update: int) -> str:
    if args.curriculum == "none":
        return args.opponents
    schedule = CURRICULA[args.curriculum]
    progress = (update - 1) / max(args.updates - 1, 1)
    mode = schedule[0][1]
    for threshold, candidate in schedule:
        if progress >= threshold:
            mode = candidate
    return mode


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_weights(value: str, n: int) -> list[float]:
    if not value.strip():
        return [1.0 / max(n, 1)] * n
    weights = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(weights) != n:
        raise SystemExit(f"--selection-weights expected {n} values, got {len(weights)}")
    total = sum(weights)
    if total <= 0:
        raise SystemExit("--selection-weights must sum to a positive value")
    return [w / total for w in weights]


def evaluate_selection(
    actor: PolicyNetwork,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    eval_actor = actor
    eval_device = device
    if getattr(args, "selection_device", "train") == "cpu" and device.type != "cpu":
        eval_actor = copy.deepcopy(actor).cpu().eval()
        eval_device = torch.device("cpu")
    suites = _parse_csv(args.selection_suites)
    if not suites:
        score = evaluate(eval_actor, args, eval_device, games=args.eval_games, opponent_mode=args.eval_opponents)
        return score, {args.eval_opponents: score}
    for suite in suites:
        if suite not in OPPONENT_MODES:
            raise SystemExit(f"Unknown selection suite {suite!r}; choices: {', '.join(OPPONENT_MODES)}")
    weights = _parse_weights(args.selection_weights, len(suites))
    scores: dict[str, float] = {}
    weighted = 0.0
    for suite, weight in zip(suites, weights):
        score = evaluate(eval_actor, args, eval_device, games=args.selection_games, opponent_mode=suite)
        scores[suite] = score
        weighted += weight * score
    return weighted, scores


def _orthogonal_init(module: nn.Module, final_gain: float = 1.0) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        gain = math.sqrt(2.0)
        if isinstance(module, nn.Linear) and module.out_features in {1, ACTION_DIM}:
            gain = final_gain if module.out_features == 1 else 0.01
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def _uses_snapshot_opponents(args: argparse.Namespace) -> bool:
    snapshot_modes = {"frozen", "mixed", "league", "selfplay"}
    if args.opponents in snapshot_modes:
        return True
    return any(mode in snapshot_modes for mode, _weight in _parse_opponent_mix(args))


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset == "default":
        return args

    if args.preset == "qualifier-best":
        args.curriculum = "pressure"
        args.opponents = "scripted"
        args.eval_opponents = "cloudsuite"
        args.selection_suites = "scripted,cloudsuite,pressure2"
        args.selection_weights = "0.30,0.45,0.25"
        args.selection_games = max(args.selection_games, 12)
        args.games_per_update = max(args.games_per_update, 16)
        args.ppo_epochs = min(args.ppo_epochs, 4)
        args.entropy_coef = max(args.entropy_coef, 0.02)
        args.entropy_final_coef = 0.004 if args.entropy_final_coef is None else args.entropy_final_coef
        args.clip_coef = min(args.clip_coef, 0.20)
        args.clip_final_coef = 0.12 if args.clip_final_coef is None else args.clip_final_coef
        args.target_kl = args.target_kl if args.target_kl > 0.0 else 0.03
        args.explore_bonus = max(args.explore_bonus, 0.15)
        args.explore_min_scale = max(args.explore_min_scale, 0.10)
        args.explore_anneal_k = max(args.explore_anneal_k, 1.2)
        args.health_delta_coef = max(args.health_delta_coef, 0.01)
        args.base_health_delta_coef = max(args.base_health_delta_coef, 0.03)
        args.bomb_action_bonus = max(args.bomb_action_bonus, 0.02)
        args.orthogonal_init = True
        return args

    if args.preset == "full-rl":
        args.curriculum = "none"
        args.opponent_mix_preset = "full-rl"
        args.opponents = "scripted"
        args.eval_opponents = "cloudsuite"
        args.selection_manager = "policy"
        args.selection_device = "cpu"
        args.selection_suites = "scripted,cloudsuite,pressure2"
        args.selection_weights = "0.35,0.40,0.25"
        args.selection_games = max(args.selection_games, 24)
        args.games_per_update = max(args.games_per_update, 16)
        args.eval_every = max(args.eval_every, 10)
        args.ppo_epochs = min(args.ppo_epochs, 3)
        args.entropy_coef = max(args.entropy_coef, 0.008)
        args.entropy_final_coef = 0.003 if args.entropy_final_coef is None else args.entropy_final_coef
        args.clip_coef = min(args.clip_coef, 0.14)
        args.clip_final_coef = 0.08 if args.clip_final_coef is None else args.clip_final_coef
        args.target_kl = args.target_kl if args.target_kl > 0.0 else 0.015
        args.explore_bonus = max(args.explore_bonus, 0.04)
        args.explore_min_scale = max(args.explore_min_scale, 0.10)
        args.health_delta_coef = max(args.health_delta_coef, 0.005)
        args.base_health_delta_coef = max(args.base_health_delta_coef, 0.025)
        args.bomb_action_bonus = max(args.bomb_action_bonus, 0.006)
        args.load_critic = True
        args.baseline_eval = True
        args.min_save_improvement = max(args.min_save_improvement, 0.015)
        return args

    return args


def train(args: argparse.Namespace) -> None:
    args = apply_preset(args)
    random.seed(args.seed or 0)
    np.random.seed(args.seed or 0)
    torch.manual_seed(args.seed or 0)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    )
    print(f"device: {device}")
    if os.environ.get("PYTHONHASHSEED") in {None, "", "random"}:
        print(
            "WARN: PYTHONHASHSEED is not fixed; AEManager set iteration can make "
            "small validation suites differ across Python processes. "
            "Use training/ae/run_full_rl_v1.py or run with PYTHONHASHSEED=0."
        )
    shortcut_tag = (
        args.selection_fixed_map_shortcut
        if args.selection_manager == "hybrid"
        else "n/a"
    )
    print(f"config: preset={args.preset} n_frames={args.n_frames} "
          f"reward_scale={args.reward_scale} return_clip={args.return_clip} "
          f"vary_maps={args.vary_maps} opponents={args.opponents} "
          f"eval_opponents={args.eval_opponents} curriculum={args.curriculum} "
          f"opponent_mix={args.opponent_mix_preset if args.opponent_mix_preset != 'none' else (args.opponent_mix or 'none')} "
          f"selection={args.selection_suites or args.eval_opponents} "
          f"selection_manager={args.selection_manager} selection_device={args.selection_device} "
          f"selection_fixed_map_shortcut={shortcut_tag}")

    actor, use_belief, warm_started, actor_ckpt = load_actor(args, device)
    if args.orthogonal_init and not warm_started:
        actor.apply(lambda m: _orthogonal_init(m, final_gain=0.01))
    critic = ValueNetwork(n_frames=args.n_frames, use_belief=use_belief).to(device)
    critic_loaded = False
    if args.load_critic and actor_ckpt is not None and actor_ckpt.get("critic_state_dict") is not None:
        try:
            critic.load_state_dict(actor_ckpt["critic_state_dict"])
            critic_loaded = True
            print("warm-started critic from checkpoint")
        except Exception as exc:
            print(f"WARN: critic warm-start skipped ({exc})")
    if args.orthogonal_init and not critic_loaded:
        critic.apply(lambda m: _orthogonal_init(m, final_gain=1.0))
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
    latest_out_path: Path | None = Path(args.latest_out) if args.latest_out else None
    if latest_out_path is None and args.preset == "full-rl":
        latest_out_path = out_path.with_name(f"{out_path.stem}-latest{out_path.suffix}")
    if latest_out_path is not None:
        latest_out_path.parent.mkdir(parents=True, exist_ok=True)

    gate_metadata: dict = {}
    save_floor = float(args.selection_min_score)
    if args.baseline_eval:
        actor.eval()
        critic.eval()
        baseline_score, baseline_parts = evaluate_selection(actor, args, device)
        gate_metadata["selection_baseline_score"] = baseline_score
        gate_metadata["selection_baseline_parts"] = baseline_parts
        if args.min_save_improvement > 0.0:
            save_floor = max(save_floor, baseline_score + args.min_save_improvement)
        print(
            f"selection baseline ({args.selection_manager})={baseline_score:.4f} "
            + " ".join(f"{k}={v:.4f}" for k, v in baseline_parts.items())
        )

    reference_results: list[dict] = []
    for ref in args.reference_checkpoint:
        ref_path = Path(ref)
        if not ref_path.exists():
            print(f"WARN: reference checkpoint missing, skipped: {ref_path}")
            continue
        ref_ckpt = torch.load(ref_path, map_location=device, weights_only=False)
        ref_args = copy.copy(args)
        ref_args.bc_checkpoint = str(ref_path)
        ref_args.n_frames = int(ref_ckpt.get("n_frames", args.n_frames))
        ref_actor, _ref_use_belief, _ref_warm_started, _ref_ckpt = load_actor(ref_args, device)
        ref_actor.eval()
        ref_score, ref_parts = evaluate_selection(ref_actor, ref_args, device)
        reference_results.append({
            "path": str(ref_path),
            "score": ref_score,
            "parts": ref_parts,
        })
        if args.min_save_improvement > 0.0:
            save_floor = max(save_floor, ref_score + args.min_save_improvement)
        print(
            f"selection reference {ref_path.name} ({args.selection_manager})={ref_score:.4f} "
            + " ".join(f"{k}={v:.4f}" for k, v in ref_parts.items())
        )
    if reference_results:
        gate_metadata["selection_references"] = reference_results
    gate_metadata["selection_save_floor"] = save_floor
    gate_metadata["min_save_improvement"] = args.min_save_improvement
    print(f"candidate save floor: {save_floor:.4f}")
    random.seed(args.seed or 0)
    np.random.seed(args.seed or 0)
    torch.manual_seed(args.seed or 0)

    best_eval = -float("inf")
    start_time = time.time()

    # Self-play snapshot pool. Workshop notebook 05 explicitly recommends this
    # pattern over training-against-shadow: ``opponent.copy_weights(agent)`` at
    # a fixed interval. We seed the pool with the actor's initial weights so
    # the first few intervals don't fall back to live-actor frozen opponents.
    snapshot_pool: SnapshotPool | None = None
    if _uses_snapshot_opponents(args) and args.snapshot_interval > 0:
        snapshot_pool = SnapshotPool(max_size=args.snapshot_pool_size)
        snapshot_pool.add(actor)
        print(
            f"self-play snapshot pool: size_cap={args.snapshot_pool_size} "
            f"interval={args.snapshot_interval} updates "
            f"(seeded with initial actor)"
        )

    # Elo-rated population (24 May 2026 experiment, arxiv 2407.00662 style).
    # Optional alongside SnapshotPool — when --elo-population is set, frozen
    # snapshot opponents are sampled with Gaussian weight on the live policy's
    # Elo instead of uniformly. Disabled by default so the shipping path /
    # legacy runs are unchanged.
    elo_pop = None
    live_rating = None
    if args.elo_population and snapshot_pool is not None:
        from elo_population import EloPopulation, LiveRating  # noqa: E402, WPS433
        elo_pop = EloPopulation(
            max_size=args.snapshot_pool_size,
            sigma=args.elo_sigma,
            k=args.elo_k,
        )
        live_rating = LiveRating(rating=args.elo_initial)
        # Seed the pool with the initial actor at the same Elo as the live
        # policy so the early matchmaking doesn't pathologically pick only
        # one snapshot.
        elo_pop.add(snapshot=copy.deepcopy(actor).cpu().eval(),
                    initial_elo=live_rating.rating, at_update=0)
        print(
            f"elo-population: size_cap={args.snapshot_pool_size} sigma={args.elo_sigma} "
            f"k={args.elo_k} initial_elo={args.elo_initial} "
            f"baseline_score={args.elo_baseline:.3f}"
        )

    for update in range(1, args.updates + 1):
        progress = (update - 1) / max(args.updates - 1, 1)
        opponent_mode = _current_opponent_mode(args, update)
        clip_coef = _linear(args.clip_coef, args.clip_final_coef or args.clip_coef, progress)
        entropy_coef = _linear(
            args.entropy_coef,
            args.entropy_final_coef if args.entropy_final_coef is not None else args.entropy_coef,
            progress,
        )
        actor.train()
        critic.train()
        transitions, rollout_score, mode_counts = collect_rollouts(
            actor, critic, args, device,
            seed_offset=update * args.games_per_update,
            snapshot_pool=snapshot_pool,
            opponent_mode=opponent_mode,
            elo_pop=elo_pop,
            live_rating=live_rating,
            elo_baseline=args.elo_baseline,
        )
        if not transitions:
            raise SystemExit("No PPO transitions collected; environment likely terminated before our agent acted.")
        stats = ppo_update(actor, critic, transitions, optimizer, args, device, clip_coef, entropy_coef)
        scheduler.step()

        eval_score = float("nan")
        eval_parts: dict[str, float] = {}
        if update == 1 or update % args.eval_every == 0:
            actor.eval()
            critic.eval()
            eval_score, eval_parts = evaluate_selection(actor, args, device)
            if latest_out_path is not None:
                save_policy_checkpoint(
                    actor, critic, args, latest_out_path, update,
                    eval_score, eval_parts, rollout_score, gate_metadata,
                )
            if eval_score >= save_floor and eval_score > best_eval:
                best_eval = eval_score
                save_policy_checkpoint(
                    actor, critic, args, out_path, update,
                    eval_score, eval_parts, rollout_score, gate_metadata,
                )
                print(f"  ✓ best PPO eval {best_eval:.4f} → saved {out_path}")
            elif eval_score > -float("inf"):
                print(
                    f"  candidate gate not met: eval {eval_score:.4f} "
                    f"< save_floor {save_floor:.4f}"
                )

        # Self-play: snapshot the actor at the configured cadence so future
        # rollouts can face this state from the pool. Done AFTER the PPO
        # update so the snapshot reflects the latest weights.
        if snapshot_pool is not None and update % args.snapshot_interval == 0:
            snapshot_pool.add(actor)
            # Mirror promotion into the Elo population at the live Elo.
            # This is the standard population-based self-play recipe: when
            # the current policy is added to the pool, it inherits the
            # live Elo rather than getting a fixed initial rating.
            if elo_pop is not None and live_rating is not None:
                elo_pop.add(
                    snapshot=copy.deepcopy(actor).cpu().eval(),
                    initial_elo=live_rating.rating,
                    at_update=update,
                )

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]
        pool_tag = f" pool={len(snapshot_pool)}" if snapshot_pool is not None else ""
        if elo_pop is not None and live_rating is not None:
            es = elo_pop.stats()
            pool_tag += (
                f" elo_live={live_rating.rating:.0f}"
                f" elo_pool=[{es.get('min', 0):.0f}-{es.get('max', 0):.0f}"
                f", n={es.get('size', 0)}]"
            )
        mode_tag = _format_mode_counts(mode_counts) if mode_counts else opponent_mode
        eval_tag = ""
        if eval_parts:
            eval_tag = " " + " ".join(f"{k}={v:.4f}" for k, v in eval_parts.items())
        print(
            f"update {update:>4}/{args.updates}  "
            f"mode={mode_tag} samples={len(transitions):>5}  rollout={rollout_score:.4f}  "
            f"eval={eval_score:.4f}{eval_tag}  best={best_eval:.4f}  "
            f"pi_loss={stats['policy_loss']:.4f}  v_loss={stats['value_loss']:.4f}  "
            f"entropy={stats['entropy']:.3f}  kl={stats['approx_kl']:.4f} "
            f"clipfrac={stats['clipfrac']:.2f}  clip={clip_coef:.2f} "
            f"entcoef={entropy_coef:.3f}  lr={current_lr:.2e}{pool_tag}  elapsed={elapsed/60:.1f}m"
        )

    print(f"\nBest saved eval score: {best_eval:.4f}")
    if best_eval == -float("inf"):
        print(f"No candidate checkpoint cleared save floor {save_floor:.4f}; leaving {out_path} untouched.")
    else:
        print(f"Checkpoint: {out_path}")
    if latest_out_path is not None:
        print(f"Latest evaluated checkpoint: {latest_out_path}")
    print("Deploy by copying a gated checkpoint to ae/models/bc.pt, then build/test.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["default", "qualifier-best", "full-rl"], default="default",
                        help="Training recipe overlay. qualifier-best enables pressure curriculum; "
                             "full-rl enables per-game opponent mixing on fixed Novice geometry.")
    parser.add_argument("--bc-checkpoint", default="training/ae/checkpoints/bc.pt")
    parser.add_argument("--out", default="training/ae/checkpoints/ppo.pt")
    parser.add_argument("--latest-out", default="",
                        help="Optional checkpoint path for the latest evaluated model. "
                             "The main --out remains gated by selection quality.")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--games-per-update", type=int, default=12)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.20)
    parser.add_argument("--clip-final-coef", type=float, default=None,
                        help="Optional linear schedule target for PPO clip coefficient.")
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--entropy-final-coef", type=float, default=None,
                        help="Optional linear schedule target for entropy coefficient.")
    parser.add_argument("--value-coef", type=float, default=0.50)
    parser.add_argument("--max-grad-norm", type=float, default=0.50)
    parser.add_argument("--target-kl", type=float, default=0.0,
                        help="Stop PPO minibatch epochs early when approximate KL exceeds this value.")
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
    parser.add_argument("--opponents", choices=OPPONENT_MODES, default="mixed")
    parser.add_argument("--eval-opponents", choices=OPPONENT_MODES, default="mixed")
    parser.add_argument("--curriculum", choices=["none", "pressure", "bomberman"], default="none",
                        help="Progressively changes rollout opponent pool across updates.")
    parser.add_argument("--opponent-mix", default="",
                        help="Per-game opponent mix as mode:weight CSV, e.g. "
                             "scripted:0.35,cloudsuite:0.35,pressure2:0.15,league:0.05. "
                             "When set, this overrides --opponents/--curriculum for rollouts.")
    parser.add_argument("--opponent-mix-preset", choices=["none", *OPPONENT_MIX_PRESETS], default="none",
                        help="Named per-game opponent mix. full-rl keeps fixed Novice geometry but "
                             "stratifies each PPO update across scripted/cloudsuite/pressure2/planner/"
                             "aggressive/league opponents.")
    parser.add_argument("--selection-suites", default="",
                        help="Comma-separated opponent pools used for checkpoint selection. "
                        "If empty, eval-opponents is used.")
    parser.add_argument("--selection-manager", choices=["policy", "hybrid"], default="policy",
                        help="Score checkpoints as pure policy or through the deployed hybrid wrapper.")
    parser.add_argument("--selection-device", choices=["train", "cpu"], default="train",
                        help="Run checkpoint selection on the training device or CPU. CPU better matches Docker.")
    parser.add_argument("--selection-fixed-map-shortcut", choices=["on", "off"], default="on",
                        help="For hybrid checkpoint selection, keep or disable the fixed-Novice-map "
                             "heuristic shortcut. Disable it when training/evaluating a candidate "
                             "whose policy must be allowed to affect fixed-map games.")
    parser.add_argument("--selection-weights", default="",
                        help="Comma-separated weights for --selection-suites; defaults to uniform.")
    parser.add_argument("--selection-games", type=int, default=6,
                        help="Games per selection suite when --selection-suites is set.")
    parser.add_argument("--baseline-eval", action="store_true",
                        help="Evaluate the starting actor before training and use it as a save gate.")
    parser.add_argument("--reference-checkpoint", action="append", default=[],
                        help="Optional checkpoint to evaluate as an additional save-floor reference. "
                             "May be repeated.")
    parser.add_argument("--min-save-improvement", type=float, default=0.0,
                        help="Require candidate eval to beat baseline/reference evals by this margin "
                             "before writing --out.")
    parser.add_argument("--selection-min-score", type=float, default=-float("inf"),
                        help="Absolute minimum selection score required before writing --out.")
    parser.add_argument("--snapshot-interval", type=int, default=10,
                        help="Add a frozen actor snapshot to the self-play pool every N PPO updates "
                             "(set to 0 to disable; falls back to live-actor frozen opponents).")
    parser.add_argument("--snapshot-pool-size", type=int, default=5,
                        help="Max historical snapshots kept in the self-play pool (FIFO).")
    # Elo population self-play (24 May 2026 experiment). When set, frozen
    # snapshot opponents are sampled with Gaussian-weight matchmaking on the
    # live policy's Elo, rather than uniformly. Snapshot pool capacity comes
    # from --snapshot-pool-size; promotion cadence from --snapshot-interval.
    parser.add_argument("--elo-population", action="store_true",
                        help="Enable Elo-rated snapshot population matchmaking. "
                             "Requires snapshot opponents (league/selfplay/mixed/frozen).")
    parser.add_argument("--elo-sigma", type=float, default=200.0,
                        help="Gaussian sigma for matchmaking weight on |snapshot_elo - live_elo|. "
                             "Smaller = stricter skill matching; larger = closer to uniform.")
    parser.add_argument("--elo-k", type=float, default=32.0,
                        help="Chess-standard Elo K-factor for per-game rating updates.")
    parser.add_argument("--elo-initial", type=float, default=1200.0,
                        help="Initial Elo for the live policy and the seed snapshot.")
    parser.add_argument("--elo-baseline", type=float, default=0.55,
                        help="Normalized score that maps to outcome=0.5 (neutral). "
                             "Scores above push live Elo up; below push it down. "
                             "0.55 ~ current heuristic baseline aggregate.")
    parser.add_argument("--use-belief", action="store_true",
                        help="Train with the belief-map architecture (16x16xK extra CNN branch). "
                             "Overridden by the BC checkpoint's use_belief flag if loading one.")
    parser.add_argument("--load-critic", action="store_true",
                        help="When the checkpoint contains critic_state_dict, warm-start the value network too.")
    parser.add_argument("--orthogonal-init", action="store_true",
                        help="Use PPO-style orthogonal init for scratch actor and critic.")
    parser.add_argument("--explore-bonus", type=float, default=0.0,
                        help="Training-only visit-count exploration bonus weight.")
    parser.add_argument("--explore-min-scale", type=float, default=0.1,
                        help="Minimum fraction of exploration bonus after adaptive annealing.")
    parser.add_argument("--explore-anneal-k", type=float, default=1.2,
                        help="Controls how quickly exploration bonus decays after high-reward episodes.")
    parser.add_argument("--explore-success-reward", type=float, default=20.0,
                        help="Raw episode reward threshold counted as success for exploration annealing.")
    parser.add_argument("--health-delta-coef", type=float, default=0.0,
                        help="Training-only coefficient for health delta shaping.")
    parser.add_argument("--base-health-delta-coef", type=float, default=0.0,
                        help="Training-only coefficient for base-health delta shaping.")
    parser.add_argument("--bomb-action-bonus", type=float, default=0.0,
                        help="Small training-only bonus for PLACE_BOMB to fight bomb timidity.")
    parser.add_argument("--novice", action="store_true", default=True)
    parser.add_argument("--no-novice", dest="novice", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
