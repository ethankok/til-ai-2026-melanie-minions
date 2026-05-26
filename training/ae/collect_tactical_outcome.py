"""Collect outcome-weighted tactical-option training data.

For each game seed, this runs:
  1. the protected heuristic baseline;
  2. a tactical-exploration agent on the same suite/seed.

The saved examples are the tactical choices from the exploration run. Their
weights come from final-score advantage over the heuristic baseline, with
extra down-weighting for decisions followed by base-health loss. This is the
"terminal reward, but learnable" version: final outcome is primary, while
nearby base damage gives credit assignment enough structure to train a
classifier before attempting PPO.
"""

from __future__ import annotations

# Pin PYTHONHASHSEED=0 so the labels we collect are reproducible. AEManager
# and TacticalExecutor both have hash-order-dependent branches in candidate
# scoring; unpinned hash seeds make repeated collection runs produce different
# labels for the same (suite, seed, game) triple.
import os
import sys

if os.environ.get("PYTHONHASHSEED") is None:
    os.environ["PYTHONHASHSEED"] = "0"
    os.execvp(sys.executable, [sys.executable, *sys.argv])

import argparse
from dataclasses import dataclass
from collections import Counter, defaultdict
import random
import time
from pathlib import Path

import numpy as np
from tqdm import trange

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
AE_SRC = REPO_ROOT / "ae" / "src"
for path in (THIS_DIR, AE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ae_manager import AEManager  # noqa: E402
from encoder import FrameStacker, rasterize_belief  # noqa: E402
from opponents import make_opponent, resolve_opponent_spec  # noqa: E402
from tactical_hybrid_manager import TacticalExecutor  # noqa: E402
from tactical_policy import (  # noqa: E402
    DEFENSIVE_TACTICAL_OPTIONS,
    NUM_TACTICAL_OPTIONS,
    TACTICAL_BOMB_BASE_THREAT,
    TACTICAL_COLLECT_MISSION_SAFE,
    TACTICAL_COUNTER_RUSH,
    TACTICAL_GUARD_BASE_LANE,
    TACTICAL_HUNT_VISIBLE_ENEMY,
    TACTICAL_INTERCEPT_BASE_THREAT,
    TACTICAL_OPTION_NAMES,
    TACTICAL_RUSH_ENEMY_BASE,
    TACTICAL_STALL_WHEN_WINNING,
    tactical_from_manager,
)
from til_environment import bomberman_env  # noqa: E402
from til_environment.config import default_config  # noqa: E402


DEFAULT_SUITES = [
    "base_rush_exploit",
    "top_seed_proxy",
    "defense_trap",
    "bracket_proxy",
    "base_rush_exploit",
    "top_seed_proxy",
    "strong_realistic",
    "cloudsuite",
    "pressure2",
]


DISTANCE_BUCKET_NEAR = 0
DISTANCE_BUCKET_MID = 1
DISTANCE_BUCKET_FAR = 2
DISTANCE_BUCKET_UNKNOWN = 3
NUM_DISTANCE_BUCKETS = 3
DISTANCE_BUCKET_NAMES = ("near", "mid", "far")


def _location_tuple(value) -> tuple[int, int] | None:
    if value is None:
        return None
    try:
        return (int(value[0]), int(value[1]))
    except Exception:
        return None


def _distance_bucket(obs_py: dict, heuristic_manager) -> int:
    """Mirror tactical_hybrid_manager._distance_bucket (near/mid/far)."""
    location = _location_tuple(obs_py.get("location"))
    base = (
        getattr(heuristic_manager, "base_location", None)
        or _location_tuple(obs_py.get("base_location"))
    )
    if location is None or base is None:
        return DISTANCE_BUCKET_UNKNOWN
    dist = abs(location[0] - base[0]) + abs(location[1] - base[1])
    if dist <= 4:
        return DISTANCE_BUCKET_NEAR
    if dist <= 8:
        return DISTANCE_BUCKET_MID
    return DISTANCE_BUCKET_FAR


@dataclass
class TacticalExample:
    agent_view: np.ndarray
    base_view: np.ndarray
    scalars: np.ndarray
    belief: np.ndarray | None
    option: int
    prior_option: int
    action: int
    suite: str
    step: int
    base_before: float
    health_before: float
    distance_bucket: int = DISTANCE_BUCKET_UNKNOWN
    base_after: float | None = None
    health_after: float | None = None


def _obs_to_python(obs) -> dict:
    out = {}
    for key, value in obs.items():
        if hasattr(value, "tolist"):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def _safe_float(value, default: float = 0.0) -> float:
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


def _has_base_threat(executor: TacticalExecutor) -> bool:
    try:
        return bool(executor._fresh_base_threats())
    except Exception:
        return False


def _sample_tactical_option(
    prior: int,
    executor: TacticalExecutor,
    rng: random.Random,
    epsilon: float,
    defense_bias: float,
) -> int:
    if rng.random() >= epsilon:
        return int(prior)
    candidates = list(range(NUM_TACTICAL_OPTIONS))
    if _has_base_threat(executor) and rng.random() < defense_bias:
        candidates = [
            TACTICAL_INTERCEPT_BASE_THREAT,
            TACTICAL_GUARD_BASE_LANE,
            TACTICAL_BOMB_BASE_THREAT,
            TACTICAL_COUNTER_RUSH,
            TACTICAL_STALL_WHEN_WINNING,
        ]
    elif prior in DEFENSIVE_TACTICAL_OPTIONS and rng.random() < 0.5:
        candidates = [
            TACTICAL_INTERCEPT_BASE_THREAT,
            TACTICAL_GUARD_BASE_LANE,
            TACTICAL_BOMB_BASE_THREAT,
            TACTICAL_COUNTER_RUSH,
        ]
    else:
        candidates = [
            TACTICAL_RUSH_ENEMY_BASE,
            TACTICAL_COLLECT_MISSION_SAFE,
            TACTICAL_HUNT_VISIBLE_ENEMY,
            TACTICAL_INTERCEPT_BASE_THREAT,
            TACTICAL_GUARD_BASE_LANE,
        ]
    return int(rng.choice(candidates))


def _run_baseline_episode(
    env,
    suite: str,
    seed: int,
    args: argparse.Namespace,
) -> tuple[float, list[TacticalExample]]:
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
    stacker = FrameStacker(args.n_frames)
    examples: list[TacticalExample] = []
    pending_idx: int | None = None
    total = 0.0
    for agent in env.agent_iter():
        obs, reward, termination, truncation, _info = env.last()
        if agent == our_agent:
            total += float(reward)
            if pending_idx is not None:
                obs_tmp = _obs_to_python(obs)
                examples[pending_idx].base_after = _safe_float(obs_tmp.get("base_health"), examples[pending_idx].base_before)
                examples[pending_idx].health_after = _safe_float(obs_tmp.get("health"), examples[pending_idx].health_before)
                pending_idx = None
        if termination or truncation:
            env.step(None)
            continue
        obs_py = _obs_to_python(obs)
        if agent == our_agent:
            if obs_py.get("step") == 0:
                planner = AEManager()
                stacker.reset()
            action = int(planner.ae(obs_py))
            option = int(tactical_from_manager(planner, action=action))
            belief = rasterize_belief(planner, obs_py) if args.with_belief else None
            stacked = stacker.observe(obs_py, belief_map=belief)
            examples.append(TacticalExample(
                agent_view=stacked["agent_view"],
                base_view=stacked["base_view"],
                scalars=stacked["scalars"],
                belief=belief,
                option=option,
                prior_option=option,
                action=action,
                suite=suite,
                step=int(_safe_float(obs_py.get("step"), 0.0)),
                base_before=_safe_float(obs_py.get("base_health"), 100.0),
                health_before=_safe_float(obs_py.get("health"), 60.0),
                distance_bucket=_distance_bucket(obs_py, planner),
            ))
            pending_idx = len(examples) - 1
        else:
            slot = other_ids.index(agent)
            action = int(opponents[slot](obs_py))
            action = _legalize_action(env, agent, obs_py, action)
        env.step(action)
    if pending_idx is not None:
        examples[pending_idx].base_after = examples[pending_idx].base_before
        examples[pending_idx].health_after = examples[pending_idx].health_before
    return total / 1000.0, examples


def _run_tactical_episode(
    env,
    suite: str,
    seed: int,
    rng: random.Random,
    args: argparse.Namespace,
) -> tuple[float, list[TacticalExample]]:
    env.reset(seed=seed)
    our_agent = env.possible_agents[0]
    other_ids = list(env.possible_agents[1:])
    opponents = _make_opponents(suite, seed)
    for op in opponents:
        if hasattr(op, "reset_for_game"):
            op.reset_for_game()
        if hasattr(op, "_reset_memory"):
            op._reset_memory()
    executor = TacticalExecutor()
    stacker = FrameStacker(args.n_frames)
    examples: list[TacticalExample] = []
    pending_idx: int | None = None
    total = 0.0

    for agent in env.agent_iter():
        obs, reward, termination, truncation, _info = env.last()
        if agent == our_agent:
            total += float(reward)
            if pending_idx is not None:
                obs_tmp = _obs_to_python(obs)
                examples[pending_idx].base_after = _safe_float(obs_tmp.get("base_health"), examples[pending_idx].base_before)
                examples[pending_idx].health_after = _safe_float(obs_tmp.get("health"), examples[pending_idx].health_before)
                pending_idx = None
        if termination or truncation:
            env.step(None)
            continue

        obs_py = _obs_to_python(obs)
        if agent == our_agent:
            if obs_py.get("step") == 0:
                executor = TacticalExecutor()
                stacker.reset()
            heuristic_action = int(executor.heuristic.ae(obs_py))
            prior = int(tactical_from_manager(executor.heuristic, action=heuristic_action))
            option = _sample_tactical_option(
                prior,
                executor,
                rng,
                epsilon=args.epsilon,
                defense_bias=args.defense_bias,
            )
            belief = rasterize_belief(executor.heuristic, obs_py) if args.with_belief else None
            stacked = stacker.observe(obs_py, belief_map=belief)
            action, _info = executor.act_with_info(
                option,
                obs_py,
                heuristic_action=heuristic_action,
                heuristic_already_run=True,
            )
            examples.append(TacticalExample(
                agent_view=stacked["agent_view"],
                base_view=stacked["base_view"],
                scalars=stacked["scalars"],
                belief=belief,
                option=option,
                prior_option=prior,
                action=int(action),
                suite=suite,
                step=int(_safe_float(obs_py.get("step"), 0.0)),
                base_before=_safe_float(obs_py.get("base_health"), 100.0),
                health_before=_safe_float(obs_py.get("health"), 60.0),
                distance_bucket=_distance_bucket(obs_py, executor.heuristic),
            ))
            pending_idx = len(examples) - 1
        else:
            slot = other_ids.index(agent)
            action = int(opponents[slot](obs_py))
            action = _legalize_action(env, agent, obs_py, action)
        env.step(action)

    if pending_idx is not None:
        examples[pending_idx].base_after = examples[pending_idx].base_before
        examples[pending_idx].health_after = examples[pending_idx].health_before
    return total / 1000.0, examples


def _example_weight(
    ex: TacticalExample,
    delta: float,
    args: argparse.Namespace,
) -> float:
    if delta <= args.min_positive_delta:
        return float(args.negative_exploration_weight)
    weight = float(np.exp(delta / max(args.advantage_temperature, 1e-6)))
    weight = float(np.clip(weight, args.min_weight, args.max_weight))
    base_after = ex.base_after if ex.base_after is not None else ex.base_before
    health_after = ex.health_after if ex.health_after is not None else ex.health_before
    base_drop = max(0.0, ex.base_before - base_after)
    health_drop = max(0.0, ex.health_before - health_after)
    if base_drop > 0.0:
        weight *= 0.75 if ex.option in DEFENSIVE_TACTICAL_OPTIONS and delta > 0 else 0.25
    elif ex.base_before <= args.base_pressure_health and ex.option in DEFENSIVE_TACTICAL_OPTIONS:
        weight *= 1.25
    if health_drop > 0.0 and ex.option not in DEFENSIVE_TACTICAL_OPTIONS:
        weight *= 0.6
    if ex.option == ex.prior_option and delta > 0:
        weight *= 1.1
    return float(np.clip(weight, args.min_weight, args.max_weight))


def _append_example(
    ex: TacticalExample,
    weight: float,
    advantage: float,
    final_score: float,
    baseline_score: float,
    is_baseline: bool,
    args: argparse.Namespace,
    buffers: dict[str, list],
) -> None:
    buffers["agent_views"].append(ex.agent_view)
    buffers["base_views"].append(ex.base_view)
    buffers["scalars"].append(ex.scalars)
    if args.with_belief and ex.belief is not None:
        buffers["beliefs"].append(ex.belief)
    buffers["options"].append(ex.option)
    buffers["prior_options"].append(ex.prior_option)
    buffers["actions"].append(ex.action)
    buffers["weights"].append(float(weight))
    buffers["advantages"].append(float(advantage))
    buffers["final_scores"].append(float(final_score))
    buffers["baseline_scores"].append(float(baseline_score))
    buffers["suites"].append(ex.suite)
    buffers["is_baseline"].append(1 if is_baseline else 0)
    buffers["distance_buckets"].append(int(ex.distance_bucket))


def collect_dataset(args: argparse.Namespace) -> None:
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    env = _make_env(novice=args.novice)

    buffers: dict[str, list] = {
        "agent_views": [],
        "base_views": [],
        "scalars": [],
        "beliefs": [],
        "options": [],
        "prior_options": [],
        "actions": [],
        "weights": [],
        "advantages": [],
        "final_scores": [],
        "baseline_scores": [],
        "suites": [],
        "is_baseline": [],
        "distance_buckets": [],
    }

    rng = random.Random(args.seed)
    start = time.time()
    suite_episode_counts: Counter[str] = Counter()
    suite_positive_counts: Counter[str] = Counter()
    suite_delta_sum: defaultdict[str, float] = defaultdict(float)
    # W1.1 harm-aware accumulators. The collector already tracked positive
    # transitions only; here we add the denominator (attempted) plus negatives
    # and signed net-delta so the inference gate can require:
    #   positive_rate = positive / attempted >= T1
    #   mean_net_delta = net_delta_sum / attempted >= T2
    #   attempted >= T3
    # per (prior_option, option) and optionally per distance bucket.
    transition_shape = (NUM_TACTICAL_OPTIONS, NUM_TACTICAL_OPTIONS)
    bucket_shape = (NUM_TACTICAL_OPTIONS, NUM_TACTICAL_OPTIONS, NUM_DISTANCE_BUCKETS)
    attempted_transition_counts = np.zeros(transition_shape, dtype=np.int64)
    positive_transition_counts_mat = np.zeros(transition_shape, dtype=np.int64)
    negative_transition_counts = np.zeros(transition_shape, dtype=np.int64)
    transition_net_delta_sum = np.zeros(transition_shape, dtype=np.float64)
    transition_weight_sum = np.zeros(transition_shape, dtype=np.float64)
    transition_weighted_delta_sum = np.zeros(transition_shape, dtype=np.float64)
    bucket_attempted = np.zeros(bucket_shape, dtype=np.int64)
    bucket_positive = np.zeros(bucket_shape, dtype=np.int64)
    bucket_net_delta_sum = np.zeros(bucket_shape, dtype=np.float64)
    # Legacy single-key counter kept for the existing console summary.
    positive_transition_counts: Counter[tuple[int, int]] = Counter()
    print(
        f"Collecting tactical outcome data; games={args.games}; n_frames={args.n_frames}; "
        f"with_belief={args.with_belief}; epsilon={args.epsilon}; suite_cycle={args.suite_cycle}"
    )
    for game in trange(args.games, desc="games"):
        suite = args.suite_cycle[game % len(args.suite_cycle)]
        seed = args.seed + game
        baseline_score, baseline_examples = _run_baseline_episode(env, suite, seed, args)
        tactical_score, examples = _run_tactical_episode(env, suite, seed, rng, args)
        delta = tactical_score - baseline_score
        suite_episode_counts[suite] += 1
        suite_delta_sum[suite] += float(delta)
        if delta > args.min_positive_delta:
            suite_positive_counts[suite] += 1
        for ex in examples:
            weight = _example_weight(ex, delta, args)
            # Harm-aware accounting: log EVERY explored transition, including
            # those with weight==0 (negative-delta episodes) and same-as-prior
            # picks. The training dataset still drops weight==0 examples, but
            # the gate denominator needs to see them.
            attempted_transition_counts[ex.prior_option, ex.option] += 1
            transition_net_delta_sum[ex.prior_option, ex.option] += float(delta)
            transition_weight_sum[ex.prior_option, ex.option] += float(max(weight, 0.0))
            transition_weighted_delta_sum[ex.prior_option, ex.option] += float(delta) * float(max(weight, 0.0))
            bucket = int(ex.distance_bucket)
            if 0 <= bucket < NUM_DISTANCE_BUCKETS:
                bucket_attempted[ex.prior_option, ex.option, bucket] += 1
                bucket_net_delta_sum[ex.prior_option, ex.option, bucket] += float(delta)
            if delta > args.min_positive_delta:
                positive_transition_counts_mat[ex.prior_option, ex.option] += 1
                if 0 <= bucket < NUM_DISTANCE_BUCKETS:
                    bucket_positive[ex.prior_option, ex.option, bucket] += 1
                if ex.option != ex.prior_option:
                    positive_transition_counts[(ex.prior_option, ex.option)] += 1
            else:
                negative_transition_counts[ex.prior_option, ex.option] += 1
            if weight <= 0.0:
                continue
            _append_example(
                ex,
                weight=weight,
                advantage=delta,
                final_score=tactical_score,
                baseline_score=baseline_score,
                is_baseline=False,
                args=args,
                buffers=buffers,
            )
        anchor_weight = args.baseline_anchor_weight * (1.0 if delta <= 0 else args.positive_baseline_anchor_scale)
        for ex in baseline_examples:
            _append_example(
                ex,
                weight=anchor_weight,
                advantage=0.0,
                final_score=baseline_score,
                baseline_score=baseline_score,
                is_baseline=True,
                args=args,
                buffers=buffers,
            )

    env.close()
    if not buffers["options"]:
        raise SystemExit("No tactical examples collected.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {
        "agent_views": np.stack(buffers["agent_views"]).astype(np.float32),
        "base_views": np.stack(buffers["base_views"]).astype(np.float32),
        "scalars": np.stack(buffers["scalars"]).astype(np.float32),
        "options": np.asarray(buffers["options"], dtype=np.int64),
        "prior_options": np.asarray(buffers["prior_options"], dtype=np.int64),
        "actions": np.asarray(buffers["actions"], dtype=np.int64),
        "weights": np.asarray(buffers["weights"], dtype=np.float32),
        "advantages": np.asarray(buffers["advantages"], dtype=np.float32),
        "final_scores": np.asarray(buffers["final_scores"], dtype=np.float32),
        "baseline_scores": np.asarray(buffers["baseline_scores"], dtype=np.float32),
        "suites": np.asarray(buffers["suites"]),
        "is_baseline": np.asarray(buffers["is_baseline"], dtype=np.int8),
        "distance_buckets": np.asarray(buffers["distance_buckets"], dtype=np.int8),
        "option_names": np.asarray(TACTICAL_OPTION_NAMES),
        "distance_bucket_names": np.asarray(DISTANCE_BUCKET_NAMES),
        "n_frames": np.asarray(args.n_frames, dtype=np.int32),
        "with_belief": np.asarray(int(args.with_belief), dtype=np.int32),
        # W1.1 harm-aware accumulators. These count every explored transition,
        # not just kept-weight samples, so the inference gate denominator is
        # honest. See tactical_hybrid_manager._delta_is_supported.
        "attempted_transition_counts": attempted_transition_counts.astype(np.int64),
        "positive_transition_counts": positive_transition_counts_mat.astype(np.int64),
        "negative_transition_counts": negative_transition_counts.astype(np.int64),
        "transition_net_delta_sum": transition_net_delta_sum.astype(np.float64),
        "transition_weight_sum": transition_weight_sum.astype(np.float64),
        "transition_weighted_delta_sum": transition_weighted_delta_sum.astype(np.float64),
        "bucket_attempted": bucket_attempted.astype(np.int64),
        "bucket_positive": bucket_positive.astype(np.int64),
        "bucket_net_delta_sum": bucket_net_delta_sum.astype(np.float64),
    }
    if args.with_belief:
        save_kwargs["beliefs"] = np.stack(buffers["beliefs"]).astype(np.float32)
    np.savez_compressed(out_path, **save_kwargs)

    elapsed = time.time() - start
    options_arr = np.asarray(buffers["options"], dtype=np.int64)
    weights_arr = np.asarray(buffers["weights"], dtype=np.float64)
    advantages_arr = np.asarray(buffers["advantages"], dtype=np.float64)
    counts = np.bincount(options_arr, minlength=NUM_TACTICAL_OPTIONS)
    weighted_counts = np.bincount(
        options_arr,
        weights=weights_arr,
        minlength=NUM_TACTICAL_OPTIONS,
    )
    print(
        f"Collected {len(options_arr):,} tactical samples across {args.games} games in {elapsed:.1f}s "
        f"({len(options_arr) / max(elapsed, 1.0):.1f} samples/sec)"
    )
    print(f"Saved -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(
        f"sample score_delta: mean={float(np.mean(advantages_arr)):.4f} "
        f"p50={float(np.median(advantages_arr)):.4f} min={float(np.min(advantages_arr)):.4f} "
        f"max={float(np.max(advantages_arr)):.4f}"
    )
    print(
        f"sample weights: mean={float(np.mean(weights_arr)):.3f} "
        f"p90={float(np.percentile(weights_arr, 90)):.3f} max={float(np.max(weights_arr)):.3f}; "
        f"baseline_anchor_samples={int(np.sum(save_kwargs['is_baseline']))}"
    )
    print("Tactical option distribution:")
    for i, name in enumerate(TACTICAL_OPTION_NAMES):
        print(f"  {name:24s} {int(counts[i]):>7d}  weighted={float(weighted_counts[i]):>9.1f}")
    print("Episode deltas by suite:")
    for suite in sorted(suite_episode_counts):
        games = int(suite_episode_counts[suite])
        positives = int(suite_positive_counts[suite])
        mean_delta = suite_delta_sum[suite] / max(games, 1)
        print(f"  {suite:24s} games={games:>4d} positive={positives:>4d} mean_delta={mean_delta:+.4f}")
    print("Positive intervention transitions (delta>0, option!=prior):")
    if positive_transition_counts:
        for (prior, option), count in positive_transition_counts.most_common():
            print(f"  {TACTICAL_OPTION_NAMES[prior]:24s} -> {TACTICAL_OPTION_NAMES[option]:24s} {count:>7d}")
    else:
        print("  none")

    print("Harm-aware per-transition stats (attempted, positive, neg, pos_rate, mean_net_delta):")
    any_row = False
    for prior in range(NUM_TACTICAL_OPTIONS):
        for option in range(NUM_TACTICAL_OPTIONS):
            attempted = int(attempted_transition_counts[prior, option])
            if attempted <= 0:
                continue
            positive = int(positive_transition_counts_mat[prior, option])
            negative = int(negative_transition_counts[prior, option])
            pos_rate = positive / attempted
            mean_delta = float(transition_net_delta_sum[prior, option]) / max(attempted, 1)
            print(
                f"  {TACTICAL_OPTION_NAMES[prior]:24s} -> {TACTICAL_OPTION_NAMES[option]:24s} "
                f"att={attempted:>5d} pos={positive:>5d} neg={negative:>5d} "
                f"pos_rate={pos_rate:.3f} mean_delta={mean_delta:+.4f}"
            )
            any_row = True
    if not any_row:
        print("  none")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--out", default="training/ae/data/tactical_outcome.npz")
    parser.add_argument("--suite-cycle", nargs="+", default=DEFAULT_SUITES)
    parser.add_argument("--novice", action="store_true", default=True)
    parser.add_argument("--no-novice", dest="novice", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-frames", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=0.35)
    parser.add_argument("--defense-bias", type=float, default=0.80)
    parser.add_argument("--advantage-temperature", type=float, default=0.07)
    parser.add_argument("--min-positive-delta", type=float, default=0.0)
    parser.add_argument("--negative-exploration-weight", type=float, default=0.0,
                        help="0 drops exploration episodes that fail to beat same-seed heuristic")
    parser.add_argument("--min-weight", type=float, default=0.05)
    parser.add_argument("--max-weight", type=float, default=8.0)
    parser.add_argument("--base-pressure-health", type=float, default=70.0)
    parser.add_argument("--baseline-anchor-weight", type=float, default=0.75)
    parser.add_argument("--positive-baseline-anchor-scale", type=float, default=0.25)
    parser.add_argument("--no-belief", dest="with_belief", action="store_false")
    parser.set_defaults(with_belief=True)
    collect_dataset(parser.parse_args())


if __name__ == "__main__":
    main()
