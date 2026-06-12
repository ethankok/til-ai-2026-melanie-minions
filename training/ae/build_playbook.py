"""Aggregate simulation trajectories into a state→action playbook.

Reads one or more `.npz` files produced by `simulate.py` and writes a single
compact `.npz` whose keys describe the best-known action at each (location,
direction, step) bucket on the fixed novice map.

Why this exists. Tier 1 #1: novice has a fixed seed for both walls AND
entity scatter. Every game starts from the same initial state, so the
"optimal action" for a given (where I am, which way I face, what tick
it is) is well defined as `argmax_a E[future_reward | s, a]`. We
approximate that expectation with Monte Carlo over thousands of
simulated rounds.

Algorithm.
  1. For each (state_key, action) pair, compute the empirical mean of
     the round score for rounds where that pair was used.
  2. For each state_key, the playbook action is the action with the
     highest mean.
  3. Coverage filter: only keep state_keys that have been visited at
     least `--min-visits` times AND whose top action's mean beats the
     fallback (heuristic) by at least `--min-margin` reward/round.

The output is keyed by ``(x, y, direction, step)`` so that:
  * Inference is a single dict lookup at each tick.
  * Storage is small (16*16*4*200 = 200K cells max, in practice much less
    because not every cell is reached).

Output `.npz` format:
    keys      (M,)  int64 packed state keys
    actions   (M,)  int8  best action 0..5
    n_visits  (M,)  int32
    mean_value (M,) float32 mean round score for the chosen action

Usage:

    python training/ae/build_playbook.py \
        --inputs training/ae/data/sim-random.npz training/ae/data/sim-library.npz \
        --out ae/models/playbook.npz \
        --min-visits 5 \
        --min-margin 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_AE = REPO_ROOT / "training" / "ae"
if str(TRAINING_AE) not in sys.path:
    sys.path.insert(0, str(TRAINING_AE))

from simulate import unpack_state_key  # noqa: E402


def aggregate(
    state_keys: np.ndarray,
    actions: np.ndarray,
    round_scores: np.ndarray,
) -> dict[int, dict[int, list[float]]]:
    """Group round scores by (state_key, action).

    Every visit of (s, a) in a round bags that round's *whole-game score*
    as its return target. Multiple visits of (s, a) in the same round each
    receive that round's score (so a state visited twice in a 0.7-score
    round contributes 0.7 twice). This is biased toward high-traffic
    states, which is exactly what we want for the playbook: high-traffic
    states are the ones whose chosen action mattered most.
    """
    table: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for k, a, score in zip(state_keys, actions, round_scores):
        table[int(k)][int(a)].append(float(score))
    return table


def select_best_actions(
    table: dict[int, dict[int, list[float]]],
    fallback_value: float,
    min_visits: int,
    min_margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pick the best action per state_key.

    Filters out state_keys with too few visits or whose best action does
    not beat the global fallback by ``min_margin``. The fallback represents
    the heuristic's mean score — playbook entries should only override
    the heuristic when there's a meaningfully better alternative.
    """
    keys = []
    actions = []
    visits = []
    values = []
    for key, by_action in table.items():
        total = sum(len(v) for v in by_action.values())
        if total < min_visits:
            continue
        # Skip state_keys where only one action was ever taken — the
        # comparison would be meaningless.
        if len(by_action) < 2:
            continue
        means = {a: float(np.mean(v)) for a, v in by_action.items() if len(v) >= 2}
        if not means:
            continue
        best_a = max(means, key=means.get)
        best_v = means[best_a]
        if best_v - fallback_value < min_margin:
            continue
        keys.append(key)
        actions.append(best_a)
        visits.append(total)
        values.append(best_v)

    if not keys:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int8),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float32),
        )

    return (
        np.asarray(keys, dtype=np.int64),
        np.asarray(actions, dtype=np.int8),
        np.asarray(visits, dtype=np.int32),
        np.asarray(values, dtype=np.float32),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", type=Path, nargs="+", required=True,
                   help="one or more .npz trajectory files from simulate.py")
    p.add_argument("--out", type=Path, default=Path("ae/models/playbook.npz"),
                   help="where to save the playbook")
    p.add_argument("--min-visits", type=int, default=5,
                   help="state_key must be visited at least this many times")
    p.add_argument("--min-margin", type=float, default=0.05,
                   help="best action must beat fallback by at least this score")
    p.add_argument("--fallback", type=float, default=None,
                   help=("fallback (heuristic) mean score; defaults to the "
                         "10th-percentile round score across input files"))
    p.add_argument("--report", type=Path, default=None,
                   help="optional JSON report path for coverage stats")
    args = p.parse_args(argv)

    all_keys: list[int] = []
    all_actions: list[int] = []
    all_round_scores: list[float] = []

    for path in args.inputs:
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path, allow_pickle=True)
        n = len(data["state_keys"])
        print(f"  {path.name}: {n} steps")
        all_keys.append(data["state_keys"])
        all_actions.append(data["actions"])
        all_round_scores.append(
            data["round_score"].astype(np.float32) / 1000.0
            if "round_score" in data.files
            else np.zeros(n, dtype=np.float32)
        )

    state_keys = np.concatenate(all_keys)
    actions = np.concatenate(all_actions)
    round_scores = np.concatenate(all_round_scores)
    print(f"total steps logged: {len(state_keys)}")

    if args.fallback is None:
        # Global mean round score = "what the heuristic does on average";
        # the playbook should only fire when it beats this.
        fallback = float(np.mean(round_scores))
        print(f"auto fallback baseline = {fallback:.4f} (mean round score)")
    else:
        fallback = float(args.fallback)
        print(f"fallback baseline      = {fallback:.4f} (--fallback)")

    table = aggregate(state_keys, actions, round_scores)
    print(f"unique state_keys: {len(table)}")

    keys, picked, visits, values = select_best_actions(
        table,
        fallback_value=fallback,
        min_visits=args.min_visits,
        min_margin=args.min_margin,
    )
    print(f"playbook entries: {len(keys)}  "
          f"(min_visits={args.min_visits} min_margin={args.min_margin})")

    if len(keys):
        print(f"  visit count  : min={visits.min()} mean={visits.mean():.1f} max={visits.max()}")
        print(f"  picked value : min={values.min():.3f} mean={values.mean():.3f} max={values.max():.3f}")

        action_counts = np.bincount(picked, minlength=6)
        print("  action mix   : "
              f"FORWARD={action_counts[0]} BACKWARD={action_counts[1]} "
              f"LEFT={action_counts[2]} RIGHT={action_counts[3]} "
              f"STAY={action_counts[4]} BOMB={action_counts[5]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        keys=keys,
        actions=picked,
        n_visits=visits,
        mean_value=values,
        fallback_baseline=np.float32(fallback),
        min_visits=np.int32(args.min_visits),
        min_margin=np.float32(args.min_margin),
    )
    print(f"playbook -> {args.out}  ({len(keys)} entries)")

    if args.report is not None:
        report = {
            "inputs": [str(p) for p in args.inputs],
            "total_steps": int(len(state_keys)),
            "unique_state_keys": int(len(table)),
            "fallback_baseline": fallback,
            "min_visits": args.min_visits,
            "min_margin": args.min_margin,
            "playbook_entries": int(len(keys)),
            "action_mix": {
                "FORWARD": int(np.sum(picked == 0)),
                "BACKWARD": int(np.sum(picked == 1)),
                "LEFT": int(np.sum(picked == 2)),
                "RIGHT": int(np.sum(picked == 3)),
                "STAY": int(np.sum(picked == 4)),
                "PLACE_BOMB": int(np.sum(picked == 5)),
            },
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"report -> {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
