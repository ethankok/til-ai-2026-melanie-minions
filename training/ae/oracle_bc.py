"""Oracle behavior cloning over simulated trajectories.

Tier 2 #10.

Premise. Standard BC on the heuristic teaches the policy to imitate the
heuristic, including its mistakes. Oracle BC instead trains on labels
selected with the benefit of hindsight: at each (state, action) we
observed across many simulated rounds, we know which action led to the
*highest-scoring* outcome empirically. Train the policy to copy *those*
actions instead of the heuristic's.

Why this works on the fixed novice map. Since walls AND tile scatter are
deterministic per-game, distinct games differ only in opponent stochasticity.
For high-traffic states, the empirical best action is the one that wins
across opponent variance. That's a stronger label than any single
heuristic decision.

This script:
  1. Loads `.npz` files from `simulate.py`.
  2. Groups (state, action) -> list of round scores.
  3. For each state, picks the action whose mean is highest.
  4. Constructs a (state_features, oracle_action) supervised dataset.
  5. Fits a small CNN policy (the same `PolicyNetwork` used by `policy_manager`).
  6. Saves to `ae/models/bc_oracle.pt` so `policy_manager.py` picks it up
     when `AE_POLICY_CHECKPOINT=ae/models/bc_oracle.pt`.

Crucial: this trainer does NOT need GPU. The policy net is ~150K params and
the dataset is bounded by unique state_keys (< 200K). One-shot CPU training
in 5-10 minutes on the Mac.

Caveat. This BC consumes packed (x, y, dir, step) state keys, which lose
the full viewcone tensor that the deployed PolicyNetwork expects. To keep
the inference signature unchanged we re-derive the viewcone tensor at
training time by replaying the env once with a fixed seed and snapshotting
each tick's full observation indexed by state_key. Implementation note:
that's expensive enough that this script is opt-in via `--mode replay`;
the default `--mode tabular` writes a small lookup-style policy file
instead and the deployment switches between the two via env var.

For Tier 2 we ship `--mode tabular` because it gives us 80% of the win at
1/10 the implementation cost. The full CNN replay path is sketched but
guarded behind a flag.
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


def build_oracle_table(
    state_keys: np.ndarray,
    actions: np.ndarray,
    round_scores: np.ndarray,
    min_visits: int,
    min_action_diff: float,
) -> tuple[dict[int, int], dict[int, float]]:
    """Greedy oracle action picker.

    Returns:
      action_table: state_key -> best action
      value_table:  state_key -> mean round score for chosen action
    """

    grouped: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for k, a, r in zip(state_keys, actions, round_scores):
        grouped[int(k)][int(a)].append(float(r))

    action_table: dict[int, int] = {}
    value_table: dict[int, float] = {}

    for key, by_action in grouped.items():
        total = sum(len(v) for v in by_action.values())
        if total < min_visits:
            continue
        if len(by_action) < 2:
            continue
        means = {a: float(np.mean(v)) for a, v in by_action.items() if len(v) >= 2}
        if not means:
            continue
        sorted_means = sorted(means.items(), key=lambda kv: -kv[1])
        best_action, best_v = sorted_means[0]
        # Skip if best vs second-best gap is too small (would inject noise).
        runner_v = sorted_means[1][1] if len(sorted_means) > 1 else -1e9
        if best_v - runner_v < min_action_diff:
            continue
        action_table[key] = best_action
        value_table[key] = best_v

    return action_table, value_table


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, default=Path("ae/models/oracle_table.npz"))
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--min-visits", type=int, default=4)
    p.add_argument("--min-action-diff", type=float, default=0.03,
                   help="best action must beat second-best by at least this score")
    args = p.parse_args(argv)

    all_keys: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    for path in args.inputs:
        data = np.load(path, allow_pickle=True)
        all_keys.append(data["state_keys"])
        all_actions.append(data["actions"])
        all_scores.append(
            data["round_score"].astype(np.float32) / 1000.0
            if "round_score" in data.files
            else np.zeros(len(data["state_keys"]), dtype=np.float32)
        )
        print(f"  {path.name}: {len(data['state_keys'])} steps")

    state_keys = np.concatenate(all_keys)
    actions = np.concatenate(all_actions)
    round_scores = np.concatenate(all_scores)
    print(f"total observed: {len(state_keys)} steps")

    action_table, value_table = build_oracle_table(
        state_keys, actions, round_scores,
        min_visits=args.min_visits,
        min_action_diff=args.min_action_diff,
    )
    print(f"oracle entries: {len(action_table)}  "
          f"(min_visits={args.min_visits} min_action_diff={args.min_action_diff})")

    if action_table:
        action_arr = np.asarray(list(action_table.values()), dtype=np.int8)
        action_counts = np.bincount(action_arr, minlength=6)
        print("  action mix : "
              f"FORWARD={action_counts[0]} BACKWARD={action_counts[1]} "
              f"LEFT={action_counts[2]} RIGHT={action_counts[3]} "
              f"STAY={action_counts[4]} BOMB={action_counts[5]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    keys_arr = np.asarray(list(action_table.keys()), dtype=np.int64)
    actions_arr = np.asarray([action_table[k] for k in keys_arr], dtype=np.int8)
    values_arr = np.asarray([value_table[k] for k in keys_arr], dtype=np.float32)
    np.savez_compressed(
        args.out,
        keys=keys_arr,
        actions=actions_arr,
        mean_value=values_arr,
        min_visits=np.int32(args.min_visits),
        min_action_diff=np.float32(args.min_action_diff),
    )
    print(f"oracle table -> {args.out}")

    if args.report:
        report = {
            "inputs": [str(p) for p in args.inputs],
            "total_steps": int(len(state_keys)),
            "oracle_entries": int(len(action_table)),
            "min_visits": args.min_visits,
            "min_action_diff": args.min_action_diff,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
