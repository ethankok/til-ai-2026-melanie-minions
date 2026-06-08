"""Tests for the finals-aligned eval revamp (multiplier overlay + per-axis floors).

Spec: docs/superpowers/specs/2026-06-09-ae-finals-aligned-eval-revamp-design.md
"""
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))


def test_run_simulation_exports_per_round():
    import simulate
    out = simulate.run_simulation(
        rounds=2, opponents_spec="greedy,greedy,bomber,defender,hunter",
        our_name="heuristic", log_traj=False, seed_start=42, novice=True,
    )
    pr = out["summary"]["per_round"]
    assert len(pr["cumulative_all"]) == 2
    assert len(pr["opening_reward"]) == 2
    assert len(pr["reward_components"]) == 2
    usid = pr["us_agent_id"]
    # our agent must be a key in every round's cumulative_all
    assert all(usid in ca for ca in pr["cumulative_all"])
    # opening_reward is a finite float per round
    assert all(isinstance(x, float) and math.isfinite(x) for x in pr["opening_reward"])
