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


def test_weighted_placement_saturates_then_contests():
    import melee_eval as M
    # Our raw reward (300) is below an opponent's (500), but our multiplier is high.
    cum = {"agent_0": 300.0, "agent_1": 500.0, "agent_2": 100.0,
           "agent_3": 50.0, "agent_4": 0.0, "agent_5": -20.0}
    # Stub field (opp_mult 0.24): 300*0.93=279 vs 500*0.24=120 -> we win (1st).
    assert M._weighted_placement(cum, "agent_0", 0.93, 0.24) == 1
    # Strong field (opp_mult 0.93): 279 vs 465 -> opponent wins, we drop.
    assert M._weighted_placement(cum, "agent_0", 0.93, 0.93) == 2


def test_axis_totals_signs():
    import melee_eval as M
    comp = {"collect_mission": 50.0, "collect_recon": 5.0, "collect_resource": 4.0,
            "own_base_destroyed": -65.0, "self_damage": -20.0, "base_damage": -3.0}
    ax = M._axis_totals(comp)
    assert ax["mission_axis"] == 55.0          # mission + recon (resource excluded)
    assert ax["base_defense_axis"] == -88.0     # all three negatives summed


def test_paired_value_stats_higher_is_better():
    import melee_eval as M
    cand = {"per_bracket": {"semis_mixed": {"run_rows": [
        {"hash": 0, "sim": 42, "raw_ae": 520.0},
        {"hash": 0, "sim": 43, "raw_ae": 480.0}]}}}
    inc = {"per_bracket": {"semis_mixed": {"run_rows": [
        {"hash": 0, "sim": 42, "raw_ae": 500.0},
        {"hash": 0, "sim": 43, "raw_ae": 500.0}]}}}
    st = M._paired_value_stats(cand, inc, "raw_ae")
    assert st["n_pairs"] == 2
    assert st["mean_delta"] == 0.0             # (+20 + -20)/2
    assert st["wins"] == 1 and st["losses"] == 1
