"""Tests for the finals-aligned eval revamp (multiplier overlay + per-axis floors).

Design spec 2026-06-09-ae-finals-aligned-eval-revamp-design (private archive).
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


def test_worker_payload_has_finals_fields(monkeypatch):
    import melee_eval as M

    fake_summary = {
        "mean_score": 0.5, "mean_placement": 1.0, "mean_margin": 10.0, "win_rate": 1.0,
        "placement_hist": {str(k): 0 for k in range(1, 7)},
        "per_round": {
            "cumulative_all": [{"agent_0": 300.0, "agent_1": 100.0, "agent_2": 0.0,
                                "agent_3": 0.0, "agent_4": 0.0, "agent_5": 0.0}],
            "opening_reward": [12.0],
            "reward_components": [{"collect_mission": 25.0, "own_base_destroyed": -65.0}],
            "us_agent_id": "agent_0",
        },
    }
    monkeypatch.setattr(M, "OUR_MULT", 0.93)
    monkeypatch.setattr(M, "OPP_MULTS", [0.24, 0.7])
    captured = {}
    monkeypatch.setattr(M, "_compute_worker_payload",
                        M._compute_worker_payload)  # ensure it exists
    payload = M._compute_worker_payload(fake_summary, "semis_mixed", "heuristic", 42)
    assert payload["raw_ae"] == 300.0
    assert payload["mission_axis"] == 25.0
    assert payload["base_defense_axis"] == -65.0
    assert payload["opening_axis"] == 12.0
    assert payload["weighted_placement"]["0.24"] == 1.0


def test_aggregate_candidate_robust_placement_and_axes():
    import melee_eval as M
    # Two brackets, one run each; weighted_placement worst over (bracket x opp_mult).
    runs_a = [{"hash": 0, "sim": 42, "mean_placement": 1.0, "win_rate": 1.0,
               "mean_margin": 10.0, "mean_score": 0.5, "raw_ae": 500.0,
               "mission_axis": 150.0, "base_defense_axis": -40.0, "opening_axis": 20.0,
               "weighted_placement": {"0.24": 1.0, "0.7": 1.0}}]
    runs_b = [{"hash": 0, "sim": 42, "mean_placement": 1.0, "win_rate": 1.0,
               "mean_margin": 5.0, "mean_score": 0.4, "raw_ae": 400.0,
               "mission_axis": 120.0, "base_defense_axis": -60.0, "opening_axis": 10.0,
               "weighted_placement": {"0.24": 1.0, "0.7": 2.0}}]
    cand = M._aggregate_candidate("cand", {"our": "heuristic"},
                                  {"semis_mixed": runs_a, "real_field": runs_b})
    assert cand["worst_robust_placement"] == 2.0          # bracket real_field @ 0.7
    assert cand["raw_ae"] == 450.0                         # mean of 500, 400
    assert cand["mission_axis"] == 135.0
    # run_rows must retain per-run raw_ae for paired stats
    assert cand["per_bracket"]["semis_mixed"]["run_rows"][0]["raw_ae"] == 500.0


def _mk(worst_robust, raw_ae, mission, base_def, rows):
    return {"worst_robust_placement": worst_robust, "worst_bracket_placement": 1.0,
            "min_margin": 5.0, "semis_mixed_score": 0.5, "raw_ae": raw_ae,
            "mission_axis": mission, "base_defense_axis": base_def, "opening_axis": 10.0,
            "per_bracket": {"semis_mixed": {"run_rows": rows}}}


def test_verdict_mission_change_passes_on_axis_gain():
    import melee_eval as M
    # 4 paired (hash,sim) rows so the Wilson-95 LB on a 4/4 win sweep clears 0.5
    # (wilson_lower(n,n) = n/(n+z^2) = 4/7.84 = 0.51 > 0.5).
    sims = [42, 137, 7, 99]
    rows_i = [{"hash": 0, "sim": s, "mean_placement": 1.0, "raw_ae": 500.0,
               "mission_axis": 100.0, "base_defense_axis": -40.0, "opening_axis": 10.0}
              for s in sims]
    rows_c = [{"hash": 0, "sim": s, "mean_placement": 1.0, "raw_ae": 520.0,
               "mission_axis": 140.0, "base_defense_axis": -40.0, "opening_axis": 10.0}
              for s in sims]
    inc = _mk(1.0, 500.0, 100.0, -40.0, rows_i)
    cand = _mk(1.0, 520.0, 140.0, -40.0, rows_c)
    v = M._promotion_verdict(cand, inc, target_axis="mission")
    assert v["placement_robust_ok"] and v["rawae_floor_ok"] and v["axis_ok"]
    assert v["promotable"]


def test_verdict_fails_when_axis_does_not_move():
    import melee_eval as M
    rows = [{"hash": 0, "sim": 42, "mean_placement": 1.0, "raw_ae": 500.0,
             "mission_axis": 100.0, "base_defense_axis": -40.0, "opening_axis": 10.0}]
    inc = _mk(1.0, 500.0, 100.0, -40.0, rows)
    cand = _mk(1.0, 500.0, 100.0, -40.0, rows)   # identical -> mission didn't move
    v = M._promotion_verdict(cand, inc, target_axis="mission")
    assert not v["axis_ok"] and not v["promotable"]


def test_verdict_fails_when_rawae_craters():
    import melee_eval as M
    rows_i = [{"hash": 0, "sim": 42, "mean_placement": 1.0, "raw_ae": 500.0,
               "mission_axis": 100.0, "base_defense_axis": -40.0, "opening_axis": 10.0}]
    rows_c = [{"hash": 0, "sim": 42, "mean_placement": 1.0, "raw_ae": 200.0,
               "mission_axis": 160.0, "base_defense_axis": -40.0, "opening_axis": 10.0}]
    inc = _mk(1.0, 500.0, 100.0, -40.0, rows_i)
    cand = _mk(1.0, 200.0, 160.0, -40.0, rows_c)   # raw_ae 200 < 0.75*500=375
    v = M._promotion_verdict(cand, inc, target_axis="mission")
    assert not v["rawae_floor_ok"] and not v["promotable"]
