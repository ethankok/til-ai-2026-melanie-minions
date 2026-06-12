"""Heuristics parameter sweep runner for AEManager.

Runs multiple configurations in parallel using multiprocessing to identify the optimal
heuristic parameters against library or mixed opponents.
"""

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TIL_AE = REPO_ROOT / "til-26-ae"
AE_SRC = REPO_ROOT / "ae" / "src"
TRAINING_AE = REPO_ROOT / "training" / "ae"
for p in (str(AE_SRC), str(TRAINING_AE), str(TIL_AE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ae_manager import AEManager
from opponents import make_opponent
from til_environment import bomberman_env
from til_environment.config import default_config
from simulate import run_one_round

CONFIGS = [
    {
        "name": "0_baseline",
        "dijkstra_no_bomb_cost": 5.0,
        "ENEMY_CHASE_VALUE": 0.0,
        "low_ammo_enemy_base_value": 130.0,
        "low_ammo_resource_value": 25.0,
        "base_health_panic_threshold": 0.0,
    },
    {
        "name": "4_low_ammo_base_80_res_40",
        "dijkstra_no_bomb_cost": 99.0,
        "low_ammo_enemy_base_value": 80.0,
        "low_ammo_resource_value": 40.0,
        "base_health_panic_threshold": 60.0,
        "base_defense_panic_radius": 12,
        "base_defense_panic_value": 150.0,
    },
    {
        "name": "9_combo_moderate",
        "dijkstra_no_bomb_cost": 20.0,
        "low_ammo_enemy_base_value": 100.0,
        "low_ammo_resource_value": 35.0,
        "base_health_panic_threshold": 40.0,
        "base_defense_panic_radius": 8,
        "base_defense_panic_value": 150.0,
        "ENEMY_CHASE_VALUE": 20.0,
        "ENEMY_CHASE_RADIUS": 3,
    },
    {
        "name": "optimal_combo",
        "dijkstra_no_bomb_cost": 25.0,
        "low_ammo_enemy_base_value": 110.0,
        "low_ammo_resource_value": 35.0,
        "base_health_panic_threshold": 40.0,
        "base_defense_panic_radius": 8,
        "base_defense_panic_value": 150.0,
        "ENEMY_CHASE_VALUE": 15.0,
        "ENEMY_CHASE_RADIUS": 3,
    },
]

def run_worker(args):
    config, rounds, opponents_spec, seed_start, novice = args
    cfg = default_config()
    cfg.env.novice = novice
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    
    config_overrides = {k: v for k, v in config.items() if k != "name"}
    our_agent = AEManager(**config_overrides)

    if opponents_spec == "random":
        names = ["random"] * 5
    elif opponents_spec == "mixed":
        names = ["mixed"] * 5
    elif opponents_spec == "library":
        names = ["random", "greedy", "bomber", "defender", "hunter"]
    else:
        names = [n.strip() for n in opponents_spec.split(",") if n.strip()]
        if len(names) == 1:
            names = names * 5
            
    opponents = [make_opponent(n, seed=seed_start + 1000 + i) for i, n in enumerate(names)]
    scores = []
    
    for r in range(rounds):
        res = run_one_round(env, our_agent, opponents, log_traj=False, seed=None)
        scores.append(res["score"])
        
    env.close()
    return config["name"], scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=50, help="Rounds per configuration")
    parser.add_argument("--opponents", type=str, default="library", help="library or mixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--non-novice", action="store_true")
    args = parser.parse_args()

    novice = not args.non_novice
    print(f"Starting sweep over {len(CONFIGS)} configurations...")
    print(f"Rounds per config: {args.rounds}")
    print(f"Opponents: {args.opponents}")
    print(f"Novice map mode: {novice}")
    print("-" * 64)

    tasks = []
    for config in CONFIGS:
        tasks.append((config, args.rounds, args.opponents, args.seed, novice))

    t0 = time.monotonic()
    
    with Pool() as pool:
        results = pool.map(run_worker, tasks)

    elapsed = time.monotonic() - t0
    print(f"Sweep complete in {elapsed:.1f}s!")
    print("=" * 64)
    print(f"{'Configuration Name':<30} | {'Mean Score':<10} | {'Std Dev':<8} | {'Min':<6} | {'Max':<6}")
    print("-" * 64)

    best_mean = -1.0
    best_name = None
    best_config = None

    for name, scores in results:
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        print(f"{name:<30} | {mean_score:<10.4f} | {std_score:<8.4f} | {min_score:<6.3f} | {max_score:<6.3f}")
        
        if mean_score > best_mean:
            best_mean = mean_score
            best_name = name
            best_config = [c for c in CONFIGS if c["name"] == name][0]

    print("=" * 64)
    print(f"WINNING CONFIGURATION: {best_name} (Mean score: {best_mean:.4f})")
    print("Parameters:")
    for k, v in best_config.items():
        if k != "name":
            print(f"  {k} = {v}")

if __name__ == "__main__":
    main()
