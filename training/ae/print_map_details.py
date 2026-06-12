import sys
import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TIL_AE = REPO_ROOT / "til-26-ae"
if str(TIL_AE) not in sys.path:
    sys.path.insert(0, str(TIL_AE))

from til_environment.arena import ArenaGenerator
from gymnasium.utils.seeding import np_random

def main():
    # Construct generator exactly like in bomberman_config.yaml (novice=true)
    arena_gen = ArenaGenerator(
        grid_size=16,
        wall_prob=1.0,
        wall_destructible_ratio=0.40,
        mission_prob=0.30,
        recon_prob=0.40,
        resource_prob=0.30,
        novice=True,
        base_respawn_steps=40,
    )
    
    episode_rng, _ = np_random(88)
    episode_seed = 88
    num_teams = 6

    result = arena_gen.generate_episode(
        episode_rng,
        episode_seed,
        num_teams=num_teams,
    )

    # Bits 0-3: wall presence (RIGHT=0, DOWN=1, LEFT=2, UP=3).
    # Bits 4-7: destructible flag.
    walls_list = []
    destructible_list = []
    for x in range(16):
        for y in range(16):
            val = result.wall_grid[x, y]
            if val > 0:
                w_dirs = []
                d_dirs = []
                for d in range(4):
                    if val & (1 << d):
                        w_dirs.append(d)
                    if val & (1 << (d + 4)):
                        d_dirs.append(d)
                if w_dirs:
                    walls_list.append(((x, y), w_dirs))
                if d_dirs:
                    destructible_list.append(((x, y), d_dirs))
                    
    static_list = []
    for spec in result.static_entities:
        static_list.append((spec.kind, spec.position.tolist()))

    data = {
        "base_locations": result.base_locations.tolist(),
        "starting_locations": result.starting_locations.tolist(),
        "starting_directions": result.starting_directions.tolist(),
        "walls_count": len(result.walls),
        "walls_list": walls_list,
        "destructible_list": destructible_list,
        "static_entities": static_list,
    }

    out_file = REPO_ROOT / "training" / "ae" / "data" / "novice_map_info.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote map details to {out_file}")

if __name__ == "__main__":
    main()
