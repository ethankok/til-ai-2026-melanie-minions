import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "ae" / "src"))

from ae_manager import AEManager


VISIBLE = 0
TILE_RECON = 6
TILE_MISSION = 7
TILE_RESOURCE = 8
ENEMY_AGENT = 10
ENEMY_BASE = 12


def empty_viewcone():
    view = [[[0.0 for _ in range(25)] for _ in range(5)] for _ in range(7)]
    for r in range(7):
        for c in range(5):
            view[r][c][VISIBLE] = 1.0
    return view


def obs(view, *, direction=0, location=(8, 8), step=1, mask=None, team_bombs=0):
    return {
        "agent_viewcone": view,
        "base_viewcone": [[[0.0 for _ in range(25)] for _ in range(7)] for _ in range(7)],
        "direction": direction,
        "location": list(location),
        "base_location": [0, 0],
        "health": [60.0],
        "frozen_ticks": 0,
        "base_health": [100.0],
        "team_resources": [0.0],
        "team_bombs": team_bombs,
        "step": step,
        "action_mask": mask or [1, 1, 1, 1, 1, 0],
    }


def test_planner_moves_forward_toward_visible_mission():
    view = empty_viewcone()
    view[3][2][TILE_MISSION] = 1.0  # one tile ahead when facing RIGHT
    manager = AEManager()

    assert manager.ae(obs(view, direction=0, location=(8, 8))) == AEManager.FORWARD


def test_planner_turns_toward_side_objective():
    view = empty_viewcone()
    view[2][1][TILE_RESOURCE] = 1.0  # one tile left when facing RIGHT
    manager = AEManager()

    assert manager.ae(obs(view, direction=0, location=(8, 8))) == AEManager.LEFT


def test_action_mask_is_always_respected():
    view = empty_viewcone()
    view[3][2][TILE_MISSION] = 1.0
    manager = AEManager()

    action = manager.ae(obs(view, mask=[0, 1, 1, 1, 1, 0]))

    assert action != AEManager.FORWARD
    assert action in {AEManager.BACKWARD, AEManager.LEFT, AEManager.RIGHT, AEManager.STAY}


def test_memory_resets_when_step_goes_back_to_zero():
    view = empty_viewcone()
    manager = AEManager()
    manager.ae(obs(view, location=(8, 8), step=15))
    assert manager.visit_count

    manager.ae(obs(view, location=(1, 1), step=0))

    assert (8, 8) not in manager.visit_count
    assert manager.visit_count.get((1, 1)) == 1


def test_no_periodic_bombing_without_tactical_value():
    view = empty_viewcone()
    manager = AEManager()

    actions = [
        manager.ae(obs(view, location=(8, 8), step=i, mask=[1, 1, 1, 1, 1, 1], team_bombs=3))
        for i in range(1, 45)
    ]

    assert AEManager.PLACE_BOMB not in actions


def test_places_bomb_when_enemy_base_visible_and_escape_exists():
    view = empty_viewcone()
    view[3][2][ENEMY_BASE] = 1.0  # adjacent ahead, inside blast reach
    manager = AEManager()

    assert manager.ae(obs(view, location=(8, 8), mask=[1, 1, 1, 1, 1, 1], team_bombs=1)) == AEManager.PLACE_BOMB


def test_accepts_raw_numpy_observation_arrays():
    view = np.array(empty_viewcone(), dtype=np.float32)
    observation = obs(view.tolist(), direction=0, location=(8, 8))
    observation["agent_viewcone"] = view
    observation["base_viewcone"] = np.zeros((7, 7, 25), dtype=np.float32)
    observation["location"] = np.array([8, 8])
    observation["base_location"] = np.array([0, 0])
    observation["health"] = np.array([60.0], dtype=np.float32)
    observation["base_health"] = np.array([100.0], dtype=np.float32)
    observation["team_resources"] = np.array([0.0], dtype=np.float32)
    observation["action_mask"] = np.array([1, 1, 1, 1, 1, 0], dtype=np.int8)
    manager = AEManager()

    action = manager.ae(observation)

    assert 0 <= action <= 5
    assert bool(observation["action_mask"][action])


def test_visible_empty_cell_clears_stale_enemy_memory_before_bombing():
    enemy_view = empty_viewcone()
    enemy_view[3][2][ENEMY_AGENT] = 1.0
    empty_after = empty_viewcone()
    manager = AEManager()
    manager.ae(obs(enemy_view, location=(8, 8), step=1, mask=[1, 1, 1, 1, 1, 0], team_bombs=0))

    action = manager.ae(obs(empty_after, location=(8, 8), step=2, mask=[1, 1, 1, 1, 1, 1], team_bombs=3))

    assert action != AEManager.PLACE_BOMB


def test_does_not_bomb_when_own_base_is_in_blast():
    view = empty_viewcone()
    view[3][2][ENEMY_BASE] = 1.0
    observation = obs(view, location=(8, 8), mask=[1, 1, 1, 1, 1, 1], team_bombs=1)
    observation["base_location"] = [9, 8]
    manager = AEManager()

    assert manager.ae(observation) != AEManager.PLACE_BOMB
