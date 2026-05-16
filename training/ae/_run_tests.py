"""One-shot test runner so we don't have to install pytest everywhere."""
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ae" / "src"))

from test.test_ae_manager_planner import (
    test_planner_moves_forward_toward_visible_mission,
    test_planner_turns_toward_side_objective,
    test_action_mask_is_always_respected,
    test_memory_resets_when_step_goes_back_to_zero,
    test_no_periodic_bombing_without_tactical_value,
    test_places_bomb_when_enemy_base_visible_and_escape_exists,
    test_accepts_raw_numpy_observation_arrays,
    test_visible_empty_cell_clears_stale_enemy_memory_before_bombing,
    test_does_not_bomb_when_own_base_is_in_blast,
)

tests = [
    test_planner_moves_forward_toward_visible_mission,
    test_planner_turns_toward_side_objective,
    test_action_mask_is_always_respected,
    test_memory_resets_when_step_goes_back_to_zero,
    test_no_periodic_bombing_without_tactical_value,
    test_places_bomb_when_enemy_base_visible_and_escape_exists,
    test_accepts_raw_numpy_observation_arrays,
    test_visible_empty_cell_clears_stale_enemy_memory_before_bombing,
    test_does_not_bomb_when_own_base_is_in_blast,
]

passed = failed = 0
for t in tests:
    try:
        t()
        print(f"PASS  {t.__name__}")
        passed += 1
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL  {t.__name__}: {exc}")
        traceback.print_exc()
        failed += 1

print(f"\n{passed}/{passed+failed} passed")
sys.exit(0 if failed == 0 else 1)
