"""W2.2c scripted_hybrid AE mode.

The wrapping manager that runs ScriptedBaseAttackPolicy first; if it returns
None or raises, falls back to AEManager.ae(). State is shared via composition:
a single AEManager instance is used by both layers, so observation parsing and
memory updates happen exactly once per tick.

Mode activation: `AE_MODE=scripted_hybrid`. See ae/src/ae_server.py.

Diagnostics: this manager exposes `decision_counts` (matching the
TacticalHybridAEManager shape) so existing per-suite diagnostic plumbing in
simulate.py / validate_cloud_suite.py keeps working. Counts:
  scripted_<reason>: the scripted policy returned an action
  fallback_<reason>: the scripted policy declined; heuristic acted
"""

from __future__ import annotations

from collections import Counter

from ae_manager import AEManager
from scripted_base_attack import ScriptedBaseAttackPolicy


class ScriptedHybridAEManager:
    """Scripted-first decision policy with heuristic fallback."""

    def __init__(self) -> None:
        self.heuristic = AEManager()
        self.scripted = ScriptedBaseAttackPolicy(self.heuristic)
        # Public diagnostic surface mirrors TacticalHybridAEManager.
        self.decision_counts: Counter[str] = Counter()
        self.option_counts: Counter[str] = Counter()  # placeholder for parity
        self.veto_counts: Counter[str] = Counter()
        self._last_step: int | None = None

    def _maybe_reset(self, observation: dict) -> None:
        step = observation.get("step")
        try:
            step_int = int(step)
        except Exception:
            return
        if step_int == 0 or (self._last_step is not None and step_int < self._last_step):
            self.scripted.reset_plan()
        self._last_step = step_int

    def ae(self, observation: dict) -> int:
        self._maybe_reset(observation)
        # IMPORTANT: heuristic.ae() does all observation parsing + memory
        # syncs. Running it FIRST guarantees scripted.act() sees fresh state.
        # The returned heuristic action is the fallback if scripted declines.
        heuristic_action = int(self.heuristic.ae(observation))
        try:
            scripted_action = self.scripted.act(observation)
        except Exception as exc:  # noqa: BLE001
            self.decision_counts[f"scripted_error_{type(exc).__name__}"] += 1
            return heuristic_action
        if scripted_action is None:
            # Aggregate decline reasons into our public surface.
            for reason, count in self.scripted.decline_reasons.items():
                self.decision_counts[f"fallback_{reason}"] = count
            return heuristic_action
        for reason, count in self.scripted.scripted_decisions.items():
            self.decision_counts[f"scripted_{reason}"] = count
        return int(scripted_action)
