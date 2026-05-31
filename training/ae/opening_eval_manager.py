"""Opening-prefix + divergence-gate wrapper around the AEManager heuristic.

On the fixed Novice map, replays a precomputed opening move sequence (from
gen_openings.py -> data/openings.json) for the agent's spawn slot, then hands
off to the live AEManager planner. The opening is computed single-agent on the
known map, so at runtime it is **divergence-gated**: before replaying opening
action k, the agent verifies its observed (position, direction) matches the
opening's predicted trajectory at step k. Any mismatch — an enemy body-blocked
us, we got frozen, an unexpected obstacle — aborts the opening permanently and
delegates the rest of the game to the heuristic. This makes the opening free
upside: it runs only while reality matches the plan.

The heuristic's belief/memory is kept warm by calling it every tick during the
opening (its action is discarded), so the hand-off planner has full world state.

This is the VALIDATION wrapper (imports the offline OpeningSim for trajectory
prediction). A deploy-packaged version that bakes sequences+trajectories into
ae/src with a self-contained replay is a follow-up.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from opening_sim import SLOT_TABLE, OpeningSim

_DEFAULT_OPENINGS = Path(__file__).resolve().parent / "data" / "openings.json"


def _as_int(v: Any, default: int = 0) -> int:
    try:
        if isinstance(v, (list, tuple)):
            return int(v[0])
        return int(v)
    except (TypeError, ValueError, IndexError):
        return default


def _as_pos(v: Any) -> tuple[int, int] | None:
    try:
        return (int(v[0]), int(v[1]))
    except (TypeError, ValueError, IndexError):
        return None


class OpeningHybridManager:
    """Heuristic planner with a divergence-gated opening prefix on the Novice map."""

    def __init__(
        self,
        heuristic=None,
        openings_path: str | Path | None = None,
        candidate_index: int = 0,
        openings: dict[str, list[dict]] | None = None,
    ) -> None:
        if heuristic is None:
            from ae_manager import AEManager  # local import: deploy path may differ
            heuristic = AEManager()
        self.heuristic = heuristic
        self.candidate_index = candidate_index

        if openings is not None:
            self.openings = openings
        else:
            path = Path(openings_path) if openings_path else _DEFAULT_OPENINGS
            self.openings = json.loads(Path(path).read_text())
        self._sim = OpeningSim()

        self._reset_opening()

    # ── opening state ─────────────────────────────────────────────────────
    def _reset_opening(self) -> None:
        self.seq: list[int] = []
        self.traj: list[tuple[tuple[int, int], int]] = []
        self.idx: int = 0
        self.aborted: bool = True  # until a valid opening is loaded at step 0
        self._loaded_for: tuple[int, int] | None = None

    def _reset_memory(self) -> None:
        # Called by the sim harness at round start.
        if hasattr(self.heuristic, "_reset_memory"):
            self.heuristic._reset_memory()
        self._reset_opening()

    def _load_opening(self, base: tuple[int, int]) -> None:
        self._reset_opening()
        key = f"{base[0]},{base[1]}"
        cands = self.openings.get(key)
        if base not in SLOT_TABLE or not cands or self.candidate_index >= len(cands):
            return  # stays aborted -> pure heuristic
        self.seq = list(cands[self.candidate_index]["actions"])
        st = self._sim.initial_state(base)
        self.traj = [(st.pos, st.dir)]
        for a in self.seq:
            st = self._sim.step(st, a)
            self.traj.append((st.pos, st.dir))
        self.idx = 0
        self.aborted = False
        self._loaded_for = base

    # ── policy ────────────────────────────────────────────────────────────
    def ae(self, observation: dict) -> int:
        step = _as_int(observation.get("step"), default=-1)
        if step == 0:
            base = _as_pos(observation.get("base_location"))
            if base is not None:
                self._load_opening(base)

        # Keep the heuristic's belief warm every tick (its action drives the
        # game once the opening ends / aborts).
        heuristic_action = int(self.heuristic.ae(observation))

        if self.aborted or self.idx >= len(self.seq):
            self.aborted = True
            return heuristic_action

        # Divergence gate: frozen, illegal, or position/direction mismatch -> abort.
        if _as_int(observation.get("frozen_ticks"), 0) > 0:
            self.aborted = True
            return heuristic_action

        pos = _as_pos(observation.get("location"))
        direction = _as_int(observation.get("direction"), 0) % 4
        pred_pos, pred_dir = self.traj[self.idx]
        if pos != pred_pos or direction != pred_dir:
            self.aborted = True
            return heuristic_action

        action = self.seq[self.idx]
        mask = observation.get("action_mask")
        if mask is not None:
            try:
                if not int(mask[action]):
                    self.aborted = True
                    return heuristic_action
            except (TypeError, ValueError, IndexError):
                pass

        self.idx += 1
        return action
