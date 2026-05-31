"""Divergence-gated opening prefix in front of a configurable planner.

On the fixed Novice map, replays a precomputed, per-spawn opening move sequence
(from training/ae/gen_openings.py + sweep, locked into ``openings_gate.json``
beside this file) for the first few ticks, then hands off to the live planner.
The opening is computed single-agent on the known map, so it is
**divergence-gated**: before replaying opening move k it verifies the observed
(x, y, direction) matches the baked predicted trajectory at step k. Any mismatch
— an enemy body-blocked us, we were frozen, an unexpected obstacle, an illegal
action — aborts the opening permanently and the planner runs the rest of the
game. This makes the opening free upside: it executes only while reality matches
the plan, and only on the spawns where it was validated to beat the planner
(others have an empty gate entry → planner from tick 0).

The wrapped ``planner`` is whatever ae_server constructs (the C+bomb7 heuristic,
or the confidence_policy_hybrid confpol agent). Its belief/policy is kept warm
by calling it every tick during the opening (its action is discarded).

Self-contained: reads the baked-trajectory gate; no OpeningSim / training-code
dependency, so it ships in the ae/src image unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_GATE_PATH = Path(__file__).resolve().parent / "openings_gate.json"


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
    """Opening-prefix + divergence-gate wrapper around an arbitrary planner."""

    def __init__(self, planner=None, gate_path: str | Path | None = None) -> None:
        if planner is None:
            from ae_manager import AEManager
            planner = AEManager()
        self.planner = planner

        path = Path(gate_path) if gate_path else _GATE_PATH
        try:
            self.gate: dict[str, list[dict]] = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001 — missing/corrupt gate -> planner only
            print(f"AE: opening_hybrid gate load failed ({exc!r}) — planner only")
            self.gate = {}

        self._reset_opening()
        enabled = [k for k, v in self.gate.items() if v]
        print(f"AE: opening_hybrid ready (planner={type(self.planner).__name__}, "
              f"opening-enabled spawns={enabled})")

    # ── opening state ─────────────────────────────────────────────────────
    def _reset_opening(self) -> None:
        self.seq: list[int] = []
        self.traj: list[tuple[int, int, int]] = []
        self.idx: int = 0
        self.aborted: bool = True

    def _reset_memory(self) -> None:
        if hasattr(self.planner, "_reset_memory"):
            self.planner._reset_memory()
        self._reset_opening()

    def _load(self, base: tuple[int, int]) -> None:
        self._reset_opening()
        cands = self.gate.get(f"{base[0]},{base[1]}")
        if not cands:
            return  # disabled spawn -> planner from tick 0
        cand = cands[0]
        self.seq = [int(a) for a in cand["actions"]]
        self.traj = [(int(t[0]), int(t[1]), int(t[2])) for t in cand["traj"]]
        self.idx = 0
        self.aborted = False

    # ── policy ────────────────────────────────────────────────────────────
    def ae(self, observation: dict) -> int:
        if _as_int(observation.get("step"), -1) == 0:
            base = _as_pos(observation.get("base_location"))
            if base is not None:
                self._load(base)

        # Keep the planner warm every tick (it drives the game after the opening).
        planner_action = int(self.planner.ae(observation))

        if self.aborted or self.idx >= len(self.seq):
            self.aborted = True
            return planner_action

        if _as_int(observation.get("frozen_ticks"), 0) > 0:
            self.aborted = True
            return planner_action

        pos = _as_pos(observation.get("location"))
        direction = _as_int(observation.get("direction"), 0) % 4
        px, py, pd = self.traj[self.idx]
        if pos != (px, py) or direction != pd:
            self.aborted = True
            return planner_action

        action = self.seq[self.idx]
        mask = observation.get("action_mask")
        if mask is not None:
            try:
                if not int(mask[action]):
                    self.aborted = True
                    return planner_action
            except (TypeError, ValueError, IndexError):
                pass

        self.idx += 1
        return action
