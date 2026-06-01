"""Tests for the SHIPPED ae/src/opening_hybrid_manager.py (deploy manager).

Self-contained: reads the baked-trajectory gate (ae/src/openings_gate.json),
wraps a generic planner, runs the divergence gate against the baked traj — no
training-code or OpeningSim dependency (mirrors the container layout where only
ae/src is on the path).

Run: PYTHONHASHSEED=0 .venv/bin/python training/ae/test_deploy_opening.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") is None:
    os.environ["PYTHONHASHSEED"] = "0"
    os.execvp(sys.executable, [sys.executable, *sys.argv])

ROOT = Path(__file__).resolve().parents[2]
# Deploy layout: only ae/src on the path (+ til-26-ae for nothing here).
for p in (str(ROOT / "ae" / "src"),):
    if p not in sys.path:
        sys.path.insert(0, p)

from opening_hybrid_manager import OpeningHybridManager  # noqa: E402  (ae/src deploy module)

GATE = json.loads((ROOT / "ae" / "src" / "openings_gate.json").read_text())
STUB = 4


class StubPlanner:
    def __init__(self):
        self.calls = 0
    def ae(self, obs):
        self.calls += 1
        return STUB
    def _reset_memory(self):
        pass


def _obs(step, loc, d, base, frozen=0):
    return {"step": step, "location": list(loc), "direction": int(d),
            "base_location": list(base), "frozen_ticks": frozen,
            "action_mask": [1, 1, 1, 1, 1, 1]}


def _key_to_base(k):
    return tuple(int(v) for v in k.split(","))


# Derive an enabled and a disabled spawn from the gate (robust to gate changes).
ENABLED_KEY = next(k for k, v in GATE.items() if v)
DISABLED_KEY = next(k for k, v in GATE.items() if not v)
ENABLED_BASE = _key_to_base(ENABLED_KEY)
DISABLED_BASE = _key_to_base(DISABLED_KEY)


def test_enabled_slot_plays_baked_opening_then_delegates():
    base = ENABLED_BASE
    cand = GATE[ENABLED_KEY][0]
    seq, traj = cand["actions"], cand["traj"]
    stub = StubPlanner()
    mgr = OpeningHybridManager(planner=stub)
    for k in range(len(seq)):
        x, y, d = traj[k]
        a = mgr.ae(_obs(k, (x, y), d, base))
        assert a == seq[k], f"tick {k}: expected {seq[k]}, got {a}"
    x, y, d = traj[len(seq)]
    a = mgr.ae(_obs(len(seq), (x, y), d, base))
    assert a == STUB and mgr.aborted


def test_disabled_slot_delegates_to_planner():
    base = DISABLED_BASE  # planner slot (gate entry is [])
    assert GATE[DISABLED_KEY] == []
    stub = StubPlanner()
    mgr = OpeningHybridManager(planner=stub)
    a = mgr.ae(_obs(0, (base[0], base[1] + 1), 0, base))
    assert a == STUB and mgr.aborted


def test_divergence_aborts():
    base = ENABLED_BASE
    traj = GATE[ENABLED_KEY][0]["traj"]
    stub = StubPlanner()
    mgr = OpeningHybridManager(planner=stub)
    mgr.ae(_obs(0, traj[0][:2], traj[0][2], base))  # ok
    wrong = (traj[1][0] + 4, traj[1][1])
    a = mgr.ae(_obs(1, wrong, traj[1][2], base))
    assert a == STUB and mgr.aborted


def test_planner_kept_warm_every_tick():
    base = ENABLED_BASE
    traj = GATE[ENABLED_KEY][0]["traj"]
    stub = StubPlanner()
    mgr = OpeningHybridManager(planner=stub)
    for k in range(3):
        mgr.ae(_obs(k, traj[k][:2], traj[k][2], base))
    assert stub.calls == 3


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL {t.__name__}: {e!r}")
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _run()
