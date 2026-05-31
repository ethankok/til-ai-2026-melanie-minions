"""Tests for OpeningHybridManager — the opening-prefix + divergence-gate wrapper.

Plain-assert runner. Run:
    PYTHONHASHSEED=0 .venv/bin/python training/ae/test_opening_hybrid.py

A stub heuristic is injected so the gate logic can be tested with minimal
observations (no full AEManager obs needed). One integration test drives the
wrapper through the real env to confirm obs-format handling.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") is None:
    os.environ["PYTHONHASHSEED"] = "0"
    os.execvp(sys.executable, [sys.executable, *sys.argv])

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "ae" / "src"), str(ROOT / "til-26-ae"), str(ROOT / "training" / "ae")):
    if p not in sys.path:
        sys.path.insert(0, p)

from opening_eval_manager import OpeningHybridManager  # noqa: E402
from opening_sim import OpeningSim, SLOT_TABLE  # noqa: E402

STUB = 4  # sentinel action the stub heuristic returns when delegated to


class StubHeuristic:
    def __init__(self):
        self.calls = 0
    def ae(self, obs):
        self.calls += 1
        return STUB
    def _reset_memory(self):
        pass


def _obs(step, loc, d, base=(13, 9), frozen=0):
    return {
        "step": step,
        "location": list(loc),
        "direction": int(d),
        "base_location": list(base),
        "frozen_ticks": frozen,
        "action_mask": [1, 1, 1, 1, 1, 1],
        "health": [60],
        "base_health": [100],
    }


def _opening_for(base):
    """Return (seq, traj) for a base's best candidate."""
    mgr = OpeningHybridManager(heuristic=StubHeuristic())
    seq = mgr.openings[f"{base[0]},{base[1]}"][0]["actions"]
    sim = OpeningSim()
    st = sim.initial_state(base)
    traj = [(st.pos, st.dir)]
    for a in seq:
        st = sim.step(st, a)
        traj.append((st.pos, st.dir))
    return seq, traj


def test_happy_path_emits_full_opening_then_delegates():
    base = (13, 9)
    seq, traj = _opening_for(base)
    stub = StubHeuristic()
    mgr = OpeningHybridManager(heuristic=stub)
    # Feed observations exactly matching the predicted trajectory (no divergence).
    for k in range(len(seq)):
        pos, d = traj[k]
        a = mgr.ae(_obs(k, pos, d, base))
        assert a == seq[k], f"tick {k}: expected opening {seq[k]}, got {a}"
    # Opening exhausted -> next tick delegates to heuristic.
    pos, d = traj[len(seq)]
    a = mgr.ae(_obs(len(seq), pos, d, base))
    assert a == STUB, f"after opening expected delegate ({STUB}), got {a}"
    assert mgr.aborted


def test_divergence_aborts_and_stays_aborted():
    base = (13, 9)
    seq, traj = _opening_for(base)
    stub = StubHeuristic()
    mgr = OpeningHybridManager(heuristic=stub)
    # tick 0 matches -> plays opening[0]
    a0 = mgr.ae(_obs(0, traj[0][0], traj[0][1], base))
    assert a0 == seq[0]
    # tick 1: feed a WRONG position (an enemy blocked us / desync) -> abort
    wrong_pos = (traj[1][0][0] + 5, traj[1][0][1])
    a1 = mgr.ae(_obs(1, wrong_pos, traj[1][1], base))
    assert a1 == STUB, "divergence should delegate to heuristic"
    assert mgr.aborted
    # stays aborted even if a later obs happens to match prediction again
    a2 = mgr.ae(_obs(2, traj[2][0], traj[2][1], base))
    assert a2 == STUB and mgr.aborted


def test_in_memory_openings_and_disabled_slot():
    """An in-memory openings dict overrides the file; an empty list = disabled
    (delegates to the planner). This is how the locked per-slot gate works."""
    seq, traj = _opening_for((13, 9))
    custom = {"13,9": [{"actions": seq}], "9,13": []}  # slot 1 disabled
    stub = StubHeuristic()
    mgr = OpeningHybridManager(heuristic=stub, openings=custom)
    a = mgr.ae(_obs(0, traj[0][0], traj[0][1], (13, 9)))
    assert a == seq[0], "enabled slot should play its opening"
    # disabled slot (empty list) -> immediate delegate
    stub2 = StubHeuristic()
    mgr2 = OpeningHybridManager(heuristic=stub2, openings=custom)
    a2 = mgr2.ae(_obs(0, (9, 14), 1, base=(9, 13)))
    assert a2 == STUB and mgr2.aborted, "disabled slot should delegate to planner"


def test_non_novice_base_delegates_immediately():
    stub = StubHeuristic()
    mgr = OpeningHybridManager(heuristic=stub)
    a = mgr.ae(_obs(0, (0, 0), 0, base=(0, 0)))  # base not in SLOT_TABLE
    assert a == STUB
    assert mgr.aborted


def test_frozen_during_opening_aborts():
    base = (13, 9)
    seq, traj = _opening_for(base)
    stub = StubHeuristic()
    mgr = OpeningHybridManager(heuristic=stub)
    mgr.ae(_obs(0, traj[0][0], traj[0][1], base))  # loads + plays opening[0]
    a = mgr.ae(_obs(1, traj[1][0], traj[1][1], base, frozen=2))  # frozen -> abort
    assert a == STUB and mgr.aborted


def test_keeps_heuristic_belief_warm_during_opening():
    """Heuristic .ae() is called every tick during the opening (belief stays warm)."""
    base = (9, 13)
    seq, traj = _opening_for(base)
    stub = StubHeuristic()
    mgr = OpeningHybridManager(heuristic=stub)
    for k in range(3):
        mgr.ae(_obs(k, traj[k][0], traj[k][1], base))
    assert stub.calls == 3, f"expected 3 warm calls, got {stub.calls}"


def test_integration_emits_opening_in_real_env():
    """Drive the wrapper as agent_0 in the real env (opponents STAY): with no
    interference it must emit exactly the opening sequence, validating that the
    wrapper reads the real env's obs format correctly."""
    from til_environment import bomberman_env
    from til_environment.config import default_config

    base = (13, 9)
    seq, _ = _opening_for(base)
    stub = StubHeuristic()
    mgr = OpeningHybridManager(heuristic=stub)

    cfg = default_config(); cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    env.reset(seed=2024)
    me = "agent_0"
    emitted = []
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        if term or trunc:
            env.step(None); continue
        n = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in obs.items()}
        if agent == me and len(emitted) < len(seq):
            a = mgr.ae(n)
            emitted.append(a)
            env.step(a)
        else:
            env.step(4)
        if len(emitted) >= len(seq):
            break
    assert emitted == seq, f"env-emitted opening != generated\n  got {emitted}\n  exp {seq}"


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e!r}")
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _run()
