"""Tests for the offline opening generator (OpeningSim + beam search).

Plain-assert runner (no pytest in the venv). Run:
    PYTHONHASHSEED=0 .venv/bin/python training/ae/test_opening_sim.py

The parity test against the real til_environment is the spine: if OpeningSim
agrees with the env on (position, direction, cumulative reward) for a random
movement sequence, then the movement physics, wall data, item positions, and
item reward values are all correct simultaneously.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") is None:
    os.environ["PYTHONHASHSEED"] = "0"
    os.execvp(sys.executable, [sys.executable, *sys.argv])

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "ae" / "src"), str(ROOT / "til-26-ae"), str(ROOT / "training" / "ae")):
    if p not in sys.path:
        sys.path.insert(0, p)

from opening_sim import OpeningSim, SLOT_TABLE, ITEM_VALUE, beam_search  # noqa: E402

FORWARD, BACKWARD, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4
MOVES = [FORWARD, BACKWARD, LEFT, RIGHT, STAY]


# ---------------------------------------------------------------------------
# OpeningSim unit tests
# ---------------------------------------------------------------------------
def test_initial_state_per_slot():
    """Each slot's start pos/dir is exposed and well-formed."""
    sim = OpeningSim()
    assert len(SLOT_TABLE) == 6
    for base, (pos, d) in SLOT_TABLE.items():
        st = sim.initial_state(base)
        assert st.pos == tuple(pos)
        assert st.dir == d
        assert st.reward == 0.0
        assert st.tick == 0
        assert len(st.collected) == 0


def test_left_right_turn_in_place():
    """LEFT/RIGHT rotate facing (dir+3 / dir+1 mod 4); position unchanged."""
    sim = OpeningSim()
    st = sim.initial_state((13, 9))  # pos (14,9), dir 0
    r = sim.step(st, RIGHT)
    assert r.pos == st.pos and r.dir == (st.dir + 1) % 4
    l = sim.step(st, LEFT)
    assert l.pos == st.pos and l.dir == (st.dir + 3) % 4


def test_wall_blocks_forward():
    """FORWARD into a known blocking edge does not change position."""
    sim = OpeningSim()
    # Find a cell with a blocking edge and an in-bounds neighbour behind it.
    (x, y, d) = next(iter(sim.blocking_edges))
    from opening_sim import State
    st = State(pos=(x, y), dir=d, collected=frozenset(), reward=0.0, tick=0)
    moved = sim.step(st, FORWARD)
    assert moved.pos == (x, y), f"expected blocked at {(x,y)} dir {d}, moved to {moved.pos}"


def test_item_pickup_once():
    """Entering an item tile collects it once; revisiting does not re-score."""
    sim = OpeningSim()
    from opening_sim import State
    # Pick an item with a free (non-blocked, in-bounds) neighbour to step from.
    for (ix, iy), kind in sim.items.items():
        for d, (dx, dy) in enumerate(sim.DELTAS):
            nx, ny = ix - dx, iy - dy
            if not (0 <= nx < 16 and 0 <= ny < 16):
                continue
            # Neighbour must not itself be an item, else the revisit's BACKWARD
            # step would legitimately collect it and confound the check.
            if (nx, ny) in sim.items:
                continue
            # step from (nx,ny) facing d should land on (ix,iy)
            st = State(pos=(nx, ny), dir=d, collected=frozenset(), reward=0.0, tick=0)
            if sim._blocked((nx, ny), d):
                continue
            after = sim.step(st, FORWARD)
            if after.pos != (ix, iy):
                continue
            assert (ix, iy) in after.collected
            assert after.reward == ITEM_VALUE[kind]
            # revisit: step back and forth, reward must not increase again
            back = sim.step(after, BACKWARD)
            again = sim.step(back, FORWARD)
            assert again.reward == after.reward
            return
    raise AssertionError("no testable item with a free neighbour found")


def test_parity_with_env():
    """OpeningSim matches the real env on (pos, dir, cumulative reward)."""
    from til_environment import bomberman_env
    from til_environment.config import default_config

    base_by_agent = {f"agent_{i}": tuple(b) for i, b in enumerate(SLOT_TABLE.keys())}
    rng = random.Random(20260601)
    HORIZON = 30

    for trial in range(3):
        seq = [rng.choice(MOVES) for _ in range(HORIZON)]
        cfg = default_config()
        cfg.env.novice = True
        env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
        env.reset(seed=1000 + trial)

        sim = OpeningSim()
        # one OpeningSim state per agent, advanced when that agent acts
        sim_state = {}
        cum_env_reward = {f"agent_{i}": 0.0 for i in range(6)}
        turn_idx = {f"agent_{i}": 0 for i in range(6)}

        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            if term or trunc:
                env.step(None)
                continue
            n = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in obs.items()}
            base = base_by_agent[agent]
            if agent not in sim_state:
                sim_state[agent] = sim.initial_state(base)
            cum_env_reward[agent] += float(reward)
            k = turn_idx[agent]
            # Compare BEFORE applying this turn's action: env obs == sim state.
            ss = sim_state[agent]
            assert tuple(n["location"]) == ss.pos, (
                f"trial{trial} {agent} turn{k}: env pos {tuple(n['location'])} != sim {ss.pos}")
            assert int(n["direction"]) == ss.dir, (
                f"trial{trial} {agent} turn{k}: env dir {n['direction']} != sim {ss.dir}")
            assert abs(cum_env_reward[agent] - ss.reward) < 1e-6, (
                f"trial{trial} {agent} turn{k}: env reward {cum_env_reward[agent]} != sim {ss.reward}")
            if k >= HORIZON:
                env.step(STAY)
                continue
            a = seq[k]
            sim_state[agent] = sim.step(ss, a)
            turn_idx[agent] = k + 1
            env.step(a)


# ---------------------------------------------------------------------------
# Beam search tests
# ---------------------------------------------------------------------------
def test_beam_respects_width_and_horizon():
    sim = OpeningSim()
    cands = beam_search(sim, (13, 9), horizon=8, beam_width=50, top_k=5)
    assert 1 <= len(cands) <= 5
    for c in cands:
        assert len(c["actions"]) == 8
        assert c["end_pos"] is not None


def test_beam_finds_reachable_reward():
    """With a horizon long enough to reach items, the best candidate scores > 0."""
    sim = OpeningSim()
    cands = beam_search(sim, (13, 9), horizon=20, beam_width=500, top_k=3)
    assert cands[0]["reward"] > 0.0
    # ranked descending by reward
    rewards = [c["reward"] for c in cands]
    assert rewards == sorted(rewards, reverse=True)


def test_beam_replay_matches_sim():
    """Replaying a candidate's action list through OpeningSim reproduces its reward."""
    sim = OpeningSim()
    c = beam_search(sim, (9, 13), horizon=16, beam_width=300, top_k=1)[0]
    st = sim.initial_state((9, 13))
    for a in c["actions"]:
        st = sim.step(st, a)
    assert abs(st.reward - c["reward"]) < 1e-6
    assert st.pos == c["end_pos"]


def test_generate_openings_structure():
    from gen_openings import generate
    result = generate(horizons=[8, 12], beam_width=200, top_k=3, position_weights=[0.0, 0.5])
    assert set(result.keys()) == {f"{bx},{by}" for (bx, by) in SLOT_TABLE}
    for slot, cands in result.items():
        assert 1 <= len(cands) <= 3, f"{slot}: {len(cands)} candidates"
        for c in cands:
            assert set(c) >= {"horizon", "actions", "reward", "end_pos", "items"}
            assert len(c["actions"]) == c["horizon"]


def test_generate_candidates_replay_truthfully():
    """Every emitted candidate reproduces its stated reward/end_pos via OpeningSim."""
    from gen_openings import generate
    sim = OpeningSim()
    result = generate(horizons=[12], beam_width=200, top_k=2, position_weights=[0.0])
    for slot, cands in result.items():
        bx, by = (int(v) for v in slot.split(","))
        for c in cands:
            st = sim.initial_state((bx, by))
            for a in c["actions"]:
                st = sim.step(st, a)
            assert abs(st.reward - c["reward"]) < 1e-6
            assert list(st.pos) == list(c["end_pos"])


def _env_replay_reward(base_idx: int, base: tuple, seq: list[int]) -> float:
    """Clean lockstep replay of an action sequence through the real env.

    Drives agent_<base_idx> with `seq` (others STAY) and returns its cumulative
    reward, bagging the final move's reward by running one extra turn (AEC
    rewards land on the agent's NEXT turn — the source of an earlier false alarm).
    """
    from til_environment import bomberman_env
    from til_environment.config import default_config
    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    env.reset(seed=777)
    me = f"agent_{base_idx}"
    cum = 0.0
    taken = 0
    extra_done = False
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        if term or trunc:
            env.step(None)
            continue
        if agent == me:
            cum += float(reward)
            if taken < len(seq):
                env.step(seq[taken]); taken += 1
            elif not extra_done:
                extra_done = True  # this turn bagged the final move's reward
                break
            else:
                env.step(STAY)
        else:
            env.step(STAY)
    return cum


def test_generated_openings_reproduce_in_env():
    """Dense generated openings reproduce their claimed reward in the real env.

    Regression for the reward-bagging false alarm: the sparse random-walk parity
    test rarely lands on items, so this stresses the dense item-collection path
    end-to-end against the official env.
    """
    from gen_openings import generate
    bases = list(SLOT_TABLE.keys())
    result = generate(horizons=[12], beam_width=400, top_k=1, position_weights=[0.0])
    for i, base in enumerate(bases):
        cand = result[f"{base[0]},{base[1]}"][0]
        env_reward = _env_replay_reward(i, base, cand["actions"])
        assert abs(env_reward - cand["reward"]) < 1e-6, (
            f"slot {base}: claimed {cand['reward']} env {env_reward}")


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
