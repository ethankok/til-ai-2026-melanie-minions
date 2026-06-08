"""Tests for the R2 phantom-bomb rollback (AE_CONFPOL_ROLLBACK_PHANTOM_BOMB).

Background: in `confidence_policy_hybrid`, the heuristic's full `ae()` runs every
tick BEFORE the confidence gate decides whether to override with the PPO policy.
When the heuristic commits to PLACE_BOMB via `_should_place_bomb`, it writes a
synthetic own-bomb into `known_bombs` (+ escape state) and returns PLACE_BOMB --
but `decision_path` stays `"target"` with the target-ranking margin. If that
margin is low, the wrapper overrides PLACE_BOMB with a policy *move*: the bomb is
never actually placed, yet the belief map now holds a phantom own-bomb that
`_danger_cells()` will route around for ~BOMB_TIMER ticks.

R2 fix (default OFF): when the wrapper overrides, revert exactly the bomb-commit
side effects the heuristic wrote this tick. Flag-off must be byte-identical.
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "ae" / "src"))

from ae_manager import AEManager
import confidence_policy_hybrid_manager as cph


# --------------------------------------------------------------------------
# Heuristic side: recording + revert_bomb_commit
# --------------------------------------------------------------------------
def _open_grid_manager():
    m = AEManager()
    m.seen = {(x, y) for x in range(16) for y in range(16)}
    m.grid_size = 16
    m.last_step = 5
    m.is_fixed_novice_map = False
    m.health = 60
    return m


def test_tick_bomb_commit_defaults_none():
    m = AEManager()
    assert m._tick_bomb_commit is None


def test_should_place_bomb_records_commit():
    m = _open_grid_manager()
    m.enemy_bases = {(2, 0): 5}      # inside the blast of (0,0)
    m.base_location = (15, 15)        # far away, not in blast
    obs = {"team_bombs": 1, "action_mask": [1, 1, 1, 1, 1, 1]}
    assert m._should_place_bomb(obs, (0, 0), None, set()) is True
    assert (0, 0) in m.known_bombs
    assert m.known_bombs[(0, 0)]["own"] is True
    commit = m._tick_bomb_commit
    assert commit is not None
    assert commit["cell"] == (0, 0)
    assert commit["prior_bomb"] is None


def test_revert_pops_synthetic_entry_and_nulls_escape():
    m = _open_grid_manager()
    m.enemy_bases = {(2, 0): 5}
    m.base_location = (15, 15)
    obs = {"team_bombs": 1, "action_mask": [1, 1, 1, 1, 1, 1]}
    assert m._should_place_bomb(obs, (0, 0), None, set()) is True
    assert (0, 0) in m.known_bombs
    assert m.escape_target is not None

    assert m.revert_bomb_commit() is True
    assert (0, 0) not in m.known_bombs       # synthetic entry removed
    assert m.escape_target is None           # escape commit reverted
    assert m.escape_until_step is None
    assert m._tick_bomb_commit is None        # record consumed


def test_revert_restores_prior_observed_bomb():
    m = _open_grid_manager()
    prior = {"timer": 2, "own": False, "last_step": 4}  # a real observed bomb
    m.known_bombs[(0, 0)] = prior
    m.enemy_bases = {(2, 0): 5}
    m.base_location = (15, 15)
    obs = {"team_bombs": 1, "action_mask": [1, 1, 1, 1, 1, 1]}
    assert m._should_place_bomb(obs, (0, 0), None, set()) is True
    # commit overwrote the prior entry with an own-bomb
    assert m.known_bombs[(0, 0)]["own"] is True

    assert m.revert_bomb_commit() is True
    assert m.known_bombs[(0, 0)] == prior     # prior restored, not popped


def test_revert_noop_without_commit():
    m = _open_grid_manager()
    m.known_bombs[(3, 3)] = {"timer": 2, "own": False, "last_step": 4}
    assert m._tick_bomb_commit is None
    assert m.revert_bomb_commit() is False    # nothing to revert
    assert (3, 3) in m.known_bombs            # untouched


def test_ae_resets_tick_commit_each_tick():
    m = _open_grid_manager()
    m._tick_bomb_commit = {"cell": (9, 9), "prior_bomb": None,
                           "prior_escape_target": None, "prior_escape_until_step": None}
    # A plain non-bomb tick on an empty open grid should not commit a bomb,
    # so the stale record must be cleared at the top of ae().
    obs = {"step": 6, "location": [0, 0], "direction": 0,
           "team_bombs": 0, "action_mask": [1, 1, 1, 1, 1, 0]}
    m.ae(obs)
    assert m._tick_bomb_commit is None


# --------------------------------------------------------------------------
# Wrapper side: gate + rollback wiring (stub the PPO policy)
# --------------------------------------------------------------------------
class _StubPolicy:
    """Stand-in for PolicyAEManager: always returns FORWARD, no torch needed."""

    def ae(self, observation):
        return AEManager.FORWARD


class _CommitHeuristic(AEManager):
    """A heuristic whose ae() simulates a low-confidence PLACE_BOMB commit."""

    def __init__(self, *, margin):
        super().__init__()
        self._sim_margin = margin

    def ae(self, observation):
        self.known_bombs = {(0, 0): {"timer": 3, "own": True, "last_step": 1}}
        self.escape_target = (0, 3)
        self.escape_until_step = 4
        self._tick_bomb_commit = {
            "cell": (0, 0), "prior_bomb": None,
            "prior_escape_target": None, "prior_escape_until_step": None,
        }
        self.last_decision_confidence = {
            "top_score": 40.0, "runner_up_score": 39.0, "margin": self._sim_margin,
            "n_candidates": 3, "decision_path": "target",
        }
        return AEManager.PLACE_BOMB


def _make_wrapper(monkeypatch, *, flag):
    if flag is None:
        monkeypatch.delenv("AE_CONFPOL_ROLLBACK_PHANTOM_BOMB", raising=False)
    else:
        monkeypatch.setenv("AE_CONFPOL_ROLLBACK_PHANTOM_BOMB", flag)
    monkeypatch.setattr(cph, "PolicyAEManager", _StubPolicy)
    return cph.ConfidencePolicyHybridAEManager()


def test_wrapper_flag_default_off_and_override(monkeypatch):
    w = _make_wrapper(monkeypatch, flag=None)
    assert w.rollback_phantom_bomb is False
    w2 = _make_wrapper(monkeypatch, flag="1")
    assert w2.rollback_phantom_bomb is True


def test_wrapper_flag_off_leaves_phantom(monkeypatch):
    w = _make_wrapper(monkeypatch, flag=None)
    w.heuristic = _CommitHeuristic(margin=1.0)   # low margin -> override
    action = w.ae({})
    assert action == AEManager.FORWARD            # policy overrode
    assert (0, 0) in w.heuristic.known_bombs       # phantom NOT reverted (flag off)


def test_wrapper_flag_on_reverts_phantom_on_override(monkeypatch):
    w = _make_wrapper(monkeypatch, flag="1")
    w.heuristic = _CommitHeuristic(margin=1.0)   # low margin -> override
    action = w.ae({})
    assert action == AEManager.FORWARD            # policy overrode
    assert (0, 0) not in w.heuristic.known_bombs   # phantom reverted
    assert w.heuristic.escape_target is None


def test_wrapper_flag_on_no_revert_when_confident(monkeypatch):
    w = _make_wrapper(monkeypatch, flag="1")
    w.heuristic = _CommitHeuristic(margin=99.0)  # high margin -> confident, no override
    action = w.ae({})
    assert action == AEManager.PLACE_BOMB         # heuristic action kept
    assert (0, 0) in w.heuristic.known_bombs       # real bomb, not reverted
