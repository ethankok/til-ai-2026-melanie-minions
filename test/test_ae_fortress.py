"""Tests for AE_FORTRESS posture + the experimental-flag de-contamination
(both AE_STUN_TAX and AE_FORTRESS must be forced OFF on opponent proxies)."""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "ae" / "src"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "training" / "ae"))

from ae_manager import AEManager


def test_self_heuristic_opponent_forces_experimental_flags_off(monkeypatch):
    monkeypatch.setenv("AE_STUN_TAX", "1")
    monkeypatch.setenv("AE_FORTRESS", "1")
    from foreign_opponents import SelfHeuristicOpponent
    op = SelfHeuristicOpponent()
    assert op._mgr.stun_tax_enabled is False
    assert op._mgr.fortress_enabled is False
