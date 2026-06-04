"""Unit test for the BC action-agreement metric (clone fidelity gate)."""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "training" / "ae"))

from bc_action_agreement import agreement_and_lift


def test_perfect_clone_has_full_agreement_and_positive_lift():
    true = np.array([0, 0, 0, 1, 2, 3])  # modal action 0 -> freq 3/6 = 0.5
    pred = true.copy()                    # perfect clone
    agreement, modal, lift = agreement_and_lift(pred, true, n_actions=6)
    assert agreement == 1.0
    assert modal == 0.5
    assert lift == 0.5


def test_always_modal_clone_has_zero_lift():
    true = np.array([0, 0, 0, 1, 2, 3])
    pred = np.zeros_like(true)             # always predict the modal action
    agreement, modal, lift = agreement_and_lift(pred, true, n_actions=6)
    assert agreement == 0.5               # matches the 3 zeros
    assert modal == 0.5
    assert lift == 0.0
