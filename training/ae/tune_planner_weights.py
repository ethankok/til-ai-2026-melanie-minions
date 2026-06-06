"""Black-box CEM tuner for AE planner scalar weights.

The tuner preserves the shipped heuristic/confpol planner core and searches only
small scalar env vars that AEManager already reads. Candidate quality is measured
by the existing melee gate: MEAN bracket placement is the optimizer objective,
guarded at promotion time only against regression on the low-variance field
brackets (semis_mixed/real_field) — the noisy adversarial probe is logged, not
gated — all relative to the same-seed incumbent, rather than cloud absolute reward.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
for _p in (str(THIS_DIR),):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@dataclass(frozen=True)
class ParamSpec:
    """One scalar planner env var in the CEM search space."""

    name: str
    default: float
    lower: float
    upper: float
    transform: str

    def encode(self, value: float) -> float:
        clipped = min(self.upper, max(self.lower, float(value)))
        if self.transform == "log":
            return math.log(clipped / self.default)
        if self.transform == "linear":
            span = self.upper - self.lower
            return (clipped - self.default) / span
        raise ValueError(f"unknown transform for {self.name}: {self.transform}")

    def decode(self, z_value: float) -> float:
        if self.transform == "log":
            value = self.default * math.exp(float(z_value))
        elif self.transform == "linear":
            value = self.default + float(z_value) * (self.upper - self.lower)
        else:
            raise ValueError(f"unknown transform for {self.name}: {self.transform}")
        return min(self.upper, max(self.lower, float(value)))


PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec("AE_ITEM_MISSION_VALUE", 80.0, 50.0, 120.0, "log"),
    ParamSpec("AE_ITEM_RESOURCE_VALUE", 40.0, 5.0, 70.0, "log"),
    ParamSpec("AE_ENEMY_BASE_VALUE", 100.0, 60.0, 160.0, "log"),
    ParamSpec("AE_DIST_PENALTY", 1.15, 0.70, 2.00, "log"),
    ParamSpec("AE_PATH_THREAT_PENALTY", 2.0, 0.50, 5.00, "log"),
    ParamSpec("AE_DIJKSTRA_BOMB_COST", 7.0, 3.00, 12.00, "log"),
    ParamSpec("AE_LEAD_TETHER_HEALTH", 60.0, 40.0, 85.0, "linear"),
    ParamSpec("AE_LEAD_TETHER_WEIGHT", 0.5, 0.00, 1.50, "linear"),
    ParamSpec("AE_CONTENTION_SCALE", 2.5, 0.50, 6.00, "log"),
    ParamSpec("AE_CONTENTION_PFLOOR", 0.15, 0.00, 0.50, "linear"),
)


FIXED_CANDIDATE_ENV: dict[str, str] = {
    "AE_MODE": "confidence_policy_hybrid",
    "AE_CONTENTION": "1",
    "AE_PLAN_RESCORE": "0",
    "AE_CONFPOL_MARGIN_EPSILON": "5.0",
    "AE_CONFPOL_TOP_FLOOR": "10.0",
    "AE_CONFPOL_OVERRIDE_TARGET_NONE": "1",
}


DEFAULT_POLICY_CKPT = str((THIS_DIR / "checkpoints" / "confpol-semis2b-u75.pt").resolve())


def incumbent_values() -> dict[str, float]:
    """Deployed scalar defaults at the center of the search."""

    return {p.name: p.default for p in PARAMS}


def default_vector() -> np.ndarray:
    """Encoded incumbent vector."""

    return encode_values(incumbent_values())


def encode_values(values: dict[str, float]) -> np.ndarray:
    """Encode concrete env values into optimizer coordinates."""

    return np.array([p.encode(values[p.name]) for p in PARAMS], dtype=float)


def decode_vector(vector: np.ndarray) -> dict[str, float]:
    """Decode optimizer coordinates into bounded concrete env values."""

    arr = np.asarray(vector, dtype=float)
    if arr.shape != (len(PARAMS),):
        raise ValueError(f"expected vector shape {(len(PARAMS),)}, got {arr.shape}")
    return {p.name: p.decode(float(arr[i])) for i, p in enumerate(PARAMS)}


def encode_env(values: dict[str, float]) -> dict[str, str]:
    """Format concrete values as stable strings for subprocess env vars."""

    return {p.name: f"{float(values[p.name]):.6f}" for p in PARAMS}
