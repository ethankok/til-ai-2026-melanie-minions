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

from foreign_opponents import CBOMB7_ENV  # noqa: E402
from melee_eval import _evaluate_candidate  # noqa: E402


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


RankKey = tuple[float, float]

# The promotion guard is applied ONLY to the low-variance, Semis-representative
# brackets. The synthetic high-variance `adversarial` probe (and the worst-bracket
# max / min-margin it dominates) is NOT gated: gating on it would reject genuinely
# better candidates that drew an unlucky adversarial run. FIELD_GUARD_TOL absorbs the
# residual noise on these stable brackets while still catching a real collapse.
FIELD_GUARD_BRACKETS: tuple[str, ...] = ("semis_mixed", "real_field")
FIELD_GUARD_TOL = 0.25
_EPS = 1e-9


def _mean_place(result: dict[str, Any]) -> float:
    brackets = result.get("per_bracket", {})
    if not brackets:
        return float("inf")
    return float(np.mean([float(b["mean_placement"]) for b in brackets.values()]))


def _bracket_place(result: dict[str, Any], bracket: str) -> float | None:
    b = result.get("per_bracket", {}).get(bracket)
    return float(b["mean_placement"]) if b else None


def _field_regression(candidate: dict[str, Any], incumbent: dict[str, Any]) -> float:
    """Total placement regression on the guarded field brackets (0.0 if none)."""

    total = 0.0
    for bracket in FIELD_GUARD_BRACKETS:
        cand = _bracket_place(candidate, bracket)
        inc = _bracket_place(incumbent, bracket)
        if cand is not None and inc is not None:
            total += max(0.0, cand - inc)
    return total


def rank_key(candidate: dict[str, Any], incumbent: dict[str, Any]) -> RankKey:
    """Lexicographic minimization key for CEM elite selection. Lower is better.

    MEAN bracket placement is the PRIMARY objective: the low-variance, most direct
    estimator of how the candidate places against the field. The synthetic
    worst-bracket max is deliberately NOT used here (it is dominated by the noisy
    `adversarial` probe and makes a poor optimizer signal). The secondary term is the
    field-bracket regression vs the incumbent, so the search is gently biased away
    from buying a better mean by sacrificing the Semis-representative brackets. The
    promotion guard lives in `promotion_ok`.
    """

    return (_mean_place(candidate), _field_regression(candidate, incumbent))


def promotion_ok(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    """Promote only if the candidate STRICTLY improves mean placement AND does not
    regress (beyond FIELD_GUARD_TOL) on the low-variance field brackets
    (semis_mixed / real_field), relative to the same-seed incumbent. Worst-bracket /
    adversarial / margin are diagnostics only and are deliberately not gated."""

    if not (_mean_place(candidate) < _mean_place(incumbent) - _EPS):
        return False
    for bracket in FIELD_GUARD_BRACKETS:
        cand = _bracket_place(candidate, bracket)
        inc = _bracket_place(incumbent, bracket)
        if cand is None or inc is None:
            return False  # fail closed if a guarded bracket was not evaluated
        if cand > inc + FIELD_GUARD_TOL + _EPS:
            return False
    return True


def cem_update(
    old_mu: np.ndarray,
    old_sigma: np.ndarray,
    elite_vectors: np.ndarray,
    *,
    smoothing: float,
    min_sigma: float,
    max_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Update CEM Gaussian parameters from elite encoded vectors."""

    if elite_vectors.ndim != 2:
        raise ValueError(f"elite_vectors must be 2D, got shape {elite_vectors.shape}")
    elite_mu = elite_vectors.mean(axis=0)
    elite_sigma = elite_vectors.std(axis=0)
    new_mu = (1.0 - smoothing) * old_mu + smoothing * elite_mu
    new_sigma = (1.0 - smoothing) * old_sigma + smoothing * elite_sigma
    new_sigma = np.clip(new_sigma, min_sigma, max_sigma)
    return new_mu.astype(float), new_sigma.astype(float)


def build_candidate_spec(values: dict[str, float], *, policy_ckpt: str) -> dict[str, Any]:
    """Build a melee_eval candidate spec for one planner-weight vector."""

    env: dict[str, str] = {}
    env.update(CBOMB7_ENV)
    env.update(FIXED_CANDIDATE_ENV)
    env["AE_POLICY_CHECKPOINT"] = str(Path(policy_ckpt).resolve())
    env.update(encode_env(values))
    return {"our": "confidence_policy_hybrid", "env": env}


def evaluate_values(
    label: str,
    values: dict[str, float],
    *,
    policy_ckpt: str,
    brackets: list[str],
    hash_seeds: list[int],
    sim_seeds: list[int],
    rounds: int,
    non_novice: bool,
) -> dict[str, Any]:
    """Evaluate one decoded parameter vector using the existing melee harness."""

    spec = build_candidate_spec(values, policy_ckpt=policy_ckpt)
    return _evaluate_candidate(
        label,
        spec,
        brackets,
        hash_seeds,
        sim_seeds,
        rounds,
        non_novice,
    )


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON record to a JSONL artifact file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records; return an empty list when the file does not exist."""

    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            records.append(json.loads(stripped))
    return records


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write a stable pretty JSON summary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
