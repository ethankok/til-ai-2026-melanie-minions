"""Black-box CEM tuner for AE planner scalar weights.

The tuner preserves the shipped heuristic/confpol planner core and searches only
small scalar env vars that AEManager already reads. Candidate quality is measured
by the existing finals-aligned melee gate: raw_ae (our cumulative reward, the
finals-proportional discriminator) is the optimizer objective, subject to a
worst_robust_placement floor — all relative to the same-seed incumbent, rather than
cloud absolute reward. The expensive held-out-composition gap + hardware A/B gate
the single finalist, not every in-loop candidate.
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

# Finals-aligned objective (2026-06-09 gate): raw_ae (our agent's cumulative reward,
# proportional to final score since our mission_multiplier is fixed) is the
# DISCRIMINATOR the CEM maximizes, subject to a worst_robust_placement FLOOR. Because
# our mult is high+fixed, robust placement saturates at 1st against a weak field, so
# it acts as a non-exploitability floor (a real regression beyond PLACEMENT_FLOOR_TOL
# is penalized) while raw_ae separates good candidates. This deliberately does NOT
# optimize mean placement (the reverted g00-fixed-03 trap: a placement-proxy that
# overfit the synthetic pool); the held-out composition gap + a hardware A/B gate the
# single finalist, not every in-loop candidate.
PLACEMENT_FLOOR_TOL = 0.25
_EPS = 1e-9


def _raw_ae(result: dict[str, Any]) -> float:
    return float(result.get("raw_ae", float("-inf")))


def _worst_robust(result: dict[str, Any]) -> float:
    return float(result.get("worst_robust_placement", float("inf")))


def _placement_violation(candidate: dict[str, Any], incumbent: dict[str, Any]) -> float:
    """Robust-placement regression beyond the noise tolerance (0.0 if floor held)."""

    return max(0.0, _worst_robust(candidate) - _worst_robust(incumbent) - PLACEMENT_FLOOR_TOL)


def rank_key(candidate: dict[str, Any], incumbent: dict[str, Any]) -> RankKey:
    """Lexicographic minimization key for CEM elite selection. Lower is better.

    PRIMARY (floor) = robust-placement violation vs the incumbent: any candidate that
    regresses ``worst_robust_placement`` beyond ``PLACEMENT_FLOOR_TOL`` sorts strictly
    worse, keeping the search inside the non-exploitable region. DISCRIMINATOR =
    ``-raw_ae`` (maximize our finals-proportional cumulative reward) among candidates
    that hold the floor. raw_ae is our documented WEAK axis and the thing that decides
    rank against a real field, so it is what the optimizer chases.
    """

    return (_placement_violation(candidate, incumbent), -_raw_ae(candidate))


def promotion_ok(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    """Promote only if the candidate STRICTLY improves raw_ae (the finals-proportional
    discriminator) AND holds the robust-placement floor (worst_robust_placement does
    not regress beyond PLACEMENT_FLOOR_TOL) vs the same-seed incumbent. The rigorous
    paired effect / Probability-of-Improvement / held-out-composition-gap checks are
    applied to the single FINALIST via ``melee_eval --heldout`` (+ hardware A/B), not
    to every in-loop candidate (too noisy at 4 rounds)."""

    if not (_raw_ae(candidate) > _raw_ae(incumbent) + _EPS):
        return False
    if _worst_robust(candidate) > _worst_robust(incumbent) + PLACEMENT_FLOOR_TOL + _EPS:
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


def _dedupe_fixed_values(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    seen: set[tuple[tuple[str, float], ...]] = set()
    out: list[dict[str, float]] = []
    for values in rows:
        key = tuple(sorted((name, round(float(value), 9)) for name, value in values.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(values)
    return out


def sample_generation(
    rng: np.random.Generator,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    pop_size: int,
    include_incumbent: bool,
    best_values: dict[str, float] | None,
    fixed_values: list[dict[str, float]],
) -> list[tuple[str, dict[str, float], np.ndarray]]:
    """Sample one CEM generation as `(label, decoded_values, encoded_vector)` rows."""

    rows: list[tuple[str, dict[str, float], np.ndarray]] = []
    fixed_rows: list[tuple[str, dict[str, float]]] = []
    if include_incumbent:
        fixed_rows.append(("incumbent", incumbent_values()))
    if best_values is not None:
        fixed_rows.append(("best", best_values))
    for i, values in enumerate(_dedupe_fixed_values([v for _label, v in fixed_rows] + fixed_values)):
        label = "incumbent" if values == incumbent_values() else "best" if best_values is not None and values == best_values else f"fixed-{i:02d}"
        rows.append((label, values, encode_values(values)))
        if len(rows) >= pop_size:
            return rows
    while len(rows) < pop_size:
        idx = len(rows)
        vector = rng.normal(mu, sigma).astype(float)
        values = decode_vector(vector)
        rows.append((f"sample-{idx:02d}", values, encode_values(values)))
    return rows


def values_from_summary(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload["best"]["values"]
    return {p.name: float(values[p.name]) for p in PARAMS}


def top_values_from_jsonl(path: Path, n: int) -> list[dict[str, float]]:
    rows = [r for r in load_jsonl(path) if r.get("rank_key") is not None]
    rows.sort(key=lambda r: tuple(r["rank_key"]))
    return [{p.name: float(r["values"][p.name]) for p in PARAMS} for r in rows[:n]]


def docker_env_lines(values: dict[str, float]) -> list[str]:
    """Return Dockerfile ENV lines for the tuned scalar params only."""

    env = encode_env(values)
    return [f"ENV {p.name}={env[p.name]}" for p in PARAMS]


def _parse_int_list(raw: str) -> list[int]:
    return [int(part) for part in raw.replace(",", " ").split() if part]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=THIS_DIR / "data" / "planner-weight-cem")
    p.add_argument("--policy-ckpt", default=DEFAULT_POLICY_CKPT)
    p.add_argument("--generations", type=int, default=4)
    p.add_argument("--pop-size", type=int, default=12)
    p.add_argument("--elite-count", type=int, default=3)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--hash-seeds", default="0")
    p.add_argument("--sim-seeds", default="42")
    p.add_argument("--brackets", nargs="+", default=None)
    p.add_argument("--seed", type=int, default=20260606)
    p.add_argument("--center-summary", type=Path, default=None,
                   help="summary.json from a prior run; its best values become the new CEM mean")
    p.add_argument("--include-top-from", type=Path, default=None,
                   help="candidates.jsonl from a prior run; top values are evaluated as fixed candidates")
    p.add_argument("--include-top-n", type=int, default=0)
    p.add_argument("--init-sigma", type=float, default=0.35)
    p.add_argument("--smoothing", type=float, default=0.70)
    p.add_argument("--min-sigma", type=float, default=0.05)
    p.add_argument("--max-sigma", type=float, default=0.80)
    p.add_argument("--non-novice", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="sample and write config without melee evaluation")
    return p


def _record_for_candidate(
    *,
    generation: int,
    label: str,
    values: dict[str, float],
    vector: np.ndarray,
    result: dict[str, Any] | None,
    incumbent_result: dict[str, Any] | None,
    elapsed_s: float,
) -> dict[str, Any]:
    key = None
    if result is not None and incumbent_result is not None:
        key = list(rank_key(result, incumbent_result))
    return {
        "generation": generation,
        "candidate": label,
        "values": values,
        "vector": [float(x) for x in vector],
        "rank_key": key,
        "result": result,
        "elapsed_s": elapsed_s,
    }


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    """Return a JSON-serializable view of parsed args.

    `argparse` stores `--out-dir`, `--center-summary`, and `--include-top-from`
    as `Path` objects; `json.dumps` cannot serialize those. Stringify every
    `Path` value so the per-generation summary write never raises on the staged
    runs (Stage-1/final/held-out) that actually pass those path flags.
    """

    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}


def run_cem(args: argparse.Namespace) -> dict[str, Any]:
    """Run the CEM search and return the final summary payload."""

    from opponents import MELEE_BRACKETS  # noqa: WPS433

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.out_dir / "candidates.jsonl"
    summary_path = args.out_dir / "summary.json"
    brackets = args.brackets or list(MELEE_BRACKETS)
    hash_seeds = _parse_int_list(args.hash_seeds)
    sim_seeds = _parse_int_list(args.sim_seeds)

    rng = np.random.default_rng(args.seed)
    initial_values = values_from_summary(args.center_summary) if args.center_summary else incumbent_values()
    fixed_values = top_values_from_jsonl(args.include_top_from, args.include_top_n) if args.include_top_from and args.include_top_n > 0 else []
    mu = encode_values(initial_values)
    sigma = np.full(len(PARAMS), float(args.init_sigma), dtype=float)
    best_values: dict[str, float] | None = None if initial_values == incumbent_values() else initial_values
    best_record: dict[str, Any] | None = None
    generation_summaries: list[dict[str, Any]] = []

    for generation in range(args.generations):
        rows = sample_generation(
            rng,
            mu,
            sigma,
            pop_size=args.pop_size,
            include_incumbent=True,
            best_values=best_values,
            fixed_values=fixed_values,
        )
        evaluated: list[dict[str, Any]] = []
        incumbent_result: dict[str, Any] | None = None

        for label, values, vector in rows:
            candidate_label = "incumbent" if label == "incumbent" else f"g{generation:02d}-{label}"
            started = time.monotonic()
            if args.dry_run:
                result = None
            else:
                result = evaluate_values(
                    candidate_label,
                    values,
                    policy_ckpt=args.policy_ckpt,
                    brackets=brackets,
                    hash_seeds=hash_seeds,
                    sim_seeds=sim_seeds,
                    rounds=args.rounds,
                    non_novice=args.non_novice,
                )
            elapsed_s = time.monotonic() - started
            if label == "incumbent":
                incumbent_result = result
            record = _record_for_candidate(
                generation=generation,
                label=candidate_label,
                values=values,
                vector=vector,
                result=result,
                incumbent_result=incumbent_result,
                elapsed_s=elapsed_s,
            )
            evaluated.append(record)
            append_jsonl(records_path, record)

        if args.dry_run:
            elite_vectors = np.array([row[2] for row in rows[: args.elite_count]], dtype=float)
            best_record = evaluated[0]
        else:
            if incumbent_result is None:
                raise RuntimeError("incumbent result missing from generation")
            for record in evaluated:
                if record["rank_key"] is None and record["result"] is not None:
                    record["rank_key"] = list(rank_key(record["result"], incumbent_result))
            ranked = sorted(evaluated, key=lambda r: tuple(r["rank_key"]))
            elites = ranked[: args.elite_count]
            elite_vectors = np.array([np.array(r["vector"], dtype=float) for r in elites], dtype=float)
            best_record = ranked[0] if best_record is None else min(
                [best_record, ranked[0]],
                key=lambda r: tuple(r["rank_key"]) if r["rank_key"] is not None else (float("inf"),),
            )
            best_values = {p.name: float(best_record["values"][p.name]) for p in PARAMS}

        mu, sigma = cem_update(
            mu,
            sigma,
            elite_vectors,
            smoothing=args.smoothing,
            min_sigma=args.min_sigma,
            max_sigma=args.max_sigma,
        )
        generation_summaries.append({
            "generation": generation,
            "mu": [float(x) for x in mu],
            "sigma": [float(x) for x in sigma],
            "best_candidate": best_record["candidate"] if best_record else None,
            "best_rank_key": best_record.get("rank_key") if best_record else None,
        })
        write_summary(summary_path, {
            "args": jsonable_args(args),
            "brackets": brackets,
            "hash_seeds": hash_seeds,
            "initial_values": initial_values,
            "fixed_values": fixed_values,
            "sim_seeds": sim_seeds,
            "generations": generation_summaries,
            "best": best_record,
            "docker_env": docker_env_lines(best_record["values"]) if best_record else [],
            "records_path": str(records_path),
        })

    return json.loads(summary_path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.elite_count <= 0 or args.elite_count > args.pop_size:
        raise SystemExit("--elite-count must be between 1 and --pop-size")
    summary = run_cem(args)
    print(json.dumps({
        "summary": str(args.out_dir / "summary.json"),
        "best_candidate": summary.get("best", {}).get("candidate"),
        "best_rank_key": summary.get("best", {}).get("rank_key"),
        "docker_env": summary.get("docker_env", []),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
