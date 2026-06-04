"""Head-to-head melee eval + minimax-plus-margin selection gate (AE Semifinals).

This is the realistic Semis gate: it scores candidate agents in the four 6-team
melee brackets (``opponents.MELEE_BRACKETS``), against the FOREIGN (non-mirror)
opponent pool, and reports **placement + margin + absolute reward** side by side
per bracket — not absolute reward vs our own mirror heuristics (the old
``validate_cloud_suite.py`` gate that mis-predicts Semis performance).

See docs/superpowers/specs/2026-06-01-ae-semis-eval-design.md.

Determinism: AEManager has hash-order-dependent branches, so — exactly like
``multi_seed_eval.py`` — we spawn a fresh interpreter per (hash_seed, sim_seed)
with ``PYTHONHASHSEED`` pinned, run ONE bracket per worker, and aggregate across
runs. One bracket per subprocess also isolates the per-process model cache
(``policy_manager._MODEL_CACHE``) to a single candidate + that bracket's
opponents.

Promotion rule (report all three; gate on the conjunction) — a candidate is
promotable over the incumbent (default ``confpol-u860``) only if ALL hold:
  1. worst-bracket ``mean_placement`` (minimax over the 4 brackets) is
     better-or-equal to the incumbent's worst-bracket placement;
  2. ``mean_margin >= 0`` in every bracket;
  3. ``mean_score`` (absolute reward) in ``semis_mixed`` >= incumbent.

Usage
-----
    # Re-rank the default candidates (incumbent + heuristic + raw policy):
    .venv/bin/python training/ae/melee_eval.py --rounds 12 \\
        --hash-seeds 0 1 2 --sim-seeds 42 137 \\
        --summary-out training/ae/data/melee-rerank.json

    # Add Pandemonium / confpol-native ladder rungs (each a confpol wrapper over
    # a different raw-policy checkpoint):
    .venv/bin/python training/ae/melee_eval.py \\
        --confpol-ckpt native-u200=training/ae/checkpoints/confpol-native-u200.pt \\
        --confpol-ckpt native-u360=training/ae/checkpoints/confpol-native-u360.pt
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
for _p in (str(THIS_DIR),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from opponents import MELEE_BRACKETS  # noqa: E402

U860 = str((THIS_DIR / "checkpoints" / "pandemonium-v1-best-u860.pt").resolve())


def _cbomb7_env() -> dict[str, str]:
    from foreign_opponents import CBOMB7_ENV
    return dict(CBOMB7_ENV)


# label=path[@eps=E,floor=F,ovr=O] -> (label, env). The optional @-suffix sets the
# confpol gate knobs for THIS candidate so one sweep registers the same checkpoint
# at several consult thresholds. No @-suffix == checkpoint only (old behavior).
_GATE_KEYS = {
    "eps": "AE_CONFPOL_MARGIN_EPSILON",
    "floor": "AE_CONFPOL_TOP_FLOOR",
    "ovr": "AE_CONFPOL_OVERRIDE_TARGET_NONE",
    "override": "AE_CONFPOL_OVERRIDE_TARGET_NONE",
}


def _parse_confpol_spec(spec_str: str) -> tuple[str, dict[str, str]]:
    label, _, rest = spec_str.partition("=")
    path_part, _, gate_part = rest.partition("@")
    env: dict[str, str] = {"AE_POLICY_CHECKPOINT": str(Path(path_part).resolve())}
    for kv in filter(None, gate_part.split(",")):
        key, _, val = kv.partition("=")
        env_name = _GATE_KEYS.get(key.strip().lower())
        if env_name is None:
            raise ValueError(f"unknown gate knob {key!r} in {spec_str!r}")
        env[env_name] = val.strip()
    return label, env


# label -> {"our": <simulate _make_our_agent name>, "env": {extra env vars}}
def _base_registry() -> dict[str, dict]:
    return {
        "confpol-u860":     {"our": "confidence_policy_hybrid", "env": {"AE_POLICY_CHECKPOINT": U860}},
        "heuristic-cbomb7": {"our": "heuristic",                "env": _cbomb7_env()},
        "self-policy-u860": {"our": "policy",                   "env": {"AE_POLICY_CHECKPOINT": U860}},
    }


DEFAULT_CANDIDATES = ["confpol-u860", "heuristic-cbomb7", "self-policy-u860"]
DEFAULT_INCUMBENT = "confpol-u860"
RESULT_SENTINEL = "MELEE_RESULT "


# ---------------------------------------------------------------------------
# Worker: run ONE (candidate, bracket, seed) and emit a one-line JSON summary.
# ---------------------------------------------------------------------------
def _run_worker(args: argparse.Namespace) -> int:
    from simulate import run_simulation
    out = run_simulation(
        rounds=args.rounds,
        opponents_spec=args.suite,
        our_name=args.our,
        log_traj=False,
        seed_start=args.seed,
        novice=not args.non_novice,
    )
    s = out["summary"]
    payload = {
        "suite": args.suite,
        "our": args.our,
        "seed": args.seed,
        "mean_score": float(s["mean_score"]),
        "mean_placement": float(s["mean_placement"]),
        "mean_margin": float(s["mean_margin"]),
        "win_rate": float(s["win_rate"]),
        "placement_hist": s["placement_hist"],
    }
    print(RESULT_SENTINEL + json.dumps(payload), flush=True)
    return 0


def _spawn_worker(our: str, suite: str, hash_seed: int, sim_seed: int,
                  rounds: int, extra_env: dict[str, str], non_novice: bool) -> dict:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(hash_seed)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env.update(extra_env)
    cmd = [
        sys.executable, "-u", str(Path(__file__).resolve()),
        "--worker", "--our", our, "--suite", suite,
        "--seed", str(sim_seed), "--rounds", str(rounds),
    ]
    if non_novice:
        cmd.append("--non-novice")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env,
                          capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"melee worker failed (our={our}, suite={suite}, hash={hash_seed}, "
            f"sim={sim_seed}, rc={proc.returncode}):\n{(proc.stderr or '')[-2500:]}"
        )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(RESULT_SENTINEL):
            return json.loads(line[len(RESULT_SENTINEL):])
    raise RuntimeError(
        f"melee worker produced no result line (our={our}, suite={suite}):\n"
        f"{(proc.stdout or '')[-2000:]}"
    )


# ---------------------------------------------------------------------------
# Aggregation + report
# ---------------------------------------------------------------------------
def _agg(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _evaluate_candidate(label: str, spec: dict, brackets: list[str],
                        hash_seeds: list[int], sim_seeds: list[int],
                        rounds: int, non_novice: bool) -> dict:
    per_bracket: dict[str, dict] = {}
    for suite in brackets:
        runs = []
        for h in hash_seeds:
            for s in sim_seeds:
                runs.append(_spawn_worker(spec["our"], suite, h, s, rounds,
                                          spec.get("env", {}), non_novice))
        per_bracket[suite] = {
            "runs": len(runs),
            "mean_placement": _agg([r["mean_placement"] for r in runs]),
            "win_rate": _agg([r["win_rate"] for r in runs]),
            "mean_margin": _agg([r["mean_margin"] for r in runs]),
            "mean_score": _agg([r["mean_score"] for r in runs]),
        }
        b = per_bracket[suite]
        print(f"    {suite:16s} place={b['mean_placement']:.2f} "
              f"win={b['win_rate']:.2f} margin={b['mean_margin']:+.1f} "
              f"score={b['mean_score']:.4f}  (n={b['runs']})", flush=True)
    worst_placement = max(b["mean_placement"] for b in per_bracket.values())
    return {
        "label": label,
        "our": spec["our"],
        "per_bracket": per_bracket,
        "worst_bracket_placement": worst_placement,
        "min_margin": min(b["mean_margin"] for b in per_bracket.values()),
        "semis_mixed_score": per_bracket.get("semis_mixed", {}).get("mean_score", 0.0),
    }


def _promotion_verdict(cand: dict, inc: dict) -> dict:
    c1 = cand["worst_bracket_placement"] <= inc["worst_bracket_placement"] + 1e-9
    c2 = cand["min_margin"] >= -1e-9
    c3 = cand["semis_mixed_score"] >= inc["semis_mixed_score"] - 1e-9
    return {
        "minimax_placement_ok": bool(c1),
        "margin_nonneg_everywhere_ok": bool(c2),
        "semis_mixed_score_ok": bool(c3),
        "promotable": bool(c1 and c2 and c3),
    }


def _print_report(results: dict[str, dict], incumbent: str, brackets: list[str]) -> None:
    inc = results[incumbent]
    print("\n" + "=" * 78)
    print(f"  MELEE RE-RANK  (incumbent = {incumbent}; minimax over {len(brackets)} brackets)")
    print("=" * 78)
    for label, cand in results.items():
        print(f"\n  {label}   [worst-bracket placement = {cand['worst_bracket_placement']:.2f}, "
              f"min-margin = {cand['min_margin']:+.1f}, "
              f"semis_mixed score = {cand['semis_mixed_score']:.4f}]")
        print(f"    {'bracket':16s} {'place':>6s} {'win':>6s} {'margin':>9s} {'score':>8s}")
        for suite in brackets:
            b = cand["per_bracket"][suite]
            print(f"    {suite:16s} {b['mean_placement']:6.2f} {b['win_rate']:6.2f} "
                  f"{b['mean_margin']:+9.1f} {b['mean_score']:8.4f}")
        if label != incumbent:
            v = _promotion_verdict(cand, inc)
            flag = "PROMOTABLE ✓" if v["promotable"] else "not promotable"
            print(f"    -> vs {incumbent}: {flag}  "
                  f"(minimax {'✓' if v['minimax_placement_ok'] else '✗'}, "
                  f"margin≥0 {'✓' if v['margin_nonneg_everywhere_ok'] else '✗'}, "
                  f"semis_mixed≥inc {'✓' if v['semis_mixed_score_ok'] else '✗'})")
    print("\n" + "=" * 78)
    print("  Headline = worst-bracket mean_placement (lower is less exploitable).")
    print("  confpol-u860 (0.626 cloud) remains the deploy floor regardless.")
    print("=" * 78)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--our", default=None, help="(worker) agent name for simulate._make_our_agent")
    p.add_argument("--suite", default=None, help="(worker) single bracket name")
    p.add_argument("--seed", type=int, default=42, help="(worker) sim seed start")
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--non-novice", action="store_true")
    # parent-only
    p.add_argument("--candidates", nargs="+", default=None,
                   help=f"registry labels to rank (default: {DEFAULT_CANDIDATES})")
    p.add_argument("--incumbent", default=DEFAULT_INCUMBENT)
    p.add_argument("--confpol-ckpt", action="append", default=[], metavar="LABEL=PATH",
                   help="register a confpol candidate over a given raw-policy checkpoint "
                        "(repeatable; e.g. native-u200=training/ae/checkpoints/confpol-native-u200.pt)")
    p.add_argument("--policy-ckpt", action="append", default=[], metavar="LABEL=PATH",
                   help="register a raw-policy (full-control) candidate over a checkpoint (repeatable)")
    p.add_argument("--brackets", nargs="+", default=None,
                   help=f"override brackets (default: {MELEE_BRACKETS})")
    p.add_argument("--hash-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--sim-seeds", nargs="+", type=int, default=[42])
    p.add_argument("--summary-out", type=Path, default=None)
    args = p.parse_args(argv)

    if args.worker:
        if not args.our or not args.suite:
            p.error("--worker requires --our and --suite")
        return _run_worker(args)

    registry = _base_registry()
    for spec_str in args.confpol_ckpt:
        label, env = _parse_confpol_spec(spec_str)
        registry[label] = {"our": "confidence_policy_hybrid", "env": env}
    for spec_str in args.policy_ckpt:
        label, _, path = spec_str.partition("=")
        registry[label] = {"our": "policy",
                           "env": {"AE_POLICY_CHECKPOINT": str(Path(path).resolve())}}

    candidates = args.candidates or DEFAULT_CANDIDATES
    brackets = args.brackets or list(MELEE_BRACKETS)
    incumbent = args.incumbent
    if incumbent not in candidates:
        candidates = [incumbent] + [c for c in candidates if c != incumbent]
    for label in candidates:
        if label not in registry:
            p.error(f"unknown candidate {label!r}; known: {sorted(registry)}")

    print(f"melee_eval: candidates={candidates} brackets={brackets} "
          f"rounds={args.rounds} hash_seeds={args.hash_seeds} sim_seeds={args.sim_seeds}",
          flush=True)
    t0 = time.monotonic()
    results: dict[str, dict] = {}
    for label in candidates:
        print(f"\n[candidate] {label} (our={registry[label]['our']})", flush=True)
        results[label] = _evaluate_candidate(
            label, registry[label], brackets,
            args.hash_seeds, args.sim_seeds, args.rounds, args.non_novice,
        )
    elapsed = time.monotonic() - t0

    _print_report(results, incumbent, brackets)
    print(f"\nelapsed_s={elapsed:.1f}")

    if args.summary_out is not None:
        report = {
            "candidates": candidates, "incumbent": incumbent, "brackets": brackets,
            "rounds": args.rounds, "hash_seeds": args.hash_seeds, "sim_seeds": args.sim_seeds,
            "elapsed_s": elapsed,
            "results": results,
            "verdicts": {label: _promotion_verdict(results[label], results[incumbent])
                         for label in candidates if label != incumbent},
        }
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(report, indent=2))
        print(f"summary -> {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
