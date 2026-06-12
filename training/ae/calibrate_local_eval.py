"""Calibrate which local AE eval suite best predicts cloud rankings.

For each heuristic config in `cloud-calibration-truth.json`, run
validate_cloud_suite.py across several opponent suites with the
config's AE_* env vars set. Aggregate per-suite local scores per
config, then offline we compute rank correlation against cloud means.

Usage (Mac):
  .venv/bin/python training/ae/calibrate_local_eval.py \\
      --rounds 12 --workers 4 \\
      --suites random library cloudsuite pressure2 mixed \\
      --out training/ae/data/cloud-calibration-local-12r.json

The script spawns one subprocess per config (env-isolated) running
validate_cloud_suite.py and waits on all of them. Aggregated output:

  {
    "rounds": 12,
    "suites": [...],
    "per_config": {
        "<config_name>": {
            "per_suite": {"<suite>": {"mean": ..., "p50": ..., ...}, ...},
            "elapsed_s": ...
        }, ...
    }
  }
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATE_SCRIPT = REPO_ROOT / "training" / "ae" / "validate_cloud_suite.py"
TRUTH_PATH = REPO_ROOT / "training" / "ae" / "data" / "cloud-calibration-truth.json"


def _run_one_config(
    name: str,
    env_extra: dict[str, str],
    suites: list[str],
    rounds: int,
    seed: int,
    our: str,
    out_dir: Path,
) -> tuple[str, dict]:
    """Spawn validate_cloud_suite.py with `env_extra` applied to os.environ."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_out = out_dir / f"calib_{name}.json"

    env = os.environ.copy()
    for k, v in env_extra.items():
        env[k] = v
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Pin PYTHONHASHSEED: AEManager's set/dict iteration order can drift
    # cloudsuite mean by 0.10+ across identical (env, seed) runs otherwise.
    # Must be set before subprocess spawn (checked at interpreter startup).
    env["PYTHONHASHSEED"] = env_extra.get("__pythonhashseed", "0")

    cmd = [
        sys.executable,
        "-u",
        str(VALIDATE_SCRIPT),
        "--rounds", str(rounds),
        "--our", our,
        "--suites", *suites,
        "--seed", str(seed),
        "--summary-out", str(summary_out),
    ]

    started = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.monotonic() - started

    if proc.returncode != 0:
        return name, {
            "error": f"returncode={proc.returncode}",
            "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
            "elapsed_s": elapsed,
        }

    if not summary_out.exists():
        return name, {
            "error": "summary file not written",
            "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
            "elapsed_s": elapsed,
        }

    raw = json.loads(summary_out.read_text())
    per_suite_in = raw.get("per_suite", [])
    # Order in per_suite matches the order of --suites passed in.
    per_suite: dict[str, dict] = {}
    for spec, entry in zip(suites, per_suite_in):
        per_suite[spec] = {
            "mean": float(entry.get("mean_score", 0.0)),
            "p50": float(entry.get("p50", 0.0)),
            "min": float(entry.get("min_score", 0.0)),
            "max": float(entry.get("max_score", 0.0)),
            "n": int(entry.get("rounds", rounds)),
            "opponents": list(entry.get("opponents", [])),
        }
    return name, {
        "env": env_extra,
        "per_suite": per_suite,
        "aggregate": raw.get("aggregate", {}),
        "elapsed_s": elapsed,
        "summary_path": str(summary_out),
    }


def _load_configs(truth_path: Path, filter_mode: str | None) -> list[dict]:
    truth = json.loads(truth_path.read_text())
    configs = truth["configs"]
    if filter_mode:
        configs = [c for c in configs if c.get("ae_mode") == filter_mode]
    return configs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, default=TRUTH_PATH)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["random", "library", "cloudsuite", "pressure2", "mixed"],
    )
    parser.add_argument("--our", choices=["heuristic", "option_v2", "hybrid"], default="heuristic")
    parser.add_argument(
        "--filter-mode",
        choices=["heuristic", "hybrid", "policy", "all"],
        default="heuristic",
        help="Only run configs whose ae_mode matches; 'all' to include policy/hybrid",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-config-dir", type=Path, default=None,
                        help="Where per-config summaries go; defaults to <out>.d/")
    args = parser.parse_args()

    filter_mode = None if args.filter_mode == "all" else args.filter_mode
    configs = _load_configs(args.truth, filter_mode)
    if not configs:
        print(f"no configs matched filter_mode={args.filter_mode}", file=sys.stderr)
        return 2

    per_config_dir = args.per_config_dir or args.out.with_suffix(".d")
    per_config_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"calibrate: rounds={args.rounds} suites={args.suites} "
        f"configs={[c['name'] for c in configs]} workers={args.workers}",
        flush=True,
    )

    results: dict[str, dict] = {}
    started = time.monotonic()

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _run_one_config,
                cfg["name"],
                cfg.get("env", {}),
                args.suites,
                args.rounds,
                args.seed,
                args.our,
                per_config_dir,
            ): cfg["name"]
            for cfg in configs
        }
        for fut in concurrent.futures.as_completed(futures):
            name = futures[fut]
            try:
                cname, result = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"[{name}] crashed: {exc}", flush=True)
                results[name] = {"error": f"exception: {exc}"}
                continue
            print(
                f"[{cname}] done in {result.get('elapsed_s', 0.0):.1f}s "
                f"err={'error' in result}",
                flush=True,
            )
            results[cname] = result

    elapsed = time.monotonic() - started
    payload = {
        "rounds": args.rounds,
        "suites": args.suites,
        "our_agent": args.our,
        "seed": args.seed,
        "filter_mode": args.filter_mode,
        "elapsed_s": elapsed,
        "per_config": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\naggregate -> {args.out}  elapsed={elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
