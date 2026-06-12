"""Compute rank correlation between local-eval scores and cloud means.

Takes the calibration output from calibrate_local_eval.py and the cloud
truth file. For each local suite (and a few weighted combinations),
prints Spearman and Pearson rank correlations + a side-by-side ranking
table. Identifies the local signal that best predicts cloud rank.

Usage:
  .venv/bin/python training/ae/analyze_calibration.py \\
      --local training/ae/data/calib-heuristic-24r-s42.json \\
      --truth training/ae/data/cloud-calibration-truth.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[2]


def spearman(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        result = [0.0] * n
        # Handle ties via average rank.
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                result[order[k]] = avg
            i = j + 1
        return result

    rx = ranks(xs)
    ry = ranks(ys)
    return pearson(rx, ry)


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs) ** 0.5
    den_y = sum((y - my) ** 2 for y in ys) ** 0.5
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--truth", type=Path, default=REPO_ROOT / "training" / "ae" / "data" / "cloud-calibration-truth.json")
    parser.add_argument("--min-cloud-n", type=int, default=2,
                        help="Drop cloud configs with fewer than N submits (default 2 — excludes heuristic-restore singleton)")
    args = parser.parse_args()

    local = json.loads(args.local.read_text())
    truth = json.loads(args.truth.read_text())

    cloud_by_name = {c["name"]: c for c in truth["configs"]}

    # Only configs present in BOTH local results and truth.
    names: list[str] = []
    cloud_means: list[float] = []
    cloud_ns: list[int] = []
    for name in local["per_config"]:
        if name not in cloud_by_name:
            continue
        cfg = cloud_by_name[name]
        if cfg.get("n", 0) < args.min_cloud_n:
            continue
        if "error" in local["per_config"][name]:
            print(f"[skip] {name}: local error {local['per_config'][name]['error']}")
            continue
        names.append(name)
        cloud_means.append(float(cfg["mean"]))
        cloud_ns.append(int(cfg.get("n", 0)))

    if len(names) < 3:
        print(f"Not enough aligned configs (got {len(names)}); need >= 3 for meaningful rank correlation.")
        return 1

    suites = local["suites"]
    print(f"\nAligned configs ({len(names)}): {names}")
    print(f"Cloud n per config:           {cloud_ns}")
    print(f"Cloud means:                  {[round(c, 4) for c in cloud_means]}")
    print(f"\nLocal suites:                 {suites}")
    print(f"Local rounds per suite:       {local['rounds']}\n")

    per_suite_local: dict[str, list[float]] = {s: [] for s in suites}
    for name in names:
        per = local["per_config"][name]["per_suite"]
        for s in suites:
            per_suite_local[s].append(float(per[s]["mean"]))

    # Cloud rank (descending = best first).
    def rank_desc(values: list[float]) -> list[int]:
        order = sorted(range(len(values)), key=lambda i: -values[i])
        rank = [0] * len(values)
        for r, idx in enumerate(order):
            rank[idx] = r + 1
        return rank

    cloud_rank = rank_desc(cloud_means)

    # Score each individual suite + a few weighted combos.
    candidates: dict[str, list[float]] = {}
    for s in suites:
        candidates[f"suite:{s}"] = per_suite_local[s]

    # mean-of-suites (equal weight) — same as `aggregate.mean_of_means`.
    mom = [
        mean(per_suite_local[s][i] for s in suites) for i in range(len(names))
    ]
    candidates["combo:mean_of_means"] = mom

    # mean of just (cloudsuite, pressure2) — pressure-focused.
    pressure_only = [s for s in suites if s in ("cloudsuite", "pressure2")]
    if len(pressure_only) >= 2:
        candidates["combo:pressure_only"] = [
            mean(per_suite_local[s][i] for s in pressure_only) for i in range(len(names))
        ]

    # weighted: 2x cloudsuite + 1x pressure2 + 1x library, no random.
    weighted_nonrandom = [s for s in suites if s in ("library", "cloudsuite", "pressure2")]
    if {"cloudsuite", "pressure2"}.issubset(set(weighted_nonrandom)):
        def w(s: str) -> float:
            return {"cloudsuite": 2.0, "pressure2": 1.0, "library": 1.0}.get(s, 0.0)
        denom = sum(w(s) for s in weighted_nonrandom)
        candidates["combo:weighted_pressure"] = [
            sum(w(s) * per_suite_local[s][i] for s in weighted_nonrandom) / denom
            for i in range(len(names))
        ]

    # all-no-random (drop random suite which is easy).
    nonrandom = [s for s in suites if s != "random"]
    if nonrandom:
        candidates["combo:all_no_random"] = [
            mean(per_suite_local[s][i] for s in nonrandom) for i in range(len(names))
        ]

    header = f"{'config':<22}" + "".join(f"{s[:10]:>12}" for s in suites) + f"{'mom':>10}" + f"{'cloud':>10}"
    print(header)
    print("-" * len(header))
    for i, name in enumerate(names):
        row = f"{name:<22}"
        for s in suites:
            row += f"{per_suite_local[s][i]:>12.4f}"
        row += f"{mom[i]:>10.4f}{cloud_means[i]:>10.4f}"
        print(row)

    print(f"\n{'signal':<32}{'spearman':>12}{'pearson':>12}")
    print("-" * 56)
    best_name, best_rho = None, -2.0
    rows = []
    for sig_name, values in candidates.items():
        rho = spearman(values, cloud_means)
        r = pearson(values, cloud_means)
        rows.append((sig_name, rho, r))
        if rho > best_rho:
            best_rho = rho
            best_name = sig_name
    rows.sort(key=lambda t: -t[1])
    for sig_name, rho, r in rows:
        marker = "  <- best" if sig_name == best_name else ""
        print(f"{sig_name:<32}{rho:>12.3f}{r:>12.3f}{marker}")

    print(f"\nRank comparison (best signal = {best_name}):")
    best_values = dict(candidates)[best_name]
    best_rank = rank_desc(best_values)
    print(f"{'config':<22}{'cloud_mean':>12}{'cloud_rank':>12}{'local_val':>12}{'local_rank':>12}")
    for i, name in enumerate(names):
        print(f"{name:<22}{cloud_means[i]:>12.4f}{cloud_rank[i]:>12d}{best_values[i]:>12.4f}{best_rank[i]:>12d}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
