#!/usr/bin/env python3
"""AE cloud variance-farm aggregation + promotion-decision tool.

Reads the canonical cloud-submission ledger (training/ae/data/cloud_samples.json)
and turns a pile of noisy cloud scores into a disciplined promote / hold / need-
more-samples verdict, using the project's established cloud noise model
(acc sigma ~= 0.053 on a deterministic image; a single score has a +-0.10 95% CI).

Why a tool: every prior AE "win" risked being a tail draw (heuristic-A's famous
0.613 is the top of a [0.613, 0.579, 0.606] series whose true mean is 0.599).
This computes means, standard errors, CIs, a two-sample test, and -- crucially --
the *power analysis* that says whether the effect you care about is even
resolvable with a feasible number of cloud submissions.

Stdlib only; runs on the Mac. Submissions themselves require the Workbench
(til build/submit) -- log each result here with `add`.

Usage:
  # See every config's mean +- SE and 95% CI, ranked.
  python training/ae/variance_farm.py summary

  # Promotion decision: is the candidate's TRUE mean above the baseline's?
  python training/ae/variance_farm.py compare --target tether-v1 --baseline heuristic-A

  # How many cloud submissions per arm to resolve a given effect size?
  python training/ae/variance_farm.py plan --effect 0.022

  # Log a fresh cloud result (do this on/after each Workbench submit).
  python training/ae/variance_farm.py add --config tether-v1 --acc 0.604 --speed 0.850
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

LEDGER = Path(__file__).resolve().parent / "data" / "cloud_samples.json"

# Standard-normal critical values (the global sigma is estimated from the n=6
# farm, so a z-based interval is more honest than a per-config small-n t).
Z_95 = 1.959963985  # two-sided 95%
Z_POWER_80 = 0.8416212  # one-sided 80% power


def load() -> dict:
    return json.loads(LEDGER.read_text())


def save(data: dict) -> None:
    LEDGER.write_text(json.dumps(data, indent=2) + "\n")


def by_name(data: dict, name: str) -> dict:
    for c in data["configs"]:
        if c["name"] == name:
            return c
    raise SystemExit(
        f"config '{name}' not found. Known: "
        + ", ".join(c["name"] for c in data["configs"])
    )


def stats(acc: list[float], sigma: float) -> dict:
    """Mean + SE/CI using the *pooled global* sigma (more reliable than per-
    config sample sigma at tiny n). Also reports the per-config sample sigma
    when n>=2 as a sanity check."""
    n = len(acc)
    mean = sum(acc) / n if n else float("nan")
    se = sigma / math.sqrt(n) if n else float("inf")
    sample_sigma = (
        math.sqrt(sum((x - mean) ** 2 for x in acc) / (n - 1)) if n >= 2 else None
    )
    return {
        "n": n,
        "mean": mean,
        "se": se,
        "ci_lo": mean - Z_95 * se,
        "ci_hi": mean + Z_95 * se,
        "sample_sigma": sample_sigma,
    }


def required_n(effect: float, sigma: float, two_sample: bool = True) -> float:
    """Samples per arm to detect `effect` at alpha=0.05 two-sided, 80% power."""
    if effect <= 0:
        return float("inf")
    k = 2.0 if two_sample else 1.0
    return k * (Z_95 + Z_POWER_80) ** 2 * sigma**2 / effect**2


def fmt(s: dict) -> str:
    extra = (
        f"  sample_sigma={s['sample_sigma']:.4f}"
        if s["sample_sigma"] is not None
        else "  sample_sigma=  n/a "
    )
    return (
        f"n={s['n']:<2d} mean={s['mean']:.4f}  SE={s['se']:.4f}  "
        f"95%CI=[{s['ci_lo']:.4f}, {s['ci_hi']:.4f}]{extra}"
    )


def cmd_summary(data: dict, args: argparse.Namespace) -> None:
    sigma = data["noise_model"]["acc_sigma"]
    rows = []
    for c in data["configs"]:
        acc = c.get("cloud_acc", [])
        if not acc:
            continue
        rows.append((c["name"], stats(acc, sigma)))
    rows.sort(key=lambda r: r[1]["mean"], reverse=True)
    print(f"\nCloud variance-farm summary  (pooled acc sigma = {sigma}, "
          f"source: {data['noise_model']['source']})")
    print(f"Incumbent / promotion bar: {data['incumbent']}\n")
    print(f"{'config':<26} {'n':>2}  {'mean':>6}  {'SE':>6}  {'95% CI':>17}  sample_sigma")
    print("-" * 86)
    for name, s in rows:
        ss = f"{s['sample_sigma']:.4f}" if s["sample_sigma"] is not None else "  -  "
        flag = "  <-- incumbent" if name == data["incumbent"] else ""
        print(f"{name:<26} {s['n']:>2}  {s['mean']:.4f}  {s['se']:.4f}  "
              f"[{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]  {ss}{flag}")
    print()


def cmd_compare(data: dict, args: argparse.Namespace) -> None:
    sigma = data["noise_model"]["acc_sigma"]
    tgt = by_name(data, args.target)
    base = by_name(data, args.baseline)
    st = stats(tgt["cloud_acc"], sigma)
    sb = stats(base["cloud_acc"], sigma)

    print(f"\n=== Promotion test: {args.target}  vs  {args.baseline} ===\n")
    print(f"  {args.target:<24} {fmt(st)}")
    print(f"  {args.baseline:<24} {fmt(sb)}")

    delta = st["mean"] - sb["mean"]
    se_diff = sigma * math.sqrt(1 / st["n"] + 1 / sb["n"])
    z = delta / se_diff if se_diff else 0.0
    # two-sided p-value from z
    p = math.erfc(abs(z) / math.sqrt(2))
    print(f"\n  delta (mean - mean) = {delta:+.4f}   ({delta/sigma:+.2f} sigma)")
    print(f"  SE(delta)           = {se_diff:.4f}")
    print(f"  z = {z:+.2f}   two-sided p = {p:.3f}")
    print(f"  95% CI on delta     = [{delta - Z_95*se_diff:+.4f}, "
          f"{delta + Z_95*se_diff:+.4f}]")

    # Power: how many per arm to resolve the *observed* delta?
    if delta > 0:
        need = required_n(delta, sigma)
        more_t = max(0.0, math.ceil(need - st["n"]))
        more_b = max(0.0, math.ceil(need - sb["n"]))
        print(f"\n  To resolve delta={delta:.4f} at 80% power: "
              f"~{math.ceil(need)} submissions PER ARM")
        print(f"    -> need ~{more_t:.0f} more for {args.target}, "
              f"~{more_b:.0f} more for {args.baseline}")

    # Verdict
    print("\n  VERDICT:", verdict(delta, se_diff, z, p, st, sb, sigma))
    print()


def verdict(delta, se_diff, z, p, st, sb, sigma) -> str:
    if delta <= 0:
        return ("HOLD / REJECT. Candidate mean is not above baseline; "
                "no evidence to promote.")
    ci_lo = delta - Z_95 * se_diff
    if ci_lo > 0:
        return ("PROMOTE. 95% CI on the difference excludes 0 -- the lift "
                "clears the noise floor.")
    need = required_n(delta, sigma)
    if need > 40:
        return (f"HOLD (unresolvable). Lift is {delta/sigma:+.2f} sigma -- "
                f"inside the noise floor. Confirming it would take ~{math.ceil(need)} "
                f"submissions/arm, which is infeasible. The candidate is SAFE "
                f"(directionally positive, no local collapse) but cannot be "
                f"statistically promoted. Ship-or-not is a judgement call, not "
                f"a measurement; do not burn the farm chasing it.")
    return (f"NEED MORE SAMPLES. Directionally positive but CI still includes 0. "
            f"Collect to ~{math.ceil(need)}/arm, then re-test.")


def cmd_plan(data: dict, args: argparse.Namespace) -> None:
    sigma = args.sigma or data["noise_model"]["acc_sigma"]
    print(f"\nPower plan  (acc sigma = {sigma}, alpha=0.05 two-sided, 80% power)\n")
    print(f"{'effect (delta)':>14}  {'sigma units':>11}  {'n/arm (2-sample)':>17}  "
          f"{'n (1-sample vs fixed bar)':>26}")
    print("-" * 76)
    effects = [args.effect] if args.effect else [0.01, 0.02, 0.022, 0.03, 0.05, 0.08, 0.10]
    for e in effects:
        n2 = required_n(e, sigma, two_sample=True)
        n1 = required_n(e, sigma, two_sample=False)
        print(f"{e:>14.4f}  {e/sigma:>11.2f}  {math.ceil(n2):>17}  {math.ceil(n1):>26}")
    print()


def cmd_add(data: dict, args: argparse.Namespace) -> None:
    cfg = by_name(data, args.config)
    cfg.setdefault("cloud_acc", []).append(round(args.acc, 4))
    if args.speed is not None:
        cfg.setdefault("cloud_speed", []).append(round(args.speed, 4))
    save(data)
    sigma = data["noise_model"]["acc_sigma"]
    s = stats(cfg["cloud_acc"], sigma)
    print(f"Logged {args.config}: acc={args.acc}"
          + (f" speed={args.speed}" if args.speed is not None else ""))
    print(f"  {args.config}: {fmt(s)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("summary", help="rank all configs by cloud mean")

    c = sub.add_parser("compare", help="promotion test: target vs baseline")
    c.add_argument("--target", required=True)
    c.add_argument("--baseline", default="heuristic-A")

    p = sub.add_parser("plan", help="power analysis: samples needed per effect size")
    p.add_argument("--effect", type=float, default=None)
    p.add_argument("--sigma", type=float, default=None)

    a = sub.add_parser("add", help="append a cloud result to the ledger")
    a.add_argument("--config", required=True)
    a.add_argument("--acc", type=float, required=True)
    a.add_argument("--speed", type=float, default=None)

    args = ap.parse_args()
    data = load()
    {"summary": cmd_summary, "compare": cmd_compare,
     "plan": cmd_plan, "add": cmd_add}[args.cmd](data, args)


if __name__ == "__main__":
    main()
