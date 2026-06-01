"""Per-slot opening sweep + locked gate.

For each spawn slot, sweeps opening horizons, full-game-validates each against a
cached per-slot baseline (plain C+bomb7 planner), and locks the per-slot gate:
the opening is enabled for a slot ONLY if its best horizon beats the planner by
a margin with confidence — otherwise the planner runs (gate = []). This realizes
max(planner, opening) per spawn: strictly >= baseline by construction.

Reuses simulate.run_one_round (score = cum_reward/1000) with us_slot rotation.
Writes training/ae/data/openings_gate.json (an openings.json-format file the
OpeningHybridManager consumes directly: enabled slots carry one candidate,
disabled slots carry []).

Run:
    PYTHONHASHSEED=0 .venv/bin/python training/ae/sweep_openings.py \
        --suite cloudsuite --rounds 8 --horizons 8,12,16,20,25
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") is None:
    os.environ["PYTHONHASHSEED"] = "0"
    os.execvp(sys.executable, [sys.executable, *sys.argv])

# Shipped C+bomb7 profile — both arms match the deployed agent.
os.environ.setdefault("AE_ITEM_MISSION_VALUE", "80")
os.environ.setdefault("AE_ITEM_RESOURCE_VALUE", "40")
os.environ.setdefault("AE_ENEMY_BASE_VALUE", "100")
os.environ.setdefault("AE_DIJKSTRA_BOMB_COST", "7.0")
os.environ.setdefault("AE_LEAD_BASE_TETHER", "1")
os.environ.setdefault("AE_LEAD_TETHER_HEALTH", "60.0")
os.environ.setdefault("AE_LEAD_TETHER_WEIGHT", "0.5")
os.environ.setdefault(
    "AE_POLICY_CHECKPOINT",
    str(Path(__file__).resolve().parents[2] / "training" / "ae" / "checkpoints"
        / "pandemonium-v1-best-u860.pt"),
)

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "ae" / "src"), str(ROOT / "til-26-ae"), str(ROOT / "training" / "ae")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from ae_manager import AEManager  # noqa: E402
from opening_eval_manager import OpeningHybridManager  # noqa: E402
from opening_sim import BASE_LOCATIONS, OpeningSim, beam_search  # noqa: E402
from opponents import make_opponent, resolve_opponent_spec  # noqa: E402
from simulate import run_one_round  # noqa: E402
from til_environment import bomberman_env  # noqa: E402
from til_environment.config import default_config  # noqa: E402

GATE_PATH = ROOT / "training" / "ae" / "data" / "openings_gate.json"

# Lock criteria: enable an opening for a slot only if its best horizon beats the
# planner by >= MARGIN with one-sided z > Z_MIN (guards against local-noise wins
# that would regress on cloud) AND the opening reliably COMPLETES (compl >=
# MIN_COMPL). The completion guard rejects non-completing openings whose delta
# is a high-variance butterfly effect of their first few moves before abort
# (observed: slot-2 H16/H20 both compl 0% swing -0.23 / +0.29).
MARGIN = 0.02
Z_MIN = 2.0
MIN_COMPL = 0.8


def select_best(rows: list[dict]) -> tuple[dict, bool]:
    """Return (chosen_row, enabled). Enable only a robust win: reliably
    completes, beats baseline by >=MARGIN, and clears z>Z_MIN. Among those,
    pick max delta. If none qualify, return the nominal best (for logging) OFF."""
    eligible = [r for r in rows
                if r["compl"] >= MIN_COMPL and r["delta"] >= MARGIN and r["z"] > Z_MIN]
    if eligible:
        return max(eligible, key=lambda r: r["delta"]), True
    return max(rows, key=lambda r: r["delta"]), False


def _make_opps(spec: str, seed: int):
    names = resolve_opponent_spec(spec)
    return [make_opponent(n, seed=seed + 1000 + i) for i, n in enumerate(names)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="cloudsuite")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--horizons", default="8,12,16,20,25")
    ap.add_argument("--slots", default="0,1,2,3,4,5")
    ap.add_argument("--seed-start", type=int, default=5000)
    ap.add_argument("--beam-width", type=int, default=4000)
    ap.add_argument("--planner", default="heuristic", choices=["heuristic", "confpol"],
                    help="the planner the opening wraps + is compared against")
    ap.add_argument("--gate-out", default=None,
                    help="where to write the locked gate (default depends on --planner)")
    args = ap.parse_args()

    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    slots = [int(s) for s in args.slots.split(",") if s.strip()]
    sim = OpeningSim()

    def make_planner():
        if args.planner == "confpol":
            from confidence_policy_hybrid_manager import ConfidencePolicyHybridAEManager
            return ConfidencePolicyHybridAEManager()
        return AEManager()

    gate_out = Path(args.gate_out) if args.gate_out else (
        ROOT / "training" / "ae" / "data" / f"openings_gate_{args.planner}.json")

    cfg = default_config(); cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)

    gate: dict[str, list[dict]] = {}
    locked_summary: list[str] = []

    print(f"planner={args.planner}  suite={args.suite}  rounds/cfg={args.rounds}  "
          f"horizons={horizons}  slots={slots}  lock: delta>={MARGIN} & z>{Z_MIN}\n")

    for slot in slots:
        base = tuple(BASE_LOCATIONS[slot])
        key = f"{base[0]},{base[1]}"

        # Cached per-slot baseline (planner only) — horizon-independent.
        baseline = make_planner()
        base_scores: dict[int, float] = {}
        for r in range(args.rounds):
            seed = args.seed_start + r
            rb = run_one_round(env, baseline, _make_opps(args.suite, seed), False,
                               seed=seed, us_slot=slot)
            base_scores[seed] = rb["score"]
        base_mean = float(np.mean(list(base_scores.values())))

        print(f"slot {slot} ({key})  planner baseline = {base_mean:.4f}")
        print(f"    {'H':>3} {'open':>8} {'delta':>9} {'z':>6} {'compl%':>7}")

        rows = []
        for H in horizons:
            cand = beam_search(sim, base, horizon=H, beam_width=args.beam_width, top_k=1)[0]
            treat = OpeningHybridManager(heuristic=make_planner(), openings={key: [cand]})
            deltas, treat_scores, compl = [], [], []
            for r in range(args.rounds):
                seed = args.seed_start + r
                rt = run_one_round(env, treat, _make_opps(args.suite, seed), False,
                                   seed=seed, us_slot=slot)
                deltas.append(rt["score"] - base_scores[seed])
                treat_scores.append(rt["score"])
                compl.append(1 if (len(treat.seq) and treat.idx == len(treat.seq)) else 0)
            d = np.array(deltas)
            se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
            z = float(d.mean() / se) if se > 0 else 0.0
            rows.append({"H": H, "cand": cand, "open_mean": float(np.mean(treat_scores)),
                         "delta": float(d.mean()), "se": se, "z": z, "compl": float(np.mean(compl))})
            print(f"    {H:>3} {np.mean(treat_scores):>8.4f} {d.mean():>+9.4f} "
                  f"{z:>6.2f} {100*np.mean(compl):>6.0f}%")

        best, enabled = select_best(rows)
        if enabled:
            c = dict(best["cand"])
            c.update({"horizon": best["H"], "delta": best["delta"], "z": best["z"]})
            gate[key] = [c]
            locked_summary.append(f"  slot {slot} ({key}): OPENING ON  H={best['H']} "
                                  f"delta={best['delta']:+.4f} z={best['z']:.1f}")
        else:
            gate[key] = []
            locked_summary.append(f"  slot {slot} ({key}): planner (best opening "
                                  f"delta={best['delta']:+.4f} z={best['z']:.1f} — below gate)")
        print(f"    -> {'OPENING ON' if enabled else 'PLANNER'} (best H={best['H']}, "
              f"delta={best['delta']:+.4f}, z={best['z']:.2f})\n")

    env.close()
    gate_out.write_text(json.dumps(gate, indent=2))

    print("=" * 60)
    print("LOCKED PER-SLOT GATE:")
    for line in locked_summary:
        print(line)
    enabled_deltas = [g[0]["delta"] for g in gate.values() if g]
    exp_lift = sum(enabled_deltas) / len(slots) if slots else 0.0
    print(f"\n  enabled slots: {sum(1 for g in gate.values() if g)}/{len(slots)}")
    print(f"  expected aggregate lift if spawns uniform: {exp_lift:+.4f}")
    print(f"  wrote {gate_out}")
    print("  (local signal — cloud variance-farm the enabled slots before shipping)")


if __name__ == "__main__":
    main()
