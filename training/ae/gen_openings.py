"""Offline driver: generate ranked opening shortlists per Novice spawn slot.

For each of the 6 spawn slots (keyed by base_location), runs the OpeningSim
beam search across a sweep of horizons and positioning weights, then merges
into a diverse, reward-ranked shortlist. The output feeds the OUTER validation
pipeline (force each candidate as a game-start prefix, hand off to the live
planner, score full games via multi_seed_eval) — this generator does NOT pick
the final opening, it produces strong candidates cheaply.

Run:
    PYTHONHASHSEED=0 .venv/bin/python training/ae/gen_openings.py \
        --horizons 8,12,16,20 --beam-width 3000 --top-k 5

Writes training/ae/data/openings.json:
    { "13,9": [ {horizon, actions, reward, score, end_pos, end_dir, items, position_weight}, ... ], ... }
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

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

from opening_sim import SLOT_TABLE, OpeningSim, beam_search  # noqa: E402

DATA_DIR = _THIS / "data"


def generate(
    *,
    horizons: list[int],
    beam_width: int,
    top_k: int,
    position_weights: list[float],
) -> dict[str, list[dict]]:
    """Return {slot_key: ranked shortlist} for all 6 spawn slots.

    A candidate is tagged with the (horizon, position_weight) that produced it.
    Shortlists are deduped by action sequence and ranked by raw reward (the real
    objective); positioning only diversifies the pool for the outer gate.
    """
    sim = OpeningSim()
    result: dict[str, list[dict]] = {}

    for base in SLOT_TABLE:
        pool: list[dict] = []
        for h in horizons:
            for pw in position_weights:
                for cand in beam_search(
                    sim, base, horizon=h, beam_width=beam_width, top_k=top_k, position_weight=pw
                ):
                    cand = dict(cand)
                    cand["horizon"] = h
                    cand["position_weight"] = pw
                    pool.append(cand)

        # Dedup by action sequence; rank by reward then shorter horizon (faster
        # to reach the same value is better), keep top_k.
        seen: set[tuple] = set()
        ranked = sorted(pool, key=lambda c: (c["reward"], -c["horizon"]), reverse=True)
        shortlist: list[dict] = []
        for c in ranked:
            key = tuple(c["actions"])
            if key in seen:
                continue
            seen.add(key)
            shortlist.append(c)
            if len(shortlist) >= top_k:
                break

        result[f"{base[0]},{base[1]}"] = shortlist

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="8,12,16,20",
                    help="comma-separated opening lengths to sweep")
    ap.add_argument("--beam-width", type=int, default=3000)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--position-weights", default="0.0,0.5",
                    help="comma-separated end-positioning weights")
    ap.add_argument("--out", default=str(DATA_DIR / "openings.json"))
    args = ap.parse_args()

    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    pweights = [float(w) for w in args.position_weights.split(",") if w.strip()]

    result = generate(
        horizons=horizons,
        beam_width=args.beam_width,
        top_k=args.top_k,
        position_weights=pweights,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2))

    print(f"wrote {out_path}")
    for slot, cands in result.items():
        if not cands:
            print(f"  slot {slot}: (no candidates)")
            continue
        best = cands[0]
        print(f"  slot {slot}: {len(cands)} cands; "
              f"best reward={best['reward']:.0f} H={best['horizon']} "
              f"items={len(best['items'])} end={best['end_pos']}")


if __name__ == "__main__":
    main()
