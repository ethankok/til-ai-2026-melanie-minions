"""Re-derive the locked per-slot opening gate from a sweep log.

Parses sweep_openings.log (per-(slot,horizon) delta/z/compl), applies the
robust selection rule (reliably completes, beats baseline by >=MARGIN, clears
z>Z_MIN; pick max delta among those), regenerates the deterministic opening
sequence for each enabled slot via beam_search, and writes openings_gate.json.

Separates measurement (the expensive sweep) from selection (cheap, re-runnable),
so the gate can be re-locked with a different rule without re-running games.

Run:
    PYTHONHASHSEED=0 .venv/bin/python training/ae/lock_gate.py \
        --log training/ae/data/sweep_openings.log
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") is None:
    os.environ["PYTHONHASHSEED"] = "0"
    os.execvp(sys.executable, [sys.executable, *sys.argv])

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "ae" / "src"), str(ROOT / "training" / "ae")):
    if p not in sys.path:
        sys.path.insert(0, p)

from opening_sim import BASE_LOCATIONS, OpeningSim, beam_search  # noqa: E402

MARGIN = 0.02
Z_MIN = 2.0
MIN_COMPL = 0.8
BEAM_WIDTH = 4000  # must match the sweep's beam width for identical sequences

_SLOT_RE = re.compile(r"slot (\d+) \((\d+),(\d+)\)\s+planner baseline = ([\d.]+)")
_ROW_RE = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([+-][\d.]+)\s+([+-]?[\d.]+)\s+(\d+)%")


def parse_log(path: Path) -> dict[int, dict]:
    """Return {slot: {"base": (x,y), "baseline": float, "rows": [...]}}."""
    out: dict[int, dict] = {}
    cur = None
    for line in path.read_text().splitlines():
        ms = _SLOT_RE.search(line)
        if ms:
            slot = int(ms.group(1))
            cur = slot
            out[slot] = {"base": (int(ms.group(2)), int(ms.group(3))),
                         "baseline": float(ms.group(4)), "rows": []}
            continue
        mr = _ROW_RE.match(line)
        if mr and cur is not None:
            out[cur]["rows"].append({
                "H": int(mr.group(1)),
                "open_mean": float(mr.group(2)),
                "delta": float(mr.group(3)),
                "z": float(mr.group(4)),
                "compl": int(mr.group(5)) / 100.0,
            })
    return out


def select_best(rows: list[dict]) -> tuple[dict, bool]:
    eligible = [r for r in rows
                if r["compl"] >= MIN_COMPL and r["delta"] >= MARGIN and r["z"] > Z_MIN]
    if eligible:
        return max(eligible, key=lambda r: r["delta"]), True
    return max(rows, key=lambda r: r["delta"]), False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(ROOT / "training" / "ae" / "data" / "sweep_openings.log"))
    ap.add_argument("--out", default=str(ROOT / "training" / "ae" / "data" / "openings_gate.json"))
    ap.add_argument("--deploy-out", default=str(ROOT / "ae" / "src" / "openings_gate.json"),
                    help="self-contained gate (with baked trajectories) shipped in the image")
    args = ap.parse_args()

    parsed = parse_log(Path(args.log))
    if not parsed:
        raise SystemExit(f"no slot data parsed from {args.log}")

    sim = OpeningSim()
    gate: dict[str, list[dict]] = {}
    print(f"re-locking from {args.log}  (rule: compl>={MIN_COMPL}, delta>={MARGIN}, z>{Z_MIN})\n")

    enabled_deltas: list[float] = []
    for slot in sorted(parsed):
        info = parsed[slot]
        base = info["base"]
        key = f"{base[0]},{base[1]}"
        best, enabled = select_best(info["rows"])
        if enabled:
            cand = beam_search(sim, base, horizon=best["H"], beam_width=BEAM_WIDTH, top_k=1)[0]
            cand = dict(cand)
            # Bake the predicted (x, y, dir) trajectory so the deploy manager can
            # run the divergence gate without importing OpeningSim.
            st = sim.initial_state(base)
            traj = [[st.pos[0], st.pos[1], st.dir]]
            for a in cand["actions"]:
                st = sim.step(st, a)
                traj.append([st.pos[0], st.pos[1], st.dir])
            cand["traj"] = traj
            cand.update({"horizon": best["H"], "delta": best["delta"], "z": best["z"],
                         "compl": best["compl"]})
            gate[key] = [cand]
            enabled_deltas.append(best["delta"])
            print(f"  slot {slot} ({key}): OPENING ON  H={best['H']:>2}  "
                  f"delta={best['delta']:+.4f}  z={best['z']:.1f}  compl={best['compl']*100:.0f}%")
        else:
            gate[key] = []
            print(f"  slot {slot} ({key}): PLANNER       "
                  f"(best robust delta={best['delta']:+.4f} z={best['z']:.1f} "
                  f"compl={best['compl']*100:.0f}% — below gate)")

    Path(args.out).write_text(json.dumps(gate, indent=2))
    Path(args.deploy_out).write_text(json.dumps(gate, indent=2))
    n_on = sum(1 for v in gate.values() if v)
    exp_lift = sum(enabled_deltas) / len(parsed) if parsed else 0.0
    print(f"\n  enabled: {n_on}/{len(parsed)} slots")
    print(f"  expected aggregate lift if spawns uniform: {exp_lift:+.4f}")
    print(f"  wrote {args.out}")
    print("  (local signal — cloud variance-farm the enabled slots before shipping)")


if __name__ == "__main__":
    main()
