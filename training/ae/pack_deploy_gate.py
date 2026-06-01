"""Bake an openings gate (actions-only, from sweep_openings.py) into the
self-contained deploy gate with predicted trajectories.

The deploy manager (ae/src/opening_hybrid_manager.py) runs the divergence gate
against a baked (x,y,dir) trajectory so it needs no OpeningSim at runtime. The
sweep writes actions only; this bakes the trajectory and writes the deploy gate.

Run:
    PYTHONHASHSEED=0 .venv/bin/python training/ae/pack_deploy_gate.py \
        --in training/ae/data/openings_gate_confpol.json \
        --out ae/src/openings_gate.json
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

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "ae" / "src"), str(ROOT / "training" / "ae")):
    if p not in sys.path:
        sys.path.insert(0, p)

from opening_sim import OpeningSim  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = json.loads(Path(args.inp).read_text())
    sim = OpeningSim()
    out: dict[str, list[dict]] = {}

    for key, cands in src.items():
        if not cands:
            out[key] = []
            continue
        cand = dict(cands[0])
        base = tuple(int(v) for v in key.split(","))
        st = sim.initial_state(base)
        traj = [[st.pos[0], st.pos[1], st.dir]]
        for a in cand["actions"]:
            st = sim.step(st, a)
            traj.append([st.pos[0], st.pos[1], st.dir])
        cand["traj"] = traj
        out[key] = [cand]

    Path(args.out).write_text(json.dumps(out, indent=2))
    enabled = [(k, v[0]["horizon"], v[0]["delta"]) for k, v in out.items() if v]
    print(f"wrote {args.out}")
    for k, h, d in enabled:
        print(f"  {k}: ON  H={h}  delta={d:+.4f}")
    print(f"  disabled: {[k for k, v in out.items() if not v]}")


if __name__ == "__main__":
    main()
