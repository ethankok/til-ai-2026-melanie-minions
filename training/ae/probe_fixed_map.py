"""Probe: does AEManager.is_fixed_novice_map actually get set on a real
novice observation?

Builds the official til_environment novice env, pulls the genuine step==0
observation for each agent (same shape the cloud evaluator emits — it's the
same package), feeds it through a fresh AEManager exactly like ae_server does,
and reports whether the fixed-Novice-map detector fired.

Run anywhere the env is importable (Mac venv OR Workbench):
    PYTHONHASHSEED=0 .venv/bin/python training/ae/probe_fixed_map.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") is None:
    os.environ["PYTHONHASHSEED"] = "0"
    os.execvp(sys.executable, [sys.executable, *sys.argv])

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (
    str(REPO_ROOT / "ae" / "src"),
    str(REPO_ROOT / "til-26-ae"),
):
    if p not in sys.path:
        sys.path.insert(0, p)

from ae_manager import AEManager  # noqa: E402
from novice_map_data import BASE_LOCATIONS  # noqa: E402
from til_environment import bomberman_env  # noqa: E402
from til_environment.config import default_config  # noqa: E402


def _native(observation: dict) -> dict:
    """Mirror the numpy->python conversion ae_server / test_ae.py do."""
    return {
        k: v if type(v) in (int, float) else (v.tolist() if hasattr(v, "tolist") else v)
        for k, v in observation.items()
    }


def main() -> None:
    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    env.reset(seed=42)

    print(f"hardcoded BASE_LOCATIONS = {BASE_LOCATIONS}")
    print("-" * 72)

    seen_agents: set = set()
    fired_any = False
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            env.step(None)
            continue
        obs = _native(observation)
        step = int(obs.get("step", -1)) if not hasattr(obs.get("step"), "__len__") else int(obs["step"][0])
        if step == 0 and agent not in seen_agents:
            seen_agents.add(agent)
            base_loc_raw = observation.get("base_location")
            mgr = AEManager()
            action = mgr.ae(obs)  # the real entry point — sets the flag if it matches
            fired = bool(getattr(mgr, "is_fixed_novice_map", False))
            fired_any = fired_any or fired
            print(f"agent={agent}")
            print(f"  raw  base_location (env)   = {base_loc_raw!r}")
            print(f"  native base_location (obs) = {obs.get('base_location')!r}")
            print(f"  parsed via _location()     = {mgr._location(obs.get('base_location'))!r}")
            print(f"  obs keys                   = {sorted(obs.keys())}")
            print(f"  ae() returned action       = {action}")
            print(f"  >>> is_fixed_novice_map    = {fired}")
            print("-" * 72)
        # advance with a legal action so the loop reaches all 6 agents at step 0
        mask = obs.get("action_mask")
        act = 0
        if mask is not None:
            for i, m in enumerate(mask):
                if int(m):
                    act = i
                    break
        env.step(act)
        if len(seen_agents) >= 6:
            break

    print("=" * 72)
    print(f"VERDICT: detector fired on {len(seen_agents)} step-0 agents — "
          f"is_fixed_novice_map set: {fired_any}")
    if not fired_any:
        print("  -> The detector does NOT fire on the real env. The exploit is "
              "silently dead; this is a format/key bug to fix.")
    else:
        print("  -> The detector DOES fire against the official env format. The "
              "'doesn't fire on cloud' note was a misdiagnosis.")


if __name__ == "__main__":
    main()
