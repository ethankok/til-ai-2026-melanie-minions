"""Empirical probe: how many of OUR decision-steps until a bomb we place detonates?

Motivation (6 Jun 2026): the heuristic hardcodes `BOMB_TIMER = 3`
(`ae/src/ae_manager.py`) with a comment claiming it matches
`til_environment/bomberman_config.yaml` (entities.bomb.timer). But:
  * we run the DATACLASS config (`default_config()`), not the yaml, and the
    dataclass has `entities.bomb.timer = 4`;
  * `Bomb.__post_init__` does `self.timer += 1` ("placed bombs tick down once
    immediately in upkeep, so compensate"), so the internal timer is 5 at
    placement;
  * the within-step phase order is place -> move -> detonation -> damage ->
    upkeep, and a bomb detonates when `expired` (timer <= 0) is seen in the
    detonation phase.

This probe drives the real AEC env, places one bomb on our agent's first legal
turn, then watches the live Bomb entity (timer / alive / env.last_explosions)
to find the exact decision-step at which the blast fires. It also confirms
whether the placer takes self-damage standing on its own bomb tile.

Run:
    .venv/bin/python training/ae/probe_bomb_timer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from til_environment import bomberman_env  # noqa: E402
from til_environment.config import default_config  # noqa: E402
from til_environment.entities.dynamic import Bomb  # noqa: E402


def main() -> int:
    cfg = default_config()
    print(f"config entities.bomb.timer = {cfg.entities.bomb.timer}")
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    env.reset(seed=42)
    dyn = env.unwrapped.dynamics
    us = env.possible_agents[0]

    def bombs() -> list[Bomb]:
        return dyn.registry.query().type(Bomb).all()

    placed_turn: int | None = None
    turn = 0
    health_on_tile = []
    for agent in env.agent_iter(max_iter=6000):
        obs, _rew, term, trunc, _info = env.last()
        if term or trunc:
            env.step(None)
            continue
        if agent == us:
            mask = [int(m) for m in obs["action_mask"]]
            if placed_turn is not None:
                bs = bombs()
                expl = len(getattr(dyn, "last_explosions", []) or [])
                health = float(np.ravel(np.asarray(obs["health"]))[0])
                health_on_tile.append(health)
                info = [(round(b.timer, 1), b.alive, b.expired) for b in bs]
                print(f"our-turn {turn:2d}: (timer,alive,expired)={info} "
                      f"last_explosions={expl} our_health={health}")
                if expl > 0:
                    print(f"\n>>> BLAST fired in the cycle before our-turn {turn} "
                          f"=> detonation DELAY = {turn - placed_turn} decision-steps "
                          f"from placement (observable at +{turn - placed_turn}).")
                    print(f">>> placer self-damage while standing on tile: "
                          f"{'NONE' if min(health_on_tile) >= 60 else 'TOOK DAMAGE'} "
                          f"(health stayed {health_on_tile}).")
                    return 0
            if placed_turn is None and mask[5] == 1:  # PLACE_BOMB legal
                env.step(5)
                placed_turn = turn
                print(f"placed bomb at our-turn {turn}")
            else:
                env.step(4)  # STAY on the tile
            turn += 1
        else:
            env.step(4)  # other agents STAY
        if turn > 20:
            print("no blast within 20 turns")
            return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
