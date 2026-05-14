"""Hybrid AE manager: policy by default, heuristic as a safety net.

Design rationale. Every approach the team has tried — pure heuristic
(planner-v3b), pure BC (bc-v1), pure PPO (ppo-v1, ppo-v2) — has hit the
same official ceiling of ~0.50 with a structural local→official gap of
0.18-0.27. The two strategies have *uncorrelated* failure modes:

- The heuristic is robust but conservative; it loses by passing on
  marginal-value bombs and by routing around enemies it would have won
  trades against.
- The policy is opportunistic but unsafe; BC overfit to local random
  opponents, PPO's action distribution can self-trap (move into bomb
  blast, place bomb without a verified escape).

This wrapper takes the policy's action by default and uses the heuristic
to *veto and replace* it when it can prove the action is unsafe or
plainly worse:

1. Always project the observation into the heuristic's belief map so
   wall/bomb/enemy/item memory stays fresh on every tick.
2. Compute the heuristic's recommended action up front.
3. Take the policy action unless any of:
   a. Policy picks an illegal action (action_mask off) — fall back.
   b. Policy picks ``PLACE_BOMB`` without a verified escape — fall back.
   c. Policy steps into a cell the heuristic flags as bomb-blast danger
      AND a heuristic-safe alternative exists — fall back.
   d. Policy picks ``STAY`` while frozen_ticks==0 and the heuristic has a
      legal non-STAY action — fall back. (The policy is known to
      degenerate into STAY under some scalar-shift conditions.)
4. Speed shortcut: if the heuristic is in an active escape path,
   trust it (heuristic owns bomb-safety mechanics) without invoking the
   policy at all.

This is the first structurally new bet the team has placed on AE since
ppo-v1. It is not a "higher local mean" submission — its premise is that
the policy and heuristic make uncorrelated mistakes, so a per-tick
arbiter strictly dominates either in isolation when their disagreements
fall in the policy's failure-mode set.
"""

from __future__ import annotations

import os

from ae_manager import AEManager


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class HybridAEManager:
    """Policy-first action selection with heuristic veto.

    Tunables (env vars, all optional):

    - ``AE_HYBRID_VETO_BOMBS`` (default true): override policy bombs that
      have no verified escape path.
    - ``AE_HYBRID_VETO_DANGER`` (default true): override policy moves
      into known bomb-blast cells.
    - ``AE_HYBRID_VETO_FROZEN_STAY`` (default true): override policy
      STAYs when frozen_ticks==0 and a non-STAY legal action exists.
    """

    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4
    PLACE_BOMB = 5

    def __init__(self, policy=None):
        # The policy is required; the wrapper has no other reason to exist.
        # Importing here keeps ae_server.py's heuristic-only path torch-free.
        if policy is None:
            from policy_manager import PolicyAEManager

            policy = PolicyAEManager()
        self.policy = policy
        self.heuristic = AEManager()

        self.veto_bombs = _env_flag("AE_HYBRID_VETO_BOMBS", True)
        self.veto_danger = _env_flag("AE_HYBRID_VETO_DANGER", True)
        self.veto_frozen_stay = _env_flag("AE_HYBRID_VETO_FROZEN_STAY", True)
        # Optional confidence gate: if set, only use the policy when its
        # top-action probability exceeds this. Default 0 = always use
        # policy (subject to vetoes). Useful for A/B experiments.
        self.policy_conf_threshold = _env_float("AE_HYBRID_CONF", 0.0)

    def ae(self, observation: dict) -> int:
        # Fast path 1: if the heuristic is currently fleeing its own bomb
        # blast, the policy's tactical action is likely to walk us into
        # the blast. Trust the heuristic here.
        heuristic_action = self.heuristic.ae(observation)
        if self.heuristic.escape_target is not None:
            return heuristic_action

        # Pull policy logits (one forward pass) and gate.
        try:
            policy_action, logits = self.policy.ae_logits(observation)
        except Exception:
            # Any policy failure → use heuristic. We *never* serve a
            # request without an action.
            return heuristic_action

        # Confidence gate (optional, default off).
        if self.policy_conf_threshold > 0.0:
            import torch

            probs = torch.softmax(logits, dim=-1)
            top_p = float(probs.max().item())
            if top_p < self.policy_conf_threshold:
                return heuristic_action

        # Vetoes — only override the policy if we can prove it's wrong.
        if not self._policy_action_legal(observation, policy_action):
            return heuristic_action

        if self.veto_bombs and policy_action == self.PLACE_BOMB:
            if not self._bomb_has_escape(observation):
                return heuristic_action

        if self.veto_danger:
            if self._steps_into_danger(observation, policy_action):
                # Only override if the heuristic's choice is safer.
                if not self._steps_into_danger(observation, heuristic_action):
                    return heuristic_action

        if self.veto_frozen_stay and policy_action == self.STAY:
            if self._frozen_ticks(observation) == 0 and heuristic_action != self.STAY:
                # Only override if heuristic action is legal.
                if self.heuristic._legal(observation, heuristic_action):
                    return heuristic_action

        return policy_action

    # ------------------------------------------------------------------
    # Veto helpers — re-use the heuristic's already-updated belief state.
    # ------------------------------------------------------------------
    def _policy_action_legal(self, observation: dict, action: int) -> bool:
        mask = observation.get("action_mask")
        if mask is None:
            return 0 <= action <= 5
        try:
            return bool(mask[action])
        except Exception:
            return 0 <= action <= 5

    def _bomb_has_escape(self, observation: dict) -> bool:
        """True if the heuristic believes bombing here has a safe exit."""

        location = self.heuristic._location(observation.get("location"))
        if location is None:
            return False
        if self.heuristic._as_int(observation.get("team_bombs"), default=0) <= 0:
            return False
        blast = self.heuristic._blast_cells(location)
        base_location = self.heuristic.base_location or self.heuristic._location(
            observation.get("base_location")
        )
        # Don't bomb our own base.
        if base_location is not None and base_location in blast:
            return False
        escape = self.heuristic._safe_escape_within(
            location, blast, self.heuristic.BOMB_TIMER
        )
        return escape is not None

    def _steps_into_danger(self, observation: dict, action: int) -> bool:
        """True if action moves the agent into a bomb-blast cell."""

        location = self.heuristic._location(observation.get("location"))
        if location is None:
            return False
        direction = (
            self.heuristic._as_int(observation.get("direction"), default=0) % 4
        )
        new_pos, _ = self.heuristic._simulate_action(location, direction, action)
        if new_pos == location:
            # STAY or rotation — danger only matters if current cell is already in danger.
            return location in self.heuristic._danger_cells()
        return new_pos in self.heuristic._danger_cells()

    def _frozen_ticks(self, observation: dict) -> int:
        return self.heuristic._as_int(observation.get("frozen_ticks"), default=0)
