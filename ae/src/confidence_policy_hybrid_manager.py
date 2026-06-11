"""Confidence-gated RAW-policy hybrid — the deployed AE composition.

  heuristic runs first (it owns movement, bomb safety, escape). Only on ticks
  where the heuristic signals LOW CONFIDENCE in its target choice do we hand the
  low-level action to the raw PPO policy. On the ~75-90% confident ticks the
  heuristic's action is returned unchanged, so the downside is bounded to the
  heuristic baseline while the policy gets to try the genuinely ambiguous ticks.

The consultant seam: this wrapper composes ``AEManager`` (planner) with
``PolicyAEManager`` (consultant adapter). The two retired siblings —
``hybrid`` (policy-first with heuristic veto) and ``confidence_hybrid``
(same gate over the 12-way tactical macro selector) — were removed with
their modules; see git history.

Confidence comes from ``AEManager.last_decision_confidence`` (written every
ae() call; see _choose_target). Non-target paths (playbook, dominant action,
escape, frozen, init) carry margin=+inf and always pass through untouched.

Env knobs (defaults match the original macro confidence_hybrid experiment):
- AE_CONFPOL_MARGIN_EPSILON (5.0)  top - runner_up must be >= this to stay heuristic
- AE_CONFPOL_TOP_FLOOR      (10.0) top_score must be >= this to stay heuristic
- AE_CONFPOL_OVERRIDE_TARGET_NONE (1) also consult policy when no scorable target
"""

from __future__ import annotations

from collections import Counter
import os

from ae_manager import AEManager
from policy_manager import PolicyAEManager


def _env_bool(name: str, default: bool) -> bool:
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


class ConfidencePolicyHybridAEManager:
    """Heuristic-first; consult the raw PPO policy only on low-confidence ticks."""

    def __init__(self) -> None:
        self.heuristic = AEManager()
        # Raises if the checkpoint is missing/incompatible -> ae_server catches
        # and falls back to pure heuristic. We assert load success loudly so a
        # silent-fallback (the 0.638 bug) is visible in the container logs.
        self.policy = PolicyAEManager()
        self.margin_epsilon = _env_float("AE_CONFPOL_MARGIN_EPSILON", 5.0)
        self.top_floor = _env_float("AE_CONFPOL_TOP_FLOOR", 10.0)
        self.override_target_none = _env_bool("AE_CONFPOL_OVERRIDE_TARGET_NONE", True)
        # R2 phantom-bomb rollback (default OFF). When the gate overrides a
        # heuristic PLACE_BOMB with a policy action, the heuristic's synthetic
        # own-bomb belief entry is phantom (no bomb was placed). When ON, revert
        # it so _danger_cells() doesn't route around a nonexistent bomb. Flag-off
        # is byte-identical to the legacy wrapper.
        self.rollback_phantom_bomb = _env_bool("AE_CONFPOL_ROLLBACK_PHANTOM_BOMB", False)
        self.tick_counts: Counter[str] = Counter()
        print(
            "AE: confidence_policy_hybrid ready "
            f"(eps={self.margin_epsilon}, floor={self.top_floor}, "
            f"override_target_none={self.override_target_none}, "
            f"rollback_phantom_bomb={self.rollback_phantom_bomb})"
        )

    def _is_low_confidence(self) -> tuple[bool, str]:
        conf = getattr(self.heuristic, "last_decision_confidence", None)
        if not isinstance(conf, dict):
            return False, "no_signal"
        path = conf.get("decision_path", "unknown")
        if path == "target_none":
            if self.override_target_none:
                return True, "target_none"
            return False, "target_none_skipped"
        if path != "target":
            # deliberate overrides (playbook/dominant/escape/frozen/init)
            return False, f"path_{path}"
        margin = float(conf.get("margin", float("inf")))
        top = float(conf.get("top_score", float("inf")))
        if margin < self.margin_epsilon:
            return True, "low_margin"
        if top < self.top_floor:
            return True, "low_top_score"
        return False, "confident"

    def ae(self, observation: dict) -> int:
        # Heuristic first -- this populates last_decision_confidence AND owns
        # all the bomb-safety / escape logic we don't want the policy to lose.
        heuristic_action = int(self.heuristic.ae(observation))
        low_conf, reason = self._is_low_confidence()
        self.tick_counts[reason] += 1
        if not low_conf:
            return heuristic_action
        # Overriding: if the heuristic committed a PLACE_BOMB this tick, the
        # bomb is never actually placed, so revert the phantom belief entry
        # before handing the action to the policy.
        if self.rollback_phantom_bomb and self.heuristic.revert_bomb_commit():
            self.tick_counts["phantom_bomb_reverted"] += 1
        # Low-confidence tick: let the raw PPO policy pick. PolicyAEManager.ae()
        # already respects observation["action_mask"], so the action is legal.
        try:
            return int(self.policy.ae(observation))
        except Exception as exc:  # never crash a round -- degrade to heuristic
            self.tick_counts["policy_error_fallback"] += 1
            print(f"AE: confidence_policy_hybrid policy error -> heuristic ({exc!r})")
            return heuristic_action
