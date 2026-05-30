"""Confidence-gated RAW-policy hybrid (for the Pandemonium CNN-PPO checkpoint).

The existing ``confidence_hybrid`` (ConfidenceHybridAEManager) gates the 12-way
*tactical macro* selector. The Pandemonium line trains the raw 6-action
``PolicyNetwork`` instead, so it cannot use that wrapper. This is the
equivalent gate for the raw policy:

  heuristic runs first (it owns movement, bomb safety, escape). Only on ticks
  where the heuristic signals LOW CONFIDENCE in its target choice do we hand the
  low-level action to the raw PPO policy. On the ~75-90% confident ticks the
  heuristic's action is returned unchanged, so the downside is bounded to the
  heuristic baseline while the policy gets to try the genuinely ambiguous ticks.

This is the most conservative of the three Pandemonium deployment modes:
  AE_MODE=policy                  -> policy in full control
  AE_MODE=hybrid                  -> policy-first, heuristic vetoes unsafe acts
  AE_MODE=confidence_policy_hybrid -> heuristic-first, policy only when unsure

Confidence comes from ``AEManager.last_decision_confidence`` (written every
ae() call; see _choose_target). Non-target paths (playbook, dominant action,
escape, frozen, init) carry margin=+inf and always pass through untouched.

Env knobs (all default to the same values as the macro confidence_hybrid):
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
        self.tick_counts: Counter[str] = Counter()
        print(
            "AE: confidence_policy_hybrid ready "
            f"(eps={self.margin_epsilon}, floor={self.top_floor}, "
            f"override_target_none={self.override_target_none})"
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
        # Low-confidence tick: let the raw PPO policy pick. PolicyAEManager.ae()
        # already respects observation["action_mask"], so the action is legal.
        try:
            return int(self.policy.ae(observation))
        except Exception as exc:  # never crash a round -- degrade to heuristic
            self.tick_counts["policy_error_fallback"] += 1
            print(f"AE: confidence_policy_hybrid policy error -> heuristic ({exc!r})")
            return heuristic_action
