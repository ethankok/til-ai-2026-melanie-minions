"""Confidence-gated tactical macro hybrid.

Variant of :class:`MacroHybridAEManager` that consults the tactical policy
*only when the heuristic itself signals low confidence* in its top action.

Why: prior macro-hybrid runs (27 May) consulted PPO on every tick and relied
on harm-aware gates to reject deviations. That requires the PPO to be net-
positive across the full state distribution; the 28 May multi-seed gate
showed it isn't. Restricting PPO to the ~10-25% of ticks where the
heuristic itself indicates ambiguity (top vs runner-up score nearly tied)
or pessimism (top absolute score below a floor) preserves the heuristic-
C+bomb7 mean on the other ~75-90% of ticks while still letting PPO try
to lift the genuinely hard situations.

Confidence signal comes from ``AEManager.last_decision_confidence``, which
the heuristic writes after every ``ae()`` call. ``_choose_target()``
exposes the top and runner-up scores; non-target paths (playbook,
dominant action, escape, etc.) carry a sentinel margin=+inf so they
always pass through untouched (those are deliberate overrides).

Tunable via env:

- ``AE_CONF_MARGIN_EPSILON`` (default 5.0) — top - runner_up must be
  >= this to keep the heuristic action without PPO consult.
- ``AE_CONF_TOP_FLOOR`` (default 10.0) — top_score must be >= this to
  keep the heuristic action without PPO consult.
- ``AE_CONF_OVERRIDE_TARGET_NONE`` (default 1) — also consult PPO when
  the heuristic returned no scorable target (`decision_path=target_none`).
  Set to 0 to skip those ticks.
"""

from __future__ import annotations

from collections import Counter
import os

import torch

from macro_hybrid_manager import MacroHybridAEManager
from tactical_hybrid_manager import TacticalPolicyAEManager
from tactical_policy import tactical_from_manager, tactical_option_name


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


class ConfidenceHybridAEManager(MacroHybridAEManager):
    """Macro-hybrid that gates PPO consultation on heuristic confidence."""

    def __init__(self, policy: TacticalPolicyAEManager | None = None, shadow: bool | None = None):
        super().__init__(policy=policy, shadow=shadow)
        self.margin_epsilon = _env_float("AE_CONF_MARGIN_EPSILON", 5.0)
        self.top_floor = _env_float("AE_CONF_TOP_FLOOR", 10.0)
        self.override_target_none = _env_bool("AE_CONF_OVERRIDE_TARGET_NONE", True)
        # Observability counters; emitted via decision_counts at end of round.
        self.confidence_tick_counts: Counter[str] = Counter()

    def _is_low_confidence(self) -> tuple[bool, str]:
        """Return (low_confidence, reason) from the heuristic's last decision."""
        conf = getattr(self.heuristic, "last_decision_confidence", None)
        if not isinstance(conf, dict):
            # Heuristic doesn't expose confidence (older AEManager). Default
            # to high-confidence -- bare heuristic behaves like the baseline.
            return False, "no_signal"
        path = conf.get("decision_path", "unknown")
        if path == "target_none":
            if self.override_target_none:
                return True, "target_none"
            return False, "target_none_skipped"
        if path != "target":
            # Deliberate overrides (playbook, dominant, escape, fallback,
            # tactical_lookahead, frozen, init): treat as high-confidence.
            return False, f"path_{path}"
        margin = float(conf.get("margin", float("inf")))
        top = float(conf.get("top_score", float("inf")))
        if margin < self.margin_epsilon:
            return True, "low_margin"
        if top < self.top_floor:
            return True, "low_top_score"
        return False, "confident"

    def ae(self, observation: dict) -> int:
        # Step 1: heuristic runs first (this populates last_decision_confidence).
        heuristic_action = int(self.heuristic.ae(observation))

        # Step 2: read confidence and gate the PPO consult.
        low_conf, reason = self._is_low_confidence()
        self.confidence_tick_counts[reason] += 1
        if not low_conf:
            self.decision_counts[f"conf_passthrough_{reason}"] += 1
            return heuristic_action

        # Step 3: low-confidence tick -- consult the tactical policy with
        # the same harm-aware acceptance gates as MacroHybridAEManager.
        prior_option = int(tactical_from_manager(self.heuristic, action=heuristic_action))
        try:
            _option, logits = self.policy.tactical_logits(observation)
        except Exception:
            self.decision_counts["conf_policy_error_fallback"] += 1
            return heuristic_action

        probs = torch.softmax(logits, dim=-1)
        prior_prob = float(probs[prior_option].item())
        ranked = torch.argsort(logits, descending=True)[: self.top_k].tolist()
        ranked_options = [int(option) for option in ranked]

        if self.shadow:
            top = ranked_options[0] if ranked_options else prior_option
            self.shadow_counts[
                f"shadow_{tactical_option_name(prior_option)}_to_{tactical_option_name(top)}"
            ] += 1
            self.decision_counts["conf_shadow_fallback"] += 1
            return heuristic_action

        bucket = self._distance_bucket(observation)
        for rank, option in enumerate(ranked_options):
            if option == prior_option:
                if rank == 0:
                    self.decision_counts[
                        f"conf_prior_top_{tactical_option_name(prior_option)}"
                    ] += 1
                continue

            ok, gate_reason = self._macro_delta_allowed(prior_option, option, bucket)
            if not ok:
                self.decision_counts[
                    f"conf_reject_{gate_reason}_{tactical_option_name(prior_option)}_to_{tactical_option_name(option)}"
                ] += 1
                continue

            option_prob = float(probs[option].item())
            if self.policy_conf_threshold > 0.0 and option_prob < self.policy_conf_threshold:
                self.decision_counts[f"conf_reject_low_policy_conf_{tactical_option_name(option)}"] += 1
                continue
            if self.delta_margin > 0.0 and (option_prob - prior_prob) < self.delta_margin:
                self.decision_counts[f"conf_reject_low_policy_margin_{tactical_option_name(option)}"] += 1
                continue

            action, info = self.executor.act_with_info(
                option,
                observation,
                heuristic_action=heuristic_action,
                heuristic_already_run=True,
            )
            if info.get("veto"):
                self.decision_counts[
                    f"conf_executor_veto_{info['veto']}_{tactical_option_name(option)}"
                ] += 1
                continue

            label = tactical_option_name(option)
            self.accept_counts[label] += 1
            self.decision_counts[
                f"conf_accept_{tactical_option_name(prior_option)}_to_{label}_via_{reason}"
            ] += 1
            if bucket is not None:
                self.decision_counts[
                    f"conf_accept_dist_{bucket}_{tactical_option_name(prior_option)}_to_{label}"
                ] += 1
            return int(action)

        self.decision_counts[f"conf_no_accepted_delta_{tactical_option_name(prior_option)}_via_{reason}"] += 1
        return heuristic_action
