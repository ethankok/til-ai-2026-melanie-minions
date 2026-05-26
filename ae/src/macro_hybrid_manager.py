"""Planner-first tactical macro hybrid.

This is the stricter hybrid path intended for post-raw-RL AE work:

1. Run the strongest heuristic baseline first.
2. Let a learned tactical policy propose high-level macros, not raw actions.
3. Execute accepted macros through planner-backed executors.
4. Fall back to the heuristic unless the proposed deviation clears confidence,
   transition-support, and harm-aware gates.

Compared with ``tactical_hybrid_manager.py``, this wrapper is deliberately
opinionated for deployment/A-B use:

- it defaults the baseline heuristic to the current calibrated
  ``heuristic-C + bomb_cost=7.0`` profile;
- it allows mapped tactical deltas such as rush/base-bomb, because those are
  the actual macro actions the policy must be allowed to choose;
- it can try a small top-k set from the tactical policy before falling back;
- it supports shadow mode for diagnostics without changing actions.
"""

from __future__ import annotations

from collections import Counter
import os

import torch

from tactical_hybrid_manager import TacticalExecutor, TacticalHybridAEManager, TacticalPolicyAEManager
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


def _env_int(name: str, default: int) -> int:
    return int(_env_float(name, float(default)))


def _apply_baseline_profile() -> None:
    """Apply the calibrated macro-hybrid default profile.

    ``setdefault`` is intentional: explicit Docker/env settings should win.
    """

    profile = os.environ.get("AE_MACRO_BASELINE_PROFILE", "combo_c_bomb7").strip().lower()
    if profile in {"combo_c_bomb7", "c_bomb7", "heuristic_c_bomb7"}:
        os.environ.setdefault("AE_MODE", "macro_hybrid")
        os.environ.setdefault("AE_ITEM_MISSION_VALUE", "80")
        os.environ.setdefault("AE_ITEM_RESOURCE_VALUE", "40")
        os.environ.setdefault("AE_ENEMY_BASE_VALUE", "100")
        os.environ.setdefault("AE_DIJKSTRA_BOMB_COST", "7.0")

    # Make this a real macro-deviation wrapper. The old tactical_hybrid
    # defaults were conservative because the checkpoint was BC-derived.
    os.environ.setdefault("AE_TACTICAL_PROFILE", "macro")
    os.environ.setdefault("AE_TACTICAL_ALLOW_MAPPED_DELTAS", "1")
    os.environ.setdefault("AE_TACTICAL_REQUIRE_DELTA_SUPPORT", "1")
    # Candidate PPO needs the wrapper to actually try safe alternatives from a
    # BC-warm-start policy, but executor-only acceptance is too broad. Require
    # at least one positive same-seed episode sample for the prior->option
    # transition, then let the outer wrapper-vs-baseline eval gate decide
    # whether the admitted deltas are worth saving.
    os.environ.setdefault("AE_TACTICAL_REQUIRE_DELTA_SUPPORT", "1")
    os.environ.setdefault("AE_TACTICAL_MIN_DELTA_SUPPORT", "1")
    os.environ.setdefault("AE_TACTICAL_MIN_ATTEMPTED", "0")
    os.environ.setdefault("AE_TACTICAL_MIN_POSITIVE_RATE", "0.0")
    os.environ.setdefault("AE_TACTICAL_MIN_NET_DELTA", "-999.0")
    os.environ.setdefault(
        "AE_TACTICAL_ALLOWED_DELTA_OPTIONS",
        "rush_enemy_base,bomb_enemy_base,collect_mission_safe,collect_resource_safe,counter_rush",
    )
    os.environ.setdefault("AE_TACTICAL_HYBRID_CONF", "0.0")


class MacroHybridAEManager(TacticalHybridAEManager):
    """Strict planner-first macro policy wrapper."""

    def __init__(self, policy: TacticalPolicyAEManager | None = None, shadow: bool | None = None):
        _apply_baseline_profile()
        super().__init__(policy=policy)
        self.top_k = max(1, _env_int("AE_MACRO_TOP_K", 4))
        self.delta_margin = _env_float("AE_MACRO_DELTA_MARGIN", -0.01)
        self.shadow = _env_bool("AE_MACRO_SHADOW", False) if shadow is None else bool(shadow)
        self.accept_counts: Counter[str] = Counter()
        self.shadow_counts: Counter[str] = Counter()

    def ae(self, observation: dict) -> int:
        heuristic_action = int(self.heuristic.ae(observation))
        prior_option = int(tactical_from_manager(self.heuristic, action=heuristic_action))
        try:
            _option, logits = self.policy.tactical_logits(observation)
        except Exception:
            self.decision_counts["macro_policy_error_fallback"] += 1
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
            self.decision_counts["macro_shadow_fallback"] += 1
            return heuristic_action

        bucket = self._distance_bucket(observation)
        for rank, option in enumerate(ranked_options):
            if option == prior_option:
                if rank == 0:
                    self.decision_counts[f"macro_prior_top_{tactical_option_name(prior_option)}"] += 1
                continue

            ok, reason = self._macro_delta_allowed(prior_option, option, bucket)
            if not ok:
                self.decision_counts[
                    f"macro_reject_{reason}_{tactical_option_name(prior_option)}_to_{tactical_option_name(option)}"
                ] += 1
                continue

            option_prob = float(probs[option].item())
            if self.policy_conf_threshold > 0.0 and option_prob < self.policy_conf_threshold:
                self.decision_counts[f"macro_reject_low_conf_{tactical_option_name(option)}"] += 1
                continue
            if self.delta_margin > 0.0 and (option_prob - prior_prob) < self.delta_margin:
                self.decision_counts[f"macro_reject_low_margin_{tactical_option_name(option)}"] += 1
                continue

            action, info = self.executor.act_with_info(
                option,
                observation,
                heuristic_action=heuristic_action,
                heuristic_already_run=True,
            )
            if info.get("veto"):
                self.decision_counts[f"macro_executor_veto_{info['veto']}_{tactical_option_name(option)}"] += 1
                continue

            label = tactical_option_name(option)
            self.accept_counts[label] += 1
            self.decision_counts[
                f"macro_accept_{tactical_option_name(prior_option)}_to_{label}"
            ] += 1
            if bucket is not None:
                self.decision_counts[
                    f"macro_accept_dist_{bucket}_{tactical_option_name(prior_option)}_to_{label}"
                ] += 1
            return int(action)

        self.decision_counts[f"macro_no_accepted_delta_{tactical_option_name(prior_option)}"] += 1
        return heuristic_action

    def _macro_delta_allowed(self, prior_option: int, option: int, bucket: str | None) -> tuple[bool, str]:
        if self.allowed_delta_options is not None and option not in self.allowed_delta_options:
            return False, "disallowed_option"
        if self.allowed_delta_transitions is not None and (prior_option, option) not in self.allowed_delta_transitions:
            return False, "disallowed_transition"
        if self.allowed_delta_distance_buckets is not None and bucket not in self.allowed_delta_distance_buckets:
            return False, f"disallowed_distance_{bucket or 'unknown'}"
        if self.require_delta_support:
            support = 0.0
            counts = self.policy.delta_transition_counts
            if counts is not None:
                support = float(counts[prior_option, option])
            if support < self.min_delta_support:
                return False, "insufficient_positive_support"
        ok, reason = self._delta_is_supported(prior_option, option, bucket)
        if not ok:
            return False, f"harm_{reason}"
        return True, "ok"
