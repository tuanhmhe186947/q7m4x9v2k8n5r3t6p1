from __future__ import annotations

from typing import Any

from pig_behavior.classification_v2.evaluation import (
    legacy_development_l6_social_relation_decision as social_decision,
)


def _comparison(
    *,
    macro_delta: float,
    ci_low: float,
    nll_delta: float,
    rare_delta: float,
) -> dict[str, Any]:
    return {
        "delta_candidate_minus_baseline": {
            "macro_f1_global_10_class": macro_delta,
            "accuracy": 0.0,
            "nll": nll_delta,
        },
        "video_cluster_bootstrap": {"ci_low": ci_low},
        "confusion_groups": {"rare": {"macro_f1_delta": rare_delta}},
    }


def _contract() -> dict[str, Any]:
    return {
        "minimum_macro_f1_gain": 0.02,
        "maximum_absolute_availability_only_gain": 0.01,
        "maximum_rare_group_macro_f1_drop": 0.02,
        "require_positive_video_cluster_ci_low": True,
        "require_nll_improvement_vs_zero": True,
    }


def _passing_comparisons() -> dict[str, dict[str, Any]]:
    return {
        "social_relation_vs_parameter_matched_zero": _comparison(
            macro_delta=0.03,
            ci_low=0.01,
            nll_delta=-0.1,
            rare_delta=-0.01,
        ),
        "social_relation_vs_availability_only": _comparison(
            macro_delta=0.025,
            ci_low=0.005,
            nll_delta=-0.05,
            rare_delta=0.0,
        ),
        "availability_only_vs_parameter_matched_zero": _comparison(
            macro_delta=0.005,
            ci_low=-0.01,
            nll_delta=0.0,
            rare_delta=0.0,
        ),
    }


def test_social_decision_promotes_only_when_every_gate_passes() -> None:
    decision = social_decision.make_social_relation_decision(
        _passing_comparisons(),
        contract=_contract(),
    )

    assert decision["full_social_relation_expansion_authorized"] is True
    assert decision["decision"] == (
        "RETAIN_SOCIAL_RELATION_FOR_FULL_LEGACY_DEVELOPMENT"
    )


def test_social_decision_rejects_an_unbounded_availability_shortcut() -> None:
    comparisons = _passing_comparisons()
    comparisons["availability_only_vs_parameter_matched_zero"] = _comparison(
        macro_delta=0.02,
        ci_low=0.01,
        nll_delta=-0.01,
        rare_delta=0.0,
    )

    decision = social_decision.make_social_relation_decision(
        comparisons,
        contract=_contract(),
    )

    assert decision["full_social_relation_expansion_authorized"] is False
    assert decision["negative_result_is_valid_evidence"] is True
    assert decision["decision"] == (
        "DO_NOT_EXPAND_SOCIAL_RELATION_FROM_CURRENT_SHORT_EVIDENCE"
    )
