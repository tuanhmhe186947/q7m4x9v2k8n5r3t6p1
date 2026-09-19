from __future__ import annotations

from typing import Any

from pig_behavior.classification_v2.evaluation.legacy_development_l6_union_context_decision import (
    make_union_context_decision,
)


def _comparison(
    macro_delta: float,
    *,
    ci_low: float,
    nll_delta: float,
    rare_delta: float,
) -> dict[str, Any]:
    return {
        "delta_candidate_minus_baseline": {
            "macro_f1_global_10_class": macro_delta,
            "nll": nll_delta,
        },
        "video_cluster_bootstrap": {"ci_low": ci_low},
        "confusion_groups": {
            "rare": {"macro_f1_delta": rare_delta},
        },
    }


def _contract() -> dict[str, Any]:
    return {
        "minimum_macro_f1_gain": 0.02,
        "maximum_absolute_availability_only_gain": 0.01,
        "maximum_rare_group_macro_f1_drop": 0.02,
        "require_positive_video_cluster_ci_low": True,
        "require_nll_improvement_vs_zero": True,
    }


def test_union_context_negative_result_is_valid_evidence() -> None:
    comparisons = {
        "union_context_vs_parameter_matched_zero": _comparison(
            -0.04,
            ci_low=-0.08,
            nll_delta=0.2,
            rare_delta=-0.03,
        ),
        "union_context_vs_availability_only": _comparison(
            -0.042,
            ci_low=-0.09,
            nll_delta=0.28,
            rare_delta=-0.03,
        ),
        "availability_only_vs_parameter_matched_zero": _comparison(
            0.002,
            ci_low=-0.02,
            nll_delta=-0.07,
            rare_delta=0.0,
        ),
    }

    decision = make_union_context_decision(
        comparisons,
        contract=_contract(),
    )

    assert decision["decision"] == (
        "DO_NOT_EXPAND_UNION_CONTEXT_FROM_CURRENT_SHORT_EVIDENCE"
    )
    assert decision["full_union_context_expansion_authorized"] is False
    assert decision["negative_result_is_valid_evidence"] is True
    assert decision["applies_to_merged_reviewed_data"] is False


def test_union_context_requires_every_promotion_criterion() -> None:
    comparisons = {
        "union_context_vs_parameter_matched_zero": _comparison(
            0.03,
            ci_low=0.005,
            nll_delta=-0.02,
            rare_delta=-0.01,
        ),
        "union_context_vs_availability_only": _comparison(
            0.025,
            ci_low=0.002,
            nll_delta=-0.01,
            rare_delta=-0.01,
        ),
        "availability_only_vs_parameter_matched_zero": _comparison(
            0.005,
            ci_low=-0.01,
            nll_delta=0.01,
            rare_delta=0.0,
        ),
    }

    decision = make_union_context_decision(
        comparisons,
        contract=_contract(),
    )

    assert decision["decision"] == (
        "RETAIN_UNION_CONTEXT_FOR_FULL_LEGACY_DEVELOPMENT"
    )
    assert decision["full_union_context_expansion_authorized"] is True
    assert all(decision["criteria"].values())
