from __future__ import annotations

from typing import Any

from pig_behavior.classification_v2.evaluation import (
    legacy_development_l6_full_frame_context_decision as decision,
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


def test_full_frame_context_rejects_mixed_short_evidence() -> None:
    comparisons = {
        "full_frame_context_vs_parameter_matched_zero": _comparison(
            0.025,
            ci_low=0.002,
            nll_delta=0.20,
            rare_delta=-0.01,
        ),
        "full_frame_context_vs_availability_only": _comparison(
            0.022,
            ci_low=0.001,
            nll_delta=0.18,
            rare_delta=-0.01,
        ),
        "availability_only_vs_parameter_matched_zero": _comparison(
            0.003,
            ci_low=-0.01,
            nll_delta=-0.02,
            rare_delta=0.0,
        ),
    }

    result = decision.make_full_frame_context_decision(
        comparisons,
        contract=_contract(),
    )

    assert result["decision"] == (
        "DO_NOT_EXPAND_FULL_FRAME_CONTEXT_FROM_CURRENT_SHORT_EVIDENCE"
    )
    assert result["full_frame_context_expansion_authorized"] is False
    assert result["criteria"]["full_frame_nll_improves_vs_zero"] is False
    assert result["negative_result_is_valid_evidence"] is True
    assert result["applies_to_merged_reviewed_data"] is False


def test_full_frame_context_requires_every_promotion_criterion() -> None:
    comparisons = {
        "full_frame_context_vs_parameter_matched_zero": _comparison(
            0.03,
            ci_low=0.005,
            nll_delta=-0.02,
            rare_delta=-0.01,
        ),
        "full_frame_context_vs_availability_only": _comparison(
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

    result = decision.make_full_frame_context_decision(
        comparisons,
        contract=_contract(),
    )

    assert result["decision"] == (
        "RETAIN_FULL_FRAME_CONTEXT_FOR_FULL_LEGACY_DEVELOPMENT"
    )
    assert result["full_frame_context_expansion_authorized"] is True
    assert all(result["criteria"].values())
