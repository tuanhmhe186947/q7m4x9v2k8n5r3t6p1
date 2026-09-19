from __future__ import annotations

from typing import Any

from pig_behavior.classification_v2.evaluation.legacy_development_l6_motion_decision import (
    make_motion_decision,
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


def test_motion_decision_promotes_only_when_every_gate_passes() -> None:
    comparisons = {
        "motion_vs_parameter_matched_zero": _comparison(
            macro_delta=0.03,
            ci_low=0.01,
            nll_delta=-0.1,
            rare_delta=-0.01,
        ),
        "motion_vs_availability_only": _comparison(
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

    decision = make_motion_decision(comparisons, contract=_contract())

    assert decision["full_motion_expansion_authorized"] is True
    assert decision["decision"] == (
        "RETAIN_MOTION_FOR_FULL_LEGACY_DEVELOPMENT"
    )


def test_motion_rejection_is_valid_evidence_and_stops_expansion() -> None:
    comparisons = {
        "motion_vs_parameter_matched_zero": _comparison(
            macro_delta=-0.001,
            ci_low=-0.02,
            nll_delta=0.01,
            rare_delta=-0.03,
        ),
        "motion_vs_availability_only": _comparison(
            macro_delta=0.001,
            ci_low=-0.02,
            nll_delta=0.0,
            rare_delta=0.0,
        ),
        "availability_only_vs_parameter_matched_zero": _comparison(
            macro_delta=0.005,
            ci_low=-0.01,
            nll_delta=0.0,
            rare_delta=0.0,
        ),
    }

    decision = make_motion_decision(comparisons, contract=_contract())

    assert decision["full_motion_expansion_authorized"] is False
    assert decision["negative_result_is_valid_evidence"] is True
    assert decision["decision"] == (
        "DO_NOT_EXPAND_MOTION_FROM_CURRENT_SHORT_EVIDENCE"
    )
    assert decision["next_action"] == (
        "retain_parameter_matched_zero_and_stop_motion_expansion"
    )
