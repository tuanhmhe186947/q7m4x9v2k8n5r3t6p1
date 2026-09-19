from __future__ import annotations

from typing import Any

from pig_behavior.classification_v2.evaluation.legacy_development_l6_pen_context_decision import (
    make_pen_context_decision,
)


def _comparison(
    *,
    macro_delta: float,
    focus_delta: float,
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
        "focus_group": {"macro_f1_delta": focus_delta},
        "confusion_groups": {
            "rare": {"macro_f1_delta": rare_delta},
        },
    }


def _contract() -> dict[str, Any]:
    return {
        "minimum_macro_f1_gain": 0.01,
        "minimum_focus_group_macro_f1_gain": 0.01,
        "maximum_absolute_availability_only_gain": 0.01,
        "maximum_rare_group_macro_f1_drop": 0.02,
        "require_positive_video_cluster_ci_low": True,
        "require_nll_improvement_vs_zero": True,
        "bootstrap_iterations": 2000,
        "bootstrap_seed": 20260717,
    }


def test_pen_context_decision_requires_every_predeclared_gate() -> None:
    comparisons = {
        "pen_context_vs_parameter_matched_zero": _comparison(
            macro_delta=0.02,
            focus_delta=0.02,
            ci_low=0.005,
            nll_delta=-0.02,
            rare_delta=-0.01,
        ),
        "pen_context_vs_availability_only": _comparison(
            macro_delta=0.015,
            focus_delta=0.01,
            ci_low=0.002,
            nll_delta=-0.01,
            rare_delta=0.0,
        ),
        "availability_only_vs_parameter_matched_zero": _comparison(
            macro_delta=0.005,
            focus_delta=0.0,
            ci_low=-0.01,
            nll_delta=0.0,
            rare_delta=0.0,
        ),
    }

    decision = make_pen_context_decision(
        comparisons,
        contract=_contract(),
    )

    assert decision["full_pen_context_expansion_authorized"] is True
    assert decision["decision"] == (
        "RETAIN_PEN_CONTEXT_FOR_FULL_LEGACY_DEVELOPMENT"
    )


def test_pen_context_rejection_is_valid_and_stops_full_expansion() -> None:
    comparisons = {
        "pen_context_vs_parameter_matched_zero": _comparison(
            macro_delta=0.0,
            focus_delta=-0.01,
            ci_low=-0.02,
            nll_delta=-0.01,
            rare_delta=-0.03,
        ),
        "pen_context_vs_availability_only": _comparison(
            macro_delta=0.001,
            focus_delta=0.0,
            ci_low=-0.02,
            nll_delta=0.0,
            rare_delta=0.0,
        ),
        "availability_only_vs_parameter_matched_zero": _comparison(
            macro_delta=0.002,
            focus_delta=0.0,
            ci_low=-0.01,
            nll_delta=0.0,
            rare_delta=0.0,
        ),
    }

    decision = make_pen_context_decision(
        comparisons,
        contract=_contract(),
    )

    assert decision["full_pen_context_expansion_authorized"] is False
    assert decision["negative_result_is_valid_evidence"] is True
    assert decision["decision"] == (
        "DO_NOT_EXPAND_PEN_CONTEXT_FROM_CURRENT_SHORT_EVIDENCE"
    )
    assert decision["next_action"] == (
        "retain_parameter_matched_zero_and_stop_pen_context_expansion"
    )
