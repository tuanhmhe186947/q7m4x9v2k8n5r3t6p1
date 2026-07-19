from __future__ import annotations

from typing import Any

from pig_behavior.classification_v2.evaluation.legacy_c6_full_development_freeze import (
    make_c6_full_development_decision,
)

CONTRACT = {
    "minimum_macro_f1_gain": 0.02,
    "maximum_absolute_availability_only_gain": 0.01,
    "maximum_rare_group_macro_f1_drop": 0.02,
    "require_positive_video_cluster_ci_low": True,
    "require_nll_improvement_vs_zero": True,
}


def test_full_gate_rejects_roi_and_union_observed_evidence() -> None:
    full = {
        "comparisons": {
            "roi__real_minus_parameter_matched_zero": _comparison(
                gain=0.0173996682,
                baseline_f1=0.3929828652,
                rare_delta=0.0,
                nll_delta=-0.0674027087,
                ci_low=-0.0453411623,
            ),
            "roi__real_minus_availability_only": _comparison(
                gain=0.0135663710,
                baseline_f1=0.3968161624,
                rare_delta=0.0,
                nll_delta=-0.0528438634,
                ci_low=-0.0483950936,
            ),
            "union_context__real_minus_parameter_matched_zero": _comparison(
                gain=0.0317916340,
                baseline_f1=0.3605383962,
                rare_delta=-0.0444444444,
                nll_delta=0.1611725788,
                ci_low=-0.0382937684,
            ),
            "union_context__real_minus_availability_only": _comparison(
                gain=0.0405772639,
                baseline_f1=0.3517527663,
                rare_delta=-0.0353535354,
                nll_delta=0.1394782465,
                ci_low=-0.0222660131,
            ),
        }
    }
    counts = {
        "roi": {control: [70833] for control in _controls()},
        "union_context": {control: [135053] for control in _controls()},
    }

    decision = make_c6_full_development_decision(
        full,
        parameter_counts=counts,
        selected_modalities=("roi", "union_context"),
        contract=CONTRACT,
    )

    assert decision["errors"] == []
    assert decision["retained_modalities"] == []
    assert decision["modality_decisions"]["roi"]["criteria"][
        "gain_vs_zero_meets_margin"
    ] is False
    union_criteria = decision["modality_decisions"]["union_context"][
        "criteria"
    ]
    assert union_criteria["nll_improves_vs_zero"] is False
    assert union_criteria["rare_group_drop_within_limit"] is False


def test_full_gate_retains_only_when_every_criterion_passes() -> None:
    full = {
        "comparisons": {
            "roi__real_minus_parameter_matched_zero": _comparison(
                gain=0.03,
                baseline_f1=0.30,
                rare_delta=-0.01,
                nll_delta=-0.05,
                ci_low=0.001,
            ),
            "roi__real_minus_availability_only": _comparison(
                gain=0.025,
                baseline_f1=0.305,
                rare_delta=-0.01,
                nll_delta=-0.04,
                ci_low=0.002,
            ),
        }
    }
    counts = {
        "roi": {control: [70833] for control in _controls()},
    }

    decision = make_c6_full_development_decision(
        full,
        parameter_counts=counts,
        selected_modalities=("roi",),
        contract=CONTRACT,
    )

    assert decision["errors"] == []
    assert decision["retained_modalities"] == ["roi"]


def _comparison(
    *,
    gain: float,
    baseline_f1: float,
    rare_delta: float,
    nll_delta: float,
    ci_low: float,
) -> dict[str, Any]:
    return {
        "macro_f1_delta": gain,
        "baseline_metrics": {
            "macro_f1_global_10_class": baseline_f1,
            "nll": 1.0,
        },
        "candidate_metrics": {
            "macro_f1_global_10_class": baseline_f1 + gain,
            "nll": 1.0 + nll_delta,
        },
        "group_deltas": {
            "rare": {"macro_f1_delta": rare_delta},
        },
        "video_cluster_bootstrap": {
            "ci_low": ci_low,
            "ci_high": ci_low + 0.1,
        },
    }


def _controls() -> tuple[str, ...]:
    return "parameter_matched_zero", "availability_only", "real"
