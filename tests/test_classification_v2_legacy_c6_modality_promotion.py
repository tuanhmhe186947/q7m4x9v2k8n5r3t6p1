from __future__ import annotations

from typing import Any

from pig_behavior.classification_v2.evaluation.legacy_c6_modality_promotion import (
    MODALITIES,
    make_c6_modality_promotion_decision,
)

CONTRACT = {
    "minimum_macro_f1_gain": 0.02,
    "pen_minimum_macro_f1_gain": 0.01,
    "pen_minimum_focus_group_macro_f1_gain": 0.01,
    "maximum_absolute_availability_only_gain": 0.01,
    "maximum_rare_group_macro_f1_drop": 0.02,
    "require_positive_video_cluster_ci_low": True,
    "require_nll_improvement_vs_zero": True,
}


def test_gate_recovers_only_roi_and_union_from_short_evidence() -> None:
    observed = {
        "geometry": (0.004719, 0.006910, -0.002191, -0.009598, 0.008775,
                     -0.006470, -0.003786),
        "motion": (0.012371, 0.017468, -0.005097, 0.005272, -0.005448,
                   -0.012469, -0.003962),
        "roi": (0.086681, 0.085218, 0.001463, 0.017962, -0.106613,
                0.042188, 0.037400),
        "numeric_social": (0.003945, 0.009663, -0.005719, 0.008724,
                           -0.044413, -0.024596, -0.018577),
        "pen_context": (0.022904, 0.010202, 0.012701, 0.023582, -0.021852,
                        0.005389, -0.007773),
        "union_context": (0.133285, 0.128320, 0.004965, 0.058277, -0.303780,
                          0.074656, 0.069622),
        "full_frame_context": (0.092414, 0.103896, -0.011482, 0.055195,
                               -0.131047, 0.032549, 0.041103),
    }
    comparisons: dict[str, Any] = {}
    for modality, values in observed.items():
        zero_delta, avail_delta, diagnostic, rare, nll, zero_ci, avail_ci = values
        comparisons[f"{modality}__real_minus_parameter_matched_zero"] = (
            _comparison(zero_delta, 0.0, rare, nll, zero_ci)
        )
        comparisons[f"{modality}__real_minus_availability_only"] = (
            _comparison(avail_delta, diagnostic, rare, nll, avail_ci)
        )
    counts = {
        modality: {
            "parameter_matched_zero": [100, 100],
            "availability_only": [100, 100],
            "real": [100, 100],
        }
        for modality in MODALITIES
    }

    decision = make_c6_modality_promotion_decision(
        {"comparisons": comparisons},
        parameter_counts=counts,
        contract=CONTRACT,
    )

    assert decision["errors"] == []
    assert decision["full_development_authorized_modalities"] == [
        "roi",
        "union_context",
    ]
    assert decision["modality_decisions"]["full_frame_context"][
        "criteria"
    ]["availability_only_is_bounded_diagnostic"] is False


def test_gate_fails_closed_on_parameter_mismatch() -> None:
    comparisons: dict[str, Any] = {}
    for modality in MODALITIES:
        comparisons[f"{modality}__real_minus_parameter_matched_zero"] = (
            _comparison(0.05, 0.0, 0.0, -0.1, 0.01)
        )
        comparisons[f"{modality}__real_minus_availability_only"] = (
            _comparison(0.05, 0.0, 0.0, -0.1, 0.01)
        )
    counts = {
        modality: {
            "parameter_matched_zero": [100, 100],
            "availability_only": [100, 100],
            "real": [100, 100],
        }
        for modality in MODALITIES
    }
    counts["roi"]["real"] = [101, 101]

    decision = make_c6_modality_promotion_decision(
        {"comparisons": comparisons},
        parameter_counts=counts,
        contract=CONTRACT,
    )

    assert decision["modality_decisions"]["roi"][
        "full_development_authorized"
    ] is False
    assert decision["modality_decisions"]["roi"]["criteria"][
        "all_modes_parameter_matched"
    ] is False


def _comparison(
    delta: float,
    baseline_shift: float,
    rare_delta: float,
    nll_delta: float,
    ci_low: float,
) -> dict[str, Any]:
    baseline_f1 = 0.2 + baseline_shift
    per_class = {
        label: {"f1_delta": 0.02}
        for label in ("stand", "move", "explore")
    }
    return {
        "macro_f1_delta": delta,
        "baseline_metrics": {
            "macro_f1_global_10_class": baseline_f1,
            "nll": 2.0,
        },
        "candidate_metrics": {
            "macro_f1_global_10_class": baseline_f1 + delta,
            "nll": 2.0 + nll_delta,
        },
        "group_deltas": {
            "rare": {"macro_f1_delta": rare_delta},
        },
        "per_class": per_class,
        "video_cluster_bootstrap": {
            "ci_low": ci_low,
            "ci_high": ci_low + 0.1,
        },
    }
