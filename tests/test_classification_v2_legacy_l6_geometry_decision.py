from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pig_behavior.classification_v2.evaluation.legacy_development_l6_geometry_decision import (
    _validate_config,
    _validate_full_config,
    make_geometry_decision,
    make_geometry_full_decision,
)


def _comparison(
    *,
    macro_delta: float,
    nll_delta: float,
    ci_low: float,
    rare_delta: float,
) -> dict[str, object]:
    return {
        "delta_candidate_minus_baseline": {
            "macro_f1_global_10_class": macro_delta,
            "accuracy": 0.01,
            "nll": nll_delta,
        },
        "video_cluster_bootstrap": {
            "ci_low": ci_low,
            "ci_high": 0.08,
        },
        "confusion_groups": {
            "rare": {
                "macro_f1_delta": rare_delta,
            }
        },
    }


def _contract() -> dict[str, object]:
    return {
        "minimum_macro_f1_gain": 0.02,
        "maximum_absolute_availability_only_gain": 0.01,
        "maximum_rare_group_macro_f1_drop": 0.02,
        "require_positive_video_cluster_ci_low": True,
        "require_nll_improvement_vs_zero": True,
    }


def test_geometry_decision_requires_paired_cluster_support() -> None:
    comparisons = {
        "geometry_vs_parameter_matched_zero": _comparison(
            macro_delta=0.03,
            nll_delta=-0.02,
            ci_low=0.001,
            rare_delta=0.0,
        ),
        "geometry_vs_availability_only": _comparison(
            macro_delta=0.031,
            nll_delta=-0.02,
            ci_low=0.002,
            rare_delta=0.0,
        ),
        "availability_only_vs_parameter_matched_zero": _comparison(
            macro_delta=-0.001,
            nll_delta=0.0,
            ci_low=-0.01,
            rare_delta=0.0,
        ),
    }

    decision = make_geometry_decision(comparisons, contract=_contract())

    assert decision["full_geometry_expansion_authorized"] is True
    assert decision["decision"] == (
        "RETAIN_GEOMETRY_FOR_FULL_LEGACY_DEVELOPMENT"
    )
    failed = copy.deepcopy(comparisons)
    failed["geometry_vs_parameter_matched_zero"][
        "video_cluster_bootstrap"
    ]["ci_low"] = -0.001
    rejected = make_geometry_decision(failed, contract=_contract())
    assert rejected["full_geometry_expansion_authorized"] is False
    assert rejected["criteria"][
        "geometry_vs_zero_cluster_ci_low_positive"
    ] is False


def test_geometry_full_decision_preserves_valid_negative_evidence() -> None:
    comparisons = {
        "geometry_vs_parameter_matched_zero": _comparison(
            macro_delta=-0.038,
            nll_delta=-0.069,
            ci_low=-0.09,
            rare_delta=-0.01,
        ),
        "geometry_vs_availability_only": _comparison(
            macro_delta=0.006,
            nll_delta=-0.037,
            ci_low=-0.03,
            rare_delta=0.0,
        ),
        "availability_only_vs_parameter_matched_zero": _comparison(
            macro_delta=-0.044,
            nll_delta=-0.032,
            ci_low=-0.09,
            rare_delta=0.0,
        ),
    }

    decision = make_geometry_full_decision(
        comparisons,
        contract=_contract(),
    )

    assert decision["geometry_retained_for_next_l6_modality"] is False
    assert decision["negative_result_is_valid_evidence"] is True
    assert decision["decision"] == (
        "REJECT_GEOMETRY_AS_UNSUPPORTED_IN_FULL_LEGACY_DEVELOPMENT"
    )
    assert decision["next_action"] == (
        "continue_l6_motion_from_parameter_matched_zero"
    )


def test_geometry_decision_config_locks_predeclared_thresholds() -> None:
    path = Path(
        "configs/classification_v2/"
        "legacy_development_l6_geometry_decision_v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    _validate_config(payload)

    changed = copy.deepcopy(payload)
    changed["decision_contract"]["minimum_macro_f1_gain"] = 0.0
    with pytest.raises(ValueError, match="decision contract"):
        _validate_config(changed)


def test_geometry_full_decision_config_locks_negative_evidence() -> None:
    path = Path(
        "configs/classification_v2/"
        "legacy_development_l6_geometry_full_decision_v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    _validate_full_config(payload)

    changed = copy.deepcopy(payload)
    changed["interpretation_boundary"][
        "negative_result_is_valid_evidence"
    ] = False
    with pytest.raises(ValueError, match="interpretation boundary"):
        _validate_full_config(changed)
