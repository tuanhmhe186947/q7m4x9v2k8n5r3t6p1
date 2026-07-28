from __future__ import annotations

import copy
import json

import pandas as pd

from pig_behavior.classification_v2.review.behavior_threshold_audit import (
    independent_threshold_candidate_audit,
    threshold_sensitivity_analysis,
)
from pig_behavior.classification_v2.review.behavior_threshold_registry import (
    threshold_by_name,
    threshold_registry_hash,
    threshold_registry_snapshot,
)


def _threshold_record(observed: float) -> dict[str, object]:
    entry = threshold_by_name("low_motion_support")
    return {
        "predicate_id": "MOTION_CONTRADICTION",
        "threshold_id": entry.threshold_id,
        "metric_id": entry.metric_id,
        "metric_version": entry.metric_version,
        "metric_units": entry.metric_units,
        "feature_name": entry.feature_name,
        "observed_feature_value": observed,
        "comparison_operator": entry.comparison_operator,
        "threshold_value": entry.threshold_value,
        "authority_hash": entry.authority_hash,
        "threshold_semantic_hash": entry.semantic_hash,
        "reason_code": "move_with_weak_motion_evidence",
    }


def _partition(record: dict[str, object]) -> tuple[pd.DataFrame, ...]:
    universe = pd.DataFrame({"review_key": ["candidate", "auto"]})
    candidates = pd.DataFrame(
        {
            "review_key": ["candidate"],
            "review_reason_codes": ["move_with_weak_motion_evidence"],
            "review_selection_predicates": [
                "review_predicate_motion_contradiction"
            ],
            "threshold_binding_details": [json.dumps([record])],
        }
    )
    auto_carry = pd.DataFrame({"review_key": ["auto"]})
    return universe, candidates, auto_carry


def test_independent_checker_accepts_authoritative_true_comparison() -> None:
    audit = independent_threshold_candidate_audit(
        *_partition(_threshold_record(0.1)),
        threshold_registry_snapshot(),
    )

    assert audit["valid"] is True
    assert audit["checked_threshold_comparisons"] == 1
    assert audit["threshold_candidates_without_authority"] == 0


def test_independent_checker_rejects_false_published_comparison() -> None:
    audit = independent_threshold_candidate_audit(
        *_partition(_threshold_record(0.9)),
        threshold_registry_snapshot(),
    )

    assert audit["valid"] is False
    assert any("comparison_false" in error for error in audit["errors"])


def test_sensitivity_analysis_does_not_mutate_frozen_registry() -> None:
    snapshot = threshold_registry_snapshot()
    before = copy.deepcopy(snapshot)
    decision = json.dumps([_threshold_record(0.2)])
    evaluations = json.dumps(
        [_threshold_record(0.2), _threshold_record(0.26)]
    )
    units = pd.DataFrame(
        {
            "review_key": ["selected", "nearby"],
            "behavior": ["move", "move"],
            "source": ["legacy", "legacy"],
            "recording_date": ["2019-11-29", "2019-11-29"],
            "include_in_review": [True, False],
            "threshold_binding_details": [decision, "[]"],
            "review_threshold_evaluations": [evaluations, evaluations],
        }
    )

    result = threshold_sensitivity_analysis(units, snapshot)

    selected = result[
        result["threshold_id"].eq(
            threshold_by_name("low_motion_support").threshold_id
        )
    ]
    assert len(selected) == 3
    assert not selected["registry_modified"].any()
    assert snapshot == before
    assert threshold_registry_hash() == snapshot["registry_hash"]
