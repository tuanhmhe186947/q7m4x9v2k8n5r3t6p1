"""Evaluation-population utilities for cross-length comparisons."""

from __future__ import annotations

import pytest

from pig_behavior.classification_v2.temporal_views.matched_cohort import (
    EVALUATION_POPULATIONS,
    EvaluationPopulationError,
    LengthComparison,
    all_eligible,
    build_view_eligibility,
    common_matched_cohort,
    evaluation_population_report,
    length_conclusion_guard,
)

VIEW_LENGTHS = {
    "T6_TARGET_CONTIGUOUS": 40,
    "T8_TARGET_CONTIGUOUS": 32,
    "T12_TARGET_CONTIGUOUS": 24,
    "T16_TARGET_CONTIGUOUS": 18,
}


def _eligibilities():
    units = [f"native_unit_{index:03d}" for index in range(40)]
    items = []
    for view, count in VIEW_LENGTHS.items():
        eligible = units[:count]
        excluded = {
            unit: "insufficient_trailing_frames" for unit in units[count:]
        }
        items.append(
            build_view_eligibility(
                view,
                eligible_native_unit_ids=eligible,
                exclusion_reasons=excluded,
            )
        )
    return items


def test_populations_are_named_exactly() -> None:
    assert EVALUATION_POPULATIONS == ("ALL_ELIGIBLE", "COMMON_MATCHED_COHORT")


def test_all_eligible_is_per_view_and_independent() -> None:
    items = _eligibilities()
    for item in items:
        assert len(all_eligible(item)) == VIEW_LENGTHS[item.view_name]


def test_common_cohort_is_the_exact_intersection() -> None:
    items = _eligibilities()
    cohort = common_matched_cohort(items)
    expected = set(all_eligible(items[0]))
    for item in items[1:]:
        expected &= set(all_eligible(item))
    assert cohort == expected
    assert len(cohort) == min(VIEW_LENGTHS.values())


def test_report_identifies_the_population_and_the_drop() -> None:
    report = evaluation_population_report(_eligibilities())
    assert report["populations"] == list(EVALUATION_POPULATIONS)
    assert report["common_matched_cohort_is_exact_intersection"] is True
    assert report["common_matched_cohort_size"] == 18
    assert report["per_view_all_eligible"]["T6_TARGET_CONTIGUOUS"] == 40
    assert report["per_view_dropped_relative_to_cohort"]["T6_TARGET_CONTIGUOUS"] == 22
    reasons = report["views"][3]["exclusion_reason_counts"]
    assert reasons == {"insufficient_trailing_frames": 22}


def test_a_longer_view_cannot_win_on_all_eligible_alone() -> None:
    guard = length_conclusion_guard(
        LengthComparison(
            reference_view="T6_TARGET_CONTIGUOUS",
            candidate_view="T16_TARGET_CONTIGUOUS",
            all_eligible_gain=0.08,
            common_matched_cohort_gain=None,
        )
    )
    assert guard["successful"] is False
    assert guard["decided_on_population"] == "COMMON_MATCHED_COHORT"
    assert any("support artifact" in reason for reason in guard["blocking_reasons"])


def test_matched_cohort_gain_below_the_minimum_effect_is_rejected() -> None:
    guard = length_conclusion_guard(
        LengthComparison(
            reference_view="T6_TARGET_CONTIGUOUS",
            candidate_view="T12_TARGET_CONTIGUOUS",
            all_eligible_gain=0.09,
            common_matched_cohort_gain=0.005,
        )
    )
    assert guard["successful"] is False

    passing = length_conclusion_guard(
        LengthComparison(
            reference_view="T6_TARGET_CONTIGUOUS",
            candidate_view="T12_TARGET_CONTIGUOUS",
            all_eligible_gain=0.09,
            common_matched_cohort_gain=0.04,
        )
    )
    assert passing["successful"] is True
    assert passing["blocking_reasons"] == []


def test_target_view_and_history_families_are_never_pooled() -> None:
    with pytest.raises(EvaluationPopulationError, match="different"):
        length_conclusion_guard(
            LengthComparison(
                reference_view="T6_TARGET_CONTIGUOUS",
                candidate_view="T6_TARGET_PLUS_H12",
                all_eligible_gain=0.05,
                common_matched_cohort_gain=0.05,
            )
        )


def test_history_family_comparisons_are_allowed_within_the_family() -> None:
    guard = length_conclusion_guard(
        LengthComparison(
            reference_view="T6_TARGET_PLUS_H6",
            candidate_view="T6_TARGET_PLUS_H24",
            all_eligible_gain=0.05,
            common_matched_cohort_gain=0.03,
        )
    )
    assert guard["family"] == "TARGET_PLUS_CAUSAL_HISTORY"
    assert guard["successful"] is True


def test_eligibility_input_is_validated() -> None:
    with pytest.raises(EvaluationPopulationError):
        build_view_eligibility(
            "T6_TARGET_CONTIGUOUS",
            eligible_native_unit_ids=["a", "a"],
        )
    with pytest.raises(EvaluationPopulationError):
        build_view_eligibility(
            "T6_TARGET_CONTIGUOUS",
            eligible_native_unit_ids=["a"],
            exclusion_reasons={"a": "also_excluded"},
        )
    with pytest.raises(ValueError, match="unknown temporal view"):
        build_view_eligibility("T7_TARGET_CONTIGUOUS", eligible_native_unit_ids=["a"])
