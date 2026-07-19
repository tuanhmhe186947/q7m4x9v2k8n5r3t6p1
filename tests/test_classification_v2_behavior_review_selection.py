from __future__ import annotations

import pandas as pd

from pig_behavior.classification_v2.review.behavior_review_selection import (
    BehaviorReviewSelectionConfig,
    assign_behavior_review_cohorts,
)


def _units(
    count: int,
    *,
    behavior: str = "sitting",
    source: str = "cvat_tracking_xml",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_unit_id": [f"unit-{index:03d}" for index in range(count)],
            "source_type": source,
            "video_key": "video-1",
            "behavior_label": behavior,
            "temporal_consistency_status": "stable",
            "review_reason": "",
            "review_priority": [float(index) for index in range(count)],
            "review_unit_type": (
                "legacy_burst_16"
                if source == "legacy_recovered"
                else "cvat_interval_6"
            ),
        }
    )


def test_selection_builds_disjoint_calibrated_random_and_control_cohorts() -> None:
    selected, audit = assign_behavior_review_cohorts(
        _units(50),
        config=BehaviorReviewSelectionConfig(
            calibrated_high_risk_fraction=0.10,
            calibrated_high_risk_max_per_stratum=10,
            calibrated_high_risk_min_pool=20,
            random_per_stratum=5,
            clean_control_per_stratum=1,
        ),
    )

    assert audit["valid"] is True
    assert audit["cohort_counts"] == {
        "behavior_not_selected": 39,
        "behavior_high_risk": 5,
        "behavior_random_audit": 5,
        "behavior_clean_control": 1,
    }
    high = selected[selected["behavior_review_cohort"].eq("behavior_high_risk")]
    assert high["review_unit_id"].tolist() == [
        "unit-045",
        "unit-046",
        "unit-047",
        "unit-048",
        "unit-049",
    ]
    random = selected[
        selected["behavior_review_cohort"].eq("behavior_random_audit")
    ]
    assert random["behavior_sampling_probability"].eq(5 / 45).all()
    assert random["behavior_sampling_weight"].eq(9.0).all()
    assert random["behavior_review_residual_estimand"].all()
    assert selected["include_in_review"].sum() == 11


def test_mandatory_and_rare_census_rules_are_explicit() -> None:
    rows = pd.concat(
        [
            _units(2, behavior="eat"),
            _units(1, behavior="fight").assign(review_unit_id="fight-1"),
            _units(1, behavior="move").assign(
                review_unit_id="transition-1",
                temporal_consistency_status="transition",
            ),
        ],
        ignore_index=True,
    )
    selected, _ = assign_behavior_review_cohorts(
        rows,
        config=BehaviorReviewSelectionConfig(
            random_per_stratum=0,
            clean_control_per_stratum=0,
            calibrated_high_risk_fraction=0.0,
        ),
    )

    assert selected["include_in_review"].all()
    assert selected["behavior_review_cohort"].eq(
        "behavior_mandatory_census"
    ).all()
    assert selected["review_reason"].str.contains(
        "mandatory_behavior_review",
        regex=False,
    ).all()


def test_legacy_sampling_requires_explicit_complete_legacy_flag() -> None:
    rows = _units(3, behavior="eat", source="legacy_recovered")
    config = BehaviorReviewSelectionConfig(
        random_per_stratum=5,
        clean_control_per_stratum=1,
        calibrated_high_risk_fraction=0.10,
    )

    selective, _ = assign_behavior_review_cohorts(rows, config=config)
    complete, _ = assign_behavior_review_cohorts(
        rows,
        config=config,
        include_all_retained_legacy_units=True,
    )

    assert not selective["include_in_review"].any()
    assert complete["include_in_review"].all()
    assert complete["behavior_review_cohort"].eq(
        "behavior_mandatory_census"
    ).all()


def test_selection_is_deterministic_and_does_not_change_labels() -> None:
    rows = _units(50, behavior="lying")
    first, _ = assign_behavior_review_cohorts(rows)
    second, _ = assign_behavior_review_cohorts(rows)

    columns = [
        "review_unit_id",
        "behavior_label",
        "behavior_review_cohort",
        "behavior_sampling_probability",
        "behavior_sampling_weight",
        "include_in_review",
    ]
    pd.testing.assert_frame_equal(first[columns], second[columns])
    assert first["behavior_label"].equals(rows["behavior_label"])
