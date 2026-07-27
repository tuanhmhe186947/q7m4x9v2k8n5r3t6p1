from __future__ import annotations

import pandas as pd
import pytest

from pig_behavior.classification_v2.review.behavior_review_selection import (
    PREDICATE_COLUMNS,
    BehaviorReviewSelectionConfig,
    assign_behavior_review_cohorts,
)
from pig_behavior.classification_v2.review.review_unit_builder import (
    _build_pilot_sample,
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
            "review_evidence_available": True,
            "review_relevant_evidence_available": True,
            "review_evidence_reason_auto": "",
            "review_evidence_status_auto": "sufficient",
            "review_unit_type": (
                "legacy_burst_16"
                if source == "legacy_recovered"
                else "cvat_interval_6"
            ),
        }
    )


def test_selection_builds_disjoint_stratified_low_risk_audit() -> None:
    selected, audit = assign_behavior_review_cohorts(
        _units(50),
        config=BehaviorReviewSelectionConfig(
            random_per_stratum=5,
        ),
    )

    assert audit["valid"] is True
    assert audit["cohort_counts"] == {
        "behavior_not_selected": 45,
        "behavior_random_audit": 5,
    }
    random = selected[
        selected["behavior_review_cohort"].eq("behavior_random_audit")
    ]
    assert random["behavior_sampling_probability"].eq(5 / 50).all()
    assert random["behavior_sampling_weight"].eq(10.0).all()
    assert random["behavior_review_residual_estimand"].all()
    assert random["candidate_tier"].eq("TIER_3_STRATIFIED_AUDIT").all()
    assert selected["include_in_review"].sum() == 5


def test_pilot_includes_stratified_audit_and_auto_carry_examples() -> None:
    selected, _ = assign_behavior_review_cohorts(
        _units(20),
        config=BehaviorReviewSelectionConfig(
            random_per_stratum=3,
        ),
    )

    pilot = _build_pilot_sample(selected)

    reasons = set(pilot["pilot_reason"])
    assert "review_predicate_stratified_low_risk_audit" in reasons
    assert "AUTO_CARRY_LOW_RISK" in reasons


def test_mandatory_and_rare_census_rules_are_explicit() -> None:
    rows = pd.concat(
        [
            _units(2, behavior="eat"),
            _units(1, behavior="fight").assign(review_unit_id="fight-1"),
            _units(1, behavior="playwithtoy").assign(
                review_unit_id="rare-1",
            ),
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

    selected_rows = selected[selected["include_in_review"]]
    assert selected_rows["review_unit_id"].tolist() == [
        "rare-1",
        "transition-1",
    ]
    assert selected_rows["behavior_review_cohort"].eq(
        "behavior_mandatory_census"
    ).all()
    assert selected.loc[
        selected["review_unit_id"].eq("rare-1"),
        "review_reason_codes",
    ].str.contains(
        "RARE_CLASS_CENSUS_PLAYWITHTOY",
        regex=False,
    ).item()
    assert not selected.loc[
        selected["behavior_label"].isin({"eat", "fight"}),
        "include_in_review",
    ].any()


def test_global_legacy_selection_flag_is_rejected() -> None:
    rows = _units(3, behavior="eat", source="legacy_recovered")
    config = BehaviorReviewSelectionConfig(
        random_per_stratum=0,
    )

    selective, _ = assign_behavior_review_cohorts(rows, config=config)
    assert not selective["include_in_review"].any()
    with pytest.raises(ValueError, match="unexplained_candidates=3"):
        assign_behavior_review_cohorts(
            rows,
            config=config,
            include_all_retained_legacy_units=True,
        )


def test_evidence_availability_and_behavior_label_do_not_select() -> None:
    rows = pd.concat(
        [
            _units(1, behavior="eat"),
            _units(1, behavior="fight").assign(review_unit_id="fight-1"),
            _units(1, behavior="move").assign(review_unit_id="move-1"),
        ],
        ignore_index=True,
    )
    selected, audit = assign_behavior_review_cohorts(
        rows,
        config=BehaviorReviewSelectionConfig(random_per_stratum=0),
    )

    assert audit["valid"]
    assert not selected["include_in_review"].any()
    assert audit["evidence_availability_only_candidates"] == 0


@pytest.mark.parametrize(
    ("behavior", "reason", "predicate"),
    [
        (
            "eat",
            "roi_label_without_persistent_target_support",
            "review_predicate_roi_contradiction",
        ),
        (
            "fight",
            "fight_without_persistent_contact_or_aggression",
            "review_predicate_interaction_contradiction",
        ),
        (
            "move",
            "move_with_weak_motion_evidence",
            "review_predicate_motion_contradiction",
        ),
        (
            "lying",
            "posture_label_with_strong_pixel_motion",
            "review_predicate_posture_contradiction",
        ),
    ],
)
def test_domain_contradiction_selects_specific_candidate(
    behavior: str,
    reason: str,
    predicate: str,
) -> None:
    rows = _units(1, behavior=behavior).assign(
        review_evidence_reason_auto=reason,
    )
    selected, audit = assign_behavior_review_cohorts(
        rows,
        config=BehaviorReviewSelectionConfig(random_per_stratum=0),
    )

    assert audit["valid"]
    assert bool(selected.iloc[0]["include_in_review"])
    assert bool(selected.iloc[0][predicate])
    assert reason in str(selected.iloc[0]["review_reason_codes"])
    assert set(PREDICATE_COLUMNS).issubset(selected.columns)


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
