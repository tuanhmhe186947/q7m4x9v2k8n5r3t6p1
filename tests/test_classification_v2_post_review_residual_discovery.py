from __future__ import annotations

import pandas as pd

from pig_behavior.classification_v2.review.post_review_residual_discovery import (
    activate_post_review_scope_for_gui,
    build_review_informed_temporal_residuals,
)


def _unit(index: int, behavior: str) -> dict[str, str]:
    start = index * 6
    key = f"key-{index}"
    return {
        "review_item_id": f"unit-{index}",
        "review_unit_id": key,
        "temporal_unit_key": key,
        "source_type": "cvat_tracking_xml",
        "dataset_id": "dataset-1",
        "video_key": "video-1",
        "object_track_key": "track-1",
        "track_id": "1",
        "unit_start_frame": str(start),
        "unit_end_frame": str(start + 5),
        "behavior_label": behavior,
        "manual_review_decision": "",
        "manual_corrected_behavior": "",
    }


def _decision(
    index: int,
    source_behavior: str,
    decision: str,
    corrected_behavior: str = "",
) -> dict[str, str]:
    key = f"key-{index}"
    return {
        "review_unit_id": key,
        "temporal_unit_key": key,
        "behavior_label": source_behavior,
        "manual_review_decision": decision,
        "manual_corrected_behavior": corrected_behavior,
    }


def test_review_correction_exposes_two_unreviewed_fight_gap_units() -> None:
    universe = pd.DataFrame(
        [
            _unit(0, "social-nose"),
            _unit(1, "move"),
            _unit(2, "move"),
            _unit(3, "fight"),
        ]
    )
    decisions = pd.DataFrame(
        [
            _decision(0, "social-nose", "corrected", "fight"),
            _decision(3, "fight", "accept"),
        ]
    )

    result = build_review_informed_temporal_residuals(universe, decisions)

    selected = result["selected_findings"]
    assert set(selected["temporal_unit_key"]) == {"key-1", "key-2"}
    assert set(selected["severity"]) == {"HIGH"}
    assert set(selected["suggested_review_hypothesis"]) == {"fight"}
    assert not selected["automatic_label_change"].any()
    assert result["audit"]["reviewed_target_overlap"] == 0
    assert result["audit"]["control_exclusion_rows"] == 4
    assert set(result["selected_scope"]["candidate_tier"]) == {
        "POST_REVIEW_RESIDUAL_TARGET"
    }
    assert result["selected_scope"]["include_in_review"].all()


def test_severity_filter_bounds_scope_without_hiding_audit_findings() -> None:
    universe = pd.DataFrame(
        [
            _unit(0, "social-nose"),
            _unit(1, "move"),
            _unit(2, "move"),
            _unit(3, "fight"),
        ]
    )
    decisions = pd.DataFrame(
        [
            _decision(0, "social-nose", "corrected", "fight"),
            _decision(3, "fight", "accept"),
        ]
    )

    result = build_review_informed_temporal_residuals(
        universe,
        decisions,
        included_severities=("MEDIUM",),
    )

    assert len(result["findings"]) == 2
    assert set(result["findings"]["severity"]) == {"HIGH"}
    assert result["selected_findings"].empty
    assert result["selected_scope"].empty
    assert result["audit"]["included_severities"] == ["MEDIUM"]


def test_unreviewed_gap_without_corrected_neighbor_is_audit_only() -> None:
    universe = pd.DataFrame(
        [
            _unit(0, "stand"),
            _unit(1, "move"),
            _unit(2, "stand"),
        ]
    )
    decisions = pd.DataFrame(
        [
            _decision(0, "stand", "accept"),
            _decision(2, "stand", "accept"),
        ]
    )

    result = build_review_informed_temporal_residuals(universe, decisions)

    assert len(result["findings"]) == 1
    assert result["selected_findings"].empty
    assert result["selected_scope"].empty


def test_gui_activation_preserves_parent_auto_carry_provenance() -> None:
    scope = pd.DataFrame(
        [
            {
                **_unit(0, "stand"),
                "candidate_tier": "AUTO_CARRY_LOW_RISK",
                "include_in_review": False,
                "review_reason": "",
                "review_reason_codes": "",
                "review_selection_predicates": "",
                "auto_carry_behavior": "stand",
                "auto_carry_provenance": "SOURCE_LABEL",
            }
        ]
    )

    activated = activate_post_review_scope_for_gui(
        scope,
        cohort="POST_REVIEW_RESIDUAL_CONTROL",
        reason_code="INDEPENDENT_RESIDUAL_CONTROL",
    )

    assert activated.at[0, "candidate_tier"] == "POST_REVIEW_RESIDUAL_CONTROL"
    assert bool(activated.at[0, "include_in_review"])
    assert not bool(activated.at[0, "review_predicate_global_mandatory"])
    assert activated.at[0, "auto_carry_behavior"] == ""
    assert (
        activated.at[0, "post_review_parent_candidate_tier"]
        == "AUTO_CARRY_LOW_RISK"
    )
    assert activated.at[0, "post_review_parent_auto_carry_behavior"] == "stand"


def test_distant_correction_inside_long_flank_does_not_expand_scope() -> None:
    universe = pd.DataFrame(
        [
            _unit(0, "social-nose"),
            _unit(1, "fight"),
            _unit(2, "move"),
            _unit(3, "fight"),
        ]
    )
    decisions = pd.DataFrame(
        [
            _decision(0, "social-nose", "corrected", "fight"),
            _decision(1, "fight", "accept"),
            _decision(3, "fight", "accept"),
        ]
    )

    result = build_review_informed_temporal_residuals(universe, decisions)

    assert len(result["findings"]) == 1
    assert result["selected_findings"].empty
