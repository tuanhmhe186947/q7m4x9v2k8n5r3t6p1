from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from pig_behavior.classification_v2.features.context_policy import (
    apply_context_policy,
)
from pig_behavior.classification_v2.features.sequence_windows import (
    build_sequence_windows,
)
from pig_behavior.classification_v2.review.hidden_review_builder import (
    DECISION_COLUMNS,
    HiddenReviewConfig,
    apply_hidden_review_decisions,
    audit_hidden_decision_coverage,
    build_hidden_review_manifest,
)


def _frame_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index in range(12):
        source = "legacy_recovered" if index < 4 else "cvat_tracking_xml"
        hidden = "Yes" if index in {0, 4} else "No"
        rows.append(
            {
                "source_type": source,
                "dataset_id": "legacy" if index < 4 else "cvat",
                "video_key": "legacy_video" if index < 4 else "cvat_video",
                "frame_uid": f"frame_{index}",
                "frame_index": index * 6,
                "pig_id": f"ID_{(index % 4) + 1}",
                "track_id": f"track_{index % 4}",
                "object_id_in_image": index,
                "behavior": "fight" if index in {5, 6} else "stand",
                "hidden": hidden,
                "bbox_valid": True,
                "bbox_was_clipped": index in {5, 6},
                "x1": 10.0,
                "y1": 10.0,
                "x2": 110.0,
                "y2": 70.0,
                "nearest_pair_iou": 0.2 if index in {5, 6} else 0.0,
                "nearest_pair_overlap_ratio": 0.3 if index in {5, 6} else 0.0,
                "nearest_dist_n": 0.03 if index in {5, 6} else 0.5,
                "pair_contact_with_nearest": index in {5, 6},
                "shape_change_score": 0.5 if index in {5, 6} else 0.0,
                "delta_area_n": 0.3 if index in {5, 6} else 0.0,
                "sentinel": f"keep_{index}",
            }
        )
    return pd.DataFrame(rows)


def _build_review() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = _frame_rows()
    config = HiddenReviewConfig(
        random_no_per_stratum=1,
        clean_control_per_stratum=1,
        high_risk_threshold=0.35,
    )
    manifest, _, _ = build_hidden_review_manifest(frames, config=config)
    return frames, manifest


def _resolved_decisions(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, item in manifest.iterrows():
        row = {column: "" for column in DECISION_COLUMNS}
        row.update(
            {
                "hidden_review_item_id": item["hidden_review_item_id"],
                "hidden_before_review": item["hidden_before_review"],
                "hidden_after_review": item["hidden_before_review"],
                "hidden_review_status": "reviewed",
                "hidden_review_confidence": "high",
                "hidden_review_reason": "synthetic_test",
                "hidden_reviewer": "pytest",
                "hidden_reviewed_at": "2026-07-13T00:00:00",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=DECISION_COLUMNS)


def test_hidden_review_manifest_audits_both_yes_and_no() -> None:
    frames, manifest = _build_review()
    cohorts = set(manifest["hidden_review_cohort"])
    assert "hidden_yes_confirmation" in cohorts
    assert "hidden_no_high_risk" in cohorts
    assert "hidden_no_random_audit" in cohorts
    assert "hidden_no_clean_control" in cohorts
    random_rows = manifest.loc[manifest["hidden_review_cohort"].eq("hidden_no_random_audit")]
    assert random_rows["hidden_sampling_design"].eq("stratified_random_hidden_no").all()
    assert random_rows["hidden_sampling_probability"].between(0, 1).all()
    assert random_rows["hidden_sampling_weight"].ge(1).all()

    second, _, _ = build_hidden_review_manifest(
        frames,
        config=HiddenReviewConfig(random_no_per_stratum=1),
    )
    assert manifest["hidden_review_item_id"].tolist() == second["hidden_review_item_id"].tolist()
    cvat = manifest["source_type"].eq("cvat_tracking_xml")
    legacy = manifest["source_type"].eq("legacy_recovered")
    assert not manifest.loc[cvat, "hidden_is_trusted_before_review"].any()
    assert manifest.loc[legacy, "hidden_is_trusted_before_review"].all()


def test_hidden_review_census_targets_untrusted_yes_and_caps_risk() -> None:
    frames = _frame_rows()
    frames.loc[frames["source_type"].eq("legacy_recovered"), "hidden"] = "Yes"
    manifest, _, audit = build_hidden_review_manifest(
        frames,
        config=HiddenReviewConfig(
            trusted_yes_per_stratum=1,
            random_no_per_stratum=0,
            clean_control_per_stratum=0,
        ),
    )

    yes_review = manifest.loc[
        manifest["hidden_review_cohort"].eq("hidden_yes_confirmation")
    ]
    legacy_yes = yes_review["source_type"].eq("legacy_recovered")
    cvat_yes = yes_review["source_type"].eq("cvat_tracking_xml")
    high_risk = manifest.loc[
        manifest["hidden_review_cohort"].eq("hidden_no_high_risk")
    ]
    assert int(legacy_yes.sum()) == 1
    assert int(cvat_yes.sum()) == 1
    assert yes_review.loc[legacy_yes, "hidden_sampling_design"].eq(
        "stratified_trusted_hidden_yes_audit"
    ).all()
    assert yes_review.loc[cvat_yes, "hidden_sampling_design"].eq(
        "census_untrusted_hidden_yes"
    ).all()
    assert len(high_risk) == 1
    assert audit["input_trusted_hidden_yes_items"] == 4
    assert audit["selected_trusted_hidden_yes_items"] == 1
    assert audit["unselected_trusted_hidden_yes_items"] == 3
    assert audit["missing_untrusted_hidden_yes_items"] == 0


def test_hidden_review_strata_group_legacy_date_but_keep_cvat_videos() -> None:
    frames = _frame_rows().iloc[:4].copy()
    frames["source_type"] = [
        "legacy_recovered",
        "legacy_recovered",
        "cvat_tracking_xml",
        "cvat_tracking_xml",
    ]
    frames["dataset_id"] = [
        "legacy_pigs281119",
        "legacy_pigs281119",
        "cvat",
        "cvat",
    ]
    frames["video_key"] = [
        r"G:\pig_data\pigs281119a\burst_001",
        r"G:\pig_data\pigs281119a\burst_002",
        "Pigs281119_000085_30fps",
        "Pigs281119_000086_30fps",
    ]
    frames["hidden"] = "No"
    manifest, _, _ = build_hidden_review_manifest(
        frames,
        config=HiddenReviewConfig(
            trusted_yes_per_stratum=0,
            random_no_per_stratum=1,
            clean_control_per_stratum=0,
            high_risk_threshold=1.0,
        ),
    )

    random_audit = manifest.loc[
        manifest["hidden_review_cohort"].eq("hidden_no_random_audit")
    ]
    legacy = random_audit.loc[
        random_audit["source_type"].eq("legacy_recovered")
    ]
    cvat = random_audit.loc[
        random_audit["source_type"].eq("cvat_tracking_xml")
    ]
    assert len(legacy) == 1
    assert legacy.iloc[0]["hidden_review_stratum_key"] == (
        "recording_date=281119"
    )
    assert len(cvat) == 2
    assert set(cvat["hidden_review_stratum_key"]) == {
        "video=pigs281119_000085_30fps",
        "video=pigs281119_000086_30fps",
    }


def test_apply_hidden_review_supports_all_four_transitions_without_row_loss() -> None:
    frames, manifest = _build_review()
    decisions = _resolved_decisions(manifest)

    yes_item = manifest.loc[manifest["hidden_before_review"].eq("Yes")].iloc[0]
    no_item = manifest.loc[manifest["hidden_review_cohort"].eq("hidden_no_high_risk")].iloc[0]
    decisions.loc[
        decisions["hidden_review_item_id"].eq(yes_item["hidden_review_item_id"]),
        "hidden_after_review",
    ] = "No"
    decisions.loc[
        decisions["hidden_review_item_id"].eq(no_item["hidden_review_item_id"]),
        "hidden_after_review",
    ] = "Yes"

    reviewed, audit, confusion = apply_hidden_review_decisions(
        frames,
        manifest,
        decisions,
    )
    assert len(reviewed) == len(frames)
    assert reviewed["sentinel"].tolist() == frames["sentinel"].tolist()
    assert audit["yes_to_no_rows"] == 1
    assert audit["no_to_yes_rows"] == 1
    assert audit["corrected_hidden_rows"] == 2
    assert audit["errors"] == []
    assert confusion["reviewed_transition_counts"]["No->Yes"] == 1
    assert confusion["reviewed_transition_counts"]["Yes->No"] == 1

    unreviewed_cvat = reviewed["source_type"].eq("cvat_tracking_xml") & ~reviewed[
        "hidden_review_item_id"
    ].isin(decisions["hidden_review_item_id"])
    assert not reviewed.loc[unreviewed_cvat, "hidden_is_trusted"].any()


def test_unclear_hidden_decision_is_fail_closed_by_default() -> None:
    frames, manifest = _build_review()
    decisions = _resolved_decisions(manifest)
    decisions.loc[0, "hidden_review_status"] = "unclear"
    decisions.loc[0, "hidden_after_review"] = ""

    audit = audit_hidden_decision_coverage(manifest, decisions)
    assert any("unclear_decision_items" in error for error in audit["errors"])
    with pytest.raises(ValueError, match="coverage failed"):
        apply_hidden_review_decisions(frames, manifest, decisions)

    reviewed, _, _ = apply_hidden_review_decisions(
        frames,
        manifest,
        decisions,
        require_resolved=False,
    )
    item_id = decisions.loc[0, "hidden_review_item_id"]
    row = reviewed.loc[reviewed["hidden_review_item_id"].eq(item_id)].iloc[0]
    assert row["hidden_review_status"] == "unclear"
    assert not bool(row["hidden_is_trusted"])


def test_duplicate_hidden_decision_is_rejected() -> None:
    _, manifest = _build_review()
    decisions = _resolved_decisions(manifest)
    decisions = pd.concat([decisions, decisions.iloc[[0]]], ignore_index=True)
    audit = audit_hidden_decision_coverage(manifest, decisions)
    assert any("duplicate_decision_items" in error for error in audit["errors"])


def test_high_risk_caps_form_nested_review_waves() -> None:
    frames = _frame_rows()
    first, _, _ = build_hidden_review_manifest(
        frames,
        config=HiddenReviewConfig(
            random_no_per_stratum=0,
            clean_control_per_stratum=0,
            max_high_risk_per_stratum=1,
        ),
    )
    second, _, _ = build_hidden_review_manifest(
        frames,
        config=HiddenReviewConfig(
            random_no_per_stratum=0,
            clean_control_per_stratum=0,
            max_high_risk_per_stratum=2,
        ),
    )
    first_ids = set(
        first.loc[
            first["hidden_review_cohort"].eq("hidden_no_high_risk"),
            "hidden_review_item_id",
        ]
    )
    second_ids = set(
        second.loc[
            second["hidden_review_cohort"].eq("hidden_no_high_risk"),
            "hidden_review_item_id",
        ]
    )
    assert first_ids
    assert first_ids.issubset(second_ids)


def test_context_policy_does_not_trust_cvat_hidden_metadata() -> None:
    frames = _frame_rows().copy()
    frames["global_context_pig_count"] = 1
    frames.loc[0, "hidden_review_status"] = "unclear"
    out = apply_context_policy(frames, recompute_context=True)
    cvat = out["source_type"].eq("cvat_tracking_xml")
    legacy = out["source_type"].eq("legacy_recovered")
    assert not out.loc[cvat, "hidden_is_trusted"].any()
    assert out.loc[legacy & (out.index != 0), "hidden_is_trusted"].all()
    assert out.loc[cvat, "hidden_trust_status"].eq("untrusted_tracking_derived").all()
    assert not bool(out.loc[0, "hidden_is_trusted"])
    assert out.loc[0, "hidden_trust_status"] == "unclear_current_review"


def _legacy_window_frames() -> pd.DataFrame:
    rows = []
    for frame_index in range(16):
        rows.append(
            {
                "source_type": "legacy_recovered",
                "dataset_id": "legacy",
                "video_key": "legacy_video",
                "frame_uid": f"legacy_frame_{frame_index}",
                "frame_index": frame_index,
                "relative_frame_index": frame_index,
                "pig_id": "ID_1",
                "track_id": "track_1",
                "behavior": "stand",
                "hidden": "Yes",
                "hidden_is_trusted": True,
                "bbox_valid": True,
                "spatiotemporal_feature_valid": True,
                "timestamp_sec": frame_index / 30.0,
                "sequence_frame_count": 16,
                "legacy_expected_sequence_length": 16,
            }
        )
    return pd.DataFrame(rows)


def test_high_hidden_ratio_is_audited_but_not_excluded_by_default() -> None:
    frames = _legacy_window_frames()
    _, _, default_windows = build_sequence_windows(
        frames,
        window_lengths=[16],
        legacy_window_stride=1,
    )
    assert len(default_windows) == 1
    row = default_windows.iloc[0]
    assert bool(row["high_hidden_ratio_window"])
    assert bool(row["window_valid_for_main_train"])
    assert not bool(row["hidden_exclusion_policy_enabled"])

    _, _, strict_windows = build_sequence_windows(
        frames,
        window_lengths=[16],
        legacy_window_stride=1,
        exclude_high_hidden_from_main=True,
    )
    strict = strict_windows.iloc[0]
    assert not bool(strict["window_valid_for_main_train"])
    assert "hidden_ratio_above_threshold" in strict["window_exclusion_reason"]
