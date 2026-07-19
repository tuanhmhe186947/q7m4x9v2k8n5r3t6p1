from __future__ import annotations

import hashlib
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pig_behavior.classification_v2.contracts.identifiers import (
    ensure_frame_object_identifiers,
)
from pig_behavior.classification_v2.features.context_policy import (
    apply_context_policy,
)
from pig_behavior.classification_v2.features.sequence_windows import (
    audit_sequence_windows,
    build_sequence_windows,
)
from pig_behavior.classification_v2.review.hidden_review_builder import (
    DECISION_COLUMNS,
    HiddenReviewConfig,
    _hidden_confusion_audit,
    _merge_hidden_review_authorities,
    apply_hidden_review_decisions,
    audit_hidden_decision_coverage,
    audit_hidden_review_manifest,
    build_hidden_review_frame_context,
    build_hidden_review_manifest,
    hidden_decision_semantic_error,
)
from pig_behavior.classification_v2.review.hidden_review_identifiers import (
    HIDDEN_REVIEW_KEY_VERSION,
    attach_hidden_review_identifiers,
)
from pig_behavior.classification_v2.review.hidden_review_migration import (
    carry_forward_hidden_review_decisions,
    migrate_hidden_review_decisions,
    upgrade_hidden_review_manifest_identifiers,
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


def test_hidden_selection_is_invariant_to_behavior_targets() -> None:
    """Behavior relabeling must not alter visibility review selection."""

    frames = _frame_rows()
    config = HiddenReviewConfig(
        random_no_per_stratum=1,
        clean_control_per_stratum=1,
    )
    first, _, first_audit = build_hidden_review_manifest(
        frames,
        config=config,
    )
    relabeled = frames.copy()
    relabeled["behavior"] = [
        "drink" if index % 2 else "social-nose"
        for index in range(len(relabeled))
    ]
    second, _, second_audit = build_hidden_review_manifest(
        relabeled,
        config=config,
    )
    signature_columns = [
        "hidden_review_item_id",
        "hidden_review_cohort",
        "hidden_false_negative_risk_score",
        "hidden_false_negative_risk_reasons",
        "hidden_false_negative_risk_band",
        "hidden_sampling_stratum",
        "hidden_sampling_probability",
        "hidden_sampling_weight",
        "hidden_review_priority",
    ]

    pd.testing.assert_frame_equal(
        first[signature_columns].reset_index(drop=True),
        second[signature_columns].reset_index(drop=True),
    )
    assert first_audit["selection_contract"]["target_independent"] is True
    assert second_audit["selection_contract"]["target_independent"] is True
    assert not first["hidden_sampling_stratum"].str.contains(
        "behavior=",
        regex=False,
    ).any()
    assert not first["hidden_false_negative_risk_reasons"].str.contains(
        "interaction_scene",
        regex=False,
    ).any()


def test_hidden_sampling_config_rejects_target_derived_strata() -> None:
    config = HiddenReviewConfig(
        stratum_columns=("source_type", "behavior"),
    )

    with pytest.raises(ValueError, match="target-independent"):
        config.validate()


def test_hidden_manifest_audit_rejects_target_derived_risk_reason() -> None:
    frames, manifest = _build_review()
    corrupted = manifest.copy()
    corrupted.loc[corrupted.index[0], "hidden_false_negative_risk_reasons"] = (
        "interaction_scene"
    )

    audit = audit_hidden_review_manifest(
        frames,
        corrupted,
        HiddenReviewConfig(
            random_no_per_stratum=1,
            clean_control_per_stratum=1,
        ),
    )

    assert audit["selection_contract"]["target_independent"] is False
    assert any(
        "target_derived_hidden_risk_reason" in error
        for error in audit["errors"]
    )


def test_hidden_review_identity_ignores_labels_and_frame_uid_schema() -> None:
    """Review identity must survive label correction and identifier migration."""

    row = _frame_rows().iloc[[0]].copy()
    row["object_track_key"] = "legacy|track=0|pig=ID_1"
    original = attach_hidden_review_identifiers(row)
    changed = row.copy()
    changed["frame_uid"] = "object-level-frame-id"
    changed["behavior"] = "fight"
    changed["hidden"] = "No"
    migrated = attach_hidden_review_identifiers(changed)

    assert original.iloc[0]["hidden_review_item_id"] == migrated.iloc[0][
        "hidden_review_item_id"
    ]
    assert original.iloc[0]["hidden_review_subject_key"] == migrated.iloc[0][
        "hidden_review_subject_key"
    ]
    assert migrated.iloc[0]["hidden_review_key_version"] == (
        HIDDEN_REVIEW_KEY_VERSION
    )


def test_hidden_review_identifier_migration_preserves_human_payload() -> None:
    """A partial decision file maps one-to-one without fabricating decisions."""

    _, generated = _build_review()
    legacy = generated.copy()
    legacy["hidden_review_item_id"] = [
        f"legacy_hidden_item_{index}" for index in range(len(legacy))
    ]
    decisions = _resolved_decisions(legacy).head(2).copy()
    upgraded = upgrade_hidden_review_manifest_identifiers(legacy)

    mapping, migrated, audit = migrate_hidden_review_decisions(
        legacy,
        upgraded,
        decisions,
    )

    assert audit["errors"] == []
    assert audit["valid"] is True
    assert audit["mapping_rows"] == len(legacy)
    assert audit["mapped_decision_rows"] == 2
    assert audit["human_payload_changed_rows"] == 0
    assert len(migrated) == len(decisions)
    assert int(mapping["has_human_decision"].sum()) == 2
    assert migrated["legacy_hidden_review_item_id"].tolist() == decisions[
        "hidden_review_item_id"
    ].tolist()
    for column in (
        "hidden_after_review",
        "hidden_review_status",
        "hidden_review_reason",
        "hidden_reviewer",
    ):
        assert migrated[column].tolist() == decisions[column].tolist()


def test_hidden_review_redesign_carries_human_payload_by_stable_item() -> None:
    _, previous = _build_review()
    decisions = _resolved_decisions(previous).head(2).copy()
    current = previous.copy()
    current["behavior"] = "drink"
    decided_ids = set(decisions["hidden_review_item_id"])
    current.loc[
        current["hidden_review_item_id"].isin(decided_ids),
        "hidden_review_cohort",
    ] = "hidden_yes_confirmation"

    carried, audit = carry_forward_hidden_review_decisions(
        previous,
        current,
        decisions,
    )

    assert audit["valid"] is True
    assert audit["carried_decision_rows"] == 2
    assert audit["human_payload_changed_rows"] == 0
    assert audit["optional_invariant_presence"]["object_track_key"] == {
        "previous": False,
        "current": False,
        "compared": False,
    }
    assert len(carried) == len(decisions)
    for column in (
        "hidden_before_review",
        "hidden_after_review",
        "hidden_review_status",
        "hidden_review_confidence",
        "hidden_review_reason",
        "hidden_reviewer",
        "hidden_reviewed_at",
    ):
        assert carried[column].tolist() == decisions[column].tolist()


def test_hidden_review_redesign_checks_optional_context_when_available() -> None:
    _, previous = _build_review()
    previous["object_track_key"] = [
        f"object-track-{index}" for index in range(len(previous))
    ]
    decisions = _resolved_decisions(previous).head(2).copy()
    current = previous.copy()
    decision_id = decisions.iloc[0]["hidden_review_item_id"]
    current.loc[
        current["hidden_review_item_id"].eq(decision_id),
        "object_track_key",
    ] = "changed-object-track"

    _, audit = carry_forward_hidden_review_decisions(
        previous,
        current,
        decisions,
    )

    assert audit["valid"] is False
    assert audit["context_changed_counts"]["object_track_key"] == 1
    assert "object_track_key" in audit["compared_invariant_columns"]


def test_hidden_review_redesign_cli_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, previous = _build_review()
    current = previous.copy()
    decisions = _resolved_decisions(previous).head(2).copy()
    previous_path = tmp_path / "previous.csv"
    current_path = tmp_path / "current.csv"
    decisions_path = tmp_path / "decisions.csv"
    output_path = tmp_path / "carried.csv"
    audit_path = tmp_path / "audit.json"
    previous.to_csv(previous_path, index=False)
    current.to_csv(current_path, index=False)
    decisions.to_csv(decisions_path, index=False)
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "classification_v2"
        / "01_review_units_gui"
        / "classification_v2_carry_forward_hidden_review_decisions.py"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script_path),
            "--previous-manifest-csv",
            str(previous_path),
            "--current-manifest-csv",
            str(current_path),
            "--decisions-csv",
            str(decisions_path),
            "--output-decisions-csv",
            str(output_path),
            "--audit-json",
            str(audit_path),
        ],
    )

    runpy.run_path(str(script_path), run_name="__main__")

    carried = pd.read_csv(output_path, low_memory=False)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["valid"] is True
    assert audit["carried_decision_rows"] == 2
    assert audit["output_written"] is True
    assert audit["output_decisions_sha256"] == hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()
    assert len(carried) == 2
    assert carried["hidden_after_review"].tolist() == decisions[
        "hidden_after_review"
    ].tolist()


def test_hidden_review_identifier_migration_cli_hashes_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, legacy = _build_review()
    legacy["hidden_review_item_id"] = [
        f"legacy-item-{index}" for index in range(len(legacy))
    ]
    decisions = _resolved_decisions(legacy).head(2).copy()
    legacy_path = tmp_path / "legacy.csv"
    decisions_path = tmp_path / "legacy_decisions.csv"
    manifest_output = tmp_path / "manifest_v2.csv"
    decisions_output = tmp_path / "decisions_v2.csv"
    mapping_output = tmp_path / "mapping.csv"
    audit_output = tmp_path / "migration_audit.json"
    legacy.to_csv(legacy_path, index=False)
    decisions.to_csv(decisions_path, index=False)
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "classification_v2"
        / "01_review_units_gui"
        / "classification_v2_migrate_hidden_review_identifiers.py"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script_path),
            "--legacy-manifest-csv",
            str(legacy_path),
            "--legacy-decisions-csv",
            str(decisions_path),
            "--output-manifest-csv",
            str(manifest_output),
            "--output-decisions-csv",
            str(decisions_output),
            "--mapping-csv",
            str(mapping_output),
            "--audit-json",
            str(audit_output),
        ],
    )

    runpy.run_path(str(script_path), run_name="__main__")

    audit = json.loads(audit_output.read_text(encoding="utf-8"))
    assert audit["valid"] is True
    assert audit["outputs_written"] is True
    for field, path in (
        ("output_manifest_sha256", manifest_output),
        ("output_decisions_sha256", decisions_output),
        ("mapping_sha256", mapping_output),
    ):
        assert audit[field] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_hidden_review_redesign_rejects_lost_human_decision() -> None:
    _, previous = _build_review()
    decisions = _resolved_decisions(previous).head(2).copy()
    lost_id = decisions.iloc[0]["hidden_review_item_id"]
    current = previous.loc[
        ~previous["hidden_review_item_id"].eq(lost_id)
    ].copy()

    _, audit = carry_forward_hidden_review_decisions(
        previous,
        current,
        decisions,
    )

    assert audit["valid"] is False
    assert audit["missing_current_decision_items"] == 1
    assert any(
        "human_decision_items_missing_from_current_manifest" in error
        for error in audit["errors"]
    )


def test_hidden_review_identifier_migration_rejects_unknown_decision() -> None:
    """Unknown legacy IDs remain visible as errors and are never dropped."""

    _, generated = _build_review()
    legacy = generated.copy()
    legacy["hidden_review_item_id"] = [
        f"legacy_hidden_item_{index}" for index in range(len(legacy))
    ]
    decisions = _resolved_decisions(legacy).head(2).copy()
    decisions.loc[0, "hidden_review_item_id"] = "unknown_hidden_item"
    upgraded = upgrade_hidden_review_manifest_identifiers(legacy)

    _, migrated, audit = migrate_hidden_review_decisions(
        legacy,
        upgraded,
        decisions,
    )

    assert audit["valid"] is False
    assert "decision_ids_missing_from_legacy=1" in audit["errors"]
    assert "unmapped_decision_rows=1" in audit["errors"]
    assert len(migrated) == len(decisions)


def test_hidden_frame_context_keeps_all_objects_in_v2_scene() -> None:
    """Selecting one actor must retain partner rows from the same scene."""

    rows = _frame_rows().iloc[:2].copy()
    rows["source_type"] = "cvat_tracking_xml"
    rows["dataset_id"] = "cvat"
    rows["video_key"] = "scene_video"
    rows["frame_uid"] = "legacy_scene_0"
    rows["frame_index"] = 0
    rows["object_track_key"] = ["scene|track=0", "scene|track=1"]
    rows = ensure_frame_object_identifiers(
        rows,
        source_name="hidden_frame_context_test",
    )
    selected_actor = rows.iloc[[0]].copy()

    context = build_hidden_review_frame_context(rows, selected_actor)

    assert len(context) == 2
    assert context["frame_uid"].nunique() == 2
    assert context["scene_frame_uid"].nunique() == 1


def test_hidden_review_census_targets_untrusted_yes_and_caps_risk() -> None:
    frames = _frame_rows()
    frames.loc[frames["source_type"].eq("legacy_recovered"), "hidden"] = "Yes"
    manifest, _, audit = build_hidden_review_manifest(
        frames,
        config=HiddenReviewConfig(
            trusted_yes_per_stratum=1,
            random_no_per_stratum=0,
            clean_control_per_stratum=0,
            max_high_risk_per_stratum=1,
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


def test_gui_metadata_keeps_manifest_cohort_authority_during_apply() -> None:
    frames, manifest = _build_review()
    decisions = _resolved_decisions(manifest)
    manifest_by_item = manifest.set_index("hidden_review_item_id")
    copied_metadata = (
        "hidden_review_cohort",
        "source_type",
        "video_key",
        "hidden_false_negative_risk_score",
    )
    for column in copied_metadata:
        decisions[column] = decisions["hidden_review_item_id"].map(
            manifest_by_item[column]
        )

    random_item = manifest.loc[
        manifest["hidden_review_cohort"].eq("hidden_no_random_audit")
    ].iloc[0]
    high_risk_item = manifest.loc[
        manifest["hidden_review_cohort"].eq("hidden_no_high_risk")
    ].iloc[0]
    corrected_ids = {
        random_item["hidden_review_item_id"],
        high_risk_item["hidden_review_item_id"],
    }
    decisions.loc[
        decisions["hidden_review_item_id"].isin(corrected_ids),
        "hidden_after_review",
    ] = "Yes"

    joined, mismatch_counts = _merge_hidden_review_authorities(
        manifest,
        decisions,
    )
    assert mismatch_counts == {}
    assert "hidden_review_cohort" in joined.columns
    assert "hidden_review_cohort_decision" in joined.columns
    assert "hidden_review_cohort_manifest" not in joined.columns

    reviewed, audit, confusion = apply_hidden_review_decisions(
        frames,
        manifest,
        decisions,
    )
    direct_confusion = _hidden_confusion_audit(manifest, decisions)
    expected_cohorts = {
        str(key): int(value)
        for key, value in manifest["hidden_review_cohort"].value_counts().items()
    }
    assert len(reviewed) == len(frames)
    assert reviewed["hidden_review_item_id"].is_unique
    assert audit["coverage"]["decision_metadata_mismatch_counts"] == {}
    assert confusion == direct_confusion
    assert confusion["cohort_counts"] == expected_cohorts
    assert confusion["reviewed_cohort_counts"] == expected_cohorts
    assert confusion["random_hidden_no_false_negatives"] == 1
    assert confusion["high_risk_hidden_no_corrected_to_yes"] == 1
    assert confusion["errors"] == []


def test_gui_metadata_mismatch_fails_closed() -> None:
    _, manifest = _build_review()
    decisions = _resolved_decisions(manifest)
    manifest_by_item = manifest.set_index("hidden_review_item_id")
    decisions["hidden_review_cohort"] = decisions["hidden_review_item_id"].map(
        manifest_by_item["hidden_review_cohort"]
    )
    decisions.loc[0, "hidden_review_cohort"] = "mismatched_gui_copy"

    audit = audit_hidden_decision_coverage(manifest, decisions)

    assert audit["decision_metadata_mismatch_counts"] == {
        "hidden_review_cohort": 1
    }
    assert "decision_metadata_mismatch:hidden_review_cohort=1" in audit["errors"]
    with pytest.raises(ValueError, match="Hidden decision metadata mismatch"):
        _hidden_confusion_audit(manifest, decisions)


def test_low_confidence_resolved_hidden_decision_fails_closed() -> None:
    frames, manifest = _build_review()
    decisions = _resolved_decisions(manifest)
    decisions.loc[0, "hidden_review_confidence"] = "low"

    audit = audit_hidden_decision_coverage(manifest, decisions)

    assert audit["decision_confidence_counts"]["low"] == 1
    assert audit["semantic_error_counts"] == {
        "low_confidence_requires_unclear": 1
    }
    with pytest.raises(ValueError, match="coverage failed"):
        apply_hidden_review_decisions(frames, manifest, decisions)


def test_medium_confidence_does_not_change_weight_or_mask() -> None:
    frames, manifest = _build_review()
    frames["sample_weight"] = 1.0
    decisions = _resolved_decisions(manifest)
    item_id = decisions.loc[0, "hidden_review_item_id"]
    decisions.loc[0, "hidden_review_confidence"] = "medium"

    reviewed, audit, _ = apply_hidden_review_decisions(
        frames,
        manifest,
        decisions,
    )
    row = reviewed.loc[reviewed["hidden_review_item_id"].eq(item_id)].iloc[0]

    assert audit["coverage"]["decision_confidence_counts"]["medium"] == 1
    assert row["hidden_review_confidence"] == "medium"
    assert bool(row["hidden_review_available_mask"])
    assert bool(row["hidden_is_trusted"])
    assert reviewed["sample_weight"].eq(1.0).all()

def test_unclear_hidden_decision_is_fail_closed_by_default() -> None:
    frames, manifest = _build_review()
    decisions = _resolved_decisions(manifest)
    decisions.loc[0, "hidden_review_status"] = "unclear"
    decisions.loc[0, "hidden_after_review"] = ""
    decisions.loc[0, "hidden_review_confidence"] = "low"

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


def test_hidden_yes_with_clearly_visible_reason_is_rejected() -> None:
    frames, manifest = _build_review()
    decisions = _resolved_decisions(manifest)
    yes_item_id = manifest.loc[
        manifest["hidden_before_review"].eq("Yes"),
        "hidden_review_item_id",
    ].iloc[0]
    target = decisions["hidden_review_item_id"].eq(yes_item_id)
    decisions.loc[target, "hidden_after_review"] = "Yes"
    decisions.loc[target, "hidden_review_reason"] = "clearly_visible"

    audit = audit_hidden_decision_coverage(manifest, decisions)

    assert audit["semantic_error_items"] == 1
    assert audit["semantic_error_counts"] == {
        "hidden_yes_with_clearly_visible_reason": 1
    }
    with pytest.raises(ValueError, match="coverage failed"):
        apply_hidden_review_decisions(frames, manifest, decisions)


def test_visible_no_rejects_hidden_only_reason() -> None:
    error = hidden_decision_semantic_error(
        hidden_after="No",
        review_status="reviewed",
        reason="occluded_by_pig;note=fixture",
    )

    assert error == "visible_no_with_hidden_only_reason"


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


def test_rebuilt_legacy_hidden_seed_stays_untrusted_through_context() -> None:
    frames = _frame_rows().iloc[[0]].copy()
    frames["global_context_pig_count"] = 1
    frames["hidden_source"] = "cvat_native_anchor"
    frames["hidden_is_trusted"] = False
    frames["hidden_review_status"] = "seed_unreviewed"
    frames["hidden_trust_status"] = "untrusted_cvat_seed"
    frames["visibility_quality"] = "cvat_anchor_seed_unreviewed"

    out = apply_context_policy(frames, recompute_context=True)

    assert not bool(out.iloc[0]["hidden_is_trusted"])
    assert out.iloc[0]["hidden_source"] == "cvat_native_anchor"
    assert out.iloc[0]["hidden_review_status"] == "seed_unreviewed"
    assert out.iloc[0]["hidden_trust_status"] == "untrusted_cvat_seed"
    assert out.iloc[0]["visibility_quality"] == "cvat_anchor_seed_unreviewed"


def test_rebuilt_legacy_hidden_seed_stays_untrusted_in_review_manifest() -> None:
    frames = _frame_rows().iloc[[0]].copy()
    frames["hidden_source"] = "cvat_native_anchor"
    frames["hidden_is_trusted"] = False
    frames["hidden_review_status"] = "seed_unreviewed"
    frames["hidden_trust_status"] = "untrusted_cvat_seed"
    frames["visibility_quality"] = "cvat_anchor_seed_unreviewed"

    manifest, _, _ = build_hidden_review_manifest(
        frames,
        config=HiddenReviewConfig(
            trusted_yes_per_stratum=0,
            random_no_per_stratum=0,
            clean_control_per_stratum=0,
        ),
    )

    row = manifest.iloc[0]
    assert not bool(row["hidden_is_trusted_before_review"])
    assert row["hidden_source"] == "cvat_native_anchor"
    assert row["hidden_trust_status_before_review"] == "untrusted_cvat_seed"


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


def test_high_hidden_ratio_is_excluded_by_default_with_explicit_opt_out() -> None:
    frames = _legacy_window_frames()
    _, _, default_windows = build_sequence_windows(
        frames,
        window_lengths=[16],
        legacy_window_stride=1,
    )
    assert len(default_windows) == 1
    row = default_windows.iloc[0]
    assert bool(row["high_hidden_ratio_window"])
    assert not bool(row["window_valid_for_main_train"])
    assert bool(row["hidden_exclusion_policy_enabled"])
    assert row["hidden_window_policy_tier"] == "exclude"
    assert row["window_training_tier_recommendation"] == "exclude"

    _, _, audit_only_windows = build_sequence_windows(
        frames,
        window_lengths=[16],
        legacy_window_stride=1,
        exclude_high_hidden_from_main=False,
    )
    audit_only = audit_only_windows.iloc[0]
    assert bool(audit_only["window_valid_for_main_train"])
    assert audit_only["hidden_window_policy_tier"] == "audit_only"


def test_untrusted_hidden_is_diagnostic_but_not_training_exclusion() -> None:
    frames = _legacy_window_frames()
    frames["hidden_is_trusted"] = False

    _, _, windows = build_sequence_windows(
        frames,
        window_lengths=[16],
        legacy_window_stride=1,
    )

    row = windows.iloc[0]
    assert row["hidden_burden_ratio_window"] == pytest.approx(1.0)
    assert row["hidden_ratio_trusted_window"] == pytest.approx(0.0)
    assert row["hidden_trusted_longest_run_ratio_window"] == pytest.approx(0.0)
    assert row["hidden_window_policy_tier"] == "main_train"
    assert bool(row["window_valid_for_main_train"])


@pytest.mark.parametrize(
    ("length", "main_indices", "robust_indices", "exclude_indices"),
    [
        (6, [0], [0, 2], [0, 1, 2, 3]),
        (8, [0, 2], [0, 2, 4], [0, 1, 2, 3, 4]),
        (12, [0, 2, 4], [0, 2, 4, 6], [0, 1, 2, 3, 4, 5, 6]),
        (16, [0, 2, 4, 6], [0, 2, 4, 6, 8], list(range(9))),
    ],
)
def test_hidden_policy_tiers_cover_all_temporal_views(
    length: int,
    main_indices: list[int],
    robust_indices: list[int],
    exclude_indices: list[int],
) -> None:
    expected = [
        (main_indices, "main_train"),
        (robust_indices, "robust_train_only"),
        (exclude_indices, "exclude"),
    ]
    for hidden_indices, tier in expected:
        frames = _legacy_window_frames()
        frames["hidden"] = "No"
        frames.loc[frames["frame_index"].isin(hidden_indices), "hidden"] = "Yes"
        _, _, windows = build_sequence_windows(
            frames,
            window_lengths=[length],
            legacy_window_stride=1,
        )
        row = windows.loc[windows["window_start_frame"].eq(0)].iloc[0]
        assert row["hidden_window_policy_tier"] == tier
        assert row["window_training_tier_recommendation"] == tier
        assert bool(row["window_valid_for_main_train"]) is (tier == "main_train")
        expected_weight = 0.0 if tier == "exclude" else 1.0
        assert row["window_sample_weight"] == pytest.approx(expected_weight)


def test_long_contiguous_hidden_run_excludes_even_below_total_ratio_limit() -> None:
    frames = _legacy_window_frames()
    frames["hidden"] = "No"
    frames.loc[frames["frame_index"].lt(7), "hidden"] = "Yes"

    _, _, windows = build_sequence_windows(
        frames,
        window_lengths=[16],
        legacy_window_stride=1,
    )

    row = windows.iloc[0]
    assert row["hidden_burden_ratio_window"] == pytest.approx(7 / 16)
    assert row["hidden_longest_run_ratio_window"] == pytest.approx(7 / 16)
    assert row["hidden_window_policy_tier"] == "exclude"
    assert "hidden_run_above_robust_threshold" in row["window_exclusion_reason"]


def test_hidden_policy_rejects_inverted_main_and_robust_thresholds() -> None:
    with pytest.raises(
        ValueError,
        match="max_hidden_ratio_main must not exceed max_hidden_ratio_robust",
    ):
        build_sequence_windows(
            _legacy_window_frames(),
            window_lengths=[16],
            max_hidden_ratio_main=0.6,
            max_hidden_ratio_robust=0.5,
        )


def test_hidden_policy_audit_rejects_nonzero_excluded_weight() -> None:
    _, intervals, windows = build_sequence_windows(
        _legacy_window_frames(),
        window_lengths=[16],
    )
    windows.loc[:, "window_sample_weight"] = 1.0

    audit = audit_sequence_windows(windows, intervals)

    assert "excluded_windows_have_nonzero_weight=1" in audit["errors"]
