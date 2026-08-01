from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pig_behavior.classification_v2.review.post_review_learning import (
    ADJUSTED_ROI_SUFFIX,
    CORRECTED_SOURCE_SCHEMA_VERSION,
    ControlSelectionConfig,
    PostReviewContractError,
    analyze_post_review_learning,
    assert_not_active_behavior_ledger_path,
    build_corrected_source_authority,
    build_final_review_integration_preflight,
    build_post_review_control_scope,
    build_review_close_authority,
    validate_review_close_authority,
)


def _population(rows: int = 180) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_unit_id": [f"unit-{index:04d}" for index in range(rows)],
            "temporal_unit_key": [
                f"temporal-{index:04d}" for index in range(rows)
            ],
            "behavior_label": [
                ("drink", "explore", "fight", "stand")[index % 4]
                for index in range(rows)
            ],
            "source_type": [
                "cvat_tracking_xml" if index % 2 else "legacy_recovered"
                for index in range(rows)
            ],
            "video_key": [
                f"Pigs{20200101 + index % 3}_example" for index in range(rows)
            ],
            "review_unit_type": ["native_temporal_unit"] * rows,
            "source_marker": [f"unchanged-{index}" for index in range(rows)],
        }
    )


def _scope(prefix: str, count: int) -> pd.DataFrame:
    behaviors = ("stand", "drink", "move", "fight")
    return pd.DataFrame(
        {
            "review_unit_id": [f"{prefix}-{index}" for index in range(count)],
            "temporal_unit_key": [f"time-{prefix}-{index}" for index in range(count)],
            "source_type": ["cvat_tracking_xml"] * count,
            "video_key": [f"video-{index % 2}" for index in range(count)],
            "behavior_label": [behaviors[index % 4] for index in range(count)],
        }
    )


def _decisions(scope: pd.DataFrame) -> pd.DataFrame:
    decision_cycle = ("accept", "corrected", "exclude", "accept")
    choices = [decision_cycle[index % 4] for index in range(len(scope))]
    return pd.DataFrame(
        {
            "review_unit_id": scope["review_unit_id"],
            "manual_review_decision": choices,
        }
    )


def _quality(scope: pd.DataFrame, *, control: bool = False) -> pd.DataFrame:
    originals = scope["behavior_label"].tolist()
    reviewed = [
        "explore" if index % 4 == 1 else behavior
        for index, behavior in enumerate(originals)
    ]
    statuses_cycle = (
        "SUPPORTED",
        "SOURCE_LABEL_ERROR_CONFIRMED",
        "TECHNICAL_DEFECT",
        "SUPPORTED",
    )
    source_errors_cycle = ("NO", "YES", "NOT_APPLICABLE", "NO")
    patterns_cycle = (
        "NONE",
        "ROI_PROXIMITY_ONLY_FALSE_POSITIVE",
        "TECHNICAL_MEDIA_OR_PRESENTATION_DEFECT",
        "NONE",
    )
    statuses = [statuses_cycle[index % 4] for index in range(len(scope))]
    source_errors = [
        source_errors_cycle[index % 4] for index in range(len(scope))
    ]
    patterns = [patterns_cycle[index % 4] for index in range(len(scope))]
    return pd.DataFrame(
        {
            "review_unit_id": scope["review_unit_id"],
            "original_behavior": originals,
            "reviewed_behavior": reviewed,
            "label_status": statuses,
            "source_label_error_confirmed": source_errors,
            "error_pattern": patterns,
            "selection_assessment": [
                "CONTROL" if control else "PRIMARY"
            ]
            * len(scope),
        }
    )


def _bindings(names: set[str]) -> dict[str, dict[str, str]]:
    return {
        name: {
            "path": (
                ADJUSTED_ROI_SUFFIX
                if name == "adjusted_roi"
                else f"outputs/frozen/{name}.json"
            ),
            "sha256": f"{index + 1:064x}",
        }
        for index, name in enumerate(sorted(names))
    }


def _review_bindings() -> dict[str, dict[str, str]]:
    return _bindings(
        {
            "primary_scope",
            "primary_decisions",
            "primary_quality",
            "control_scope",
            "control_decisions",
            "control_quality",
        }
    )


def _close_authority() -> dict[str, object]:
    primary = _scope("primary", 4)
    control = _scope("control", 120)
    return build_review_close_authority(
        primary_scope=primary,
        primary_decisions=_decisions(primary),
        primary_quality=_quality(primary),
        control_scope=control,
        control_decisions=_decisions(control),
        control_quality=_quality(control, control=True),
        artifact_bindings=_review_bindings(),
        expected_primary_count=4,
        minimum_control_count=120,
    )


def test_control_scope_is_deterministic_disjoint_and_preserves_source_rows() -> None:
    population = _population()
    primary = population.iloc[:30].copy()
    config = ControlSelectionConfig(target_count=120, seed=47)

    first, first_audit = build_post_review_control_scope(
        population,
        primary,
        config=config,
    )
    second, second_audit = build_post_review_control_scope(
        population,
        primary,
        config=config,
    )

    assert first["review_unit_id"].tolist() == second["review_unit_id"].tolist()
    assert first_audit["selected_control_key_hash"] == second_audit[
        "selected_control_key_hash"
    ]
    assert len(first) == 120
    assert set(first["review_unit_id"]).isdisjoint(primary["review_unit_id"])
    expected = population.set_index("review_unit_id").loc[
        first["review_unit_id"], "source_marker"
    ]
    assert first["source_marker"].tolist() == expected.tolist()
    assert first_audit["sampling_outcomes_used"] is False
    assert (first["post_review_control_sampling_weight"] >= 1.0).all()
    added = set(first.columns) - set(population.columns)
    assert all(column.startswith("post_review_control_") for column in added)


def test_control_scope_rejects_review_outcomes_and_fewer_than_120() -> None:
    population = _population()
    contaminated = population.assign(manual_review_decision="accept")
    with pytest.raises(PostReviewContractError, match="review_outcomes"):
        build_post_review_control_scope(
            contaminated,
            population.iloc[:10],
        )
    with pytest.raises(PostReviewContractError, match="below_minimum"):
        build_post_review_control_scope(
            population,
            population.iloc[:10],
            config=ControlSelectionConfig(target_count=119),
        )


def test_close_authority_requires_complete_resolved_review() -> None:
    authority = _close_authority()
    validate_review_close_authority(authority)

    primary = _scope("primary", 4)
    control = _scope("control", 120)
    incomplete = _decisions(primary).iloc[:-1].copy()
    with pytest.raises(PostReviewContractError, match="coverage_mismatch"):
        build_review_close_authority(
            primary_scope=primary,
            primary_decisions=incomplete,
            primary_quality=_quality(primary),
            control_scope=control,
            control_decisions=_decisions(control),
            control_quality=_quality(control, control=True),
            artifact_bindings=_review_bindings(),
            expected_primary_count=4,
            minimum_control_count=120,
        )


def test_learning_separates_changes_controls_and_technical_exclusions() -> None:
    primary = _scope("primary", 4)
    control = _scope("control", 120).assign(
        post_review_control_sampling_weight=[
            2.0 if index % 2 == 0 else 3.0 for index in range(120)
        ]
    )
    all_keys = [
        *primary["temporal_unit_key"],
        *control["temporal_unit_key"],
    ]
    features = pd.DataFrame(
        {
            "temporal_unit_key": all_keys,
            "roi_distance": [index / 10.0 for index in range(len(all_keys))],
            "motion_speed": [
                None if index % 4 == 2 else index / 20.0
                for index in range(len(all_keys))
            ],
        }
    )
    result = analyze_post_review_learning(
        review_close_authority=_close_authority(),
        primary_scope=primary,
        primary_quality=_quality(primary),
        control_scope=control,
        control_quality=_quality(control, control=True),
        frame_features=features,
        feature_columns=["roi_distance", "motion_speed"],
    )

    summary = result["summary"]
    assert summary["changed_labels"] == 31
    assert summary["technical_exclusions"] == 31
    assert summary["review_fields_entering_model_x"] == 0
    assert summary["selector_diagnostics"]["CONTROL"][
        "source_label_errors"
    ] == 30
    assert summary["selector_diagnostics"][
        "estimated_selector_recall_within_explicit_population"
    ] is not None
    assert summary["selector_diagnostics"]["CONTROL"][
        "weighted_rate_wilson_95"
    ] is not None
    assert set(result["feature_contrasts"]["interpretation"]) == {
        "HYPOTHESIS_ONLY"
    }
    assert {"ALL", "original_behavior", "source_type"}.issubset(
        set(result["feature_contrasts"]["stratum_type"])
    )
    transition = result["transition_matrix"]
    assert not transition["original_behavior"].eq("move").any()


def test_learning_rejects_review_fields_as_features() -> None:
    primary = _scope("primary", 4)
    control = _scope("control", 120).assign(
        post_review_control_sampling_weight=1.0
    )
    features = pd.DataFrame(
        {
            "temporal_unit_key": [
                *primary["temporal_unit_key"],
                *control["temporal_unit_key"],
            ],
            "source_label_error_confirmed": [0.0] * 124,
        }
    )
    with pytest.raises(PostReviewContractError, match="forbidden_from_model_x"):
        analyze_post_review_learning(
            review_close_authority=_close_authority(),
            primary_scope=primary,
            primary_quality=_quality(primary),
            control_scope=control,
            control_quality=_quality(control, control=True),
            frame_features=features,
            feature_columns=["source_label_error_confirmed"],
        )


def test_active_behavior_ledger_path_is_rejected() -> None:
    active = Path(
        "human_review_workspace/classification_v2/run-1/"
        "human_decisions/behavior/decisions.csv"
    )
    with pytest.raises(PostReviewContractError, match="forbidden"):
        assert_not_active_behavior_ledger_path(active)


def test_identity_behavior_conflict_blocks_but_bbox_only_does_not() -> None:
    close = _close_authority()
    required = {
        *close["artifacts"],
        "adjusted_roi",
        "corrected_source_authority",
        "rebuilt_frame_features",
    }
    bindings = _bindings(required)
    bindings.update(close["artifacts"])
    manifest_hash = "a" * 64
    behavior_manifest = {
        "manifest_sha256": manifest_hash,
        "status": "APPLIED",
        "targets": [
            {
                "path": "data/source.xml",
                "before_sha256": "c" * 64,
                "after_sha256": "b" * 64,
                "bbox_updates": 2,
                "identity_updates": 1,
                "behavior_updates": 1,
                "hidden_updates": 0,
            }
        ],
    }
    corrected_source = {
        "schema_version": CORRECTED_SOURCE_SCHEMA_VERSION,
        "status": "FROZEN",
        "target_after_hashes": {"data/source.xml": "b" * 64}
    }
    blocked = build_final_review_integration_preflight(
        review_close_authority=close,
        artifact_bindings=bindings,
        identity_apply_manifests=[behavior_manifest],
        corrected_source_authority=corrected_source,
    )
    assert blocked["status"] == "BLOCKED"
    assert any("behavior_conflict_unresolved" in item for item in blocked["blockers"])

    ready = build_final_review_integration_preflight(
        review_close_authority=close,
        artifact_bindings=bindings,
        identity_apply_manifests=[behavior_manifest],
        corrected_source_authority=corrected_source,
        conflict_resolutions=[
            {
                "manifest_sha256": manifest_hash,
                "field": "behavior",
                "resolution": "BEHAVIOR_REVIEW_WINS",
            }
        ],
    )
    assert ready["status"] == "READY_FOR_REVIEWED_WINDOW_REBUILD"
    assert ready["window_lengths"] == [6, 8, 12, 16]
    assert ready["window_structure_reuse_allowed"] is False

    bbox_manifest = {
        **behavior_manifest,
        "targets": [
            {
                **behavior_manifest["targets"][0],
                "behavior_updates": 0,
            }
        ],
    }
    bbox_ready = build_final_review_integration_preflight(
        review_close_authority=close,
        artifact_bindings=bindings,
        identity_apply_manifests=[bbox_manifest],
        corrected_source_authority=corrected_source,
    )
    assert bbox_ready["status"] == "READY_FOR_REVIEWED_WINDOW_REBUILD"


def test_corrected_source_authority_requires_unbroken_sequential_chain() -> None:
    path = "data/source.xml"
    first = {
        "manifest_sha256": "1" * 64,
        "status": "APPLIED",
        "targets": [
            {
                "path": path,
                "before_sha256": "a" * 64,
                "after_sha256": "b" * 64,
            }
        ],
    }
    second = {
        "manifest_sha256": "2" * 64,
        "status": "APPLIED",
        "targets": [
            {
                "path": path,
                "before_sha256": "b" * 64,
                "after_sha256": "c" * 64,
            }
        ],
    }
    authority = build_corrected_source_authority(
        identity_apply_manifests=[first, second],
        observed_target_hashes={path: "c" * 64},
    )
    assert authority["status"] == "FROZEN"
    assert authority["target_after_hashes"][path] == "c" * 64

    broken = {
        **second,
        "targets": [
            {
                **second["targets"][0],
                "before_sha256": "d" * 64,
            }
        ],
    }
    with pytest.raises(PostReviewContractError, match="chain_break"):
        build_corrected_source_authority(
            identity_apply_manifests=[first, broken],
            observed_target_hashes={path: "c" * 64},
        )
