from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.review.post_review_learning import (
    build_review_close_authority,
)
from pig_behavior.classification_v2.review.post_review_selector_candidate import (
    PostReviewSelectorContractError,
    SelectorCandidateConfig,
    SelectorFeatureSpec,
    aggregate_masked_selector_features,
    build_selector_outcomes,
    canonical_selector_feature_specs,
    run_post_review_selector_candidate,
    selector_feature_contract,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS


def _scope(prefix: str, count: int) -> pd.DataFrame:
    behaviors = list(VALID_BEHAVIORS)
    rows = []
    for index in range(count):
        date_index = index // len(behaviors)
        rows.append(
            {
                "review_unit_id": f"{prefix}-unit-{index:04d}",
                "temporal_unit_key": f"{prefix}-time-{index:04d}",
                "source_type": (
                    "cvat_tracking_xml" if index % 2 else "legacy_recovered"
                ),
                "video_key": f"Pigs202001{date_index + 1:02d}_video_{index % 2}",
                "recording_date": f"2020-01-{date_index + 1:02d}",
                "behavior_label": behaviors[index % len(behaviors)],
            }
        )
    return pd.DataFrame(rows)


def _quality(scope: pd.DataFrame) -> pd.DataFrame:
    behaviors = list(VALID_BEHAVIORS)
    rows = []
    for index, row in scope.reset_index(drop=True).iterrows():
        original = str(row["behavior_label"])
        changed = _is_changed(index)
        reviewed = (
            behaviors[(behaviors.index(original) + 1) % len(behaviors)]
            if changed
            else original
        )
        rows.append(
            {
                "review_unit_id": row["review_unit_id"],
                "original_behavior": original,
                "reviewed_behavior": reviewed,
                "label_status": (
                    "SOURCE_LABEL_ERROR_CONFIRMED" if changed else "SUPPORTED"
                ),
                "source_label_error_confirmed": "YES" if changed else "NO",
                "error_pattern": (
                    "OTHER_CLEAR_SOURCE_LABEL_ERROR" if changed else "NONE"
                ),
                "selection_assessment": "PRIMARY",
            }
        )
    return pd.DataFrame(rows)


def _decisions(scope: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_unit_id": scope["review_unit_id"],
            "manual_review_decision": [
                "corrected" if _is_changed(index) else "accept"
                for index in range(len(scope))
            ],
        }
    )


def _is_changed(index: int) -> bool:
    behavior_count = len(VALID_BEHAVIORS)
    return (index // behavior_count + index % behavior_count) % 5 == 0


def _bindings() -> dict[str, dict[str, str]]:
    names = {
        "primary_scope",
        "primary_decisions",
        "primary_quality",
        "control_scope",
        "control_decisions",
        "control_quality",
    }
    return {
        name: {"path": f"outputs/frozen/{name}.csv", "sha256": f"{i:064x}"}
        for i, name in enumerate(sorted(names), start=1)
    }


def _review_fixture() -> tuple[
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    primary = _scope("primary", 40)
    control = _scope("control", 120)
    control["post_review_control_sampling_weight"] = 2.0 + (
        np.arange(len(control)) % 3
    )
    primary_quality = _quality(primary)
    control_quality = _quality(control)
    authority = build_review_close_authority(
        primary_scope=primary,
        primary_decisions=_decisions(primary),
        primary_quality=primary_quality,
        control_scope=control,
        control_decisions=_decisions(control),
        control_quality=control_quality,
        artifact_bindings=_bindings(),
        expected_primary_count=40,
        minimum_control_count=120,
    )
    return authority, primary, primary_quality, control, control_quality


def test_canonical_selector_contract_covers_exact_spatial_46d() -> None:
    specs = canonical_selector_feature_specs()
    assert len(specs) == 46
    assert len({spec.feature_name for spec in specs}) == 46
    assert all(spec.validity_columns for spec in specs)
    contract = selector_feature_contract()
    assert len(contract["ordered_features"]) == 46
    assert len(contract["contract_hash"]) == 64
    assert contract["score_context_only"] == ["original_behavior"]


def test_masked_aggregation_distinguishes_valid_zero_from_placeholder() -> None:
    frame = pd.DataFrame(
        {
            "temporal_unit_key": ["valid", "valid", "invalid", "invalid"],
            "frame_index": [0, 1, 0, 1],
            "speed": [0.0, 0.0, 0.0, 0.0],
            "speed_valid": [True, True, False, False],
        }
    )
    aggregate, audit = aggregate_masked_selector_features(
        frame,
        temporal_unit_keys=["valid", "invalid"],
        specs=(SelectorFeatureSpec("speed", ("speed_valid",)),),
    )
    assert aggregate.loc["valid", "speed::masked_mean"] == 0.0
    assert aggregate.loc["valid", "speed::valid_coverage"] == 1.0
    assert np.isnan(aggregate.loc["invalid", "speed::masked_mean"])
    assert aggregate.loc["invalid", "speed::valid_coverage"] == 0.0
    feature_audit = audit["features"][0]
    assert feature_audit["valid_zero_observations"] == 2
    assert feature_audit["invalid_zero_placeholders"] == 2


def test_masked_aggregation_fails_when_required_mask_is_absent() -> None:
    frame = pd.DataFrame(
        {
            "temporal_unit_key": ["unit"],
            "frame_index": [0],
            "speed": [0.0],
        }
    )
    with pytest.raises(PostReviewSelectorContractError, match="missing_columns"):
        aggregate_masked_selector_features(
            frame,
            temporal_unit_keys=["unit"],
            specs=(SelectorFeatureSpec("speed", ("speed_valid",)),),
        )


def test_grouped_selector_is_deterministic_and_has_no_date_video_leakage() -> None:
    authority, primary, primary_quality, control, control_quality = _review_fixture()
    outcomes, audit = build_selector_outcomes(
        review_close_authority=authority,
        primary_scope=primary,
        primary_quality=primary_quality,
        control_scope=control,
        control_quality=control_quality,
    )
    behaviors = list(VALID_BEHAVIORS)
    aggregate = pd.DataFrame(
        0.0,
        index=outcomes["temporal_unit_key"],
        columns=[f"evidence_{index}" for index in range(len(behaviors))],
    )
    aggregate.index.name = "temporal_unit_key"
    for row in outcomes.itertuples():
        class_index = behaviors.index(row.reviewed_behavior)
        aggregate.loc[row.temporal_unit_key, f"evidence_{class_index}"] = 1.0
    aggregate["sometimes_missing"] = np.nan
    aggregate.loc[aggregate.index[::2], "sometimes_missing"] = 1.0
    config = SelectorCandidateConfig(seed=13, fold_count=4, max_iter=1000)
    first = run_post_review_selector_candidate(
        outcomes=outcomes,
        aggregates=aggregate,
        config=config,
    )
    second = run_post_review_selector_candidate(
        outcomes=outcomes,
        aggregates=aggregate,
        config=config,
    )
    columns = ["review_unit_id", "fold_id", "selector_suspicion_score"]
    pd.testing.assert_frame_equal(
        first["predictions"][columns],
        second["predictions"][columns],
    )
    assert audit["control_validation_authority_for_candidate"] is False
    assert first["leakage_audit"]["date_group_overlap"] == 0
    assert first["leakage_audit"]["video_group_overlap"] == 0
    assert first["leakage_audit"]["source_label_entering_model_x"] == 0
    assert first["metrics"]["fresh_holdout_required"] is True
    assert len(first["predictions"]) == len(outcomes)
