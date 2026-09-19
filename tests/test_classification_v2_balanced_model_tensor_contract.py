"""Fail-closed batch/tensor contract tests for the balanced causal model."""

from __future__ import annotations

import pytest
import torch

from pig_behavior.classification_v2.models.balanced.baselines import baseline_config
from pig_behavior.classification_v2.models.balanced.contracts import (
    BATCH_CONTRACT_CHECKS,
    SPATIAL_PREDICTIVE_DIMENSION,
    BatchContract,
    ModelBatch,
    SequenceSegment,
    TensorContractError,
    numeric_group_dimensions,
    require_batch,
    spatial_predictive_contract,
    validate_batch,
)
from pig_behavior.classification_v2.models.balanced.synthetic import (
    SyntheticBatchSpec,
    replace_numeric_group,
    synthetic_batch,
)

B3 = "B3_ACTOR_T6_PLUS_GEOMETRY_MOTION"


def _b3_batch(**kwargs: object) -> tuple[ModelBatch, BatchContract]:
    config = baseline_config(B3)
    spec = SyntheticBatchSpec(contract=config.batch_contract, batch_size=3, **kwargs)
    return synthetic_batch(spec), config.batch_contract


def test_every_declared_check_is_reported() -> None:
    batch, contract = _b3_batch()
    report = validate_batch(batch, contract)
    assert tuple(check.name for check in report.checks) == BATCH_CONTRACT_CHECKS
    assert report.passed, report.errors


def test_spatial_predictive_dimension_excludes_controls() -> None:
    dimensions = numeric_group_dimensions()
    assert dimensions == {
        "bbox_xywh_n": 4,
        "bbox_shape_n": 2,
        "motion_delta": 12,
        "roi_class_relation": 18,
        "social_relation": 10,
    }
    assert SPATIAL_PREDICTIVE_DIMENSION == 46
    contract = spatial_predictive_contract()
    assert contract["spatial_predictive_dimension"] == 46
    assert contract["controls_counted_as_predictive_features"] is False


def test_motion_width_mismatch_is_rejected() -> None:
    batch, contract = _b3_batch()
    wrong = torch.zeros((3, contract.target_length, 10), dtype=torch.float32)
    broken = replace_numeric_group(batch, "motion_delta", tensor=wrong)
    report = validate_batch(broken, contract)
    check = report.check("MOTION_DIMENSION_CONTRACT")
    assert not check.passed
    assert "width=10" in check.errors[0]
    assert "expected=12" in check.errors[0]


def test_missing_extra_duplicate_and_reordered_features_are_rejected() -> None:
    batch, contract = _b3_batch()
    canonical = tuple(batch.numeric_feature_names["motion_delta"])

    missing = replace_numeric_group(batch, "motion_delta", names=canonical[:-1])
    assert not validate_batch(missing, contract).check("FEATURE_ORDER_CONTRACT").passed

    extra = replace_numeric_group(
        batch,
        "motion_delta",
        names=(*canonical, "unexpected_motion_feature"),
    )
    assert not validate_batch(extra, contract).check("FEATURE_ORDER_CONTRACT").passed

    duplicated = replace_numeric_group(
        batch,
        "motion_delta",
        names=(canonical[0], *canonical[1:-1], canonical[0]),
    )
    duplicate_check = validate_batch(duplicated, contract).check(
        "FEATURE_ORDER_CONTRACT"
    )
    assert not duplicate_check.passed
    assert any("duplicated" in error for error in duplicate_check.errors)

    reordered = replace_numeric_group(
        batch,
        "motion_delta",
        names=(canonical[1], canonical[0], *canonical[2:]),
    )
    reorder_check = validate_batch(reordered, contract).check("FEATURE_ORDER_CONTRACT")
    assert not reorder_check.passed
    assert any("order mismatch" in error for error in reorder_check.errors)


def test_absent_required_modality_is_rejected() -> None:
    batch, contract = _b3_batch()
    dropped = replace_numeric_group(batch, "motion_delta", drop=True)
    report = validate_batch(dropped, contract)
    assert not report.passed
    assert not report.check("MOTION_DIMENSION_CONTRACT").passed
    with pytest.raises(TensorContractError):
        require_batch(dropped, contract)


def test_maskable_modality_requires_explicit_availability() -> None:
    contract = BatchContract(
        required_modalities=("actor_images", "social_relation"),
        target_length=6,
        maskable_modalities=("social_relation",),
    )
    batch = synthetic_batch(SyntheticBatchSpec(contract=contract, batch_size=2))
    without_availability = ModelBatch(
        target=batch.target,
        numeric_feature_names=batch.numeric_feature_names,
        quality_mask_names=batch.quality_mask_names,
        modality_availability={},
        labels=batch.labels,
        native_unit_id=batch.native_unit_id,
        window_id=batch.window_id,
        motion_schema_hash=batch.motion_schema_hash,
        motion_schema_version=batch.motion_schema_version,
    )
    check = validate_batch(without_availability, contract).check("MASK_SHAPE_CONTRACT")
    assert not check.passed
    assert any("modality_availability" in error for error in check.errors)


def test_nonfinite_values_are_rejected() -> None:
    batch, contract = _b3_batch()
    poisoned = batch.target.numeric_groups["bbox_xywh_n"].clone()
    poisoned[0, 0, 0] = float("nan")
    broken = replace_numeric_group(batch, "bbox_xywh_n", tensor=poisoned)
    check = validate_batch(broken, contract).check("FINITE_VALUE_CONTRACT")
    assert not check.passed
    assert "nonfinite" in check.errors[0]


def test_batch_alignment_and_forbidden_features() -> None:
    batch, contract = _b3_batch()
    misaligned = ModelBatch(
        target=batch.target,
        numeric_feature_names=batch.numeric_feature_names,
        quality_mask_names=batch.quality_mask_names,
        modality_availability=batch.modality_availability,
        labels=batch.labels,
        native_unit_id=batch.native_unit_id[:-1],
        window_id=batch.window_id,
    )
    assert not validate_batch(misaligned, contract).check(
        "BATCH_ALIGNMENT_CONTRACT"
    ).passed

    leaked = replace_numeric_group(
        batch,
        "bbox_xywh_n",
        names=("cx_n", "cy_n", "bw_n", "reviewed_behavior"),
    )
    forbidden = validate_batch(leaked, contract).check("FORBIDDEN_FEATURE_CONTRACT")
    assert not forbidden.passed
    assert "reviewed_behavior" in forbidden.errors[0]


def test_target_and_history_masks_are_separate() -> None:
    contract = BatchContract(
        required_modalities=("actor_images",),
        target_length=6,
        history_length=12,
    )
    batch = synthetic_batch(
        SyntheticBatchSpec(
            contract=BatchContract(
                required_modalities=("actor_images",),
                target_length=6,
            ),
            batch_size=2,
        )
    )
    report = validate_batch(batch, contract)
    check = report.check("HISTORY_LENGTH_CONTRACT")
    assert not check.passed
    assert "history_length=12" in check.errors[0]


def test_history_after_prediction_endpoint_is_rejected() -> None:
    contract = BatchContract(
        required_modalities=("actor_images",),
        target_length=6,
        history_length=6,
    )
    base = synthetic_batch(
        SyntheticBatchSpec(
            contract=BatchContract(
                required_modalities=("actor_images",),
                target_length=6,
            ),
            batch_size=2,
        )
    )
    valid = torch.ones((2, 6), dtype=torch.bool)
    good_history = SequenceSegment(
        valid_mask=valid,
        frame_offsets=torch.arange(-11, -5).repeat(2, 1),
        images=torch.zeros((2, 6, 3, 16, 16)),
    )
    bad_history = SequenceSegment(
        valid_mask=valid,
        frame_offsets=torch.arange(-3, 3).repeat(2, 1),
        images=torch.zeros((2, 6, 3, 16, 16)),
    )
    for history, expect_valid in ((good_history, True), (bad_history, False)):
        candidate = ModelBatch(
            target=base.target,
            history=history,
            numeric_feature_names=base.numeric_feature_names,
            quality_mask_names=base.quality_mask_names,
            modality_availability=base.modality_availability,
            labels=base.labels,
            native_unit_id=base.native_unit_id,
            window_id=base.window_id,
        )
        check = validate_batch(candidate, contract).check("HISTORY_LENGTH_CONTRACT")
        assert check.passed is expect_valid, check.errors
