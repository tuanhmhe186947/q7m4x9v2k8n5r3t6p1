"""Crash-bounded runtime adapter for legacy L6 ROI-relation controls."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.training.legacy_development_l6_cached_modality_runtime import (
    CachedModalityRuntimeSpec,
    audit_cached_modality_repeat_gate,
    audit_cached_modality_run,
    audit_cached_modality_short_matrix,
    run_cached_modality_mode,
    write_cached_modality_repeat_gate,
    write_cached_modality_short_matrix,
)
from pig_behavior.classification_v2.training.legacy_development_l6_roi_relation import (
    LINEAGE_SCOPE,
    MODES,
    SHORT_SCOPE,
    LegacyL6ROIRelationConfig,
    fit_roi_relation_normalization,
    l6_roi_relation_feature_whitelist,
    load_roi_relation_training_inputs,
    preflight_roi_relation_mode,
    roi_relation_implementation_hashes,
    roi_relation_training_git_guard,
    train_roi_relation_core,
)

RUN_RESULT_SCHEMA = (
    "classification_v2.legacy_development_l6.roi_relation_run_result.v1"
)
RUN_MANIFEST_SCHEMA = (
    "classification_v2.legacy_development_l6.roi_relation_run_manifest.v1"
)
ARTIFACT_MANIFEST_SCHEMA = (
    "classification_v2.legacy_development_l6.roi_relation_artifacts.v1"
)
REPEAT_GATE_SCHEMA = (
    "classification_v2.legacy_development_l6.roi_relation_repeat_gate.v1"
)
MATRIX_GATE_SCHEMA = (
    "classification_v2.legacy_development_l6.roi_relation_short_matrix.v1"
)
RUN_AUDIT_SCHEMA = (
    "classification_v2.legacy_development_l6.roi_relation_run_audit.v1"
)
FAILURE_SCHEMA = (
    "classification_v2.legacy_development_l6.roi_relation_failure.v1"
)
CHECKPOINT_SCHEMA = (
    "classification_v2.legacy_development_l6.roi_relation_checkpoint.v1"
)
CHECKPOINT_MANIFEST_SCHEMA = (
    "classification_v2.legacy_development_l6.roi_relation_checkpoint_manifest.v1"
)
PREDICTION_MANIFEST_SCHEMA = (
    "classification_v2.legacy_development_l6.roi_relation_prediction_manifest.v1"
)
ENVIRONMENT_SCHEMA = (
    "classification_v2.legacy_development_l6.roi_relation_environment.v1"
)

PASS_TRAINING_STATUS = (
    "PASS_LEGACY_DEVELOPMENT_L6_ROI_RELATION_TRAINING"
)
PASS_REPEAT_STATUS = "PASS_LEGACY_DEVELOPMENT_L6_ROI_RELATION_REPEAT"
PASS_MATRIX_STATUS = "PASS_LEGACY_DEVELOPMENT_L6_ROI_RELATION_SHORT_MATRIX"

ROI_RELATION_RUNTIME_SPEC = CachedModalityRuntimeSpec(
    modality_name="roi_relation",
    modes=MODES,
    lineage_scope=LINEAGE_SCOPE,
    short_scope=SHORT_SCOPE,
    run_result_schema=RUN_RESULT_SCHEMA,
    run_manifest_schema=RUN_MANIFEST_SCHEMA,
    artifact_manifest_schema=ARTIFACT_MANIFEST_SCHEMA,
    repeat_gate_schema=REPEAT_GATE_SCHEMA,
    matrix_gate_schema=MATRIX_GATE_SCHEMA,
    run_audit_schema=RUN_AUDIT_SCHEMA,
    failure_schema=FAILURE_SCHEMA,
    checkpoint_schema=CHECKPOINT_SCHEMA,
    checkpoint_manifest_schema=CHECKPOINT_MANIFEST_SCHEMA,
    prediction_manifest_schema=PREDICTION_MANIFEST_SCHEMA,
    environment_schema=ENVIRONMENT_SCHEMA,
    pass_training_status=PASS_TRAINING_STATUS,
    pass_repeat_status=PASS_REPEAT_STATUS,
    pass_matrix_status=PASS_MATRIX_STATUS,
    preflight=preflight_roi_relation_mode,
    load_inputs=load_roi_relation_training_inputs,
    fit_normalization=fit_roi_relation_normalization,
    feature_whitelist=l6_roi_relation_feature_whitelist,
    train_core=train_roi_relation_core,
    git_guard=roi_relation_training_git_guard,
    implementation_hashes=roi_relation_implementation_hashes,
)


def run_roi_relation_mode(
    config: LegacyL6ROIRelationConfig,
    *,
    mode: str,
    run_id: str,
) -> dict[str, Any]:
    return run_cached_modality_mode(
        ROI_RELATION_RUNTIME_SPEC,
        config,
        mode=mode,
        run_id=run_id,
    )


def audit_roi_relation_run(
    config: LegacyL6ROIRelationConfig,
    *,
    result_path: Path,
) -> dict[str, Any]:
    return audit_cached_modality_run(
        ROI_RELATION_RUNTIME_SPEC,
        config,
        result_path=result_path,
    )


def audit_roi_relation_repeat_gate(
    config: LegacyL6ROIRelationConfig,
    *,
    mode: str,
    primary_result_path: Path,
    repeat_result_path: Path,
) -> dict[str, Any]:
    return audit_cached_modality_repeat_gate(
        ROI_RELATION_RUNTIME_SPEC,
        config,
        mode=mode,
        primary_result_path=primary_result_path,
        repeat_result_path=repeat_result_path,
    )


def write_roi_relation_repeat_gate(
    config: LegacyL6ROIRelationConfig,
    *,
    mode: str,
    primary_result_path: Path,
    repeat_result_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    return write_cached_modality_repeat_gate(
        ROI_RELATION_RUNTIME_SPEC,
        config,
        mode=mode,
        primary_result_path=primary_result_path,
        repeat_result_path=repeat_result_path,
        output_path=output_path,
    )


def audit_roi_relation_short_matrix(
    config: LegacyL6ROIRelationConfig,
    *,
    repeat_gate_paths: dict[str, Path],
) -> dict[str, Any]:
    return audit_cached_modality_short_matrix(
        ROI_RELATION_RUNTIME_SPEC,
        config,
        repeat_gate_paths=repeat_gate_paths,
    )


def write_roi_relation_short_matrix(
    config: LegacyL6ROIRelationConfig,
    *,
    repeat_gate_paths: dict[str, Path],
    output_path: Path,
) -> dict[str, Any]:
    return write_cached_modality_short_matrix(
        ROI_RELATION_RUNTIME_SPEC,
        config,
        repeat_gate_paths=repeat_gate_paths,
        output_path=output_path,
    )
