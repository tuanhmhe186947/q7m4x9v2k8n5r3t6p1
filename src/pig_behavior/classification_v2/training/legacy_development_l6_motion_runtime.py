"""Crash-bounded runtime adapter for legacy L6 motion controls."""

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
from pig_behavior.classification_v2.training.legacy_development_l6_motion import (
    LINEAGE_SCOPE,
    MODES,
    SHORT_SCOPE,
    LegacyL6MotionConfig,
    fit_motion_normalization,
    l6_motion_feature_whitelist,
    load_motion_training_inputs,
    motion_implementation_hashes,
    motion_training_git_guard,
    preflight_motion_mode,
    train_motion_core,
)

RUN_RESULT_SCHEMA = (
    "classification_v2.legacy_development_l6.motion_run_result.v1"
)
RUN_MANIFEST_SCHEMA = (
    "classification_v2.legacy_development_l6.motion_run_manifest.v1"
)
ARTIFACT_MANIFEST_SCHEMA = (
    "classification_v2.legacy_development_l6.motion_artifacts.v1"
)
REPEAT_GATE_SCHEMA = (
    "classification_v2.legacy_development_l6.motion_repeat_gate.v1"
)
MATRIX_GATE_SCHEMA = (
    "classification_v2.legacy_development_l6.motion_short_matrix.v1"
)
RUN_AUDIT_SCHEMA = (
    "classification_v2.legacy_development_l6.motion_run_audit.v1"
)
FAILURE_SCHEMA = (
    "classification_v2.legacy_development_l6.motion_failure.v1"
)
CHECKPOINT_SCHEMA = (
    "classification_v2.legacy_development_l6.motion_checkpoint.v1"
)
CHECKPOINT_MANIFEST_SCHEMA = (
    "classification_v2.legacy_development_l6.motion_checkpoint_manifest.v1"
)
PREDICTION_MANIFEST_SCHEMA = (
    "classification_v2.legacy_development_l6.motion_prediction_manifest.v1"
)
ENVIRONMENT_SCHEMA = (
    "classification_v2.legacy_development_l6.motion_environment.v1"
)

PASS_TRAINING_STATUS = "PASS_LEGACY_DEVELOPMENT_L6_MOTION_TRAINING"
PASS_REPEAT_STATUS = "PASS_LEGACY_DEVELOPMENT_L6_MOTION_REPEAT"
PASS_MATRIX_STATUS = "PASS_LEGACY_DEVELOPMENT_L6_MOTION_SHORT_MATRIX"

MOTION_RUNTIME_SPEC = CachedModalityRuntimeSpec(
    modality_name="motion",
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
    preflight=preflight_motion_mode,
    load_inputs=load_motion_training_inputs,
    fit_normalization=fit_motion_normalization,
    feature_whitelist=l6_motion_feature_whitelist,
    train_core=train_motion_core,
    git_guard=motion_training_git_guard,
    implementation_hashes=motion_implementation_hashes,
)


def run_motion_mode(
    config: LegacyL6MotionConfig,
    *,
    mode: str,
    run_id: str,
) -> dict[str, Any]:
    return run_cached_modality_mode(
        MOTION_RUNTIME_SPEC,
        config,
        mode=mode,
        run_id=run_id,
    )


def audit_motion_run(
    config: LegacyL6MotionConfig,
    *,
    result_path: Path,
) -> dict[str, Any]:
    return audit_cached_modality_run(
        MOTION_RUNTIME_SPEC,
        config,
        result_path=result_path,
    )


def audit_motion_repeat_gate(
    config: LegacyL6MotionConfig,
    *,
    mode: str,
    primary_result_path: Path,
    repeat_result_path: Path,
) -> dict[str, Any]:
    return audit_cached_modality_repeat_gate(
        MOTION_RUNTIME_SPEC,
        config,
        mode=mode,
        primary_result_path=primary_result_path,
        repeat_result_path=repeat_result_path,
    )


def write_motion_repeat_gate(
    config: LegacyL6MotionConfig,
    *,
    mode: str,
    primary_result_path: Path,
    repeat_result_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    return write_cached_modality_repeat_gate(
        MOTION_RUNTIME_SPEC,
        config,
        mode=mode,
        primary_result_path=primary_result_path,
        repeat_result_path=repeat_result_path,
        output_path=output_path,
    )


def audit_motion_short_matrix(
    config: LegacyL6MotionConfig,
    *,
    repeat_gate_paths: dict[str, Path],
) -> dict[str, Any]:
    return audit_cached_modality_short_matrix(
        MOTION_RUNTIME_SPEC,
        config,
        repeat_gate_paths=repeat_gate_paths,
    )


def write_motion_short_matrix(
    config: LegacyL6MotionConfig,
    *,
    repeat_gate_paths: dict[str, Path],
    output_path: Path,
) -> dict[str, Any]:
    return write_cached_modality_short_matrix(
        MOTION_RUNTIME_SPEC,
        config,
        repeat_gate_paths=repeat_gate_paths,
        output_path=output_path,
    )
