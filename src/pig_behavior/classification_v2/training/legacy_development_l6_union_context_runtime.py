"""Crash-bounded runtime adapter for legacy L6 union-context controls."""

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
from pig_behavior.classification_v2.training.legacy_development_l6_union_context import (
    LINEAGE_SCOPE,
    MODES,
    SHORT_SCOPE,
    LegacyL6UnionContextConfig,
    fit_union_context_normalization,
    load_union_context_training_inputs,
    preflight_union_context_mode,
    train_union_context_core,
    union_context_feature_whitelist,
    union_context_implementation_hashes,
    union_context_training_git_guard,
)

RUN_RESULT_SCHEMA = (
    "classification_v2.legacy_development_l6."
    "union_context_run_result.v1"
)
RUN_MANIFEST_SCHEMA = (
    "classification_v2.legacy_development_l6."
    "union_context_run_manifest.v1"
)
ARTIFACT_MANIFEST_SCHEMA = (
    "classification_v2.legacy_development_l6."
    "union_context_artifacts.v1"
)
REPEAT_GATE_SCHEMA = (
    "classification_v2.legacy_development_l6."
    "union_context_repeat_gate.v1"
)
MATRIX_GATE_SCHEMA = (
    "classification_v2.legacy_development_l6."
    "union_context_short_matrix.v1"
)
RUN_AUDIT_SCHEMA = (
    "classification_v2.legacy_development_l6."
    "union_context_run_audit.v1"
)
FAILURE_SCHEMA = (
    "classification_v2.legacy_development_l6."
    "union_context_failure.v1"
)
CHECKPOINT_SCHEMA = (
    "classification_v2.legacy_development_l6."
    "union_context_checkpoint.v1"
)
CHECKPOINT_MANIFEST_SCHEMA = (
    "classification_v2.legacy_development_l6."
    "union_context_checkpoint_manifest.v1"
)
PREDICTION_MANIFEST_SCHEMA = (
    "classification_v2.legacy_development_l6."
    "union_context_prediction_manifest.v1"
)
ENVIRONMENT_SCHEMA = (
    "classification_v2.legacy_development_l6."
    "union_context_environment.v1"
)

PASS_TRAINING_STATUS = (
    "PASS_LEGACY_DEVELOPMENT_L6_UNION_CONTEXT_TRAINING"
)
PASS_REPEAT_STATUS = "PASS_LEGACY_DEVELOPMENT_L6_UNION_CONTEXT_REPEAT"
PASS_MATRIX_STATUS = (
    "PASS_LEGACY_DEVELOPMENT_L6_UNION_CONTEXT_SHORT_MATRIX"
)

UNION_CONTEXT_RUNTIME_SPEC = CachedModalityRuntimeSpec(
    modality_name="union_context",
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
    preflight=preflight_union_context_mode,
    load_inputs=load_union_context_training_inputs,
    fit_normalization=fit_union_context_normalization,
    feature_whitelist=union_context_feature_whitelist,
    train_core=train_union_context_core,
    git_guard=union_context_training_git_guard,
    implementation_hashes=union_context_implementation_hashes,
)


def run_union_context_mode(
    config: LegacyL6UnionContextConfig,
    *,
    mode: str,
    run_id: str,
) -> dict[str, Any]:
    return run_cached_modality_mode(
        UNION_CONTEXT_RUNTIME_SPEC,
        config,
        mode=mode,
        run_id=run_id,
    )


def audit_union_context_run(
    config: LegacyL6UnionContextConfig,
    *,
    result_path: Path,
) -> dict[str, Any]:
    return audit_cached_modality_run(
        UNION_CONTEXT_RUNTIME_SPEC,
        config,
        result_path=result_path,
    )


def audit_union_context_repeat_gate(
    config: LegacyL6UnionContextConfig,
    *,
    mode: str,
    primary_result_path: Path,
    repeat_result_path: Path,
) -> dict[str, Any]:
    return audit_cached_modality_repeat_gate(
        UNION_CONTEXT_RUNTIME_SPEC,
        config,
        mode=mode,
        primary_result_path=primary_result_path,
        repeat_result_path=repeat_result_path,
    )


def write_union_context_repeat_gate(
    config: LegacyL6UnionContextConfig,
    *,
    mode: str,
    primary_result_path: Path,
    repeat_result_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    return write_cached_modality_repeat_gate(
        UNION_CONTEXT_RUNTIME_SPEC,
        config,
        mode=mode,
        primary_result_path=primary_result_path,
        repeat_result_path=repeat_result_path,
        output_path=output_path,
    )


def audit_union_context_short_matrix(
    config: LegacyL6UnionContextConfig,
    *,
    repeat_gate_paths: dict[str, Path],
) -> dict[str, Any]:
    return audit_cached_modality_short_matrix(
        UNION_CONTEXT_RUNTIME_SPEC,
        config,
        repeat_gate_paths=repeat_gate_paths,
    )


def write_union_context_short_matrix(
    config: LegacyL6UnionContextConfig,
    *,
    repeat_gate_paths: dict[str, Path],
    output_path: Path,
) -> dict[str, Any]:
    return write_cached_modality_short_matrix(
        UNION_CONTEXT_RUNTIME_SPEC,
        config,
        repeat_gate_paths=repeat_gate_paths,
        output_path=output_path,
    )
