"""Bind source-to-window evidence to one immutable training snapshot."""

from __future__ import annotations

from typing import Any

LINEAGE_SCHEMA_VERSION = "classification_v2.source_to_window_lineage.v1"
DEFAULT_LINEAGE_ARTIFACT = "identifier_v2_lineage_audit"
REQUIRED_TRAINING_AUTHORIZATIONS = (
    "reviewed_dataset_authorized",
    "model_training_authorized",
)

BASE_ARTIFACT_HASH_BINDINGS = {
    "tables.image_frame_manifest": "image_frame_context_manifest",
    "tables.image_window_manifest": "image_window_context_manifest",
    "train_ready_tables.X_window_features": "tabular_X",
    "train_ready_tables.y_behavior": "y_behavior",
    "train_ready_tables.train_mask": "train_mask",
    "train_ready_tables.sample_weight": "sample_weight",
    "audits.train_ready": "train_ready_audit",
    "audits.spatial": "spatial_sequence_audit",
    "audits.image_context": "image_context_index_audit",
    "spatial_npz": "spatial_sequences",
}
INTERACTION_ARTIFACT_HASH_BINDINGS = {
    "tables.interaction_window_manifest": (
        "interaction_window_context_manifest"
    ),
    "audits.interaction_context": "interaction_context_audit",
}


def audit_training_lineage_packet(
    lineage: dict[str, Any],
    snapshot_check: dict[str, Any],
    *,
    lineage_file_sha256: str | None,
    expected_git_commit: str | None,
    require_full_multimodal: bool,
    require_clean_code: bool,
    require_training_authorization: bool,
) -> dict[str, Any]:
    """Validate ordered keys, artifact hashes, code, and review authority."""

    technical_errors: list[str] = []
    authorization_errors: list[str] = []
    current = snapshot_check.get("current") or {}
    artifacts = current.get("artifacts") or {}

    if snapshot_check.get("valid") is not True:
        technical_errors.append(
            f"training_snapshot_invalid={snapshot_check.get('errors')}"
        )
    if lineage.get("schema_version") != LINEAGE_SCHEMA_VERSION:
        technical_errors.append(
            "lineage_schema_version_mismatch="
            f"{lineage.get('schema_version')}"
        )
    if lineage.get("technical_pass") is not True or lineage.get("errors"):
        technical_errors.append(
            f"identifier_lineage_not_technical_pass={lineage.get('errors')}"
        )
    if (
        require_full_multimodal
        and lineage.get("full_multimodal_lineage_complete") is not True
    ):
        technical_errors.append("full_multimodal_lineage_incomplete")

    ordered_audit = _audit_ordered_window_hashes(
        lineage,
        current,
        artifacts,
    )
    technical_errors.extend(ordered_audit["errors"])

    hash_audit = _audit_artifact_hash_bindings(
        lineage,
        artifacts,
        require_full_multimodal=require_full_multimodal,
    )
    technical_errors.extend(hash_audit["errors"])

    lineage_artifact_name = (
        current.get("lineage_audit_artifact")
        or DEFAULT_LINEAGE_ARTIFACT
    )
    snapshot_lineage_sha = artifacts.get(lineage_artifact_name, {}).get(
        "sha256"
    )
    if not lineage_file_sha256 or lineage_file_sha256 != snapshot_lineage_sha:
        technical_errors.append(
            "lineage_file_snapshot_hash_mismatch="
            f"file:{lineage_file_sha256},snapshot:{snapshot_lineage_sha}"
        )

    code_audit = _audit_code_state(
        lineage,
        expected_git_commit=expected_git_commit,
        require_clean_code=require_clean_code,
    )
    technical_errors.extend(code_audit["errors"])

    authorization = lineage.get("authorization") or {}
    for field in REQUIRED_TRAINING_AUTHORIZATIONS:
        if authorization.get(field) is not True:
            authorization_errors.append(
                f"lineage_requires_{field}_true"
            )
    if require_training_authorization and lineage.get(
        "human_review_blockers"
    ):
        authorization_errors.append("lineage_has_human_review_blockers")

    active_authorization_errors = (
        authorization_errors if require_training_authorization else []
    )
    errors = [*technical_errors, *active_authorization_errors]
    return {
        "schema_version": "classification_v2.training_lineage_binding.v1",
        "snapshot_id": snapshot_check.get("expected_snapshot_id"),
        "snapshot_contract_digest": current.get("contract_digest"),
        "lineage_artifact_name": lineage_artifact_name,
        "lineage_file_sha256": lineage_file_sha256,
        "expected_ordered_window_id_sha256": ordered_audit[
            "expected_sha256"
        ],
        "ordered_window_audit": ordered_audit,
        "artifact_hash_audit": hash_audit,
        "code_audit": code_audit,
        "authorization": authorization,
        "technical_errors": technical_errors,
        "authorization_errors": authorization_errors,
        "technical_valid": not technical_errors,
        "training_authorized": not authorization_errors,
        "errors": errors,
        "valid": not errors,
    }


def _audit_ordered_window_hashes(
    lineage: dict[str, Any],
    current: dict[str, Any],
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    """Require split, image, interaction, and exporter positional identity."""

    errors: list[str] = []
    required = current.get("required_ordered_window_artifacts") or []
    if not required:
        errors.append("snapshot_missing_required_ordered_window_artifacts")
    expected = artifacts.get("split_manifest", {}).get(
        "ordered_key_sha256"
    )
    if not expected:
        errors.append("snapshot_missing_split_ordered_window_hash")
    artifact_hashes = {
        name: artifacts.get(name, {}).get("ordered_key_sha256")
        for name in required
    }
    for name, value in artifact_hashes.items():
        if not expected or value != expected:
            errors.append(f"snapshot_ordered_window_hash_mismatch={name}")

    key_alignment = current.get("key_alignment") or {}
    if key_alignment.get("aligned") is not True:
        errors.append(
            f"snapshot_key_alignment_invalid={key_alignment.get('mismatched')}"
        )

    lineage_hash = (lineage.get("window_lineage") or {}).get(
        "ordered_window_id_sha256"
    )
    if not expected or lineage_hash != expected:
        errors.append("lineage_ordered_window_hash_mismatch")
    exporter_hashes = lineage.get("exported_window_hashes") or {}
    if exporter_hashes.get("expected_sha256") != expected:
        errors.append("lineage_exporter_expected_hash_mismatch")
    for name, item in (exporter_hashes.get("artifacts") or {}).items():
        if item.get("audited") is False:
            continue
        if item.get("matches_sequence") is not True:
            errors.append(f"lineage_exporter_hash_mismatch={name}")
    return {
        "expected_sha256": expected,
        "snapshot_artifact_sha256": artifact_hashes,
        "lineage_sha256": lineage_hash,
        "errors": errors,
        "valid": not errors,
    }


def _audit_artifact_hash_bindings(
    lineage: dict[str, Any],
    artifacts: dict[str, Any],
    *,
    require_full_multimodal: bool,
) -> dict[str, Any]:
    """Bind audit input bytes to the exact files frozen in the snapshot."""

    bindings = dict(BASE_ARTIFACT_HASH_BINDINGS)
    if require_full_multimodal:
        bindings.update(INTERACTION_ARTIFACT_HASH_BINDINGS)
    audit_hashes = lineage.get("artifact_sha256") or {}
    comparisons: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for audit_name, snapshot_name in bindings.items():
        audit_sha = audit_hashes.get(audit_name)
        snapshot_sha = artifacts.get(snapshot_name, {}).get("sha256")
        matches = bool(audit_sha) and audit_sha == snapshot_sha
        comparisons[audit_name] = {
            "snapshot_artifact": snapshot_name,
            "audit_sha256": audit_sha,
            "snapshot_sha256": snapshot_sha,
            "matches": matches,
        }
        if not matches:
            errors.append(
                "lineage_artifact_hash_mismatch="
                f"{audit_name}->{snapshot_name}"
            )
    return {
        "comparisons": comparisons,
        "errors": errors,
        "valid": not errors,
    }


def _audit_code_state(
    lineage: dict[str, Any],
    *,
    expected_git_commit: str | None,
    require_clean_code: bool,
) -> dict[str, Any]:
    """Reject stale or dirty lineage evidence before expensive execution."""

    code_state = lineage.get("code_state") or {}
    audit_commit = code_state.get("git_sha")
    dirty = code_state.get("dirty_worktree")
    errors: list[str] = []
    if not expected_git_commit or audit_commit != expected_git_commit:
        errors.append(
            "lineage_git_commit_mismatch="
            f"audit:{audit_commit},expected:{expected_git_commit}"
        )
    if require_clean_code and dirty is not False:
        errors.append(f"lineage_requires_clean_worktree={dirty}")
    return {
        "audit_git_commit": audit_commit,
        "expected_git_commit": expected_git_commit,
        "audit_dirty_worktree": dirty,
        "errors": errors,
        "valid": not errors,
    }


__all__ = [
    "DEFAULT_LINEAGE_ARTIFACT",
    "LINEAGE_SCHEMA_VERSION",
    "audit_training_lineage_packet",
]
