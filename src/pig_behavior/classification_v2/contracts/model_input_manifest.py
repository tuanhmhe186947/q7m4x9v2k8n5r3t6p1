"""Build one run-bound, inference-compatible classifier input manifest.

The generated data contract is the only path authority. This module never
joins a root with conventional child filenames and never falls back to the
historical canonical output tree.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.contracts.versioned_data_contract import (
    GENERATED_CONTRACT_SCHEMA_VERSION,
    validate_generated_data_contract,
)

MODEL_INPUT_MANIFEST_SCHEMA_VERSION = (
    "classification_v2.model_input_manifest.v3"
)
FILE_CHUNK_BYTES = 1024 * 1024

PREDICTIVE_ARTIFACTS = (
    "tabular_X",
    "spatial_sequences",
    "actor_packed_cache_tensor",
    "visual_packed_cache_tensor",
)
INDEX_ARTIFACTS = (
    "image_frame_context_manifest",
    "image_window_context_manifest",
    "actor_cache_manifest",
    "actor_packed_cache_index",
    "visual_context_manifest",
    "visual_packed_cache_index",
)
MASK_CONTROL_ARTIFACTS = (
    "interaction_window_context_manifest",
    "train_mask",
    "sample_weight",
    "event_weight_manifest",
    "fold_event_weight_manifest",
    "temporal_view_selection_manifest",
    "fixed6_observed_time_manifest",
)
TARGET_ARTIFACTS = (
    "y_behavior",
    "y_auxiliary_targets",
)
SPLIT_ARTIFACTS = (
    "split_manifest",
    "q2_outer_fold_assignments",
    "q2_outer_inner_roles",
)
FEATURE_CONTRACT_ARTIFACTS = (
    "feature_whitelist",
    "feature_blacklist",
    "feature_whitelist_audit",
)
REVIEW_AUTHORITY_ARTIFACTS = (
    "hidden_review_unit_manifest",
    "hidden_review_decisions",
    "hidden_review_decision_coverage_audit",
    "hidden_review_scientific_gate",
    "hidden_apply_audit",
    "hidden_confusion_audit",
    "hidden_reviewed_frame_features",
    "full_review_unit_manifest",
    "roi_behavior_decisions",
    "motion_behavior_decisions",
    "posture_behavior_decisions",
    "interaction_behavior_decisions",
    "behavior_decision_coverage_audit",
    "review_unit_decisions_combined",
    "behavior_apply_audit",
    "reviewed_frame_features",
    "cvat_anchor_1020_audit",
    "temporal_evidence_audit",
)
DATA_AUDIT_ARTIFACTS = (
    "temporal_unit_audit",
    "q2_grouped_fold_audit",
    "train_ready_audit",
    "spatial_sequence_audit",
    "identifier_lineage_audit",
    "source_to_window_lineage_audit",
    "leakage_audit",
    "class_by_fold_support",
    "source_by_fold_support",
    "domain_controls_audit",
)

ARTIFACT_GROUPS = {
    "predictive": PREDICTIVE_ARTIFACTS,
    "index_only": INDEX_ARTIFACTS,
    "mask_and_control": MASK_CONTROL_ARTIFACTS,
    "targets": TARGET_ARTIFACTS,
    "splits": SPLIT_ARTIFACTS,
    "feature_contract": FEATURE_CONTRACT_ARTIFACTS,
    "review_authority": REVIEW_AUTHORITY_ARTIFACTS,
    "data_audits": DATA_AUDIT_ARTIFACTS,
}
REQUIRED_ARTIFACTS = tuple(
    dict.fromkeys(
        name
        for names in ARTIFACT_GROUPS.values()
        for name in names
    )
)
EXPECTED_ARTIFACT_SCOPES = {
    name: "agent_derived"
    for name in (
        *PREDICTIVE_ARTIFACTS,
        *INDEX_ARTIFACTS,
        *MASK_CONTROL_ARTIFACTS,
        *TARGET_ARTIFACTS,
        *SPLIT_ARTIFACTS,
        *FEATURE_CONTRACT_ARTIFACTS,
        *DATA_AUDIT_ARTIFACTS,
    )
}
EXPECTED_ARTIFACT_SCOPES.update(
    {name: "human_review" for name in REVIEW_AUTHORITY_ARTIFACTS}
)
VALID_BEHAVIORS = (
    "drink",
    "eat",
    "fight",
    "social-nose",
    "explore",
    "lying",
    "stand",
    "move",
    "sitting",
    "playwithtoy",
)


class ModelInputManifestError(ValueError):
    """Expose stable validation errors to CLI callers and tests."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = tuple(errors)
        super().__init__(f"invalid model input manifest: {errors}")


@dataclass(frozen=True, slots=True)
class ModelInputManifestBuild:
    """Validated manifest payload and its contract-declared destination."""

    manifest: dict[str, Any]
    audit: dict[str, Any]
    output_path: Path


def build_model_input_manifest(
    contract_path: Path,
    *,
    output_path: Path,
    project_root: Path,
) -> ModelInputManifestBuild:
    """Bind explicit contract artifacts without inferring any child paths."""

    root = project_root.resolve()
    contract_file, contract_relative = _project_file(
        contract_path,
        root,
        label="data_contract",
    )
    destination, destination_relative = _project_path(
        output_path,
        root,
        label="output_json",
    )
    contract = _load_json_object(contract_file, "data_contract")
    errors = validate_generated_data_contract(
        contract_file,
        project_root=root,
    )
    if (
        contract.get("generated_contract_schema_version")
        != GENERATED_CONTRACT_SCHEMA_VERSION
    ):
        errors.append("generated_contract_schema_version_mismatch")

    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("contract_artifacts_must_be_object")
        artifacts = {}
    missing_specs = sorted(set(REQUIRED_ARTIFACTS).difference(artifacts))
    if missing_specs:
        errors.append(f"contract_missing_model_artifacts={missing_specs}")

    output_spec = artifacts.get("model_input_contract")
    errors.extend(
        _validate_output_binding(
            output_spec,
            destination_relative,
        )
    )
    bindings: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_ARTIFACTS:
        spec = artifacts.get(name)
        binding, binding_errors = _artifact_binding(
            name,
            spec,
            root=root,
        )
        errors.extend(binding_errors)
        if binding is not None:
            bindings[name] = binding

    for name, expected_scope in EXPECTED_ARTIFACT_SCOPES.items():
        scope = bindings.get(name, {}).get("scope")
        if scope != expected_scope:
            errors.append(
                f"model_input_artifact_scope_mismatch:{name}:{scope}"
            )

    errors = sorted(set(errors))
    if errors:
        raise ModelInputManifestError(errors)

    manifest = _manifest_payload(
        contract,
        contract_relative=contract_relative,
        contract_sha256=_sha256_file(contract_file),
        output_relative=destination_relative,
        bindings=bindings,
    )
    audit = {
        "schema_version": (
            "classification_v2.model_input_manifest_build_audit.v1"
        ),
        "status": "PASS",
        "valid": True,
        "errors": [],
        "run_id": contract["run_id"],
        "profile": contract["profile"],
        "data_contract": contract_relative,
        "data_contract_sha256": _sha256_file(contract_file),
        "output_json": destination_relative,
        "bound_artifact_count": len(bindings),
        "canonical_fallback_used": False,
        "dataset_rows_read": 0,
        "dataset_rows_written": 0,
    }
    return ModelInputManifestBuild(
        manifest=manifest,
        audit=audit,
        output_path=destination,
    )


def write_model_input_manifest(
    build: ModelInputManifestBuild,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Write only the contract-declared agent artifact, or audit in memory."""

    existed = build.output_path.exists()
    audit = {
        **build.audit,
        "dry_run": bool(dry_run),
        "overwrite": bool(overwrite),
        "output_existed_before_write": existed,
        "artifact_written": False,
    }
    if dry_run:
        return audit
    require_output_paths_available(
        [build.output_path],
        overwrite=overwrite,
    )
    build.output_path.parent.mkdir(parents=True, exist_ok=True)
    build.output_path.write_text(
        _stable_json(build.manifest),
        encoding="utf-8",
    )
    return {
        **audit,
        "artifact_written": True,
        "output_sha256": _sha256_file(build.output_path),
    }


def _manifest_payload(
    contract: dict[str, Any],
    *,
    contract_relative: str,
    contract_sha256: str,
    output_relative: str,
    bindings: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    grouped = {
        group: {
            name: bindings[name]
            for name in names
        }
        for group, names in ARTIFACT_GROUPS.items()
    }
    return {
        "schema_version": MODEL_INPUT_MANIFEST_SCHEMA_VERSION,
        "version": MODEL_INPUT_MANIFEST_SCHEMA_VERSION,
        "run_id": contract["run_id"],
        "profile": contract["profile"],
        "data_contract": contract_relative,
        "data_contract_sha256": contract_sha256,
        "template_sha256": contract["template_sha256"],
        "artifact_map_sha256": contract["artifact_map_sha256"],
        "lineage_ids": contract["lineage_ids"],
        "lineage_roots": contract["lineage_roots"],
        "train_ready_root": contract["train_ready_root"],
        "output_json": output_relative,
        "path_policy": contract["path_policy"],
        "artifacts": {
            name: binding["path"]
            for name, binding in bindings.items()
        },
        "artifact_groups": grouped,
        "model_input_branches": {
            "actor_rgb": {
                "tensor": "actor_packed_cache_tensor",
                "index": "actor_packed_cache_index",
                "frame_manifest": "image_frame_context_manifest",
                "window_manifest": "image_window_context_manifest",
                "resize_policy": "letterbox_preserve_aspect",
            },
            "tabular": {
                "tensor": "tabular_X",
                "selection": "feature_whitelist",
                "all_numeric_selection_allowed": False,
            },
            "spatial_sequence": {
                "tensor": "spatial_sequences",
                "selection": "feature_whitelist",
                "missing_values_require_masks": True,
            },
            "interaction_visual": {
                "tensor": "visual_packed_cache_tensor",
                "index": "visual_packed_cache_index",
                "context_manifest": "visual_context_manifest",
                "availability_manifest": (
                    "interaction_window_context_manifest"
                ),
                "availability_is_behavior_evidence": False,
            },
        },
        "target_contract": {
            "primary_artifact": "y_behavior",
            "auxiliary_artifact": "y_auxiliary_targets",
            "final_head_directly_supervised": True,
            "allowed_behaviors": list(VALID_BEHAVIORS),
            "auxiliary_argmax_fed_to_final_head": False,
        },
        "temporal_contract": {
            "primary_view": "fixed6_observed_time",
            "legacy_native_frames": 16,
            "cvat_native_frames": 6,
            "windows_after_harmonization": True,
            "primary_prediction_unit": "native_temporal_unit",
        },
        "split_contract": {
            "outer_roles": "q2_outer_inner_roles",
            "outer_assignments": "q2_outer_fold_assignments",
            "random_frame_or_window_split_allowed": False,
            "outer_predictions_for_model_selection_allowed": False,
            "pig_id_cross_video_identity_allowed": False,
        },
        "feature_selection": {
            "whitelist": "feature_whitelist",
            "blacklist": "feature_blacklist",
            "audit": "feature_whitelist_audit",
            "forbidden_patterns": contract["forbidden_x_patterns"],
            "all_numeric_selection_allowed": False,
        },
        "forbidden_model_inputs": contract["forbidden_x_patterns"],
        "missing_artifacts": [],
        "inference_contract": {
            "ground_truth_only_fields_allowed": False,
            "review_fields_allowed": False,
            "missing_modalities_require_masks": True,
            "partner_selection_may_use_target_behavior": False,
        },
        "errors": [],
    }


def _artifact_binding(
    name: str,
    spec: Any,
    *,
    root: Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(spec, dict):
        return None, [f"artifact_spec_missing_or_invalid:{name}"]
    if spec.get("required") is not True:
        return None, [f"model_manifest_artifact_must_be_required:{name}"]
    path_value = spec.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        return None, [f"artifact_path_missing:{name}"]
    try:
        path, relative = _project_path(
            Path(path_value),
            root,
            label=f"artifact:{name}",
        )
    except ValueError as exc:
        return None, [f"artifact_path_invalid:{name}:{exc}"]
    if not path.is_file():
        return None, [f"required_artifact_missing:{name}:{relative}"]
    return {
        "path": relative,
        "scope": spec.get("scope"),
        "type": spec.get("type"),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }, []


def _validate_output_binding(
    spec: Any,
    destination_relative: str,
) -> list[str]:
    if not isinstance(spec, dict):
        return ["model_input_contract_artifact_missing"]
    errors: list[str] = []
    if spec.get("scope") != "agent_derived":
        errors.append("model_input_contract_scope_must_be_agent_derived")
    if spec.get("type") != "json":
        errors.append("model_input_contract_type_must_be_json")
    if spec.get("path") != destination_relative:
        errors.append("output_json_does_not_match_contract_artifact")
    return errors


def _project_file(
    path: Path,
    root: Path,
    *,
    label: str,
) -> tuple[Path, str]:
    resolved, relative = _project_path(path, root, label=label)
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} file not found: {relative}")
    return resolved, relative


def _project_path(
    path: Path,
    root: Path,
    *,
    label: str,
) -> tuple[Path, str]:
    if not str(path).strip():
        raise ValueError(f"{label} path must not be blank")
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} path is outside project root") from exc
    return resolved, relative.as_posix()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(FILE_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


__all__ = [
    "MODEL_INPUT_MANIFEST_SCHEMA_VERSION",
    "REQUIRED_ARTIFACTS",
    "ModelInputManifestBuild",
    "ModelInputManifestError",
    "build_model_input_manifest",
    "write_model_input_manifest",
]
