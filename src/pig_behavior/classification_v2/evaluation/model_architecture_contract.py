"""Model-architecture contract checks for classification_v2.

This gate validates the scientific design boundary for the multimodal roadmap.
It does not train a model. Instead, it verifies that the declared architecture
uses explicit branches, masks, leakage controls, native-temporal evaluation, and
Q2-only claim language before later learned-model work can be treated as a
paper-facing candidate.
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.models.model_factory import MODEL_MODE_NAMES

REQUIRED_BRANCHES: tuple[str, ...] = (
    "actor_image_sequence",
    "spatial_sequence_bbox_motion_roi_social_quality",
    "label_independent_partner_context",
    "actor_partner_union_context",
)

IMPLEMENTED_STATUSES: tuple[str, ...] = (
    "implemented",
    "implemented_manifest",
    "implemented_smoke",
)

PLANNED_STATUSES: tuple[str, ...] = (
    "planned_required_before_paper_candidate",
    "planned_exploratory",
)

REQUIRED_FORBIDDEN_PATTERNS: tuple[str, ...] = (
    "manual_*",
    "review_*",
    "*behavior*",
    "*label*",
    "review_unit_id",
    "window_id",
    "temporal_unit_key",
    "video_key",
    "dataset_id",
    "pig_id",
    "track_id",
    "object_track_key",
    "source_type",
    "split",
    "split_*",
    "*_path",
)


def check_model_architecture_contract(
    contract_json: Path,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Validate a multimodal architecture contract JSON artifact.

    ``valid`` means the roadmap contract is internally consistent and aligned
    with the Q2 leakage-safe boundary. ``paper_candidate_ready`` is stricter: it
    requires every paper-candidate branch to be implemented and all declared
    blockers cleared.
    """

    root = project_root or Path(".")
    errors: list[str] = []
    warnings: list[str] = []
    contract = _read_json(contract_json, errors)
    if errors:
        return _audit(contract_json, contract, errors, warnings, paper_ready=False)

    _check_claim_boundary(contract, errors, warnings)
    _check_model_modes(contract, errors)
    _check_implemented_modules(contract, root, errors, warnings)
    branch_report = _check_branches(contract, root, errors, warnings)
    _check_fusion_policy(contract, errors, warnings)
    _check_evaluation_contract(contract, errors, warnings)

    blockers = list(contract.get("paper_candidate_blockers", []))
    paper_candidate = bool(contract.get("paper_candidate", False))
    missing_required = [
        item["name"]
        for item in branch_report
        if item["paper_candidate_required"] and item["status"] not in IMPLEMENTED_STATUSES
    ]
    paper_ready = not blockers and not missing_required and not errors
    if paper_candidate and not paper_ready:
        errors.append(
            "paper_candidate_true_but_not_ready="
            f"missing_required_branches={missing_required}; blockers={blockers}"
        )
    if not paper_candidate:
        warnings.append("paper_candidate_false_design_contract_only")
    if missing_required:
        warnings.append(f"paper_candidate_missing_required_branches={missing_required}")
    if blockers:
        warnings.append(f"paper_candidate_blockers={blockers}")

    result = _audit(contract_json, contract, errors, warnings, paper_ready=paper_ready)
    result["branch_report"] = branch_report
    result["paper_candidate_blockers"] = blockers
    result["missing_required_paper_branches"] = missing_required
    return result


def _read_json(path: Path, errors: list[str]) -> dict[str, Any]:
    """Read contract JSON and report missing or malformed files as errors."""

    if not path.exists():
        errors.append(f"missing_model_architecture_contract={path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"invalid_model_architecture_contract_json={path}:{exc}")
        return {}


def _check_claim_boundary(contract: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    """Ensure declared claims stay inside the user-approved Q2 boundary."""

    claim = contract.get("claim_boundary", {})
    if claim.get("target_strength") != "Q2_strong":
        errors.append(f"target_strength_must_be_Q2_strong={claim.get('target_strength')}")
    primary = str(claim.get("primary_claim", "")).lower()
    if "session" not in primary and "video-safe" not in primary:
        errors.append("primary_claim_must_reference_session_or_video_safe_validation")
    prohibited = " ".join(str(item).lower() for item in claim.get("prohibited_claims", []))
    if "q1" not in prohibited or "external" not in prohibited:
        warnings.append("prohibited_claims_should_explicitly_block_q1_external_generalization")
    if "pig_id" not in prohibited:
        warnings.append("prohibited_claims_should_block_pig_id_biological_identity_generalization")


def _check_model_modes(contract: dict[str, Any], errors: list[str]) -> None:
    """Require the machine-readable contract to match the factory registry."""

    observed = contract.get("model_modes")
    if not isinstance(observed, list):
        errors.append("model_modes_missing")
        return
    values = [str(value) for value in observed]
    if len(values) != len(set(values)):
        errors.append("model_modes_duplicate")
    missing = sorted(MODEL_MODE_NAMES.difference(values))
    unknown = sorted(set(values).difference(MODEL_MODE_NAMES))
    if missing or unknown:
        errors.append(
            f"model_modes_registry_mismatch=missing:{missing},unknown:{unknown}"
        )


def _check_implemented_modules(
    contract: dict[str, Any],
    root: Path,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Confirm declared implementation/checker modules exist in the worktree."""

    modules = contract.get("implemented_modules", {})
    if not isinstance(modules, dict) or not modules:
        errors.append("implemented_modules_missing")
        return
    for name, rel_path in sorted(modules.items()):
        path = root / str(rel_path)
        if not path.exists():
            errors.append(f"implemented_module_missing={name}:{rel_path}")
        elif path.suffix == ".py" and path.stat().st_size == 0:
            errors.append(f"implemented_module_empty={name}:{rel_path}")
    if "prediction_schema_checker" not in modules:
        warnings.append("prediction_schema_checker_not_declared")


def _check_branches(
    contract: dict[str, Any],
    root: Path,
    errors: list[str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Validate branch declarations and artifact expectations."""

    branches = contract.get("input_branches", [])
    if not isinstance(branches, list) or not branches:
        errors.append("input_branches_missing")
        return []
    by_name = {str(branch.get("name")): branch for branch in branches if isinstance(branch, dict)}
    missing = sorted(set(REQUIRED_BRANCHES).difference(by_name))
    if missing:
        errors.append(f"missing_required_input_branches={missing}")

    report: list[dict[str, Any]] = []
    for name, branch in sorted(by_name.items()):
        status = str(branch.get("status", ""))
        if status not in (*IMPLEMENTED_STATUSES, *PLANNED_STATUSES):
            errors.append(f"unsupported_branch_status={name}:{status}")
        mask_artifacts = branch.get("mask_artifacts", [])
        if not mask_artifacts:
            errors.append(f"branch_missing_mask_artifacts={name}")
        source_artifacts = [str(item) for item in branch.get("source_artifacts", [])]
        existing_sources = [item for item in source_artifacts if (root / item).exists()]
        if status in IMPLEMENTED_STATUSES and not existing_sources:
            errors.append(f"implemented_branch_missing_source_artifacts={name}:{source_artifacts}")
        if status in PLANNED_STATUSES:
            warnings.append(f"planned_branch_not_yet_implemented={name}")
        report.append(
            {
                "name": name,
                "status": status,
                "paper_candidate_required": bool(branch.get("paper_candidate_required")),
                "source_artifacts": source_artifacts,
                "existing_source_artifact_count": len(existing_sources),
                "mask_artifacts": list(mask_artifacts),
            }
        )
    return report


def _check_fusion_policy(contract: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    """Check fusion and leakage-control policy fields."""

    policy = contract.get("fusion_policy", {})
    if policy.get("type") != "late_fusion_with_masks":
        errors.append(f"fusion_policy_type_must_be_late_fusion_with_masks={policy.get('type')}")
    controls = {str(item) for item in policy.get("required_controls", [])}
    required_controls = (
        "branch-specific masks",
        "branch ablation",
        "shortcut controls before paper claim",
    )
    for required in required_controls:
        if required not in controls:
            errors.append(f"missing_fusion_required_control={required}")
    forbidden = [str(item) for item in policy.get("forbidden_inputs", [])]
    missing_patterns = [
        pattern
        for pattern in REQUIRED_FORBIDDEN_PATTERNS
        if not any(fnmatch.fnmatchcase(pattern, item) for item in forbidden)
    ]
    if missing_patterns:
        errors.append(f"missing_forbidden_input_patterns={missing_patterns}")
    if "pig_id" not in forbidden:
        warnings.append("pig_id_not_explicitly_forbidden")


def _check_evaluation_contract(
    contract: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate native-temporal leakage-safe evaluation declarations."""

    evaluation = contract.get("evaluation_contract", {})
    if evaluation.get("split_policy") != "recording_group_oof":
        errors.append(f"split_policy_must_be_recording_group_oof={evaluation.get('split_policy')}")
    if evaluation.get("metric_unit") != "native_temporal_unit":
        errors.append(f"metric_unit_must_be_native_temporal_unit={evaluation.get('metric_unit')}")
    if (
        evaluation.get("prediction_schema_contract")
        != "classification_v2_prediction_schema_contract_v1"
    ):
        errors.append("prediction_schema_contract_must_reference_S16_contract")
    if "macro_f1_supported" not in evaluation.get("primary_metrics", []):
        errors.append("primary_metrics_missing_macro_f1_supported")
    try:
        minimum_effect = float(evaluation.get("minimum_effect_size"))
    except (TypeError, ValueError):
        errors.append("minimum_effect_size_missing_or_non_numeric")
        return
    if minimum_effect <= 0:
        errors.append("minimum_effect_size_must_be_positive")
    if "bootstrap" not in str(evaluation.get("uncertainty", "")).lower():
        warnings.append("uncertainty_should_name_bootstrap_or_equivalent_cluster_method")


def _audit(
    contract_json: Path,
    contract: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    *,
    paper_ready: bool,
) -> dict[str, Any]:
    """Return a compact audit payload suitable for versioned evidence."""

    return {
        "schema_version": "classification_v2_model_architecture_contract_audit_v1",
        "contract_json": str(contract_json),
        "contract_version": contract.get("version"),
        "model_family": contract.get("model_family"),
        "paper_candidate": bool(contract.get("paper_candidate", False)),
        "paper_candidate_ready": bool(paper_ready),
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }
