"""Machine-readable paper-grade protocol checks for classification_v2.

The long-form roadmap documents the scientific plan, but a checker is needed to
keep future work honest. This module verifies that the declared Q2-oriented
claim boundary is backed by current contracts, leakage controls, native OOF
folds, and source-domain audits before any result is described as
publication-facing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.contracts.training_snapshot import check_training_snapshot
from pig_behavior.classification_v2.training.trainer_contract import check_trainer_contract


def check_paper_grade_protocol(protocol_path: Path) -> dict[str, Any]:
    """Validate the Q2-oriented research protocol and its evidence artifacts."""

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []
    _check_claim_boundary(protocol.get("claim_boundary", {}), errors)
    _check_documents(protocol.get("required_documents", []), errors)
    artifact_audit = _check_artifacts(protocol.get("required_artifacts", {}), errors, warnings)
    _check_primary_evaluation(protocol.get("primary_evaluation", {}), artifact_audit, errors)
    _check_confusion_pairs(protocol.get("mandatory_confusion_pairs", []), errors)
    _check_ablation_ladder(protocol.get("ablation_ladder", []), errors)
    _check_module_design(protocol.get("required_module_design", []), errors)
    return {
        "protocol_path": str(protocol_path),
        "version": protocol.get("version"),
        "target_strength": protocol.get("claim_boundary", {}).get("target_strength"),
        "primary_claim": protocol.get("claim_boundary", {}).get("primary_claim"),
        "artifact_audit": artifact_audit,
        "confusion_pair_count": len(protocol.get("mandatory_confusion_pairs", [])),
        "ablation_count": len(protocol.get("ablation_ladder", [])),
        "module_design_count": len(protocol.get("required_module_design", [])),
        "warnings": warnings,
        "errors": errors,
        "valid": not errors,
    }


def write_paper_grade_protocol_audit(protocol_path: Path, output_path: Path) -> dict[str, Any]:
    """Run the protocol checker and write its audit JSON."""

    audit = check_paper_grade_protocol(protocol_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit


def _check_claim_boundary(boundary: dict[str, Any], errors: list[str]) -> None:
    if boundary.get("target_strength") != "Q2_strong":
        errors.append("claim_boundary_target_strength_must_be_Q2_strong")
    primary = str(boundary.get("primary_claim", "")).lower()
    if "session" not in primary and "video" not in primary:
        errors.append("primary_claim_must_be_session_or_video_safe")
    prohibited = " ".join(str(item).lower() for item in boundary.get("prohibited_claims", []))
    for token in ["cross-farm", "unseen-animal", "sota"]:
        if token not in prohibited:
            errors.append(f"missing_prohibited_claim_boundary={token}")


def _check_documents(paths: list[str], errors: list[str]) -> None:
    for path in paths:
        if not Path(path).exists():
            errors.append(f"missing_required_document={path}")


def _check_artifacts(
    artifacts: dict[str, str],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for name, raw_path in artifacts.items():
        path = Path(raw_path)
        exists = path.exists()
        audit[name] = {"path": raw_path, "exists": exists}
        if not exists:
            errors.append(f"missing_required_artifact={name}:{raw_path}")
            continue
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            audit[name].update(_summarize_json_artifact(name, payload))
            if payload.get("errors"):
                errors.append(f"artifact_errors={name}:{payload.get('errors')}")

    snapshot_path = artifacts.get("training_snapshot")
    if snapshot_path and Path(snapshot_path).exists():
        snapshot_check = check_training_snapshot(Path(snapshot_path))
        audit["training_snapshot_check"] = {
            "valid": snapshot_check["valid"],
            "errors": snapshot_check["errors"],
            "warnings": snapshot_check["warnings"],
            "expected_snapshot_id": snapshot_check["expected_snapshot_id"],
            "current_snapshot_id": snapshot_check["current_snapshot_id"],
        }
        if not snapshot_check["valid"]:
            errors.append(f"training_snapshot_check_failed={snapshot_check['errors']}")
        warnings.extend(f"training_snapshot_warning={w}" for w in snapshot_check["warnings"])

    trainer_path = artifacts.get("trainer_contract")
    if trainer_path and Path(trainer_path).exists():
        trainer_check = check_trainer_contract(Path(trainer_path))
        audit["trainer_contract_check"] = {
            "valid": trainer_check["valid"],
            "errors": trainer_check["errors"],
            "feature_count": trainer_check["feature_count"],
            "forbidden_x_columns": trainer_check["forbidden_x_columns"],
        }
        if not trainer_check["valid"]:
            errors.append(f"trainer_contract_check_failed={trainer_check['errors']}")
    return audit


def _summarize_json_artifact(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "errors": payload.get("errors", []),
        "valid": payload.get("valid"),
    }
    for key in [
        "rows",
        "kept_rows",
        "balanced_accuracy",
        "fold_count",
        "duplicate_temporal_unit_key",
        "forbidden_x_columns",
    ]:
        if key in payload:
            summary[key] = payload[key]
    if name == "spatial_shortcut":
        summary["controls"] = {
            control: {
                "accuracy": values.get("accuracy"),
                "balanced_accuracy": values.get("balanced_accuracy"),
            }
            for control, values in payload.get("controls", {}).items()
        }
    return summary


def _check_primary_evaluation(
    primary: dict[str, Any],
    artifact_audit: dict[str, Any],
    errors: list[str],
) -> None:
    if primary.get("unit") != "native_temporal_unit":
        errors.append("primary_evaluation_unit_must_be_native_temporal_unit")
    if "recording" not in str(primary.get("split_policy", "")).lower():
        errors.append("primary_split_policy_must_be_recording_group_safe")
    fold_count = artifact_audit.get("native_oof_folds", {}).get("fold_count", 0)
    if int(fold_count or 0) < int(primary.get("minimum_fold_count", 3)):
        errors.append(f"native_oof_fold_count_too_low={fold_count}")
    duplicate_units = artifact_audit.get("native_oof_folds", {}).get("duplicate_temporal_unit_key", 0)
    if int(duplicate_units or 0):
        errors.append(f"native_oof_duplicate_temporal_unit_key={duplicate_units}")


def _check_confusion_pairs(pairs: list[list[str]], errors: list[str]) -> None:
    required = {
        ("fight", "social-nose"),
        ("eat", "stand"),
        ("drink", "stand"),
        ("playwithtoy", "explore"),
        ("lying", "sitting"),
        ("move", "stand"),
    }
    observed = {tuple(pair) for pair in pairs}
    missing = sorted(required.difference(observed))
    if missing:
        errors.append(f"missing_mandatory_confusion_pairs={missing}")


def _check_ablation_ladder(ladder: list[dict[str, Any]], errors: list[str]) -> None:
    ids = {str(item.get("id")) for item in ladder}
    required = {"B0", "B1", "B2", "B3", "B5", "P1", "P2", "P3", "P4"}
    missing = sorted(required.difference(ids))
    if missing:
        errors.append(f"missing_ablation_ids={missing}")


def _check_module_design(modules: list[str], errors: list[str]) -> None:
    required_tokens = ["recording_groups", "sequence_dataset", "image_cache", "multimodal_fusion", "metrics"]
    joined = " ".join(modules)
    missing = [token for token in required_tokens if token not in joined]
    if missing:
        errors.append(f"missing_module_design_tokens={missing}")
