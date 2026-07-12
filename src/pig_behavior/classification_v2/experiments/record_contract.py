"""Experiment-record gate for classification_v2.

Engineering smoke records only need basic provenance artifact hashes. A
paper-facing record has a stricter contract: it must point to the frozen
training snapshot, paper-grade protocol, source-domain control, native OOF
folds, trainer input contract, and a native-temporal evaluation contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.evaluation.native_temporal_metrics_gate import (
    check_native_temporal_metrics_gate,
)

PAPER_FACING_REQUIRED_PROVENANCE = (
    "dataset_snapshot_json",
    "paper_protocol_json",
    "paper_protocol_audit_json",
    "source_domain_audit_json",
    "native_oof_audit_json",
    "trainer_contract_json",
    "loader_input_audit_json",
)

PAPER_MODEL_REQUIRED_PROVENANCE = (
    "run_audit_json",
    "calibration_audit_json",
    "source_balanced_metrics_json",
    "confusion_comparison_json",
    "ablation_report_json",
    "runtime_benchmark_audit_json",
)

PAPER_BASELINE_REQUIRED_PROVENANCE = (
    "run_audit_json",
    "source_balanced_metrics_json",
)

PAPER_ABLATION_REQUIRED_PROVENANCE = (
    "run_audit_json",
    "source_balanced_metrics_json",
    "ablation_report_json",
)


def check_experiment_record(record_path: Path, _visited: set[Path] | None = None) -> dict[str, Any]:
    """Validate one registry record and enforce Q2 provenance requirements."""

    resolved_path = record_path.resolve()
    visited = set(_visited or set())
    if resolved_path in visited:
        return {
            "record_json": str(record_path),
            "errors": [f"cyclic_parent_experiment_record={record_path}"],
            "warnings": [],
            "valid": False,
        }
    visited.add(resolved_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []
    required = [
        "schema_version",
        "name",
        "created_at_utc",
        "git_commit",
        "git_dirty",
        "artifacts",
        "record_path",
    ]
    missing = [key for key in required if key not in record]
    if missing:
        errors.append(f"missing_record_keys={missing}")
    for artifact in record.get("artifacts", []):
        _check_artifact_record(artifact, errors)

    paper_facing = bool(record.get("paper_facing", False))
    stage = str(record.get("experiment_stage", "engineering_smoke"))
    if paper_facing or stage == "paper_facing_candidate":
        if record.get("git_dirty") is not False:
            errors.append(f"paper_facing_record_requires_clean_git={record.get('git_dirty')}")
        if not record.get("git_commit"):
            errors.append("paper_facing_record_missing_git_commit")
        if record.get("evaluation_contract", {}).get("external_generalization_claim") is True:
            errors.append("external_generalization_claim_forbidden_without_external_domain_test")
        _check_paper_facing_provenance(
            record.get("provenance", {}),
            result_kind=str(record.get("evaluation_contract", {}).get("result_kind", "")),
            visited=visited,
            errors=errors,
            warnings=warnings,
        )
    else:
        warnings.append("engineering_smoke_record_not_valid_for_paper_claims")

    native_gate = check_native_temporal_metrics_gate(
        evaluation_contract=record.get("evaluation_contract"),
        metrics_payload=record.get("metrics"),
        paper_facing=paper_facing,
        experiment_stage=stage,
    )
    errors.extend(f"native_temporal_metrics_gate:{error}" for error in native_gate["errors"])
    warnings.extend(
        f"native_temporal_metrics_gate:{warning}"
        for warning in native_gate["warnings"]
    )

    return {
        "record_json": str(record_path),
        "name": record.get("name"),
        "schema_version": record.get("schema_version"),
        "experiment_stage": stage,
        "paper_facing": paper_facing,
        "artifact_count": len(record.get("artifacts", [])),
        "git_commit": record.get("git_commit"),
        "git_dirty": record.get("git_dirty"),
        "native_temporal_metrics_gate": native_gate,
        "warnings": warnings,
        "errors": errors,
        "valid": not errors,
    }


def _check_artifact_record(artifact: dict[str, Any], errors: list[str]) -> None:
    """Validate one artifact entry written by the experiment registry."""

    if not artifact.get("exists"):
        errors.append(f"missing_artifact={artifact.get('path')}")
    if artifact.get("hash_status") == "ok" and not artifact.get("sha256"):
        errors.append(f"missing_sha256={artifact.get('path')}")


def check_parent_record_link(record_path: Path) -> dict[str, Any]:
    """Validate a linked parent/control record without requiring its parents."""

    errors: list[str] = []
    if not record_path.exists():
        errors.append(f"missing_parent_experiment_record={record_path}")
    else:
        _check_parent_record_link(record_path, errors)
    return {
        "record_json": str(record_path),
        "valid": not errors,
        "errors": errors,
    }


def _check_paper_facing_provenance(
    provenance: dict[str, Any],
    *,
    result_kind: str,
    visited: set[Path],
    errors: list[str],
    warnings: list[str],
) -> None:
    """Require all upstream audit gates before a record can support Q2 claims."""

    required = list(PAPER_FACING_REQUIRED_PROVENANCE)
    if result_kind == "model_evaluation":
        required.extend(PAPER_MODEL_REQUIRED_PROVENANCE)
    elif result_kind == "baseline_evaluation":
        required.extend(PAPER_BASELINE_REQUIRED_PROVENANCE)
    elif result_kind == "ablation_evaluation":
        required.extend(PAPER_ABLATION_REQUIRED_PROVENANCE)
    missing = [name for name in required if not provenance.get(name)]
    if missing:
        errors.append(f"missing_paper_facing_provenance={missing}")
    for name in required:
        artifact = provenance.get(name)
        if not artifact:
            continue
        if artifact.get("hash_status") == "skipped_large_file":
            warnings.append(f"provenance_hash_skipped={name}")
        _check_artifact_record(artifact, errors)
        if name.endswith("_audit_json"):
            _check_json_payload(artifact, name, errors)
        elif (
            name.endswith("_metrics_json")
            or name.endswith("_comparison_json")
            or name.endswith("_report_json")
        ):
            _check_json_payload(artifact, name, errors)
    parent_records = provenance.get("parent_record_jsons", [])
    if result_kind in {"model_evaluation", "ablation_evaluation"} and not parent_records:
        errors.append("missing_parent_experiment_records")
    for artifact in parent_records:
        _check_artifact_record(artifact, errors)
        path = Path(str(artifact.get("path", "")))
        if path.exists():
            _check_parent_record_link(path, errors)

    _check_semantic_paper_payloads(provenance, result_kind, errors)


def _check_parent_record_link(path: Path, errors: list[str]) -> None:
    """Validate parent records as comparable native-temporal controls.

    Parent records can be older baseline/control records. For a full learned
    model claim, the link must prove clean, paper-facing native-temporal metrics
    without recursively requiring the parent to have its own parent graph.
    """

    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("paper_facing") is not True:
        errors.append(f"parent_experiment_not_paper_facing={path}")
    if record.get("git_dirty") is not False:
        errors.append(f"parent_experiment_git_dirty={path}")
    evaluation = record.get("evaluation_contract") or {}
    if evaluation.get("external_generalization_claim") is True:
        errors.append(f"parent_external_generalization_claim={path}")
    if evaluation.get("primary_metric_unit") != "native_temporal_unit":
        errors.append(f"parent_metric_unit_not_native_temporal={path}")
    metrics = record.get("metrics") or {}
    native_metrics = metrics.get("native_temporal_metrics") or {}
    if not native_metrics:
        errors.append(f"parent_missing_native_temporal_metrics={path}")
    parent_gate = check_native_temporal_metrics_gate(
        evaluation_contract=evaluation,
        metrics_payload=metrics,
        paper_facing=bool(record.get("paper_facing")),
        experiment_stage=str(record.get("experiment_stage", "")),
    )
    errors.extend(
        f"parent_native_temporal_metrics_gate={path}:{error}"
        for error in parent_gate["errors"]
    )


def _check_semantic_paper_payloads(
    provenance: dict[str, Any],
    result_kind: str,
    errors: list[str],
) -> None:
    """Verify named downstream gates, not merely that their JSON files exist."""

    if result_kind not in {"model_evaluation", "baseline_evaluation", "ablation_evaluation"}:
        return
    run_payload = _payload(provenance.get("run_audit_json"))
    if run_payload and (
        run_payload.get("run_mode") != "full" or run_payload.get("paper_facing_result") is not True
    ):
        errors.append("run_audit_not_full_paper_facing")
    if result_kind == "model_evaluation":
        calibration = _payload(provenance.get("calibration_audit_json"))
        comparison = _payload(provenance.get("confusion_comparison_json"))
        source_report = _payload(provenance.get("source_balanced_metrics_json"))
        if calibration and calibration.get("complete_oof_fold_coverage") is not True:
            errors.append("calibration_incomplete_oof_fold_coverage")
        if comparison and comparison.get("paper_facing_ready") is not True:
            errors.append("confusion_comparison_not_paper_facing_ready")
        if source_report and source_report.get("paper_facing_ready") is not True:
            errors.append("source_balanced_report_not_paper_facing_ready")


def _payload(artifact: dict[str, Any] | None) -> dict[str, Any]:
    """Read a registered JSON artifact for semantic paper-gate checks."""

    if not artifact:
        return {}
    path = Path(str(artifact.get("path", "")))
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _check_json_payload(artifact: dict[str, Any], name: str, errors: list[str]) -> None:
    """Open small JSON audit payloads and surface their recorded failures."""

    path = Path(str(artifact.get("path", "")))
    if not path.exists():
        errors.append(f"missing_json_payload={name}:{path}")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("errors"):
        errors.append(f"{name}_errors={payload.get('errors')}")
    if payload.get("valid") is False:
        errors.append(f"{name}_invalid")
