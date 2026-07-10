"""Ablation and shortcut-control contract for classification_v2.

This checker is intentionally conservative. Shortcut audits may show high source
predictability; that is evidence of a scientific risk, not an error to hide. A
paper-facing learned-model claim only becomes ready when required ablations,
native-temporal metrics, prediction-schema checks, and shortcut controls are all
recorded with explicit evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

READY_STATUSES = {"recorded", "recorded_smoke", "implemented_smoke"}
PLANNED_STATUSES = {"planned_required", "planned_exploratory"}
REQUIRED_ABLATION_IDS = {"B0", "B1", "B2", "B3", "B5", "P1", "P2"}


def check_ablation_shortcut_contract(contract_json: Path) -> dict[str, Any]:
    """Validate shortcut-control and ablation evidence readiness."""

    errors: list[str] = []
    warnings: list[str] = []
    contract = _read_json(contract_json, errors)
    if errors:
        return _audit(contract_json, contract, errors, warnings, paper_ready=False)

    shortcut_report = _check_shortcut_audits(contract.get("required_shortcut_audits", {}), errors, warnings)
    smoke_report = _check_smoke_evidence(contract.get("required_smoke_evidence", {}), errors, warnings)
    ablation_report = _check_ablation_ladder(contract.get("required_ablation_ladder", []), errors, warnings)

    blockers = list(contract.get("paper_candidate_blockers", []))
    planned_required = [
        item["id"]
        for item in ablation_report
        if item["required_before_paper_candidate"] and item["status"] not in READY_STATUSES
    ]
    paper_candidate = bool(contract.get("paper_candidate", False))
    paper_ready = not errors and not blockers and not planned_required
    if paper_candidate and not paper_ready:
        errors.append(f"paper_candidate_true_but_not_ready=planned_required={planned_required}; blockers={blockers}")
    if not paper_candidate:
        warnings.append("paper_candidate_false_ablation_contract_only")
    if planned_required:
        warnings.append(f"planned_required_ablations_not_recorded={planned_required}")
    if blockers:
        warnings.append(f"paper_candidate_blockers={blockers}")

    result = _audit(contract_json, contract, errors, warnings, paper_ready=paper_ready)
    result["shortcut_report"] = shortcut_report
    result["smoke_evidence_report"] = smoke_report
    result["ablation_report"] = ablation_report
    result["planned_required_ablations_not_recorded"] = planned_required
    result["paper_candidate_blockers"] = blockers
    return result


def _read_json(path: Path, errors: list[str]) -> dict[str, Any]:
    """Read a JSON contract while converting file failures into audit errors."""

    if not path.exists():
        errors.append(f"missing_ablation_shortcut_contract={path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"invalid_ablation_shortcut_contract={path}:{exc}")
        return {}


def _check_shortcut_audits(
    audits: dict[str, str],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    """Check source/spatial shortcut audits and summarize shortcut risk."""

    report: dict[str, Any] = {}
    for name, raw_path in sorted(audits.items()):
        path = Path(raw_path)
        entry: dict[str, Any] = {"path": raw_path, "exists": path.exists()}
        if not path.exists():
            errors.append(f"missing_shortcut_audit={name}:{raw_path}")
            report[name] = entry
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        entry["errors"] = payload.get("errors", [])
        if payload.get("errors"):
            errors.append(f"shortcut_audit_errors={name}:{payload.get('errors')}")
        if name == "source_shortcut":
            balanced = float(payload.get("balanced_accuracy", 0.0))
            entry["balanced_accuracy"] = balanced
            entry["risk_level"] = "high" if balanced >= 0.8 else "moderate_or_low"
            if balanced >= 0.8:
                warnings.append(f"source_shortcut_high_balanced_accuracy={balanced:.4f}")
        if name == "spatial_control_shortcut":
            controls = payload.get("controls", {})
            entry["controls"] = {
                key: {
                    "balanced_accuracy": values.get("balanced_accuracy"),
                    "risk_level": "high"
                    if float(values.get("balanced_accuracy", 0.0)) >= 0.8
                    else "moderate_or_low",
                }
                for key, values in controls.items()
            }
            high = {
                key: values["balanced_accuracy"]
                for key, values in entry["controls"].items()
                if float(values.get("balanced_accuracy", 0.0)) >= 0.8
            }
            if high:
                warnings.append(f"spatial_control_shortcut_high_balanced_accuracy={high}")
        report[name] = entry
    return report


def _check_smoke_evidence(
    evidence: dict[str, str],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    """Validate smoke/baseline evidence artifacts used by the ablation ladder."""

    report: dict[str, Any] = {}
    for name, raw_path in sorted(evidence.items()):
        path = Path(raw_path)
        entry: dict[str, Any] = {"path": raw_path, "exists": path.exists()}
        if not path.exists():
            errors.append(f"missing_smoke_evidence={name}:{raw_path}")
            report[name] = entry
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        entry["errors"] = payload.get("errors", [])
        entry["valid"] = payload.get("valid")
        if payload.get("errors"):
            errors.append(f"smoke_evidence_errors={name}:{payload.get('errors')}")
        if name.endswith("prediction_schema") and payload.get("valid") is not True:
            errors.append(f"prediction_schema_evidence_invalid={name}")
        if name == "multimodal_smoke_train":
            entry["train_rows"] = payload.get("train_rows")
            entry["eval_rows"] = payload.get("eval_rows")
            entry["loss_reduction"] = payload.get("loss_reduction")
            warnings.append("multimodal_smoke_train_is_not_full_oof_evaluation")
        report[name] = entry
    return report


def _check_ablation_ladder(
    ladder: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Validate predeclared ablation IDs, statuses, and evidence paths."""

    if not isinstance(ladder, list) or not ladder:
        errors.append("required_ablation_ladder_missing")
        return []
    ids = [str(item.get("id")) for item in ladder]
    missing = sorted(REQUIRED_ABLATION_IDS.difference(ids))
    if missing:
        errors.append(f"missing_required_ablation_ids={missing}")
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        errors.append(f"duplicate_ablation_ids={duplicates}")
    report: list[dict[str, Any]] = []
    for item in ladder:
        status = str(item.get("status", ""))
        if status not in READY_STATUSES.union(PLANNED_STATUSES):
            errors.append(f"unsupported_ablation_status={item.get('id')}:{status}")
        evidence = str(item.get("evidence", ""))
        required = bool(item.get("required_before_paper_candidate", False))
        evidence_exists = bool(evidence and Path(evidence).exists())
        if status in READY_STATUSES and not evidence_exists:
            errors.append(f"ready_ablation_missing_evidence={item.get('id')}:{evidence}")
        if required and status in PLANNED_STATUSES:
            warnings.append(f"required_ablation_planned_not_recorded={item.get('id')}")
        report.append(
            {
                "id": str(item.get("id")),
                "name": str(item.get("name")),
                "status": status,
                "required_before_paper_candidate": required,
                "evidence": evidence,
                "evidence_exists": evidence_exists,
            }
        )
    return report


def _audit(
    contract_json: Path,
    contract: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    *,
    paper_ready: bool,
) -> dict[str, Any]:
    """Return a compact contract audit suitable for experiment provenance."""

    return {
        "schema_version": "classification_v2_ablation_shortcut_contract_audit_v1",
        "contract_json": str(contract_json),
        "contract_version": contract.get("version"),
        "paper_candidate": bool(contract.get("paper_candidate", False)),
        "paper_candidate_ready": bool(paper_ready),
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }
