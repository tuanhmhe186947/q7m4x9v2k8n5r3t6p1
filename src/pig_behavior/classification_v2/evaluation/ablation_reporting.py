"""Ablation and shortcut-aware reporting for classification_v2.

The report is deliberately conservative: only records with native-temporal
metrics are treated as comparable paper-facing evidence. Smoke-only branches are
listed as engineering evidence so they cannot be mistaken for full OOF model
results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AblationReportingConfig:
    """Input and output locations for the Q2 ablation reporting gate."""

    contract_json: Path = Path("configs/classification_v2/ablation_shortcut_contract_v1.json")
    output_json: Path = Path("outputs/classification_v2/model_design/ablation_reporting_audit.json")
    primary_metric: str = "macro_f1_supported"
    sesoi: float = 0.02


def build_ablation_reporting_audit(config: AblationReportingConfig) -> dict[str, Any]:
    """Build a paper-safe summary of ablation metrics, deltas, and shortcut risks."""

    errors: list[str] = []
    warnings: list[str] = []
    contract = _read_json(config.contract_json, errors, "contract")
    ladder = list(contract.get("required_ablation_ladder", []))
    records = [_load_ablation_record(item, errors, warnings, config.primary_metric) for item in ladder]
    comparable = [record for record in records if record["native_temporal_metrics_available"]]
    comparable_by_id = {record["id"]: record for record in comparable}
    deltas = _build_metric_deltas(comparable_by_id, config.primary_metric, config.sesoi)
    shortcut_report = _shortcut_report(contract, errors, warnings)

    if "B0" not in comparable_by_id:
        errors.append("missing_comparable_native_majority_baseline_B0")
    if "B1" not in comparable_by_id:
        errors.append("missing_comparable_linear_tabular_B1")
    if "B2" not in comparable_by_id:
        errors.append("missing_comparable_nonlinear_tabular_B2")
    smoke_only_required = [
        record["id"]
        for record in records
        if record["required_before_paper_candidate"] and not record["native_temporal_metrics_available"]
    ]
    if smoke_only_required:
        warnings.append(f"required_ablation_records_smoke_only_not_native_oof={smoke_only_required}")
    if shortcut_report["source_shortcut_risk_level"] == "high":
        warnings.append("source_shortcut_high_risk_requires_source_balanced_reporting")
    if shortcut_report["spatial_shortcut_high_risk_controls"]:
        warnings.append(
            "spatial_shortcut_high_risk_controls="
            f"{sorted(shortcut_report['spatial_shortcut_high_risk_controls'])}"
        )

    audit = {
        "schema_version": "classification_v2_ablation_reporting_audit_v1",
        "contract_json": str(config.contract_json),
        "primary_metric": config.primary_metric,
        "sesoi": float(config.sesoi),
        "comparable_metric_unit": "native_temporal_unit",
        "paper_claim_level": "Q2_strong",
        "external_generalization_claim": False,
        "records": records,
        "native_oof_comparable_ids": [record["id"] for record in comparable],
        "smoke_only_ids": [
            record["id"] for record in records if record["status"] and not record["native_temporal_metrics_available"]
        ],
        "metric_deltas": deltas,
        "shortcut_report": shortcut_report,
        "paper_safe_interpretation": (
            "Only B0/B1/B2 native-temporal OOF metrics are directly comparable. "
            "B3/B5/P-branch evidence remains smoke-only until full OOF learned evaluation is recorded."
        ),
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def _load_ablation_record(
    item: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    primary_metric: str,
) -> dict[str, Any]:
    """Load one declared ablation record and extract paper-comparable metrics if present."""

    evidence_text = str(item.get("evidence", "") or "").strip()
    base = {
        "id": item.get("id"),
        "name": item.get("name"),
        "status": item.get("status"),
        "required_before_paper_candidate": bool(item.get("required_before_paper_candidate")),
        "evidence": evidence_text,
        "evidence_exists": bool(evidence_text and Path(evidence_text).exists()),
        "record_valid": None,
        "git_dirty": None,
        "result_kind": None,
        "native_temporal_metrics_available": False,
        "rows": None,
        primary_metric: None,
        "accuracy": None,
        "macro_recall_supported": None,
        "metric_ci": None,
        "comparability_note": "missing_evidence",
    }
    if not evidence_text:
        base["comparability_note"] = "planned_without_evidence"
        return base
    evidence = Path(evidence_text)
    if not evidence.exists():
        errors.append(f"missing_ablation_record={item.get('id')}:{evidence}")
        return base
    record = _read_json(evidence, errors, f"record_{item.get('id')}")
    base["record_valid"] = _record_is_valid(record)
    base["git_dirty"] = record.get("git_dirty")
    base["result_kind"] = record.get("evaluation_contract", {}).get("result_kind")
    metrics = record.get("metrics", {}).get("native_temporal_metrics")
    if not metrics:
        base["comparability_note"] = "smoke_or_non_native_temporal_record"
        warnings.append(f"ablation_not_native_temporal_comparable={item.get('id')}")
        return base
    base.update(
        {
            "native_temporal_metrics_available": True,
            "rows": metrics.get("rows"),
            primary_metric: metrics.get(primary_metric),
            "accuracy": metrics.get("accuracy"),
            "macro_recall_supported": metrics.get("macro_recall_supported"),
            "metric_ci": record.get("metrics", {}).get("confidence_intervals", {}).get(primary_metric),
            "comparability_note": "native_temporal_oof_comparable",
        }
    )
    return base


def _build_metric_deltas(
    comparable_by_id: dict[str, dict[str, Any]],
    primary_metric: str,
    sesoi: float,
) -> list[dict[str, Any]]:
    """Compute predeclared primary-metric deltas for comparable native OOF records."""

    comparisons = [("B1", "B0"), ("B2", "B0"), ("B2", "B1")]
    rows: list[dict[str, Any]] = []
    for left_id, right_id in comparisons:
        left = comparable_by_id.get(left_id)
        right = comparable_by_id.get(right_id)
        if not left or not right:
            rows.append({"comparison": f"{left_id}_minus_{right_id}", "available": False})
            continue
        delta = float(left[primary_metric]) - float(right[primary_metric])
        rows.append(
            {
                "comparison": f"{left_id}_minus_{right_id}",
                "available": True,
                "left_metric": left[primary_metric],
                "right_metric": right[primary_metric],
                "delta": delta,
                "meets_sesoi": abs(delta) >= float(sesoi),
                "interpretation": _delta_interpretation(delta, sesoi),
            }
        )
    return rows


def _shortcut_report(contract: dict[str, Any], errors: list[str], warnings: list[str]) -> dict[str, Any]:
    """Summarize source and spatial shortcut audits that bound paper claims."""

    shortcuts = contract.get("required_shortcut_audits", {})
    source = _read_json(Path(shortcuts.get("source_shortcut", "")), errors, "source_shortcut")
    spatial = _read_json(Path(shortcuts.get("spatial_control_shortcut", "")), errors, "spatial_control_shortcut")
    source_bal_acc = source.get("balanced_accuracy")
    source_risk = source.get("risk_level") or _risk_from_balanced_accuracy(source_bal_acc)
    spatial_controls = spatial.get("controls", {})
    high_risk_controls = [
        name
        for name, control in spatial_controls.items()
        if (control.get("risk_level") or _risk_from_balanced_accuracy(control.get("balanced_accuracy"))) == "high"
    ]
    if not source:
        warnings.append("missing_source_shortcut_report")
    if not spatial:
        warnings.append("missing_spatial_control_shortcut_report")
    return {
        "source_shortcut_balanced_accuracy": source_bal_acc,
        "source_shortcut_risk_level": source_risk,
        "spatial_control_shortcuts": spatial_controls,
        "spatial_shortcut_high_risk_controls": high_risk_controls,
        "claim_guardrail": (
            "Shortcut risks are high; report source-balanced metrics and ablation deltas "
            "before any learned-model claim."
        ),
    }


def _record_is_valid(record: dict[str, Any]) -> bool:
    """Return whether a registry record is clean enough to cite as evidence."""

    if not record:
        return False
    gate = record.get("native_temporal_metrics_gate", {})
    return record.get("git_dirty") is False and not record.get("errors") and gate.get("valid") is not False


def _risk_from_balanced_accuracy(value: Any) -> str | None:
    """Map shortcut balanced accuracy to a conservative risk level."""

    if value is None:
        return None
    score = float(value)
    if score >= 0.8:
        return "high"
    if score >= 0.65:
        return "medium"
    return "low"


def _read_json(path: Path, errors: list[str], name: str) -> dict[str, Any]:
    """Read a JSON file into a dict and preserve errors in the caller audit."""

    if not str(path) or str(path) == "." or not path.exists():
        errors.append(f"missing_{name}={path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _delta_interpretation(delta: float, sesoi: float) -> str:
    if delta >= sesoi:
        return "meaningful_improvement"
    if delta <= -sesoi:
        return "meaningful_regression"
    return "within_sesoi"
