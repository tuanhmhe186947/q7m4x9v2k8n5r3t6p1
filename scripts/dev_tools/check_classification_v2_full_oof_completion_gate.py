from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.evaluation.metrics_payload_contract import (
    check_paper_metrics_payload,
)
from pig_behavior.classification_v2.evaluation.prediction_schema_contract import (
    check_prediction_schema_csv,
)


def main() -> None:
    """Check full OOF completion evidence without weakening the no-claim gate."""

    parser = argparse.ArgumentParser(
        description="Check classification_v2 full OOF completion gate."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_v2/model_full/full_multimodal_oof"),
    )
    parser.add_argument(
        "--contract-json",
        type=Path,
        default=Path("configs/classification_v2/full_learned_oof_contract_v1.json"),
    )
    parser.add_argument(
        "--preflight-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_multimodal_oof_preflight.json"
        ),
    )
    parser.add_argument(
        "--registry-record-json",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "full_oof_completion_gate_audit.json"
        ),
    )
    args = parser.parse_args()

    audit = build_completion_gate_audit(
        output_dir=args.output_dir,
        contract_json=args.contract_json,
        preflight_json=args.preflight_json,
        registry_record_json=args.registry_record_json,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["errors"]:
        raise SystemExit(1)


def build_completion_gate_audit(
    *,
    output_dir: Path,
    contract_json: Path,
    preflight_json: Path,
    registry_record_json: Path | None,
) -> dict[str, Any]:
    """Summarize whether full OOF evidence is complete enough for Q2 claim."""

    errors: list[str] = []
    contract = _load_json(contract_json, errors, "contract")
    preflight = _load_json(preflight_json, errors, "preflight")
    record_path = registry_record_json or Path(str(contract.get("required_record", "")))
    paths = _expected_paths(output_dir, record_path)
    reports = {name: _path_report(path) for name, path in paths.items()}
    missing = sorted(name for name, report in reports.items() if not report["exists"])

    artifact_blockers: list[str] = [f"missing_artifacts={missing}"] if missing else []
    validation = _validate_complete_artifacts(paths, preflight) if not missing else {}
    validation_blockers = validation.get("blocking_reasons", [])
    blocking_reasons = artifact_blockers + validation_blockers
    completion_ready = not blocking_reasons and not errors
    q2_claim_allowed = bool(completion_ready)
    fail_closed = (completion_ready and q2_claim_allowed) or (
        not completion_ready and not q2_claim_allowed
    )
    if not fail_closed:
        errors.append("completion_gate_not_fail_closed")

    return {
        "schema_version": "classification_v2_full_oof_completion_gate_v1",
        "valid": not errors,
        "errors": errors,
        "completion_ready": completion_ready,
        "q2_claim_allowed": q2_claim_allowed,
        "fail_closed": fail_closed,
        "blocking_reasons": blocking_reasons,
        "missing_artifact_count": len(missing),
        "missing_artifacts": missing,
        "output_dir": str(output_dir),
        "registry_record_json": str(record_path),
        "artifact_reports": reports,
        "validation": validation,
        "preflight_config_sha256": preflight.get("config_sha256"),
        "preflight_git_commit": preflight.get("git_commit"),
        "claim_boundary": (
            "Q2 internal recording-date/video-safe improvement only; "
            "external generalization claim remains false."
        ),
    }


def _validate_complete_artifacts(
    paths: dict[str, Path],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    """Validate full outputs only after every required artifact exists."""

    blocking: list[str] = []
    run_audit = _read_existing_json(paths["run_audit"])
    metrics = _read_existing_json(paths["metrics"])
    source_report = _read_existing_json(paths["source_balanced_report"])
    registry = _read_existing_json(paths["registry_record"])
    schema_file_audit = _read_existing_json(paths["prediction_schema_audit"])
    schema_check = check_prediction_schema_csv(paths["predictions"])
    metrics_check = check_paper_metrics_payload(metrics)

    _check_run_audit(run_audit, preflight, blocking)
    _check_source_report(source_report, blocking)
    _check_registry_record(registry, paths, blocking)
    if schema_check.get("valid") is not True:
        blocking.append(f"prediction_schema_invalid={schema_check.get('errors')}")
    if schema_file_audit.get("valid") is not True:
        blocking.append(
            f"prediction_schema_file_invalid={schema_file_audit.get('errors')}"
        )
    if metrics_check.get("valid") is not True:
        blocking.append(f"metrics_payload_invalid={metrics_check.get('errors')}")

    expected_native_rows = run_audit.get("native_temporal_unit_rows")
    metrics_native_rows = (metrics.get("native_temporal_prediction_audit") or {}).get(
        "native_temporal_unit_rows"
    )
    if expected_native_rows != metrics_native_rows:
        blocking.append(
            "native_temporal_row_count_mismatch="
            f"{expected_native_rows}!={metrics_native_rows}"
        )

    return {
        "blocking_reasons": blocking,
        "run_mode": run_audit.get("run_mode"),
        "paper_facing_result": run_audit.get("paper_facing_result"),
        "full_oof_training_verified": run_audit.get("full_oof_training_verified"),
        "prediction_rows": schema_check.get("prediction_rows"),
        "native_temporal_unit_rows": metrics_native_rows,
        "source_balanced_ready": source_report.get("paper_facing_ready"),
        "source_balanced_valid": source_report.get("valid"),
        "registry_paper_facing": registry.get("paper_facing"),
        "registry_stage": registry.get("experiment_stage"),
        "prediction_schema_valid": schema_check.get("valid"),
        "metrics_payload_valid": metrics_check.get("valid"),
    }


def _check_run_audit(
    audit: dict[str, Any],
    preflight: dict[str, Any],
    blocking: list[str],
) -> None:
    """Require the learned OOF run to match full-run and preflight contracts."""

    if audit.get("errors"):
        blocking.append(f"run_audit_errors={audit.get('errors')}")
    if audit.get("run_mode") != "full":
        blocking.append(f"run_mode_not_full={audit.get('run_mode')}")
    if audit.get("paper_facing_result") is not True:
        blocking.append("run_audit_not_paper_facing_result")
    if audit.get("full_oof_training_verified") is not True:
        blocking.append("full_oof_training_not_verified")
    if audit.get("git_commit") != preflight.get("git_commit"):
        blocking.append(
            "run_preflight_git_commit_mismatch="
            f"{audit.get('git_commit')}!={preflight.get('git_commit')}"
        )
    fold_audits = audit.get("fold_audits") or []
    if not fold_audits:
        blocking.append("run_audit_missing_fold_audits")
    for fold in fold_audits:
        completed = int(fold.get("training_steps_completed", -1))
        expected = int(fold.get("expected_training_steps", -2))
        coverage = float(fold.get("train_row_coverage_ratio", 0.0))
        if completed != expected or coverage < 1.0:
            blocking.append(f"incomplete_fold_training={fold.get('oof_fold_id')}")


def _check_source_report(report: dict[str, Any], blocking: list[str]) -> None:
    """Require source-balanced metrics before any Q2 model claim."""

    if report.get("valid") is not True or report.get("errors"):
        blocking.append(f"source_balanced_report_invalid={report.get('errors')}")
    if report.get("paper_facing_ready") is not True:
        blocking.append("source_balanced_report_not_paper_ready")
    if report.get("complete_oof_fold_coverage") is not True:
        blocking.append("source_balanced_incomplete_oof_fold_coverage")
    if len(report.get("source_labels") or []) < 2:
        blocking.append(f"source_balanced_requires_two_sources={report.get('source_labels')}")
    if int(report.get("matched_native_unit_rows") or 0) <= 0:
        blocking.append("source_balanced_matched_rows_zero")


def _check_registry_record(
    record: dict[str, Any],
    paths: dict[str, Path],
    blocking: list[str],
) -> None:
    """Require a paper-facing registry record binding metrics and run artifacts."""

    if record.get("paper_facing") is not True:
        blocking.append("registry_record_not_paper_facing")
    if record.get("experiment_stage") != "paper_facing_candidate":
        blocking.append(f"registry_stage_invalid={record.get('experiment_stage')}")
    if record.get("git_dirty") is not False:
        blocking.append(f"registry_git_dirty={record.get('git_dirty')}")
    evaluation = record.get("evaluation_contract") or {}
    if evaluation.get("external_generalization_claim") is not False:
        blocking.append("registry_external_generalization_claim_true")
    if evaluation.get("primary_metric_unit") != "native_temporal_unit":
        blocking.append("registry_metric_unit_not_native_temporal_unit")
    provenance = record.get("provenance") or {}
    _check_provenance_path(
        provenance,
        "run_audit_json",
        paths["run_audit"],
        blocking,
    )
    _check_provenance_path(
        provenance,
        "source_balanced_metrics_json",
        paths["source_balanced_report"],
        blocking,
    )


def _check_provenance_path(
    provenance: dict[str, Any],
    key: str,
    expected: Path,
    blocking: list[str],
) -> None:
    value = provenance.get(key) or {}
    if Path(str(value.get("path", ""))) != expected:
        blocking.append(f"registry_provenance_path_mismatch={key}")
    if value.get("exists") is not True:
        blocking.append(f"registry_provenance_missing={key}")


def _expected_paths(output_dir: Path, registry_record: Path) -> dict[str, Path]:
    return {
        "run_audit": output_dir / "full_multimodal_oof_audit.json",
        "predictions": output_dir / "full_multimodal_oof_predictions.csv",
        "unit_predictions": output_dir / "full_multimodal_oof_unit_predictions.csv",
        "metrics": output_dir / "full_multimodal_oof_metrics.json",
        "prediction_schema_audit": (
            output_dir / "full_multimodal_oof_prediction_schema_audit.json"
        ),
        "source_balanced_report": output_dir / "source_balanced_report.json",
        "source_balanced_native_units": (
            output_dir / "source_balanced_native_units.csv"
        ),
        "source_balanced_selection": output_dir / "source_balanced_selection.csv",
        "registry_record": registry_record,
    }


def _path_report(path: Path) -> dict[str, Any]:
    exists = path.exists()
    report: dict[str, Any] = {"path": str(path), "exists": exists}
    if exists:
        report["size_bytes"] = int(path.stat().st_size)
    return report


def _load_json(path: Path, errors: list[str], name: str) -> dict[str, Any]:
    if not path.exists():
        errors.append(f"missing_{name}={path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_existing_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
