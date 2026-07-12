from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pig_behavior.classification_v2.evaluation.metrics_payload_contract import (
    check_paper_metrics_payload,
)
from pig_behavior.classification_v2.evaluation.prediction_schema_contract import (
    check_prediction_schema_csv,
)

REQUIRED_AUDIT_ARTIFACT_KEYS = (
    "metrics_json",
    "predictions_csv",
    "native_unit_predictions_csv",
    "prediction_schema_audit_json",
)
FULL_RUN_REQUIRED_AUDIT_ARTIFACT_KEYS = (
    "source_balanced_report_json",
    "source_balanced_native_units_csv",
    "source_balanced_selection_csv",
)


def main() -> None:
    """Validate learned multimodal OOF pilot/full artifacts without promoting claims."""

    parser = argparse.ArgumentParser(
        description="Check classification_v2 learned multimodal OOF artifacts."
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_smoke/full_multimodal_oof_pilot/"
            "full_multimodal_oof_audit.json"
        ),
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--require-cache-only",
        action="store_true",
        help="Fail unless all sampled images came from the configured disk cache.",
    )
    args = parser.parse_args()
    errors: list[str] = []
    audit = _read_json(args.audit_json, errors, "audit")
    predictions_csv = args.predictions_csv or _path_from_audit(audit, "predictions_csv")
    metrics_json = args.metrics_json or _path_from_audit(audit, "metrics_json")
    metrics = _read_json(metrics_json, errors, "metrics") if metrics_json is not None else {}
    schema_check = (
        check_prediction_schema_csv(predictions_csv)
        if predictions_csv is not None
        else {
            "valid": False,
            "errors": ["missing_predictions_csv"],
        }
    )
    metrics_check = (
        check_paper_metrics_payload(metrics)
        if metrics
        else {"errors": ["missing_metrics"]}
    )
    errors.extend(f"audit:{error}" for error in audit.get("errors", []))
    errors.extend(
        f"prediction_schema:{error}" for error in schema_check.get("errors", [])
    )
    errors.extend(f"metrics_payload:{error}" for error in metrics_check.get("errors", []))
    _check_audit_artifact_paths(audit, args.audit_json, errors)
    _check_row_counts(audit, schema_check, metrics, errors)
    if audit.get("run_mode") == "pilot" and audit.get("paper_facing_result") is True:
        errors.append("pilot_marked_paper_facing")
    if audit.get("run_mode") == "full":
        incomplete_folds = [
            fold.get("oof_fold_id")
            for fold in audit.get("fold_audits", [])
            if int(fold.get("training_steps_completed", -1))
            != int(fold.get("expected_training_steps", -2))
            or float(fold.get("train_row_coverage_ratio", 0.0)) < 1.0
        ]
        if incomplete_folds:
            errors.append(f"incomplete_full_training_coverage={incomplete_folds}")
        if audit.get("paper_facing_result") is not True:
            errors.append("full_run_not_marked_paper_facing")
        if audit.get("git_dirty") is not False:
            errors.append(f"full_run_git_dirty={audit.get('git_dirty')}")
        if audit.get("full_oof_training_verified") is not True:
            errors.append("full_run_training_not_verified")
        if audit.get("source_balanced_report_valid") is not True:
            errors.append("full_run_source_balanced_report_invalid")
        if audit.get("source_balanced_paper_facing_ready") is not True:
            errors.append("full_run_source_balanced_not_paper_ready")
        for metric_name, interval in metrics.get("confidence_intervals", {}).items():
            if interval.get("resample_unit") != "oof_fold_id":
                errors.append(f"full_run_ci_not_fold_clustered={metric_name}")
            if int(interval.get("n_bootstrap", 0)) < 1000:
                errors.append(f"full_run_ci_bootstrap_below_1000={metric_name}")
    audit_config = audit.get("config", {})
    has_weight_contract = "sample_weight_policy" in audit_config
    has_performance_contract = "precision" in audit_config
    weight_policy = str(audit_config.get("sample_weight_policy", ""))
    if (
        has_weight_contract
        and audit.get("run_mode") == "full"
        and weight_policy not in {"event", "event_class"}
    ):
        errors.append(f"full_run_uses_non_event_weight_policy={weight_policy}")
    for fold in audit.get("fold_audits", []):
        fold_id = fold.get("oof_fold_id")
        if has_weight_contract and str(fold.get("sample_weight_policy", "")) != weight_policy:
            errors.append(f"fold_weight_policy_mismatch={fold_id}")
        weight_fields = (
            ("training_weight_min", "training_weight_max", "training_weight_mean")
            if has_weight_contract
            else ()
        )
        for field in weight_fields:
            value = fold.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                errors.append(f"invalid_{field}={fold_id}:{value}")
        if (
            has_weight_contract
            and weight_policy in {"event", "event_class"}
            and int(fold.get("training_zero_weight_rows", -1)) != 0
        ):
            errors.append(
                "event_weight_zero_training_rows="
                f"{fold_id}:{fold.get('training_zero_weight_rows')}"
            )
        if has_weight_contract and weight_policy == "event_class":
            class_weights = fold.get("fold_local_class_weights", {})
            if sorted(class_weights) != sorted(audit.get("label_order", [])):
                errors.append(f"fold_local_class_weight_labels_mismatch={fold_id}")
        if has_performance_contract:
            precision = str(audit_config.get("precision"))
            if str(fold.get("precision", "")) != precision:
                errors.append(f"fold_precision_mismatch={fold_id}")
            if precision == "amp" and fold.get("amp_enabled") is not True:
                errors.append(f"amp_not_enabled={fold_id}")
            if float(fold.get("optimizer_steps_per_sec", 0.0)) <= 0.0:
                errors.append(f"nonpositive_optimizer_throughput={fold_id}")
            if float(fold.get("training_rows_per_sec", 0.0)) <= 0.0:
                errors.append(f"nonpositive_training_row_throughput={fold_id}")
            if str(audit.get("device", "")).startswith("cuda") and float(
                fold.get("cuda_peak_memory_allocated_mb", 0.0)
            ) <= 0.0:
                errors.append(f"missing_cuda_peak_memory={fold_id}")
    image_load_audit = audit.get("image_load_audit", {})
    if args.require_cache_only:
        if image_load_audit.get("cache_manifest_configured") is not True:
            errors.append("image_cache_manifest_not_configured")
        if image_load_audit.get("require_cached_images") is not True:
            errors.append("strict_image_cache_not_enabled")
        if int(image_load_audit.get("disk_image_cache_misses", -1)) != 0:
            errors.append(
                f"disk_image_cache_misses={image_load_audit.get('disk_image_cache_misses')}"
            )
        if int(image_load_audit.get("source_image_loads", -1)) != 0:
            errors.append(
                f"source_image_loads={image_load_audit.get('source_image_loads')}"
            )
        # Packed hits are included in disk_image_cache_hits; summing double-counts.
        total_cache_hits = int(
            image_load_audit.get("packed_image_cache_hits", 0)
            if image_load_audit.get("packed_cache_configured") is True
            else image_load_audit.get("disk_image_cache_hits", 0)
        )
        if total_cache_hits <= 0:
            errors.append("image_cache_hits_not_positive")
    result = {
        "audit_json": str(args.audit_json),
        "predictions_csv": str(predictions_csv) if predictions_csv is not None else None,
        "metrics_json": str(metrics_json) if metrics_json is not None else None,
        "run_mode": audit.get("run_mode"),
        "paper_facing_result": audit.get("paper_facing_result"),
        "prediction_rows": schema_check.get("prediction_rows"),
        "native_temporal_rows": metrics.get(
            "native_temporal_prediction_audit",
            {},
        ).get("native_temporal_unit_rows")
        if metrics
        else None,
        "macro_f1_supported": metrics.get("native_temporal_metrics", {}).get(
            "macro_f1_supported"
        )
        if metrics
        else None,
        "prediction_schema_valid": bool(schema_check.get("valid")),
        "metrics_payload_valid": not metrics_check.get("errors"),
        "cache_only_required": bool(args.require_cache_only),
        "image_load_audit": image_load_audit,
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


def _check_audit_artifact_paths(
    audit: dict,
    audit_json: Path,
    errors: list[str],
) -> None:
    """Require run audits to name every downstream full-OOF artifact path."""

    if not audit:
        return
    required = list(REQUIRED_AUDIT_ARTIFACT_KEYS)
    if audit.get("run_mode") == "full":
        required.extend(FULL_RUN_REQUIRED_AUDIT_ARTIFACT_KEYS)
    for key in required:
        path = _path_from_audit(audit, key)
        if path is None:
            errors.append(f"audit_missing_artifact_path={key}")
            continue
        if not path.exists():
            errors.append(f"audit_artifact_missing={key}:{path}")
    if audit.get("run_mode") == "full":
        expected_parent = Path(
            "outputs/classification_v2/model_full/full_multimodal_oof"
        )
        if audit_json.parent != expected_parent:
            errors.append(f"full_audit_not_in_model_full_dir={audit_json.parent}")


def _check_row_counts(
    audit: dict,
    schema_check: dict,
    metrics: dict,
    errors: list[str],
) -> None:
    """Keep prediction, native-unit metric, and audit row counts aligned."""

    if not audit:
        return
    prediction_rows = schema_check.get("prediction_rows")
    audit_prediction_rows = audit.get("prediction_rows")
    if prediction_rows is not None and audit_prediction_rows is not None:
        if int(prediction_rows) != int(audit_prediction_rows):
            errors.append(
                "prediction_row_count_mismatch="
                f"{prediction_rows}!={audit_prediction_rows}"
            )
    native_metrics = metrics.get("native_temporal_prediction_audit") or {}
    native_rows = native_metrics.get("native_temporal_unit_rows")
    audit_native_rows = audit.get("native_temporal_rows")
    if native_rows is not None and audit_native_rows is not None:
        if int(native_rows) != int(audit_native_rows):
            errors.append(
                "native_temporal_row_count_mismatch="
                f"{native_rows}!={audit_native_rows}"
            )
    if audit.get("run_mode") == "full" and int(native_rows or 0) <= 0:
        errors.append("full_run_native_temporal_rows_zero")


def _read_json(path: Path, errors: list[str], name: str) -> dict:
    """Read JSON evidence while preserving missing-file failures."""

    if not path.exists():
        errors.append(f"missing_{name}={path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _path_from_audit(audit: dict, key: str) -> Path | None:
    """Resolve artifact paths from the audited run instead of a smoke default."""

    value = audit.get(key)
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value))


if __name__ == "__main__":
    main()
