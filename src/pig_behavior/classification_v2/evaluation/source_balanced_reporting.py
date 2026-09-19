"""Source-balanced native-unit reporting without treating overlapping windows as samples."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.evaluation.metrics import (
    DEFAULT_LABEL_ORDER,
    evaluate_predictions,
    evaluate_predictions_by_slice,
)
from pig_behavior.classification_v2.evaluation.native_temporal_metrics import (
    build_native_temporal_predictions,
)


def build_source_balanced_native_report(
    window_predictions: pd.DataFrame,
    window_metadata: pd.DataFrame,
    *,
    expected_fold_count: int | None = None,
    paper_facing_run_verified: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build deterministic source-matched native units and full/source/matched metrics."""

    prediction_required = {
        "window_id",
        "temporal_unit_key",
        "behavior_true",
        "behavior_pred",
        "oof_fold_id",
    }
    metadata_required = {"window_id", "source_type"}
    missing_prediction = sorted(prediction_required - set(window_predictions.columns))
    missing_metadata = sorted(metadata_required - set(window_metadata.columns))
    if missing_prediction or missing_metadata:
        raise ValueError(
            f"source report input contract failed: predictions={missing_prediction}, metadata={missing_metadata}"
        )
    if window_predictions["window_id"].duplicated().any():
        raise ValueError("window predictions contain duplicate window_id")
    if window_metadata["window_id"].duplicated().any():
        raise ValueError("window metadata contain duplicate window_id")

    metadata = window_metadata[["window_id", "source_type"]].copy()
    merged = window_predictions.merge(metadata, on="window_id", how="left", validate="one_to_one")
    missing_source = int(merged["source_type"].isna().sum())
    if missing_source:
        raise ValueError(f"prediction windows missing source metadata={missing_source}")
    native_units, native_audit = build_native_temporal_predictions(merged)
    if not native_audit.get("valid"):
        raise ValueError(f"native temporal aggregation failed: {native_audit.get('errors')}")

    unit_metadata = _native_unit_metadata(merged)
    native_units = native_units.merge(unit_metadata, on="temporal_unit_key", how="left", validate="one_to_one")
    selection = _source_matched_selection(native_units)
    matched_ids = set(selection.loc[selection["source_balance_keep"], "temporal_unit_key"])
    matched_units = native_units.loc[native_units["temporal_unit_key"].isin(matched_ids)].copy()

    full_metrics = evaluate_predictions(
        native_units,
        y_true_col="behavior_true",
        y_pred_col="native_predicted_behavior",
        label_order=DEFAULT_LABEL_ORDER,
    )
    by_source = evaluate_predictions_by_slice(
        native_units,
        y_true_col="behavior_true",
        y_pred_col="native_predicted_behavior",
        slice_col="source_type",
        label_order=DEFAULT_LABEL_ORDER,
    )
    matched_metrics = evaluate_predictions(
        matched_units,
        y_true_col="behavior_true",
        y_pred_col="native_predicted_behavior",
        label_order=DEFAULT_LABEL_ORDER,
    ) if not matched_units.empty else {}
    matched_by_source = evaluate_predictions_by_slice(
        matched_units,
        y_true_col="behavior_true",
        y_pred_col="native_predicted_behavior",
        slice_col="source_type",
        label_order=DEFAULT_LABEL_ORDER,
    ) if not matched_units.empty else {}

    fold_count = int(native_units["oof_fold_id"].nunique())
    source_labels = sorted(native_units["source_type"].astype(str).unique())
    errors: list[str] = []
    warnings: list[str] = []
    if len(source_labels) < 2:
        errors.append(f"source_balanced_reporting_requires_two_sources={source_labels}")
    if expected_fold_count is None:
        warnings.append("expected_fold_count_not_declared_complete_oof_coverage_unproven")
    elif fold_count != int(expected_fold_count):
        errors.append(f"oof_fold_count_mismatch=expected:{expected_fold_count},observed:{fold_count}")
    zero_quota_labels = sorted(
        selection.loc[selection["source_balance_quota"].eq(0), "behavior_true"].unique()
    )
    if zero_quota_labels:
        warnings.append(f"labels_missing_one_or_more_sources={zero_quota_labels}")
    if not paper_facing_run_verified:
        warnings.append("full_run_audit_not_verified_source_report_blocked_for_paper")
    complete_folds = expected_fold_count is not None and fold_count == int(expected_fold_count)
    report = {
        "schema_version": "classification_v2_source_balanced_native_report_v1",
        "statistical_unit": "native_temporal_unit",
        "selection_policy": "deterministic_min_source_quota_within_behavior",
        "native_unit_rows": int(len(native_units)),
        "matched_native_unit_rows": int(len(matched_units)),
        "excluded_native_unit_rows": int(len(native_units) - len(matched_units)),
        "source_labels": source_labels,
        "oof_fold_count": fold_count,
        "expected_fold_count": expected_fold_count,
        "complete_oof_fold_coverage": bool(complete_folds),
        "paper_facing_run_verified": bool(paper_facing_run_verified),
        "full_metrics": full_metrics,
        "full_metrics_by_source": by_source,
        "source_matched_metrics": matched_metrics,
        "source_matched_metrics_by_source": matched_by_source,
        "selection_reason_counts": selection["source_balance_reason"].value_counts().to_dict(),
        "source_label_counts_before": _source_label_counts(selection, keep_only=False),
        "source_label_counts_matched": _source_label_counts(selection, keep_only=True),
        "selection_unit_ids_sha256": _ids_hash(selection["temporal_unit_key"]),
        "paper_facing_ready": bool(
            complete_folds
            and paper_facing_run_verified
            and len(source_labels) >= 2
            and not zero_quota_labels
            and not errors
        ),
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }
    return native_units, selection, report


def _native_unit_metadata(merged: pd.DataFrame) -> pd.DataFrame:
    """Require source, label, and fold to be constant inside each native temporal unit."""

    rows: list[dict[str, Any]] = []
    conflicts: list[str] = []
    for unit_id, group in merged.groupby("temporal_unit_key", sort=True):
        source_values = sorted(group["source_type"].dropna().astype(str).unique())
        label_values = sorted(group["behavior_true"].dropna().astype(str).unique())
        fold_values = sorted(group["oof_fold_id"].dropna().astype(str).unique())
        if len(source_values) != 1 or len(label_values) != 1 or len(fold_values) != 1:
            conflicts.append(str(unit_id))
            continue
        rows.append(
            {
                "temporal_unit_key": str(unit_id),
                "source_type": source_values[0],
                "behavior_true_metadata": label_values[0],
                "oof_fold_id_metadata": fold_values[0],
            }
        )
    if conflicts:
        raise ValueError(f"native unit source/label/fold conflicts={len(conflicts)}, examples={conflicts[:10]}")
    return pd.DataFrame(rows)


def _source_matched_selection(native_units: pd.DataFrame) -> pd.DataFrame:
    """Retain every unit and mark a deterministic equal-source quota within each label."""

    selection = native_units[
        ["temporal_unit_key", "oof_fold_id", "source_type", "behavior_true"]
    ].copy()
    source_labels = sorted(selection["source_type"].astype(str).unique())
    counts = selection.groupby(["behavior_true", "source_type"])["temporal_unit_key"].count()
    quotas: dict[str, int] = {}
    for label in sorted(selection["behavior_true"].unique()):
        values = [int(counts.get((label, source), 0)) for source in source_labels]
        quotas[str(label)] = (
            min(values) if len(source_labels) >= 2 and values and all(value > 0 for value in values) else 0
        )
    selection = selection.sort_values(
        ["behavior_true", "source_type", "temporal_unit_key"], kind="mergesort"
    ).reset_index(drop=True)
    selection["source_balance_rank"] = (
        selection.groupby(["behavior_true", "source_type"]).cumcount() + 1
    )
    selection["source_balance_quota"] = selection["behavior_true"].map(quotas).astype(int)
    selection["source_balance_keep"] = selection["source_balance_rank"].le(
        selection["source_balance_quota"]
    ) & selection["source_balance_quota"].gt(0)
    selection["source_balance_reason"] = "above_source_matched_native_unit_quota"
    selection.loc[selection["source_balance_keep"], "source_balance_reason"] = "source_matched_keep"
    selection.loc[
        selection["source_balance_quota"].eq(0), "source_balance_reason"
    ] = "label_missing_one_or_more_sources"
    return selection.sort_values(["oof_fold_id", "temporal_unit_key"], kind="mergesort").reset_index(drop=True)


def _source_label_counts(selection: pd.DataFrame, *, keep_only: bool) -> dict[str, int]:
    """Serialize source×label counts before or after deterministic matching."""

    frame = selection.loc[selection["source_balance_keep"]] if keep_only else selection
    counts = frame.groupby(["source_type", "behavior_true"])["temporal_unit_key"].count()
    return {f"{source}|{label}": int(count) for (source, label), count in counts.items()}


def _ids_hash(values: pd.Series) -> str:
    """Hash the complete selection identity set for reproducibility."""

    return hashlib.sha256("\n".join(sorted(values.astype(str))).encode("utf-8")).hexdigest()
