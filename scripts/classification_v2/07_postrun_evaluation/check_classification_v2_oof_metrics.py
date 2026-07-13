"""Check Q2 native metrics and statistics on a deterministic complete synthetic OOF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.evaluation.native_unit_metrics import (
    evaluate_native_oof,
)
from pig_behavior.classification_v2.evaluation.statistics import (
    holm_adjust,
    paired_cluster_bootstrap,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check Q2 OOF metric contracts without model training."
    )
    parser.add_argument(
        "--fold-assignments",
        type=Path,
        default=Path("outputs/classification_v2/q2_grouped_folds/q2_outer_fold_assignments.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/q2_grouped_folds/"
            "check_q2_oof_metrics.json"
        ),
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    assignments = (
        _synthetic_assignments()
        if args.synthetic
        else pd.read_csv(args.fold_assignments, low_memory=False)
    )
    valid = assignments.loc[_to_bool(assignments["native_unit_valid_for_main_eval"])].copy()
    windows = _perfect_window_predictions(valid)
    units, audit = evaluate_native_oof(windows, assignments)
    evaluated = units.loc[_to_bool(units["native_metric_include"])].copy()
    baseline = evaluated.copy()
    labels = list(VALID_BEHAVIORS)
    baseline["native_predicted_behavior"] = baseline["true_label"].map(
        {label: labels[(index + 1) % len(labels)] for index, label in enumerate(labels)}
    )
    bootstrap = paired_cluster_bootstrap(
        evaluated,
        baseline,
        iterations=100,
        seed=123,
    )
    holm = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    rejection_audit = _rejection_audit(
        assignments,
        windows,
        evaluated,
    )
    errors = list(audit["errors"])
    supported_labels = audit["pooled_metrics"]["supported_label_count"]
    expected_fixed_macro_f1 = supported_labels / len(VALID_BEHAVIORS)
    if not np.isclose(
        audit["pooled_metrics"]["macro_f1"],
        expected_fixed_macro_f1,
    ):
        errors.append("perfect_oof_fixed_macro_f1_mismatch")
    if bootstrap["macro_f1_delta"] <= 0.0:
        errors.append("paired_bootstrap_effect_not_positive")
    if holm != {"a": 0.03, "b": 0.06, "c": 0.06}:
        errors.append(f"holm_contract_mismatch={holm}")
    if not all(rejection_audit.values()):
        errors.append(f"fail_closed_rejections={rejection_audit}")
    result = {
        "schema_version": "classification_v2_q2_oof_metrics_check_v2",
        "native_unit_rows": int(len(units)),
        "complete_oof_valid_units": audit["predicted_valid_native_units"],
        "pooled_macro_f1": audit["pooled_metrics"]["macro_f1"],
        "pooled_macro_f1_supported": audit["pooled_metrics"][
            "macro_f1_supported"
        ],
        "balanced_accuracy": audit["pooled_metrics"]["balanced_accuracy"],
        "multiclass_mcc": audit["pooled_metrics"]["multiclass_mcc"],
        "negative_log_likelihood": audit["pooled_metrics"]["negative_log_likelihood"],
        "multiclass_brier": audit["pooled_metrics"]["multiclass_brier"],
        "top_label_ece": audit["pooled_metrics"]["top_label_ece"],
        "paired_bootstrap": bootstrap,
        "holm_adjusted": holm,
        "fail_closed_rejections": rejection_audit,
        "synthetic_fixture": bool(args.synthetic),
        "full_dataset_read": not bool(args.synthetic),
        "optimizer_steps": 0,
        "errors": errors,
        "valid": not errors,
    }
    require_output_paths_available(
        [args.output_json],
        overwrite=args.overwrite,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


def _rejection_audit(
    assignments: pd.DataFrame,
    windows: pd.DataFrame,
    evaluated: pd.DataFrame,
) -> dict[str, bool]:
    """Prove missing units, fold drift, and outer tuning fail closed."""

    first_unit = str(assignments.iloc[0]["temporal_unit_key"])
    missing_windows = windows.loc[
        ~windows["temporal_unit_key"].astype(str).eq(first_unit)
    ].copy()
    _, missing_audit = evaluate_native_oof(missing_windows, assignments)

    baseline = evaluated.copy()
    first_cluster = str(baseline.iloc[0]["recording_group_id"])
    baseline.loc[
        baseline["recording_group_id"].astype(str).eq(first_cluster),
        "outer_fold_id",
    ] = "different-fold"
    fold_drift_rejected = False
    try:
        paired_cluster_bootstrap(evaluated, baseline, iterations=10)
    except ValueError:
        fold_drift_rejected = True

    outer_tuning_rejected = False
    try:
        paired_cluster_bootstrap(
            evaluated,
            evaluated.copy(),
            iterations=10,
            outer_predictions_used_for_model_selection=True,
        )
    except ValueError:
        outer_tuning_rejected = True
    return {
        "missing_native_unit_rejected": not missing_audit["valid"],
        "paired_fold_drift_rejected": fold_drift_rejected,
        "outer_model_selection_rejected": outer_tuning_rejected,
    }


def _synthetic_assignments() -> pd.DataFrame:
    """Build two recording-safe folds without reading project artifacts."""

    labels = ["drink", "eat", "fight", "social-nose"]
    return pd.DataFrame(
        {
            "temporal_unit_key": [f"synthetic-unit-{index}" for index in range(4)],
            "recording_group_id": ["recording-a"] * 2 + ["recording-b"] * 2,
            "outer_fold_id": ["fold-a"] * 2 + ["fold-b"] * 2,
            "behavior_label": labels,
            "source_type": ["legacy_recovered"] * 2
            + ["cvat_tracking_xml"] * 2,
            "video_key": ["video-a"] * 2 + ["video-b"] * 2,
            "native_unit_valid_for_main_eval": [True] * 4,
        }
    )


def _perfect_window_predictions(valid: pd.DataFrame) -> pd.DataFrame:
    label_to_index = {label: index for index, label in enumerate(VALID_BEHAVIORS)}
    rows = []
    for row in valid.itertuples(index=False):
        item = {
            "window_id": f"metric_smoke|{row.temporal_unit_key}",
            "temporal_unit_key": row.temporal_unit_key,
            "oof_fold_id": row.outer_fold_id,
            "true_label": row.behavior_label,
            "predicted_label": row.behavior_label,
        }
        probabilities = np.full(
            len(VALID_BEHAVIORS),
            0.01 / (len(VALID_BEHAVIORS) - 1),
        )
        probabilities[label_to_index[row.behavior_label]] = 0.99
        item.update(
            {
                f"prob_{label}": float(probabilities[index])
                for index, label in enumerate(VALID_BEHAVIORS)
            }
        )
        rows.append(item)
    return pd.DataFrame(rows)


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


if __name__ == "__main__":
    main()
