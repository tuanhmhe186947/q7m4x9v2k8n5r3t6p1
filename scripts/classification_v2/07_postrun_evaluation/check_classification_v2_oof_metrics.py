"""Check Q2 native metrics and statistics on a deterministic complete synthetic OOF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.evaluation.native_unit_metrics import evaluate_native_oof
from pig_behavior.classification_v2.evaluation.statistics import holm_adjust, paired_cluster_bootstrap
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Q2 OOF metric contracts without model training.")
    parser.add_argument(
        "--fold-assignments",
        type=Path,
        default=Path("outputs/classification_v2/q2_grouped_folds/q2_outer_fold_assignments.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/q2_grouped_folds/check_q2_oof_metrics.json"),
    )
    args = parser.parse_args()
    assignments = pd.read_csv(args.fold_assignments, low_memory=False)
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
    errors = list(audit["errors"])
    if audit["pooled_metrics"]["macro_f1"] != 1.0:
        errors.append("perfect_oof_macro_f1_not_one")
    if bootstrap["macro_f1_delta"] <= 0.0:
        errors.append("paired_bootstrap_effect_not_positive")
    if holm != {"a": 0.03, "b": 0.06, "c": 0.06}:
        errors.append(f"holm_contract_mismatch={holm}")
    result = {
        "schema_version": "classification_v2_q2_oof_metrics_check_v1",
        "native_unit_rows": int(len(units)),
        "complete_oof_valid_units": audit["predicted_valid_native_units"],
        "pooled_macro_f1": audit["pooled_metrics"]["macro_f1"],
        "balanced_accuracy": audit["pooled_metrics"]["balanced_accuracy"],
        "multiclass_mcc": audit["pooled_metrics"]["multiclass_mcc"],
        "negative_log_likelihood": audit["pooled_metrics"]["negative_log_likelihood"],
        "multiclass_brier": audit["pooled_metrics"]["multiclass_brier"],
        "top_label_ece": audit["pooled_metrics"]["top_label_ece"],
        "paired_bootstrap": bootstrap,
        "holm_adjusted": holm,
        "errors": errors,
        "valid": not errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


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
        probabilities = np.full(len(VALID_BEHAVIORS), 0.01 / (len(VALID_BEHAVIORS) - 1))
        probabilities[label_to_index[row.behavior_label]] = 0.99
        item.update({f"prob_{label}": float(probabilities[index]) for index, label in enumerate(VALID_BEHAVIORS)})
        rows.append(item)
    return pd.DataFrame(rows)


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


if __name__ == "__main__":
    main()
