import os
import sys
import pandas as pd
import numpy as np

from pig_behavior.classification_v2.training.validation_selection import (
    build_native_split_evaluation,
    resolve_source_aware_native_unit_key,
)
from pig_behavior.classification_v2.evaluation.metrics import evaluate_predictions
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

labels = list(VALID_BEHAVIORS)

full_t6_dir = "outputs/classification_v2/full_t6_training_authority_20260817"
df_t6 = pd.read_csv(os.path.join(full_t6_dir, "full_t6_row_manifest.csv"), low_memory=False)
df_rel = pd.read_csv(os.path.join(full_t6_dir, "full_temporal_window_manifest_release.csv"), low_memory=False)
df_merged = df_t6.merge(df_rel, on="target_id", how="left", suffixes=("_t6", "_rel"))

df_val = df_merged[df_merged["split"] == "validation"].copy().reset_index(drop=True)
val_windows = len(df_val)

# Resolve source-aware native unit key for each validation row
df_val["source_type"] = df_val["source_type_t6"]
df_val["window_id"] = df_val["target_id"]
df_val["native_unit_key"] = [resolve_source_aware_native_unit_key(r) for _, r in df_val.iterrows()]

# Check populations
total_native_units = int(df_val["native_unit_key"].nunique())
counts = df_val["native_unit_key"].value_counts()
min_w = int(counts.min())
max_w = int(counts.max())
gt1_w = int((counts > 1).sum())

cvat_val = df_val[df_val["source_type"] == "cvat_tracking_xml"]
cvat_windows = len(cvat_val)
cvat_native_units = int(cvat_val["native_unit_key"].nunique())

legacy_val = df_val[df_val["source_type"] == "legacy_recovered"]
legacy_windows = len(legacy_val)
legacy_native_units = int(legacy_val["native_unit_key"].nunique())

missing_keys = int(df_val["native_unit_key"].isna().sum() + (df_val["native_unit_key"] == "").sum())
dup_window_ids = int(df_val["window_id"].duplicated().sum())

print("=== POPULATION AUDIT ===")
print(f"VALIDATION_WINDOWS = {val_windows}")
print(f"NATIVE_UNITS = {total_native_units}")
print(f"CVAT_WINDOWS = {cvat_windows}")
print(f"CVAT_NATIVE_UNITS = {cvat_native_units}")
print(f"LEGACY_WINDOWS = {legacy_windows}")
print(f"LEGACY_NATIVE_UNITS = {legacy_native_units}")
print(f"SUPPORT_WINDOWS_PER_NATIVE_MIN = {min_w}")
print(f"SUPPORT_WINDOWS_PER_NATIVE_MAX = {max_w}")
print(f"NATIVE_UNITS_WITH_GT1_WINDOW = {gt1_w}")
print(f"MISSING_NATIVE_KEYS = {missing_keys}")
print(f"DUPLICATE_WINDOW_IDS = {dup_window_ids}")

# Non-degenerate Multi-Class Evaluation Test
np.random.seed(20260818)
mock_logits = np.random.randn(val_windows, len(labels))
mock_probs = np.exp(mock_logits) / np.sum(np.exp(mock_logits), axis=1, keepdims=True)
mock_preds = mock_probs.argmax(axis=1)
mock_conf = mock_probs.max(axis=1)

prob_cols = [f"prob_{lbl}" for lbl in labels]

rows = []
for i in range(val_windows):
    row_data = df_val.iloc[i]
    st = str(row_data["source_type"])
    sgk = str(row_data.get("video_key_t6", f"group_{i}"))
    row_dict = {
        "schema_version": "classification_v2_training_predictions_v2",
        "window_id": str(row_data["window_id"]),
        "temporal_unit_key": str(row_data["native_unit_key"]),
        "fold_id": "r128_full_t6",
        "oof_fold_id": "r128_full_t6",
        "split": "validation",
        "source_type": st,
        "split_group_key": sgk,
        "true_label": str(row_data["behavior_t6"]),
        "predicted_label": labels[mock_preds[i]],
        "y_true": str(row_data["behavior_t6"]),
        "y_pred": labels[mock_preds[i]],
        "prediction_split": "validation",
        "confidence": float(mock_conf[i]),
        "model_version": "FullMultimodal-R34-T6-Concat",
        "snapshot_id": "full_t6_20260817",
    }
    for c_idx, col in enumerate(prob_cols):
        row_dict[col] = float(mock_probs[i, c_idx])
    rows.append(row_dict)

df_pred = pd.DataFrame(rows)

# 1. Repaired Production Evaluator
native_preds_prod, metrics_prod, audit_prod = build_native_split_evaluation(
    df_pred,
    split="validation",
    min_supported_classes=10,
    label_order=tuple(labels),
)

# 2. Direct Authoritative 1:1 Calculation
direct_rows = []
for unit_id, grp in df_pred.groupby("temporal_unit_key", sort=True):
    mean_p = grp[prob_cols].mean(axis=0).to_numpy()
    winner_idx = int(np.argmax(mean_p))
    winner_label = labels[winner_idx]
    direct_rows.append({
        "temporal_unit_key": unit_id,
        "true_label": grp["true_label"].iloc[0],
        "native_predicted_behavior": winner_label,
        **{col: mean_p[c_idx] for c_idx, col in enumerate(prob_cols)}
    })
df_direct = pd.DataFrame(direct_rows)
metrics_direct = evaluate_predictions(
    df_direct,
    y_true_col="true_label",
    y_pred_col="native_predicted_behavior",
    label_order=labels,
)

window_macro_f1 = metrics_prod["validation_window_macro_f1"]
primary_native_macro_f1 = metrics_prod["validation_native_unit_macro_f1_supported"]
direct_macro_f1 = metrics_direct["macro_f1_supported"]

id_parity = (sorted(native_preds_prod["temporal_unit_key"].tolist()) == sorted(df_direct["temporal_unit_key"].tolist()))
merged = native_preds_prod.merge(df_direct, on="temporal_unit_key", suffixes=("_prod", "_direct"))
pred_parity = (merged["native_predicted_behavior_prod"] == merged["native_predicted_behavior_direct"]).all()
f1_parity = (abs(primary_native_macro_f1 - direct_macro_f1) < 1e-9)
one_to_one_parity = (abs(window_macro_f1 - primary_native_macro_f1) < 1e-9)

print("\n=== NON-DEGENERATE METRIC TEST ===")
print(f"WINDOW_MACRO_F1 = {window_macro_f1:.6f}")
print(f"PRIMARY_NATIVE_MACRO_F1 = {primary_native_macro_f1:.6f}")
print(f"DIRECT_NATIVE_MACRO_F1 = {direct_macro_f1:.6f}")
print(f"NATIVE_ID_PARITY = {'PASS' if id_parity else 'FAIL'}")
print(f"NATIVE_PREDICTION_PARITY = {'PASS' if pred_parity else 'FAIL'}")
print(f"PRIMARY_NATIVE_MACRO_F1_PARITY = {'PASS' if f1_parity else 'FAIL'}")
print(f"ONE_TO_ONE_METRIC_PARITY = {'PASS' if one_to_one_parity else 'FAIL'}")
