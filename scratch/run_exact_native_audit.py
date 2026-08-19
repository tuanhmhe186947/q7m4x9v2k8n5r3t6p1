import base64
import json
import lightning_sdk

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

script_code = r'''
import os, sys, json
sys.path.insert(0, "/teamspace/studios/this_studio/runtime_ea392e2d/src")
import pandas as pd
import numpy as np

full_t6_dir = "/teamspace/uploads/classification_v2/full_t6_training_authority_20260817"
manifest_path = os.path.join(full_t6_dir, "full_t6_row_manifest.csv")
df = pd.read_csv(manifest_path, low_memory=False)

print("Columns in full_t6_row_manifest.csv:", list(df.columns))
df_val = df[df["split"] == "validation"].copy().reset_index(drop=True)
val_windows = len(df_val)
print(f"Total Validation Windows: {val_windows}")

# Native unit ID analysis
native_key = "native_unit_id"
unique_native_units = int(df_val[native_key].nunique())
counts = df_val[native_key].value_counts()
min_w = int(counts.min())
med_w = float(counts.median())
max_w = int(counts.max())
n_1_w = int((counts == 1).sum())
n_gt1_w = int((counts > 1).sum())

print(f"UNIQUE_NATIVE_UNITS: {unique_native_units}")
print(f"WINDOWS_PER_NATIVE_UNIT_MIN: {min_w}")
print(f"WINDOWS_PER_NATIVE_UNIT_MEDIAN: {med_w}")
print(f"WINDOWS_PER_NATIVE_UNIT_MAX: {max_w}")
print(f"NATIVE_UNITS_WITH_1_WINDOW: {n_1_w}")
print(f"NATIVE_UNITS_WITH_GT1_WINDOW: {n_gt1_w}")

# Source breakdown
cvat_val = df_val[df_val["source_type"] == "cvat_tracking_xml"]
cvat_val_windows = len(cvat_val)
cvat_native_units = int(cvat_val[native_key].nunique())

legacy_val = df_val[df_val["source_type"] == "legacy_recovered"]
legacy_val_windows = len(legacy_val)
legacy_native_units = int(legacy_val[native_key].nunique())

print(f"CVAT_VALIDATION_WINDOWS: {cvat_val_windows}")
print(f"CVAT_NATIVE_UNITS: {cvat_native_units}")
print(f"LEGACY_VALIDATION_WINDOWS: {legacy_val_windows}")
print(f"LEGACY_NATIVE_UNITS: {legacy_native_units}")

# Sample native_unit_ids
print("\nSample 5 CVAT native_unit_id values:")
for val in cvat_val[native_key].head(5):
    print(" ", val)
    
print("\nSample 5 Legacy native_unit_id values:")
for val in legacy_val[native_key].head(5):
    print(" ", val)

# Inspect validation_selection.py implementation
from pig_behavior.classification_v2.training.validation_selection import build_native_split_evaluation
from pig_behavior.classification_v2.evaluation.native_temporal_collapse import collapse_window_predictions_to_native_units
from pig_behavior.classification_v2.evaluation.metrics import evaluate_predictions
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

labels = list(VALID_BEHAVIORS)

# Non-degenerate test
np.random.seed(20260818)
mock_logits = np.random.randn(val_windows, len(labels))
mock_probs = np.exp(mock_logits) / np.sum(np.exp(mock_logits), axis=1, keepdims=True)
mock_preds = mock_probs.argmax(axis=1)
mock_conf = mock_probs.max(axis=1)

predictions_rows = []
for i in range(val_windows):
    row_data = df_val.iloc[i]
    st = row_data.get("source_type", "cvat_tracking_xml")
    sgk = row_data.get("video_key", f"group_{i}")
    unit_id = str(row_data[native_key])
    
    pred_row = {
        "schema_version": "classification_v2_training_predictions_v2",
        "window_id": str(row_data["target_id"]),
        "temporal_unit_key": unit_id, # Evaluator uses temporal_unit_key
        "fold_id": "r128_full_t6",
        "oof_fold_id": "r128_full_t6",
        "split": "validation",
        "source_type": str(st),
        "split_group_key": str(sgk),
        "true_label": str(row_data["behavior"]),
        "predicted_label": labels[mock_preds[i]],
        "y_true": str(row_data["behavior"]),
        "y_pred": labels[mock_preds[i]],
        "prediction_split": "validation",
        "confidence": float(mock_conf[i]),
        "model_version": "FullMultimodal-R34-T6-Concat",
        "snapshot_id": "full_t6_20260817",
    }
    for c_idx, lbl in enumerate(labels):
        pred_row[f"prob_{lbl}"] = float(mock_probs[i, c_idx])
    predictions_rows.append(pred_row)

df_pred = pd.DataFrame(predictions_rows)

# 1. Production Evaluator
native_preds_prod, metrics_prod, audit_prod = build_native_split_evaluation(
    df_pred,
    split="validation",
    min_supported_classes=1,
    label_order=tuple(labels),
)

# 2. Direct Authority Grouping
direct_rows = []
for unit_k, grp in df_pred.groupby("temporal_unit_key", sort=True):
    scores = grp.groupby("y_pred", sort=True)["confidence"].sum()
    rank = {label: index for index, label in enumerate(labels)}
    ordered = sorted(
        ((str(lbl), float(score)) for lbl, score in scores.items()),
        key=lambda item: (-item[1], rank.get(item[0], len(rank)), item[0]),
    )
    winner_label = ordered[0][0]
    true_label = grp["true_label"].iloc[0]
    direct_rows.append({
        "temporal_unit_key": unit_k,
        "y_true": true_label,
        "y_pred": winner_label,
    })
df_direct = pd.DataFrame(direct_rows)

eval_dir = evaluate_predictions(df_direct, y_true_col="y_true", y_pred_col="y_pred", label_order=tuple(labels))

print("\n--- Production Metrics ---")
print("Macro-F1 (prod):", metrics_prod.get("primary_native_macro_f1"))
print("Window Macro-F1 (prod):", metrics_prod.get("window_macro_f1"))
print("Native units (prod):", len(native_preds_prod))

print("\n--- Direct Metrics ---")
print("Macro-F1 (direct):", eval_dir.get("macro_f1"))
print("Native units (direct):", len(df_direct))

native_id_parity = (sorted(native_preds_prod["temporal_unit_key"].tolist()) == sorted(df_direct["temporal_unit_key"].tolist()))
merged = native_preds_prod.merge(df_direct, on="temporal_unit_key", suffixes=("_prod", "_direct"))
native_pred_parity = (merged["y_pred_prod"] == merged["y_pred_direct"]).all()
prod_f1 = metrics_prod.get("primary_native_macro_f1", 0.0)
direct_f1 = eval_dir.get("macro_f1", 0.0)
macro_f1_parity = (abs(prod_f1 - direct_f1) < 1e-6)

print(f"NATIVE_ID_PARITY: {'PASS' if native_id_parity else 'FAIL'}")
print(f"NATIVE_PREDICTION_PARITY: {'PASS' if native_pred_parity else 'FAIL'}")
print(f"PRIMARY_NATIVE_MACRO_F1_PARITY: {'PASS' if macro_f1_parity else 'FAIL'}")

# Trace how native_unit_id was constructed
print("\n=== Lineage of native_unit_id ===")
print("CVAT native_unit_id format:", cvat_val["native_unit_id"].iloc[0])
print("Legacy native_unit_id format:", legacy_val["native_unit_id"].iloc[0])
'''

b64 = base64.b64encode(script_code.encode("utf-8")).decode("ascii")
cmd = f"python3 -c \"import base64; open('/teamspace/studios/this_studio/run_native_audit_clean.py', 'w').write(base64.b64decode('{b64}').decode('utf-8'))\""
studio.run(cmd)
print("Uploaded run_native_audit_clean.py.")
out = studio.run("python3 /teamspace/studios/this_studio/run_native_audit_clean.py")
print("=== REMOTE OUTPUT ===")
print(out)
