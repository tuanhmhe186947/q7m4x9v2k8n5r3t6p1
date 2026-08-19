import base64
import json
import time
import lightning_sdk
from lightning_sdk import Machine

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

print(f"Current studio status: {studio.status}", flush=True)
while str(studio.status).lower() in ["stopping", "status.stopping"]:
    print("Studio is stopping, waiting 5s...", flush=True)
    time.sleep(5)

if str(studio.status).lower() not in ["running", "status.running"]:
    print("Starting studio on Machine.CPU...", flush=True)
    studio.start(machine=Machine.CPU)
    print(f"Studio started. Status: {studio.status}, Machine: {studio.machine}", flush=True)
else:
    print(f"Studio is already running on {studio.machine}", flush=True)

# Upload the updated validation_selection.py to Studio
val_sel_content = open("src/pig_behavior/classification_v2/training/validation_selection.py", "r", encoding="utf-8").read()
b64_val = base64.b64encode(val_sel_content.encode("utf-8")).decode("ascii")
studio.run(f"python3 -c \"import base64; open('/teamspace/studios/this_studio/runtime_ea392e2d/src/pig_behavior/classification_v2/training/validation_selection.py', 'w').write(base64.b64decode('{b64_val}').decode('utf-8'))\"")
print("Uploaded updated validation_selection.py to Studio.", flush=True)

audit_script = r'''
import os, sys, json
sys.path.insert(0, "/teamspace/studios/this_studio/runtime_ea392e2d/src")
import pandas as pd
import numpy as np

from pig_behavior.classification_v2.training.validation_selection import (
    build_native_split_evaluation,
    resolve_source_aware_native_unit_key,
)
from pig_behavior.classification_v2.evaluation.metrics import evaluate_predictions
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

labels = list(VALID_BEHAVIORS)

full_t6_dir = "/teamspace/uploads/classification_v2/full_t6_training_authority_20260817"
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

print("=== POPULATION AUDIT ===", flush=True)
print(f"VALIDATION_WINDOWS = {val_windows}", flush=True)
print(f"NATIVE_UNITS = {total_native_units}", flush=True)
print(f"CVAT_WINDOWS = {cvat_windows}", flush=True)
print(f"CVAT_NATIVE_UNITS = {cvat_native_units}", flush=True)
print(f"LEGACY_WINDOWS = {legacy_windows}", flush=True)
print(f"LEGACY_NATIVE_UNITS = {legacy_native_units}", flush=True)
print(f"SUPPORT_WINDOWS_PER_NATIVE_MIN = {min_w}", flush=True)
print(f"SUPPORT_WINDOWS_PER_NATIVE_MAX = {max_w}", flush=True)
print(f"NATIVE_UNITS_WITH_GT1_WINDOW = {gt1_w}", flush=True)
print(f"MISSING_NATIVE_KEYS = {missing_keys}", flush=True)
print(f"DUPLICATE_WINDOW_IDS = {dup_window_ids}", flush=True)

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

window_macro_f1 = float(metrics_prod["validation_window_macro_f1"])
primary_native_macro_f1 = float(metrics_prod["validation_native_unit_macro_f1_supported"])
direct_macro_f1 = float(metrics_direct["macro_f1_supported"])

id_parity = (sorted(native_preds_prod["temporal_unit_key"].tolist()) == sorted(df_direct["temporal_unit_key"].tolist()))
merged = native_preds_prod.merge(df_direct, on="temporal_unit_key", suffixes=("_prod", "_direct"))
pred_parity = bool((merged["native_predicted_behavior_prod"] == merged["native_predicted_behavior_direct"]).all())
f1_parity = bool(abs(primary_native_macro_f1 - direct_macro_f1) < 1e-9)
one_to_one_parity = bool(abs(window_macro_f1 - primary_native_macro_f1) < 1e-9)

print("\n=== NON-DEGENERATE METRIC TEST ===", flush=True)
print(f"WINDOW_MACRO_F1 = {window_macro_f1:.6f}", flush=True)
print(f"PRIMARY_NATIVE_MACRO_F1 = {primary_native_macro_f1:.6f}", flush=True)
print(f"DIRECT_NATIVE_MACRO_F1 = {direct_macro_f1:.6f}", flush=True)
print(f"NATIVE_ID_PARITY = {'PASS' if id_parity else 'FAIL'}", flush=True)
print(f"NATIVE_PREDICTION_PARITY = {'PASS' if pred_parity else 'FAIL'}", flush=True)
print(f"PRIMARY_NATIVE_MACRO_F1_PARITY = {'PASS' if f1_parity else 'FAIL'}", flush=True)
print(f"ONE_TO_ONE_METRIC_PARITY = {'PASS' if one_to_one_parity else 'FAIL'}", flush=True)
'''

b64_audit = base64.b64encode(audit_script.encode("utf-8")).decode("ascii")
studio.run(f"python3 -c \"import base64; open('/teamspace/studios/this_studio/run_full_t6_val_audit.py', 'w').write(base64.b64decode('{b64_audit}').decode('utf-8'))\"")
print("Uploaded run_full_t6_val_audit.py.", flush=True)
out = studio.run("python3 /teamspace/studios/this_studio/run_full_t6_val_audit.py")
print("=== REMOTE OUTPUT ===", flush=True)
print(out, flush=True)

print("Stopping studio...", flush=True)
studio.stop()
print("Studio stopped.", flush=True)
