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

from pig_behavior.classification_v2.training.validation_selection import build_native_split_evaluation
from pig_behavior.classification_v2.evaluation.native_temporal_collapse import collapse_window_predictions_to_native_units
from pig_behavior.classification_v2.evaluation.metrics import evaluate_predictions
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

labels = list(VALID_BEHAVIORS)

full_t6_dir = "/teamspace/uploads/classification_v2/full_t6_training_authority_20260817"
df_t6 = pd.read_csv(os.path.join(full_t6_dir, "full_t6_row_manifest.csv"), low_memory=False)
df_rel = pd.read_csv(os.path.join(full_t6_dir, "full_temporal_window_manifest_release.csv"), low_memory=False)
df_merged = df_t6.merge(df_rel, on="target_id", how="left", suffixes=("_t6", "_rel"))

df_val = df_merged[df_merged["split"] == "validation"].copy().reset_index(drop=True)
val_windows = len(df_val)

# Definition A: Continuous Behavior Run (cvat_behavior_run_id for CVAT, native_unit_id for Legacy)
def get_run_level_id(row):
    if row["source_type_t6"] == "cvat_tracking_xml":
        return str(row["cvat_behavior_run_id"])
    else:
        return str(row["native_unit_id_t6"])

df_val["run_level_native_id"] = df_val.apply(get_run_level_id, axis=1)

# Definition B: Anchor Interval (6-frame anchor interval for CVAT, 16-frame burst for Legacy)
# In FULL_NONOVERLAP_VIEW_POOL, each 6-frame window is target_id (ordinal chunk)
df_val["anchor_level_native_id"] = df_val["target_id"]

print("=== NATIVE UNIT AUDIT STATS ===")
print("Validation Windows:", val_windows)
print("Run-Level Unique Native Units:", df_val["run_level_native_id"].nunique())
print("Anchor-Level Unique Native Units:", df_val["anchor_level_native_id"].nunique())

cvat_val = df_val[df_val["source_type_t6"] == "cvat_tracking_xml"]
cvat_run_counts = cvat_val["cvat_behavior_run_id"].value_counts()
print(f"CVAT: targets={len(cvat_val)}, unique runs={cvat_val['cvat_behavior_run_id'].nunique()}, max targets/run={cvat_run_counts.max()}")

legacy_val = df_val[df_val["source_type_t6"] == "legacy_recovered"]
legacy_counts = legacy_val["native_unit_id_t6"].value_counts()
print(f"Legacy: targets={len(legacy_val)}, unique native bursts={legacy_val['native_unit_id_t6'].nunique()}, max targets/burst={legacy_counts.max()}")
print(f"Legacy units with >1 T6 target: {(legacy_counts > 1).sum()}")

# Non-degenerate evaluator test with run-level grouping
np.random.seed(20260818)
mock_logits = np.random.randn(val_windows, len(labels))
mock_probs = np.exp(mock_logits) / np.sum(np.exp(mock_logits), axis=1, keepdims=True)
mock_preds = mock_probs.argmax(axis=1)
mock_conf = mock_probs.max(axis=1)

# Create prediction dataframe
rows = []
for i in range(val_windows):
    row_data = df_val.iloc[i]
    st = row_data["source_type_t6"]
    sgk = row_data.get("video_key_t6", f"group_{i}")
    rows.append({
        "schema_version": "classification_v2_training_predictions_v2",
        "window_id": str(row_data["target_id"]),
        "temporal_unit_key": str(row_data["run_level_native_id"]),
        "fold_id": "r128_full_t6",
        "oof_fold_id": "r128_full_t6",
        "split": "validation",
        "source_type": str(st),
        "split_group_key": str(sgk),
        "true_label": str(row_data["behavior_t6"]),
        "predicted_label": labels[mock_preds[i]],
        "y_true": str(row_data["behavior_t6"]),
        "y_pred": labels[mock_preds[i]],
        "prediction_split": "validation",
        "confidence": float(mock_conf[i]),
        "model_version": "FullMultimodal-R34-T6-Concat",
        "snapshot_id": "full_t6_20260817",
        **{f"prob_{lbl}": float(mock_probs[i, c_idx]) for c_idx, lbl in enumerate(labels)}
    })

df_pred = pd.DataFrame(rows)

# 1. Production Evaluator
native_preds_prod, metrics_prod, audit_prod = build_native_split_evaluation(
    df_pred,
    split="validation",
    min_supported_classes=1,
    label_order=tuple(labels),
)

# 2. Direct Authority Aggregation
direct_rows = []
for unit_id, grp in df_pred.groupby("temporal_unit_key", sort=True):
    scores = grp.groupby("y_pred", sort=True)["confidence"].sum()
    rank = {label: index for index, label in enumerate(labels)}
    ordered = sorted(
        ((str(lbl), float(score)) for lbl, score in scores.items()),
        key=lambda item: (-item[1], rank.get(item[0], len(rank)), item[0]),
    )
    winner = ordered[0][0]
    direct_rows.append({
        "temporal_unit_key": unit_id,
        "y_true": grp["true_label"].iloc[0],
        "y_pred": winner,
    })
df_direct = pd.DataFrame(direct_rows)
eval_direct = evaluate_predictions(df_direct, y_true_col="y_true", y_pred_col="y_pred", label_order=tuple(labels))

print(f"\n--- Non-degenerate Evaluation Comparison ---")
print(f"Production Native Units: {len(native_preds_prod)}")
print(f"Direct Native Units: {len(df_direct)}")
print(f"Production Native Macro-F1: {metrics_prod.get('primary_native_macro_f1', 0.0):.6f}")
print(f"Direct Native Macro-F1: {eval_direct.get('macro_f1', 0.0):.6f}")
print(f"Production Window Macro-F1: {metrics_prod.get('window_macro_f1', 0.0):.6f}")

id_parity = (sorted(native_preds_prod["temporal_unit_key"].tolist()) == sorted(df_direct["temporal_unit_key"].tolist()))
merged = native_preds_prod.merge(df_direct, on="temporal_unit_key", suffixes=("_prod", "_direct"))
pred_parity = (merged["y_pred_prod"] == merged["y_pred_direct"]).all()
f1_parity = (abs(metrics_prod.get("primary_native_macro_f1", 0.0) - eval_direct.get("macro_f1", 0.0)) < 1e-6)

print(f"NATIVE_ID_PARITY: {'PASS' if id_parity else 'FAIL'}")
print(f"NATIVE_PREDICTION_PARITY: {'PASS' if pred_parity else 'FAIL'}")
print(f"PRIMARY_NATIVE_MACRO_F1_PARITY: {'PASS' if f1_parity else 'FAIL'}")
'''

b64 = base64.b64encode(script_code.encode("utf-8")).decode("ascii")
cmd = f"python3 -c \"import base64; open('/teamspace/studios/this_studio/run_evaluator_parity_audit.py', 'w').write(base64.b64decode('{b64}').decode('utf-8'))\""
studio.run(cmd)
print("Uploaded run_evaluator_parity_audit.py.")
out = studio.run("python3 /teamspace/studios/this_studio/run_evaluator_parity_audit.py")
print("=== EVALUATOR PARITY OUTPUT ===")
print(out)
