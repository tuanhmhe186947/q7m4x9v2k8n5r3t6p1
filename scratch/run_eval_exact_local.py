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
from pig_behavior.classification_v2.evaluation.metrics import evaluate_predictions
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

labels = list(VALID_BEHAVIORS)

full_t6_dir = "/teamspace/uploads/classification_v2/full_t6_training_authority_20260817"
df_t6 = pd.read_csv(os.path.join(full_t6_dir, "full_t6_row_manifest.csv"), low_memory=False)
df_rel = pd.read_csv(os.path.join(full_t6_dir, "full_temporal_window_manifest_release.csv"), low_memory=False)
df_merged = df_t6.merge(df_rel, on="target_id", how="left", suffixes=("_t6", "_rel"))

df_val = df_merged[df_merged["split"] == "validation"].copy().reset_index(drop=True)
val_windows = len(df_val)

def get_run_level_id(row):
    if row["source_type_t6"] == "cvat_tracking_xml":
        return str(row["cvat_behavior_run_id"])
    else:
        return str(row["native_unit_id_t6"])

df_val["auth_native_unit_id"] = df_val.apply(get_run_level_id, axis=1)

# Generate non-degenerate deterministic predictions
np.random.seed(20260818)
mock_logits = np.random.randn(val_windows, len(labels))
mock_probs = np.exp(mock_logits) / np.sum(np.exp(mock_logits), axis=1, keepdims=True)
mock_preds = mock_probs.argmax(axis=1)
mock_conf = mock_probs.max(axis=1)

prob_cols = [f"prob_{lbl}" for lbl in labels]

rows = []
for i in range(val_windows):
    row_data = df_val.iloc[i]
    st = row_data["source_type_t6"]
    sgk = row_data.get("video_key_t6", f"group_{i}")
    unit_id = str(row_data["auth_native_unit_id"])
    row_dict = {
        "schema_version": "classification_v2_training_predictions_v2",
        "window_id": str(row_data["target_id"]),
        "temporal_unit_key": unit_id,
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
    }
    for c_idx, col in enumerate(prob_cols):
        row_dict[col] = float(mock_probs[i, c_idx])
    rows.append(row_dict)

df_pred = pd.DataFrame(rows)

# 1. Production Evaluator
native_preds_prod, metrics_prod, audit_prod = build_native_split_evaluation(
    df_pred,
    split="validation",
    min_supported_classes=1,
    label_order=tuple(labels),
)

# 2. Direct Authority Calculation with mean probability vector aggregation
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

print("\n--- Production Metrics Keys ---")
for k, v in metrics_prod.items():
    print(f"  {k}: {v}")

print("\n--- Direct Metrics ---")
print(f"  direct native macro_f1: {metrics_direct['macro_f1']}")
print(f"  direct native macro_f1_supported: {metrics_direct['macro_f1_supported']}")

prod_global_f1 = metrics_prod["validation_native_unit_macro_f1_global"]
prod_supp_f1 = metrics_prod["validation_native_unit_macro_f1_supported"]
dir_global_f1 = metrics_direct["macro_f1"]
dir_supp_f1 = metrics_direct["macro_f1_supported"]

id_parity = (sorted(native_preds_prod["temporal_unit_key"].tolist()) == sorted(df_direct["temporal_unit_key"].tolist()))
merged = native_preds_prod.merge(df_direct, on="temporal_unit_key", suffixes=("_prod", "_direct"))
pred_parity = (merged["native_predicted_behavior_prod"] == merged["native_predicted_behavior_direct"]).all()
f1_parity = (abs(prod_global_f1 - dir_global_f1) < 1e-9) and (abs(prod_supp_f1 - dir_supp_f1) < 1e-9)

print(f"\nPARITY RESULTS:")
print(f"NATIVE_ID_PARITY: {'PASS' if id_parity else 'FAIL'}")
print(f"NATIVE_PREDICTION_PARITY: {'PASS' if pred_parity else 'FAIL'}")
print(f"PRIMARY_NATIVE_MACRO_F1_PARITY: {'PASS' if f1_parity else 'FAIL'}")
'''

b64 = base64.b64encode(script_code.encode("utf-8")).decode("ascii")
cmd = f"python3 -c \"import base64; open('/teamspace/studios/this_studio/run_eval_exact_check.py', 'w').write(base64.b64decode('{b64}').decode('utf-8'))\""
studio.run(cmd)
print("Uploaded run_eval_exact_check.py.")
out = studio.run("python3 /teamspace/studios/this_studio/run_eval_exact_check.py")
print("=== REMOTE OUTPUT ===")
print(out)
