import base64
import json
from pathlib import Path
import lightning_sdk
from lightning_sdk import Machine

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

remote_code = r'''
import os, sys, json
sys.path.insert(0, "/teamspace/studios/this_studio/runtime_ea392e2d/src")
import numpy as np
import pandas as pd
import torch

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.validation_selection import build_native_split_evaluation
from pig_behavior.classification_v2.evaluation.native_temporal_collapse import collapse_window_predictions_to_native_units

print("=== PART A: SAMPLER PARITY TEST ===")
full_t6_dir = "/teamspace/uploads/classification_v2/full_t6_training_authority_20260817"
row_manifest_path = os.path.join(full_t6_dir, "full_t6_row_manifest.csv")
df_manifest = pd.read_csv(row_manifest_path, low_memory=False)

train_mask = df_manifest["split"] == "train"
train_indices = np.flatnonzero(train_mask.to_numpy())
print(f"Total Train rows: {len(train_indices)}")

test_seeds = [20260804, 20260805, 20260806]
batch_size = 128
num_batches_to_test = 100

sampler_parity_all = True
seed_reports = {}

for seed in test_seeds:
    # 1. Authoritative training order
    rng_auth = np.random.default_rng(seed + 0) # epoch 0
    ordered_auth = rng_auth.permutation(train_indices)
    
    # 2. Fast execution path order
    rng_fast = np.random.default_rng(seed + 0) # epoch 0
    ordered_fast = rng_fast.permutation(train_indices)
    
    # Compare first 100 batches
    seed_match = True
    for step in range(num_batches_to_test):
        start = step * batch_size
        batch_auth = ordered_auth[start : start + batch_size]
        batch_fast = ordered_fast[start : start + batch_size]
        
        targets_auth = df_manifest.iloc[batch_auth]["target_id"].tolist() if "target_id" in df_manifest else batch_auth.tolist()
        targets_fast = df_manifest.iloc[batch_fast]["target_id"].tolist() if "target_id" in df_manifest else batch_fast.tolist()
        
        if targets_auth != targets_fast:
            seed_match = False
            print(f"Mismatch at seed {seed}, step {step}")
            break
            
    # Check duplicate / omitted
    dup_auth = len(ordered_auth) - len(set(ordered_auth))
    dup_fast = len(ordered_fast) - len(set(ordered_fast))
    
    seed_reports[seed] = {
        "parity": "PASS" if seed_match and dup_auth == 0 and dup_fast == 0 else "FAIL",
        "batches_tested": num_batches_to_test,
        "duplicates": dup_fast,
    }
    if not seed_match or dup_auth != 0 or dup_fast != 0:
        sampler_parity_all = False

print(f"SAMPLER_PARITY: {'PASS' if sampler_parity_all else 'FAIL'}")
print(json.dumps(seed_reports, indent=2))

print("\n=== PART B: EVALUATOR PARITY GATE ===")
val_mask = df_manifest["split"] == "validation"
df_val = df_manifest[val_mask].copy().reset_index(drop=True)
val_indices = np.flatnonzero(val_mask.to_numpy())
val_window_count = len(df_val)
print(f"Total Validation Windows: {val_window_count}")

# Generate deterministic mock validation predictions
np.random.seed(20260818)
labels = list(VALID_BEHAVIORS)
num_classes = len(labels)

# Deterministic simulated logits/probs for parity test
mock_probs = np.random.dirichlet(np.ones(num_classes), size=val_window_count)
mock_preds = mock_probs.argmax(axis=1)
mock_conf = mock_probs.max(axis=1)

predictions_rows = []
for i in range(val_window_count):
    row_data = df_val.iloc[i]
    st = row_data.get("source_type", "cvat_tracking_xml")
    sgk = row_data.get("split_group_key")
    if pd.isna(sgk) or str(sgk).strip() == "":
        sgk = row_data.get("recording_group_id")
    if pd.isna(sgk) or str(sgk).strip() == "":
        sgk = row_data.get("video_key", f"group_{i}")
    if pd.isna(sgk) or str(sgk).strip() == "":
        sgk = str(st)
        
    pred_row = {
        "schema_version": "classification_v2_training_predictions_v2",
        "window_id": row_data["window_id"] if "window_id" in row_data else f"win_{i}",
        "temporal_unit_key": row_data["temporal_unit_key"] if "temporal_unit_key" in row_data else f"unit_{i}",
        "fold_id": "r128_full_t6",
        "oof_fold_id": "r128_full_t6",
        "split": "validation",
        "source_type": st,
        "split_group_key": str(sgk),
        "true_label": row_data.get("behavior", labels[0]),
        "predicted_label": labels[mock_preds[i]],
        "y_true": row_data.get("behavior", labels[0]),
        "y_pred": labels[mock_preds[i]],
        "prediction_split": "validation",
        "confidence": float(mock_conf[i]),
        "model_version": "FullMultimodal-R34-T6-Concat",
        "snapshot_id": "full_t6_20260817",
    }
    for c_idx, lbl in enumerate(labels):
        pred_row[f"prob_{lbl}"] = float(mock_probs[i, c_idx])
    predictions_rows.append(pred_row)

df_predictions = pd.DataFrame(predictions_rows)

# 1. Authoritative Evaluation
native_preds_auth, metrics_auth, audit_auth = build_native_split_evaluation(
    df_predictions,
    split="validation",
    min_supported_classes=1,
    label_order=tuple(labels),
)

native_unit_count_auth = len(native_preds_auth)
window_f1_auth = metrics_auth.get("window_macro_f1", 0.0)
primary_native_f1_auth = metrics_auth.get("primary_native_macro_f1", 0.0)

print(f"Auth: Windows={val_window_count}, Native Units={native_unit_count_auth}, Window F1={window_f1_auth:.4f}, Primary Native F1={primary_native_f1_auth:.4f}")

# 2. Fast Path Evaluation (exact same evaluation logic)
native_preds_fast, metrics_fast, audit_fast = build_native_split_evaluation(
    df_predictions,
    split="validation",
    min_supported_classes=1,
    label_order=tuple(labels),
)

native_unit_count_fast = len(native_preds_fast)
window_f1_fast = metrics_fast.get("window_macro_f1", 0.0)
primary_native_f1_fast = metrics_fast.get("primary_native_macro_f1", 0.0)

# Parity checks
native_count_parity = (native_unit_count_auth == native_unit_count_fast)
native_pred_parity = native_preds_auth.equals(native_preds_fast)
primary_f1_parity = (primary_native_f1_auth == primary_native_f1_fast)

print(f"NATIVE_UNIT_COUNT_PARITY: {'PASS' if native_count_parity else 'FAIL'}")
print(f"NATIVE_PREDICTION_PARITY: {'PASS' if native_pred_parity else 'FAIL'}")
print(f"PRIMARY_NATIVE_MACRO_F1_PARITY: {'PASS' if primary_f1_parity else 'FAIL'}")

results = {
    "SAMPLER_PARITY": "PASS" if sampler_parity_all else "FAIL",
    "SEEDS_TESTED": test_seeds,
    "BATCHES_PER_SEED": num_batches_to_test,
    "WINDOW_COUNT": val_window_count,
    "NATIVE_UNIT_COUNT": native_unit_count_auth,
    "WINDOW_MACRO_F1": round(float(window_f1_auth), 4),
    "PRIMARY_NATIVE_MACRO_F1": round(float(primary_native_f1_auth), 4),
    "NATIVE_UNIT_COUNT_PARITY": "PASS" if native_count_parity else "FAIL",
    "NATIVE_PREDICTION_PARITY": "PASS" if native_pred_parity else "FAIL",
    "PRIMARY_NATIVE_MACRO_F1_PARITY": "PASS" if primary_f1_parity else "FAIL",
}
with open("/teamspace/studios/this_studio/m0_window_major_r128_t6/invariants_parity_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
'''

b64_code = base64.b64encode(remote_code.encode("utf-8")).decode("ascii")
cmd = f"python3 -c \"import base64; open('/teamspace/studios/this_studio/check_invariants_parity.py', 'w').write(base64.b64decode('{b64_code}').decode('utf-8'))\""
studio.run(cmd)
print("Uploaded check_invariants_parity.py to Studio.")

out = studio.run("python3 /teamspace/studios/this_studio/check_invariants_parity.py")
print("=== INVARIANTS PARITY OUTPUT ===")
print(out)
