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
df_t6 = pd.read_csv(manifest_path, low_memory=False)

rel_manifest_path = os.path.join(full_t6_dir, "full_temporal_window_manifest_release.csv")
df_rel = pd.read_csv(rel_manifest_path, low_memory=False)

print("df_t6 length:", len(df_t6))
print("df_rel length:", len(df_rel))

# Merge df_t6 with df_rel on target_id to see native_unit_id, cvat_behavior_run_id, within_run_or_support_ordinal
df_merged = df_t6.merge(df_rel, on="target_id", how="left", suffixes=("_t6", "_rel"))
print("Merged columns:", list(df_merged.columns))

# Inspect validation split
df_val = df_merged[df_merged["split"] == "validation"].copy().reset_index(drop=True)
print("Total Validation targets:", len(df_val))

# For CVAT targets in validation
cvat_val = df_val[df_val["source_type_t6"] == "cvat_tracking_xml"]
print("CVAT validation targets:", len(cvat_val))
print("CVAT df_rel native_unit_id unique:", cvat_val["native_unit_id_rel"].nunique())
print("CVAT df_rel cvat_behavior_run_id unique:", cvat_val["cvat_behavior_run_id"].nunique())

cvat_unit_key = cvat_val["cvat_behavior_run_id"]
cvat_counts = cvat_unit_key.value_counts()
print(f"CVAT targets per cvat_behavior_run_id: min={cvat_counts.min()}, median={cvat_counts.median()}, max={cvat_counts.max()}")
print(f"CVAT units with 1 target: {(cvat_counts == 1).sum()}, >1 target: {(cvat_counts > 1).sum()}")

# For Legacy targets in validation
legacy_val = df_val[df_val["source_type_t6"] == "legacy_recovered"]
print("\nLegacy validation targets:", len(legacy_val))
print("Legacy df_t6 native_unit_id unique:", legacy_val["native_unit_id_t6"].nunique())
print("Legacy df_rel native_unit_id unique:", legacy_val["native_unit_id_rel"].nunique())
print("Legacy df_rel matched_support_id unique:", legacy_val["matched_support_id_rel"].nunique())

legacy_unit_key = legacy_val["native_unit_id_t6"]
legacy_counts = legacy_unit_key.value_counts()
print(f"Legacy targets per native burst: min={legacy_counts.min()}, median={legacy_counts.median()}, max={legacy_counts.max()}")
print(f"Legacy units with 1 target: {(legacy_counts == 1).sum()}, >1 target: {(legacy_counts > 1).sum()}")

# Check all 33,287 FULL-T6 rows (Train + Validation)
print("\n=== ACROSS FULL 33,287 FULL-T6 DATASET ===")
cvat_all = df_merged[df_merged["source_type_t6"] == "cvat_tracking_xml"]
print("Total CVAT T6 targets:", len(cvat_all))
print("Total CVAT unique cvat_behavior_run_id:", cvat_all["cvat_behavior_run_id"].nunique())
cvat_all_counts = cvat_all["cvat_behavior_run_id"].value_counts()
print(f"All CVAT targets per run: max={cvat_all_counts.max()}, >1 count={(cvat_all_counts > 1).sum()}")

legacy_all = df_merged[df_merged["source_type_t6"] == "legacy_recovered"]
print("Total Legacy T6 targets:", len(legacy_all))
print("Total Legacy unique native bursts:", legacy_all["native_unit_id_t6"].nunique())
legacy_all_counts = legacy_all["native_unit_id_t6"].value_counts()
print(f"All Legacy targets per burst: max={legacy_all_counts.max()}, >1 count={(legacy_all_counts > 1).sum()}")

# Construct authoritative native_unit_id for every row
def get_authoritative_native_id(row):
    if row["source_type_t6"] == "cvat_tracking_xml":
        return str(row["cvat_behavior_run_id"])
    else:
        return str(row["native_unit_id_t6"])

df_val["auth_native_unit_id"] = df_val.apply(get_authoritative_native_id, axis=1)
print("\nTotal Unique Authoritative Native Units in Validation:", df_val["auth_native_unit_id"].nunique())
val_counts = df_val["auth_native_unit_id"].value_counts()
print(f"Validation targets per authoritative native unit: min={val_counts.min()}, median={val_counts.median()}, max={val_counts.max()}")
print(f"Validation units with 1 target: {(val_counts == 1).sum()}, >1 target: {(val_counts > 1).sum()}")
'''

b64 = base64.b64encode(script_code.encode("utf-8")).decode("ascii")
cmd = f"python3 -c \"import base64; open('/teamspace/studios/this_studio/run_rel_audit.py', 'w').write(base64.b64decode('{b64}').decode('utf-8'))\""
studio.run(cmd)
print("Uploaded run_rel_audit.py.")
out = studio.run("python3 /teamspace/studios/this_studio/run_rel_audit.py")
print("=== REMOTE OUTPUT ===")
print(out)
