import base64
import lightning_sdk

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

test_script = """
import os, json, sys
import pandas as pd

full_t6_dir = "/teamspace/uploads/classification_v2/full_t6_training_authority_20260817"
row_man_p = os.path.join(full_t6_dir, "full_t6_row_manifest.csv")
df_row = pd.read_csv(row_man_p, nrows=5)
print("Columns:", df_row.columns.tolist())
print("Sample row:")
for k, v in df_row.iloc[0].items():
    print(f"  {k}: {v}")

# Check union index matching
union_idx_p = "/teamspace/studios/this_studio/full_t6_union_r128_20260818/packed_image_cache_index.csv"
df_u_idx = pd.read_csv(union_idx_p, nrows=5)
print("Union index sample:")
for cid in df_u_idx["image_context_id"].head():
    print("  ", cid)
"""

b64 = base64.b64encode(test_script.encode("utf-8")).decode("ascii")
res = studio.run(f'python3 -c "import base64; exec(base64.b64decode(\'{b64}\').decode(\'utf-8\'))"')
print(res)
