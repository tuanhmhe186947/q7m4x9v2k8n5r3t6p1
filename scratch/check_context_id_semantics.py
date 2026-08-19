import base64
import lightning_sdk

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

remote_script = """
import os
import pandas as pd

r128_dir = "/teamspace/uploads/classification_v2/cloud_r128_recovery_20260817_gcp/r128_cache"
idx_p = os.path.join(r128_dir, "packed_image_cache_index.csv")
df_idx = pd.read_csv(idx_p, low_memory=False)

print("=== Total rows in packed_image_cache_index.csv ===", len(df_idx))
print("Sample rows:")
for i in range(10):
    row = df_idx.iloc[i]
    print(f"Row {i}: {row['packed_row']} | {row['image_context_id']}")

print("\\nSample rows from index 100000:")
for i in range(100000, 100010):
    row = df_idx.iloc[i]
    print(f"Row {i}: {row['packed_row']} | {row['image_context_id']}")

# Check prefixes
prefixes = [x.split('|')[0] if '|' in str(x) else str(x) for x in df_idx['image_context_id']]
df_idx['prefix'] = prefixes
print("\\nPrefix counts:")
print(df_idx['prefix'].value_counts())
"""

b64_code = base64.b64encode(remote_script.encode("utf-8")).decode("ascii")
cmd = f'python3 -c "import base64; exec(base64.b64decode(\'{b64_code}\').decode(\'utf-8\'))"'

print("Executing context ID semantics check with base64 encoding...")
res = studio.run(cmd)
print("=== REMOTE OUTPUT ===")
print(res)
