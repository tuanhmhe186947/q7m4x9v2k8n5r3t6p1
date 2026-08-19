import lightning_sdk

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

remote_script = """
import os
import json
import pandas as pd

r128_dir = "/teamspace/uploads/classification_v2/cloud_r128_recovery_20260817_gcp/r128_cache"
full_t6_dir = "/teamspace/uploads/classification_v2/full_t6_training_authority_20260817"

idx_p = os.path.join(r128_dir, "packed_image_cache_index.csv")
df_idx = pd.read_csv(idx_p, low_memory=False)

print("=== Sample image_context_id in packed_image_cache_index.csv ===")
print("First 10:")
for x in df_idx["image_context_id"].head(10):
    print(" ", x)

print("\\nChecking if any image_context_id has union/partner/context:")
has_union = df_idx["image_context_id"].str.contains("union|partner|interaction|visual_context", case=False).sum()
print("Matches for union/partner/interaction/visual_context:", has_union)

print("\\n=== Inspecting image_window_context / frame_context files ===")
for root, dirs, files in os.walk("/teamspace"):
    for f in files:
        if "context_manifest" in f or "interaction" in f or "visual" in f:
            p = os.path.join(root, f)
            print(f"{p} ({os.path.getsize(p):,} bytes)")

print("\\n=== Checking Git Repo status on this_studio ===")
os.chdir("/teamspace/studios/this_studio")
print("Cwd:", os.getcwd())
if os.path.exists(".git"):
    import subprocess
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    print("Git HEAD SHA:", git_sha)
    git_status = subprocess.check_output(["git", "status", "-s"]).decode().strip()
    print("Git status:", git_status)
else:
    print("No .git directory in this_studio directly.")
"""

print("Executing union and context semantics inspection...")
res = studio.run(f"python3 -c '{remote_script}'")
print("=== REMOTE OUTPUT ===")
print(res)
