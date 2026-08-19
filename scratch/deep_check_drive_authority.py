import base64
import lightning_sdk

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

remote_script = """
import os
import numpy as np
import pandas as pd

base_path = "/teamspace/uploads/classification_v2/full_t6_training_authority_20260817"
print(f"=== FULL LISTING OF {base_path} ===")
if os.path.exists(base_path):
    for root, dirs, files in os.walk(base_path):
        rel = os.path.relpath(root, base_path)
        print(f"Directory: {rel} (dirs={dirs})")
        for f in files:
            p = os.path.join(root, f)
            sz = os.path.getsize(p)
            extra = ""
            if f.endswith(".npy"):
                try:
                    arr = np.load(p, mmap_mode="r")
                    extra = f", shape={arr.shape}, dtype={arr.dtype}"
                except Exception as e:
                    extra = f", err={e}"
            elif f.endswith(".npz"):
                try:
                    with np.load(p) as npz:
                        extra = f", keys={list(npz.keys())[:5]}... (total keys={len(npz.keys())})"
                except Exception as e:
                    extra = f", err={e}"
            elif f.endswith(".csv"):
                try:
                    df = pd.read_csv(p, nrows=2)
                    extra = f", cols={list(df.columns)[:5]}..."
                except Exception as e:
                    extra = f", err={e}"
            print(f"  - {f}: {sz:,} bytes{extra}")
else:
    print("Base path does not exist!")

print("\\n=== LISTING ALL UNDER /teamspace/uploads/classification_v2/ ===")
root_upload = "/teamspace/uploads/classification_v2"
for root, dirs, files in os.walk(root_upload):
    for f in files:
        p = os.path.join(root, f)
        sz = os.path.getsize(p)
        print(f"  {p} ({sz:,} B)")
"""

b64_code = base64.b64encode(remote_script.encode("utf-8")).decode("ascii")
cmd = f'python3 -c "import base64; exec(base64.b64decode(\'{b64_code}\').decode(\'utf-8\'))"'

print("Inspecting /teamspace/uploads/classification_v2/ recursively...")
res = studio.run(cmd)
print("=== REMOTE OUTPUT ===")
print(res)
