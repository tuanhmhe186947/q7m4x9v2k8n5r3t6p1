import base64
import lightning_sdk

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

remote_script = """
import os
import tarfile
import shutil

src_file = None
for f in os.listdir("/teamspace/studios/this_studio"):
    if "runtime_m0_ea392e2d" in f:
        src_file = os.path.join("/teamspace/studios/this_studio", f)
        break

print("Found uploaded archive:", src_file)
assert src_file is not None, "Archive not found!"

target_dir = "/teamspace/studios/this_studio/runtime_ea392e2d"
os.makedirs(target_dir, exist_ok=True)

print(f"Extracting {src_file} into {target_dir}...")
with tarfile.open(src_file, "r:gz") as tar:
    tar.extractall(target_dir)

print("Extracted successfully!")
print("Checking contents of target_dir:")
for d in ["src", "configs", "scripts"]:
    p = os.path.join(target_dir, d)
    print(f"  {d} exists: {os.path.exists(p)}")

m0_cfg_p = os.path.join(target_dir, "configs/classification_v2/m0_full_multimodal_r34_t6_concat.json")
print("M0 config exists:", os.path.exists(m0_cfg_p))
"""

b64_code = base64.b64encode(remote_script.encode("utf-8")).decode("ascii")
cmd = f'python3 -c "import base64; exec(base64.b64decode(\'{b64_code}\').decode(\'utf-8\'))"'

print("Executing extraction on Studio...")
res = studio.run(cmd)
print("=== REMOTE OUTPUT ===")
print(res)
