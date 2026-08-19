import base64
import hashlib
import lightning_sdk
from pathlib import Path

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

local_tar = Path("scratch/runtime_m0_ea392e2d.tar.gz")
with open(local_tar, "rb") as f:
    local_sha = hashlib.sha256(f.read()).hexdigest()

print(f"Uploading {local_tar} ({local_tar.stat().st_size:,} bytes, sha256={local_sha})...")
studio.upload_file(str(local_tar), "/teamspace/studios/this_studio/runtime_m0_ea392e2d.tar.gz")
print("Upload complete!")

remote_script = f"""
import os
import tarfile
import hashlib

tar_path = "/teamspace/studios/this_studio/runtime_m0_ea392e2d.tar.gz"
with open(tar_path, "rb") as f:
    remote_sha = hashlib.sha256(f.read()).hexdigest()
print(f"Remote tar SHA256: {{remote_sha}}")
assert remote_sha == "{local_sha}", "SHA mismatch!"

target_dir = "/teamspace/studios/this_studio/runtime_ea392e2d"
os.makedirs(target_dir, exist_ok=True)

print(f"Extracting to {{target_dir}}...")
with tarfile.open(tar_path, "r:gz") as tar:
    tar.extractall(target_dir)

print("Extraction complete! Verifying files:")
print("  M0 config exists:", os.path.exists(os.path.join(target_dir, "configs/classification_v2/m0_full_multimodal_r34_t6_concat.json")))
print("  src/ exists:", os.path.exists(os.path.join(target_dir, "src/pig_behavior")))
print("  verify_m0_contract.py exists:", os.path.exists(os.path.join(target_dir, "scripts/classification_v2/04_baselines_smokes/verify_m0_contract.py")))
"""

b64_code = base64.b64encode(remote_script.encode("utf-8")).decode("ascii")
cmd = f'python3 -c "import base64; exec(base64.b64decode(\'{b64_code}\').decode(\'utf-8\'))"'

print("Executing remote extraction and verification...")
res = studio.run(cmd)
print("=== REMOTE OUTPUT ===")
print(res)
