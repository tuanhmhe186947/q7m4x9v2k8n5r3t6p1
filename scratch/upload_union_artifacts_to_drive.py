import base64
import hashlib
import time
from pathlib import Path
import lightning_sdk
from lightning_sdk import Machine

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

print(f"Current Studio status: {studio.status}")
if studio.status != "Running":
    print("Starting Studio on Machine.CPU...")
    studio.start(machine=Machine.CPU)
    print("Studio is now Running!")

local_dir = Path("outputs/classification_v2/full_t6_union_r128_20260818")
files_to_upload = [
    "full_t6_union_selection.csv",
    "visual_context_manifest.csv",
    "visual_context_cache_audit.json",
    "packed_image_cache_index.csv",
    "packed_image_cache_audit.json",
    "packed_rgb_128_letterbox.npy",
]

remote_dest_dir = "/teamspace/uploads/classification_v2/full_t6_union_r128_20260818"

# 1. Create target directory on remote Studio
mkdir_script = f"""
import os
os.makedirs("{remote_dest_dir}", exist_ok=True)
print("Created remote directory: {remote_dest_dir}")
"""
b64_mkdir = base64.b64encode(mkdir_script.encode("utf-8")).decode("ascii")
studio.run(f'python3 -c "import base64; exec(base64.b64decode(\'{b64_mkdir}\').decode(\'utf-8\'))"')

# 2. Upload each file and move to /teamspace/uploads/
for fname in files_to_upload:
    local_p = local_dir / fname
    assert local_p.exists(), f"File missing: {local_p}"
    sz = local_p.stat().st_size
    print(f"\nUploading {fname} ({sz:,} bytes)...")
    t0 = time.perf_counter()
    studio.upload_file(str(local_p), f"/teamspace/studios/this_studio/{fname}")
    elapsed = time.perf_counter() - t0
    print(f"Uploaded in {elapsed:.1f}s ({sz / (1024*1024) / max(elapsed, 0.001):.1f} MB/s)!")

    # Move from this_studio to destination directory on Drive
    move_script = f"""
import os, shutil
src = None
for f in os.listdir("/teamspace/studios/this_studio"):
    if "{fname}" in f:
        src = os.path.join("/teamspace/studios/this_studio", f)
        break
if src is not None:
    dest = os.path.join("{remote_dest_dir}", "{fname}")
    shutil.move(src, dest)
    print(f"Moved {{src}} -> {{dest}} (size: {{os.path.getsize(dest):,}} B)")
else:
    print(f"ERROR: {fname} not found in this_studio!")
"""
    b64_move = base64.b64encode(move_script.encode("utf-8")).decode("ascii")
    res_move = studio.run(f'python3 -c "import base64; exec(base64.b64decode(\'{b64_move}\').decode(\'utf-8\'))"')
    print(res_move)

# 3. Final verification on remote Drive
verify_script = f"""
import os
print("=== VERIFYING REMOTE DRIVE DIRECTORY {remote_dest_dir} ===")
for f in sorted(os.listdir("{remote_dest_dir}")):
    p = os.path.join("{remote_dest_dir}", f)
    print(f"  {{f}}: {{os.path.getsize(p):,}} bytes")
"""
b64_verify = base64.b64encode(verify_script.encode("utf-8")).decode("ascii")
res_verify = studio.run(f'python3 -c "import base64; exec(base64.b64decode(\'{b64_verify}\').decode(\'utf-8\'))"')
print(res_verify)
