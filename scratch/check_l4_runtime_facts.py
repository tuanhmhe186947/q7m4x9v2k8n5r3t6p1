import base64
import json
import lightning_sdk

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

remote_script = """
import os, sys, subprocess, json
import torch
import torchvision

info = {}
info["cuda_available"] = torch.cuda.is_available()
if torch.cuda.is_available():
    info["gpu_name"] = torch.cuda.get_device_name(0)
    info["device_count"] = torch.cuda.device_count()
    info["total_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    info["total_vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
    info["cuda_version"] = torch.version.cuda
    info["torch_version"] = torch.__version__
    info["torchvision_version"] = torchvision.__version__
    info["arch_list"] = torch.cuda.get_arch_list()

# Check Git SHA in runtime_ea392e2d
runtime_dir = "/teamspace/studios/this_studio/runtime_ea392e2d"
sys.path.insert(0, os.path.join(runtime_dir, "src"))

# Check files
union_dir = "/teamspace/studios/this_studio/full_t6_union_r128_20260818"
actor_dir = "/teamspace/uploads/classification_v2/cloud_r128_recovery_20260817_gcp/r128_cache"
full_t6_dir = "/teamspace/uploads/classification_v2/full_t6_training_authority_20260817"

info["union_cache_exists"] = os.path.exists(os.path.join(union_dir, "packed_rgb_128_letterbox.npy"))
info["union_cache_size"] = os.path.getsize(os.path.join(union_dir, "packed_rgb_128_letterbox.npy")) if info["union_cache_exists"] else 0

info["actor_cache_exists"] = os.path.exists(os.path.join(actor_dir, "packed_rgb_128_letterbox.npy"))
info["actor_cache_size"] = os.path.getsize(os.path.join(actor_dir, "packed_rgb_128_letterbox.npy")) if info["actor_cache_exists"] else 0

info["canon_46d_exists"] = os.path.exists(os.path.join(full_t6_dir, "full_t6_canonical_46d.npz"))
info["canon_46d_size"] = os.path.getsize(os.path.join(full_t6_dir, "full_t6_canonical_46d.npz")) if info["canon_46d_exists"] else 0

info["row_manifest_exists"] = os.path.exists(os.path.join(full_t6_dir, "full_t6_row_manifest.csv"))
info["row_manifest_size"] = os.path.getsize(os.path.join(full_t6_dir, "full_t6_row_manifest.csv")) if info["row_manifest_exists"] else 0

print("=== L4 RUNTIME FACTS ===")
print(json.dumps(info, indent=2))
"""

b64 = base64.b64encode(remote_script.encode("utf-8")).decode("ascii")
res = studio.run(f'python3 -c "import base64; exec(base64.b64decode(\'{b64}\').decode(\'utf-8\'))"')
print(res)
