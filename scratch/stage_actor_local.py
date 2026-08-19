import lightning_sdk

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

remote_code = r'''
import os, time, hashlib
from pathlib import Path

src_dir = "/teamspace/uploads/classification_v2/cloud_r128_recovery_20260817_gcp/r128_cache"
dst_dir = "/teamspace/studios/this_studio/m0_actor_r128_local"
os.makedirs(dst_dir, exist_ok=True)

src_npy = os.path.join(src_dir, "packed_rgb_128_letterbox.npy")
src_csv = os.path.join(src_dir, "packed_image_cache_index.csv")

dst_npy = os.path.join(dst_dir, "packed_rgb_128_letterbox.npy")
dst_csv = os.path.join(dst_dir, "packed_image_cache_index.csv")

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(64 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

print("1. Checking source sizes...")
src_npy_size = os.path.getsize(src_npy)
src_csv_size = os.path.getsize(src_csv)
print(f"Source NPY Size: {src_npy_size} bytes")
print(f"Source CSV Size: {src_csv_size} bytes")

print("2. Copying files to local studio SSD...")
t0 = time.perf_counter()
# Copy CSV
with open(src_csv, "rb") as fsrc, open(dst_csv, "wb") as fdst:
    while chunk := fsrc.read(16 * 1024 * 1024):
        fdst.write(chunk)
t_csv = time.perf_counter() - t0

# Copy NPY
t_npy_start = time.perf_counter()
with open(src_npy, "rb") as fsrc, open(dst_npy, "wb") as fdst:
    while chunk := fsrc.read(64 * 1024 * 1024):
        fdst.write(chunk)
t_npy = time.perf_counter() - t_npy_start
t_total = time.perf_counter() - t0

print(f"Copy completed in {t_total:.2f}s (CSV: {t_csv:.2f}s, NPY: {t_npy:.2f}s)")

dst_npy_size = os.path.getsize(dst_npy)
dst_csv_size = os.path.getsize(dst_csv)
print(f"Dest NPY Size: {dst_npy_size} bytes (Match: {dst_npy_size == src_npy_size})")
print(f"Dest CSV Size: {dst_csv_size} bytes (Match: {dst_csv_size == src_csv_size})")

print("3. Computing SHA256 checksums...")
t_sha = time.perf_counter()
src_csv_sha = sha256_file(src_csv)
dst_csv_sha = sha256_file(dst_csv)
print(f"Source CSV SHA256: {src_csv_sha}")
print(f"Dest   CSV SHA256: {dst_csv_sha}")
print(f"CSV SHA Match: {src_csv_sha == dst_csv_sha}")

src_npy_sha = sha256_file(src_npy)
dst_npy_sha = sha256_file(dst_npy)
print(f"Source NPY SHA256: {src_npy_sha}")
print(f"Dest   NPY SHA256: {dst_npy_sha}")
print(f"NPY SHA Match: {src_npy_sha == dst_npy_sha}")
print(f"SHA calculation time: {time.perf_counter() - t_sha:.2f}s")
'''

import base64
b64_code = base64.b64encode(remote_code.encode("utf-8")).decode("ascii")
cmd = f"python3 -c \"import base64; open('/teamspace/studios/this_studio/stage_actor_cache.py', 'w').write(base64.b64decode('{b64_code}').decode('utf-8'))\""
studio.run(cmd)
print("Uploaded stage_actor_cache.py to Studio.")

out = studio.run("python3 /teamspace/studios/this_studio/stage_actor_cache.py")
print("=== STAGING OUTPUT ===")
print(out)
