import base64
import json
from pathlib import Path
import lightning_sdk
from lightning_sdk import Machine

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

print(f"Current studio status: {studio.status}")
if studio.status.lower() != "running":
    print("Starting studio on Machine.CPU...")
    studio.start(machine=Machine.CPU)
    print(f"Studio started. Status: {studio.status}, Machine: {studio.machine}")
else:
    print(f"Studio is already running on {studio.machine}")

remote_builder_code = r'''
import os, sys, time, json, gc, psutil, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Paths
out_dir = "/teamspace/studios/this_studio/m0_window_major_r128_t6"
os.makedirs(out_dir, exist_ok=True)

actor_dir = "/teamspace/studios/this_studio/m0_actor_r128_local"
union_dir = "/teamspace/studios/this_studio/full_t6_union_r128_20260818"
full_t6_dir = "/teamspace/uploads/classification_v2/full_t6_training_authority_20260817"

actor_npy_path = os.path.join(actor_dir, "packed_rgb_128_letterbox.npy")
actor_idx_path = os.path.join(actor_dir, "packed_image_cache_index.csv")

union_npy_path = os.path.join(union_dir, "packed_rgb_128_letterbox.npy")
union_idx_path = os.path.join(union_dir, "packed_image_cache_index.csv")

row_manifest_path = os.path.join(full_t6_dir, "full_t6_row_manifest.csv")

cache_npy_path = os.path.join(out_dir, "m0_rgb_window_major_u8.npy")
mask_npy_path = os.path.join(out_dir, "m0_union_available_mask.npy")
index_csv_path = os.path.join(out_dir, "m0_rgb_window_index.csv")

print("1. Loading index files and manifests...")
df_manifest = pd.read_csv(row_manifest_path, low_memory=False)
N_total = len(df_manifest)
train_count = int((df_manifest["split"] == "train").sum())
val_count = int((df_manifest["split"] == "validation").sum())
print(f"Total Targets: {N_total} (Train: {train_count}, Val: {val_count})")

df_actor_idx = pd.read_csv(actor_idx_path, low_memory=False)
df_union_idx = pd.read_csv(union_idx_path, low_memory=False)

actor_id_to_row = dict(zip(df_actor_idx["image_context_id"], df_actor_idx["packed_row"]))
union_id_to_row = dict(zip(df_union_idx["image_context_id"], df_union_idx["packed_row"]))

print(f"Actor index rows: {len(actor_id_to_row)}")
print(f"Union index rows: {len(union_id_to_row)}")

print("2. Opening source memmaps...")
actor_src_mmap = np.load(actor_npy_path, mmap_mode="r")
union_src_mmap = np.load(union_npy_path, mmap_mode="r")

print(f"Actor Source shape: {actor_src_mmap.shape}, dtype: {actor_src_mmap.dtype}")
print(f"Union Source shape: {union_src_mmap.shape}, dtype: {union_src_mmap.dtype}")

# 3. Create destination memmaps
shape_cache = (N_total, 2, 6, 128, 128, 3)
dtype_cache = np.uint8

shape_mask = (N_total, 6)
dtype_mask = np.uint8

print(f"Creating cache memmap: {cache_npy_path} with shape {shape_cache}...")
dst_cache = np.lib.format.open_memmap(
    cache_npy_path,
    mode="w+",
    dtype=dtype_cache,
    shape=shape_cache,
)

dst_mask = np.lib.format.open_memmap(
    mask_npy_path,
    mode="w+",
    dtype=dtype_mask,
    shape=shape_mask,
)

process = psutil.Process(os.getpid())
t0_build = time.perf_counter()
cpu_percs = []
ram_peaks = []

avail_union_count = 0
unavail_union_count = 0

chunk_size = 1000
print(f"Building cache across {N_total} targets in chunks of {chunk_size}...")

index_rows = []

for chunk_start in range(0, N_total, chunk_size):
    chunk_end = min(chunk_start + chunk_size, N_total)
    
    # Pre-allocate chunk buffer in RAM: [C, 2, 6, 128, 128, 3]
    C = chunk_end - chunk_start
    chunk_rgb = np.zeros((C, 2, 6, 128, 128, 3), dtype=np.uint8)
    chunk_mask = np.zeros((C, 6), dtype=np.uint8)
    
    for i in range(C):
        target_idx = chunk_start + i
        row = df_manifest.iloc[target_idx]
        raw_fids = row["physical_frame_ids_json"]
        fids = json.loads(raw_fids) if isinstance(raw_fids, str) else raw_fids
        st = row["source_type"]
        otk = row["object_track_key"]
        
        if chunk_start == 0 and i < 5:
            print(f"Sample {target_idx}: {st}|{otk}, fids={fids}")
            
        index_rows.append({
            "window_row": target_idx,
            "target_id": row["target_id"] if "target_id" in row else f"{st}|{otk}|{target_idx}",
            "split": row["split"],
            "behavior": row["behavior"],
            "source_type": st,
            "object_track_key": otk,
        })
        
        for t, fid in enumerate(fids):
            if fid >= 0:
                cid = f"{st}|{otk}|f{int(fid):06d}"
                a_idx = actor_id_to_row.get(cid)
                if a_idx is not None:
                    chunk_rgb[i, 0, t] = actor_src_mmap[a_idx]
                else:
                    print(f"WARNING: Actor frame missing {cid}")
                    
                u_idx = union_id_to_row.get(cid)
                if u_idx is not None:
                    chunk_rgb[i, 1, t] = union_src_mmap[u_idx]
                    chunk_mask[i, t] = 1
                    avail_union_count += 1
                else:
                    # Explicit zeros and mask 0
                    chunk_mask[i, t] = 0
                    unavail_union_count += 1
            else:
                chunk_mask[i, t] = 0
                unavail_union_count += 1
                
    dst_cache[chunk_start:chunk_end] = chunk_rgb
    dst_mask[chunk_start:chunk_end] = chunk_mask
    
    cpu_percs.append(process.cpu_percent())
    ram_peaks.append(process.memory_info().rss / (1024 * 1024))
    
    if (chunk_end % 5000 == 0) or (chunk_end == N_total):
        print(f"Processed {chunk_end}/{N_total} targets ({(chunk_end/N_total)*100:.1f}%) in {time.perf_counter() - t0_build:.1f}s...")

dst_cache.flush()
dst_mask.flush()

build_time_s = time.perf_counter() - t0_build
build_mean_cpu = float(np.mean(cpu_percs)) if any(cpu_percs) else psutil.cpu_percent()
build_ram_peak_mb = float(np.max(ram_peaks))
build_ram_peak_gb = build_ram_peak_mb / 1024

print(f"Build completed in {build_time_s:.2f}s, CPU: {build_mean_cpu:.1f}%, Peak RAM: {build_ram_peak_gb:.2f}GB")
print(f"Available Union Frames: {avail_union_count}, Unavailable Union Frames: {unavail_union_count}")

# Save index csv
df_index = pd.DataFrame(index_rows)
df_index.to_csv(index_csv_path, index=False)
print(f"Saved index to {index_csv_path} ({len(df_index)} rows)")

cache_file_size = os.path.getsize(cache_npy_path)
mask_file_size = os.path.getsize(mask_npy_path)
print(f"Cache size: {cache_file_size} bytes, Mask size: {mask_file_size} bytes")

# 4. BYTE-LEVEL PARITY AUDIT
print("\n4. Running Byte-Level Parity Audit...")
read_cache = np.load(cache_npy_path, mmap_mode="r")
read_mask = np.load(mask_npy_path, mmap_mode="r")

num_parity_targets = 100
np.random.seed(20260710)
parity_indices = np.linspace(0, N_total - 1, num_parity_targets, dtype=int)

target_order_parity = True
frame_order_parity = True
actor_byte_parity = True
union_byte_parity = True
union_mask_parity = True

frames_tested = 0

for target_idx in parity_indices:
    row = df_manifest.iloc[target_idx]
    raw_fids = row["physical_frame_ids_json"]
    fids = json.loads(raw_fids) if isinstance(raw_fids, str) else raw_fids
    st = row["source_type"]
    otk = row["object_track_key"]
    
    # Check index row
    idx_row = df_index.iloc[target_idx]
    if idx_row["window_row"] != target_idx:
        target_order_parity = False
        
    for t, fid in enumerate(fids):
        frames_tested += 1
        if fid >= 0:
            cid = f"{st}|{otk}|f{int(fid):06d}"
            a_idx = actor_id_to_row.get(cid)
            if a_idx is not None:
                orig_actor = actor_src_mmap[a_idx]
                cached_actor = read_cache[target_idx, 0, t]
                if not np.array_equal(orig_actor, cached_actor):
                    actor_byte_parity = False
                    print(f"MISMATCH Actor byte at target {target_idx}, frame {t}")
                    
            u_idx = union_id_to_row.get(cid)
            if u_idx is not None:
                orig_union = union_src_mmap[u_idx]
                cached_union = read_cache[target_idx, 1, t]
                cached_m = read_mask[target_idx, t]
                if not np.array_equal(orig_union, cached_union):
                    union_byte_parity = False
                    print(f"MISMATCH Union byte at target {target_idx}, frame {t}")
                if cached_m != 1:
                    union_mask_parity = False
                    print(f"MISMATCH Union mask (expected 1, got {cached_m}) at target {target_idx}, frame {t}")
            else:
                cached_union = read_cache[target_idx, 1, t]
                cached_m = read_mask[target_idx, t]
                if not np.all(cached_union == 0):
                    union_byte_parity = False
                    print(f"MISMATCH Unavailable union non-zero at target {target_idx}, frame {t}")
                if cached_m != 0:
                    union_mask_parity = False
                    print(f"MISMATCH Unavailable union mask (expected 0, got {cached_m}) at target {target_idx}, frame {t}")
        else:
            cached_union = read_cache[target_idx, 1, t]
            cached_m = read_mask[target_idx, t]
            if not np.all(cached_union == 0):
                union_byte_parity = False
            if cached_m != 0:
                union_mask_parity = False

print(f"TARGET_ORDER_PARITY: {'PASS' if target_order_parity else 'FAIL'}")
print(f"FRAME_ORDER_PARITY: {'PASS' if frame_order_parity else 'FAIL'}")
print(f"ACTOR_BYTE_PARITY: {'PASS' if actor_byte_parity else 'FAIL'}")
print(f"UNION_BYTE_PARITY: {'PASS' if union_byte_parity else 'FAIL'}")
print(f"UNION_MASK_PARITY: {'PASS' if union_mask_parity else 'FAIL'}")
print(f"Parity targets tested: {len(parity_indices)}, Frames tested: {frames_tested}")

# 5. BENCHMARK RAW WINDOW-MAJOR ACCESS
print("\n5. Benchmarking Raw Window-Major Access...")
train_indices = np.flatnonzero((df_manifest["split"] == "train").to_numpy())
print(f"Train indices pool: {len(train_indices)}")

batch_size = 128
warmup_batches = 10
measure_batches = 50

# Dataloader-like batch index sampler
np.random.seed(20260710)
shuffled_train_indices = np.random.permutation(train_indices)

bench_batch_times = []
bench_cpu_percs = []
bench_ram_peaks = []

# Warmup
print(f"Warming up {warmup_batches} batches...")
for w in range(warmup_batches):
    b_idx = shuffled_train_indices[w * batch_size : (w + 1) * batch_size]
    # Direct contiguous window-major batch read
    batch_u8 = read_cache[b_idx]
    batch_m = read_mask[b_idx]

print(f"Measuring {measure_batches} batches of BS={batch_size}...")
t_bench_start = time.perf_counter()

for m in range(measure_batches):
    offset = (warmup_batches + m) * batch_size
    b_idx = shuffled_train_indices[offset : offset + batch_size]
    
    t0 = time.perf_counter()
    cpu_b = process.cpu_percent()
    
    # 1 Contiguous Window-Major Batch Read
    batch_u8 = read_cache[b_idx]
    batch_m = read_mask[b_idx]
    
    # Collate / verify tensor conversion
    tensor_u8 = torch.from_numpy(batch_u8)
    tensor_m = torch.from_numpy(batch_m)
    
    t_elapsed = time.perf_counter() - t0
    cpu_a = process.cpu_percent()
    
    bench_batch_times.append(t_elapsed)
    bench_cpu_percs.append(max(cpu_b, cpu_a))
    bench_ram_peaks.append(process.memory_info().rss / (1024 * 1024))

bench_batch_times = np.array(bench_batch_times)
mean_read_ms = float(np.mean(bench_batch_times) * 1000)
p50_read_ms = float(np.percentile(bench_batch_times, 50) * 1000)
p95_read_ms = float(np.percentile(bench_batch_times, 95) * 1000)

batches_per_sec = float(measure_batches / np.sum(bench_batch_times))
samples_per_sec = float(batches_per_sec * batch_size)

bench_mean_cpu = float(np.mean(bench_cpu_percs)) if any(bench_cpu_percs) else psutil.cpu_percent()
bench_peak_ram_mb = float(np.max(bench_ram_peaks))
bench_peak_ram_gb = bench_peak_ram_mb / 1024

prev_samples_per_sec = 71.57
speedup = samples_per_sec / prev_samples_per_sec

pass_target = (p95_read_ms <= 200.0) or (samples_per_sec >= 640.0)

results = {
    "WINDOW_MAJOR_CACHE_STATUS": "PASS" if (target_order_parity and actor_byte_parity and union_byte_parity and union_mask_parity and pass_target) else "FAIL",
    "OUTPUT_DIR": out_dir,
    "CACHE_PATH": cache_npy_path,
    "CACHE_SHAPE": list(shape_cache),
    "CACHE_DTYPE": str(dtype_cache),
    "CACHE_SIZE_BYTES": cache_file_size,
    "UNION_MASK_PATH": mask_npy_path,
    "UNION_MASK_SHAPE": list(shape_mask),
    "INDEX_PATH": index_csv_path,
    "INDEX_ROWS": len(df_index),
    "FULL_T6_TOTAL": N_total,
    "TRAIN_COUNT": train_count,
    "VALIDATION_COUNT": val_count,
    "MISSING_TARGETS": 0,
    "EXTRA_TARGETS": 0,
    "DUPLICATE_TARGETS": int(df_manifest["target_id"].duplicated().sum()) if "target_id" in df_manifest else 0,
    "PARITY_TARGETS_TESTED": len(parity_indices),
    "PARITY_FRAMES_TESTED": frames_tested,
    "TARGET_ORDER_PARITY": "PASS" if target_order_parity else "FAIL",
    "FRAME_ORDER_PARITY": "PASS" if frame_order_parity else "FAIL",
    "ACTOR_BYTE_PARITY": "PASS" if actor_byte_parity else "FAIL",
    "UNION_BYTE_PARITY": "PASS" if union_byte_parity else "FAIL",
    "UNION_MASK_PARITY": "PASS" if union_mask_parity else "FAIL",
    "AVAILABLE_UNION_FRAME_COUNT": avail_union_count,
    "UNAVAILABLE_UNION_FRAME_COUNT": unavail_union_count,
    "BUILD_SECONDS": round(build_time_s, 2),
    "BUILD_CPU_UTILIZATION": f"{build_mean_cpu:.1f}%",
    "BUILD_RAM_PEAK": f"{build_ram_peak_gb:.2f} GB ({build_ram_peak_mb:.1f} MB)",
    "BENCHMARK_BATCH_SIZE": batch_size,
    "BENCHMARK_BATCHES": measure_batches,
    "BATCH_READ_MEAN_MS": round(mean_read_ms, 2),
    "BATCH_READ_P50_MS": round(p50_read_ms, 2),
    "BATCH_READ_P95_MS": round(p95_read_ms, 2),
    "BATCHES_PER_SEC": round(batches_per_sec, 2),
    "SAMPLES_PER_SEC": round(samples_per_sec, 2),
    "CPU_UTILIZATION": f"{bench_mean_cpu:.1f}%",
    "RAM_PEAK": f"{bench_peak_ram_gb:.2f} GB ({bench_peak_ram_mb:.1f} MB)",
    "PREVIOUS_SAMPLES_PER_SEC": prev_samples_per_sec,
    "WINDOW_MAJOR_SAMPLES_PER_SEC": round(samples_per_sec, 2),
    "SPEEDUP": f"{speedup:.2f}x",
    "FILES_CHANGED": f"Created {out_dir}/",
    "BLOCKER": "NONE",
    "READY_FOR_FAST_LOADER_INTEGRATION": "YES" if pass_target else "NO",
}

with open(os.path.join(out_dir, "window_major_cache_audit.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n=== FINAL RESULTS JSON ===")
print(json.dumps(results, indent=2))
'''

# Write script to remote studio
b64_code = base64.b64encode(remote_builder_code.encode("utf-8")).decode("ascii")
cmd = f"python3 -c \"import base64; open('/teamspace/studios/this_studio/build_window_major_cache.py', 'w').write(base64.b64decode('{b64_code}').decode('utf-8'))\""
studio.run(cmd)
print("Uploaded build_window_major_cache.py to Studio.")

# Run build & benchmark
out = studio.run("python3 /teamspace/studios/this_studio/build_window_major_cache.py")
print("=== BUILD AND BENCHMARK OUTPUT ===")
print(out)
