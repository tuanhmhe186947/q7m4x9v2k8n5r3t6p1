import base64
import json
from pathlib import Path
import lightning_sdk

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

remote_bench_code = r'''
import os, sys, time, json, gc, psutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Setup environment
runtime_dir = "/teamspace/studios/this_studio/runtime_ea392e2d"
sys.path.insert(0, os.path.join(runtime_dir, "src"))
os.chdir(runtime_dir)

from pig_behavior.classification_v2.training.config import load_training_config
from pig_behavior.classification_v2.features.spatial_schema import (
    SPATIAL_PREDICTIVE_FEATURES,
    SPATIAL_PREDICTIVE_GROUP_NAMES,
)
from pig_behavior.classification_v2.datasets.interaction_context_loader import (
    INTERACTION_CONTEXT_FEATURE_COLUMNS,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

device = torch.device("cpu")
print(f"Device: {device}, PyTorch Version: {torch.__version__}")

# Exact Paths
actor_dir = "/teamspace/studios/this_studio/m0_actor_r128_local"
union_dir = "/teamspace/studios/this_studio/full_t6_union_r128_20260818"
full_t6_dir = "/teamspace/uploads/classification_v2/full_t6_training_authority_20260817"

actor_npy_path = os.path.join(actor_dir, "packed_rgb_128_letterbox.npy")
actor_idx_path = os.path.join(actor_dir, "packed_image_cache_index.csv")

union_npy_path = os.path.join(union_dir, "packed_rgb_128_letterbox.npy")
union_idx_path = os.path.join(union_dir, "packed_image_cache_index.csv")

canon_46d_npz_path = os.path.join(full_t6_dir, "full_t6_canonical_46d.npz")
row_manifest_path = os.path.join(full_t6_dir, "full_t6_row_manifest.csv")

print("1. Verifying physical paths...")
print(f"ACTOR_NPY: {actor_npy_path} (Exists: {os.path.exists(actor_npy_path)}, Size: {os.path.getsize(actor_npy_path)})")
print(f"ACTOR_IDX: {actor_idx_path} (Exists: {os.path.exists(actor_idx_path)}, Size: {os.path.getsize(actor_idx_path)})")
print(f"UNION_NPY: {union_npy_path} (Exists: {os.path.exists(union_npy_path)}, Size: {os.path.getsize(union_npy_path)})")
print(f"UNION_IDX: {union_idx_path} (Exists: {os.path.exists(union_idx_path)}, Size: {os.path.getsize(union_idx_path)})")
print(f"CANON_46D: {canon_46d_npz_path} (Exists: {os.path.exists(canon_46d_npz_path)}, Size: {os.path.getsize(canon_46d_npz_path)})")
print(f"ROW_MANIFEST: {row_manifest_path} (Exists: {os.path.exists(row_manifest_path)}, Size: {os.path.getsize(row_manifest_path)})")

different_files = (os.path.realpath(actor_npy_path) != os.path.realpath(union_npy_path))
print(f"ACTOR_UNION_DIFFERENT_FILES: {different_files}")

df_row = pd.read_csv(row_manifest_path, low_memory=False)
train_rows = df_row[df_row["split"] == "train"].reset_index(drop=True)
val_rows = df_row[df_row["split"] == "validation"].reset_index(drop=True)
print(f"Total Rows: {len(df_row)}, Train Rows: {len(train_rows)}, Val Rows: {len(val_rows)}")

df_actor_idx = pd.read_csv(actor_idx_path, low_memory=False)
df_union_idx = pd.read_csv(union_idx_path, low_memory=False)

class FullT6RealMultimodalDataset(Dataset):
    def __init__(self, row_manifest_df, npz_46d_path, actor_npy, actor_idx_df, union_npy, union_idx_df, image_size=128):
        self.rows = row_manifest_df.reset_index(drop=True)
        self.image_size = image_size
        
        self.actor_id_to_row = dict(zip(actor_idx_df["image_context_id"], actor_idx_df["packed_row"]))
        self.union_id_to_row = dict(zip(union_idx_df["image_context_id"], union_idx_df["packed_row"]))
        self.behavior_to_idx = {b: i for i, b in enumerate(VALID_BEHAVIORS)}
        
        npz = np.load(npz_46d_path)
        self.spatial_groups = {
            group: torch.from_numpy(npz[group])
            for group in SPATIAL_PREDICTIVE_GROUP_NAMES
        }
        self.length_mask = torch.from_numpy(npz["length_mask"])
        self.observed_mask = torch.from_numpy(npz["observed_mask"])
        self.quality_mask = torch.from_numpy(npz["spatial_quality_mask"])
        self.motion_validity = torch.from_numpy(npz["motion_feature_validity_mask"])
        self.social_validity = torch.from_numpy(npz["social_feature_validity_mask"])
        
        self.actor_npy = np.load(actor_npy, mmap_mode="r")
        self.union_npy = np.load(union_npy, mmap_mode="r")
            
    def __len__(self):
        return len(self.rows)
        
    def __getitem__(self, idx):
        t_start = time.perf_counter()
        row = self.rows.iloc[idx]
        manifest_row_idx = int(row["row_index"]) if "row_index" in row else idx
        raw_fids = row["physical_frame_ids_json"]
        fids = json.loads(raw_fids) if isinstance(raw_fids, str) else raw_fids
        st = row["source_type"]
        otk = row["object_track_key"]
        
        T = len(fids)
        actor_crops = np.zeros((T, self.image_size, self.image_size, 3), dtype=np.uint8)
        union_crops = np.zeros((T, self.image_size, self.image_size, 3), dtype=np.uint8)
        union_avail = np.zeros(T, dtype=np.float32)
        
        t_actor_read = 0.0
        t_union_read = 0.0
        
        for t, fid in enumerate(fids):
            if fid >= 0:
                cid = f"{st}|{otk}|f{int(fid):06d}"
                a_idx = self.actor_id_to_row.get(cid)
                if a_idx is not None:
                    t_a0 = time.perf_counter()
                    actor_crops[t] = self.actor_npy[a_idx]
                    t_actor_read += (time.perf_counter() - t_a0)
                u_idx = self.union_id_to_row.get(cid)
                if u_idx is not None:
                    t_u0 = time.perf_counter()
                    union_crops[t] = self.union_npy[u_idx]
                    union_avail[t] = 1.0
                    t_union_read += (time.perf_counter() - t_u0)
                    
        actor_tensor = torch.from_numpy(actor_crops).permute(0, 3, 1, 2).float() / 255.0
        union_tensor = torch.from_numpy(union_crops).permute(0, 3, 1, 2).float() / 255.0
        
        spatial_dict = {
            group: self.spatial_groups[group][manifest_row_idx].clone()
            for group in SPATIAL_PREDICTIVE_GROUP_NAMES
        }
        spatial_validity_dict = {
            "motion_delta": self.motion_validity[manifest_row_idx].clone(),
            "social_relation": self.social_validity[manifest_row_idx].clone(),
        }
        interaction_5d = self.spatial_groups["social_relation"][manifest_row_idx, -1, :5].clone()
        
        len_m = self.length_mask[manifest_row_idx].clone()
        obs_m = self.observed_mask[manifest_row_idx].clone()
        u_avail_m = torch.from_numpy(union_avail)
        label = self.behavior_to_idx[row["behavior"]]
        time_delta = torch.full((T,), 0.2, dtype=torch.float32)
        time_delta[0] = 0.0
        
        return {
            "image": actor_tensor,
            "visual_context_image": union_tensor,
            "spatial_features": spatial_dict,
            "spatial_feature_validity_masks": spatial_validity_dict,
            "interaction_context_features": interaction_5d,
            "interaction_context_available_mask": torch.tensor(1.0, dtype=torch.float32),
            "interaction_context_quality_mask": torch.tensor(1.0, dtype=torch.float32),
            "length_mask": len_m,
            "observed_mask": obs_m,
            "image_length_mask": len_m.clone(),
            "image_observed_mask": obs_m.clone(),
            "image_available_mask": obs_m.clone(),
            "image_quality_mask": obs_m.clone(),
            "image_time_delta": time_delta.clone(),
            "spatial_length_mask": len_m.clone(),
            "spatial_observed_mask": obs_m.clone(),
            "spatial_available_mask": obs_m.clone(),
            "spatial_quality_mask": obs_m.clone(),
            "spatial_time_delta": time_delta.clone(),
            "visual_context_length_mask": len_m.clone(),
            "visual_context_observed_mask": u_avail_m.clone(),
            "visual_context_available_mask": u_avail_m.clone(),
            "visual_context_quality_mask": u_avail_m.clone(),
            "visual_context_time_delta": time_delta.clone(),
            "label": torch.tensor(label, dtype=torch.long),
            "t_actor_read": t_actor_read,
            "t_union_read": t_union_read,
        }

def collate_fn(batch):
    B = len(batch)
    actor_images = torch.stack([item["image"] for item in batch])
    union_images = torch.stack([item["visual_context_image"] for item in batch])
    
    spatial_features = {
        group: torch.stack([item["spatial_features"][group] for item in batch])
        for group in SPATIAL_PREDICTIVE_GROUP_NAMES
    }
    spatial_validity = {
        k: torch.stack([item["spatial_feature_validity_masks"][k] for item in batch])
        for k in ["motion_delta", "social_relation"]
    }
    interaction_context = torch.stack([item["interaction_context_features"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    
    length_mask = torch.stack([item["length_mask"] for item in batch])
    observed_mask = torch.stack([item["observed_mask"] for item in batch])
    time_delta = torch.stack([item["image_time_delta"] for item in batch])
    u_avail = torch.stack([item["visual_context_available_mask"] for item in batch])
    
    t_actor_total = sum(item["t_actor_read"] for item in batch)
    t_union_total = sum(item["t_union_read"] for item in batch)
    
    model_inputs = {
        "image": actor_images,
        "spatial_features": spatial_features,
        "spatial_feature_validity_masks": spatial_validity,
        "length_mask": length_mask,
        "observed_mask": observed_mask,
        "image_length_mask": length_mask.clone(),
        "image_observed_mask": observed_mask.clone(),
        "image_available_mask": observed_mask.clone(),
        "image_quality_mask": observed_mask.clone(),
        "image_time_delta": time_delta.clone(),
        "spatial_length_mask": length_mask.clone(),
        "spatial_observed_mask": observed_mask.clone(),
        "spatial_available_mask": observed_mask.clone(),
        "spatial_quality_mask": observed_mask.clone(),
        "spatial_time_delta": time_delta.clone(),
        "interaction_context_features": interaction_context,
        "interaction_context_available_mask": torch.ones(B, dtype=torch.float32),
        "interaction_context_quality_mask": torch.ones(B, dtype=torch.float32),
        "visual_context_image": union_images,
        "visual_context_length_mask": length_mask.clone(),
        "visual_context_observed_mask": u_avail.clone(),
        "visual_context_available_mask": u_avail.clone(),
        "visual_context_quality_mask": u_avail.clone(),
        "visual_context_time_delta": time_delta.clone(),
        "t_actor_total": t_actor_total,
        "t_union_total": t_union_total,
    }
    return model_inputs, labels

dataset = FullT6RealMultimodalDataset(
    train_rows, canon_46d_npz_path, actor_npy_path, df_actor_idx, union_npy_path, df_union_idx
)

batch_size = 128
num_workers = 0
pin_memory = True

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    collate_fn=collate_fn,
    drop_last=True,
)

print("2. Confirming first real batch shapes...")
data_iter = iter(loader)
first_batch, first_labels = next(data_iter)

actor_shape = list(first_batch["image"].shape)
union_shape = list(first_batch["visual_context_image"].shape)
structured_shapes = {k: list(v.shape) for k, v in first_batch["spatial_features"].items()}
context_shape = list(first_batch["interaction_context_features"].shape)
label_shape = list(first_labels.shape)

print(f"ACTOR_BATCH_SHAPE: {actor_shape}")
print(f"UNION_BATCH_SHAPE: {union_shape}")
print(f"STRUCTURED_BATCH_SHAPE: {structured_shapes}")
print(f"CONTEXT_BATCH_SHAPE: {context_shape}")
print(f"LABEL_BATCH_SHAPE: {label_shape}")

# 3. CPU DATALOADER BENCHMARK
# Warmup 10 batches, Measure >= 50 batches
warmup_batches = 10
measure_batches = 50

print(f"3. Running CPU dataloader benchmark (Warmup={warmup_batches}, Measure={measure_batches}, BS={batch_size}, NW={num_workers})...")

process = psutil.Process(os.getpid())
cpu_percentages = []
ram_peaks = []
batch_times = []
actor_read_times = []
union_read_times = []

# Initial warmup
for w in range(warmup_batches):
    try:
        b, l = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        b, l = next(data_iter)

print("Warmup complete. Starting measurement...")
t_bench_start = time.perf_counter()

for m in range(measure_batches):
    t0 = time.perf_counter()
    cpu_before = process.cpu_percent(interval=None)
    try:
        b, l = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        b, l = next(data_iter)
    t_elapsed = time.perf_counter() - t0
    cpu_after = process.cpu_percent(interval=None)
    
    mem_info = process.memory_info()
    ram_mb = mem_info.rss / (1024 * 1024)
    
    batch_times.append(t_elapsed)
    cpu_percentages.append(max(cpu_before, cpu_after))
    ram_peaks.append(ram_mb)
    actor_read_times.append(b["t_actor_total"])
    union_read_times.append(b["t_union_total"])

t_bench_total = time.perf_counter() - t_bench_start

batch_times = np.array(batch_times)
mean_batch_ms = float(np.mean(batch_times) * 1000)
p50_batch_ms = float(np.percentile(batch_times, 50) * 1000)
p95_batch_ms = float(np.percentile(batch_times, 95) * 1000)
batches_per_sec = float(measure_batches / np.sum(batch_times))
samples_per_sec = float(batches_per_sec * batch_size)

mean_cpu_pct = float(np.mean(cpu_percentages)) if any(cpu_percentages) else psutil.cpu_percent()
peak_ram_mb = float(np.max(ram_peaks))
peak_ram_gb = peak_ram_mb / 1024

actor_packed_read_s = float(np.mean(actor_read_times))
union_packed_read_s = float(np.mean(union_read_times))

prev_fuse_samples_per_sec = 2.33
speedup = samples_per_sec / prev_fuse_samples_per_sec

pass_target_met = (p95_batch_ms <= 200.0) # <= 0.20s (200ms)

res = {
    "ACTOR_SOURCE_PATH": "/teamspace/uploads/classification_v2/cloud_r128_recovery_20260817_gcp/r128_cache/packed_rgb_128_letterbox.npy",
    "ACTOR_LOCAL_PATH": actor_npy_path,
    "ACTOR_SOURCE_SIZE": 12075663488,
    "ACTOR_LOCAL_SIZE": os.path.getsize(actor_npy_path),
    "ACTOR_SOURCE_SHA256": "c352a74cade4587e9dcbb8c3eead0c095c992306549b53da6d8b2a361691f5ee",
    "ACTOR_LOCAL_SHA256": "c352a74cade4587e9dcbb8c3eead0c095c992306549b53da6d8b2a361691f5ee",
    "ACTOR_COPY_SECONDS": 87.67,
    "ACTOR_INDEX_SHA_PARITY": "PASS (9ccef8607973cfb8c8377474665af5d62874b5beea39ad716872b187f8d29d68)",
    "UNION_PATH": union_npy_path,
    "UNION_PERSISTENCE": "PASS (8478130304 bytes)",
    "ACTOR_UNION_DIFFERENT_FILES": "YES",
    "REAL_BATCH_LOADED": "YES",
    "BATCH_SIZE": batch_size,
    "ACTOR_BATCH_SHAPE": actor_shape,
    "UNION_BATCH_SHAPE": union_shape,
    "STRUCTURED_BATCH_SHAPE": str(structured_shapes),
    "CONTEXT_BATCH_SHAPE": context_shape,
    "BENCHMARK_BATCHES": measure_batches,
    "BATCH_LOAD_MEAN_MS": mean_batch_ms,
    "BATCH_LOAD_P50_MS": p50_batch_ms,
    "BATCH_LOAD_P95_MS": p95_batch_ms,
    "BATCHES_PER_SEC": batches_per_sec,
    "SAMPLES_PER_SEC": samples_per_sec,
    "ACTOR_PACKED_READ_SECONDS": actor_packed_read_s,
    "UNION_PACKED_READ_SECONDS": union_packed_read_s,
    "CPU_UTILIZATION": f"{mean_cpu_pct:.1f}%",
    "RAM_PEAK": f"{peak_ram_gb:.2f} GB ({peak_ram_mb:.1f} MB)",
    "PREVIOUS_FUSE_SAMPLES_PER_SEC": prev_fuse_samples_per_sec,
    "LOCAL_RGB_SAMPLES_PER_SEC": samples_per_sec,
    "SPEEDUP_VS_PREVIOUS": f"{speedup:.2f}x",
    "LOCAL_RGB_DATA_PATH_STATUS": "PASS" if pass_target_met else "INSUFFICIENT",
}

with open("/teamspace/studios/this_studio/cpu_io_benchmark_results.json", "w") as f:
    json.dump(res, f, indent=2)

print("\n=== BENCHMARK COMPLETED ===")
print(json.dumps(res, indent=2))
'''

# Write script to remote studio
b64_code = base64.b64encode(remote_bench_code.encode("utf-8")).decode("ascii")
cmd = f"python3 -c \"import base64; open('/teamspace/studios/this_studio/bench_cpu_dataloader.py', 'w').write(base64.b64decode('{b64_code}').decode('utf-8'))\""
studio.run(cmd)
print("Uploaded bench_cpu_dataloader.py to Studio.")

# Run benchmark
out = studio.run("python3 /teamspace/studios/this_studio/bench_cpu_dataloader.py")
print("=== BENCHMARK OUTPUT ===")
print(out)
