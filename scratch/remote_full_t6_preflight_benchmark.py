import base64
import lightning_sdk

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

remote_script = """
import os
import sys
import time
import json
import psutil
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Setup paths
target_dir = "/teamspace/studios/this_studio/runtime_ea392e2d"
sys.path.insert(0, os.path.join(target_dir, "src"))
os.chdir(target_dir)

from pig_behavior.classification_v2.models.model_factory import build_multimodal_model
from pig_behavior.classification_v2.training.config import load_training_config
from pig_behavior.classification_v2.features.spatial_schema import (
    SPATIAL_PREDICTIVE_FEATURES,
    SPATIAL_PREDICTIVE_GROUP_NAMES,
)
from pig_behavior.classification_v2.datasets.interaction_context_loader import (
    INTERACTION_CONTEXT_FEATURE_COLUMNS,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

# 1. Authority Paths
r128_dir = "/teamspace/uploads/classification_v2/cloud_r128_recovery_20260817_gcp/r128_cache"
full_t6_dir = "/teamspace/uploads/classification_v2/full_t6_training_authority_20260817"

packed_npy_path = os.path.join(r128_dir, "packed_rgb_128_letterbox.npy")
packed_idx_path = os.path.join(r128_dir, "packed_image_cache_index.csv")
canon_46d_npz_path = os.path.join(full_t6_dir, "full_t6_canonical_46d.npz")
row_manifest_path = os.path.join(full_t6_dir, "full_t6_row_manifest.csv")
full_temporal_manifest_path = os.path.join(full_t6_dir, "full_temporal_window_manifest_release.csv")
target_split_roles_path = os.path.join(full_t6_dir, "target_split_roles_release.csv")

print("=== 1. FULL-T6 REMOTE AUTHORITY PHYSICAL VERIFICATION ===")
df_full_temporal = pd.read_csv(full_temporal_manifest_path, low_memory=False)
df_t6 = df_full_temporal[df_full_temporal["view_id"] == "T6"].copy()
print(f"Full T6 total rows: {len(df_t6):,}")

df_row = pd.read_csv(row_manifest_path, low_memory=False)
print(f"Row manifest total rows: {len(df_row):,}")
split_counts = df_row["split"].value_counts().to_dict()
print(f"Split counts: {split_counts}")
train_count = split_counts.get("train", 0)
val_count = split_counts.get("validation", 0)

# Verify 46D NPZ
with np.load(canon_46d_npz_path) as npz_46d:
    npz_keys = list(npz_46d.keys())
    print(f"46D NPZ keys count: {len(npz_keys)}")
    print(f"46D bbox_xywh_n shape: {npz_46d['bbox_xywh_n'].shape}")
    print(f"46D motion_delta shape: {npz_46d['motion_delta'].shape}")
    print(f"46D roi_class_relation shape: {npz_46d['roi_class_relation'].shape}")
    print(f"46D social_relation shape: {npz_46d['social_relation'].shape}")

# Verify Actor RGB Cache
actor_arr = np.load(packed_npy_path, mmap_mode="r")
actor_size = os.path.getsize(packed_npy_path)
print(f"Actor Cache: shape={actor_arr.shape}, dtype={actor_arr.dtype}, size={actor_size:,} B")
df_idx = pd.read_csv(packed_idx_path, low_memory=False)
print(f"Actor Index: rows={len(df_idx):,}")

# 2. Build Fast Production Dataset
print("\n=== 2. PRODUCTION FULL-T6 REAL MULTIMODAL DATASET ===")

class FullT6ProductionDataset(Dataset):
    def __init__(self, row_manifest_df, npz_46d_path, actor_npy_path, actor_idx_df, image_size=128):
        self.rows = row_manifest_df.reset_index(drop=True)
        self.npz = np.load(npz_46d_path)
        self.actor_npy = np.load(actor_npy_path, mmap_mode="r")
        self.image_size = image_size
        self.context_id_to_row = dict(zip(actor_idx_df["image_context_id"], actor_idx_df["packed_row"]))
        self.behavior_to_idx = {b: i for i, b in enumerate(VALID_BEHAVIORS)}
        
        # Pre-cache spatial arrays in memory (46D NPZ is only 16.8 MB)
        self.spatial_groups = {
            group: torch.from_numpy(self.npz[group])
            for group in SPATIAL_PREDICTIVE_GROUP_NAMES
        }
        self.length_mask = torch.from_numpy(self.npz["length_mask"])
        self.observed_mask = torch.from_numpy(self.npz["observed_mask"])
        self.quality_mask = torch.from_numpy(self.npz["spatial_quality_mask"])
        self.motion_validity = torch.from_numpy(self.npz["motion_feature_validity_mask"])
        self.social_validity = torch.from_numpy(self.npz["social_feature_validity_mask"])
        
        # Parse physical frame IDs for image sequence indexing
        self.frame_ids = [json.loads(s) for s in self.rows["physical_frame_ids_json"]]
        self.target_ids = self.rows["target_id"].tolist()
        self.behaviors = torch.tensor([self.behavior_to_idx[b] for b in self.rows["behavior"]], dtype=torch.long)
        self.interaction_context_dim = len(INTERACTION_CONTEXT_FEATURE_COLUMNS)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        # 1. Image sequence (T=6, 3, H, W)
        frames_tensors = []
        target_id = self.target_ids[idx]
        f_ids = self.frame_ids[idx]
        
        # Parse prefix / video / track from target_id:
        # e.g. cvat_tracking_xml|cvat_tracking_Pigs281119_000085_30fps|video=Pigs281119_000085_30fps|track_id=0|...
        parts = target_id.split("|")
        src_kind = parts[0]
        ds_name = parts[1]
        video_name = parts[2] if len(parts) > 2 else ""
        track_part = parts[3] if len(parts) > 3 else ""
        
        for fid in f_ids:
            if fid >= 0:
                ctx_id = f"{src_kind}|source={src_kind}|dataset={ds_name}|{video_name}|{track_part}|f{fid:06d}"
                row_idx = self.context_id_to_row.get(ctx_id, -1)
                if row_idx >= 0:
                    img_np = self.actor_npy[row_idx]  # [128, 128, 3] uint8
                    # Convert to float tensor [3, 128, 128] normalized to [0, 1]
                    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
                else:
                    img_t = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
            else:
                img_t = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
            frames_tensors.append(img_t)
            
        actor_seq = torch.stack(frames_tensors, dim=0)  # [6, 3, 128, 128]
        
        # 2. Spatial features & masks
        spatial_dict = {g: self.spatial_groups[g][idx] for g in SPATIAL_PREDICTIVE_GROUP_NAMES}
        l_mask = self.length_mask[idx]
        obs_mask = self.observed_mask[idx]
        q_mask = self.quality_mask[idx]
        val_masks = {
            "motion_delta": self.motion_validity[idx],
            "social_relation": self.social_validity[idx],
        }
        
        time_delta = torch.full((6,), 0.2, dtype=torch.float32)
        time_delta[0] = 0.0
        
        # 3. Context & Union features
        ctx_feat = torch.zeros(self.interaction_context_dim, dtype=torch.float32)
        union_seq = actor_seq.clone()  # If union cache missing, tensor shape for interface
        
        return {
            "image": actor_seq,
            "spatial_features": spatial_dict,
            "spatial_feature_validity_masks": val_masks,
            "length_mask": l_mask,
            "observed_mask": obs_mask,
            "image_length_mask": l_mask,
            "image_observed_mask": obs_mask,
            "image_available_mask": obs_mask,
            "image_quality_mask": q_mask,
            "image_time_delta": time_delta,
            "spatial_length_mask": l_mask,
            "spatial_observed_mask": obs_mask,
            "spatial_available_mask": obs_mask,
            "spatial_quality_mask": q_mask,
            "spatial_time_delta": time_delta,
            "interaction_context_features": ctx_feat,
            "interaction_context_available_mask": torch.tensor(1.0),
            "interaction_context_quality_mask": torch.tensor(1.0),
            "visual_context_image": union_seq,
            "visual_context_length_mask": l_mask,
            "visual_context_observed_mask": obs_mask,
            "visual_context_available_mask": obs_mask,
            "visual_context_quality_mask": q_mask,
            "visual_context_time_delta": time_delta,
            "target": self.behaviors[idx],
        }

def collate_multimodal(batch):
    out = {}
    for key in batch[0]:
        if key == "spatial_features":
            out[key] = {
                g: torch.stack([b[key][g] for b in batch], dim=0)
                for g in SPATIAL_PREDICTIVE_GROUP_NAMES
            }
        elif key == "spatial_feature_validity_masks":
            out[key] = {
                g: torch.stack([b[key][g] for b in batch], dim=0)
                for g in ["motion_delta", "social_relation"]
            }
        else:
            out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out

train_rows_df = df_row[df_row["split"] == "train"].reset_index(drop=True)
ds_train = FullT6ProductionDataset(
    train_rows_df,
    canon_46d_npz_path,
    packed_npy_path,
    df_idx,
    image_size=128
)
print(f"Train dataset initialized: {len(ds_train):,} samples.")

# 3. Test ONE REAL BATCH through exact M0
print("\n=== 3. REAL PRODUCTION BATCH & M0 FORWARD PASS ===")
loader_single = DataLoader(ds_train, batch_size=16, shuffle=False, collate_fn=collate_multimodal)
batch0 = next(iter(loader_single))

print(f"Batch actor image shape: {list(batch0['image'].shape)}")
print(f"Batch spatial bbox_xywh_n: {list(batch0['spatial_features']['bbox_xywh_n'].shape)}")
print(f"Batch spatial motion_delta: {list(batch0['spatial_features']['motion_delta'].shape)}")
print(f"Batch spatial roi_class_relation: {list(batch0['spatial_features']['roi_class_relation'].shape)}")
print(f"Batch spatial social_relation: {list(batch0['spatial_features']['social_relation'].shape)}")
print(f"Batch interaction context: {list(batch0['interaction_context_features'].shape)}")
print(f"Batch visual context image: {list(batch0['visual_context_image'].shape)}")
print(f"Batch targets: {list(batch0['target'].shape)}")
print(f"Batch time delta: {list(batch0['image_time_delta'].shape)}")

# Build M0
config_path = os.path.join(target_dir, "configs/classification_v2/m0_full_multimodal_r34_t6_concat.json")
training_config = load_training_config(config_path)
model_cfg = training_config.model
group_dims = {
    group: len(SPATIAL_PREDICTIVE_FEATURES[group])
    for group in SPATIAL_PREDICTIVE_GROUP_NAMES
}
interaction_dim = len(INTERACTION_CONTEXT_FEATURE_COLUMNS)

model = build_multimodal_model(
    model_cfg,
    spatial_input_dims=group_dims,
    interaction_context_dim=interaction_dim,
    num_classes=10,
)
model.eval()

inputs = {k: v for k, v in batch0.items() if k != "target"}
with torch.no_grad():
    output = model(**inputs)
    logits = output.behavior
    loss = F.cross_entropy(logits, batch0["target"])

print(f"\nReal batch logits shape: {list(logits.shape)}")
print(f"Real batch CE loss: {loss.item():.4f}")
print(f"Logits finite: {torch.isfinite(logits).all().item()}")
print(f"Loss finite: {torch.isfinite(loss).item()}")

# 4. BENCHMARK DATALOADER THROUGHPUT (Drive vs Local Staging)
print("\n=== 4. BENCHMARK DATALOADER THROUGHPUT ON CPU ===")

def benchmark_loader(dataset, num_workers, batch_size=16, n_batches=30, persistent_workers=False, prefetch_factor=2):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_multimodal,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    
    # Warm-up (3 batches)
    it = iter(loader)
    for _ in range(3):
        try:
            next(it)
        except StopIteration:
            break
            
    times = []
    process = psutil.Process()
    cpu_percents = []
    
    for _ in range(n_batches):
        t0 = time.perf_counter()
        try:
            next(it)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            cpu_percents.append(process.cpu_percent(interval=None))
        except StopIteration:
            break
            
    times = np.array(times)
    mean_t = float(np.mean(times))
    p50_t = float(np.percentile(times, 50))
    p95_t = float(np.percentile(times, 95))
    batches_per_sec = float(1.0 / mean_t) if mean_t > 0 else 0.0
    samples_per_sec = float(batch_size / mean_t) if mean_t > 0 else 0.0
    ram_mb = process.memory_info().rss / (1024 * 1024)
    cpu_util = float(np.mean(cpu_percents)) if cpu_percents else 0.0
    
    return {
        "num_workers": num_workers,
        "batch_size": batch_size,
        "measured_batches": len(times),
        "mean_s": mean_t,
        "p50_s": p50_t,
        "p95_s": p95_t,
        "batches_per_sec": round(batches_per_sec, 2),
        "samples_per_sec": round(samples_per_sec, 2),
        "cpu_percent": round(cpu_util, 1),
        "ram_mb": round(ram_mb, 1),
    }

print("--- Testing on Teamspace Drive ---")
results_drive = {}
for nw in [0, 2, 4]:
    res = benchmark_loader(ds_train, num_workers=nw, persistent_workers=(nw > 0))
    results_drive[f"nw{nw}"] = res
    print(f"  Drive NW={nw}: {res['batches_per_sec']} batches/s ({res['samples_per_sec']} samples/s) | p50={res['p50_s']*1000:.1f}ms | p95={res['p95_s']*1000:.1f}ms")

# 5. Local Staging Test
print("\n--- Testing Local Staging on Studio Fast Disk ---")
local_stage_dir = "/teamspace/studios/this_studio/staged_data"
os.makedirs(local_stage_dir, exist_ok=True)

staged_npy = os.path.join(local_stage_dir, "packed_rgb_128_letterbox.npy")
staged_idx = os.path.join(local_stage_dir, "packed_image_cache_index.csv")

if not os.path.exists(staged_npy) or os.path.getsize(staged_npy) != os.path.getsize(packed_npy_path):
    print("Staging packed_rgb_128_letterbox.npy to local fast storage...")
    t_copy_start = time.perf_counter()
    shutil.copy2(packed_npy_path, staged_npy)
    print(f"Copied in {time.perf_counter() - t_copy_start:.1f}s")
if not os.path.exists(staged_idx):
    shutil.copy2(packed_idx_path, staged_idx)

ds_local = FullT6ProductionDataset(
    train_rows_df,
    canon_46d_npz_path,
    staged_npy,
    df_idx,
    image_size=128
)

results_local = {}
for nw in [0, 2, 4]:
    res = benchmark_loader(ds_local, num_workers=nw, persistent_workers=(nw > 0))
    results_local[f"nw{nw}"] = res
    print(f"  Local Stage NW={nw}: {res['batches_per_sec']} batches/s ({res['samples_per_sec']} samples/s) | p50={res['p50_s']*1000:.1f}ms | p95={res['p95_s']*1000:.1f}ms")

summary_report = {
    "drive_benchmark": results_drive,
    "local_benchmark": results_local,
    "logits_shape": list(logits.shape),
    "ce_loss": round(float(loss.item()), 4),
    "loss_finite": bool(torch.isfinite(loss).item()),
}
print("\n=== FINAL BENCHMARK SUMMARY ===")
print(json.dumps(summary_report, indent=2))
"""

b64_code = base64.b64encode(remote_script.encode("utf-8")).decode("ascii")
cmd = f'python3 -c "import base64; exec(base64.b64decode(\'{b64_code}\').decode(\'utf-8\'))"'

print("Executing comprehensive FULL-T6 preflight & benchmark on Studio...")
res = studio.run(cmd)
print("=== REMOTE OUTPUT ===")
print(res.encode("ascii", errors="replace").decode("ascii"))
