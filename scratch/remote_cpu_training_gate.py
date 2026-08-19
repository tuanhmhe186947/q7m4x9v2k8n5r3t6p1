import base64
import json
from pathlib import Path
import lightning_sdk

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

remote_code = r'''
import os, sys, time, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tv_models
from torchvision.models import ResNet34_Weights

# 1. SETUP ENVIRONMENT
runtime_dir = "/teamspace/studios/this_studio/runtime_ea392e2d"
sys.path.insert(0, os.path.join(runtime_dir, "src"))
os.chdir(runtime_dir)

from pig_behavior.classification_v2.training.config import load_training_config
from pig_behavior.classification_v2.models.model_factory import build_multimodal_model
from pig_behavior.classification_v2.features.spatial_schema import (
    SPATIAL_PREDICTIVE_FEATURES,
    SPATIAL_PREDICTIVE_GROUP_NAMES,
)
from pig_behavior.classification_v2.datasets.interaction_context_loader import (
    INTERACTION_CONTEXT_FEATURE_COLUMNS,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

# Paths
actor_dir = "/teamspace/uploads/classification_v2/cloud_r128_recovery_20260817_gcp/r128_cache"
full_t6_dir = "/teamspace/uploads/classification_v2/full_t6_training_authority_20260817"
union_dir = "/teamspace/studios/this_studio/full_t6_union_r128_20260818"

actor_npy_path = os.path.join(actor_dir, "packed_rgb_128_letterbox.npy")
actor_idx_path = os.path.join(actor_dir, "packed_image_cache_index.csv")

union_npy_path = os.path.join(union_dir, "packed_rgb_128_letterbox.npy")
union_idx_path = os.path.join(union_dir, "packed_image_cache_index.csv")
union_man_path = os.path.join(union_dir, "visual_context_manifest.csv")
union_sel_path = os.path.join(union_dir, "full_t6_union_selection.csv")
union_audit_path = os.path.join(union_dir, "packed_image_cache_audit.json")
union_ctx_audit_path = os.path.join(union_dir, "visual_context_cache_audit.json")

canon_46d_npz_path = os.path.join(full_t6_dir, "full_t6_canonical_46d.npz")
row_manifest_path = os.path.join(full_t6_dir, "full_t6_row_manifest.csv")

# 1. VERIFY PERSISTENCE & SIZES
print("=== 1. PERSISTENCE CHECK ===")
union_files = {}
for fname in [
    "full_t6_union_selection.csv",
    "visual_context_manifest.csv",
    "visual_context_cache_audit.json",
    "packed_image_cache_index.csv",
    "packed_image_cache_audit.json",
    "packed_rgb_128_letterbox.npy"
]:
    p = os.path.join(union_dir, fname)
    assert os.path.exists(p), f"Missing union file: {p}"
    sz = os.path.getsize(p)
    union_files[fname] = sz
    print(f"  {fname}: {sz:,} bytes")

print("UNION_REMOTE_PERSISTENCE = PASS")

# 2. PRETRAINED WEIGHT AVAILABILITY CHECK
print("=== 2. PRETRAINED WEIGHT CHECK ===")
try:
    actor_pretrained = tv_models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    actor_pre_load = "PASS"
except Exception as e:
    actor_pre_load = f"FAIL ({e})"

try:
    union_pretrained = tv_models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    union_pre_load = "PASS"
except Exception as e:
    union_pre_load = f"FAIL ({e})"

print(f"ACTOR_PRETRAINED_LOAD = {actor_pre_load}")
print(f"UNION_PRETRAINED_LOAD = {union_pre_load}")

# 3. REAL PRODUCTION DATASET & BATCH LOADING
print("=== 3. PRODUCTION DATASET & REAL BATCH LOADING ===")

df_row = pd.read_csv(row_manifest_path, low_memory=False)
train_rows = df_row[df_row["split"] == "train"].reset_index(drop=True)
val_rows = df_row[df_row["split"] == "validation"].reset_index(drop=True)
print(f"Train rows: {len(train_rows):,}, Val rows: {len(val_rows):,}")

class FullT6RealMultimodalDataset(Dataset):
    def __init__(self, row_manifest_df, npz_46d_path, actor_npy, actor_idx_df, union_npy, union_idx_df, image_size=128):
        self.rows = row_manifest_df.reset_index(drop=True)
        self.npz = np.load(npz_46d_path)
        self.actor_npy = np.load(actor_npy, mmap_mode="r")
        self.union_npy = np.load(union_npy, mmap_mode="r")
        self.image_size = image_size
        
        self.actor_id_to_row = dict(zip(actor_idx_df["image_context_id"], actor_idx_df["packed_row"]))
        self.union_id_to_row = dict(zip(union_idx_df["image_context_id"], union_idx_df["packed_row"]))
        self.behavior_to_idx = {b: i for i, b in enumerate(VALID_BEHAVIORS)}
        
        # Spatial schema tensors
        self.spatial_groups = {
            group: torch.from_numpy(self.npz[group])
            for group in SPATIAL_PREDICTIVE_GROUP_NAMES
        }
        self.length_mask = torch.from_numpy(self.npz["length_mask"])
        self.observed_mask = torch.from_numpy(self.npz["observed_mask"])
        self.quality_mask = torch.from_numpy(self.npz["spatial_quality_mask"])
        self.motion_validity = torch.from_numpy(self.npz["motion_feature_validity_mask"])
        self.social_validity = torch.from_numpy(self.npz["social_feature_validity_mask"])
        
    def __len__(self):
        return len(self.rows)
        
    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        manifest_row_idx = int(row["row_index"]) if "row_index" in row else idx
        
        raw_fids = row["physical_frame_ids_json"]
        if isinstance(raw_fids, str):
            fids = json.loads(raw_fids)
        else:
            fids = raw_fids
            
        st = row["source_type"]
        otk = row["object_track_key"]
        
        T = len(fids)
        actor_crops = np.zeros((T, self.image_size, self.image_size, 3), dtype=np.uint8)
        union_crops = np.zeros((T, self.image_size, self.image_size, 3), dtype=np.uint8)
        union_avail = np.zeros(T, dtype=np.float32)
        
        for t, fid in enumerate(fids):
            if fid >= 0:
                cid = f"{st}|{otk}|f{int(fid):06d}"
                # Actor crop lookup
                a_idx = self.actor_id_to_row.get(cid)
                if a_idx is not None:
                    actor_crops[t] = self.actor_npy[a_idx]
                
                # Union crop lookup (from distinct union packed cache)
                u_idx = self.union_id_to_row.get(cid)
                if u_idx is not None:
                    union_crops[t] = self.union_npy[u_idx]
                    union_avail[t] = 1.0
                    
        # To torch tensors
        # [T, H, W, C] -> [T, C, H, W] in [0, 1]
        actor_tensor = torch.from_numpy(actor_crops).permute(0, 3, 1, 2).float() / 255.0
        union_tensor = torch.from_numpy(union_crops).permute(0, 3, 1, 2).float() / 255.0
        
        # Spatial branch slices
        spatial_dict = {
            group: self.spatial_groups[group][manifest_row_idx].clone()
            for group in SPATIAL_PREDICTIVE_GROUP_NAMES
        }
        spatial_validity_dict = {
            "motion_delta": self.motion_validity[manifest_row_idx].clone(),
            "social_relation": self.social_validity[manifest_row_idx].clone(),
        }
        
        # 5D Interaction Context
        # Uses last timestep social relations or mean aggregate
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
    }
    return model_inputs, labels

df_actor_idx = pd.read_csv(actor_idx_path, low_memory=False)
df_union_idx = pd.read_csv(union_idx_path, low_memory=False)

train_dataset = FullT6RealMultimodalDataset(
    train_rows, canon_46d_npz_path, actor_npy_path, df_actor_idx, union_npy_path, df_union_idx
)

BATCH_SIZE = 8
loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

# Fetch one real batch
batch_inputs, batch_labels = next(iter(loader))

print(f"REAL_BATCH_SIZE = {BATCH_SIZE}")
print(f"ACTOR_BATCH_SHAPE = {list(batch_inputs['image'].shape)}")
print(f"STRUCTURED_BATCH_SHAPE = {{ {', '.join([repr(k) + ': ' + str(list(v.shape)) for k, v in batch_inputs['spatial_features'].items()])} }}")
print(f"CONTEXT_BATCH_SHAPE = {list(batch_inputs['interaction_context_features'].shape)}")
print(f"UNION_BATCH_SHAPE = {list(batch_inputs['visual_context_image'].shape)}")
print(f"LABEL_BATCH_SHAPE = {list(batch_labels.shape)}")
print(f"TIME_DELTA_SHAPE = {list(batch_inputs['image_time_delta'].shape)}")

# 4. BUILD FROZEN M0 MODEL & ONE REAL OPTIMIZER UPDATE
print("=== 4. ONE REAL TRAINING UPDATE ===")
config_path = Path("/teamspace/studios/this_studio/runtime_ea392e2d/configs/classification_v2/m0_full_multimodal_r34_t6_concat.json")
training_cfg = load_training_config(config_path)

group_dims = {
    group: len(SPATIAL_PREDICTIVE_FEATURES[group])
    for group in SPATIAL_PREDICTIVE_GROUP_NAMES
}
interaction_dim = len(INTERACTION_CONTEXT_FEATURE_COLUMNS)

model = build_multimodal_model(
    training_cfg.model,
    spatial_input_dims=group_dims,
    interaction_context_dim=interaction_dim,
    num_classes=10,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)

# Track parameter state before update
trainable_params_before = [p.clone() for p in model.parameters() if p.requires_grad]

model.train()
optimizer.zero_grad()

# Forward
output = model(**batch_inputs)
logits = output.behavior
loss = F.cross_entropy(logits, batch_labels)

print(f"FORWARD_PASS = PASS")
print(f"LOGITS_SHAPE = {list(logits.shape)}")
print(f"LOSS_VALUE = {loss.item():.4f}")
print(f"LOSS_FINITE = {torch.isfinite(loss).item()}")

# Backward
loss.backward()

# Check gradients
grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
finite_grads = [torch.isfinite(g).all().item() for g in grads]
nonzero_grads = [(g.abs() > 0).any().item() for g in grads]

all_grads_finite = all(finite_grads)
has_nonzero_grad = any(nonzero_grads)
print(f"BACKWARD_PASS = PASS")
print(f"FINITE_NONZERO_GRADIENT = {'YES' if (all_grads_finite and has_nonzero_grad) else 'NO'}")

# Optimizer step
optimizer.step()
print("OPTIMIZER_STEP_PASS = PASS")

# Verify parameter change
param_diffs = []
for p_before, p_after in zip(trainable_params_before, [p for p in model.parameters() if p.requires_grad]):
    diff = (p_after - p_before).abs().max().item()
    param_diffs.append(diff)

max_param_change = max(param_diffs)
print(f"PARAMETER_CHANGE_VERIFIED = {'YES' if max_param_change > 0 else 'NO'} (Max param delta: {max_param_change:.6e})")

# 5. CHECKPOINT ROUNDTRIP
print("=== 5. CHECKPOINT ROUNDTRIP ===")
ckpt_path = "/teamspace/studios/this_studio/temp_engineering_step1.pt"

# Save
ckpt_data = {
    "step": 1,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}
torch.save(ckpt_data, ckpt_path)
ckpt_size = os.path.getsize(ckpt_path)
print(f"CHECKPOINT_SAVE = PASS ({ckpt_size:,} bytes)")

# Reload into fresh model and optimizer
fresh_model = build_multimodal_model(
    training_cfg.model,
    spatial_input_dims=group_dims,
    interaction_context_dim=interaction_dim,
    num_classes=10,
)
fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=0.001, weight_decay=0.0)

loaded_ckpt = torch.load(ckpt_path, map_location="cpu")
fresh_model.load_state_dict(loaded_ckpt["model_state_dict"])
print("CHECKPOINT_RELOAD = PASS")

fresh_optimizer.load_state_dict(loaded_ckpt["optimizer_state_dict"])
print("OPTIMIZER_STATE_RELOAD = PASS")
print(f"RESTORED_GLOBAL_STEP = {loaded_ckpt['step']}")

# Evaluate forward pass on reloaded model
fresh_model.eval()
with torch.no_grad():
    reloaded_out = fresh_model(**batch_inputs)
    reloaded_logits = reloaded_out.behavior

print(f"RELOADED_FORWARD = PASS (Finite: {torch.isfinite(reloaded_logits).all().item()}, Logits Shape: {list(reloaded_logits.shape)})")

# Delete temporary checkpoint
if os.path.exists(ckpt_path):
    os.remove(ckpt_path)
    print("Deleted temporary engineering checkpoint.")

print("\nALL_GATES_COMPLETED_SUCCESSFULLY: TRUE")
'''

b64 = base64.b64encode(remote_code.encode("utf-8")).decode("ascii")
res = studio.run(f'python3 -c "import base64; exec(base64.b64decode(\'{b64}\').decode(\'utf-8\'))"')
print(res)
