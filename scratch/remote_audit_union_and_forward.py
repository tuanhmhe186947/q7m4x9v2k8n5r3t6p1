import base64
import json
import lightning_sdk

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

remote_code = """
import os, sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Checkpaths
actor_dir = "/teamspace/uploads/classification_v2/cloud_r128_recovery_20260817_gcp/r128_cache"
union_dir = "/teamspace/studios/this_studio/full_t6_union_r128_20260818"

actor_npy = os.path.join(actor_dir, "packed_rgb_128_letterbox.npy")
actor_idx = os.path.join(actor_dir, "packed_image_cache_index.csv")

union_npy = os.path.join(union_dir, "packed_rgb_128_letterbox.npy")
union_idx = os.path.join(union_dir, "packed_image_cache_index.csv")
union_manifest = os.path.join(union_dir, "visual_context_manifest.csv")

# 1. Load arrays
arr_actor = np.load(actor_npy, mmap_mode="r")
arr_union = np.load(union_npy, mmap_mode="r")
df_actor_idx = pd.read_csv(actor_idx, low_memory=False)
df_union_idx = pd.read_csv(union_idx, low_memory=False)
df_union_man = pd.read_csv(union_manifest, low_memory=False)

print(f"Actor Cache Array Shape: {arr_actor.shape}, Dtype: {arr_actor.dtype}")
print(f"Union Cache Array Shape: {arr_union.shape}, Dtype: {arr_union.dtype}")

# 2. Sample 20 available contexts for pixel MAD
actor_map = dict(zip(df_actor_idx["image_context_id"], df_actor_idx["packed_row"]))
union_map = dict(zip(df_union_idx["image_context_id"], df_union_idx["packed_row"]))

avail_rows = df_union_man[df_union_man["visual_context_available"] == True].sample(n=20, random_state=20260818)

mads = []
max_diffs = []
identicals = 0

for _, r in avail_rows.iterrows():
    cid = r["image_context_id"]
    a_row = actor_map[cid]
    u_row = union_map[cid]
    
    img_a = arr_actor[a_row].astype(float)
    img_u = arr_union[u_row].astype(float)
    
    mad = float(np.mean(np.abs(img_a - img_u)))
    max_d = float(np.max(np.abs(img_a - img_u)))
    mads.append(mad)
    max_diffs.append(max_d)
    if np.array_equal(img_a, img_u):
        identicals += 1

print(f"Remote Sampled 20 Contexts: Identical = {identicals}/20, Mean MAD = {np.mean(mads):.2f}, Mean MaxDiff = {np.mean(max_diffs):.1f}")
assert identicals == 0, "Actor and Union must not be identical!"

# 3. Test M0 Model Forward Pass with Real Union Input
sys.path.insert(0, "/teamspace/studios/this_studio/runtime_ea392e2d/src")

from pig_behavior.classification_v2.training.config import load_training_config
from pig_behavior.classification_v2.models.model_factory import build_multimodal_model
from pig_behavior.classification_v2.features.spatial_schema import (
    SPATIAL_PREDICTIVE_FEATURES,
    SPATIAL_PREDICTIVE_GROUP_NAMES,
)
from pig_behavior.classification_v2.datasets.interaction_context_loader import (
    INTERACTION_CONTEXT_FEATURE_COLUMNS,
)

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
model.eval()

# Construct a real batch with B=16
batch_size = 16
sequence_length = 6
image_size = 128

# Real actor and real union tensors
actor_batch = torch.from_numpy(arr_actor[:batch_size*sequence_length].copy().reshape(batch_size, sequence_length, image_size, image_size, 3)).permute(0, 1, 4, 2, 3).float() / 255.0
union_batch = torch.from_numpy(arr_union[:batch_size*sequence_length].copy().reshape(batch_size, sequence_length, image_size, image_size, 3)).permute(0, 1, 4, 2, 3).float() / 255.0

length = torch.ones(batch_size, sequence_length)
observed = length.clone()
time_delta = torch.full((batch_size, sequence_length), 0.2)
time_delta[:, 0] = 0.0

inputs = {
    "image": actor_batch,
    "spatial_features": {
        name: torch.rand(batch_size, sequence_length, dim)
        for name, dim in group_dims.items()
    },
    "spatial_feature_validity_masks": {
        name: torch.ones(batch_size, sequence_length, dim)
        for name, dim in group_dims.items()
        if name in {"motion_delta", "social_relation"}
    },
    "length_mask": length,
    "observed_mask": observed,
    "image_length_mask": length.clone(),
    "image_observed_mask": observed.clone(),
    "image_available_mask": observed.clone(),
    "image_quality_mask": observed.clone(),
    "image_time_delta": time_delta.clone(),
    "spatial_length_mask": length.clone(),
    "spatial_observed_mask": observed.clone(),
    "spatial_available_mask": observed.clone(),
    "spatial_quality_mask": observed.clone(),
    "spatial_time_delta": time_delta.clone(),
    "interaction_context_features": torch.rand(batch_size, interaction_dim),
    "interaction_context_available_mask": torch.ones(batch_size),
    "interaction_context_quality_mask": torch.ones(batch_size),
    "visual_context_image": union_batch,
    "visual_context_length_mask": length.clone(),
    "visual_context_observed_mask": observed.clone(),
    "visual_context_available_mask": observed.clone(),
    "visual_context_quality_mask": observed.clone(),
    "visual_context_time_delta": time_delta.clone(),
}

labels = torch.randint(0, 10, (batch_size,), dtype=torch.long)

with torch.inference_mode():
    output = model(**inputs)
    logits = output.behavior
    loss = F.cross_entropy(logits, labels)

print(f"M0 Forward Logits Shape: {logits.shape}")
print(f"M0 Forward Loss: {loss.item():.4f} (Finite: {torch.isfinite(loss).item()})")
print("REMOTE_AUDIT_SUCCESS: TRUE")
"""

b64 = base64.b64encode(remote_code.encode("utf-8")).decode("ascii")
res = studio.run(f'python3 -c "import base64; exec(base64.b64decode(\'{b64}\').decode(\'utf-8\'))"')
print(res)
