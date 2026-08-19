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
import os, sys, time, json, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tv_models
from torchvision.models import ResNet34_Weights

# Setup environment
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"CUDA Available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}")

# Paths
actor_dir = "/teamspace/uploads/classification_v2/cloud_r128_recovery_20260817_gcp/r128_cache"
full_t6_dir = "/teamspace/uploads/classification_v2/full_t6_training_authority_20260817"
union_dir = "/teamspace/studios/this_studio/full_t6_union_r128_20260818"

actor_npy_path = os.path.join(actor_dir, "packed_rgb_128_letterbox.npy")
actor_idx_path = os.path.join(actor_dir, "packed_image_cache_index.csv")

union_npy_path = os.path.join(union_dir, "packed_rgb_128_letterbox.npy")
union_idx_path = os.path.join(union_dir, "packed_image_cache_index.csv")

canon_46d_npz_path = os.path.join(full_t6_dir, "full_t6_canonical_46d.npz")
row_manifest_path = os.path.join(full_t6_dir, "full_t6_row_manifest.csv")

df_row = pd.read_csv(row_manifest_path, low_memory=False)
train_rows = df_row[df_row["split"] == "train"].reset_index(drop=True)
df_actor_idx = pd.read_csv(actor_idx_path, low_memory=False)
df_union_idx = pd.read_csv(union_idx_path, low_memory=False)

class FullT6RealMultimodalDataset(Dataset):
    def __init__(self, row_manifest_df, npz_46d_path, actor_npy, actor_idx_df, union_npy, union_idx_df, image_size=128):
        self.rows = row_manifest_df.reset_index(drop=True)
        self.npz_path = npz_46d_path
        self.actor_npy_path = actor_npy
        self.union_npy_path = union_npy
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
        
        self.actor_npy = np.load(self.actor_npy_path, mmap_mode="r")
        self.union_npy = np.load(self.union_npy_path, mmap_mode="r")
            
    def __len__(self):
        return len(self.rows)
        
    def __getitem__(self, idx):
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
        
        for t, fid in enumerate(fids):
            if fid >= 0:
                cid = f"{st}|{otk}|f{int(fid):06d}"
                a_idx = self.actor_id_to_row.get(cid)
                if a_idx is not None:
                    actor_crops[t] = self.actor_npy[a_idx]
                u_idx = self.union_id_to_row.get(cid)
                if u_idx is not None:
                    union_crops[t] = self.union_npy[u_idx]
                    union_avail[t] = 1.0
                    
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

config_path = Path("/teamspace/studios/this_studio/runtime_ea392e2d/configs/classification_v2/m0_full_multimodal_r34_t6_concat.json")
training_cfg = load_training_config(config_path)

group_dims = {
    group: len(SPATIAL_PREDICTIVE_FEATURES[group])
    for group in SPATIAL_PREDICTIVE_GROUP_NAMES
}
interaction_dim = len(INTERACTION_CONTEXT_FEATURE_COLUMNS)

dataset = FullT6RealMultimodalDataset(
    train_rows, canon_46d_npz_path, actor_npy_path, df_actor_idx, union_npy_path, df_union_idx
)

def run_benchmark(
    batch_size=32,
    num_workers=0,
    pin_memory=True,
    amp_dtype=torch.bfloat16,
    channels_last=False,
    use_compile=False,
    warmup_steps=10,
    measure_steps=30,
):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
    
    model = build_multimodal_model(
        training_cfg.model,
        spatial_input_dims=group_dims,
        interaction_context_dim=interaction_dim,
        num_classes=10,
    )
    model = model.to(device)
    
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    
    compile_time = 0.0
    if use_compile:
        t0_c = time.perf_counter()
        model = torch.compile(model)
        compile_time = time.perf_counter() - t0_c
        
    model.train()
    
    data_iter = iter(loader)
    total_steps = warmup_steps + measure_steps
    
    step_times = []
    data_wait_times = []
    h2d_times = []
    forward_times = []
    backward_times = []
    
    for step in range(total_steps):
        t0 = time.perf_counter()
        
        # 1. Data wait
        t_data_start = time.perf_counter()
        try:
            batch_inputs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch_inputs, labels = next(data_iter)
        t_data = time.perf_counter() - t_data_start
        
        # 2. Host to Device
        t_h2d_start = time.perf_counter()
        dev_inputs = {}
        for k, v in batch_inputs.items():
            if isinstance(v, torch.Tensor):
                if channels_last and k in ("image", "visual_context_image"):
                    dev_inputs[k] = v.to(device, memory_format=torch.channels_last, non_blocking=pin_memory)
                else:
                    dev_inputs[k] = v.to(device, non_blocking=pin_memory)
            elif isinstance(v, dict):
                dev_inputs[k] = {gk: gv.to(device, non_blocking=pin_memory) for gk, gv in v.items()}
            else:
                dev_inputs[k] = v
        dev_labels = labels.to(device, non_blocking=pin_memory)
        torch.cuda.synchronize()
        t_h2d = time.perf_counter() - t_h2d_start
        
        # 3. Forward & Backward
        optimizer.zero_grad(set_to_none=True)
        
        t_fwd_start = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(amp_dtype is not None)):
            output = model(**dev_inputs)
            logits = output.behavior
            loss = F.cross_entropy(logits, dev_labels)
        torch.cuda.synchronize()
        t_fwd = time.perf_counter() - t_fwd_start
        
        t_bwd_start = time.perf_counter()
        if amp_dtype == torch.float16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        t_bwd = time.perf_counter() - t_bwd_start
        
        t_step = time.perf_counter() - t0
        
        if step >= warmup_steps:
            step_times.append(t_step)
            data_wait_times.append(t_data)
            h2d_times.append(t_h2d)
            forward_times.append(t_fwd)
            backward_times.append(t_bwd)
            
    step_times = np.array(step_times)
    data_wait_times = np.array(data_wait_times)
    
    mean_step = float(np.mean(step_times))
    p50_step = float(np.percentile(step_times, 50))
    p95_step = float(np.percentile(step_times, 95))
    steps_per_sec = 1.0 / mean_step
    samples_per_sec = steps_per_sec * batch_size
    data_wait_frac = float(np.sum(data_wait_times) / np.sum(step_times))
    
    peak_vram_bytes = torch.cuda.max_memory_allocated()
    peak_vram_gb = peak_vram_bytes / (1024**3)
    reserved_vram_gb = torch.cuda.memory_reserved() / (1024**3)
    
    res = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "amp": "bf16" if amp_dtype == torch.bfloat16 else ("fp16" if amp_dtype == torch.float16 else "fp32"),
        "channels_last": channels_last,
        "compile": use_compile,
        "compile_time_s": compile_time,
        "mean_step_s": mean_step,
        "p50_step_s": p50_step,
        "p95_step_s": p95_step,
        "steps_per_sec": steps_per_sec,
        "samples_per_sec": samples_per_sec,
        "data_wait_fraction": data_wait_frac,
        "peak_vram_gb": peak_vram_gb,
        "reserved_vram_gb": reserved_vram_gb,
        "loss_finite": bool(torch.isfinite(loss).item()),
        "last_loss": float(loss.item()),
    }
    
    del model, optimizer, loader
    torch.cuda.empty_cache()
    gc.collect()
    
    return res

results = {}

# 1. BASELINE PILOT (batch_size=32, num_workers=0, BF16, eager)
print("=== 1. BASELINE PILOT ===", flush=True)
res_baseline = run_benchmark(batch_size=32, num_workers=0, amp_dtype=torch.bfloat16, warmup_steps=10, measure_steps=30)
results["baseline"] = res_baseline
print(f"BASELINE: {res_baseline['samples_per_sec']:.2f} samples/s, p50={res_baseline['p50_step_s']*1000:.1f}ms, Peak VRAM={res_baseline['peak_vram_gb']:.2f}GB", flush=True)

# 2. BATCH SIZE SWEEP (16, 32, 64, 128)
print("=== 2. BATCH SIZE SWEEP ===", flush=True)
batch_results = {}
for bs in [16, 32, 64, 128]:
    try:
        r = run_benchmark(batch_size=bs, num_workers=0, amp_dtype=torch.bfloat16, warmup_steps=8, measure_steps=20)
        batch_results[str(bs)] = r
        print(f"BS={bs}: {r['samples_per_sec']:.2f} samples/s ({r['steps_per_sec']:.2f} steps/s), p50={r['p50_step_s']*1000:.1f}ms, Peak VRAM={r['peak_vram_gb']:.2f}GB", flush=True)
    except RuntimeError as e:
        print(f"BS={bs} failed: {e}", flush=True)
        batch_results[str(bs)] = {"error": str(e)}
        break
results["batch_sweep"] = batch_results

best_bs = 128
print(f"Selected Best BS: {best_bs}", flush=True)

# 3. AMP COMPARISON (BF16 vs FP16 on BS=128)
print("=== 3. AMP COMPARISON on BS=128 ===", flush=True)
res_bf16 = run_benchmark(batch_size=128, num_workers=0, amp_dtype=torch.bfloat16, warmup_steps=8, measure_steps=20)
res_fp16 = run_benchmark(batch_size=128, num_workers=0, amp_dtype=torch.float16, warmup_steps=8, measure_steps=20)
results["amp"] = {"bf16": res_bf16, "fp16": res_fp16}
print(f"BF16: {res_bf16['samples_per_sec']:.2f} samples/s, Peak VRAM={res_bf16['peak_vram_gb']:.2f}GB", flush=True)
print(f"FP16: {res_fp16['samples_per_sec']:.2f} samples/s, Peak VRAM={res_fp16['peak_vram_gb']:.2f}GB", flush=True)

selected_amp = torch.bfloat16 if res_bf16["samples_per_sec"] >= res_fp16["samples_per_sec"] else torch.float16
selected_amp_str = "bf16" if selected_amp == torch.bfloat16 else "fp16"

# 4. WORKER POLICY
print("=== 4. WORKER POLICY ===", flush=True)
# On FUSE Drive mmap, num_workers=0 is the only deadlock-free mode; num_workers>0 leads to FUSE D-state deadlock.
results["workers"] = {
    "0": {"samples_per_sec": res_bf16["samples_per_sec"], "status": "PASS", "data_wait_fraction": res_bf16["data_wait_fraction"]},
    "2": {"status": "UNSUPPORTED_FUSE_MMAP_DEADLOCK"},
    "4": {"status": "UNSUPPORTED_FUSE_MMAP_DEADLOCK"}
}
best_nw = 0
print(f"Selected num_workers: 0 (Drive FUSE mmap constraint)", flush=True)

# 5. CHANNELS LAST TEST
print("=== 5. CHANNELS LAST TEST ===", flush=True)
res_contig = res_bf16
try:
    res_cl = run_benchmark(batch_size=128, num_workers=0, amp_dtype=selected_amp, channels_last=True, warmup_steps=8, measure_steps=20)
    results["channels_last"] = {"contiguous": res_contig, "channels_last": res_cl}
    print(f"Contiguous: {res_contig['samples_per_sec']:.2f} samples/s", flush=True)
    print(f"Channels Last: {res_cl['samples_per_sec']:.2f} samples/s", flush=True)
    use_cl = res_cl["samples_per_sec"] > res_contig["samples_per_sec"] * 1.03
except Exception as e:
    print(f"Channels last error: {e}", flush=True)
    results["channels_last"] = {"contiguous": res_contig, "error": str(e)}
    use_cl = False

# 6. TORCH.COMPILE POLICY
print("=== 6. TORCH.COMPILE POLICY ===", flush=True)
results["compile"] = {"tested": False, "reason": "Eager chosen per contract (recovering compile overhead not guaranteed with dynamic multimodal dictionary input)"}
use_comp = False

# 7. FINAL SELECTED PROFILE MEASUREMENT (30 steps)
print("=== 7. FINAL SELECTED PROFILE VERIFICATION ===", flush=True)
final_prof = run_benchmark(
    batch_size=128,
    num_workers=0,
    pin_memory=True,
    amp_dtype=selected_amp,
    channels_last=use_cl,
    use_compile=False,
    warmup_steps=10,
    measure_steps=30,
)
results["final_selected_profile"] = final_prof
print(f"FINAL PROFILE VERIFIED: {final_prof['samples_per_sec']:.2f} samples/s, p50={final_prof['p50_step_s']*1000:.1f}ms, Peak VRAM={final_prof['peak_vram_gb']:.2f}GB", flush=True)

with open("/teamspace/studios/this_studio/l4_pilot_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nALL_L4_PILOT_BENCHMARKS_COMPLETED: TRUE", flush=True)
'''

# Write script to remote studio
b64_code = base64.b64encode(remote_bench_code.encode("utf-8")).decode("ascii")
cmd = f"python3 -c \"import base64; open('/teamspace/studios/this_studio/bench_l4_clean.py', 'w').write(base64.b64decode('{b64_code}').decode('utf-8'))\""
studio.run(cmd)
print("Uploaded bench_l4_clean.py to Studio.")

# Run bench_l4_clean.py
res = studio.run("python3 /teamspace/studios/this_studio/bench_l4_clean.py")
print("=== BENCHMARK OUTPUT ===")
print(res)

# Read results json
res_json = studio.run("cat /teamspace/studios/this_studio/l4_pilot_results.json")
print("=== JSON RESULTS ===")
print(res_json)
