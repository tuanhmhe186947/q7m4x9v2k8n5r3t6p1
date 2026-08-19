import base64
import json
import lightning_sdk
from lightning_sdk import Machine

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

print(f"Current studio status: {studio.status}")
if str(studio.status).lower() != "running" and str(studio.status).lower() != "status.running":
    print("Starting studio on Machine.CPU...")
    studio.start(machine=Machine.CPU)
    print(f"Studio started. Status: {studio.status}, Machine: {studio.machine}")
else:
    print(f"Studio is already running on {studio.machine}")

script_code = r'''
import os, sys, json, re
import pandas as pd
import numpy as np

full_t6_dir = "/teamspace/uploads/classification_v2/full_t6_training_authority_20260817"
df_t6 = pd.read_csv(os.path.join(full_t6_dir, "full_t6_row_manifest.csv"), low_memory=False)
df_rel = pd.read_csv(os.path.join(full_t6_dir, "full_temporal_window_manifest_release.csv"), low_memory=False)

df_merged = df_t6.merge(df_rel, on="target_id", how="left", suffixes=("_t6", "_rel"))
df_val = df_merged[df_merged["split"] == "validation"].copy().reset_index(drop=True)

# Filter for CVAT validation only
cvat_val = df_val[df_val["source_type_t6"] == "cvat_tracking_xml"].copy().reset_index(drop=True)
print(f"Total CVAT Validation Windows: {len(cvat_val)}")
print(f"Unique cvat_behavior_run_id in CVAT Validation: {cvat_val['cvat_behavior_run_id'].nunique()}")

# For each run in CVAT validation:
runs_stats = []
for run_id, grp in cvat_val.groupby("cvat_behavior_run_id"):
    n_windows = len(grp)
    
    # Parse run start and end from cvat_behavior_run_id or target_id
    # Format typically: ...|run=START-END|... or parse from physical_frame_ids_json
    all_fids = []
    for raw_fids in grp["physical_frame_ids_json"]:
        fids = json.loads(raw_fids) if isinstance(raw_fids, str) else raw_fids
        all_fids.extend([f for f in fids if f >= 0])
    
    min_fid = min(all_fids) if all_fids else -1
    max_fid = max(all_fids) if all_fids else -1
    
    # Try parsing run=X-Y from run_id
    m = re.search(r"run=(\d+)-(\d+)", str(run_id))
    if m:
        run_start = int(m.group(1))
        run_end = int(m.group(2))
        duration_frames = (run_end - run_start + 1)
    else:
        run_start = min_fid
        run_end = max_fid
        duration_frames = (max_fid - min_fid + 1) if (max_fid >= min_fid >= 0) else (n_windows * 6)
        
    first_row = grp.iloc[0]
    runs_stats.append({
        "cvat_behavior_run_id": run_id,
        "dataset_id": first_row.get("dataset_id_t6", first_row.get("dataset_id_rel")),
        "video_key": first_row.get("video_key_t6", first_row.get("video_key_rel")),
        "track_id": first_row.get("object_track_key_t6", first_row.get("object_track_key_rel")),
        "behavior_label": first_row.get("behavior_t6", first_row.get("behavior_rel")),
        "run_start": run_start,
        "run_end": run_end,
        "number_of_T6_windows": n_windows,
        "duration_frames": duration_frames,
    })

df_runs = pd.DataFrame(runs_stats)

# Windows per run statistics
w_vals = df_runs["number_of_T6_windows"].to_numpy()
w_min = int(np.min(w_vals))
w_p25 = float(np.percentile(w_vals, 25))
w_med = float(np.percentile(w_vals, 50))
w_p75 = float(np.percentile(w_vals, 75))
w_p90 = float(np.percentile(w_vals, 90))
w_p95 = float(np.percentile(w_vals, 95))
w_p99 = float(np.percentile(w_vals, 99))
w_max = int(np.max(w_vals))

# Duration frames per run statistics
d_vals = df_runs["duration_frames"].to_numpy()
d_min = int(np.min(d_vals))
d_p25 = float(np.percentile(d_vals, 25))
d_med = float(np.percentile(d_vals, 50))
d_p75 = float(np.percentile(d_vals, 75))
d_p90 = float(np.percentile(d_vals, 90))
d_p95 = float(np.percentile(d_vals, 95))
d_p99 = float(np.percentile(d_vals, 99))
d_max = int(np.max(d_vals))

print("\n--- WINDOWS PER RUN DISTRIBUTION ---")
print(f"min: {w_min}")
print(f"p25: {w_p25:.1f}")
print(f"median: {w_med:.1f}")
print(f"p75: {w_p75:.1f}")
print(f"p90: {w_p90:.1f}")
print(f"p95: {w_p95:.1f}")
print(f"p99: {w_p99:.1f}")
print(f"max: {w_max}")

print("\n--- DURATION FRAMES PER RUN DISTRIBUTION ---")
print(f"min: {d_min}")
print(f"p25: {d_p25:.1f}")
print(f"median: {d_med:.1f}")
print(f"p75: {d_p75:.1f}")
print(f"p90: {d_p90:.1f}")
print(f"p95: {d_p95:.1f}")
print(f"p99: {d_p99:.1f}")
print(f"max: {d_max}")

# Top 20 longest runs
df_top20 = df_runs.sort_values(by="number_of_T6_windows", ascending=False).head(20).reset_index(drop=True)
print("\n--- TOP 20 LONGEST RUNS ---")
for idx, r in df_top20.iterrows():
    print(f"{idx+1}. dataset={r['dataset_id']}, video={r['video_key']}, track={r['track_id']}, behavior={r['behavior_label']}, run={r['run_start']}-{r['run_end']}, windows={r['number_of_T6_windows']}, duration_frames={r['duration_frames']}")

out_results = {
    "cvat_val_windows": len(cvat_val),
    "cvat_runs_count": len(df_runs),
    "windows_per_run": {
        "min": w_min,
        "p25": w_p25,
        "median": w_med,
        "p75": w_p75,
        "p90": w_p90,
        "p95": w_p95,
        "p99": w_p99,
        "max": w_max,
    },
    "duration_frames_per_run": {
        "min": d_min,
        "p25": d_p25,
        "median": d_med,
        "p75": d_p75,
        "p90": d_p90,
        "p95": d_p95,
        "p99": d_p99,
        "max": d_max,
    },
    "top20_longest_runs": df_top20.to_dict(orient="records"),
}

with open("/teamspace/studios/this_studio/cvat_run_audit_results.json", "w") as f:
    json.dump(out_results, f, indent=2)
'''

b64 = base64.b64encode(script_code.encode("utf-8")).decode("ascii")
cmd = f"python3 -c \"import base64; open('/teamspace/studios/this_studio/run_cvat_run_audit.py', 'w').write(base64.b64decode('{b64}').decode('utf-8'))\""
studio.run(cmd)
print("Uploaded run_cvat_run_audit.py.")
out = studio.run("python3 /teamspace/studios/this_studio/run_cvat_run_audit.py")
print("=== REMOTE OUTPUT ===")
print(out)
