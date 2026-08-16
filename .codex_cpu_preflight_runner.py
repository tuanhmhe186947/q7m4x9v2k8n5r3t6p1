import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.datasets.resolution_pipeline import (
    build_inner_resolution_binding_from_dataframes,
)
from pig_behavior.classification_v2.training.stage1_rgb_binding import (
    resolve_stage1_execution_rgb_binding,
)


base = Path("/home/zeus/c2v2_overnight_20260812")
runtime = base / "runtime"
host_dir = runtime / (
    "outputs/classification_v2/post_s1_cpu_host_bindings_20260812/"
    "t6_binding_host_bound_v4/post_s1_cpu_20260812_t6"
)
wrapper = json.loads((host_dir / "post_s1_host_binding.json").read_text())
data_path = host_dir / wrapper["artifacts"]["data_bindings"]
requested = pd.read_csv(
    runtime
    / (
        "outputs/classification_v2/post_s1_cpu_host_bindings_20260812/"
        "t6_binding_host_bound_v3h/post_s1_cpu_20260812_t6/stage1_window_context.csv"
    ),
    low_memory=False,
)
requested = requested.rename(columns={"stage1_role": "primary_s1_role"})
rgb = resolve_stage1_execution_rgb_binding(
    data_bindings_path=data_path,
    requested_roles=requested[["window_id", "primary_s1_role"]],
    authority_sha256=(
        "9daf6a3bda89678c2b3ddd6ba9a0132fa1be16b583bfafb1cc76eb610ff8b1e4"
    ),
    provenance_hashes=wrapper["scientific_identity"].get(
        "t6_population_provenance_hashes", {}
    ),
    view="T6",
    sequence_length=6,
)
frames = pd.read_csv(rgb.frame_context_path, low_memory=False)
windows = pd.read_csv(rgb.window_context_path, low_memory=False)
selection = requested.copy()
window_rows = {str(value): index for index, value in enumerate(windows["window_id"])}
selection["window_row_index"] = selection["window_id"].astype(str).map(window_rows)
selection = selection.dropna(subset=["window_row_index"]).copy()
selection["window_row_index"] = selection["window_row_index"].astype(int)
selection["source_type"] = selection["window_id"].map(
    windows.set_index("window_id")["source_type"]
)
selection["view_type"] = selection["window_id"].map(
    windows.set_index("window_id")["view_type"]
)
selection["behavior_window_label"] = selection.get(
    "behavior_window_label", pd.Series("unknown", index=selection.index)
)
selection["window_valid_for_main_train"] = True
selection["primary_s1_eligible"] = True
binding = build_inner_resolution_binding_from_dataframes(
    frames=frames,
    windows=windows,
    selection=selection[
        [
            "window_row_index",
            "window_id",
            "view_type",
            "source_type",
            "behavior_window_label",
            "window_valid_for_main_train",
            "primary_s1_role",
            "primary_s1_eligible",
        ]
    ],
    media_root=Path("/teamspace/studios/this_studio/pig_e0_r3/inputs"),
    expected_window_count=39454,
    expected_observation_count=201792,
)
reports = {}
for resolution in (64, 128, 160):
    dataset = binding.build_dataset(resolution, image_cache_size=0)
    try:
        for source in ("cvat_tracking_xml", "legacy_recovered"):
            indices = binding.windows.index[
                binding.windows["source_type"].astype(str).eq(source)
            ].tolist()[:16]
            checks = []
            for index in indices:
                item = dataset[int(index)]
                checks.append(
                    {
                        "window_id": item["window_id"],
                        "shape": [int(value) for value in item["image"].shape],
                        "errors": item["errors"],
                        "observed_frames": int(item["observed_mask"].sum().item()),
                    }
                )
            expected_shape = [6, 3, resolution, resolution]
            passed = bool(
                len(checks) == 16
                and all(row["shape"] == expected_shape for row in checks)
                and all(not row["errors"] for row in checks)
                and all(row["observed_frames"] == 6 for row in checks)
            )
            reports[f"{source}_r{resolution}"] = {
                "status": "PASS" if passed else "FAIL",
                "sample_windows": len(checks),
                "sample": checks,
            }
    finally:
        dataset.close()
print(json.dumps({"status": "PASS" if all(item["status"] == "PASS" for item in reports.values()) else "FAIL", "binding_windows": binding.window_count, "binding_observations": binding.observation_count, "reports": reports}, indent=2))
