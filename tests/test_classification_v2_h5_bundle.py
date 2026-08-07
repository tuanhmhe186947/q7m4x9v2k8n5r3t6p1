"""Focused H5 bundle tests using the current 46D exporter and loader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.features.motion_schema import (
    MOTION_FEATURE_NAMES,
    MOTION_REQUIRED_MASKS,
    MOTION_SCHEMA_DIMENSION,
    MOTION_SCHEMA_HASH,
    MOTION_SCHEMA_ID,
    MOTION_SCHEMA_VERSION,
)
from pig_behavior.classification_v2.features.spatial_schema import (
    SPATIAL_PREDICTIVE_FEATURES,
    SPATIAL_PREDICTIVE_GROUP_NAMES,
    load_current_spatial_tensor_bundle,
)
from pig_behavior.classification_v2.spatial_sequence_export import (
    export_spatial_sequences,
)
from pig_behavior.classification_v2.temporal_views.h5_bundle import (
    H5_SAMPLING_PATTERN,
    H5_VIEW_TYPE,
    build_h5_window_manifest,
)


def _cohort() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "h5_target_id": ["cvat-target", "legacy-target"],
            "source_type": ["cvat_tracking_xml", "legacy_recovered"],
            "object_track_key": ["actor-cvat", "actor-legacy"],
            "history_frame_indices_json": ["[5,6,7,8,9]", "[20,21,22,23,24]"],
            "target_frame_indices_json": ["[10,11,12,13,14,15]", "[25,26,27,28,29,30]"],
            "h5_valid": [True, True],
        }
    )


def _frames() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for track, source, indices in (
        ("actor-cvat", "cvat_tracking_xml", range(5, 16)),
        ("actor-legacy", "legacy_recovered", range(20, 31)),
    ):
        for frame_index in indices:
            row: dict[str, object] = {
                "object_track_key": track,
                "source_type": source,
                "frame_index": frame_index,
                "timestamp_sec": frame_index / 30.0,
                "cx_n": 0.2,
                "cy_n": 0.4,
                "bw_n": 0.2,
                "bh_n": 0.1,
                "area_n": 0.02,
                "aspect_ratio": 2.0,
                "bbox_valid": True,
                "actor_bbox_valid": True,
                "geometry_feature_valid": True,
                "spatiotemporal_feature_valid": True,
                "roi_feeder_available": False,
                "roi_drinker_available": False,
                "roi_toy_available": False,
                "social_neighbor_available": False,
                "nearest_partner_key": "",
                "velocity_sample_time_sec": np.nan,
                "acceleration_delta_t_sec": np.nan,
                "motion_schema_id": MOTION_SCHEMA_ID,
                "motion_schema_version": MOTION_SCHEMA_VERSION,
                "motion_schema_dimension": MOTION_SCHEMA_DIMENSION,
                "motion_schema_feature_names": json.dumps(
                    list(MOTION_FEATURE_NAMES), separators=(",", ":")
                ),
                "motion_schema_hash": MOTION_SCHEMA_HASH,
            }
            for group in SPATIAL_PREDICTIVE_GROUP_NAMES:
                for name in SPATIAL_PREDICTIVE_FEATURES[group]:
                    row.setdefault(name, 0.0)
            for name in MOTION_REQUIRED_MASKS:
                row[name] = False
            row["motion_feature_available"] = True
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def test_h5_bundle_preserves_order_and_excludes_history_labels(
    tmp_path: Path,
) -> None:
    frames = _frames()
    windows = build_h5_window_manifest(_cohort(), frames)

    assert windows["window_id"].tolist() == ["cvat-target", "legacy-target"]
    assert windows["view_type"].eq(H5_VIEW_TYPE).all()
    assert windows["sampling_pattern"].eq(H5_SAMPLING_PATTERN).all()
    assert windows["history_length"].eq(5).all()
    assert windows["target_length"].eq(6).all()
    assert "behavior_target_label" not in windows
    assert json.loads(windows.loc[1, "selected_frame_indices"]) == list(range(20, 31))

    exported = export_spatial_sequences(windows, frames)
    npz_path = tmp_path / "X_spatial_sequences.npz"
    audit_path = tmp_path / "spatial_sequence_audit.json"
    np.savez_compressed(npz_path, **exported.arrays)
    audit_path.write_text(json.dumps(exported.audit), encoding="utf-8")
    arrays, audit = load_current_spatial_tensor_bundle(npz_path, audit_path)

    assert arrays["motion_delta"].shape[:2] == (2, 11)
    assert audit["forbidden_selected"] == []
    assert not any(
        "behavior" in name or "label" in name
        for names in audit["feature_names"].values()
        for name in names
    )


def test_existing_t6_spatial_export_contract_is_unchanged() -> None:
    frames = _frames().loc[lambda frame: frame["object_track_key"].eq("actor-cvat")]
    windows = pd.DataFrame(
        {
            "window_id": ["t6-control"],
            "object_track_key": ["actor-cvat"],
            "window_start_frame": [10],
            "window_end_frame": [15],
            "window_length_frames": [6],
            "feature_computation_grain": ["FINAL_VIEW_FEATURES"],
            "pair_scope_key": ["t6-control"],
            "view_type": ["T6_contiguous"],
            "sampling_pattern": ["contiguous"],
            "selected_frame_offsets": ["[0,1,2,3,4,5]"],
            "selected_frame_indices": ["[10,11,12,13,14,15]"],
            "selected_timestamps_seconds": [
                json.dumps([frame / 30.0 for frame in range(10, 16)])
            ],
            "pair_delta_frames": ["[1,1,1,1,1]"],
            "pair_delta_seconds": [json.dumps([1.0 / 30.0] * 5)],
            "pair_recomputed_for_view": [True],
            "aggregate_recomputed_for_view": [True],
        }
    )

    exported = export_spatial_sequences(windows, frames)

    assert exported.arrays["motion_delta"].shape[:2] == (1, 6)
    assert exported.audit["errors"] == []
