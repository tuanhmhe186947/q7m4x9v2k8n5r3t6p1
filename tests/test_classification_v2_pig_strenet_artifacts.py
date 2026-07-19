from __future__ import annotations

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.features.pig_strenet_artifacts import (
    availability_columns,
    build_pig_strenet_artifacts,
    compute_stabilized_difference_maps,
    model_x_columns,
)


def _frames(
    *,
    source_type: str = "legacy_recovered",
    missing_history: bool = False,
    actual_offset: int = 0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for relative_index in range(16 if source_type == "legacy_recovered" else 12):
        if missing_history and relative_index < 6:
            continue
        frame_index = actual_offset + relative_index
        scene = f"scene-{frame_index}"
        for actor_index in range(2):
            track = f"track-{actor_index}"
            unit = f"{source_type}|unit-{actor_index}"
            row: dict[str, object] = {
                "object_track_key": track,
                "temporal_unit_key": unit,
                "source_type": source_type,
                "dataset_id": "synthetic",
                "video_key": "video-1",
                "frame_index": frame_index,
                "frame_uid": f"{track}|frame-{frame_index}",
                "scene_frame_uid": scene,
                "relative_frame_index": relative_index,
                "label_window_start": 6 if source_type != "legacy_recovered" else 0,
                "label_window_end": 11 if source_type != "legacy_recovered" else 15,
                "human_review_complete": False,
                "behavior_label": "fight" if actor_index == 0 else "explore",
                "lineage_scope": "synthetic",
                "crop_path": "",
                "cx_n": 0.2 + 0.2 * actor_index + 0.01 * frame_index,
                "cy_n": 0.3,
                "speed_n_per_frame": 0.01 * (frame_index + 1),
                "displacement_n": 0.01,
                "abs_accel_n_per_frame2": 0.001,
                "abs_direction_change_rad": 0.1,
                "nearest_dist_n": 0.2 - 0.001 * frame_index,
                "nearest_dist_delta": -0.001,
                "approach_speed_n_per_frame": 0.001,
                "separation_speed_n_per_frame": 0.0,
                "nearest_pair_iou": 0.1,
                "nearest_pair_overlap_ratio": 0.2,
                "pair_contact_with_nearest": frame_index >= 8,
                "nearest_track_id": f"track-{1 - actor_index}",
                "timestamp_sec": relative_index / 6.0,
            }
            for roi in ("feeder", "drinker", "toy"):
                row.update(
                    {
                        f"roi_{roi}_available": True,
                        f"roi_{roi}_min_dist_n": 0.1 + 0.01 * frame_index,
                        f"roi_{roi}_max_overlap_ratio": 0.2,
                        f"roi_{roi}_max_iou": 0.1,
                        f"roi_{roi}_center_inside": False,
                        f"roi_{roi}_near": frame_index >= 4,
                        f"roi_{roi}_contact": frame_index >= 8,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def test_legacy_history_artifacts_preserve_event_mass_and_controls() -> None:
    artifacts = build_pig_strenet_artifacts(_frames())

    assert len(artifacts.pair_manifest) == 2
    assert artifacts.pair_manifest["derived_view"].eq("legacy_derived_6f").all()
    mass = artifacts.pair_manifest.groupby("native_event_id")["event_weight"].sum()
    assert mass.tolist() == [1.0, 1.0]
    assert len(artifacts.slot_manifest) == 24
    assert len(artifacts.roi_dynamics) == 72
    assert len(artifacts.control_matrix) == 16
    assert set(artifacts.control_matrix["control_id"]) == {
        "T0",
        "T1",
        "H0",
        "HA",
        "HS",
        "HR",
        "HRev",
        "PM",
    }
    assert artifacts.audit["target_selected_roi_used"] is False
    assert artifacts.audit["behavior_selected_partner_used"] is False
    assert not any("target_roi" in name for name in model_x_columns(artifacts.history_features))
    assert not set(availability_columns(artifacts.history_features)).intersection(
        model_x_columns(artifacts.history_features)
    )
    for _, group in artifacts.social_edges.groupby(
        ["pair_id", "slot_index", "actor_node_key"], sort=False
    ):
        assert group["neighbor_rank"].tolist() == [1, 2, 3]
    assert artifacts.social_edges["edge_available"].sum() == 24


def test_legacy_actual_frame_coordinates_are_not_relative_coordinates() -> None:
    artifacts = build_pig_strenet_artifacts(_frames(actual_offset=100))

    pair = artifacts.pair_manifest.iloc[0]
    assert pair["history_start_relative"] == 0
    assert pair["history_end_relative"] == 5
    assert pair["target_start_relative"] == 6
    assert pair["target_end_relative"] == 11
    assert pair["history_window_start_frame"] == 100
    assert pair["history_window_end_frame"] == 105
    assert pair["target_window_start_frame"] == 106
    assert pair["target_window_end_frame"] == 111
    assert artifacts.slot_manifest.iloc[0]["frame_index"] == 100


def test_multiple_legacy_views_conserve_native_event_mass() -> None:
    artifacts = build_pig_strenet_artifacts(
        _frames(),
        legacy_target_starts=(6, 8),
    )

    assert len(artifacts.pair_manifest) == 4
    assert artifacts.pair_manifest["event_weight"].eq(0.5).all()
    mass = artifacts.pair_manifest.groupby("native_event_id")["event_weight"].sum()
    assert mass.eq(1.0).all()


def test_cvat_history_is_missing_without_prior_frames() -> None:
    artifacts = build_pig_strenet_artifacts(
        _frames(source_type="cvat_tracking_xml", missing_history=True)
    )

    assert len(artifacts.pair_manifest) == 2
    assert artifacts.pair_manifest["derived_view"].eq("cvat_target_6f").all()
    assert not artifacts.pair_manifest["history_complete"].any()
    assert artifacts.pair_manifest["history_available_ratio"].eq(0.0).all()
    assert artifacts.history_features["history_expected_frame_count"].eq(6).all()
    assert artifacts.history_features["history_gap_count"].eq(6).all()
    xml_hr = artifacts.control_matrix.query("control_id == 'HR'")
    assert xml_hr["history_window_spec"].str.startswith("actual[").all()


def test_difference_maps_mask_missing_slots() -> None:
    crops = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    crops[1, 2:5, 2:5] = 255
    maps, summary, pair_valid = compute_stabilized_difference_maps(
        crops,
        np.array([True, True, False]),
    )

    assert maps.shape == (2, 8, 8)
    assert pair_valid.tolist() == [True, False]
    assert summary["pair_valid"].tolist() == [True, False]
    assert summary.loc[1, "diff_mean"] == 0.0
