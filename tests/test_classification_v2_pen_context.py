from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.features.pen_context import (
    DEFAULT_PEN_MASK_SHA256,
    PEN_CONTEXT_MODEL_FEATURE_COLUMNS,
    audit_pen_context_features,
    build_pen_context_features,
    summarize_pen_context,
)
from pig_behavior.classification_v2.features.spatial_schema import (
    SpatialSchemaError,
)
from pig_behavior.classification_v2.spatial_sequence_export import (
    export_spatial_sequences,
)


def _write_mask(path: Path, *, width: int = 20, height: int = 20) -> Path:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[2 : height - 2, 2 : width - 2] = 255
    assert cv2.imwrite(str(path), mask)
    return path


def _frame_rows() -> pd.DataFrame:
    centers = [(7.0, 10.0), (5.0, 10.0), (3.0, 10.0), (3.0, 12.0)]
    rows: list[dict[str, object]] = []
    for frame_index, (center_x, center_y) in enumerate(centers):
        rows.append(
            {
                "source_type": "legacy_recovered",
                "dataset_id": "legacy",
                "video_key": "video-a",
                "frame_uid": f"frame-{frame_index}",
                "frame_index": frame_index,
                "timestamp_sec": frame_index / 30.0,
                "object_track_key": "legacy|video-a|track=1",
                "temporal_unit_key": "legacy|video-a|track=1|unit=0-15",
                "bbox_valid": True,
                "x1": center_x - 2.0,
                "y1": center_y - 2.0,
                "x2": center_x + 2.0,
                "y2": center_y + 2.0,
                "image_width": 20,
                "image_height": 20,
                "cx_n": center_x / 20.0,
                "cy_n": center_y / 20.0,
                "bw_n": 0.2,
                "bh_n": 0.2,
                "area_n": 0.04,
                "aspect_ratio": 1.0,
                "geometry_feature_valid": True,
                "spatiotemporal_feature_valid": True,
            }
        )
    return pd.DataFrame(rows)


def test_pen_context_preserves_rows_and_encodes_boundary_motion(
    tmp_path: Path,
) -> None:
    mask_path = _write_mask(tmp_path / "pen-mask.png")
    source = _frame_rows()

    result = build_pen_context_features(source, mask_path=mask_path)

    assert len(result) == len(source)
    assert result["frame_uid"].tolist() == source["frame_uid"].tolist()
    assert result["pen_context_available"].all()
    assert result["pen_center_inside"].all()
    assert result.loc[2, "pen_bbox_inside_ratio"] < 1.0
    assert bool(result.loc[2, "pen_near_boundary"])
    assert result.loc[1, "pen_approach_speed_n_per_frame"] > 0.0
    assert result.loc[2, "pen_approach_speed_n_per_frame"] > 0.0
    assert result.loc[3, "pen_parallel_speed_n_per_frame"] > 0.0
    assert result.loc[3, "pen_approach_speed_n_per_frame"] == pytest.approx(
        0.0
    )


def test_pen_context_invalid_bbox_is_auditable_without_row_drop(
    tmp_path: Path,
) -> None:
    mask_path = _write_mask(tmp_path / "pen-mask.png")
    source = _frame_rows()
    source.loc[1, "bbox_valid"] = False

    result = build_pen_context_features(source, mask_path=mask_path)

    assert len(result) == len(source)
    assert not bool(result.loc[1, "pen_context_available"])
    assert pd.isna(result.loc[1, "pen_center_signed_distance_n"])
    assert not bool(result.loc[1, "pen_motion_context_valid"])
    assert not bool(result.loc[2, "pen_motion_context_valid"])


def test_pen_context_resizes_calibration_mask_with_nearest_neighbor(
    tmp_path: Path,
) -> None:
    mask_path = _write_mask(tmp_path / "pen-mask.png")
    source = _frame_rows().iloc[[0]].copy()
    source[["x1", "y1", "x2", "y2"]] *= 2.0
    source[["cx_n", "cy_n", "bw_n", "bh_n"]] = [0.35, 0.25, 0.2, 0.2]
    source[["image_width", "image_height"]] = [40, 40]

    result = build_pen_context_features(source, mask_path=mask_path)

    assert bool(result.iloc[0]["pen_context_available"])
    assert bool(result.iloc[0]["pen_center_inside"])


def test_pen_context_fails_closed_for_missing_mask(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Pen mask does not exist"):
        build_pen_context_features(
            _frame_rows(),
            mask_path=tmp_path / "missing.png",
        )


def test_pen_context_fails_before_build_for_wrong_mask_hash(
    tmp_path: Path,
) -> None:
    mask_path = _write_mask(tmp_path / "pen-mask.png")

    with pytest.raises(ValueError, match="Pen mask SHA-256 mismatch"):
        build_pen_context_features(
            _frame_rows(),
            mask_path=mask_path,
            expected_mask_sha256="0" * 64,
        )


def test_pen_context_audit_binds_mask_hash_and_feature_contract(
    tmp_path: Path,
) -> None:
    mask_path = _write_mask(tmp_path / "pen-mask.png")
    result = build_pen_context_features(_frame_rows(), mask_path=mask_path)

    audit = audit_pen_context_features(
        result,
        mask_path=mask_path,
        input_rows=4,
    )

    assert audit["errors"] == []
    assert audit["row_count_preserved"] is True
    assert len(audit["mask"]["sha256"]) == 64
    assert audit["mask"]["threshold"] == 127
    assert audit["model_feature_columns"] == list(
        PEN_CONTEXT_MODEL_FEATURE_COLUMNS
    )
    assert "pen_context_available" not in audit["model_feature_columns"]
    assert "pen_boundary_inward_normal_x" not in audit["model_feature_columns"]

    mismatch = audit_pen_context_features(
        result,
        mask_path=mask_path,
        input_rows=4,
        expected_mask_sha256="0" * 64,
    )
    assert any(
        error.startswith("pen_mask_sha256_mismatch=")
        for error in mismatch["errors"]
    )


def test_pen_context_window_summary_separates_transient_and_persistent_contact(
    tmp_path: Path,
) -> None:
    mask_path = _write_mask(tmp_path / "pen-mask.png")
    result = build_pen_context_features(_frame_rows(), mask_path=mask_path)

    summary = summarize_pen_context(result)

    assert summary["pen_near_boundary_ratio_window"] == pytest.approx(0.5)
    assert summary["pen_near_boundary_longest_run_ratio_window"] == pytest.approx(
        0.5
    )
    assert summary["pen_near_boundary_episode_count_window"] == 1
    assert summary["pen_approach_speed_max_window"] > 0.0
    assert summary["pen_parallel_speed_max_window"] > 0.0


def test_current_spatial_export_rejects_experimental_pen_group(
    tmp_path: Path,
) -> None:
    mask_path = _write_mask(tmp_path / "pen-mask.png")
    frames = build_pen_context_features(_frame_rows(), mask_path=mask_path)
    windows = pd.DataFrame(
        {
            "window_id": ["legacy|video-a|track=1|win=4|0-3"],
            "object_track_key": ["legacy|video-a|track=1"],
            "window_start_frame": [0],
            "window_end_frame": [3],
            "window_length_frames": [4],
            "feature_computation_grain": ["FINAL_VIEW_FEATURES"],
            "pair_scope_key": ["legacy|video-a|track=1|win=4|0-3"],
            "view_type": ["T4_contiguous"],
            "sampling_pattern": ["contiguous"],
            "selected_frame_offsets": ["[0,1,2,3]"],
            "selected_frame_indices": ["[0,1,2,3]"],
            "selected_timestamps_seconds": [
                "[0.0,0.03333333333333333,0.06666666666666667,0.1]"
            ],
            "pair_delta_frames": ["[1,1,1]"],
            "pair_delta_seconds": [
                "[0.03333333333333333,0.03333333333333333,"
                "0.03333333333333333]"
            ],
            "pair_recomputed_for_view": [True],
            "aggregate_recomputed_for_view": [True],
        }
    )

    with pytest.raises(
        SpatialSchemaError,
        match="unexpected_spatial_groups",
    ):
        export_spatial_sequences(
            windows,
            frames,
            feature_schema={
                "pen_boundary_context": list(
                    PEN_CONTEXT_MODEL_FEATURE_COLUMNS
                ),
            },
        )


def test_pen_context_is_not_in_the_default_trainer_whitelist() -> None:
    trainer = json.loads(
        Path("configs/classification_v2/trainer_contract_v2.json").read_text(
            encoding="utf-8"
        )
    )
    semantics = json.loads(
        Path("configs/classification_v2/feature_semantics_v2.json").read_text(
            encoding="utf-8"
        )
    )

    assert "pen_boundary_context" not in trainer[
        "spatial_sequence_feature_whitelist"
    ]
    pen_semantics = semantics["spatial_arrays"]["pen_boundary_context"]
    assert pen_semantics["model_input_role"] == "model_input_candidate"
    assert pen_semantics["model_input_allowed"] is True


def test_pen_context_ablation_is_hash_bound_and_motion_controlled() -> None:
    contract = json.loads(
        Path("configs/classification_v2/pen_context_ablation_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert contract["mask_contract"]["sha256"] == DEFAULT_PEN_MASK_SHA256
    comparison = contract["paired_comparison"]
    assert comparison["reference_model_mode"] == "actor_geometry_motion"
    assert comparison["candidate_model_mode"] == "actor_geometry_motion_pen"
    assert comparison["only_changed_spatial_group"] == "pen_boundary_context"
