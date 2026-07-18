"""Tests for mask- and perspective-aware tracking strata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pig_behavior.evaluation.tracking.hard_scene import (
    HardSceneEvalConfig,
    run_hard_scene_evaluation,
)
from pig_behavior.evaluation.tracking.spatial_strata import (
    PERSPECTIVE_AXIS,
    SpatialStrataThresholds,
    calibrate_perspective_small,
    load_spatial_scene_context,
    spatial_features_for_bbox,
    summarize_spatial_strata,
)


def _write_mask(path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:90, 10:90] = 255
    assert cv2.imwrite(str(path), mask)


def _write_cvat_xml(
    path: Path,
    tracks: dict[int, dict[int, list[float]]],
) -> None:
    parts = [
        "<annotations><meta><task><name>spatial_test</name>"
        "<size>5</size></task></meta>"
    ]
    for track_id, by_frame in tracks.items():
        parts.append(f'<track id="{track_id}" label="Pig_{track_id}">')
        for frame, box in sorted(by_frame.items()):
            x1, y1, x2, y2 = box
            hidden = "Yes" if frame == 2 and track_id == 1 else "No"
            parts.append(
                f'<box frame="{frame}" xtl="{x1}" ytl="{y1}" '
                f'xbr="{x2}" ybr="{y2}" outside="0">'
                f'<attribute name="ID">ID_{track_id}</attribute>'
                f'<attribute name="Hidden">{hidden}</attribute></box>'
            )
        parts.append("</track>")
    parts.append("</annotations>")
    path.write_text("".join(parts), encoding="utf-8")


def test_spatial_features_follow_mask_and_left_to_right_perspective(
    tmp_path: Path,
) -> None:
    mask_path = tmp_path / "mask.png"
    _write_mask(mask_path)
    context = load_spatial_scene_context(mask_path)
    thresholds = SpatialStrataThresholds()

    near_left = spatial_features_for_bbox(
        context,
        (10.0, 40.0, 20.0, 60.0),
        thresholds,
    )
    far_right = spatial_features_for_bbox(
        context,
        (70.0, 40.0, 80.0, 60.0),
        thresholds,
    )

    assert near_left["is_near_wall"] is True
    assert far_right["is_far_camera_proxy"] is True
    assert float(far_right["far_camera_score"]) > float(
        near_left["far_camera_score"]
    )
    assert far_right["perspective_axis"] == PERSPECTIVE_AXIS
    assert far_right["is_absolute_small"] is True


def test_perspective_small_is_residual_after_expected_size_fit() -> None:
    rows = [
        {"pen_relative_x": 0.10, "bbox_area_ratio": 0.08},
        {"pen_relative_x": 0.30, "bbox_area_ratio": 0.06},
        {"pen_relative_x": 0.70, "bbox_area_ratio": 0.03},
        {"pen_relative_x": 0.90, "bbox_area_ratio": 0.005},
    ]
    calibration = calibrate_perspective_small(
        rows,
        SpatialStrataThresholds(perspective_small_quantile=0.25),
    )

    assert calibration["sample_count"] == 4
    assert float(calibration["slope"]) < 0.0
    assert rows[-1]["is_perspective_residual_small"] is True
    assert "perspective_area_log_residual" in rows[0]


def test_spatial_strata_report_geometry_quality_at_primary_iou(
    tmp_path: Path,
) -> None:
    mask_path = tmp_path / "mask.png"
    _write_mask(mask_path)
    context = load_spatial_scene_context(mask_path)
    rows = [
        {
            "is_near_wall": True,
            "is_matched": True,
            "is_id_correct": True,
            "is_id_wrong": False,
            "is_missing": False,
            "matched_iou": 0.90,
        },
        {
            "is_near_wall": True,
            "is_matched": True,
            "is_id_correct": False,
            "is_id_wrong": True,
            "is_missing": False,
            "matched_iou": 0.49,
        },
        {
            "is_near_wall": True,
            "is_matched": False,
            "is_id_correct": False,
            "is_id_wrong": False,
            "is_missing": True,
            "matched_iou": float("nan"),
        },
    ]

    summary = summarize_spatial_strata(
        rows,
        SpatialStrataThresholds(),
        {},
        context,
        source_match_iou_threshold=0.30,
        quality_iou_threshold=0.50,
    )
    near_wall = summary["near_wall"]

    assert summary["all_instances"] == near_wall
    assert near_wall["instance_count"] == 3
    assert near_wall["quality_match_count"] == 1
    assert near_wall["quality_match_rate"] == 0.333333
    assert near_wall["low_iou_match_count"] == 1
    assert near_wall["low_iou_match_rate"] == 0.333333
    assert near_wall["mean_matched_iou"] == 0.695
    assert near_wall["median_matched_iou"] == 0.695
    assert near_wall["p10_matched_iou"] == 0.531


def test_hard_scene_spatial_strata_are_opt_in_and_no_mp4(
    tmp_path: Path,
) -> None:
    tracks = {
        1: {frame: [10, 40, 20, 60] for frame in range(5)},
        2: {frame: [70, 40, 80, 60] for frame in range(5)},
    }
    gt_xml = tmp_path / "gt.xml"
    pred_xml = tmp_path / "pred.xml"
    mask_path = tmp_path / "mask.png"
    _write_cvat_xml(gt_xml, tracks)
    _write_cvat_xml(pred_xml, tracks)
    _write_mask(mask_path)

    output_dir = tmp_path / "spatial_output"
    metrics = run_hard_scene_evaluation(
        HardSceneEvalConfig(
            gt_xml=gt_xml,
            pred_xml=pred_xml,
            output_dir=output_dir,
            include_hidden=True,
            mask_path=mask_path,
        )
    )

    rows = pd.read_csv(output_dir / "per_frame_identity_analysis.csv")
    stored = json.loads(
        (output_dir / "hard_scene_metrics.json").read_text(encoding="utf-8")
    )
    assert len(rows) == 10
    assert "is_near_wall" in rows.columns
    assert "is_far_camera_proxy" in rows.columns
    assert "is_perspective_residual_small" in rows.columns
    assert metrics["spatial_strata"]["enabled"] is True
    assert metrics["spatial_strata"]["mask_sha256"] == hashlib.sha256(
        mask_path.read_bytes()
    ).hexdigest()
    assert metrics["spatial_strata"]["mask_width"] == 100
    assert metrics["spatial_strata"]["mask_height"] == 100
    assert metrics["spatial_strata"]["all_instances"][
        "quality_match_count"
    ] == 10
    assert metrics["spatial_strata"]["all_instances"][
        "mean_matched_iou"
    ] == 1.0
    assert stored["config"]["include_hidden"] is True
    assert stored["spatial_strata"]["perspective_axis"] == PERSPECTIVE_AXIS
    assert not list(output_dir.rglob("*.mp4"))


def test_hard_scene_without_mask_preserves_legacy_spatial_schema(
    tmp_path: Path,
) -> None:
    tracks = {1: {frame: [10, 40, 20, 60] for frame in range(5)}}
    gt_xml = tmp_path / "gt.xml"
    pred_xml = tmp_path / "pred.xml"
    _write_cvat_xml(gt_xml, tracks)
    _write_cvat_xml(pred_xml, tracks)

    output_dir = tmp_path / "legacy_output"
    metrics = run_hard_scene_evaluation(
        HardSceneEvalConfig(
            gt_xml=gt_xml,
            pred_xml=pred_xml,
            output_dir=output_dir,
        )
    )
    rows = pd.read_csv(output_dir / "per_frame_identity_analysis.csv")

    assert "spatial_strata" not in metrics
    assert "is_near_wall" not in rows.columns


def test_spatial_thresholds_fail_closed() -> None:
    with pytest.raises(ValueError, match="far_pen_relative_x"):
        SpatialStrataThresholds(far_pen_relative_x=1.0)
