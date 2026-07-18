from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pig_behavior.classification_v2.training.legacy_development_l6_pen_context_cache import (
    PEN_DIM,
    PEN_FEATURE_NAMES,
    PEN_STATIC_FEATURE_COUNT,
    materialize_pen_context_cache,
)


def _write_mask(path: Path) -> Path:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:18, 2:18] = 255
    assert cv2.imwrite(str(path), mask)
    return path


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _order() -> tuple[pd.DataFrame, pd.DataFrame]:
    window_id = "track-a|win=6|3-8"
    windows = pd.DataFrame(
        {
            "cache_row": [0],
            "window_id": [window_id],
            "temporal_unit_key": ["unit-a"],
            "l5_role": ["train"],
            "source_type": ["legacy_recovered"],
            "dataset_id": ["legacy_recovered_16f"],
            "lineage_scope": ["legacy-only-unreviewed-development"],
            "human_review_complete": [False],
            "sequence_length": [6],
        }
    )
    slots = pd.DataFrame(
        {
            "cache_row": [0] * 6,
            "window_id": [window_id] * 6,
            "slot_index": np.arange(6),
            "frame_uid": [f"frame-{value}" for value in range(3, 9)],
            "object_track_key": ["track-a"] * 6,
            "frame_index": np.arange(3, 9),
            "source_type": ["legacy_recovered"] * 6,
            "dataset_id": ["legacy_recovered_16f"] * 6,
            "lineage_scope": ["legacy-only-unreviewed-development"] * 6,
            "human_review_complete": [False] * 6,
        }
    )
    return windows, slots


def _frames() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for frame_index in range(9):
        center_x = 13.0 - frame_index * 0.75
        center_y = 10.0
        rows.append(
            {
                "source_type": "legacy_recovered",
                "dataset_id": "legacy_recovered_16f",
                "video_key": "video-a",
                "frame_uid": f"frame-{frame_index}",
                "frame_index": frame_index,
                "object_track_key": "track-a",
                "bbox_valid": True,
                "x1": center_x - 1.5,
                "y1": center_y - 1.5,
                "x2": center_x + 1.5,
                "y2": center_y + 1.5,
                "image_width": 20,
                "image_height": 20,
                "cx_n": center_x / 20.0,
                "cy_n": center_y / 20.0,
                "bw_n": 0.15,
                "bh_n": 0.15,
                "area_n": 0.0225,
                "aspect_ratio": 1.0,
                "actor_bbox_valid": True,
                "geometry_feature_valid": True,
                "spatiotemporal_feature_valid": True,
                "lineage_scope": "legacy-only-unreviewed-development",
                "human_review_complete": False,
            }
        )
    return pd.DataFrame.from_records(rows)


def test_pen_cache_separates_static_quality_and_window_pair_masks(
    tmp_path: Path,
) -> None:
    mask_path = _write_mask(tmp_path / "pen.png")
    windows, slots = _order()

    result = materialize_pen_context_cache(
        windows,
        slots,
        _frames(),
        mask_path=mask_path,
        expected_mask_sha256=_sha256(mask_path),
    )

    pen = result["pen"]
    feature_mask = result["feature_availability"]
    branch = result["availability"]
    pair = result["motion_availability"]
    assert pen.shape == (1, 6, PEN_DIM)
    assert feature_mask.shape == pen.shape
    assert branch.shape == (1, 6)
    assert branch.all()
    assert not pair[0, 0]
    assert pair[0, 1:].all()
    assert feature_mask[0, 0, :PEN_STATIC_FEATURE_COUNT].all()
    assert not feature_mask[0, 0, PEN_STATIC_FEATURE_COUNT:].any()
    assert np.count_nonzero(pen[0, 0, PEN_STATIC_FEATURE_COUNT:]) == 0
    approach = PEN_FEATURE_NAMES.index("pen_approach_speed_n_per_frame")
    assert pen[0, 1:, approach].max() > 0.0
    assert not result["slot_index"][["window_id", "slot_index"]].duplicated().any()
    audit = result["content_audit"]
    assert audit["binary_near_boundary_selected"] is False
    assert audit["paths_ids_review_labels_selected"] is False
    assert audit["motion_pair_available_slots"] == 5
    assert audit["errors"] == []


def test_pen_cache_invalid_quality_masks_static_and_adjacent_pairs(
    tmp_path: Path,
) -> None:
    mask_path = _write_mask(tmp_path / "pen.png")
    windows, slots = _order()
    frames = _frames()
    frames.loc[5, "bbox_valid"] = False

    result = materialize_pen_context_cache(
        windows,
        slots,
        frames,
        mask_path=mask_path,
        expected_mask_sha256=_sha256(mask_path),
    )

    # Source frame 5 is slot 2 in the 3--8 window. Its static values and the
    # pairs ending at slots 2 and 3 must not enter model X.
    assert not result["availability"][0, 2]
    assert not result["feature_availability"][0, 2].any()
    assert not result["motion_availability"][0, 2]
    assert not result["motion_availability"][0, 3]
    assert np.count_nonzero(result["pen"][0, 2]) == 0
