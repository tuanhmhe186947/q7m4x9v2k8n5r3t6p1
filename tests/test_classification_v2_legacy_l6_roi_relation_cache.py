from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.training import (
    legacy_development_l6_roi_relation_cache as roi_cache,
)
from pig_behavior.classification_v2.training.legacy_development_l6_roi_relation_cache import (
    LINEAGE_SCOPE,
    ROI_RELATION_DIM,
    ROI_RELATION_FEATURE_NAMES,
    SEQUENCE_LENGTH,
    _validate_relation_bounds,
    materialize_roi_relation_cache,
)

EXPECTED_FEATURES = (
    "roi_feeder_min_dist_n",
    "roi_feeder_max_overlap_ratio",
    "roi_feeder_max_iou",
    "roi_feeder_center_inside",
    "roi_feeder_near",
    "roi_feeder_contact",
    "roi_drinker_min_dist_n",
    "roi_drinker_max_overlap_ratio",
    "roi_drinker_max_iou",
    "roi_drinker_center_inside",
    "roi_drinker_near",
    "roi_drinker_contact",
    "roi_toy_min_dist_n",
    "roi_toy_max_overlap_ratio",
    "roi_toy_max_iou",
    "roi_toy_center_inside",
    "roi_toy_near",
    "roi_toy_contact",
)


def _synthetic_order() -> tuple[pd.DataFrame, pd.DataFrame]:
    window_id = "track-a|win=6|0-5"
    window_index = pd.DataFrame(
        {
            "cache_row": [0],
            "window_id": [window_id],
            "temporal_unit_key": ["unit-a"],
            "l5_role": ["train"],
            "source_type": ["legacy_recovered"],
            "dataset_id": ["legacy_recovered_16f"],
            "lineage_scope": [LINEAGE_SCOPE],
            "human_review_complete": [False],
            "ordered_frame_uid_sha256": ["a" * 64],
            "sequence_length": [SEQUENCE_LENGTH],
        }
    )
    slot_index = pd.DataFrame(
        {
            "cache_row": [0] * SEQUENCE_LENGTH,
            "window_id": [window_id] * SEQUENCE_LENGTH,
            "slot_index": np.arange(SEQUENCE_LENGTH),
            "frame_uid": [f"frame-{index}" for index in range(SEQUENCE_LENGTH)],
            "object_track_key": ["track-a"] * SEQUENCE_LENGTH,
            "frame_index": np.arange(SEQUENCE_LENGTH),
            "source_type": ["legacy_recovered"] * SEQUENCE_LENGTH,
            "dataset_id": ["legacy_recovered_16f"] * SEQUENCE_LENGTH,
            "geometry_available": [True] * SEQUENCE_LENGTH,
            "lineage_scope": [LINEAGE_SCOPE] * SEQUENCE_LENGTH,
            "human_review_complete": [False] * SEQUENCE_LENGTH,
        }
    )
    return window_index, slot_index


def _synthetic_frames() -> pd.DataFrame:
    rows = SEQUENCE_LENGTH
    frame = pd.DataFrame(
        {
            "frame_uid": [f"frame-{index}" for index in range(rows)],
            "source_type": ["legacy_recovered"] * rows,
            "dataset_id": ["legacy_recovered_16f"] * rows,
            "lineage_scope": [LINEAGE_SCOPE] * rows,
            "human_review_complete": [False] * rows,
            "roi_feeder_available": [True] * rows,
            "roi_drinker_available": [True] * rows,
            "roi_toy_available": [True] * rows,
            "target_selected_roi": np.full(rows, 999.0),
            "behavior": ["forbidden"] * rows,
            "unit_aggregate": np.full(rows, 998.0),
            "geometry_value": np.full(rows, 997.0),
            "motion_value": np.full(rows, 996.0),
        }
    )
    for feature_index, name in enumerate(ROI_RELATION_FEATURE_NAMES):
        if name.endswith("_min_dist_n"):
            frame[name] = np.arange(rows, dtype=np.float64) / 10.0
        elif name.endswith(("_max_overlap_ratio", "_max_iou")):
            frame[name] = (feature_index + 1) / 100.0
        else:
            frame[name] = np.arange(rows) % 2
    return frame


def _patch_expected_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(roi_cache, "EXPECTED_MODEL_WINDOWS", 1)
    monkeypatch.setattr(roi_cache, "EXPECTED_MODEL_SLOTS", SEQUENCE_LENGTH)
    monkeypatch.setattr(roi_cache, "EXPECTED_RAW_ROWS", SEQUENCE_LENGTH)


def test_roi_relation_feature_order_is_exact() -> None:
    assert ROI_RELATION_FEATURE_NAMES == EXPECTED_FEATURES
    assert ROI_RELATION_DIM == 18


def test_roi_relation_cache_locks_legacy_16f_counts() -> None:
    assert roi_cache.CANONICAL_SOURCE_NAME == "legacy_16f"
    assert roi_cache.EXPECTED_RAW_ROWS == 72_864
    assert roi_cache.EXPECTED_MODEL_WINDOWS == 15_588
    assert roi_cache.EXPECTED_MODEL_SLOTS == 93_528


def test_roi_relation_cache_materializes_only_whitelisted_frame_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_expected_counts(monkeypatch)
    window_index, slot_index = _synthetic_order()

    roi, availability, windows, slots = materialize_roi_relation_cache(
        window_index,
        slot_index,
        _synthetic_frames(),
    )

    assert roi.shape == (1, SEQUENCE_LENGTH, ROI_RELATION_DIM)
    assert availability.shape == (1, SEQUENCE_LENGTH)
    assert availability.all()
    assert roi[0, 5, EXPECTED_FEATURES.index("roi_feeder_min_dist_n")] == 0.5
    assert 999.0 not in roi
    assert "geometry_available" not in slots.columns
    assert "roi_relation_available" in slots.columns
    assert windows.equals(window_index)


def test_roi_relation_cache_rejects_duplicate_and_missing_frame_uid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_expected_counts(monkeypatch)
    window_index, slot_index = _synthetic_order()
    frames = _synthetic_frames()
    frames.loc[5, "frame_uid"] = "frame-4"
    with pytest.raises(ValueError, match="blank or duplicated"):
        materialize_roi_relation_cache(window_index, slot_index, frames)

    frames = _synthetic_frames()
    slot_index.loc[5, "frame_uid"] = "missing-frame"
    with pytest.raises(ValueError, match="unmatched frame_uid"):
        materialize_roi_relation_cache(window_index, slot_index, frames)


def test_roi_relation_bounds_and_binary_fields_fail_closed() -> None:
    valid = np.zeros((1, ROI_RELATION_DIM), dtype=np.float64)
    _validate_relation_bounds(valid)

    distance = valid.copy()
    distance[0, EXPECTED_FEATURES.index("roi_feeder_min_dist_n")] = -0.1
    with pytest.raises(ValueError, match="negative ROI distance"):
        _validate_relation_bounds(distance)

    ratio = valid.copy()
    ratio[0, EXPECTED_FEATURES.index("roi_drinker_max_iou")] = 1.1
    with pytest.raises(ValueError, match="ROI ratio out of bounds"):
        _validate_relation_bounds(ratio)

    binary = valid.copy()
    binary[0, EXPECTED_FEATURES.index("roi_toy_contact")] = 0.5
    with pytest.raises(ValueError, match="not binary"):
        _validate_relation_bounds(binary)
