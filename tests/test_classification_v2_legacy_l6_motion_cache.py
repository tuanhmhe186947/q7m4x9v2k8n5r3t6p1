from __future__ import annotations

import copy
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.training import (
    legacy_development_l6_motion_cache as motion_cache,
)
from pig_behavior.classification_v2.training.legacy_development_l6_motion_cache import (
    LINEAGE_SCOPE,
    MOTION_DIM,
    MOTION_FEATURE_NAMES,
    SEQUENCE_LENGTH,
    _validate_cache_config_payload,
    _validate_motion_slot_pairs,
    materialize_motion_cache,
)


@dataclass
class _FakeOrder:
    window_index: pd.DataFrame
    slot_index: pd.DataFrame
    manifest: dict[str, Any]
    audit: dict[str, Any]


def _synthetic_order() -> _FakeOrder:
    window_id = "track-a|win=6|3-8"
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
            "frame_uid": [f"frame-{value}" for value in range(3, 9)],
            "object_track_key": ["track-a"] * SEQUENCE_LENGTH,
            "frame_index": np.arange(3, 9),
            "source_type": ["legacy_recovered"] * SEQUENCE_LENGTH,
            "dataset_id": ["legacy_recovered_16f"] * SEQUENCE_LENGTH,
            "geometry_available": [True] * SEQUENCE_LENGTH,
            "lineage_scope": [LINEAGE_SCOPE] * SEQUENCE_LENGTH,
            "human_review_complete": [False] * SEQUENCE_LENGTH,
        }
    )
    return _FakeOrder(
        window_index=window_index,
        slot_index=slot_index,
        manifest={},
        audit={},
    )


def _synthetic_frames() -> pd.DataFrame:
    frame_index = np.arange(9, dtype=np.int64)
    cx = frame_index.astype(np.float64) / 10.0
    frame = pd.DataFrame(
        {
            "source_type": ["legacy_recovered"] * len(frame_index),
            "dataset_id": ["legacy_recovered_16f"] * len(frame_index),
            "frame_uid": [f"frame-{value}" for value in frame_index],
            "object_track_key": ["track-a"] * len(frame_index),
            "frame_index": frame_index,
            "timestamp_sec": frame_index / 30.0,
            "lineage_scope": [LINEAGE_SCOPE] * len(frame_index),
            "human_review_complete": [False] * len(frame_index),
            "cx_n": cx,
            "cy_n": np.zeros(len(frame_index)),
            "bw_n": np.full(len(frame_index), 0.2),
            "bh_n": np.full(len(frame_index), 0.1),
            "area_n": np.full(len(frame_index), 0.02),
            "aspect_ratio": np.full(len(frame_index), 2.0),
            "bbox_valid": [True] * len(frame_index),
            "actor_bbox_valid": [True] * len(frame_index),
            "geometry_feature_valid": [True] * len(frame_index),
            "spatiotemporal_feature_valid": [True] * len(frame_index),
            "speed_mean_unit": np.full(len(frame_index), 999.0),
        }
    )
    for name in MOTION_FEATURE_NAMES:
        frame[name] = np.full(len(frame_index), 77.0)
    frame["delta_cx_n"] = np.r_[0.0, np.full(8, 0.1)]
    frame["delta_cy_n"] = 0.0
    frame["delta_bw_n"] = 0.0
    frame["delta_bh_n"] = 0.0
    frame["delta_area_n"] = 0.0
    frame["delta_aspect_ratio"] = 0.0
    frame["speed_n_per_frame"] = np.r_[0.0, np.full(8, 0.1)]
    frame["speed_n_per_sec"] = np.r_[0.0, np.full(8, 3.0)]
    frame["abs_accel_n_per_frame2"] = 0.0
    frame["abs_direction_change_rad"] = 0.0
    return frame


def test_motion_cache_rebases_window_start_at_frame_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(motion_cache, "EXPECTED_MODEL_WINDOWS", 1)
    monkeypatch.setattr(
        motion_cache,
        "EXPECTED_MODEL_SLOTS",
        SEQUENCE_LENGTH,
    )
    monkeypatch.setattr(motion_cache, "EXPECTED_RAW_ROWS", 9)

    result = materialize_motion_cache(_synthetic_order(), _synthetic_frames())
    motion = result["motion"]
    available = result["availability"]
    names = list(MOTION_FEATURE_NAMES)

    assert motion.shape == (1, SEQUENCE_LENGTH, MOTION_DIM)
    assert not available[0, 0]
    assert available[0, 1:].all()
    assert np.count_nonzero(motion[0, 0]) == 0
    assert motion[0, 1, names.index("delta_cx_n")] == pytest.approx(0.1)
    assert motion[0, 1, names.index("speed_n_per_frame")] == pytest.approx(0.1)
    assert motion[0, 1, names.index("speed_n_per_sec")] == pytest.approx(3.0)
    audit = result["content_audit"]
    assert audit["unit_aggregate_features_selected"] == []
    assert audit["availability_pattern"] == [0, 1, 1, 1, 1, 1]
    reset = audit["window_start_reset_audit"]
    assert reset["windows_starting_after_frame_zero"] == 1
    assert reset["raw_nonzero_after_frame_zero_start_rows"] == 1
    assert reset["cached_nonzero_first_slot_rows"] == 0

    serialized = result["slot_index"].to_csv(index=False)
    round_trip = pd.read_csv(io.StringIO(serialized))
    _validate_motion_slot_pairs(round_trip)


def test_motion_cache_rejects_noncontiguous_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(motion_cache, "EXPECTED_MODEL_WINDOWS", 1)
    monkeypatch.setattr(
        motion_cache,
        "EXPECTED_MODEL_SLOTS",
        SEQUENCE_LENGTH,
    )
    monkeypatch.setattr(motion_cache, "EXPECTED_RAW_ROWS", 9)
    order = _synthetic_order()
    order.slot_index.loc[3, "frame_index"] = 7

    with pytest.raises(ValueError, match="not contiguous"):
        materialize_motion_cache(order, _synthetic_frames())


@pytest.mark.parametrize(
    "path",
    [
        Path(
            "configs/classification_v2/"
            "legacy_development_l6_motion_cache_v1.json"
        ),
        Path(
            "configs/classification_v2/"
            "legacy_development_l6_motion_cache_repeat_v1.json"
        ),
        Path(
            "configs/classification_v2/"
            "legacy_development_l6_motion_cache_v2.json"
        ),
        Path(
            "configs/classification_v2/"
            "legacy_development_l6_motion_cache_repeat_v2.json"
        ),
    ],
)
def test_motion_cache_config_locks_source_and_window_local_contract(
    path: Path,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))

    _validate_cache_config_payload(payload)

    assert payload["source_identity"]["canonical_short_name"] == "legacy_16f"
    assert payload["features"]["window_local_rebase"] is True
    assert payload["features"]["unit_aggregate_features_allowed"] is False
    changed = copy.deepcopy(payload)
    changed["order_authority"]["geometry_values_used"] = True
    with pytest.raises(ValueError, match="must not use geometry values"):
        _validate_cache_config_payload(changed)
