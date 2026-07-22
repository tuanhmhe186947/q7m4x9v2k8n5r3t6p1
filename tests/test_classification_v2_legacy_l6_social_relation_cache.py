from __future__ import annotations

import copy
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
from pig_behavior.classification_v2.training import (
    legacy_development_l6_social_relation_cache as social_cache,
)
from pig_behavior.classification_v2.training.legacy_development_l6_social_relation_cache import (
    LINEAGE_SCOPE,
    SEQUENCE_LENGTH,
    SOCIAL_RELATION_DIM,
    SOCIAL_RELATION_FEATURE_NAMES,
    _validate_cache_config_payload,
    materialize_social_relation_cache,
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
            "nearest_pig_id": [
                "ID_2",
                "ID_2",
                "ID_2",
                "ID_2",
                "ID_2",
                "ID_2",
                "ID_3",
                "ID_3",
                "",
            ],
            "nearest_track_id": [""] * len(frame_index),
            "cx_n": frame_index.astype(float) / 10.0,
            "cy_n": np.full(len(frame_index), 0.5),
            "bw_n": np.full(len(frame_index), 0.2),
            "bh_n": np.full(len(frame_index), 0.1),
            "speed_n_per_frame": np.full(len(frame_index), 0.01),
            "bbox_valid": [True] * len(frame_index),
            "actor_bbox_valid": [True] * len(frame_index),
            "geometry_feature_valid": [True] * len(frame_index),
            "spatiotemporal_feature_valid": [True] * len(frame_index),
        }
    )
    for name in SOCIAL_RELATION_FEATURE_NAMES:
        frame[name] = np.full(len(frame_index), 77.0)
    frame["nearest_dist_n"] = 0.5 - frame_index.astype(float) / 100.0
    frame["nearest_pair_iou"] = 0.2
    frame["nearest_pair_overlap_ratio"] = 0.3
    frame["social_density_near_count"] = 1.0
    frame["social_contact_count"] = 1.0
    frame["pair_contact_with_nearest"] = 1.0
    return frame


def test_social_cache_rebases_partner_motion_within_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for module in (social_cache, motion_cache):
        monkeypatch.setattr(module, "EXPECTED_MODEL_WINDOWS", 1)
        monkeypatch.setattr(module, "EXPECTED_MODEL_SLOTS", SEQUENCE_LENGTH)
        monkeypatch.setattr(module, "EXPECTED_RAW_ROWS", 9)

    result = materialize_social_relation_cache(
        _synthetic_order(),
        _synthetic_frames(),
    )
    social = result["social_relation"]
    available = result["availability"]
    names = list(SOCIAL_RELATION_FEATURE_NAMES)
    delta = names.index("nearest_dist_delta")
    approach = names.index("approach_speed_n_per_frame")

    assert social.shape == (1, SEQUENCE_LENGTH, SOCIAL_RELATION_DIM)
    assert available[0].tolist() == [True, True, True, True, True, False]
    assert social[0, 0, delta] == 0.0
    assert social[0, 1, delta] == pytest.approx(-0.01)
    assert social[0, 1, approach] == pytest.approx(0.01)
    assert social[0, 3, delta] == 0.0
    assert social[0, 4, delta] == pytest.approx(-0.01)
    assert np.count_nonzero(social[0, 5]) == 0
    assert result["content_audit"]["top_k_partner_features_used"] is False
    assert result["content_audit"]["unit_aggregate_features_used"] is False
    assert result["slot_index"]["social_window_slot_uid"].is_unique


def test_social_feature_contract_excludes_partner_identity() -> None:
    assert SOCIAL_RELATION_DIM == 10
    assert not any(
        token in name
        for name in SOCIAL_RELATION_FEATURE_NAMES
        for token in ("pig_id", "track_id", "nearest_pig_id")
    )


@pytest.mark.parametrize(
    "path",
    [
        Path(
            "configs/classification_v2/"
            "legacy_development_l6_social_relation_cache_v1.json"
        ),
        Path(
            "configs/classification_v2/"
            "legacy_development_l6_social_relation_cache_repeat_v1.json"
        ),
    ],
)
def test_social_cache_config_locks_numeric_only_contract(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))

    _validate_cache_config_payload(payload)

    assert payload["source_identity"]["canonical_short_name"] == "legacy_16f"
    assert payload["features"]["window_local_rebase"] is True
    assert payload["features"]["numeric_social_only"] is True
    assert payload["features"]["top_k_partner_features_allowed"] is False
    changed = copy.deepcopy(payload)
    changed["features"]["top_k_partner_features_allowed"] = True
    with pytest.raises(ValueError, match="feature contract drift"):
        _validate_cache_config_payload(changed)
