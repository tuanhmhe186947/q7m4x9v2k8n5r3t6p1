from __future__ import annotations

import copy
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation.legacy_development_l6_motion_cache_repeat import (
    ARTIFACT_NAMES,
    _artifact_comparison,
    _content_comparison,
    _validate_config,
)

CONFIG_PATH = Path(
    "configs/classification_v2/"
    "legacy_development_l6_motion_cache_repeat_gate_v1.json"
)
def _synthetic_manifest() -> dict[str, object]:
    return {
        "artifacts": {
            name: {"sha256": str(index) * 64, "size_bytes": index}
            for index, name in enumerate(ARTIFACT_NAMES, start=1)
        },
        "content_audit": {
            "model_window_rows": 2,
            "model_slot_rows": 12,
            "role_window_counts": {"train": 2},
            "available_pair_slots": 10,
            "unavailable_slots": 2,
            "unavailable_first_slots": 2,
            "unavailable_nonfirst_slots": 0,
            "motion_shape": [2, 6, 12],
            "availability_shape": [2, 6],
            "motion_dtype": "float32",
            "availability_dtype": "bool",
            "motion_statistics": {"mean": 0.0},
            "availability_pattern_count": 1,
            "availability_pattern": [False, True, True, True, True, True],
            "ordered_window_id_sha256": "a" * 64,
            "window_index_content_sha256": "b" * 64,
            "slot_index_content_sha256": "c" * 64,
            "spatial_export_audit": {"valid": True},
            "window_start_reset_audit": {"valid": True},
            "source_probe": {"status": "PASS"},
            "unit_aggregate_features_selected": False,
            "geometry_values_consumed": False,
        },
    }


def test_motion_cache_repeat_config_is_strict() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    _validate_config(payload)

    assert payload["lineage_scope"] == "legacy-only-unreviewed-development"
    assert payload["human_review_complete"] is False


def test_motion_cache_repeat_comparison_detects_drift() -> None:
    primary = _synthetic_manifest()
    repeat = copy.deepcopy(primary)

    artifacts = _artifact_comparison(primary, repeat)
    content = _content_comparison(primary, repeat)

    assert artifacts["valid"]
    assert artifacts["artifact_count"] == len(ARTIFACT_NAMES)
    assert content["valid"]
    repeat["artifacts"]["motion"]["sha256"] = "0" * 64
    assert not _artifact_comparison(primary, repeat)["valid"]
