"""Focused tests for central-T6 H5 feasibility semantics."""

from __future__ import annotations

import json

import pandas as pd

from pig_behavior.classification_v2.temporal_views.h5_feasibility import (
    LEGACY_T6_OFFSETS,
    build_h5_targets,
    evaluate_h5_targets,
)
from pig_behavior.classification_v2.temporal_views.registry import temporal_view_spec


def _window(window_id: str, source: str, frames: list[int]) -> dict[str, object]:
    return {
        "window_id": window_id, "source_type": source, "object_track_key": f"actor={source}",
        "selected_frame_indices": json.dumps(frames), "temporal_unit_keys_json": "[\"unit\"]",
        "behavior_window_label": "move", "window_valid_for_main_train": True,
        "window_sample_weight": 1.0,
        "view_type": "T6_contiguous" if len(frames) == 6 else "T16_contiguous",
    }


def _split(*ids: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ids,
            "model_split_role": ["train"] * len(ids),
            "outer_fold_id": ["FOLD_3"] * len(ids),
        }
    )


def _frames(source: str, values: range, label: str = "move") -> pd.DataFrame:
    return pd.DataFrame({
        "object_track_key": [f"actor={source}"] * len(values), "frame_index": list(values),
        "source_type": [source] * len(values), "video_key": ["video"] * len(values),
        "temporal_unit_key": ["unit"] * len(values),
        "timestamp_sec": [index / 30 for index in values],
        "bbox_valid": [True] * len(values), "actor_bbox_valid": [True] * len(values),
        "behavior_reviewed_final": [label] * len(values),
    })


def test_legacy_h5_uses_frozen_central_target_and_native_history() -> None:
    spec = temporal_view_spec("T6_TARGET_PLUS_H5")
    assert spec.history_offsets_from_endpoint == (-10, -9, -8, -7, -6)
    assert spec.primary_cross_source_eligible
    targets = build_h5_targets(
        pd.DataFrame([_window("legacy", "legacy_recovered", list(range(13, 29)))]),
        _split("legacy"),
    )
    row = targets.iloc[0]
    assert json.loads(row["target_frame_indices_json"]) == [18, 19, 20, 21, 22, 23]
    assert json.loads(row["history_frame_indices_json"]) == [13, 14, 15, 16, 17]
    assert LEGACY_T6_OFFSETS == (5, 6, 7, 8, 9, 10)
    result, audit = evaluate_h5_targets(targets, _frames("legacy_recovered", range(13, 29)))
    assert result.loc[0, "h5_valid"]
    assert audit["h5_future_frame_dependence"] == 0


def test_cvat_h5_rejects_missing_or_future_history_and_keeps_labels_diagnostic() -> None:
    targets = build_h5_targets(
        pd.DataFrame([_window("cvat", "cvat_tracking_xml", list(range(6, 12)))]),
        _split("cvat"),
    )
    rows = _frames("cvat_tracking_xml", range(1, 12), label="lying")
    result, audit = evaluate_h5_targets(targets, rows)
    assert result.loc[0, "h5_valid"]
    assert result.loc[0, "h5_context_classification"] == "different_behavior_context"
    assert audit["h5_video_crossings"] == 0
    missing = rows.loc[rows["frame_index"].ne(4)].copy()
    result, missing_audit = evaluate_h5_targets(targets, missing)
    assert not result.loc[0, "h5_valid"]
    assert result.loc[0, "missing_history_observations"] == 1
    assert missing_audit["h5_video_crossings"] == 0
    assert missing_audit["h5_actor_scope_violations"] == 0
