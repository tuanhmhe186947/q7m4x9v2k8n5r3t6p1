from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pig_behavior.classification_v2.datasets.visual_interaction_selection import (
    LINEAGE_SCOPE,
    _dataframe_sha256,
    derive_visual_interaction_short_selection,
    load_visual_interaction_selection_config,
)

CONFIG_PATH = Path(
    "configs/classification_v2/"
    "legacy_development_l6_union_context_short_selection_v1.json"
)


def _fixture() -> dict[str, Any]:
    training = pd.DataFrame(
        [
            _selection_row(
                order=0,
                window_id="window-train",
                unit="unit-train",
                recording="recording-train",
                video="video-train",
                role="train",
                behavior="eat",
            ),
            _selection_row(
                order=1,
                window_id="window-validation",
                unit="unit-validation",
                recording="recording-validation",
                video="video-validation",
                role="validation",
                behavior="drink",
            ),
        ]
    )
    parent_audit = {
        "valid": True,
        "outer_holdout_rows": 0,
        "selection_content_sha256": _dataframe_sha256(training),
    }
    folds = pd.DataFrame(
        [
            _fold_row(
                window_id="window-train",
                unit="unit-train",
                recording="recording-train",
                video="video-train",
                fold="native_oof_001",
            ),
            _fold_row(
                window_id="window-validation",
                unit="unit-validation",
                recording="recording-validation",
                video="video-validation",
                fold="native_oof_006",
            ),
        ]
    )
    image_windows = pd.DataFrame(
        [
            _window_row("window-train", ["target-a0", "target-a1"]),
            _window_row(
                "window-validation",
                ["target-b0", "target-b1"],
            ),
        ]
    )
    frame_rows: list[dict[str, Any]] = []
    for prefix, video in (("a", "video-train"), ("b", "video-validation")):
        for frame_index in range(2):
            frame_rows.extend(
                [
                    _frame_row(
                        context_id=f"target-{prefix}{frame_index}",
                        video=video,
                        frame_index=frame_index,
                        track_id=f"actor-{prefix}",
                        nearest_track_id=f"partner-{prefix}",
                        partner_available=True,
                        x1=10.0,
                        x2=30.0,
                    ),
                    _frame_row(
                        context_id=f"partner-{prefix}{frame_index}",
                        video=video,
                        frame_index=frame_index,
                        track_id=f"partner-{prefix}",
                        nearest_track_id="",
                        partner_available=False,
                        x1=40.0,
                        x2=60.0,
                    ),
                ]
            )
    contract = {
        "training_scope": "short_repeat_gate",
        "view_id": "t6_sliding",
        "sampling_protocol": "all_sliding_event_balanced",
        "sequence_length": 2,
        "windows_per_native_unit": 1,
        "selected_windows": 2,
        "train_windows": 1,
        "validation_windows": 1,
        "train_native_units": 1,
        "validation_native_units": 1,
        "selected_image_context_ids": 4,
        "frame_context_rows": 8,
        "outer_holdout_fold_id": "native_oof_005",
        "development_validation_fold_id": "native_oof_006",
    }
    return {
        "training": training,
        "parent_audit": parent_audit,
        "folds": folds,
        "image_windows": image_windows,
        "frame_context": pd.DataFrame(frame_rows),
        "contract": contract,
    }


def _selection_row(
    *,
    order: int,
    window_id: str,
    unit: str,
    recording: str,
    video: str,
    role: str,
    behavior: str,
) -> dict[str, Any]:
    return {
        "selection_order": order,
        "position": order,
        "window_id": window_id,
        "temporal_unit_key": unit,
        "recording_group_id": recording,
        "video_key": video,
        "source_type": "legacy_recovered",
        "dataset_id": "legacy_recovered_16f",
        "behavior_label": behavior,
        "l5_role": role,
        "view_id": "t6_sliding",
        "sampling_protocol": "all_sliding_event_balanced",
        "sequence_length": 2,
        "training_scope": "short_repeat_gate",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
    }


def _fold_row(
    *,
    window_id: str,
    unit: str,
    recording: str,
    video: str,
    fold: str,
) -> dict[str, Any]:
    return {
        "window_id": window_id,
        "temporal_unit_key": unit,
        "recording_group_id": recording,
        "video_key": video,
        "oof_fold_id": fold,
        "source_type": "legacy_recovered",
        "dataset_id": "legacy_recovered_16f",
        "legacy_t6_all_sliding_keep": True,
    }


def _window_row(window_id: str, context_ids: list[str]) -> dict[str, Any]:
    return {
        "window_id": window_id,
        "source_type": "legacy_recovered",
        "dataset_id": "legacy_recovered_16f",
        "window_length_frames": 2,
        "image_context_id_sequence": ";;".join(context_ids),
        "observed_image_context_rows": 2,
        "loadable_image_context_rows": 2,
        "missing_image_context_slots": 0,
        "window_image_context_complete": True,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
    }


def _frame_row(
    *,
    context_id: str,
    video: str,
    frame_index: int,
    track_id: str,
    nearest_track_id: str,
    partner_available: bool,
    x1: float,
    x2: float,
) -> dict[str, Any]:
    return {
        "image_context_id": context_id,
        "source_type": "legacy_recovered",
        "dataset_id": "legacy_recovered_16f",
        "video_key": video,
        "clip_id": f"clip-{video}",
        "resolved_media_path": f"{video}.mp4",
        "resolved_media_exists": True,
        "frame_index": frame_index,
        "track_id": track_id,
        "nearest_track_id": nearest_track_id,
        "x1": x1,
        "y1": 10.0,
        "x2": x2,
        "y2": 40.0,
        "partner_context_available": partner_available,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
    }


def test_short_selection_is_holdout_free_and_identifier_only() -> None:
    fixture = _fixture()

    selection, audit = derive_visual_interaction_short_selection(**fixture)

    assert list(selection.columns) == ["image_context_id"]
    assert selection["image_context_id"].tolist() == [
        "target-a0",
        "target-a1",
        "target-b0",
        "target-b1",
    ]
    assert audit["outer_holdout_windows"] == 0
    assert audit["outer_holdout_image_context_ids"] == 0
    assert audit["train_validation_group_overlap"] == []
    assert audit["static_union_geometry_ready"] == 4
    assert audit["static_union_geometry_unavailable"] == 0


def test_short_selection_config_is_fail_closed_and_legacy_only() -> None:
    config = load_visual_interaction_selection_config(CONFIG_PATH)

    assert config.payload["canonical_source_name"] == "legacy_16f"
    assert config.payload["canonical_full_oof_authorized"] is False
    assert config.payload["outer_holdout_predictions_authorized"] is False
    assert config.payload["reviewed_or_final_claim_allowed"] is False
    assert config.payload["q2_claim_allowed"] is False


def test_short_selection_rejects_outer_holdout_window() -> None:
    fixture = _fixture()
    fixture["folds"].loc[
        fixture["folds"]["window_id"].eq("window-train"),
        "oof_fold_id",
    ] = "native_oof_005"

    with pytest.raises(ValueError, match="held-out fold entered"):
        derive_visual_interaction_short_selection(**fixture)


def test_short_selection_rejects_partner_availability_drift() -> None:
    fixture = _fixture()
    fixture["frame_context"].loc[
        fixture["frame_context"]["image_context_id"].eq("target-a0"),
        "partner_context_available",
    ] = False

    with pytest.raises(ValueError, match="partner availability differs"):
        derive_visual_interaction_short_selection(**fixture)
