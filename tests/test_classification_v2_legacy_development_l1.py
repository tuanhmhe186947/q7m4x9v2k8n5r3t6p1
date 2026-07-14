from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from pig_behavior.classification_v2.contracts.legacy_development_l1 import (
    LETTERBOX_POLICY,
    audit_legacy_l1_tables,
    build_legacy_development_folds,
)
from pig_behavior.classification_v2.datasets.image_context_index import (
    IMAGE_CONTEXT_SEQUENCE_DELIMITER,
)
from pig_behavior.classification_v2.datasets.legacy_unreviewed_development import (
    LEGACY_DEVELOPMENT_SCOPE,
    LEGACY_TEMPORAL_MODEL_VIEW_SPECS,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

_DATES = ("291119", "301119", "031219")
_STARTS = {
    6: (0, 3, 6, 9),
    8: (0, 3, 6),
    12: (0, 3),
    16: (0,),
}
_CENTERED_START = {6: 6, 8: 3, 12: 3, 16: 0}


def _claim() -> dict[str, Any]:
    return {
        "lineage_scope": LEGACY_DEVELOPMENT_SCOPE,
        "human_review_complete": False,
    }


def _fold_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    native_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for unit_index, date_token in enumerate(_DATES):
        unit_key = f"unit-{unit_index}"
        video_key = f"pigs{date_token}/00010{unit_index}/color.mp4"
        native_rows.append(
            {
                "temporal_unit_key": unit_key,
                "source_type": "legacy_recovered",
                "dataset_id": "legacy_recovered_16f",
                "video_key": video_key,
                "behavior_label": VALID_BEHAVIORS[unit_index],
                "native_unit_valid_for_main_eval": True,
                **_claim(),
            }
        )
        for length, starts in _STARTS.items():
            for start in starts:
                row = {
                    "window_id": f"{unit_key}|T{length}|s{start}",
                    "temporal_unit_key": unit_key,
                    "temporal_tier": f"T{length}",
                    "window_length_frames": length,
                    "window_start_frame": start,
                    **_claim(),
                }
                for spec in LEGACY_TEMPORAL_MODEL_VIEW_SPECS.values():
                    column = str(spec["selection_column"])
                    same_length = int(spec["sequence_length"]) == length
                    if str(spec["sampling_view"]) == (
                        "all_sliding_event_balanced"
                    ):
                        row[column] = same_length
                    else:
                        row[column] = (
                            same_length and start == _CENTERED_START[length]
                        )
                window_rows.append(row)
    return pd.DataFrame(native_rows), pd.DataFrame(window_rows)


def _image_tables(
    selection: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame_rows: list[dict[str, Any]] = []
    context_ids_by_unit: dict[str, list[str]] = {}
    for unit_key in selection["temporal_unit_key"].drop_duplicates():
        context_ids = [f"{unit_key}|f{index:02d}" for index in range(16)]
        context_ids_by_unit[str(unit_key)] = context_ids
        for frame_index, context_id in enumerate(context_ids):
            frame_rows.append(
                {
                    "image_context_id": context_id,
                    "image_context_source": "legacy_video_bbox",
                    "image_context_loadable": True,
                    "temporal_unit_key": str(unit_key),
                    "frame_index": frame_index,
                    **_claim(),
                }
            )
    window_rows: list[dict[str, Any]] = []
    for row in selection.itertuples(index=False):
        start = int(row.window_start_frame)
        end = start + int(row.window_length_frames)
        context_ids = context_ids_by_unit[str(row.temporal_unit_key)][start:end]
        window_rows.append(
            {
                "window_id": str(row.window_id),
                "image_context_id_sequence": (
                    IMAGE_CONTEXT_SEQUENCE_DELIMITER.join(context_ids)
                ),
                "expected_frame_indices": "|".join(
                    str(frame_index) for frame_index in range(start, end)
                ),
                "window_image_context_complete": True,
                **_claim(),
            }
        )
    sorted_ids = sorted(row["image_context_id"] for row in frame_rows)
    cache_rows = [_cache_row(context_id) for context_id in sorted_ids]
    packed_rows = [
        {
            "image_context_id": context_id,
            "packed_row": row_index,
            **_claim(),
        }
        for row_index, context_id in enumerate(sorted_ids)
    ]
    return (
        pd.DataFrame(frame_rows),
        pd.DataFrame(window_rows),
        pd.DataFrame(cache_rows),
        pd.DataFrame(packed_rows),
    )


def _cache_row(context_id: str) -> dict[str, Any]:
    return {
        "image_context_id": context_id,
        "image_size": 160,
        "resize_policy": LETTERBOX_POLICY,
        "cache_path": f"cache/{context_id}.npy",
        "source_crop_width": 80.0,
        "source_crop_height": 40.0,
        "source_crop_aspect_ratio": 2.0,
        "letterbox_scale": 2.0,
        "letterbox_resized_width": 160,
        "letterbox_resized_height": 80,
        "letterbox_pad_left": 0,
        "letterbox_pad_top": 40,
        "letterbox_pad_right": 0,
        "letterbox_pad_bottom": 40,
        **_claim(),
    }


def _audit_inputs() -> dict[str, pd.DataFrame]:
    native, selection = _fold_inputs()
    folds = build_legacy_development_folds(native, selection)
    image_frames, image_windows, cache, packed = _image_tables(selection)
    return {
        "temporal_selection": selection,
        "image_frames": image_frames,
        "image_windows": image_windows,
        "cache_manifest": cache,
        "packed_index": packed,
        "recording_groups": folds.recording_groups,
        "native_folds": folds.native_folds,
        "window_folds": folds.window_folds,
        "class_support": folds.class_by_fold_support,
        "source_support": folds.source_by_fold_support,
    }


def test_builds_three_recording_date_folds_with_exact_window_inheritance() -> None:
    native, selection = _fold_inputs()

    result = build_legacy_development_folds(native, selection)

    assert result.audit["valid"] is True
    assert result.audit["fold_count"] == 3
    assert result.audit["recording_group_count"] == 3
    assert result.audit["recording_group_leakage"] == 0
    assert result.audit["video_leakage"] == 0
    assert result.audit["wrong_window_count_units"] == 0
    assert len(result.class_by_fold_support) == 30
    assert result.class_by_fold_support["test_native_units"].eq(0).sum() == 27
    expected = result.window_folds["temporal_unit_key"].map(
        result.native_folds.set_index("temporal_unit_key")["oof_fold_id"]
    )
    assert result.window_folds["oof_fold_id"].equals(expected)


def test_fold_builder_rejects_native_unit_without_tier_windows() -> None:
    native, selection = _fold_inputs()
    missing_unit = str(native.loc[0, "temporal_unit_key"])
    selection = selection.loc[
        selection["temporal_unit_key"].astype(str).ne(missing_unit)
    ].copy()

    with pytest.raises(
        ValueError,
        match="native_units_without_ten_tier_windows=1",
    ):
        build_legacy_development_folds(native, selection)


def test_l1_relational_audit_accepts_cache_slots_folds_and_letterbox() -> None:
    inputs = _audit_inputs()

    audit = audit_legacy_l1_tables(**inputs, image_size=160)

    assert audit["valid"] is True
    assert audit["errors"] == []
    assert audit["fold_count"] == 3
    assert audit["frame_cache_id_set_equal"] is True
    assert audit["frame_packed_id_set_equal"] is True
    assert audit["ordered_cache_packed_ids_match"] is True
    assert audit["total_selected_image_slots"] == 264
    assert audit["letterbox_metadata_audit"]["invalid_rows"] == 0


def test_l1_relational_audit_rejects_missing_packed_and_window_slot() -> None:
    inputs = _audit_inputs()
    inputs["packed_index"] = inputs["packed_index"].iloc[:-1].copy()
    image_windows = inputs["image_windows"].copy()
    first_sequence = image_windows.loc[0, "image_context_id_sequence"]
    first_ids = first_sequence.split(IMAGE_CONTEXT_SEQUENCE_DELIMITER)
    first_ids[0] = "missing-context"
    image_windows.loc[0, "image_context_id_sequence"] = (
        IMAGE_CONTEXT_SEQUENCE_DELIMITER.join(first_ids)
    )
    inputs["image_windows"] = image_windows

    audit = audit_legacy_l1_tables(**inputs, image_size=160)

    assert audit["valid"] is False
    assert any(error.startswith("frame_packed_id_set_mismatch") for error in audit["errors"])
    assert "missing_cache_slot_rows=1" in audit["errors"]


def test_l1_relational_audit_rejects_window_fold_drift() -> None:
    inputs = _audit_inputs()
    window_folds = inputs["window_folds"].copy()
    original = str(window_folds.loc[0, "oof_fold_id"])
    replacement = next(
        fold
        for fold in window_folds["oof_fold_id"].astype(str).unique()
        if fold != original
    )
    window_folds.loc[0, "oof_fold_id"] = replacement
    inputs["window_folds"] = window_folds

    audit = audit_legacy_l1_tables(**inputs, image_size=160)

    assert audit["valid"] is False
    assert "fold_inheritance_mismatches=1" in audit["errors"]


def test_l1_relational_audit_rejects_cross_unit_image_slot() -> None:
    inputs = _audit_inputs()
    image_windows = inputs["image_windows"].copy()
    context_ids = image_windows.loc[0, "image_context_id_sequence"].split(
        IMAGE_CONTEXT_SEQUENCE_DELIMITER
    )
    context_ids[0] = "unit-1|f00"
    image_windows.loc[0, "image_context_id_sequence"] = (
        IMAGE_CONTEXT_SEQUENCE_DELIMITER.join(context_ids)
    )
    inputs["image_windows"] = image_windows

    audit = audit_legacy_l1_tables(**inputs, image_size=160)

    assert audit["valid"] is False
    assert "slot_native_unit_mismatches=1" in audit["errors"]


def test_l1_relational_audit_rejects_support_and_packed_row_drift() -> None:
    inputs = _audit_inputs()
    class_support = inputs["class_support"].copy()
    class_support.loc[0, "test_native_units"] += 1
    inputs["class_support"] = class_support
    packed_index = inputs["packed_index"].copy()
    packed_index["packed_row"] = packed_index["packed_row"].astype(float)
    packed_index.loc[0, "packed_row"] = 0.5
    inputs["packed_index"] = packed_index

    audit = audit_legacy_l1_tables(**inputs, image_size=160)

    assert audit["valid"] is False
    assert "class_support_mismatches=1" in audit["errors"]
    assert "packed_rows_not_contiguous" in audit["errors"]


def test_l1_relational_audit_rejects_claim_or_source_schema_drift() -> None:
    inputs = _audit_inputs()
    inputs["cache_manifest"] = inputs["cache_manifest"].copy()
    inputs["cache_manifest"].loc[0, "human_review_complete"] = True

    with pytest.raises(ValueError, match="must remain unreviewed"):
        audit_legacy_l1_tables(**inputs, image_size=160)

    inputs = _audit_inputs()
    inputs["native_folds"] = inputs["native_folds"].drop(columns="source_type")
    with pytest.raises(ValueError, match="native_folds missing columns"):
        audit_legacy_l1_tables(**inputs, image_size=160)
