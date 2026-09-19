"""Compose canonical cache and fold contracts for the legacy L1 packet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.datasets.image_context_index import (
    IMAGE_CONTEXT_SEQUENCE_DELIMITER,
)
from pig_behavior.classification_v2.datasets.legacy_unreviewed_development import (
    LEGACY_DEVELOPMENT_SCOPE,
    LEGACY_SOURCE,
    LEGACY_TEMPORAL_MODEL_VIEW_SPECS,
)
from pig_behavior.classification_v2.evaluation.native_oof_folds import (
    build_native_oof_folds,
)
from pig_behavior.classification_v2.metadata.recording_groups import (
    build_recording_group_manifest,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

LEGACY_L1_SCHEMA_VERSION = "classification_v2.legacy_development_l1.v1"
LETTERBOX_POLICY = "letterbox_preserve_aspect_rgb_pad_black_v1"
LETTERBOX_METADATA_COLUMNS = {
    "source_crop_width",
    "source_crop_height",
    "source_crop_aspect_ratio",
    "letterbox_scale",
    "letterbox_resized_width",
    "letterbox_resized_height",
    "letterbox_pad_left",
    "letterbox_pad_top",
    "letterbox_pad_right",
    "letterbox_pad_bottom",
}


@dataclass(slots=True)
class LegacyDevelopmentFoldTables:
    """Recording-safe fold artifacts for one legacy temporal-tier universe."""

    recording_groups: pd.DataFrame
    native_folds: pd.DataFrame
    window_folds: pd.DataFrame
    class_by_fold_support: pd.DataFrame
    source_by_fold_support: pd.DataFrame
    audit: dict[str, Any]


def build_legacy_development_folds(
    native_units: pd.DataFrame,
    temporal_selection: pd.DataFrame,
    *,
    group_level: str = "recording_date",
) -> LegacyDevelopmentFoldTables:
    """Compose shared recording metadata and native OOF fold implementations."""

    _require_columns(
        native_units,
        {
            "temporal_unit_key",
            "source_type",
            "dataset_id",
            "video_key",
            "behavior_label",
            "native_unit_valid_for_main_eval",
            "lineage_scope",
            "human_review_complete",
        },
        "native_units",
    )
    _require_columns(
        temporal_selection,
        {
            "window_id",
            "temporal_unit_key",
            "temporal_tier",
            "window_length_frames",
            "lineage_scope",
            "human_review_complete",
        },
        "temporal_selection",
    )
    _require_legacy_scope(native_units, "native_units")
    _require_legacy_scope(temporal_selection, "temporal_selection")
    native_sources = set(native_units["source_type"].astype(str))
    if native_sources != {LEGACY_SOURCE}:
        raise ValueError(f"legacy native source values={sorted(native_sources)}")
    invalid_labels = sorted(
        set(native_units["behavior_label"].astype(str)).difference(
            VALID_BEHAVIORS
        )
    )
    if invalid_labels:
        raise ValueError(f"invalid legacy behavior labels={invalid_labels}")
    if native_units["temporal_unit_key"].duplicated().any():
        raise ValueError("duplicate temporal_unit_key in legacy native units")
    if temporal_selection["window_id"].duplicated().any():
        raise ValueError("duplicate window_id in legacy temporal selection")

    recording = build_recording_group_manifest(
        native_units,
        group_level=group_level,
    )
    recording_groups = recording.manifest.copy()
    recording_groups["lineage_scope"] = LEGACY_DEVELOPMENT_SCOPE
    recording_groups["human_review_complete"] = False
    group_columns = [
        "source_type",
        "dataset_id",
        "video_key",
        "recording_group_id",
    ]
    native_with_groups = native_units.merge(
        recording.manifest[group_columns],
        on=["source_type", "dataset_id", "video_key"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if len(native_with_groups) != len(native_units):
        raise RuntimeError("recording-group merge changed native-unit row count")
    missing_groups = int(native_with_groups["recording_group_id"].isna().sum())
    if missing_groups:
        raise ValueError(f"legacy native units without recording group={missing_groups}")

    fold_result = build_native_oof_folds(native_with_groups)
    native_metadata = native_units[
        [
            "temporal_unit_key",
            "dataset_id",
            "lineage_scope",
            "human_review_complete",
        ]
    ]
    native_folds = fold_result.manifest.merge(
        native_metadata,
        on="temporal_unit_key",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    fold_lookup = native_folds[
        [
            "temporal_unit_key",
            "recording_group_id",
            "oof_fold_id",
            "behavior_label",
            "source_type",
            "dataset_id",
            "video_key",
        ]
    ]
    window_folds = temporal_selection.merge(
        fold_lookup,
        on="temporal_unit_key",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    class_support = _class_by_fold_support(native_folds)
    source_support = _source_by_fold_support(native_folds)
    audit = _audit_fold_tables(
        native_units=native_units,
        temporal_selection=temporal_selection,
        recording_audit=recording.audit,
        fold_audit=fold_result.audit,
        native_folds=native_folds,
        window_folds=window_folds,
        class_support=class_support,
        source_support=source_support,
        group_level=group_level,
    )
    if audit["errors"]:
        raise ValueError("legacy development fold contract failed: " + "; ".join(audit["errors"]))
    return LegacyDevelopmentFoldTables(
        recording_groups=recording_groups,
        native_folds=native_folds,
        window_folds=window_folds,
        class_by_fold_support=class_support,
        source_by_fold_support=source_support,
        audit=audit,
    )


def audit_legacy_l1_tables(
    *,
    temporal_selection: pd.DataFrame,
    image_frames: pd.DataFrame,
    image_windows: pd.DataFrame,
    cache_manifest: pd.DataFrame,
    packed_index: pd.DataFrame,
    recording_groups: pd.DataFrame,
    native_folds: pd.DataFrame,
    window_folds: pd.DataFrame,
    class_support: pd.DataFrame,
    source_support: pd.DataFrame,
    image_size: int,
) -> dict[str, Any]:
    """Audit slot, cache, packed-index, and fold joins without reading pixels."""

    claim_columns = {"lineage_scope", "human_review_complete"}
    required = {
        "selection": (
            temporal_selection,
            {
                "window_id",
                "temporal_unit_key",
                "window_length_frames",
                *claim_columns,
            },
        ),
        "image_frames": (
            image_frames,
            {
                "image_context_id",
                "image_context_source",
                "image_context_loadable",
                "temporal_unit_key",
                "frame_index",
                *claim_columns,
            },
        ),
        "image_windows": (
            image_windows,
            {
                "window_id",
                "image_context_id_sequence",
                "expected_frame_indices",
                "window_image_context_complete",
                *claim_columns,
            },
        ),
        "cache_manifest": (
            cache_manifest,
            {
                "image_context_id",
                "image_size",
                "resize_policy",
                "cache_path",
                *LETTERBOX_METADATA_COLUMNS,
                *claim_columns,
            },
        ),
        "packed_index": (
            packed_index,
            {"image_context_id", "packed_row", *claim_columns},
        ),
        "recording_groups": (
            recording_groups,
            {
                "source_type",
                "dataset_id",
                "video_key",
                "recording_group_id",
                *claim_columns,
            },
        ),
        "native_folds": (
            native_folds,
            {
                "temporal_unit_key",
                "recording_group_id",
                "oof_fold_id",
                "behavior_label",
                "source_type",
                "dataset_id",
                "video_key",
                *claim_columns,
            },
        ),
        "window_folds": (
            window_folds,
            {
                "window_id",
                "temporal_unit_key",
                "oof_fold_id",
                *claim_columns,
            },
        ),
        "class_support": (
            class_support,
            {
                "oof_fold_id",
                "behavior_label",
                "test_native_units",
                "train_native_units",
                "test_supported",
                "train_supported",
                *claim_columns,
            },
        ),
        "source_support": (
            source_support,
            {
                "oof_fold_id",
                "source_type",
                "test_native_units",
                "train_native_units",
                "test_supported",
                "train_supported",
                *claim_columns,
            },
        ),
    }
    for name, (frame, columns) in required.items():
        _require_columns(frame, columns, name)
        _require_legacy_scope(frame, name)

    errors: list[str] = []
    duplicate_counts = {
        "selection_window_id": int(temporal_selection["window_id"].duplicated().sum()),
        "image_window_id": int(image_windows["window_id"].duplicated().sum()),
        "image_context_id": int(image_frames["image_context_id"].duplicated().sum()),
        "cache_context_id": int(cache_manifest["image_context_id"].duplicated().sum()),
        "packed_context_id": int(packed_index["image_context_id"].duplicated().sum()),
        "recording_group_video": int(
            recording_groups.duplicated(
                ["source_type", "dataset_id", "video_key"]
            ).sum()
        ),
        "native_fold_unit": int(native_folds["temporal_unit_key"].duplicated().sum()),
        "window_fold_id": int(window_folds["window_id"].duplicated().sum()),
        "class_support_key": int(
            class_support.duplicated(["oof_fold_id", "behavior_label"]).sum()
        ),
        "source_support_key": int(
            source_support.duplicated(["oof_fold_id", "source_type"]).sum()
        ),
    }
    for name, count in duplicate_counts.items():
        if count:
            errors.append(f"duplicate_{name}={count}")
    native_source_types = sorted(
        native_folds["source_type"].fillna("").astype(str).unique()
    )
    native_behavior_labels = sorted(
        native_folds["behavior_label"].fillna("").astype(str).unique()
    )
    invalid_native_labels = sorted(
        set(native_behavior_labels).difference(VALID_BEHAVIORS)
    )
    if native_source_types != [LEGACY_SOURCE]:
        errors.append(f"invalid_native_source_types={native_source_types}")
    if invalid_native_labels:
        errors.append(f"invalid_native_behavior_labels={invalid_native_labels}")

    selection_ids = temporal_selection["window_id"].astype(str).tolist()
    image_window_ids = image_windows["window_id"].astype(str).tolist()
    fold_window_ids = window_folds["window_id"].astype(str).tolist()
    ordered_image_windows_match = selection_ids == image_window_ids
    ordered_fold_windows_match = selection_ids == fold_window_ids
    if not ordered_image_windows_match:
        errors.append("ordered_image_window_ids_do_not_match_selection")
    if not ordered_fold_windows_match:
        errors.append("ordered_fold_window_ids_do_not_match_selection")

    frame_ids = set(image_frames["image_context_id"].astype(str))
    cache_ids = set(cache_manifest["image_context_id"].astype(str))
    packed_ids = set(packed_index["image_context_id"].astype(str))
    ordered_cache_packed_ids_match = (
        cache_manifest["image_context_id"].astype(str).tolist()
        == packed_index["image_context_id"].astype(str).tolist()
    )
    if not ordered_cache_packed_ids_match:
        errors.append("ordered_cache_packed_ids_do_not_match")
    if frame_ids != cache_ids:
        errors.append(
            "frame_cache_id_set_mismatch="
            f"missing:{len(frame_ids - cache_ids)},extra:{len(cache_ids - frame_ids)}"
        )
    if frame_ids != packed_ids:
        errors.append(
            "frame_packed_id_set_mismatch="
            f"missing:{len(frame_ids - packed_ids)},extra:{len(packed_ids - frame_ids)}"
        )

    image_by_window = image_windows.set_index("window_id", drop=False)
    missing_slot_rows = 0
    wrong_slot_count_rows = 0
    slot_native_unit_mismatches = 0
    slot_frame_order_mismatches = 0
    total_slot_rows = 0
    frame_by_context = image_frames.set_index("image_context_id", drop=False)
    for row in temporal_selection.itertuples(index=False):
        window_id = str(row.window_id)
        if window_id not in image_by_window.index:
            missing_slot_rows += 1
            continue
        image_row = image_by_window.loc[window_id]
        context_ids = _context_ids(str(image_row["image_context_id_sequence"]))
        expected_slots = int(row.window_length_frames)
        total_slot_rows += len(context_ids)
        if len(context_ids) != expected_slots:
            wrong_slot_count_rows += 1
        missing_slot_rows += sum(context_id not in cache_ids for context_id in context_ids)
        available_ids = [
            context_id for context_id in context_ids if context_id in frame_ids
        ]
        slot_units = [
            str(frame_by_context.loc[context_id, "temporal_unit_key"])
            for context_id in available_ids
        ]
        if any(unit != str(row.temporal_unit_key) for unit in slot_units):
            slot_native_unit_mismatches += 1
        expected_frames = _frame_indices(
            str(image_row["expected_frame_indices"])
        )
        observed_frames = [
            int(frame_by_context.loc[context_id, "frame_index"])
            for context_id in available_ids
        ]
        if expected_frames != observed_frames:
            slot_frame_order_mismatches += 1
    if missing_slot_rows:
        errors.append(f"missing_cache_slot_rows={missing_slot_rows}")
    if wrong_slot_count_rows:
        errors.append(f"wrong_window_slot_count_rows={wrong_slot_count_rows}")
    if slot_native_unit_mismatches:
        errors.append(
            f"slot_native_unit_mismatches={slot_native_unit_mismatches}"
        )
    if slot_frame_order_mismatches:
        errors.append(
            f"slot_frame_order_mismatches={slot_frame_order_mismatches}"
        )

    invalid_view_selection_values = 0
    selected_view_length_mismatches = 0
    selected_windows_by_view: dict[str, int] = {}
    for view_name, spec in LEGACY_TEMPORAL_MODEL_VIEW_SPECS.items():
        column = str(spec["selection_column"])
        if column not in temporal_selection.columns:
            errors.append(f"missing_temporal_selection_column={column}")
            continue
        selected, invalid = _strict_bool(temporal_selection[column])
        invalid_view_selection_values += invalid
        expected_length = int(spec["sequence_length"])
        selected_view_length_mismatches += int(
            temporal_selection.loc[selected, "window_length_frames"]
            .astype(int)
            .ne(expected_length)
            .sum()
        )
        selected_windows_by_view[view_name] = int(selected.sum())
    if invalid_view_selection_values:
        errors.append(f"invalid_view_selection_values={invalid_view_selection_values}")
    if selected_view_length_mismatches:
        errors.append(
            f"selected_view_length_mismatches={selected_view_length_mismatches}"
        )

    image_loadable, invalid_image_loadable = _strict_bool(
        image_frames["image_context_loadable"]
    )
    window_complete, invalid_window_complete = _strict_bool(
        image_windows["window_image_context_complete"]
    )
    if invalid_image_loadable or not image_loadable.all():
        errors.append(
            "image_context_not_fully_loadable="
            f"invalid:{invalid_image_loadable},false:{int((~image_loadable).sum())}"
        )
    if invalid_window_complete or not window_complete.all():
        errors.append(
            "image_windows_not_complete="
            f"invalid:{invalid_window_complete},false:{int((~window_complete).sum())}"
        )

    size_mismatches = int(
        pd.to_numeric(cache_manifest["image_size"], errors="coerce")
        .ne(image_size)
        .sum()
    )
    policies = sorted(cache_manifest["resize_policy"].astype(str).unique())
    if size_mismatches:
        errors.append(f"cache_image_size_mismatches={size_mismatches}")
    if policies != [LETTERBOX_POLICY]:
        errors.append(f"cache_resize_policy_mismatch={policies}")
    letterbox_audit = _audit_letterbox_metadata(cache_manifest, image_size)
    errors.extend(letterbox_audit["errors"])
    packed_rows = pd.to_numeric(packed_index["packed_row"], errors="coerce")
    packed_rows_contiguous = bool(
        packed_rows.notna().all()
        and packed_rows.mod(1).eq(0).all()
        and np.array_equal(
            packed_rows.to_numpy(dtype=np.int64),
            np.arange(len(packed_index), dtype=np.int64),
        )
    )
    if not packed_rows_contiguous:
        errors.append("packed_rows_not_contiguous")

    recording_lookup = recording_groups[
        ["source_type", "dataset_id", "video_key", "recording_group_id"]
    ]
    native_recording = native_folds.merge(
        recording_lookup,
        on=["source_type", "dataset_id", "video_key"],
        how="left",
        suffixes=("_native", "_recording"),
        validate="many_to_one",
        sort=False,
    )
    missing_native_recording_groups = int(
        native_recording["recording_group_id_recording"].isna().sum()
    )
    recording_group_mismatches = int(
        native_recording["recording_group_id_native"]
        .astype(str)
        .ne(native_recording["recording_group_id_recording"].astype(str))
        .sum()
    )
    if missing_native_recording_groups:
        errors.append(
            "missing_native_recording_groups="
            f"{missing_native_recording_groups}"
        )
    if recording_group_mismatches:
        errors.append(
            f"recording_group_mismatches={recording_group_mismatches}"
        )

    fold_lookup = native_folds.set_index("temporal_unit_key")["oof_fold_id"]
    expected_window_folds = window_folds["temporal_unit_key"].map(fold_lookup)
    missing_window_fold_rows = int(expected_window_folds.isna().sum())
    fold_inheritance_mismatches = int(
        expected_window_folds.astype(str)
        .ne(window_folds["oof_fold_id"].astype(str))
        .sum()
    )
    if missing_window_fold_rows:
        errors.append(f"missing_window_fold_rows={missing_window_fold_rows}")
    if fold_inheritance_mismatches:
        errors.append(f"fold_inheritance_mismatches={fold_inheritance_mismatches}")
    fold_count = int(native_folds["oof_fold_id"].nunique())
    expected_class_support_rows = fold_count * len(VALID_BEHAVIORS)
    if len(class_support) != expected_class_support_rows:
        errors.append(
            "class_support_row_mismatch="
            f"expected:{expected_class_support_rows},observed:{len(class_support)}"
        )
    expected_source_support_rows = fold_count * int(
        native_folds["source_type"].nunique()
    )
    if len(source_support) != expected_source_support_rows:
        errors.append(
            "source_support_row_mismatch="
            f"expected:{expected_source_support_rows},observed:{len(source_support)}"
        )
    expected_class_support = _class_by_fold_support(native_folds)
    expected_source_support = _source_by_fold_support(native_folds)
    class_support_mismatches = _support_mismatch_count(
        class_support,
        expected_class_support,
        ["oof_fold_id", "behavior_label"],
    )
    source_support_mismatches = _support_mismatch_count(
        source_support,
        expected_source_support,
        ["oof_fold_id", "source_type"],
    )
    if class_support_mismatches:
        errors.append(
            f"class_support_mismatches={class_support_mismatches}"
        )
    if source_support_mismatches:
        errors.append(
            f"source_support_mismatches={source_support_mismatches}"
        )

    return {
        "schema_version": LEGACY_L1_SCHEMA_VERSION,
        "lineage_scope": LEGACY_DEVELOPMENT_SCOPE,
        "human_review_complete": False,
        "image_size": int(image_size),
        "selection_rows": int(len(temporal_selection)),
        "image_frame_rows": int(len(image_frames)),
        "image_window_rows": int(len(image_windows)),
        "cache_manifest_rows": int(len(cache_manifest)),
        "packed_index_rows": int(len(packed_index)),
        "recording_group_rows": int(len(recording_groups)),
        "native_fold_rows": int(len(native_folds)),
        "window_fold_rows": int(len(window_folds)),
        "fold_count": fold_count,
        "native_source_types": native_source_types,
        "native_behavior_labels": native_behavior_labels,
        "total_selected_image_slots": int(total_slot_rows),
        "selected_windows_by_view": selected_windows_by_view,
        "image_context_source_counts": image_frames["image_context_source"]
        .value_counts(dropna=False)
        .to_dict(),
        "ordered_image_windows_match": ordered_image_windows_match,
        "ordered_fold_windows_match": ordered_fold_windows_match,
        "frame_cache_id_set_equal": frame_ids == cache_ids,
        "frame_packed_id_set_equal": frame_ids == packed_ids,
        "ordered_cache_packed_ids_match": ordered_cache_packed_ids_match,
        "wrong_slot_count_rows": int(wrong_slot_count_rows),
        "missing_cache_slot_rows": int(missing_slot_rows),
        "slot_native_unit_mismatches": slot_native_unit_mismatches,
        "slot_frame_order_mismatches": slot_frame_order_mismatches,
        "packed_rows_contiguous": packed_rows_contiguous,
        "letterbox_metadata_audit": letterbox_audit,
        "missing_native_recording_groups": missing_native_recording_groups,
        "recording_group_mismatches": recording_group_mismatches,
        "missing_window_fold_rows": missing_window_fold_rows,
        "fold_inheritance_mismatches": fold_inheritance_mismatches,
        "class_support_mismatches": class_support_mismatches,
        "source_support_mismatches": source_support_mismatches,
        "duplicate_counts": duplicate_counts,
        "errors": errors,
        "valid": not errors,
    }


def _audit_fold_tables(
    *,
    native_units: pd.DataFrame,
    temporal_selection: pd.DataFrame,
    recording_audit: dict[str, Any],
    fold_audit: dict[str, Any],
    native_folds: pd.DataFrame,
    window_folds: pd.DataFrame,
    class_support: pd.DataFrame,
    source_support: pd.DataFrame,
    group_level: str,
) -> dict[str, Any]:
    errors = list(recording_audit.get("errors", []))
    errors.extend(fold_audit.get("errors", []))
    missing_window_folds = int(window_folds["oof_fold_id"].isna().sum())
    native_keys = native_units["temporal_unit_key"].astype(str)
    window_count_by_unit = (
        window_folds.groupby("temporal_unit_key")
        .size()
        .reindex(native_keys, fill_value=0)
    )
    wrong_window_count_units = int(window_count_by_unit.ne(10).sum())
    group_leakage = int(
        native_folds.groupby("recording_group_id")["oof_fold_id"]
        .nunique()
        .gt(1)
        .sum()
    )
    video_leakage = int(
        native_folds.groupby(["source_type", "dataset_id", "video_key"])[
            "oof_fold_id"
        ]
        .nunique()
        .gt(1)
        .sum()
    )
    if missing_window_folds:
        errors.append(f"windows_without_fold={missing_window_folds}")
    if wrong_window_count_units:
        errors.append(f"native_units_without_ten_tier_windows={wrong_window_count_units}")
    if group_leakage:
        errors.append(f"recording_groups_cross_folds={group_leakage}")
    if video_leakage:
        errors.append(f"videos_cross_folds={video_leakage}")
    unsupported_test = int(class_support["test_native_units"].eq(0).sum())
    unsupported_train = int(class_support["train_native_units"].eq(0).sum())
    return {
        "schema_version": "classification_v2.legacy_development_folds.v1",
        "lineage_scope": LEGACY_DEVELOPMENT_SCOPE,
        "human_review_complete": False,
        "group_level": group_level,
        "native_input_rows": int(len(native_units)),
        "native_fold_rows": int(len(native_folds)),
        "window_input_rows": int(len(temporal_selection)),
        "window_fold_rows": int(len(window_folds)),
        "recording_group_count": int(native_folds["recording_group_id"].nunique()),
        "fold_count": int(native_folds["oof_fold_id"].nunique()),
        "missing_window_folds": missing_window_folds,
        "wrong_window_count_units": wrong_window_count_units,
        "recording_group_leakage": group_leakage,
        "video_leakage": video_leakage,
        "class_support_rows": int(len(class_support)),
        "source_support_rows": int(len(source_support)),
        "unsupported_test_class_fold_rows": unsupported_test,
        "unsupported_train_class_fold_rows": unsupported_train,
        "pig_id_used_as_cross_video_identity": False,
        "recording_group_audit": recording_audit,
        "native_oof_fold_audit": fold_audit,
        "warnings": [
            "unsupported short-packet class/fold cells are reported, not hidden",
            "short-packet folds authorize data expansion only, not model selection",
        ],
        "errors": errors,
        "valid": not errors,
    }


def _audit_letterbox_metadata(
    cache_manifest: pd.DataFrame,
    image_size: int,
) -> dict[str, Any]:
    """Verify that every cached crop preserves aspect inside its square canvas."""

    numeric = cache_manifest[sorted(LETTERBOX_METADATA_COLUMNS)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    finite = pd.Series(
        np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1),
        index=numeric.index,
    )
    width = numeric["source_crop_width"]
    height = numeric["source_crop_height"]
    expected_scale = pd.concat(
        [image_size / width, image_size / height],
        axis=1,
    ).min(axis=1)
    expected_width = (width * expected_scale).round().clip(lower=1)
    expected_height = (height * expected_scale).round().clip(lower=1)
    expected_left = ((image_size - expected_width) // 2).astype("Int64")
    expected_top = ((image_size - expected_height) // 2).astype("Int64")
    expected_right = image_size - expected_width.astype("Int64") - expected_left
    expected_bottom = image_size - expected_height.astype("Int64") - expected_top
    canvas_width = (
        numeric["letterbox_pad_left"]
        + numeric["letterbox_resized_width"]
        + numeric["letterbox_pad_right"]
    )
    canvas_height = (
        numeric["letterbox_pad_top"]
        + numeric["letterbox_resized_height"]
        + numeric["letterbox_pad_bottom"]
    )
    invalid = _invalid_letterbox_rows(
        numeric=numeric,
        finite=finite,
        width=width,
        height=height,
        expected_scale=expected_scale,
        expected_width=expected_width,
        expected_height=expected_height,
        expected_left=expected_left,
        expected_top=expected_top,
        expected_right=expected_right,
        expected_bottom=expected_bottom,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        image_size=image_size,
    )
    padded = numeric[
        [
            "letterbox_pad_left",
            "letterbox_pad_top",
            "letterbox_pad_right",
            "letterbox_pad_bottom",
        ]
    ].gt(0).any(axis=1)
    invalid_count = int(invalid.sum())
    return {
        "rows": int(len(cache_manifest)),
        "non_square_source_crop_rows": int(width.ne(height).sum()),
        "padded_canvas_rows": int(padded.sum()),
        "invalid_rows": invalid_count,
        "errors": (
            [f"invalid_letterbox_metadata_rows={invalid_count}"]
            if invalid_count
            else []
        ),
        "valid": invalid_count == 0,
    }


def _invalid_letterbox_rows(
    *,
    numeric: pd.DataFrame,
    finite: pd.Series,
    width: pd.Series,
    height: pd.Series,
    expected_scale: pd.Series,
    expected_width: pd.Series,
    expected_height: pd.Series,
    expected_left: pd.Series,
    expected_top: pd.Series,
    expected_right: pd.Series,
    expected_bottom: pd.Series,
    canvas_width: pd.Series,
    canvas_height: pd.Series,
    image_size: int,
) -> pd.Series:
    expected_aspect = width / height
    return (
        ~finite
        | width.le(0)
        | height.le(0)
        | numeric["source_crop_aspect_ratio"]
        .sub(expected_aspect)
        .abs()
        .gt(1e-9)
        | numeric["letterbox_scale"].sub(expected_scale).abs().gt(1e-9)
        | numeric["letterbox_resized_width"].ne(expected_width)
        | numeric["letterbox_resized_height"].ne(expected_height)
        | numeric["letterbox_pad_left"].ne(expected_left)
        | numeric["letterbox_pad_top"].ne(expected_top)
        | numeric["letterbox_pad_right"].ne(expected_right)
        | numeric["letterbox_pad_bottom"].ne(expected_bottom)
        | numeric["letterbox_pad_left"].lt(0)
        | numeric["letterbox_pad_top"].lt(0)
        | numeric["letterbox_pad_right"].lt(0)
        | numeric["letterbox_pad_bottom"].lt(0)
        | canvas_width.ne(image_size)
        | canvas_height.ne(image_size)
    )


def _class_by_fold_support(native_folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fold_ids = sorted(native_folds["oof_fold_id"].astype(str).unique())
    for fold_id in fold_ids:
        is_test = native_folds["oof_fold_id"].astype(str).eq(fold_id)
        for behavior in VALID_BEHAVIORS:
            is_behavior = native_folds["behavior_label"].astype(str).eq(behavior)
            rows.append(
                {
                    "oof_fold_id": fold_id,
                    "behavior_label": behavior,
                    "test_native_units": int((is_test & is_behavior).sum()),
                    "train_native_units": int((~is_test & is_behavior).sum()),
                    "test_supported": bool((is_test & is_behavior).any()),
                    "train_supported": bool((~is_test & is_behavior).any()),
                    "lineage_scope": LEGACY_DEVELOPMENT_SCOPE,
                    "human_review_complete": False,
                }
            )
    return pd.DataFrame(rows)


def _source_by_fold_support(native_folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fold_ids = sorted(native_folds["oof_fold_id"].astype(str).unique())
    sources = sorted(native_folds["source_type"].astype(str).unique())
    for fold_id in fold_ids:
        is_test = native_folds["oof_fold_id"].astype(str).eq(fold_id)
        for source in sources:
            is_source = native_folds["source_type"].astype(str).eq(source)
            rows.append(
                {
                    "oof_fold_id": fold_id,
                    "source_type": source,
                    "test_native_units": int((is_test & is_source).sum()),
                    "train_native_units": int((~is_test & is_source).sum()),
                    "test_supported": bool((is_test & is_source).any()),
                    "train_supported": bool((~is_test & is_source).any()),
                    "lineage_scope": LEGACY_DEVELOPMENT_SCOPE,
                    "human_review_complete": False,
                }
            )
    return pd.DataFrame(rows)


def _support_mismatch_count(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    keys: list[str],
) -> int:
    value_columns = [
        "test_native_units",
        "train_native_units",
        "test_supported",
        "train_supported",
    ]
    observed_values = observed[keys + value_columns].copy()
    expected_values = expected[keys + value_columns].copy()
    invalid_boolean_values = 0
    for column in ["test_supported", "train_supported"]:
        observed_bool, invalid = _strict_bool(observed_values[column])
        invalid_boolean_values += invalid
        observed_values[column] = observed_bool.astype(int)
        expected_values[column] = expected_values[column].astype(bool).astype(int)
    merged = expected_values.merge(
        observed_values,
        on=keys,
        how="outer",
        suffixes=("_expected", "_observed"),
        indicator=True,
        validate="one_to_one",
    )
    mismatched = merged["_merge"].ne("both")
    for column in value_columns:
        expected_column = f"{column}_expected"
        observed_column = f"{column}_observed"
        mismatched |= merged[expected_column].ne(merged[observed_column])
    return int(mismatched.sum()) + invalid_boolean_values


def _context_ids(value: str) -> list[str]:
    if not value or value.lower() in {"nan", "none", "<na>"}:
        return []
    return [item for item in value.split(IMAGE_CONTEXT_SEQUENCE_DELIMITER) if item]


def _frame_indices(value: str) -> list[int]:
    if not value or value.lower() in {"nan", "none", "<na>"}:
        return []
    return [int(item) for item in value.split("|") if item]


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _require_legacy_scope(frame: pd.DataFrame, name: str) -> None:
    scopes = set(frame["lineage_scope"].fillna("").astype(str))
    if scopes != {LEGACY_DEVELOPMENT_SCOPE}:
        raise ValueError(f"{name} has invalid lineage_scope values={sorted(scopes)}")
    reviewed, invalid = _strict_bool(frame["human_review_complete"])
    if invalid or reviewed.any():
        raise ValueError(
            f"{name} must remain unreviewed: invalid={invalid} true={int(reviewed.sum())}"
        )


def _strict_bool(series: pd.Series) -> tuple[pd.Series, int]:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool), int(series.isna().sum())
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "t"}
    falsy = {"false", "0", "no", "n", "f"}
    invalid = int((~normalized.isin(truthy | falsy)).sum())
    return normalized.isin(truthy), invalid
