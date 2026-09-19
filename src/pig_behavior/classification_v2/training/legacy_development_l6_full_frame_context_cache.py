"""Read-only ResNet feature adapter for legacy L6 full-frame context."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

SEQUENCE_LENGTH = 6
FEATURE_DIM = 512
LINEAGE_SCOPE = "legacy-only-unreviewed-development"
CANONICAL_SOURCE_NAME = "legacy_16f"
SOURCE_TYPE = "legacy_recovered"
DATASET_ID = "legacy_recovered_16f"
MODALITY_NAME = "full_frame_context"
FEATURE_NAMES = tuple(
    f"feature_{index:03d}" for index in range(FEATURE_DIM)
)


@dataclass(frozen=True, slots=True)
class LegacyL6FullFrameContextCache:
    """Window-aligned mapping into the immutable full-frame feature tensor."""

    feature_tensor_path: Path
    feature_row_index: np.ndarray
    availability: np.ndarray
    window_index: pd.DataFrame
    slot_index: pd.DataFrame
    manifest: dict[str, Any]
    audit: dict[str, Any]

    def load_full_frame_context(
        self,
        rows: np.ndarray | None = None,
    ) -> np.ndarray:
        positions = _validated_rows(rows, len(self.window_index))
        tensor = np.load(self.feature_tensor_path, mmap_mode="r")
        try:
            result = np.zeros(
                (len(positions), SEQUENCE_LENGTH, FEATURE_DIM),
                dtype=np.float32,
            )
            row_map = self.feature_row_index[positions]
            valid = row_map >= 0
            if valid.any():
                result[valid] = np.asarray(
                    tensor[row_map[valid]],
                    dtype=np.float32,
                )
        finally:
            _close_memmap(tensor)
        if not np.isfinite(result).all():
            raise ValueError("full-frame context features are nonfinite")
        return result

    def load_availability(
        self,
        rows: np.ndarray | None = None,
    ) -> np.ndarray:
        positions = _validated_rows(rows, len(self.window_index))
        return self.availability[positions].copy()


def load_full_frame_context_cache(
    config: Any,
    *,
    base_windows: pd.DataFrame,
) -> LegacyL6FullFrameContextCache:
    """Bind selected scene IDs to the exact frozen T6 window order."""

    inputs = _object(config.payload["inputs"], "inputs")
    feature_tensor_path = _bound_path(config, inputs, "feature_tensor")
    feature_index_path = _bound_path(config, inputs, "feature_index")
    feature_audit_path = _bound_path(config, inputs, "feature_audit")
    context_path = _bound_path(
        config,
        inputs,
        "image_window_context_manifest",
    )
    subset_audit_path = _bound_path(
        config,
        inputs,
        "window_subset_audit",
    )
    feature_audit = _read_json(feature_audit_path)
    subset_audit = _read_json(subset_audit_path)
    _validate_feature_audit(
        feature_audit,
        config=config,
        feature_tensor_path=feature_tensor_path,
        feature_index_path=feature_index_path,
    )
    _validate_subset_audit(
        subset_audit,
        config=config,
        context_path=context_path,
    )
    feature_index = pd.read_csv(feature_index_path, low_memory=False)
    _validate_feature_index(feature_index)
    feature_rows = feature_index["feature_row"].astype(np.int64).to_numpy()
    tensor = np.load(feature_tensor_path, mmap_mode="r")
    try:
        expected_shape = (len(feature_index), FEATURE_DIM)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"full-frame tensor shape={tuple(tensor.shape)}"
                f"!={expected_shape}"
            )
        if tensor.dtype != np.float32:
            raise ValueError(f"full-frame feature dtype={tensor.dtype}")
        if not np.isfinite(tensor).all():
            raise ValueError("full-frame feature tensor is nonfinite")
    finally:
        _close_memmap(tensor)
    feature_row_by_id = dict(
        zip(
            feature_index["scene_frame_uid"].astype(str),
            feature_rows,
            strict=True,
        )
    )
    context = pd.read_csv(context_path, low_memory=False)
    _validate_context(context, config=config, subset_audit=subset_audit)
    selected = context.set_index("window_id", drop=False)
    selected_rows = set(selected.index.astype(str))
    base = base_windows.reset_index(drop=True)
    row_index = np.full(
        (len(base), SEQUENCE_LENGTH),
        -1,
        dtype=np.int64,
    )
    available = np.zeros(
        (len(base), SEQUENCE_LENGTH),
        dtype=np.bool_,
    )
    scene_ids_by_row: list[list[str]] = []
    for cache_row, window_id in enumerate(base["window_id"].astype(str)):
        if window_id not in selected_rows:
            scene_ids = [""] * SEQUENCE_LENGTH
        else:
            item = selected.loc[window_id]
            scene_ids = _scene_id_sequence(
                item["scene_frame_uid_sequence"]
            )
            if len(scene_ids) != SEQUENCE_LENGTH:
                raise ValueError("full-frame scene sequence length drift")
            for slot, scene_id in enumerate(scene_ids):
                if scene_id not in feature_row_by_id:
                    raise ValueError(
                        "selected scene ID is absent from full-frame cache"
                    )
                row_index[cache_row, slot] = feature_row_by_id[scene_id]
                available[cache_row, slot] = True
        scene_ids_by_row.append(scene_ids)
    expected_selected = int(subset_audit["selected_windows"])
    available_window_ids = set(
        base.loc[available.any(axis=1), "window_id"].astype(str)
    )
    if selected_rows != available_window_ids:
        raise ValueError("full-frame selected-window membership drift")
    selected_scene_ids = {
        scene_id
        for scene_ids in scene_ids_by_row
        for scene_id in scene_ids
        if scene_id
    }
    if selected_scene_ids != set(feature_row_by_id):
        raise ValueError("full-frame selected scene membership drift")
    if int(available.any(axis=1).sum()) != expected_selected:
        raise ValueError("full-frame selected-window count drift")
    if int(available.sum()) != expected_selected * SEQUENCE_LENGTH:
        raise ValueError("full-frame selected-slot count drift")
    slot_index = _slot_index(base, scene_ids_by_row, available)
    content = _content_audit(available, expected_selected)
    semantic = {
        "schema_version": (
            "classification_v2.legacy_development_l6."
            "full_frame_context_reader.v1"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "canonical_source_name": CANONICAL_SOURCE_NAME,
        "base_window_id_sha256": _ordered_sha256(base["window_id"]),
        "selection_content_sha256": str(
            config.payload["selection"]["selection_content_sha256"]
        ),
        "feature_tensor_sha256": file_sha256(feature_tensor_path),
        "feature_index_sha256": file_sha256(feature_index_path),
        "content": content,
    }
    manifest_sha = _payload_sha256(semantic)
    manifest = {
        "schema_version": semantic["schema_version"],
        "status": "PASS_LEGACY_DEVELOPMENT_L6_FULL_FRAME_CONTEXT_READER",
        "lineage_scope": LINEAGE_SCOPE,
        "canonical_source_name": CANONICAL_SOURCE_NAME,
        "source_type": SOURCE_TYPE,
        "dataset_id": DATASET_ID,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "feature_audit_sha256": file_sha256(feature_audit_path),
        "window_subset_audit_sha256": file_sha256(subset_audit_path),
        "content_audit": content,
        "semantic_sha256": manifest_sha,
        "valid": True,
    }
    audit = {
        "manifest_sha256": manifest_sha,
        "source_media_reads": 0,
        "outer_holdout_slots_materialized": 0,
        "content_audit": content,
        "source_feature_tensor_sha256": file_sha256(feature_tensor_path),
        "source_feature_index_sha256": file_sha256(feature_index_path),
        "source_feature_audit_sha256": file_sha256(feature_audit_path),
        "valid": True,
    }
    return LegacyL6FullFrameContextCache(
        feature_tensor_path=feature_tensor_path,
        feature_row_index=row_index,
        availability=available,
        window_index=base.copy(),
        slot_index=slot_index,
        manifest=manifest,
        audit=audit,
    )


def _validate_feature_index(frame: pd.DataFrame) -> None:
    required = {
        "scene_frame_uid",
        "feature_row",
        "packed_row",
        "selection_order",
        "video_key",
        "frame_index",
        "lineage_scope",
        "human_review_complete",
        "source_width",
        "source_height",
        "resized_width",
        "resized_height",
        "pad_left",
        "pad_right",
        "pad_top",
        "pad_bottom",
        "resize_policy",
        "control_id",
        "backbone_name",
        "pretrained_weight_enum",
        "image_size",
        "feature_dim",
        "feature_dtype",
    }
    if set(frame.columns) != required:
        raise ValueError("full-frame feature index columns drift")
    if frame["scene_frame_uid"].astype(str).duplicated().any():
        raise ValueError("full-frame feature index has duplicate scene IDs")
    expected_rows = np.arange(len(frame), dtype=np.int64)
    if not np.array_equal(frame["feature_row"].astype(np.int64), expected_rows):
        raise ValueError("full-frame feature rows are not contiguous")
    if not np.array_equal(frame["packed_row"].astype(np.int64), expected_rows):
        raise ValueError("full-frame packed rows are not contiguous")
    expected = {
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "resize_policy": "full_frame_letterbox_rgb_pad_black_v1",
        "control_id": "FULL_FRAME_CONTEXT",
        "backbone_name": "resnet18",
        "pretrained_weight_enum": "ResNet18_Weights.IMAGENET1K_V1",
        "image_size": 224,
        "feature_dim": FEATURE_DIM,
        "feature_dtype": "float32",
    }
    _require_constant_columns(frame, expected, "full-frame feature index")


def _validate_feature_audit(
    audit: dict[str, Any],
    *,
    config: Any,
    feature_tensor_path: Path,
    feature_index_path: Path,
) -> None:
    selection = _object(config.payload["selection"], "selection")
    expected = {
        "schema_version": (
            "classification_v2.legacy_development_l6."
            "full_frame_resnet18_features.v1"
        ),
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_FULL_FRAME_RESNET18_FEATURES"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "canonical_source_name": CANONICAL_SOURCE_NAME,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "backbone_name": "resnet18",
        "pretrained_weight_enum": "ResNet18_Weights.IMAGENET1K_V1",
        "image_size": 224,
        "feature_dim": FEATURE_DIM,
        "feature_dtype": "float32",
        "rows": selection["selected_scene_frames"],
        "source_media_reads": 0,
        "outer_holdout_rows": 0,
        "feature_tensor_sha256": file_sha256(feature_tensor_path),
        "feature_index_sha256": file_sha256(feature_index_path),
        "errors": [],
        "valid": True,
    }
    _require_expected_values(audit, expected, "full-frame feature audit")


def _validate_subset_audit(
    audit: dict[str, Any],
    *,
    config: Any,
    context_path: Path,
) -> None:
    selection = _object(config.payload["selection"], "selection")
    expected = {
        "schema_version": (
            "classification_v2.legacy_development_l6."
            "union_context_window_subset.v1"
        ),
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_UNION_CONTEXT_WINDOW_SUBSET"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "canonical_source_name": CANONICAL_SOURCE_NAME,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "selection_manifest_sha256": selection["selection_content_sha256"],
        "selected_windows": selection["selected_windows"],
        "selected_native_units": (
            selection["short_train_native_units"]
            + selection["validation_native_units"]
        ),
        "train_windows": selection["short_train_windows"],
        "validation_windows": selection["validation_windows"],
        "validation_native_units": selection["validation_native_units"],
        "missing_context_ids": 0,
        "outer_holdout_windows": 0,
        "output_manifest_sha256": file_sha256(context_path),
        "valid": True,
    }
    _require_expected_values(audit, expected, "full-frame subset audit")


def _validate_context(
    frame: pd.DataFrame,
    *,
    config: Any,
    subset_audit: dict[str, Any],
) -> None:
    required = {
        "window_id",
        "source_type",
        "dataset_id",
        "lineage_scope",
        "human_review_complete",
        "scene_frame_uid_sequence",
        "observed_image_context_rows",
        "loadable_image_context_rows",
        "missing_image_context_slots",
        "window_image_context_complete",
    }
    if not required.issubset(frame.columns):
        raise ValueError("full-frame context columns are incomplete")
    if frame["window_id"].astype(str).duplicated().any():
        raise ValueError("full-frame context has duplicate windows")
    selection = _object(config.payload["selection"], "selection")
    if len(frame) != int(selection["selected_windows"]):
        raise ValueError("full-frame context row-count drift")
    expected = {
        "source_type": SOURCE_TYPE,
        "dataset_id": DATASET_ID,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "observed_image_context_rows": SEQUENCE_LENGTH,
        "loadable_image_context_rows": SEQUENCE_LENGTH,
        "missing_image_context_slots": 0,
        "window_image_context_complete": True,
    }
    _require_constant_columns(frame, expected, "full-frame context")
    if int(subset_audit["missing_context_ids"]) != 0:
        raise ValueError("full-frame subset reports missing context IDs")
    if int(subset_audit["outer_holdout_windows"]) != 0:
        raise ValueError("full-frame subset reports outer-holdout windows")


def _slot_index(
    base: pd.DataFrame,
    scene_ids_by_row: list[list[str]],
    available: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cache_row, window in base.iterrows():
        window_id = str(window["window_id"])
        for slot in range(SEQUENCE_LENGTH):
            rows.append(
                {
                    "cache_row": cache_row,
                    "window_id": window_id,
                    "slot_index": slot,
                    "scene_frame_uid": scene_ids_by_row[cache_row][slot],
                    "full_frame_context_available": bool(
                        available[cache_row, slot]
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def _content_audit(
    available: np.ndarray,
    expected_selected: int,
) -> dict[str, Any]:
    return {
        "full_frame_context_only": True,
        "scene_full_frame_only": True,
        "selected_windows": expected_selected,
        "selected_slots": int(available.sum()),
        "available_windows": int(available.any(axis=1).sum()),
        "available_slots": int(available.sum()),
        "unavailable_windows": int((~available.any(axis=1)).sum()),
        "availability_patterns": [
            {
                "pattern": [0] * SEQUENCE_LENGTH,
                "windows": int((~available.any(axis=1)).sum()),
            },
            {
                "pattern": [1] * SEQUENCE_LENGTH,
                "windows": int(available.all(axis=1).sum()),
            },
        ],
        "source_scene_ids_in_model_x": False,
        "union_context_values_in_model_x": False,
        "geometry_values_in_model_x": False,
        "motion_values_in_model_x": False,
        "roi_values_in_model_x": False,
        "social_values_in_model_x": False,
        "unit_aggregate_features_in_model_x": False,
        "labels_paths_ids_folds_review_fields_in_model_x": False,
        "availability_is_behavior_evidence": False,
        "source_media_reads": 0,
        "outer_holdout_slots_materialized": 0,
    }


def _bound_path(config: Any, inputs: dict[str, Any], name: str) -> Path:
    spec = _object(inputs[name], f"inputs.{name}")
    path = (config.repo_root / str(spec["path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"full-frame input missing={path}")
    if file_sha256(path) != str(spec["sha256"]):
        raise ValueError(f"full-frame input hash drift={name}")
    return path


def _require_constant_columns(
    frame: pd.DataFrame,
    expected: dict[str, Any],
    name: str,
) -> None:
    for column, value in expected.items():
        if column not in frame or not frame[column].eq(value).all():
            raise ValueError(f"{name} {column} drift")


def _require_expected_values(
    payload: dict[str, Any],
    expected: dict[str, Any],
    name: str,
) -> None:
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"{name} {field} drift")


def _scene_id_sequence(value: object) -> list[str]:
    text = "" if value is None else str(value)
    values = text.split("|") if text else []
    return [item for item in values if item]


def _validated_rows(
    values: np.ndarray | None,
    maximum: int,
) -> np.ndarray:
    if values is None:
        return np.arange(maximum, dtype=np.int64)
    rows = np.asarray(values, dtype=np.int64)
    if rows.ndim != 1 or len(rows) == 0:
        raise ValueError("full-frame rows must be a nonempty vector")
    if rows.min() < 0 or rows.max() >= maximum:
        raise ValueError("full-frame rows are out of bounds")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("full-frame rows contain duplicates")
    return rows


def _ordered_sha256(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values.fillna("").astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return _object(json.loads(path.read_text(encoding="utf-8")), str(path))


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"full-frame {name} must be an object")
    return value


def _close_memmap(value: Any) -> None:
    close = getattr(value, "_mmap", None)
    if close is not None:
        close.close()
