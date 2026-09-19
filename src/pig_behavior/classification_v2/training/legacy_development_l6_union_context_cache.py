"""Read-only ResNet feature adapter for the legacy L6 union context."""

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
MODALITY_NAME = "union_context"
FEATURE_NAMES = tuple(
    f"union_context_feature_{index:03d}" for index in range(FEATURE_DIM)
)


@dataclass(frozen=True, slots=True)
class LegacyL6UnionContextCache:
    """Window-aligned mapping into the immutable ResNet feature tensor."""

    feature_tensor_path: Path
    feature_row_index: np.ndarray
    availability: np.ndarray
    window_index: pd.DataFrame
    slot_index: pd.DataFrame
    manifest: dict[str, Any]
    audit: dict[str, Any]

    def load_union_context(
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
            raise ValueError("union context feature rows are nonfinite")
        return result

    def load_availability(
        self,
        rows: np.ndarray | None = None,
    ) -> np.ndarray:
        positions = _validated_rows(rows, len(self.window_index))
        return self.availability[positions].copy()


def load_union_context_cache(
    config: Any,
    *,
    base_windows: pd.DataFrame,
) -> LegacyL6UnionContextCache:
    """Bind selected context IDs to the exact frozen T6 window order."""

    inputs = _object(config.payload["inputs"], "inputs")
    feature_tensor_path = _bound_path(config, inputs, "feature_tensor")
    feature_index_path = _bound_path(config, inputs, "feature_index")
    feature_audit_path = _bound_path(config, inputs, "feature_audit")
    context_path = _bound_path(config, inputs, "image_window_context_manifest")
    subset_audit_path = _bound_path(config, inputs, "window_subset_audit")
    feature_audit = _read_json(feature_audit_path)
    subset_audit = _read_json(subset_audit_path)
    _validate_source_hashes(
        inputs,
        feature_tensor_path=feature_tensor_path,
        feature_index_path=feature_index_path,
        feature_audit_path=feature_audit_path,
        context_path=context_path,
        subset_audit_path=subset_audit_path,
    )
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
    feature_index = pd.read_csv(feature_index_path)
    required_index = {
        "image_context_id",
        "feature_row",
        "packed_row",
        "lineage_scope",
        "human_review_complete",
        "control_id",
        "backbone_name",
        "pretrained_weight_enum",
        "image_size",
        "feature_dim",
        "feature_dtype",
    }
    if set(feature_index.columns) != required_index:
        raise ValueError("union feature index columns are incomplete")
    if feature_index["image_context_id"].astype(str).duplicated().any():
        raise ValueError("union feature index contains duplicate context IDs")
    feature_rows = feature_index["feature_row"].astype(np.int64).to_numpy()
    if not np.array_equal(feature_rows, np.arange(len(feature_index))):
        raise ValueError("union feature rows are not contiguous")
    packed_rows = feature_index["packed_row"].astype(np.int64).to_numpy()
    if not np.array_equal(packed_rows, feature_rows):
        raise ValueError("union feature packed rows are not contiguous")
    _validate_feature_index_metadata(feature_index)
    tensor = np.load(feature_tensor_path, mmap_mode="r")
    try:
        expected_shape = (len(feature_index), FEATURE_DIM)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"union feature tensor shape={tuple(tensor.shape)}"
                f"!={expected_shape}"
            )
        if tensor.dtype != np.float32:
            raise ValueError(f"union feature dtype={tensor.dtype}")
        if not np.isfinite(tensor).all():
            raise ValueError("union feature tensor contains nonfinite values")
    finally:
        _close_memmap(tensor)
    feature_row_by_id = dict(
        zip(
            feature_index["image_context_id"].astype(str),
            feature_rows,
            strict=True,
        )
    )
    context = pd.read_csv(context_path)
    required_context = {
        "window_id",
        "source_type",
        "dataset_id",
        "lineage_scope",
        "human_review_complete",
        "frame_uid_sequence",
        "image_context_id_sequence",
        "observed_image_context_rows",
        "loadable_image_context_rows",
        "missing_image_context_slots",
        "window_image_context_complete",
    }
    if not required_context.issubset(context.columns):
        raise ValueError("union context manifest columns are incomplete")
    if context["window_id"].astype(str).duplicated().any():
        raise ValueError("union context manifest has duplicate windows")
    _validate_context_metadata(
        context,
        config=config,
        subset_audit=subset_audit,
    )
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
    context_ids_by_row: list[list[str]] = []
    frame_ids_by_row: list[list[str]] = []
    for cache_row, window_id in enumerate(base["window_id"].astype(str)):
        if window_id not in selected_rows:
            ids = [""] * SEQUENCE_LENGTH
            frames = [""] * SEQUENCE_LENGTH
        else:
            item = selected.loc[window_id]
            ids = _context_id_sequence(item["image_context_id_sequence"])
            frames = _frame_uid_sequence(item["frame_uid_sequence"])
            if len(ids) != SEQUENCE_LENGTH or len(frames) != SEQUENCE_LENGTH:
                raise ValueError("union context sequence length drift")
            for slot, context_id in enumerate(ids):
                if context_id not in feature_row_by_id:
                    raise ValueError(
                        "selected union context ID is absent from feature cache"
                    )
                row_index[cache_row, slot] = feature_row_by_id[context_id]
                available[cache_row, slot] = True
        context_ids_by_row.append(ids)
        frame_ids_by_row.append(frames)
    expected_selected = int(subset_audit["selected_windows"])
    available_window_ids = set(
        base.loc[available.any(axis=1), "window_id"].astype(str)
    )
    if selected_rows != available_window_ids:
        raise ValueError("union selected-window membership drift")
    selected_context_ids = {
        context_id
        for ids in context_ids_by_row
        for context_id in ids
        if context_id
    }
    if selected_context_ids != set(feature_row_by_id):
        raise ValueError("union selected context-ID membership drift")
    if int(available.any(axis=1).sum()) != expected_selected:
        raise ValueError("union selected-window count drift")
    if int(available.sum()) != expected_selected * SEQUENCE_LENGTH:
        raise ValueError("union selected-slot count drift")
    slot_rows: list[dict[str, Any]] = []
    for cache_row, window in base.iterrows():
        window_id = str(window["window_id"])
        for slot in range(SEQUENCE_LENGTH):
            slot_rows.append(
                {
                    "cache_row": cache_row,
                    "window_id": window_id,
                    "slot_index": slot,
                    "frame_uid": frame_ids_by_row[cache_row][slot],
                    "image_context_id": context_ids_by_row[cache_row][slot],
                    "union_context_window_slot_uid": (
                        f"{window_id}::slot={slot}"
                    ),
                    "union_context_available": bool(
                        available[cache_row, slot]
                    ),
                }
            )
    slot_index = pd.DataFrame.from_records(slot_rows)
    content = {
        "union_context_only": True,
        "actor_partner_union_crop_only": True,
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
        "source_context_ids_in_model_x": False,
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
    semantic = {
        "schema_version": "classification_v2.legacy_development_l6."
        "union_context_reader.v1",
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
        "status": "PASS_LEGACY_DEVELOPMENT_L6_UNION_CONTEXT_READER",
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
    return LegacyL6UnionContextCache(
        feature_tensor_path=feature_tensor_path,
        feature_row_index=row_index,
        availability=available,
        window_index=base.copy(),
        slot_index=slot_index,
        manifest=manifest,
        audit=audit,
    )


def _bound_path(config: Any, inputs: dict[str, Any], name: str) -> Path:
    spec = _object(inputs[name], f"inputs.{name}")
    path = config.repo_root / str(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(f"union context input missing={path}")
    if file_sha256(path) != str(spec["sha256"]):
        raise ValueError(f"union context input hash drift={name}")
    return path.resolve()


def _validate_source_hashes(
    inputs: dict[str, Any],
    **paths: Path,
) -> None:
    for name, path in paths.items():
        key = {
            "feature_tensor_path": "feature_tensor",
            "feature_index_path": "feature_index",
            "feature_audit_path": "feature_audit",
            "context_path": "image_window_context_manifest",
            "subset_audit_path": "window_subset_audit",
        }[name]
        expected = str(_object(inputs[key], f"inputs.{key}")["sha256"])
        if file_sha256(path) != expected:
            raise ValueError(f"union context source hash mismatch={key}")


def _validate_feature_audit(
    audit: dict[str, Any],
    *,
    config: Any,
    feature_tensor_path: Path,
    feature_index_path: Path,
) -> None:
    selection = _object(config.payload["selection"], "selection")
    model = _object(config.payload["model"], "model")
    expected = {
        "schema_version": (
            "classification_v2.legacy_development_l6."
            "union_context_resnet18_features.v1"
        ),
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_UNION_CONTEXT_RESNET18_FEATURES"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "canonical_source_name": CANONICAL_SOURCE_NAME,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "backbone_name": model["backbone_name"],
        "pretrained_weight_enum": "ResNet18_Weights.IMAGENET1K_V1",
        "image_size": model["input_resolution"],
        "feature_dim": FEATURE_DIM,
        "feature_dtype": "float32",
        "rows": selection["selected_image_context_ids"],
        "feature_tensor_sha256": file_sha256(feature_tensor_path),
        "feature_index_sha256": file_sha256(feature_index_path),
        "errors": [],
        "valid": True,
    }
    _require_expected_values(audit, expected, "union feature audit")
    source_tensor = _object(audit.get("source_tensor"), "source_tensor")
    if int(source_tensor.get("rows", -1)) != int(expected["rows"]):
        raise ValueError("union feature source tensor row-count drift")


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
    _require_expected_values(audit, expected, "union subset audit")


def _validate_feature_index_metadata(feature_index: pd.DataFrame) -> None:
    expected = {
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "control_id": "UNION_CONTEXT",
        "backbone_name": "resnet18",
        "pretrained_weight_enum": "ResNet18_Weights.IMAGENET1K_V1",
        "image_size": 224,
        "feature_dim": FEATURE_DIM,
        "feature_dtype": "float32",
    }
    _require_constant_columns(
        feature_index,
        expected,
        "union feature index",
    )


def _validate_context_metadata(
    context: pd.DataFrame,
    *,
    config: Any,
    subset_audit: dict[str, Any],
) -> None:
    selection = _object(config.payload["selection"], "selection")
    if len(context) != int(selection["selected_windows"]):
        raise ValueError("union context manifest row-count drift")
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
    _require_constant_columns(context, expected, "union context manifest")
    if int(subset_audit["missing_context_ids"]) != 0:
        raise ValueError("union context audit reports missing context IDs")
    if int(subset_audit["outer_holdout_windows"]) != 0:
        raise ValueError("union context audit reports outer-holdout windows")


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


def _context_id_sequence(value: object) -> list[str]:
    text = "" if value is None else str(value)
    values = text.split(";;") if text else []
    return [value for value in values if value != ""]


def _frame_uid_sequence(value: object) -> list[str]:
    text = "" if value is None else str(value)
    values = text.split("|") if text else []
    return [value for value in values if value != ""]


def _validated_rows(values: np.ndarray | None, maximum: int) -> np.ndarray:
    if values is None:
        return np.arange(maximum, dtype=np.int64)
    rows = np.asarray(values, dtype=np.int64)
    if rows.ndim != 1 or len(rows) == 0:
        raise ValueError("union context rows must be a nonempty vector")
    if rows.min() < 0 or rows.max() >= maximum:
        raise ValueError("union context rows are out of bounds")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("union context rows contain duplicates")
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _object(payload, str(path))


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"union context {name} must be an object")
    return value


def _close_memmap(value: Any) -> None:
    close = getattr(value, "_mmap", None)
    if close is not None:
        close.close()
