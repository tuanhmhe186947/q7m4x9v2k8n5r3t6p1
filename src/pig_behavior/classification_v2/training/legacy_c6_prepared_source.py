"""Prepare and load one clean, bounded legacy C6 development source."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.legacy_development_l5_cached_data import (
    FEATURE_DIM,
    LegacyL5CachedFeatureView,
)
from pig_behavior.classification_v2.training.legacy_development_l5_cached_training import (
    LegacyL5CachedShortSelection,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

PACKET_SCHEMA = "classification_v2.legacy_c6_prepared_source.v1"
LINEAGE_SCOPE = "legacy-only-unreviewed-development"
SEQUENCE_LENGTH = 16


@dataclass(frozen=True, slots=True)
class LegacyC6PreparedTables:
    """Selected native units and their complete harmonized frame rows."""

    units: pd.DataFrame
    frames: pd.DataFrame
    audit: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LegacyC6PreparedSource:
    """Model-visible T16 view and deterministic short selection."""

    base_view: LegacyL5CachedFeatureView
    selection: LegacyL5CachedShortSelection
    parent_audit: dict[str, Any]
    source_config_sha256: str


def prepare_legacy_c6_tables(
    harmonized_frames: pd.DataFrame,
    native_units: pd.DataFrame,
    native_folds: pd.DataFrame,
    *,
    outer_fold_id: str = "native_oof_005",
    validation_fold_id: str = "native_oof_006",
    train_units_per_class: int | None = 8,
    train_selection_policy: str = "fixed_per_class",
    selection_salt: str = "legacy_c6_rebuild_20260719_v1",
) -> LegacyC6PreparedTables:
    """Select train/validation units without exposing outer-fold media."""

    if outer_fold_id == validation_fold_id:
        raise ValueError("outer and validation fold IDs must differ")
    if train_selection_policy not in {"fixed_per_class", "all_eligible"}:
        raise ValueError("unsupported C6 train selection policy")
    if train_selection_policy == "fixed_per_class":
        if train_units_per_class is None or train_units_per_class <= 0:
            raise ValueError("train_units_per_class must be positive")
    elif train_units_per_class is not None:
        raise ValueError(
            "all_eligible C6 selection must not declare a class cap"
        )
    _require_columns(
        native_units,
        {
            "temporal_unit_key",
            "source_type",
            "dataset_id",
            "video_key",
            "behavior_label",
            "native_unit_valid_for_development",
            "native_unit_valid_for_main_eval",
            "lineage_scope",
            "human_review_complete",
        },
        "native units",
    )
    _require_columns(
        native_folds,
        {
            "temporal_unit_key",
            "recording_group_id",
            "oof_fold_id",
            "behavior_label",
            "native_unit_valid_for_main_eval",
            "lineage_scope",
            "human_review_complete",
        },
        "native folds",
    )
    _require_columns(
        harmonized_frames,
        {
            "temporal_unit_key",
            "frame_uid",
            "scene_frame_uid",
            "relative_frame_index",
            "timestamp_sec",
            "behavior_temporal_final",
            "crop_path",
            "bbox_valid",
            "spatiotemporal_feature_valid",
            "include_in_training",
            "use_for_main_eval",
            "lineage_scope",
            "human_review_complete",
        },
        "harmonized frames",
    )
    for name, frame in {
        "native units": native_units,
        "native folds": native_folds,
        "harmonized frames": harmonized_frames,
    }.items():
        _require_unreviewed_claim(frame, name)
    _require_unique(native_units, "temporal_unit_key", "native units")
    _require_unique(native_folds, "temporal_unit_key", "native folds")

    native_keys = set(native_units["temporal_unit_key"].astype(str))
    fold_keys = set(native_folds["temporal_unit_key"].astype(str))
    if native_keys != fold_keys:
        raise ValueError("native unit and fold universes differ")
    fold_columns = [
        "temporal_unit_key",
        "recording_group_id",
        "oof_fold_id",
        "behavior_label",
        "native_unit_valid_for_main_eval",
    ]
    fold = native_folds[fold_columns].copy()
    fold = fold.rename(
        columns={
            "behavior_label": "fold_behavior_label",
            "native_unit_valid_for_main_eval": "fold_main_eval",
        }
    )
    units = native_units.merge(
        fold,
        on="temporal_unit_key",
        how="inner",
        validate="one_to_one",
    )
    if units["behavior_label"].astype(str).ne(
        units["fold_behavior_label"].astype(str)
    ).any():
        raise ValueError("native unit behavior differs from fold behavior")
    valid = (
        _strict_bool(units["native_unit_valid_for_development"])
        & _strict_bool(units["native_unit_valid_for_main_eval"])
        & _strict_bool(units["fold_main_eval"])
    )
    units = units.loc[valid].copy()
    observed_folds = set(units["oof_fold_id"].astype(str))
    if outer_fold_id not in observed_folds or validation_fold_id not in observed_folds:
        raise ValueError("declared outer or validation fold is absent")

    train_pool = units.loc[
        ~units["oof_fold_id"].astype(str).isin(
            {outer_fold_id, validation_fold_id}
        )
    ].copy()
    train_pool["selection_score"] = train_pool["temporal_unit_key"].map(
        lambda value: _selection_score(selection_salt, str(value))
    )
    train_pool = train_pool.sort_values(
        ["behavior_label", "selection_score", "temporal_unit_key"],
        kind="mergesort",
    )
    if train_selection_policy == "all_eligible":
        train = train_pool.copy()
    else:
        train = train_pool.groupby(
            "behavior_label",
            sort=False,
            group_keys=False,
        ).head(int(train_units_per_class))
    train_counts = train["behavior_label"].value_counts().to_dict()
    if train_selection_policy == "fixed_per_class":
        expected_counts = {
            label: int(train_units_per_class) for label in VALID_BEHAVIORS
        }
        if train_counts != expected_counts:
            raise ValueError(
                "C6 train class support="
                f"{train_counts} expected={expected_counts}"
            )
    elif set(train_counts) != set(VALID_BEHAVIORS):
        raise ValueError(f"C6 all-train class support={train_counts}")
    train = train.copy()
    train["l5_role"] = "train"
    validation = units.loc[
        units["oof_fold_id"].astype(str).eq(validation_fold_id)
    ].copy()
    if validation.empty:
        raise ValueError("C6 validation fold is empty")
    validation["selection_score"] = validation["temporal_unit_key"].map(
        lambda value: _selection_score("all_validation", str(value))
    )
    validation["l5_role"] = "validation"
    validation = validation.sort_values("temporal_unit_key", kind="mergesort")
    selected = pd.concat([train, validation], ignore_index=True)
    selected["position"] = np.arange(len(selected), dtype=np.int64)
    selected["window_id"] = selected["temporal_unit_key"].map(
        lambda value: "legacy_c6_t16::" + hashlib.sha256(
            str(value).encode("utf-8")
        ).hexdigest()[:24]
    )
    selected["lineage_scope"] = LINEAGE_SCOPE
    selected["human_review_complete"] = False
    selected["review_status"] = (
        "operator_cvat_checked_pending_hidden_behavior_double_check"
    )
    selected_keys = set(selected["temporal_unit_key"].astype(str))
    outer_keys = set(
        units.loc[
            units["oof_fold_id"].astype(str).eq(outer_fold_id),
            "temporal_unit_key",
        ].astype(str)
    )
    if selected_keys.intersection(outer_keys):
        raise ValueError("C6 model-visible selection contains outer units")

    frames = harmonized_frames.loc[
        harmonized_frames["temporal_unit_key"].astype(str).isin(selected_keys)
    ].copy()
    frames = frames.merge(
        selected[
            [
                "temporal_unit_key",
                "position",
                "l5_role",
                "recording_group_id",
                "behavior_label",
            ]
        ],
        on="temporal_unit_key",
        how="inner",
        validate="many_to_one",
    )
    frames["relative_frame_index"] = pd.to_numeric(
        frames["relative_frame_index"],
        errors="coerce",
    )
    frames = frames.sort_values(
        ["position", "relative_frame_index", "frame_uid"],
        kind="mergesort",
    ).reset_index(drop=True)
    frame_summary = frames.groupby("temporal_unit_key", sort=False).agg(
        rows=("relative_frame_index", "size"),
        frames=("relative_frame_index", "nunique"),
        frame_min=("relative_frame_index", "min"),
        frame_max=("relative_frame_index", "max"),
        behavior_count=("behavior_temporal_final", "nunique"),
        bbox_all=("bbox_valid", _all_bool),
        spatial_all=("spatiotemporal_feature_valid", _all_bool),
        include_all=("include_in_training", _all_bool),
        main_eval_all=("use_for_main_eval", _all_bool),
        crop_count=("crop_path", _nonblank_count),
    )
    complete = (
        frame_summary["rows"].eq(SEQUENCE_LENGTH)
        & frame_summary["frames"].eq(SEQUENCE_LENGTH)
        & frame_summary["frame_min"].eq(0)
        & frame_summary["frame_max"].eq(SEQUENCE_LENGTH - 1)
        & frame_summary["behavior_count"].eq(1)
        & frame_summary["bbox_all"]
        & frame_summary["spatial_all"]
        & frame_summary["include_all"]
        & frame_summary["main_eval_all"]
        & frame_summary["crop_count"].eq(SEQUENCE_LENGTH)
    )
    if not bool(complete.all()):
        bad = frame_summary.index[~complete].astype(str).tolist()
        raise ValueError(f"C6 selected units are incomplete={bad[:10]}")
    frame_labels = frames.groupby("temporal_unit_key", sort=False)[
        "behavior_temporal_final"
    ].first()
    unit_labels = selected.set_index("temporal_unit_key")["behavior_label"]
    if frame_labels.astype(str).ne(unit_labels.loc[frame_labels.index].astype(str)).any():
        raise ValueError("C6 selected frame and native labels differ")
    frames["slot_index"] = frames["relative_frame_index"].astype(np.int64)
    frames["feature_row"] = np.arange(len(frames), dtype=np.int64)
    frames["image_context_id"] = frames["frame_uid"].astype(str)

    audit = {
        "schema_version": "classification_v2.legacy_c6_selection.v1",
        "status": "PASS_LEGACY_C6_PREPARED_SELECTION",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "outer_fold_id": outer_fold_id,
        "validation_fold_id": validation_fold_id,
        "outer_metadata_units_read": int(len(outer_keys)),
        "outer_holdout_media_reads": 0,
        "outer_holdout_features_created": 0,
        "outer_holdout_predictions_created": 0,
        "train_selection_policy": train_selection_policy,
        "train_units_per_class": train_units_per_class,
        "train_native_units": int(len(train)),
        "validation_native_units": int(len(validation)),
        "model_visible_native_units": int(len(selected)),
        "model_visible_frame_rows": int(len(frames)),
        "train_class_counts": {
            label: int(train_counts[label]) for label in VALID_BEHAVIORS
        },
        "selection_salt": selection_salt,
        "recording_group_overlap_train_validation": int(
            len(
                set(train["recording_group_id"].astype(str)).intersection(
                    set(validation["recording_group_id"].astype(str))
                )
            )
        ),
        "outer_units_in_model_view": 0,
        "errors": [],
        "valid": True,
    }
    if audit["recording_group_overlap_train_validation"]:
        raise ValueError("C6 train and validation recording groups overlap")
    return LegacyC6PreparedTables(
        units=selected.reset_index(drop=True),
        frames=frames,
        audit=audit,
    )


def load_legacy_c6_prepared_source(
    packet_path: Path,
    *,
    repo_root: Path,
) -> LegacyC6PreparedSource:
    """Load a hash-bound prepared packet without reading source media."""

    packet_path = packet_path.resolve()
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    if packet.get("schema_version") != PACKET_SCHEMA:
        raise ValueError("legacy C6 prepared packet schema drift")
    expected_claims = {
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "outer_holdout_media_reads": 0,
        "outer_holdout_features_created": 0,
        "outer_holdout_predictions_created": 0,
        "valid": True,
    }
    for name, value in expected_claims.items():
        if packet.get(name) != value:
            raise ValueError(f"legacy C6 packet {name} drift")
    artifacts = packet.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("legacy C6 packet artifacts are missing")
    required_artifacts = {
        "selected_native_units",
        "selected_frames",
        "actor_feature_tensor",
        "actor_feature_index",
    }
    if set(artifacts) != required_artifacts:
        raise ValueError("legacy C6 packet artifact set drift")
    paths = {
        name: _verify_artifact(repo_root, spec, name)
        for name, spec in artifacts.items()
    }
    units = pd.read_csv(paths["selected_native_units"], low_memory=False)
    index = pd.read_csv(paths["actor_feature_index"], low_memory=False)
    _require_unique(units, "temporal_unit_key", "prepared native units")
    _require_unique(index, "feature_row", "prepared actor feature index")
    if units["position"].tolist() != list(range(len(units))):
        raise ValueError("legacy C6 prepared unit positions drift")
    if index["feature_row"].tolist() != list(range(len(index))):
        raise ValueError("legacy C6 prepared feature rows drift")
    _require_unreviewed_claim(units, "prepared native units")
    _require_unreviewed_claim(index, "prepared actor feature index")
    expected_rows = len(units) * SEQUENCE_LENGTH
    if len(index) != expected_rows:
        raise ValueError("legacy C6 prepared feature index row count drift")
    tensor = np.load(paths["actor_feature_tensor"], mmap_mode="r")
    try:
        if tensor.shape != (expected_rows, FEATURE_DIM):
            raise ValueError("legacy C6 prepared feature tensor shape drift")
        if tensor.dtype != np.float32 or not np.isfinite(tensor).all():
            raise ValueError("legacy C6 prepared feature tensor values drift")
    finally:
        _close_memmap(tensor)
    index = index.sort_values(
        ["position", "slot_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    grouped = index.groupby("position", sort=True)
    if not grouped.size().eq(SEQUENCE_LENGTH).all():
        raise ValueError("legacy C6 prepared slot count drift")
    slot_values = grouped["slot_index"].apply(list)
    expected_slots = list(range(SEQUENCE_LENGTH))
    if not slot_values.map(lambda value: value == expected_slots).all():
        raise ValueError("legacy C6 prepared slot order drift")
    feature_rows = index["feature_row"].to_numpy(dtype=np.int64).reshape(
        len(units),
        SEQUENCE_LENGTH,
    )
    timestamps = pd.to_numeric(index["timestamp_sec"], errors="coerce").to_numpy(
        dtype=np.float64
    ).reshape(len(units), SEQUENCE_LENGTH)
    if not np.isfinite(timestamps).all():
        raise ValueError("legacy C6 prepared timestamps are nonfinite")
    time_delta = np.zeros_like(timestamps, dtype=np.float32)
    time_delta[:, 1:] = np.diff(timestamps, axis=1).astype(np.float32)
    if (time_delta[:, 1:] <= 0.0).any():
        raise ValueError("legacy C6 prepared timestamps are not increasing")
    label_to_index = {
        label: index for index, label in enumerate(VALID_BEHAVIORS)
    }
    labels = units["behavior_label"].astype(str)
    if not labels.isin(label_to_index).all():
        raise ValueError("legacy C6 prepared labels drift")
    targets = labels.map(label_to_index).to_numpy(dtype=np.int64)
    windows = units.copy(deep=True)
    windows["lineage_scope"] = LINEAGE_SCOPE
    windows["human_review_complete"] = False
    observed = np.ones(feature_rows.shape, dtype=bool)
    sample_weights = np.ones(len(windows), dtype=np.float64)
    audit = {
        **packet,
        "packet_path": str(packet_path),
        "packet_sha256": file_sha256(packet_path),
        "source_media_reads": 0,
        "outer_holdout_rows_loaded": 0,
        "model_visible_native_units": int(len(windows)),
        "errors": [],
        "valid": True,
    }
    view = LegacyL5CachedFeatureView(
        feature_tensor_path=paths["actor_feature_tensor"],
        feature_tensor_sha256=file_sha256(paths["actor_feature_tensor"]),
        control_id="V1",
        temporal_view_name="legacy_c6_rebuild_t16_prepared_v1",
        sequence_length=SEQUENCE_LENGTH,
        windows=windows,
        fold_manifest=windows.copy(deep=True),
        feature_rows=feature_rows,
        observed_mask=observed,
        time_delta=time_delta,
        targets=targets,
        sample_weights=sample_weights,
        audit=audit,
    )
    train_positions = np.flatnonzero(
        windows["l5_role"].astype(str).to_numpy() == "train"
    ).astype(np.int64)
    validation_positions = np.flatnonzero(
        windows["l5_role"].astype(str).to_numpy() == "validation"
    ).astype(np.int64)
    expected_train = int(packet.get("train_native_units", len(train_positions)))
    if len(train_positions) != expected_train:
        raise ValueError("legacy C6 prepared train count drift")
    if len(validation_positions) != int(packet["validation_native_units"]):
        raise ValueError("legacy C6 prepared validation count drift")
    selection_audit = {
        "selection_content_sha256": _dataframe_sha256(windows),
        "train_native_units": int(len(train_positions)),
        "train_selection_policy": packet.get(
            "train_selection_policy", "fixed_per_class"
        ),
        "train_units_per_class": packet.get("train_units_per_class"),
        "validation_native_units": int(len(validation_positions)),
        "outer_holdout_rows": 0,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "errors": [],
        "valid": True,
    }
    selection = LegacyL5CachedShortSelection(
        manifest=windows.copy(deep=True),
        train_positions=train_positions,
        validation_positions=validation_positions,
        audit=selection_audit,
    )
    return LegacyC6PreparedSource(
        base_view=view,
        selection=selection,
        parent_audit=audit,
        source_config_sha256=file_sha256(packet_path),
    )


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns={missing}")


def _require_unique(frame: pd.DataFrame, column: str, name: str) -> None:
    if frame[column].astype(str).duplicated().any():
        raise ValueError(f"{name} contains duplicate {column}")


def _require_unreviewed_claim(frame: pd.DataFrame, name: str) -> None:
    scopes = set(frame["lineage_scope"].fillna("").astype(str))
    if scopes != {LINEAGE_SCOPE}:
        raise ValueError(f"{name} lineage scopes={sorted(scopes)}")
    reviewed = set(_strict_bool(frame["human_review_complete"]))
    if reviewed != {False}:
        raise ValueError(f"{name} review claim drift")


def _strict_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise ValueError("boolean column contains missing values")
        return series.astype(bool)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    unknown = sorted(set(normalized).difference(mapping))
    if unknown:
        raise ValueError(f"invalid boolean values={unknown}")
    return normalized.map(mapping).astype(bool)


def _all_bool(series: pd.Series) -> bool:
    return bool(_strict_bool(series).all())


def _nonblank_count(series: pd.Series) -> int:
    return int(series.fillna("").astype(str).str.strip().ne("").sum())


def _selection_score(salt: str, value: str) -> str:
    return hashlib.sha256(f"{salt}|{value}".encode()).hexdigest()


def _verify_artifact(repo_root: Path, spec: object, name: str) -> Path:
    if not isinstance(spec, dict) or set(spec) != {"path", "sha256", "size_bytes"}:
        raise ValueError(f"legacy C6 packet artifact spec drift={name}")
    path = Path(str(spec["path"]))
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(spec["sha256"]):
        raise ValueError(f"legacy C6 packet artifact hash drift={name}")
    if path.stat().st_size != int(spec["size_bytes"]):
        raise ValueError(f"legacy C6 packet artifact size drift={name}")
    return path


def _dataframe_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _close_memmap(array: np.ndarray) -> None:
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


__all__ = [
    "LINEAGE_SCOPE",
    "PACKET_SCHEMA",
    "LegacyC6PreparedSource",
    "LegacyC6PreparedTables",
    "load_legacy_c6_prepared_source",
    "prepare_legacy_c6_tables",
]
