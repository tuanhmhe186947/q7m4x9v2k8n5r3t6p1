"""Bounded real-data correctness ladder for legacy-only development L4."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from pig_behavior.classification_v2.datasets.image_sequence_dataset import (
    ClassificationV2ImageSequenceDataset,
    ImageSequenceDatasetConfig,
)
from pig_behavior.classification_v2.models.model_factory import (
    build_multimodal_model,
)
from pig_behavior.classification_v2.models.multimodal_fusion import (
    ImageSequenceEncoder,
)
from pig_behavior.classification_v2.models.multitask_fusion import (
    MULTITASK_ARCHITECTURE_VERSION,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.config import ModelConfig

L4_CONFIG_SCHEMA_VERSION = "classification_v2.legacy_development_l4.config.v1"
L4_SHORT_SCHEMA_VERSION = "classification_v2.legacy_development_l4.short.v1"
L4_SCHEMA_VERSION = "classification_v2.legacy_development_l4.v1"
LINEAGE_SCOPE = "legacy-only-unreviewed-development"


@dataclass(frozen=True, slots=True)
class LegacyL4ModelSettings:
    model_mode: str
    backbone_name: str
    pretrained_weight_enum: str
    image_size: int
    sequence_length: int
    hidden_dim: int
    temporal_encoder_name: str
    dropout: float


@dataclass(frozen=True, slots=True)
class LegacyL4ShortSettings:
    seed: int
    device: str
    gradient_learning_rate: float
    tiny_events_per_class: int
    tiny_steps: int
    tiny_learning_rate: float
    tiny_minimum_accuracy: float
    tiny_maximum_loss_ratio: float
    frame_batch_events: int
    maximum_peak_vram_fraction: float


@dataclass(frozen=True, slots=True)
class LegacyL4FoldSettings:
    epochs: int
    frame_batch_events: int
    train_batch_size: int
    learning_rate: float
    weight_decay: float
    maximum_peak_vram_fraction: float
    visual_policy: str


@dataclass(frozen=True, slots=True)
class LegacyL4Config:
    path: Path
    payload: dict[str, Any]
    development_root: Path
    primary_run_id: str
    l3_audit_relative_path: Path
    fold_id: str
    temporal_view_name: str
    temporal_selection_column: str
    expected_selected_native_units: int
    expected_development_valid_native_units: int
    model: LegacyL4ModelSettings
    short: LegacyL4ShortSettings
    fold_epoch: LegacyL4FoldSettings

    @property
    def primary_root(self) -> Path:
        return self.development_root / self.primary_run_id

    @property
    def l3_audit_json(self) -> Path:
        return self.primary_root / self.l3_audit_relative_path

    @property
    def sha256(self) -> str:
        return file_sha256(self.path)


@dataclass(slots=True)
class LegacyL4DataBundle:
    config: LegacyL4Config
    actor_dataset: ClassificationV2ImageSequenceDataset
    selected_rows: pd.DataFrame
    train_rows: pd.DataFrame
    test_rows: pd.DataFrame
    train_development_rows: pd.DataFrame
    test_development_rows: pd.DataFrame
    dataset_index_by_window: dict[str, int]
    time_delta_by_window: dict[str, np.ndarray]
    support_audit: dict[str, Any]
    lineage_audit: dict[str, Any]

    def close(self) -> None:
        self.actor_dataset.close()

    def __enter__(self) -> LegacyL4DataBundle:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(slots=True)
class LegacyL4Batch:
    images: torch.Tensor
    masks: torch.Tensor
    time_delta: torch.Tensor
    targets: torch.Tensor
    metadata: dict[str, list[str]]


def load_legacy_l4_config(path: Path) -> LegacyL4Config:
    """Load one exact config and reject any claim or schema drift."""

    payload = _read_json(path)
    required = {
        "schema_version",
        "lineage_scope",
        "human_review_complete",
        "q2_claim_allowed",
        "canonical_full_oof_authorized",
        "accuracy_f1_comparison_authorized",
        "development_root",
        "primary_run_id",
        "l3_audit_relative_path",
        "fold_id",
        "temporal_view_name",
        "temporal_selection_column",
        "expected_selected_native_units",
        "expected_development_valid_native_units",
        "model",
        "short_gate",
        "fold_epoch",
    }
    _require_exact_keys(payload, required, name="legacy L4 config")
    if payload["schema_version"] != L4_CONFIG_SCHEMA_VERSION:
        raise ValueError("legacy L4 config schema mismatch")
    if payload["lineage_scope"] != LINEAGE_SCOPE:
        raise ValueError("legacy L4 lineage scope mismatch")
    false_claims = (
        "human_review_complete",
        "q2_claim_allowed",
        "canonical_full_oof_authorized",
        "accuracy_f1_comparison_authorized",
    )
    if any(payload[name] is not False for name in false_claims):
        raise ValueError("legacy L4 config exceeds its claim boundary")
    model = _model_settings(payload["model"])
    short = _short_settings(payload["short_gate"])
    fold = _fold_settings(payload["fold_epoch"])
    config = LegacyL4Config(
        path=path,
        payload=payload,
        development_root=Path(str(payload["development_root"])),
        primary_run_id=str(payload["primary_run_id"]),
        l3_audit_relative_path=Path(str(payload["l3_audit_relative_path"])),
        fold_id=str(payload["fold_id"]),
        temporal_view_name=str(payload["temporal_view_name"]),
        temporal_selection_column=str(payload["temporal_selection_column"]),
        expected_selected_native_units=int(
            payload["expected_selected_native_units"]
        ),
        expected_development_valid_native_units=int(
            payload["expected_development_valid_native_units"]
        ),
        model=model,
        short=short,
        fold_epoch=fold,
    )
    _validate_config_values(config)
    return config


def _model_settings(payload: object) -> LegacyL4ModelSettings:
    row = _object(payload, name="model")
    fields = {
        "model_mode",
        "backbone_name",
        "pretrained_weight_enum",
        "image_size",
        "sequence_length",
        "hidden_dim",
        "temporal_encoder_name",
        "dropout",
    }
    _require_exact_keys(row, fields, name="model")
    return LegacyL4ModelSettings(
        model_mode=str(row["model_mode"]),
        backbone_name=str(row["backbone_name"]),
        pretrained_weight_enum=str(row["pretrained_weight_enum"]),
        image_size=int(row["image_size"]),
        sequence_length=int(row["sequence_length"]),
        hidden_dim=int(row["hidden_dim"]),
        temporal_encoder_name=str(row["temporal_encoder_name"]),
        dropout=float(row["dropout"]),
    )


def _short_settings(payload: object) -> LegacyL4ShortSettings:
    row = _object(payload, name="short_gate")
    fields = {
        "seed",
        "device",
        "gradient_learning_rate",
        "tiny_events_per_class",
        "tiny_steps",
        "tiny_learning_rate",
        "tiny_minimum_accuracy",
        "tiny_maximum_loss_ratio",
        "frame_batch_events",
        "maximum_peak_vram_fraction",
    }
    _require_exact_keys(row, fields, name="short_gate")
    return LegacyL4ShortSettings(
        seed=int(row["seed"]),
        device=str(row["device"]),
        gradient_learning_rate=float(row["gradient_learning_rate"]),
        tiny_events_per_class=int(row["tiny_events_per_class"]),
        tiny_steps=int(row["tiny_steps"]),
        tiny_learning_rate=float(row["tiny_learning_rate"]),
        tiny_minimum_accuracy=float(row["tiny_minimum_accuracy"]),
        tiny_maximum_loss_ratio=float(row["tiny_maximum_loss_ratio"]),
        frame_batch_events=int(row["frame_batch_events"]),
        maximum_peak_vram_fraction=float(
            row["maximum_peak_vram_fraction"]
        ),
    )


def _fold_settings(payload: object) -> LegacyL4FoldSettings:
    row = _object(payload, name="fold_epoch")
    fields = {
        "epochs",
        "frame_batch_events",
        "train_batch_size",
        "learning_rate",
        "weight_decay",
        "maximum_peak_vram_fraction",
        "visual_policy",
    }
    _require_exact_keys(row, fields, name="fold_epoch")
    return LegacyL4FoldSettings(
        epochs=int(row["epochs"]),
        frame_batch_events=int(row["frame_batch_events"]),
        train_batch_size=int(row["train_batch_size"]),
        learning_rate=float(row["learning_rate"]),
        weight_decay=float(row["weight_decay"]),
        maximum_peak_vram_fraction=float(
            row["maximum_peak_vram_fraction"]
        ),
        visual_policy=str(row["visual_policy"]),
    )


def _validate_config_values(config: LegacyL4Config) -> None:
    values = (
        config.model.image_size,
        config.model.sequence_length,
        config.model.hidden_dim,
        config.short.tiny_events_per_class,
        config.short.tiny_steps,
        config.short.frame_batch_events,
        config.fold_epoch.epochs,
        config.fold_epoch.frame_batch_events,
        config.fold_epoch.train_batch_size,
    )
    if min(values) <= 0:
        raise ValueError("legacy L4 integer settings must be positive")
    rates = (
        config.short.gradient_learning_rate,
        config.short.tiny_learning_rate,
        config.fold_epoch.learning_rate,
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in rates):
        raise ValueError("legacy L4 learning rates must be finite and positive")
    fractions = (
        config.short.maximum_peak_vram_fraction,
        config.fold_epoch.maximum_peak_vram_fraction,
    )
    if any(not 0.0 < value < 1.0 for value in fractions):
        raise ValueError("legacy L4 VRAM fractions must be in (0,1)")
    if not 0.0 <= config.short.tiny_minimum_accuracy <= 1.0:
        raise ValueError("tiny minimum accuracy must be in [0,1]")
    if not 0.0 < config.short.tiny_maximum_loss_ratio < 1.0:
        raise ValueError("tiny maximum loss ratio must be in (0,1)")
    if config.fold_epoch.epochs != 1:
        raise ValueError("legacy L4 fold gate must remain exactly one epoch")
    if config.fold_epoch.weight_decay < 0.0:
        raise ValueError("legacy L4 weight decay must be nonnegative")
    expected_policy = "frozen_random_resnet_correctness_only"
    if config.fold_epoch.visual_policy != expected_policy:
        raise ValueError("legacy L4 visual policy drift")


def load_legacy_l4_data(config: LegacyL4Config) -> LegacyL4DataBundle:
    """Load frozen manifests and expose only development-valid optimizer rows."""

    l3_audit = _read_json(config.l3_audit_json)
    _validate_l3_parent(config, l3_audit)
    paths = _legacy_paths(
        config.primary_root,
        config.temporal_view_name,
        config.model.image_size,
    )
    fold_rows = pd.read_csv(paths["window_folds"], low_memory=False)
    required_fold = {
        "window_id",
        "temporal_unit_key",
        "oof_fold_id",
        "behavior_label",
        "source_type",
        "video_key",
        "recording_group_id",
        "lineage_scope",
        "human_review_complete",
        config.temporal_selection_column,
    }
    _require_columns(fold_rows, required_fold, name="window fold manifest")
    selected_mask = _strict_bool(
        fold_rows[config.temporal_selection_column],
        name=config.temporal_selection_column,
    )
    selected = fold_rows.loc[selected_mask].copy()
    native = pd.read_csv(
        paths["native_units"],
        usecols=[
            "temporal_unit_key",
            "native_unit_valid_for_development",
            "lineage_scope",
            "human_review_complete",
        ],
        low_memory=False,
    )
    if native["temporal_unit_key"].duplicated().any():
        raise ValueError("native temporal unit keys are not unique")
    native_valid = _strict_bool(
        native["native_unit_valid_for_development"],
        name="native_unit_valid_for_development",
    )
    native = native.assign(development_valid=native_valid.to_numpy())
    selected = selected.merge(
        native[["temporal_unit_key", "development_valid"]],
        on="temporal_unit_key",
        how="left",
        validate="one_to_one",
    )
    if selected["development_valid"].isna().any():
        raise ValueError("selected fold rows are missing native validity")
    selected["development_valid"] = selected["development_valid"].astype(bool)
    _validate_selected_rows(config, selected)
    train = selected.loc[selected["oof_fold_id"].astype(str).ne(config.fold_id)].copy()
    test = selected.loc[selected["oof_fold_id"].astype(str).eq(config.fold_id)].copy()
    train_valid = train.loc[train["development_valid"]].copy()
    test_valid = test.loc[test["development_valid"]].copy()
    support = _support_audit(config, selected, train, test, paths)
    lineage = _lineage_audit(config, selected, train, test, train_valid, test_valid)
    actor = ClassificationV2ImageSequenceDataset(
        ImageSequenceDatasetConfig(
            frame_context_csv=paths["image_frames"],
            window_context_csv=paths["image_windows"],
            packed_image_cache_npy=paths["packed_tensor"],
            packed_image_cache_index_csv=paths["packed_index"],
            image_size=config.model.image_size,
            require_complete=True,
            require_cached_images=True,
            image_cache_size=0,
        )
    )
    index_by_window = {
        str(window_id): int(index)
        for index, window_id in enumerate(actor.windows["window_id"])
    }
    missing_windows = sorted(
        set(selected["window_id"].astype(str)).difference(index_by_window)
    )
    if missing_windows:
        actor.close()
        raise ValueError(f"selected image windows are missing: {missing_windows[:5]}")
    time_delta = _load_time_delta(paths["temporal_view"], config, selected)
    return LegacyL4DataBundle(
        config=config,
        actor_dataset=actor,
        selected_rows=selected.reset_index(drop=True),
        train_rows=train.reset_index(drop=True),
        test_rows=test.reset_index(drop=True),
        train_development_rows=train_valid.reset_index(drop=True),
        test_development_rows=test_valid.reset_index(drop=True),
        dataset_index_by_window=index_by_window,
        time_delta_by_window=time_delta,
        support_audit=support,
        lineage_audit=lineage,
    )


def _legacy_paths(
    root: Path,
    temporal_view_name: str,
    image_size: int,
) -> dict[str, Path]:
    cache_root = root / f"10_actor_cache_{image_size}"
    return {
        "image_frames": root / "09_image_context" / "image_frame_context_manifest.csv",
        "image_windows": root / "09_image_context" / "image_window_context_manifest.csv",
        "packed_tensor": cache_root / f"packed_rgb_{image_size}_letterbox.npy",
        "packed_index": cache_root / "packed_image_cache_index.csv",
        "window_folds": root / "11_folds" / "window_oof_fold_manifest.csv",
        "class_support": root / "11_folds" / "class_by_fold_support.csv",
        "source_support": root / "11_folds" / "source_by_fold_support.csv",
        "native_units": (
            root
            / "06_temporal_tier_contract"
            / "native_temporal_unit_manifest.csv"
        ),
        "temporal_view": (
            root
            / "06_temporal_tier_contract"
            / f"{temporal_view_name}_manifest.csv"
        ),
    }


def _validate_l3_parent(
    config: LegacyL4Config,
    audit: dict[str, Any],
) -> None:
    expected = {
        "status": "PASS_LEGACY_DEVELOPMENT_L3",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "accuracy_f1_comparison_authorized": False,
        "l4_model_correctness_authorized": True,
        "bounded_model_correctness_training_authorized": True,
        "primary_root": str(config.primary_root).replace("\\", "/"),
        "valid": True,
    }
    mismatches = {
        key: {"expected": value, "observed": audit.get(key)}
        for key, value in expected.items()
        if str(audit.get(key)).replace("\\", "/") != str(value)
    }
    if audit.get("errors"):
        mismatches["errors"] = {"expected": [], "observed": audit["errors"]}
    if mismatches:
        raise ValueError(f"legacy L3 parent gate mismatch: {mismatches}")


def _validate_selected_rows(
    config: LegacyL4Config,
    selected: pd.DataFrame,
) -> None:
    errors: list[str] = []
    if len(selected) != config.expected_selected_native_units:
        errors.append(
            "selected_native_unit_count="
            f"{len(selected)} expected={config.expected_selected_native_units}"
        )
    valid_count = int(selected["development_valid"].sum())
    if valid_count != config.expected_development_valid_native_units:
        errors.append(
            "development_valid_native_unit_count="
            f"{valid_count} "
            f"expected={config.expected_development_valid_native_units}"
        )
    for column in ("window_id", "temporal_unit_key"):
        duplicate = int(selected[column].astype(str).duplicated().sum())
        if duplicate:
            errors.append(f"duplicate_{column}={duplicate}")
    if set(selected["behavior_label"].astype(str)) != set(VALID_BEHAVIORS):
        errors.append("selected_behavior_label_set_mismatch")
    if set(selected["lineage_scope"].astype(str)) != {LINEAGE_SCOPE}:
        errors.append("selected_lineage_scope_mismatch")
    reviewed = _strict_bool(
        selected["human_review_complete"],
        name="selected human_review_complete",
    )
    if reviewed.any():
        errors.append("selected_rows_claim_human_review")
    if errors:
        raise ValueError(f"invalid legacy L4 selected rows: {errors}")


def _support_audit(
    config: LegacyL4Config,
    selected: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    paths: dict[str, Path],
) -> dict[str, Any]:
    classes = pd.read_csv(paths["class_support"], low_memory=False)
    sources = pd.read_csv(paths["source_support"], low_memory=False)
    class_fold = classes.loc[
        classes["oof_fold_id"].astype(str).eq(config.fold_id)
    ].copy()
    source_fold = sources.loc[
        sources["oof_fold_id"].astype(str).eq(config.fold_id)
    ].copy()
    class_rows: dict[str, Any] = {}
    errors: list[str] = []
    for label in VALID_BEHAVIORS:
        observed = class_fold.loc[
            class_fold["behavior_label"].astype(str).eq(label)
        ]
        if len(observed) != 1:
            errors.append(f"class_support_row_count={label}:{len(observed)}")
            continue
        row = observed.iloc[0]
        train_count = int(train["behavior_label"].astype(str).eq(label).sum())
        test_count = int(test["behavior_label"].astype(str).eq(label).sum())
        declared_train = int(row["train_native_units"])
        declared_test = int(row["test_native_units"])
        matches = train_count == declared_train and test_count == declared_test
        if not matches:
            errors.append(f"class_support_count_mismatch={label}")
        class_rows[label] = {
            "train_native_units": train_count,
            "test_native_units": test_count,
            "declared_train_native_units": declared_train,
            "declared_test_native_units": declared_test,
            "train_supported": _bool_scalar(
                row["train_supported"],
                name=f"train_supported:{label}",
            ),
            "test_supported": _bool_scalar(
                row["test_supported"],
                name=f"test_supported:{label}",
            ),
            "counts_match": matches,
        }
    source_rows: dict[str, Any] = {}
    for row in source_fold.itertuples(index=False):
        source = str(row.source_type)
        train_count = int(train["source_type"].astype(str).eq(source).sum())
        test_count = int(test["source_type"].astype(str).eq(source).sum())
        declared_train = int(row.train_native_units)
        declared_test = int(row.test_native_units)
        matches = train_count == declared_train and test_count == declared_test
        if not matches:
            errors.append(f"source_support_count_mismatch={source}")
        source_rows[source] = {
            "train_native_units": train_count,
            "test_native_units": test_count,
            "declared_train_native_units": declared_train,
            "declared_test_native_units": declared_test,
            "train_supported": _bool_scalar(
                row.train_supported,
                name=f"source_train_supported:{source}",
            ),
            "test_supported": _bool_scalar(
                row.test_supported,
                name=f"source_test_supported:{source}",
            ),
            "counts_match": matches,
        }
    if sum(row["train_native_units"] for row in class_rows.values()) != len(train):
        errors.append("class_support_train_total_mismatch")
    if sum(row["test_native_units"] for row in class_rows.values()) != len(test):
        errors.append("class_support_test_total_mismatch")
    return {
        "fold_id": config.fold_id,
        "selected_native_units": int(len(selected)),
        "class_support": class_rows,
        "source_support": source_rows,
        "errors": errors,
        "valid": not errors,
    }


def _lineage_audit(
    config: LegacyL4Config,
    selected: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_valid: pd.DataFrame,
    test_valid: pd.DataFrame,
) -> dict[str, Any]:
    train_units = set(train["temporal_unit_key"].astype(str))
    test_units = set(test["temporal_unit_key"].astype(str))
    train_groups = set(train["recording_group_id"].astype(str))
    test_groups = set(test["recording_group_id"].astype(str))
    train_videos = set(train["video_key"].astype(str))
    test_videos = set(test["video_key"].astype(str))
    errors: list[str] = []
    if train_units.intersection(test_units):
        errors.append("native_unit_train_test_overlap")
    if train_groups.intersection(test_groups):
        errors.append("recording_group_train_test_overlap")
    if train_videos.intersection(test_videos):
        errors.append("video_train_test_overlap")
    train_labels = set(train_valid["behavior_label"].astype(str))
    if train_labels != set(VALID_BEHAVIORS):
        errors.append("development_train_class_set_mismatch")
    return {
        "fold_id": config.fold_id,
        "selected_native_units": int(len(selected)),
        "development_valid_native_units": int(selected["development_valid"].sum()),
        "retained_policy_invalid_native_units": int(
            (~selected["development_valid"]).sum()
        ),
        "declared_train_native_units": int(len(train)),
        "declared_test_native_units": int(len(test)),
        "optimizer_train_native_units": int(len(train_valid)),
        "development_test_native_units": int(len(test_valid)),
        "train_class_count": len(train_labels),
        "train_test_native_overlap": len(train_units.intersection(test_units)),
        "train_test_recording_group_overlap": len(
            train_groups.intersection(test_groups)
        ),
        "train_test_video_overlap": len(train_videos.intersection(test_videos)),
        "errors": errors,
        "valid": not errors,
    }


def _load_time_delta(
    path: Path,
    config: LegacyL4Config,
    selected: pd.DataFrame,
) -> dict[str, np.ndarray]:
    view = pd.read_csv(
        path,
        usecols=[
            "temporal_view_name",
            "parent_window_id",
            "slot_index",
            "declared_sequence_length",
            "time_delta",
            "length_mask",
            "observed_mask",
            "timing_valid_mask",
            "padding_mask",
            "lineage_scope",
            "human_review_complete",
        ],
        low_memory=False,
    )
    if set(view["temporal_view_name"].astype(str)) != {
        config.temporal_view_name
    }:
        raise ValueError("legacy L4 temporal view name mismatch")
    expected_rows = len(selected) * config.model.sequence_length
    if len(view) != expected_rows:
        raise ValueError(
            f"legacy L4 temporal slot rows={len(view)} expected={expected_rows}"
        )
    if set(view["lineage_scope"].astype(str)) != {LINEAGE_SCOPE}:
        raise ValueError("legacy L4 temporal view lineage mismatch")
    if _strict_bool(
        view["human_review_complete"],
        name="temporal human_review_complete",
    ).any():
        raise ValueError("legacy L4 temporal view claims human review")
    masks = {
        "length_mask": True,
        "observed_mask": True,
        "timing_valid_mask": True,
        "padding_mask": False,
    }
    for name, expected in masks.items():
        observed = _strict_bool(view[name], name=name)
        if not bool((observed == expected).all()):
            raise ValueError(f"legacy L4 temporal mask mismatch={name}")
    lengths = pd.to_numeric(view["declared_sequence_length"], errors="coerce")
    if not lengths.eq(config.model.sequence_length).all():
        raise ValueError("legacy L4 declared sequence length mismatch")
    view = view.sort_values(
        ["parent_window_id", "slot_index"],
        kind="mergesort",
    )
    expected_ids = set(selected["window_id"].astype(str))
    if set(view["parent_window_id"].astype(str)) != expected_ids:
        raise ValueError("legacy L4 temporal window ID set mismatch")
    out: dict[str, np.ndarray] = {}
    for window_id, group in view.groupby("parent_window_id", sort=False):
        slots = pd.to_numeric(group["slot_index"], errors="coerce").to_numpy()
        expected_slots = np.arange(config.model.sequence_length)
        if not np.array_equal(slots, expected_slots):
            raise ValueError(f"legacy L4 slot order mismatch={window_id}")
        delta = pd.to_numeric(group["time_delta"], errors="coerce").to_numpy(
            dtype=np.float32
        )
        if not np.isfinite(delta).all() or (delta < 0.0).any():
            raise ValueError(f"invalid legacy L4 time delta={window_id}")
        out[str(window_id)] = delta
    return out


def load_legacy_l4_batch(
    bundle: LegacyL4DataBundle,
    rows: pd.DataFrame,
    *,
    device: torch.device,
) -> LegacyL4Batch:
    """Load one cache-only batch while keeping metadata outside model inputs."""

    items: list[dict[str, Any]] = []
    for window_id in rows["window_id"].astype(str):
        index = bundle.dataset_index_by_window[window_id]
        item = bundle.actor_dataset[index]
        if item["errors"]:
            raise ValueError(
                f"legacy L4 actor image load errors={window_id}:{item['errors']}"
            )
        items.append(item)
    images = torch.stack([item["image"] for item in items]).to(device)
    masks = torch.stack([item["observed_mask"] for item in items]).to(device)
    lengths = torch.stack([item["length_mask"] for item in items]).to(device)
    if images.shape[1:] != (
        bundle.config.model.sequence_length,
        3,
        bundle.config.model.image_size,
        bundle.config.model.image_size,
    ):
        raise ValueError(f"legacy L4 image batch shape mismatch={tuple(images.shape)}")
    if not torch.equal(masks, lengths) or not torch.all(masks == 1):
        raise ValueError("legacy L4 selected batch is not fully observed")
    deltas = torch.from_numpy(
        np.stack(
            [
                bundle.time_delta_by_window[window_id]
                for window_id in rows["window_id"].astype(str)
            ]
        )
    ).to(device)
    label_to_index = {
        label: index for index, label in enumerate(VALID_BEHAVIORS)
    }
    targets = torch.tensor(
        [label_to_index[str(label)] for label in rows["behavior_label"]],
        dtype=torch.long,
        device=device,
    )
    return LegacyL4Batch(
        images=images,
        masks=masks,
        time_delta=deltas,
        targets=targets,
        metadata={
            "window_id": rows["window_id"].astype(str).tolist(),
            "temporal_unit_key": rows["temporal_unit_key"].astype(str).tolist(),
            "source_type": rows["source_type"].astype(str).tolist(),
            "video_key": rows["video_key"].astype(str).tolist(),
            "oof_fold_id": rows["oof_fold_id"].astype(str).tolist(),
        },
    )


def build_legacy_l4_model(config: LegacyL4Config) -> nn.Module:
    model = ModelConfig(
        architecture_version=MULTITASK_ARCHITECTURE_VERSION,
        model_mode=config.model.model_mode,
        backbone_name=config.model.backbone_name,
        pretrained_weight_enum=config.model.pretrained_weight_enum,
        temporal_view=config.temporal_view_name,
        temporal_input_frames=config.model.sequence_length,
        temporal_encoder_name=config.model.temporal_encoder_name,
        image_size=config.model.image_size,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
        transformer_layers=1,
        transformer_heads=2,
        visual_freeze_policy="all_trainable",
        visual_frozen_warmup_epochs=0,
        visual_layer4_only_epochs=0,
        visual_backbone_lr_multiplier=1.0,
        spatial_feature_groups=(),
        standardize_spatial_groups=(),
        enable_image=True,
        enable_spatial=False,
        enable_interaction_context=False,
        enable_visual_context=False,
        enable_multitask=False,
    )
    return build_multimodal_model(
        model,
        spatial_input_dims={},
        interaction_context_dim=None,
        num_classes=len(VALID_BEHAVIORS),
    )


def legacy_l4_model_inputs(
    batch: LegacyL4Batch,
    *,
    masks: torch.Tensor | None = None,
    images: torch.Tensor | None = None,
) -> dict[str, Any]:
    resolved_mask = batch.masks if masks is None else masks
    resolved_image = batch.images if images is None else images
    return {
        "image": resolved_image,
        "spatial_features": {},
        "length_mask": resolved_mask,
        "observed_mask": resolved_mask,
        "image_length_mask": resolved_mask,
        "image_observed_mask": resolved_mask,
        "image_available_mask": resolved_mask,
        "image_quality_mask": resolved_mask,
        "image_time_delta": batch.time_delta,
    }


def _select_tiny_rows(
    rows: pd.DataFrame,
    *,
    events_per_class: int,
) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for label in VALID_BEHAVIORS:
        candidates = rows.loc[
            rows["behavior_label"].astype(str).eq(label)
        ].sort_values("temporal_unit_key", kind="mergesort")
        if len(candidates) < events_per_class:
            raise ValueError(
                f"tiny overfit lacks class support={label}:{len(candidates)}"
            )
        selected.append(candidates.head(events_per_class))
    out = pd.concat(selected, ignore_index=True)
    if out["temporal_unit_key"].duplicated().any():
        raise ValueError("tiny overfit native units are not unique")
    return out


def run_legacy_l4_short(
    config: LegacyL4Config,
    *,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Run the ordered real-cache L4 correctness ladder before a fold epoch."""

    device = _resolve_device(config.short.device)
    _seed_all(config.short.seed)
    _reset_peak_memory(device)
    started = time.perf_counter()
    l3_sha256 = file_sha256(config.l3_audit_json)
    errors: list[str] = []
    with load_legacy_l4_data(config) as bundle:
        one_row = bundle.train_development_rows.sort_values(
            "temporal_unit_key",
            kind="mergesort",
        ).head(1)
        batch = load_legacy_l4_batch(bundle, one_row, device=device)
        input_contract = _input_contract_audit(config, batch)
        mask_audit = _mask_and_order_audit(config, batch, device=device)
        gradient_audit = _one_batch_gradient_audit(
            config,
            batch,
            device=device,
        )
        deterministic_audit = _deterministic_repeat_audit(
            config,
            batch,
            device=device,
        )
        resume_audit = _checkpoint_resume_audit(
            config,
            batch,
            checkpoint_path=checkpoint_path,
            l3_sha256=l3_sha256,
            device=device,
        )
        tiny_rows = _select_tiny_rows(
            bundle.train_development_rows,
            events_per_class=config.short.tiny_events_per_class,
        )
        tiny_audit = _tiny_overfit_audit(
            config,
            bundle,
            tiny_rows,
            device=device,
        )
        cache_audit = _cache_only_audit(bundle)
        sections = {
            "lineage_audit": bundle.lineage_audit,
            "support_audit": bundle.support_audit,
            "input_contract_audit": input_contract,
            "mask_and_order_audit": mask_audit,
            "one_batch_gradient_audit": gradient_audit,
            "deterministic_repeat_audit": deterministic_audit,
            "checkpoint_resume_audit": resume_audit,
            "tiny_overfit_audit": tiny_audit,
            "cache_only_audit": cache_audit,
        }
        for name, section in sections.items():
            if not section.get("valid", False):
                errors.append(f"{name}_failed")
            errors.extend(str(error) for error in section.get("errors", []))
    memory_audit = _memory_audit(
        device,
        config.short.maximum_peak_vram_fraction,
    )
    if not memory_audit["valid"]:
        errors.extend(memory_audit["errors"])
    runtime_sec = float(time.perf_counter() - started)
    valid = not errors
    return {
        "schema_version": L4_SHORT_SCHEMA_VERSION,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L4_SHORT"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L4_SHORT"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "accuracy_f1_comparison_authorized": False,
        "fold_epoch_authorized": valid,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "l3_audit_path": str(config.l3_audit_json),
        "l3_audit_sha256": l3_sha256,
        "implementation_source_sha256": file_sha256(Path(__file__)),
        "git_state": git_state(),
        "fold_id": config.fold_id,
        "temporal_view_name": config.temporal_view_name,
        "device": str(device),
        "runtime_sec": runtime_sec,
        "memory_audit": memory_audit,
        **sections,
        "errors": errors,
        "valid": valid,
    }


def _input_contract_audit(
    config: LegacyL4Config,
    batch: LegacyL4Batch,
) -> dict[str, Any]:
    inputs = legacy_l4_model_inputs(batch)
    expected_keys = {
        "image",
        "spatial_features",
        "length_mask",
        "observed_mask",
        "image_length_mask",
        "image_observed_mask",
        "image_available_mask",
        "image_quality_mask",
        "image_time_delta",
    }
    metadata_keys = set(batch.metadata)
    forbidden = {
        "target",
        "behavior_label",
        "window_id",
        "temporal_unit_key",
        "source_type",
        "video_key",
        "oof_fold_id",
        "path",
        "review_status",
    }
    errors: list[str] = []
    if set(inputs) != expected_keys:
        errors.append("legacy_l4_model_input_key_mismatch")
    if set(inputs).intersection(forbidden):
        errors.append("legacy_l4_forbidden_metadata_in_model_inputs")
    if not metadata_keys.issubset(forbidden):
        errors.append("legacy_l4_unknown_audit_metadata")
    if tuple(batch.images.shape[1:]) != (
        config.model.sequence_length,
        3,
        config.model.image_size,
        config.model.image_size,
    ):
        errors.append("legacy_l4_image_shape_mismatch")
    if batch.masks.shape != batch.images.shape[:2]:
        errors.append("legacy_l4_mask_shape_mismatch")
    if not torch.all(batch.masks == 1):
        errors.append("legacy_l4_short_batch_incomplete")
    return {
        "model_input_keys": sorted(inputs),
        "metadata_keys": sorted(metadata_keys),
        "metadata_separated_from_x": not set(inputs).intersection(forbidden),
        "image_shape": list(batch.images.shape),
        "mask_shape": list(batch.masks.shape),
        "mask_only_controls": [
            "length_mask",
            "observed_mask",
            "image_available_mask",
            "image_quality_mask",
        ],
        "errors": errors,
        "valid": not errors,
    }


def _mask_and_order_audit(
    config: LegacyL4Config,
    batch: LegacyL4Batch,
    *,
    device: torch.device,
) -> dict[str, Any]:
    _seed_all(config.short.seed)
    model = build_legacy_l4_model(config).to(device).eval()
    masked = batch.masks.clone()
    masked[:, -1] = 0.0
    first = batch.images.clone()
    second = batch.images.clone()
    first[:, -1] = 0.0
    second[:, -1] = 1.0
    with torch.no_grad():
        first_logits = model(
            **legacy_l4_model_inputs(batch, masks=masked, images=first)
        ).behavior
        second_logits = model(
            **legacy_l4_model_inputs(batch, masks=masked, images=second)
        ).behavior
        base_logits = model(**legacy_l4_model_inputs(batch)).behavior
        reversed_inputs = legacy_l4_model_inputs(
            batch,
            images=batch.images.flip(1),
        )
        reversed_inputs["image_time_delta"] = batch.time_delta.flip(1)
        reversed_logits = model(**reversed_inputs).behavior
    masked_delta = float((first_logits - second_logits).abs().max().cpu())
    reverse_delta = float((base_logits - reversed_logits).abs().max().cpu())
    bad_inputs = legacy_l4_model_inputs(batch)
    bad_length = batch.masks.clone()
    bad_length[:, -1] = 0.0
    bad_inputs["length_mask"] = bad_length
    bad_inputs["image_length_mask"] = bad_length
    invalid_mask_rejected = False
    try:
        model(**bad_inputs)
    except ValueError:
        invalid_mask_rejected = True
    errors: list[str] = []
    if masked_delta != 0.0:
        errors.append(f"masked_value_changed_logits={masked_delta}")
    if reverse_delta <= 1e-8:
        errors.append(f"temporal_order_insensitive={reverse_delta}")
    if not invalid_mask_rejected:
        errors.append("observed_outside_length_not_rejected")
    del model
    _empty_cuda_cache(device)
    return {
        "masked_value_max_logit_delta": masked_delta,
        "reversed_frame_max_logit_delta": reverse_delta,
        "invalid_mask_rejected": invalid_mask_rejected,
        "errors": errors,
        "valid": not errors,
    }


def _one_batch_gradient_audit(
    config: LegacyL4Config,
    batch: LegacyL4Batch,
    *,
    device: torch.device,
) -> dict[str, Any]:
    _seed_all(config.short.seed)
    model = build_legacy_l4_model(config).to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.short.gradient_learning_rate,
        weight_decay=0.0,
    )
    optimizer.zero_grad(set_to_none=True)
    logits = model(**legacy_l4_model_inputs(batch)).behavior
    loss = nn.functional.cross_entropy(logits, batch.targets)
    loss.backward()
    gradients = _gradient_groups(model)
    finite_logits = bool(torch.isfinite(logits).all())
    finite_loss = bool(torch.isfinite(loss))
    optimizer.step()
    errors: list[str] = []
    if not finite_logits:
        errors.append("one_batch_nonfinite_logits")
    if not finite_loss:
        errors.append("one_batch_nonfinite_loss")
    if not gradients["valid"]:
        errors.append("one_batch_gradient_group_failure")
    result = {
        "batch_size": int(batch.targets.numel()),
        "loss": float(loss.detach().cpu()),
        "finite_logits": finite_logits,
        "finite_loss": finite_loss,
        "gradient_groups": gradients,
        "optimizer_steps": 1,
        "errors": errors,
        "valid": not errors,
    }
    del optimizer, model, logits, loss
    _empty_cuda_cache(device)
    return result


def _gradient_groups(model: nn.Module) -> dict[str, Any]:
    prefixes = {
        "visual_backbone": "backbone.image_encoder.frame_encoder",
        "temporal_projection": "backbone.image_encoder.temporal_projection",
        "temporal_encoder": "backbone.image_encoder.temporal_encoder",
        "behavior_head": "backbone.classifier",
    }
    rows: dict[str, Any] = {}
    for name, prefix in prefixes.items():
        values = [
            parameter.grad
            for parameter_name, parameter in model.named_parameters()
            if parameter_name.startswith(prefix) and parameter.requires_grad
        ]
        finite = bool(values) and all(
            value is not None and torch.isfinite(value).all()
            for value in values
        )
        absolute_sum = float(
            sum(
                value.detach().abs().sum().cpu().item()
                for value in values
                if value is not None
            )
        )
        rows[name] = {
            "parameter_tensors": len(values),
            "finite": finite,
            "absolute_gradient_sum": absolute_sum,
            "nonzero": absolute_sum > 0.0,
        }
    return {
        "groups": rows,
        "valid": all(row["finite"] and row["nonzero"] for row in rows.values()),
    }


def _deterministic_repeat_audit(
    config: LegacyL4Config,
    batch: LegacyL4Batch,
    *,
    device: torch.device,
) -> dict[str, Any]:
    first = _single_step_evidence(config, batch, device=device)
    second = _single_step_evidence(config, batch, device=device)
    logit_delta = float((first.pop("logits") - second.pop("logits")).abs().max())
    semantic_match = first == second
    errors: list[str] = []
    if logit_delta != 0.0:
        errors.append(f"deterministic_repeat_logit_delta={logit_delta}")
    if not semantic_match:
        errors.append("deterministic_repeat_semantic_mismatch")
    return {
        "runs": 2,
        "max_logit_delta": logit_delta,
        "semantic_match": semantic_match,
        "evidence": first,
        "errors": errors,
        "valid": not errors,
    }


def _single_step_evidence(
    config: LegacyL4Config,
    batch: LegacyL4Batch,
    *,
    device: torch.device,
) -> dict[str, Any]:
    _seed_all(config.short.seed)
    model = build_legacy_l4_model(config).to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.short.gradient_learning_rate,
        weight_decay=0.0,
    )
    loss = _optimizer_step(model, optimizer, batch)
    model.eval()
    with torch.no_grad():
        logits = model(**legacy_l4_model_inputs(batch)).behavior.detach().cpu()
    evidence = {
        "loss": loss,
        "model_state_sha256": state_sha256(model.state_dict()),
        "optimizer_state_sha256": state_sha256(optimizer.state_dict()),
        "logits": logits,
    }
    del optimizer, model
    _empty_cuda_cache(device)
    return evidence


def _optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: LegacyL4Batch,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(**legacy_l4_model_inputs(batch)).behavior
    loss = nn.functional.cross_entropy(logits, batch.targets)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def _checkpoint_resume_audit(
    config: LegacyL4Config,
    batch: LegacyL4Batch,
    *,
    checkpoint_path: Path,
    l3_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    _seed_all(config.short.seed)
    model = build_legacy_l4_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.short.gradient_learning_rate,
        weight_decay=0.0,
    )
    first_loss = _optimizer_step(model, optimizer, batch)
    model.eval()
    with torch.no_grad():
        expected_logits = model(**legacy_l4_model_inputs(batch)).behavior.detach()
    payload = {
        "schema_version": "classification_v2.legacy_l4_resume_probe.v1",
        "config_sha256": config.sha256,
        "l3_audit_sha256": l3_sha256,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": _rng_state(),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _validate_resume_payload(loaded, config, l3_sha256=l3_sha256)
    resumed_model = build_legacy_l4_model(config).to(device)
    resumed_optimizer = torch.optim.AdamW(
        resumed_model.parameters(),
        lr=config.short.gradient_learning_rate,
        weight_decay=0.0,
    )
    resumed_model.load_state_dict(loaded["model_state_dict"])
    resumed_optimizer.load_state_dict(loaded["optimizer_state_dict"])
    resumed_model.eval()
    with torch.no_grad():
        resumed_logits = resumed_model(
            **legacy_l4_model_inputs(batch)
        ).behavior.detach()
    logit_delta = float((expected_logits - resumed_logits).abs().max().cpu())
    model_state_match = state_sha256(model.state_dict()) == state_sha256(
        resumed_model.state_dict()
    )
    optimizer_state_match = state_sha256(optimizer.state_dict()) == state_sha256(
        resumed_optimizer.state_dict()
    )
    original_next_loss = _optimizer_step(model, optimizer, batch)
    resumed_next_loss = _optimizer_step(resumed_model, resumed_optimizer, batch)
    next_model_match = state_sha256(model.state_dict()) == state_sha256(
        resumed_model.state_dict()
    )
    next_optimizer_match = state_sha256(optimizer.state_dict()) == state_sha256(
        resumed_optimizer.state_dict()
    )
    errors: list[str] = []
    if logit_delta != 0.0:
        errors.append(f"resume_logit_delta={logit_delta}")
    if not model_state_match:
        errors.append("resume_model_state_mismatch")
    if not optimizer_state_match:
        errors.append("resume_optimizer_state_mismatch")
    if original_next_loss != resumed_next_loss:
        errors.append("resume_next_loss_mismatch")
    if not next_model_match:
        errors.append("resume_next_model_state_mismatch")
    if not next_optimizer_match:
        errors.append("resume_next_optimizer_state_mismatch")
    result = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size_bytes": int(checkpoint_path.stat().st_size),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "first_loss": first_loss,
        "max_logit_delta": logit_delta,
        "model_state_match": model_state_match,
        "optimizer_state_match": optimizer_state_match,
        "next_step_loss_match": original_next_loss == resumed_next_loss,
        "next_model_state_match": next_model_match,
        "next_optimizer_state_match": next_optimizer_match,
        "errors": errors,
        "valid": not errors,
    }
    del optimizer, model, resumed_optimizer, resumed_model, loaded, payload
    _empty_cuda_cache(device)
    return result


def _tiny_overfit_audit(
    config: LegacyL4Config,
    bundle: LegacyL4DataBundle,
    rows: pd.DataFrame,
    *,
    device: torch.device,
) -> dict[str, Any]:
    _seed_all(config.short.seed)
    model = build_legacy_l4_model(config).to(device)
    features, masks, deltas, targets = _precompute_frame_features(
        model,
        bundle,
        rows,
        batch_events=config.short.frame_batch_events,
        device=device,
    )
    trainable = _freeze_frame_encoder(model)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.short.tiny_learning_rate,
        weight_decay=0.0,
    )
    model.eval()
    with torch.no_grad():
        initial_logits = _forward_from_frame_features(
            model,
            features,
            masks,
            deltas,
        )
        initial_loss = float(
            nn.functional.cross_entropy(initial_logits, targets).cpu()
        )
    losses: list[float] = []
    first_gradient: dict[str, Any] | None = None
    for _ in range(config.short.tiny_steps):
        model.train()
        _image_encoder(model).frame_encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        logits = _forward_from_frame_features(
            model,
            features,
            masks,
            deltas,
        )
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()
        if first_gradient is None:
            first_gradient = _frozen_path_gradient_audit(model)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        final_logits = _forward_from_frame_features(
            model,
            features,
            masks,
            deltas,
        )
        final_loss = float(nn.functional.cross_entropy(final_logits, targets).cpu())
        accuracy = float(final_logits.argmax(dim=1).eq(targets).float().mean().cpu())
    ratio = final_loss / initial_loss if initial_loss > 0.0 else float("inf")
    errors: list[str] = []
    if not np.isfinite([initial_loss, final_loss, *losses]).all():
        errors.append("tiny_overfit_nonfinite_loss")
    if accuracy < config.short.tiny_minimum_accuracy:
        errors.append(f"tiny_overfit_accuracy={accuracy}")
    if ratio > config.short.tiny_maximum_loss_ratio:
        errors.append(f"tiny_overfit_loss_ratio={ratio}")
    if first_gradient is None or not first_gradient["valid"]:
        errors.append("tiny_overfit_gradient_failure")
    class_counts = {
        label: int(rows["behavior_label"].astype(str).eq(label).sum())
        for label in VALID_BEHAVIORS
    }
    if set(class_counts.values()) != {config.short.tiny_events_per_class}:
        errors.append("tiny_overfit_class_balance_mismatch")
    result = {
        "native_event_count": int(len(rows)),
        "unique_native_event_count": int(rows["temporal_unit_key"].nunique()),
        "class_counts": class_counts,
        "frame_features_shape": list(features.shape),
        "frame_encoder_policy": "frozen_after_real_cache_feature_extraction",
        "optimizer_steps": config.short.tiny_steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_ratio": ratio,
        "memorization_accuracy": accuracy,
        "minimum_memorization_accuracy": config.short.tiny_minimum_accuracy,
        "maximum_loss_ratio": config.short.tiny_maximum_loss_ratio,
        "first_gradient": first_gradient,
        "final_trainable_state_sha256": state_sha256(
            _trainable_path_state(model)
        ),
        "errors": errors,
        "valid": not errors,
    }
    del optimizer, model, features, masks, deltas, targets
    _empty_cuda_cache(device)
    return result


def _precompute_frame_features(
    model: nn.Module,
    bundle: LegacyL4DataBundle,
    rows: pd.DataFrame,
    *,
    batch_events: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    encoder = _image_encoder(model)
    encoder.frame_encoder.eval()
    feature_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    delta_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_events):
            batch_rows = rows.iloc[start : start + batch_events]
            batch = load_legacy_l4_batch(bundle, batch_rows, device=device)
            batch_size, sequence_length = batch.images.shape[:2]
            normalized = encoder._normalize(
                batch.images.reshape(
                    batch_size * sequence_length,
                    *batch.images.shape[2:],
                )
            )
            encoded = encoder.frame_encoder(normalized).reshape(
                batch_size,
                sequence_length,
                -1,
            )
            feature_rows.append(encoded.detach().cpu())
            mask_rows.append(batch.masks.detach().cpu())
            delta_rows.append(batch.time_delta.detach().cpu())
            target_rows.append(batch.targets.detach().cpu())
    return (
        torch.cat(feature_rows).to(device),
        torch.cat(mask_rows).to(device),
        torch.cat(delta_rows).to(device),
        torch.cat(target_rows).to(device),
    )


def _forward_from_frame_features(
    model: nn.Module,
    frame_features: torch.Tensor,
    masks: torch.Tensor,
    time_delta: torch.Tensor,
) -> torch.Tensor:
    encoder = _image_encoder(model)
    projected = encoder.temporal_projection(frame_features)
    sequence = encoder.temporal_encoder(
        projected,
        masks.bool(),
        time_delta=time_delta,
    )
    return model.backbone.classifier(sequence)


def _image_encoder(model: nn.Module) -> ImageSequenceEncoder:
    encoder = getattr(getattr(model, "backbone", None), "image_encoder", None)
    if not isinstance(encoder, ImageSequenceEncoder):
        raise ValueError("legacy L4 actor image encoder is missing")
    return encoder


def _freeze_frame_encoder(model: nn.Module) -> list[nn.Parameter]:
    encoder = _image_encoder(model)
    for parameter in encoder.frame_encoder.parameters():
        parameter.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("legacy L4 frozen path has no trainable parameters")
    return trainable


def _frozen_path_gradient_audit(model: nn.Module) -> dict[str, Any]:
    groups = {
        "temporal_projection": "backbone.image_encoder.temporal_projection",
        "temporal_encoder": "backbone.image_encoder.temporal_encoder",
        "behavior_head": "backbone.classifier",
    }
    rows: dict[str, Any] = {}
    for name, prefix in groups.items():
        gradients = [
            parameter.grad
            for parameter_name, parameter in model.named_parameters()
            if parameter_name.startswith(prefix) and parameter.requires_grad
        ]
        finite = bool(gradients) and all(
            value is not None and torch.isfinite(value).all()
            for value in gradients
        )
        total = float(
            sum(
                value.detach().abs().sum().cpu().item()
                for value in gradients
                if value is not None
            )
        )
        rows[name] = {
            "parameter_tensors": len(gradients),
            "finite": finite,
            "absolute_gradient_sum": total,
            "nonzero": total > 0.0,
        }
    return {
        "groups": rows,
        "valid": all(row["finite"] and row["nonzero"] for row in rows.values()),
    }


def _trainable_path_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in model.state_dict().items()
        if "frame_encoder" not in name
    }


def _cache_only_audit(bundle: LegacyL4DataBundle) -> dict[str, Any]:
    audit = bundle.actor_dataset.image_load_audit()
    errors: list[str] = []
    if audit["packed_image_cache_hits"] <= 0:
        errors.append("legacy_l4_no_packed_cache_hits")
    if audit["disk_image_cache_misses"] != 0:
        errors.append("legacy_l4_cache_miss")
    if audit["source_image_loads"] != 0:
        errors.append("legacy_l4_source_media_load")
    if bundle.actor_dataset.video_decode_count != 0:
        errors.append("legacy_l4_video_decode")
    if bundle.actor_dataset.video_seek_count != 0:
        errors.append("legacy_l4_video_seek")
    return {
        **audit,
        "video_decode_count": int(bundle.actor_dataset.video_decode_count),
        "video_seek_count": int(bundle.actor_dataset.video_seek_count),
        "errors": errors,
        "valid": not errors,
    }


def run_legacy_l4_fold_epoch(
    config: LegacyL4Config,
    *,
    short_audit_path: Path,
) -> dict[str, Any]:
    """Run one exact fold epoch without held-out performance comparison."""

    short = _read_json(short_audit_path)
    _validate_short_parent(config, short)
    device = _resolve_device(config.short.device)
    _seed_all(config.short.seed)
    _reset_peak_memory(device)
    started = time.perf_counter()
    errors: list[str] = []
    with load_legacy_l4_data(config) as bundle:
        rows = bundle.train_development_rows.sort_values(
            "temporal_unit_key",
            kind="mergesort",
        ).reset_index(drop=True)
        model = build_legacy_l4_model(config).to(device)
        frame_state_sha256 = state_sha256(
            _image_encoder(model).frame_encoder.state_dict()
        )
        features, masks, deltas, targets = _precompute_frame_features(
            model,
            bundle,
            rows,
            batch_events=config.fold_epoch.frame_batch_events,
            device=device,
        )
        trainable = _freeze_frame_encoder(model)
        optimizer = torch.optim.AdamW(
            trainable,
            lr=config.fold_epoch.learning_rate,
            weight_decay=config.fold_epoch.weight_decay,
        )
        epoch_audit = _train_one_feature_epoch(
            model,
            optimizer,
            features,
            masks,
            deltas,
            targets,
            batch_size=config.fold_epoch.train_batch_size,
            seed=config.short.seed,
        )
        cache_audit = _cache_only_audit(bundle)
        if cache_audit["packed_image_cache_hits"] != (
            len(rows) * config.model.sequence_length
        ):
            cache_audit["errors"].append("fold_epoch_packed_hit_count_mismatch")
            cache_audit["valid"] = False
        if not epoch_audit["valid"]:
            errors.extend(epoch_audit["errors"])
        if not cache_audit["valid"]:
            errors.extend(cache_audit["errors"])
        support = _eligible_support_audit(rows, config)
        if not support["valid"]:
            errors.extend(support["errors"])
        feature_audit = {
            "shape": list(features.shape),
            "dtype": str(features.dtype),
            "finite": bool(torch.isfinite(features).all()),
            "frame_encoder_state_sha256": frame_state_sha256,
            "visual_policy": config.fold_epoch.visual_policy,
            "source_media_fallback_allowed": False,
        }
        if not feature_audit["finite"]:
            errors.append("fold_epoch_nonfinite_frame_features")
        final_trainable_sha256 = state_sha256(_trainable_path_state(model))
        lineage_audit = bundle.lineage_audit
        declared_support_audit = bundle.support_audit
        del optimizer, model, features, masks, deltas, targets
        _empty_cuda_cache(device)
    memory_audit = _memory_audit(
        device,
        config.fold_epoch.maximum_peak_vram_fraction,
    )
    if not memory_audit["valid"]:
        errors.extend(memory_audit["errors"])
    runtime_sec = float(time.perf_counter() - started)
    valid = not errors
    return {
        "schema_version": L4_SCHEMA_VERSION,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L4"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L4"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "accuracy_f1_comparison_authorized": False,
        "l5_controlled_baselines_authorized": valid,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "l3_audit_path": str(config.l3_audit_json),
        "l3_audit_sha256": file_sha256(config.l3_audit_json),
        "short_audit_path": str(short_audit_path),
        "short_audit_sha256": file_sha256(short_audit_path),
        "implementation_source_sha256": file_sha256(Path(__file__)),
        "git_state": git_state(),
        "fold_id": config.fold_id,
        "temporal_view_name": config.temporal_view_name,
        "device": str(device),
        "runtime_sec": runtime_sec,
        "lineage_audit": lineage_audit,
        "declared_support_audit": declared_support_audit,
        "optimizer_support_audit": support,
        "feature_extraction_audit": feature_audit,
        "one_fold_one_epoch_audit": epoch_audit,
        "cache_only_audit": cache_audit,
        "memory_audit": memory_audit,
        "final_trainable_state_sha256": final_trainable_sha256,
        "held_out_predictions_computed": False,
        "held_out_accuracy_f1_computed": False,
        "errors": errors,
        "valid": valid,
    }


def _train_one_feature_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    masks: torch.Tensor,
    deltas: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(targets))
    losses: list[float] = []
    first_gradient: dict[str, Any] | None = None
    seen: list[int] = []
    model.train()
    _image_encoder(model).frame_encoder.eval()
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        index = torch.from_numpy(indices).long().to(features.device)
        optimizer.zero_grad(set_to_none=True)
        logits = _forward_from_frame_features(
            model,
            features.index_select(0, index),
            masks.index_select(0, index),
            deltas.index_select(0, index),
        )
        loss = nn.functional.cross_entropy(
            logits,
            targets.index_select(0, index),
        )
        loss.backward()
        if first_gradient is None:
            first_gradient = _frozen_path_gradient_audit(model)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        seen.extend(int(value) for value in indices)
    errors: list[str] = []
    if len(seen) != len(targets) or len(set(seen)) != len(targets):
        errors.append("fold_epoch_row_coverage_mismatch")
    if not np.isfinite(losses).all():
        errors.append("fold_epoch_nonfinite_loss")
    if first_gradient is None or not first_gradient["valid"]:
        errors.append("fold_epoch_gradient_failure")
    return {
        "epochs": 1,
        "native_event_rows": int(len(targets)),
        "unique_rows_seen": len(set(seen)),
        "batch_size": batch_size,
        "optimizer_steps": len(losses),
        "mean_training_loss": float(np.mean(losses)),
        "minimum_training_loss": float(np.min(losses)),
        "maximum_training_loss": float(np.max(losses)),
        "losses": losses,
        "first_gradient": first_gradient,
        "held_out_metrics_computed": False,
        "errors": errors,
        "valid": not errors,
    }


def _eligible_support_audit(
    rows: pd.DataFrame,
    config: LegacyL4Config,
) -> dict[str, Any]:
    class_counts = {
        label: int(rows["behavior_label"].astype(str).eq(label).sum())
        for label in VALID_BEHAVIORS
    }
    source_counts = {
        str(key): int(value)
        for key, value in rows["source_type"].astype(str).value_counts().items()
    }
    errors: list[str] = []
    if any(value <= 0 for value in class_counts.values()):
        errors.append("fold_epoch_missing_training_class")
    if not source_counts or any(value <= 0 for value in source_counts.values()):
        errors.append("fold_epoch_missing_training_source")
    if sum(class_counts.values()) != len(rows):
        errors.append("fold_epoch_class_support_total_mismatch")
    if sum(source_counts.values()) != len(rows):
        errors.append("fold_epoch_source_support_total_mismatch")
    if set(rows["oof_fold_id"].astype(str)) == {config.fold_id}:
        errors.append("fold_epoch_optimizer_rows_are_test_fold")
    return {
        "fold_id": config.fold_id,
        "optimizer_native_units": int(len(rows)),
        "class_counts": class_counts,
        "source_counts": source_counts,
        "all_classes_supported": all(value > 0 for value in class_counts.values()),
        "all_sources_supported": all(value > 0 for value in source_counts.values()),
        "errors": errors,
        "valid": not errors,
    }


def _validate_short_parent(
    config: LegacyL4Config,
    short: dict[str, Any],
) -> None:
    expected = {
        "schema_version": L4_SHORT_SCHEMA_VERSION,
        "status": "PASS_LEGACY_DEVELOPMENT_L4_SHORT",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "accuracy_f1_comparison_authorized": False,
        "fold_epoch_authorized": True,
        "config_sha256": config.sha256,
        "l3_audit_sha256": file_sha256(config.l3_audit_json),
        "implementation_source_sha256": file_sha256(Path(__file__)),
        "valid": True,
    }
    mismatches = {
        key: {"expected": value, "observed": short.get(key)}
        for key, value in expected.items()
        if short.get(key) != value
    }
    if short.get("errors"):
        mismatches["errors"] = {"expected": [], "observed": short["errors"]}
    resume = short.get("checkpoint_resume_audit", {})
    checkpoint_path = Path(str(resume.get("checkpoint_path", "")))
    if not checkpoint_path.is_file():
        mismatches["resume_checkpoint"] = {
            "expected": "existing file",
            "observed": str(checkpoint_path),
        }
    elif file_sha256(checkpoint_path) != resume.get("checkpoint_sha256"):
        mismatches["resume_checkpoint_sha256"] = {
            "expected": resume.get("checkpoint_sha256"),
            "observed": file_sha256(checkpoint_path),
        }
    if mismatches:
        raise ValueError(f"legacy L4 short parent mismatch: {mismatches}")


def _validate_resume_payload(
    payload: object,
    config: LegacyL4Config,
    *,
    l3_sha256: str,
) -> None:
    if not isinstance(payload, dict):
        raise ValueError("legacy L4 resume checkpoint is not an object")
    expected = {
        "schema_version": "classification_v2.legacy_l4_resume_probe.v1",
        "config_sha256": config.sha256,
        "l3_audit_sha256": l3_sha256,
    }
    mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    required = {
        "model_state_dict",
        "optimizer_state_dict",
        "rng_state",
    }
    missing = sorted(required.difference(payload))
    if missing:
        mismatches["missing"] = missing
    if mismatches:
        raise ValueError(f"legacy L4 resume checkpoint mismatch: {mismatches}")


def _memory_audit(
    device: torch.device,
    maximum_fraction: float,
) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "device": str(device),
            "peak_vram_bytes": 0,
            "total_vram_bytes": None,
            "peak_vram_fraction": 0.0,
            "maximum_peak_vram_fraction": maximum_fraction,
            "errors": [],
            "valid": True,
        }
    peak = int(torch.cuda.max_memory_allocated(device))
    total = int(torch.cuda.get_device_properties(device).total_memory)
    fraction = peak / total
    errors = (
        []
        if fraction <= maximum_fraction
        else [f"peak_vram_fraction={fraction:.6f}"]
    )
    return {
        "device": str(device),
        "peak_vram_bytes": peak,
        "total_vram_bytes": total,
        "peak_vram_fraction": fraction,
        "maximum_peak_vram_fraction": maximum_fraction,
        "errors": errors,
        "valid": not errors,
    }


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("legacy L4 requested CUDA but it is unavailable")
    return device


def _seed_all(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _empty_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def state_sha256(value: object) -> str:
    digest = hashlib.sha256()
    _update_state_digest(digest, value)
    return digest.hexdigest()


def _update_state_digest(digest: Any, value: object) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor")
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(json.dumps(list(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray")
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(json.dumps(list(array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
        return
    if isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value, key=lambda item: str(item)):
            _update_state_digest(digest, key)
            _update_state_digest(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode("utf-8"))
        for item in value:
            _update_state_digest(digest, item)
        return
    digest.update(type(value).__name__.encode("utf-8"))
    digest.update(repr(value).encode("utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(dirty), "dirty_entries": dirty}


def _strict_bool(series: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    valid = normalized.isin({"true", "false", "1", "0"})
    if not valid.all():
        raise ValueError(f"invalid boolean values in {name}")
    return normalized.isin({"true", "1"})


def _bool_scalar(value: object, *, name: str) -> bool:
    return bool(_strict_bool(pd.Series([value]), name=name).iloc[0])


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _object(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_exact_keys(
    payload: dict[str, Any],
    expected: set[str],
    *,
    name: str,
) -> None:
    missing = sorted(expected.difference(payload))
    unknown = sorted(set(payload).difference(expected))
    if missing or unknown:
        raise ValueError(f"{name} key mismatch: missing={missing}, unknown={unknown}")


def _require_columns(
    frame: pd.DataFrame,
    expected: set[str],
    *,
    name: str,
) -> None:
    missing = sorted(expected.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns={missing}")


__all__ = [
    "L4_CONFIG_SCHEMA_VERSION",
    "L4_SCHEMA_VERSION",
    "L4_SHORT_SCHEMA_VERSION",
    "LegacyL4Batch",
    "LegacyL4Config",
    "LegacyL4DataBundle",
    "build_legacy_l4_model",
    "file_sha256",
    "legacy_l4_model_inputs",
    "load_legacy_l4_batch",
    "load_legacy_l4_config",
    "load_legacy_l4_data",
    "run_legacy_l4_fold_epoch",
    "run_legacy_l4_short",
    "state_sha256",
]
