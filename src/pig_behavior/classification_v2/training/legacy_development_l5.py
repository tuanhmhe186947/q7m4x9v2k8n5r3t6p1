"""Controlled L5 baselines for legacy-only unreviewed development."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.contracts.temporal_tier_contract import (
    LEGACY_TEMPORAL_MODEL_VIEW_SPECS,
)
from pig_behavior.classification_v2.datasets.image_sequence_dataset import (
    ClassificationV2ImageSequenceDataset,
    ImageSequenceDatasetConfig,
)
from pig_behavior.classification_v2.evaluation.native_unit_metrics import (
    CLASS_GROUPS,
)
from pig_behavior.classification_v2.models.temporal_encoders import (
    TEMPORAL_ENCODER_NAMES,
)
from pig_behavior.classification_v2.models.visual_backbones import (
    visual_backbone_contract,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.lineage_hashing import (
    file_sha256,
)

L5_CONFIG_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.config.v1"
)
L5_READINESS_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.readiness.v1"
)
L5_CACHE_SCHEMA_VERSION = "classification_v2.legacy_development_l5.cache.v1"
LINEAGE_SCOPE = "legacy-only-unreviewed-development"
PACKED_AUDIT_REOPEN_ROWS = 2048
VISUAL_CONTROL_IDS = ("Vtiny", "V0", "V1", "V2")
VISUAL_PRETRAINED_IDS = ("V0", "V1", "V2")
TEMPORAL_LENGTHS = (6, 8, 12, 16)
SAMPLING_PROTOCOLS = (
    "all_sliding_event_balanced",
    "one_centered_window_matched",
)
TEMPORAL_ENCODERS = ("masked_mean", "masked_tcn", "small_transformer")
CANONICAL_RARE_CLASSES = tuple(str(label) for label in CLASS_GROUPS["rare"])


@dataclass(frozen=True, slots=True)
class LegacyL5Config:
    """Resolved strict config plus immutable lineage locations."""

    path: Path
    payload: dict[str, Any]
    development_root: Path
    primary_run_id: str
    l3_audit_relative_path: Path
    l4_audit_relative_path: Path
    l5_output_relative_path: Path

    @property
    def primary_root(self) -> Path:
        return self.development_root / self.primary_run_id

    @property
    def l3_audit_json(self) -> Path:
        return self.primary_root / self.l3_audit_relative_path

    @property
    def l4_audit_json(self) -> Path:
        return self.primary_root / self.l4_audit_relative_path

    @property
    def l5_output_root(self) -> Path:
        return self.primary_root / self.l5_output_relative_path

    @property
    def full_cache_224_root(self) -> Path:
        cache = self.payload["cache_contract"]
        return self.primary_root / str(cache["cache_224_relative_path"])

    @property
    def short_cache_224_root(self) -> Path:
        cache = self.payload["cache_contract"]
        return (
            self.development_root
            / str(cache["short_cache_224_run_id"])
            / str(cache["short_cache_224_relative_path"])
        )

    @property
    def short_cache_224_reference_root(self) -> Path:
        cache = self.payload["cache_contract"]
        return (
            self.development_root
            / str(cache["short_cache_224_run_id"])
            / str(cache["short_cache_224_reference_relative_path"])
        )

    @property
    def short_image_context_root(self) -> Path:
        cache = self.payload["cache_contract"]
        return (
            self.development_root
            / str(cache["short_cache_224_run_id"])
            / str(cache["short_image_context_relative_path"])
        )

    @property
    def sha256(self) -> str:
        return file_sha256(self.path)


def load_legacy_l5_config(path: Path) -> LegacyL5Config:
    """Load one exact L5 semantic contract and reject uncontrolled drift."""

    payload = _read_json(path)
    required = {
        "schema_version",
        "lineage_scope",
        "human_review_complete",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        "canonical_full_oof_authorized",
        "outer_holdout_predictions_authorized",
        "development_accuracy_f1_authorized",
        "development_root",
        "primary_run_id",
        "l3_audit_relative_path",
        "l4_audit_relative_path",
        "l5_output_relative_path",
        "expected_counts",
        "split_contract",
        "visual_view",
        "temporal_matrix",
        "cache_contract",
        "visual_controls",
        "common_model",
        "optimization",
        "feature_cache",
        "metric_contract",
        "promotion_contract",
    }
    _require_exact_keys(payload, required, name="legacy L5 config")
    if payload["schema_version"] != L5_CONFIG_SCHEMA_VERSION:
        raise ValueError("legacy L5 config schema mismatch")
    if payload["lineage_scope"] != LINEAGE_SCOPE:
        raise ValueError("legacy L5 lineage scope mismatch")
    false_claims = (
        "human_review_complete",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        "canonical_full_oof_authorized",
        "outer_holdout_predictions_authorized",
    )
    if any(payload[name] is not False for name in false_claims):
        raise ValueError("legacy L5 config exceeds its claim boundary")
    if payload["development_accuracy_f1_authorized"] is not True:
        raise ValueError("legacy L5 development metrics must be explicit")
    config = LegacyL5Config(
        path=path,
        payload=payload,
        development_root=Path(str(payload["development_root"])),
        primary_run_id=str(payload["primary_run_id"]),
        l3_audit_relative_path=Path(str(payload["l3_audit_relative_path"])),
        l4_audit_relative_path=Path(str(payload["l4_audit_relative_path"])),
        l5_output_relative_path=Path(str(payload["l5_output_relative_path"])),
    )
    _validate_expected_counts(payload["expected_counts"])
    _validate_split_contract(payload["split_contract"])
    _validate_visual_view(payload["visual_view"])
    _validate_temporal_matrix(payload["temporal_matrix"])
    _validate_cache_contract(payload["cache_contract"])
    _validate_visual_controls(payload["visual_controls"])
    _validate_common_model(payload["common_model"])
    _validate_optimization(payload["optimization"])
    _validate_feature_cache(payload["feature_cache"])
    _validate_metric_contract(payload["metric_contract"])
    _validate_promotion_contract(payload["promotion_contract"])
    return config


def _validate_expected_counts(value: object) -> None:
    payload = _object(value, name="expected_counts")
    fields = {
        "selected_native_units",
        "development_valid_native_units",
        "policy_invalid_native_units",
        "train_native_units",
        "validation_native_units",
        "outer_holdout_native_units",
        "image_context_rows",
    }
    _require_exact_keys(payload, fields, name="expected_counts")
    if any(int(payload[name]) <= 0 for name in fields):
        raise ValueError("legacy L5 expected counts must be positive")
    total = int(payload["development_valid_native_units"])
    roles = sum(
        int(payload[name])
        for name in (
            "train_native_units",
            "validation_native_units",
            "outer_holdout_native_units",
        )
    )
    if total != roles:
        raise ValueError("legacy L5 development role counts do not reconcile")
    selected = int(payload["selected_native_units"])
    invalid = int(payload["policy_invalid_native_units"])
    if selected != total + invalid:
        raise ValueError("legacy L5 valid and policy-invalid counts do not reconcile")


def _validate_split_contract(value: object) -> None:
    payload = _object(value, name="split_contract")
    fields = {
        "outer_holdout_fold_id",
        "development_validation_fold_id",
        "training_role",
        "grouping_columns",
        "outer_predictions_forbidden",
        "outer_metrics_forbidden",
    }
    _require_exact_keys(payload, fields, name="split_contract")
    outer = str(payload["outer_holdout_fold_id"])
    validation = str(payload["development_validation_fold_id"])
    if outer != "native_oof_005" or validation != "native_oof_006":
        raise ValueError("legacy L5 sealed split IDs drifted")
    if outer == validation:
        raise ValueError("legacy L5 validation and outer folds must differ")
    if payload["training_role"] != (
        "development_valid_and_in_neither_held_out_fold"
    ):
        raise ValueError("legacy L5 training-role policy drift")
    if list(payload["grouping_columns"]) != [
        "recording_group_id",
        "video_key",
    ]:
        raise ValueError("legacy L5 grouping columns drifted")
    if payload["outer_predictions_forbidden"] is not True:
        raise ValueError("legacy L5 outer predictions must remain forbidden")
    if payload["outer_metrics_forbidden"] is not True:
        raise ValueError("legacy L5 outer metrics must remain forbidden")


def _validate_visual_view(value: object) -> None:
    payload = _object(value, name="visual_view")
    fields = {
        "temporal_view_name",
        "selection_column",
        "sequence_length",
        "sampling_protocol",
    }
    _require_exact_keys(payload, fields, name="visual_view")
    view_name = "legacy_t16_centered_matched_observed_time"
    spec = LEGACY_TEMPORAL_MODEL_VIEW_SPECS[view_name]
    expected = {
        "temporal_view_name": view_name,
        "selection_column": str(spec["selection_column"]),
        "sequence_length": int(spec["sequence_length"]),
        "sampling_protocol": "one_centered_window_matched",
    }
    _require_mapping_equal(payload, expected, name="visual_view")


def _validate_temporal_matrix(value: object) -> None:
    payload = _object(value, name="temporal_matrix")
    fields = {
        "sequence_lengths",
        "sampling_protocols",
        "temporal_encoders",
        "all_sliding_stride",
        "native_aggregation",
    }
    _require_exact_keys(payload, fields, name="temporal_matrix")
    if tuple(int(item) for item in payload["sequence_lengths"]) != TEMPORAL_LENGTHS:
        raise ValueError("legacy L5 temporal length matrix drift")
    if tuple(str(item) for item in payload["sampling_protocols"]) != (
        SAMPLING_PROTOCOLS
    ):
        raise ValueError("legacy L5 sampling protocol matrix drift")
    encoders = tuple(str(item) for item in payload["temporal_encoders"])
    if encoders != TEMPORAL_ENCODERS:
        raise ValueError("legacy L5 temporal encoder matrix drift")
    if not set(encoders).issubset(TEMPORAL_ENCODER_NAMES):
        raise ValueError("legacy L5 contains an unsupported temporal encoder")
    if int(payload["all_sliding_stride"]) != 3:
        raise ValueError("legacy L5 all-sliding stride must remain three")
    if payload["native_aggregation"] != (
        "mean_probability_by_temporal_unit_key"
    ):
        raise ValueError("legacy L5 native aggregation policy drift")


def _validate_cache_contract(value: object) -> None:
    payload = _object(value, name="cache_contract")
    fields = {
        "source",
        "resize_policy",
        "dtype",
        "channel_order",
        "short_context_rows",
        "short_preview_rows",
        "cache_160_relative_path",
        "cache_224_relative_path",
        "short_cache_224_run_id",
        "short_image_context_relative_path",
        "short_cache_224_reference_relative_path",
        "short_cache_224_relative_path",
        "upscale_from_160_forbidden",
        "source_media_fallback_during_training",
    }
    _require_exact_keys(payload, fields, name="cache_contract")
    expected = {
        "source": "original_video_and_bbox_context",
        "resize_policy": "letterbox_preserve_aspect_rgb_pad_black_v1",
        "dtype": "uint8",
        "channel_order": "RGB_HWC",
        "upscale_from_160_forbidden": True,
        "source_media_fallback_during_training": False,
    }
    _require_mapping_equal(payload, expected, name="cache_contract")
    if int(payload["short_context_rows"]) <= 0:
        raise ValueError("legacy L5 short cache row count must be positive")
    if int(payload["short_preview_rows"]) <= 0:
        raise ValueError("legacy L5 short preview count must be positive")
    cache_160 = Path(str(payload["cache_160_relative_path"]))
    cache_224 = Path(str(payload["cache_224_relative_path"]))
    short_run = Path(str(payload["short_cache_224_run_id"]))
    short_context = Path(str(payload["short_image_context_relative_path"]))
    short_reference = Path(
        str(payload["short_cache_224_reference_relative_path"])
    )
    short_relative = Path(str(payload["short_cache_224_relative_path"]))
    path_parts = (
        cache_160,
        cache_224,
        short_run,
        short_context,
        short_reference,
        short_relative,
    )
    if any(path.is_absolute() or ".." in path.parts for path in path_parts):
        raise ValueError("legacy L5 cache paths must stay relative and local")
    paths = (
        str(cache_160),
        str(cache_224),
        str(short_run / short_reference),
        str(short_run / short_relative),
    )
    if len(set(paths)) != len(paths):
        raise ValueError("legacy L5 cache paths must be distinct")
    if (
        "160" not in paths[0]
        or "224" not in paths[1]
        or any("224" not in path for path in paths[2:])
    ):
        raise ValueError("legacy L5 cache paths do not bind their resolution")


def _validate_visual_controls(value: object) -> None:
    if not isinstance(value, list):
        raise ValueError("visual_controls must be a list")
    controls = [_object(row, name="visual control") for row in value]
    fields = {
        "control_id",
        "role",
        "backbone_name",
        "pretrained_family",
        "pretrained_weight_enum",
        "image_size",
        "frame_encoder_schedule",
    }
    for row in controls:
        _require_exact_keys(row, fields, name="visual control")
    ids = tuple(str(row["control_id"]) for row in controls)
    if ids != VISUAL_CONTROL_IDS:
        raise ValueError("legacy L5 visual control order drift")
    expected = {
        "Vtiny": (
            "smoke_cnn",
            "NONE_RANDOM_INIT",
            "NONE_RANDOM_INIT",
            160,
            "all_trainable_reference",
        ),
        "V0": (
            "resnet18",
            "IMAGENET1K_V1",
            "ResNet18_Weights.IMAGENET1K_V1",
            160,
            "frozen_all_epochs_v1",
        ),
        "V1": (
            "resnet18",
            "IMAGENET1K_V1",
            "ResNet18_Weights.IMAGENET1K_V1",
            224,
            "frozen_all_epochs_v1",
        ),
        "V2": (
            "resnet34",
            "IMAGENET1K_V1",
            "ResNet34_Weights.IMAGENET1K_V1",
            224,
            "frozen_all_epochs_v1",
        ),
    }
    contracts: dict[str, Any] = {}
    for row in controls:
        control_id = str(row["control_id"])
        observed = (
            str(row["backbone_name"]),
            str(row["pretrained_family"]),
            str(row["pretrained_weight_enum"]),
            int(row["image_size"]),
            str(row["frame_encoder_schedule"]),
        )
        if observed != expected[control_id]:
            raise ValueError(f"legacy L5 visual control drift={control_id}")
        contracts[control_id] = visual_backbone_contract(
            observed[0],
            observed[2],
        )
    pretrained = [contracts[name] for name in VISUAL_PRETRAINED_IDS]
    if {contract.normalization_name for contract in pretrained} != {
        "imagenet_1k_rgb"
    }:
        raise ValueError("legacy L5 pretrained normalization name drift")
    if len({contract.input_mean for contract in pretrained}) != 1:
        raise ValueError("legacy L5 pretrained input mean drift")
    if len({contract.input_std for contract in pretrained}) != 1:
        raise ValueError("legacy L5 pretrained input std drift")
    if not all(contract.uses_pretrained_weights for contract in pretrained):
        raise ValueError("legacy L5 V0/V1/V2 must all use pretrained weights")


def _validate_common_model(value: object) -> None:
    payload = _object(value, name="common_model")
    fields = {
        "model_mode",
        "temporal_encoder_name",
        "hidden_dim",
        "dropout",
        "transformer_layers",
        "transformer_heads",
        "direct_ten_class_supervision",
        "spatial_enabled",
        "interaction_context_enabled",
        "visual_context_enabled",
        "multitask_enabled",
    }
    _require_exact_keys(payload, fields, name="common_model")
    expected = {
        "model_mode": "actor_temporal",
        "temporal_encoder_name": "masked_mean",
        "direct_ten_class_supervision": True,
        "spatial_enabled": False,
        "interaction_context_enabled": False,
        "visual_context_enabled": False,
        "multitask_enabled": False,
    }
    _require_mapping_equal(payload, expected, name="common_model")
    if int(payload["hidden_dim"]) <= 0:
        raise ValueError("legacy L5 hidden dimension must be positive")
    if not 0.0 <= float(payload["dropout"]) < 1.0:
        raise ValueError("legacy L5 dropout must be in [0,1)")
    layers = int(payload["transformer_layers"])
    heads = int(payload["transformer_heads"])
    hidden = int(payload["hidden_dim"])
    if layers not in {1, 2} or heads <= 0 or hidden % heads:
        raise ValueError("legacy L5 small Transformer dimensions are invalid")


def _validate_optimization(value: object) -> None:
    payload = _object(value, name="optimization")
    fields = {
        "seeds",
        "maximum_epochs",
        "minimum_epochs",
        "early_stopping_patience",
        "head_batch_size",
        "evaluation_batch_size",
        "tiny_event_batch_size",
        "learning_rate",
        "tiny_learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "loss",
        "sampler",
        "augmentation",
        "deterministic",
        "declared_local_gpu_vram_gib",
        "maximum_peak_vram_fraction",
        "short_vram_probe_required",
        "oom_retry_allowed",
    }
    _require_exact_keys(payload, fields, name="optimization")
    seeds = [int(seed) for seed in payload["seeds"]]
    if len(seeds) != 3 or len(set(seeds)) != 3 or min(seeds) <= 0:
        raise ValueError("legacy L5 requires exactly three distinct seeds")
    integer_fields = (
        "maximum_epochs",
        "minimum_epochs",
        "early_stopping_patience",
        "head_batch_size",
        "evaluation_batch_size",
        "tiny_event_batch_size",
    )
    if any(int(payload[name]) <= 0 for name in integer_fields):
        raise ValueError("legacy L5 optimization integers must be positive")
    if int(payload["minimum_epochs"]) > int(payload["maximum_epochs"]):
        raise ValueError("legacy L5 minimum epochs exceed maximum epochs")
    positive_floats = (
        "learning_rate",
        "tiny_learning_rate",
        "gradient_clip_norm",
    )
    if any(float(payload[name]) <= 0.0 for name in positive_floats):
        raise ValueError("legacy L5 optimization rates must be positive")
    if float(payload["weight_decay"]) < 0.0:
        raise ValueError("legacy L5 weight decay must be nonnegative")
    expected = {
        "loss": "event_mass_balanced_cross_entropy",
        "sampler": "deterministic_seeded_shuffle",
        "augmentation": "none",
        "deterministic": True,
        "declared_local_gpu_vram_gib": 4,
        "short_vram_probe_required": True,
        "oom_retry_allowed": False,
    }
    _require_mapping_equal(payload, expected, name="optimization")
    fraction = float(payload["maximum_peak_vram_fraction"])
    if not 0.0 < fraction <= 0.7:
        raise ValueError("legacy L5 maximum VRAM fraction must be in (0,0.7]")


def _validate_feature_cache(value: object) -> None:
    payload = _object(value, name="feature_cache")
    fields = {
        "dtype",
        "resnet18_frame_batch_size",
        "resnet34_frame_batch_size",
        "require_packed_actor_cache",
        "repeat_sample_rows",
    }
    _require_exact_keys(payload, fields, name="feature_cache")
    if payload["dtype"] != "float32":
        raise ValueError("legacy L5 feature cache must remain float32")
    if payload["require_packed_actor_cache"] is not True:
        raise ValueError("legacy L5 feature cache must require packed actors")
    expected_batches = {
        "resnet18_frame_batch_size": 16,
        "resnet34_frame_batch_size": 8,
    }
    _require_mapping_equal(payload, expected_batches, name="feature_cache")
    for name in (
        "resnet18_frame_batch_size",
        "resnet34_frame_batch_size",
        "repeat_sample_rows",
    ):
        if int(payload[name]) <= 0:
            raise ValueError(f"legacy L5 {name} must be positive")


def _validate_metric_contract(value: object) -> None:
    payload = _object(value, name="metric_contract")
    fields = {
        "primary_metric",
        "secondary_metrics",
        "class_order",
        "uncertainty_cluster",
        "bootstrap_iterations",
        "outer_predictions_used_for_model_selection",
    }
    _require_exact_keys(payload, fields, name="metric_contract")
    expected = {
        "primary_metric": (
            "validation_native_unit_macro_f1_global_10_class"
        ),
        "secondary_metrics": [
            "validation_native_unit_accuracy",
            "validation_native_unit_nll",
        ],
        "class_order": list(VALID_BEHAVIORS),
        "uncertainty_cluster": "video_key",
        "outer_predictions_used_for_model_selection": False,
    }
    _require_mapping_equal(payload, expected, name="metric_contract")
    if int(payload["bootstrap_iterations"]) < 1000:
        raise ValueError("legacy L5 paired bootstrap is too small")


def _validate_promotion_contract(value: object) -> None:
    payload = _object(value, name="promotion_contract")
    fields = {
        "material_macro_f1_gain_over_tiny",
        "maximum_rare_group_recall_drop",
        "rare_classes",
        "validation_rare_class_warning_threshold",
        "minimum_positive_seed_deltas",
        "maximum_runtime_ratio_to_parent",
        "prefer_simpler_within_macro_f1",
        "pretrained_fallback_if_frozen_gate_fails",
        "fallback_requires_new_short_gate",
    }
    _require_exact_keys(payload, fields, name="promotion_contract")
    bounded = (
        "material_macro_f1_gain_over_tiny",
        "maximum_rare_group_recall_drop",
        "prefer_simpler_within_macro_f1",
    )
    if any(not 0.0 <= float(payload[name]) <= 1.0 for name in bounded):
        raise ValueError("legacy L5 promotion thresholds must be in [0,1]")
    rare_classes = tuple(str(item) for item in payload["rare_classes"])
    if rare_classes != CANONICAL_RARE_CLASSES:
        raise ValueError("legacy L5 rare-class guardrail drift")
    if int(payload["validation_rare_class_warning_threshold"]) <= 0:
        raise ValueError("legacy L5 rare-class warning threshold must be positive")
    if int(payload["minimum_positive_seed_deltas"]) not in {2, 3}:
        raise ValueError("legacy L5 positive-seed threshold must be two or three")
    if float(payload["maximum_runtime_ratio_to_parent"]) < 1.0:
        raise ValueError("legacy L5 runtime ratio must be at least one")
    if payload["pretrained_fallback_if_frozen_gate_fails"] != (
        "frozen_then_layer4_v1"
    ):
        raise ValueError("legacy L5 fallback schedule drift")
    if payload["fallback_requires_new_short_gate"] is not True:
        raise ValueError("legacy L5 fallback must require a new short gate")


def audit_legacy_l5_readiness(config: LegacyL5Config) -> dict[str, Any]:
    """Prove the sealed L5 development universe without model execution."""

    errors: list[str] = []
    l3 = _read_json(config.l3_audit_json)
    l4 = _read_json(config.l4_audit_json)
    _validate_parent_audits(config, l3, l4, errors)
    paths = _legacy_paths(config)
    native = pd.read_csv(paths["native_units"], low_memory=False)
    folds = pd.read_csv(paths["window_folds"], low_memory=False)
    split = _split_role_audit(config, native=native, folds=folds)
    errors.extend(split["errors"])
    views = _temporal_view_matrix_audit(config, native=native, folds=folds)
    errors.extend(views["errors"])
    models = _visual_control_audit(config)
    errors.extend(models["errors"])
    inputs = _input_hash_audit(config, paths)
    errors.extend(inputs["errors"])
    warnings = list(split["warnings"])
    valid = not errors
    return {
        "schema_version": L5_READINESS_SCHEMA_VERSION,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L5_READINESS"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L5_READINESS"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "development_accuracy_f1_authorized": valid,
        "outer_holdout_predictions_authorized": False,
        "outer_holdout_predictions_computed": False,
        "outer_holdout_metrics_computed": False,
        "short_224_cache_build_authorized": valid,
        "pretrained_weight_prepare_authorized": valid,
        "full_224_cache_build_authorized": False,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "l3_audit_path": str(config.l3_audit_json),
        "l3_audit_sha256": file_sha256(config.l3_audit_json),
        "l4_audit_path": str(config.l4_audit_json),
        "l4_audit_sha256": file_sha256(config.l4_audit_json),
        "implementation_source_sha256": file_sha256(Path(__file__)),
        "git_state": git_state(),
        "split_role_audit": split,
        "temporal_view_matrix_audit": views,
        "visual_control_audit": models,
        "input_hash_audit": inputs,
        "cache_root_contract": {
            "short_cache_224_root": str(config.short_cache_224_root),
            "short_cache_224_reference_root": str(
                config.short_cache_224_reference_root
            ),
            "short_image_context_root": str(config.short_image_context_root),
            "full_cache_224_root": str(config.full_cache_224_root),
        },
        "accuracy_f1_computed": False,
        "optimizer_steps": 0,
        "pretrained_weight_downloads": 0,
        "warnings": warnings,
        "errors": errors,
        "valid": valid,
    }


def _validate_parent_audits(
    config: LegacyL5Config,
    l3: dict[str, Any],
    l4: dict[str, Any],
    errors: list[str],
) -> None:
    l3_expected = {
        "status": "PASS_LEGACY_DEVELOPMENT_L3",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "valid": True,
    }
    l4_expected = {
        "status": "PASS_LEGACY_DEVELOPMENT_L4",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "l5_controlled_baselines_authorized": True,
        "valid": True,
    }
    errors.extend(_mapping_errors(l3, l3_expected, prefix="l3_parent"))
    errors.extend(_mapping_errors(l4, l4_expected, prefix="l4_parent"))
    if l3.get("errors"):
        errors.append("l3_parent_has_errors")
    if l4.get("errors"):
        errors.append("l4_parent_has_errors")
    if l4.get("l3_audit_sha256") != file_sha256(config.l3_audit_json):
        errors.append("l4_parent_l3_hash_mismatch")


def _split_role_audit(
    config: LegacyL5Config,
    *,
    native: pd.DataFrame,
    folds: pd.DataFrame,
) -> dict[str, Any]:
    split = _object(config.payload["split_contract"], name="split_contract")
    expected = _object(config.payload["expected_counts"], name="expected_counts")
    view = _object(config.payload["visual_view"], name="visual_view")
    selection_column = str(view["selection_column"])
    _require_columns(
        native,
        {
            "temporal_unit_key",
            "native_unit_valid_for_development",
            "lineage_scope",
            "human_review_complete",
        },
        name="native units",
    )
    _require_columns(
        folds,
        {
            "window_id",
            "temporal_unit_key",
            "oof_fold_id",
            "behavior_label",
            "source_type",
            "video_key",
            "recording_group_id",
            "lineage_scope",
            "human_review_complete",
            selection_column,
        },
        name="window folds",
    )
    errors: list[str] = []
    if native["temporal_unit_key"].astype(str).duplicated().any():
        errors.append("duplicate_native_temporal_unit_key")
    selected_mask = _strict_bool(
        folds[selection_column],
        name=selection_column,
    )
    selected = folds.loc[selected_mask].copy()
    valid = _strict_bool(
        native["native_unit_valid_for_development"],
        name="native_unit_valid_for_development",
    )
    native_valid = native.assign(development_valid=valid.to_numpy())
    selected = selected.merge(
        native_valid[["temporal_unit_key", "development_valid"]],
        on="temporal_unit_key",
        how="left",
        validate="one_to_one",
    )
    if selected["development_valid"].isna().any():
        errors.append("selected_units_missing_native_validity")
    selected["development_valid"] = selected["development_valid"].fillna(False)
    outer_fold = str(split["outer_holdout_fold_id"])
    validation_fold = str(split["development_validation_fold_id"])
    role = np.full(len(selected), "policy_invalid", dtype=object)
    valid_mask = selected["development_valid"].astype(bool).to_numpy()
    fold_ids = selected["oof_fold_id"].astype(str).to_numpy()
    role[valid_mask & (fold_ids == outer_fold)] = "outer_holdout"
    role[valid_mask & (fold_ids == validation_fold)] = "validation"
    role[
        valid_mask
        & (fold_ids != outer_fold)
        & (fold_ids != validation_fold)
    ] = "train"
    selected["l5_role"] = role
    roles = {
        name: selected.loc[selected["l5_role"].eq(name)].copy()
        for name in ("train", "validation", "outer_holdout", "policy_invalid")
    }
    observed_counts = {
        "selected_native_units": int(len(selected)),
        "development_valid_native_units": int(valid_mask.sum()),
        "policy_invalid_native_units": int(len(roles["policy_invalid"])),
        "train_native_units": int(len(roles["train"])),
        "validation_native_units": int(len(roles["validation"])),
        "outer_holdout_native_units": int(len(roles["outer_holdout"])),
    }
    for name, observed in observed_counts.items():
        if observed != int(expected[name]):
            errors.append(
                f"split_count_mismatch={name}:{observed}!={int(expected[name])}"
            )
    if set(selected["lineage_scope"].astype(str)) != {LINEAGE_SCOPE}:
        errors.append("split_lineage_scope_mismatch")
    if _strict_bool(
        selected["human_review_complete"],
        name="split human_review_complete",
    ).any():
        errors.append("split_rows_claim_human_review")
    overlap = _role_overlap_audit(roles)
    errors.extend(overlap["errors"])
    promotion = config.payload["promotion_contract"]
    support = _role_support_audit(
        roles,
        rare_classes=tuple(str(item) for item in promotion["rare_classes"]),
        warning_threshold=int(
            promotion["validation_rare_class_warning_threshold"]
        ),
    )
    errors.extend(support["errors"])
    role_unit_hashes = {
        name: _ordered_id_sha256(frame["temporal_unit_key"])
        for name, frame in roles.items()
    }
    return {
        "outer_holdout_fold_id": outer_fold,
        "development_validation_fold_id": validation_fold,
        "role_assignment_policy": str(split["training_role"]),
        "observed_counts": observed_counts,
        "role_unit_sha256": role_unit_hashes,
        "role_overlap": overlap,
        "class_support": support,
        "outer_holdout_prediction_access": "FORBIDDEN",
        "outer_holdout_metric_access": "FORBIDDEN",
        "warnings": support["warnings"],
        "errors": errors,
        "valid": not errors,
    }


def _role_overlap_audit(roles: dict[str, pd.DataFrame]) -> dict[str, Any]:
    errors: list[str] = []
    pairs: dict[str, Any] = {}
    role_names = ("train", "validation", "outer_holdout")
    for left_index, left in enumerate(role_names):
        for right in role_names[left_index + 1 :]:
            key = f"{left}_vs_{right}"
            values: dict[str, int] = {}
            for column in (
                "temporal_unit_key",
                "recording_group_id",
                "video_key",
            ):
                overlap = set(roles[left][column].astype(str)).intersection(
                    roles[right][column].astype(str)
                )
                values[f"{column}_overlap"] = len(overlap)
                if overlap:
                    errors.append(f"role_overlap={key}:{column}:{len(overlap)}")
            pairs[key] = values
    return {"pairs": pairs, "errors": errors, "valid": not errors}


def _role_support_audit(
    roles: dict[str, pd.DataFrame],
    *,
    rare_classes: tuple[str, ...],
    warning_threshold: int,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    reports: dict[str, Any] = {}
    for role_name in ("train", "validation", "outer_holdout"):
        frame = roles[role_name]
        counts = {
            label: int(frame["behavior_label"].astype(str).eq(label).sum())
            for label in VALID_BEHAVIORS
        }
        missing = [label for label, count in counts.items() if count <= 0]
        if missing:
            errors.append(f"role_missing_classes={role_name}:{missing}")
        reports[role_name] = {
            "native_units": int(len(frame)),
            "class_counts": counts,
            "supported_class_count": sum(count > 0 for count in counts.values()),
            "video_count": int(frame["video_key"].astype(str).nunique()),
            "recording_group_count": int(
                frame["recording_group_id"].astype(str).nunique()
            ),
        }
    validation_counts = reports["validation"]["class_counts"]
    for label in rare_classes:
        count = int(validation_counts[label])
        if count < warning_threshold:
            warnings.append(
                "validation_rare_class_low_support="
                f"{label}:{count}<{warning_threshold}"
            )
    return {
        "roles": reports,
        "rare_classes": list(rare_classes),
        "validation_warning_threshold": warning_threshold,
        "warnings": warnings,
        "errors": errors,
        "valid": not errors,
    }


def _temporal_view_matrix_audit(
    config: LegacyL5Config,
    *,
    native: pd.DataFrame,
    folds: pd.DataFrame,
) -> dict[str, Any]:
    errors: list[str] = []
    expected_native = int(
        config.payload["expected_counts"]["selected_native_units"]
    )
    reports: dict[str, Any] = {}
    artifact_hashes: dict[str, str] = {}
    for length in TEMPORAL_LENGTHS:
        for protocol in SAMPLING_PROTOCOLS:
            view_name = _view_name(length, protocol)
            spec = LEGACY_TEMPORAL_MODEL_VIEW_SPECS[view_name]
            selection_column = str(spec["selection_column"])
            selected_mask = _strict_bool(
                folds[selection_column],
                name=selection_column,
            )
            selected = folds.loc[selected_mask].copy()
            expected_placements = (
                ((16 - length) // 3) + 1
                if protocol == "all_sliding_event_balanced"
                else 1
            )
            counts = selected.groupby("temporal_unit_key", sort=False).size()
            wrong_placements = int(counts.ne(expected_placements).sum())
            missing_units = int(expected_native - len(counts))
            duplicate_windows = int(selected["window_id"].astype(str).duplicated().sum())
            expected_rows = expected_native * expected_placements
            if len(selected) != expected_rows:
                errors.append(
                    f"temporal_view_row_count={view_name}:{len(selected)}"
                    f"!={expected_rows}"
                )
            if wrong_placements:
                errors.append(
                    f"temporal_view_wrong_placements={view_name}:"
                    f"{wrong_placements}"
                )
            if missing_units:
                errors.append(
                    f"temporal_view_missing_native_units={view_name}:"
                    f"{missing_units}"
                )
            if duplicate_windows:
                errors.append(
                    f"temporal_view_duplicate_windows={view_name}:"
                    f"{duplicate_windows}"
                )
            view_path = (
                config.primary_root
                / "06_temporal_tier_contract"
                / str(spec["slot_manifest_filename"])
            )
            if not view_path.is_file():
                errors.append(f"temporal_view_manifest_missing={view_name}")
                view_hash = ""
                slot_rows = -1
            else:
                view_hash = file_sha256(view_path)
                slot_rows = _csv_row_count(view_path)
            expected_slot_rows = expected_rows * length
            if slot_rows != expected_slot_rows:
                errors.append(
                    f"temporal_view_slot_rows={view_name}:{slot_rows}"
                    f"!={expected_slot_rows}"
                )
            artifact_hashes[view_name] = view_hash
            reports[view_name] = {
                "sequence_length": length,
                "sampling_protocol": protocol,
                "selection_column": selection_column,
                "selected_window_rows": int(len(selected)),
                "native_unit_rows": int(len(counts)),
                "windows_per_native_unit": expected_placements,
                "wrong_placement_units": wrong_placements,
                "duplicate_window_rows": duplicate_windows,
                "slot_manifest_path": str(view_path),
                "slot_manifest_sha256": view_hash,
                "slot_rows": slot_rows,
                "expected_slot_rows": expected_slot_rows,
                "event_mass_per_native_unit": 1.0,
            }
    native_keys = set(native["temporal_unit_key"].astype(str))
    fold_keys = set(folds["temporal_unit_key"].astype(str))
    if not fold_keys.issubset(native_keys):
        errors.append("temporal_views_contain_unknown_native_units")
    return {
        "sequence_lengths": list(TEMPORAL_LENGTHS),
        "sampling_protocols": list(SAMPLING_PROTOCOLS),
        "temporal_encoders": list(TEMPORAL_ENCODERS),
        "native_aggregation": "mean_probability_by_temporal_unit_key",
        "reports": reports,
        "artifact_hashes": artifact_hashes,
        "errors": errors,
        "valid": not errors,
    }


def _visual_control_audit(config: LegacyL5Config) -> dict[str, Any]:
    errors: list[str] = []
    reports: dict[str, Any] = {}
    controls = config.payload["visual_controls"]
    common = config.payload["common_model"]
    for row in controls:
        contract = visual_backbone_contract(
            str(row["backbone_name"]),
            str(row["pretrained_weight_enum"]),
        )
        reports[str(row["control_id"])] = {
            **row,
            "normalization_name": contract.normalization_name,
            "input_mean": list(contract.input_mean),
            "input_std": list(contract.input_std),
            "frame_output_dim": contract.output_dim,
            "uses_pretrained_weights": contract.uses_pretrained_weights,
            "trainable_head": {
                "hidden_dim": int(common["hidden_dim"]),
                "temporal_encoder_name": str(common["temporal_encoder_name"]),
                "dropout": float(common["dropout"]),
                "behavior_classes": len(VALID_BEHAVIORS),
            },
        }
    pretrained = [reports[name] for name in VISUAL_PRETRAINED_IDS]
    controlled_fields = (
        "pretrained_family",
        "normalization_name",
        "input_mean",
        "input_std",
        "frame_encoder_schedule",
        "trainable_head",
    )
    for field in controlled_fields:
        serialized = {
            json.dumps(row[field], sort_keys=True) for row in pretrained
        }
        if len(serialized) != 1:
            errors.append(f"visual_control_field_drift={field}")
    return {
        "controls": reports,
        "controlled_fields": list(controlled_fields),
        "V0_to_V1_changed_family": "input_resolution_only",
        "V1_to_V2_changed_family": "backbone_capacity_only",
        "tiny_reference_is_controlled_visual_matrix_member": False,
        "errors": errors,
        "valid": not errors,
    }


def _input_hash_audit(
    config: LegacyL5Config,
    paths: dict[str, Path],
) -> dict[str, Any]:
    errors: list[str] = []
    required_paths = {
        "native_units": paths["native_units"],
        "window_folds": paths["window_folds"],
        "image_frames": paths["image_frames"],
        "image_windows": paths["image_windows"],
        "cache_160_manifest": paths["cache_160_manifest"],
        "cache_160_packed_index": paths["cache_160_packed_index"],
        "cache_160_packed_tensor": paths["cache_160_packed_tensor"],
        "feature_contract": Path(
            "configs/classification_v2/legacy_development_input_contract_v1.json"
        ),
        "feature_audit": (
            config.primary_root
            / "12_input_freeze"
            / "legacy_feature_contract_audit.json"
        ),
        "shortcut_audit": (
            config.primary_root
            / "12_input_freeze"
            / "legacy_shortcut_audit.json"
        ),
    }
    rows: dict[str, Any] = {}
    for name, path in required_paths.items():
        exists = path.is_file()
        if not exists:
            errors.append(f"l5_input_missing={name}")
        rows[name] = {
            "path": str(path),
            "sha256": file_sha256(path) if exists else "",
            "size_bytes": int(path.stat().st_size) if exists else -1,
        }
    cache_tensor = paths["cache_160_packed_tensor"]
    if cache_tensor.is_file():
        tensor = np.load(cache_tensor, mmap_mode="r")
        shape = tuple(int(item) for item in tensor.shape)
        expected_rows = int(config.payload["expected_counts"]["image_context_rows"])
        if shape != (expected_rows, 160, 160, 3):
            errors.append(f"l5_cache_160_shape_mismatch={shape}")
        if tensor.dtype != np.uint8:
            errors.append(f"l5_cache_160_dtype_mismatch={tensor.dtype}")
        rows["cache_160_packed_tensor"].update(
            {"shape": list(shape), "dtype": str(tensor.dtype)}
        )
    return {"artifacts": rows, "errors": errors, "valid": not errors}


def _expected_cache_root(config: LegacyL5Config, mode: str) -> Path:
    if mode == "short":
        return config.short_cache_224_root
    if mode == "full":
        return config.full_cache_224_root
    raise ValueError(f"unsupported legacy L5 cache audit mode={mode}")


def _expected_cache_source_paths(
    config: LegacyL5Config,
    mode: str,
) -> dict[str, Path]:
    if mode == "short":
        root = config.short_image_context_root
        return {
            "image_frames": root / "image_frame_context_manifest.csv",
            "image_windows": root / "image_window_context_manifest.csv",
        }
    if mode == "full":
        paths = _legacy_paths(config)
        return {
            "image_frames": paths["image_frames"],
            "image_windows": paths["image_windows"],
        }
    raise ValueError(f"unsupported legacy L5 cache audit mode={mode}")


def _short_cache_reference_audit(
    config: LegacyL5Config,
    candidate_paths: dict[str, Path],
) -> dict[str, Any]:
    reference_root = config.short_cache_224_reference_root
    reference_paths = _cache_artifact_paths(reference_root)
    compared_names = ("packed_index", "packed_tensor")
    errors: list[str] = []
    pairs: dict[str, Any] = {}
    for name in compared_names:
        reference = reference_paths[name]
        candidate = candidate_paths[name]
        if not reference.is_file():
            errors.append(f"short_reference_artifact_missing={name}")
            continue
        reference_hash = file_sha256(reference)
        candidate_hash = file_sha256(candidate)
        identical = reference_hash == candidate_hash
        if not identical:
            errors.append(f"short_reference_hash_mismatch={name}")
        pairs[name] = {
            "reference_path": str(reference),
            "reference_sha256": reference_hash,
            "candidate_path": str(candidate),
            "candidate_sha256": candidate_hash,
            "byte_identical": identical,
        }
    return {
        "reference_cache_root": str(reference_root),
        "candidate_cache_root": str(config.short_cache_224_root),
        "compared_artifacts": list(compared_names),
        "pairs": pairs,
        "errors": errors,
        "valid": not errors,
    }


def audit_legacy_l5_cache(
    config: LegacyL5Config,
    *,
    cache_root: Path,
    mode: str,
    readiness_audit_path: Path,
    short_cache_audit_path: Path | None = None,
) -> dict[str, Any]:
    """Verify a direct-source 224px cache and strict packed-only loading."""

    if mode not in {"short", "full"}:
        raise ValueError(f"unsupported legacy L5 cache audit mode={mode}")
    readiness = _read_json(readiness_audit_path)
    _validate_readiness_parent(config, readiness)
    short_parent: dict[str, Any] | None = None
    if mode == "full":
        if short_cache_audit_path is None:
            raise ValueError("full cache audit requires the short cache audit")
        short_parent = _read_json(short_cache_audit_path)
        _validate_short_cache_parent(config, short_parent)
    errors: list[str] = []
    expected_cache_root = _expected_cache_root(config, mode)
    if cache_root.resolve() != expected_cache_root.resolve():
        errors.append(
            "cache_root_binding_mismatch="
            f"{cache_root}!={expected_cache_root}"
        )
    paths = _cache_artifact_paths(cache_root)
    missing = [name for name, path in paths.items() if not path.is_file()]
    errors.extend(f"cache_artifact_missing={name}" for name in missing)
    if errors:
        return _cache_failure_payload(
            config,
            cache_root=cache_root,
            mode=mode,
            readiness_audit_path=readiness_audit_path,
            short_cache_audit_path=short_cache_audit_path,
            errors=errors,
        )
    cache_audit = _read_json(paths["cache_audit"])
    packed_audit = _read_json(paths["packed_audit"])
    manifest = pd.read_csv(paths["manifest"], low_memory=False)
    index = pd.read_csv(paths["packed_index"], low_memory=False)
    tensor = np.load(paths["packed_tensor"], mmap_mode="r")
    expected_rows = (
        int(config.payload["cache_contract"]["short_context_rows"])
        if mode == "short"
        else int(config.payload["expected_counts"]["image_context_rows"])
    )
    try:
        contract = _cache_contract_audit(
            config,
            cache_root=cache_root,
            mode=mode,
            expected_rows=expected_rows,
            cache_audit=cache_audit,
            packed_audit=packed_audit,
            manifest=manifest,
            index=index,
            tensor=tensor,
            paths=paths,
        )
    finally:
        _close_memmap(tensor)
    errors.extend(contract["errors"])
    short_reference = (
        _short_cache_reference_audit(config, paths)
        if mode == "short"
        else None
    )
    if short_reference is not None:
        errors.extend(short_reference["errors"])
    letterbox = _letterbox_audit(manifest, image_size=224)
    errors.extend(letterbox["errors"])
    packed_equivalence = _packed_equivalence_audit(
        cache_root=cache_root,
        manifest=manifest,
        index=index,
        packed_tensor_path=paths["packed_tensor"],
    )
    errors.extend(packed_equivalence["errors"])
    loader = _strict_cache_loader_audit(
        config,
        cache_root=cache_root,
        paths=paths,
        available_context_ids=set(manifest["image_context_id"].astype(str)),
    )
    errors.extend(loader["errors"])
    valid = not errors
    return {
        "schema_version": L5_CACHE_SCHEMA_VERSION,
        "status": (
            f"PASS_LEGACY_DEVELOPMENT_L5_CACHE_{mode.upper()}"
            if valid
            else f"FAIL_LEGACY_DEVELOPMENT_L5_CACHE_{mode.upper()}"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "mode": mode,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "readiness_audit_path": str(readiness_audit_path),
        "readiness_audit_sha256": file_sha256(readiness_audit_path),
        "short_cache_audit_path": (
            str(short_cache_audit_path)
            if short_cache_audit_path is not None
            else None
        ),
        "short_cache_audit_sha256": (
            file_sha256(short_cache_audit_path)
            if short_cache_audit_path is not None
            else None
        ),
        "implementation_source_sha256": file_sha256(Path(__file__)),
        "git_state": git_state(),
        "cache_root": str(cache_root),
        "expected_cache_root": str(expected_cache_root),
        "cache_artifact_hashes": {
            name: file_sha256(path) for name, path in paths.items()
        },
        "cache_contract_audit": contract,
        "short_reference_equivalence_audit": short_reference,
        "letterbox_audit": letterbox,
        "packed_equivalence_audit": packed_equivalence,
        "strict_cache_loader_audit": loader,
        "source_media_fallback_during_training": False,
        "full_224_cache_build_authorized": mode == "short" and valid,
        "pretrained_feature_cache_authorized": mode == "full" and valid,
        "accuracy_f1_computed": False,
        "optimizer_steps": 0,
        "errors": errors,
        "valid": valid,
    }


def _validate_readiness_parent(
    config: LegacyL5Config,
    readiness: dict[str, Any],
) -> None:
    expected = {
        "schema_version": L5_READINESS_SCHEMA_VERSION,
        "status": "PASS_LEGACY_DEVELOPMENT_L5_READINESS",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "short_224_cache_build_authorized": True,
        "config_sha256": config.sha256,
        "implementation_source_sha256": file_sha256(Path(__file__)),
        "valid": True,
    }
    errors = _mapping_errors(readiness, expected, prefix="readiness_parent")
    if readiness.get("errors"):
        errors.append("readiness_parent_has_errors")
    if errors:
        raise ValueError(f"legacy L5 readiness parent mismatch: {errors}")


def _validate_short_cache_parent(
    config: LegacyL5Config,
    short: dict[str, Any],
) -> None:
    expected = {
        "schema_version": L5_CACHE_SCHEMA_VERSION,
        "status": "PASS_LEGACY_DEVELOPMENT_L5_CACHE_SHORT",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "mode": "short",
        "config_sha256": config.sha256,
        "implementation_source_sha256": file_sha256(Path(__file__)),
        "full_224_cache_build_authorized": True,
        "valid": True,
    }
    errors = _mapping_errors(short, expected, prefix="short_cache_parent")
    if short.get("errors"):
        errors.append("short_cache_parent_has_errors")
    short_root = Path(str(short.get("cache_root", "")))
    if short_root.resolve() != config.short_cache_224_root.resolve():
        errors.append("short_cache_parent_root_mismatch")
    if errors:
        raise ValueError(f"legacy L5 short cache parent mismatch: {errors}")


def _cache_contract_audit(
    config: LegacyL5Config,
    *,
    cache_root: Path,
    mode: str,
    expected_rows: int,
    cache_audit: dict[str, Any],
    packed_audit: dict[str, Any],
    manifest: pd.DataFrame,
    index: pd.DataFrame,
    tensor: np.ndarray,
    paths: dict[str, Path],
) -> dict[str, Any]:
    errors: list[str] = []
    required_manifest = {
        "image_context_id",
        "cache_path",
        "image_size",
        "cache_format",
        "resize_policy",
        "source_crop_width",
        "source_crop_height",
        "source_crop_aspect_ratio",
        "letterbox_resized_width",
        "letterbox_resized_height",
        "letterbox_pad_left",
        "letterbox_pad_top",
        "letterbox_pad_right",
        "letterbox_pad_bottom",
        "lineage_scope",
        "human_review_complete",
    }
    required_index = {
        "image_context_id",
        "packed_row",
        "lineage_scope",
        "human_review_complete",
    }
    missing_manifest = sorted(required_manifest.difference(manifest.columns))
    missing_index = sorted(required_index.difference(index.columns))
    if missing_manifest:
        errors.append(f"cache_manifest_missing_columns={missing_manifest}")
    if missing_index:
        errors.append(f"cache_index_missing_columns={missing_index}")
    source_paths = _expected_cache_source_paths(config, mode)
    expected_cache_root = _expected_cache_root(config, mode)
    if cache_root.resolve() != expected_cache_root.resolve():
        errors.append("cache_root_does_not_match_config")
    observed_frame_source = Path(str(cache_audit.get("frame_context_csv", "")))
    observed_window_source = Path(str(cache_audit.get("window_context_csv", "")))
    if observed_frame_source.resolve() != source_paths["image_frames"].resolve():
        errors.append("cache_frame_source_is_not_original_context")
    if observed_window_source.resolve() != source_paths["image_windows"].resolve():
        errors.append("cache_window_source_is_not_original_context")
    expected_audit = {
        "schema_version": "classification_v2_image_cache_audit_v1",
        "image_size": 224,
        "selected_context_rows": expected_rows,
        "manifest_rows": expected_rows,
        "missing_context_rows": 0,
        "duplicate_context_rows": 0,
        "failed_rows": 0,
        "cache_format": "npy_uint8_rgb_hwc",
        "resize_policy": "letterbox_preserve_aspect_rgb_pad_black_v1",
        "processing_order": "source_media_frame_context_v1",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "valid": True,
    }
    errors.extend(
        _mapping_errors(cache_audit, expected_audit, prefix="cache_builder")
    )
    expected_packed = {
        "schema_version": "classification_v2_packed_image_cache_audit_v1",
        "source_manifest_sha256": file_sha256(paths["manifest"]),
        "shape": [expected_rows, 224, 224, 3],
        "dtype": "uint8",
        "source_rows": expected_rows,
        "packed_rows": expected_rows,
        "index_rows": expected_rows,
        "failed_rows": 0,
        "duplicate_index_ids": 0,
        "verification_mismatches": 0,
        "working_set_release_policy": (
            "flush_close_reopen_each_checkpoint_v1"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "valid": True,
    }
    errors.extend(
        _mapping_errors(packed_audit, expected_packed, prefix="packed_builder")
    )
    if int(packed_audit.get("mapping_reopen_count", 0)) <= 0:
        errors.append("packed_builder_has_no_mapping_reopen_evidence")
    if len(manifest) != expected_rows:
        errors.append(f"cache_manifest_rows={len(manifest)}!={expected_rows}")
    if len(index) != expected_rows:
        errors.append(f"cache_index_rows={len(index)}!={expected_rows}")
    if tuple(tensor.shape) != (expected_rows, 224, 224, 3):
        errors.append(f"cache_tensor_shape={tuple(tensor.shape)}")
    if tensor.dtype != np.uint8:
        errors.append(f"cache_tensor_dtype={tensor.dtype}")
    if not missing_manifest:
        if manifest["image_context_id"].astype(str).duplicated().any():
            errors.append("cache_manifest_duplicate_context_ids")
        if set(manifest["lineage_scope"].astype(str)) != {LINEAGE_SCOPE}:
            errors.append("cache_manifest_lineage_scope_mismatch")
        if _strict_bool(
            manifest["human_review_complete"],
            name="cache manifest human_review_complete",
        ).any():
            errors.append("cache_manifest_claims_human_review")
        path_values = manifest["cache_path"].fillna("").astype(str)
        bad_resolution_paths = int(
            (
                ~path_values.str.contains(
                    "actor_rgb_224_letterbox",
                    regex=False,
                )
            ).sum()
        )
        contains_160 = int(
            path_values.str.contains(
                "actor_rgb_160_letterbox",
                regex=False,
            ).sum()
        )
        if bad_resolution_paths:
            errors.append(
                f"cache_manifest_non_224_paths={bad_resolution_paths}"
            )
        if contains_160:
            errors.append(f"cache_manifest_160_path_references={contains_160}")
    direct_source_proof = {
        "frame_context_path": str(observed_frame_source),
        "frame_context_sha256": file_sha256(source_paths["image_frames"]),
        "window_context_path": str(observed_window_source),
        "window_context_sha256": file_sha256(source_paths["image_windows"]),
        "processing_order": cache_audit.get("processing_order"),
        "video_decode_count": int(cache_audit.get("video_decode_count", 0)),
        "video_seek_count": int(cache_audit.get("video_seek_count", 0)),
        "upscaled_from_160": False,
        "video_capture_audit": cache_audit.get("video_capture_audit"),
    }
    if direct_source_proof["video_decode_count"] <= 0:
        errors.append("cache_builder_has_no_direct_video_decode_evidence")
    if mode in {"short", "full"}:
        capture = cache_audit.get("video_capture_audit")
        if not isinstance(capture, dict):
            errors.append("cache_builder_missing_video_capture_audit")
        else:
            expected_capture = {
                "video_capture_cache_size": 1,
                "active_video_captures": 0,
                "peak_open_video_captures": 1,
            }
            errors.extend(
                _mapping_errors(
                    capture,
                    expected_capture,
                    prefix="cache_builder_video_capture",
                )
            )
            if int(capture.get("video_capture_open_count", 0)) <= 0:
                errors.append("cache_builder_has_no_video_capture_open_evidence")
    return {
        "mode": mode,
        "expected_rows": expected_rows,
        "manifest_rows": int(len(manifest)),
        "index_rows": int(len(index)),
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
        "direct_source_proof": direct_source_proof,
        "cache_builder_audit": cache_audit,
        "packed_builder_audit": packed_audit,
        "errors": errors,
        "valid": not errors,
    }


def _letterbox_audit(
    manifest: pd.DataFrame,
    *,
    image_size: int,
) -> dict[str, Any]:
    numeric_columns = (
        "source_crop_width",
        "source_crop_height",
        "source_crop_aspect_ratio",
        "letterbox_resized_width",
        "letterbox_resized_height",
        "letterbox_pad_left",
        "letterbox_pad_top",
        "letterbox_pad_right",
        "letterbox_pad_bottom",
    )
    values = {
        name: pd.to_numeric(manifest[name], errors="coerce")
        for name in numeric_columns
    }
    invalid = pd.Series(False, index=manifest.index)
    invalid |= values["source_crop_width"].le(0)
    invalid |= values["source_crop_height"].le(0)
    invalid |= values["source_crop_aspect_ratio"].le(0)
    for name in numeric_columns:
        invalid |= values[name].isna()
    invalid |= (
        values["letterbox_resized_width"]
        + values["letterbox_pad_left"]
        + values["letterbox_pad_right"]
    ).ne(image_size)
    invalid |= (
        values["letterbox_resized_height"]
        + values["letterbox_pad_top"]
        + values["letterbox_pad_bottom"]
    ).ne(image_size)
    invalid |= values["letterbox_pad_left"].lt(0)
    invalid |= values["letterbox_pad_top"].lt(0)
    invalid |= values["letterbox_pad_right"].lt(0)
    invalid |= values["letterbox_pad_bottom"].lt(0)
    aspect = values["source_crop_width"] / values["source_crop_height"]
    aspect_error = (aspect - values["source_crop_aspect_ratio"]).abs()
    invalid |= aspect_error.gt(1e-9)
    padded = (
        values["letterbox_pad_left"]
        + values["letterbox_pad_top"]
        + values["letterbox_pad_right"]
        + values["letterbox_pad_bottom"]
    ).gt(0)
    invalid_rows = int(invalid.sum())
    errors = [] if invalid_rows == 0 else [f"letterbox_invalid_rows={invalid_rows}"]
    return {
        "rows": int(len(manifest)),
        "image_size": image_size,
        "non_square_source_rows": int(
            values["source_crop_width"].ne(values["source_crop_height"]).sum()
        ),
        "padded_canvas_rows": int(padded.sum()),
        "maximum_source_aspect_error": float(aspect_error.max()),
        "invalid_rows": invalid_rows,
        "errors": errors,
        "valid": not errors,
    }


def _packed_equivalence_audit(
    *,
    cache_root: Path,
    manifest: pd.DataFrame,
    index: pd.DataFrame,
    packed_tensor_path: Path,
    reopen_every_rows: int = PACKED_AUDIT_REOPEN_ROWS,
) -> dict[str, Any]:
    if reopen_every_rows <= 0:
        raise ValueError("reopen_every_rows must be positive")
    errors: list[str] = []
    ordered_manifest = manifest.sort_values(
        "image_context_id",
        kind="mergesort",
    ).reset_index(drop=True)
    ordered_index = index.sort_values(
        "packed_row",
        kind="mergesort",
    ).reset_index(drop=True)
    manifest_ids = ordered_manifest["image_context_id"].astype(str)
    index_ids = ordered_index["image_context_id"].astype(str)
    ordered_ids_match = bool(manifest_ids.equals(index_ids))
    if not ordered_ids_match:
        errors.append("packed_index_manifest_order_mismatch")
    packed_rows = pd.to_numeric(ordered_index["packed_row"], errors="coerce")
    expected_rows = pd.Series(np.arange(len(ordered_index)), dtype=np.int64)
    contiguous = bool(
        not packed_rows.isna().any()
        and np.array_equal(
            packed_rows.to_numpy(dtype=np.int64),
            expected_rows.to_numpy(),
        )
    )
    if not contiguous:
        errors.append("packed_index_rows_not_contiguous")
    mismatches = 0
    missing_files = 0
    shape_mismatches = 0
    dtype_mismatches = 0
    mapping_open_count = 0
    tensor: np.ndarray | None = None
    if ordered_ids_match and contiguous:
        try:
            for row_index, row in enumerate(
                ordered_manifest.itertuples(index=False)
            ):
                if row_index % reopen_every_rows == 0:
                    if tensor is not None:
                        _close_memmap(tensor)
                    tensor = np.load(packed_tensor_path, mmap_mode="r")
                    mapping_open_count += 1
                cache_path = Path(str(row.cache_path))
                if not cache_path.is_absolute():
                    cache_path = cache_root / cache_path
                if not cache_path.is_file():
                    missing_files += 1
                    continue
                image = np.load(cache_path)
                if image.shape != (224, 224, 3):
                    shape_mismatches += 1
                    continue
                if image.dtype != np.uint8:
                    dtype_mismatches += 1
                    continue
                if tensor is None:
                    raise AssertionError("packed tensor mapping was not opened")
                if not np.array_equal(image, np.asarray(tensor[row_index])):
                    mismatches += 1
        finally:
            if tensor is not None:
                _close_memmap(tensor)
    if missing_files:
        errors.append(f"cache_source_tensor_files_missing={missing_files}")
    if shape_mismatches:
        errors.append(f"cache_source_tensor_shape_mismatches={shape_mismatches}")
    if dtype_mismatches:
        errors.append(f"cache_source_tensor_dtype_mismatches={dtype_mismatches}")
    if mismatches:
        errors.append(f"cache_packed_pixel_mismatches={mismatches}")
    return {
        "verified_rows": int(len(ordered_manifest)),
        "ordered_context_ids_match": ordered_ids_match,
        "packed_rows_contiguous": contiguous,
        "missing_source_tensor_files": missing_files,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "pixel_mismatches": mismatches,
        "mapping_reopen_interval_rows": reopen_every_rows,
        "mapping_open_count": mapping_open_count,
        "errors": errors,
        "valid": not errors,
    }


def _strict_cache_loader_audit(
    config: LegacyL5Config,
    *,
    cache_root: Path,
    paths: dict[str, Path],
    available_context_ids: set[str],
) -> dict[str, Any]:
    source_paths = _legacy_paths(config)
    dataset = ClassificationV2ImageSequenceDataset(
        ImageSequenceDatasetConfig(
            frame_context_csv=source_paths["image_frames"],
            window_context_csv=source_paths["image_windows"],
            image_cache_manifest_csv=paths["manifest"],
            packed_image_cache_npy=paths["packed_tensor"],
            packed_image_cache_index_csv=paths["packed_index"],
            image_size=224,
            require_complete=True,
            require_cached_images=True,
            image_cache_size=0,
        )
    )
    errors: list[str] = []
    selected_indices: list[int] = []
    try:
        for index, sequence in enumerate(
            dataset.windows["image_context_id_sequence"].fillna("").astype(str)
        ):
            context_ids = [
                value.strip() for value in sequence.split(";;") if value.strip()
            ]
            if context_ids and set(context_ids).issubset(available_context_ids):
                selected_indices.append(index)
                if len(selected_indices) >= 8:
                    break
        if not selected_indices:
            errors.append("strict_cache_loader_has_no_complete_short_window")
        loaded_slots = 0
        for index in selected_indices:
            item = dataset[index]
            if item["errors"]:
                errors.append(
                    "strict_cache_loader_item_errors="
                    f"{item['window_id']}:{item['errors']}"
                )
            if tuple(item["image"].shape[1:]) != (3, 224, 224):
                errors.append(
                    f"strict_cache_loader_shape={tuple(item['image'].shape)}"
                )
            loaded_slots += int(item["length_mask"].sum().item())
        load_audit = dataset.image_load_audit()
        video_decode_count = int(dataset.video_decode_count)
        video_seek_count = int(dataset.video_seek_count)
    finally:
        dataset.close()
    if load_audit["packed_image_cache_hits"] != loaded_slots:
        errors.append("strict_cache_loader_packed_hit_count_mismatch")
    if load_audit["disk_image_cache_misses"] != 0:
        errors.append("strict_cache_loader_cache_miss")
    if load_audit["source_image_loads"] != 0:
        errors.append("strict_cache_loader_source_media_fallback")
    if video_decode_count != 0 or video_seek_count != 0:
        errors.append("strict_cache_loader_video_io")
    return {
        "cache_root": str(cache_root),
        "sample_window_rows": len(selected_indices),
        "loaded_image_slots": loaded_slots,
        **load_audit,
        "video_decode_count": video_decode_count,
        "video_seek_count": video_seek_count,
        "errors": errors,
        "valid": not errors,
    }


def _cache_failure_payload(
    config: LegacyL5Config,
    *,
    cache_root: Path,
    mode: str,
    readiness_audit_path: Path,
    short_cache_audit_path: Path | None,
    errors: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": L5_CACHE_SCHEMA_VERSION,
        "status": f"FAIL_LEGACY_DEVELOPMENT_L5_CACHE_{mode.upper()}",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "mode": mode,
        "config_sha256": config.sha256,
        "readiness_audit_path": str(readiness_audit_path),
        "short_cache_audit_path": (
            str(short_cache_audit_path)
            if short_cache_audit_path is not None
            else None
        ),
        "cache_root": str(cache_root),
        "expected_cache_root": str(_expected_cache_root(config, mode)),
        "full_224_cache_build_authorized": False,
        "pretrained_feature_cache_authorized": False,
        "errors": errors,
        "valid": False,
    }


def _cache_artifact_paths(cache_root: Path) -> dict[str, Path]:
    return {
        "cache_audit": cache_root / "cache_audit.json",
        "manifest": cache_root / "manifest.csv",
        "packed_audit": cache_root / "packed_image_cache_audit.json",
        "packed_index": cache_root / "packed_image_cache_index.csv",
        "packed_tensor": cache_root / "packed_rgb_224_letterbox.npy",
    }


def _legacy_paths(config: LegacyL5Config) -> dict[str, Path]:
    root = config.primary_root
    return {
        "native_units": (
            root
            / "06_temporal_tier_contract"
            / "native_temporal_unit_manifest.csv"
        ),
        "window_folds": root / "11_folds" / "window_oof_fold_manifest.csv",
        "image_frames": (
            root / "09_image_context" / "image_frame_context_manifest.csv"
        ),
        "image_windows": (
            root / "09_image_context" / "image_window_context_manifest.csv"
        ),
        "cache_160_manifest": root / "10_actor_cache_160" / "manifest.csv",
        "cache_160_packed_index": (
            root / "10_actor_cache_160" / "packed_image_cache_index.csv"
        ),
        "cache_160_packed_tensor": (
            root / "10_actor_cache_160" / "packed_rgb_160_letterbox.npy"
        ),
    }


def _view_name(length: int, protocol: str) -> str:
    suffix = {
        "all_sliding_event_balanced": "all_sliding",
        "one_centered_window_matched": "centered_matched",
    }[protocol]
    return f"legacy_t{length}_{suffix}_observed_time"


def _close_memmap(array: np.ndarray) -> None:
    """Close a NumPy mmap without retaining all audited pages in memory."""

    mmap_handle = getattr(array, "_mmap", None)
    if mmap_handle is not None:
        mmap_handle.close()


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


def _ordered_id_sha256(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values.astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _csv_row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def _strict_bool(series: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "t"}
    falsy = {"false", "0", "no", "n", "f"}
    valid = normalized.isin(truthy | falsy)
    if not valid.all():
        raise ValueError(f"invalid boolean values in {name}")
    return normalized.isin(truthy)


def _mapping_errors(
    observed: dict[str, Any],
    expected: dict[str, Any],
    *,
    prefix: str,
) -> list[str]:
    return [
        f"{prefix}_mismatch={name}:{observed.get(name)!r}!={value!r}"
        for name, value in expected.items()
        if observed.get(name) != value
    ]


def _require_mapping_equal(
    observed: dict[str, Any],
    expected: dict[str, Any],
    *,
    name: str,
) -> None:
    errors = _mapping_errors(observed, expected, prefix=name)
    if errors:
        raise ValueError(f"{name} contract mismatch: {errors}")


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
    "L5_CACHE_SCHEMA_VERSION",
    "L5_CONFIG_SCHEMA_VERSION",
    "L5_READINESS_SCHEMA_VERSION",
    "LINEAGE_SCOPE",
    "LegacyL5Config",
    "audit_legacy_l5_cache",
    "audit_legacy_l5_readiness",
    "load_legacy_l5_config",
]
