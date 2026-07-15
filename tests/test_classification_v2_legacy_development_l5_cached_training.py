from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training import (
    legacy_development_l5_cached_training as cached_training,
)
from pig_behavior.classification_v2.training.legacy_development_l5_cached_data import (
    GPU_ALLOCATOR_LIMIT_BYTES,
    LegacyL5CachedFeatureView,
)
from pig_behavior.classification_v2.training.legacy_development_l5_cached_training import (
    audit_legacy_l5_cached_training_repeat_gate,
    build_legacy_l5_cached_short_selection,
    compute_legacy_l5_native_metrics,
    load_legacy_l5_cached_training_config,
    train_legacy_l5_cached_short_core,
)

CONFIG_PATH = Path(
    "configs/classification_v2/legacy_development_l5_cached_training_v1.json"
)
SHORT_V2_CONFIG_PATH = Path(
    "configs/classification_v2/"
    "legacy_development_l5_cached_training_short_v2.json"
)
V1_SHORT_V3_CONFIG_PATH = Path(
    "configs/classification_v2/"
    "legacy_development_l5_cached_training_v1_t16_short_v3.json"
)
V2_SHORT_V4_CONFIG_PATH = Path(
    "configs/classification_v2/"
    "legacy_development_l5_cached_training_v2_t16_short_v4.json"
)
VALIDATION_COUNTS = {
    "drink": 20,
    "eat": 17,
    "fight": 2,
    "social-nose": 9,
    "explore": 45,
    "lying": 62,
    "stand": 11,
    "move": 8,
    "sitting": 70,
    "playwithtoy": 1,
}


def _synthetic_view(
    tmp_path: Path,
    *,
    with_features: bool = False,
    train_counts: dict[str, int] | None = None,
) -> LegacyL5CachedFeatureView:
    records: list[dict[str, object]] = []
    targets: list[int] = []
    item_index = 0
    train_counts = train_counts or {
        behavior: 10 for behavior in VALID_BEHAVIORS
    }
    for behavior_index, behavior in enumerate(VALID_BEHAVIORS):
        for _ in range(train_counts[behavior]):
            records.append(
                _window_record(
                    item_index,
                    role="train",
                    behavior=behavior,
                )
            )
            targets.append(behavior_index)
            item_index += 1
    for behavior_index, behavior in enumerate(VALID_BEHAVIORS):
        for _ in range(VALIDATION_COUNTS[behavior]):
            records.append(
                _window_record(
                    item_index,
                    role="validation",
                    behavior=behavior,
                )
            )
            targets.append(behavior_index)
            item_index += 1
    windows = pd.DataFrame.from_records(records)
    rows = len(windows)
    feature_rows = np.arange(rows * 16, dtype=np.int64).reshape(rows, 16)
    feature_path = tmp_path / "frame_features_f32.npy"
    if with_features:
        features = np.zeros((rows * 16, 512), dtype=np.float32)
        for position, target in enumerate(targets):
            features[feature_rows[position], target] = 4.0
        np.save(feature_path, features, allow_pickle=False)
    return LegacyL5CachedFeatureView(
        feature_tensor_path=feature_path,
        feature_tensor_sha256="1" * 64,
        control_id="V0",
        temporal_view_name="legacy_t16_centered_matched_observed_time",
        sequence_length=16,
        windows=windows,
        fold_manifest=windows[
            [
                "temporal_unit_key",
                "recording_group_id",
                "video_key",
                "source_type",
                "dataset_id",
                "behavior_label",
                "l5_role",
            ]
        ].copy(),
        feature_rows=feature_rows,
        observed_mask=np.ones((rows, 16), dtype=np.bool_),
        time_delta=np.zeros((rows, 16), dtype=np.float32),
        targets=np.asarray(targets, dtype=np.int64),
        sample_weights=np.ones(rows, dtype=np.float32),
        audit={},
    )


def _window_record(
    item_index: int,
    *,
    role: str,
    behavior: str,
) -> dict[str, object]:
    return {
        "window_id": f"window_{item_index:04d}",
        "temporal_unit_key": f"unit_{item_index:04d}",
        "recording_group_id": f"recording_{item_index:04d}",
        "video_key": f"video_{item_index:04d}",
        "source_type": "legacy_recovered",
        "dataset_id": "legacy_fixture",
        "behavior_label": behavior,
        "oof_fold_id": "fold_train" if role == "train" else "fold_validation",
        "l5_role": role,
    }


def test_cached_training_config_freezes_four_gib_short_semantics() -> None:
    config = load_legacy_l5_cached_training_config(CONFIG_PATH)

    assert config.payload["data"]["expected_train_native_units"] == 80
    assert config.payload["data"]["expected_validation_native_units"] == 245
    assert config.payload["optimization"]["maximum_optimizer_steps"] == 9
    assert config.payload["optimization"]["maximum_loaded_batch_bytes"] == (
        2_103_552
    )
    assert config.payload["optimization"]["allocator_limit_bytes"] == (
        GPU_ALLOCATOR_LIMIT_BYTES
    )
    assert config.payload["optimization"]["autocast_enabled"] is False
    assert config.payload["optimization"]["oom_retry_allowed"] is False
    assert config.payload["outer_holdout_predictions_authorized"] is False
    assert config.payload["execution_guard"]["require_fresh_process"] is True
    assert config.payload["execution_guard"][
        "clear_cublas_workspaces_after_training"
    ] is True


def test_cached_training_v2_short_freezes_expansion_contract(
    tmp_path: Path,
) -> None:
    config = load_legacy_l5_cached_training_config(SHORT_V2_CONFIG_PATH)
    view = _synthetic_view(tmp_path)
    selection = build_legacy_l5_cached_short_selection(view, config)

    assert config.training_scope == cached_training.SHORT_TRAINING_SCOPE
    assert config.payload["expansion_contract"][
        "full_expected_train_native_units"
    ] == 3_652
    assert config.payload["expansion_contract"][
        "full_maximum_optimizer_steps"
    ] == 345
    assert selection.audit["training_scope"] == config.training_scope
    assert tuple(selection.manifest.columns) == cached_training.SELECTION_FIELDS_V2
    assert selection.manifest["training_scope"].eq(config.training_scope).all()


def test_cached_training_v3_freezes_v1_resolution_only(
    tmp_path: Path,
) -> None:
    path = _write_v1_short_config(tmp_path)
    config = load_legacy_l5_cached_training_config(path)
    v1_view = replace(_synthetic_view(tmp_path / "v1"), control_id="V1")
    v1_selection = build_legacy_l5_cached_short_selection(v1_view, config)
    v0_config = load_legacy_l5_cached_training_config(CONFIG_PATH)
    v0_selection = build_legacy_l5_cached_short_selection(
        replace(v1_view, control_id="V0"),
        v0_config,
    )

    assert config.payload["ablation_contract"]["changed_variable"] == (
        "input_resolution"
    )
    assert config.payload["ablation_contract"]["reference_value"] == 160
    assert config.payload["ablation_contract"]["candidate_value"] == 224
    assert config.payload["data"]["control_id"] == "V1"
    assert config.payload["expansion_contract"][
        "frozen_semantic_family"
    ] == cached_training.V1_RESOLUTION_FROZEN_FAMILY
    assert v1_selection.manifest["temporal_unit_key"].tolist() == (
        v0_selection.manifest["temporal_unit_key"].tolist()
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["ablation_contract"]["candidate_value"] = 225
    drifted = path.with_name("v1_resolution_drift.json")
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="V1 resolution ablation drift"):
        load_legacy_l5_cached_training_config(drifted)


def test_cached_training_v4_freezes_v2_backbone_only(
    tmp_path: Path,
) -> None:
    path = _write_v2_short_config(tmp_path)
    config = load_legacy_l5_cached_training_config(path)
    v2_view = replace(_synthetic_view(tmp_path / "v2"), control_id="V2")
    v2_selection = build_legacy_l5_cached_short_selection(v2_view, config)
    v1_config = load_legacy_l5_cached_training_config(
        _write_v1_short_config(tmp_path / "v1_contract")
    )
    v1_selection = build_legacy_l5_cached_short_selection(
        replace(v2_view, control_id="V1"),
        v1_config,
    )

    contract = config.payload["ablation_contract"]
    assert contract["changed_variable"] == "backbone_name"
    assert contract["reference_value"] == "resnet18"
    assert contract["candidate_value"] == "resnet34"
    assert contract["fixed_input_resolution"] == 224
    assert contract["fixed_pretrained_weight_family"] == "ImageNet-1K V1"
    assert config.payload["data"]["control_id"] == "V2"
    assert config.payload["expansion_contract"][
        "frozen_semantic_family"
    ] == cached_training.V2_BACKBONE_FROZEN_FAMILY
    assert v2_selection.manifest["temporal_unit_key"].tolist() == (
        v1_selection.manifest["temporal_unit_key"].tolist()
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["ablation_contract"]["fixed_input_resolution"] = 225
    drifted = path.with_name("v2_backbone_drift.json")
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="V2 backbone ablation drift"):
        load_legacy_l5_cached_training_config(drifted)


def test_cached_training_config_rejects_allocator_drift(tmp_path: Path) -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["optimization"]["allocator_limit_bytes"] += 1
    path = tmp_path / "drifted_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cached training optimization drift"):
        load_legacy_l5_cached_training_config(path)


def test_cached_training_config_rejects_loaded_batch_drift(
    tmp_path: Path,
) -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["optimization"]["maximum_loaded_batch_bytes"] += 1
    path = tmp_path / "drifted_batch_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cached training optimization drift"):
        load_legacy_l5_cached_training_config(path)


def test_cached_training_config_rejects_fresh_process_weakening(
    tmp_path: Path,
) -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["execution_guard"]["require_fresh_process"] = False
    path = tmp_path / "weakened_process_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fresh-process guard is disabled"):
        load_legacy_l5_cached_training_config(path)


def test_cached_training_config_rejects_cublas_cleanup_weakening(
    tmp_path: Path,
) -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["execution_guard"][
        "clear_cublas_workspaces_after_training"
    ] = False
    path = tmp_path / "weakened_cublas_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cuBLAS cleanup guard is disabled"):
        load_legacy_l5_cached_training_config(path)


def test_cached_short_selection_is_deterministic_and_native_safe(
    tmp_path: Path,
) -> None:
    config = load_legacy_l5_cached_training_config(CONFIG_PATH)
    view = _synthetic_view(tmp_path)
    first = build_legacy_l5_cached_short_selection(view, config)
    second = build_legacy_l5_cached_short_selection(view, config)

    assert len(first.train_positions) == 80
    assert len(first.validation_positions) == 245
    assert first.audit["train_class_counts"] == {
        behavior: 8 for behavior in VALID_BEHAVIORS
    }
    assert first.audit["validation_class_counts"] == VALIDATION_COUNTS
    assert first.audit["outer_holdout_rows"] == 0
    assert first.audit["selection_content_sha256"] == second.audit[
        "selection_content_sha256"
    ]
    pd.testing.assert_frame_equal(first.manifest, second.manifest)


def test_cached_short_selection_rejects_recording_overlap(
    tmp_path: Path,
) -> None:
    config = load_legacy_l5_cached_training_config(CONFIG_PATH)
    view = _synthetic_view(tmp_path)
    windows = view.windows.copy()
    train_group = windows.loc[
        windows["l5_role"].eq("train"),
        "recording_group_id",
    ].iloc[0]
    validation_index = windows.index[windows["l5_role"].eq("validation")][0]
    windows.loc[validation_index, "recording_group_id"] = train_group
    bad_view = replace(view, windows=windows)

    with pytest.raises(ValueError, match="selection group overlap"):
        build_legacy_l5_cached_short_selection(bad_view, config)


def test_native_metrics_use_global_ten_class_order() -> None:
    targets = np.arange(len(VALID_BEHAVIORS), dtype=np.int64)
    probabilities = np.eye(len(VALID_BEHAVIORS), dtype=np.float64)
    keys = pd.Series([f"unit_{index}" for index in targets])

    metrics, per_class, confusion = compute_legacy_l5_native_metrics(
        probabilities,
        targets,
        keys,
    )

    assert metrics["macro_f1_global_10_class"] == 1.0
    assert metrics["macro_f1_supported_classes"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["supported_class_count"] == 10
    assert per_class["f1"].eq(1.0).all()
    np.testing.assert_array_equal(
        confusion[list(VALID_BEHAVIORS)].to_numpy(),
        np.eye(len(VALID_BEHAVIORS), dtype=np.int64),
    )


def test_native_metrics_reject_duplicate_units() -> None:
    probabilities = np.full((2, len(VALID_BEHAVIORS)), 0.1)
    targets = np.asarray([0, 1], dtype=np.int64)

    with pytest.raises(ValueError, match="duplicate temporal units"):
        compute_legacy_l5_native_metrics(
            probabilities,
            targets,
            pd.Series(["same", "same"]),
        )


def test_cached_short_training_core_is_deterministic_and_mmap_bounded(
    tmp_path: Path,
) -> None:
    config = load_legacy_l5_cached_training_config(CONFIG_PATH)
    payload = json.loads(json.dumps(config.payload))
    payload["model"]["hidden_dim"] = 8
    payload["optimization"]["epochs"] = 2
    payload["optimization"]["batch_size"] = 40
    payload["optimization"]["maximum_optimizer_steps"] = 4
    test_config = replace(config, payload=payload)
    view = _synthetic_view(tmp_path, with_features=True)
    selection = build_legacy_l5_cached_short_selection(view, config)
    cuda_initialized_before = torch.cuda.is_initialized()

    first = train_legacy_l5_cached_short_core(
        view,
        selection,
        test_config,
        device="cpu",
    )
    second = train_legacy_l5_cached_short_core(
        view,
        selection,
        test_config,
        device="cpu",
    )

    assert first.optimizer_steps == 4
    assert first.maximum_loaded_batch_bytes == 2_103_552
    assert len(first.predictions) == 245
    assert first.parameter_sha256 == second.parameter_sha256
    assert first.prediction_sha256 == second.prediction_sha256
    assert first.epoch_metrics_sha256 == second.epoch_metrics_sha256
    pd.testing.assert_frame_equal(first.epoch_metrics, second.epoch_metrics)
    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    assert np.isfinite(first.metrics["macro_f1_global_10_class"])
    assert torch.cuda.is_initialized() is cuda_initialized_before


def test_cached_training_v2_metrics_preserve_scope(tmp_path: Path) -> None:
    config = load_legacy_l5_cached_training_config(SHORT_V2_CONFIG_PATH)
    payload = json.loads(json.dumps(config.payload))
    payload["model"]["hidden_dim"] = 8
    payload["optimization"]["epochs"] = 1
    payload["optimization"]["batch_size"] = 32
    payload["optimization"]["maximum_optimizer_steps"] = 3
    test_config = replace(config, payload=payload)
    view = _synthetic_view(tmp_path, with_features=True)
    selection = build_legacy_l5_cached_short_selection(view, config)

    outcome = train_legacy_l5_cached_short_core(
        view,
        selection,
        test_config,
        device="cpu",
    )

    assert outcome.metrics["training_scope"] == config.training_scope
    assert outcome.predictions["training_scope"].eq(config.training_scope).all()
    assert outcome.epoch_metrics["training_scope"].eq(config.training_scope).all()
    assert outcome.per_class_metrics["lineage_scope"].eq(
        cached_training.LINEAGE_SCOPE
    ).all()
    assert outcome.confusion["q2_claim_allowed"].eq(False).all()


def test_cached_short_repeat_gate_validates_two_immutable_packets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_legacy_l5_cached_training_config(CONFIG_PATH)
    payload = json.loads(json.dumps(config.payload))
    payload["model"]["hidden_dim"] = 8
    payload["optimization"]["epochs"] = 1
    payload["optimization"]["batch_size"] = 32
    payload["optimization"]["maximum_optimizer_steps"] = 3
    test_config = replace(config, payload=payload)
    view = _synthetic_view(tmp_path, with_features=True)
    selection = build_legacy_l5_cached_short_selection(view, config)
    outcome = train_legacy_l5_cached_short_core(
        view,
        selection,
        test_config,
        device="cpu",
    )
    primary = _write_synthetic_packet(
        tmp_path / "primary",
        config=test_config,
        selection=selection,
        outcome=outcome,
        run_id="primary",
        process_id=101,
    )
    repeat = _write_synthetic_packet(
        tmp_path / "repeat",
        config=test_config,
        selection=selection,
        outcome=outcome,
        run_id="repeat",
        process_id=202,
    )
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)

    audit = audit_legacy_l5_cached_training_repeat_gate(
        test_config,
        primary_result_path=primary,
        repeat_result_path=repeat,
    )

    assert audit["valid"] is True
    assert audit["exact_full_v0_t16_centered_expansion_authorized"] is True
    assert audit["other_visual_or_temporal_controls_authorized"] is False
    assert all(audit["equality"].values())


def test_cached_v1_repeat_gate_authorizes_only_exact_v1_full(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_legacy_l5_cached_training_config(
        _write_v1_short_config(tmp_path / "contract")
    )
    payload = json.loads(json.dumps(config.payload))
    payload["model"]["hidden_dim"] = 8
    payload["optimization"]["epochs"] = 1
    payload["optimization"]["batch_size"] = 32
    payload["optimization"]["maximum_optimizer_steps"] = 3
    test_config = replace(config, payload=payload)
    view_root = tmp_path / "view"
    view_root.mkdir()
    view = replace(
        _synthetic_view(view_root, with_features=True),
        control_id="V1",
    )
    selection = build_legacy_l5_cached_short_selection(view, config)
    outcome = train_legacy_l5_cached_short_core(
        view,
        selection,
        test_config,
        device="cpu",
    )
    primary = _write_synthetic_packet(
        tmp_path / "v1_primary",
        config=test_config,
        selection=selection,
        outcome=outcome,
        run_id="v1_primary",
        process_id=303,
    )
    repeat = _write_synthetic_packet(
        tmp_path / "v1_repeat",
        config=test_config,
        selection=selection,
        outcome=outcome,
        run_id="v1_repeat",
        process_id=404,
    )
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)

    audit = audit_legacy_l5_cached_training_repeat_gate(
        test_config,
        primary_result_path=primary,
        repeat_result_path=repeat,
    )

    assert audit["valid"] is True
    assert audit["authorized_control_id"] == "V1"
    assert audit["authorized_frozen_semantic_family"] == (
        cached_training.V1_RESOLUTION_FROZEN_FAMILY
    )
    assert audit["exact_full_control_expansion_authorized"] is True
    assert "exact_full_v0_t16_centered_expansion_authorized" not in audit
    assert audit["other_visual_or_temporal_controls_authorized"] is False


def test_cached_v2_repeat_gate_authorizes_only_exact_v2_full(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_legacy_l5_cached_training_config(
        _write_v2_short_config(tmp_path / "contract")
    )
    payload = json.loads(json.dumps(config.payload))
    payload["model"]["hidden_dim"] = 8
    payload["optimization"]["epochs"] = 1
    payload["optimization"]["batch_size"] = 32
    payload["optimization"]["maximum_optimizer_steps"] = 3
    test_config = replace(config, payload=payload)
    view_root = tmp_path / "view"
    view_root.mkdir()
    view = replace(
        _synthetic_view(view_root, with_features=True),
        control_id="V2",
    )
    selection = build_legacy_l5_cached_short_selection(view, config)
    outcome = train_legacy_l5_cached_short_core(
        view,
        selection,
        test_config,
        device="cpu",
    )
    primary = _write_synthetic_packet(
        tmp_path / "v2_primary",
        config=test_config,
        selection=selection,
        outcome=outcome,
        run_id="v2_primary",
        process_id=505,
    )
    repeat = _write_synthetic_packet(
        tmp_path / "v2_repeat",
        config=test_config,
        selection=selection,
        outcome=outcome,
        run_id="v2_repeat",
        process_id=606,
    )
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)

    audit = audit_legacy_l5_cached_training_repeat_gate(
        test_config,
        primary_result_path=primary,
        repeat_result_path=repeat,
    )

    assert audit["valid"] is True
    assert audit["authorized_control_id"] == "V2"
    assert audit["authorized_frozen_semantic_family"] == (
        cached_training.V2_BACKBONE_FROZEN_FAMILY
    )
    assert audit["exact_full_control_expansion_authorized"] is True
    assert "exact_full_v0_t16_centered_expansion_authorized" not in audit
    assert audit["other_visual_or_temporal_controls_authorized"] is False


def test_cached_training_v2_full_requires_bound_short_gate(
    tmp_path: Path,
) -> None:
    full_path = _write_authorized_v2_full_config(tmp_path)
    config = load_legacy_l5_cached_training_config(full_path)
    train_counts = {
        behavior: 366 if index < 2 else 365
        for index, behavior in enumerate(VALID_BEHAVIORS)
    }
    view = _synthetic_view(tmp_path / "view", train_counts=train_counts)
    selection = build_legacy_l5_cached_short_selection(view, config)

    assert config.training_scope == cached_training.FULL_TRAINING_SCOPE
    assert len(selection.train_positions) == 3_652
    assert len(selection.validation_positions) == 245
    assert selection.audit["train_class_counts"] == train_counts
    assert selection.audit["training_scope"] == config.training_scope
    assert selection.manifest["training_scope"].eq(config.training_scope).all()

    payload = json.loads(full_path.read_text(encoding="utf-8"))
    payload["short_gate_parent"]["gate_sha256"] = "0" * 64
    drifted = full_path.with_name("full_gate_drift.json")
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="short gate parent hash drift"):
        load_legacy_l5_cached_training_config(drifted)


def _write_v1_short_config(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    config_dir = repo / "configs" / "classification_v2"
    reference_dir = repo / "references" / "v0_full"
    config_dir.mkdir(parents=True)
    reference_dir.mkdir(parents=True)
    payload = json.loads(V1_SHORT_V3_CONFIG_PATH.read_text(encoding="utf-8"))

    base_path = config_dir / "legacy_development_l5_v1.json"
    base_payload = json.loads(
        Path("configs/classification_v2/legacy_development_l5_v1.json")
        .read_text(encoding="utf-8")
    )
    base_payload["development_root"] = str(repo / "outputs" / "legacy")
    base_path.write_text(json.dumps(base_payload), encoding="utf-8")
    payload["base_config"]["sha256"] = cached_training.file_sha256(base_path)

    reference_config_path = reference_dir / "config.json"
    reference_config = {
        "schema_version": (
            cached_training.CACHED_TRAINING_CONFIG_SCHEMA_VERSION_V2
        ),
        "training_scope": cached_training.FULL_TRAINING_SCOPE,
        "data": {
            "control_id": "V0",
            "temporal_view_name": (
                "legacy_t16_centered_matched_observed_time"
            ),
            "sampling_protocol": "one_centered_window_matched",
            "sequence_length": 16,
        },
        "base_config": payload["base_config"],
        "model": payload["model"],
        "optimization": {
            **payload["optimization"],
            "maximum_optimizer_steps": 345,
        },
    }
    reference_config_path.write_text(
        json.dumps(reference_config),
        encoding="utf-8",
    )
    reference_config_sha = cached_training.file_sha256(reference_config_path)

    reference_result_path = reference_dir / "run_result.json"
    reference_result = {
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L5_CACHED_FULL_DEVELOPMENT_TRAINING"
        ),
        "training_scope": cached_training.FULL_TRAINING_SCOPE,
        "config_sha256": reference_config_sha,
        "train_native_units": 3_652,
        "validation_native_units": 245,
        "outer_holdout_rows_loaded": 0,
        "optimizer_steps": 345,
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "valid": True,
    }
    reference_result_path.write_text(
        json.dumps(reference_result),
        encoding="utf-8",
    )
    reference_result_sha = cached_training.file_sha256(reference_result_path)

    reference_manifest_path = reference_dir / "run_manifest.json"
    reference_manifest = {
        "status": "completed",
        "config_hash": reference_config_sha,
        "control_id": "V0",
        "backbone_name": "resnet18",
        "pretrained_weight_enum": "ResNet18_Weights.IMAGENET1K_V1",
        "resolution": 160,
        "temporal_view_name": "legacy_t16_centered_matched_observed_time",
        "sequence_length": 16,
        "temporal_encoder_name": "masked_mean",
        "run_result_sha256": reference_result_sha,
    }
    reference_manifest_path.write_text(
        json.dumps(reference_manifest),
        encoding="utf-8",
    )
    contract = payload["ablation_contract"]
    contract.update(
        {
            "reference_full_config_path": (
                reference_config_path.relative_to(repo).as_posix()
            ),
            "reference_full_config_sha256": reference_config_sha,
            "reference_full_result_path": (
                reference_result_path.relative_to(repo).as_posix()
            ),
            "reference_full_result_sha256": reference_result_sha,
            "reference_full_run_manifest_path": (
                reference_manifest_path.relative_to(repo).as_posix()
            ),
            "reference_full_run_manifest_sha256": (
                cached_training.file_sha256(reference_manifest_path)
            ),
        }
    )
    path = config_dir / "cached_training_v1_short_v3.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_v2_short_config(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    config_dir = repo / "configs" / "classification_v2"
    reference_dir = repo / "references" / "v1_full"
    config_dir.mkdir(parents=True)
    reference_dir.mkdir(parents=True)
    payload = json.loads(V2_SHORT_V4_CONFIG_PATH.read_text(encoding="utf-8"))

    base_path = config_dir / "legacy_development_l5_v1.json"
    base_payload = json.loads(
        Path("configs/classification_v2/legacy_development_l5_v1.json")
        .read_text(encoding="utf-8")
    )
    base_payload["development_root"] = str(repo / "outputs" / "legacy")
    base_path.write_text(json.dumps(base_payload), encoding="utf-8")
    payload["base_config"]["sha256"] = cached_training.file_sha256(base_path)

    reference_config_path = reference_dir / "config.json"
    reference_config = {
        "schema_version": (
            cached_training.CACHED_TRAINING_CONFIG_SCHEMA_VERSION_V3
        ),
        "training_scope": cached_training.FULL_TRAINING_SCOPE,
        "data": {
            "control_id": "V1",
            "temporal_view_name": (
                "legacy_t16_centered_matched_observed_time"
            ),
            "sampling_protocol": "one_centered_window_matched",
            "sequence_length": 16,
        },
        "base_config": payload["base_config"],
        "model": payload["model"],
        "optimization": {
            **payload["optimization"],
            "maximum_optimizer_steps": 345,
        },
    }
    reference_config_path.write_text(
        json.dumps(reference_config),
        encoding="utf-8",
    )
    reference_config_sha = cached_training.file_sha256(reference_config_path)

    reference_result_path = reference_dir / "run_result.json"
    reference_result = {
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L5_CACHED_FULL_DEVELOPMENT_TRAINING"
        ),
        "training_scope": cached_training.FULL_TRAINING_SCOPE,
        "config_sha256": reference_config_sha,
        "train_native_units": 3_652,
        "validation_native_units": 245,
        "outer_holdout_rows_loaded": 0,
        "optimizer_steps": 345,
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "valid": True,
    }
    reference_result_path.write_text(
        json.dumps(reference_result),
        encoding="utf-8",
    )
    reference_result_sha = cached_training.file_sha256(reference_result_path)

    reference_manifest_path = reference_dir / "run_manifest.json"
    reference_manifest = {
        "status": "completed",
        "config_hash": reference_config_sha,
        "control_id": "V1",
        "backbone_name": "resnet18",
        "pretrained_weight_enum": "ResNet18_Weights.IMAGENET1K_V1",
        "resolution": 224,
        "temporal_view_name": "legacy_t16_centered_matched_observed_time",
        "sequence_length": 16,
        "temporal_encoder_name": "masked_mean",
        "run_result_sha256": reference_result_sha,
    }
    reference_manifest_path.write_text(
        json.dumps(reference_manifest),
        encoding="utf-8",
    )
    contract = payload["ablation_contract"]
    contract.update(
        {
            "reference_full_config_path": (
                reference_config_path.relative_to(repo).as_posix()
            ),
            "reference_full_config_sha256": reference_config_sha,
            "reference_full_result_path": (
                reference_result_path.relative_to(repo).as_posix()
            ),
            "reference_full_result_sha256": reference_result_sha,
            "reference_full_run_manifest_path": (
                reference_manifest_path.relative_to(repo).as_posix()
            ),
            "reference_full_run_manifest_sha256": (
                cached_training.file_sha256(reference_manifest_path)
            ),
        }
    )
    path = config_dir / "cached_training_v2_short_v4.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_authorized_v2_full_config(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    config_dir = repo / "configs" / "classification_v2"
    config_dir.mkdir(parents=True)
    base_path = config_dir / "legacy_development_l5_v1.json"
    base_payload = json.loads(
        Path("configs/classification_v2/legacy_development_l5_v1.json")
        .read_text(encoding="utf-8")
    )
    base_payload["development_root"] = str(repo / "outputs" / "legacy")
    base_path.write_text(json.dumps(base_payload), encoding="utf-8")

    source_relative = Path(
        "src/pig_behavior/classification_v2/training/"
        "legacy_development_l5_cached_training.py"
    )
    source_path = repo / source_relative
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(Path(cached_training.__file__).read_bytes())
    source_sha256 = cached_training.file_sha256(source_path)

    short_payload = json.loads(SHORT_V2_CONFIG_PATH.read_text(encoding="utf-8"))
    short_payload["base_config"]["sha256"] = cached_training.file_sha256(
        base_path
    )
    short_path = config_dir / "cached_training_short_v2.json"
    short_path.write_text(json.dumps(short_payload), encoding="utf-8")
    short_config = load_legacy_l5_cached_training_config(short_path)
    gate_path = short_config.output_root / short_payload["output"][
        "short_gate_filename"
    ]
    gate_path.parent.mkdir(parents=True)
    gate_result = {
        "training_scope": cached_training.SHORT_TRAINING_SCOPE,
        "config_sha256": cached_training.file_sha256(short_path),
        "implementation_source_sha256": source_sha256,
        "valid": True,
    }
    gate = {
        "schema_version": cached_training.CACHED_TRAINING_REPEAT_GATE_SCHEMA_VERSION_V2,
        "status": "PASS_LEGACY_DEVELOPMENT_L5_CACHED_TRAINING_SHORT_GATE",
        "lineage_scope": cached_training.LINEAGE_SCOPE,
        "training_scope": cached_training.SHORT_TRAINING_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "config_path": str(short_path.resolve()),
        "config_sha256": cached_training.file_sha256(short_path),
        "implementation_source_sha256": source_sha256,
        "required_runs": 2,
        "reports": {
            "primary": {
                "result_sha256": "1" * 64,
                "result": gate_result,
                "errors": [],
                "valid": True,
            },
            "repeat": {
                "result_sha256": "2" * 64,
                "result": gate_result,
                "errors": [],
                "valid": True,
            },
        },
        "equality": {
            name: True for name in cached_training.REPEAT_EQUALITY_FIELDS
        },
        "non_overlapping_execution": {"errors": [], "valid": True},
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "exact_full_v0_t16_centered_expansion_authorized": True,
        "other_visual_or_temporal_controls_authorized": False,
        "errors": [],
        "valid": True,
    }
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    full_payload = json.loads(json.dumps(short_payload))
    full_payload["training_scope"] = cached_training.FULL_TRAINING_SCOPE
    full_payload["experiment_name"] = (
        "legacy_l5_v0_t16_centered_masked_mean_full_development"
    )
    full_payload["experiment_contract"] = {
        "experiment_id": "L5_V0_T16_FULL_DEVELOPMENT",
        "parent_id": "L5_V0_T16_SHORT",
        "scientific_role": "foundational_full_development_baseline",
        "changed_family": "bounded_to_full_train_cardinality",
        "hypothesis": (
            "The unchanged V0 cached T16 head trains on every eligible "
            "development-training native unit without outer access."
        ),
        "compute_cap": (
            "one isolated run, 3652 train units, three epochs and 345 steps"
        ),
        "stop_rule": (
            "Stop on short-gate, lineage, memory, finite, or outer-access "
            "failure."
        ),
    }
    full_payload["data"].update(
        {
            "train_selection_policy": "all_train_native_units_v1",
            "train_selection_salt": "legacy_l5_cached_full_v1",
            "train_native_units_per_class": None,
            "expected_train_native_units": 3_652,
        }
    )
    full_payload["optimization"]["maximum_optimizer_steps"] = 345
    full_payload["repeat_gate"] = {
        "required_runs": 1,
        "require_distinct_run_ids": False,
        "require_distinct_process_ids": False,
        "require_non_overlapping_execution": False,
        "require_identical_subset_hash": False,
        "require_identical_parameter_hash": False,
        "require_identical_prediction_hash": False,
        "require_identical_epoch_metric_hash": False,
    }
    full_payload["short_gate_parent"] = {
        "gate_path": gate_path.relative_to(repo).as_posix(),
        "gate_sha256": cached_training.file_sha256(gate_path),
        "short_config_path": short_path.relative_to(repo).as_posix(),
        "short_config_sha256": cached_training.file_sha256(short_path),
        "implementation_source_sha256": source_sha256,
        "authorized_training_scope": cached_training.FULL_TRAINING_SCOPE,
    }
    full_path = config_dir / "cached_training_full_v2.json"
    full_path.write_text(json.dumps(full_payload), encoding="utf-8")
    return full_path


def _write_synthetic_packet(
    root: Path,
    *,
    config: cached_training.LegacyL5CachedTrainingConfig,
    selection: cached_training.LegacyL5CachedShortSelection,
    outcome: cached_training.LegacyL5CachedTrainingOutcome,
    run_id: str,
    process_id: int,
) -> Path:
    root.mkdir()
    paths = cached_training._cached_training_run_paths(root, config=config)
    control_id = config.payload["data"]["control_id"]
    if control_id == "V2":
        backbone_name = "resnet34"
        weight_enum = "ResNet34_Weights.IMAGENET1K_V1"
        image_size = 224
    elif control_id == "V1":
        backbone_name = "resnet18"
        weight_enum = "ResNet18_Weights.IMAGENET1K_V1"
        image_size = 224
    else:
        backbone_name = "resnet18"
        weight_enum = "ResNet18_Weights.IMAGENET1K_V1"
        image_size = 160
    git_guard = {
        "code_sha": "a" * 40,
        "dirty_worktree": False,
        "dirty_entries": [],
    }
    planned = cached_training._planned_training_manifest(
        config,
        selection=selection,
        parent={
            "valid": True,
            "feature_manifest": {
                "backbone_name": backbone_name,
                "pretrained_weight_enum": weight_enum,
                "pretrained_weight_sha256": "b" * 64,
                "normalization_name": "imagenet_1k_rgb",
                "image_size": image_size,
            },
        },
        preflight={"valid": True},
        git_guard=git_guard,
        run_id=run_id,
        started_at=cached_training._utc_now(),
    )
    planned["process_id"] = process_id
    cached_training._write_json_exclusive(paths["run_manifest"], planned)
    planned_hash = cached_training.file_sha256(paths["run_manifest"])
    cached_training._write_json_exclusive(
        paths["environment"],
        cached_training._training_environment_payload(planned),
    )
    cached_training._write_json_exclusive(
        paths["preflight"],
        {"valid": True},
    )
    cached_training._write_dataframe_exclusive(
        paths["selection_manifest"],
        selection.manifest,
    )
    cached_training._write_json_exclusive(
        paths["selection_audit"],
        selection.audit,
    )
    optimization = config.payload["optimization"]
    execution = {
        "device": "cuda:0",
        "device_name": "synthetic",
        "process_id": process_id,
        "actual_total_vram_bytes": optimization[
            "validated_local_gpu_vram_bytes"
        ],
        "mem_info_total_vram_bytes": optimization[
            "validated_local_gpu_vram_bytes"
        ],
        "free_vram_before_bytes": optimization["allocator_limit_bytes"],
        "allocated_before_bytes": 0,
        "reserved_before_bytes": 0,
        "allocator_fraction": optimization["maximum_peak_vram_fraction"],
        "allocator_limit_bytes": optimization["allocator_limit_bytes"],
        "peak_allocated_bytes": 1,
        "peak_reserved_bytes": 1,
        "post_cleanup_allocated_bytes": 0,
        "post_cleanup_reserved_bytes": 0,
        "cublas_workspaces_cleared": True,
        "precision": "float32",
        "autocast_enabled": False,
        "oom": False,
        "oom_message": None,
        "oom_retry_allowed": False,
        "oom_retry_count": 0,
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "errors": [],
        "valid": True,
    }
    cached_training._finalize_cached_training_run(
        paths,
        config=config,
        planned=planned,
        planned_sha256=planned_hash,
        selection=selection,
        outcome=outcome,
        execution=execution,
        failure=None,
        runtime_seconds=1.0,
    )
    return paths["run_result"]
