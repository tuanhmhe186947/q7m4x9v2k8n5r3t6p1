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
) -> LegacyL5CachedFeatureView:
    records: list[dict[str, object]] = []
    targets: list[int] = []
    item_index = 0
    for behavior_index, behavior in enumerate(VALID_BEHAVIORS):
        for _ in range(10):
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
    paths = cached_training._cached_training_run_paths(root)
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
                "backbone_name": "resnet18",
                "pretrained_weight_enum": "ResNet18_Weights.IMAGENET1K_V1",
                "pretrained_weight_sha256": "b" * 64,
                "normalization_name": "imagenet_1k_rgb",
                "image_size": 160,
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
