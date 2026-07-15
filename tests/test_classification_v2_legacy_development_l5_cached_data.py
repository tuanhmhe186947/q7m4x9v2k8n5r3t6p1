from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training import (
    legacy_development_l5_cached_data as cached_data_module,
)
from pig_behavior.classification_v2.training.legacy_development_l5 import (
    LINEAGE_SCOPE,
    LegacyL5Config,
)
from pig_behavior.classification_v2.training.legacy_development_l5_cached_data import (
    LegacyL5CachedFeatureClassifier,
    audit_legacy_l5_cached_feature_batches,
    build_legacy_l5_cached_feature_view,
    write_legacy_l5_cached_data_packet,
)
from pig_behavior.classification_v2.training.legacy_development_l5_feature_cache import (
    FEATURE_ARTIFACT_MANIFEST_SCHEMA_VERSION,
    FEATURE_CHECKPOINT_MANIFEST_SCHEMA_VERSION,
    FEATURE_DIM,
    FEATURE_DTYPE,
    FEATURE_ENVIRONMENT_SCHEMA_VERSION,
    FEATURE_INDEX_FIELDS,
    FEATURE_PREDICTION_MANIFEST_SCHEMA_VERSION,
    FEATURE_RUN_MANIFEST_SCHEMA_VERSION,
    FEATURE_RUN_RESULT_SCHEMA_VERSION,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

VIEW_NAME = "legacy_t6_centered_matched_observed_time"
SELECTION_COLUMN = "legacy_t6_centered_matched_keep"
SEQUENCE_LENGTH = 6


@dataclass(frozen=True, slots=True)
class CachedDataFixture:
    config: LegacyL5Config
    feature_result_path: Path
    feature_index_path: Path
    fold_manifest_path: Path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _role_units() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    role_folds = (
        ("train", "fold_train", True),
        ("validation", "fold_validation", True),
        ("outer_holdout", "fold_outer", True),
    )
    unit_index = 0
    for role, fold_id, valid in role_folds:
        for behavior in VALID_BEHAVIORS:
            records.append(
                {
                    "unit_index": unit_index,
                    "role": role,
                    "fold_id": fold_id,
                    "valid": valid,
                    "behavior_label": behavior,
                }
            )
            unit_index += 1
    records.append(
        {
            "unit_index": unit_index,
            "role": "policy_invalid",
            "fold_id": "fold_train",
            "valid": False,
            "behavior_label": VALID_BEHAVIORS[0],
        }
    )
    return records


def _build_fixture(tmp_path: Path) -> CachedDataFixture:
    units = _role_units()
    development_root = tmp_path / "development"
    primary_run_id = "lineage"
    primary_root = development_root / primary_run_id
    temporal_root = primary_root / "06_temporal_tier_contract"
    fold_root = primary_root / "11_folds"
    image_root = primary_root / "09_image_context"
    temporal_root.mkdir(parents=True)
    fold_root.mkdir(parents=True)
    image_root.mkdir(parents=True)

    native_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    context_ids: list[str] = []
    for item_order, unit in enumerate(units):
        unit_key = f"unit_{item_order:03d}"
        window_id = f"window_{item_order:03d}"
        video_key = f"video_{item_order:03d}"
        group_id = f"recording_{item_order:03d}"
        behavior = str(unit["behavior_label"])
        native_rows.append(
            {
                "temporal_unit_key": unit_key,
                "source_type": "legacy_recovered",
                "dataset_id": "legacy_fixture",
                "video_key": video_key,
                "label_frame_count": 16,
                "behavior_label": behavior,
                "native_unit_valid_for_development": bool(unit["valid"]),
                "lineage_scope": LINEAGE_SCOPE,
                "human_review_complete": False,
            }
        )
        fold_rows.append(
            {
                "window_id": window_id,
                "temporal_unit_key": unit_key,
                "window_length_frames": SEQUENCE_LENGTH,
                "tier_window_valid": True,
                SELECTION_COLUMN: True,
                "recording_group_id": group_id,
                "oof_fold_id": str(unit["fold_id"]),
                "behavior_label": behavior,
                "source_type": "legacy_recovered",
                "dataset_id": "legacy_fixture",
                "video_key": video_key,
                "lineage_scope": LINEAGE_SCOPE,
                "human_review_complete": False,
            }
        )
        unit_context_ids = [
            f"context_{item_order:03d}_{slot_index:02d}"
            for slot_index in range(SEQUENCE_LENGTH)
        ]
        frame_indices = list(range(5, 5 + SEQUENCE_LENGTH))
        context_ids.extend(unit_context_ids)
        image_rows.append(
            {
                "window_id": window_id,
                "window_length_frames": SEQUENCE_LENGTH,
                "expected_frame_indices": "|".join(
                    str(value) for value in frame_indices
                ),
                "image_context_id_sequence": ";;".join(unit_context_ids),
                "observed_image_context_rows": SEQUENCE_LENGTH,
                "loadable_image_context_rows": SEQUENCE_LENGTH,
                "missing_image_context_slots": 0,
                "window_image_context_complete": True,
                "lineage_scope": LINEAGE_SCOPE,
                "human_review_complete": False,
            }
        )
        for slot_index, frame_index in enumerate(frame_indices):
            slot_rows.append(
                {
                    "temporal_view_name": VIEW_NAME,
                    "view_item_id": window_id,
                    "parent_window_id": window_id,
                    "temporal_unit_key": unit_key,
                    "item_order": item_order,
                    "slot_index": slot_index,
                    "declared_sequence_length": SEQUENCE_LENGTH,
                    "frame_index_expected_audit": frame_index,
                    "time_delta": slot_index / 30.0,
                    "length_mask": True,
                    "observed_mask": True,
                    "timing_valid_mask": True,
                    "padding_mask": False,
                    "lineage_scope": LINEAGE_SCOPE,
                    "human_review_complete": False,
                }
            )

    pd.DataFrame(native_rows).to_csv(
        temporal_root / "native_temporal_unit_manifest.csv",
        index=False,
        lineterminator="\n",
    )
    fold_manifest_path = fold_root / "window_oof_fold_manifest.csv"
    pd.DataFrame(fold_rows).to_csv(
        fold_manifest_path,
        index=False,
        lineterminator="\n",
    )
    pd.DataFrame(image_rows).to_csv(
        image_root / "image_window_context_manifest.csv",
        index=False,
        lineterminator="\n",
    )
    pd.DataFrame(slot_rows).to_csv(
        temporal_root / "legacy_t6_centered_matched_observed_time_manifest.csv",
        index=False,
        lineterminator="\n",
    )

    config_path = tmp_path / "fixture_config.json"
    _write_json(config_path, {"fixture": "cached_data"})
    expected_counts = {
        "image_context_rows": len(context_ids),
        "selected_native_units": len(units),
        "train_native_units": len(VALID_BEHAVIORS),
        "validation_native_units": len(VALID_BEHAVIORS),
        "outer_holdout_native_units": len(VALID_BEHAVIORS),
        "policy_invalid_native_units": 1,
    }
    config = LegacyL5Config(
        path=config_path,
        payload={
            "expected_counts": expected_counts,
            "split_contract": {
                "outer_holdout_fold_id": "fold_outer",
                "development_validation_fold_id": "fold_validation",
            },
        },
        development_root=development_root,
        primary_run_id=primary_run_id,
        l3_audit_relative_path=Path("unused_l3.json"),
        l4_audit_relative_path=Path("unused_l4.json"),
        l5_output_relative_path=Path("unused_l5"),
    )

    feature_root = tmp_path / "feature_run"
    feature_root.mkdir()
    tensor_path = feature_root / "frame_features_f32.npy"
    features = np.repeat(
        np.arange(len(context_ids), dtype=FEATURE_DTYPE)[:, None],
        FEATURE_DIM,
        axis=1,
    )
    np.save(tensor_path, features)
    feature_index_path = feature_root / "frame_feature_index.csv"
    index_rows = [
        {
            "image_context_id": context_id,
            "feature_row": row_index,
            "control_id": "V0",
            "backbone_name": "resnet18",
            "pretrained_weight_enum": "ResNet18_Weights.IMAGENET1K_V1",
            "image_size": 160,
            "feature_dim": FEATURE_DIM,
            "feature_dtype": str(FEATURE_DTYPE),
            "lineage_scope": LINEAGE_SCOPE,
            "human_review_complete": False,
        }
        for row_index, context_id in enumerate(context_ids)
    ]
    pd.DataFrame(index_rows, columns=FEATURE_INDEX_FIELDS).to_csv(
        feature_index_path,
        index=False,
        lineterminator="\n",
    )
    packet_files = {
        "artifact_manifest": feature_root / "artifact_manifest.json",
        "checkpoint_manifest": feature_root / "checkpoint_manifest.json",
        "prediction_manifest": feature_root / "prediction_manifest.json",
        "environment": feature_root / "environment.json",
    }
    parent_run_id = "feature_fixture"
    _write_json(
        packet_files["artifact_manifest"],
        {
            "schema_version": FEATURE_ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "run_id": parent_run_id,
            "status": "completed",
            "artifacts": [],
        },
    )
    _write_json(
        packet_files["checkpoint_manifest"],
        {
            "schema_version": FEATURE_CHECKPOINT_MANIFEST_SCHEMA_VERSION,
            "run_id": parent_run_id,
            "status": "completed",
            "checkpoints": [],
            "checkpoint_creation_authorized": False,
        },
    )
    _write_json(
        packet_files["prediction_manifest"],
        {
            "schema_version": FEATURE_PREDICTION_MANIFEST_SCHEMA_VERSION,
            "run_id": parent_run_id,
            "status": "completed",
            "predictions": [],
            "prediction_creation_authorized": False,
            "outer_holdout_predictions_authorized": False,
        },
    )
    _write_json(
        packet_files["environment"],
        {
            "schema_version": FEATURE_ENVIRONMENT_SCHEMA_VERSION,
            "declared_gpu_vram_gib": 4,
            "gpu_vram_bytes": 4_294_443_008,
            "maximum_peak_vram_fraction": 0.7,
            "precision": "float32",
            "autocast_enabled": False,
            "oom_retry_allowed": False,
        },
    )
    feature_result_path = feature_root / "run_result.json"
    result = {
        "schema_version": FEATURE_RUN_RESULT_SCHEMA_VERSION,
        "status": "PASS_LEGACY_DEVELOPMENT_L5_FEATURE_CACHE",
        "scope": "full",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "feature_dtype": str(FEATURE_DTYPE),
        "precision": "float32",
        "autocast_enabled": False,
        "oom_retry_allowed": False,
        "gradient_enabled": False,
        "optimizer_steps": 0,
        "accuracy_f1_computed": False,
        "baseline_metrics_authorized": False,
        "nonfinite_feature_values": 0,
        "video_decode_count": 0,
        "video_seek_count": 0,
        "full_control_complete": True,
        "working_set_release_policy": (
            "flush_close_reopen_input_output_each_checkpoint_v1"
        ),
        "valid": True,
        "run_id": parent_run_id,
        "config_sha256": config.sha256,
        "control_id": "V0",
        "backbone_name": "resnet18",
        "pretrained_weight_enum": "ResNet18_Weights.IMAGENET1K_V1",
        "image_size": 160,
        "feature_shape": [len(context_ids), FEATURE_DIM],
        "completed_rows": len(context_ids),
        "oom": False,
        "oom_retry_count": 0,
        "actual_total_vram_bytes": 4_294_443_008,
        "allocator_limit_bytes": 3_006_110_105,
        "source_media_loads": 0,
        "post_cleanup_allocated_bytes": 0,
        "post_cleanup_reserved_bytes": 0,
        "feature_tensor_path": str(tensor_path.resolve()),
        "feature_tensor_sha256": file_sha256(tensor_path),
        "feature_index_path": str(feature_index_path.resolve()),
        "feature_index_sha256": file_sha256(feature_index_path),
        "environment_sha256": file_sha256(packet_files["environment"]),
    }
    _write_json(feature_result_path, result)
    manifest = {
        "schema_version": FEATURE_RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": parent_run_id,
        "status": "completed",
        "config_hash": config.sha256,
        "control_id": "V0",
        "backbone_name": "resnet18",
        "pretrained_weight_enum": "ResNet18_Weights.IMAGENET1K_V1",
        "run_result_sha256": file_sha256(feature_result_path),
        "artifact_manifest_sha256": file_sha256(
            packet_files["artifact_manifest"]
        ),
        "checkpoint_manifest_sha256": file_sha256(
            packet_files["checkpoint_manifest"]
        ),
        "prediction_manifest_sha256": file_sha256(
            packet_files["prediction_manifest"]
        ),
    }
    _write_json(feature_root / "run_manifest.json", manifest)
    return CachedDataFixture(
        config=config,
        feature_result_path=feature_result_path,
        feature_index_path=feature_index_path,
        fold_manifest_path=fold_manifest_path,
    )


def _refresh_feature_index_hashes(fixture: CachedDataFixture) -> None:
    result = _read_json(fixture.feature_result_path)
    result["feature_index_sha256"] = file_sha256(fixture.feature_index_path)
    _write_json(fixture.feature_result_path, result)
    manifest_path = fixture.feature_result_path.parent / "run_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["run_result_sha256"] = file_sha256(fixture.feature_result_path)
    _write_json(manifest_path, manifest)


def test_cached_feature_view_joins_in_exact_order_and_closes_each_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(tmp_path)
    view = build_legacy_l5_cached_feature_view(
        fixture.config,
        feature_result_path=fixture.feature_result_path,
        temporal_view_name=VIEW_NAME,
    )

    assert view.windows["l5_role"].value_counts().to_dict() == {
        "train": 10,
        "validation": 10,
    }
    assert view.feature_rows.shape == (20, SEQUENCE_LENGTH)
    assert set(view.fold_manifest["outer_fold_id"].astype(str)) == {
        "fold_outer"
    }
    assert view.fold_manifest["role"].equals(view.fold_manifest["l5_role"])
    loaded = view.load_sequences(np.asarray([0, 10], dtype=np.int64))
    np.testing.assert_array_equal(loaded[:, :, 0], view.feature_rows[[0, 10]])
    np.testing.assert_allclose(view.sample_weights, 1.0)
    assert view.audit["join_audit"]["outer_holdout_feature_slots"] == 0
    assert view.audit["source_media_loads"] == 0

    close_calls = 0
    original_close = cached_data_module._close_memmap

    def _counted_close(array: np.ndarray) -> None:
        nonlocal close_calls
        close_calls += 1
        original_close(array)

    monkeypatch.setattr(cached_data_module, "_close_memmap", _counted_close)
    batches = list(
        view.iter_role_batches(
            "train",
            batch_size=3,
            seed=20260715,
            shuffle=False,
        )
    )
    assert [len(batch["positions"]) for batch in batches] == [3, 3, 3, 1]
    assert close_calls == len(batches)


def test_cached_feature_view_rejects_missing_and_duplicate_features(
    tmp_path: Path,
) -> None:
    missing_fixture = _build_fixture(tmp_path / "missing")
    missing_index = pd.read_csv(missing_fixture.feature_index_path)
    missing_index.loc[0, "image_context_id"] = "unused_replacement_context"
    missing_index.to_csv(
        missing_fixture.feature_index_path,
        index=False,
        lineterminator="\n",
    )
    _refresh_feature_index_hashes(missing_fixture)
    with pytest.raises(ValueError, match="missing cached features"):
        build_legacy_l5_cached_feature_view(
            missing_fixture.config,
            feature_result_path=missing_fixture.feature_result_path,
            temporal_view_name=VIEW_NAME,
        )

    duplicate_fixture = _build_fixture(tmp_path / "duplicate")
    duplicate_index = pd.read_csv(duplicate_fixture.feature_index_path)
    duplicate_index.loc[1, "image_context_id"] = duplicate_index.loc[
        0, "image_context_id"
    ]
    duplicate_index.to_csv(
        duplicate_fixture.feature_index_path,
        index=False,
        lineterminator="\n",
    )
    _refresh_feature_index_hashes(duplicate_fixture)
    with pytest.raises(ValueError, match="duplicate context IDs"):
        build_legacy_l5_cached_feature_view(
            duplicate_fixture.config,
            feature_result_path=duplicate_fixture.feature_result_path,
            temporal_view_name=VIEW_NAME,
        )


def test_cached_feature_view_rejects_selected_invalid_tier(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    folds = pd.read_csv(fixture.fold_manifest_path)
    folds.loc[0, "tier_window_valid"] = False
    folds.to_csv(
        fixture.fold_manifest_path,
        index=False,
        lineterminator="\n",
    )

    with pytest.raises(ValueError, match="invalid tier rows"):
        build_legacy_l5_cached_feature_view(
            fixture.config,
            feature_result_path=fixture.feature_result_path,
            temporal_view_name=VIEW_NAME,
        )


def test_cached_feature_view_rejects_group_leakage_and_outer_access(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    folds = pd.read_csv(fixture.fold_manifest_path)
    train_group = str(
        folds.loc[folds["oof_fold_id"].eq("fold_train"), "recording_group_id"].iloc[
            0
        ]
    )
    validation_index = folds.index[
        folds["oof_fold_id"].eq("fold_validation")
    ][0]
    folds.loc[validation_index, "recording_group_id"] = train_group
    folds.to_csv(
        fixture.fold_manifest_path,
        index=False,
        lineterminator="\n",
    )
    with pytest.raises(ValueError, match="role overlap"):
        build_legacy_l5_cached_feature_view(
            fixture.config,
            feature_result_path=fixture.feature_result_path,
            temporal_view_name=VIEW_NAME,
        )

    clean_fixture = _build_fixture(tmp_path / "clean")
    view = build_legacy_l5_cached_feature_view(
        clean_fixture.config,
        feature_result_path=clean_fixture.feature_result_path,
        temporal_view_name=VIEW_NAME,
    )
    with pytest.raises(ValueError, match="access forbidden"):
        view.indices_for_role("outer_holdout")
    assert "outer_holdout" not in set(view.windows["l5_role"].astype(str))
    assert view.audit["role_audit"]["native_counts"]["outer_holdout"] == 10


def test_cached_feature_view_rejects_parent_outside_four_gib_contract(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    result = _read_json(fixture.feature_result_path)
    result["actual_total_vram_bytes"] = 8 * 1024**3
    _write_json(fixture.feature_result_path, result)
    _refresh_feature_index_hashes(fixture)

    with pytest.raises(ValueError, match="frozen 4 GB contract"):
        build_legacy_l5_cached_feature_view(
            fixture.config,
            feature_result_path=fixture.feature_result_path,
            temporal_view_name=VIEW_NAME,
        )


def test_cached_feature_classifier_is_mask_invariant_and_has_gradients() -> None:
    torch.manual_seed(20260715)
    model = LegacyL5CachedFeatureClassifier(
        temporal_encoder_name="masked_mean",
        hidden_dim=16,
        dropout=0.0,
        transformer_layers=1,
        transformer_heads=4,
    )
    model.train()
    features = torch.randn(2, SEQUENCE_LENGTH, FEATURE_DIM, requires_grad=True)
    observed_mask = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, True, True, True, False, False],
        ]
    )
    time_delta = torch.arange(SEQUENCE_LENGTH, dtype=torch.float32).repeat(2, 1)
    altered = features.detach().clone()
    altered[~observed_mask] = 1_000_000.0

    logits = model(features, observed_mask, time_delta=time_delta)
    altered_logits = model(altered, observed_mask, time_delta=time_delta)
    assert logits.shape == (2, len(VALID_BEHAVIORS))
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits, altered_logits)

    logits.square().mean().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    projection_gradient = model.projection.weight.grad
    head_gradient = model.behavior_head.weight.grad
    assert projection_gradient is not None
    assert head_gradient is not None
    assert torch.isfinite(projection_gradient).all()
    assert torch.isfinite(head_gradient).all()
    assert projection_gradient.abs().sum().item() > 0.0
    assert head_gradient.abs().sum().item() > 0.0


def test_cached_data_packet_requires_the_bounded_crash_gate(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    view = build_legacy_l5_cached_feature_view(
        fixture.config,
        feature_result_path=fixture.feature_result_path,
        temporal_view_name=VIEW_NAME,
    )
    with pytest.raises(ValueError, match="batch size"):
        audit_legacy_l5_cached_feature_batches(
            view,
            batch_size=257,
            max_batches_per_role=1,
        )
    with pytest.raises(ValueError, match="passing bounded audit"):
        write_legacy_l5_cached_data_packet(
            view,
            output_dir=tmp_path / "unaudited_packet",
            run_id="unaudited_packet",
            runtime_seconds=0.1,
        )


def test_cached_data_packet_writes_fail_closed_zero_cuda_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(tmp_path)
    monkeypatch.setattr(
        cached_data_module,
        "git_state",
        lambda: {"commit": "fixture_sha", "dirty": False, "dirty_entries": []},
    )
    cuda_checks = 0

    def _cuda_is_not_initialized() -> bool:
        nonlocal cuda_checks
        cuda_checks += 1
        return False

    monkeypatch.setattr(torch.cuda, "is_initialized", _cuda_is_not_initialized)
    view = build_legacy_l5_cached_feature_view(
        fixture.config,
        feature_result_path=fixture.feature_result_path,
        temporal_view_name=VIEW_NAME,
    )
    view = audit_legacy_l5_cached_feature_batches(
        view,
        batch_size=4,
        max_batches_per_role=2,
    )
    output_dir = tmp_path / "cached_fixture_001"
    manifest = write_legacy_l5_cached_data_packet(
        view,
        output_dir=output_dir,
        run_id="cached_fixture_001",
        runtime_seconds=0.25,
    )

    assert manifest["status"] == "completed"
    assert manifest["peak_vram_bytes"] == 0
    assert cuda_checks == 3
    bounded = view.audit["bounded_batch_audit"]
    assert bounded["total_loaded_windows"] == 16
    assert bounded["outer_holdout_rows_loaded"] == 0
    assert bounded["maximum_loaded_batch_bytes"] < 1_000_000
    environment = _read_json(output_dir / "environment.json")
    assert environment["cuda_runtime_initialized"] is False
    assert environment["gpu_execution_performed"] is False
    assert environment["declared_local_gpu_vram_gib"] == 4
    assert environment["maximum_peak_vram_fraction"] == 0.7
    assert environment["dataloader_num_workers"] == 0
    assert environment["pin_memory"] is False
    assert environment["oom_retry_allowed"] is False
    fold_manifest = pd.read_csv(output_dir / "fold_manifest.csv")
    native_routing = pd.read_csv(output_dir / "native_routing_manifest.csv")
    assert len(fold_manifest) == 30
    assert len(native_routing) == 31
    assert "policy_invalid" not in set(fold_manifest["role"].astype(str))
    assert "policy_invalid" in set(native_routing["role"].astype(str))
    packet_audit = _read_json(output_dir / "cached_data_audit.json")[
        "packet_manifest_audit"
    ]
    assert packet_audit["eligible_fold_manifest_rows"] == 30
    assert packet_audit["native_routing_manifest_rows"] == 31
    assert packet_audit["policy_invalid_rows_preserved"] == 1
    assert packet_audit["silent_row_drop"] is False
    assert _read_json(output_dir / "checkpoint_manifest.json")[
        "checkpoints"
    ] == []
    assert _read_json(output_dir / "prediction_manifest.json")[
        "predictions"
    ] == []
    with (output_dir / "runs_registry.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        registry_rows = list(csv.DictReader(handle))
    assert len(registry_rows) == 1
    assert registry_rows[0]["outer_predictions_created"] == "0"
    assert registry_rows[0]["peak_vram_bytes"] == "0"
