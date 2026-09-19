from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training import (
    legacy_development_l5_temporal_ladder_runtime as ladder_runtime,
)
from pig_behavior.classification_v2.training.legacy_development_l5_cached_data import (
    LegacyL5CachedFeatureView,
)
from pig_behavior.classification_v2.training.legacy_development_l5_temporal_ladder import (
    CANONICAL_VIEWS,
    FULL_CONFIG_SCHEMA,
    FULL_SCOPE,
    LINEAGE_SCOPE,
    SHORT_CONFIG_SCHEMA,
    SHORT_SCOPE,
    TemporalLadderConfig,
    _validate_config_payload,
    _validate_full_semantic_binding,
    aggregate_temporal_ladder_predictions,
    build_temporal_ladder_selection,
    load_temporal_ladder_config,
    train_temporal_ladder_core,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256


def test_sliding_selection_is_native_first_and_event_mass_balanced(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    view = _view(tmp_path, view_id="t6_sliding")

    selected = build_temporal_ladder_selection(view, config, "t6_sliding")

    assert selected.audit["train_native_units"] == 80
    assert selected.audit["validation_native_units"] == 245
    assert selected.audit["train_windows"] == 320
    assert selected.audit["validation_windows"] == 980
    train = selected.manifest.loc[selected.manifest["l5_role"].eq("train")]
    validation = selected.manifest.loc[
        selected.manifest["l5_role"].eq("validation")
    ]
    assert train.groupby("temporal_unit_key").size().eq(4).all()
    assert validation.groupby("temporal_unit_key").size().eq(4).all()
    assert train.groupby("temporal_unit_key")["sample_weight"].sum().eq(1.0).all()
    assert (
        validation.groupby("temporal_unit_key")["sample_weight"].sum().eq(1.0).all()
    )
    assert train.groupby("behavior_label")["temporal_unit_key"].nunique().to_dict() == {
        label: 8 for label in VALID_BEHAVIORS
    }


def test_native_aggregation_means_windows_and_rejects_mass_drift() -> None:
    windows = _window_predictions()

    native, metrics, per_class, confusion = aggregate_temporal_ladder_predictions(
        windows,
        expected_windows_per_native=4,
        training_scope=SHORT_SCOPE,
    )

    assert len(native) == 2
    assert native["aggregated_window_count"].tolist() == [4, 4]
    assert native["predicted_label"].tolist() == ["drink", "eat"]
    assert metrics["native_unit_rows"] == 2
    assert metrics["window_rows"] == 8
    assert metrics["macro_f1_global_10_class"] == pytest.approx(0.2)
    assert metrics["accuracy"] == 1.0
    assert len(per_class) == 10
    assert confusion.shape[0] == 10
    assert set(native["lineage_scope"]) == {LINEAGE_SCOPE}

    drifted = windows.copy()
    drifted.loc[0, "sample_weight"] = 0.5
    with pytest.raises(ValueError, match="event mass"):
        aggregate_temporal_ladder_predictions(
            drifted,
            expected_windows_per_native=4,
            training_scope=SHORT_SCOPE,
        )

    probability_drifted = windows.copy()
    probability_columns = [
        column for column in probability_drifted if column.startswith("prob_")
    ]
    probability_drifted.loc[:, probability_columns] *= 0.9
    with pytest.raises(ValueError, match="native probability mass"):
        aggregate_temporal_ladder_predictions(
            probability_drifted,
            expected_windows_per_native=4,
            training_scope=SHORT_SCOPE,
        )


def test_centered_core_runs_three_deterministic_cpu_epochs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    view = _view(tmp_path, view_id="t6_centered", materialize_features=True)
    selected = build_temporal_ladder_selection(view, config, "t6_centered")

    outcome = train_temporal_ladder_core(
        view,
        selected,
        config,
        "t6_centered",
        device=torch.device("cpu"),
    )

    assert outcome.optimizer_steps == 9
    assert outcome.best_epoch in {1, 2, 3}
    assert len(outcome.window_predictions) == 245
    assert len(outcome.native_predictions) == 245
    assert outcome.metrics["native_unit_rows"] == 245
    assert outcome.metrics["window_rows"] == 245
    assert len(outcome.parameter_sha256) == 64
    assert len(outcome.native_prediction_sha256) == 64
    assert outcome.maximum_loaded_batch_bytes <= 2_103_552


def test_config_contract_rejects_hardware_or_merged_scope_drift() -> None:
    payload = _config_payload()
    _validate_config_payload(payload)

    changed_hardware = _config_payload()
    changed_hardware["experiment_contract"]["local_vram_is_architecture_limit"] = True
    with pytest.raises(ValueError, match="experiment contract"):
        _validate_config_payload(changed_hardware)

    changed_cublas = _config_payload()
    changed_cublas["optimization"]["cublas_workspace_config"] = ":16:8"
    with pytest.raises(ValueError, match="optimization contract"):
        _validate_config_payload(changed_cublas)

    changed_claim = _config_payload()
    changed_claim["q2_claim_allowed"] = True
    with pytest.raises(ValueError, match="q2_claim_allowed"):
        _validate_config_payload(changed_claim)


def test_loader_binds_feature_run_manifest_hash(tmp_path: Path) -> None:
    payload = _config_payload()
    bound_files = {
        "base.json": "base\n",
        "feature_result.json": "feature result\n",
        "feature_manifest.json": "feature manifest\n",
        "core.py": "core\n",
        "runtime.py": "runtime\n",
        "engine.py": "engine\n",
    }
    for name, content in bound_files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    payload["base_config"] = {
        "path": "base.json",
        "sha256": file_sha256(tmp_path / "base.json"),
    }
    feature = payload["feature_parent"]
    feature["result_sha256"] = file_sha256(tmp_path / "feature_result.json")
    feature["run_manifest_sha256"] = file_sha256(
        tmp_path / "feature_manifest.json"
    )
    implementation = payload["implementation"]
    for prefix, filename in (
        ("core", "core.py"),
        ("runtime", "runtime.py"),
        ("frozen_engine", "engine.py"),
    ):
        implementation[f"{prefix}_sha256"] = file_sha256(tmp_path / filename)
    config_path = tmp_path / "configs" / "classification_v2" / "short.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_temporal_ladder_config(config_path).training_scope == SHORT_SCOPE
    (tmp_path / "feature_manifest.json").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="feature run manifest hash"):
        load_temporal_ladder_config(config_path)


def test_full_config_is_scientifically_bound_to_short_config() -> None:
    short = _config_payload()
    full = copy.deepcopy(short)
    full["schema_version"] = FULL_CONFIG_SCHEMA
    full["training_scope"] = FULL_SCOPE

    _validate_full_semantic_binding(full, short)

    full["model"]["hidden_dim"] = 256
    with pytest.raises(ValueError, match="full/short scientific binding.model"):
        _validate_full_semantic_binding(full, short)


def test_runtime_artifact_manifest_detects_tampering(tmp_path: Path) -> None:
    paths: dict[str, Path] = {}
    for name, filename in ladder_runtime.ARTIFACT_FILES.items():
        path = tmp_path / filename
        path.write_bytes(f"artifact:{name}\n".encode())
        paths[name] = path
    manifest = ladder_runtime._build_artifact_manifest(
        paths,
        run_id="synthetic-run",
        view_id="t6_centered",
    )

    assert ladder_runtime._audit_artifacts(tmp_path, manifest)["valid"] is True
    paths["native_predictions"].write_bytes(b"tampered\n")
    audit = ladder_runtime._audit_artifacts(tmp_path, manifest)
    assert audit["valid"] is False
    assert "artifact_hash_mismatch=native_predictions" in audit["errors"]


def test_repeat_gate_requires_distinct_exact_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.path.write_text("{}\n", encoding="utf-8")
    common = _repeat_result(config)
    primary = _repeat_audit(
        {
            **common,
            "process_id": 101,
            "started_at_utc": "2026-07-15T00:00:00+00:00",
            "completed_at_utc": "2026-07-15T00:01:00+00:00",
        },
        "primary",
    )
    repeat = _repeat_audit(
        {
            **common,
            "process_id": 202,
            "started_at_utc": "2026-07-15T00:02:00+00:00",
            "completed_at_utc": "2026-07-15T00:03:00+00:00",
        },
        "repeat",
    )

    def fake_audit(
        _config_value: TemporalLadderConfig,
        *,
        result_path: Path,
    ) -> dict[str, object]:
        return primary if result_path.stem == "primary" else repeat

    monkeypatch.setattr(ladder_runtime, "audit_temporal_ladder_run", fake_audit)
    gate = ladder_runtime.audit_temporal_ladder_repeat_gate(
        config,
        view_id="t6_centered",
        primary_result_path=tmp_path / "primary.json",
        repeat_result_path=tmp_path / "repeat.json",
    )
    assert gate["valid"] is True
    assert gate["full_view_expansion_authorized"] is True

    repeat["result"]["parameter_sha256"] = "f" * 64
    drifted = ladder_runtime.audit_temporal_ladder_repeat_gate(
        config,
        view_id="t6_centered",
        primary_result_path=tmp_path / "primary.json",
        repeat_result_path=tmp_path / "repeat.json",
    )
    assert drifted["valid"] is False
    assert "repeat_field_mismatch=parameter_sha256" in drifted["errors"]


def _repeat_result(config: TemporalLadderConfig) -> dict[str, object]:
    implementation = config.payload["implementation"]
    return {
        "run_id": "synthetic",
        "view_id": "t6_centered",
        "config_sha256": config.sha256,
        "implementation_hashes": {
            "core_sha256": implementation["core_sha256"],
            "runtime_sha256": implementation["runtime_sha256"],
            "frozen_engine_sha256": implementation["frozen_engine_sha256"],
        },
        "training_scope": SHORT_SCOPE,
        "selection_content_sha256": "a" * 64,
        "train_native_unit_sha256": "b" * 64,
        "validation_native_unit_sha256": "c" * 64,
        "train_native_units": 80,
        "validation_native_units": 245,
        "train_windows": 80,
        "validation_windows": 245,
        "optimizer_steps": 9,
        "best_epoch": 2,
        "parameter_sha256": "d" * 64,
        "window_prediction_content_sha256": "e" * 64,
        "native_prediction_content_sha256": "1" * 64,
        "epoch_metrics_content_sha256": "2" * 64,
        "validation_metrics": {"macro_f1_global_10_class": 0.2},
    }


def _repeat_audit(
    result: dict[str, object],
    name: str,
) -> dict[str, object]:
    return {
        "run_id": name,
        "view_id": "t6_centered",
        "result_path": f"{name}.json",
        "result_sha256": "3" * 64,
        "run_manifest_sha256": "4" * 64,
        "artifact_manifest_sha256": "5" * 64,
        "verified_artifacts": len(ladder_runtime.ARTIFACT_FILES),
        "result": result,
        "errors": [],
        "valid": True,
    }


def _config(tmp_path: Path) -> TemporalLadderConfig:
    return TemporalLadderConfig(
        path=tmp_path / "short_config.json",
        payload=_config_payload(),
        repo_root=tmp_path,
    )


def _config_payload() -> dict[str, object]:
    return {
        "schema_version": SHORT_CONFIG_SCHEMA,
        "training_scope": SHORT_SCOPE,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "development_metrics_authorized": True,
        "experiment_contract": {
            "experiment_id": "L5_V1_TEMPORAL_LENGTH_PROTOCOL_LADDER_V1",
            "parent_decision": "RETAIN_V1_REJECT_T1_FOR_LEGACY_T16_SEARCH",
            "changed_family": "temporal_input_length_and_declared_protocol_matrix",
            "primary_metric": "validation_native_unit_macro_f1_global_10_class",
            "uncertainty_cluster": "video_key",
            "outer_predictions_used_for_model_selection": False,
            "legacy_only_decision": True,
            "merged_reviewed_reassessment_required": True,
            "local_vram_is_architecture_limit": False,
        },
        "base_config": {"path": "base.json", "sha256": "a" * 64},
        "feature_parent": {
            "result_path": "feature_result.json",
            "result_sha256": "b" * 64,
            "run_manifest_path": "feature_manifest.json",
            "run_manifest_sha256": "c" * 64,
            "feature_tensor_sha256": "d" * 64,
            "feature_index_sha256": "e" * 64,
        },
        "implementation": {
            "core_path": "core.py",
            "core_sha256": "1" * 64,
            "runtime_path": "runtime.py",
            "runtime_sha256": "2" * 64,
            "frozen_engine_path": "engine.py",
            "frozen_engine_sha256": "3" * 64,
        },
        "views": {
            view_id: {
                "temporal_view_name": expected["temporal_view_name"],
                "sampling_protocol": expected["sampling_protocol"],
                "sequence_length": expected["sequence_length"],
                "windows_per_native_unit": expected["windows_per_native_unit"],
                "consumer_parent": {
                    "run_id": f"consumer-{view_id}",
                    "code_sha": "f" * 40,
                    "run_relative_path": f"consumers/{view_id}",
                    "run_manifest_sha256": "4" * 64,
                    "cached_data_audit_sha256": "5" * 64,
                    "artifact_manifest_sha256": "6" * 64,
                },
            }
            for view_id, expected in CANONICAL_VIEWS.items()
        },
        "selection": {
            "native_unit": "complete_legacy_16_frame_burst",
            "short_train_selection_policy": "sha256_rank_per_class_native_first_v1",
            "short_native_selection_salt": "legacy_l5_temporal_ladder_short_v1",
            "short_train_native_units_per_class": 8,
            "short_train_native_units": 80,
            "full_train_native_units": 3_652,
            "validation_native_units": 245,
            "event_mass_per_native_unit": 1.0,
            "window_expansion_after_native_selection": True,
            "validation_selection_policy": "all_validation_native_units_v1",
            "outer_holdout_access": "FORBIDDEN_DURING_MODEL_SELECTION",
        },
        "model": {
            "architecture": "cached_frame_feature_temporal_classifier_v1",
            "feature_control_id": "V1",
            "backbone_name": "resnet18",
            "input_resolution": 224,
            "temporal_encoder_name": "masked_mean",
            "hidden_dim": 128,
            "dropout": 0.1,
            "transformer_layers": 1,
            "transformer_heads": 4,
            "parameter_count": 68_234,
            "native_probability_aggregation": "mean_window_probability_v1",
        },
        "optimization": {
            "seed": 20260714,
            "epochs": 3,
            "batch_size": 32,
            "evaluation_batch_size": 64,
            "learning_rate": 0.003,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
            "loss": "event_mass_balanced_cross_entropy",
            "sampler": "deterministic_seeded_window_shuffle_after_native_selection",
            "checkpoint_selection": "native_global_10_class_macro_f1_then_nll",
            "precision": "float32",
            "autocast_enabled": False,
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
            "dataloader_num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": None,
            "device": "cuda:0",
            "declared_local_gpu_vram_gib": 4,
            "validated_local_gpu_vram_bytes": 4_294_443_008,
            "maximum_peak_vram_fraction": 0.7,
            "allocator_limit_bytes": 3_006_110_105,
            "oom_retry_allowed": False,
            "maximum_loaded_batch_bytes": 2_103_552,
        },
        "repeat_gate": {
            "required_runs": 2,
            "require_fresh_process": True,
            "require_distinct_process_ids": True,
            "require_non_overlapping_execution": True,
            "require_identical_selection_hash": True,
            "require_identical_parameter_hash": True,
            "require_identical_window_prediction_hash": True,
            "require_identical_native_prediction_hash": True,
            "require_identical_epoch_metric_hash": True,
        },
        "execution_guard": {
            "allowed_dirty_paths": [],
            "required_tracked_paths": [],
        },
        "output": {
            "run_root_relative_path": "15_l5_core_baselines",
            "matrix_gate_filename": "short_matrix.json",
        },
    }


def _view(
    tmp_path: Path,
    *,
    view_id: str,
    materialize_features: bool = False,
) -> LegacyL5CachedFeatureView:
    expected = CANONICAL_VIEWS[view_id]
    multiplier = int(expected["windows_per_native_unit"])
    sequence_length = int(expected["sequence_length"])
    native_rows: list[tuple[str, str, str]] = []
    for index in range(3_652):
        native_rows.append((f"train-{index:04d}", "train", VALID_BEHAVIORS[index % 10]))
    for index in range(245):
        native_rows.append(
            (f"validation-{index:04d}", "validation", VALID_BEHAVIORS[index % 10])
        )
    rows: list[dict[str, object]] = []
    targets: list[int] = []
    weights: list[float] = []
    label_to_index = {label: index for index, label in enumerate(VALID_BEHAVIORS)}
    for unit_id, role, label in native_rows:
        for window_index in range(multiplier):
            rows.append(
                {
                    "window_id": f"{unit_id}-window-{window_index}",
                    "temporal_unit_key": unit_id,
                    "recording_group_id": f"recording-{role}",
                    "video_key": f"video-{role}-{int(unit_id.split('-')[-1]) // 10}",
                    "source_type": "legacy_recovered",
                    "dataset_id": "legacy_recovered_16f",
                    "behavior_label": label,
                    "l5_role": role,
                    "lineage_scope": LINEAGE_SCOPE,
                    "human_review_complete": False,
                }
            )
            targets.append(label_to_index[label])
            weights.append(1.0 / multiplier)
    windows = pd.DataFrame.from_records(rows)
    feature_path = tmp_path / f"{view_id}_features.npy"
    if materialize_features:
        features = np.random.default_rng(7).normal(
            size=(sequence_length, 512)
        ).astype(np.float32)
        np.save(feature_path, features, allow_pickle=False)
    feature_rows = np.tile(
        np.arange(sequence_length, dtype=np.int64),
        (len(windows), 1),
    )
    return LegacyL5CachedFeatureView(
        feature_tensor_path=feature_path,
        feature_tensor_sha256="0" * 64,
        control_id="V1",
        temporal_view_name=str(expected["temporal_view_name"]),
        sequence_length=sequence_length,
        windows=windows,
        fold_manifest=pd.DataFrame(),
        feature_rows=feature_rows,
        observed_mask=np.ones((len(windows), sequence_length), dtype=bool),
        time_delta=np.tile(
            np.arange(sequence_length, dtype=np.float32),
            (len(windows), 1),
        ),
        targets=np.asarray(targets, dtype=np.int64),
        sample_weights=np.asarray(weights, dtype=np.float64),
        audit={},
    )


def _window_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for unit_index, label in enumerate(("drink", "eat")):
        target_index = VALID_BEHAVIORS.index(label)
        for window_index in range(4):
            probabilities = np.full(10, 0.01 / 9.0)
            probabilities[target_index] = 0.99
            row: dict[str, object] = {
                "window_id": f"window-{unit_index}-{window_index}",
                "temporal_unit_key": f"unit-{unit_index}",
                "recording_group_id": "recording-a",
                "video_key": f"video-{unit_index}",
                "source_type": "legacy_recovered",
                "dataset_id": "legacy_recovered_16f",
                "behavior_label": label,
                "target_index": target_index,
                "sample_weight": 0.25,
            }
            row.update(
                {
                    "prob_" + behavior.replace("-", "_"): float(
                        probabilities[index]
                    )
                    for index, behavior in enumerate(VALID_BEHAVIORS)
                }
            )
            rows.append(row)
    return pd.DataFrame.from_records(rows)
