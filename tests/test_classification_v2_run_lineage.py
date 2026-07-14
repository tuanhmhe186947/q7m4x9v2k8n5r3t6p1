from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from pig_behavior.classification_v2.models.multitask_fusion import (
    MULTITASK_ARCHITECTURE_VERSION,
)
from pig_behavior.classification_v2.training import run_identity as identity_module
from pig_behavior.classification_v2.training import run_lineage as lineage_module
from pig_behavior.classification_v2.training.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    RUN_IDENTITY_REQUIRED_FIELDS,
)
from pig_behavior.classification_v2.training.config import (
    ClassificationV2TrainingConfig,
    DatasetConfig,
    ExecutionConfig,
    LossConfig,
    ModelConfig,
    OptimizationConfig,
)
from pig_behavior.classification_v2.training.run_lineage import (
    PredictionArtifact,
    finalize_run_lineage,
    initialize_run_lineage,
    merge_registry_entries,
)
from pig_behavior.classification_v2.training.run_lineage_audit import (
    audit_run_lineage,
)
from pig_behavior.classification_v2.training.trainer import training_run_dir

ARTIFACT_CONSUMING_TRAINING_CALLERS = (
    Path("scripts")
    / "classification_v2"
    / "04_baselines_smokes"
    / "check_classification_v2_behavior_only_baselines.py",
    Path("scripts")
    / "classification_v2"
    / "04_baselines_smokes"
    / "check_classification_v2_training_reproducibility.py",
    Path("scripts")
    / "classification_v2"
    / "04_baselines_smokes"
    / "classification_v2_run_q2_baseline_smokes.py",
    Path("scripts")
    / "classification_v2"
    / "07_postrun_evaluation"
    / "classification_v2_estimate_b4_seed_variance.py",
)


def test_training_run_dir_uses_lineage_owned_directory(tmp_path: Path) -> None:
    expected = tmp_path / "fold-0" / "run-a"

    assert training_run_dir({"run_lineage": {"run_dir": str(expected)}}) == expected
    for audit in ({}, {"run_lineage": {}}, {"run_lineage": {"run_dir": ""}}):
        with pytest.raises(ValueError, match="run_lineage|run_dir"):
            training_run_dir(audit)


def test_artifact_consuming_training_callers_use_lineage_resolver() -> None:
    root = Path(__file__).parents[1]

    for relative_path in ARTIFACT_CONSUMING_TRAINING_CALLERS:
        tree = ast.parse((root / relative_path).read_text(encoding="utf-8"))
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "run_training" in called_names, relative_path
        assert "training_run_dir" in called_names, relative_path


def test_initialize_writes_complete_planned_packet(tmp_path: Path) -> None:
    config, snapshot = _fixture(tmp_path)

    session = initialize_run_lineage(
        config,
        snapshot_check=snapshot,
        environment=_environment(),
    )

    expected = {
        "run_manifest.json",
        "environment.json",
        "artifact_manifest.json",
        "checkpoint_manifest.json",
        "prediction_manifest.json",
        "resolved_config.json",
    }
    assert expected.issubset(path.name for path in session.run_dir.iterdir())
    assert session.run_dir.parent.name == config.execution.fold_id
    assert config.execution.runs_registry_csv.exists()


def test_initialize_skips_absent_optional_snapshot_artifact(
    tmp_path: Path,
) -> None:
    config, snapshot = _fixture(tmp_path)
    optional_path = tmp_path / "optional-context.bin"
    snapshot["current"]["artifacts"]["optional_context"] = {
        "path": str(optional_path.resolve()),
        "type": "binary",
        "required": False,
        "exists": False,
    }

    session = initialize_run_lineage(
        config,
        snapshot_check=snapshot,
        environment=_environment(),
    )

    names = {item["name"] for item in session.input_artifacts}
    assert "optional_context" not in names


def test_existing_run_requires_resume_and_exact_identity(tmp_path: Path) -> None:
    config, snapshot = _fixture(tmp_path)
    first = initialize_run_lineage(
        config,
        snapshot_check=snapshot,
        environment=_environment(),
    )
    manifest_before = (first.run_dir / "run_manifest.json").read_bytes()
    no_resume = replace(
        config,
        execution=replace(config.execution, resume=False),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        initialize_run_lineage(
            no_resume,
            snapshot_check=snapshot,
            environment=_environment(),
        )

    drifted = replace(
        config,
        model=replace(config.model, hidden_dim=99),
    )
    with pytest.raises(ValueError, match="resume run identity mismatch"):
        initialize_run_lineage(
            drifted,
            snapshot_check=snapshot,
            environment=_environment(),
        )
    freeze_drift = replace(
        config,
        model=replace(
            config.model,
            visual_backbone_lr_multiplier=0.5,
        ),
    )
    with pytest.raises(ValueError, match="resume run identity mismatch"):
        initialize_run_lineage(
            freeze_drift,
            snapshot_check=snapshot,
            environment=_environment(),
        )
    assert (first.run_dir / "run_manifest.json").read_bytes() == manifest_before

    resumed = initialize_run_lineage(
        config,
        snapshot_check=snapshot,
        environment={**_environment(), "gpu_model": "resume-device"},
    )
    packet = json.loads((resumed.run_dir / "environment.json").read_text(encoding="utf-8"))
    assert packet["initial"]["gpu_model"] == "NONE"
    assert packet["resume_events"][0]["environment"]["gpu_model"] == ("resume-device")


def test_resume_rejects_dataset_and_cache_hash_drift(tmp_path: Path) -> None:
    config, snapshot = _fixture(tmp_path)
    session = initialize_run_lineage(
        config,
        snapshot_check=snapshot,
        environment=_environment(),
    )

    config.dataset.snapshot_json.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset_snapshot_sha256"):
        initialize_run_lineage(
            config,
            snapshot_check=snapshot,
            environment=_environment(),
        )
    config.dataset.snapshot_json.write_text("snapshot", encoding="utf-8")

    cache_drift = json.loads(json.dumps(snapshot))
    actor = cache_drift["current"]["artifacts"]["actor_packed_cache"]
    actor["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="cache_sha256"):
        initialize_run_lineage(
            config,
            snapshot_check=cache_drift,
            environment=_environment(),
        )
    assert session.terminal is False


def test_finalize_links_prediction_to_exact_checkpoint(tmp_path: Path) -> None:
    config, snapshot = _fixture(tmp_path)
    session = initialize_run_lineage(
        config,
        snapshot_check=snapshot,
        environment=_environment(),
    )
    checkpoint = _checkpoint(session.run_dir, session.identity.to_payload())
    prediction = _prediction(session.run_dir, rows=2)
    metric = session.run_dir / "metrics.json"
    metric.write_text('{"macro_f1": 0.5}', encoding="utf-8")

    result = finalize_run_lineage(
        session,
        checkpoint_paths=[checkpoint],
        predictions=[
            PredictionArtifact(
                path=prediction,
                checkpoint_path=checkpoint,
                split="test",
                expected_rows=2,
            )
        ],
        metric_paths=[metric],
        runtime_seconds=2.5,
        peak_vram_bytes=123,
    )

    manifest = json.loads(
        (session.run_dir / "prediction_manifest.json").read_text(encoding="utf-8")
    )
    assert result["valid"] is True
    assert Path(result["run_dir"]) == session.run_dir
    assert manifest["predictions"][0]["checkpoint_path"] == str(checkpoint.resolve())
    registry = pd.read_csv(config.execution.runs_registry_csv)
    assert registry["run_id"].tolist() == [session.identity.run_id]
    assert registry["status"].tolist() == ["completed"]
    assert registry["visual_freeze_policy"].tolist() == ["all_trainable"]
    assert registry["visual_backbone_lr_multiplier"].tolist() == [1.0]


def test_registry_append_failure_preserves_completed_packet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, snapshot = _fixture(tmp_path)
    session = initialize_run_lineage(
        config,
        snapshot_check=snapshot,
        environment=_environment(),
    )
    checkpoint = _checkpoint(session.run_dir, session.identity.to_payload())
    prediction = _prediction(session.run_dir, rows=1)

    def reject_append(_path: Path, _entry: dict[str, object]) -> None:
        raise OSError("synthetic registry failure")

    monkeypatch.setattr(
        lineage_module,
        "_append_registry_entry",
        reject_append,
    )
    with pytest.raises(RuntimeError, match="packet completed"):
        with session:
            finalize_run_lineage(
                session,
                checkpoint_paths=[checkpoint],
                predictions=[PredictionArtifact(prediction, checkpoint, "test", 1)],
                metric_paths=[],
            )

    manifest = json.loads((session.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert session.terminal is True
    assert manifest["status"] == "completed"
    audit = audit_run_lineage(session.run_dir)
    assert audit["packet_integrity_valid"] is True
    assert audit["registry_registered"] is False
    assert audit["run_succeeded"] is False

    recovered_registry = tmp_path / "recovered" / "runs_registry.csv"
    merge_registry_entries(
        [session.run_dir / "registry_entry.json"],
        recovered_registry,
    )
    recovered = audit_run_lineage(
        session.run_dir,
        registry_csv=recovered_registry,
    )
    assert recovered["registry_registered"] is True
    assert recovered["run_succeeded"] is True


def test_failure_registry_error_does_not_mask_training_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, snapshot = _fixture(tmp_path)
    session = initialize_run_lineage(
        config,
        snapshot_check=snapshot,
        environment=_environment(),
    )

    def reject_append(_path: Path, _entry: dict[str, object]) -> None:
        raise OSError("synthetic registry failure")

    monkeypatch.setattr(
        lineage_module,
        "_append_registry_entry",
        reject_append,
    )
    with pytest.raises(RuntimeError, match="original training failure"):
        with session:
            raise RuntimeError("original training failure")

    manifest = json.loads((session.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "original training failure" in manifest["failure_reason"]


def test_prediction_without_declared_checkpoint_is_rejected(tmp_path: Path) -> None:
    config, snapshot = _fixture(tmp_path)
    session = initialize_run_lineage(
        config,
        snapshot_check=snapshot,
        environment=_environment(),
    )
    checkpoint = _checkpoint(session.run_dir, session.identity.to_payload())
    other = session.run_dir / "other.pt"
    other.write_bytes(b"other")
    prediction = _prediction(session.run_dir, rows=1)

    with pytest.raises(ValueError, match="not in checkpoint manifest"):
        finalize_run_lineage(
            session,
            checkpoint_paths=[checkpoint],
            predictions=[
                PredictionArtifact(
                    path=prediction,
                    checkpoint_path=other,
                    split="test",
                    expected_rows=1,
                )
            ],
            metric_paths=[],
        )


def test_terminal_registry_row_is_immutable(tmp_path: Path) -> None:
    config, snapshot = _fixture(tmp_path)
    session = initialize_run_lineage(
        config,
        snapshot_check=snapshot,
        environment=_environment(),
    )
    checkpoint = _checkpoint(session.run_dir, session.identity.to_payload())
    prediction = _prediction(session.run_dir, rows=1)
    finalize_run_lineage(
        session,
        checkpoint_paths=[checkpoint],
        predictions=[PredictionArtifact(prediction, checkpoint, "test", 1)],
        metric_paths=[],
    )
    registry_before = config.execution.runs_registry_csv.read_bytes()

    with pytest.raises(FileExistsError, match="terminal run"):
        initialize_run_lineage(
            config,
            snapshot_check=snapshot,
            environment=_environment(),
        )
    assert config.execution.runs_registry_csv.read_bytes() == registry_before


def test_packet_audit_detects_prediction_tampering(tmp_path: Path) -> None:
    config, snapshot = _fixture(tmp_path)
    session = initialize_run_lineage(
        config,
        snapshot_check=snapshot,
        environment=_environment(),
    )
    checkpoint = _checkpoint(session.run_dir, session.identity.to_payload())
    prediction = _prediction(session.run_dir, rows=1)
    finalize_run_lineage(
        session,
        checkpoint_paths=[checkpoint],
        predictions=[PredictionArtifact(prediction, checkpoint, "test", 1)],
        metric_paths=[],
    )

    assert audit_run_lineage(session.run_dir)["run_succeeded"] is True
    prediction.write_text("tampered", encoding="utf-8")
    audit = audit_run_lineage(session.run_dir)
    assert audit["integrity_valid"] is False
    assert any("prediction_" in error for error in audit["errors"])


def test_packet_audit_detects_resolved_config_tampering(tmp_path: Path) -> None:
    config, snapshot = _fixture(tmp_path)
    session = initialize_run_lineage(
        config,
        snapshot_check=snapshot,
        environment=_environment(),
    )
    checkpoint = _checkpoint(session.run_dir, session.identity.to_payload())
    prediction = _prediction(session.run_dir, rows=1)
    finalize_run_lineage(
        session,
        checkpoint_paths=[checkpoint],
        predictions=[PredictionArtifact(prediction, checkpoint, "test", 1)],
        metric_paths=[],
    )
    config_path = session.run_dir / "resolved_config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["optimization"]["seed"] = 999
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    audit = audit_run_lineage(session.run_dir)

    assert audit["integrity_valid"] is False
    assert "resolved_config_sha256_mismatch" in audit["errors"]


def test_registry_merge_checks_all_collisions_before_append(
    tmp_path: Path,
) -> None:
    config, snapshot = _fixture(tmp_path / "source")
    session = initialize_run_lineage(
        config,
        snapshot_check=snapshot,
        environment=_environment(),
    )
    with pytest.raises(RuntimeError):
        with session:
            raise RuntimeError("record failure")
    entry = session.run_dir / "registry_entry.json"
    target = tmp_path / "merged" / "runs_registry.csv"

    result = merge_registry_entries([entry], target)
    before = target.read_bytes()
    assert result["entry_count"] == 1
    with pytest.raises(FileExistsError, match="collisions"):
        merge_registry_entries([entry], target)
    assert target.read_bytes() == before


def test_registry_merge_validates_every_schema_before_creating_target(
    tmp_path: Path,
) -> None:
    config, snapshot = _fixture(tmp_path / "source")
    session = initialize_run_lineage(
        config,
        snapshot_check=snapshot,
        environment=_environment(),
    )
    with pytest.raises(RuntimeError):
        with session:
            raise RuntimeError("record failure")
    good = session.run_dir / "registry_entry.json"
    malformed = tmp_path / "malformed-entry.json"
    payload = json.loads(good.read_text(encoding="utf-8"))
    payload.pop("config_sha256")
    malformed.write_text(json.dumps(payload), encoding="utf-8")
    target = tmp_path / "merged" / "runs_registry.csv"

    with pytest.raises(ValueError, match="schema mismatch"):
        merge_registry_entries([good, malformed], target)

    assert not target.exists()


def test_lineage_cli_dry_runs_do_not_write_outputs(tmp_path: Path) -> None:
    config, snapshot = _fixture(tmp_path / "source")
    session = initialize_run_lineage(
        config,
        snapshot_check=snapshot,
        environment=_environment(),
    )
    checkpoint = _checkpoint(session.run_dir, session.identity.to_payload())
    prediction = _prediction(session.run_dir, rows=1)
    finalize_run_lineage(
        session,
        checkpoint_paths=[checkpoint],
        predictions=[PredictionArtifact(prediction, checkpoint, "test", 1)],
        metric_paths=[],
    )
    root = Path(__file__).parents[1]
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}
    audit_output = tmp_path / "lineage-audit.json"
    checker = (
        root
        / "scripts"
        / "classification_v2"
        / ("04_baselines_smokes/check_classification_v2_run_lineage.py")
    )
    checked = subprocess.run(
        [
            sys.executable,
            str(checker),
            "--run-dir",
            str(session.run_dir),
            "--output-json",
            str(audit_output),
            "--require-success",
            "--dry-run",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stderr
    assert not audit_output.exists()

    merged = tmp_path / "merged-registry.csv"
    merger = (
        root
        / "scripts"
        / "classification_v2"
        / ("06_full_oof_training/classification_v2_merge_run_registry.py")
    )
    merge_check = subprocess.run(
        [
            sys.executable,
            str(merger),
            "--entry-json",
            str(session.run_dir / "registry_entry.json"),
            "--registry-csv",
            str(merged),
            "--dry-run",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert merge_check.returncode == 0, merge_check.stderr
    assert not merged.exists()


def test_context_preserves_failure_reason(tmp_path: Path) -> None:
    config, snapshot = _fixture(tmp_path)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        with initialize_run_lineage(
            config,
            snapshot_check=snapshot,
            environment=_environment(),
        ):
            raise RuntimeError("synthetic failure")

    run_dir = next((config.execution.output_dir / "fold-0").iterdir())
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "synthetic failure" in manifest["failure_reason"]
    registry = pd.read_csv(config.execution.runs_registry_csv)
    assert registry["status"].tolist() == ["failed"]


def test_automatic_run_ids_isolate_independent_folds(tmp_path: Path) -> None:
    config, snapshot = _fixture(tmp_path, run_id=None)
    fold_zero = initialize_run_lineage(
        config,
        snapshot_check=snapshot,
        environment=_environment(),
    )
    fold_one_config = replace(
        config,
        execution=replace(config.execution, fold_id="fold-1"),
    )
    fold_one = initialize_run_lineage(
        fold_one_config,
        snapshot_check=snapshot,
        environment=_environment(),
    )

    assert fold_zero.run_dir != fold_one.run_dir
    assert fold_zero.identity.run_id != fold_one.identity.run_id
    assert fold_zero.run_dir.parent.name == "fold-0"
    assert fold_one.run_dir.parent.name == "fold-1"


def test_automatic_run_id_changes_with_code_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, snapshot = _fixture(tmp_path, run_id=None)
    monkeypatch.setattr(
        identity_module,
        "_git_state",
        lambda: ("a" * 40, False, "1" * 64),
    )
    first = initialize_run_lineage(
        config,
        snapshot_check=snapshot,
        environment=_environment(),
    )
    monkeypatch.setattr(
        identity_module,
        "_git_state",
        lambda: ("b" * 40, False, "2" * 64),
    )
    second = initialize_run_lineage(
        config,
        snapshot_check=snapshot,
        environment=_environment(),
    )

    assert first.identity.run_id != second.identity.run_id
    assert first.run_dir != second.run_dir


def _fixture(
    tmp_path: Path,
    *,
    run_id: str | None = "run-a",
) -> tuple[ClassificationV2TrainingConfig, dict[str, object]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {
        name: tmp_path / f"{name}.bin"
        for name in [
            "snapshot",
            "trainer_contract",
            "actor_packed_cache",
            "actor_packed_index",
            "visual_cache_manifest",
            "visual_packed_cache",
            "visual_packed_index",
            "native_folds",
            "grouped_roles",
            "temporal_selection",
            "temporal_manifest",
            "auxiliary",
            "fold_event_weights",
        ]
    }
    for name, path in paths.items():
        path.write_text(name, encoding="utf-8")
    paths["snapshot"].write_text("snapshot", encoding="utf-8")
    dataset = DatasetConfig(
        snapshot_json=paths["snapshot"],
        trainer_contract_json=paths["trainer_contract"],
        train_ready_root=tmp_path,
        actor_packed_cache=paths["actor_packed_cache"],
        actor_packed_index=paths["actor_packed_index"],
        visual_cache_manifest=paths["visual_cache_manifest"],
        visual_packed_cache=paths["visual_packed_cache"],
        visual_packed_index=paths["visual_packed_index"],
        native_oof_fold_manifest=paths["native_folds"],
        grouped_fold_roles=paths["grouped_roles"],
        temporal_view_selection_manifest=paths["temporal_selection"],
        temporal_view_manifest=paths["temporal_manifest"],
        auxiliary_targets_csv=paths["auxiliary"],
        fold_event_weight_manifest=paths["fold_event_weights"],
    )
    config = ClassificationV2TrainingConfig(
        version="classification_v2_training_config_v1",
        dataset=dataset,
        model=ModelConfig(
            architecture_version=MULTITASK_ARCHITECTURE_VERSION,
        ),
        optimization=OptimizationConfig(seed=17),
        loss=LossConfig(),
        execution=ExecutionConfig(
            experiment_name="lineage-test",
            fold_id="fold-0",
            run_id=run_id,
            output_dir=tmp_path / "runs",
            runs_registry_csv=tmp_path / "registry" / "runs_registry.csv",
        ),
    )
    artifacts = {
        name: {
            "path": str(path.resolve()),
            "type": "binary",
            "required": True,
            "exists": True,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for name, path in paths.items()
        if name != "snapshot"
    }
    snapshot = {
        "valid": True,
        "current_snapshot_id": "snapshot-id",
        "current": {"artifacts": artifacts},
        "errors": [],
    }
    return config, snapshot


def _checkpoint(run_dir: Path, identity: dict[str, object]) -> Path:
    path = run_dir / "best_validation.pt"
    path.write_bytes(b"checkpoint")
    lineage = {key: identity[key] for key in RUN_IDENTITY_REQUIRED_FIELDS}
    audit = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "lineage": lineage,
        "valid": True,
        "errors": [],
    }
    path.with_suffix(path.suffix + ".audit.json").write_text(
        json.dumps(audit),
        encoding="utf-8",
    )
    return path


def _prediction(run_dir: Path, *, rows: int) -> Path:
    path = run_dir / "oof_test_predictions.csv"
    pd.DataFrame(
        {
            "window_id": [f"window-{index}" for index in range(rows)],
            "y_true": ["stand"] * rows,
            "y_pred": ["stand"] * rows,
            "prediction_split": ["test"] * rows,
        }
    ).to_csv(path, index=False)
    return path


def _environment() -> dict[str, object]:
    return {
        "captured_at_utc": "2026-07-13T00:00:00+00:00",
        "os": "test",
        "python_version": "3.11",
        "python_executable": "python",
        "torch_version": "test",
        "torchvision_version": "test",
        "cuda_available": False,
        "cuda_version": None,
        "cudnn_version": None,
        "gpu_model": "NONE",
        "gpu_vram_bytes": 0,
        "device_count": 0,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
