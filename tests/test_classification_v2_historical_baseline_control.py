from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pig_behavior.classification_v2.experiments.historical_baseline import (
    FULL_CONTROL_STATUS,
    HistoricalFullOOFConfig,
    LegacySequenceCheckpointConfig,
    build_historical_baseline_reconciliation,
    check_historical_baseline_reconciliation,
    write_historical_baseline_reconciliation,
)

ORIGIN_COMMIT = "1" * 40
FIX_COMMIT = "2" * 40


def test_historical_control_reproduces_defect_without_promoting_metrics(
    tmp_path: Path,
) -> None:
    config = _full_config(tmp_path)

    payload = build_historical_baseline_reconciliation(config)

    full = payload["historical_full_oof"]
    assert payload["valid"] is True
    assert full["status"] == FULL_CONTROL_STATUS
    assert full["performance_evidence_valid"] is False
    assert all(value is False for value in full["claim_flags"].values())
    assert full["alignment_evidence"]["defect_reproduced"] is True
    assert full["diagnostic_metrics_snapshot"]["status"] == (
        "INVALID_FOR_MODEL_QUALITY"
    )
    assert full["artifact_hash_scope"] == "registration_time_integrity_only"
    assert full["known_lineage_gaps"]
    assert payload["io_scope"]["full_manifest_window_id_scan"] is True
    assert payload["io_scope"]["raw_data_read"] is False
    assert payload["optimizer_steps"] == 0


def test_historical_control_rejects_unexpected_alignment(
    tmp_path: Path,
) -> None:
    config = _full_config(tmp_path, expected_mismatch_rows=2)

    payload = build_historical_baseline_reconciliation(config)

    assert payload["valid"] is False
    assert any(
        "expected_historical_alignment_defect_not_reproduced" in error
        for error in payload["errors"]
    )


def test_saved_historical_control_detects_artifact_hash_drift(
    tmp_path: Path,
) -> None:
    config = _full_config(tmp_path)
    output = tmp_path / "historical_control.json"
    payload = build_historical_baseline_reconciliation(config)
    write_historical_baseline_reconciliation(payload, output)
    with pytest.raises(FileExistsError, match="already exist"):
        write_historical_baseline_reconciliation(payload, output)
    config.metrics_json.write_text("{}", encoding="utf-8")

    result = check_historical_baseline_reconciliation(output)

    assert result["valid"] is False
    assert any("artifact_sha256_drift" in error for error in result["errors"])


def test_legacy_checkpoint_is_architecture_only_and_loaded_safely(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "legacy.pt"
    state = {
        "head.3.weight": torch.zeros((10, 32)),
        "head.0.weight": torch.zeros((32, 32)),
        "cnn_proj.weight": torch.zeros((22, 512)),
        "transformer.layers.1.linear1.weight": torch.zeros((64, 32)),
        "cnn.6.5.conv1.weight": torch.zeros((1, 1, 1, 1)),
    }
    torch.save({"model_state": state}, checkpoint)
    expected_hash = _sha256(checkpoint)
    full_config = _full_config(tmp_path / "full")

    payload = build_historical_baseline_reconciliation(
        full_config,
        LegacySequenceCheckpointConfig(checkpoint, expected_hash),
    )

    legacy = payload["legacy_sequence_checkpoint"]
    assert payload["valid"] is True
    assert legacy["performance_evidence_valid"] is False
    assert legacy["safe_load_policy"] == "torch_load_weights_only_true"
    assert legacy["model_spec"] == {
        "num_classes": 10,
        "d_model": 32,
        "extra_dim": 10,
        "num_layers": 2,
        "backbone_name": "resnet34",
    }
    assert legacy["state_tensor_element_count"] > 0
    assert len(legacy["label_order"]) == 10


def test_historical_control_rejects_origin_commit_drift(tmp_path: Path) -> None:
    config = _full_config(tmp_path, origin_commit="3" * 40)

    payload = build_historical_baseline_reconciliation(config)

    assert payload["valid"] is False
    assert any("origin_git_commit_mismatch" in error for error in payload["errors"])


def _full_config(
    root: Path,
    *,
    expected_mismatch_rows: int = 4,
    origin_commit: str = ORIGIN_COMMIT,
) -> HistoricalFullOOFConfig:
    root.mkdir(parents=True, exist_ok=True)
    split = root / "split.csv"
    image = root / "image.csv"
    interaction = root / "interaction.csv"
    pd.DataFrame({"window_id": ["a", "b", "c", "d"]}).to_csv(
        split,
        index=False,
    )
    pd.DataFrame({"window_id": ["b", "a", "d", "c"]}).to_csv(
        image,
        index=False,
    )
    pd.DataFrame({"window_id": ["b", "a", "d", "c"]}).to_csv(
        interaction,
        index=False,
    )

    fold_root = root / "folds"
    fold_work = fold_root / "native_oof_000_work"
    fold_work.mkdir(parents=True)
    (fold_work / "trained_model.pt").write_bytes(b"checkpoint")
    (fold_work / "training_audit.json").write_text("{}", encoding="utf-8")
    run_audit = root / "run.json"
    run_audit.write_text(
        json.dumps(
            {
                "git_commit": ORIGIN_COMMIT,
                "run_mode": "full",
                "full_oof_training_verified": True,
                "prediction_rows": 4,
                "native_temporal_rows": 2,
                "load_audit": {"row_counts": {"split": 4}},
                "fold_audits": [{"oof_fold_id": "native_oof_000"}],
            }
        ),
        encoding="utf-8",
    )
    metrics = root / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "native_temporal_metrics": {
                    "rows": 2,
                    "accuracy": 0.5,
                    "macro_f1": 0.4,
                    "macro_recall": 0.45,
                }
            }
        ),
        encoding="utf-8",
    )
    schema = root / "schema.json"
    schema.write_text(json.dumps({"prediction_rows": 4}), encoding="utf-8")
    window_predictions = root / "window_predictions.csv"
    window_predictions.write_text("window_id\na\nb\nc\nd\n", encoding="utf-8")
    native_predictions = root / "native_predictions.csv"
    native_predictions.write_text("unit_id\nu1\nu2\n", encoding="utf-8")
    return HistoricalFullOOFConfig(
        split_manifest_csv=split,
        image_manifest_csv=image,
        interaction_manifest_csv=interaction,
        run_audit_json=run_audit,
        metrics_json=metrics,
        prediction_schema_json=schema,
        window_predictions_csv=window_predictions,
        native_predictions_csv=native_predictions,
        fold_artifact_dir=fold_root,
        origin_git_commit=origin_commit,
        alignment_fix_commit=FIX_COMMIT,
        expected_manifest_rows=4,
        expected_positional_mismatch_rows=expected_mismatch_rows,
    )


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
