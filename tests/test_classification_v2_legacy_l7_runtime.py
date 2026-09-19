from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from pig_behavior.classification_v2.training import (
    legacy_development_l7_imbalance_runtime as runtime,
)
from pig_behavior.classification_v2.training.imbalance_losses import LOSS_POLICIES


def test_partial_artifact_manifest_is_explicit(tmp_path: Path) -> None:
    paths = {
        name: tmp_path / filename
        for name, filename in runtime.ARTIFACT_FILES.items()
    }
    paths["preflight"].write_text("{}\n", encoding="utf-8")

    manifest = runtime._artifact_manifest(
        paths,
        run_id="failed-run",
        policy="event_balanced_ce",
        allow_partial=True,
    )

    assert manifest["status"] == "partial"
    assert manifest["valid"] is False
    assert manifest["artifact_count"] == 1
    assert "loss_fit" in manifest["missing_artifacts"]
    assert "artifact_missing=loss_fit" in manifest["errors"]


def test_matrix_uses_shared_loss_fit_source_hash(tmp_path: Path) -> None:
    config = SimpleNamespace(
        sha256="c" * 64,
        training_scope="short_repeat_gate",
    )
    gate_paths: dict[str, Path] = {}
    for index, policy in enumerate(LOSS_POLICIES):
        primary = _result(config.sha256, policy, process_id=100 + index)
        repeat = _result(config.sha256, policy, process_id=200 + index)
        payload = {
            "schema_version": runtime.REPEAT_GATE_SCHEMA,
            "status": runtime.PASS_REPEAT_STATUS,
            "loss_policy": policy,
            "short_config_sha256": config.sha256,
            "full_matrix_authorized": True,
            "valid": True,
            "primary": {"result": primary},
            "repeat": {"result": repeat},
        }
        path = tmp_path / f"{policy}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        gate_paths[policy] = path

    matrix = runtime.audit_l7_imbalance_short_matrix(
        config,
        repeat_gate_paths=gate_paths,
    )

    assert matrix["valid"] is True
    assert matrix["loss_fit_audit_sha256"] == "s" * 64


def _result(config_sha256: str, policy: str, *, process_id: int) -> dict[str, object]:
    return {
        "config_sha256": config_sha256,
        "loss_policy": policy,
        "selection_content_sha256": "a" * 64,
        "loss_fit_audit_sha256": ("a" if policy == "event_balanced_ce" else "b") * 64,
        "loss_fit_source_sha256": "s" * 64,
        "loss_state_sha256": ("x" if policy == "event_balanced_ce" else "y") * 64,
        "parameter_sha256": "p" * 64,
        "window_prediction_sha256": "w" * 64,
        "native_prediction_sha256": "n" * 64,
        "epoch_metrics_sha256": "e" * 64,
        "process_id": process_id,
    }
