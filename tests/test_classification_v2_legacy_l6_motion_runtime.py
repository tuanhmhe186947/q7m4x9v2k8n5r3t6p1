from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from pig_behavior.classification_v2.training import (
    legacy_development_l6_cached_modality_runtime as runtime,
)
from pig_behavior.classification_v2.training.legacy_development_l6_cached_modality_runtime import (
    _audit_artifacts,
    _build_artifact_manifest,
    _finalize_run,
    _non_overlapping_intervals,
    _run_paths,
    _safe_run_id,
    _validate_run_path_lengths,
)
from pig_behavior.classification_v2.training.legacy_development_l6_motion import (
    LINEAGE_SCOPE,
    MODES,
    SHORT_SCOPE,
    LegacyL6MotionConfig,
)
from pig_behavior.classification_v2.training.legacy_development_l6_motion_runtime import (
    MOTION_RUNTIME_SPEC,
    PASS_REPEAT_STATUS,
    REPEAT_GATE_SCHEMA,
    audit_motion_short_matrix,
)


def _short_config(tmp_path: Path) -> LegacyL6MotionConfig:
    path = tmp_path / "short.json"
    path.write_text("{}\n", encoding="utf-8")
    return LegacyL6MotionConfig(
        path=path,
        payload={
            "training_scope": SHORT_SCOPE,
            "output": {"run_root_relative_path": "short"},
        },
        repo_root=tmp_path,
    )


def _repeat_gate(
    config: LegacyL6MotionConfig,
    *,
    mode: str,
    primary_pid: int,
    repeat_pid: int,
) -> dict[str, object]:
    summary = {
        "run_id": "run",
        "result_path": "result.json",
        "result_sha256": "a" * 64,
        "run_manifest_sha256": "b" * 64,
        "artifact_manifest_sha256": "c" * 64,
        "started_at_utc": "2026-07-15T00:00:00+00:00",
        "completed_at_utc": "2026-07-15T00:01:00+00:00",
        "runtime_seconds": 60.0,
        "selection_content_sha256": "d" * 64,
        "normalization_state_sha256": "e" * 64,
        "parameter_sha256": "f" * 64,
        "native_prediction_sha256": "0" * 64,
        "valid": True,
    }
    primary = {**summary, "process_id": primary_pid}
    repeat = {
        **summary,
        "run_id": "repeat",
        "process_id": repeat_pid,
        "started_at_utc": "2026-07-15T00:02:00+00:00",
        "completed_at_utc": "2026-07-15T00:03:00+00:00",
    }
    return {
        "schema_version": REPEAT_GATE_SCHEMA,
        "status": PASS_REPEAT_STATUS,
        "lineage_scope": LINEAGE_SCOPE,
        "training_scope": SHORT_SCOPE,
        "mode": mode,
        "short_config_sha256": config.sha256,
        "full_mode_expansion_authorized": True,
        "primary": primary,
        "repeat": repeat,
        "errors": [],
        "valid": True,
    }


def test_runtime_run_id_and_non_overlap_guards() -> None:
    assert _safe_run_id("motion_repeat-01")
    assert not _safe_run_id("../escape")
    start = datetime(2026, 7, 15, tzinfo=timezone.utc)
    primary = {
        "started_at_utc": start.isoformat(),
        "completed_at_utc": (start + timedelta(minutes=1)).isoformat(),
    }
    repeat = {
        "started_at_utc": (start + timedelta(minutes=2)).isoformat(),
        "completed_at_utc": (start + timedelta(minutes=3)).isoformat(),
    }

    assert _non_overlapping_intervals(primary, repeat)["valid"] is True
    overlapping = copy.deepcopy(repeat)
    overlapping["started_at_utc"] = (
        start + timedelta(seconds=30)
    ).isoformat()
    assert _non_overlapping_intervals(primary, overlapping)["valid"] is False


def test_windows_path_length_fails_before_run_creation(tmp_path: Path) -> None:
    config = _short_config(tmp_path)
    audit = _validate_run_path_lengths(
        MOTION_RUNTIME_SPEC,
        config,
        mode="motion",
        run_id="run",
    )

    assert audit["valid"] is True
    if os.name == "nt":
        long_config = LegacyL6MotionConfig(
            path=config.path,
            payload={
                "training_scope": SHORT_SCOPE,
                "output": {"run_root_relative_path": "x" * 220},
            },
            repo_root=tmp_path,
        )
        with pytest.raises(ValueError, match="Windows artifact path too long"):
            _validate_run_path_lengths(
                MOTION_RUNTIME_SPEC,
                long_config,
                mode="motion",
                run_id="run",
            )


def test_artifact_manifest_detects_hash_drift(tmp_path: Path) -> None:
    artifact_path = tmp_path / "metrics.json"
    artifact_path.write_text('{"valid":true}\n', encoding="utf-8")
    paths = {
        "run_manifest": tmp_path / "run_manifest.json",
        "artifact_manifest": tmp_path / "artifact_manifest.json",
        "metrics": artifact_path,
    }
    manifest = _build_artifact_manifest(
        MOTION_RUNTIME_SPEC,
        paths,
        run_id="run",
        mode="motion",
        status="completed",
    )

    assert _audit_artifacts(tmp_path, manifest)["valid"] is True
    artifact_path.write_text('{"valid":false}\n', encoding="utf-8")
    audit = _audit_artifacts(tmp_path, manifest)
    assert audit["valid"] is False
    assert audit["errors"] == ["artifact_hash_mismatch=metrics"]


def test_finalization_error_becomes_failed_packet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _short_config(tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    paths = _run_paths(MOTION_RUNTIME_SPEC, root)
    paths["run_manifest"].write_text("{}\n", encoding="utf-8")

    def _fail_write(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("synthetic artifact path failure")

    monkeypatch.setattr(runtime, "_write_outcome", _fail_write)
    result = _finalize_run(
        MOTION_RUNTIME_SPEC,
        paths,
        config=config,
        mode="motion",
        run_id="run",
        planned={"started_at_utc": "2026-07-15T00:00:00+00:00"},
        planned_sha="a" * 64,
        selection={"selection_content_sha256": "b" * 64},
        outcome=object(),
        execution={"errors": [], "valid": True},
        failure=None,
        runtime_seconds=1.0,
    )

    assert result["valid"] is False
    assert result["failure"]["failure_stage"] == "artifact_finalization"
    assert paths["unexpected_failure"].is_file()
    assert paths["run_result"].is_file()
    assert paths["artifact_manifest"].is_file()
    final_manifest = json.loads(
        paths["run_manifest"].read_text(encoding="utf-8")
    )
    assert final_manifest["status"] == "failed"


def test_short_matrix_requires_six_distinct_processes(tmp_path: Path) -> None:
    config = _short_config(tmp_path)
    paths: dict[str, Path] = {}
    for index, mode in enumerate(MODES):
        path = tmp_path / f"{mode}.json"
        payload = _repeat_gate(
            config,
            mode=mode,
            primary_pid=10 + index * 2,
            repeat_pid=11 + index * 2,
        )
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[mode] = path

    matrix = audit_motion_short_matrix(config, repeat_gate_paths=paths)

    assert matrix["valid"] is True
    assert matrix["all_process_ids_distinct"] is True
    assert matrix["full_expansion_authorized"] is True
    duplicate = json.loads(paths[MODES[-1]].read_text(encoding="utf-8"))
    duplicate["repeat"]["process_id"] = 10
    paths[MODES[-1]].write_text(json.dumps(duplicate), encoding="utf-8")
    failed = audit_motion_short_matrix(config, repeat_gate_paths=paths)
    assert failed["valid"] is False
    assert "short_matrix_process_ids_not_all_distinct" in failed["errors"]
