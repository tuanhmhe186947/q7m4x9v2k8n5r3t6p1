from __future__ import annotations

import csv
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pig_behavior.classification_v2.training import (
    legacy_development_l5_feature_cache as feature_cache_module,
)
from pig_behavior.classification_v2.training.legacy_development_l5 import (
    LINEAGE_SCOPE,
    LegacyL5Config,
    load_legacy_l5_config,
)
from pig_behavior.classification_v2.training.legacy_development_l5_feature_cache import (
    FEATURE_CONTROL_IDS,
    FEATURE_ENVIRONMENT_SCHEMA_VERSION,
    FEATURE_INDEX_FIELDS,
    FEATURE_REGISTRY_FIELDS,
    FEATURE_RUN_MANIFEST_SCHEMA_VERSION,
    FEATURE_RUN_RESULT_SCHEMA_VERSION,
    FEATURE_SHORT_GATE_SCHEMA_VERSION,
    FeatureCacheSource,
    _environment_payload,
    _feature_run_paths,
    _finalize_feature_run,
    _prepare_feature_run,
    _short_run_sequence_audit,
    _write_feature_index,
    _write_progress,
    audit_legacy_l5_feature_preflight,
    write_legacy_l5_feature_short_gate,
)
from pig_behavior.classification_v2.training.legacy_development_l5_visual import (
    LegacyVisualProbeControl,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

CONFIG_PATH = Path("configs/classification_v2/legacy_development_l5_v1.json")


def _control() -> LegacyVisualProbeControl:
    return LegacyVisualProbeControl(
        control_id="V0",
        backbone_name="resnet18",
        pretrained_weight_enum="ResNet18_Weights.IMAGENET1K_V1",
        image_size=160,
        frame_batch_size=16,
    )


def _source(tmp_path: Path, *, rows: int = 3) -> FeatureCacheSource:
    tensor_path = tmp_path / "packed.npy"
    np.save(tensor_path, np.zeros((rows, 2, 2, 3), dtype=np.uint8))
    index_path = tmp_path / "packed_index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_context_id",
                "packed_row",
                "lineage_scope",
                "human_review_complete",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in range(rows):
            writer.writerow(
                {
                    "image_context_id": f"context_{row}",
                    "packed_row": row,
                    "lineage_scope": LINEAGE_SCOPE,
                    "human_review_complete": False,
                }
            )
    return FeatureCacheSource(
        scope="short",
        root=tmp_path,
        tensor_path=tensor_path,
        index_path=index_path,
        rows=rows,
        image_size=2,
        tensor_sha256=file_sha256(tensor_path),
        index_sha256=file_sha256(index_path),
        parent_audit_hashes={},
    )


def _input_file(tmp_path: Path, name: str) -> tuple[str, str]:
    path = tmp_path / "inputs" / f"{name}.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"fixture:{name}".encode())
    return str(path), file_sha256(path)


def _planned_manifest(
    tmp_path: Path,
    *,
    source: FeatureCacheSource,
    control: LegacyVisualProbeControl,
    run_id: str,
) -> dict[str, Any]:
    artifact_fields = {
        name: _input_file(tmp_path, name)
        for name in (
            "config",
            "dataset_snapshot",
            "fold_manifest",
            "feature_whitelist",
            "pretrained_weight",
            "readiness_audit",
            "short_cache_audit",
            "full_cache_audit",
            "weights_audit",
            "vram_probe_audit",
        )
    }
    semantic = "1" * 64
    scientific = "2" * 64
    return {
        "schema_version": FEATURE_RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "execution_mode": "local_smoke",
        "experiment_name": "legacy_l5_pretrained_frame_feature_cache",
        "code_sha": "abc123",
        "dirty_worktree": True,
        "config_path": artifact_fields["config"][0],
        "config_hash": artifact_fields["config"][1],
        "dataset_snapshot_path": artifact_fields["dataset_snapshot"][0],
        "dataset_snapshot_hash": artifact_fields["dataset_snapshot"][1],
        "cache_hash": source.tensor_sha256,
        "fold_manifest_path": artifact_fields["fold_manifest"][0],
        "fold_manifest_hash": artifact_fields["fold_manifest"][1],
        "feature_whitelist_path": artifact_fields["feature_whitelist"][0],
        "feature_whitelist_hash": artifact_fields["feature_whitelist"][1],
        "fold": "not_applicable_frame_feature_cache",
        "seed": 20260714,
        "scope": "short",
        "control_id": control.control_id,
        "backbone_name": control.backbone_name,
        "pretrained_weight_enum": control.pretrained_weight_enum,
        "pretrained_weight_path": artifact_fields["pretrained_weight"][0],
        "pretrained_weight_sha256": artifact_fields["pretrained_weight"][1],
        "image_size": control.image_size,
        "frame_batch_size": control.frame_batch_size,
        "source_tensor_path": str(source.tensor_path),
        "source_tensor_sha256": source.tensor_sha256,
        "source_index_path": str(source.index_path),
        "source_index_sha256": source.index_sha256,
        "readiness_audit_path": artifact_fields["readiness_audit"][0],
        "readiness_audit_sha256": artifact_fields["readiness_audit"][1],
        "short_cache_audit_path": artifact_fields["short_cache_audit"][0],
        "short_cache_audit_sha256": artifact_fields["short_cache_audit"][1],
        "full_cache_audit_path": artifact_fields["full_cache_audit"][0],
        "full_cache_audit_sha256": artifact_fields["full_cache_audit"][1],
        "weights_audit_path": artifact_fields["weights_audit"][0],
        "weights_audit_sha256": artifact_fields["weights_audit"][1],
        "vram_probe_audit_path": artifact_fields["vram_probe_audit"][0],
        "vram_probe_audit_sha256": artifact_fields["vram_probe_audit"][1],
        "short_gate_audit_path": None,
        "short_gate_audit_sha256": None,
        "implementation_source_sha256": "3" * 64,
        "gpu_model": "NVIDIA GeForce RTX 3050 Laptop GPU",
        "gpu_vram_bytes": 4 * 1024**3,
        "declared_gpu_vram_gib": 4,
        "maximum_peak_vram_fraction": 0.7,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "oom_retry_allowed": False,
        "semantic_identity_sha256": semantic,
        "scientific_identity_sha256": scientific,
        "status": "planned",
        "created_at_utc": "2026-07-15T00:00:00+00:00",
    }


def _passing_result(
    paths: dict[str, Path],
    *,
    manifest: dict[str, Any],
    source: FeatureCacheSource,
    control: LegacyVisualProbeControl,
) -> dict[str, Any]:
    return {
        "schema_version": FEATURE_RUN_RESULT_SCHEMA_VERSION,
        "status": "PASS_LEGACY_DEVELOPMENT_L5_FEATURE_CACHE",
        "run_id": manifest["run_id"],
        "semantic_identity_sha256": manifest["semantic_identity_sha256"],
        "scientific_identity_sha256": manifest["scientific_identity_sha256"],
        "planned_run_manifest_sha256": file_sha256(paths["run_manifest"]),
        "environment_sha256": file_sha256(paths["environment"]),
        "lineage_scope": LINEAGE_SCOPE,
        "scope": "short",
        "control_id": control.control_id,
        "backbone_name": control.backbone_name,
        "pretrained_weight_enum": control.pretrained_weight_enum,
        "image_size": control.image_size,
        "frame_batch_size": control.frame_batch_size,
        "source_rows": source.rows,
        "completed_rows": source.rows,
        "feature_tensor_path": str(paths["feature_tensor"]),
        "feature_tensor_sha256": file_sha256(paths["feature_tensor"]),
        "feature_index_path": str(paths["feature_index"]),
        "feature_index_sha256": file_sha256(paths["feature_index"]),
        "source_tensor_sha256": source.tensor_sha256,
        "source_index_sha256": source.index_sha256,
        "precision": "float32",
        "device_name": "NVIDIA GeForce RTX 3050 Laptop GPU",
        "actual_total_vram_bytes": 4 * 1024**3,
        "runtime_sec": 0.1,
        "peak_reserved_bytes": 128 * 1024**2,
        "resumed": False,
        "errors": [],
        "valid": True,
    }


def test_feature_environment_defers_cuda_probe_until_allocator_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unexpected_cuda_probe() -> bool:
        raise AssertionError("environment capture initialized CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", _unexpected_cuda_probe)
    payload = _environment_payload(
        {
            "execution_mode": "local_smoke",
            "gpu_model": "NVIDIA GeForce RTX 3050 Laptop GPU",
            "gpu_vram_bytes": 4 * 1024**3,
            "declared_gpu_vram_gib": 4,
            "maximum_peak_vram_fraction": 0.7,
        }
    )

    assert payload["schema_version"] == FEATURE_ENVIRONMENT_SCHEMA_VERSION
    assert payload["cuda_runtime_probe_deferred_until_allocator_gate"] is True
    assert payload["maximum_peak_vram_fraction"] == 0.7
    assert payload["oom_retry_allowed"] is False


def test_feature_preflight_resolves_six_sources_without_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path, rows=3)
    config = load_legacy_l5_config(CONFIG_PATH)

    def _resolve(*_args: object, **_kwargs: object) -> FeatureCacheSource:
        return source

    monkeypatch.setattr(
        feature_cache_module,
        "resolve_legacy_l5_feature_source",
        _resolve,
    )
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)

    result = audit_legacy_l5_feature_preflight(
        config,
        parents={
            "readiness": {},
            "short_cache": {},
            "full_cache": {},
        },
    )

    assert result["valid"] is True
    assert len(result["sources"]) == 6
    assert result["cuda_initialized_before"] is False
    assert result["cuda_initialized_after"] is False
    assert result["full_feature_cache_expansion_authorized"] is False


def test_feature_index_is_deterministic_and_binds_control(tmp_path: Path) -> None:
    source = _source(tmp_path, rows=4)
    control = _control()
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"

    _write_feature_index(
        source.index_path,
        first,
        control=control,
        expected_rows=source.rows,
    )
    _write_feature_index(
        source.index_path,
        second,
        control=control,
        expected_rows=source.rows,
    )

    assert first.read_bytes() == second.read_bytes()
    with first.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == FEATURE_INDEX_FIELDS
    assert [int(row["feature_row"]) for row in rows] == list(range(4))
    assert {row["control_id"] for row in rows} == {"V0"}


def test_feature_resume_uses_only_durable_checkpoint(tmp_path: Path) -> None:
    source = _source(tmp_path, rows=4)
    control = _control()
    paths = _feature_run_paths(tmp_path / "run")
    manifest = _planned_manifest(
        tmp_path,
        source=source,
        control=control,
        run_id="run",
    )
    assert _prepare_feature_run(
        paths,
        run_manifest=manifest,
        source=source,
        resume=False,
    ) == 0
    _write_progress(
        paths,
        run_manifest=manifest,
        source=source,
        completed_rows=2,
        input_mapping_open_count=1,
        output_mapping_open_count=2,
        status="running",
    )

    resumed_row = _prepare_feature_run(
        paths,
        run_manifest=manifest,
        source=source,
        resume=True,
    )

    assert resumed_row == 2
    paths["unexpected_failure"].write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be resumed implicitly"):
        _prepare_feature_run(
            paths,
            run_manifest=manifest,
            source=source,
            resume=True,
        )


def test_feature_finalize_writes_all_required_lineage(tmp_path: Path) -> None:
    source = _source(tmp_path, rows=3)
    control = _control()
    paths = _feature_run_paths(tmp_path / "run")
    manifest = _planned_manifest(
        tmp_path,
        source=source,
        control=control,
        run_id="run",
    )
    _prepare_feature_run(
        paths,
        run_manifest=manifest,
        source=source,
        resume=False,
    )
    _write_feature_index(
        source.index_path,
        paths["feature_index"],
        control=control,
        expected_rows=source.rows,
    )
    _write_progress(
        paths,
        run_manifest=manifest,
        source=source,
        completed_rows=source.rows,
        input_mapping_open_count=1,
        output_mapping_open_count=2,
        status="complete",
    )
    result = _passing_result(
        paths,
        manifest=manifest,
        source=source,
        control=control,
    )

    _finalize_feature_run(paths, result=result)

    required = (
        "run_manifest",
        "environment",
        "artifact_manifest",
        "checkpoint_manifest",
        "prediction_manifest",
        "registry_entry",
        "runs_registry",
        "run_result",
    )
    assert all(paths[name].is_file() for name in required)
    final_manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
    assert final_manifest["status"] == "completed"
    with paths["runs_registry"].open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        registry_rows = list(csv.DictReader(handle))
    assert tuple(registry_rows[0]) == FEATURE_REGISTRY_FIELDS
    assert registry_rows[0]["status"] == "completed"
    artifact_manifest = json.loads(
        paths["artifact_manifest"].read_text(encoding="utf-8")
    )
    artifact_names = {item["name"] for item in artifact_manifest["artifacts"]}
    assert {"feature_tensor", "feature_index", "pretrained_weight"}.issubset(
        artifact_names
    )


def test_short_gate_sequence_rejects_overlapping_gpu_runs() -> None:
    start = datetime(2026, 7, 15, tzinfo=timezone.utc)
    reports: dict[str, dict[str, Any]] = {}
    offset = 0
    for control_id in FEATURE_CONTROL_IDS:
        runs: dict[str, Any] = {}
        for role in ("primary", "repeat"):
            created = start + timedelta(seconds=offset)
            completed = created + timedelta(seconds=1)
            runs[role] = {
                "run_id": f"{control_id}_{role}",
                "created_at_utc": created.isoformat(),
                "completed_at_utc": completed.isoformat(),
                "post_cleanup_vram_zero": True,
            }
            offset += 2
        reports[control_id] = {
            "control_id": control_id,
            **runs,
            "errors": [],
            "valid": True,
        }

    passing = _short_run_sequence_audit(reports)
    overlapping = deepcopy(reports)
    overlapping["V1"]["primary"]["created_at_utc"] = (
        start + timedelta(milliseconds=500)
    ).isoformat()
    failing = _short_run_sequence_audit(overlapping)

    assert passing["valid"] is True
    assert passing["run_count"] == 6
    assert passing["all_post_cleanup_vram_zero"] is True
    assert failing["valid"] is False
    assert failing["interval_overlap_count"] >= 1


def test_short_gate_writer_preserves_fail_evidence(tmp_path: Path) -> None:
    base = load_legacy_l5_config(CONFIG_PATH)
    config = LegacyL5Config(
        path=base.path,
        payload=base.payload,
        development_root=tmp_path / "development",
        primary_run_id=base.primary_run_id,
        l3_audit_relative_path=base.l3_audit_relative_path,
        l4_audit_relative_path=base.l4_audit_relative_path,
        l5_output_relative_path=base.l5_output_relative_path,
    )
    payload = {
        "schema_version": FEATURE_SHORT_GATE_SCHEMA_VERSION,
        "config_sha256": config.sha256,
        "implementation_source_sha256": file_sha256(
            Path(str(feature_cache_module.__file__))
        ),
        "status": "FAIL_LEGACY_DEVELOPMENT_L5_FEATURE_SHORT_GATE",
        "errors": ["synthetic_mismatch"],
        "valid": False,
    }
    output = config.l5_output_root / "failed_gate.json"

    write_legacy_l5_feature_short_gate(
        config,
        output_path=output,
        payload=payload,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_config_still_freezes_four_gib_feature_batches() -> None:
    config = load_legacy_l5_config(CONFIG_PATH)

    assert config.payload["optimization"]["declared_local_gpu_vram_gib"] == 4
    assert config.payload["optimization"]["maximum_peak_vram_fraction"] == 0.7
    assert config.payload["feature_cache"]["resnet18_frame_batch_size"] == 16
    assert config.payload["feature_cache"]["resnet34_frame_batch_size"] == 8
    assert config.payload["optimization"]["oom_retry_allowed"] is False
