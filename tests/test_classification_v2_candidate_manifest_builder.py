from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import pig_behavior.classification_v2.contracts.candidate_manifest as candidate_manifest_module
from pig_behavior.classification_v2.contracts.candidate_manifest import (
    CANDIDATE_MANIFEST_DEVELOPMENT_CONTRACT_VERSION,
    build_candidate_artifact_manifest,
    inspect_candidate_output,
    output_inspector_registry,
)
from pig_behavior.classification_v2.contracts.semantic_lineage import (
    MANIFEST_BUILDER_ID,
    MANIFEST_BUILDER_VERSION,
    file_sha256,
    load_scientific_contract,
    validate_artifact_manifest,
)
from pig_behavior.classification_v2.features.motion_schema import (
    MOTION_FEATURE_NAMES,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ROOT = (
    REPO_ROOT / "docs/classification_v2/scientific_contract_v1"
)
CONTRACT = load_scientific_contract(
    CONTRACT_ROOT / "00_pipeline_contract.yaml"
)
DEVELOPMENT = CANDIDATE_MANIFEST_DEVELOPMENT_CONTRACT_VERSION
INTEGRATED_STAGES = (
    "stage.legacy_cvat_source_merge",
    "stage.frame_local_primitives",
    "stage.hidden_review_design",
    "stage.hidden_decision_migration",
    "stage.hidden_coverage_scientific_gate",
    "stage.hidden_apply",
    "stage.temporal_harmonization",
    "stage.native_review_evidence",
    "stage.pig_strenet_evidence",
    "stage.behavior_review_unit_construction",
    "stage.behavior_decision_apply",
    "stage.train_ready_export",
    "stage.tensor_export",
    "stage.model_input",
)


def _stage(stage_id: str) -> dict[str, object]:
    return next(
        stage
        for stage in CONTRACT["stages"]
        if stage["stage_id"] == stage_id
    )


def _columns(stage_id: str) -> list[str]:
    stage = _stage(stage_id)
    columns = [
        str(value)
        for value in stage.get("produced_columns", [])
        if isinstance(value, str)
        and value
        and " " not in value
        and "*" not in value
    ]
    if stage_id in {
        "stage.legacy_cvat_source_merge",
        "stage.frame_local_primitives",
        "stage.hidden_review_design",
        "stage.hidden_apply",
        "stage.temporal_harmonization",
        "stage.native_review_evidence",
        "stage.pig_strenet_evidence",
        "stage.behavior_review_unit_construction",
        "stage.behavior_decision_apply",
    }:
        columns.append("object_track_key")
    return list(dict.fromkeys(columns or ["record_id"]))


def _write_csv(path: Path, stage_id: str) -> None:
    values = {
        column: (
            "track:source:dataset:video:1"
            if column == "object_track_key"
            else 1
        )
        for column in _columns(stage_id)
    }
    pd.DataFrame([values]).to_csv(path, index=False)


def _build(
    tmp_path: Path,
    stage_id: str,
    *,
    upstream: tuple[Path, ...] = (),
    metadata: dict[str, object] | None = None,
    expected: dict[str, object] | None = None,
    suffix: str = ".csv",
    before_replace=None,
):
    stage = _stage(stage_id)
    tmp_path.mkdir(parents=True, exist_ok=True)
    output = tmp_path / f"{stage_id.removeprefix('stage.')}{suffix}"
    if suffix == ".csv":
        _write_csv(output, stage_id)
    else:
        output.write_text("unsupported", encoding="utf-8")
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    result = build_candidate_artifact_manifest(
        repo_root=REPO_ROOT,
        contract_root=CONTRACT_ROOT,
        stage_id=stage_id,
        artifact_id=f"test.{stage_id}.{tmp_path.name}",
        artifact_class="SYNTHETIC_INTEGRATION_TEST_ONLY",
        output_path=output,
        candidate_manifest_path=manifest_path,
        upstream_manifest_paths=upstream,
        output_schema_id=str(stage["output_artifacts"][0]),
        output_schema_version=str(stage["schema_version"]),
        stage_specific_metadata=metadata,
        expected_authority=expected,
        development_contract_version=DEVELOPMENT,
        before_atomic_replace=before_replace,
    )
    return output, result


def test_source_merge_candidate_is_production_built_and_reread_valid(
    tmp_path: Path,
) -> None:
    output, result = _build(
        tmp_path,
        "stage.legacy_cvat_source_merge",
    )
    manifest = result.manifest
    assert result.production_builder_owned
    assert manifest["manifest_builder_id"] == MANIFEST_BUILDER_ID
    assert manifest["manifest_builder_version"] == MANIFEST_BUILDER_VERSION
    assert manifest["authority_state"] == "CANDIDATE_VALIDATED"
    assert manifest["status"] == "VALIDATED"
    assert manifest["output_file_sha256"] == file_sha256(output)
    assert manifest["row_count"] == 1
    assert manifest["column_count"] == len(_columns(result.manifest["stage_id"]))
    assert len(manifest["created_by_code_authority_sha"]) == 40
    assert all(
        len(manifest[field]) == 64
        for field in (
            "manifest_builder_code_hash",
            "stage_code_hash",
            "stage_semantics_hash",
            "stage_input_fingerprint",
            "stage_execution_fingerprint",
            "execution_parameters_hash",
            "output_file_sha256",
            "output_schema_hash",
        )
    )
    reread = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert validate_artifact_manifest(reread, output_path=output)["valid"]


def test_frame_local_candidate_validates_its_source_upstream(
    tmp_path: Path,
) -> None:
    _, source = _build(
        tmp_path / "source",
        "stage.legacy_cvat_source_merge",
    )
    _, frame_local = _build(
        tmp_path / "frame",
        "stage.frame_local_primitives",
        upstream=(source.manifest_path,),
    )
    assert frame_local.manifest["input_artifact_ids"] == [
        source.manifest["artifact_id"]
    ]
    assert frame_local.manifest["input_artifact_fingerprints"] == {
        source.manifest["artifact_id"]: source.manifest[
            "stage_execution_fingerprint"
        ]
    }


def test_output_inspector_registry_is_typed_and_fails_unsupported(
    tmp_path: Path,
) -> None:
    registry = output_inspector_registry()
    assert set(registry["inspectors"]) == {
        ".csv",
        ".json",
        ".jsonl",
        ".npy",
        ".npz",
    }
    unsupported = tmp_path / "artifact.parquet"
    unsupported.write_bytes(b"not parquet")
    with pytest.raises(ValueError, match="UNSUPPORTED_OUTPUT_INSPECTOR"):
        inspect_candidate_output(unsupported)


def test_expected_load_bearing_authority_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="EXPECTED_AUTHORITY_MISMATCH:stage_code_hash",
    ):
        _build(
            tmp_path,
            "stage.legacy_cvat_source_merge",
            expected={"stage_code_hash": "0" * 64},
        )
    assert not list(tmp_path.glob("*.manifest.json"))


def test_supplied_row_count_cannot_override_inspection(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="EXPECTED_AUTHORITY_MISMATCH:row_count",
    ):
        _build(
            tmp_path,
            "stage.legacy_cvat_source_merge",
            expected={"row_count": 99},
        )
    assert not list(tmp_path.glob("*.manifest.json"))


def test_missing_and_stale_upstream_fail_before_manifest_write(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.manifest.json"
    with pytest.raises(FileNotFoundError, match="UPSTREAM_MANIFEST_MISSING"):
        _build(
            tmp_path / "missing_case",
            "stage.frame_local_primitives",
            upstream=(missing,),
        )
    source_output, source = _build(
        tmp_path / "source",
        "stage.legacy_cvat_source_merge",
    )
    source_output.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="UPSTREAM_MANIFEST_INVALID"):
        _build(
            tmp_path / "stale_case",
            "stage.frame_local_primitives",
            upstream=(source.manifest_path,),
        )
    assert not list((tmp_path / "stale_case").glob("*.manifest.json"))


def test_wrong_stage_schema_and_forbidden_roi_fail_closed(
    tmp_path: Path,
) -> None:
    output = tmp_path / "wrong.csv"
    _write_csv(output, "stage.legacy_cvat_source_merge")
    with pytest.raises(ValueError, match="OUTPUT_SCHEMA_NOT_PERMITTED"):
        build_candidate_artifact_manifest(
            repo_root=REPO_ROOT,
            contract_root=CONTRACT_ROOT,
            stage_id="stage.legacy_cvat_source_merge",
            artifact_id="test.wrong-schema",
            artifact_class="SYNTHETIC_INTEGRATION_TEST_ONLY",
            output_path=output,
            candidate_manifest_path=tmp_path / "wrong.manifest.json",
            output_schema_id="artifact.model_input",
            development_contract_version=DEVELOPMENT,
        )
    with pytest.raises(ValueError, match="FORBIDDEN_MODEL_INPUT_COLUMNS"):
        _build(
            tmp_path / "roi",
            "stage.model_input",
            metadata={"model_feature_names": ["target_roi_contact"]},
        )
    assert not list(tmp_path.rglob("*.manifest.json"))


def test_motion_order_and_object_identity_gates_fail_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="MOTION_SCHEMA_FEATURE_ORDER_MISMATCH",
    ):
        _build(
            tmp_path / "motion",
            "stage.tensor_export",
            metadata={"motion_feature_names": ["speed_n_per_second"]},
        )
    output = tmp_path / "identity.csv"
    pd.DataFrame([{"source_frame_index": 0}]).to_csv(output, index=False)
    with pytest.raises(ValueError, match="MISSING_CONTRACT_OUTPUT_COLUMNS"):
        build_candidate_artifact_manifest(
            repo_root=REPO_ROOT,
            contract_root=CONTRACT_ROOT,
            stage_id="stage.legacy_cvat_source_merge",
            artifact_id="test.missing-identity",
            artifact_class="SYNTHETIC_INTEGRATION_TEST_ONLY",
            output_path=output,
            candidate_manifest_path=tmp_path / "identity.manifest.json",
            development_contract_version=DEVELOPMENT,
        )


def test_atomic_write_interruption_leaves_no_candidate(
    tmp_path: Path,
) -> None:
    def interrupt() -> None:
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        _build(
            tmp_path,
            "stage.legacy_cvat_source_merge",
            before_replace=interrupt,
        )
    assert not list(tmp_path.glob("*.manifest.json"))
    assert not list(tmp_path.glob("*.candidate-staging"))


def test_dirty_scientific_authority_requires_versioned_development_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        candidate_manifest_module,
        "_tracked_scientific_changes",
        lambda _repo_root: [
            "src/pig_behavior/classification_v2/contracts/example.py"
        ],
    )
    stage = _stage("stage.legacy_cvat_source_merge")
    output = tmp_path / "source.csv"
    _write_csv(output, "stage.legacy_cvat_source_merge")
    with pytest.raises(ValueError, match="DIRTY_SCIENTIFIC_WORKTREE"):
        build_candidate_artifact_manifest(
            repo_root=REPO_ROOT,
            contract_root=CONTRACT_ROOT,
            stage_id="stage.legacy_cvat_source_merge",
            artifact_id="test.dirty",
            artifact_class="SYNTHETIC_INTEGRATION_TEST_ONLY",
            output_path=output,
            candidate_manifest_path=tmp_path / "source.manifest.json",
            output_schema_id=str(stage["output_artifacts"][0]),
        )


def test_complete_fourteen_stage_candidate_chain_is_production_built(
    tmp_path: Path,
) -> None:
    previous: tuple[Path, ...] = ()
    manifests: list[dict[str, object]] = []
    for index, stage_id in enumerate(INTEGRATED_STAGES):
        metadata: dict[str, object] = {}
        if stage_id in {
            "stage.native_review_evidence",
            "stage.tensor_export",
        }:
            metadata["motion_feature_names"] = list(MOTION_FEATURE_NAMES)
        if stage_id in {
            "stage.train_ready_export",
            "stage.tensor_export",
            "stage.model_input",
        }:
            metadata["model_feature_names"] = ["cx_n"]
        _, result = _build(
            tmp_path / f"{index:02d}",
            stage_id,
            upstream=previous,
            metadata=metadata,
        )
        manifests.append(dict(result.manifest))
        previous = (result.manifest_path,)
    assert len(manifests) == 14
    assert all(
        item["manifest_builder_id"] == MANIFEST_BUILDER_ID
        and item["authority_state"] == "CANDIDATE_VALIDATED"
        for item in manifests
    )


def test_upstream_class_self_dependency_and_stage_cycle_fail_closed(
    tmp_path: Path,
) -> None:
    _, source = _build(
        tmp_path / "source",
        "stage.legacy_cvat_source_merge",
    )
    altered = dict(source.manifest)
    altered["artifact_class"] = "FAILED_DIAGNOSTIC"
    failed_path = source.manifest_path.parent / "failed.manifest.json"
    failed_path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="INVALID_UPSTREAM_ARTIFACT_CLASS"):
        _build(
            tmp_path / "failed_case",
            "stage.frame_local_primitives",
            upstream=(failed_path,),
        )

    with pytest.raises(ValueError, match="SELF_DEPENDENCY"):
        stage = _stage("stage.frame_local_primitives")
        output = tmp_path / "self.csv"
        _write_csv(output, "stage.frame_local_primitives")
        build_candidate_artifact_manifest(
            repo_root=REPO_ROOT,
            contract_root=CONTRACT_ROOT,
            stage_id="stage.frame_local_primitives",
            artifact_id=str(source.manifest["artifact_id"]),
            artifact_class="SYNTHETIC_INTEGRATION_TEST_ONLY",
            output_path=output,
            candidate_manifest_path=tmp_path / "self.manifest.json",
            upstream_manifest_paths=(source.manifest_path,),
            output_schema_id=str(stage["output_artifacts"][0]),
            development_contract_version=DEVELOPMENT,
        )

    _, downstream = _build(
        tmp_path / "downstream",
        "stage.model_input",
        metadata={"model_feature_names": ["cx_n"]},
    )
    with pytest.raises(ValueError, match="DOWNSTREAM_TO_UPSTREAM_CYCLE"):
        _build(
            tmp_path / "cycle",
            "stage.frame_local_primitives",
            upstream=(downstream.manifest_path,),
        )


def test_official_candidate_rejects_nonofficial_upstream(
    tmp_path: Path,
) -> None:
    _, source = _build(
        tmp_path / "source",
        "stage.legacy_cvat_source_merge",
    )
    stage = _stage("stage.frame_local_primitives")
    output = tmp_path / "official.csv"
    _write_csv(output, "stage.frame_local_primitives")
    with pytest.raises(ValueError, match="AUDIT_ONLY_UPSTREAM_FOR_OFFICIAL"):
        build_candidate_artifact_manifest(
            repo_root=REPO_ROOT,
            contract_root=CONTRACT_ROOT,
            stage_id="stage.frame_local_primitives",
            artifact_id="test.official",
            artifact_class="OFFICIAL_SCIENTIFIC",
            output_path=output,
            candidate_manifest_path=tmp_path / "official.manifest.json",
            upstream_manifest_paths=(source.manifest_path,),
            output_schema_id=str(stage["output_artifacts"][0]),
            development_contract_version=DEVELOPMENT,
        )
