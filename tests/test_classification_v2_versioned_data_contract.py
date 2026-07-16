from __future__ import annotations

import json
from pathlib import Path

import pytest

from pig_behavior.classification_v2.contracts.versioned_data_contract import (
    ARTIFACT_MAP_SCHEMA_VERSION,
    DATA_CONTRACT_TEMPLATE_SCHEMA_VERSION,
    GENERATED_CONTRACT_SCHEMA_VERSION,
    VersionedDataContractError,
    build_versioned_data_contract,
    validate_generated_data_contract,
    write_versioned_data_contract,
)

RUN_ID = "c2v2_reviewed_20260716_v1"


def test_explicit_artifact_map_builds_run_bound_contract(
    tmp_path: Path,
) -> None:
    template_path, map_path, output_path = _fixture(tmp_path)

    build = build_versioned_data_contract(
        template_path,
        map_path,
        output_path=output_path,
        project_root=tmp_path,
    )

    assert (
        build.contract["generated_contract_schema_version"]
        == GENERATED_CONTRACT_SCHEMA_VERSION
    )
    assert build.contract["run_id"] == RUN_ID
    assert build.contract["path_policy"]["canonical_fallback_allowed"] is False
    assert build.contract["artifacts"]["split_manifest"]["path"].startswith(
        f"lineages/{RUN_ID}/"
    )
    assert build.contract["artifacts"]["trainer_contract"]["scope"] == (
        "project_static"
    )
    assert len(build.contract["template_sha256"]) == 64
    assert len(build.contract["artifact_map_sha256"]) == 64
    assert build.audit["dataset_rows_read"] == 0


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    template_path, map_path, output_path = _fixture(tmp_path)
    build = build_versioned_data_contract(
        template_path,
        map_path,
        output_path=output_path,
        project_root=tmp_path,
    )

    audit = write_versioned_data_contract(
        build,
        dry_run=True,
        overwrite=False,
    )

    assert audit["artifact_written"] is False
    assert output_path.exists() is False


def test_write_requires_explicit_overwrite(tmp_path: Path) -> None:
    template_path, map_path, output_path = _fixture(tmp_path)
    build = build_versioned_data_contract(
        template_path,
        map_path,
        output_path=output_path,
        project_root=tmp_path,
    )
    write_versioned_data_contract(build, dry_run=False, overwrite=False)

    with pytest.raises(FileExistsError, match="--overwrite"):
        write_versioned_data_contract(
            build,
            dry_run=False,
            overwrite=False,
        )

    audit = write_versioned_data_contract(
        build,
        dry_run=False,
        overwrite=True,
    )
    assert audit["artifact_written"] is True


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (
            lambda payload: payload["artifacts"].pop("split_manifest"),
            "artifact_map_missing_artifacts",
        ),
        (
            lambda payload: payload["artifacts"].update(
                {"unknown": {"path": "unknown.csv", "scope": "lineage"}}
            ),
            "artifact_map_unknown_artifacts",
        ),
        (
            lambda payload: payload["artifacts"]["split_manifest"].update(
                {
                    "path": (
                        "outputs/classification_v2/train_ready_windows/"
                        "split_manifest.csv"
                    )
                }
            ),
            "lineage_artifact_outside_roots",
        ),
        (
            lambda payload: payload["artifacts"]["trainer_contract"].update(
                {"scope": "lineage"}
            ),
            "artifact_scope_mismatch",
        ),
    ],
)
def test_artifact_map_rejects_implicit_or_misclassified_paths(
    tmp_path: Path,
    mutation,
    expected_error: str,
) -> None:
    template_path, map_path, output_path = _fixture(tmp_path)
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    mutation(payload)
    map_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VersionedDataContractError) as exc_info:
        build_versioned_data_contract(
            template_path,
            map_path,
            output_path=output_path,
            project_root=tmp_path,
        )

    assert any(expected_error in error for error in exc_info.value.errors)


def test_template_cannot_supply_fallback_artifact_path(tmp_path: Path) -> None:
    template_path, map_path, output_path = _fixture(tmp_path)
    template = json.loads(template_path.read_text(encoding="utf-8"))
    template["artifacts"]["split_manifest"]["path"] = (
        "outputs/classification_v2/train_ready_windows/split_manifest.csv"
    )
    template_path.write_text(json.dumps(template), encoding="utf-8")

    with pytest.raises(VersionedDataContractError) as exc_info:
        build_versioned_data_contract(
            template_path,
            map_path,
            output_path=output_path,
            project_root=tmp_path,
        )

    assert "template_artifact_contains_path:split_manifest" in (
        exc_info.value.errors
    )


def test_generated_contract_detects_artifact_map_drift(tmp_path: Path) -> None:
    template_path, map_path, output_path = _fixture(tmp_path)
    build = build_versioned_data_contract(
        template_path,
        map_path,
        output_path=output_path,
        project_root=tmp_path,
    )
    write_versioned_data_contract(build, dry_run=False, overwrite=False)
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    payload["snapshot_output_dir"] = (
        f"lineages/{RUN_ID}/snapshots_changed"
    )
    map_path.write_text(json.dumps(payload), encoding="utf-8")

    errors = validate_generated_data_contract(
        output_path,
        project_root=tmp_path,
    )

    assert errors == [
        "artifact_map_sha256_mismatch",
        "generated_contract_payload_drift",
    ]


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    static_dir = tmp_path / "configs" / "classification_v2"
    lineage_dir = tmp_path / "lineages" / RUN_ID
    static_dir.mkdir(parents=True)
    lineage_dir.mkdir(parents=True)
    static_contract = static_dir / "trainer_contract.json"
    static_contract.write_text("{}", encoding="utf-8")
    template_path = static_dir / "data_contract_template.json"
    template_path.write_text(
        json.dumps(
            {
                "template_schema_version": (
                    DATA_CONTRACT_TEMPLATE_SCHEMA_VERSION
                ),
                "contract_version": "fixture_contract_v1",
                "snapshot_name": "fixture_snapshot",
                "allowed_profiles": ["mixed-reviewed"],
                "primary_key": "window_id",
                "artifacts": {
                    "split_manifest": {
                        "scope": "lineage",
                        "type": "csv",
                        "required": True,
                        "key_column": "window_id",
                    },
                    "trainer_contract": {
                        "scope": "project_static",
                        "type": "json",
                        "required": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    map_path = lineage_dir / "artifact_map.json"
    map_path.write_text(
        json.dumps(
            {
                "schema_version": ARTIFACT_MAP_SCHEMA_VERSION,
                "run_id": RUN_ID,
                "profile": "mixed-reviewed",
                "lineage_roots": [f"lineages/{RUN_ID}"],
                "train_ready_root": f"lineages/{RUN_ID}/train_ready",
                "snapshot_output_dir": f"lineages/{RUN_ID}/snapshots",
                "artifacts": {
                    "split_manifest": {
                        "path": f"lineages/{RUN_ID}/train_ready/split.csv",
                        "scope": "lineage",
                    },
                    "trainer_contract": {
                        "path": (
                            "configs/classification_v2/"
                            "trainer_contract.json"
                        ),
                        "scope": "project_static",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    output_path = lineage_dir / "contracts" / "data_contract.json"
    return template_path, map_path, output_path
