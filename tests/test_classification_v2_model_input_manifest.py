from __future__ import annotations

import json
from pathlib import Path

import pytest

from pig_behavior.classification_v2.contracts.model_input_manifest import (
    MODEL_INPUT_MANIFEST_SCHEMA_VERSION,
    REQUIRED_ARTIFACTS,
    ModelInputManifestError,
    build_model_input_manifest,
    write_model_input_manifest,
)
from pig_behavior.classification_v2.contracts.versioned_data_contract import (
    ARTIFACT_MAP_SCHEMA_VERSION,
    build_versioned_data_contract,
    write_versioned_data_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_TEMPLATE = (
    PROJECT_ROOT
    / "configs"
    / "classification_v2"
    / "reviewed_q2_data_contract_template_v1.json"
)
SOURCE_FEATURE_SPEC = (
    PROJECT_ROOT
    / "configs"
    / "classification_v2"
    / "reviewed_q2_tabular_feature_spec_v1.json"
)
RUN_ID = "c2v2_model_input_manifest_fixture_v1"
HUMAN_RUN_ID = "c2v2_human_review_manifest_fixture_v1"


def test_model_input_manifest_uses_only_contract_declared_paths(
    tmp_path: Path,
) -> None:
    contract_path, output_path, _ = _fixture(tmp_path)

    build = build_model_input_manifest(
        contract_path,
        output_path=output_path,
        project_root=tmp_path,
    )
    dry_run = write_model_input_manifest(
        build,
        dry_run=True,
        overwrite=False,
    )

    assert dry_run["status"] == "PASS"
    assert dry_run["artifact_written"] is False
    assert output_path.exists() is False
    assert build.manifest["schema_version"] == (
        MODEL_INPUT_MANIFEST_SCHEMA_VERSION
    )
    assert build.manifest["run_id"] == RUN_ID
    assert build.manifest["path_policy"][
        "canonical_fallback_allowed"
    ] is False
    assert build.manifest["feature_selection"][
        "all_numeric_selection_allowed"
    ] is False
    assert build.manifest["forbidden_model_inputs"]
    assert build.manifest["missing_artifacts"] == []
    assert build.manifest["artifacts"]["tabular_X"] == (
        build.manifest["artifact_groups"]["predictive"]["tabular_X"][
            "path"
        ]
    )
    serialized = json.dumps(build.manifest)
    assert "outputs/classification_v2/train_ready_windows" not in serialized
    assert "native_temporal_units_oof_folds" not in serialized

    written = write_model_input_manifest(
        build,
        dry_run=False,
        overwrite=False,
    )

    assert written["artifact_written"] is True
    assert output_path.is_file()
    assert len(written["output_sha256"]) == 64


def test_loader_audit_is_post_manifest_evidence_not_manifest_input() -> None:
    assert "loader_input_audit" not in REQUIRED_ARTIFACTS


def test_model_input_manifest_rejects_human_owned_destination(
    tmp_path: Path,
) -> None:
    contract_path, output_path, _ = _fixture(tmp_path)
    human_path = (
        tmp_path
        / "human_review_workspace"
        / "classification_v2"
        / HUMAN_RUN_ID
        / "model_input_contract.json"
    )

    with pytest.raises(
        ModelInputManifestError,
        match="output_json_does_not_match_contract_artifact",
    ):
        build_model_input_manifest(
            contract_path,
            output_path=human_path,
            project_root=tmp_path,
        )

    assert output_path.exists() is False
    assert human_path.exists() is False


def test_model_input_manifest_rejects_missing_required_artifact(
    tmp_path: Path,
) -> None:
    contract_path, output_path, contract = _fixture(tmp_path)
    tabular_path = tmp_path / contract["artifacts"]["tabular_X"]["path"]
    tabular_path.unlink()

    with pytest.raises(
        ModelInputManifestError,
        match="required_artifact_missing:tabular_X",
    ):
        build_model_input_manifest(
            contract_path,
            output_path=output_path,
            project_root=tmp_path,
        )

    assert output_path.exists() is False


def test_model_input_manifest_rejects_artifact_map_drift(
    tmp_path: Path,
) -> None:
    contract_path, output_path, contract = _fixture(tmp_path)
    map_path = tmp_path / contract["artifact_map_path"]
    artifact_map = json.loads(map_path.read_text(encoding="utf-8"))
    artifact_map["train_ready_root"] += "_changed"
    map_path.write_text(json.dumps(artifact_map), encoding="utf-8")

    with pytest.raises(
        ModelInputManifestError,
        match="artifact_map_sha256_mismatch",
    ):
        build_model_input_manifest(
            contract_path,
            output_path=output_path,
            project_root=tmp_path,
        )

    assert output_path.exists() is False


def test_model_input_manifest_refuses_unflagged_overwrite(
    tmp_path: Path,
) -> None:
    contract_path, output_path, _ = _fixture(tmp_path)
    build = build_model_input_manifest(
        contract_path,
        output_path=output_path,
        project_root=tmp_path,
    )
    write_model_input_manifest(
        build,
        dry_run=False,
        overwrite=False,
    )

    with pytest.raises(FileExistsError):
        write_model_input_manifest(
            build,
            dry_run=False,
            overwrite=False,
        )


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    template = json.loads(SOURCE_TEMPLATE.read_text(encoding="utf-8"))
    config_dir = tmp_path / "configs" / "classification_v2"
    config_dir.mkdir(parents=True)
    template_path = config_dir / SOURCE_TEMPLATE.name
    template_path.write_text(json.dumps(template), encoding="utf-8")
    policy_path = config_dir / "hidden_review_scientific_policy_v1.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "classification_v2.hidden_scientific_policy.v1"
                )
            }
        ),
        encoding="utf-8",
    )
    feature_spec_path = config_dir / SOURCE_FEATURE_SPEC.name
    feature_spec_path.write_text(
        SOURCE_FEATURE_SPEC.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    agent_root = f"outputs/classification_v2/agent_audits/{RUN_ID}"
    human_root = (
        "human_review_workspace/classification_v2/"
        f"{HUMAN_RUN_ID}"
    )
    artifacts = {
        name: {
            "path": _artifact_path(
                name,
                spec,
                agent_root=agent_root,
                human_root=human_root,
            ),
            "scope": spec["scope"],
        }
        for name, spec in template["artifacts"].items()
    }
    map_path = tmp_path / agent_root / "contracts" / "artifact_map.json"
    map_path.parent.mkdir(parents=True)
    map_path.write_text(
        json.dumps(
            {
                "schema_version": ARTIFACT_MAP_SCHEMA_VERSION,
                "run_id": RUN_ID,
                "profile": "mixed-reviewed",
                "lineage_ids": {
                    "agent_derived": RUN_ID,
                    "human_review": HUMAN_RUN_ID,
                },
                "lineage_roots": {
                    "agent_derived": agent_root,
                    "human_review": human_root,
                },
                "train_ready_root": f"{agent_root}/train_ready",
                "snapshot_output_dir": f"{agent_root}/snapshots",
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    contract_path = tmp_path / agent_root / "contracts" / "data_contract.json"
    contract_build = build_versioned_data_contract(
        template_path,
        map_path,
        output_path=contract_path,
        project_root=tmp_path,
    )
    write_versioned_data_contract(
        contract_build,
        dry_run=False,
        overwrite=False,
    )
    contract = contract_build.contract
    for name in REQUIRED_ARTIFACTS:
        path = tmp_path / contract["artifacts"][name]["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture:{name}\n".encode("ascii"))
    output_path = (
        tmp_path
        / contract["artifacts"]["model_input_contract"]["path"]
    )
    return contract_path, output_path, contract


def _artifact_path(
    name: str,
    spec: dict[str, object],
    *,
    agent_root: str,
    human_root: str,
) -> str:
    if spec["scope"] == "project_static":
        static_paths = {
            "hidden_review_scientific_policy": (
                "configs/classification_v2/"
                "hidden_review_scientific_policy_v1.json"
            ),
            "tabular_feature_spec": (
                "configs/classification_v2/"
                "reviewed_q2_tabular_feature_spec_v1.json"
            ),
        }
        return static_paths[name]
    suffix = {
        "binary": ".npy",
        "csv": ".csv",
        "json": ".json",
        "npz": ".npz",
    }[str(spec["type"])]
    root = human_root if spec["scope"] == "human_review" else agent_root
    return f"{root}/artifacts/{name}{suffix}"
