from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pig_behavior.classification_v2.contracts.training_snapshot import (
    check_training_snapshot,
    freeze_training_snapshot,
)
from pig_behavior.classification_v2.contracts.versioned_data_contract import (
    ARTIFACT_MAP_SCHEMA_VERSION,
    DATA_CONTRACT_TEMPLATE_SCHEMA_VERSION,
    build_versioned_data_contract,
    write_versioned_data_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "c2v2_agent_snapshot_fixture_v1"
HUMAN_RUN_ID = "c2v2_human_snapshot_fixture_v1"


def test_historical_snapshot_payload_remains_unversioned(
    tmp_path: Path,
) -> None:
    contract_path = tmp_path / "historical_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "contract_version": "historical-v1",
                "snapshot_name": "historical",
                "root": str(tmp_path),
                "snapshot_output_dir": "configured_snapshots",
                "artifacts": {},
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "legacy_override" / "snapshot.json"

    snapshot = freeze_training_snapshot(
        contract_path,
        output_path=output_path,
    )

    assert Path(snapshot["snapshot_path"]) == output_path
    assert {
        "generated_contract_schema_version",
        "run_id",
        "profile",
        "lineage_ids",
        "lineage_roots",
        "template_sha256",
        "artifact_map_sha256",
        "path_policy",
        "versioned_contract_audit",
    }.isdisjoint(snapshot)


def test_versioned_snapshot_binds_run_map_and_owner_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path, snapshot_dir, _ = _fixture(tmp_path)
    monkeypatch.chdir(tmp_path)

    snapshot = freeze_training_snapshot(contract_path)

    assert Path(snapshot["snapshot_path"]).parent == snapshot_dir
    assert snapshot["run_id"] == RUN_ID
    assert snapshot["profile"] == "mixed-reviewed"
    assert snapshot["lineage_ids"] == {
        "agent_derived": RUN_ID,
        "human_review": HUMAN_RUN_ID,
    }
    assert snapshot["versioned_contract_audit"]["valid"] is True
    assert snapshot["versioned_contract_audit"]["applicable"] is True
    assert len(snapshot["artifact_map_sha256"]) == 64


def test_versioned_snapshot_rejects_map_drift_before_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path, snapshot_dir, map_path = _fixture(tmp_path)
    map_payload = json.loads(map_path.read_text(encoding="utf-8"))
    map_payload["train_ready_root"] = (
        "outputs/classification_v2/agent_audits/"
        f"{RUN_ID}/changed_train_ready"
    )
    map_path.write_text(json.dumps(map_payload), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="artifact_map_sha256_mismatch"):
        freeze_training_snapshot(contract_path)

    assert list(snapshot_dir.glob("*.json")) == []


def test_versioned_snapshot_rejects_human_workspace_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path, snapshot_dir, _ = _fixture(tmp_path)
    human_output = (
        tmp_path
        / "human_review_workspace"
        / "classification_v2"
        / HUMAN_RUN_ID
        / "snapshot.json"
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(
        ValueError,
        match="snapshot_destination_outside_declared_output_dir",
    ):
        freeze_training_snapshot(
            contract_path,
            output_path=human_output,
        )

    assert human_output.exists() is False
    assert list(snapshot_dir.glob("*.json")) == []


def test_snapshot_checker_honors_explicit_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path, snapshot_dir, _ = _fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    snapshot = freeze_training_snapshot(contract_path)
    snapshot_path = Path(snapshot["snapshot_path"])

    monkeypatch.chdir(PROJECT_ROOT)
    checked = check_training_snapshot(
        snapshot_path,
        contract_path=contract_path,
        project_root=tmp_path,
    )

    assert checked["valid"] is True


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    config_dir = tmp_path / "configs" / "classification_v2"
    agent_root = (
        tmp_path
        / "outputs"
        / "classification_v2"
        / "agent_audits"
        / RUN_ID
    )
    human_root = (
        tmp_path
        / "human_review_workspace"
        / "classification_v2"
        / HUMAN_RUN_ID
    )
    config_dir.mkdir(parents=True)
    agent_root.mkdir(parents=True)
    human_root.mkdir(parents=True)
    split_path = agent_root / "train_ready" / "split_manifest.csv"
    split_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "window_id": ["window-0", "window-1"],
            "split": ["train", "test"],
            "split_group_key": ["recording-a", "recording-b"],
        }
    ).to_csv(split_path, index=False)

    template_path = config_dir / "data_contract_template.json"
    template_path.write_text(
        json.dumps(
            {
                "template_schema_version": (
                    DATA_CONTRACT_TEMPLATE_SCHEMA_VERSION
                ),
                "contract_version": "fixture_versioned_snapshot_v1",
                "snapshot_name": "fixture_versioned_snapshot",
                "allowed_profiles": ["mixed-reviewed"],
                "primary_key": "window_id",
                "row_count_alignment_group": ["split_manifest"],
                "window_id_source_artifact": "split_manifest",
                "required_ordered_window_artifacts": ["split_manifest"],
                "key_alignment_group": ["split_manifest"],
                "artifacts": {
                    "split_manifest": {
                        "scope": "agent_derived",
                        "type": "csv",
                        "required": True,
                        "key_column": "window_id",
                        "required_columns": [
                            "window_id",
                            "split",
                            "split_group_key",
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    map_path = agent_root / "contracts" / "artifact_map.json"
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
                    "agent_derived": (
                        "outputs/classification_v2/agent_audits/"
                        f"{RUN_ID}"
                    ),
                    "human_review": (
                        "human_review_workspace/classification_v2/"
                        f"{HUMAN_RUN_ID}"
                    ),
                },
                "train_ready_root": (
                    "outputs/classification_v2/agent_audits/"
                    f"{RUN_ID}/train_ready"
                ),
                "snapshot_output_dir": (
                    "outputs/classification_v2/agent_audits/"
                    f"{RUN_ID}/snapshots"
                ),
                "artifacts": {
                    "split_manifest": {
                        "path": (
                            "outputs/classification_v2/agent_audits/"
                            f"{RUN_ID}/train_ready/"
                            "split_manifest.csv"
                        ),
                        "scope": "agent_derived",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    contract_path = agent_root / "contracts" / "data_contract.json"
    build = build_versioned_data_contract(
        template_path,
        map_path,
        output_path=contract_path,
        project_root=tmp_path,
    )
    write_versioned_data_contract(
        build,
        dry_run=False,
        overwrite=False,
    )
    return contract_path, agent_root / "snapshots", map_path
