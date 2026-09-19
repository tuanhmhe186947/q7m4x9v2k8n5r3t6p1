from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from pig_behavior.classification_v2.contracts.model_input_manifest import (
    MODEL_INPUT_MANIFEST_SCHEMA_VERSION,
)
from pig_behavior.classification_v2.evaluation.loader_input_audit import (
    LOADER_INPUT_AUDIT_SCHEMA_VERSION,
    audit_loader_input_contract,
    write_loader_input_audit,
)

RUN_ID = "c2v2_agent_loader_fixture_v1"


def test_loader_audit_uses_only_hash_bound_manifest_paths(
    tmp_path: Path,
) -> None:
    manifest = _fixture(tmp_path)

    audit = audit_loader_input_contract(
        model_input_contract_json=manifest,
        project_root=tmp_path,
    )

    assert audit["schema_version"] == LOADER_INPUT_AUDIT_SCHEMA_VERSION
    assert audit["valid"] is True
    assert audit["errors"] == []
    assert audit["canonical_fallback_used"] is False
    assert audit["tabular_x_column_count"] == 2
    assert audit["source_control_rows"] == 2
    assert "trainer_contract" not in json.dumps(audit)

    output = _agent_root(tmp_path) / "audits" / "loader_input_audit.json"
    dry_run = write_loader_input_audit(
        audit,
        output_path=output,
        project_root=tmp_path,
        dry_run=True,
        overwrite=False,
    )
    assert dry_run["artifact_written"] is False
    assert output.exists() is False

    written = write_loader_input_audit(
        audit,
        output_path=output,
        project_root=tmp_path,
        dry_run=False,
        overwrite=False,
    )
    assert written["artifact_written"] is True
    assert output.is_file()


def test_loader_audit_rejects_tabular_column_order_drift(
    tmp_path: Path,
) -> None:
    manifest = _fixture(tmp_path)
    root = _agent_root(tmp_path)
    x_path = root / "data" / "X.csv"
    pd.DataFrame({"feature_b": [2.0], "feature_a": [1.0]}).to_csv(
        x_path,
        index=False,
    )
    _refresh_binding(manifest, "predictive", "tabular_X", x_path, tmp_path)

    audit = audit_loader_input_contract(
        model_input_contract_json=manifest,
        project_root=tmp_path,
    )

    assert audit["valid"] is False
    assert "tabular_x_column_order_does_not_match_whitelist" in audit["errors"]


def test_loader_audit_rejects_bound_artifact_hash_drift(
    tmp_path: Path,
) -> None:
    manifest = _fixture(tmp_path)
    x_path = _agent_root(tmp_path) / "data" / "X.csv"
    x_path.write_text("feature_a,feature_b\n9,9\n", encoding="utf-8")

    audit = audit_loader_input_contract(
        model_input_contract_json=manifest,
        project_root=tmp_path,
    )

    assert audit["valid"] is False
    assert any(
        error == "model_manifest_bound_hash_mismatch:predictive:tabular_X"
        for error in audit["errors"]
    )


def test_loader_audit_cannot_write_human_review_root(
    tmp_path: Path,
) -> None:
    manifest = _fixture(tmp_path)
    audit = audit_loader_input_contract(
        model_input_contract_json=manifest,
        project_root=tmp_path,
    )
    human_output = (
        tmp_path
        / "human_review_workspace"
        / "classification_v2"
        / "c2v2_human_loader_fixture_v1"
        / "loader_input_audit.json"
    )

    with pytest.raises(
        ValueError,
        match="outside agent_derived_root",
    ):
        write_loader_input_audit(
            audit,
            output_path=human_output,
            project_root=tmp_path,
            dry_run=False,
            overwrite=False,
        )

    assert human_output.exists() is False


def _fixture(tmp_path: Path) -> Path:
    root = _agent_root(tmp_path)
    data = root / "data"
    data.mkdir(parents=True)
    x_path = data / "X.csv"
    whitelist_path = data / "feature_whitelist.json"
    blacklist_path = data / "feature_blacklist.json"
    source_path = data / "source_matched_view_manifest.csv"
    source_audit_path = data / "source_matched_view_audit.json"
    source_check_path = data / "source_matched_view_check_audit.json"
    domain_path = data / "domain_controls_audit.json"
    pd.DataFrame(
        {
            "feature_a": [1.0, 2.0],
            "feature_b": [3.0, 4.0],
        }
    ).to_csv(x_path, index=False)
    whitelist_path.write_text(
        json.dumps({"features": ["feature_a", "feature_b"]}),
        encoding="utf-8",
    )
    blacklist_path.write_text(
        json.dumps({"forbidden_patterns": ["*label*", "source_type"]}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "window_id": ["w0", "w1"],
            "source_type": ["legacy_recovered", "cvat_tracking_xml"],
            "view_matched_6frame": [True, True],
            "source_class_balance_keep": [True, True],
        }
    ).to_csv(source_path, index=False)
    source_audit_path.write_text(
        json.dumps({"rows": 2, "valid": True, "errors": []}),
        encoding="utf-8",
    )
    source_check_path.write_text(
        json.dumps({"rows": 2, "valid": True, "errors": []}),
        encoding="utf-8",
    )
    domain_path.write_text(
        json.dumps({"valid": True, "errors": []}),
        encoding="utf-8",
    )
    artifacts = {
        "predictive": {"tabular_X": x_path},
        "feature_contract": {
            "feature_whitelist": whitelist_path,
            "feature_blacklist": blacklist_path,
        },
        "mask_and_control": {
            "source_matched_view_manifest": source_path,
        },
        "data_audits": {
            "source_matched_view_audit": source_audit_path,
            "source_matched_view_check_audit": source_check_path,
            "domain_controls_audit": domain_path,
        },
    }
    groups = {
        group: {
            name: _binding(path, tmp_path)
            for name, path in values.items()
        }
        for group, values in artifacts.items()
    }
    manifest = root / "contracts" / "model_input_contract.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": MODEL_INPUT_MANIFEST_SCHEMA_VERSION,
                "run_id": RUN_ID,
                "lineage_roots": {
                    "agent_derived": (
                        "outputs/classification_v2/agent_audits/"
                        f"{RUN_ID}"
                    )
                },
                "path_policy": {"canonical_fallback_allowed": False},
                "artifact_groups": groups,
                "forbidden_model_inputs": ["*label*", "source_type"],
                "missing_artifacts": [],
                "model_input_branches": {"tabular": {}},
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _refresh_binding(
    manifest: Path,
    group: str,
    name: str,
    path: Path,
    project_root: Path,
) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["artifact_groups"][group][name] = _binding(path, project_root)
    manifest.write_text(json.dumps(payload), encoding="utf-8")


def _binding(path: Path, project_root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(project_root).as_posix(),
        "scope": "agent_derived",
        "type": "csv" if path.suffix == ".csv" else "json",
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _agent_root(tmp_path: Path) -> Path:
    return (
        tmp_path
        / "outputs"
        / "classification_v2"
        / "agent_audits"
        / RUN_ID
    )
