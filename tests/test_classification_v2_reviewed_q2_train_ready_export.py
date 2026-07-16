from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pig_behavior.classification_v2.contracts.versioned_data_contract import (
    ARTIFACT_MAP_SCHEMA_VERSION,
    DATA_CONTRACT_TEMPLATE_SCHEMA_VERSION,
    build_versioned_data_contract,
    write_versioned_data_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORTER_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "classification_v2"
    / "02_train_ready_exports"
    / "classification_v2_export_train_ready_windows.py"
)
FEATURE_SPEC_SOURCE = (
    PROJECT_ROOT
    / "configs"
    / "classification_v2"
    / "reviewed_q2_tabular_feature_spec_v1.json"
)
EXPORT_TRAIN_READY = runpy.run_path(str(EXPORTER_PATH))[
    "export_train_ready_from_contract"
]
AGENT_RUN_ID = "c2v2_agent_export_fixture_v1"
HUMAN_RUN_ID = "c2v2_human_export_fixture_v1"
OUTPUT_NAMES = (
    "tabular_X",
    "y_behavior",
    "train_mask",
    "sample_weight",
    "feature_whitelist",
    "feature_blacklist",
    "feature_whitelist_audit",
    "train_ready_audit",
)


def test_export_uses_run_bound_paths_and_preserves_all_rows(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    audit = _export(fixture["contract_path"], tmp_path)

    contract = _read_json(fixture["contract_path"])
    train_ready_root = tmp_path / contract["train_ready_root"]
    feature_spec = _read_json(fixture["feature_spec_path"])
    exported_x = pd.read_csv(
        tmp_path / contract["artifacts"]["tabular_X"]["path"]
    )
    assert list(exported_x.columns) == feature_spec["features"]
    assert len(exported_x) == 3
    assert "review_confidence" not in exported_x.columns
    assert "new_numeric_noise" not in exported_x.columns
    assert audit["complete_export"] is True
    assert audit["canonical_fallback_used"] is False
    assert audit["rows"]["source_input"] == 3
    assert audit["rows"]["selected_input"] == 3
    assert audit["rows"]["row_count_preserved"] is True
    for name in OUTPUT_NAMES:
        output = tmp_path / contract["artifacts"][name]["path"]
        assert output.is_file()
        assert output.resolve().is_relative_to(train_ready_root.resolve())
    assert fixture["human_root"].exists() is False


def test_bounded_export_is_marked_incomplete(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    audit = _export(
        fixture["contract_path"],
        tmp_path,
        max_rows=2,
    )

    assert audit["complete_export"] is False
    assert audit["max_rows"] == 2
    assert audit["rows"]["source_input"] == 3
    assert audit["rows"]["selected_input"] == 2
    assert audit["rows"]["row_count_preserved"] is True


def test_export_rejects_output_outside_declared_train_ready_root(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, x_outside_train_ready=True)

    with pytest.raises(
        ValueError,
        match="outside declared train_ready_root.*tabular_X",
    ):
        _export(fixture["contract_path"], tmp_path)

    assert fixture["human_root"].exists() is False


def test_export_rejects_human_owned_model_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, x_scope="human_review")

    with pytest.raises(
        ValueError,
        match="tabular_X must have scope=agent_derived",
    ):
        _export(fixture["contract_path"], tmp_path)

    assert fixture["human_root"].exists() is False


def test_export_rejects_target_derived_feature_spec(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, leaky_feature=True)

    with pytest.raises(
        ValueError,
        match="matches forbidden X patterns.*behavior_numeric_proxy",
    ):
        _export(fixture["contract_path"], tmp_path)

    assert fixture["train_ready_root"].exists() is False


def _export(
    contract_path: Path,
    project_root: Path,
    *,
    max_rows: int | None = None,
) -> dict[str, Any]:
    return EXPORT_TRAIN_READY(
        contract_path,
        project_root=project_root,
        label_col="behavior_window_label",
        mask_col="window_valid_for_main_train",
        sample_weight_col="window_sample_weight",
        max_rows=max_rows,
        overwrite=False,
    )


def _fixture(
    tmp_path: Path,
    *,
    x_scope: str = "agent_derived",
    x_outside_train_ready: bool = False,
    leaky_feature: bool = False,
) -> dict[str, Path]:
    config_dir = tmp_path / "configs" / "classification_v2"
    agent_root = (
        tmp_path
        / "outputs"
        / "classification_v2"
        / "agent_audits"
        / AGENT_RUN_ID
    )
    human_root = (
        tmp_path
        / "human_review_workspace"
        / "classification_v2"
        / HUMAN_RUN_ID
    )
    config_dir.mkdir(parents=True)
    feature_spec_path = config_dir / FEATURE_SPEC_SOURCE.name
    feature_spec = _read_json(FEATURE_SPEC_SOURCE)
    if leaky_feature:
        feature_spec["features"][0] = "behavior_numeric_proxy"
    _write_json(feature_spec_path, feature_spec)

    agent_relative = (
        f"outputs/classification_v2/agent_audits/{AGENT_RUN_ID}"
    )
    human_relative = (
        f"human_review_workspace/classification_v2/{HUMAN_RUN_ID}"
    )
    train_ready_relative = f"{agent_relative}/data/11_train_ready"
    source_relative = (
        f"{agent_relative}/data/08_sequence_reviewed/"
        "sequence_window_features.csv"
    )
    source_path = tmp_path / source_relative
    source_path.parent.mkdir(parents=True)
    _window_fixture(feature_spec["features"]).to_csv(source_path, index=False)

    artifacts = _artifact_specs(x_scope=x_scope)
    mapped = {
        name: {
            "path": _artifact_relative_path(
                name,
                scope=str(spec["scope"]),
                agent_relative=agent_relative,
                human_relative=human_relative,
                train_ready_relative=train_ready_relative,
                source_relative=source_relative,
                x_outside_train_ready=x_outside_train_ready,
            ),
            "scope": spec["scope"],
        }
        for name, spec in artifacts.items()
    }
    template_path = config_dir / "fixture_data_contract_template.json"
    _write_json(
        template_path,
        {
            "template_schema_version": DATA_CONTRACT_TEMPLATE_SCHEMA_VERSION,
            "contract_version": "classification_v2.export_fixture.v1",
            "snapshot_name": "classification_v2_export_fixture",
            "allowed_profiles": ["mixed-reviewed"],
            "primary_key": "window_id",
            "forbidden_x_patterns": [
                "manual_*",
                "review_*",
                "*review*",
                "*decision*",
                "*corrected*",
                "*behavior*",
                "*label*",
                "*_path",
                "source_*",
                "split_*",
                "target_roi_*",
                "roi_target_*",
            ],
            "artifacts": artifacts,
        },
    )
    map_path = agent_root / "contracts" / "artifact_map.json"
    map_path.parent.mkdir(parents=True)
    _write_json(
        map_path,
        {
            "schema_version": ARTIFACT_MAP_SCHEMA_VERSION,
            "run_id": AGENT_RUN_ID,
            "profile": "mixed-reviewed",
            "lineage_ids": {
                "agent_derived": AGENT_RUN_ID,
                "human_review": HUMAN_RUN_ID,
            },
            "lineage_roots": {
                "agent_derived": agent_relative,
                "human_review": human_relative,
            },
            "train_ready_root": train_ready_relative,
            "snapshot_output_dir": (
                f"{agent_relative}/data/14_training_snapshot"
            ),
            "artifacts": mapped,
        },
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
    return {
        "contract_path": contract_path,
        "feature_spec_path": feature_spec_path,
        "human_root": human_root,
        "train_ready_root": tmp_path / train_ready_relative,
    }


def _artifact_specs(*, x_scope: str) -> dict[str, dict[str, object]]:
    specs = {
        "sequence_window_features": {
            "scope": "agent_derived",
            "type": "csv",
            "required": True,
        },
        "tabular_feature_spec": {
            "scope": "project_static",
            "type": "json",
            "required": True,
        },
    }
    for name in OUTPUT_NAMES:
        specs[name] = {
            "scope": x_scope if name == "tabular_X" else "agent_derived",
            "type": "csv" if name in OUTPUT_NAMES[:4] else "json",
            "required": True,
        }
    return specs


def _artifact_relative_path(
    name: str,
    *,
    scope: str,
    agent_relative: str,
    human_relative: str,
    train_ready_relative: str,
    source_relative: str,
    x_outside_train_ready: bool,
) -> str:
    if name == "sequence_window_features":
        return source_relative
    if name == "tabular_feature_spec":
        return (
            "configs/classification_v2/"
            "reviewed_q2_tabular_feature_spec_v1.json"
        )
    suffix = ".csv" if name in OUTPUT_NAMES[:4] else ".json"
    if name == "tabular_X" and x_outside_train_ready:
        return f"{agent_relative}/data/15_model_development/{name}{suffix}"
    if scope == "human_review":
        return f"{human_relative}/invalid_model_output/{name}{suffix}"
    return f"{train_ready_relative}/{name}{suffix}"


def _window_fixture(features: list[str]) -> pd.DataFrame:
    data = {
        feature: [
            float(row_index + feature_index / 1000.0)
            for row_index in range(3)
        ]
        for feature_index, feature in enumerate(features)
    }
    data.update(
        {
            "window_id": ["window-0", "window-1", "window-2"],
            "behavior_window_label": ["stand", "move", "lying"],
            "window_valid_for_main_train": [True, False, True],
            "window_sample_weight": [1.0, 0.4, 0.8],
            "review_confidence": [1.0, 1.0, 1.0],
            "new_numeric_noise": [99.0, 98.0, 97.0],
        }
    )
    return pd.DataFrame(data)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
