from __future__ import annotations

import json
from pathlib import Path

import pytest

from pig_behavior.classification_v2.contracts.reviewed_q2_artifact_map import (
    ReviewedQ2ArtifactMapError,
    build_reviewed_q2_artifact_map,
    write_reviewed_q2_artifact_map,
)
from pig_behavior.classification_v2.contracts.versioned_data_contract import (
    build_versioned_data_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_TEMPLATE = (
    PROJECT_ROOT
    / "configs"
    / "classification_v2"
    / "reviewed_q2_data_contract_template_v1.json"
)
SOURCE_LAYOUT = (
    PROJECT_ROOT
    / "configs"
    / "classification_v2"
    / "reviewed_q2_artifact_layout_v1.json"
)
HUMAN_RUN_ID = "c2v2_human_review_fixture_v1"
AGENT_RUN_ID = "c2v2_agent_audit_fixture_v1"


def test_map_resolves_separate_owner_roots_without_human_writes(
    tmp_path: Path,
) -> None:
    template, layout, output = _fixture(tmp_path)

    build = build_reviewed_q2_artifact_map(
        template,
        layout,
        human_review_run_id=HUMAN_RUN_ID,
        agent_audit_run_id=AGENT_RUN_ID,
        output_path=output,
        project_root=tmp_path,
    )
    dry_run = write_reviewed_q2_artifact_map(
        build,
        dry_run=True,
        overwrite=False,
    )

    assert dry_run["status"] == "PASS"
    assert output.exists() is False
    assert _human_root(tmp_path).exists() is False
    assert build.artifact_map["lineage_ids"] == {
        "human_review": HUMAN_RUN_ID,
        "agent_derived": AGENT_RUN_ID,
    }
    reviewed_path = build.artifact_map["artifacts"][
        "reviewed_frame_features"
    ]["path"]
    split_path = build.artifact_map["artifacts"]["split_manifest"]["path"]
    assert reviewed_path.startswith(
        f"human_review_workspace/classification_v2/{HUMAN_RUN_ID}/"
    )
    assert split_path.startswith(
        "outputs/classification_v2/agent_audits/"
        f"{AGENT_RUN_ID}/"
    )

    written = write_reviewed_q2_artifact_map(
        build,
        dry_run=False,
        overwrite=False,
    )

    assert written["artifact_written"] is True
    assert output.is_file()
    assert _human_root(tmp_path).exists() is False


def test_generated_map_builds_versioned_contract(tmp_path: Path) -> None:
    template, layout, output = _fixture(tmp_path)
    build = build_reviewed_q2_artifact_map(
        template,
        layout,
        human_review_run_id=HUMAN_RUN_ID,
        agent_audit_run_id=AGENT_RUN_ID,
        output_path=output,
        project_root=tmp_path,
    )
    write_reviewed_q2_artifact_map(
        build,
        dry_run=False,
        overwrite=False,
    )
    contract_path = output.parent / "reviewed_q2_data_contract.json"

    contract_build = build_versioned_data_contract(
        template,
        output,
        output_path=contract_path,
        project_root=tmp_path,
    )

    assert contract_build.contract["run_id"] == AGENT_RUN_ID
    assert contract_build.contract["lineage_ids"]["human_review"] == (
        HUMAN_RUN_ID
    )
    assert contract_build.output_path == contract_path


def test_map_rejects_equal_owner_ids(tmp_path: Path) -> None:
    template, layout, output = _fixture(tmp_path)

    with pytest.raises(
        ReviewedQ2ArtifactMapError,
        match="human_and_agent_run_ids_must_be_distinct",
    ):
        build_reviewed_q2_artifact_map(
            template,
            layout,
            human_review_run_id=AGENT_RUN_ID,
            agent_audit_run_id=AGENT_RUN_ID,
            output_path=output,
            project_root=tmp_path,
        )


def test_map_rejects_output_outside_agent_contract_path(
    tmp_path: Path,
) -> None:
    template, layout, _ = _fixture(tmp_path)
    wrong = _human_root(tmp_path) / "artifact_map.json"

    with pytest.raises(
        ReviewedQ2ArtifactMapError,
        match="output_json_must_equal_agent_contract_path",
    ):
        build_reviewed_q2_artifact_map(
            template,
            layout,
            human_review_run_id=HUMAN_RUN_ID,
            agent_audit_run_id=AGENT_RUN_ID,
            output_path=wrong,
            project_root=tmp_path,
        )

    assert wrong.exists() is False


def test_map_rejects_incomplete_layout(tmp_path: Path) -> None:
    template, layout, output = _fixture(tmp_path)
    payload = json.loads(layout.read_text(encoding="utf-8"))
    payload["artifacts"].pop("split_manifest")
    layout.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ReviewedQ2ArtifactMapError,
        match="layout_missing_artifacts",
    ):
        build_reviewed_q2_artifact_map(
            template,
            layout,
            human_review_run_id=HUMAN_RUN_ID,
            agent_audit_run_id=AGENT_RUN_ID,
            output_path=output,
            project_root=tmp_path,
        )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    config = tmp_path / "configs" / "classification_v2"
    config.mkdir(parents=True)
    template = config / SOURCE_TEMPLATE.name
    layout = config / SOURCE_LAYOUT.name
    template.write_text(
        SOURCE_TEMPLATE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    layout.write_text(
        SOURCE_LAYOUT.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    policy = config / "hidden_review_scientific_policy_v1.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": (
                    "classification_v2.hidden_scientific_policy.v1"
                )
            }
        ),
        encoding="utf-8",
    )
    output = (
        tmp_path
        / "outputs"
        / "classification_v2"
        / "agent_audits"
        / AGENT_RUN_ID
        / "contracts"
        / "reviewed_q2_artifact_map.json"
    )
    return template, layout, output


def _human_root(tmp_path: Path) -> Path:
    return (
        tmp_path
        / "human_review_workspace"
        / "classification_v2"
        / HUMAN_RUN_ID
    )
