from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import pig_behavior.classification_v2.evaluation.reviewed_q2_p0_preflight as p0

AGENT_RUN_ID = "c2v2_agent_p0_fixture_v1"
HUMAN_RUN_ID = "c2v2_human_p0_fixture_v1"


def test_p0_fails_closed_without_review_and_never_authorizes_full_oof(
    tmp_path: Path,
) -> None:
    contract_path, snapshot_path, output_path, human_root = _fixture(tmp_path)

    result = p0.build_reviewed_q2_p0_preflight(
        contract_path,
        snapshot_path,
        project_root=tmp_path,
        output_json=output_path,
    )

    assert result["valid"] is False
    assert result["model_smoke_authorized"] is False
    assert result["full_oof_authorized"] is False
    assert result["full_oof_authorization_required"] is True
    assert result["human_review_root_write_attempted"] is False
    assert result["errors"]
    assert output_path.exists() is False
    assert human_root.exists() is False


def test_p0_frame_parity_detects_key_loss_and_reordering(tmp_path: Path) -> None:
    paths = {}
    for name in p0.FRAME_PARITY_ARTIFACTS:
        path = tmp_path / f"{name}.csv"
        pd.DataFrame(
            {
                "scene_frame_uid": ["frame-0", "frame-1"],
                "value": [1.0, 2.0],
            }
        ).to_csv(path, index=False)
        paths[name] = path

    valid = p0._audit_frame_parity(paths)
    assert valid["valid"] is True

    pd.DataFrame(
        {
            "scene_frame_uid": ["frame-1", "frame-0"],
            "value": [2.0, 1.0],
        }
    ).to_csv(paths["reviewed_frame_features"], index=False)
    changed = p0._audit_frame_parity(paths)
    assert changed["valid"] is False
    assert "frame_ordered_key_mismatch:reviewed_frame_features" in (
        changed["errors"]
    )


def test_p0_pass_contract_still_blocks_full_oof(
    tmp_path: Path,
    monkeypatch,
) -> None:
    contract_path, snapshot_path, output_path, _ = _fixture(tmp_path)
    valid_check = {"valid": True, "errors": []}
    monkeypatch.setattr(p0, "_validate_contract", lambda *_: [])
    monkeypatch.setattr(p0, "_resolve_artifact_paths", lambda *_: {})
    monkeypatch.setattr(p0, "_audit_declared_artifacts", lambda *_: [])
    monkeypatch.setattr(
        p0,
        "_audit_model_input_manifest",
        lambda *_: valid_check.copy(),
    )
    monkeypatch.setattr(
        p0,
        "_audit_loader_input",
        lambda *_: valid_check.copy(),
    )
    monkeypatch.setattr(
        p0,
        "_audit_snapshot",
        lambda *_: valid_check.copy(),
    )
    monkeypatch.setattr(
        p0,
        "_audit_review_layer",
        lambda *_: valid_check.copy(),
    )
    monkeypatch.setattr(
        p0,
        "_audit_frame_parity",
        lambda *_: valid_check.copy(),
    )
    monkeypatch.setattr(
        p0,
        "_audit_scientific_artifacts",
        lambda *_: valid_check.copy(),
    )
    monkeypatch.setattr(
        p0,
        "_audit_keyed_artifacts",
        lambda *_: valid_check.copy(),
    )
    monkeypatch.setattr(
        p0,
        "_required_values_check",
        lambda *_: valid_check.copy(),
    )

    result = p0.build_reviewed_q2_p0_preflight(
        contract_path,
        snapshot_path,
        project_root=tmp_path,
        output_json=output_path,
    )

    assert result["valid"] is True
    assert result["model_smoke_authorized"] is True
    assert result["full_oof_authorized"] is False
    assert result["next_allowed_action"] == "model_smoke_after_short_gate"


def test_p0_writer_rejects_human_owned_output(tmp_path: Path) -> None:
    contract_path, snapshot_path, _, human_root = _fixture(tmp_path)
    result = p0.build_reviewed_q2_p0_preflight(
        contract_path,
        snapshot_path,
        project_root=tmp_path,
    )
    human_output = human_root / "p0.json"

    try:
        p0.write_reviewed_q2_p0_preflight(
            result,
            data_contract_json=contract_path,
            output_json=human_output,
            project_root=tmp_path,
            overwrite=False,
        )
    except ValueError as exc:
        assert "agent-owned" in str(exc)
    else:
        raise AssertionError("human-owned P0 output was accepted")
    assert human_output.exists() is False


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
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
    contract_path = agent_root / "contracts" / "data_contract.json"
    snapshot_path = agent_root / "data" / "14_training_snapshot" / "snapshot.json"
    output_path = agent_root / "preflight" / "reviewed_q2_p0.json"
    contract_path.parent.mkdir(parents=True)
    snapshot_path.parent.mkdir(parents=True)
    contract = {
        "generated_contract_schema_version": (
            "classification_v2.versioned_data_contract.v2"
        ),
        "profile": "mixed-reviewed",
        "run_id": AGENT_RUN_ID,
        "lineage_ids": {
            "agent_derived": AGENT_RUN_ID,
            "human_review": HUMAN_RUN_ID,
        },
        "lineage_roots": {
            "agent_derived": (
                "outputs/classification_v2/agent_audits/"
                f"{AGENT_RUN_ID}"
            ),
            "human_review": (
                "human_review_workspace/classification_v2/"
                f"{HUMAN_RUN_ID}"
            ),
        },
        "path_policy": {
            "canonical_fallback_allowed": False,
        },
        "artifacts": {},
        "template_path": "configs/classification_v2/template.json",
        "artifact_map_path": (
            "outputs/classification_v2/agent_audits/"
            f"{AGENT_RUN_ID}/contracts/artifact_map.json"
        ),
        "train_ready_root": (
            "outputs/classification_v2/agent_audits/"
            f"{AGENT_RUN_ID}/data/11_train_ready"
        ),
        "snapshot_output_dir": (
            "outputs/classification_v2/agent_audits/"
            f"{AGENT_RUN_ID}/data/14_training_snapshot"
        ),
    }
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    snapshot_path.write_text("{}", encoding="utf-8")
    return contract_path, snapshot_path, output_path, human_root
