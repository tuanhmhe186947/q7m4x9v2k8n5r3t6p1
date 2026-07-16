from __future__ import annotations

import json
from pathlib import Path

from pig_behavior.classification_v2.contracts.versioned_data_contract import (
    ARTIFACT_MAP_SCHEMA_VERSION,
    DATA_CONTRACT_TEMPLATE_SCHEMA_VERSION,
    build_versioned_data_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = (
    PROJECT_ROOT
    / "configs"
    / "classification_v2"
    / "reviewed_q2_data_contract_template_v1.json"
)
FEATURE_SPEC_PATH = (
    PROJECT_ROOT
    / "configs"
    / "classification_v2"
    / "reviewed_q2_tabular_feature_spec_v1.json"
)
TRAINER_CONTRACT_PATH = (
    PROJECT_ROOT
    / "configs"
    / "classification_v2"
    / "trainer_contract_v1.json"
)
RUN_ID = "c2v2_reviewed_q2_contract_fixture_v1"
HUMAN_RUN_ID = "c2v2_human_review_contract_fixture_v1"


def test_reviewed_q2_template_is_path_free_and_scope_complete() -> None:
    template = _load_template()
    artifacts = template["artifacts"]

    assert (
        template["template_schema_version"]
        == DATA_CONTRACT_TEMPLATE_SCHEMA_VERSION
    )
    assert template["allowed_profiles"] == ["mixed-reviewed"]
    assert {
        "root",
        "train_ready_root",
        "snapshot_output_dir",
    }.isdisjoint(template)
    assert all("path" not in spec for spec in artifacts.values())
    assert all(
        spec["scope"]
        in {"human_review", "agent_derived", "project_static"}
        for spec in artifacts.values()
    )

    required_human = {
        "hidden_review_unit_manifest",
        "hidden_review_decisions",
        "hidden_review_decision_coverage_audit",
        "hidden_review_scientific_gate",
        "hidden_apply_audit",
        "hidden_reviewed_frame_features",
        "full_review_unit_manifest",
        "roi_behavior_decisions",
        "motion_behavior_decisions",
        "posture_behavior_decisions",
        "interaction_behavior_decisions",
        "behavior_decision_coverage_audit",
        "behavior_apply_audit",
        "reviewed_frame_features",
    }
    required_agent = {
        "harmonized_frames",
        "temporal_label_intervals",
        "sequence_window_manifest",
        "native_temporal_unit_manifest",
        "q2_outer_inner_roles",
        "split_manifest",
        "tabular_X",
        "spatial_sequences",
        "feature_whitelist",
        "feature_blacklist",
        "leakage_audit",
        "loader_input_audit",
        "source_image_loader_audit",
    }
    assert required_human.issubset(artifacts)
    assert required_agent.issubset(artifacts)
    assert all(
        artifacts[name]["scope"] == "human_review"
        for name in required_human
    )
    assert all(
        artifacts[name]["scope"] == "agent_derived"
        for name in required_agent
    )
    assert not any(
        "native_oof" in name or "loro" in name
        for name in artifacts
    )


def test_reviewed_q2_template_locks_regression_case_evidence() -> None:
    artifacts = _load_template()["artifacts"]
    anchor = artifacts["cvat_anchor_1020_audit"]["required_json_values"]
    resolver = artifacts["source_image_loader_audit"]["required_json_values"]

    assert anchor["video_key"] == "Pigs281119_000085_30fps"
    assert anchor["pig_id"] == "ID_4"
    assert anchor["anchor"] == 1020
    assert anchor["expected_behavior"] == "social-nose"
    assert anchor["expected_template"] == "interaction"
    assert anchor["valid"] is True
    assert resolver["mandatory_gui_video_case.rows"] == 6
    assert resolver[
        "mandatory_gui_video_case.expected_media_basename"
    ] == "Pigs291119_000231_30fps.mp4"
    assert resolver["mandatory_gui_video_case.ok"] is True


def test_reviewed_q2_tabular_spec_is_explicit_unique_and_migrated() -> None:
    feature_spec = json.loads(FEATURE_SPEC_PATH.read_text(encoding="utf-8"))
    trainer_contract = json.loads(
        TRAINER_CONTRACT_PATH.read_text(encoding="utf-8")
    )
    features = feature_spec["features"]

    assert feature_spec["profile"] == "mixed-reviewed"
    assert feature_spec["selection_policy"] == {
        "explicit_ordered_whitelist": True,
        "all_numeric_selection_allowed": False,
        "unknown_feature_fails_closed": True,
        "inference_available_only": True,
    }
    assert len(features) == 110
    assert len(features) == len(set(features))
    assert features == trainer_contract["tabular_feature_whitelist"]


def test_train_ready_audit_requires_complete_run_bound_export() -> None:
    required = _load_template()["artifacts"]["train_ready_audit"][
        "required_json_values"
    ]

    assert required["complete_export"] is True
    assert required["canonical_fallback_used"] is False
    assert required["valid"] is True


def test_reviewed_q2_template_builds_with_owner_separated_map(
    tmp_path: Path,
) -> None:
    template = _load_template()
    template_path = (
        tmp_path
        / "configs"
        / "classification_v2"
        / TEMPLATE_PATH.name
    )
    template_path.parent.mkdir(parents=True)
    template_path.write_text(
        json.dumps(template),
        encoding="utf-8",
    )
    policy_path = (
        tmp_path
        / "configs"
        / "classification_v2"
        / "hidden_review_scientific_policy_v1.json"
    )
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
    feature_spec_path = (
        tmp_path
        / "configs"
        / "classification_v2"
        / FEATURE_SPEC_PATH.name
    )
    feature_spec_path.write_text(
        FEATURE_SPEC_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    agent_root = f"outputs/classification_v2/agent_audits/{RUN_ID}"
    human_root = (
        "human_review_workspace/classification_v2/"
        f"{HUMAN_RUN_ID}"
    )
    mapped = {
        name: {
            "path": _mapped_path(
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
                "artifacts": mapped,
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / agent_root / "contracts" / "data_contract.json"

    build = build_versioned_data_contract(
        template_path,
        map_path,
        output_path=output_path,
        project_root=tmp_path,
    )

    assert build.audit["status"] == "PASS"
    assert build.audit["artifact_count"] == len(template["artifacts"])
    assert build.audit["human_review_artifact_count"] > 0
    assert build.audit["agent_derived_artifact_count"] > 0
    assert build.contract["owner_policy"] == template["owner_policy"]
    assert (
        build.contract["path_policy"]["canonical_fallback_allowed"]
        is False
    )


def _mapped_path(
    name: str,
    spec: dict[str, object],
    *,
    agent_root: str,
    human_root: str,
) -> str:
    scope = str(spec["scope"])
    if scope == "project_static":
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
    root = human_root if scope == "human_review" else agent_root
    return f"{root}/artifacts/{name}{suffix}"


def _load_template() -> dict[str, object]:
    return json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
