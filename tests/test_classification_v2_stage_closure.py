from __future__ import annotations

import copy
import sys
from pathlib import Path

from pig_behavior.classification_v2.lineage_config import load_config

SCRIPTS = Path(__file__).parents[1] / "scripts" / "classification_v2"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import check_lineage_stage_closure  # noqa: E402


def test_remaining_stage_publication_closure_is_complete() -> None:
    root, config = load_config()
    closure = check_lineage_stage_closure.build_closure(root, config)

    assert closure["stage_count"] == 8
    assert closure["errors"] == []
    assert closure["required_result"] == {
        "REMAINING_STAGES_AUDITED": 8,
        "UNCLASSIFIED_PATH_ARGUMENTS": 0,
        "INPUTS_IN_OUTPUT_COLLISION_SETS": 0,
        "OUTPUTS_WITHOUT_ARTIFACT_ID": 0,
        "OUTPUTS_WITHOUT_SCHEMA_ID": 0,
        "SCHEMA_IDS_WITHOUT_REGISTRY_ENTRY": 0,
        "OUTPUTS_WITHOUT_MANIFEST_MAPPING": 0,
        "OUTPUTS_WITHOUT_VALIDATOR_MAPPING": 0,
        "STAGE_PUBLICATION_FAILURES": 0,
    }


def test_temporal_stage_publishes_both_declared_outputs() -> None:
    root, config = load_config()
    temporal = config["stages"]["temporal_harmonization"]
    specs = temporal["output_schemas"]

    assert [spec["artifact_id"] for spec in specs] == [
        "artifact.harmonized_frames",
        "artifact.temporal_intervals",
    ]
    assert [spec["schema_id"] for spec in specs] == [
        "artifact.harmonized_frames",
        "artifact.temporal_intervals",
    ]
    assert all(spec["schema_version"] for spec in specs)


def test_required_production_cli_arguments_are_closed() -> None:
    root, config = load_config()

    for stage_id in (
        "temporal_harmonization",
        "native_evidence",
        "pig_strenet_evidence",
        "behavior_review_units",
        "behavior_decision_apply",
        "train_ready",
        "tensor_export",
        "model_input",
    ):
        assert check_lineage_stage_closure._command_interface_errors(
            root,
            config,
            stage_id,
        ) == []


def test_missing_required_production_cli_argument_fails_closure(
    monkeypatch,
) -> None:
    root, config = load_config()
    config = copy.deepcopy(config)
    original = check_lineage_stage_closure._commands

    def missing_run_scope(root_arg, config_arg, stage_id):
        commands = original(root_arg, config_arg, stage_id)
        if stage_id == "pig_strenet_evidence":
            command = commands[0]
            index = command.index("--run-scope")
            commands[0] = command[:index] + command[index + 2 :]
        return commands

    monkeypatch.setattr(
        check_lineage_stage_closure,
        "_commands",
        missing_run_scope,
    )

    closure = check_lineage_stage_closure.build_closure(root, config)

    assert any(
        error.endswith(
            "pig_strenet_evidence:"
            "classification_v2_build_pig_strenet_artifacts.py:"
            "--run-scope"
        )
        for error in closure["errors"]
    )
    assert closure["required_result"]["STAGE_PUBLICATION_FAILURES"] == 1
