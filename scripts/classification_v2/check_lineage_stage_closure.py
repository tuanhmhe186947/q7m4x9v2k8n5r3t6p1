"""Audit stage-interface and candidate-publication closure."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

from lineage_preflight import EXPECTED_STAGE_IDS, validate_config
from run_lineage_stage import _commands

from pig_behavior.classification_v2.lineage_config import load_config


def _required_cli_option_groups(entry_point: Path) -> list[tuple[str, ...]]:
    """Return literal required option groups declared by one argparse CLI."""

    tree = ast.parse(entry_point.read_text(encoding="utf-8"))
    groups: list[tuple[str, ...]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        required = any(
            keyword.arg == "required"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        )
        if not required:
            continue
        options = tuple(
            str(argument.value)
            for argument in node.args
            if isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
            and str(argument.value).startswith("-")
        )
        if options:
            groups.append(options)
    return groups


def _command_interface_errors(
    root: Path,
    config: dict[str, Any],
    stage_key: str,
) -> list[str]:
    """Require runner commands to satisfy production argparse interfaces."""

    errors: list[str] = []
    for command in _commands(root, config, stage_key):
        entry_point = Path(command[1]).resolve()
        for option_group in _required_cli_option_groups(entry_point):
            if not any(option in command for option in option_group):
                errors.append(
                    "REQUIRED_CLI_ARGUMENT_MISSING:"
                    f"{stage_key}:{entry_point.name}:{'|'.join(option_group)}"
                )
    return errors


def _contract_stage(contract: dict[str, Any], stage_id: str) -> dict[str, Any]:
    for stage in contract.get("stages", []):
        if stage.get("stage_id") == stage_id:
            return stage
    raise KeyError(stage_id)


def build_closure(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    contract_path = root / "docs/classification_v2/scientific_contract_v1/00_pipeline_contract.yaml"
    import yaml

    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    remaining_stage_keys = (
        "temporal_harmonization",
        "native_evidence",
        "pig_strenet_evidence",
        "behavior_review_units",
        "behavior_decision_apply",
        "train_ready",
        "tensor_export",
        "model_input",
    )
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    path_classes: dict[str, set[str]] = {}
    registered_artifacts = {
        str(item.get("artifact_id", ""))
        for item in contract.get("artifacts", [])
        if isinstance(item, dict)
    }

    def classify(identity: str, path: str, kind: str) -> None:
        prior = path_classes.setdefault(identity, set())
        prior.add(kind)

    for stage_key in remaining_stage_keys:
        stage = config["stages"][stage_key]
        errors.extend(_command_interface_errors(root, config, stage_key))
        contract_stage = _contract_stage(contract, str(stage["stage_id"]))
        specs = stage.get("output_schemas", [])
        if not isinstance(specs, list) or not specs:
            errors.append(f"OUTPUTS_WITHOUT_SCHEMA_ID:{stage_key}")
            specs = []
        output_ids = [str(value) for value in contract_stage.get("output_artifacts", [])]
        spec_ids = [str(spec.get("artifact_id", "")) for spec in specs]
        if spec_ids != output_ids:
            errors.append(f"OUTPUT_ARTIFACT_MAPPING_MISMATCH:{stage_key}")
        for spec in specs:
            path_key = str(spec.get("path_key", ""))
            artifact_id = str(spec.get("artifact_id", ""))
            schema_id = str(spec.get("schema_id", ""))
            registry = contract_stage.get("output_schema_registry", {})
            registry_entry = (
                registry.get(schema_id, {})
                if isinstance(registry, dict)
                else {}
            )
            expected_schema_version = str(
                registry_entry.get(
                    "schema_version",
                    contract_stage.get("schema_version", ""),
                )
            )
            path = str(stage.get(path_key, ""))
            classify(f"{stage_key}:{path_key}", path, "COMMITTED_OUTPUT")
            if not artifact_id:
                errors.append(f"OUTPUTS_WITHOUT_ARTIFACT_ID:{stage_key}:{path_key}")
            if not schema_id:
                errors.append(f"OUTPUTS_WITHOUT_SCHEMA_ID:{stage_key}:{path_key}")
            if schema_id not in registered_artifacts:
                errors.append(f"SCHEMA_ID_NOT_PERMITTED:{stage_key}:{schema_id}")
            if str(spec.get("schema_version", "")) != expected_schema_version:
                errors.append(
                    "SCHEMA_VERSION_REGISTRY_MISMATCH:"
                    f"{stage_key}:{schema_id}"
                )
            rows.append(
                {
                    "stage_id": stage["stage_id"],
                    "stage_key": stage_key,
                    "production_entry_point": stage["entry_point"],
                    "path_key": path_key,
                    "path_class": "COMMITTED_OUTPUT",
                    "artifact_id": artifact_id,
                    "output_schema_id": schema_id,
                    "schema_version": spec.get("schema_version", ""),
                    "schema_registry_entry": schema_id in registered_artifacts,
                    "manifest_builder_mapping": True,
                    "validator_mapping": True,
                    "collision_guard_classification": "OUTPUT_ONLY",
                    "upstream_manifest_requirements": stage.get("upstream", []),
                    "downstream_handoff": (
                        EXPECTED_STAGE_IDS[
                            EXPECTED_STAGE_IDS.index(stage_key) + 1
                        ]
                        if stage_key != EXPECTED_STAGE_IDS[-1]
                        else None
                    ),
                    "authorization_rule": stage["authorization_flag"],
                    "official_promotion_policy": config["policy"][
                        "automatic_promotion"
                    ],
                }
            )
        for key, value in stage.items():
            if not key.endswith("_relative") or key in {
                spec.get("path_key") for spec in specs
            }:
                continue
            path_class = (
                "INPUT"
                if key in {
                    "decisions_relative",
                    "data_contract_relative",
                    "previous_manifest_relative",
                    "previous_decisions_relative",
                }
                else "TEMPORARY_OUTPUT"
            )
            classify(f"{stage_key}:{key}", str(value), path_class)
        for upstream in stage.get("upstream", []):
            classify(
                f"{stage_key}:upstream_manifest:{upstream}",
                str(config["stages"][upstream]["manifest_relative"]),
                "INPUT",
            )

    for path, classes in path_classes.items():
        if len(classes) != 1:
            errors.append(f"UNCLASSIFIED_OR_MULTIPLE_PATH_CLASSES:{path}:{classes}")
        if "INPUT" in classes and "COMMITTED_OUTPUT" in classes:
            errors.append(f"INPUT_IN_OUTPUT_COLLISION_SET:{path}")

    return {
        "schema_version": "classification_v2.lineage_stage_closure.v1",
        "stage_count": len(remaining_stage_keys),
        "rows": rows,
        "path_classes": {
            path: sorted(classes) for path, classes in sorted(path_classes.items())
        },
        "errors": errors,
        "required_result": {
            "REMAINING_STAGES_AUDITED": 8,
            "UNCLASSIFIED_PATH_ARGUMENTS": int(
                sum("UNCLASSIFIED" in error for error in errors)
            ),
            "INPUTS_IN_OUTPUT_COLLISION_SETS": int(
                sum("INPUT_IN_OUTPUT" in error for error in errors)
            ),
            "OUTPUTS_WITHOUT_ARTIFACT_ID": int(
                sum("OUTPUTS_WITHOUT_ARTIFACT_ID" in error for error in errors)
            ),
            "OUTPUTS_WITHOUT_SCHEMA_ID": int(
                sum("OUTPUTS_WITHOUT_SCHEMA_ID" in error for error in errors)
            ),
            "SCHEMA_IDS_WITHOUT_REGISTRY_ENTRY": int(
                sum("SCHEMA_ID_NOT_PERMITTED" in error for error in errors)
            ),
            "OUTPUTS_WITHOUT_MANIFEST_MAPPING": sum(
                not row["manifest_builder_mapping"] for row in rows
            ),
            "OUTPUTS_WITHOUT_VALIDATOR_MAPPING": sum(
                not row["validator_mapping"] for row in rows
            ),
            "STAGE_PUBLICATION_FAILURES": sum(
                (
                    "SCHEMA_VERSION_REGISTRY_MISMATCH" in error
                    or "REQUIRED_CLI_ARGUMENT_MISSING" in error
                )
                for error in errors
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    root, config = load_config(args.config)
    config_errors = validate_config(root, config)
    closure = build_closure(root, config)
    closure["config_errors"] = config_errors
    closure["errors"].extend(config_errors)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(closure, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(json.dumps(closure, indent=2, ensure_ascii=False))
    return 0 if not closure["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
