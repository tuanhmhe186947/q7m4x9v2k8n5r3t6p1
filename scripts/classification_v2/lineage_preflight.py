"""Read-only operational preflight for the Classification V2 lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.lineage_config import (
    load_config,
    reject_stale_path,
    source_bundle_report,
    stable_stage_ids,
)

EXPECTED_STAGE_IDS = (
    "source_merge",
    "frame_local",
    "hidden_design",
    "hidden_decision_migration",
    "hidden_coverage_gate",
    "hidden_apply",
    "temporal_harmonization",
    "native_evidence",
    "pig_strenet_evidence",
    "behavior_review_units",
    "behavior_decision_apply",
    "train_ready",
    "tensor_export",
    "model_input",
)


def validate_config(root: Path, config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if tuple(config.get("stage_order", ())) != EXPECTED_STAGE_IDS:
        errors.append("STAGE_ORDER_MISMATCH")
    if tuple(stable_stage_ids(config)) != EXPECTED_STAGE_IDS:
        errors.append("STAGE_ID_MISMATCH")
    for stage_id in EXPECTED_STAGE_IDS:
        stage = config.get("stages", {}).get(stage_id, {})
        entry = root / str(stage.get("entry_point", ""))
        if not entry.is_file():
            errors.append(f"MISSING_ENTRY_POINT:{stage_id}")
        post_entry = stage.get("post_entry_point")
        if post_entry and not (root / str(post_entry)).is_file():
            errors.append(f"MISSING_POST_ENTRY_POINT:{stage_id}")
        for key in ("output_relative", "manifest_relative"):
            if key not in stage:
                errors.append(f"MISSING_STAGE_PATH:{stage_id}:{key}")
        output_schemas = stage.get("output_schemas")
        if not isinstance(output_schemas, list) or not output_schemas:
            errors.append(f"MISSING_OUTPUT_SCHEMA_REGISTRY:{stage_id}")
        else:
            seen_paths: set[str] = set()
            seen_artifacts: set[str] = set()
            for spec in output_schemas:
                if not isinstance(spec, dict):
                    errors.append(f"INVALID_OUTPUT_SCHEMA_SPEC:{stage_id}")
                    continue
                path_key = str(spec.get("path_key", ""))
                artifact_id = str(spec.get("artifact_id", ""))
                schema_id = str(spec.get("schema_id", ""))
                if path_key not in stage:
                    errors.append(
                        f"OUTPUT_SCHEMA_PATH_MISSING:{stage_id}:{path_key}"
                    )
                if not artifact_id or not schema_id:
                    errors.append(f"OUTPUT_SCHEMA_AUTHORITY_MISSING:{stage_id}")
                if artifact_id in seen_artifacts:
                    errors.append(f"DUPLICATE_OUTPUT_ARTIFACT:{stage_id}:{artifact_id}")
                if path_key in seen_paths:
                    errors.append(f"DUPLICATE_OUTPUT_PATH:{stage_id}:{path_key}")
                seen_artifacts.add(artifact_id)
                seen_paths.add(path_key)
    source = config.get("source", {})
    xmls = source.get("cvat_behavior_xml", [])
    if len(xmls) != 12:
        errors.append("CVAT_XML_SELECTION_COUNT_NOT_12")
    if len(set(xmls)) != len(xmls):
        errors.append("CVAT_XML_SELECTION_DUPLICATE")
    for value in _flatten_strings(config):
        try:
            reject_stale_path(value)
        except ValueError:
            errors.append(f"STALE_PATH_IN_CONFIG:{value}")
    if config.get("policy", {}).get("no_overwrite") is not True:
        errors.append("NO_OVERWRITE_POLICY_REQUIRED")
    if config.get("policy", {}).get("automatic_promotion") is not False:
        errors.append("AUTOMATIC_PROMOTION_MUST_BE_FALSE")
    if config.get("policy", {}).get("automatic_downstream_execution") is not False:
        errors.append("AUTOMATIC_DOWNSTREAM_EXECUTION_MUST_BE_FALSE")
    authorization = config.get("authorization", {})
    if not authorization or any(
        value is not False for value in authorization.values()
    ):
        errors.append("CANONICAL_AUTHORIZATION_FLAGS_MUST_REMAIN_FALSE")
    return errors


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        result: list[str] = []
        for key, child in value.items():
            result.extend(_flatten_strings(key))
            result.extend(_flatten_strings(child))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for child in value:
            result.extend(_flatten_strings(child))
        return result
    return [str(value)] if isinstance(value, (str, Path)) else []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root, config = load_config(args.config)
    errors = validate_config(root, config)
    source_report = source_bundle_report(root, config)
    valid = not errors and bool(source_report["valid"])
    report = {
        "status": "PASS" if valid else "FAIL",
        "config": str(args.config.resolve()),
        "repository_root": str(root),
        "stage_count": len(config["stage_order"]),
        "source_bundle": source_report,
        "release_authority_all_false": all(
            value is False for value in config["authorization"].values()
        ),
        "errors": errors,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
