"""Validate one Classification V2 stage without rerunning global acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lineage_preflight import EXPECTED_STAGE_IDS, validate_config

from pig_behavior.classification_v2.contracts.candidate_manifest import (
    validate_upstream_manifest_for_current_authority,
)
from pig_behavior.classification_v2.contracts.semantic_lineage import (
    validate_artifact_manifest,
)
from pig_behavior.classification_v2.lineage_config import (
    load_config,
    resolve_stage_path,
    source_bundle_report,
)


def _artifact_path(
    root: Path,
    config: dict[str, object],
    stage_key: str,
) -> Path:
    stages = config["stages"]
    if not isinstance(stages, dict):
        raise ValueError("stages must be a mapping")
    stage = stages[stage_key]
    if not isinstance(stage, dict):
        raise ValueError(f"invalid stage config: {stage_key}")
    key = "artifact_relative" if "artifact_relative" in stage else "output_relative"
    return resolve_stage_path(root, config, stage_key, key)


def _audit_errors(path: Path) -> list[str]:
    if not path.is_file():
        return [f"AUDIT_MISSING:{path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"AUDIT_INVALID:{type(exc).__name__}"]
    errors = list(payload.get("errors") or [])
    if payload.get("valid") is False or payload.get("status") == "FAIL":
        errors.append("AUDIT_DECLARED_FAILURE")
    return [str(error) for error in errors]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", required=True, choices=EXPECTED_STAGE_IDS)
    args = parser.parse_args()
    root, config = load_config(args.config)
    errors = validate_config(root, config)
    stage = config["stages"][args.stage]
    output = _artifact_path(root, config, args.stage)
    manifest = resolve_stage_path(root, config, args.stage, "manifest_relative")
    manifest_errors = []
    upstream_manifests = {}
    contract_root = root / "docs/classification_v2/scientific_contract_v1"
    for upstream_key in stage["upstream"]:
        upstream_path = resolve_stage_path(
            root,
            config,
            upstream_key,
            "manifest_relative",
        )
        try:
            upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
            upstream_manifests[str(upstream["artifact_id"])] = upstream
        except (KeyError, OSError, ValueError) as exc:
            manifest_errors.append(
                f"UPSTREAM_MANIFEST_INVALID:{upstream_key}:{type(exc).__name__}"
            )
            continue
        authority = validate_upstream_manifest_for_current_authority(
            manifest_path=upstream_path,
            repo_root=root,
            contract_root=contract_root,
            intended_downstream_stage_id=str(stage["stage_id"]),
        )
        if not authority.current_authoritative:
            manifest_errors.extend(authority.reason_codes)
    if manifest.is_file():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            validation = validate_artifact_manifest(
                payload,
                output_path=output,
                upstream_manifests=upstream_manifests,
            )
            manifest_errors.extend(validation["errors"])
            if payload.get("stage_id") != stage["stage_id"]:
                manifest_errors.append("MANIFEST_STAGE_ID_MISMATCH")
        except (OSError, ValueError) as exc:
            manifest_errors.append(f"MANIFEST_INVALID:{type(exc).__name__}")
    else:
        manifest_errors.append("MANIFEST_MISSING")
    audit_errors = []
    if "audit_relative" in stage:
        audit_errors = _audit_errors(
            resolve_stage_path(root, config, args.stage, "audit_relative")
        )
    source_valid = bool(source_bundle_report(root, config)["valid"])
    report = {
        "stage": args.stage,
        "stage_id": stage["stage_id"],
        "stage_local": True,
        "output_exists": output.is_file(),
        "manifest_exists": manifest.is_file(),
        "manifest_errors": manifest_errors,
        "audit_errors": audit_errors,
        "source_fingerprints_valid": source_valid,
        "config_errors": errors,
        "valid": (
            not errors
            and output.is_file()
            and manifest.is_file()
            and not manifest_errors
            and not audit_errors
            and source_valid
        ),
    }
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
