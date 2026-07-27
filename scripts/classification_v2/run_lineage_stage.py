"""Run exactly one authorized Classification V2 lineage stage."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from lineage_preflight import EXPECTED_STAGE_IDS, validate_config

from pig_behavior.classification_v2.contracts.candidate_manifest import (
    build_candidate_artifact_manifest,
    validate_upstream_manifest_for_current_authority,
)
from pig_behavior.classification_v2.features.motion_schema import (
    MOTION_FEATURE_NAMES,
)
from pig_behavior.classification_v2.lineage_authorization import (
    consume_stage_authorization,
    validate_stage_authorization,
)
from pig_behavior.classification_v2.lineage_config import (
    load_config,
    resolve_source_path,
    resolve_stage_path,
    source_bundle_report,
)


def _stage_path(
    root: Path,
    config: dict[str, Any],
    stage_id: str,
    key: str = "output_relative",
) -> str:
    return str(resolve_stage_path(root, config, stage_id, key))


def _output_specs(
    root: Path,
    config: dict[str, Any],
    stage_id: str,
) -> list[dict[str, Any]]:
    """Resolve the explicit committed-output schema registry for a stage."""

    stage = config["stages"][stage_id]
    specs = stage.get("output_schemas", [])
    if not isinstance(specs, list) or not specs:
        raise ValueError(f"OUTPUT_SCHEMA_REGISTRY_MISSING:{stage_id}")
    resolved: list[dict[str, Any]] = []
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError(f"OUTPUT_SCHEMA_SPEC_INVALID:{stage_id}")
        item = dict(spec)
        item["path"] = resolve_stage_path(
            root,
            config,
            stage_id,
            str(item["path_key"]),
        )
        resolved.append(item)
    return resolved


def _command(root: Path, config: dict[str, Any], stage_id: str) -> list[str]:
    stage = config["stages"][stage_id]
    entry = str(root / stage["entry_point"])
    source = config["source"]
    if stage_id == "source_merge":
        command = [
            sys.executable,
            entry,
            "--legacy-csv",
            str(resolve_source_path(root, config, "legacy_export")),
        ]
        for xml in source["cvat_behavior_xml"]:
            command.extend(["--cvat-tracking-xml", str(root / xml)])
        command.extend(
            [
                "--fps",
                "30",
                "--require-full-8-for-eval",
                "--output-csv",
                str(resolve_stage_path(root, config, stage_id, "output_relative")),
                "--audit-json",
                str(resolve_stage_path(root, config, stage_id, "audit_relative")),
                "--lineage-json",
                str(
                    resolve_stage_path(
                        root,
                        config,
                        stage_id,
                        "lineage_relative",
                    )
                ),
            ]
        )
        return command
    if stage_id == "frame_local":
        return [
            sys.executable,
            entry,
            "--input-csv",
            str(resolve_stage_path(root, config, "source_merge", "output_relative")),
            "--roi-coco",
            str(resolve_source_path(root, config, "roi")),
            "--pen-mask",
            str(resolve_source_path(root, config, "pen_mask")),
            "--output-csv",
            str(resolve_stage_path(root, config, stage_id, "output_relative")),
            "--schema-json",
            str(resolve_stage_path(root, config, stage_id, "schema_relative")),
            "--audit-json",
            str(resolve_stage_path(root, config, stage_id, "audit_relative")),
            "--lineage-id",
            str(config["lineage_id"]),
            "--code-authority-sha",
            str(config["scientific_accepted_sha"]),
        ]
    if stage_id == "hidden_design":
        return [
            sys.executable,
            entry,
            "--input-csv",
            str(resolve_stage_path(root, config, "frame_local", "output_relative")),
            "--output-dir",
            str(resolve_stage_path(root, config, stage_id, "output_relative")),
            "--design-scope",
            "full",
        ]
    if stage_id == "hidden_decision_migration":
        return [
            sys.executable,
            entry,
            "--previous-manifest-csv",
            _stage_path(root, config, stage_id, "previous_manifest_relative"),
            "--current-manifest-csv",
            _stage_path(root, config, "hidden_design", "artifact_relative"),
            "--decisions-csv",
            _stage_path(root, config, stage_id, "previous_decisions_relative"),
            "--output-decisions-csv",
            _stage_path(root, config, stage_id, "artifact_relative"),
            "--audit-json",
            _stage_path(root, config, stage_id, "audit_relative"),
            "--apply",
        ]
    if stage_id == "hidden_coverage_gate":
        return [
            sys.executable,
            entry,
            "--manifest-csv",
            _stage_path(root, config, "hidden_design", "artifact_relative"),
            "--decisions-csv",
            _stage_path(
                root,
                config,
                "hidden_decision_migration",
                "artifact_relative",
            ),
            "--design-json",
            str(
                resolve_stage_path(
                    root,
                    config,
                    "hidden_design",
                    "output_relative",
                )
                / "hidden_review_scientific_design.json"
            ),
            "--audit-json",
            _stage_path(root, config, stage_id),
        ]
    if stage_id == "hidden_apply":
        return [
            sys.executable,
            entry,
            "--input-csv",
            _stage_path(root, config, "frame_local"),
            "--manifest-csv",
            _stage_path(root, config, "hidden_design", "artifact_relative"),
            "--decisions-csv",
            _stage_path(
                root,
                config,
                "hidden_decision_migration",
                "artifact_relative",
            ),
            "--output-csv",
            _stage_path(root, config, stage_id),
            "--audit-json",
            _stage_path(root, config, stage_id, "audit_relative"),
            "--confusion-audit-json",
            _stage_path(root, config, stage_id, "confusion_audit_relative"),
        ]
    if stage_id == "temporal_harmonization":
        return [
            sys.executable,
            entry,
            "--input-csv",
            _stage_path(root, config, "hidden_apply"),
            "--output-csv",
            _stage_path(root, config, stage_id),
            "--intervals-csv",
            _stage_path(root, config, stage_id, "intervals_relative"),
            "--audit-json",
            _stage_path(root, config, stage_id, "audit_relative"),
            "--cvat-label-stride",
            "6",
            "--legacy-expected-sequence-length",
            "16",
        ]
    if stage_id == "native_evidence":
        return [
            sys.executable,
            entry,
            "--input-csv",
            _stage_path(root, config, "temporal_harmonization"),
            "--output-csv",
            _stage_path(root, config, stage_id),
            "--audit-json",
            _stage_path(root, config, stage_id, "audit_relative"),
            "--code-sha",
            str(config["scientific_accepted_sha"]),
            "--pen-mask",
            str(resolve_source_path(root, config, "pen_mask")),
        ]
    if stage_id == "pig_strenet_evidence":
        return [
            sys.executable,
            entry,
            "--input-csv",
            _stage_path(root, config, "native_evidence"),
            "--output-dir",
            _stage_path(root, config, stage_id),
            "--roi-coco",
            str(resolve_source_path(root, config, "roi")),
            "--video-root",
            str(resolve_source_path(root, config, "video_root")),
            "--legacy-crop-root",
            str(resolve_source_path(root, config, "legacy_crop_root")),
            "--run-scope",
            "full",
            "--lineage-scope",
            str(config["lineage_id"]),
            "--human-review-complete",
            "false",
        ]
    if stage_id == "behavior_review_units":
        return [
            sys.executable,
            entry,
            "--intervals-csv",
            _stage_path(
                root,
                config,
                "temporal_harmonization",
                "intervals_relative",
            ),
            "--native-only",
            "--output-dir",
            _stage_path(root, config, stage_id),
            "--include-all-retained-legacy-units",
            "--full-native-unit-behavior-review",
            "--pig-strenet-artifact-dir",
            _stage_path(root, config, "pig_strenet_evidence"),
        ]
    if stage_id == "behavior_decision_apply":
        return [
            sys.executable,
            entry,
            "--frame-features-csv",
            _stage_path(root, config, "native_evidence"),
            "--review-unit-manifest-csv",
            _stage_path(
                root,
                config,
                "behavior_review_units",
                "artifact_relative",
            ),
            "--decisions-csv",
            _stage_path(root, config, stage_id, "decisions_relative"),
            "--output-csv",
            _stage_path(root, config, stage_id),
            "--audit-json",
            _stage_path(root, config, stage_id, "audit_relative"),
            "--combined-decisions-csv",
            _stage_path(
                root,
                config,
                stage_id,
                "combined_decisions_relative",
            ),
        ]
    if stage_id == "train_ready":
        return [
            sys.executable,
            entry,
            "--input-csv",
            _stage_path(root, config, "behavior_decision_apply"),
            "--output-dir",
            _stage_path(root, config, stage_id),
            "--window-lengths",
            "6,8,12,16",
            "--include-legacy-sparse-s6-at16",
            "--behavior-review-requirement",
            "full_native_unit_review_required",
            "--cvat-label-stride",
            "6",
            "--legacy-expected-sequence-length",
            "16",
            "--disable-fast-reuse",
        ]
    if stage_id == "tensor_export":
        return [
            sys.executable,
            entry,
            "--window-manifest-csv",
            str(
                resolve_stage_path(
                    root,
                    config,
                    "train_ready",
                    "output_relative",
                )
                / "sequence_window_manifest.csv"
            ),
            "--frame-features-csv",
            _stage_path(root, config, "behavior_decision_apply"),
            "--output-dir",
            _stage_path(root, config, stage_id),
            "--compress",
        ]
    if stage_id == "model_input":
        return [
            sys.executable,
            entry,
            "--data-contract-json",
            _stage_path(root, config, stage_id, "data_contract_relative"),
            "--output-json",
            _stage_path(root, config, stage_id),
            "--project-root",
            str(root),
        ]
    raise ValueError(f"UNKNOWN_STAGE:{stage_id}")


def _commands(
    root: Path,
    config: dict[str, Any],
    stage_id: str,
) -> list[list[str]]:
    commands = [_command(root, config, stage_id)]
    if stage_id == "train_ready":
        stage = config["stages"][stage_id]
        commands.append(
            [
                sys.executable,
                str(root / stage["post_entry_point"]),
                "--input-csv",
                str(
                    resolve_stage_path(
                        root,
                        config,
                        stage_id,
                        "output_relative",
                    )
                    / "sequence_window_features.csv"
                ),
                "--output-dir",
                _stage_path(root, config, stage_id),
                "--trainer-contract-json",
                str(
                    root
                    / "configs/classification_v2/trainer_contract_v1.json"
                ),
            ]
        )
    return commands


def _output_collision(root: Path, config: dict[str, Any], stage_id: str) -> bool:
    stage = config["stages"][stage_id]
    paths = [spec["path"] for spec in _output_specs(root, config, stage_id)]
    for key in (
        "audit_relative",
        "manifest_relative",
        "lineage_relative",
        "confusion_audit_relative",
        "combined_decisions_relative",
        "sequence_audit_relative",
        "schema_relative",
    ):
        if stage.get(key):
            paths.append(resolve_stage_path(root, config, stage_id, key))
    return any(path.exists() for path in paths)


def _existing_output_errors(
    root: Path,
    config: dict[str, Any],
    stage_id: str,
) -> list[str]:
    """Validate outputs eligible for publication-only recovery."""

    errors = []
    for spec in _output_specs(root, config, stage_id):
        path = Path(spec["path"])
        if not path.is_file():
            errors.append(f"COMMITTED_OUTPUT_MISSING:{path}")
    stage = config["stages"][stage_id]
    for key in ("audit_relative", "lineage_relative"):
        if stage.get(key):
            path = resolve_stage_path(root, config, stage_id, key)
            if not path.is_file():
                errors.append(f"PUBLICATION_EVIDENCE_MISSING:{path}")
    manifest = resolve_stage_path(root, config, stage_id, "manifest_relative")
    if manifest.exists():
        errors.append(f"CANDIDATE_MANIFEST_COLLISION:{manifest}")
    return errors


def _resume_errors(
    root: Path,
    config: dict[str, Any],
    stage_id: str,
) -> list[str]:
    """Validate the narrow checkpoint-resume exception to collision refusal."""

    if stage_id != "pig_strenet_evidence":
        return [f"STAGE_RESUME_UNSUPPORTED:{stage_id}"]
    output_dir = resolve_stage_path(
        root,
        config,
        stage_id,
        "output_relative",
    )
    identity = output_dir / ".checkpoints" / "checkpoint_identity.json"
    manifest = resolve_stage_path(
        root,
        config,
        stage_id,
        "manifest_relative",
    )
    run_manifest = output_dir / "run_manifest.json"
    errors = []
    if not identity.is_file():
        errors.append(f"RESUME_CHECKPOINT_IDENTITY_MISSING:{identity}")
    if manifest.exists():
        errors.append(f"CANDIDATE_MANIFEST_COLLISION:{manifest}")
    if run_manifest.exists():
        errors.append(
            f"COMPUTATION_ALREADY_COMPLETE_USE_PUBLISH_EXISTING:{run_manifest}"
        )
    return errors


def _publication_recovery_errors(
    root: Path,
    config: dict[str, Any],
    stage_id: str,
) -> list[str]:
    """Validate the computation-free Pig-STRENet publication recovery."""

    if stage_id != "pig_strenet_evidence":
        return [f"PUBLICATION_RECOVERY_UNSUPPORTED:{stage_id}"]
    output_dir = resolve_stage_path(
        root,
        config,
        stage_id,
        "output_relative",
    )
    required = (
        "pair_manifest.csv",
        "slot_manifest.csv",
        "history_features.csv",
        "roi_dynamics.csv",
        "roi_visual_selection.csv",
        "social_nodes.csv",
        "social_edges.csv",
        "history_control_matrix.csv",
        "difference_pixel_index.csv",
        "stabilized_difference_summary.csv",
        "stabilized_difference_maps_f32.npy",
        "roi_visual_union_patch_index.csv",
        "roi_visual_union_patches_uint8.npy",
        "roi_visual_union_patch_mask_bool.npy",
    )
    errors = [
        f"PUBLICATION_RECOVERY_OUTPUT_MISSING:{output_dir / name}"
        for name in required
        if not (output_dir / name).is_file()
    ]
    identity = output_dir / ".checkpoints" / "checkpoint_identity.json"
    if not identity.is_file():
        errors.append(
            f"PUBLICATION_RECOVERY_IDENTITY_MISSING:{identity}"
        )
    manifest = resolve_stage_path(
        root,
        config,
        stage_id,
        "manifest_relative",
    )
    if manifest.exists():
        errors.append(f"CANDIDATE_MANIFEST_COLLISION:{manifest}")
    return errors


def _artifact_path(
    root: Path,
    config: dict[str, Any],
    stage_id: str,
) -> Path:
    return Path(_output_specs(root, config, stage_id)[0]["path"])


def _stage_specific_metadata(
    config: dict[str, Any],
    stage_id: str,
) -> dict[str, Any]:
    """Return canonical publication metadata required by the stage contract."""

    metadata: dict[str, Any] = {
        "operational_lineage_id": str(config["lineage_id"]),
        "operational_stage_key": stage_id,
    }
    if stage_id in {"native_evidence", "tensor_export"}:
        metadata["motion_feature_names"] = list(MOTION_FEATURE_NAMES)
    return metadata


def _upstream_manifest_paths(
    root: Path,
    config: dict[str, Any],
    stage_id: str,
) -> list[Path]:
    return [
        resolve_stage_path(root, config, upstream, "manifest_relative")
        for upstream in config["stages"][stage_id]["upstream"]
    ]


def _upstream_errors(
    root: Path,
    config: dict[str, Any],
    stage_id: str,
) -> list[str]:
    contract_root = root / "docs/classification_v2/scientific_contract_v1"
    downstream = str(config["stages"][stage_id]["stage_id"])
    errors = []
    for manifest_path in _upstream_manifest_paths(root, config, stage_id):
        result = validate_upstream_manifest_for_current_authority(
            manifest_path=manifest_path,
            repo_root=root,
            contract_root=contract_root,
            intended_downstream_stage_id=downstream,
        )
        if not result.current_authoritative:
            errors.extend(result.reason_codes)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", required=True, choices=EXPECTED_STAGE_IDS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a supported stage from an exact checkpoint.",
    )
    parser.add_argument(
        "--publish-existing",
        action="store_true",
        help=(
            "Publish already-computed declared outputs after a publication-only "
            "failure; never invokes stage computation."
        ),
    )
    parser.add_argument(
        "--recover-publication",
        action="store_true",
        help=(
            "Recover Pig-STRENet self-publication from completed checkpoint-"
            "bound outputs, then publish the candidate manifest."
        ),
    )
    args = parser.parse_args()
    root, config = load_config(args.config)
    errors = validate_config(root, config)
    if errors:
        print(json.dumps({"status": "FAIL", "errors": errors}))
        return 1
    stage_id = args.stage
    flag = str(config["stages"][stage_id]["authorization_flag"])
    valid, reason, authorization_path = validate_stage_authorization(
        root=root,
        config_path=args.config.resolve(),
        config=config,
        stage_id=stage_id,
    )
    if not valid:
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "reason": f"UNAUTHORIZED:{flag}",
                    "authorization_detail": reason,
                }
            )
        )
        return 2
    authorization_source = "run_local_single_use"
    selected_modes = sum(
        bool(value)
        for value in (
            args.publish_existing,
            args.resume,
            args.recover_publication,
        )
    )
    if selected_modes > 1:
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "reason": "RECOVERY_MODES_ARE_MUTUALLY_EXCLUSIVE",
                }
            )
        )
        return 3
    if args.publish_existing:
        existing_output_errors = _existing_output_errors(
            root,
            config,
            stage_id,
        )
        if existing_output_errors:
            print(
                json.dumps(
                    {
                        "status": "BLOCKED",
                        "reason": "EXISTING_OUTPUT_NOT_PUBLISHABLE",
                        "errors": existing_output_errors,
                    }
                )
            )
            return 3
    elif args.resume:
        resume_errors = _resume_errors(root, config, stage_id)
        if resume_errors:
            print(
                json.dumps(
                    {
                        "status": "BLOCKED",
                        "reason": "CHECKPOINT_RESUME_NOT_VALID",
                        "errors": resume_errors,
                    }
                )
            )
            return 3
    elif args.recover_publication:
        recovery_errors = _publication_recovery_errors(
            root,
            config,
            stage_id,
        )
        if recovery_errors:
            print(
                json.dumps(
                    {
                        "status": "BLOCKED",
                        "reason": "PUBLICATION_RECOVERY_NOT_VALID",
                        "errors": recovery_errors,
                    }
                )
            )
            return 3
    elif _output_collision(root, config, stage_id):
        print(json.dumps({"status": "BLOCKED", "reason": "OUTPUT_COLLISION"}))
        return 3
    upstream_errors = _upstream_errors(root, config, stage_id)
    if upstream_errors:
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "reason": "UPSTREAM_NOT_CURRENT_AUTHORITATIVE",
                    "errors": upstream_errors,
                }
            )
        )
        return 4
    source_report = source_bundle_report(
        root,
        config,
        verification_mode="fast",
        config_path=args.config,
    )
    if not source_report["valid"]:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "reason": "SOURCE_FINGERPRINT_MISMATCH",
                    "verification_mode": source_report.get(
                        "verification_mode"
                    ),
                    "verification_errors": source_report.get(
                        "verification_errors",
                        [],
                    ),
                }
            )
        )
        return 1
    commands = _commands(root, config, stage_id)
    if args.resume:
        commands[0].append("--resume")
    if args.recover_publication:
        commands[0].append("--recover-publication")
    consumed_authorization: Path | None = None
    if not args.dry_run and authorization_path is not None:
        try:
            consumed_authorization = consume_stage_authorization(
                authorization_path
            )
        except (FileNotFoundError, KeyError, OSError, json.JSONDecodeError):
            print(
                json.dumps(
                    {
                        "status": "BLOCKED",
                        "reason": "RUN_LOCAL_AUTHORIZATION_CONSUME_FAILED",
                    }
                )
            )
            return 2
    result = {
        "status": "PLANNED" if args.dry_run else "RUNNING",
        "stage": stage_id,
        "commands": commands,
        "authorization_source": authorization_source,
        "consumed_authorization": (
            None
            if consumed_authorization is None
            else str(consumed_authorization)
        ),
        "automatic_downstream_execution": False,
        "automatic_promotion": False,
        "publication_only": bool(args.publish_existing),
        "publication_recovery": bool(args.recover_publication),
        "checkpoint_resume": bool(args.resume),
    }
    if args.dry_run:
        print(json.dumps(result, indent=2))
        return 0
    if not args.publish_existing:
        for command in commands:
            completed = subprocess.run(command, cwd=root, check=False)
            if completed.returncode != 0:
                result.update(status="FAIL", returncode=completed.returncode)
                print(json.dumps(result))
                return completed.returncode
    output = _artifact_path(root, config, stage_id)
    manifest = resolve_stage_path(root, config, stage_id, "manifest_relative")
    if not output.is_file():
        result.update(status="FAIL", reason="PRIMARY_OUTPUT_MISSING")
        print(json.dumps(result))
        return 5
    specs = _output_specs(root, config, stage_id)
    primary = specs[0]
    build_candidate_artifact_manifest(
        repo_root=root,
        contract_root=root / "docs/classification_v2/scientific_contract_v1",
        stage_id=config["stages"][stage_id]["stage_id"],
        artifact_id=f"{config['lineage_id']}.{stage_id}",
        artifact_class=config["policy"]["candidate_class"],
        output_path=output,
        candidate_manifest_path=manifest,
        upstream_manifest_paths=_upstream_manifest_paths(
            root,
            config,
            stage_id,
        ),
        output_schema_id=str(primary["schema_id"]),
        output_schema_version=str(primary["schema_version"]),
        additional_outputs=[
            {
                **spec,
                "path": spec["path"],
            }
            for spec in specs[1:]
        ],
        stage_specific_metadata=_stage_specific_metadata(config, stage_id),
    )
    result.update(status="PASS", returncode=0, manifest=str(manifest))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
