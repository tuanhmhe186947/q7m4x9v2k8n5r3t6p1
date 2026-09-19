"""Freeze, derive, prepare and audit the post-review rebuild lineage."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from pig_behavior.classification_v2.lineage_config import load_config
from pig_behavior.classification_v2.review.post_review_learning import (
    assert_not_active_behavior_ledger_path,
    bindings_from_paths,
    sha256_file,
    write_json,
)
from pig_behavior.classification_v2.review.reviewed_rebuild import (
    audit_reviewed_label_overlay,
    build_final_review_autocarry,
    build_reviewed_application_views,
    derive_reviewed_lineage_config,
    freeze_reviewed_training_application_authority,
    stable_payload_hash,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze-authority")
    _add_review_paths(freeze)
    freeze.add_argument("--review-close-authority-json", type=Path, required=True)
    freeze.add_argument("--corrected-source-authority-json", type=Path, required=True)
    freeze.add_argument("--fixed-point-audit-json", type=Path, required=True)
    freeze.add_argument("--adjusted-roi-json", type=Path, required=True)
    freeze.add_argument("--output-json", type=Path, required=True)

    derive = subparsers.add_parser("derive-config")
    derive.add_argument("--base-config", type=Path, required=True)
    derive.add_argument("--lineage-id", required=True)
    derive.add_argument("--run-root", type=Path, required=True)
    derive.add_argument("--adjusted-roi-json", type=Path, required=True)
    derive.add_argument("--scientific-accepted-sha")
    derive.add_argument("--output-yaml", type=Path, required=True)
    derive.add_argument("--output-manifest-json", type=Path, required=True)

    prepare = subparsers.add_parser("prepare-overlay")
    prepare.add_argument("--application-authority-json", type=Path, required=True)
    prepare.add_argument("--frame-features-csv", type=Path, required=True)
    prepare.add_argument("--composite-scope-csv", type=Path, required=True)
    prepare.add_argument("--composite-decisions-csv", type=Path, required=True)
    prepare.add_argument("--composite-quality-csv", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)

    audit = subparsers.add_parser("audit-overlay")
    audit.add_argument("--application-authority-json", type=Path, required=True)
    audit.add_argument("--before-frame-features-csv", type=Path, required=True)
    audit.add_argument("--after-frame-features-csv", type=Path, required=True)
    audit.add_argument("--apply-audit-json", type=Path, required=True)
    audit.add_argument("--composite-scope-csv", type=Path, required=True)
    audit.add_argument("--composite-quality-csv", type=Path, required=True)
    audit.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _add_review_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--primary-scope-csv", type=Path, required=True)
    parser.add_argument("--primary-decisions-csv", type=Path, required=True)
    parser.add_argument("--primary-quality-csv", type=Path, required=True)
    parser.add_argument("--control-scope-csv", type=Path, required=True)
    parser.add_argument("--control-decisions-csv", type=Path, required=True)
    parser.add_argument("--control-quality-csv", type=Path, required=True)
    parser.add_argument("--composite-scope-csv", type=Path, required=True)
    parser.add_argument("--composite-decisions-csv", type=Path, required=True)
    parser.add_argument("--composite-quality-csv", type=Path, required=True)


def main() -> None:
    args = parse_args()
    for value in vars(args).values():
        if isinstance(value, Path):
            assert_not_active_behavior_ledger_path(value)
    if args.command == "freeze-authority":
        _freeze(args)
    elif args.command == "derive-config":
        _derive(args)
    elif args.command == "prepare-overlay":
        _prepare(args)
    elif args.command == "audit-overlay":
        _audit(args)
    else:  # pragma: no cover
        raise ValueError(f"unsupported_command={args.command}")


def _freeze(args: argparse.Namespace) -> None:
    paths = {
        "primary_scope": args.primary_scope_csv,
        "primary_decisions": args.primary_decisions_csv,
        "primary_quality": args.primary_quality_csv,
        "control_scope": args.control_scope_csv,
        "control_decisions": args.control_decisions_csv,
        "control_quality": args.control_quality_csv,
        "composite_scope": args.composite_scope_csv,
        "composite_decisions": args.composite_decisions_csv,
        "composite_quality": args.composite_quality_csv,
        "review_close_authority": args.review_close_authority_json,
        "corrected_source_authority": args.corrected_source_authority_json,
        "fixed_point_audit": args.fixed_point_audit_json,
        "adjusted_roi": args.adjusted_roi_json,
    }
    bindings = bindings_from_paths(paths)
    review_close = _read_json(args.review_close_authority_json)
    corrected_source = _read_json(args.corrected_source_authority_json)
    fixed_point = _read_json(args.fixed_point_audit_json)
    payload = freeze_reviewed_training_application_authority(
        review_close_authority=review_close,
        primary_scope=pd.read_csv(args.primary_scope_csv, low_memory=False),
        primary_decisions=pd.read_csv(args.primary_decisions_csv, low_memory=False),
        primary_quality=pd.read_csv(args.primary_quality_csv, low_memory=False),
        control_scope=pd.read_csv(args.control_scope_csv, low_memory=False),
        control_decisions=pd.read_csv(args.control_decisions_csv, low_memory=False),
        control_quality=pd.read_csv(args.control_quality_csv, low_memory=False),
        composite_scope=pd.read_csv(args.composite_scope_csv, low_memory=False),
        composite_decisions=pd.read_csv(args.composite_decisions_csv, low_memory=False),
        composite_quality=pd.read_csv(args.composite_quality_csv, low_memory=False),
        corrected_source_authority=corrected_source,
        fixed_point_audit=fixed_point,
        artifact_bindings=bindings,
    )
    payload["code_sha"] = _git_head()
    payload["authority_payload_hash"] = stable_payload_hash(payload)
    write_json(args.output_json, payload)
    print("PASS: reviewed training application authority frozen")
    print(args.output_json)


def _derive(args: argparse.Namespace) -> None:
    root, base = load_config(args.base_config)
    code_sha = args.scientific_accepted_sha or _git_head()
    config, manifest = derive_reviewed_lineage_config(
        repository_root=root,
        base_config=base,
        base_config_path=args.base_config,
        lineage_id=args.lineage_id,
        run_root=args.run_root,
        scientific_accepted_sha=code_sha,
        adjusted_roi_path=args.adjusted_roi_json,
    )
    args.output_yaml.parent.mkdir(parents=True, exist_ok=False)
    args.output_yaml.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    manifest["derived_config"] = {
        "path": str(args.output_yaml.resolve()),
        "sha256": sha256_file(args.output_yaml),
    }
    write_json(args.output_manifest_json, manifest)
    print("PASS: reviewed lineage config derived")
    print(args.output_yaml)


def _prepare(args: argparse.Namespace) -> None:
    authority = _read_json(args.application_authority_json)
    if authority.get("status") != "FROZEN":
        raise ValueError("application_authority_not_frozen")
    _validate_application_binding(authority, "composite_scope", args.composite_scope_csv)
    _validate_application_binding(
        authority,
        "composite_decisions",
        args.composite_decisions_csv,
    )
    _validate_application_binding(
        authority,
        "composite_quality",
        args.composite_quality_csv,
    )
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    frames = pd.read_csv(args.frame_features_csv, low_memory=False)
    scope = pd.read_csv(args.composite_scope_csv, low_memory=False)
    decisions = pd.read_csv(args.composite_decisions_csv, low_memory=False)
    quality = pd.read_csv(args.composite_quality_csv, low_memory=False)
    application_scope, application_decisions, view_audit = (
        build_reviewed_application_views(
            frame_features=frames,
            composite_scope=scope,
            composite_decisions=decisions,
            composite_quality=quality,
        )
    )
    autocarry, carry_audit = build_final_review_autocarry(
        frames,
        application_scope,
    )
    args.output_dir.mkdir(parents=True)
    outputs = {
        "application_scope": args.output_dir / "application_review_scope.csv",
        "application_decisions": (
            args.output_dir / "application_behavior_decisions.csv"
        ),
        "autocarry": args.output_dir / "final_review_autocarry_manifest.csv",
        "pre_apply_audit": args.output_dir / "reviewed_overlay_pre_apply_audit.json",
    }
    application_scope.to_csv(outputs["application_scope"], index=False)
    application_decisions.to_csv(outputs["application_decisions"], index=False)
    autocarry.to_csv(outputs["autocarry"], index=False)
    write_json(
        outputs["pre_apply_audit"],
        {
            "status": "READY_TO_APPLY",
            "view_audit": view_audit,
            "autocarry_audit": carry_audit,
            "inputs": {
                "frame_features": {
                    "path": str(args.frame_features_csv.resolve()),
                    "sha256": sha256_file(args.frame_features_csv),
                },
                "application_authority": {
                    "path": str(args.application_authority_json.resolve()),
                    "sha256": sha256_file(args.application_authority_json),
                },
            },
        },
    )
    _write_inventory(args.output_dir, outputs)
    print("PASS: reviewed overlay inputs prepared")
    print(args.output_dir)


def _audit(args: argparse.Namespace) -> None:
    authority = _read_json(args.application_authority_json)
    _validate_application_binding(authority, "composite_scope", args.composite_scope_csv)
    _validate_application_binding(
        authority,
        "composite_quality",
        args.composite_quality_csv,
    )
    payload = audit_reviewed_label_overlay(
        before_frames=pd.read_csv(args.before_frame_features_csv, low_memory=False),
        after_frames=pd.read_csv(args.after_frame_features_csv, low_memory=False),
        composite_scope=pd.read_csv(args.composite_scope_csv, low_memory=False),
        composite_quality=pd.read_csv(args.composite_quality_csv, low_memory=False),
        apply_audit=_read_json(args.apply_audit_json),
    )
    payload["inputs"] = bindings_from_paths(
        {
            "application_authority": args.application_authority_json,
            "before_frames": args.before_frame_features_csv,
            "after_frames": args.after_frame_features_csv,
            "apply_audit": args.apply_audit_json,
            "composite_scope": args.composite_scope_csv,
            "composite_quality": args.composite_quality_csv,
        }
    )
    write_json(args.output_json, payload)
    print("PASS: reviewed label overlay audited")
    print(args.output_json)


def _validate_application_binding(
    authority: dict[str, object],
    name: str,
    path: Path,
) -> None:
    artifacts = authority.get("artifacts")
    if not isinstance(artifacts, dict) or name not in artifacts:
        raise ValueError(f"application_binding_missing={name}")
    binding = artifacts[name]
    if not isinstance(binding, dict) or sha256_file(path) != binding.get("sha256"):
        raise ValueError(f"application_binding_drift={name}")


def _write_inventory(output_dir: Path, outputs: dict[str, Path]) -> None:
    write_json(
        output_dir / "artifact_inventory.json",
        {
            "artifacts": [
                {
                    "name": name,
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for name, path in sorted(outputs.items())
            ]
        },
    )


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_mapping_required={path}")
    return payload


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    main()
