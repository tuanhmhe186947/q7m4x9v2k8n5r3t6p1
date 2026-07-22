"""Independently rebuild and verify official behavior-review authority."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from pig_behavior.classification_v2.review.review_authority import (
    OFFICIAL_SCOPE,
    build_review_authority_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--code-authority-sha", required=True)
    parser.add_argument("--lineage-id", required=True)
    parser.add_argument(
        "--source-artifact",
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--component-gate",
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument("--frame-local-csv", required=True, type=Path)
    parser.add_argument("--hidden-reviewed-frame-csv", required=True, type=Path)
    parser.add_argument("--harmonized-frame-csv", required=True, type=Path)
    parser.add_argument(
        "--temporal-native-unit-manifest-csv",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--pig-strenet-evidence-manifest",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--behavior-review-unit-manifest-csv",
        required=True,
        type=Path,
    )
    parser.add_argument("--media-authority-manifest", required=True, type=Path)
    parser.add_argument("--timestamp-fps-contract-json", required=True, type=Path)
    parser.add_argument("--evidence-semantics-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    observed = _read_json(args.manifest_json)
    expected = build_review_authority_manifest(
        code_authority_sha=args.code_authority_sha,
        code_dirty=False,
        lineage_id=args.lineage_id,
        authority_scope=OFFICIAL_SCOPE,
        source_artifacts=_named_paths(args.source_artifact),
        artifacts={
            "frame_local": args.frame_local_csv,
            "hidden_reviewed_frames": args.hidden_reviewed_frame_csv,
            "harmonized_frames": args.harmonized_frame_csv,
            "temporal_native_units": args.temporal_native_unit_manifest_csv,
            "pig_strenet_evidence": args.pig_strenet_evidence_manifest,
            "behavior_review_units": args.behavior_review_unit_manifest_csv,
            "media_authority": args.media_authority_manifest,
            "timestamp_fps_contract": args.timestamp_fps_contract_json,
            "evidence_semantics": args.evidence_semantics_json,
        },
        timestamp_fps_contract=_read_json(args.timestamp_fps_contract_json),
        evidence_semantics=_read_json(args.evidence_semantics_json),
        component_gates=_named_paths(args.component_gate),
        actual_head_sha=_git_output("rev-parse", "HEAD"),
        tracked_code_clean=not bool(
            _git_output("status", "--porcelain", "--untracked-files=no")
        ),
        require_full_component_gates=True,
    )
    errors = list(expected["errors"])
    if observed != expected:
        errors.append("official_review_authority_content_or_hash_drift")
    if observed.get("authorizes_behavior_gui") is not True:
        errors.append("manifest_does_not_authorize_behavior_gui")
    if observed.get("authorizes_final_view_build") is not False:
        errors.append("pre_review_manifest_authorizes_final_view_build")
    if observed.get("authorizes_training") is not False:
        errors.append("pre_review_manifest_authorizes_training")
    audit = {
        "lineage_id": args.lineage_id,
        "valid": not errors,
        "errors": errors,
        "authorizes_behavior_gui": not errors,
        "manifest_exact_match": observed == expected,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if errors:
        raise SystemExit(2)
    print(json.dumps(audit, indent=2))


def _named_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError("named path must use NAME=PATH")
        result[name.strip()] = Path(raw_path.strip())
    return result


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


if __name__ == "__main__":
    main()
