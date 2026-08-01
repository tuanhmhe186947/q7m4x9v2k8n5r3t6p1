"""Bind frozen review, identity corrections, ROI, and rebuild artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.review.post_review_learning import (
    assert_not_active_behavior_ledger_path,
    bindings_from_paths,
    build_final_review_integration_preflight,
    sha256_file,
    write_json,
)


def _binding(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("binding must be NAME=PATH")
    return name.strip(), Path(raw_path.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-close-authority-json", type=Path, required=True)
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        type=_binding,
        help="Repeat NAME=PATH for every required artifact binding.",
    )
    parser.add_argument(
        "--identity-apply-manifest-json",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument("--conflict-resolutions-json", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_paths = dict(args.artifact)
    if len(artifact_paths) != len(args.artifact):
        raise ValueError("duplicate_artifact_binding_name")
    for path in (
        args.review_close_authority_json,
        *artifact_paths.values(),
        *args.identity_apply_manifest_json,
        args.output_json,
    ):
        assert_not_active_behavior_ledger_path(path)
    if args.conflict_resolutions_json is not None:
        assert_not_active_behavior_ledger_path(args.conflict_resolutions_json)

    close = json.loads(
        args.review_close_authority_json.read_text(encoding="utf-8")
    )
    manifests = []
    for path in args.identity_apply_manifest_json:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["manifest_sha256"] = sha256_file(path)
        payload["manifest_path"] = str(path.resolve())
        manifests.append(payload)
    resolutions = []
    if args.conflict_resolutions_json is not None:
        resolutions = json.loads(
            args.conflict_resolutions_json.read_text(encoding="utf-8")
        )
        if not isinstance(resolutions, list):
            raise ValueError("conflict resolutions must be a JSON list")

    corrected_source_path = artifact_paths.get("corrected_source_authority")
    corrected_source = None
    if corrected_source_path is not None:
        corrected_source = json.loads(
            corrected_source_path.read_text(encoding="utf-8")
        )

    preflight = build_final_review_integration_preflight(
        review_close_authority=close,
        artifact_bindings=bindings_from_paths(artifact_paths),
        identity_apply_manifests=manifests,
        corrected_source_authority=corrected_source,
        conflict_resolutions=resolutions,
    )
    write_json(args.output_json, preflight)
    print(f"{preflight['status']}: final review integration preflight")
    print(args.output_json)
    if preflight["status"] != "READY_FOR_REVIEWED_WINDOW_REBUILD":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
