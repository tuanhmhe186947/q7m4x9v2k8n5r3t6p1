"""Build a hash-bound behavior-review authority manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.review.review_authority import (
    OFFICIAL_SCOPE,
    SMOKE_SCOPE,
    build_review_authority_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bind review-critical Group A artifacts before behavior review. "
            "Final temporal views are intentionally excluded."
        )
    )
    parser.add_argument("--code-authority-sha", required=True)
    parser.add_argument("--code-dirty", action="store_true")
    parser.add_argument("--lineage-id", required=True)
    parser.add_argument(
        "--authority-scope",
        choices=[OFFICIAL_SCOPE, SMOKE_SCOPE],
        required=True,
    )
    parser.add_argument(
        "--source-artifact",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Repeat for every immutable source artifact.",
    )
    parser.add_argument("--frame-local-csv", required=True, type=Path)
    parser.add_argument(
        "--hidden-reviewed-frame-csv",
        required=True,
        type=Path,
    )
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
    parser.add_argument(
        "--media-authority-manifest",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--timestamp-fps-contract-json",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--evidence-semantics-json",
        required=True,
        type=Path,
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_output_paths_available(
        [args.output_json],
        overwrite=args.overwrite,
    )
    source_artifacts = _named_paths(args.source_artifact)
    timestamp_contract = _read_json(args.timestamp_fps_contract_json)
    evidence_semantics = _read_json(args.evidence_semantics_json)
    artifacts = {
        "frame_local": args.frame_local_csv,
        "hidden_reviewed_frames": args.hidden_reviewed_frame_csv,
        "harmonized_frames": args.harmonized_frame_csv,
        "temporal_native_units": (
            args.temporal_native_unit_manifest_csv
        ),
        "pig_strenet_evidence": args.pig_strenet_evidence_manifest,
        "behavior_review_units": args.behavior_review_unit_manifest_csv,
        "media_authority": args.media_authority_manifest,
    }
    manifest = build_review_authority_manifest(
        code_authority_sha=args.code_authority_sha,
        code_dirty=args.code_dirty,
        lineage_id=args.lineage_id,
        authority_scope=args.authority_scope,
        source_artifacts=source_artifacts,
        artifacts=artifacts,
        timestamp_fps_contract=timestamp_contract,
        evidence_semantics=evidence_semantics,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "valid": manifest["valid"],
                "official_review_authority": manifest[
                    "official_review_authority"
                ],
                "authorizes_behavior_gui": manifest[
                    "authorizes_behavior_gui"
                ],
                "review_authority_sha256": manifest[
                    "review_authority_sha256"
                ],
                "errors": manifest["errors"],
            },
            indent=2,
        )
    )
    if not manifest["valid"]:
        raise SystemExit(2)


def _named_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError(
                "--source-artifact must use nonblank NAME=PATH"
            )
        name = name.strip()
        if name in result:
            raise ValueError(f"duplicate source artifact name: {name}")
        result[name] = Path(raw_path.strip())
    return result


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


if __name__ == "__main__":
    main()
