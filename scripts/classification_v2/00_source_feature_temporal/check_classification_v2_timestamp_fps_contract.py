"""Independently reproduce and check the timestamp/FPS contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.contracts.timestamp_fps import (
    build_timestamp_fps_contract,
    inspect_video_fps_authority,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-local-csv", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--lineage-id", required=True)
    parser.add_argument("--code-authority-sha", required=True)
    parser.add_argument(
        "--source-lineage-artifact",
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument("--contract-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.frame_local_csv, low_memory=False)
    expected = build_timestamp_fps_contract(
        frame,
        lineage_id=args.lineage_id,
        code_authority_sha=args.code_authority_sha,
        source_lineage_artifacts=_named_paths(args.source_lineage_artifact),
        video_fps_authority=inspect_video_fps_authority(
            frame,
            args.video_root,
        ),
    )
    observed = json.loads(args.contract_json.read_text(encoding="utf-8"))
    errors = list(expected["errors"])
    if observed != expected:
        errors.append("timestamp_fps_contract_content_drift")
    audit = {
        "lineage_id": args.lineage_id,
        "code_authority_sha": args.code_authority_sha.lower(),
        "valid": not errors,
        "errors": errors,
        "contract_matches_independent_rebuild": observed == expected,
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
            raise ValueError("source lineage artifact must use NAME=PATH")
        result[name.strip()] = Path(raw_path.strip())
    if not result:
        raise ValueError("at least one --source-lineage-artifact is required")
    return result


if __name__ == "__main__":
    main()
