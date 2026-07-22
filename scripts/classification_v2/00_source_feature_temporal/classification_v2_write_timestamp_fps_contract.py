"""Write the hash-bound per-video timestamp/FPS contract."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.contracts.timestamp_fps import (
    build_timestamp_fps_contract,
    inspect_video_fps_authority,
)


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_output_paths_available(
        [args.output_json],
        overwrite=args.overwrite,
    )
    frame = pd.read_csv(args.frame_local_csv, low_memory=False)
    authority = inspect_video_fps_authority(frame, args.video_root)
    contract = build_timestamp_fps_contract(
        frame,
        lineage_id=args.lineage_id,
        code_authority_sha=args.code_authority_sha,
        source_lineage_artifacts=_named_paths(args.source_lineage_artifact),
        video_fps_authority=authority,
    )
    _atomic_json(args.output_json, contract)
    if contract["errors"]:
        raise SystemExit(2)
    print(json.dumps({"valid": True, "videos": len(authority)}, indent=2))


def _named_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name.strip() or not path.strip():
            raise ValueError("source lineage artifact must use NAME=PATH")
        if name.strip() in result:
            raise ValueError(f"duplicate source lineage artifact: {name}")
        result[name.strip()] = Path(path.strip())
    if not result:
        raise ValueError("at least one --source-lineage-artifact is required")
    return result


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
