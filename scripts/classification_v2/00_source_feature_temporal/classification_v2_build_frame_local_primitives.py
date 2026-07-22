"""Build the production FRAME_LOCAL_PRIMITIVES artifact atomically."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.features.frame_local import (
    ACTIVE_SOURCE_FPS,
    audit_frame_local_primitives,
    build_frame_local_primitives,
    frame_local_schema_payload,
)
from pig_behavior.classification_v2.features.pen_context import (
    DEFAULT_PEN_MASK_SHA256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--roi-coco", required=True, type=Path)
    parser.add_argument("--pen-mask", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--schema-json", required=True, type=Path)
    parser.add_argument("--audit-json", required=True, type=Path)
    parser.add_argument("--lineage-id", required=True)
    parser.add_argument("--code-authority-sha", required=True)
    parser.add_argument("--source-fps", type=float, default=ACTIVE_SOURCE_FPS)
    parser.add_argument(
        "--expected-pen-mask-sha256",
        default=DEFAULT_PEN_MASK_SHA256,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = [args.output_csv, args.schema_json, args.audit_json]
    require_output_paths_available(outputs, overwrite=args.overwrite)
    source = pd.read_csv(args.input_csv, low_memory=False)
    output = build_frame_local_primitives(
        source,
        roi_coco_path=args.roi_coco,
        pen_mask_path=args.pen_mask,
        source_fps=args.source_fps,
        expected_pen_mask_sha256=args.expected_pen_mask_sha256,
    )
    audit = audit_frame_local_primitives(source, output)
    audit["lineage_id"] = args.lineage_id
    audit["code_authority_sha"] = args.code_authority_sha.lower()
    if audit["errors"]:
        raise ValueError(f"frame-local self-audit failed: {audit['errors']}")
    schema = frame_local_schema_payload(output)
    schema["lineage_id"] = args.lineage_id
    schema["code_authority_sha"] = args.code_authority_sha.lower()
    _atomic_bundle(output, schema, audit, outputs)
    print(json.dumps({"valid": True, "rows": len(output)}, indent=2))


def _atomic_bundle(
    frame: pd.DataFrame,
    schema: dict[str, object],
    audit: dict[str, object],
    paths: list[Path],
) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary = [path.with_name(f".{path.name}.{token}.tmp") for path in paths]
    try:
        frame.to_csv(temporary[0], index=False)
        for path, payload in zip(temporary[1:], (schema, audit), strict=True):
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        for source, target in zip(temporary, paths, strict=True):
            source.replace(target)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
