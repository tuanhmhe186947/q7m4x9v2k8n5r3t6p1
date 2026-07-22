"""Independently check a production FRAME_LOCAL_PRIMITIVES artifact."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.features.context_policy import (
    normalize_hidden_provenance,
)
from pig_behavior.classification_v2.features.frame_local import (
    ACTIVE_SOURCE_FPS,
    FRAME_LOCAL_GRAIN,
    audit_frame_local_primitives,
    frame_local_schema_payload,
)
from pig_behavior.classification_v2.features.geometry import build_geometry_features
from pig_behavior.classification_v2.features.pen_context import (
    DEFAULT_PEN_MASK_SHA256,
    build_static_pen_context_features,
)
from pig_behavior.classification_v2.features.roi import build_roi_features
from pig_behavior.classification_v2.features.social import (
    build_static_social_context_features,
)
from pig_behavior.classification_v2.sources.temporal_provenance import (
    apply_source_frame_clock,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", required=True, type=Path)
    parser.add_argument("--frame-local-csv", required=True, type=Path)
    parser.add_argument("--roi-coco", required=True, type=Path)
    parser.add_argument("--pen-mask", required=True, type=Path)
    parser.add_argument(
        "--expected-pen-mask-sha256",
        default=DEFAULT_PEN_MASK_SHA256,
    )
    parser.add_argument("--schema-json", required=True, type=Path)
    parser.add_argument("--builder-audit-json", required=True, type=Path)
    parser.add_argument("--lineage-id", required=True)
    parser.add_argument("--code-authority-sha", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    source = pd.read_csv(args.source_csv, low_memory=False)
    output = pd.read_csv(args.frame_local_csv, low_memory=False)
    audit = audit_frame_local_primitives(source, output)
    expected = _independent_rebuild(
        source,
        roi_coco=args.roi_coco,
        pen_mask=args.pen_mask,
        expected_pen_mask_sha256=args.expected_pen_mask_sha256,
    )
    expected = pd.read_csv(
        io.StringIO(expected.to_csv(index=False)),
        low_memory=False,
    )
    try:
        pd.testing.assert_frame_equal(
            output,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=1e-9,
            atol=1e-12,
        )
    except AssertionError as exc:
        audit["errors"].append(f"frame_local_content_drift={exc}")
    declared_schema = _read_json(args.schema_json)
    builder_audit = _read_json(args.builder_audit_json)
    observed_schema = frame_local_schema_payload(output)
    observed_schema["lineage_id"] = args.lineage_id
    observed_schema["code_authority_sha"] = args.code_authority_sha.lower()
    if declared_schema != observed_schema:
        audit["errors"].append("frame_local_schema_json_drift")
    if builder_audit.get("errors") != [] or not builder_audit.get("valid"):
        audit["errors"].append("frame_local_builder_audit_not_pass")
    if builder_audit.get("lineage_id") != args.lineage_id:
        audit["errors"].append("frame_local_builder_lineage_mismatch")
    if builder_audit.get("code_authority_sha") != args.code_authority_sha.lower():
        audit["errors"].append("frame_local_builder_code_authority_mismatch")
    audit["lineage_id"] = args.lineage_id
    audit["code_authority_sha"] = args.code_authority_sha.lower()
    audit["valid"] = not audit["errors"]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if audit["errors"]:
        raise SystemExit(2)
    print(json.dumps({"valid": True, "rows": len(output)}, indent=2))


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _independent_rebuild(
    source: pd.DataFrame,
    *,
    roi_coco: Path,
    pen_mask: Path,
    expected_pen_mask_sha256: str | None,
) -> pd.DataFrame:
    out = source.copy().reset_index(drop=True)
    out["source_row_ordinal"] = np.arange(len(out), dtype="int64")
    if {"source_type", "hidden"}.issubset(out.columns):
        out = normalize_hidden_provenance(out)
    out = apply_source_frame_clock(
        out,
        source_fps=ACTIVE_SOURCE_FPS,
        preserve_input_as_acquisition=True,
    )
    out = build_geometry_features(out)
    out = build_roi_features(out, roi_coco_path=roi_coco)
    out["object_track_key"] = (
        out["source_type"].astype(str)
        + "|"
        + out["dataset_id"].astype(str)
        + "|"
        + out["video_key"].astype(str)
        + "|track="
        + out["track_id"].fillna("").astype(str)
        + "|pig="
        + out["pig_id"].fillna("").astype(str)
    )
    out = build_static_social_context_features(out)
    out = build_static_pen_context_features(
        out,
        mask_path=pen_mask,
        expected_mask_sha256=expected_pen_mask_sha256,
    )
    out["feature_computation_grain"] = FRAME_LOCAL_GRAIN
    out["pair_scope_key"] = ""
    return out


if __name__ == "__main__":
    main()
