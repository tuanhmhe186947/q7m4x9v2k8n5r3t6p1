"""Independently check a production FRAME_LOCAL_PRIMITIVES artifact."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

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

CVAT_STRUCTURAL_SOURCES = {"cvat_tracking_xml", "cvat_selected_native"}
LEGACY_STRUCTURAL_SOURCE = "legacy_recovered"
DEFAULT_CONTRACT_PATH = Path(
    "docs/classification_v2/scientific_contract_v1/"
    "00_pipeline_contract.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", required=True, type=Path)
    parser.add_argument("--frame-local-csv", required=True, type=Path)
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT_PATH,
    )
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
    object_track_key_contract = _load_object_track_key_contract(
        args.contract
    )
    reference_keys = _reference_object_track_keys(
        source.reset_index(drop=True),
        object_track_key_contract,
    )
    key_check = _audit_object_track_keys(output, reference_keys)
    audit["object_track_key_check"] = key_check
    if key_check["mismatches"]:
        audit["errors"].append(
            "object_track_key_mismatch_rows="
            f"{key_check['mismatches']}"
        )
    expected = _independent_rebuild(
        source,
        roi_coco=args.roi_coco,
        pen_mask=args.pen_mask,
        expected_pen_mask_sha256=args.expected_pen_mask_sha256,
        object_track_key_contract=object_track_key_contract,
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


def _load_object_track_key_contract(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    contract = payload.get("object_track_key_contract")
    if not isinstance(contract, dict):
        raise ValueError(
            "missing object_track_key_contract in primary contract"
        )
    required = {
        "schema_id",
        "schema_version",
        "identity_scope_components",
        "existing_key_field",
        "identity_fallback_order",
        "identity_discriminators",
        "component_order",
        "component_names",
        "component_delimiter",
        "name_value_delimiter",
        "serialization_templates",
        "escaping_policy",
        "blank_policy",
        "pig_id_authoritative",
        "row_order_authoritative",
    }
    missing = sorted(required.difference(contract))
    if missing:
        raise ValueError(
            f"object_track_key_contract missing fields: {missing}"
        )
    if contract["pig_id_authoritative"] is not False:
        raise ValueError("object_track_key_contract authorizes pig_id")
    if contract["row_order_authoritative"] is not False:
        raise ValueError("object_track_key_contract authorizes row order")
    return contract


def _independent_rebuild(
    source: pd.DataFrame,
    *,
    roi_coco: Path,
    pen_mask: Path,
    expected_pen_mask_sha256: str | None,
    object_track_key_contract: dict[str, Any],
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
    reference_keys = _reference_object_track_keys(
        out,
        object_track_key_contract,
    )
    invalid = reference_keys["reason_code"].ne("OK")
    if invalid.any():
        details = reference_keys.loc[
            invalid,
            [
                "row_authority_key",
                "selected_identity_type",
                "selected_identity_value",
                "reason_code",
            ],
        ].to_dict(orient="records")
        raise ValueError(
            "object_track_key independent reconstruction failed: "
            f"{details[:10]}"
        )
    out["object_track_key"] = reference_keys[
        "expected_canonical_key"
    ].to_numpy()
    frame_index = pd.to_numeric(out["frame_index"], errors="coerce")
    source_type = out["source_type"].fillna("").astype(str)
    cvat = source_type.isin(CVAT_STRUCTURAL_SOURCES)
    legacy = source_type.eq(LEGACY_STRUCTURAL_SOURCE)
    other = ~(cvat | legacy)
    out["temporal_unit_key"] = ""
    cvat_anchor = np.floor(frame_index / 6).astype("Int64") * 6
    out.loc[cvat, "temporal_unit_key"] = (
        out.loc[cvat, "object_track_key"].astype(str)
        + "|anchor="
        + cvat_anchor.loc[cvat].astype(str)
    )
    out.loc[legacy, "temporal_unit_key"] = (
        out.loc[legacy, "object_track_key"].astype(str)
        + "|legacy_sequence"
    )
    out.loc[other, "temporal_unit_key"] = (
        out.loc[other, "object_track_key"].astype(str)
        + "|frame="
        + frame_index.loc[other].round().astype("Int64").astype(str)
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


def _reference_object_track_keys(
    rows: pd.DataFrame,
    contract: dict[str, Any],
) -> pd.DataFrame:
    scope_fields = list(contract["identity_scope_components"])
    component_names = dict(contract["component_names"])
    fallback_order = list(contract["identity_fallback_order"])
    discriminators = dict(contract["identity_discriminators"])
    templates = dict(contract["serialization_templates"])
    safe = str(contract["escaping_policy"]["safe_characters"])
    existing_field = str(contract["existing_key_field"])

    cleaned = {
        field: _clean_reference_series(rows, field)
        for field in {
            *scope_fields,
            *fallback_order,
            existing_field,
            "frame_uid",
        }
    }
    records: list[dict[str, str]] = []
    for position, index in enumerate(rows.index):
        scope = {field: cleaned[field].loc[index] for field in scope_fields}
        existing = cleaned[existing_field].loc[index]
        selected_field = ""
        selected_value = ""
        for field in fallback_order:
            value = cleaned[field].loc[index]
            if value:
                selected_field = field
                selected_value = value
                break
        reason_code = "OK"
        expected = ""
        missing_scope = [
            field for field in scope_fields if not scope[field]
        ]
        if missing_scope:
            reason_code = "MISSING_SCOPE_AUTHORITY"
        elif selected_field:
            discriminator = str(discriminators[selected_field])
            template = str(templates[discriminator])
            escaped_scope = {
                component_names[field]: quote(
                    scope[field],
                    safe=safe,
                )
                for field in scope_fields
            }
            expected = template.format(
                **escaped_scope,
                value=quote(selected_value, safe=safe),
            )
            if existing and existing != expected:
                reason_code = "EXISTING_KEY_MISMATCH"
        elif existing:
            selected_field = existing_field
            selected_value = existing
            expected = existing
        else:
            reason_code = "MISSING_IDENTITY_AUTHORITY"
        frame_uid = cleaned["frame_uid"].loc[index]
        row_authority_key = frame_uid or f"source_row_ordinal={position}"
        records.append(
            {
                "row_authority_key": row_authority_key,
                "expected_canonical_key": expected,
                "selected_identity_type": selected_field,
                "selected_identity_value": selected_value,
                "source": scope.get("source_type", ""),
                "dataset": scope.get("dataset_id", ""),
                "video": scope.get("video_key", ""),
                "reason_code": reason_code,
            }
        )
    return pd.DataFrame.from_records(records, index=rows.index)


def _audit_object_track_keys(
    output: pd.DataFrame,
    reference: pd.DataFrame,
) -> dict[str, Any]:
    actual = _clean_reference_series(
        output.reset_index(drop=True),
        "object_track_key",
    )
    expected = reference.reset_index(drop=True)
    details: list[dict[str, str]] = []
    row_count = max(len(actual), len(expected))
    for position in range(row_count):
        if position >= len(expected):
            details.append(
                {
                    "row_authority_key": f"output_row={position}",
                    "expected_canonical_key": "",
                    "actual_object_track_key": actual.iloc[position],
                    "selected_identity_type": "",
                    "selected_identity_value": "",
                    "source": "",
                    "dataset": "",
                    "video": "",
                    "reason_code": "UNEXPECTED_OUTPUT_ROW",
                }
            )
            continue
        row = expected.iloc[position]
        actual_key = actual.iloc[position] if position < len(actual) else ""
        reason_code = str(row["reason_code"])
        if reason_code == "OK" and actual_key == row["expected_canonical_key"]:
            continue
        if reason_code == "OK":
            reason_code = (
                "ACTUAL_KEY_BLANK"
                if not actual_key
                else "CANONICAL_SERIALIZATION_MISMATCH"
            )
        details.append(
            {
                "row_authority_key": str(row["row_authority_key"]),
                "expected_canonical_key": str(
                    row["expected_canonical_key"]
                ),
                "actual_object_track_key": actual_key,
                "selected_identity_type": str(
                    row["selected_identity_type"]
                ),
                "selected_identity_value": str(
                    row["selected_identity_value"]
                ),
                "source": str(row["source"]),
                "dataset": str(row["dataset"]),
                "video": str(row["video"]),
                "reason_code": reason_code,
            }
        )
    return {
        "schema_id": "schema.classification_v2.object_track_key",
        "schema_version": "classification_v2.object_track_key.v1",
        "rows_checked": row_count,
        "matches": row_count - len(details),
        "mismatches": len(details),
        "details": details,
    }


def _clean_reference_series(
    rows: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in rows.columns:
        return pd.Series("", index=rows.index, dtype=object)
    values = rows[column].fillna("").astype(str).str.strip()
    return values.mask(values.isin({"nan", "None", "<NA>"}), "")


if __name__ == "__main__":
    main()
