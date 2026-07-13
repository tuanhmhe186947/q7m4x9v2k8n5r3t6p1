"""End-to-end identifier and positional lineage audit for bounded rebuilds."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.contracts.identifiers import (
    FRAME_OBJECT_IDENTIFIER_VERSION,
    audit_frame_object_identifiers,
)
from pig_behavior.classification_v2.contracts.window_alignment import (
    audit_ordered_window_ids,
    ordered_window_id_sha256,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

EXPECTED_FRAME_STAGES = (
    "context",
    "geometry",
    "roi",
    "enhanced",
    "harmonized",
)
EXPECTED_SOURCES = {"cvat_tracking_xml", "legacy_recovered"}
FRAME_LOCATOR_COLUMNS = (
    "identifier_schema_version",
    "scene_frame_uid",
    "source_type",
    "dataset_id",
    "video_key",
    "clip_id",
    "task_id",
    "pig_id",
    "track_id",
    "frame_index",
)
MODEL_IDENTIFIER_COLUMNS = {
    "dataset_id",
    "video_key",
    "source_video_key",
    "source_type",
    "clip_id",
    "task_id",
    "identifier_schema_version",
    "scene_frame_uid",
    "frame_uid",
    "image_key",
    "object_id_in_image",
    "pig_id",
    "track_id",
    "object_track_key",
    "temporal_unit_key",
    "review_unit_id",
    "review_item_id",
    "window_id",
    "fold_id",
    "oof_fold_id",
}


def audit_source_to_window_lineage(
    *,
    frame_stages: dict[str, pd.DataFrame],
    sequence_manifest: pd.DataFrame,
    sequence_features: pd.DataFrame,
    image_frame_manifest: pd.DataFrame,
    image_window_manifest: pd.DataFrame,
    x_columns: list[str],
    artifact_audits: dict[str, dict[str, Any]],
    artifact_row_counts: dict[str, int],
    spatial_array_rows: dict[str, int],
    preload_errors: list[str] | None = None,
) -> dict[str, Any]:
    """Prove frame IDs and ordered window IDs survive every positional export."""

    errors = list(preload_errors or [])
    warnings: list[str] = []
    frame_lineage = _audit_frame_lineage(frame_stages)
    errors.extend(frame_lineage["errors"])

    image_frame_mapping = _audit_image_frame_mapping(
        frame_stages.get("harmonized", pd.DataFrame()),
        image_frame_manifest,
    )
    errors.extend(image_frame_mapping["errors"])

    window_lineage = _audit_window_lineage(
        sequence_manifest,
        sequence_features,
        image_window_manifest,
    )
    errors.extend(window_lineage["errors"])
    expected_window_hash = window_lineage.get("ordered_window_id_sha256")

    audit_hashes = _audit_exported_window_hashes(
        artifact_audits,
        expected_window_hash,
    )
    errors.extend(audit_hashes["errors"])

    row_lineage = _audit_row_counts(
        len(sequence_manifest),
        artifact_row_counts,
        spatial_array_rows,
    )
    errors.extend(row_lineage["errors"])

    feature_guard = _audit_model_identifier_columns(
        x_columns,
        spatial_array_rows,
        artifact_audits,
    )
    errors.extend(feature_guard["errors"])

    coverage = _audit_coverage(
        frame_stages.get("context", pd.DataFrame()),
        sequence_manifest,
        sequence_features,
    )
    errors.extend(coverage["errors"])

    artifact_errors = _artifact_errors(artifact_audits)
    errors.extend(artifact_errors)
    stage_warnings = _artifact_warnings(artifact_audits)
    warnings.extend(stage_warnings)
    technical_pass = not errors
    human_blockers = [
        "Hidden human review is incomplete and is not applied in this smoke.",
        "Behavior human review is incomplete and is not applied in this smoke.",
    ]
    return {
        "schema_version": "classification_v2.source_to_window_lineage.v1",
        "technical_pass": technical_pass,
        "status": (
            "PASS_IDENTIFIER_V2_TECHNICAL_HUMAN_REVIEW_BLOCKED"
            if technical_pass
            else "FAIL_IDENTIFIER_V2_LINEAGE"
        ),
        "authorization": {
            "reviewed_dataset_authorized": False,
            "model_training_authorized": False,
            "full_oof_authorized": False,
            "q2_claim_authorized": False,
        },
        "frame_lineage": frame_lineage,
        "image_frame_mapping": image_frame_mapping,
        "window_lineage": window_lineage,
        "exported_window_hashes": audit_hashes,
        "row_lineage": row_lineage,
        "feature_identifier_guard": feature_guard,
        "coverage": coverage,
        "artifact_contract_errors": artifact_errors,
        "human_review_blockers": human_blockers,
        "errors": errors,
        "warnings": warnings,
    }


def _audit_frame_lineage(
    frame_stages: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Compare every frame-feature stage against the context-stage actor rows."""

    errors: list[str] = []
    missing_stages = [name for name in EXPECTED_FRAME_STAGES if name not in frame_stages]
    if missing_stages:
        errors.append(f"missing_frame_stages={missing_stages}")
    reference = frame_stages.get("context", pd.DataFrame())
    reference_audit = _identifier_stage_audit("context", reference)
    errors.extend(reference_audit["errors"])
    comparisons: dict[str, dict[str, Any]] = {}
    for name in EXPECTED_FRAME_STAGES[1:]:
        candidate = frame_stages.get(name, pd.DataFrame())
        comparison = _compare_frame_tables(
            reference,
            candidate,
            candidate_name=name,
            require_order=True,
        )
        comparisons[name] = comparison
        errors.extend(f"{name}:{value}" for value in comparison["errors"])
    return {
        "reference_stage": "context",
        "reference": reference_audit,
        "comparisons": comparisons,
        "errors": errors,
        "valid": not errors,
    }


def _audit_image_frame_mapping(
    harmonized: pd.DataFrame,
    image_frames: pd.DataFrame,
) -> dict[str, Any]:
    """Require keyed image-frame coverage while allowing its deterministic sort."""

    return _compare_frame_tables(
        harmonized,
        image_frames,
        candidate_name="image_frame_manifest",
        require_order=False,
    )


def _identifier_stage_audit(name: str, rows: pd.DataFrame) -> dict[str, Any]:
    """Return identifier evidence without throwing on a malformed fixture."""

    required = {"identifier_schema_version", "scene_frame_uid", "frame_uid"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        return {
            "stage": name,
            "rows": int(len(rows)),
            "ordered_frame_uid_sha256": None,
            "errors": [f"missing_identifier_columns={missing}"],
            "valid": False,
        }
    audit = audit_frame_object_identifiers(rows)
    frame_ids = _clean_text(rows["frame_uid"])
    return {
        "stage": name,
        **audit,
        "ordered_frame_uid_sha256": _ordered_text_sha256(frame_ids),
    }


def _compare_frame_tables(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    candidate_name: str,
    require_order: bool,
) -> dict[str, Any]:
    """Compare actor keys, locator values, and optionally positional order."""

    reference_audit = _identifier_stage_audit("reference", reference)
    candidate_audit = _identifier_stage_audit(candidate_name, candidate)
    errors = list(candidate_audit["errors"])
    if reference_audit["errors"]:
        errors.append("reference_identifier_contract_invalid")
    if errors:
        return {
            "candidate": candidate_name,
            "rows": int(len(candidate)),
            "missing_frame_uid_count": None,
            "extra_frame_uid_count": None,
            "order_mismatch_rows": None,
            "locator_mismatch_counts": {},
            "identifier_audit": candidate_audit,
            "errors": errors,
            "valid": False,
        }

    reference_ids = _clean_text(reference["frame_uid"])
    candidate_ids = _clean_text(candidate["frame_uid"])
    reference_set = set(reference_ids)
    candidate_set = set(candidate_ids)
    missing = sorted(reference_set.difference(candidate_set))
    extra = sorted(candidate_set.difference(reference_set))
    order_mismatch = (
        _ordered_mismatch_count(reference_ids, candidate_ids)
        if require_order
        else 0
    )
    if missing:
        errors.append(f"missing_frame_uid_count={len(missing)}")
    if extra:
        errors.append(f"extra_frame_uid_count={len(extra)}")
    if order_mismatch:
        errors.append(f"frame_uid_order_mismatch_rows={order_mismatch}")

    missing_locator_columns = sorted(
        column
        for column in FRAME_LOCATOR_COLUMNS
        if column not in reference.columns or column not in candidate.columns
    )
    locator_mismatches: dict[str, int] = {}
    if missing_locator_columns:
        errors.append(f"missing_locator_columns={missing_locator_columns}")
    else:
        locator_mismatches = _frame_locator_mismatches(
            reference,
            candidate,
            reference_ids,
            candidate_set,
        )
    for column, count in locator_mismatches.items():
        if count:
            errors.append(f"locator_mismatch_{column}={count}")
    return {
        "candidate": candidate_name,
        "rows": int(len(candidate)),
        "ordered_frame_uid_sha256": _ordered_text_sha256(candidate_ids),
        "missing_frame_uid_count": int(len(missing)),
        "extra_frame_uid_count": int(len(extra)),
        "order_mismatch_rows": int(order_mismatch),
        "locator_mismatch_counts": locator_mismatches,
        "identifier_audit": candidate_audit,
        "errors": errors,
        "valid": not errors,
    }


def _frame_locator_mismatches(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    reference_ids: pd.Series,
    candidate_set: set[str],
) -> dict[str, int]:
    """Count changed inference locators for actor keys shared by both tables."""

    mismatch: dict[str, int] = {}
    common = [value for value in reference_ids if value in candidate_set]
    selected = ["frame_uid", *FRAME_LOCATOR_COLUMNS]
    reference_lookup = reference.loc[:, selected].copy()
    candidate_lookup = candidate.loc[:, selected].copy()
    reference_lookup["_frame_uid"] = reference_ids.to_numpy()
    candidate_lookup["_frame_uid"] = _clean_text(
        candidate["frame_uid"]
    ).to_numpy()
    reference_lookup = reference_lookup.set_index("_frame_uid")
    candidate_lookup = candidate_lookup.set_index("_frame_uid")
    for column in FRAME_LOCATOR_COLUMNS:
        left = _normalized_locator(reference_lookup.loc[common, column], column)
        right = _normalized_locator(candidate_lookup.loc[common, column], column)
        mismatch[column] = int(left.reset_index(drop=True).ne(
            right.reset_index(drop=True)
        ).sum())
    return mismatch


def _audit_window_lineage(
    sequence_manifest: pd.DataFrame,
    sequence_features: pd.DataFrame,
    image_windows: pd.DataFrame,
) -> dict[str, Any]:
    """Audit sequence, tabular-feature, and image window positional identity."""

    tables = {
        "sequence_manifest": sequence_manifest,
        "sequence_features": sequence_features,
        "image_window_manifest": image_windows,
    }
    missing_columns = [name for name, rows in tables.items() if "window_id" not in rows]
    if missing_columns:
        return {
            "ordered_window_id_sha256": None,
            "errors": [f"missing_window_id_columns={missing_columns}"],
            "valid": False,
        }
    audit = audit_ordered_window_ids(
        "sequence_manifest",
        sequence_manifest["window_id"],
        {
            "sequence_features": sequence_features["window_id"],
            "image_window_manifest": image_windows["window_id"],
        },
    )
    return {
        **audit,
        "ordered_window_id_sha256": ordered_window_id_sha256(
            sequence_manifest["window_id"]
        ),
    }


def _audit_exported_window_hashes(
    artifact_audits: dict[str, dict[str, Any]],
    expected_hash: str | None,
) -> dict[str, Any]:
    """Reconcile ordered-key hashes emitted by every positional exporter."""

    paths = {
        "train_ready": ("window_alignment", "reference_ordered_window_id_sha256"),
        "spatial": ("window_alignment", "reference_ordered_window_id_sha256"),
        "image_context_input": (
            "window_alignment",
            "reference_ordered_window_id_sha256",
        ),
        "image_context_output": (
            "window_alignment",
            "comparisons",
            "image_context_windows",
            "ordered_window_id_sha256",
        ),
    }
    source_names = {
        "train_ready": "train_ready",
        "spatial": "spatial",
        "image_context_input": "image_context",
        "image_context_output": "image_context",
    }
    values: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for name, path in paths.items():
        payload = artifact_audits.get(source_names[name], {})
        value = _nested_value(payload, path)
        matches = bool(expected_hash) and value == expected_hash
        values[name] = {"sha256": value, "matches_sequence": matches}
        if not matches:
            errors.append(f"ordered_window_hash_mismatch={name}")
    return {
        "expected_sha256": expected_hash,
        "artifacts": values,
        "errors": errors,
        "valid": not errors,
    }


def _audit_row_counts(
    expected_rows: int,
    artifact_row_counts: dict[str, int],
    spatial_array_rows: dict[str, int],
) -> dict[str, Any]:
    """Reject row loss across tabular targets, masks, and spatial arrays."""

    errors: list[str] = []
    for name, rows in artifact_row_counts.items():
        if int(rows) != int(expected_rows):
            errors.append(f"artifact_row_mismatch={name}:{rows}!={expected_rows}")
    for name, rows in spatial_array_rows.items():
        if int(rows) != int(expected_rows):
            errors.append(f"spatial_row_mismatch={name}:{rows}!={expected_rows}")
    return {
        "expected_window_rows": int(expected_rows),
        "artifact_rows": artifact_row_counts,
        "spatial_array_rows": spatial_array_rows,
        "errors": errors,
        "valid": not errors,
    }


def _audit_model_identifier_columns(
    x_columns: list[str],
    spatial_array_rows: dict[str, int],
    artifact_audits: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Keep stable keys and source metadata outside all model-input tensors."""

    forbidden_x = sorted(column for column in x_columns if _is_identifier_name(column))
    forbidden_arrays = sorted(
        name for name in spatial_array_rows if _is_identifier_name(name)
    )
    feature_errors = _nested_value(
        artifact_audits.get("train_ready", {}),
        ("feature_selection", "errors"),
    )
    errors: list[str] = []
    if forbidden_x:
        errors.append(f"identifier_columns_in_tabular_x={forbidden_x}")
    if forbidden_arrays:
        errors.append(f"identifier_arrays_in_spatial_x={forbidden_arrays}")
    if feature_errors:
        errors.append("train_ready_feature_selection_has_errors")
    return {
        "tabular_x_columns": int(len(x_columns)),
        "forbidden_tabular_columns": forbidden_x,
        "forbidden_spatial_arrays": forbidden_arrays,
        "errors": errors,
        "valid": not errors,
    }


def _audit_coverage(
    context: pd.DataFrame,
    sequence_manifest: pd.DataFrame,
    sequence_features: pd.DataFrame,
) -> dict[str, Any]:
    """Require both sources and every canonical behavior in the bounded chain."""

    errors: list[str] = []
    frame_sources = _value_counts(context, "source_type")
    window_sources = _value_counts(sequence_manifest, "source_type")
    behavior_counts = _value_counts(sequence_features, "behavior_window_label")
    if set(frame_sources) != EXPECTED_SOURCES:
        errors.append(f"frame_source_coverage={sorted(frame_sources)}")
    if set(window_sources) != EXPECTED_SOURCES:
        errors.append(f"window_source_coverage={sorted(window_sources)}")
    expected_behaviors = set(VALID_BEHAVIORS)
    if set(behavior_counts) != expected_behaviors:
        missing = sorted(expected_behaviors.difference(behavior_counts))
        unexpected = sorted(set(behavior_counts).difference(expected_behaviors))
        errors.append(
            f"behavior_coverage_mismatch=missing:{missing},unexpected:{unexpected}"
        )
    return {
        "frame_source_counts": frame_sources,
        "window_source_counts": window_sources,
        "behavior_window_counts": behavior_counts,
        "errors": errors,
        "valid": not errors,
    }


def _artifact_warnings(
    artifact_audits: dict[str, dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    for name, payload in artifact_audits.items():
        for value in payload.get("warnings") or []:
            warnings.append(f"{name}:{value}")
    return warnings


def _artifact_errors(
    artifact_audits: dict[str, dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    for name, payload in artifact_audits.items():
        for value in payload.get("errors") or []:
            errors.append(f"{name}:{value}")
    return errors


def _is_identifier_name(column: str) -> bool:
    lowered = str(column).strip().lower()
    return (
        lowered in MODEL_IDENTIFIER_COLUMNS
        or lowered.endswith("_uid")
        or lowered.endswith("_id")
        or lowered.endswith("_key")
        or lowered.endswith("_path")
        or lowered.startswith("fold_")
    )


def _value_counts(rows: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in rows.columns:
        return {}
    values = _clean_text(rows[column])
    values = values[values.ne("")]
    return {str(key): int(value) for key, value in values.value_counts().items()}


def _nested_value(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _normalized_locator(values: pd.Series, column: str) -> pd.Series:
    if column == "frame_index":
        numeric = pd.to_numeric(values, errors="coerce")
        return numeric.map(lambda value: "" if pd.isna(value) else f"{value:.12g}")
    return _clean_text(values)


def _clean_text(values: pd.Series) -> pd.Series:
    cleaned = values.fillna("").astype(str).str.strip().reset_index(drop=True)
    return cleaned.mask(cleaned.isin({"nan", "None", "<NA>"}), "")


def _ordered_mismatch_count(reference: pd.Series, candidate: pd.Series) -> int:
    size = max(len(reference), len(candidate))
    left = reference.reindex(range(size), fill_value="")
    right = candidate.reindex(range(size), fill_value="")
    return int(left.ne(right).sum())


def _ordered_text_sha256(values: pd.Series) -> str:
    payload = "\n".join(_clean_text(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "EXPECTED_FRAME_STAGES",
    "FRAME_OBJECT_IDENTIFIER_VERSION",
    "audit_source_to_window_lineage",
]
