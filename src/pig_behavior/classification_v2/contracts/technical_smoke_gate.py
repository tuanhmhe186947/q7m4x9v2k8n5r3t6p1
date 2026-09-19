"""Cross-stage contract for bounded classification_v2 scientific smoke runs."""

from __future__ import annotations

from typing import Any

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

EXPECTED_SOURCES = {"cvat_tracking_xml", "legacy_recovered"}


def audit_technical_smoke_gate(
    payloads: dict[str, dict[str, Any]],
    *,
    repeatability: dict[str, Any],
    decision_files: list[str],
    preload_errors: list[str] | None = None,
) -> dict[str, Any]:
    """Reconcile bounded artifacts without authorizing reviewed training data."""

    errors = list(preload_errors or [])
    warnings: list[str] = []
    for name, payload in payloads.items():
        stage_errors = payload.get("errors") or []
        stage_warnings = payload.get("warnings") or []
        errors.extend(f"{name}:{value}" for value in stage_errors)
        warnings.extend(f"{name}:{value}" for value in stage_warnings)

    scope = payloads.get("scope", {})
    enhanced = payloads.get("enhanced", {})
    sequence = payloads.get("sequence", {})
    review = payloads.get("review_units", {})
    temporal = payloads.get("temporal_evidence", {})
    train_ready = payloads.get("train_ready", {})
    semantics = payloads.get("feature_semantics", {})
    spatial = payloads.get("spatial_validation", {})

    selected_rows = _int_value(scope.get("selected_rows"))
    enhanced_rows = _int_value(enhanced.get("rows"))
    temporal_rows = _int_value(
        sequence.get("temporal_harmonization", {}).get("rows")
    )
    interval_rows = _int_value(
        sequence.get("temporal_harmonization", {}).get("temporal_intervals")
    )
    window_rows = _int_value(
        sequence.get("sequence_windows", {}).get("window_rows")
    )
    review_rows = _int_value(review.get("rows", {}).get("review_units"))
    x_rows = _int_value(train_ready.get("rows", {}).get("X"))
    y_rows = _int_value(train_ready.get("rows", {}).get("y"))
    spatial_rows = _int_value(spatial.get("rows"))
    _require_equal(
        "frame_row_lineage",
        {
            "scope": selected_rows,
            "enhanced": enhanced_rows,
            "harmonized": temporal_rows,
        },
        errors,
    )
    _require_equal(
        "native_unit_lineage",
        {
            "intervals": interval_rows,
            "review_units": review_rows,
            "temporal_audit_intervals": _int_value(
                temporal.get("rows", {}).get("intervals")
            ),
        },
        errors,
    )
    _require_equal(
        "window_row_lineage",
        {
            "windows": window_rows,
            "tabular_X": x_rows,
            "target_y": y_rows,
            "spatial_X": spatial_rows,
        },
        errors,
    )

    source_counts = scope.get("selected_source_counts") or {}
    if set(source_counts) != EXPECTED_SOURCES:
        errors.append(f"source_coverage_mismatch={sorted(source_counts)}")
    behavior_counts = scope.get("selected_behavior_counts") or {}
    if set(behavior_counts) != set(VALID_BEHAVIORS):
        missing = sorted(set(VALID_BEHAVIORS).difference(behavior_counts))
        unexpected = sorted(set(behavior_counts).difference(VALID_BEHAVIORS))
        errors.append(
            f"behavior_coverage_mismatch=missing:{missing} unexpected:{unexpected}"
        )
    nonpositive_behaviors = sorted(
        behavior
        for behavior, count in behavior_counts.items()
        if _int_value(count) in {None, 0}
    )
    if nonpositive_behaviors:
        errors.append(f"behaviors_without_rows={nonpositive_behaviors}")

    _check_temporal_contract(temporal, errors)
    _check_train_ready_contract(train_ready, window_rows, errors)
    _check_feature_semantics(semantics, errors)
    _check_spatial_contract(spatial, x_rows, errors)
    if decision_files:
        errors.append(f"unexpected_human_decision_files={decision_files}")
    if repeatability.get("all_match") is not True:
        errors.append("repeatability_csv_mismatch")

    technical_pass = not errors
    human_blockers = [
        "Hidden decisions are not applied in this bounded technical smoke.",
        "Behavior review decisions are not applied in this bounded technical smoke.",
    ]
    return {
        "schema_version": "classification_v2_technical_smoke_gate_v1",
        "technical_pass": technical_pass,
        "status": (
            "PASS_TECHNICAL_SMOKE_HUMAN_REVIEW_BLOCKED"
            if technical_pass
            else "FAIL_TECHNICAL_SMOKE"
        ),
        "authorization": {
            "reviewed_dataset_authorized": False,
            "full_training_authorized": False,
            "full_oof_authorized": False,
        },
        "counts": {
            "selected_frame_rows": selected_rows,
            "enhanced_frame_rows": enhanced_rows,
            "harmonized_frame_rows": temporal_rows,
            "native_interval_rows": interval_rows,
            "review_unit_rows": review_rows,
            "window_rows": window_rows,
            "tabular_x_rows": x_rows,
            "spatial_x_rows": spatial_rows,
            "train_mask_true": _int_value(
                train_ready.get("rows", {}).get("mask_true")
            ),
            "train_mask_false": _int_value(
                train_ready.get("rows", {}).get("mask_false")
            ),
        },
        "source_counts": source_counts,
        "behavior_counts": behavior_counts,
        "feature_contract": {
            "tabular_feature_count": semantics.get("tabular_feature_count"),
            "tabular_contract_match": semantics.get("tabular_contract_match"),
            "spatial_model_input_array_count": semantics.get(
                "spatial_model_input_array_count"
            ),
            "temporal_evidence_feature_count": temporal.get(
                "evidence_column_counts", {}
            ).get("window"),
        },
        "native_key_contract": temporal.get("keys", {}),
        "spatial_train_mask_contract": spatial.get(
            "train_mask_completeness", {}
        ),
        "repeatability": repeatability,
        "human_decision_files": decision_files,
        "human_gate_blockers": human_blockers,
        "errors": errors,
        "warnings": warnings,
    }


def _check_temporal_contract(
    temporal: dict[str, Any],
    errors: list[str],
) -> None:
    """Validate native lengths, keys, and evidence-to-whitelist separation."""

    if temporal.get("valid") is not True:
        errors.append("temporal_evidence_audit_not_valid")
    keys = temporal.get("keys", {})
    if _int_value(keys.get("duplicate_temporal_unit_key")) != 0:
        errors.append("duplicate_temporal_unit_key_nonzero")
    if _int_value(keys.get("duplicate_window_id")) != 0:
        errors.append("duplicate_window_id_nonzero")
    review = temporal.get("review_units", {})
    if _int_value(review.get("duplicate_review_unit_id")) != 0:
        errors.append("duplicate_review_unit_id_nonzero")
    native = temporal.get("native_lengths", {})
    if _int_value(native.get("cvat_invalid")) != 0:
        errors.append("invalid_cvat_native_lengths")
    if _int_value(native.get("legacy_invalid")) != 0:
        errors.append("invalid_legacy_native_lengths")
    if temporal.get("missing_temporal_evidence_from_whitelist"):
        errors.append("temporal_evidence_missing_from_whitelist")
    if temporal.get("review_evidence_in_model_whitelist"):
        errors.append("review_evidence_in_model_whitelist")


def _check_train_ready_contract(
    train_ready: dict[str, Any],
    window_rows: int | None,
    errors: list[str],
) -> None:
    """Validate exact X whitelist and mask preservation."""

    features = train_ready.get("feature_selection", {})
    if features.get("explicit_whitelist_used") is not True:
        errors.append("train_ready_explicit_whitelist_not_used")
    if features.get("feature_whitelist_match") is not True:
        errors.append("train_ready_feature_whitelist_mismatch")
    if features.get("forbidden_selected"):
        errors.append("train_ready_forbidden_features_selected")
    rows = train_ready.get("rows", {})
    mask_total = _sum_optional(
        _int_value(rows.get("mask_true")),
        _int_value(rows.get("mask_false")),
    )
    if mask_total != window_rows:
        errors.append(
            f"train_mask_row_mismatch=mask:{mask_total} windows:{window_rows}"
        )


def _check_feature_semantics(
    semantics: dict[str, Any],
    errors: list[str],
) -> None:
    """Validate model-input roles and exact tabular contract."""

    if semantics.get("valid") is not True:
        errors.append("feature_semantics_not_valid")
    if semantics.get("tabular_contract_match") is not True:
        errors.append("tabular_contract_not_exact")
    if semantics.get("forbidden_tabular_features"):
        errors.append("forbidden_tabular_features_present")
    if semantics.get("undeclared_spatial_arrays"):
        errors.append("undeclared_spatial_arrays_present")
    if semantics.get("spatial_model_input_role_errors"):
        errors.append("spatial_model_input_role_errors_present")


def _check_spatial_contract(
    spatial: dict[str, Any],
    x_rows: int | None,
    errors: list[str],
) -> None:
    """Ensure missing observed slots cannot enter the trainable subset."""

    if _int_value(spatial.get("rows")) != x_rows:
        errors.append("spatial_row_count_does_not_match_tabular_x")
    train_mask = spatial.get("train_mask_completeness", {})
    if train_mask.get("available") is not True:
        errors.append("spatial_train_mask_completeness_not_audited")
    if _int_value(train_mask.get("trainable_rows_with_missing_slots")) != 0:
        errors.append("trainable_spatial_rows_have_missing_slots")
    if _int_value(train_mask.get("trainable_missing_slots")) != 0:
        errors.append("trainable_spatial_missing_slot_count_nonzero")


def _require_equal(
    name: str,
    values: dict[str, int | None],
    errors: list[str],
) -> None:
    """Require known, equal counts across one lineage boundary."""

    if any(value is None for value in values.values()):
        errors.append(f"{name}_has_missing_counts={values}")
        return
    if len(set(values.values())) != 1:
        errors.append(f"{name}_mismatch={values}")


def _int_value(value: object) -> int | None:
    """Return an exact integer count without silently coercing bad values."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _sum_optional(*values: int | None) -> int | None:
    """Sum counts only when every component exists."""

    return sum(values) if all(value is not None for value in values) else None
