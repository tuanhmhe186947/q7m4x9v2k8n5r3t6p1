"""Structural source-shortcut audits for classification_v2 temporal views."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.contracts.model_io import (
    validate_model_input_columns,
)
from pig_behavior.classification_v2.features.temporal_views import (
    FIXED6_NORMALIZED_PHASE,
    FIXED6_OBSERVED_TIME,
    MODEL_TENSOR_COLUMNS,
    NATIVE6_16,
)

SIGNATURE_FAMILIES = {
    "length": ["declared_sequence_length"],
    "padding": ["padding_mask"],
    "observed": ["observed_mask"],
    "quality": ["bbox_quality_mask", "spatiotemporal_quality_mask"],
    "timing": ["timing_valid_mask", "time_value", "time_delta"],
    "availability": [
        "roi_feeder_available_mask",
        "roi_drinker_available_mask",
        "roi_toy_available_mask",
        "social_neighbor_available_mask",
        "actor_context_available_mask",
        "partner_context_available_mask",
    ],
}


def audit_temporal_view_shortcuts(
    selection: pd.DataFrame,
    observed: pd.DataFrame,
    phase: pd.DataFrame,
    native: pd.DataFrame,
    contract: dict[str, Any],
    *,
    direct_accuracy_threshold: float = 0.95,
    minimum_uplift: float = 0.10,
    mitigated_families: Iterable[str] = (),
    require_artifact_contract: bool = False,
) -> dict[str, Any]:
    """Audit whether structural temporal patterns nearly determine source.

    This is a deterministic pattern audit, not a behavior classifier. A family
    may be declared mitigated only after a separate, versioned control provides
    evidence. Native length is reported as an expected ablation confound.
    """

    _validate_thresholds(direct_accuracy_threshold, minimum_uplift)
    errors: list[str] = []
    warnings: list[str] = []
    mitigated = {str(value).strip() for value in mitigated_families if str(value).strip()}
    _validate_contract(contract, errors)
    _validate_selection(selection, errors)
    for name, slots in [
        (FIXED6_OBSERVED_TIME, observed),
        (FIXED6_NORMALIZED_PHASE, phase),
        (NATIVE6_16, native),
    ]:
        _validate_slots(name, slots, errors)
    _validate_fixed_pair(observed, phase, errors)
    _validate_artifact_contract(
        contract,
        {
            "selection": selection,
            "fixed6_observed_time": observed,
            "fixed6_normalized_phase": phase,
            "native6_16": native,
        },
        errors,
        required=require_artifact_contract,
    )

    reports: dict[str, dict[str, Any]] = {}
    for view_name, slots in [
        (FIXED6_OBSERVED_TIME, observed),
        (FIXED6_NORMALIZED_PHASE, phase),
        (NATIVE6_16, native),
    ]:
        reports[view_name] = _view_shortcut_report(
            slots,
            direct_accuracy_threshold=direct_accuracy_threshold,
            minimum_uplift=minimum_uplift,
        )

    hard_families = {
        (FIXED6_OBSERVED_TIME, "length"),
        (FIXED6_OBSERVED_TIME, "padding"),
        (FIXED6_NORMALIZED_PHASE, "length"),
        (FIXED6_NORMALIZED_PHASE, "padding"),
        (FIXED6_NORMALIZED_PHASE, "timing"),
    }
    monitored_families = {
        (FIXED6_OBSERVED_TIME, "observed"),
        (FIXED6_OBSERVED_TIME, "quality"),
        (FIXED6_OBSERVED_TIME, "timing"),
        (FIXED6_OBSERVED_TIME, "availability"),
        (FIXED6_NORMALIZED_PHASE, "observed"),
        (FIXED6_NORMALIZED_PHASE, "quality"),
        (FIXED6_NORMALIZED_PHASE, "availability"),
    }
    unmitigated: list[str] = []
    for view_name, family in sorted(hard_families | monitored_families):
        report = reports.get(view_name, {}).get("families", {}).get(family, {})
        if not report.get("near_direct_source_signature", False):
            continue
        key = f"{view_name}:{family}"
        if key in mitigated or family in mitigated:
            warnings.append(f"declared_mitigated_source_shortcut={key}")
            continue
        unmitigated.append(key)
        errors.append(f"unmitigated_near_direct_source_shortcut={key}")

    native_length = reports.get(NATIVE6_16, {}).get("families", {}).get("length", {})
    if native_length.get("near_direct_source_signature", False):
        warnings.append("native6_16_length_is_expected_source_confound_ablation_only")

    label_shortcuts = _label_shortcut_report(
        selection,
        direct_accuracy_threshold=direct_accuracy_threshold,
        minimum_uplift=minimum_uplift,
    )
    for name, report in label_shortcuts.items():
        if report.get("near_direct_target_signature", False):
            errors.append(f"audit_metadata_nearly_determines_behavior={name}")

    primary = reports.get(FIXED6_OBSERVED_TIME, {})
    phase_report = reports.get(FIXED6_NORMALIZED_PHASE, {})
    result = {
        "schema_version": "classification_v2_temporal_shortcut_audit_v1",
        "method": "deterministic_signature_mapping_without_behavior_model_fit",
        "thresholds": {
            "direct_accuracy": direct_accuracy_threshold,
            "minimum_uplift_over_majority": minimum_uplift,
        },
        "mitigated_families": sorted(mitigated),
        "persisted_artifact_contract_required": bool(require_artifact_contract),
        "persisted_artifact_contract_present": bool(contract.get("artifact_contract")),
        "view_reports": reports,
        "label_shortcut_reports": label_shortcuts,
        "fixed6_length_pattern_shared_across_sources": _family_safe(
            primary,
            "length",
        ),
        "fixed6_padding_pattern_shared_across_sources": _family_safe(
            primary,
            "padding",
        ),
        "phase_timing_pattern_shared_across_sources": _family_safe(
            phase_report,
            "timing",
        ),
        "native_length_confound_expected": bool(
            native_length.get("near_direct_source_signature", False)
        ),
        "unmitigated_shortcuts": unmitigated,
        "model_input_fields": list(contract.get("model_tensor_columns", [])),
        "source_metadata_in_model_inputs": bool(
            validate_model_input_columns(
                list(contract.get("model_tensor_columns", []))
            )["forbidden_columns"]
        ),
        "training_stop_required": bool(errors),
        "training_authorized": False,
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }
    return result


def write_temporal_shortcut_audit(
    audit: dict[str, Any],
    output_json: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Persist a strict audit without silently replacing earlier evidence."""

    if output_json.exists() and not overwrite:
        raise FileExistsError(f"temporal shortcut audit already exists: {output_json}")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )


def _validate_thresholds(direct_accuracy: float, minimum_uplift: float) -> None:
    """Reject thresholds that cannot express a bounded probability rule."""

    if not 0.5 <= direct_accuracy <= 1.0:
        raise ValueError("direct_accuracy_threshold must be in [0.5, 1.0]")
    if not 0.0 <= minimum_uplift <= 0.5:
        raise ValueError("minimum_uplift must be in [0.0, 0.5]")


def _validate_contract(contract: dict[str, Any], errors: list[str]) -> None:
    """Validate explicit tensor fields and fixed-view policy declarations."""

    model_columns = list(contract.get("model_tensor_columns", []))
    schema_audit = validate_model_input_columns(model_columns)
    if not model_columns:
        errors.append("temporal_contract_missing_model_tensor_columns")
    if schema_audit["forbidden_columns"]:
        errors.append(
            "forbidden_temporal_model_fields="
            f"{schema_audit['forbidden_columns']}"
        )
    missing_tensor_fields = sorted(set(MODEL_TENSOR_COLUMNS) - set(model_columns))
    if missing_tensor_fields:
        errors.append(f"missing_temporal_tensor_fields={missing_tensor_fields}")
    if contract.get("selection_rule") != "exact_final_T6_contiguous_only":
        errors.append("unsupported_fixed6_selection_rule")
    if contract.get("window_uid_created") is not False:
        errors.append("temporal_contract_must_forbid_window_uid")


def _validate_selection(selection: pd.DataFrame, errors: list[str]) -> None:
    """Prove the ledger has one unique row and one explicit decision per window."""

    required = {
        "input_window_order",
        "window_id",
        "source_type",
        "window_length_frames",
        "view_type",
        "sampling_pattern",
        "fixed6_keep",
        "fixed6_reason",
    }
    missing = sorted(required - set(selection.columns))
    if missing:
        errors.append(f"selection_missing_columns={missing}")
        return
    blank = int(selection["window_id"].fillna("").astype(str).str.strip().eq("").sum())
    duplicate = int(selection["window_id"].duplicated(keep=False).sum())
    if blank:
        errors.append(f"selection_blank_window_ids={blank}")
    if duplicate:
        errors.append(f"selection_duplicate_window_rows={duplicate}")
    expected_order = np.arange(len(selection), dtype=np.int64)
    observed_order = pd.to_numeric(
        selection["input_window_order"],
        errors="coerce",
    ).to_numpy()
    if not np.array_equal(observed_order, expected_order):
        errors.append("selection_input_window_order_not_contiguous")
    keep = selection["fixed6_keep"].map(_bool_scalar)
    lengths = pd.to_numeric(selection["window_length_frames"], errors="coerce")
    expected_keep = (
        lengths.eq(6)
        & selection["view_type"].astype(str).eq("T6_contiguous")
        & selection["sampling_pattern"].astype(str).eq("contiguous")
    )
    if not keep.equals(expected_keep):
        errors.append(
            "selection_fixed6_keep_does_not_match_exact_T6_contiguous"
        )
    selected_reason = selection.loc[keep, "fixed6_reason"].astype(str)
    if not selected_reason.eq("selected_exact_T6_contiguous_window").all():
        errors.append("selection_fixed6_selected_reason_mismatch")


def _validate_artifact_contract(
    contract: dict[str, Any],
    artifacts: dict[str, pd.DataFrame],
    errors: list[str],
    *,
    required: bool,
) -> None:
    """Verify persisted row counts and ordered keys against the build contract."""

    expected = contract.get("artifact_contract")
    if not isinstance(expected, dict):
        if required:
            errors.append("missing_persisted_temporal_artifact_contract")
        return
    for name, frame in artifacts.items():
        profile = expected.get(name)
        if not isinstance(profile, dict):
            errors.append(f"missing_temporal_artifact_profile={name}")
            continue
        key_column = str(profile.get("key_column", ""))
        if key_column not in frame.columns:
            errors.append(f"artifact_contract_key_missing={name}:{key_column}")
            continue
        if int(profile.get("rows", -1)) != len(frame):
            errors.append(
                f"artifact_contract_row_mismatch={name}:"
                f"{profile.get('rows')}!={len(frame)}"
            )
        observed_hash = _ordered_digest(frame[key_column])
        if profile.get("ordered_key_sha256") != observed_hash:
            errors.append(f"artifact_contract_ordered_key_mismatch={name}")


def _validate_slots(
    expected_view: str,
    slots: pd.DataFrame,
    errors: list[str],
) -> None:
    """Validate one ordered, unique slot table and its explicit masks."""

    required = {
        "temporal_view_name",
        "view_item_id",
        "source_type",
        "item_order",
        "slot_index",
        "slot_key",
        "declared_sequence_length",
        *MODEL_TENSOR_COLUMNS,
        "padding_mask",
    }
    missing = sorted(required - set(slots.columns))
    if missing:
        errors.append(f"{expected_view}:missing_slot_columns={missing}")
        return
    names = set(slots["temporal_view_name"].astype(str))
    if names != {expected_view}:
        errors.append(f"{expected_view}:unexpected_view_names={sorted(names)}")
    blank = int(slots["slot_key"].fillna("").astype(str).str.strip().eq("").sum())
    duplicate = int(slots["slot_key"].duplicated(keep=False).sum())
    if blank:
        errors.append(f"{expected_view}:blank_slot_keys={blank}")
    if duplicate:
        errors.append(f"{expected_view}:duplicate_slot_rows={duplicate}")
    for item_id, group in slots.groupby("view_item_id", sort=False):
        lengths = pd.to_numeric(
            group["declared_sequence_length"],
            errors="coerce",
        ).dropna().unique()
        if len(lengths) != 1:
            errors.append(f"{expected_view}:length_conflict={item_id}")
            continue
        length = int(lengths[0])
        slots_observed = pd.to_numeric(group["slot_index"], errors="coerce").tolist()
        if len(group) != length or slots_observed != list(range(length)):
            errors.append(f"{expected_view}:slot_sequence_invalid={item_id}")
    source_count = slots["source_type"].astype(str).nunique()
    if expected_view != NATIVE6_16 and source_count != 2:
        errors.append(f"{expected_view}:requires_two_sources={source_count}")


def _validate_fixed_pair(
    observed: pd.DataFrame,
    phase: pd.DataFrame,
    errors: list[str],
) -> None:
    """Require fixed views to differ only in declared timing coordinates."""

    ignored = {
        "temporal_view_name",
        "time_coordinate_kind",
        "time_value",
        "time_delta",
        "timing_valid_mask",
    }
    shared = [column for column in observed.columns if column not in ignored]
    missing = sorted(set(shared) - set(phase.columns))
    if missing:
        errors.append(f"phase_missing_observed_identity_columns={missing}")
        return
    if len(observed) != len(phase) or not observed[shared].equals(phase[shared]):
        errors.append("fixed6_views_do_not_share_exact_membership_and_order")


def _view_shortcut_report(
    slots: pd.DataFrame,
    *,
    direct_accuracy_threshold: float,
    minimum_uplift: float,
) -> dict[str, Any]:
    """Summarize source predictability from each structural pattern family."""

    if slots.empty or "view_item_id" not in slots:
        return {"item_rows": 0, "families": {}}
    source_by_item = _source_by_item(slots)
    families: dict[str, Any] = {}
    for family, columns in SIGNATURE_FAMILIES.items():
        available = [column for column in columns if column in slots.columns]
        signatures = _item_signatures(slots, available)
        families[family] = _source_signature_report(
            signatures,
            source_by_item,
            direct_accuracy_threshold=direct_accuracy_threshold,
            minimum_uplift=minimum_uplift,
        )
        families[family]["columns"] = available
    return {
        "item_rows": int(len(source_by_item)),
        "slot_rows": int(len(slots)),
        "source_counts": source_by_item.value_counts().sort_index().to_dict(),
        "families": families,
    }


def _source_by_item(slots: pd.DataFrame) -> pd.Series:
    """Return one source per item and reject source conflicts inside a sequence."""

    source_counts = slots.groupby("view_item_id", sort=False)["source_type"].nunique()
    if source_counts.ne(1).any():
        conflicts = int(source_counts.ne(1).sum())
        raise ValueError(f"source conflicts inside temporal items={conflicts}")
    return slots.groupby("view_item_id", sort=False)["source_type"].first().astype(str)


def _item_signatures(slots: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Encode ordered per-slot values into deterministic structural signatures."""

    signatures: dict[str, str] = {}
    for item_id, group in slots.groupby("view_item_id", sort=False):
        ordered = group.sort_values("slot_index", kind="mergesort")
        values = [
            [_canonical_value(value) for value in ordered[column].tolist()]
            for column in columns
        ]
        signatures[str(item_id)] = json.dumps(
            values,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    return pd.Series(signatures, dtype=str)


def _source_signature_report(
    signatures: pd.Series,
    sources: pd.Series,
    *,
    direct_accuracy_threshold: float,
    minimum_uplift: float,
) -> dict[str, Any]:
    """Measure deterministic signature-to-source mapping against majority base."""

    aligned = pd.DataFrame(
        {
            "signature": signatures.reindex(sources.index),
            "source": sources,
        }
    )
    if aligned.empty:
        return _empty_predictability_report()
    table = pd.crosstab(aligned["signature"], aligned["source"])
    correct = int(table.max(axis=1).sum())
    rows = int(table.to_numpy().sum())
    baseline = float(aligned["source"].value_counts().max() / rows)
    accuracy = float(correct / rows)
    pure_signatures = table.gt(0).sum(axis=1).eq(1)
    pure_rows = int(table.loc[pure_signatures].to_numpy().sum())
    shared_rows = rows - pure_rows
    uplift = accuracy - baseline
    near_direct = bool(
        aligned["source"].nunique() >= 2
        and accuracy >= direct_accuracy_threshold
        and uplift >= minimum_uplift
    )
    return {
        "rows": rows,
        "unique_signatures": int(len(table)),
        "source_count": int(aligned["source"].nunique()),
        "majority_source_baseline": baseline,
        "signature_mapping_accuracy": accuracy,
        "uplift_over_majority": uplift,
        "source_pure_signature_row_ratio": float(pure_rows / rows),
        "cross_source_shared_signature_row_ratio": float(shared_rows / rows),
        "near_direct_source_signature": near_direct,
    }


def _label_shortcut_report(
    selection: pd.DataFrame,
    *,
    direct_accuracy_threshold: float,
    minimum_uplift: float,
) -> dict[str, dict[str, Any]]:
    """Audit target association with source/length metadata outside model X."""

    if "behavior_window_label" not in selection.columns:
        return {}
    reports: dict[str, dict[str, Any]] = {}
    fixed = selection.loc[selection["fixed6_keep"].map(_bool_scalar)].copy()
    specifications = [
        ("fixed6_source_to_behavior", fixed, ["source_type"]),
        (
            "all_window_length_to_behavior",
            selection,
            ["window_length_frames"],
        ),
        (
            "all_source_and_length_to_behavior",
            selection,
            ["source_type", "window_length_frames"],
        ),
    ]
    for name, frame, columns in specifications:
        target = frame["behavior_window_label"].fillna("").astype(str)
        signatures = frame[columns].astype(str).agg("|".join, axis=1)
        reports[name] = _target_signature_report(
            signatures,
            target,
            direct_accuracy_threshold=direct_accuracy_threshold,
            minimum_uplift=minimum_uplift,
        )
        reports[name]["predictor_columns_audit_only"] = columns
    return reports


def _target_signature_report(
    signatures: pd.Series,
    target: pd.Series,
    *,
    direct_accuracy_threshold: float,
    minimum_uplift: float,
) -> dict[str, Any]:
    """Measure metadata-to-label purity without fitting a classifier."""

    frame = pd.DataFrame({"signature": signatures, "target": target})
    frame = frame.loc[frame["target"].ne("")]
    if frame.empty:
        return {
            "rows": 0,
            "near_direct_target_signature": False,
        }
    table = pd.crosstab(frame["signature"], frame["target"])
    rows = int(table.to_numpy().sum())
    accuracy = float(table.max(axis=1).sum() / rows)
    baseline = float(frame["target"].value_counts().max() / rows)
    uplift = accuracy - baseline
    return {
        "rows": rows,
        "target_class_count": int(frame["target"].nunique()),
        "majority_target_baseline": baseline,
        "signature_mapping_accuracy": accuracy,
        "uplift_over_majority": uplift,
        "near_direct_target_signature": bool(
            frame["target"].nunique() >= 2
            and accuracy >= direct_accuracy_threshold
            and uplift >= minimum_uplift
        ),
    }


def _family_safe(report: dict[str, Any], family: str) -> bool:
    """Return whether a signature family is not near-direct for source."""

    details = report.get("families", {}).get(family, {})
    return not bool(details.get("near_direct_source_signature", False))


def _canonical_value(value: object) -> str | int | float | None:
    """Canonicalize values before constructing exact pattern signatures."""

    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(bool(value))
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 6)
    return str(value)


def _bool_scalar(value: object) -> bool:
    """Normalize persisted boolean values for independent audit reads."""

    if value is None or pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(float(value))
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def _empty_predictability_report() -> dict[str, Any]:
    """Return a stable zero-row schema for incomplete diagnostic inputs."""

    return {
        "rows": 0,
        "unique_signatures": 0,
        "source_count": 0,
        "majority_source_baseline": 0.0,
        "signature_mapping_accuracy": 0.0,
        "uplift_over_majority": 0.0,
        "source_pure_signature_row_ratio": 0.0,
        "cross_source_shared_signature_row_ratio": 0.0,
        "near_direct_source_signature": False,
    }


def _ordered_digest(values: pd.Series) -> str:
    """Hash normalized ordered keys using the project newline-join contract."""

    payload = "\n".join(values.fillna("").astype(str).str.strip()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
