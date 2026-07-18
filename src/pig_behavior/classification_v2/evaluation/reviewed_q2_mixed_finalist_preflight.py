"""Fail-closed preflight for the reviewed mixed-data SF128/A128 gate.

The preflight reuses the reviewed-Q2 P0 gate, then proves that both temporal
arms consume one identical, inference-safe comparison universe. It never
starts training and never authorizes full OOF.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.contracts.model_io import (
    validate_model_input_columns,
)
from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.evaluation.metrics import (
    DEFAULT_LABEL_ORDER,
)
from pig_behavior.classification_v2.evaluation.reviewed_q2_p0_preflight import (
    build_reviewed_q2_p0_preflight,
)

CONTRACT_SCHEMA_VERSION = (
    "classification_v2.reviewed_q2_mixed_finalist_contract.v1"
)
PREFLIGHT_SCHEMA_VERSION = (
    "classification_v2.reviewed_q2_mixed_finalist_preflight.v1"
)
REQUIRED_SOURCES = ("cvat_tracking_xml", "legacy_recovered")
REQUIRED_ARMS = ("SF128", "A128")
FIXED_VIEW = "fixed6_observed_time"
FILE_CHUNK_BYTES = 1024 * 1024
AVAILABILITY_COLUMNS = (
    "roi_feeder_available_mask",
    "roi_drinker_available_mask",
    "roi_toy_available_mask",
    "social_neighbor_available_mask",
    "actor_context_available_mask",
    "partner_context_available_mask",
)
ARTIFACT_BINDINGS = {
    "native_unit_manifest_sha256": "native_temporal_unit_manifest",
    "fold_manifest_sha256": "q2_outer_fold_assignments",
    "fixed6_view_manifest_sha256": "fixed6_observed_time_manifest",
    "feature_whitelist_sha256": "feature_whitelist",
    "model_input_contract_sha256": "model_input_contract",
    "temporal_view_audit_sha256": "temporal_view_audit",
}
ALLOWED_MODEL_DIFFS = {
    "expected_parameter_count",
    "selected_slot_indices",
    "temporal_encoder_name",
}


def build_reviewed_q2_mixed_finalist_preflight(
    data_contract_json: Path,
    snapshot_json: Path,
    comparison_contract_json: Path,
    handoff_json: Path,
    *,
    project_root: Path,
    output_json: Path | None = None,
) -> dict[str, Any]:
    """Build a read-only authorization record for one paired short gate."""

    root = project_root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    data_contract_path = _input_path(
        data_contract_json,
        root,
        errors,
        "data_contract",
    )
    snapshot_path = _input_path(snapshot_json, root, errors, "snapshot")
    comparison_path = _input_path(
        comparison_contract_json,
        root,
        errors,
        "comparison_contract",
    )
    handoff_path = _input_path(handoff_json, root, errors, "handoff")
    data_contract = _read_json(data_contract_path, errors, "data_contract")
    comparison = _read_json(
        comparison_path,
        errors,
        "comparison_contract",
    )
    handoff = _read_json(handoff_path, errors, "handoff")

    try:
        p0_result = build_reviewed_q2_p0_preflight(
            data_contract_json,
            snapshot_json,
            project_root=root,
            output_json=output_json,
        )
    except Exception as exc:
        p0_result = {
            "valid": False,
            "errors": [f"p0_execution_failed={exc}"],
        }
    if p0_result.get("valid") is not True:
        errors.append(f"reviewed_q2_p0_invalid={p0_result.get('errors')}")

    namespace = _audit_namespace(
        data_contract,
        comparison_path,
        handoff_path,
        output_json,
        root,
    )
    errors.extend(namespace["errors"])
    handoff_check = _audit_handoff(handoff, data_contract)
    errors.extend(handoff_check["errors"])

    artifact_paths = _artifact_paths(data_contract, root, errors)
    actual_bindings = _actual_bindings(
        data_contract_path,
        snapshot_path,
        artifact_paths,
    )
    pairing = _audit_pairing(comparison, actual_bindings)
    errors.extend(pairing["errors"])
    warnings.extend(pairing["warnings"])

    universe = _audit_comparison_universe(
        artifact_paths.get("native_temporal_unit_manifest"),
        artifact_paths.get("q2_outer_fold_assignments"),
        artifact_paths.get("fixed6_observed_time_manifest"),
    )
    errors.extend(universe["errors"])
    warnings.extend(universe["warnings"])

    shortcut = _audit_shortcut_contract(
        artifact_paths.get("temporal_view_audit"),
    )
    errors.extend(shortcut["errors"])
    warnings.extend(shortcut["warnings"])
    inference = _audit_inference_inputs(
        artifact_paths.get("feature_whitelist"),
        artifact_paths.get("model_input_contract"),
        data_contract,
    )
    errors.extend(inference["errors"])

    errors = sorted(set(errors))
    warnings = sorted(set(warnings))
    valid = not errors
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "comparison_id": comparison.get("comparison_id"),
        "profile": data_contract.get("profile"),
        "review_stage": handoff.get("review_stage"),
        "p0": {
            "valid": p0_result.get("valid") is True,
            "errors": p0_result.get("errors", []),
        },
        "checks": {
            "namespace": namespace,
            "behavior_complete_handoff": handoff_check,
            "paired_contract": pairing,
            "comparison_universe": universe,
            "temporal_shortcuts": shortcut,
            "inference_inputs": inference,
        },
        "actual_bindings": actual_bindings,
        "scientific_claim_boundary": {
            "comparison_purpose": "select_temporal_finalist",
            "composite_temporal_change": True,
            "attention_mechanism_claim_allowed": False,
            "reason": (
                "SF128 and A128 differ in selected slots and temporal "
                "pooling; the pair selects a base but does not isolate "
                "attention alone."
            ),
        },
        "short_paired_gate_authorized": valid,
        "development_pilot_authorized": False,
        "full_oof_authorized": False,
        "full_oof_authorization_required": True,
        "next_allowed_action": (
            "run_sf128_a128_short_paired_gate"
            if valid
            else "resolve_review_or_contract_blockers"
        ),
        "errors": errors,
        "warnings": warnings,
        "valid": valid,
    }


def write_reviewed_q2_mixed_finalist_preflight(
    result: dict[str, Any],
    *,
    data_contract_json: Path,
    output_json: Path,
    project_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    """Write the audit only under the generated contract's agent root."""

    root = project_root.resolve()
    errors: list[str] = []
    contract_path = _input_path(
        data_contract_json,
        root,
        errors,
        "data_contract",
    )
    contract = _read_json(contract_path, errors, "data_contract")
    output_path = _project_path(output_json, root, "output")
    agent_root, human_root = _lineage_roots(contract, root, errors)
    if errors or agent_root is None or not _is_under(output_path, agent_root):
        raise ValueError(f"preflight output is not agent-owned: {errors}")
    if human_root is not None and _is_under(output_path, human_root):
        raise ValueError("preflight output cannot be inside human-review root")
    require_output_paths_available([output_path], overwrite=overwrite)
    payload = {
        **result,
        "output_json": output_path.relative_to(root).as_posix(),
        "artifact_written": True,
        "overwrite": bool(overwrite),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_stable_json(payload), encoding="utf-8")
    return payload


def _audit_handoff(
    handoff: dict[str, Any],
    data_contract: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    lineage = data_contract.get("lineage_ids")
    expected_run = lineage.get("human_review") if isinstance(lineage, dict) else None
    if handoff.get("review_stage") != "behavior_complete":
        errors.append(
            "behavior_complete_handoff_required="
            f"{handoff.get('review_stage')}"
        )
    if handoff.get("run_id") != expected_run:
        errors.append("handoff_human_review_run_id_mismatch")
    for field in ("reviewer_name", "review_code_sha"):
        value = handoff.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"handoff_missing_{field}")
    return {
        "valid": not errors,
        "run_id": handoff.get("run_id"),
        "review_stage": handoff.get("review_stage"),
        "errors": errors,
    }


def _audit_pairing(
    comparison: dict[str, Any],
    actual_bindings: dict[str, str | None],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if comparison.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        errors.append("mixed_finalist_contract_schema_mismatch")
    if comparison.get("template_only") is True:
        errors.append("template_contract_cannot_authorize_run")
    expected_scalars = {
        "profile": "mixed-reviewed",
        "scientific_family": "temporal_base_finalist",
        "temporal_view": FIXED_VIEW,
        "outer_predictions_used_for_model_selection": False,
        "full_oof_requested": False,
    }
    for field, expected in expected_scalars.items():
        if comparison.get(field) != expected:
            errors.append(
                f"comparison_contract_value_mismatch:{field}="
                f"{comparison.get(field)}"
            )
    required_controls = {
        "paired_native_unit_evaluation": True,
        "source_stratified_evaluation": True,
        "source_matched_evaluation": True,
        "missingness_stratified_evaluation": True,
        "availability_only_control_predeclared": True,
    }
    controls = comparison.get("evaluation_controls")
    controls = controls if isinstance(controls, dict) else {}
    for field, expected in required_controls.items():
        if controls.get(field) != expected:
            errors.append(f"missing_evaluation_control:{field}")

    arms = comparison.get("arms")
    arms = arms if isinstance(arms, dict) else {}
    if set(arms) != set(REQUIRED_ARMS):
        errors.append(f"required_comparison_arms={list(REQUIRED_ARMS)}")
        return {
            "valid": False,
            "changed_model_fields": [],
            "errors": errors,
            "warnings": warnings,
        }
    sf = arms["SF128"] if isinstance(arms["SF128"], dict) else {}
    attention = arms["A128"] if isinstance(arms["A128"], dict) else {}
    _audit_arm_identity("SF128", sf, "control", errors)
    _audit_arm_identity("A128", attention, "candidate", errors)

    sf_bindings = sf.get("bindings")
    a_bindings = attention.get("bindings")
    if sf_bindings != a_bindings:
        errors.append("candidate_artifact_bindings_differ")
    for field, observed in actual_bindings.items():
        if not observed:
            errors.append(f"actual_binding_missing:{field}")
            continue
        if not isinstance(sf_bindings, dict) or sf_bindings.get(field) != observed:
            errors.append(f"candidate_binding_mismatch:SF128:{field}")
        if not isinstance(a_bindings, dict) or a_bindings.get(field) != observed:
            errors.append(f"candidate_binding_mismatch:A128:{field}")

    sf_protocol = sf.get("protocol")
    a_protocol = attention.get("protocol")
    if sf_protocol != a_protocol:
        errors.append(
            "candidate_protocols_differ="
            f"{_diff_paths(sf_protocol, a_protocol)}"
        )
    protocol = sf_protocol if isinstance(sf_protocol, dict) else {}
    if protocol.get("label_order") != list(DEFAULT_LABEL_ORDER):
        errors.append("candidate_global_label_order_mismatch")
    for field in ("preprocessing", "seed", "loss", "sampler", "optimizer"):
        if field not in protocol:
            errors.append(f"candidate_protocol_missing:{field}")
    exposure = protocol.get("optimizer_exposure")
    if not isinstance(exposure, dict) or not exposure:
        errors.append("candidate_protocol_missing:optimizer_exposure")

    sf_model = sf.get("model") if isinstance(sf.get("model"), dict) else {}
    a_model = (
        attention.get("model")
        if isinstance(attention.get("model"), dict)
        else {}
    )
    model_diffs = _diff_paths(sf_model, a_model)
    unexpected = sorted(set(model_diffs) - ALLOWED_MODEL_DIFFS)
    if unexpected:
        errors.append(f"uncontrolled_model_differences={unexpected}")
    required_diffs = {"selected_slot_indices", "temporal_encoder_name"}
    if not required_diffs.issubset(model_diffs):
        errors.append("declared_temporal_finalist_difference_missing")
    _audit_model_arm("SF128", sf_model, errors)
    _audit_model_arm("A128", a_model, errors)
    warnings.append(
        "sf128_a128_is_composite_temporal_selection_not_attention_ablation"
    )
    return {
        "valid": not errors,
        "arms": list(REQUIRED_ARMS),
        "changed_scientific_family": "temporal_base_finalist",
        "changed_model_fields": model_diffs,
        "attention_mechanism_claim_allowed": False,
        "errors": errors,
        "warnings": warnings,
    }


def _audit_arm_identity(
    name: str,
    arm: dict[str, Any],
    expected_role: str,
    errors: list[str],
) -> None:
    if arm.get("candidate_id") != name:
        errors.append(f"candidate_id_mismatch:{name}")
    if arm.get("role") != expected_role:
        errors.append(f"candidate_role_mismatch:{name}")


def _audit_model_arm(
    name: str,
    model: dict[str, Any],
    errors: list[str],
) -> None:
    common = {
        "architecture": "cached_frame_feature_temporal_classifier_v1",
        "backbone_name": "resnet18",
        "input_resolution": 224,
        "hidden_dim": 128,
        "temporal_view": FIXED_VIEW,
    }
    for field, expected in common.items():
        if model.get(field) != expected:
            errors.append(f"candidate_model_mismatch:{name}:{field}")
    expected = {
        "SF128": {
            "selected_slot_indices": [2],
            "temporal_encoder_name": "masked_mean",
        },
        "A128": {
            "selected_slot_indices": [0, 1, 2, 3, 4, 5],
            "temporal_encoder_name": "masked_attention",
        },
    }[name]
    for field, value in expected.items():
        if model.get(field) != value:
            errors.append(f"candidate_model_mismatch:{name}:{field}")


def _audit_comparison_universe(
    native_path: Path | None,
    fold_path: Path | None,
    view_path: Path | None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    native = _read_csv(native_path, errors, "native_units")
    folds = _read_csv(fold_path, errors, "fold_manifest")
    view = _read_csv(view_path, errors, "fixed6_view")
    native_required = {
        "temporal_unit_key",
        "source_type",
        "behavior_label",
        "native_unit_valid_for_main_eval",
    }
    fold_required = {"temporal_unit_key", "outer_fold_id", "oof_role"}
    view_required = {
        "temporal_view_name",
        "view_item_id",
        "temporal_unit_key",
        "source_type",
        "slot_index",
    }
    if not _require_columns(native, native_required, errors, "native_units"):
        return _invalid_universe(errors, warnings)
    if not _require_columns(folds, fold_required, errors, "fold_manifest"):
        return _invalid_universe(errors, warnings)
    if not _require_columns(view, view_required, errors, "fixed6_view"):
        return _invalid_universe(errors, warnings)

    eligible = native.loc[
        native["native_unit_valid_for_main_eval"].map(_bool_scalar)
    ].copy()
    if eligible.empty:
        errors.append("zero_eligible_native_units")
    for label, frame in (("native", native), ("eligible_native", eligible)):
        duplicate = int(frame["temporal_unit_key"].duplicated(keep=False).sum())
        if duplicate:
            errors.append(f"duplicate_{label}_unit_keys={duplicate}")
    fold_duplicates = int(
        folds["temporal_unit_key"].duplicated(keep=False).sum()
    )
    if fold_duplicates:
        errors.append(f"duplicate_fold_unit_keys={fold_duplicates}")

    source_labels = sorted(eligible["source_type"].astype(str).unique())
    if source_labels != list(REQUIRED_SOURCES):
        errors.append(f"mixed_comparison_requires_two_sources={source_labels}")
    unknown_labels = sorted(
        set(eligible["behavior_label"].astype(str)) - set(DEFAULT_LABEL_ORDER)
    )
    if unknown_labels:
        errors.append(f"unknown_behavior_labels={unknown_labels}")

    joined = eligible.merge(
        folds[["temporal_unit_key", "outer_fold_id", "oof_role"]],
        on="temporal_unit_key",
        how="left",
        validate="one_to_one",
    )
    missing_fold = int(joined["outer_fold_id"].isna().sum())
    if missing_fold:
        errors.append(f"eligible_native_units_without_fold={missing_fold}")
    if joined["outer_fold_id"].nunique(dropna=True) < 2:
        errors.append("grouped_comparison_requires_multiple_folds")

    view_names = sorted(view["temporal_view_name"].astype(str).unique())
    if view_names != [FIXED_VIEW]:
        errors.append(f"fixed6_view_name_mismatch={view_names}")
    duplicate_slots = int(
        view.duplicated(["view_item_id", "slot_index"], keep=False).sum()
    )
    if duplicate_slots:
        errors.append(f"duplicate_fixed6_slots={duplicate_slots}")
    item_sizes = view.groupby("view_item_id", sort=False).size()
    bad_sizes = int(item_sizes.ne(6).sum())
    if bad_sizes:
        errors.append(f"fixed6_items_without_six_slots={bad_sizes}")
    bad_slot_order = 0
    metadata_conflicts = 0
    for _, group in view.groupby("view_item_id", sort=False):
        slots = sorted(pd.to_numeric(group["slot_index"], errors="coerce"))
        if slots != [0, 1, 2, 3, 4, 5]:
            bad_slot_order += 1
        if (
            group["temporal_unit_key"].astype(str).nunique() != 1
            or group["source_type"].astype(str).nunique() != 1
        ):
            metadata_conflicts += 1
    if bad_slot_order:
        errors.append(f"fixed6_slot_order_invalid_items={bad_slot_order}")
    if metadata_conflicts:
        errors.append(f"fixed6_item_metadata_conflicts={metadata_conflicts}")

    native_keys = set(native["temporal_unit_key"].astype(str))
    view_keys = set(view["temporal_unit_key"].astype(str))
    unknown_view_keys = sorted(view_keys - native_keys)
    if unknown_view_keys:
        errors.append(f"fixed6_view_unknown_native_units={len(unknown_view_keys)}")
    eligible_keys = set(eligible["temporal_unit_key"].astype(str))
    missing_view = eligible_keys - view_keys
    if missing_view:
        errors.append(f"eligible_native_units_without_fixed6_view={len(missing_view)}")

    availability = [column for column in AVAILABILITY_COLUMNS if column in view]
    if not availability:
        errors.append("fixed6_view_missing_availability_masks")
    missingness = _missingness_support(view, availability)
    class_source = _support_table(
        joined,
        ["source_type", "behavior_label"],
    )
    class_fold = _support_table(
        joined.dropna(subset=["outer_fold_id"]),
        ["outer_fold_id", "behavior_label"],
    )
    for source in REQUIRED_SOURCES:
        supported = set(
            eligible.loc[
                eligible["source_type"].astype(str).eq(source),
                "behavior_label",
            ].astype(str)
        )
        missing = sorted(set(DEFAULT_LABEL_ORDER) - supported)
        if missing:
            warnings.append(f"classes_missing_source_support:{source}={missing}")
    return {
        "valid": not errors,
        "eligible_native_units": int(len(eligible)),
        "fixed6_view_items": int(view["view_item_id"].nunique()),
        "fixed6_slot_rows": int(len(view)),
        "source_labels": source_labels,
        "fold_count": int(joined["outer_fold_id"].nunique(dropna=True)),
        "class_by_source_support": class_source,
        "class_by_fold_support": class_fold,
        "missingness_support": missingness,
        "availability_mask_columns": availability,
        "errors": errors,
        "warnings": warnings,
    }


def _audit_shortcut_contract(path: Path | None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    audit = _read_json(path, errors, "temporal_view_audit")
    if audit.get("schema_version") != "classification_v2_temporal_shortcut_audit_v1":
        errors.append("temporal_view_audit_schema_mismatch")
    if audit.get("valid") is not True or audit.get("errors") != []:
        errors.append("temporal_view_audit_invalid")
    if audit.get("training_stop_required") is not False:
        errors.append("temporal_shortcut_training_stop_required")
    if audit.get("source_metadata_in_model_inputs") is not False:
        errors.append("source_metadata_present_in_model_inputs")
    view = audit.get("view_reports", {}).get(FIXED_VIEW, {})
    if view.get("source_counts") is None or len(view.get("source_counts", {})) != 2:
        errors.append("fixed6_shortcut_audit_requires_two_sources")
    mitigated = set(audit.get("mitigated_families", []))
    families = view.get("families", {})
    for family in ("length", "padding", "availability"):
        report = families.get(family)
        if not isinstance(report, dict):
            errors.append(f"fixed6_shortcut_family_missing:{family}")
            continue
        if report.get("near_direct_source_signature") is not True:
            continue
        key = f"{FIXED_VIEW}:{family}"
        if family in mitigated or key in mitigated:
            warnings.append(f"declared_mitigated_source_shortcut={key}")
        else:
            errors.append(f"unmitigated_fixed6_source_shortcut={family}")
    label_reports = audit.get("label_shortcut_reports", {})
    if not isinstance(label_reports, dict):
        errors.append("label_shortcut_reports_missing")
    else:
        for name, report in label_reports.items():
            if isinstance(report, dict) and report.get(
                "near_direct_target_signature"
            ):
                errors.append(f"audit_metadata_nearly_determines_behavior={name}")
    return {
        "valid": not errors,
        "fixed6_source_counts": view.get("source_counts", {}),
        "availability": families.get("availability", {}),
        "errors": errors,
        "warnings": warnings,
    }


def _audit_inference_inputs(
    whitelist_path: Path | None,
    model_input_path: Path | None,
    data_contract: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    whitelist = _read_json(whitelist_path, errors, "feature_whitelist")
    model_input = _read_json(model_input_path, errors, "model_input_contract")
    features = whitelist.get("features")
    features = features if isinstance(features, list) else []
    patterns = data_contract.get("forbidden_x_patterns")
    pattern_values = patterns if isinstance(patterns, list) else None
    feature_audit = validate_model_input_columns(
        [str(value) for value in features],
        forbidden_patterns=pattern_values,
    )
    if whitelist.get("schema_version") != "classification_v2.feature_whitelist.v1":
        errors.append("feature_whitelist_schema_mismatch")
    if feature_audit.get("valid") is not True:
        errors.append(
            f"feature_whitelist_not_inference_safe="
            f"{feature_audit.get('forbidden_columns')}"
        )
    if model_input.get("errors") != []:
        errors.append("model_input_contract_has_errors")
    temporal = model_input.get("temporal_contract", {})
    target = model_input.get("target_contract", {})
    inference = model_input.get("inference_contract", {})
    if temporal.get("primary_view") != FIXED_VIEW:
        errors.append("model_input_primary_temporal_view_mismatch")
    if temporal.get("windows_after_harmonization") is not True:
        errors.append("model_input_windows_must_follow_harmonization")
    if target.get("allowed_behaviors") != list(DEFAULT_LABEL_ORDER):
        errors.append("model_input_label_order_mismatch")
    if target.get("final_head_directly_supervised") is not True:
        errors.append("final_behavior_head_must_be_directly_supervised")
    expected_inference = {
        "ground_truth_only_fields_allowed": False,
        "review_fields_allowed": False,
        "missing_modalities_require_masks": True,
        "partner_selection_may_use_target_behavior": False,
    }
    for field, expected in expected_inference.items():
        if inference.get(field) != expected:
            errors.append(f"inference_contract_mismatch:{field}")
    return {
        "valid": not errors,
        "feature_count": len(features),
        "forbidden_features": feature_audit.get("forbidden_columns", []),
        "errors": errors,
    }


def _audit_namespace(
    contract: dict[str, Any],
    comparison_path: Path | None,
    handoff_path: Path | None,
    output_json: Path | None,
    root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    agent_root, human_root = _lineage_roots(contract, root, errors)
    if agent_root is not None:
        for label, path in (("comparison", comparison_path), ("handoff", handoff_path)):
            if path is None or not _is_under(path, agent_root):
                errors.append(f"{label}_outside_agent_derived_root")
        expected_handoff = agent_root / "review_handoff"
        if handoff_path is None or not _is_under(handoff_path, expected_handoff):
            errors.append("handoff_outside_agent_review_handoff_root")
        if output_json is not None:
            try:
                output_path = _project_path(output_json, root, "output")
            except ValueError as exc:
                errors.append(f"output_path_invalid={exc}")
            else:
                if not _is_under(output_path, agent_root):
                    errors.append("output_outside_agent_derived_root")
                if human_root is not None and _is_under(output_path, human_root):
                    errors.append("output_inside_human_review_root")
    return {
        "valid": not errors,
        "agent_derived_root": _relative(agent_root, root),
        "human_review_root": _relative(human_root, root),
        "errors": errors,
    }


def _lineage_roots(
    contract: dict[str, Any],
    root: Path,
    errors: list[str],
) -> tuple[Path | None, Path | None]:
    roots = contract.get("lineage_roots")
    roots = roots if isinstance(roots, dict) else {}
    resolved: list[Path | None] = []
    for label in ("agent_derived", "human_review"):
        value = roots.get(label)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"missing_{label}_root")
            resolved.append(None)
            continue
        try:
            resolved.append(_project_path(Path(value), root, label))
        except ValueError as exc:
            errors.append(f"invalid_{label}_root={exc}")
            resolved.append(None)
    return resolved[0], resolved[1]


def _artifact_paths(
    contract: dict[str, Any],
    root: Path,
    errors: list[str],
) -> dict[str, Path]:
    artifacts = contract.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, dict) else {}
    required = set(ARTIFACT_BINDINGS.values())
    paths: dict[str, Path] = {}
    for name in sorted(required):
        spec = artifacts.get(name)
        value = spec.get("path") if isinstance(spec, dict) else None
        if not isinstance(value, str) or not value.strip():
            errors.append(f"comparison_artifact_path_missing:{name}")
            continue
        try:
            path = _project_path(Path(value), root, f"artifact:{name}")
        except ValueError as exc:
            errors.append(f"comparison_artifact_path_invalid:{name}:{exc}")
            continue
        if not path.is_file():
            errors.append(f"comparison_artifact_missing:{name}")
        paths[name] = path
    return paths


def _actual_bindings(
    contract_path: Path | None,
    snapshot_path: Path | None,
    artifacts: dict[str, Path],
) -> dict[str, str | None]:
    bindings: dict[str, str | None] = {
        "data_contract_sha256": _optional_sha256(contract_path),
        "training_snapshot_sha256": _optional_sha256(snapshot_path),
    }
    for field, artifact in ARTIFACT_BINDINGS.items():
        bindings[field] = _optional_sha256(artifacts.get(artifact))
    return bindings


def _support_table(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    grouped = frame.groupby(columns, dropna=False).size().rename("native_unit_count")
    out = grouped.reset_index().sort_values(columns, kind="mergesort")
    return out.to_dict(orient="records")


def _missingness_support(
    view: pd.DataFrame,
    availability: list[str],
) -> list[dict[str, Any]]:
    if not availability:
        return []
    rows: list[dict[str, Any]] = []
    for item_id, group in view.groupby("view_item_id", sort=False):
        signature_parts = []
        for column in availability:
            available_count = int(group[column].map(_bool_scalar).sum())
            signature_parts.append(f"{column}:{available_count}/6")
        rows.append(
            {
                "view_item_id": str(item_id),
                "source_type": str(group["source_type"].iloc[0]),
                "availability_signature": "|".join(signature_parts),
            }
        )
    frame = pd.DataFrame(rows)
    grouped = (
        frame.groupby(["source_type", "availability_signature"])
        .size()
        .rename("view_item_count")
        .reset_index()
        .sort_values(["source_type", "availability_signature"], kind="mergesort")
    )
    return grouped.to_dict(orient="records")


def _invalid_universe(
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "valid": False,
        "eligible_native_units": 0,
        "fixed6_view_items": 0,
        "source_labels": [],
        "class_by_source_support": [],
        "class_by_fold_support": [],
        "missingness_support": [],
        "errors": errors,
        "warnings": warnings,
    }


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    errors: list[str],
    label: str,
) -> bool:
    missing = sorted(required - set(frame.columns))
    if missing:
        errors.append(f"{label}_missing_columns={missing}")
    return not missing


def _diff_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_diff_paths(left.get(key), right.get(key), child))
        return paths
    return [] if left == right else [prefix or "<root>"]


def _read_json(
    path: Path | None,
    errors: list[str],
    label: str,
) -> dict[str, Any]:
    if path is None or not path.is_file():
        errors.append(f"missing_json:{label}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid_json:{label}:{exc}")
        return {}
    if not isinstance(payload, dict):
        errors.append(f"json_must_be_object:{label}")
        return {}
    return payload


def _read_csv(
    path: Path | None,
    errors: list[str],
    label: str,
) -> pd.DataFrame:
    if path is None or not path.is_file():
        errors.append(f"missing_csv:{label}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        errors.append(f"invalid_csv:{label}:{exc}")
        return pd.DataFrame()


def _input_path(
    value: Path,
    root: Path,
    errors: list[str],
    label: str,
) -> Path | None:
    try:
        path = _project_path(value, root, label)
    except ValueError as exc:
        errors.append(f"{label}_path_invalid={exc}")
        return None
    if not path.is_file():
        errors.append(f"missing_{label}={path}")
        return None
    return path


def _project_path(value: Path, root: Path, label: str) -> Path:
    candidate = Path(value)
    if ".." in candidate.parts:
        raise ValueError(f"{label} contains parent traversal")
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not _is_under(path, root):
        raise ValueError(f"{label} is outside project root")
    return path


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _relative(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return None


def _bool_scalar(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(FILE_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return _sha256_file(path)


def _stable_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


__all__ = [
    "CONTRACT_SCHEMA_VERSION",
    "PREFLIGHT_SCHEMA_VERSION",
    "build_reviewed_q2_mixed_finalist_preflight",
    "write_reviewed_q2_mixed_finalist_preflight",
]
