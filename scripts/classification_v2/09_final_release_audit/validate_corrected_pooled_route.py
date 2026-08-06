"""Validate the bounded corrected Classification V2 route package.

This validator checks route semantics and package structure only. It never
loads training data, starts a model, accesses an outer fold, or changes a
scientific authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_FILES = (
    "authority_binding.json",
    "plan_model_traceability_matrix.json",
    "reuse_ledger.json",
    "experiment_funnel.json",
    "matched_ablation_registry.json",
    "temporal_view_screening_manifest.json",
    "feature_family_registry.json",
    "imbalance_strategy_registry.json",
    "justified_extension_registry.json",
    "posture_ablation_contract.json",
    "candidate_model_family_registry.json",
    "autoresearch_search_space.json",
    "permit_policy.json",
    "outer_oof_contract.json",
    "dependency_invalidation_graph.json",
    "compute_cost_estimate.json",
    "delta_test_plan.json",
    "readiness_decision.json",
    "scientific_question_amendment.md",
    "a12_supersession_notice.md",
    "thesis_mapping.md",
)

ALLOWED_REUSE = {
    "REUSE_AS_CURRENT_PASS",
    "REUSE_AS_SCIENTIFIC_COMPONENT",
    "REUSE_AS_ENGINEERING_EVIDENCE",
    "REUSE_AS_DIAGNOSTIC_ONLY",
    "RERUN_REQUIRED_AFTER_DEPENDENCY_CHANGE",
    "STALE_FOR_CURRENT_AUTHORITY",
    "SUPERSEDED_GATE_DESIGN",
    "MISSING_OR_UNVERIFIABLE",
}

ALLOWED_IMPLEMENTATION = {
    "COMPLETE",
    "PARTIAL",
    "DECLARED_EXTENSION_POINT",
    "NOT_IMPLEMENTED",
    "SUPERSEDED",
    "NOT_REQUIRED",
}

ALLOWED_EXECUTION = {
    "NOT_RUN",
    "ENGINEERING_SMOKE_ONLY",
    "BOUNDED_DIAGNOSTIC",
    "INNER_SCREENING_INCOMPLETE",
    "INNER_SCREENING_COMPLETE",
    "CLAIM_GRADE_COMPLETE",
}

ALLOWED_EVIDENCE = {
    "NONE",
    "ENGINEERING_ONLY",
    "DIAGNOSTIC_ONLY",
    "CURRENT_INNER_EVIDENCE",
    "CURRENT_CLAIM_GRADE_EVIDENCE",
    "STALE_FOR_CURRENT_AUTHORITY",
    "SUPERSEDED_QUESTION",
}

REQUIRED_PLAN_ITEMS = {
    "authority and reviewed snapshot",
    "loader tensor masks causality leakage",
    "B0-B3 baseline ladder",
    "imbalance-loss study",
    "ROI and causal history",
    "social relation development",
    "availability and quality-aware fusion",
    "integrated retained-module BALANCED model",
    "temporal-view experiments",
    "post-review reproduction",
    "B0",
    "B1",
    "B2",
    "B3",
    "L0-L7 loss families",
    "T6",
    "T8",
    "T12",
    "T16",
    "S6@16",
    "geometry",
    "motion",
    "ROI",
    "social",
    "full 46D",
    "causal history and two-timescale history",
    "ROI concat and ROI-conditioned FiLM",
    "nearest-partner, Top-K, and GAT",
    "concat, availability-aware, and quality-gated fusion",
    "posture auxiliary head",
    "integrated finalist",
    "outer OOF and calibration",
}

REQUIRED_REUSE_ITEMS = {
    "pooled_reviewed_dataset",
    "hidden_review_completion",
    "behavior_review_completion",
    "corrected_source_apply",
    "temporal_harmonization",
    "T6_builder",
    "T8_builder",
    "T12_builder",
    "T16_builder",
    "S6@16_support",
    "exact_view_feature_recomputation",
    "canonical_46D_schema",
    "validity_masks",
    "availability_masks",
    "predictive_whitelist",
    "grouped_folds",
    "overlap_removal_evidence",
    "event_weights",
    "zero_weight_filtering",
    "B0_B3_implementations",
    "B0_B3_pilots",
    "imbalance_infrastructure",
    "ROI_infrastructure",
    "history_infrastructure",
    "social_infrastructure",
    "fusion_infrastructure",
    "checkpoint_resume",
    "compute_profile",
    "VRAM_profile",
    "prediction_exporter",
    "native_unit_collapse",
    "bounded_OOF",
    "bounded_calibration",
    "source_probes",
    "old_A12_plus_0_038",
    "current_A12_plus_0_005264",
    "S0_no_social",
    "S1_social_10D",
    "S2_topK",
    "S3_GAT_closure",
    "posture_authority_and_implementation",
    "independent_checkers",
    "protected_authority_audit",
    "paid_GPU_packaging",
    "old_final_readiness_gate",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _error(errors: list[str], message: str) -> None:
    errors.append(message)


def validate_forbidden_input_fields(fields: list[str]) -> list[str]:
    forbidden = (
        "source",
        "dataset",
        "video",
        "path",
        "review",
        "reviewer",
        "decision",
        "partition",
        "fold",
        "pig_id",
        "track_id",
        "object_track_key",
        "behavior_label",
        "posture_label",
        "target",
        "future",
    )
    return [
        field
        for field in fields
        if any(token in field.lower() for token in forbidden)
    ]


def validate_plan_matrix(matrix: dict[str, Any], errors: list[str]) -> int:
    items = matrix.get("items")
    if not isinstance(items, list):
        _error(errors, "plan matrix items must be a list")
        return 0
    seen = {item.get("ORIGINAL_ITEM") for item in items if isinstance(item, dict)}
    missing = sorted(REQUIRED_PLAN_ITEMS - seen)
    if missing:
        _error(errors, f"plan matrix missing items: {missing}")
    required_fields = {
        "ORIGINAL_PHASE",
        "ORIGINAL_ITEM",
        "ORIGINAL_HYPOTHESIS",
        "ORIGINAL_PLAN_FILE",
        "ORIGINAL_PLAN_SECTION",
        "CURRENT_CODE_PATH",
        "CURRENT_CONFIG_PATH",
        "CURRENT_ARTIFACT_PATH",
        "CURRENT_CODE_SHA",
        "CURRENT_DEPENDENCY_HASHES",
        "IMPLEMENTATION_STATUS",
        "EXECUTION_STATUS",
        "SCIENTIFIC_EVIDENCE_STATUS",
        "AUTHORITY_COMPATIBILITY",
        "CURRENT_SCIENTIFIC_RELEVANCE",
        "REUSE_DECISION",
        "MINIMUM_DELTA_REQUIRED",
        "SUPERSEDED_SEMANTICS",
        "DEPENDENT_NEXT_STAGE",
    }
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            _error(errors, f"plan matrix item {index} is not an object")
            continue
        missing_fields = sorted(required_fields - set(item))
        if missing_fields:
            _error(errors, f"plan matrix item {index} missing {missing_fields}")
        if item.get("IMPLEMENTATION_STATUS") not in ALLOWED_IMPLEMENTATION:
            _error(errors, f"invalid implementation status at plan item {index}")
        if item.get("EXECUTION_STATUS") not in ALLOWED_EXECUTION:
            _error(errors, f"invalid execution status at plan item {index}")
        if item.get("SCIENTIFIC_EVIDENCE_STATUS") not in ALLOWED_EVIDENCE:
            _error(errors, f"invalid evidence status at plan item {index}")
        if item.get("REUSE_DECISION") not in {
            "REUSE_WITHOUT_RERUN",
            "REUSE_AS_ENGINEERING_EVIDENCE",
            "REUSE_AS_DIAGNOSTIC_HYPOTHESIS",
            "RERUN_ONLY_UNDER_CURRENT_AUTHORITY",
            "IMPLEMENT_MINIMUM_MISSING_COMPONENT",
            "DROP_FROM_PRIMARY_FUNNEL",
            "SUPERSEDED_BY_CURRENT_PLAN",
        }:
            _error(errors, f"invalid reuse decision at plan item {index}")
    return len(items)


def validate_reuse_ledger(ledger: dict[str, Any], errors: list[str]) -> int:
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        _error(errors, "reuse ledger entries must be a list")
        return 0
    names = [entry.get("item") for entry in entries if isinstance(entry, dict)]
    missing = sorted(REQUIRED_REUSE_ITEMS - set(names))
    if missing:
        _error(errors, f"reuse ledger missing items: {missing}")
    if len(names) != len(set(names)):
        _error(errors, "reuse ledger contains duplicate item names")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            _error(errors, f"reuse ledger item {index} is not an object")
            continue
        if entry.get("reuse_status") not in ALLOWED_REUSE:
            _error(errors, f"invalid reuse status at ledger item {index}")
        if not entry.get("exact_local_paths"):
            _error(errors, f"missing exact local path at ledger item {index}")
        if "artifact_hashes" not in entry:
            _error(errors, f"missing artifact hash field at ledger item {index}")
        if entry.get("rerun_required") and not entry.get("rerun_spec"):
            _error(errors, f"rerun specification missing at ledger item {index}")
    return len(entries)


def validate_temporal_views(manifest: dict[str, Any], errors: list[str]) -> None:
    expected = {"T6", "T8", "T12", "T16", "S6@16"}
    views = manifest.get("views", [])
    ids = {view.get("id") for view in views if isinstance(view, dict)}
    if ids != expected:
        _error(errors, f"temporal view set mismatch: {sorted(ids)}")
    for view in views:
        if not isinstance(view, dict):
            _error(errors, "temporal view entry is not an object")
            continue
        if not view.get("feature_recompute_required"):
            _error(errors, f"view lacks exact-view recomputation: {view.get('id')}")
        if not view.get("feature_binding_key"):
            _error(errors, f"view lacks feature binding: {view.get('id')}")
        offsets = view.get("offsets_from_endpoint", view.get("offsets_from_native_start", []))
        if view.get("id") != "S6@16" and any(int(offset) > 0 for offset in offsets):
            _error(errors, f"future offset in pooled view: {view.get('id')}")
        if view.get("id") == "S6@16" and view.get("scope") != "legacy-only":
            _error(errors, "S6@16 is not legacy-only")
        if view.get("id") == "S6@16" and not view.get("diagnostic_only"):
            _error(errors, "S6@16 is not diagnostic-only")


def validate_feature_registry(registry: dict[str, Any], errors: list[str]) -> None:
    families = registry.get("families", [])
    family_ids = {family.get("id") for family in families if isinstance(family, dict)}
    if family_ids != {"F0", "F1", "F2", "F3", "F4", "F5"}:
        _error(errors, f"feature family set mismatch: {sorted(family_ids)}")
    for family in families:
        if not isinstance(family, dict):
            _error(errors, "feature family entry is not an object")
            continue
        forbidden = validate_forbidden_input_fields(family.get("inputs", []))
        if forbidden:
            _error(errors, f"forbidden feature-family inputs for {family.get('id')}: {forbidden}")
        if not family.get("producer"):
            _error(errors, f"missing producer for {family.get('id')}")
    if registry.get("exact_view_contract", {}).get("feature_cache_reuse_across_views"):
        _error(errors, "feature cache reuse across views is enabled")


def validate_search_space(search: dict[str, Any], errors: list[str]) -> None:
    budget = search.get("trial_budget", {})
    for key in (
        "max_trials",
        "max_gpu_hours",
        "max_monetary_cost_usd",
        "per_trial_timeout_hours",
        "failed_run_ceiling",
        "disk_ceiling_gb",
    ):
        value = budget.get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            _error(errors, f"search-space budget is not finite and positive: {key}")
    if search.get("execution_mode") != "inner_only":
        _error(errors, "autoresearch execution mode is not inner_only")
    if search.get("outer_test_access", {}).get("technical_enforcement_required") is not True:
        _error(errors, "outer-test technical enforcement is not required")
    if search.get("promotion", {}).get("automatic") is not False:
        _error(errors, "autoresearch automatic promotion is enabled")
    excluded = set(search.get("excluded_dimensions", []))
    if not {"outer_fold_membership", "outer_metrics", "outer_predictions"}.issubset(excluded):
        _error(errors, "outer-test dimensions are not excluded from search")


def validate_outer_isolation(
    search: dict[str, Any], outer: dict[str, Any], errors: list[str]
) -> None:
    access = search.get("outer_test_access", {})
    access_keys = ("data_mount", "labels", "metrics", "predictions", "errors")
    if any(access.get(key) is not False for key in access_keys):
        _error(errors, "search package exposes an outer-test resource")
    nested = outer.get("nested_selection", {})
    nested_keys = (
        "outer_test_visible_during_selection",
        "outer_test_metrics_visible_during_selection",
        "outer_test_predictions_visible_during_selection",
    )
    for key in nested_keys:
        if nested.get(key) is not False:
            _error(errors, f"outer contract exposes {key}")
    if outer.get("native_unit_contract", {}).get("silent_row_or_unit_drop") is not False:
        _error(errors, "outer contract permits silent unit drop")


def validate_permits(permits: dict[str, Any], errors: list[str]) -> None:
    by_id = {
        permit.get("id"): permit
        for permit in permits.get("permits", [])
        if isinstance(permit, dict)
    }
    for required in ("E0", "S1", "C2"):
        if required not in by_id:
            _error(errors, f"missing permit {required}")
    e0 = by_id.get("E0", {})
    if not str(e0.get("status", "")).startswith("READY"):
        _error(errors, "E0 is not ready")
    if e0.get("execution", {}).get("outer_test_mount") is not False:
        _error(errors, "E0 permits outer-test mount")
    if e0.get("budget", {}).get("max_cost_usd", 0) <= 0:
        _error(errors, "E0 has no positive cost cap")
    for permit_id in ("S1", "C2"):
        if by_id.get(permit_id, {}).get("status") != "BLOCKED":
            _error(errors, f"{permit_id} is not fail-closed")
    if not permits.get("single_use_permit_rule"):
        _error(errors, "permit reuse is not prohibited")


def calculate_run_counts(funnel: dict[str, Any]) -> dict[str, int]:
    accounting = funnel["run_accounting"]
    return {
        "stage_1_to_5": int(accounting["new_stage_1_to_5_max"]),
        "s1": int(accounting["new_s1_trials_max"]),
        "c2": int(accounting["new_c2_training_runs"]),
        "e0": int(accounting["new_e0_runs"]),
        "total": int(accounting["new_total_max_including_e0_s1_c2"]),
    }


def calculate_cost(estimate: dict[str, Any]) -> dict[str, float]:
    rate = float(estimate["provider_assumption"]["gpu_rate_usd_per_hour"])
    stages = estimate["stages"]
    p50_hours = sum(float(stage["p50_gpu_hours"]) for stage in stages)
    p90_hours = sum(float(stage["p90_gpu_hours"]) for stage in stages)
    fraction = float(estimate["totals"]["planning_contingency_fraction"])
    return {
        "p50_gpu_hours": p50_hours,
        "p90_gpu_hours": p90_hours,
        "p50_cost_usd_before_contingency": p50_hours * rate,
        "p90_cost_usd_before_contingency": p90_hours * rate,
        "p50_estimated_cost_usd": p50_hours * rate * (1 + fraction),
        "p90_estimated_cost_usd": p90_hours * rate * (1 + fraction),
    }


def validate_readiness(readiness: dict[str, Any], errors: list[str]) -> None:
    required = {
        "SCIENTIFIC_QUESTION_CORRECTED": "YES",
        "OLD_SOURCE_COMPARISON_OBJECTIVE_REMOVED": "YES",
        "POOLED_REVIEWED_DATA_AUTHORITY": "PASS",
        "A12_DIRECT_SOURCE_LEAKAGE_SAFETY": "PASS",
        "A12_OVERLAP_AND_GROUPING_INTEGRITY": "INCONCLUSIVE",
        "A12_SOURCE_BALANCED_GAIN_GATE": "REMOVED",
        "READY_FOR_PAID_GPU_E0": "YES",
        "READY_FOR_PAID_INNER_AUTORESEARCH_S1": "NO",
        "READY_FOR_CLAIM_GRADE_OUTER_OOF_C2": "NO",
        "PAPER_GRADE_RESULT_AVAILABLE": "NO",
    }
    for key, value in required.items():
        if readiness.get(key) != value:
            _error(errors, f"readiness mismatch {key}: expected {value}, got {readiness.get(key)}")
    if (
        not isinstance(readiness.get("NEXT_AUTHORIZED_ACTION"), str)
        or not readiness["NEXT_AUTHORIZED_ACTION"].strip()
    ):
        _error(errors, "readiness has no exact next action")
    if len(readiness.get("BLOCKERS", [])) == 0:
        _error(errors, "readiness has no blockers while S1/C2 are blocked")


def validate_package(package_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    missing_files = [name for name in REQUIRED_FILES if not (package_dir / name).exists()]
    if missing_files:
        _error(errors, f"missing package files: {missing_files}")
    payloads: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_FILES:
        path = package_dir / name
        if path.suffix == ".json" and path.exists():
            try:
                payloads[name] = load_json(path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                _error(errors, f"cannot load {name}: {exc}")
    if payloads:
        authority = payloads.get("authority_binding.json", {})
        expected_code_sha = "884016aff7d7f23608adcc81a6c138a46351c57e"
        if authority.get("classification_v2_code_sha") != expected_code_sha:
            _error(errors, "classification code authority does not match")
        expected_snapshot_sha = (
            "ab86e2e04267cfdc8248f9bdb8774615479d67a3589f7a25844bb1a4c93a639e"
        )
        if authority.get("reviewed_snapshot_sha256") != expected_snapshot_sha:
            _error(errors, "reviewed snapshot hash does not match")
        matrix_count = validate_plan_matrix(
            payloads.get("plan_model_traceability_matrix.json", {}), errors
        )
        ledger_count = validate_reuse_ledger(payloads.get("reuse_ledger.json", {}), errors)
        validate_temporal_views(payloads.get("temporal_view_screening_manifest.json", {}), errors)
        validate_feature_registry(payloads.get("feature_family_registry.json", {}), errors)
        validate_search_space(payloads.get("autoresearch_search_space.json", {}), errors)
        validate_outer_isolation(
            payloads.get("autoresearch_search_space.json", {}),
            payloads.get("outer_oof_contract.json", {}),
            errors,
        )
        validate_permits(payloads.get("permit_policy.json", {}), errors)
        validate_readiness(payloads.get("readiness_decision.json", {}), errors)
        funnel = payloads.get("experiment_funnel.json", {})
        counts = calculate_run_counts(funnel)
        if counts != {"stage_1_to_5": 47, "s1": 24, "c2": 12, "e0": 1, "total": 84}:
            _error(errors, f"run count mismatch: {counts}")
        estimate = payloads.get("compute_cost_estimate.json", {})
        calculated_cost = calculate_cost(estimate)
        totals = estimate.get("totals", {})
        for key, expected in calculated_cost.items():
            recorded_key = key
            if abs(float(totals.get(recorded_key, -1)) - expected) > 1e-6:
                _error(
                    errors,
                    f"cost mismatch {recorded_key}: {totals.get(recorded_key)} != {expected}",
                )
        reusable_items = matrix_count + ledger_count - ledger_count
        if (
            payloads.get("readiness_decision.json", {}).get(
                "PLAN_MODEL_REUSABLE_ITEMS"
            )
            != reusable_items
        ):
            warnings.append(
                "PLAN_MODEL_REUSABLE_ITEMS is a declared count and should be "
                "reconciled after matrix review"
            )
    return {
        "schema_version": "classification_v2.corrected_pooled_route.validator_report.v1",
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "package_dir": str(package_dir),
        "required_file_count": len(REQUIRED_FILES),
        "loaded_json_count": len(payloads),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = validate_package(args.package_dir)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["valid"] or not args.check else 1


if __name__ == "__main__":
    raise SystemExit(main())
