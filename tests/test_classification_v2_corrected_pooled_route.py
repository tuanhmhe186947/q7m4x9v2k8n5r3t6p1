from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PROJECT_ROOT / "docs" / "classification_v2" / "corrected_pooled_route_20260806"
VALIDATOR_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "classification_v2"
    / "09_final_release_audit"
    / "validate_corrected_pooled_route.py"
)


def _load_validator():
    spec = importlib.util.spec_from_file_location("corrected_route_validator", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_validator()


def _load(name: str) -> dict:
    return json.loads((PACKAGE_DIR / name).read_text(encoding="utf-8"))


def test_current_route_package_passes_structural_validation() -> None:
    report = VALIDATOR.validate_package(PACKAGE_DIR)

    assert report["valid"], report["errors"]
    assert report["required_file_count"] == 21


def test_forbidden_source_review_and_target_fields_are_rejected() -> None:
    fields = [
        "geometry_6D",
        "source_type",
        "reviewer_id",
        "file_path",
        "behavior_label",
        "future_motion",
    ]

    assert set(VALIDATOR.validate_forbidden_input_fields(fields)) == {
        "source_type",
        "reviewer_id",
        "file_path",
        "behavior_label",
        "future_motion",
    }


def test_temporal_manifest_requires_exact_view_feature_recomputation() -> None:
    manifest = _load("temporal_view_screening_manifest.json")
    broken = copy.deepcopy(manifest)
    broken["views"][0]["feature_recompute_required"] = False
    errors: list[str] = []

    VALIDATOR.validate_temporal_views(broken, errors)

    assert any("exact-view recomputation" in error for error in errors)


def test_s6_is_legacy_diagnostic_and_not_a_pooled_target() -> None:
    manifest = _load("temporal_view_screening_manifest.json")
    s6 = next(view for view in manifest["views"] if view["id"] == "S6@16")

    assert s6["scope"] == "legacy-only"
    assert s6["diagnostic_only"] is True
    assert s6["primary_candidate"] is False


def test_feature_family_isolation_rejects_forbidden_model_input() -> None:
    registry = _load("feature_family_registry.json")
    broken = copy.deepcopy(registry)
    broken["families"][0]["inputs"].append("source_type")
    errors: list[str] = []

    VALIDATOR.validate_feature_registry(broken, errors)

    assert any("forbidden feature-family inputs" in error for error in errors)


def test_search_space_rejects_outer_test_access() -> None:
    search = _load("autoresearch_search_space.json")
    broken = copy.deepcopy(search)
    broken["outer_test_access"]["metrics"] = True
    errors: list[str] = []

    VALIDATOR.validate_search_space(broken, errors)
    VALIDATOR.validate_outer_isolation(broken, _load("outer_oof_contract.json"), errors)

    assert any("outer-test" in error for error in errors)


def test_permit_dependencies_keep_s1_and_c2_fail_closed() -> None:
    permits = _load("permit_policy.json")
    errors: list[str] = []

    VALIDATOR.validate_permits(permits, errors)

    assert errors == []
    statuses = {permit["id"]: permit["status"] for permit in permits["permits"]}
    assert statuses == {
        "E0": "READY_TO_ISSUE_AFTER_PACKAGE_VALIDATION",
        "S1": "BLOCKED",
        "C2": "BLOCKED",
    }


def test_run_count_calculation_excludes_reused_runs() -> None:
    counts = VALIDATOR.calculate_run_counts(_load("experiment_funnel.json"))

    assert counts == {"stage_1_to_5": 47, "s1": 24, "c2": 12, "e0": 1, "total": 84}


def test_cost_calculation_matches_recorded_p50_and_p90() -> None:
    estimate = _load("compute_cost_estimate.json")
    calculated = VALIDATOR.calculate_cost(estimate)

    assert calculated["p50_gpu_hours"] == pytest.approx(75.0)
    assert calculated["p90_gpu_hours"] == pytest.approx(151.0)
    assert calculated["p50_estimated_cost_usd"] == pytest.approx(225.0)
    assert calculated["p90_estimated_cost_usd"] == pytest.approx(453.0)


def test_readiness_has_one_next_action_and_no_paper_result() -> None:
    readiness = _load("readiness_decision.json")
    errors: list[str] = []

    VALIDATOR.validate_readiness(readiness, errors)

    assert errors == []
    assert readiness["READY_FOR_PAID_GPU_E0"] == "YES"
    assert readiness["READY_FOR_CLAIM_GRADE_OUTER_OOF_C2"] == "NO"
    assert readiness["PAPER_GRADE_RESULT_AVAILABLE"] == "NO"
    assert readiness["NEXT_AUTHORIZED_ACTION"].count("E0") == 1
