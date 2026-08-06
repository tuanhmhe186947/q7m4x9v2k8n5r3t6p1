"""Focused next-phase contract tests.

Changed invariants covered here:

* A12-B keeps unsupported duplicate/interval edges INCONCLUSIVE.
* posture authority cannot enter S1 without a snapshot-compatible binding.
* E0 cannot be ready when its exact inner fold is absent.
* the S1 decision remains inner-only and keeps C2 blocked.

The smallest inputs are copied JSON dictionaries; the produced authority is
the next-phase validator decision, not a model result.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "classification_v2"
    / "09_final_release_audit"
    / "validate_next_phase.py"
)


def _load_validator():
    spec = importlib.util.spec_from_file_location("next_phase_validator", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_validator()


def _proof() -> dict:
    statuses = {
        "construction_source_overlap": "PASS",
        "exact_duplicate_isolation": "PASS",
        "near_duplicate_isolation": "INCONCLUSIVE",
        "exact_temporal_interval_isolation": "INCONCLUSIVE",
        "native_unit_isolation": "PASS",
        "video_group_isolation": "PASS",
        "recording_date_group_isolation": "PASS",
        "window_role_inheritance": "PASS",
    }
    return {
        "decision": "INCONCLUSIVE",
        "checks": [
            {"id": key, "status": value, "evidence": ["synthetic"]}
            for key, value in statuses.items()
        ],
    }


def _posture() -> dict:
    return {
        "status": "INCONCLUSIVE",
        "class_order": ["lying", "sitting", "upright"],
        "review_reopened": False,
        "candidate_behavior_to_posture": {
            "lying": "lying",
            "sitting": "sitting",
            "stand": "upright",
            "eat": "upright",
        },
        "included_in_s1": False,
        "missing_machine_readable_items": ["snapshot_binding"],
    }


def _e0() -> dict:
    return {
        "paid_execution_authorization": "NO",
        "e0_status": "NOT_EXECUTED",
        "outer_test_access": "BLOCKED",
        "registered_inner_fold": None,
        "ready_to_launch_e0": False,
        "blocker_code": "E0_CONFIG_INCOMPLETE",
        "outer_access_negative_test": {"status": "PASS"},
        "outer_access_policy": {
            "data_mount": False,
            "labels": False,
            "metrics": False,
            "predictions": False,
            "errors": False,
            "confusion_matrices": False,
            "registered_outer_resources": [],
        },
    }


def _s1() -> dict:
    return {
        "A12_A_STATUS": "PASS",
        "A12_B_STATUS": "INCONCLUSIVE",
        "E0_STATUS": "NOT_EXECUTED",
        "POSTURE_AUTHORITY_STATUS": "INCONCLUSIVE",
        "POSTURE_INCLUDED_IN_S1": False,
        "OUTER_TEST_ISOLATION_STATUS": "PASS",
        "READY_FOR_PAID_INNER_AUTORESEARCH_S1": "NO",
        "S1_PERMIT_STATUS": "BLOCKED",
        "READY_FOR_CLAIM_GRADE_OUTER_OOF_C2": "NO",
        "BLOCKERS": ["A12-B"],
        "NEXT_AUTHORIZED_ACTION": "Resolve the bounded provenance edge.",
    }


def test_a12b_rejects_a_false_pass_for_unresolved_edges() -> None:
    errors: list[str] = []
    broken = copy.deepcopy(_proof())
    broken["decision"] = "PASS"

    VALIDATOR.validate_a12b_proof(broken, errors)

    assert any("must remain INCONCLUSIVE" in error for error in errors)


def test_posture_inconclusive_is_excluded_from_s1() -> None:
    errors: list[str] = []
    broken = copy.deepcopy(_posture())
    broken["included_in_s1"] = True

    VALIDATOR.validate_posture_binding(broken, errors)

    assert any("entered S1" in error for error in errors)


def test_e0_requires_an_exact_inner_fold() -> None:
    errors: list[str] = []
    broken = copy.deepcopy(_e0())
    broken["ready_to_launch_e0"] = True

    VALIDATOR.validate_e0_preflight(broken, errors)

    assert any("exact inner fold" in error for error in errors)


def test_outer_access_negative_policy_rejects_a_permitted_metric() -> None:
    errors: list[str] = []
    policy = copy.deepcopy(_e0()["outer_access_policy"])
    policy["metrics"] = True

    VALIDATOR.validate_outer_access_policy(policy, errors)

    assert any("permits metrics" in error for error in errors)


def test_s1_readiness_keeps_c2_blocked() -> None:
    errors: list[str] = []
    broken = copy.deepcopy(_s1())
    broken["READY_FOR_CLAIM_GRADE_OUTER_OOF_C2"] = "YES"

    VALIDATOR.validate_s1_readiness(broken, errors)

    assert any("C2" in error for error in errors)
