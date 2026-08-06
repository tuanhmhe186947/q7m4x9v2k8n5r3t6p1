"""Focused next-phase contract tests.

Changed invariants covered here:

* A12-B uses the source-video construction contract as the hard gate and
  leaves unregistered near-duplicate checks NOT_APPLICABLE.
* posture authority cannot enter S1 without a snapshot-compatible binding.
* E0 cannot be ready when its exact inner fold is absent.
* the S1 decision remains inner-only and keeps C2 blocked.
* extensionless current CVAT display keys bind to the same canonical source key.

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
HANDOFF_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "classification_v2"
    / "09_final_release_audit"
    / "prepare_execution_handoff.py"
)


def _load_validator():
    spec = importlib.util.spec_from_file_location("next_phase_validator", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_validator()


def _load_handoff():
    spec = importlib.util.spec_from_file_location("execution_handoff", HANDOFF_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HANDOFF = _load_handoff()


def _proof() -> dict:
    statuses = {
        "construction_source_overlap": "PASS",
        "exact_duplicate_isolation": "PASS",
        "near_duplicate_isolation": "NOT_APPLICABLE",
        "exact_temporal_interval_isolation": "PASS",
        "native_unit_isolation": "PASS",
        "video_group_isolation": "PASS",
        "recording_date_group_isolation": "PASS",
        "window_role_inheritance": "PASS",
        "direct_predictive_source_leakage": "PASS",
    }
    return {
        "decision": "PASS",
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
        "registered_inner_fold": "FOLD_3",
        "preflight_status": "PASS",
        "ready_to_launch_e0": False,
        "blocker_code": "PAID_EXECUTION_NOT_AUTHORIZED",
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
        "A12_B_STATUS": "PASS",
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


def test_a12b_accepts_the_registered_source_video_contract() -> None:
    errors: list[str] = []
    proof = copy.deepcopy(_proof())

    VALIDATOR.validate_a12b_proof(proof, errors)

    assert errors == []


def test_current_extensionless_cvat_key_uses_source_video_canonicalization() -> None:
    assert HANDOFF.canonical_video_key("Pigs281119_000085_30fps") == (
        "pigs281119/000085"
    )
    assert HANDOFF.canonical_video_key("Pigs281119_000085_30fps.mp4") == (
        "pigs281119/000085"
    )


def test_posture_inconclusive_is_excluded_from_s1() -> None:
    errors: list[str] = []
    broken = copy.deepcopy(_posture())
    broken["included_in_s1"] = True

    VALIDATOR.validate_posture_binding(broken, errors)

    assert any("entered S1" in error for error in errors)


def test_e0_requires_an_exact_inner_fold() -> None:
    errors: list[str] = []
    broken = copy.deepcopy(_e0())
    broken["registered_inner_fold"] = None

    VALIDATOR.validate_e0_preflight(broken, errors)

    assert any("lacks an exact inner fold" in error for error in errors)


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
