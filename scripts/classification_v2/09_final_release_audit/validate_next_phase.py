"""Validate the bounded post-readiness Classification V2 handoff.

The checks in this module are deliberately package-local. They validate
provenance decisions, posture binding, E0 permit completeness, and S1
fail-closed semantics without loading model data or starting remote work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


A12_CHECK_STATUSES = {"PASS", "FAIL", "NOT_APPLICABLE", "INCONCLUSIVE"}
POSTURE_STATUSES = {"PASS", "INCONCLUSIVE", "MISSING", "FAIL"}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_a12b_proof(proof: dict[str, Any], errors: list[str]) -> None:
    required_ids = {
        "construction_source_overlap",
        "exact_duplicate_isolation",
        "near_duplicate_isolation",
        "exact_temporal_interval_isolation",
        "native_unit_isolation",
        "video_group_isolation",
        "recording_date_group_isolation",
        "window_role_inheritance",
    }
    checks = proof.get("checks")
    if not isinstance(checks, list):
        errors.append("A12-B checks must be a list")
        return
    seen = set()
    for check in checks:
        if not isinstance(check, dict):
            errors.append("A12-B check is not an object")
            continue
        check_id = check.get("id")
        seen.add(check_id)
        if check.get("status") not in A12_CHECK_STATUSES:
            errors.append(f"invalid A12-B status for {check_id}")
        if not check.get("evidence"):
            errors.append(f"missing A12-B evidence for {check_id}")
    missing = sorted(required_ids - seen)
    if missing:
        errors.append(f"missing A12-B checks: {missing}")
    decision = proof.get("decision")
    if decision not in {"PASS", "INCONCLUSIVE"}:
        errors.append(f"invalid A12-B decision: {decision!r}")
    statuses = {check.get("id"): check.get("status") for check in checks}
    if decision == "PASS":
        required_pass = {
            "construction_source_overlap",
            "exact_duplicate_isolation",
            "exact_temporal_interval_isolation",
            "native_unit_isolation",
            "video_group_isolation",
            "recording_date_group_isolation",
            "window_role_inheritance",
            "direct_predictive_source_leakage",
        }
        for check_id in sorted(required_pass):
            if statuses.get(check_id) != "PASS":
                errors.append(f"A12-B PASS requires {check_id}=PASS")
        if statuses.get("near_duplicate_isolation") != "NOT_APPLICABLE":
            errors.append("A12-B PASS requires near_duplicate_isolation=NOT_APPLICABLE")
    elif not {
        "near_duplicate_isolation",
        "exact_temporal_interval_isolation",
    }.issubset(
        {
            check.get("id")
            for check in checks
            if isinstance(check, dict)
            and check.get("status") == "INCONCLUSIVE"
        }
    ):
        errors.append("A12-B INCONCLUSIVE proof must preserve unresolved content edges")


def validate_posture_binding(binding: dict[str, Any], errors: list[str]) -> None:
    if binding.get("status") not in POSTURE_STATUSES:
        errors.append("invalid posture authority status")
    if binding.get("class_order") != ["lying", "sitting", "upright"]:
        errors.append("posture class order is not the current machine order")
    if binding.get("review_reopened") is not False:
        errors.append("posture review was reopened")
    mapping = binding.get("candidate_behavior_to_posture", {})
    allowed = {
        "lying": "lying",
        "sitting": "sitting",
        "stand": "upright",
        "eat": "upright",
    }
    if mapping != allowed:
        errors.append("posture candidate mapping is not the bounded safe mapping")
    if binding.get("status") != "PASS" and binding.get("included_in_s1") is not False:
        errors.append("unresolved posture authority entered S1")
    if not binding.get("missing_machine_readable_items"):
        errors.append("posture binding has no finite missing-item list")


def validate_e0_preflight(decision: dict[str, Any], errors: list[str]) -> None:
    if decision.get("paid_execution_authorization") != "NO":
        errors.append("paid execution authorization is not NO")
    if decision.get("e0_status") != "NOT_EXECUTED":
        errors.append("E0 must be NOT_EXECUTED in this phase")
    if decision.get("outer_test_access") != "BLOCKED":
        errors.append("E0 outer-test access is not blocked")
    fold = decision.get("registered_inner_fold")
    if not fold:
        errors.append("E0 preflight lacks an exact inner fold")
    if decision.get("preflight_status") != "PASS":
        errors.append("E0 technical preflight is not PASS")
    if decision.get("ready_to_launch_e0") is True and decision.get(
        "paid_execution_authorization"
    ) != "YES":
        errors.append("E0 cannot launch without paid authorization")
    negative_test = decision.get("outer_access_negative_test", {})
    if negative_test.get("status") != "PASS":
        errors.append("local E0 outer-access negative test did not pass")
    validate_outer_access_policy(decision.get("outer_access_policy", {}), errors)


def validate_outer_access_policy(policy: dict[str, Any], errors: list[str]) -> None:
    access_keys = (
        "data_mount",
        "labels",
        "metrics",
        "predictions",
        "errors",
        "confusion_matrices",
    )
    for key in access_keys:
        if policy.get(key) is not False:
            errors.append(f"E0 outer access policy permits {key}")
    if policy.get("registered_outer_resources") not in ([], None):
        errors.append("E0 outer resources are registered in the input package")


def validate_s1_readiness(decision: dict[str, Any], errors: list[str]) -> None:
    required = {
        "A12_A_STATUS": "PASS",
        "A12_B_STATUS": "PASS",
        "E0_STATUS": "NOT_EXECUTED",
        "POSTURE_AUTHORITY_STATUS": "INCONCLUSIVE",
        "POSTURE_INCLUDED_IN_S1": False,
        "OUTER_TEST_ISOLATION_STATUS": "PASS",
        "READY_FOR_PAID_INNER_AUTORESEARCH_S1": "NO",
        "S1_PERMIT_STATUS": "BLOCKED",
    }
    for key, expected in required.items():
        if decision.get(key) != expected:
            errors.append(f"S1 readiness mismatch {key}: {decision.get(key)!r}")
    if decision.get("READY_FOR_CLAIM_GRADE_OUTER_OOF_C2") != "NO":
        errors.append("C2 was authorized by the S1 decision")
    if not isinstance(decision.get("BLOCKERS"), list) or not decision["BLOCKERS"]:
        errors.append("S1 readiness has no finite blockers")
    if not isinstance(decision.get("NEXT_AUTHORIZED_ACTION"), str):
        errors.append("S1 readiness has no exact next action")


def validate_artifact_hashes(payloads: dict[str, dict[str, Any]], errors: list[str]) -> None:
    bindings: dict[str, str] = {}
    proof = payloads.get("revised_a12b_construction_overlap_proof.json", {})
    for check in proof.get("checks", []):
        for evidence in check.get("evidence", []):
            if isinstance(evidence, dict) and evidence.get("path"):
                bindings[str(evidence["path"])] = str(evidence.get("sha256", ""))
    posture = payloads.get("posture_authority_binding.json", {}).get(
        "observed_artifact", {}
    )
    if posture.get("path"):
        bindings[str(posture["path"])] = str(posture.get("sha256", ""))
    route = payloads.get("authority_recheck.json", {}).get("route_validator_pre_change", {})
    if route.get("path"):
        bindings[str(route["path"])] = str(route.get("sha256", ""))
    for raw_path, expected in bindings.items():
        path = Path(raw_path)
        if not path.exists():
            errors.append(f"bound artifact is missing: {raw_path}")
            continue
        if not expected or sha256_file(path).lower() != expected.lower():
            errors.append(f"bound artifact hash mismatch: {raw_path}")


def validate_next_phase(next_phase_dir: Path, verify_hashes: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    required = (
        "authority_recheck.json",
        "revised_a12b_construction_overlap_proof.json",
        "posture_authority_binding.json",
        "e0_preflight_decision.json",
        "s1_readiness_decision.json",
    )
    payloads: dict[str, dict[str, Any]] = {}
    for name in required:
        path = next_phase_dir / name
        if not path.exists():
            errors.append(f"missing next-phase artifact: {name}")
            continue
        try:
            payloads[name] = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"cannot load {name}: {exc}")
    if payloads:
        validate_a12b_proof(
            payloads.get("revised_a12b_construction_overlap_proof.json", {}),
            errors,
        )
        validate_posture_binding(payloads.get("posture_authority_binding.json", {}), errors)
        validate_e0_preflight(payloads.get("e0_preflight_decision.json", {}), errors)
        validate_s1_readiness(payloads.get("s1_readiness_decision.json", {}), errors)
        if verify_hashes:
            validate_artifact_hashes(payloads, errors)
    return {
        "schema_version": "classification_v2.next_phase.validator_report.v1",
        "valid": not errors,
        "errors": errors,
        "next_phase_dir": str(next_phase_dir),
        "required_json_count": len(required),
        "loaded_json_count": len(payloads),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--next-phase-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    report = validate_next_phase(args.next_phase_dir, verify_hashes=args.verify_hashes)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["valid"] or not args.check else 1


if __name__ == "__main__":
    raise SystemExit(main())
