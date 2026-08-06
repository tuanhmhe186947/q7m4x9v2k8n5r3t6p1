"""Write the small, versioned post-readiness execution decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


CLASSIFICATION_CODE_SHA = "884016aff7d7f23608adcc81a6c138a46351c57e"
CLASSIFICATION_TREE_HASH = "5dfaa841ee5c0162a77807ceaf44983f3f174ae6"
SNAPSHOT_ID = "reviewed_engineering_amendment_992f34c0204a85a1"
SNAPSHOT_SHA256 = "ab86e2e04267cfdc8248f9bdb8774615479d67a3589f7a25844bb1a4c93a639e"
SPLIT_HASH = "557156a7eb6cceeb6a91f667f7c51dcb286e3111f35f414970fa7431acc7e63b"
EVENT_WEIGHT_HASH = "92a901b8bb431102f5e32fd73c899930f5f3f4c83a9eac6945f9609cdd84938d"
SCHEMA_HASH = "18377d825ba84974e49305e46561ada81353f9ffd0f2d2526471af1c199daad4"
ENVIRONMENT_HASH = "6b783d5296094e0be94b0e553e3c83376a462eec3278285b076b35761bc103ca"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path: Path, role: str) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "semantic_role": role,
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, encoding="utf-8"
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--posture-binding", type=Path, required=True)
    parser.add_argument("--queue-audit", type=Path, required=True)
    parser.add_argument("--session-manifest", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    output = args.output_dir.resolve()
    package = args.package_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    authority = load_json(output / "cvat_legacy_duplicate_removal_authority.json")
    proof = load_json(output / "revised_a12b_construction_overlap_proof.json")
    eligibility = load_json(output / "eligibility_reconciliation.json")
    e0_handoff = load_json(output / "e0_l4_handoff.json")
    old_posture = load_json(args.posture_binding)
    queue_audit = load_json(args.queue_audit)
    session = load_json(args.session_manifest)
    permit_policy = load_json(package / "permit_policy.json")
    search_space_path = package / "autoresearch_search_space.json"
    search_space = load_json(search_space_path)

    session_root = Path(session["session_path"])
    queue_path = Path(session["queue_path"])
    decisions_path = Path(session["decisions_path"])
    actual_session_manifest = session_root / "session_manifest.json"
    queue_binding = binding(queue_path, "open posture-review queue")
    decisions_binding = binding(decisions_path, "live posture decision ledger")
    session_binding = binding(actual_session_manifest, "open posture session manifest")

    posture = dict(old_posture)
    posture.update(
        {
            "schema_version": "classification_v2.next_phase.posture_authority_binding.v2",
            "status": "INCONCLUSIVE",
            "included_in_s1": False,
            "review_reopened": False,
            "campaign_status": "HUMAN_REVIEW_IN_PROGRESS",
            "current_queue": {
                "queue_audit": binding(args.queue_audit, "500-item posture queue audit"),
                "session_manifest": session_binding,
                "queue": queue_binding,
                "decisions": decisions_binding,
                "target_rows": queue_audit["target_rows"],
                "queue_sha256": queue_audit["queue_sha256"],
                "initial_prefilled_decisions": session["prefilled_human_decisions"],
                "human_review_pending": True,
            },
        }
    )
    posture["missing_machine_readable_items"] = sorted(
        set(posture.get("missing_machine_readable_items", []))
        | {"closed reviewed posture authority for the current snapshot"}
    )
    write_json(output / "posture_authority_binding.json", posture)

    matched_contract = load_json(
        args.posture_binding.parent / "posture_matched_ablation_contract.json"
    )
    matched_contract.update(
        {
            "schema_version": "classification_v2.next_phase.posture_matched_ablation_contract.v2",
            "status": "BLOCKED_UNTIL_POSTURE_AUTHORITY_PASS",
            "executable": False,
            "posture_included_in_s1": False,
            "review_reopened": False,
            "current_queue_binding": queue_binding,
        }
    )
    write_json(output / "posture_matched_ablation_contract.json", matched_contract)
    support_path = args.posture_binding.parent / "posture_support_by_group.csv"
    if support_path.exists():
        shutil.copyfile(support_path, output / "posture_support_by_group.csv")

    route_report = package / "next_phase_20260806" / "next_phase_validator_report.json"
    current_head = git_value(repo, "rev-parse", "HEAD")
    authority_recheck = {
        "schema_version": "classification_v2.next_phase.authority_recheck.v2",
        "decision_date": "2026-08-06",
        "execution_worktree": str(repo),
        "execution_branch": git_value(repo, "branch", "--show-current"),
        "base_main_sha_measured": git_value(repo, "rev-parse", "main"),
        "execution_head_sha_measured": current_head,
        "worktree_status_at_generation": "EXPECTED_NEXT_PHASE_CHANGES",
        "current_authorities": {
            "classification_code_sha": CLASSIFICATION_CODE_SHA,
            "classification_tree_hash": CLASSIFICATION_TREE_HASH,
            "reviewed_snapshot_id": SNAPSHOT_ID,
            "reviewed_snapshot_sha256": SNAPSHOT_SHA256,
            "split_hash": SPLIT_HASH,
            "event_weight_hash": EVENT_WEIGHT_HASH,
            "canonical_46d_schema_hash": SCHEMA_HASH,
            "environment_lock_hash": ENVIRONMENT_HASH,
        },
        "protected_state": {
            "data_rebuild": False,
            "hidden_review_reopened": False,
            "behavior_review_reopened": False,
            "outer_split_changed": False,
            "paid_compute_started": False,
            "s1_started": False,
            "c2_started": False,
        },
        "route_validator_pre_change": binding(
            route_report, "committed corrected-route validator report"
        ),
        "replacement_artifacts": {
            "construction_authority": binding(
                output / "cvat_legacy_duplicate_removal_authority.json",
                "replacement construction authority",
            ),
            "filtered_burst_binding": binding(
                output / "cvat_legacy_filtered_burst_binding.json",
                "replacement filtered legacy binding",
            ),
            "a12b_proof": binding(
                output / "revised_a12b_construction_overlap_proof.json",
                "replacement A12-B proof",
            ),
            "eligibility_reconciliation": binding(
                output / "eligibility_reconciliation.json",
                "eligibility reconciliation only",
            ),
            "e0_l4_handoff": binding(
                output / "e0_l4_handoff.json", "non-executed L4 handoff"
            ),
        },
        "scope_reconciliation": {
            "construction_overlap": "PASS",
            "a12_b": "PASS",
            "old_inconclusive_a12_b": "SUPERSEDED_BY_DIRECT_CONSTRUCTION_LINEAGE",
            "old_e0_missing_fold": "SUPERSEDED_BY_FOLD_3_BINDING",
            "posture": "INCONCLUSIVE_HUMAN_REVIEW_IN_PROGRESS",
        },
        "no_data_rebuild": True,
    }
    write_json(output / "authority_recheck.json", authority_recheck)

    e0_descriptor = e0_handoff["descriptor"]
    e0_preflight = {
        "schema_version": "classification_v2.next_phase.e0_preflight_decision.v2",
        "decision_date": "2026-08-06",
        "registered_model": e0_descriptor["model"],
        "temporal_view": e0_descriptor["temporal_view"],
        "modalities": e0_descriptor["modalities"],
        "seed": e0_descriptor["seed"],
        "registered_inner_fold": "FOLD_3",
        "fold_selection": {
            "order_source": "frozen split manifest canonical outer-fold order",
            "selected_first_valid_inner_development_fold": "FOLD_3",
            "selection_used_metrics": False,
            "split_unchanged": True,
        },
        "preflight_status": "PASS",
        "e0_status": "NOT_EXECUTED",
        "paid_execution_authorization": "NO",
        "ready_to_launch_e0": False,
        "blocker_code": "PAID_EXECUTION_NOT_AUTHORIZED",
        "blockers": ["PAID_EXECUTION_AUTHORIZATION=NO; no remote process launched."],
        "route_hashes": {
            "classification_code_sha": CLASSIFICATION_CODE_SHA,
            "classification_tree_hash": CLASSIFICATION_TREE_HASH,
            "snapshot_sha256": SNAPSHOT_SHA256,
            "split_hash": SPLIT_HASH,
            "event_weight_hash": EVENT_WEIGHT_HASH,
            "schema_hash": SCHEMA_HASH,
            "environment_lock_hash": ENVIRONMENT_HASH,
        },
        "input_package": e0_handoff["inputs"],
        "package_descriptor_sha256": e0_handoff["descriptor_sha256"],
        "checks": {
            "exact_inner_fold": "PASS",
            "t6_exact_view_feature_binding": "PASS",
            "geometry_and_motion_schema": "PASS",
            "predictive_whitelist": "PASS",
            "masks_and_zero_weight_filtering": "PASS",
            "checkpoint_contract": "PASS_REUSED_ENGINEERING_EVIDENCE",
            "prediction_exporter": "PASS_REUSED_ENGINEERING_EVIDENCE",
            "native_unit_metric_path": "PASS_REUSED_ENGINEERING_EVIDENCE",
            "outer_access_negative_test": "PASS",
            "remote_package_upload": "NOT_EXECUTED",
        },
        "outer_test_access": "BLOCKED",
        "outer_access_negative_test": {
            "status": "PASS",
            "process_started": False,
            "scope": "local E0 package policy and path-resolution negative control",
            "result": (
                "outer data, labels, metrics, predictions, errors, and confusion "
                "matrices are denied"
            ),
        },
        "outer_access_policy": e0_handoff["outer_test_exclusion"],
        "output_contract": {
            "launch_command": e0_handoff["launch_command"],
            "checkpoint_path": e0_handoff["checkpoint_path"],
            "resume_command": e0_handoff["resume_command"],
            "prediction_export": e0_handoff["prediction_export"],
            "download_manifest": e0_handoff["download_manifest"],
        },
        "limits": e0_handoff["provider"],
        "no_model_execution": True,
    }
    write_json(output / "e0_preflight_decision.json", e0_preflight)

    s1_permit = next(item for item in permit_policy["permits"] if item["id"] == "S1")
    budget_hash = hashlib.sha256(
        json.dumps(
            s1_permit["budget"], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    s1 = {
        "schema_version": "classification_v2.next_phase.s1_readiness_decision.v2",
        "decision_date": "2026-08-06",
        "A12_A_STATUS": "PASS",
        "A12_B_STATUS": proof["decision"],
        "E0_STATUS": "NOT_EXECUTED",
        "POSTURE_AUTHORITY_STATUS": "INCONCLUSIVE",
        "POSTURE_INCLUDED_IN_S1": False,
        "OUTER_TEST_ISOLATION_STATUS": "PASS",
        "SEARCH_SPACE_HASH": sha256_file(search_space_path),
        "BUDGET_HASH": budget_hash,
        "search_space_status": search_space["status"],
        "S1_PERMIT_STATUS": "BLOCKED",
        "READY_FOR_PAID_INNER_AUTORESEARCH_S1": "NO",
        "READY_FOR_CLAIM_GRADE_OUTER_OOF_C2": "NO",
        "PAPER_GRADE_RESULT_AVAILABLE": "NO",
        "POSTURE_CAMPAIGN_STATUS": "HUMAN_REVIEW_IN_PROGRESS",
        "POSTURE_MATCHED_ABLATION_EXECUTABLE": False,
        "BLOCKERS": [
            "E0_STATUS=NOT_EXECUTED; paid authorization is NO.",
            "POSTURE_AUTHORITY=INCONCLUSIVE; posture supervision is excluded from S1.",
        ],
        "no_auto_promotion": True,
        "outer_test_access": "BLOCKED",
        "NEXT_AUTHORIZED_ACTION": "Human review of the opened 500-item posture session.",
    }
    write_json(output / "s1_readiness_decision.json", s1)

    superseded = [
        {
            "old_path": str(args.posture_binding.parent / name),
            "status": "SUPERSEDED",
            "reason": reason,
            "replacement": binding(output / replacement, reason),
            "scope_of_invalidation": scope,
        }
        for name, reason, replacement, scope in [
            (
                "a12b_construction_overlap_proof.json",
                "direct check_duplicate_videos.py lineage resolved the construction question",
                "revised_a12b_construction_overlap_proof.json",
                "old A12-B inconclusive construction decision only",
            ),
            (
                "a12b_broken_provenance_edge.json",
                "the previously unresolved construction edge is now directly bound",
                "cvat_legacy_duplicate_removal_authority.json",
                "old construction-provenance blocker only",
            ),
            (
                "e0_preflight_decision.json",
                "the exact registered inner fold is now FOLD_3",
                "e0_preflight_decision.json",
                "old missing-fold E0 readiness status only",
            ),
            (
                "s1_readiness_decision.json",
                "A12-B and the E0 fold status were revised; posture remains inconclusive",
                "s1_readiness_decision.json",
                "old S1 gate inputs only; no scientific result invalidated",
            ),
        ]
    ]
    supersession_notice = {
        "schema_version": "classification_v2.next_phase.supersession_notice.v1",
        "status": "SUPERSEDED_AUTHORITIES_EXPLICITLY_BOUND",
        "replacement_directory": str(output),
        "superseded": superseded,
        "replacement_hash": sha256_file(output / "s1_readiness_decision.json"),
        "no_data_rebuild": True,
    }
    write_json(
        args.posture_binding.parent / "supersession_notice_20260806_r2.json",
        supersession_notice,
    )
    write_json(output / "supersession_notice.json", supersession_notice)

    summary = {
        "schema_version": "classification_v2.next_phase.execution_handoff_summary.v2",
        "current_head": current_head,
        "construction_decision": authority["decision"],
        "a12b_decision": proof["decision"],
        "eligibility_changed_native_units": eligibility["eligibility_changed_unit_count"],
        "e0_preflight": e0_preflight["preflight_status"],
        "e0_status": e0_preflight["e0_status"],
        "s1_status": s1["S1_PERMIT_STATUS"],
        "posture_status": posture["status"],
        "posture_campaign_status": posture["campaign_status"],
        "no_training_or_paid_execution": True,
    }
    write_json(output / "execution_handoff_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
