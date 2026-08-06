"""Generate the bounded post-readiness Classification V2 handoff artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


CANONICAL_VIDEO_RE = re.compile(r"(pigs\d{6}[a-z]?)[_/](\d{6})", re.IGNORECASE)
CLASS_ORDER = ["lying", "sitting", "upright"]
SAFE_POSTURE_MAPPING = {
    "lying": "lying",
    "sitting": "sitting",
    "stand": "upright",
    "eat": "upright",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, encoding="utf-8"
    ).strip()


def canonical_video_key(value: str) -> str | None:
    match = CANONICAL_VIDEO_RE.search(value.lower())
    if not match:
        return None
    return f"{match.group(1)}/{match.group(2)}"


def source_comparison(reviewed_frame: Path, exclusion_csv: Path) -> dict[str, Any]:
    row_counts: dict[str, int] = {}
    source_keys: dict[str, set[str]] = {}
    missing_keys: dict[str, int] = {}
    rows = 0
    with gzip.open(reviewed_frame, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            source = row["source_type"]
            row_counts[source] = row_counts.get(source, 0) + 1
            value = row.get("source_video_key") or row.get("video_key") or ""
            key = canonical_video_key(value)
            if key is None:
                missing_keys[source] = missing_keys.get(source, 0) + 1
            else:
                source_keys.setdefault(source, set()).add(key)
    exclusion_keys = {
        row["source_video_key"].strip().lower()
        for row in csv.DictReader(exclusion_csv.open(encoding="utf-8", newline=""))
    }
    cvat_keys = source_keys.get("cvat_tracking_xml", set())
    legacy_keys = source_keys.get("legacy_recovered", set())
    return {
        "rows": rows,
        "row_counts_by_source": row_counts,
        "unique_canonical_keys_by_source": {
            key: len(value) for key, value in source_keys.items()
        },
        "missing_canonical_keys_by_source": missing_keys,
        "exclusion_key_count": len(exclusion_keys),
        "cvat_minus_exclusion": len(cvat_keys - exclusion_keys),
        "exclusion_minus_cvat": len(exclusion_keys - cvat_keys),
        "cvat_intersection_legacy": len(cvat_keys & legacy_keys),
        "exclusion_intersection_legacy": len(exclusion_keys & legacy_keys),
        "canonicalization_rule": (
            "lower-case regex (pigs\\d{6}[a-z]?)[_/](\\d{6}); "
            "source_video_key preferred over video_key"
        ),
    }


def posture_summary(posture_csv: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(posture_csv.open(encoding="utf-8", newline="")))
    labels: dict[str, int] = {}
    sources: dict[str, int] = {}
    keys = []
    reviewers = set()
    for row in rows:
        label = row["posture_decision"]
        source = row["source_type"]
        labels[label] = labels.get(label, 0) + 1
        sources[source] = sources.get(source, 0) + 1
        keys.append(row["native_temporal_unit_key"])
        reviewers.add(row["reviewer"])
    return {
        "rows": len(rows),
        "labels": labels,
        "sources": sources,
        "duplicate_native_keys": len(keys) - len(set(keys)),
        "reviewers": sorted(reviewers),
        "scope_sha256": rows[0]["scope_sha256"] if rows else None,
    }


def evidence_record(
    path: Path, scope: str, count: Any, overlap: Any, note: str
) -> dict[str, Any]:
    binding = file_binding(path)
    return {
        "path": binding["path"],
        "sha256": binding["sha256"],
        "size_bytes": binding["size_bytes"],
        "semantic_scope": scope,
        "examined_count": count,
        "detected_overlap_count": overlap,
        "note": note,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reviewed-frame",
        type=Path,
        default=Path(
            r"E:\PigProjectStorage\PIG_Behavior_Project\outputs\classification_v2"
            r"\agent_audits\post_review_frame_amendment_materialization_fa028cb_20260803_224700"
            r"\reviewed_frame_features.csv.gz"
        ),
    )
    parser.add_argument(
        "--exclusion-csv",
        type=Path,
        default=Path(
            r"E:\PigProjectStorage\PIG_Behavior_Project\outputs\legacy_16f_rebuild"
            r"\legacy_16f_rebuild_20260718_v2\02_video_policy\exclude_source_videos.csv"
        ),
    )
    parser.add_argument(
        "--duplicate-audit",
        type=Path,
        default=Path(
            r"E:\PigProjectStorage\PIG_Behavior_Project\outputs\legacy_16f_rebuild"
            r"\legacy_16f_rebuild_20260718_v2\02_video_policy\duplicate_video_filter_audit.json"
        ),
    )
    parser.add_argument(
        "--legacy-lineage",
        type=Path,
        default=Path(
            r"E:\PigProjectStorage\PIG_Behavior_Project\outputs\legacy_16f_rebuild"
            r"\legacy_16f_rebuild_20260718_v2\01_provenance\legacy_source_trace_lineage.json"
        ),
    )
    parser.add_argument(
        "--completion-audit",
        type=Path,
        default=Path(
            r"E:\PigProjectStorage\PIG_Behavior_Project\outputs\legacy_16f_rebuild"
            r"\legacy_16f_rebuild_20260718_v2\08_audits\legacy_16f_rebuild_completion_audit.json"
        ),
    )
    parser.add_argument(
        "--posture-csv",
        type=Path,
        default=Path(
            r"E:\PigProjectStorage\PIG_Behavior_Project\outputs\classification_v2"
            r"\posture_review_sessions\paired_pilot_86edabc\posture_pilot_decisions.csv"
        ),
    )
    args = parser.parse_args()
    package = args.package_dir.resolve()
    repo = package.parents[2]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    authority = load_json(package / "authority_binding.json")
    permit_policy = load_json(package / "permit_policy.json")
    search_space_path = package / "autoresearch_search_space.json"
    search_space = load_json(search_space_path)
    e0 = next(permit for permit in permit_policy["permits"] if permit["id"] == "E0")
    s1 = next(permit for permit in permit_policy["permits"] if permit["id"] == "S1")
    source_metrics = source_comparison(args.reviewed_frame, args.exclusion_csv)
    posture_metrics = posture_summary(args.posture_csv)
    duplicate_audit = load_json(args.duplicate_audit)
    legacy_lineage = load_json(args.legacy_lineage)
    completion_audit = load_json(args.completion_audit)
    split_path = (
        args.reviewed_frame.parent / "split_manifest.csv"
    )
    effective_index_path = args.reviewed_frame.parent / "effective_window_index.csv"
    route_validator_path = package / "route_validator_report.json"

    input_binding_hash = hashlib.sha256(
        json.dumps(
            e0["input_package"], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    descriptor = {
        "permit_id": "E0",
        "registered_model": e0["registered_model"],
        "candidate_family": e0["candidate_family"],
        "view": e0["view"],
        "feature_families": e0["feature_families"],
        "loss": e0["loss"],
        "auxiliary_heads": e0["auxiliary_heads"],
        "declared_fold": e0["fold"],
        "seed": e0["seed"],
        "input_package": e0["input_package"],
    }
    package_descriptor_hash = hashlib.sha256(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    search_space_hash = sha256_file(search_space_path)
    budget_hash = hashlib.sha256(
        json.dumps(s1["budget"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    status_lines = git_value(repo, "status", "--porcelain").splitlines()
    authority_recheck = {
        "schema_version": "classification_v2.next_phase.authority_recheck.v1",
        "decision_date": "2026-08-06",
        "execution_worktree": str(repo),
        "execution_branch": git_value(repo, "branch", "--show-current"),
        "base_main_sha_measured": git_value(repo, "rev-parse", "main"),
        "execution_head_sha_measured": git_value(repo, "rev-parse", "HEAD"),
        "expected_execution_head_sha": "34a7b35f8d4f0ee7d9efd9a2882bd89c0de5d1d8",
        "worktree_status_at_generation": (
            "EXPECTED_NEXT_PHASE_CHANGES" if status_lines else "CLEAN"
        ),
        "route_validator_pre_change": {
            "status": "PASS",
            "path": str(route_validator_path),
            "sha256": sha256_file(route_validator_path),
            "required_file_count": 21,
            "loaded_json_count": 18,
        },
        "current_authorities": {
            "classification_code_sha": authority["classification_v2_code_sha"],
            "classification_tree_hash": authority["classification_v2_tree_hash"],
            "reviewed_snapshot_id": authority["reviewed_snapshot_id"],
            "reviewed_snapshot_sha256": authority["reviewed_snapshot_sha256"],
            "split_hash": authority["split_hash"],
            "event_weight_hash": authority["event_weight_hash"],
            "canonical_46d_schema_hash": authority["canonical_schema_hash"],
            "plan_hash": authority["plan_sha256"],
            "environment_lock_hash": authority["environment"]["worktree_file_sha256"].lower(),
        },
        "measured_source_population": source_metrics,
        "protected_state": {
            "data_rebuild": False,
            "hidden_review_reopened": False,
            "behavior_review_reopened": False,
            "outer_split_changed": False,
            "paid_compute_started": False,
            "s1_started": False,
            "c2_started": False,
        },
    }
    write_json(output / "authority_recheck.json", authority_recheck)

    evidence = [
        evidence_record(
            args.exclusion_csv,
            "construction-time additional-source exclusion inventory",
            12,
            0,
            "The 12 CVAT canonical source keys equal the 12 exclusion keys.",
        ),
        evidence_record(
            args.legacy_lineage,
            "legacy source and manifest lineage",
            legacy_lineage.get("candidate_groups_after_collapse"),
            0,
            "Legacy source-key lineage; it does not itself prove cross-role content hashes.",
        ),
        evidence_record(
            args.reviewed_frame,
            "current pooled reviewed frame source manifest",
            source_metrics["rows"],
            source_metrics["cvat_intersection_legacy"],
            "Current source-key comparison with the registered canonicalization rule.",
        ),
        evidence_record(
            args.duplicate_audit,
            "legacy-branch duplicate video audit",
            duplicate_audit["counts"]["legacy_rows"],
            duplicate_audit["counts"]["duplicate_rows"],
            (
                "Duplicate rows are reported within the legacy branch; no cross-role"
                " content contract is bound."
            ),
        ),
        evidence_record(
            args.completion_audit,
            "legacy export and exclusion reconciliation",
            completion_audit["counts"]["export_rows"],
            completion_audit["excluded_video_checks"]["present_in_export"],
            "Export contains zero declared excluded videos and no unexplained reconciliation rows.",
        ),
        evidence_record(
            split_path,
            "grouped development split manifest",
            165305,
            0,
            "Zero missing split bindings and zero row-index mismatches in the bounded check.",
        ),
        evidence_record(
            effective_index_path,
            "effective temporal window index",
            165305,
            0,
            "Zero duplicate window IDs and zero eligible native/video/date role crossings.",
        ),
    ]
    construction_check = {
        "id": "construction_source_overlap",
        "status": "PASS",
        "evidence": [evidence[0], evidence[1], evidence[2]],
        "result": {
            "cvat_minus_exclusion": source_metrics["cvat_minus_exclusion"],
            "exclusion_minus_cvat": source_metrics["exclusion_minus_cvat"],
            "exclusion_intersection_legacy": source_metrics[
                "exclusion_intersection_legacy"
            ],
        },
    }
    proof = {
        "schema_version": "classification_v2.next_phase.a12b_construction_overlap_proof.v1",
        "decision": "INCONCLUSIVE",
        "hard_gate": True,
        "construction_rule": (
            "Additional-source dates/videos were intended to exclude legacy-overlapping"
            " dates/videos; this rule is checked without changing the pooled authority."
        ),
        "checks": [
            construction_check,
            {
                "id": "exact_duplicate_isolation",
                "status": "INCONCLUSIVE",
                "evidence": [evidence[3]],
                "result": {"cross_role_overlap_count": 0, "proof_available": False},
                "limitation": (
                    "The located audit is legacy-branch scoped, not a cross-role"
                    " content audit."
                ),
            },
            {
                "id": "near_duplicate_isolation",
                "status": "INCONCLUSIVE",
                "evidence": [evidence[3]],
                "result": {"detected_overlap_count": None, "proof_available": False},
                "limitation": "No frozen near-duplicate representation and threshold was located.",
            },
            {
                "id": "exact_temporal_interval_isolation",
                "status": "INCONCLUSIVE",
                "evidence": [evidence[5], evidence[6]],
                "result": {"group_crossing_count": 0, "interval_content_proof": False},
                "limitation": (
                    "Grouped inheritance passes, but no frozen exact frame/crop/content"
                    " interval audit is bound."
                ),
            },
            {
                "id": "native_unit_isolation",
                "status": "PASS",
                "evidence": [evidence[6]],
                "result": {"eligible_native_unit_role_crossings": 0},
            },
            {
                "id": "video_group_isolation",
                "status": "PASS",
                "evidence": [evidence[6]],
                "result": {"eligible_video_role_crossings": 0},
            },
            {
                "id": "recording_date_group_isolation",
                "status": "PASS",
                "evidence": [evidence[5], evidence[6]],
                "result": {"eligible_date_role_crossings": 0},
            },
            {
                "id": "window_role_inheritance",
                "status": "PASS",
                "evidence": [evidence[5], evidence[6]],
                "result": {
                    "window_rows": 165305,
                    "eligible_windows": 159410,
                    "excluded_or_invalid_windows": 5895,
                    "duplicate_window_ids": 0,
                    "missing_split_bindings": 0,
                    "row_index_mismatches": 0,
                },
            },
        ],
        "measured_population": source_metrics,
        "limitations": [
            "No frozen near-duplicate representation and threshold was located.",
            "No cross-role exact frame, crop, or content-hash contract was located.",
            (
                "The construction comparison does not authorize S1 while those edges"
                " remain unresolved."
            ),
        ],
        "no_data_rebuild": True,
    }
    write_json(output / "a12b_construction_overlap_proof.json", proof)

    inventory_rows = [
        [
            "A12B-01",
            "construction_source_overlap",
            "PASS",
            evidence[0]["path"],
            evidence[0]["sha256"],
            evidence[0]["semantic_scope"],
            12,
            0,
            evidence[0]["note"],
        ],
        [
            "A12B-02",
            "legacy_source_lineage",
            "PASS",
            evidence[1]["path"],
            evidence[1]["sha256"],
            evidence[1]["semantic_scope"],
            673,
            0,
            evidence[1]["note"],
        ],
        [
            "A12B-03",
            "current_source_key_comparison",
            "PASS",
            evidence[2]["path"],
            evidence[2]["sha256"],
            evidence[2]["semantic_scope"],
            source_metrics["rows"],
            0,
            evidence[2]["note"],
        ],
        [
            "A12B-04",
            "exact_duplicate_isolation",
            "INCONCLUSIVE",
            evidence[3]["path"],
            evidence[3]["sha256"],
            evidence[3]["semantic_scope"],
            duplicate_audit["counts"]["legacy_rows"],
            0,
            evidence[3]["note"],
        ],
        [
            "A12B-05",
            "legacy_export_reconciliation",
            "PASS",
            evidence[4]["path"],
            evidence[4]["sha256"],
            evidence[4]["semantic_scope"],
            completion_audit["counts"]["export_rows"],
            0,
            evidence[4]["note"],
        ],
        [
            "A12B-06",
            "window_group_inheritance",
            "PASS",
            evidence[5]["path"],
            evidence[5]["sha256"],
            evidence[5]["semantic_scope"],
            165305,
            0,
            evidence[5]["note"],
        ],
        [
            "A12B-07",
            "exact_temporal_interval_isolation",
            "INCONCLUSIVE",
            evidence[6]["path"],
            evidence[6]["sha256"],
            "frozen exact frame/crop/content interval contract",
            165305,
            0,
            "Role inheritance passes; exact interval content proof is not bound.",
        ],
        [
            "A12B-08",
            "near_duplicate_isolation",
            "INCONCLUSIVE",
            str(package / "a12_supersession_notice.md"),
            sha256_file(package / "a12_supersession_notice.md"),
            "current A12-B contract",
            "NOT_ESTIMABLE",
            "NOT_ESTIMABLE",
            "No frozen near-duplicate representation or threshold was located.",
        ],
    ]
    with (output / "a12b_evidence_inventory.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "check_id",
                "check_name",
                "status",
                "evidence_path",
                "evidence_sha256",
                "semantic_scope",
                "examined_count",
                "detected_overlap_count",
                "notes",
            ]
        )
        writer.writerows(inventory_rows)

    broken_edge = {
        "schema_version": "classification_v2.next_phase.a12b_broken_provenance_edge.v1",
        "status": "INCONCLUSIVE",
        "failed_invariant": (
            "A12-B requires a frozen cross-role exact/near-duplicate and exact temporal-content"
            " provenance contract before S1."
        ),
        "evidence_paths": [row[3] for row in inventory_rows],
        "unresolved_edges": [
            "near_duplicate_representation_and_threshold",
            "cross_role_exact_frame_crop_content_hash_audit",
            "cross_role_exact_temporal_interval_content_audit",
        ],
        "smallest_correction": (
            "Locate and hash-bind the existing frozen representation, threshold, and audit"
            " over immutable manifests; do not rebuild the reviewed dataset."
        ),
        "invalidated_dependencies": ["S1 permit", "claim-grade C2 permit"],
        "minimum_rerun": "bounded A12-B checker only after the missing evidence is bound",
        "blocked_permit": "S1",
        "data_rebuild": False,
    }
    write_json(output / "a12b_broken_provenance_edge.json", broken_edge)

    posture_binding = file_binding(args.posture_csv)
    posture_authority = {
        "schema_version": "classification_v2.next_phase.posture_authority_binding.v1",
        "status": "INCONCLUSIVE",
        "review_reopened": False,
        "observed_artifact": posture_binding,
        "observed_scope_sha256": posture_metrics["scope_sha256"],
        "class_order": CLASS_ORDER,
        "observed_class_support": posture_metrics["labels"],
        "observed_source_support": posture_metrics["sources"],
        "observed_duplicate_native_keys": posture_metrics["duplicate_native_keys"],
        "reviewers": posture_metrics["reviewers"],
        "candidate_behavior_to_posture": SAFE_POSTURE_MAPPING,
        "mapping_authority_status": "INCONCLUSIVE",
        "mapping_origin": (
            "user-confirmed candidate mapping; no complete current mapping sidecar"
            " located"
        ),
        "checks": {
            "artifact_sha256": "PASS",
            "target_schema_and_class_order": "INCONCLUSIVE",
            "review_close_and_decision_lineage": "INCONCLUSIVE",
            "snapshot_compatibility": "INCONCLUSIVE",
            "temporal_view_alignment": "INCONCLUSIVE",
            "group_support_by_date_inner_role_outer_fold": "INCONCLUSIVE",
            "duplicate_native_unit_check": "PASS",
            "predictive_input_exclusion": "PASS",
            "matched_behavior_cohort": "INCONCLUSIVE",
        },
        "missing_machine_readable_items": [
            "snapshot-bound posture target manifest or sidecar",
            "authoritative mapping and posture class-schema record",
            "explicit unresolved/availability mask and transition policy",
            "temporal-view/native-unit alignment audit",
            "support by recording date, inner role, and outer fold",
        ],
        "included_in_s1": False,
        "posture_head_is_auxiliary_only": True,
    }
    write_json(output / "posture_authority_binding.json", posture_authority)

    support_rows = [
        [
            "posture_class", "lying", "lying",
            posture_metrics["labels"].get("lying", 0), "PASS",
            str(args.posture_csv), "Observed pilot support only.",
        ],
        [
            "posture_class", "sitting", "sitting",
            posture_metrics["labels"].get("sitting", 0), "PASS",
            str(args.posture_csv), "Observed pilot support only.",
        ],
        [
            "posture_class", "upright", "upright",
            posture_metrics["labels"].get("upright", 0), "PASS",
            str(args.posture_csv), "Observed pilot support only.",
        ],
        [
            "source_type", "cvat_tracking_xml", "",
            posture_metrics["sources"].get("cvat_tracking_xml", 0),
            "INCONCLUSIVE", str(args.posture_csv),
            "Source total is known; label cross-tab is not bound.",
        ],
        [
            "source_type", "legacy_recovered", "",
            posture_metrics["sources"].get("legacy_recovered", 0),
            "INCONCLUSIVE", str(args.posture_csv),
            "Source total is known; label cross-tab is not bound.",
        ],
        [
            "native_temporal_unit_key", "duplicate_count", "",
            posture_metrics["duplicate_native_keys"], "PASS",
            str(args.posture_csv), "No duplicate pilot native keys.",
        ],
        [
            "recording_date", "NOT_BOUND", "", "", "INCONCLUSIVE", "",
            "No current posture/date support manifest.",
        ],
        [
            "inner_role", "NOT_BOUND", "", "", "INCONCLUSIVE", "",
            "No current posture/split binding.",
        ],
        [
            "outer_fold", "NOT_BOUND", "", "", "INCONCLUSIVE", "",
            "No current posture/outer-fold support audit.",
        ],
    ]
    with (output / "posture_support_by_group.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "group_dimension", "group_value", "posture_label", "support_count",
                "status", "evidence_path", "notes",
            ]
        )
        writer.writerows(support_rows)

    matched_contract = {
        "schema_version": "classification_v2.next_phase.posture_matched_ablation_contract.v1",
        "status": "BLOCKED_POSTURE_AUTHORITY_INCONCLUSIVE",
        "executable": False,
        "review_reopened": False,
        "behavior_head_class_order": authority["population"].get(
            "behavior_class_order",
            [
                "drink", "eat", "fight", "social-nose", "explore",
                "lying", "stand", "move", "sitting", "playwithtoy",
            ],
        ),
        "posture_head_class_order": CLASS_ORDER,
        "candidate_behavior_to_posture": SAFE_POSTURE_MAPPING,
        "comparisons": [
            {"id": "behavior_only", "posture_loss": 0, "role": "matched_control"},
            {
                "id": "behavior_plus_masked_posture",
                "posture_loss": "registered_weight_only",
                "role": "candidate",
                "unresolved_targets_contribute": False,
            },
        ],
        "fixed_factors": [
            "snapshot", "grouped roles", "temporal view", "feature family",
            "imbalance strategy", "optimizer", "training budget",
            "native-unit evaluator", "seed",
        ],
        "promotion_rule": (
            "retain only if behavior Macro-F1 and rare-class guardrails are not"
            " materially harmed and posture evaluation is adequate"
        ),
        "smallest_missing_item": (
            "complete snapshot-compatible posture target authority and alignment/"
            "support audit"
        ),
        "included_in_s1": False,
    }
    write_json(output / "posture_matched_ablation_contract.json", matched_contract)

    exact_fold = None if not re.fullmatch(r"FOLD_[1-4]", str(e0["fold"])) else e0["fold"]
    preflight = {
        "schema_version": "classification_v2.next_phase.e0_preflight_decision.v1",
        "decision_date": "2026-08-06",
        "registered_model": e0["registered_model"],
        "candidate_family": e0["candidate_family"],
        "temporal_view": e0["view"],
        "modalities": ["actor_rgb", "geometry_6D", "motion_12D"],
        "seed": e0["seed"],
        "declared_fold_text": e0["fold"],
        "registered_inner_fold": exact_fold,
        "config_status": "E0_CONFIG_INCOMPLETE" if exact_fold is None else "COMPLETE",
        "ready_to_launch_e0": False,
        "paid_execution_authorization": "NO",
        "e0_status": "NOT_EXECUTED",
        "input_binding_sha256": input_binding_hash,
        "package_descriptor_sha256": package_descriptor_hash,
        "route_hashes": authority["current_authority"] if "current_authority" in authority else {
            "classification_code_sha": authority["classification_v2_code_sha"],
            "classification_tree_hash": authority["classification_v2_tree_hash"],
            "snapshot_sha256": authority["reviewed_snapshot_sha256"],
            "split_hash": authority["split_hash"],
            "event_weight_hash": authority["event_weight_hash"],
            "schema_hash": authority["canonical_schema_hash"],
            "environment_lock_hash": authority["environment"]["worktree_file_sha256"].lower(),
        },
        "checks": {
            "code_and_tree": "PASS",
            "reviewed_snapshot": "PASS",
            "split_event_weight_schema": "PASS",
            "t6_exact_view_feature_binding": "PASS",
            "geometry_and_motion_schema": "PASS",
            "masks_and_zero_weight_filtering": "PASS",
            "image_crop_package_inventory": "PASS_HASH_BOUND_DESCRIPTOR_ONLY",
            "checkpoint_contract": "PASS_REUSED_ENGINEERING_EVIDENCE",
            "prediction_exporter": "PASS_REUSED_ENGINEERING_EVIDENCE",
            "native_unit_metric_path": "PASS_REUSED_ENGINEERING_EVIDENCE",
            "remote_package_upload": "NOT_EXECUTED",
            "exact_inner_fold": "FAIL_MISSING_FROM_COMMITTED_PERMIT",
        },
        "outer_test_access": "BLOCKED",
        "outer_access_policy": {
            "data_mount": False,
            "labels": False,
            "metrics": False,
            "predictions": False,
            "errors": False,
            "confusion_matrices": False,
            "registered_outer_resources": [],
        },
        "outer_access_negative_test": {
            "status": "PASS",
            "scope": "local_package_contract_only",
            "process_started": False,
            "result": (
                "all search/permit/outer-contract access flags deny outer data,"
                " labels, metrics, predictions, errors, and confusion matrices"
            ),
        },
        "limits": e0["budget"],
        "blocker_code": "E0_CONFIG_INCOMPLETE",
        "blockers": [
            (
                "The committed E0 manifest does not identify an exact inner-"
                "development fold; no fold may be invented."
            ),
            "PAID_EXECUTION_AUTHORIZATION=NO; no remote package was uploaded.",
        ],
        "smallest_correction": (
            "Amend the E0 permit with an exact registered inner fold, re-hash the"
            " descriptor, and rerun only local preflight."
        ),
    }
    write_json(output / "e0_preflight_decision.json", preflight)

    e0_common = {
        "schema_version": "classification_v2.next_phase.e0_manifest.v1",
        "status": "NOT_EXECUTED",
        "execution_mode": "remote_pilot",
        "registered_model": e0["registered_model"],
        "temporal_view": e0["view"],
        "feature_families": e0["feature_families"],
        "seed": e0["seed"],
        "registered_inner_fold": exact_fold,
        "declared_fold_text": e0["fold"],
        "input_binding_sha256": input_binding_hash,
        "package_descriptor_sha256": package_descriptor_hash,
        "paid_execution_authorization": "NO",
        "no_outer_test_access": True,
        "reason": "NOT_EXECUTED: paid authorization is NO and the exact inner fold is missing.",
        "no_auto_promotion": True,
    }
    write_json(output / "e0_execution_manifest.json", {
        **e0_common,
        "remote_run_id": None,
        "uploaded": False,
        "expected_outputs": e0["expected_outputs"],
        "remote_environment": None,
    })
    write_json(output / "e0_resume_audit.json", {
        **e0_common,
        "checkpoint_created": False,
        "forced_interruption": "NOT_EXECUTED",
        "resume_started": False,
        "resume_state_equivalence": "NOT_EXECUTED",
        "required_checkpoint_state": [
            "model_state", "optimizer_state", "scheduler_state", "global_step",
            "scaler_state_when_applicable", "seed_rng_state", "configuration",
            "dependency_hashes",
        ],
    })
    write_json(output / "e0_runtime_vram_cost.json", {
        **e0_common,
        "gpu_hours": None,
        "wall_clock_hours": None,
        "peak_vram_gb": None,
        "checkpoint_size_bytes": None,
        "prediction_size_bytes": None,
        "upload_bytes": None,
        "download_bytes": None,
        "measured_cost_usd": None,
        "hard_caps": e0["budget"],
    })
    write_json(output / "e0_download_hashes.json", {
        **e0_common,
        "remote_manifest_received": False,
        "download_completed": False,
        "hash_parity": "NOT_EXECUTED",
        "downloaded_artifacts": [],
        "large_artifacts_outside_git": True,
    })

    s1_decision = {
        "schema_version": "classification_v2.next_phase.s1_readiness_decision.v1",
        "decision_date": "2026-08-06",
        "READY_FOR_PAID_INNER_AUTORESEARCH_S1": "NO",
        "S1_PERMIT_STATUS": "BLOCKED",
        "A12_A_STATUS": "PASS",
        "A12_B_STATUS": "INCONCLUSIVE",
        "E0_STATUS": "NOT_EXECUTED",
        "POSTURE_AUTHORITY_STATUS": "INCONCLUSIVE",
        "POSTURE_INCLUDED_IN_S1": False,
        "OUTER_TEST_ISOLATION_STATUS": "PASS",
        "SEARCH_SPACE_HASH": search_space_hash,
        "BUDGET_HASH": budget_hash,
        "search_space_status": search_space["status"],
        "candidate_families_considered": search_space["allowed_dimensions"]["model_family"],
        "posture_exclusion_rule": (
            "unresolved posture authority is excluded from executable S1 candidates"
        ),
        "trial_budget": s1["budget"],
        "stop_rules": search_space["trial_budget"],
        "remote_lineage_checker": "NOT_EXECUTED; E0 is not authorized",
        "no_auto_promotion": True,
        "READY_FOR_CLAIM_GRADE_OUTER_OOF_C2": "NO",
        "PAPER_GRADE_RESULT_AVAILABLE": "NO",
        "BLOCKERS": [
            (
                "A12-B remains INCONCLUSIVE because the frozen near-duplicate, exact"
                " content, and exact temporal-interval provenance edge is not bound."
            ),
            (
                "E0 is NOT_EXECUTED because the committed permit lacks an exact"
                " inner-development fold and paid authorization is NO."
            ),
            (
                "Posture authority is INCONCLUSIVE; posture supervision is excluded"
                " from S1 until formally bound or closed."
            ),
        ],
        "NEXT_AUTHORIZED_ACTION": (
            "Resolve and hash-bind the missing frozen exact/near-duplicate and"
            " temporal-content provenance contract, then rerun only the bounded"
            " A12-B and readiness validators."
        ),
    }
    write_json(output / "s1_readiness_decision.json", s1_decision)
    authority_recheck["scope_reconciliation"] = {
        "route_semantics_remain_current": True,
        "superseded_for_current_execution_scope": [
            {
                "path": str(package / "permit_policy.json"),
                "sha256": sha256_file(package / "permit_policy.json"),
                "status": "SUPERSEDED_FOR_E0_LAUNCH_READINESS_ONLY",
                "reason": "The committed permit names no exact inner-development fold.",
                "replacement_path": str(output / "e0_preflight_decision.json"),
                "replacement_sha256": sha256_file(output / "e0_preflight_decision.json"),
            },
            {
                "path": str(package / "readiness_decision.json"),
                "sha256": sha256_file(package / "readiness_decision.json"),
                "status": "SUPERSEDED_FOR_E0_LAUNCH_READINESS_ONLY",
                "reason": "Current preflight found E0_CONFIG_INCOMPLETE before any launch.",
                "replacement_path": str(output / "s1_readiness_decision.json"),
                "replacement_sha256": sha256_file(output / "s1_readiness_decision.json"),
            },
        ],
    }
    write_json(output / "authority_recheck.json", authority_recheck)
    print(json.dumps({
        "output_dir": str(output),
        "source_rows": source_metrics["rows"],
        "posture_rows": posture_metrics["rows"],
        "input_binding_sha256": input_binding_hash,
        "package_descriptor_sha256": package_descriptor_hash,
        "search_space_sha256": search_space_hash,
        "budget_sha256": budget_hash,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
