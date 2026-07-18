"""Run the final hash-bound gate for a legacy 16-frame rebuild."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import pandas as pd

from legacy_burst_recovery.check_duplicate_videos import (
    normalize_source_video_key,
)
from legacy_burst_recovery.cvat_behavior_overlay import (
    load_cvat_legacy_rows,
    select_first_task_frame_authority,
)
from pig_behavior.classification_v2.schema import normalize_behavior, normalize_pig_id

BEHAVIOR_CLASSES = {
    "drink",
    "eat",
    "fight",
    "social-nose",
    "explore",
    "lying",
    "stand",
    "move",
    "sitting",
    "playwithtoy",
}
ANCHOR_OFFSETS = [0, 3, 6, 9, 12, 15]
ANCHOR_COLUMNS = ["x1", "y1", "x2", "y2"]
PROVENANCE_COLUMNS = [
    "hidden",
    "hidden_source",
    "hidden_review_status",
    "hidden_is_trusted",
    "hidden_trust_status",
    "visibility_quality",
    "hidden_seed_method",
]
ANCHOR_PROVENANCE_COLUMNS = [
    column for column in PROVENANCE_COLUMNS if column != "hidden_seed_method"
]
CODE_STATE_PATHS = [
    "src/legacy_burst_recovery/cvat_anchor_rebuild.py",
    "src/legacy_burst_recovery/cvat_behavior_overlay.py",
    "src/legacy_burst_recovery/export_legacy_annotations.py",
    (
        "scripts/classification_v2/00_source_feature_temporal/"
        "classification_v2_rebuild_legacy_cvat_recovery_inputs.py"
    ),
    (
        "scripts/classification_v2/09_final_release_audit/"
        "check_legacy_16f_rebuild_completion.py"
    ),
    "tests/test_legacy_cvat_behavior_overlay.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def actor_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(
        zip(
            frame["group_id"].astype(str),
            frame["pig_id"].map(normalize_pig_id),
            strict=True,
        )
    )


def add_error(errors: list[str], name: str, count: int) -> None:
    if count:
        errors.append(f"{name}={count}")


def compare_anchor_evidence(
    anchors: pd.DataFrame,
    exported: pd.DataFrame,
) -> dict[str, int]:
    raw_bbox_mismatches = 0
    clipped_bbox_coordinates = 0
    hidden_mismatches = 0
    missing = 0
    export_index = exported.set_index(
        ["group_id", "pig_id", "relative_frame_index"],
        drop=False,
    )
    for row in anchors.itertuples(index=False):
        key = (str(row.group_id), normalize_pig_id(row.pig_id), int(row.legacy_order) * 3)
        try:
            matched = export_index.loc[key]
        except KeyError:
            missing += 1
            continue
        if isinstance(matched, pd.DataFrame):
            missing += 1
            continue
        for column in ANCHOR_COLUMNS:
            left = float(getattr(row, column))
            raw = float(matched[f"{column}_raw"])
            operational = float(matched[column])
            if not math.isclose(left, raw, rel_tol=0.0, abs_tol=1e-6):
                raw_bbox_mismatches += 1
            if not math.isclose(left, operational, rel_tol=0.0, abs_tol=1e-6):
                clipped_bbox_coordinates += 1
        for column in ANCHOR_PROVENANCE_COLUMNS:
            left = str(getattr(row, column, "")).strip()
            right = str(matched[column]).strip()
            if column == "hidden_is_trusted":
                if bool_value(left) != bool_value(right):
                    hidden_mismatches += 1
            elif left != right:
                hidden_mismatches += 1
    return {
        "missing_anchor_rows": missing,
        "anchor_raw_bbox_mismatches": raw_bbox_mismatches,
        "anchor_operational_bbox_clipped_coordinates": (
            clipped_bbox_coordinates
        ),
        "anchor_hidden_provenance_mismatches": hidden_mismatches,
    }


def compare_dense_export_evidence(
    dense: pd.DataFrame,
    exported: pd.DataFrame,
) -> dict[str, int]:
    key = ["group_id", "pig_id", "frame_index"]
    dense_columns = key + ANCHOR_COLUMNS + PROVENANCE_COLUMNS
    export_columns = key + [f"{column}_raw" for column in ANCHOR_COLUMNS]
    export_columns += PROVENANCE_COLUMNS
    dense_evidence = dense[dense_columns].rename(
        columns={column: f"{column}_dense" for column in ANCHOR_COLUMNS}
    )
    joined = dense_evidence.merge(
        exported[export_columns],
        on=key,
        how="outer",
        suffixes=("_dense", "_export"),
        indicator=True,
        validate="one_to_one",
    )
    missing_rows = int(joined["_merge"].ne("both").sum())
    bbox_mismatches = 0
    for column in ANCHOR_COLUMNS:
        dense_values = pd.to_numeric(
            joined[f"{column}_dense"],
            errors="coerce",
        )
        export_values = pd.to_numeric(
            joined[f"{column}_raw"],
            errors="coerce",
        )
        bbox_mismatches += int(
            (dense_values - export_values).abs().gt(1e-6).sum()
        )
    provenance_mismatches = 0
    for column in PROVENANCE_COLUMNS:
        before = joined[f"{column}_dense"].fillna("").astype(str).str.strip()
        after = joined[f"{column}_export"].fillna("").astype(str).str.strip()
        if column == "hidden_is_trusted":
            before = before.map(bool_value)
            after = after.map(bool_value)
        provenance_mismatches += int(before.ne(after).sum())
    return {
        "dense_export_missing_rows": missing_rows,
        "dense_export_raw_bbox_mismatches": bbox_mismatches,
        "dense_export_hidden_provenance_mismatches": provenance_mismatches,
    }


def authority_behavior_mismatches(
    exported: pd.DataFrame,
    cvat_root: Path,
) -> dict[str, int]:
    prepared, _ = load_cvat_legacy_rows(cvat_root)
    authority = select_first_task_frame_authority(prepared)
    authority_map = {
        (str(row.group_id), normalize_pig_id(row.pig_id)): normalize_behavior(
            row.behavior
        )
        for row in authority.itertuples(index=False)
    }
    exported_map = (
        exported.assign(
            _behavior=exported["behavior"].map(normalize_behavior),
            _pig=exported["pig_id"].map(normalize_pig_id),
        )
        .groupby(["group_id", "_pig"], dropna=False)["_behavior"]
        .agg(lambda values: set(values.dropna()))
    )
    export_missing_authority = 0
    native_authority_missing_export = 0
    mismatches = 0
    for key, observed in exported_map.items():
        if key not in authority_map:
            export_missing_authority += 1
            continue
        expected = authority_map[key]
        if observed != {expected}:
            mismatches += 1
    for key in authority_map:
        if key not in exported_map.index:
            native_authority_missing_export += 1
    return {
        "export_actor_keys_missing_native_authority": export_missing_authority,
        "export_actor_behavior_authority_mismatches": mismatches,
        "native_authority_actor_keys_missing_export": (
            native_authority_missing_export
        ),
        "native_authority_actor_keys": len(authority_map),
    }


def collect_hashes(run_root: Path) -> dict[str, str]:
    roots = [
        run_root / name
        for name in [
            "00_behavior_source",
            "01_provenance",
            "02_video_policy",
            "03_cvat_audit",
            "04_cvat_inputs",
            "08_audits",
        ]
    ]
    roots.extend(
        [
            run_root / "05_short_smoke" / "recovery" / "legacy_dense_tracklet_map.csv",
            run_root / "05_short_smoke" / "recovery" / "cvat_recovery_output_audit.json",
            run_root / "06_full_recovery" / "legacy_dense_tracklet_map.csv",
            run_root / "06_full_recovery" / "legacy_training_sequence_manifest.csv",
            run_root / "06_full_recovery" / "qa_summary.json",
            run_root / "06_full_recovery" / "timing_report.json",
            run_root / "07_export" / "legacy_frame_object_annotations.csv",
            run_root / "07_export" / "legacy_frame_object_export_audit.json",
            run_root / "07_export" / "legacy_cvat_behavior_authority_audit.json",
            run_root / "07_export" / "legacy_cvat_behavior_discrepancies.csv",
        ]
    )
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(path for path in root.rglob("*") if path.is_file())
        elif root.is_file():
            files.append(root)
    return {
        str(path.relative_to(run_root)): sha256_file(path)
        for path in sorted(set(files))
        if path.name != "legacy_16f_rebuild_completion_audit.json"
    }


def collect_code_state(repo_root: Path) -> dict[str, object]:
    """Bind the final audit to code and the complete dirty-worktree state."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status_bytes = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"],
        cwd=repo_root,
        capture_output=True,
        check=True,
    ).stdout
    status_entries = [item for item in status_bytes.split(b"\0") if item]
    code_hashes = {
        path: sha256_file(repo_root / path) for path in CODE_STATE_PATHS
    }
    return {
        "git_head": head,
        "worktree_dirty": bool(status_entries),
        "worktree_status_entry_count": len(status_entries),
        "worktree_status_sha256": hashlib.sha256(status_bytes).hexdigest(),
        "code_sha256": code_hashes,
    }


def build_completion_audit(
    run_root: Path,
    cvat_root: Path,
    repo_root: Path,
) -> dict[str, object]:
    errors: list[str] = []
    required = {
        "center": run_root / "04_cvat_inputs" / "legacy_center_keyframes_from_cvat.csv",
        "anchors": run_root / "04_cvat_inputs" / "legacy_six_anchor_bboxes_from_cvat.csv",
        "p5_audit": run_root / "04_cvat_inputs" / "legacy_cvat_recovery_input_audit.json",
        "dense": run_root / "06_full_recovery" / "legacy_dense_tracklet_map.csv",
        "export": run_root / "07_export" / "legacy_frame_object_annotations.csv",
        "export_audit": run_root / "07_export" / "legacy_frame_object_export_audit.json",
        "authority_audit": run_root / "07_export" / "legacy_cvat_behavior_authority_audit.json",
        "discrepancies": run_root / "07_export" / "legacy_cvat_behavior_discrepancies.csv",
        "p8_audit": run_root / "08_audits" / "full_cvat_recovery_output_audit.json",
        "actor_support": run_root / "08_audits" / "full_actor_exclusion_support.csv",
        "video_support": run_root / "08_audits" / "full_video_exclusion_support.csv",
        "actor_policy": run_root / "02_video_policy" / "excluded_actor_keys.csv",
        "video_policy": run_root / "02_video_policy" / "exclude_source_videos.csv",
        "behavior_source": run_root / "00_behavior_source" / "behavior_with_feats_rectROI.csv",
        "provenance_center": run_root / "01_provenance" / "old_burst_center_keyframes_combined.csv",
        "provenance_bbox": (
            run_root
            / "01_provenance"
            / "old_burst_all_keyframe_bboxes_combined.csv"
        ),
    }
    missing_files = [str(path) for path in required.values() if not path.is_file()]
    if missing_files:
        errors.append(f"missing_required_files={len(missing_files)}")
        return {
            "schema_version": 1,
            "status": "FAIL",
            "errors": errors,
            "missing_required_files": missing_files,
        }

    center = pd.read_csv(required["center"], low_memory=False)
    anchors = pd.read_csv(required["anchors"], low_memory=False)
    dense = pd.read_csv(required["dense"], low_memory=False)
    exported = pd.read_csv(required["export"], low_memory=False)
    p5_audit = json.loads(required["p5_audit"].read_text(encoding="utf-8"))
    export_audit = json.loads(required["export_audit"].read_text(encoding="utf-8"))
    authority_audit = json.loads(
        required["authority_audit"].read_text(encoding="utf-8")
    )
    p8_audit = json.loads(required["p8_audit"].read_text(encoding="utf-8"))
    center_keys = actor_keys(center)
    dense_keys = actor_keys(dense)
    export_keys = actor_keys(exported)
    add_error(errors, "center_dense_actor_key_difference", len(center_keys ^ dense_keys))
    add_error(errors, "dense_export_actor_key_difference", len(dense_keys ^ export_keys))
    add_error(
        errors,
        "center_actor_key_duplicates",
        center.duplicated(list(center_keys_columns())).sum(),
    )
    frame_key = ["group_id", "pig_id", "frame_index"]
    add_error(
        errors,
        "dense_frame_key_duplicates",
        dense.duplicated(frame_key).sum(),
    )
    add_error(
        errors,
        "export_frame_key_duplicates",
        exported.duplicated(frame_key).sum(),
    )

    dense_counts = dense.groupby(["group_id", "pig_id"]).size()
    export_counts = exported.groupby(["group_id", "pig_id"]).size()
    add_error(errors, "dense_actor_sequence_length_mismatches", int((dense_counts != 16).sum()))
    add_error(errors, "export_actor_sequence_length_mismatches", int((export_counts != 16).sum()))
    invalid_bbox = ~(
        pd.to_numeric(exported["x1"], errors="coerce").notna()
        & pd.to_numeric(exported["y1"], errors="coerce").notna()
        & pd.to_numeric(exported["x2"], errors="coerce").notna()
        & pd.to_numeric(exported["y2"], errors="coerce").notna()
        & exported["x2"].gt(exported["x1"])
        & exported["y2"].gt(exported["y1"])
    )
    add_error(errors, "export_invalid_bbox_rows", int(invalid_bbox.sum()))
    relative = sorted(exported["relative_frame_index"].unique().tolist())
    if relative != list(range(16)):
        errors.append(f"export_relative_frame_set={relative}")
    anchor_rows = exported[exported["is_legacy_gt_anchor"].map(bool_value)]
    anchor_offsets = sorted(anchor_rows["relative_frame_index"].unique().tolist())
    if anchor_offsets != ANCHOR_OFFSETS:
        errors.append(f"export_anchor_offsets={anchor_offsets}")
    noncanonical_behavior = ~exported["behavior"].isin(BEHAVIOR_CLASSES)
    add_error(
        errors,
        "noncanonical_behavior_rows",
        int(noncanonical_behavior.sum()),
    )
    anchor_checks = compare_anchor_evidence(anchors, exported)
    for name in [
        "missing_anchor_rows",
        "anchor_raw_bbox_mismatches",
        "anchor_hidden_provenance_mismatches",
    ]:
        count = anchor_checks[name]
        add_error(errors, name, count)
    dense_export_checks = compare_dense_export_evidence(dense, exported)
    for name, count in dense_export_checks.items():
        add_error(errors, name, count)
    authority_checks = authority_behavior_mismatches(exported, cvat_root)
    for name in [
        "export_actor_keys_missing_native_authority",
        "export_actor_behavior_authority_mismatches",
    ]:
        add_error(errors, name, int(authority_checks[name]))
    actor_support = pd.read_csv(required["actor_support"], low_memory=False)
    excluded_actors_present = int(
        actor_support["present_in_dense"].map(bool_value).sum()
    )
    add_error(
        errors,
        "excluded_actor_support_rows_present_in_dense",
        excluded_actors_present,
    )
    video_support = pd.read_csv(required["video_support"], low_memory=False)
    excluded_video_keys = set(video_support["source_video_key"].astype(str))
    retained_video_keys = set(exported["source_video_key"].astype(str))
    excluded_videos_present = len(excluded_video_keys & retained_video_keys)
    add_error(
        errors,
        "excluded_video_keys_present_in_export",
        excluded_videos_present,
    )
    actor_policy = pd.read_csv(required["actor_policy"], low_memory=False)
    actor_policy_keys = actor_keys(actor_policy)
    actor_support_keys = actor_keys(actor_support)
    add_error(
        errors,
        "actor_policy_support_key_difference",
        len(actor_policy_keys ^ actor_support_keys),
    )
    behavior_source = pd.read_csv(required["behavior_source"], low_memory=False)
    filtered_frames = {
        "behavior_source": behavior_source,
        "p5_center": center,
        "dense": dense,
        "export": exported,
    }
    actor_policy_presence = {
        name: len(actor_keys(frame) & actor_policy_keys)
        for name, frame in filtered_frames.items()
    }
    for name, count in actor_policy_presence.items():
        add_error(errors, f"actor_policy_keys_present_in_{name}", count)
    video_policy = pd.read_csv(required["video_policy"], low_memory=False)
    video_policy_keys = set(video_policy["source_video_key"].astype(str))
    add_error(
        errors,
        "video_policy_support_key_difference",
        len(video_policy_keys ^ excluded_video_keys),
    )
    discrepancies = pd.read_csv(required["discrepancies"], low_memory=False)
    provenance = pd.read_csv(required["provenance_center"], low_memory=False)
    provenance["_actor_key"] = list(
        zip(
            provenance["group_id"].astype(str),
            provenance["pig_id"].map(normalize_pig_id),
            strict=True,
        )
    )
    source_by_actor = dict(
        zip(
            provenance["_actor_key"],
            provenance["video_final"].map(normalize_source_video_key),
            strict=True,
        )
    )
    provenance_keys = set(source_by_actor)
    provenance_keys_missing_export = provenance_keys - export_keys
    unexplained_provenance_keys = {
        key
        for key in provenance_keys_missing_export
        if source_by_actor.get(key) not in video_policy_keys
    }
    add_error(
        errors,
        "unexplained_provenance_actor_keys_missing_export",
        len(unexplained_provenance_keys),
    )
    source_prepared, _ = load_cvat_legacy_rows(cvat_root)
    source_actor_keys = list(
        zip(
            source_prepared["group_id"].astype(str),
            source_prepared["pig_id"].map(normalize_pig_id),
            strict=True,
        )
    )
    source_actor_policy_rows = sum(
        key in actor_policy_keys for key in source_actor_keys
    )
    p5_counts = p5_audit.get("counts", {})
    if p5_audit.get("status") != "PASS":
        errors.append(f"p5_audit_status={p5_audit.get('status')}")
    if p5_audit.get("errors"):
        errors.append(f"p5_audit_errors={len(p5_audit['errors'])}")
    if p5_audit.get("warnings"):
        errors.append(f"p5_audit_warnings={len(p5_audit['warnings'])}")
    add_error(
        errors,
        "p5_incomplete_retained_authority_keys",
        int(p5_counts.get("incomplete_authority_keys", -1)),
    )
    add_error(
        errors,
        "p5_actor_filter_row_accounting_delta",
        abs(
            int(p5_counts.get("actor_exclusion_rows_removed", -1))
            - source_actor_policy_rows
        ),
    )
    add_error(
        errors,
        "p5_retained_raw_row_accounting_delta",
        abs(int(p5_counts.get("cvat_rows_all_groups", -1)) - len(behavior_source)),
    )
    provenance_bbox = pd.read_csv(required["provenance_bbox"], low_memory=False)
    provenance_bbox_video_keys = provenance_bbox["video_final"].map(
        normalize_source_video_key
    )
    behavior_video_policy_rows = int(
        provenance_bbox_video_keys.isin(video_policy_keys).sum()
    )
    source_authority = select_first_task_frame_authority(source_prepared)
    source_authority_policy_keys = actor_keys(source_authority) & actor_policy_keys
    expected_native_authority_missing_export = (
        len(provenance_keys_missing_export) + len(source_authority_policy_keys)
    )
    add_error(
        errors,
        "native_authority_missing_export_reconciliation_delta",
        abs(
            int(authority_checks["native_authority_actor_keys_missing_export"])
            - expected_native_authority_missing_export
        ),
    )
    add_error(
        errors,
        "p2_actor_filter_row_accounting_delta",
        abs(
            len(source_prepared)
            - source_actor_policy_rows
            - len(behavior_source)
        ),
    )
    add_error(
        errors,
        "p4_video_filter_row_accounting_delta",
        abs(len(provenance_bbox) - behavior_video_policy_rows - len(anchors)),
    )
    filtered_before_authority = int(
        authority_audit.get("counts", {}).get(
            "cvat_object_rows_filtered_before_authority_audit",
            -1,
        )
    )
    add_error(
        errors,
        "p9_filter_row_accounting_delta",
        abs(
            filtered_before_authority
            - source_actor_policy_rows
            - behavior_video_policy_rows
        ),
    )
    discrepancy_actor_exclusions = 0
    discrepancy_video_exclusions = 0
    unexplained_discrepancies = 0
    for row in discrepancies.itertuples(index=False):
        key = (str(row.group_id), normalize_pig_id(row.pig_id))
        if key in actor_policy_keys:
            discrepancy_actor_exclusions += 1
        elif source_by_actor.get(key) in video_policy_keys:
            discrepancy_video_exclusions += 1
        else:
            unexplained_discrepancies += 1
    add_error(
        errors,
        "unexplained_native_authority_keys_missing_export",
        unexplained_discrepancies,
    )
    if export_audit.get("status") != "PASS":
        errors.append(f"export_audit_status={export_audit.get('status')}")
    if export_audit.get("errors"):
        errors.append(f"export_audit_errors={len(export_audit['errors'])}")
    if p8_audit.get("status") != "PASS":
        errors.append(f"p8_audit_status={p8_audit.get('status')}")
    if authority_audit.get("status") != "PASS":
        errors.append(
            f"authority_audit_status={authority_audit.get('status')}"
        )
    if authority_audit.get("errors"):
        errors.append(
            f"authority_audit_errors={len(authority_audit['errors'])}"
        )
    if authority_audit.get("warnings"):
        errors.append(
            f"authority_audit_warnings={len(authority_audit['warnings'])}"
        )
    if authority_audit.get("counts", {}).get("dense_keys_missing_authority", 0):
        errors.append("dense_keys_missing_native_authority")
    data_status = subprocess.run(
        ["git", "status", "--short", "--", "data"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if data_status:
        errors.append("tracked_or_visible_data_changes")
    return {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "run_root": str(run_root),
        "policy": {
            "canonical_export": "all_dense_rows_no_training_only",
            "sequence_length": 16,
            "anchor_relative_frames": ANCHOR_OFFSETS,
            "behavior_authority": "independently_reloaded_first_task_frame_per_group",
            "raw_data_immutable": True,
            "training_or_oof": "not_run",
        },
        "counts": {
            "center_rows": len(center),
            "anchor_rows": len(anchors),
            "dense_rows": len(dense),
            "export_rows": len(exported),
            "center_actor_keys": len(center_keys),
            "dense_actor_keys": len(dense_keys),
            "export_actor_keys": len(export_keys),
            "native_authority_actor_keys": authority_checks["native_authority_actor_keys"],
            "declared_excluded_actors": len(actor_support),
            "declared_excluded_videos": len(video_support),
            "authority_keys_missing_dense": authority_audit.get(
                "counts",
                {},
            ).get("authority_keys_missing_dense", 0),
        },
        "anchor_checks": anchor_checks,
        "dense_export_checks": dense_export_checks,
        "authority_checks": authority_checks,
        "excluded_actor_checks": {
            "rows": len(actor_support),
            "present_in_dense": excluded_actors_present,
            "policy_presence_by_stage": actor_policy_presence,
        },
        "excluded_video_checks": {
            "policy_rows": len(video_support),
            "present_in_export": excluded_videos_present,
        },
        "authority_exclusion_reconciliation": {
            "discrepancy_rows": len(discrepancies),
            "explained_by_actor_policy": discrepancy_actor_exclusions,
            "explained_by_video_policy": discrepancy_video_exclusions,
            "unexplained": unexplained_discrepancies,
            "native_authority_keys_missing_export": authority_checks[
                "native_authority_actor_keys_missing_export"
            ],
            "actor_policy_authority_keys": len(source_authority_policy_keys),
            "video_policy_authority_keys": len(
                provenance_keys_missing_export
            ),
        },
        "filter_row_reconciliation": {
            "raw_cvat_rows": len(source_prepared),
            "actor_policy_rows_removed": source_actor_policy_rows,
            "behavior_source_rows": len(behavior_source),
            "video_policy_rows_removed": behavior_video_policy_rows,
            "retained_anchor_rows": len(anchors),
            "p9_rows_filtered_before_authority_audit": (
                filtered_before_authority
            ),
        },
        "p5_filter_gate": {
            "status": p5_audit.get("status"),
            "errors": p5_audit.get("errors", []),
            "warnings": p5_audit.get("warnings", []),
            "actor_exclusion_stage": p5_audit.get("policy", {}).get(
                "actor_exclusion_stage"
            ),
            "actor_exclusion_rows_removed": p5_counts.get(
                "actor_exclusion_rows_removed"
            ),
            "retained_incomplete_authority_keys": p5_counts.get(
                "incomplete_authority_keys"
            ),
        },
        "data_git_status": data_status,
        "code_and_worktree_state": collect_code_state(repo_root),
        "source_and_artifact_sha256": collect_hashes(run_root),
        "errors": errors,
    }


def center_keys_columns() -> tuple[str, str]:
    return ("group_id", "pig_id")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cvat-export-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    audit = build_completion_audit(
        args.run_root,
        args.cvat_export_root,
        args.repo_root.resolve(),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
