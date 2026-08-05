from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(r"C:/Users/ironh/Downloads/PIG_Behavior_Project")
E_ROOT = Path(r"E:/PigProjectStorage/PIG_Behavior_Project/.codex_tmp/worktrees/")
WORKTREE = E_ROOT / "classification_v2_pre_gpu_autoresearch_q2"
MATERIALIZED = ROOT / (
    "outputs/classification_v2/agent_audits/"
    "post_review_frame_amendment_materialization_fa028cb_20260803_224700"
)
CURRENT_AUDIT = WORKTREE / (
    "outputs/classification_v2/model_readiness_audit/"
    "px2_rebind_59684f7_20260805_020000"
)
OLD_ROLES = ROOT / (
    "outputs/classification_v2/model_readiness_audit/"
    "pre_gpu_autoresearch_q2_59684f7_20260805_011800/"
    "four_fold_roles_47103f6_20260804_153800"
)
OLD_WEIGHTS = ROOT / (
    "outputs/classification_v2/model_readiness_audit/"
    "pre_gpu_autoresearch_q2_59684f7_20260805_011800/"
    "fold_event_weights_47103f6_20260804_154500"
)
RGB_MANIFEST = ROOT / (
    "outputs/classification_v2/model_readiness_audit/"
    "pre_gpu_autoresearch_q2_6c2f204_20260804_084638/reviewed_rgb_v1/"
    "rgb_authority_manifest_v2.json"
)
SOCIAL_MANIFEST = ROOT / (
    "outputs/classification_v2/agent_audits/"
    "social_topk_k3_bundle_a8f727a_20260804_070500/manifest.json"
)
OUTPUT = CURRENT_AUDIT / "px2_rebind_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_csv(path: Path, key: str) -> tuple[list[str], dict[str, object]]:
    values: list[str] = []
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or key not in reader.fieldnames:
            raise ValueError(f"missing key column={key} path={path}")
        for row in reader:
            value = str(row.get(key, "")).strip()
            values.append(value)
            counts[value] += 1
    digest = hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()
    duplicates = sum(count for value, count in counts.items() if value and count > 1)
    blanks = counts.get("", 0)
    return values, {
        "path": str(path),
        "sha256": sha256(path),
        "rows": len(values),
        "blank_keys": blanks,
        "duplicate_key_rows": duplicates,
        "ordered_key_sha256": digest,
    }


def compare_order(left: list[str], right: list[str]) -> dict[str, object]:
    size = max(len(left), len(right))
    mismatch = sum(
        (left[i] if i < len(left) else "") != (right[i] if i < len(right) else "")
        for i in range(size)
    )
    return {
        "left_rows": len(left),
        "right_rows": len(right),
        "order_mismatch_rows": mismatch,
        "exact_order_equal": mismatch == 0 and len(left) == len(right),
        "set_equal": set(left) == set(right),
    }


def scan_aligned_column(path: Path, column: str) -> tuple[list[str], dict[str, object]]:
    values: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"missing column={column} path={path}")
        values.extend(str(row.get(column, "")).strip() for row in reader)
    return values, {"path": str(path), "sha256": sha256(path), "rows": len(values)}


def scan_effective(path: Path) -> tuple[list[str], dict[str, object]]:
    values: list[str] = []
    labels: list[str] = []
    valid: list[str] = []
    source_counts: Counter[str] = Counter()
    length_counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "window_id",
            "behavior_window_label",
            "window_valid_for_main_train",
            "source_type",
            "window_length_frames",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"effective index schema missing={required - set(reader.fieldnames or [])}")
        for row in reader:
            value = str(row.get("window_id", "")).strip()
            values.append(value)
            labels.append(str(row.get("behavior_window_label", "")).strip())
            valid.append(str(row.get("window_valid_for_main_train", "")).strip())
            source_counts[str(row.get("source_type", "")).strip()] += 1
            length_counts[str(row.get("window_length_frames", "")).strip()] += 1
    counts = Counter(values)
    digest = hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()
    return values, {
        "path": str(path),
        "sha256": sha256(path),
        "rows": len(values),
        "blank_keys": counts.get("", 0),
        "duplicate_key_rows": sum(c for v, c in counts.items() if v and c > 1),
        "ordered_key_sha256": digest,
        "behavior_counts": dict(Counter(labels)),
        "trainable_count": sum(v.lower() in {"true", "1", "yes"} for v in valid),
        "source_counts": dict(source_counts),
        "window_length_counts": dict(length_counts),
        "labels": labels,
        "valid": valid,
    }


def compare_file_hashes(old: Path, new: Path) -> dict[str, object]:
    return {
        "old": {"path": str(old), "sha256": sha256(old), "size_bytes": old.stat().st_size},
        "new": {"path": str(new), "sha256": sha256(new), "size_bytes": new.stat().st_size},
        "byte_equal": sha256(old) == sha256(new),
    }


def main() -> None:
    effective_path = MATERIALIZED / "effective_window_index.csv"
    split_path = MATERIALIZED / "split_manifest.csv"
    y_path = MATERIALIZED / "y_behavior.csv"
    train_mask_path = MATERIALIZED / "train_mask.csv"
    base_sequence_path = Path(
        r"C:/pig_runs/classification_v2_reviewed_rebuild_20260802_v1/"
        r"candidates/train_ready/sequence_window_manifest.csv"
    )
    effective_ids, effective = scan_effective(effective_path)
    split_ids, split = scan_csv(split_path, "window_id")
    y_values, y = scan_aligned_column(y_path, "behavior_window_label")
    mask_values, mask = scan_aligned_column(train_mask_path, "window_valid_for_main_train")
    sequence_ids, sequence = scan_csv(base_sequence_path, "window_id")
    effective_labels = effective.pop("labels")
    effective_valid = effective.pop("valid")
    y_match = len(y_values) == len(effective_labels) and y_values == effective_labels
    mask_match = len(mask_values) == len(effective_valid) and mask_values == effective_valid

    role_old = OLD_ROLES / "native_unit_outer_roles.csv"
    role_new = CURRENT_AUDIT / "four_fold_roles_inner4/native_unit_outer_roles.csv"
    assignment_old = OLD_ROLES / "native_unit_outer_assignments.csv"
    assignment_new = CURRENT_AUDIT / "four_fold_roles_inner4/native_unit_outer_assignments.csv"
    weight_old = OLD_WEIGHTS / "fold_event_weight_manifest.csv"
    weight_new = CURRENT_AUDIT / "fold_event_weights_inner4/fold_event_weight_manifest.csv"
    role_manifest = json.loads((CURRENT_AUDIT / "four_fold_roles_inner4/four_fold_role_manifest.json").read_text())
    rgb = json.loads(RGB_MANIFEST.read_text())
    social = json.loads(SOCIAL_MANIFEST.read_text())
    expected_order_hash = effective["ordered_key_sha256"]
    social_shapes = {}
    for name, item in social.get("files", {}).items():
        path = Path(str(item["path"]))
        if path.exists():
            array = np.load(path, mmap_mode="r")
            social_shapes[name] = {
                "exists": True,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "expected_shape": item.get("shape"),
                "shape_match": list(array.shape) == item.get("shape"),
            }
        else:
            social_shapes[name] = {"exists": False, "path": str(path)}

    checks = {
        "effective_vs_split": compare_order(effective_ids, split_ids),
        "effective_vs_base_sequence": compare_order(effective_ids, sequence_ids),
        "y_row_and_label_alignment": {"rows_equal": len(y_values) == len(effective_labels), "values_equal": y_match},
        "train_mask_row_and_value_alignment": {"rows_equal": len(mask_values) == len(effective_valid), "values_equal": mask_match},
        "rgb_window_order": {
            "manifest_status": rgb.get("status"),
            "manifest_window_order_matches_input": rgb.get("window_order_matches_input"),
            "manifest_window_order_hash": rgb.get("window_order_hash"),
            "expected_window_order_hash": expected_order_hash,
            "hash_equal": rgb.get("window_order_hash") == expected_order_hash,
            "window_rows": rgb.get("window_rows"),
            "missing_or_extra_window_ids": rgb.get("missing_or_extra_window_ids"),
            "duplicate_window_ids": rgb.get("duplicate_window_ids"),
        },
        "social_window_order": {
            "manifest_window_source_sha256": social.get("window_source", {}).get("sha256"),
            "effective_sha256": effective["sha256"],
            "source_hash_equal": social.get("window_source", {}).get("sha256") == effective["sha256"],
            "manifest_window_order_hash": social.get("window_id_order_sha256"),
            "expected_window_order_hash": expected_order_hash,
            "hash_equal": social.get("window_id_order_sha256") == expected_order_hash,
            "window_count": social.get("window_count"),
            "top_k": social.get("top_k"),
            "schema_version": social.get("social_schema_version"),
            "array_checks": social_shapes,
        },
        "rebound_roles": {
            "manifest_code_sha": role_manifest.get("code_authority", {}).get("git_sha"),
            "current_code_sha": "59684f7c8e232d6b54e0d28fd117453b771da4ab",
            "inner_fold_count": role_manifest.get("inner_fold_count"),
            "role_manifest_valid": role_manifest.get("audit_valid"),
            "role_assignment": compare_file_hashes(assignment_old, assignment_new),
            "role_rows": compare_file_hashes(role_old, role_new),
        },
        "rebound_event_weights": {
            "weight_rows": compare_file_hashes(weight_old, weight_new),
            "audit_valid": json.loads((CURRENT_AUDIT / "fold_event_weights_inner4/fold_event_weight_audit.json").read_text()).get("valid"),
        },
    }
    result = {
        "schema_version": "classification_v2.px2_rebind_audit.v1",
        "status": "PASS" if all(
            [
                checks["effective_vs_split"]["exact_order_equal"],
                checks["effective_vs_base_sequence"]["exact_order_equal"],
                y_match,
                mask_match,
                checks["rgb_window_order"]["hash_equal"],
                checks["social_window_order"]["source_hash_equal"],
                checks["social_window_order"]["hash_equal"],
                checks["rebound_roles"]["role_rows"]["byte_equal"],
                checks["rebound_event_weights"]["weight_rows"]["byte_equal"],
                checks["rebound_event_weights"]["audit_valid"],
            ]
        ) else "FAIL",
        "current_code_sha": "59684f7c8e232d6b54e0d28fd117453b771da4ab",
        "effective": effective,
        "split": split,
        "y_behavior": y,
        "train_mask": mask,
        "base_sequence": sequence,
        "checks": checks,
        "authority_paths": {
            "effective_window_index": str(effective_path),
            "base_sequence_manifest": str(base_sequence_path),
            "rgb_manifest": str(RGB_MANIFEST),
            "social_manifest": str(SOCIAL_MANIFEST),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
