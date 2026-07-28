"""Audit Hidden-manifest sensitivity without mutating review authorities."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.review.hidden_review_builder import (
    HiddenReviewConfig,
    build_hidden_review_manifest,
)
from pig_behavior.classification_v2.review.hidden_review_science import (
    build_hidden_scientific_design,
    load_hidden_scientific_policy,
)

IDENTITY_COLUMNS = (
    "source_type",
    "dataset_id",
    "video_key",
    "frame_index",
    "pig_id",
    "object_track_key",
    "track_id",
    "object_id_in_image",
)
MEDIA_COLUMNS = (
    "source_video_path",
    "image_name",
    "crop_path",
    "scene_frame_uid",
    "frame_uid",
)
SELECTION_COLUMNS = (
    "hidden_review_cohort",
    "hidden_sampling_design",
    "hidden_review_priority",
    "hidden_review_stratum_key",
    "hidden_false_negative_risk_band",
    "hidden_sampling_stratum",
    "hidden_false_negative_risk_score",
    "hidden_false_negative_risk_reasons",
)
REQUIRED_INPUT_COLUMNS = (
    "source_type",
    "dataset_id",
    "video_key",
    "frame_uid",
    "frame_index",
    "pig_id",
    "behavior",
    "hidden",
    "object_track_key",
    "track_id",
    "object_id_in_image",
    "temporal_unit_key",
    "bbox_was_clipped",
    "nearest_pair_iou",
    "nearest_pair_overlap_ratio",
    "nearest_dist_n",
    "pair_contact_with_nearest",
    "cx_n",
    "cy_n",
    "bw_n",
    "bh_n",
    "area_n",
    "aspect_ratio",
    "hidden_source",
    "hidden_is_trusted",
    "hidden_review_status",
    "hidden_trust_status",
)
OPTIONAL_INPUT_COLUMNS = (
    "cx",
    "cy",
    "bbox_w",
    "bbox_h",
    *MEDIA_COLUMNS,
)
MOTION_PERTURBATION_COLUMNS = (
    "shape_change_score",
    "delta_area_n",
    "displacement_n",
    "speed_n_per_frame",
    "speed_n_per_second",
    "tangential_acceleration_n_per_second2",
    "path_length_n_unit",
    "approach_speed_n_per_frame",
    "approach_speed_n_per_second",
    "roi_target_entry_event",
    "pen_approach_speed_n_per_second",
)

SELECTION_INPUT_COLUMN_CLASSES = {
    "image_visibility_only": (
        "hidden",
        "bbox_was_clipped",
        "hidden_source",
        "hidden_is_trusted",
        "hidden_review_status",
        "hidden_trust_status",
    ),
    "geometry": (
        "cx_n",
        "cx",
        "cy_n",
        "cy",
        "bw_n",
        "bbox_w",
        "bh_n",
        "bbox_h",
        "area_n",
        "aspect_ratio",
    ),
    "motion": (),
    "temporal": ("frame_index", "temporal_unit_key"),
    "roi": (),
    "social": (
        "nearest_pair_iou",
        "nearest_pair_overlap_ratio",
        "nearest_dist_n",
        "pair_contact_with_nearest",
    ),
    "source_metadata": ("source_type", "dataset_id", "video_key"),
    "review_identity": (
        "frame_uid",
        "pig_id",
        "object_track_key",
        "track_id",
        "object_id_in_image",
    ),
    "descriptive_not_selection": ("behavior",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-features-csv", type=Path, required=True)
    parser.add_argument("--old-manifest-csv", type=Path, required=True)
    parser.add_argument("--old-template-audit-json", type=Path, required=True)
    parser.add_argument("--old-decisions-csv", type=Path)
    parser.add_argument("--scientific-policy-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        args.frame_features_csv,
        args.old_manifest_csv,
        args.old_template_audit_json,
        args.scientific_policy_json,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.old_decisions_csv is not None and not args.old_decisions_csv.is_file():
        raise FileNotFoundError(args.old_decisions_csv)

    header = pd.read_csv(args.frame_features_csv, nrows=0).columns
    missing = sorted(set(REQUIRED_INPUT_COLUMNS).difference(header))
    if missing:
        raise ValueError(f"Hidden impact input missing columns: {missing}")
    requested = set(REQUIRED_INPUT_COLUMNS)
    requested.update(OPTIONAL_INPUT_COLUMNS)
    requested.update(MOTION_PERTURBATION_COLUMNS)
    usecols = [column for column in header if column in requested]
    frames = pd.read_csv(
        args.frame_features_csv,
        usecols=usecols,
        low_memory=False,
    )

    config = _load_config(args.old_template_audit_json)
    rebuilt, _, rebuilt_audit = build_hidden_review_manifest(
        frames,
        config=config,
    )
    _, policy_payload, policy_sha256 = load_hidden_scientific_policy(
        args.scientific_policy_json
    )
    rebuilt_bytes = rebuilt.to_csv(
        index=False,
        lineterminator="\n",
    ).encode("utf-8")
    scientific_design = build_hidden_scientific_design(
        rebuilt,
        manifest_sha256=hashlib.sha256(rebuilt_bytes).hexdigest(),
        policy_payload=policy_payload,
        policy_sha256=policy_sha256,
        selection_contract=rebuilt_audit["selection_contract"],
        require_final_support=True,
    )
    perturbed = frames.copy()
    perturbed_columns = [
        column
        for column in MOTION_PERTURBATION_COLUMNS
        if column in perturbed.columns
    ]
    for column in perturbed_columns:
        perturbed[column] = 1_000_000.0
    perturbed_manifest, _, _ = build_hidden_review_manifest(
        perturbed,
        config=config,
    )

    old_columns = [
        "hidden_review_item_id",
        *IDENTITY_COLUMNS,
        *MEDIA_COLUMNS,
        *SELECTION_COLUMNS,
    ]
    old_header = pd.read_csv(args.old_manifest_csv, nrows=0).columns
    old_usecols = [column for column in old_columns if column in old_header]
    old_manifest = pd.read_csv(
        args.old_manifest_csv,
        usecols=old_usecols,
        low_memory=False,
    )
    comparison = _compare_manifests(old_manifest, rebuilt)
    perturbation = _compare_manifests(rebuilt, perturbed_manifest)
    decision_audit = _decision_carry_audit(
        args.old_decisions_csv,
        old_manifest,
        rebuilt,
        comparison,
    )
    classified_columns = {
        category: [column for column in columns if column in frames.columns]
        for category, columns in SELECTION_INPUT_COLUMN_CLASSES.items()
    }
    payload = {
        "schema_version": "classification_v2.hidden_motion_impact_audit.v1",
        "frame_features_csv": str(args.frame_features_csv),
        "old_manifest_csv": str(args.old_manifest_csv),
        "old_decisions_csv": (
            str(args.old_decisions_csv)
            if args.old_decisions_csv is not None
            else None
        ),
        "loaded_columns": usecols,
        "selection_input_columns_by_class": classified_columns,
        "motion_perturbation_columns": perturbed_columns,
        "external_motion_dependency_after_patch": False,
        "rebuilt_manifest_audit_errors": rebuilt_audit["errors"],
        "rebuilt_scientific_design": scientific_design,
        "old_vs_rebuilt": comparison,
        "motion_perturbation": perturbation,
        "motion_perturbation_invariant": _is_exact_selection_match(
            perturbation
        ),
        "decision_carry_forward": decision_audit,
        "errors": [],
    }
    if rebuilt_audit["errors"]:
        payload["errors"].append("rebuilt_manifest_audit_failed")
    if not scientific_design["planned_support_meets_final_gate"]:
        payload["errors"].append("rebuilt_scientific_support_failed")
    if not payload["motion_perturbation_invariant"]:
        payload["errors"].append("motion_perturbation_changed_manifest")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if payload["errors"]:
        raise ValueError(f"Hidden impact audit failed: {payload['errors']}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _load_config(path: Path) -> HiddenReviewConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("config", {})
    allowed = {field.name for field in fields(HiddenReviewConfig)}
    values = {key: value for key, value in raw.items() if key in allowed}
    if "stratum_columns" in values:
        values["stratum_columns"] = tuple(values["stratum_columns"])
    config = HiddenReviewConfig(**values)
    config.validate()
    return config


def _compare_manifests(
    old: pd.DataFrame,
    new: pd.DataFrame,
) -> dict[str, Any]:
    key = "hidden_review_item_id"
    old_keys = set(old[key].astype(str))
    new_keys = set(new[key].astype(str))
    shared = old_keys.intersection(new_keys)
    joined = old.loc[old[key].astype(str).isin(shared)].merge(
        new.loc[new[key].astype(str).isin(shared)],
        on=key,
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    comparable = [
        column
        for column in (*IDENTITY_COLUMNS, *MEDIA_COLUMNS, *SELECTION_COLUMNS)
        if f"{column}_old" in joined and f"{column}_new" in joined
    ]
    changes = {
        column: _difference_count(
            joined[f"{column}_old"],
            joined[f"{column}_new"],
        )
        for column in comparable
    }
    changed_item_ids = {
        column: _changed_item_ids(joined, key, column)
        for column in comparable
    }
    identity_mismatch_ids = _union_changed_ids(
        changed_item_ids,
        IDENTITY_COLUMNS,
    )
    media_mismatch_ids = _union_changed_ids(
        changed_item_ids,
        MEDIA_COLUMNS,
    )
    return {
        "old_item_count": int(len(old)),
        "new_item_count": int(len(new)),
        "exact_review_key_intersection": int(len(shared)),
        "old_only": int(len(old_keys.difference(new_keys))),
        "new_only": int(len(new_keys.difference(old_keys))),
        "old_only_item_ids": sorted(old_keys.difference(new_keys)),
        "new_only_item_ids": sorted(new_keys.difference(old_keys)),
        "span_mismatch": changes.get("frame_index", 0),
        "source_video_actor_mismatch": len(identity_mismatch_ids),
        "visual_media_authority_mismatch": len(media_mismatch_ids),
        "stratum_changes": {
            column: changes.get(column, 0)
            for column in (
                "hidden_review_stratum_key",
                "hidden_false_negative_risk_band",
                "hidden_sampling_stratum",
            )
        },
        "priority_changes": changes.get("hidden_review_priority", 0),
        "selection_reason_changes": {
            "cohort": changes.get("hidden_review_cohort", 0),
            "sampling_design": changes.get("hidden_sampling_design", 0),
            "risk_score": changes.get("hidden_false_negative_risk_score", 0),
            "risk_reasons": changes.get(
                "hidden_false_negative_risk_reasons",
                0,
            ),
        },
        "column_change_counts": changes,
        "changed_item_ids": {
            column: changed_item_ids.get(column, [])
            for column in SELECTION_COLUMNS
        },
        "old_cohort_counts": _counts(old, "hidden_review_cohort"),
        "new_cohort_counts": _counts(new, "hidden_review_cohort"),
    }


def _difference_count(left: pd.Series, right: pd.Series) -> int:
    left_text = left.fillna("").astype(str)
    right_text = right.fillna("").astype(str)
    return int(left_text.ne(right_text).sum())


def _changed_item_ids(
    joined: pd.DataFrame,
    key: str,
    column: str,
) -> list[str]:
    left = joined[f"{column}_old"].fillna("").astype(str)
    right = joined[f"{column}_new"].fillna("").astype(str)
    return sorted(joined.loc[left.ne(right), key].astype(str).tolist())


def _union_changed_ids(
    changed_item_ids: dict[str, list[str]],
    columns: tuple[str, ...],
) -> set[str]:
    combined: set[str] = set()
    for column in columns:
        combined.update(changed_item_ids.get(column, []))
    return combined


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    counts = frame[column].fillna("<NA>").astype(str).value_counts()
    return {str(key): int(value) for key, value in counts.sort_index().items()}


def _is_exact_selection_match(comparison: dict[str, Any]) -> bool:
    selection_changes = comparison["selection_reason_changes"]
    return bool(
        comparison["old_only"] == 0
        and comparison["new_only"] == 0
        and comparison["priority_changes"] == 0
        and all(value == 0 for value in comparison["stratum_changes"].values())
        and all(value == 0 for value in selection_changes.values())
    )


def _decision_carry_audit(
    decisions_path: Path | None,
    old_manifest: pd.DataFrame,
    new_manifest: pd.DataFrame,
    comparison: dict[str, Any],
) -> dict[str, Any]:
    key = "hidden_review_item_id"
    old_keys = set(old_manifest[key].astype(str))
    new_keys = set(new_manifest[key].astype(str))
    exact_identity_and_media = bool(
        comparison["source_video_actor_mismatch"] == 0
        and comparison["visual_media_authority_mismatch"] == 0
        and comparison["span_mismatch"] == 0
    )
    if decisions_path is None:
        return {
            "decision_rows": None,
            "exact_carryable_rows": None,
            "new_items_requiring_review": int(len(new_keys.difference(old_keys))),
            "policy": "INCONCLUSIVE_NO_DECISIONS",
        }
    decisions = pd.read_csv(decisions_path, low_memory=False)
    decision_ids = decisions[key].astype(str)
    duplicate = int(decision_ids.duplicated(keep=False).sum())
    carryable = set(decision_ids).intersection(new_keys)
    new_items = new_keys.difference(old_keys)
    full = bool(
        duplicate == 0
        and exact_identity_and_media
        and new_keys == old_keys
        and new_keys.issubset(set(decision_ids))
    )
    return {
        "decision_rows": int(len(decisions)),
        "duplicate_decision_rows": duplicate,
        "exact_carryable_rows": int(len(carryable)),
        "old_decisions_without_new_item": int(
            len(set(decision_ids).difference(new_keys))
        ),
        "new_items_requiring_review": int(len(new_items)),
        "fully_carryable": full,
        "sampling_rationale_changed": not _is_exact_selection_match(
            comparison
        ),
        "policy": (
            "FULL_EXACT_MEDIA_CARRY"
            if full
            else "EXACT_INTERSECTION_ONLY"
        ),
    }


if __name__ == "__main__":
    main()
