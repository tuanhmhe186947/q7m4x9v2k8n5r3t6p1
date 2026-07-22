"""Run bounded Group A and exact-view smoke before behavior review.

This script reads an existing frame artifact without modifying it.  Human
review fields used for final-view checks are an explicit in-memory smoke
overlay and are never published as decisions or official review authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.features.pen_context import (
    DEFAULT_PEN_MASK_SHA256,
    audit_pen_context_features,
    build_pen_context_features,
)
from pig_behavior.classification_v2.features.pig_strenet_artifacts import (
    build_pig_strenet_artifacts,
)
from pig_behavior.classification_v2.features.sequence_windows import (
    audit_sequence_windows,
    build_sequence_windows,
)
from pig_behavior.classification_v2.features.spatiotemporal import (
    audit_enhanced_spatiotemporal_features,
    build_enhanced_spatiotemporal_features,
)
from pig_behavior.classification_v2.features.temporal_harmonization import (
    build_temporal_label_intervals,
    harmonize_temporal_labels,
)
from pig_behavior.classification_v2.review.review_authority import (
    SMOKE_SCOPE,
    audit_frame_local_schema,
    build_review_authority_manifest,
)
from pig_behavior.classification_v2.sources.temporal_provenance import (
    CANONICAL_TIMESTAMP_SOURCE,
)

INDEX_COLUMNS = (
    "source_type",
    "dataset_id",
    "video_key",
    "object_track_key",
    "temporal_unit_key",
    "frame_index",
    "behavior_temporal_final",
    "behavior",
    "hidden",
    "hidden_is_trusted",
    "bbox_valid",
    "is_interaction_behavior",
)

FRAME_LOCAL_COLUMNS = (
    "source_type",
    "dataset_id",
    "video_key",
    "scene_frame_uid",
    "frame_uid",
    "image_key",
    "image_name",
    "object_id_in_image",
    "frame_index",
    "relative_frame_index",
    "native_offset",
    "image_width",
    "image_height",
    "pig_id",
    "track_id",
    "track_label",
    "x1_raw",
    "y1_raw",
    "x2_raw",
    "y2_raw",
    "x1",
    "y1",
    "x2",
    "y2",
    "bbox_valid",
    "bbox_was_clipped",
    "bbox_w",
    "bbox_h",
    "bbox_area",
    "cx",
    "cy",
    "cx_n",
    "cy_n",
    "bw_n",
    "bh_n",
    "area_n",
    "aspect_ratio",
    "box_diag",
    "box_diag_n",
    "box_compactness",
    "behavior",
    "behavior_coarse",
    "hidden",
    "hidden_source",
    "hidden_review_status",
    "hidden_is_trusted",
    "hidden_trust_status",
    "visibility_quality",
    "crop_path",
    "source_video_path",
    "global_context_pig_count",
    "local_context_pig_count",
    "geometry_feature_valid",
    "geometry_quality",
    "roi_feeder_available",
    "roi_feeder_min_dist_n",
    "roi_feeder_max_overlap_ratio",
    "roi_feeder_max_iou",
    "roi_feeder_center_inside",
    "roi_drinker_available",
    "roi_drinker_min_dist_n",
    "roi_drinker_max_overlap_ratio",
    "roi_drinker_max_iou",
    "roi_drinker_center_inside",
    "roi_toy_available",
    "roi_toy_min_dist_n",
    "roi_toy_max_overlap_ratio",
    "roi_toy_max_iou",
    "roi_toy_center_inside",
    "roi_feeder_near",
    "roi_feeder_contact",
    "roi_drinker_near",
    "roi_drinker_contact",
    "roi_toy_near",
    "roi_toy_contact",
    "roi_feature_required",
    "roi_target_class",
    "roi_target_available",
    "roi_target_min_dist_n",
    "roi_target_max_overlap_ratio",
    "roi_target_max_iou",
    "roi_target_center_inside",
    "roi_target_near",
    "roi_target_contact",
    "roi_context_quality",
    "roi_feature_valid",
    "object_track_key",
    "temporal_label_mode",
    "label_anchor_frame_index",
    "label_window_start",
    "label_window_end",
    "temporal_unit_key",
    "nearest_pig_id",
    "nearest_track_id",
    "nearest_dist_n",
    "nearest_pair_iou",
    "nearest_pair_overlap_ratio",
    "social_density_near_count",
    "social_contact_count",
    "social_context_frame_size",
    "pair_contact_with_nearest",
    "pen_center_signed_distance_n",
    "pen_center_clearance_box_ratio",
    "pen_bbox_inside_ratio",
    "pen_boundary_inward_normal_x",
    "pen_boundary_inward_normal_y",
    "pen_center_inside",
    "pen_near_boundary",
    "pen_context_available",
    "pen_context_quality_valid",
    "spatiotemporal_feature_valid",
    "include_in_training",
    "sample_weight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--code-authority-sha", required=True)
    parser.add_argument("--code-dirty", action="store_true")
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument(
        "--pen-mask",
        type=Path,
        default=Path("data/annotations/scene/mask.png"),
    )
    parser.add_argument(
        "--expected-pen-mask-sha256",
        default=DEFAULT_PEN_MASK_SHA256,
    )
    parser.add_argument("--chunksize", type=int, default=25_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.frame_csv.is_file():
        raise FileNotFoundError(args.frame_csv)
    if not args.pen_mask.is_file():
        raise FileNotFoundError(args.pen_mask)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    if args.source_fps <= 0:
        raise ValueError("--source-fps must be positive")
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be positive")

    index = pd.read_csv(
        args.frame_csv,
        usecols=lambda column: column in INDEX_COLUMNS,
        low_memory=False,
    )
    selection = _select_representative_scope(index)
    selected_rows = _read_selected_rows(
        args.frame_csv,
        set(selection["selected_unit_keys"]),
        chunksize=args.chunksize,
    )
    frame_local = _frame_local_primitives(
        selected_rows,
        source_fps=args.source_fps,
    )
    frame_local_errors = audit_frame_local_schema(frame_local.columns)
    if frame_local_errors:
        raise ValueError(frame_local_errors)

    native_evidence = build_enhanced_spatiotemporal_features(frame_local)
    native_evidence = build_pen_context_features(
        native_evidence,
        mask_path=args.pen_mask,
        expected_mask_sha256=args.expected_pen_mask_sha256,
    )
    native_audit = audit_enhanced_spatiotemporal_features(native_evidence)
    pen_audit = audit_pen_context_features(
        native_evidence,
        mask_path=args.pen_mask,
        expected_mask_sha256=args.expected_pen_mask_sha256,
        input_rows=len(frame_local),
    )
    native_boundary_audit = _native_boundary_audit(native_evidence)
    native_sample_manifest = _native_sample_manifest(
        native_evidence,
        selection["native_sample_keys"],
    )

    view_samples, view_errors = _build_view_smoke(
        frame_local,
        selection,
    )
    pig_artifacts = build_pig_strenet_artifacts(native_evidence)
    pig_manifest = {
        "schema_version": "classification_v2.pig_strenet_smoke.v1",
        "smoke_only": True,
        "pair_rows": int(len(pig_artifacts.pair_manifest)),
        "slot_rows": int(len(pig_artifacts.slot_manifest)),
        "history_feature_rows": int(len(pig_artifacts.history_features)),
        "roi_dynamic_rows": int(len(pig_artifacts.roi_dynamics)),
        "social_node_rows": int(len(pig_artifacts.social_nodes)),
        "social_edge_rows": int(len(pig_artifacts.social_edges)),
        "audit": pig_artifacts.audit,
    }

    harmonized = harmonize_temporal_labels(frame_local)
    intervals = build_temporal_label_intervals(harmonized)
    review_units = _smoke_review_units(intervals)
    media = _media_authority(frame_local)
    hidden = _hidden_authority(frame_local)
    timestamp_contract = {
        "schema_version": "classification_v2.timestamp_fps_contract.v1",
        "lineage_id": "pre_behavior_review_representative_smoke",
        "source_fps": float(args.source_fps),
        "source_frame_index_authority": "decoded_video_frame_index",
        "formula": "timestamp_sec=source_frame_index/source_fps",
        "times_txt_role": "acquisition_audit_only_not_motion_clock",
        "errors": [],
        "valid": True,
    }
    evidence_semantics = {
        "lineage_id": "pre_behavior_review_representative_smoke",
        "evidence_column_semantic_version": (
            "classification_v2.native_review_evidence.v2"
        ),
        "feature_computation_grain": "NATIVE_UNIT_REVIEW_EVIDENCE",
        "pair_scope": "exact_temporal_unit_key",
        "final_view_aggregate_reuse_allowed": False,
        "smoke_review_overlay_is_human_authority": False,
        "errors": [],
        "valid": True,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = _write_smoke_artifacts(
        args.output_dir,
        frame_local=frame_local,
        hidden=hidden,
        harmonized=harmonized,
        intervals=intervals,
        review_units=review_units,
        media=media,
        native_evidence=native_evidence,
        native_samples=native_sample_manifest,
        view_samples=view_samples,
        pig_manifest=pig_manifest,
        timestamp_contract=timestamp_contract,
        evidence_semantics=evidence_semantics,
    )
    authority = build_review_authority_manifest(
        code_authority_sha=args.code_authority_sha,
        code_dirty=args.code_dirty,
        lineage_id="pre_behavior_review_representative_smoke",
        authority_scope=SMOKE_SCOPE,
        source_artifacts={
            "input_frame_artifact": args.frame_csv,
            "pen_mask": args.pen_mask,
        },
        artifacts={
            "frame_local": paths["frame_local"],
            "hidden_reviewed_frames": paths["hidden"],
            "harmonized_frames": paths["harmonized"],
            "temporal_native_units": paths["intervals"],
            "pig_strenet_evidence": paths["pig_manifest"],
            "behavior_review_units": paths["review_units"],
            "media_authority": paths["media"],
        },
        timestamp_fps_contract=timestamp_contract,
        evidence_semantics=evidence_semantics,
    )
    authority_path = args.output_dir / "review_authority_smoke.json"
    _write_json(authority_path, authority)

    errors = [
        *frame_local_errors,
        *native_audit.get("errors", []),
        *pen_audit.get("errors", []),
        *native_boundary_audit["errors"],
        *view_errors,
        *pig_artifacts.audit.get("errors", []),
        *authority["errors"],
    ]
    report = {
        "schema_version": "classification_v2.pre_behavior_review_gate.v1",
        "code_authority_sha": args.code_authority_sha,
        "code_dirty": bool(args.code_dirty),
        "input_frame_csv": str(args.frame_csv),
        "input_read_only": True,
        "failed_v3_reused_as_authority": False,
        "official_lineage_created": False,
        "full_final_view_corpus_built": False,
        "review_overlay": "SMOKE_ONLY_NOT_HUMAN_DECISIONS",
        "selected_scope": selection,
        "frame_local_rows": int(len(frame_local)),
        "native_evidence_rows": int(len(native_evidence)),
        "native_evidence_audit": native_audit,
        "pen_context_audit": pen_audit,
        "native_boundary_audit": native_boundary_audit,
        "view_results": _view_result_summary(view_samples),
        "pig_strenet_smoke": pig_manifest,
        "review_authority_smoke_sha256": authority[
            "review_authority_sha256"
        ],
        "official_review_authority_created": False,
        "errors": errors,
        "valid": not errors,
    }
    report_path = args.output_dir / "pre_behavior_review_gate_audit.json"
    _write_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(2)


def _select_representative_scope(index: pd.DataFrame) -> dict[str, Any]:
    work = index.copy()
    work["frame_index"] = pd.to_numeric(work["frame_index"], errors="coerce")
    work["behavior_effective"] = work.get(
        "behavior_temporal_final",
        work["behavior"],
    ).fillna(work["behavior"]).astype(str)
    work["hidden_flag"] = _to_bool(work.get("hidden", False))
    work["trusted_flag"] = _to_bool(work.get("hidden_is_trusted", True))
    work["bbox_flag"] = _to_bool(work.get("bbox_valid", True))
    work["interaction_flag"] = _to_bool(
        work.get("is_interaction_behavior", False)
    ) | work["behavior_effective"].isin({"fight", "social-nose"})
    group_columns = [
        "source_type",
        "dataset_id",
        "video_key",
        "object_track_key",
        "temporal_unit_key",
    ]
    units = (
        work.groupby(group_columns, dropna=False, sort=True)
        .agg(
            start_frame=("frame_index", "min"),
            end_frame=("frame_index", "max"),
            observed_frames=("frame_index", "nunique"),
            behavior=("behavior_effective", "first"),
            hidden_risk=("hidden_flag", "max"),
            trusted=("trusted_flag", "min"),
            bbox_valid=("bbox_flag", "min"),
            interaction=("interaction_flag", "max"),
        )
        .reset_index()
    )
    units = units.sort_values(
        ["source_type", "video_key", "object_track_key", "start_frame"],
        kind="mergesort",
    ).reset_index(drop=True)
    legacy = units[
        units["source_type"].eq("legacy_recovered")
        & units["observed_frames"].eq(16)
    ]
    cvat = units[
        units["source_type"].eq("cvat_tracking_xml")
        & units["observed_frames"].eq(6)
    ]
    if len(legacy) < 5 or len(cvat) < 5:
        raise ValueError("representative smoke requires five 16f/6f units")

    stable_run = _find_cvat_run(cvat, same_behavior=True, units_needed=3)
    transition_run = _find_cvat_run(
        cvat,
        same_behavior=False,
        units_needed=2,
    )
    hidden_unit = _first_unit(units[units["hidden_risk"]])
    interaction_unit = _first_unit(units[units["interaction"]])
    legacy_samples = _distinct_unit_keys(
        pd.concat(
            [
                legacy[legacy["interaction"]].head(1),
                legacy[legacy["hidden_risk"]].head(1),
                legacy,
            ],
            ignore_index=True,
        ),
        count=5,
    )
    cvat_samples = _distinct_unit_keys(
        pd.concat(
            [
                stable_run,
                transition_run,
                cvat[cvat["interaction"]].head(1),
                cvat[cvat["hidden_risk"]].head(1),
                cvat,
            ],
            ignore_index=True,
        ),
        count=5,
    )
    native_sample_keys = [*legacy_samples, *cvat_samples]
    selected = set(native_sample_keys)
    selected.update(stable_run["temporal_unit_key"].astype(str))
    selected.update(transition_run["temporal_unit_key"].astype(str))
    selected.add(str(hidden_unit["temporal_unit_key"]))
    selected.add(str(interaction_unit["temporal_unit_key"]))
    return {
        "native_sample_keys": native_sample_keys,
        "selected_unit_keys": sorted(selected),
        "stable_cvat_unit_keys": stable_run[
            "temporal_unit_key"
        ].astype(str).tolist(),
        "transition_cvat_unit_keys": transition_run[
            "temporal_unit_key"
        ].astype(str).tolist(),
        "hidden_risk_unit_key": str(hidden_unit["temporal_unit_key"]),
        "interaction_unit_key": str(interaction_unit["temporal_unit_key"]),
    }


def _find_cvat_run(
    units: pd.DataFrame,
    *,
    same_behavior: bool,
    units_needed: int,
) -> pd.DataFrame:
    for _, track in units.groupby(
        ["video_key", "object_track_key"],
        sort=True,
    ):
        ordered = track.sort_values("start_frame", kind="mergesort")
        records = list(ordered.index)
        for offset in range(0, len(records) - units_needed + 1):
            candidate = ordered.loc[records[offset : offset + units_needed]]
            starts = candidate["start_frame"].astype(int).tolist()
            ends = candidate["end_frame"].astype(int).tolist()
            contiguous = all(
                starts[index] == ends[index - 1] + 1
                for index in range(1, units_needed)
            )
            behaviors = candidate["behavior"].astype(str).tolist()
            behavior_match = len(set(behaviors)) == 1
            quality = (
                bool(candidate["bbox_valid"].all())
                and not bool(candidate["hidden_risk"].any())
            )
            if (
                contiguous
                and quality
                and behavior_match is same_behavior
            ):
                return candidate.copy()
    kind = "stable" if same_behavior else "transition"
    raise ValueError(f"no representative CVAT {kind} run")


def _first_unit(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        raise ValueError("required representative unit not found")
    return frame.sort_values(
        ["source_type", "video_key", "object_track_key", "start_frame"],
        kind="mergesort",
    ).iloc[0]


def _distinct_unit_keys(frame: pd.DataFrame, *, count: int) -> list[str]:
    values = frame["temporal_unit_key"].fillna("").astype(str)
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
        if len(result) == count:
            return result
    raise ValueError(f"could not select {count} distinct native units")


def _read_selected_rows(
    path: Path,
    unit_keys: set[str],
    *,
    chunksize: int,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        mask = chunk["temporal_unit_key"].fillna("").astype(str).isin(unit_keys)
        if mask.any():
            parts.append(chunk.loc[mask].copy())
    if not parts:
        raise ValueError("representative unit selection loaded zero rows")
    return pd.concat(parts, ignore_index=True)


def _frame_local_primitives(
    frame: pd.DataFrame,
    *,
    source_fps: float,
) -> pd.DataFrame:
    available = [column for column in FRAME_LOCAL_COLUMNS if column in frame]
    out = frame.loc[:, available].copy()
    out["source_frame_index"] = pd.to_numeric(
        out["frame_index"],
        errors="raise",
    ).astype(int)
    out["native_offset"] = pd.to_numeric(
        out.get("relative_frame_index", 0),
        errors="coerce",
    )
    out["source_fps"] = float(source_fps)
    out["timestamp_sec"] = out["source_frame_index"] / float(source_fps)
    out["timestamp_source"] = CANONICAL_TIMESTAMP_SOURCE
    out["feature_computation_grain"] = "FRAME_LOCAL_PRIMITIVES"
    out["pair_scope_key"] = ""
    out["pair_recomputed_for_view"] = False
    out["aggregate_recomputed_for_view"] = False
    return out.sort_values(
        ["source_type", "video_key", "object_track_key", "frame_index"],
        kind="mergesort",
    ).reset_index(drop=True)


def _native_boundary_audit(frame: pd.DataFrame) -> dict[str, Any]:
    ordered = frame.sort_values(
        ["temporal_unit_key", "frame_index"],
        kind="mergesort",
    )
    starts = ordered.groupby("temporal_unit_key", sort=False).head(1)
    checks = {
        "unit_start_adjacent_pair_valid": int(
            _to_bool(starts["adjacent_motion_pair_valid"]).sum()
        ),
        "unit_start_sparse_pair_valid": int(
            _to_bool(starts["sparse_velocity_pair_valid"]).sum()
        ),
        "unit_start_nonzero_speed": int(
            pd.to_numeric(starts["speed_n_per_second"], errors="coerce")
            .fillna(0.0)
            .ne(0.0)
            .sum()
        ),
        "unit_start_nonzero_acceleration": int(
            pd.to_numeric(
                starts["acceleration_n_per_second2"],
                errors="coerce",
            )
            .fillna(0.0)
            .ne(0.0)
            .sum()
        ),
        "unit_start_roi_transition": int(
            _to_bool(starts["roi_target_entry_event"]).sum()
        ),
        "unit_start_pen_motion_valid": int(
            _to_bool(starts["pen_velocity_context_valid"]).sum()
        ),
    }
    errors = [f"{name}={value}" for name, value in checks.items() if value]
    return {
        "native_units": int(frame["temporal_unit_key"].nunique()),
        **checks,
        "errors": errors,
    }


def _native_sample_manifest(
    frame: pd.DataFrame,
    unit_keys: list[str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for key in unit_keys:
        unit = frame.loc[frame["temporal_unit_key"].astype(str).eq(key)].sort_values(
            "frame_index",
            kind="mergesort",
        )
        first = unit.iloc[0]
        records.append(
            {
                "source_type": first["source_type"],
                "video_key": first["video_key"],
                "object_track_key": first["object_track_key"],
                "temporal_unit_key": key,
                "behavior": first.get("behavior_temporal_final", first["behavior"]),
                "selected_source_frame_indices": _json_ints(unit["frame_index"]),
                "selected_timestamps_seconds": _json_floats(
                    unit["timestamp_sec"]
                ),
                "feature_computation_grain": first[
                    "feature_computation_grain"
                ],
                "pair_scope_matches_unit": bool(
                    unit["pair_scope_key"].astype(str).eq(key).all()
                ),
                "first_pair_valid": bool(
                    _to_bool(unit.head(1)["motion_velocity_pair_valid"]).iloc[0]
                ),
                "adjacent_pair_count": int(
                    _to_bool(unit["adjacent_motion_pair_valid"]).sum()
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _build_view_smoke(
    frame_local: pd.DataFrame,
    selection: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    stable_keys = set(selection["stable_cvat_unit_keys"])
    stable = frame_local.loc[
        frame_local["temporal_unit_key"].astype(str).isin(stable_keys)
    ].copy()
    stable = _review_overlay(stable)
    start = int(stable["frame_index"].min())
    _, _, stable_windows = build_sequence_windows(
        stable,
        window_lengths=[6, 8, 12, 16],
        behavior_review_requirement="full_native_unit_review_required",
    )
    for view in ("T6_contiguous", "T8_contiguous", "T12_contiguous", "T16_contiguous"):
        candidate = stable_windows.loc[
            stable_windows["view_type"].eq(view)
            & stable_windows["window_start_frame"].eq(start)
        ]
        if candidate.empty:
            errors.append(f"representative_view_missing={view}")
            continue
        records.append(_view_record(candidate.iloc[0], scenario="stable_reviewed"))
    errors.extend(audit_sequence_windows(stable_windows)["errors"])

    legacy_key = next(
        key
        for key in selection["native_sample_keys"]
        if key.startswith("legacy_recovered|")
    )
    legacy = _review_overlay(
        frame_local.loc[
            frame_local["temporal_unit_key"].astype(str).eq(legacy_key)
        ].copy()
    )
    _, _, legacy_windows = build_sequence_windows(
        legacy,
        window_lengths=[16],
        behavior_review_requirement="full_native_unit_review_required",
        include_legacy_sparse_s6_at16=True,
    )
    sparse = legacy_windows.loc[legacy_windows["view_type"].eq("S6@16")]
    if sparse.empty:
        errors.append("representative_view_missing=S6@16")
    else:
        records.append(_view_record(sparse.iloc[0], scenario="stable_reviewed"))
    errors.extend(audit_sequence_windows(legacy_windows)["errors"])

    transition_keys = set(selection["transition_cvat_unit_keys"])
    transition = frame_local.loc[
        frame_local["temporal_unit_key"].astype(str).isin(transition_keys)
    ].copy()
    transition = _review_overlay(transition)
    _, _, transition_windows = build_sequence_windows(
        transition,
        window_lengths=[8],
        behavior_review_requirement="full_native_unit_review_required",
    )
    transition_row = transition_windows.sort_values("window_start_frame").iloc[0]
    records.append(_view_record(transition_row, scenario="behavior_transition"))
    if bool(transition_row["window_valid_for_main_train"]):
        errors.append("transition_window_silently_main_train_eligible")

    pending = stable.copy()
    second_key = selection["stable_cvat_unit_keys"][1]
    pending.loc[
        pending["temporal_unit_key"].astype(str).eq(second_key),
        "behavior_review_decision_present",
    ] = False
    _, _, pending_windows = build_sequence_windows(
        pending,
        window_lengths=[8],
        behavior_review_requirement="full_native_unit_review_required",
    )
    pending_row = pending_windows.loc[
        pending_windows["window_start_frame"].eq(start)
    ].iloc[0]
    records.append(_view_record(pending_row, scenario="constituent_pending"))
    if bool(pending_row["window_valid_for_main_train"]):
        errors.append("pending_constituent_silently_main_train_eligible")

    return pd.DataFrame.from_records(records), errors


def _review_overlay(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    labels = out.get("behavior_temporal_final", out["behavior"]).fillna(
        out["behavior"]
    )
    out["behavior_review_decision_present"] = True
    out["behavior_review_label_resolved"] = True
    out["behavior_review_include_in_training"] = True
    out["behavior_reviewed_final"] = labels.astype(str)
    return out


def _view_record(row: pd.Series, *, scenario: str) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "source_type": row["source_type"],
        "video_key": row["video_key"],
        "object_track_key": row["object_track_key"],
        "constituent_native_unit_keys": row[
            "constituent_native_unit_keys"
        ],
        "selected_source_frame_indices": row["selected_frame_indices"],
        "selected_timestamps_seconds": row[
            "selected_timestamps_seconds"
        ],
        "view_type": row["view_type"],
        "sampling_pattern": row["sampling_pattern"],
        "pair_delta_frames": row["pair_delta_frames"],
        "pair_delta_seconds": row["pair_delta_seconds"],
        "physical_span_seconds": row["physical_span_seconds"],
        "adjacent_pair_count": row[
            "adjacent_motion_pair_count_window"
        ],
        "sparse_pair_count": row["sparse_velocity_pair_count_window"],
        "behavior_review_status": row[
            "human_reviewed_behavior_consistency_status"
        ],
        "hidden_policy_tier": row["hidden_window_policy_tier"],
        "pair_recomputed_for_view": row["pair_recomputed_for_view"],
        "aggregate_recomputed_for_view": row[
            "aggregate_recomputed_for_view"
        ],
        "final_structural_eligibility": row[
            "window_valid_for_main_train"
        ],
        "window_exclusion_reason": row["window_exclusion_reason"],
    }


def _smoke_review_units(intervals: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in (
            "temporal_unit_key",
            "source_type",
            "dataset_id",
            "video_key",
            "object_track_key",
            "pig_id",
            "track_id",
            "label_window_start",
            "label_window_end",
        )
        if column in intervals
    ]
    out = intervals.loc[:, columns].copy()
    out.insert(0, "review_unit_id", out["temporal_unit_key"].astype(str))
    out = out.rename(
        columns={
            "label_window_start": "unit_start_frame",
            "label_window_end": "unit_end_frame",
        }
    )
    return out.sort_values("review_unit_id", kind="mergesort").reset_index(
        drop=True
    )


def _media_authority(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in (
            "frame_uid",
            "scene_frame_uid",
            "image_name",
            "crop_path",
            "source_video_path",
        )
        if column in frame
    ]
    return frame.loc[:, columns].sort_values(
        columns[0],
        kind="mergesort",
    ).reset_index(drop=True)


def _hidden_authority(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in (
            "frame_uid",
            "object_track_key",
            "frame_index",
            "hidden",
            "hidden_review_status",
            "hidden_is_trusted",
            "hidden_trust_status",
        )
        if column in frame
    ]
    return frame.loc[:, columns].sort_values(
        ["object_track_key", "frame_index"],
        kind="mergesort",
    ).reset_index(drop=True)


def _write_smoke_artifacts(
    output_dir: Path,
    **payloads: Any,
) -> dict[str, Path]:
    names = {
        "frame_local": "representative_frame_local_primitives.csv",
        "hidden": "representative_hidden_reviewed_frames.csv",
        "harmonized": "representative_harmonized_frames.csv",
        "intervals": "representative_temporal_native_units.csv",
        "review_units": "representative_behavior_review_units.csv",
        "media": "representative_media_authority.csv",
        "native_evidence": "representative_native_review_evidence.csv",
        "native_samples": "representative_native_unit_samples.csv",
        "view_samples": "representative_exact_view_smoke.csv",
        "pig_manifest": "representative_pig_strenet_evidence_manifest.json",
        "timestamp_contract": "timestamp_fps_contract.json",
        "evidence_semantics": "evidence_semantics.json",
    }
    paths = {key: output_dir / name for key, name in names.items()}
    for key, payload in payloads.items():
        path = paths[key]
        if isinstance(payload, pd.DataFrame):
            payload.to_csv(path, index=False)
        else:
            _write_json(path, payload)
    return paths


def _view_result_summary(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row in frame.itertuples(index=False):
        key = f"{row.view_type}:{row.scenario}"
        result[key] = {
            "selected_source_frame_indices": row.selected_source_frame_indices,
            "pair_delta_frames": row.pair_delta_frames,
            "pair_delta_seconds": row.pair_delta_seconds,
            "physical_span_seconds": row.physical_span_seconds,
            "behavior_review_status": row.behavior_review_status,
            "pair_recomputed_for_view": bool(row.pair_recomputed_for_view),
            "aggregate_recomputed_for_view": bool(
                row.aggregate_recomputed_for_view
            ),
            "final_structural_eligibility": bool(
                row.final_structural_eligibility
            ),
        }
    return result


def _to_bool(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.fillna(False).astype(str).str.strip().str.lower().isin(
            {"true", "1", "yes", "y", "t"}
        )
    return pd.Series(bool(values))


def _json_ints(values: pd.Series) -> str:
    return json.dumps(
        pd.to_numeric(values, errors="raise").astype(int).tolist(),
        separators=(",", ":"),
    )


def _json_floats(values: pd.Series) -> str:
    return json.dumps(
        pd.to_numeric(values, errors="raise").astype(float).tolist(),
        separators=(",", ":"),
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
