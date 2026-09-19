"""Build bounded, non-promoted Phase 3 social and ROI audit evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.features.native_evidence_contract import (
    check_native_review_evidence,
)
from pig_behavior.classification_v2.features.spatial_semantics import (
    ROI_AGGREGATION_VERSION,
    SOCIAL_IDENTITY_VERSION,
    SOCIAL_TIE_BREAK_VERSION,
)
from pig_behavior.classification_v2.features.spatiotemporal import (
    audit_enhanced_spatiotemporal_features,
    build_enhanced_spatiotemporal_features,
)
from pig_behavior.classification_v2.spatial_sequence_export import (
    export_spatial_sequences,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--contract-manifest", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--units-per-source", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    stage = args.output_dir.with_name(args.output_dir.name + ".staging")
    if stage.exists():
        raise FileExistsError(stage)
    stage.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir()

    header = pd.read_csv(args.input_csv, nrows=0).columns.tolist()
    discovery_columns = [
        column
        for column in (
            "source_type",
            "dataset_id",
            "video_key",
            "temporal_unit_key",
            "object_track_key",
            "frame_index",
            "pig_id",
            "roi_target_available",
            "roi_target_contact",
        )
        if column in header
    ]
    discovery = pd.read_csv(
        args.input_csv,
        usecols=discovery_columns,
        low_memory=False,
    )
    selected_units = _select_units(
        discovery,
        units_per_source=args.units_per_source,
    )
    selected_set = set(selected_units)
    chunks: list[pd.DataFrame] = []
    input_rows = 0
    for chunk in pd.read_csv(
        args.input_csv,
        low_memory=False,
        chunksize=20_000,
    ):
        input_rows += len(chunk)
        keep = chunk["temporal_unit_key"].astype(str).isin(selected_set)
        if keep.any():
            chunks.append(chunk.loc[keep].copy())
    selected = pd.concat(chunks, ignore_index=True)
    selected = selected.sort_values(
        [
            "source_type",
            "temporal_unit_key",
            "object_track_key",
            "frame_index",
        ],
        kind="mergesort",
    )
    produced = build_enhanced_spatiotemporal_features(selected)
    input_sha = _file_sha256(args.input_csv)
    contract_sha = _file_sha256(args.contract_manifest)
    produced["code_authority_sha"] = args.code_sha.lower()
    produced["input_sha256"] = input_sha
    produced["contract_manifest_sha256"] = contract_sha
    producer_audit = audit_enhanced_spatiotemporal_features(
        produced,
        input_rows=len(selected),
        code_sha=args.code_sha,
        input_sha256=input_sha,
        contract_manifest_sha256=contract_sha,
    )
    independent = check_native_review_evidence(
        selected,
        produced,
        producer_audit=producer_audit,
        code_sha=args.code_sha,
        input_sha256=input_sha,
        contract_manifest_sha256=contract_sha,
    )
    trace = _social_trace(produced)
    roi_summary = _roi_summary(produced)
    permutation = _permutation_check(selected)
    leakage = _leakage_preflight()
    consistency = _consistency_checks(trace, roi_summary)
    errors = [
        *producer_audit["errors"],
        *independent["errors"],
        *[
            name
            for name, passed in consistency.items()
            if not passed
        ],
    ]
    if not permutation["pass"]:
        errors.append("row_permutation_invariance")
    if not leakage["pass"]:
        errors.append("model_leakage_preflight")
    audit = {
        "schema_version": "classification_v2.phase3_bounded_audit.v1",
        "input_csv": str(args.input_csv),
        "input_sha256": input_sha,
        "contract_manifest_sha256": contract_sha,
        "code_authority_sha": args.code_sha.lower(),
        "input_rows": input_rows,
        "selected_rows": len(produced),
        "selected_units": int(produced["temporal_unit_key"].nunique()),
        "source_unit_counts": {
            source: int(
                roi_summary["source_type"].eq(source).sum()
            )
            for source in (
                "cvat_tracking_xml",
                "legacy_recovered",
            )
        },
        "social_identity_version": SOCIAL_IDENTITY_VERSION,
        "social_tie_break_version": SOCIAL_TIE_BREAK_VERSION,
        "roi_aggregation_version": ROI_AGGREGATION_VERSION,
        "consistency_checks": consistency,
        "producer_audit_errors": producer_audit["errors"],
        "independent_checker": independent,
        "errors": errors,
    }
    _write_json(stage / "phase3_bounded_regression_audit.json", audit)
    trace.to_csv(stage / "phase3_production_social_trace.csv", index=False)
    roi_summary.to_csv(
        stage / "phase3_production_roi_summary.csv",
        index=False,
    )
    _write_json(
        stage / "phase3_row_permutation_results.json",
        permutation,
    )
    _write_json(
        stage / "phase3_model_leakage_preflight.json",
        leakage,
    )
    if errors:
        raise SystemExit(1)
    stage.replace(args.output_dir)


def _select_units(
    discovery: pd.DataFrame,
    *,
    units_per_source: int,
) -> list[str]:
    work = discovery.copy()
    frame_group = [
        "source_type",
        "dataset_id",
        "video_key",
        "frame_index",
    ]
    work["_frame_actor_count"] = work.groupby(
        frame_group,
        dropna=False,
    )["object_track_key"].transform("nunique")
    if "roi_target_available" in work:
        work["_roi_available"] = _bool_series(
            work["roi_target_available"]
        )
    else:
        work["_roi_available"] = False
    if "pig_id" in work:
        work["_blank_pig"] = (
            work["pig_id"].fillna("").astype(str).str.strip().eq("")
        )
    else:
        work["_blank_pig"] = False
    unit_score = work.groupby(
        ["source_type", "temporal_unit_key"],
        dropna=False,
    ).agg(
        has_social=("_frame_actor_count", lambda s: bool((s > 1).any())),
        roi_available_count=("_roi_available", "sum"),
        observed_count=("_roi_available", "size"),
        has_blank_pig=("_blank_pig", "any"),
    )
    selected: list[str] = []
    for source in ("cvat_tracking_xml", "legacy_recovered"):
        local = unit_score.loc[source].reset_index()
        local["roi_state"] = np.select(
            [
                local["roi_available_count"].eq(0),
                local["roi_available_count"].eq(local["observed_count"]),
            ],
            ["none", "full"],
            default="partial",
        )
        local = local.sort_values(
            [
                "has_social",
                "has_blank_pig",
                "temporal_unit_key",
            ],
            ascending=[False, False, True],
            kind="mergesort",
        )
        source_selection: list[str] = []
        for roi_state in ("partial", "none", "full"):
            candidates = local.loc[local["roi_state"].eq(roi_state)]
            if not candidates.empty:
                source_selection.append(
                    str(candidates.iloc[0]["temporal_unit_key"])
                )
        source_selection.extend(
            [
                str(key)
                for key in local["temporal_unit_key"].tolist()
                if str(key) not in set(source_selection)
            ]
        )
        selected.extend(source_selection[:units_per_source])
    return selected


def _social_trace(produced: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "temporal_unit_key": produced["temporal_unit_key"],
        "canonical_actor_key": produced["object_track_key"],
        "source_type": produced["source_type"],
        "video_key": produced["video_key"],
        "source_frame_index": produced["frame_index"],
        "actor_geometry_valid": produced["current_geometry_valid"],
        "partner_candidate_count": produced["partner_candidate_count"],
        "nearest_partner_key": produced["nearest_partner_key"],
        "nearest_track_id": produced["nearest_track_id"],
        "nearest_object_id": produced["nearest_object_id"],
        "nearest_pig_id": produced["nearest_pig_id"],
        "nearest_distance_axis": produced["nearest_dist_n"],
        "nearest_distance_diagonal": produced[
            "nearest_distance_diagonal"
        ],
        "distance_available": produced["distance_available"],
        "nearest_neighbor_available": produced[
            "nearest_neighbor_available"
        ],
        "nearest_tie_count": produced["nearest_tie_count"],
        "nearest_tie_break_rule": produced["nearest_tie_break_rule"],
        "previous_partner_key": produced["previous_partner_key"],
        "same_partner_as_previous": produced[
            "same_partner_as_previous"
        ],
        "partner_switch": produced["partner_switch"],
        "partner_continuity_valid": produced[
            "partner_continuity_valid"
        ],
        "social_exclusion_reason": produced[
            "social_exclusion_reason"
        ],
    }
    return pd.DataFrame(columns).sort_values(
        [
            "source_type",
            "temporal_unit_key",
            "source_frame_index",
            "canonical_actor_key",
        ],
        kind="mergesort",
    )


def _roi_summary(produced: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "temporal_unit_key",
        "source_type",
        "behavior",
        "observed_frame_count",
        "target_roi_available_frame_count",
        "target_roi_contact_frame_count",
        "target_roi_availability_ratio_unit",
        "target_roi_contact_ratio_unit",
        "target_roi_unit_available",
        "roi_aggregation_version",
    ]
    return (
        produced[columns]
        .drop_duplicates("temporal_unit_key")
        .sort_values(
            ["source_type", "temporal_unit_key"],
            kind="mergesort",
        )
    )


def _permutation_check(selected: pd.DataFrame) -> dict[str, object]:
    first = build_enhanced_spatiotemporal_features(selected)
    permuted_input = selected.sample(frac=1.0, random_state=20260724)
    second = build_enhanced_spatiotemporal_features(permuted_input)
    columns = [
        "source_type",
        "temporal_unit_key",
        "object_track_key",
        "frame_index",
        "nearest_partner_key",
        "nearest_dist_n",
        "nearest_distance_diagonal",
        "nearest_tie_count",
        "partner_continuity_valid",
        "same_partner_as_previous",
        "partner_switch",
    ]
    order = columns[:4]
    first = first[columns].sort_values(order).reset_index(drop=True)
    second = second[columns].sort_values(order).reset_index(drop=True)
    first_digest = hashlib.sha256(
        first.to_csv(index=False, lineterminator="\n").encode()
    ).hexdigest()
    second_digest = hashlib.sha256(
        second.to_csv(index=False, lineterminator="\n").encode()
    ).hexdigest()
    return {
        "schema_version": "classification_v2.phase3_permutation.v1",
        "seed": 20260724,
        "rows": len(first),
        "first_digest": first_digest,
        "permuted_digest": second_digest,
        "pass": first_digest == second_digest,
    }


def _leakage_preflight() -> dict[str, object]:
    forbidden = [
        "target_roi_contact",
        "target_roi_distance",
        "target_roi_contact_ratio_unit",
        "label_selected_roi_class_indicator",
    ]
    windows = pd.DataFrame(
        {
            "window_id": ["w"],
            "object_track_key": ["a"],
            "window_start_frame": [0],
            "window_end_frame": [0],
            "window_length_frames": [1],
        }
    )
    failures: dict[str, str] = {}
    for column in forbidden:
        frames = pd.DataFrame(
            {
                "object_track_key": ["a"],
                "frame_index": [0],
                column: [1.0],
            }
        )
        try:
            export_spatial_sequences(
                windows,
                frames,
                feature_schema={"requested": [column]},
            )
        except ValueError as exc:
            failures[column] = str(exc)
    allowed_frames = pd.DataFrame(
        {
            "object_track_key": ["a"],
            "frame_index": [0],
            "roi_feeder_contact": [1.0],
        }
    )
    allowed = export_spatial_sequences(
        windows,
        allowed_frames,
        feature_schema={
            "roi_class_relation": ["roi_feeder_contact"]
        },
    )
    return {
        "schema_version": "classification_v2.phase3_leakage_preflight.v1",
        "forbidden_requested": forbidden,
        "forbidden_failures": failures,
        "label_independent_allowed": (
            allowed.feature_names["roi_class_relation"]
            == ["roi_feeder_contact"]
        ),
        "pass": (
            len(failures) == len(forbidden)
            and allowed.feature_names["roi_class_relation"]
            == ["roi_feeder_contact"]
        ),
    }


def _consistency_checks(
    trace: pd.DataFrame,
    roi: pd.DataFrame,
) -> dict[str, bool]:
    first = trace.groupby(
        ["temporal_unit_key", "canonical_actor_key"],
        sort=False,
    ).head(1)
    available_roi = roi["target_roi_unit_available"].astype(bool)
    return {
        "self_neighbor_count_zero": bool(
            (
                trace["nearest_partner_key"].fillna("").ne(
                    trace["canonical_actor_key"].fillna("")
                )
            ).all()
        ),
        "first_social_row_has_no_continuity": bool(
            (~first["partner_continuity_valid"].astype(bool)).all()
        ),
        "no_neighbor_not_partner_switch": bool(
            (
                ~trace.loc[
                    ~trace["nearest_neighbor_available"].astype(bool),
                    "partner_switch",
                ].astype(bool)
            ).all()
        ),
        "distance_mask_matches_neighbor": bool(
            trace["distance_available"].astype(bool).equals(
                trace["nearest_neighbor_available"].astype(bool)
            )
        ),
        "roi_contact_not_above_available": bool(
            (
                roi["target_roi_contact_frame_count"]
                <= roi["target_roi_available_frame_count"]
            ).all()
        ),
        "roi_availability_ratio_bounded": bool(
            roi["target_roi_availability_ratio_unit"].between(0, 1).all()
        ),
        "available_roi_contact_ratio_bounded": bool(
            roi.loc[
                available_roi,
                "target_roi_contact_ratio_unit",
            ].between(0, 1).all()
        ),
        "zero_roi_availability_marked_unavailable": bool(
            (
                ~roi.loc[
                    roi["target_roi_available_frame_count"].eq(0),
                    "target_roi_unit_available",
                ].astype(bool)
            ).all()
        ),
    }


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.fillna("").astype(str).str.strip().str.casefold().isin(
        {"true", "1", "yes", "y"}
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
