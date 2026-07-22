"""Run a bounded dual-source smoke through the production Group-A APIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.contracts.timestamp_fps import (
    build_timestamp_fps_contract,
    inspect_video_fps_authority,
)
from pig_behavior.classification_v2.features.frame_local import (
    audit_frame_local_primitives,
    build_frame_local_primitives,
    frame_local_schema_payload,
)
from pig_behavior.classification_v2.features.pen_context import (
    DEFAULT_PEN_MASK_SHA256,
    build_pen_context_features,
)
from pig_behavior.classification_v2.features.spatiotemporal import (
    audit_enhanced_spatiotemporal_features,
    build_enhanced_spatiotemporal_features,
)
from pig_behavior.classification_v2.review.evidence_semantics import (
    build_evidence_semantics,
)
from pig_behavior.classification_v2.review.hidden_review_builder import (
    audit_hidden_input_structure,
)
from pig_behavior.classification_v2.review.media_authority import (
    build_behavior_review_media_authority,
    finalize_media_authority_summary,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

SMOKE_LINEAGE = "agent_group_a_production_smoke"
INDEX_COLUMNS = {
    "source_type",
    "dataset_id",
    "video_key",
    "object_track_key",
    "temporal_unit_key",
    "frame_index",
    "label_anchor_frame_index",
    "behavior",
    "behavior_temporal_final",
    "hidden",
    "bbox_valid",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-frame-objects-csv", required=True, type=Path)
    parser.add_argument("--selection-index-csv", required=True, type=Path)
    parser.add_argument("--roi-coco", required=True, type=Path)
    parser.add_argument("--pen-mask", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--legacy-crop-root", required=True, type=Path)
    parser.add_argument("--code-authority-sha", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunksize", type=int, default=25_000)
    parser.add_argument(
        "--expected-pen-mask-sha256",
        default=DEFAULT_PEN_MASK_SHA256,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    index = pd.read_csv(
        args.selection_index_csv,
        usecols=lambda column: column in INDEX_COLUMNS,
        low_memory=False,
    )
    selection, selected_index = select_representative_units(index)
    source_rows = _read_scene_rows(
        args.merged_frame_objects_csv,
        selected_index,
        chunksize=args.chunksize,
    )
    source_rows = source_rows.drop(
        columns=["temporal_unit_key"],
        errors="ignore",
    )
    frame_local = build_frame_local_primitives(
        source_rows,
        roi_coco_path=args.roi_coco,
        pen_mask_path=args.pen_mask,
        expected_pen_mask_sha256=args.expected_pen_mask_sha256,
    )
    frame_audit = audit_frame_local_primitives(source_rows, frame_local)
    selected_keys = set(selection["selected_unit_keys"])
    hidden_scope = frame_local.loc[
        frame_local["temporal_unit_key"].astype(str).isin(selected_keys)
    ].copy()
    hidden_structural_audit = audit_hidden_input_structure(hidden_scope)
    native_all = build_enhanced_spatiotemporal_features(frame_local)
    native_all = build_pen_context_features(
        native_all,
        mask_path=args.pen_mask,
        expected_mask_sha256=args.expected_pen_mask_sha256,
    )
    native = native_all.loc[
        native_all["temporal_unit_key"].astype(str).isin(selected_keys)
    ].copy()
    missing_units = sorted(
        selected_keys.difference(native["temporal_unit_key"].astype(str))
    )
    native_audit = audit_enhanced_spatiotemporal_features(native)
    if missing_units:
        native_audit["errors"].append(f"selected_units_missing={missing_units}")
    semantics = build_evidence_semantics(
        frame_local,
        native,
        lineage_id=SMOKE_LINEAGE,
        code_authority_sha=args.code_authority_sha,
    )
    timestamp = build_timestamp_fps_contract(
        frame_local,
        lineage_id=SMOKE_LINEAGE,
        code_authority_sha=args.code_authority_sha,
        source_lineage_artifacts={
            "merged_frame_objects": args.merged_frame_objects_csv,
            "selection_index": args.selection_index_csv,
        },
        video_fps_authority=inspect_video_fps_authority(
            frame_local,
            args.video_root,
        ),
    )
    units = _review_units(native)
    media_index, media_summary = build_behavior_review_media_authority(
        units,
        native,
        video_root=args.video_root,
        legacy_crop_root=args.legacy_crop_root,
        lineage_id=SMOKE_LINEAGE,
        code_authority_sha=args.code_authority_sha,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "source": args.output_dir / "representative_merged_frame_objects.csv",
        "frame_local": args.output_dir / "frame_local_primitives.csv",
        "frame_schema": args.output_dir / "frame_local_schema.json",
        "frame_audit": args.output_dir / "frame_local_audit.json",
        "hidden_structure": args.output_dir / "hidden_structure_audit.json",
        "native": args.output_dir / "native_review_evidence.csv",
        "timestamp": args.output_dir / "timestamp_fps_contract.json",
        "semantics": args.output_dir / "evidence_semantics.json",
        "review_units": args.output_dir / "review_units.csv",
        "media_index": args.output_dir / "behavior_review_media_index.csv",
        "media": args.output_dir / "behavior_review_media_authority.json",
    }
    source_rows.to_csv(paths["source"], index=False)
    frame_local.to_csv(paths["frame_local"], index=False)
    native.to_csv(paths["native"], index=False)
    units.to_csv(paths["review_units"], index=False)
    media_index.to_csv(paths["media_index"], index=False)
    frame_schema = frame_local_schema_payload(frame_local)
    frame_schema["lineage_id"] = SMOKE_LINEAGE
    frame_schema["code_authority_sha"] = args.code_authority_sha.lower()
    frame_audit["lineage_id"] = SMOKE_LINEAGE
    frame_audit["code_authority_sha"] = args.code_authority_sha.lower()
    media = finalize_media_authority_summary(
        media_summary,
        index_csv=paths["media_index"],
    )
    for key, payload in (
        ("frame_schema", frame_schema),
        ("frame_audit", frame_audit),
        ("hidden_structure", hidden_structural_audit),
        ("timestamp", timestamp),
        ("semantics", semantics),
        ("media", media),
    ):
        _write_json(paths[key], payload)

    errors = [
        *frame_audit["errors"],
        *hidden_structural_audit["errors"],
        *native_audit["errors"],
        *timestamp["errors"],
        *semantics["errors"],
        *media["errors"],
    ]
    report = {
        "schema_version": "classification_v2.group_a_production_smoke.v2",
        "lineage_id": SMOKE_LINEAGE,
        "code_authority_sha": args.code_authority_sha.lower(),
        "selection": selection,
        "source_artifacts_read_only": True,
        "v3_reused": False,
        "official_v5_created": False,
        "decisions_written": False,
        "gui_opened": False,
        "authorizes_behavior_gui": False,
        "frame_local_rows": len(frame_local),
        "hidden_structural_units": hidden_structural_audit.get(
            "temporal_unit_count",
            0,
        ),
        "native_evidence_rows": len(native),
        "review_units": len(units),
        "artifact_hashes": {
            key: file_sha256(path) for key, path in paths.items()
        },
        "errors": errors,
        "valid": not errors,
    }
    _write_json(args.output_dir / "group_a_production_smoke.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(2)


def select_representative_units(
    index: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    work = index.copy()
    work["frame_index"] = pd.to_numeric(work["frame_index"], errors="coerce")
    work["behavior_effective"] = work.get(
        "behavior_temporal_final",
        work["behavior"],
    ).fillna(work["behavior"]).astype(str)
    work["hidden_flag"] = _to_bool(work.get("hidden", False))
    units = (
        work.groupby(
            [
                "source_type",
                "dataset_id",
                "video_key",
                "object_track_key",
                "temporal_unit_key",
            ],
            dropna=False,
            sort=True,
        )
        .agg(
            start_frame=("frame_index", "min"),
            end_frame=("frame_index", "max"),
            frame_count=("frame_index", "nunique"),
            behavior=("behavior_effective", "first"),
            hidden=("hidden_flag", "max"),
        )
        .reset_index()
    )
    expected = units["source_type"].map(
        {"legacy_recovered": 16, "cvat_tracking_xml": 6}
    )
    complete = units.loc[units["frame_count"].eq(expected)].copy()
    selected: list[str] = []
    cases: dict[str, str] = {}

    def add_case(name: str, candidates: pd.DataFrame) -> None:
        if candidates.empty:
            raise ValueError(f"representative case unavailable: {name}")
        key = str(candidates.iloc[0]["temporal_unit_key"])
        cases[name] = key
        if key not in selected:
            selected.append(key)

    add_case("anchor_1020", complete.loc[complete["start_frame"].eq(1020)])
    add_case(
        "video_000231",
        complete.loc[complete["video_key"].astype(str).str.contains("000231")],
    )
    behavior_cases = {
        "roi": {"eat", "drink", "playwithtoy"},
        "motion": {"move", "explore", "stand"},
        "posture": {"lying", "sitting"},
        "interaction": {"fight", "social-nose"},
    }
    for name, behaviors in behavior_cases.items():
        add_case(name, complete.loc[complete["behavior"].isin(behaviors)])
    add_case("hidden", complete.loc[complete["hidden"]])
    for source, count in (("legacy_recovered", 5), ("cvat_tracking_xml", 5)):
        candidates = complete.loc[complete["source_type"].eq(source)]
        source_count = 0
        for key in candidates["temporal_unit_key"].astype(str):
            if key not in selected:
                selected.append(key)
            source_count = complete.loc[
                complete["temporal_unit_key"].astype(str).isin(selected)
                & complete["source_type"].eq(source)
            ]["temporal_unit_key"].nunique()
            if source_count >= count:
                break
        if source_count < count:
            raise ValueError(f"representative source support unavailable: {source}")
    selected_rows = work.loc[
        work["temporal_unit_key"].astype(str).isin(selected)
    ].copy()
    return {
        "selected_unit_keys": selected,
        "case_unit_keys": cases,
        "source_unit_counts": complete.loc[
            complete["temporal_unit_key"].astype(str).isin(selected),
            "source_type",
        ].value_counts().sort_index().to_dict(),
    }, selected_rows


def _read_scene_rows(
    path: Path,
    selected_index: pd.DataFrame,
    *,
    chunksize: int,
) -> pd.DataFrame:
    key_columns = ["source_type", "dataset_id", "video_key", "frame_index"]
    keys = set(
        map(tuple, selected_index[key_columns].astype(str).to_numpy().tolist())
    )
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        key = pd.MultiIndex.from_frame(chunk[key_columns].astype(str))
        mask = key.isin(keys)
        if mask.any():
            parts.append(chunk.loc[mask].copy())
    if not parts:
        raise ValueError("representative scene selection loaded zero source rows")
    return pd.concat(parts, ignore_index=True)


def _review_units(native: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for key, unit in native.groupby("temporal_unit_key", sort=True):
        ordered = unit.sort_values("source_frame_index", kind="mergesort")
        first = ordered.iloc[0]
        frames = ordered["source_frame_index"].astype(int).tolist()
        records.append(
            {
                "review_unit_id": str(key),
                "temporal_unit_key": str(key),
                "source_type": first["source_type"],
                "dataset_id": first["dataset_id"],
                "video_key": first["video_key"],
                "pig_id": first["pig_id"],
                "track_id": first["track_id"],
                "object_track_key": first["object_track_key"],
                "unit_start_frame": min(frames),
                "unit_end_frame": max(frames),
                "display_frame_indices": json.dumps(
                    frames,
                    separators=(",", ":"),
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _to_bool(value: Any) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.fillna("").astype(str).str.casefold().isin(
            {"1", "true", "yes", "y"}
        )
    return pd.Series(False, index=[])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
