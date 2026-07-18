"""Read-only quality audits for CVAT behavior annotations.

The module supports two source contracts:

* legacy six-anchor image tasks with a per-task manifest and XML/JSON authority;
* CVAT interpolation tracking XML used by the all-source classifier.

It never edits annotations. Identity repair candidates are evidence for human
inspection only.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd

from pig_behavior.classification_v2.schema import (
    DEFAULT_PIG_IDS,
    VALID_BEHAVIOR_SET,
    normalize_behavior,
    normalize_pig_id,
)
from pig_behavior.data.cvat_native import (
    frame_file_name,
    load_cvat_task,
    load_manifest,
    parse_burst_from_filename,
    select_cvat_annotation_source,
)

EXPECTED_LEGACY_SLOTS = tuple(range(6))
VALID_HIDDEN_VALUES = {"yes", "no", "true", "false", "1", "0", "y", "n"}
REQUIRED_BOX_ATTRIBUTES = ("ID", "Behavior", "Hidden")
SOURCE_LEGACY_TASK = "legacy_six_anchor_task_export"
SOURCE_TRACKING_XML = "classification_tracking_xml"


def audit_legacy_task_export(
    export_root: str | Path,
    *,
    expected_slots: Sequence[int] = EXPECTED_LEGACY_SLOTS,
) -> dict[str, Any]:
    """Audit mixed XML/JSON legacy CVAT tasks without changing source files."""
    root = Path(export_root)
    task_dirs = sorted(path for path in root.glob("task_*") if path.is_dir())
    if not task_dirs:
        raise FileNotFoundError(f"No task_* folders found under {root}")

    issues: list[dict[str, Any]] = []
    loaded: list[pd.DataFrame] = []
    task_metadata: dict[str, dict[str, Any]] = {}
    manifest_by_slot: dict[tuple[str, str, int], dict[str, Any]] = {}

    for task_dir in task_dirs:
        task = task_dir.name
        try:
            annotation_format, annotation_path = select_cvat_annotation_source(
                task_dir
            )
            manifest = load_manifest(task_dir / "data" / "manifest.jsonl")
            task_metadata[task] = {
                "annotation_format": annotation_format,
                "annotation_path": str(annotation_path),
                "total_frames": len(manifest),
            }
            raw_audit = _audit_raw_task_rectangles(
                task=task,
                annotation_format=annotation_format,
                annotation_path=annotation_path,
                manifest=manifest,
            )
            task_metadata[task].update(raw_audit["summary"])
            issues.extend(raw_audit["issues"])
            _index_legacy_manifest(
                task=task,
                manifest=manifest,
                manifest_by_slot=manifest_by_slot,
                issues=issues,
                annotation_path=annotation_path,
            )
            frame = load_cvat_task(task_dir)
            if frame.empty:
                issues.append(
                    _issue(
                        severity="error",
                        code="empty_task_annotations",
                        source_kind=SOURCE_LEGACY_TASK,
                        annotation_path=annotation_path,
                        task=task,
                        suggested_action="Export non-empty CVAT annotations.",
                    )
                )
            else:
                expected_rows = raw_audit["summary"]["valid_pig_rectangles"]
                if len(frame) != expected_rows:
                    issues.append(
                        _issue(
                            severity="error",
                            code="loaded_row_count_mismatch",
                            source_kind=SOURCE_LEGACY_TASK,
                            annotation_path=annotation_path,
                            task=task,
                            evidence={
                                "loaded_rows": len(frame),
                                "valid_raw_pig_rectangles": expected_rows,
                            },
                            suggested_action=(
                                "Resolve every skipped raw shape before using "
                                "this task."
                            ),
                        )
                    )
                loaded.append(frame)
        except (FileNotFoundError, OSError, ET.ParseError, ValueError) as exc:
            issues.append(
                _issue(
                    severity="error",
                    code="task_load_error",
                    source_kind=SOURCE_LEGACY_TASK,
                    annotation_path=task_dir,
                    task=task,
                    evidence={"error": str(exc)},
                    suggested_action=(
                        "Repair the task export or manifest binding before "
                        "using this task."
                    ),
                )
            )

    if not loaded:
        return _source_report(
            source_kind=SOURCE_LEGACY_TASK,
            source_path=root,
            summary={
                "task_count": len(task_dirs),
                "loaded_rows": 0,
                "authority_actor_keys": 0,
                "complete_authority_actor_keys": 0,
                "incomplete_authority_actor_keys": 0,
                "actors_absent_authority_frame": 0,
            },
            issues=issues,
        )

    frame = pd.concat(loaded, ignore_index=True)
    prepared = _prepare_legacy_frame(frame)
    _audit_legacy_row_fields(
        prepared,
        task_metadata=task_metadata,
        issues=issues,
    )
    _audit_legacy_duplicate_identities(
        prepared,
        task_metadata=task_metadata,
        issues=issues,
    )

    authority = _legacy_authority_rows(prepared, issues)
    authority_keys = authority[
        ["task", "group_id", "_pig_id"]
    ].drop_duplicates()
    all_keys = prepared[["task", "group_id", "_pig_id"]].drop_duplicates()

    absent_keys = all_keys.merge(
        authority_keys,
        on=["task", "group_id", "_pig_id"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    absent_keys = absent_keys.loc[absent_keys["_merge"].eq("left_only")].copy()

    complete_authority_keys = 0
    incomplete_authority_keys = 0
    expected_slot_set = set(map(int, expected_slots))
    authority_key_set = set(
        authority_keys.itertuples(index=False, name=None)
    )

    for key, actor_rows in prepared.groupby(
        ["task", "group_id", "_pig_id"],
        dropna=False,
        sort=True,
    ):
        task, group_id, pig_id = key
        observed_slots = sorted(set(map(int, actor_rows["_slot"])))
        missing_slots = sorted(expected_slot_set.difference(observed_slots))
        is_authority_actor = key in authority_key_set

        if is_authority_actor and missing_slots:
            incomplete_authority_keys += 1
        elif is_authority_actor:
            complete_authority_keys += 1

        for slot in missing_slots:
            manifest_row = manifest_by_slot.get((task, group_id, slot))
            if manifest_row is None:
                issues.append(
                    _issue(
                        severity="error",
                        code="missing_manifest_slot",
                        source_kind=SOURCE_LEGACY_TASK,
                        annotation_path=task_metadata[task][
                            "annotation_path"
                        ],
                        task=task,
                        group_id=group_id,
                        pig_id=pig_id,
                        slot=slot,
                        observed_slots=observed_slots,
                        missing_slots=missing_slots,
                        suggested_action=(
                            "Repair the task manifest before annotating boxes."
                        ),
                    )
                )
                continue

            issues.append(
                _issue(
                    severity="review",
                    code=(
                        "missing_anchor"
                        if is_authority_actor
                        else "missing_anchor_non_authority_actor"
                    ),
                    source_kind=SOURCE_LEGACY_TASK,
                    annotation_path=task_metadata[task]["annotation_path"],
                    task=task,
                    group_id=group_id,
                    pig_id=pig_id,
                    slot=slot,
                    observed_slots=observed_slots,
                    missing_slots=missing_slots,
                    suggested_action=(
                        "Add a frame-specific bbox/Hidden annotation if the "
                        "pig is identifiable; otherwise declare this actor "
                        "key as an exclusion."
                    ),
                    **manifest_row,
                )
            )

    _audit_absent_authority_actors(
        prepared,
        absent_keys=absent_keys,
        manifest_by_slot=manifest_by_slot,
        task_metadata=task_metadata,
        issues=issues,
    )
    _add_duplicate_identity_candidates(
        prepared,
        authority=authority,
        task_metadata=task_metadata,
        issues=issues,
    )
    _add_sequence_identity_candidates(
        prepared,
        authority=authority,
        task_metadata=task_metadata,
        issues=issues,
    )

    duplicate_rows = int(
        prepared.duplicated(
            ["task", "group_id", "_slot", "_pig_id"],
            keep=False,
        ).sum()
    )
    summary = {
        "task_count": len(task_dirs),
        "task_sources": {
            task: metadata["annotation_format"]
            for task, metadata in sorted(task_metadata.items())
        },
        "task_total_frames": {
            task: metadata["total_frames"]
            for task, metadata in sorted(task_metadata.items())
        },
        "task_raw_pig_rectangle_candidates": {
            task: metadata["raw_pig_rectangle_candidates"]
            for task, metadata in sorted(task_metadata.items())
        },
        "task_valid_pig_rectangles": {
            task: metadata["valid_pig_rectangles"]
            for task, metadata in sorted(task_metadata.items())
        },
        "loaded_rows": int(len(prepared)),
        "authority_actor_keys": int(len(authority_keys)),
        "complete_authority_actor_keys": complete_authority_keys,
        "incomplete_authority_actor_keys": incomplete_authority_keys,
        "actors_absent_authority_frame": int(len(absent_keys)),
        "duplicate_anchor_identity_rows": duplicate_rows,
        "duplicate_anchor_identity_keys": int(
            prepared.loc[
                prepared.duplicated(
                    ["task", "group_id", "_slot", "_pig_id"],
                    keep=False,
                ),
                ["task", "group_id", "_slot", "_pig_id"],
            ]
            .drop_duplicates()
            .shape[0]
        ),
        "missing_anchor_issue_rows": sum(
            issue["code"].startswith("missing_anchor")
            for issue in issues
        ),
    }
    return _source_report(
        source_kind=SOURCE_LEGACY_TASK,
        source_path=root,
        summary=summary,
        issues=issues,
    )


def audit_tracking_xml(
    xml_path: str | Path,
    *,
    expected_pig_ids: Sequence[str] = DEFAULT_PIG_IDS,
) -> dict[str, Any]:
    """Audit one CVAT interpolation XML with raw attribute preservation."""
    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(f"CVAT tracking XML not found: {path}")

    root = ET.parse(path).getroot()
    task_name = _xml_text(root, "./meta/task/name", path.stem)
    video_key = _tracking_video_key(root, path)
    task_size = _safe_int(
        _xml_text(root, "./meta/task/size", "0"),
        default=0,
    )
    start_frame = _safe_int(
        _xml_text(root, "./meta/task/start_frame", "0"),
        default=0,
    )
    stop_frame = _safe_int(
        _xml_text(root, "./meta/task/stop_frame", "-1"),
        default=-1,
    )
    image_width = _safe_int(
        _xml_text(root, "./meta/task/original_size/width", "0"),
        default=0,
    )
    image_height = _safe_int(
        _xml_text(root, "./meta/task/original_size/height", "0"),
        default=0,
    )
    if stop_frame < start_frame and task_size > 0:
        stop_frame = start_frame + task_size - 1
    total_frames = (
        task_size
        if task_size > 0
        else max(stop_frame - start_frame + 1, 0)
    )

    normalized_expected = tuple(
        pig_id
        for pig_id in map(normalize_pig_id, expected_pig_ids)
        if pig_id is not None
    )
    expected_set = set(normalized_expected)
    issues: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    outside_boxes = 0
    track_ids: list[str] = []
    declared_range_size = max(stop_frame - start_frame + 1, 0)
    if task_size > 0 and declared_range_size != task_size:
        issues.append(
            _issue(
                severity="error",
                code="task_size_range_mismatch",
                source_kind=SOURCE_TRACKING_XML,
                annotation_path=path,
                task=task_name,
                video_key=video_key,
                evidence={
                    "task_size": task_size,
                    "start_frame": start_frame,
                    "stop_frame": stop_frame,
                    "range_size": declared_range_size,
                },
                suggested_action=(
                    "Re-export CVAT XML with consistent task frame metadata."
                ),
            )
        )

    for track in root.findall("track"):
        track_id = str(track.attrib.get("id", "")).strip()
        track_label = str(track.attrib.get("label", "")).strip()
        label_pig_id = _pig_id_from_track_label(track_label)
        track_ids.append(track_id)
        seen_frames: set[int] = set()

        for box in track.findall("box"):
            frame_id = _safe_int(box.attrib.get("frame"), default=-1)
            if frame_id in seen_frames:
                issues.append(
                    _tracking_issue(
                        severity="error",
                        code="duplicate_frame_in_track",
                        path=path,
                        task_name=task_name,
                        video_key=video_key,
                        frame_id=frame_id,
                        start_frame=start_frame,
                        total_frames=total_frames,
                        pig_id=label_pig_id,
                        evidence={"track_id": track_id},
                        suggested_action=(
                            "Keep only one track box for this frame."
                        ),
                    )
                )
            seen_frames.add(frame_id)

            if _is_true(box.attrib.get("outside", "0")):
                outside_boxes += 1
                continue

            attr_pairs = _raw_box_attributes(box)
            attr_counts = Counter(name for name, _ in attr_pairs)
            attrs = {name: value for name, value in attr_pairs}
            for attribute in REQUIRED_BOX_ATTRIBUTES:
                if attr_counts[attribute] == 0:
                    issues.append(
                        _tracking_issue(
                            severity="error",
                            code="missing_required_box_attribute",
                            path=path,
                            task_name=task_name,
                            video_key=video_key,
                            frame_id=frame_id,
                            start_frame=start_frame,
                            total_frames=total_frames,
                            pig_id=label_pig_id,
                            evidence={
                                "track_id": track_id,
                                "attribute": attribute,
                            },
                            suggested_action=(
                                f"Add the {attribute} attribute in CVAT."
                            ),
                        )
                    )
                elif attr_counts[attribute] > 1:
                    issues.append(
                        _tracking_issue(
                            severity="error",
                            code="duplicate_box_attribute",
                            path=path,
                            task_name=task_name,
                            video_key=video_key,
                            frame_id=frame_id,
                            start_frame=start_frame,
                            total_frames=total_frames,
                            pig_id=label_pig_id,
                            evidence={
                                "track_id": track_id,
                                "attribute": attribute,
                                "count": attr_counts[attribute],
                            },
                            suggested_action=(
                                f"Keep one {attribute} attribute."
                            ),
                        )
                    )

            raw_pig_id = attrs.get("ID", "")
            pig_id = normalize_pig_id(raw_pig_id) or label_pig_id
            behavior = normalize_behavior(attrs.get("Behavior"))
            raw_hidden = str(attrs.get("Hidden", "")).strip()

            if normalize_pig_id(raw_pig_id) is None:
                issues.append(
                    _tracking_issue(
                        severity="error",
                        code="invalid_pig_id",
                        path=path,
                        task_name=task_name,
                        video_key=video_key,
                        frame_id=frame_id,
                        start_frame=start_frame,
                        total_frames=total_frames,
                        pig_id=pig_id,
                        evidence={
                            "track_id": track_id,
                            "raw_pig_id": raw_pig_id,
                        },
                        suggested_action="Set ID to ID_1 through ID_8.",
                    )
                )
            if behavior not in VALID_BEHAVIOR_SET:
                issues.append(
                    _tracking_issue(
                        severity="error",
                        code="invalid_behavior",
                        path=path,
                        task_name=task_name,
                        video_key=video_key,
                        frame_id=frame_id,
                        start_frame=start_frame,
                        total_frames=total_frames,
                        pig_id=pig_id,
                        evidence={
                            "track_id": track_id,
                            "raw_behavior": attrs.get("Behavior", ""),
                        },
                        suggested_action="Set one canonical 10-class behavior.",
                    )
                )
            if raw_hidden.lower() not in VALID_HIDDEN_VALUES:
                issues.append(
                    _tracking_issue(
                        severity="error",
                        code="invalid_hidden",
                        path=path,
                        task_name=task_name,
                        video_key=video_key,
                        frame_id=frame_id,
                        start_frame=start_frame,
                        total_frames=total_frames,
                        pig_id=pig_id,
                        evidence={
                            "track_id": track_id,
                            "raw_hidden": raw_hidden,
                        },
                        suggested_action="Set Hidden explicitly to Yes or No.",
                    )
                )
            if (
                label_pig_id is not None
                and pig_id is not None
                and label_pig_id != pig_id
            ):
                issues.append(
                    _tracking_issue(
                        severity="error",
                        code="track_label_id_mismatch",
                        path=path,
                        task_name=task_name,
                        video_key=video_key,
                        frame_id=frame_id,
                        start_frame=start_frame,
                        total_frames=total_frames,
                        pig_id=pig_id,
                        evidence={
                            "track_id": track_id,
                            "track_label": track_label,
                            "label_pig_id": label_pig_id,
                        },
                        suggested_action=(
                            "Make the box ID agree with the CVAT track label."
                        ),
                    )
                )

            bbox = _tracking_bbox(box)
            if bbox is None:
                issues.append(
                    _tracking_issue(
                        severity="error",
                        code="invalid_bbox",
                        path=path,
                        task_name=task_name,
                        video_key=video_key,
                        frame_id=frame_id,
                        start_frame=start_frame,
                        total_frames=total_frames,
                        pig_id=pig_id,
                        evidence={"track_id": track_id},
                        suggested_action="Repair bbox coordinates in CVAT.",
                    )
                )
            else:
                overshoot = _bbox_boundary_overshoot(
                    bbox,
                    width=image_width,
                    height=image_height,
                )
            if bbox is not None and overshoot > 0:
                minor_tolerance = _minor_boundary_tolerance(
                    width=image_width,
                    height=image_height,
                )
                is_minor = overshoot <= minor_tolerance
                issues.append(
                    _tracking_issue(
                        severity="info",
                        code=(
                            "bbox_minor_boundary_overshoot"
                            if is_minor
                            else "bbox_boundary_attention"
                        ),
                        path=path,
                        task_name=task_name,
                        video_key=video_key,
                        frame_id=frame_id,
                        start_frame=start_frame,
                        total_frames=total_frames,
                        pig_id=pig_id,
                        evidence={
                            "track_id": track_id,
                            "bbox": bbox,
                            "image_width": image_width,
                            "image_height": image_height,
                            "max_overshoot_px": overshoot,
                            "minor_tolerance_px": minor_tolerance,
                            "requires_visual_attention": not is_minor,
                            "review_status_inferred": False,
                        },
                        suggested_action=(
                            "Minor CVAT interpolation overshoot is recorded "
                            "and may use deterministic downstream clipping."
                            if is_minor
                            else "Attention only: retain the bbox when human "
                            "review confirmed it; otherwise inspect it before "
                            "relying on deterministic downstream clipping."
                        ),
                    )
                )

            if not start_frame <= frame_id <= stop_frame:
                issues.append(
                    _tracking_issue(
                        severity="error",
                        code="frame_outside_task_range",
                        path=path,
                        task_name=task_name,
                        video_key=video_key,
                        frame_id=frame_id,
                        start_frame=start_frame,
                        total_frames=total_frames,
                        pig_id=pig_id,
                        evidence={
                            "track_id": track_id,
                            "stop_frame": stop_frame,
                        },
                        suggested_action="Move or remove the out-of-range box.",
                    )
                )

            rows.append(
                {
                    "frame_id": frame_id,
                    "pig_id": pig_id,
                    "behavior": behavior,
                    "hidden": raw_hidden,
                    "track_id": track_id,
                    "track_label": track_label,
                    "label_pig_id": label_pig_id,
                    "bbox": bbox,
                }
            )

    if not track_ids:
        issues.append(
            _issue(
                severity="error",
                code="tracking_xml_has_no_tracks",
                source_kind=SOURCE_TRACKING_XML,
                annotation_path=path,
                task=task_name,
                video_key=video_key,
                suggested_action="Export interpolation tracks from CVAT.",
            )
        )

    duplicate_track_ids = sorted(
        track_id
        for track_id, count in Counter(track_ids).items()
        if count > 1
    )
    for track_id in duplicate_track_ids:
        issues.append(
            _issue(
                severity="error",
                code="duplicate_track_id",
                source_kind=SOURCE_TRACKING_XML,
                annotation_path=path,
                video_key=video_key,
                evidence={"track_id": track_id},
                suggested_action="Assign unique CVAT track IDs.",
            )
        )

    frame_rows = pd.DataFrame(rows)
    duplicate_identity_rows = 0
    if not frame_rows.empty:
        duplicate_mask = frame_rows.duplicated(
            ["frame_id", "pig_id"],
            keep=False,
        )
        duplicate_identity_rows = int(duplicate_mask.sum())
        for (frame_id, pig_id), duplicate in frame_rows.loc[
            duplicate_mask
        ].groupby(["frame_id", "pig_id"], dropna=False, sort=True):
            evidence_rows = duplicate[
                ["track_id", "track_label", "label_pig_id", "bbox"]
            ].to_dict("records")
            issues.append(
                _tracking_issue(
                    severity="error",
                    code="duplicate_pig_id_in_frame",
                    path=path,
                    task_name=task_name,
                    video_key=video_key,
                    frame_id=int(frame_id),
                    start_frame=start_frame,
                    total_frames=total_frames,
                    pig_id=pig_id,
                    evidence={"rows": evidence_rows},
                    suggested_action=(
                        "Correct the duplicated ID; do not delete a valid "
                        "different pig bbox."
                    ),
                )
            )
            _add_tracking_duplicate_candidate(
                duplicate=duplicate,
                expected_set=expected_set,
                all_frame_rows=frame_rows,
                path=path,
                task_name=task_name,
                video_key=video_key,
                start_frame=start_frame,
                total_frames=total_frames,
                issues=issues,
            )

    observed_frames = set(
        map(int, frame_rows["frame_id"])
    ) if not frame_rows.empty else set()
    missing_expected_rows = 0
    for frame_id in range(start_frame, stop_frame + 1):
        current = (
            frame_rows.loc[frame_rows["frame_id"].eq(frame_id)]
            if not frame_rows.empty
            else frame_rows
        )
        present = set(current["pig_id"].dropna().astype(str))
        for pig_id in sorted(expected_set.difference(present)):
            missing_expected_rows += 1
            issues.append(
                _tracking_issue(
                    severity="review",
                    code="missing_expected_pig_id_in_frame",
                    path=path,
                    task_name=task_name,
                    video_key=video_key,
                    frame_id=frame_id,
                    start_frame=start_frame,
                    total_frames=total_frames,
                    pig_id=pig_id,
                    evidence={
                        "present_pig_ids": sorted(present),
                        "frame_has_annotations": frame_id in observed_frames,
                    },
                    suggested_action=(
                        "Inspect the frame. Add a valid bbox/Hidden annotation "
                        "or record a deliberate exclusion."
                    ),
                )
            )

    summary = {
        "task_name": task_name,
        "video_key": video_key,
        "task_size": task_size,
        "start_frame": start_frame,
        "stop_frame": stop_frame,
        "total_frames": total_frames,
        "image_width": image_width,
        "image_height": image_height,
        "inside_box_rows": int(len(frame_rows)),
        "outside_box_rows": outside_boxes,
        "annotated_frames": len(observed_frames),
        "track_count": len(track_ids),
        "expected_pig_ids": list(normalized_expected),
        "duplicate_identity_rows": duplicate_identity_rows,
        "missing_expected_pig_id_rows": missing_expected_rows,
    }
    return _source_report(
        source_kind=SOURCE_TRACKING_XML,
        source_path=path,
        summary=summary,
        issues=issues,
    )


def combine_annotation_audits(
    reports: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Combine source reports and derive a deterministic overall status."""
    source_reports = list(reports)
    issues = [
        issue
        for report in source_reports
        for issue in report.get("issues", [])
    ]
    issues = sorted(issues, key=_issue_sort_key)
    severity_counts = Counter(issue["severity"] for issue in issues)
    code_counts = Counter(issue["code"] for issue in issues)
    return {
        "schema_version": 1,
        "status": _status(issues),
        "source_count": len(source_reports),
        "summary": {
            "issue_count": len(issues),
            "error_count": severity_counts["error"],
            "review_count": severity_counts["review"],
            "info_count": severity_counts["info"],
            "issue_counts_by_code": dict(sorted(code_counts.items())),
        },
        "sources": [
            {
                "source_kind": report["source_kind"],
                "source_path": report["source_path"],
                "status": report["status"],
                "summary": report["summary"],
            }
            for report in source_reports
        ],
        "issues": issues,
    }


def _prepare_legacy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["_pig_id"] = out["pig_id"].map(normalize_pig_id)
    out["_behavior"] = out["behavior"].map(normalize_behavior)
    out["_slot"] = pd.to_numeric(out["order"], errors="coerce")
    out["_frame_id"] = pd.to_numeric(out["frame"], errors="coerce")
    return out


def _audit_raw_task_rectangles(
    *,
    task: str,
    annotation_format: str,
    annotation_path: Path,
    manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    candidate_rows = 0
    valid_rows = 0
    if annotation_format == "xml":
        root = ET.parse(annotation_path).getroot()
        raw_shapes = [
            {
                "frame": _safe_int(image.attrib.get("id"), default=-1),
                "label": box.attrib.get("label", ""),
                "points_valid": all(
                    box.attrib.get(name) is not None
                    for name in ["xtl", "ytl", "xbr", "ybr"]
                ),
            }
            for image in root.findall("image")
            for box in image.findall("box")
        ]
    else:
        with annotation_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        raw_shapes = [
            {
                "frame": _safe_int(shape.get("frame"), default=-1),
                "label": shape.get("label", ""),
                "points_valid": len(shape.get("points", [])) == 4,
            }
            for annotation in payload
            for shape in annotation.get("shapes", [])
            if shape.get("type") == "rectangle"
            and not _is_true(shape.get("outside", False))
        ]

    for shape in raw_shapes:
        if not str(shape["label"]).lower().startswith("pig"):
            continue
        candidate_rows += 1
        frame_id = int(shape["frame"])
        frame_valid = 0 <= frame_id < len(manifest)
        if not frame_valid:
            issues.append(
                _issue(
                    severity="error",
                    code="raw_shape_frame_out_of_range",
                    source_kind=SOURCE_LEGACY_TASK,
                    annotation_path=annotation_path,
                    task=task,
                    frame_id=frame_id,
                    total_frames=len(manifest),
                    evidence={"raw_label": shape["label"]},
                    suggested_action=(
                        "Move or remove the shape with an invalid task frame."
                    ),
                )
            )
        if not shape["points_valid"]:
            image_name = (
                frame_file_name(manifest[frame_id])
                if frame_valid
                else ""
            )
            issues.append(
                _issue(
                    severity="error",
                    code="raw_shape_invalid_point_count",
                    source_kind=SOURCE_LEGACY_TASK,
                    annotation_path=annotation_path,
                    task=task,
                    frame_id=frame_id,
                    frame_position_1based=(
                        frame_id + 1 if frame_valid else None
                    ),
                    total_frames=len(manifest),
                    image_name=image_name,
                    evidence={"raw_label": shape["label"]},
                    suggested_action="Repair the raw rectangle coordinates.",
                )
            )
        if frame_valid and shape["points_valid"]:
            valid_rows += 1

    return {
        "summary": {
            "raw_pig_rectangle_candidates": candidate_rows,
            "valid_pig_rectangles": valid_rows,
        },
        "issues": issues,
    }


def _index_legacy_manifest(
    *,
    task: str,
    manifest: list[dict[str, Any]],
    manifest_by_slot: dict[tuple[str, str, int], dict[str, Any]],
    issues: list[dict[str, Any]],
    annotation_path: Path,
) -> None:
    for frame_id, item in enumerate(manifest):
        image_name = frame_file_name(item)
        group_id, slot = parse_burst_from_filename(image_name)
        key = (task, group_id, int(slot))
        value = {
            "frame_id": frame_id,
            "frame_position_1based": frame_id + 1,
            "total_frames": len(manifest),
            "image_name": image_name,
        }
        if key in manifest_by_slot:
            issues.append(
                _issue(
                    severity="error",
                    code="duplicate_manifest_group_slot",
                    source_kind=SOURCE_LEGACY_TASK,
                    annotation_path=annotation_path,
                    task=task,
                    group_id=group_id,
                    slot=int(slot),
                    evidence={
                        "first": manifest_by_slot[key],
                        "second": value,
                    },
                    suggested_action=(
                        "Keep one manifest image for each burst slot."
                    ),
                )
            )
        else:
            manifest_by_slot[key] = value


def _audit_legacy_row_fields(
    frame: pd.DataFrame,
    *,
    task_metadata: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    for _, row in frame.iterrows():
        context = _legacy_series_context(row, task_metadata)
        if row["_pig_id"] is None:
            issues.append(
                _issue(
                    severity="error",
                    code="invalid_pig_id",
                    pig_id=row["pig_id"],
                    evidence={"raw_pig_id": row["pig_id"]},
                    suggested_action="Set ID to ID_1 through ID_8.",
                    **context,
                )
            )
        if row["_behavior"] not in VALID_BEHAVIOR_SET:
            issues.append(
                _issue(
                    severity="error",
                    code="invalid_behavior",
                    pig_id=row["_pig_id"],
                    evidence={"raw_behavior": row["behavior"]},
                    suggested_action="Set one canonical 10-class behavior.",
                    **context,
                )
            )
        hidden_present = bool(row["hidden_attribute_present"])
        hidden_valid = (
            str(row["hidden"]).strip().lower() in VALID_HIDDEN_VALUES
        )
        if not hidden_present or not hidden_valid:
            issues.append(
                _issue(
                    severity="error",
                    code="invalid_hidden",
                    pig_id=row["_pig_id"],
                    evidence={
                        "attribute_present": hidden_present,
                        "raw_hidden": row["hidden"],
                    },
                    suggested_action="Set Hidden explicitly to Yes or No.",
                    **context,
                )
            )
        bbox = _bbox_from_values(
            row["x1"],
            row["y1"],
            row["x2"],
            row["y2"],
        )
        if bbox is None:
            issues.append(
                _issue(
                    severity="error",
                    code="invalid_bbox",
                    pig_id=row["_pig_id"],
                    suggested_action="Repair bbox coordinates in CVAT.",
                    **context,
                )
            )
        else:
            width = _safe_int(row["width"], default=0)
            height = _safe_int(row["height"], default=0)
            overshoot = _bbox_boundary_overshoot(
                bbox,
                width=width,
                height=height,
            )
        if bbox is not None and overshoot > 0:
            minor_tolerance = _minor_boundary_tolerance(
                width=width,
                height=height,
            )
            is_minor = overshoot <= minor_tolerance
            issues.append(
                _issue(
                    severity="info" if is_minor else "review",
                    code=(
                        "bbox_minor_boundary_overshoot"
                        if is_minor
                        else "bbox_out_of_image_bounds"
                    ),
                    pig_id=row["_pig_id"],
                    evidence={
                        "bbox": bbox,
                        "image_width": width,
                        "image_height": height,
                        "max_overshoot_px": overshoot,
                        "minor_tolerance_px": minor_tolerance,
                    },
                    suggested_action=(
                        "Minor CVAT interpolation overshoot is recorded and "
                        "may use deterministic downstream clipping."
                        if is_minor
                        else "Inspect and clamp the source bbox in CVAT; do "
                        "not rely on downstream clipping."
                    ),
                    **context,
                )
            )


def _audit_legacy_duplicate_identities(
    frame: pd.DataFrame,
    *,
    task_metadata: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    key = ["task", "group_id", "_slot", "_pig_id"]
    duplicate_mask = frame.duplicated(key, keep=False)
    for _, duplicate in frame.loc[duplicate_mask].groupby(
        key,
        dropna=False,
        sort=True,
    ):
        first = duplicate.iloc[0]
        evidence_rows = []
        for _, row in duplicate.iterrows():
            evidence_rows.append(
                {
                    "behavior": row["_behavior"],
                    "hidden": row["hidden"],
                    "bbox": _bbox_from_values(
                        row["x1"],
                        row["y1"],
                        row["x2"],
                        row["y2"],
                    ),
                }
            )
        issues.append(
            _issue(
                severity="error",
                code="duplicate_anchor_identity",
                pig_id=first["_pig_id"],
                evidence={"rows": evidence_rows},
                suggested_action=(
                    "Correct the duplicated ID; retain every valid distinct "
                    "pig bbox."
                ),
                **_legacy_series_context(first, task_metadata),
            )
        )


def _legacy_authority_rows(
    frame: pd.DataFrame,
    issues: list[dict[str, Any]],
) -> pd.DataFrame:
    tasks_per_group = frame.groupby("group_id", dropna=False)["task"].nunique()
    for group_id in tasks_per_group.loc[tasks_per_group.gt(1)].index:
        issues.append(
            _issue(
                severity="error",
                code="group_in_multiple_tasks",
                source_kind=SOURCE_LEGACY_TASK,
                group_id=group_id,
                suggested_action="Keep each burst in exactly one task.",
            )
        )

    authority = frame.loc[
        frame["_frame_id"].eq(frame["burst_first_task_frame"])
    ].copy()
    duplicate_mask = authority.duplicated(
        ["task", "group_id", "_pig_id"],
        keep=False,
    )
    for _, duplicate in authority.loc[duplicate_mask].groupby(
        ["task", "group_id", "_pig_id"],
        dropna=False,
        sort=True,
    ):
        row = duplicate.iloc[0]
        issues.append(
            _issue(
                severity="error",
                code="duplicate_first_frame_authority",
                source_kind=SOURCE_LEGACY_TASK,
                task=row["task"],
                group_id=row["group_id"],
                pig_id=row["_pig_id"],
                frame_id=int(row["_frame_id"]),
                image_name=row["img_name"],
                evidence={"row_count": len(duplicate)},
                suggested_action=(
                    "Keep one authority bbox per actor on the first task frame."
                ),
            )
        )
    return authority


def _audit_absent_authority_actors(
    frame: pd.DataFrame,
    *,
    absent_keys: pd.DataFrame,
    manifest_by_slot: dict[tuple[str, str, int], dict[str, Any]],
    task_metadata: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    for _, row in absent_keys.iterrows():
        task = row["task"]
        group_id = row["group_id"]
        pig_id = row["_pig_id"]
        actor_rows = frame.loc[
            frame["task"].eq(task)
            & frame["group_id"].eq(group_id)
            & frame["_pig_id"].eq(pig_id)
        ].sort_values("_frame_id")
        first_frame_id = int(actor_rows["burst_first_task_frame"].iloc[0])
        manifest_rows = [
            value
            for (item_task, item_group, _), value in manifest_by_slot.items()
            if item_task == task
            and item_group == group_id
            and value["frame_id"] == first_frame_id
        ]
        first_manifest = manifest_rows[0] if manifest_rows else {
            "frame_id": first_frame_id,
            "frame_position_1based": first_frame_id + 1,
            "total_frames": task_metadata[task]["total_frames"],
            "image_name": "",
        }
        observations = [
            {
                "slot": int(item["_slot"]),
                "frame_id": int(item["_frame_id"]),
                "image_name": item["img_name"],
                "behavior": item["_behavior"],
            }
            for _, item in actor_rows.iterrows()
        ]
        issues.append(
            _issue(
                severity="review",
                code="actor_absent_authority_frame",
                source_kind=SOURCE_LEGACY_TASK,
                annotation_path=task_metadata[task]["annotation_path"],
                task=task,
                group_id=group_id,
                pig_id=pig_id,
                evidence={"observed_later": observations},
                suggested_action=(
                    "Inspect the first task frame. Add an accurate bbox and "
                    "behavior authority if the pig is identifiable; otherwise "
                    "declare this actor key as an exclusion."
                ),
                **first_manifest,
            )
        )


def _add_duplicate_identity_candidates(
    frame: pd.DataFrame,
    *,
    authority: pd.DataFrame,
    task_metadata: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    expected_by_group = (
        authority.groupby(["task", "group_id"])["_pig_id"]
        .apply(lambda values: set(values.dropna()))
        .to_dict()
    )
    duplicate_mask = frame.duplicated(
        ["task", "group_id", "_slot", "_pig_id"],
        keep=False,
    )
    group_columns = ["task", "group_id", "_slot", "_pig_id"]
    for key, duplicate in frame.loc[duplicate_mask].groupby(
        group_columns,
        dropna=False,
        sort=True,
    ):
        task, group_id, slot, duplicate_id = key
        slot_rows = frame.loc[
            frame["task"].eq(task)
            & frame["group_id"].eq(group_id)
            & frame["_slot"].eq(slot)
        ]
        present = set(slot_rows["_pig_id"].dropna())
        missing = sorted(expected_by_group.get((task, group_id), set()) - present)
        if len(duplicate) != 2 or len(missing) != 1:
            continue

        missing_id = missing[0]
        candidates = [duplicate_id, missing_id]
        duplicate_records = [
            row for _, row in duplicate.iterrows()
        ]
        direct_cost = sum(
            _row_actor_distance(
                duplicate_records[index],
                actor_id=candidates[index],
                group_rows=frame.loc[
                    frame["task"].eq(task)
                    & frame["group_id"].eq(group_id)
                ],
            )
            for index in range(2)
        )
        swapped_cost = sum(
            _row_actor_distance(
                duplicate_records[index],
                actor_id=candidates[1 - index],
                group_rows=frame.loc[
                    frame["task"].eq(task)
                    & frame["group_id"].eq(group_id)
                ],
            )
            for index in range(2)
        )
        if not all(map(math.isfinite, [direct_cost, swapped_cost])):
            continue

        assignments = (
            candidates
            if direct_cost <= swapped_cost
            else list(reversed(candidates))
        )
        best_cost = min(direct_cost, swapped_cost)
        alternative_cost = max(direct_cost, swapped_cost)
        if alternative_cost <= 0 or best_cost > alternative_cost * 0.75:
            continue

        mapping = []
        for actor_row, suggested_id in zip(
            duplicate_records,
            assignments,
            strict=True,
        ):
            mapping.append(
                {
                    "bbox": _bbox_from_values(
                        actor_row["x1"],
                        actor_row["y1"],
                        actor_row["x2"],
                        actor_row["y2"],
                    ),
                    "behavior": actor_row["_behavior"],
                    "current_id": duplicate_id,
                    "suggested_id": suggested_id,
                }
            )
        first = duplicate.iloc[0]
        issues.append(
            _issue(
                severity="review",
                code="probable_duplicate_identity_substitution",
                pig_id=duplicate_id,
                evidence={
                    "missing_expected_id": missing_id,
                    "suggested_mapping": mapping,
                    "best_normalized_cost": best_cost,
                    "alternative_normalized_cost": alternative_cost,
                    "auto_fix_safe": False,
                },
                suggested_action=(
                    "Verify the suggested ID mapping against adjacent anchors; "
                    "never apply it automatically."
                ),
                **_legacy_series_context(first, task_metadata),
            )
        )


def _add_sequence_identity_candidates(
    frame: pd.DataFrame,
    *,
    authority: pd.DataFrame,
    task_metadata: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    expected_by_group = (
        authority.groupby(["task", "group_id"])["_pig_id"]
        .apply(lambda values: set(values.dropna()))
        .to_dict()
    )
    for (task, group_id), group_rows in frame.groupby(
        ["task", "group_id"],
        sort=True,
    ):
        expected = expected_by_group.get((task, group_id), set())
        present = set(group_rows["_pig_id"].dropna())
        unexpected = sorted(present - expected)
        for unexpected_id in unexpected:
            unexpected_rows = group_rows.loc[
                group_rows["_pig_id"].eq(unexpected_id)
            ]
            unexpected_slots = set(map(int, unexpected_rows["_slot"]))
            if len(unexpected_slots) < 2:
                continue

            matches: list[tuple[str, float]] = []
            for expected_id in sorted(expected):
                expected_rows = group_rows.loc[
                    group_rows["_pig_id"].eq(expected_id)
                ]
                expected_slots = set(map(int, expected_rows["_slot"]))
                missing_slots = set(EXPECTED_LEGACY_SLOTS) - expected_slots
                if unexpected_slots != missing_slots:
                    continue
                costs = [
                    _row_actor_distance(
                        row,
                        actor_id=expected_id,
                        group_rows=group_rows,
                    )
                    for _, row in unexpected_rows.iterrows()
                ]
                if costs and all(map(math.isfinite, costs)):
                    matches.append((expected_id, sum(costs) / len(costs)))

            if len(matches) != 1 or matches[0][1] > 0.5:
                continue
            expected_id, mean_cost = matches[0]
            first = unexpected_rows.sort_values("_frame_id").iloc[0]
            issues.append(
                _issue(
                    severity="review",
                    code="probable_sequence_identity_substitution",
                    pig_id=unexpected_id,
                    observed_slots=sorted(unexpected_slots),
                    evidence={
                        "suggested_id": expected_id,
                        "mean_normalized_center_distance": mean_cost,
                        "auto_fix_safe": False,
                    },
                    suggested_action=(
                        "Verify whether the unexpected ID should be changed to "
                        "the missing authority actor on these anchors."
                    ),
                    **_legacy_series_context(first, task_metadata),
                )
            )


def _add_tracking_duplicate_candidate(
    *,
    duplicate: pd.DataFrame,
    expected_set: set[str],
    all_frame_rows: pd.DataFrame,
    path: Path,
    task_name: str,
    video_key: str,
    start_frame: int,
    total_frames: int,
    issues: list[dict[str, Any]],
) -> None:
    frame_id = int(duplicate["frame_id"].iloc[0])
    current = all_frame_rows.loc[all_frame_rows["frame_id"].eq(frame_id)]
    present = set(current["pig_id"].dropna().astype(str))
    missing = sorted(expected_set - present)
    label_ids = list(duplicate["label_pig_id"])
    if len(missing) != 1 or missing[0] not in label_ids:
        return

    mapping = [
        {
            "track_id": row.track_id,
            "track_label": row.track_label,
            "current_id": row.pig_id,
            "suggested_id": row.label_pig_id,
            "bbox": row.bbox,
        }
        for row in duplicate.itertuples()
    ]
    issues.append(
        _tracking_issue(
            severity="review",
            code="probable_duplicate_identity_substitution",
            path=path,
            task_name=task_name,
            video_key=video_key,
            frame_id=frame_id,
            start_frame=start_frame,
            total_frames=total_frames,
            pig_id=duplicate["pig_id"].iloc[0],
            evidence={
                "missing_expected_id": missing[0],
                "suggested_mapping": mapping,
                "auto_fix_safe": False,
            },
            suggested_action=(
                "Verify the box ID against its CVAT track label; never apply "
                "the suggestion automatically."
            ),
        )
    )


def _row_actor_distance(
    row: pd.Series,
    *,
    actor_id: str,
    group_rows: pd.DataFrame,
) -> float:
    references = group_rows.loc[
        group_rows["_pig_id"].eq(actor_id)
        & group_rows["_slot"].ne(row["_slot"])
    ].copy()
    if references.empty:
        return math.inf
    references["_slot_distance"] = (
        references["_slot"].astype(float) - float(row["_slot"])
    ).abs()
    references = references.loc[
        references["_slot_distance"].eq(references["_slot_distance"].min())
    ]
    row_center = _center(
        row["x1"],
        row["y1"],
        row["x2"],
        row["y2"],
    )
    distances = []
    for reference in references.itertuples():
        reference_center = _center(
            reference.x1,
            reference.y1,
            reference.x2,
            reference.y2,
        )
        diagonal = math.hypot(
            float(reference.x2) - float(reference.x1),
            float(reference.y2) - float(reference.y1),
        )
        distances.append(
            math.dist(row_center, reference_center) / max(diagonal, 1.0)
        )
    return min(distances)


def _legacy_series_context(
    row: pd.Series,
    task_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "source_kind": SOURCE_LEGACY_TASK,
        "annotation_path": task_metadata[row["task"]]["annotation_path"],
        "task": row["task"],
        "group_id": row["group_id"],
        "slot": int(row["_slot"]),
        "frame_id": int(row["_frame_id"]),
        "frame_position_1based": int(row["_frame_id"]) + 1,
        "total_frames": task_metadata[row["task"]]["total_frames"],
        "image_name": row["img_name"],
    }


def _tracking_issue(
    *,
    severity: str,
    code: str,
    path: Path,
    task_name: str,
    video_key: str,
    frame_id: int,
    start_frame: int,
    total_frames: int,
    pig_id: str | None,
    evidence: dict[str, Any],
    suggested_action: str,
) -> dict[str, Any]:
    return _issue(
        severity=severity,
        code=code,
        source_kind=SOURCE_TRACKING_XML,
        annotation_path=path,
        task=task_name,
        video_key=video_key,
        pig_id=pig_id,
        frame_id=frame_id,
        frame_position_1based=frame_id - start_frame + 1,
        total_frames=total_frames,
        image_name=f"{video_key}__f{frame_id:06d}.jpg",
        evidence=evidence,
        suggested_action=suggested_action,
    )


def _source_report(
    *,
    source_kind: str,
    source_path: Path,
    summary: dict[str, Any],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    sorted_issues = sorted(issues, key=_issue_sort_key)
    return {
        "source_kind": source_kind,
        "source_path": str(source_path),
        "status": _status(sorted_issues),
        "summary": summary,
        "issues": sorted_issues,
    }


def _issue(
    *,
    severity: str,
    code: str,
    source_kind: str,
    annotation_path: str | Path = "",
    task: str = "",
    video_key: str = "",
    group_id: str = "",
    pig_id: Any = None,
    slot: int | None = None,
    frame_id: int | None = None,
    frame_position_1based: int | None = None,
    total_frames: int | None = None,
    image_name: str = "",
    observed_slots: Sequence[int] | None = None,
    missing_slots: Sequence[int] | None = None,
    evidence: dict[str, Any] | None = None,
    suggested_action: str = "",
) -> dict[str, Any]:
    return {
        "severity": severity,
        "code": code,
        "source_kind": source_kind,
        "annotation_path": str(annotation_path),
        "task": task,
        "video_key": video_key,
        "group_id": group_id,
        "pig_id": None if pig_id is None else str(pig_id),
        "slot": slot,
        "frame_id": frame_id,
        "frame_position_1based": frame_position_1based,
        "total_frames": total_frames,
        "image_name": image_name,
        "observed_slots": list(observed_slots or []),
        "missing_slots": list(missing_slots or []),
        "evidence": _json_safe(evidence or {}),
        "suggested_action": suggested_action,
    }


def _issue_sort_key(issue: dict[str, Any]) -> tuple[Any, ...]:
    return (
        issue.get("source_kind", ""),
        issue.get("task", ""),
        issue.get("video_key", ""),
        issue.get("group_id", ""),
        issue.get("frame_id")
        if issue.get("frame_id") is not None
        else -1,
        issue.get("pig_id") or "",
        issue.get("code", ""),
    )


def _status(issues: Sequence[dict[str, Any]]) -> str:
    if any(issue["severity"] == "error" for issue in issues):
        return "FAIL"
    if any(issue["severity"] == "review" for issue in issues):
        return "REVIEW_REQUIRED"
    return "PASS"


def _raw_box_attributes(box: ET.Element) -> list[tuple[str, str]]:
    return [
        (
            str(attribute.attrib.get("name", "")).strip(),
            "" if attribute.text is None else str(attribute.text).strip(),
        )
        for attribute in box.findall("attribute")
    ]


def _tracking_bbox(box: ET.Element) -> list[float] | None:
    return _bbox_from_values(
        box.attrib.get("xtl"),
        box.attrib.get("ytl"),
        box.attrib.get("xbr"),
        box.attrib.get("ybr"),
    )


def _bbox_from_values(
    x1: Any,
    y1: Any,
    x2: Any,
    y2: Any,
) -> list[float] | None:
    try:
        values = [float(x1), float(y1), float(x2), float(y2)]
    except (TypeError, ValueError):
        return None
    if not all(map(math.isfinite, values)):
        return None
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values


def _bbox_boundary_overshoot(
    bbox: Sequence[float],
    *,
    width: int,
    height: int,
) -> float:
    if width <= 0 or height <= 0:
        return 0.0
    return max(
        0.0,
        -bbox[0],
        -bbox[1],
        bbox[2] - width,
        bbox[3] - height,
    )


def _minor_boundary_tolerance(*, width: int, height: int) -> float:
    if width <= 0 or height <= 0:
        return 0.0
    return max(2.0, min(width, height) * 0.005)


def _center(x1: Any, y1: Any, x2: Any, y2: Any) -> tuple[float, float]:
    return (
        (float(x1) + float(x2)) / 2.0,
        (float(y1) + float(y2)) / 2.0,
    )


def _pig_id_from_track_label(label: str) -> str | None:
    match = re.search(r"(\d+)$", label.strip())
    if match is None:
        return normalize_pig_id(label)
    return normalize_pig_id(f"ID_{match.group(1)}")


def _tracking_video_key(root: ET.Element, path: Path) -> str:
    source = _xml_text(root, "./meta/task/source", "")
    task_name = _xml_text(root, "./meta/task/name", path.stem)
    for candidate in [task_name, Path(source).stem, path.stem]:
        match = re.search(
            r"(?i)\bpigs\d+_\d+(?:_\d+fps)?",
            candidate,
        )
        if match is not None:
            text = match.group(0)
            return f"Pigs{text[4:]}"
    if source:
        return Path(source).stem
    return re.sub(r"\s*\([^)]*\)\s*$", "", task_name).strip() or path.stem


def _xml_text(root: ET.Element, query: str, default: str) -> str:
    value = root.findtext(query)
    if value is None:
        return default
    text = value.strip()
    return text if text else default


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            pass
    return value
