"""Apply isolated identity adjudication to explicit CSV and CVAT XML sources."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.review.identity_continuity_adjudication import (
    assert_safe_identity_input_path,
)
from pig_behavior.classification_v2.review.mini_cvat_adjudication import (
    MINI_CVAT_SCHEMA,
    MiniCvatActorAttributes,
    MiniCvatFrameAnnotation,
    validate_mini_cvat_state,
)
from pig_behavior.classification_v2.schema import behavior_to_coarse

APPLY_SCHEMA = "classification_v2.identity_source_apply.v1"
APPLY_MARKER_COLUMNS = (
    "identity_review_status",
    "identity_reviewer",
    "identity_sidecar_sha256",
    "identity_actor_scope_id",
    "identity_frame_annotation_applied",
    "identity_source_apply_version",
)
DERIVED_FEATURE_MARKERS = {
    "evidence_semantics_version",
    "motion_schema_version",
    "nearest_dist_n",
    "roi_target_available",
    "spatiotemporal_feature_valid",
}
IMAGE_FRAME_PATTERN = re.compile(
    r"^(?P<group>.+)_f(?P<frame>\d+)(?:_k\d+)?\.[^.]+$",
    re.IGNORECASE,
)


class IdentitySourceApplyError(ValueError):
    """Raised when identity corrections cannot be applied without ambiguity."""


@dataclass(frozen=True)
class IdentitySnapshot:
    """Validated mini-CVAT sidecar state."""

    path: Path
    sha256: str
    reviewer: str
    source_type: str
    dataset_id: str
    video_key: str
    editable_actor_ids: tuple[str, ...]
    frame_indices: tuple[int, ...]
    actor_attributes: Mapping[str, MiniCvatActorAttributes]
    frame_annotations: Mapping[
        tuple[str, int],
        MiniCvatFrameAnnotation,
    ]


@dataclass(frozen=True)
class PreparedTarget:
    """One fully rendered target ready for an atomic transaction."""

    path: Path
    kind: str
    temporary_path: Path
    before_sha256: str
    after_sha256: str
    row_count_before: int
    row_count_after: int
    behavior_updates: int
    bbox_updates: int
    identity_updates: int
    hidden_updates: int
    matched_actor_rows: int
    backup_path: Path


@dataclass(frozen=True)
class IdentityApplyResult:
    """Completed apply transaction."""

    manifest_path: Path
    generation_dir: Path
    group_id: str
    changed_target_count: int
    targets: tuple[PreparedTarget, ...]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _integer(value: Any, *, field: str) -> int:
    try:
        return int(float(_text(value)))
    except (TypeError, ValueError) as exc:
        raise IdentitySourceApplyError(
            f"invalid_integer={field}:{value}"
        ) from exc


def _load_snapshot(sidecar_path: Path) -> IdentitySnapshot:
    path = assert_safe_identity_input_path(
        sidecar_path,
        role="identity_sidecar",
    )
    if not path.is_file():
        raise IdentitySourceApplyError(f"sidecar_not_found={path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IdentitySourceApplyError("sidecar_unreadable") from exc
    if not isinstance(payload, dict):
        raise IdentitySourceApplyError("sidecar_not_object")
    if payload.get("schema") != MINI_CVAT_SCHEMA:
        raise IdentitySourceApplyError("sidecar_schema_not_v2")
    if payload.get("model_x_forbidden") != "YES":
        raise IdentitySourceApplyError("sidecar_model_x_boundary_mismatch")
    try:
        actor_attributes = {
            _text(row["actor_scope_id"]): MiniCvatActorAttributes(**row)
            for row in payload["actor_attributes"]
        }
        frame_annotations = {}
        for row in payload["frame_annotations"]:
            annotation = MiniCvatFrameAnnotation(**row)
            key = (annotation.actor_scope_id, annotation.frame_index)
            if key in frame_annotations:
                raise IdentitySourceApplyError(
                    f"duplicate_sidecar_frame_annotation={key}"
                )
            frame_annotations[key] = annotation
        editable_actor_ids = tuple(
            _text(value) for value in payload["editable_actor_ids"]
        )
        frame_indices = tuple(int(value) for value in payload["frame_indices"])
    except (KeyError, TypeError, ValueError) as exc:
        raise IdentitySourceApplyError("sidecar_payload_invalid") from exc
    errors = validate_mini_cvat_state(
        actor_attributes,
        frame_annotations,
        editable_actor_ids=editable_actor_ids,
        frame_indices=frame_indices,
        require_complete=False,
    )
    if errors:
        raise IdentitySourceApplyError(";".join(errors))
    reviewer = _text(payload.get("reviewer"))
    if not reviewer:
        raise IdentitySourceApplyError("sidecar_reviewer_required")
    return IdentitySnapshot(
        path=path,
        sha256=sha256_file(path),
        reviewer=reviewer,
        source_type=_text(payload.get("source_type")),
        dataset_id=_text(payload.get("dataset_id")),
        video_key=_text(payload.get("video_key")),
        editable_actor_ids=editable_actor_ids,
        frame_indices=frame_indices,
        actor_attributes=actor_attributes,
        frame_annotations=frame_annotations,
    )


def _image_scope(value: str) -> tuple[str, int] | None:
    match = IMAGE_FRAME_PATTERN.match(Path(value).name)
    if match is None:
        return None
    return match.group("group"), int(match.group("frame"))


def _csv_group_and_frame(row: Mapping[str, str]) -> tuple[str, int] | None:
    group_id = _text(row.get("group_id")) or _text(row.get("clip_id"))
    frame_value = _text(row.get("frame_index"))
    if group_id and frame_value:
        return group_id, _integer(frame_value, field="frame_index")
    for column in ("img_name", "image_name"):
        parsed = _image_scope(_text(row.get(column)))
        if parsed is not None:
            if group_id and parsed[0] != group_id:
                raise IdentitySourceApplyError(
                    "csv_group_image_name_mismatch="
                    f"{group_id}:{parsed[0]}"
                )
            return parsed
    return None


def _csv_kind(fieldnames: Sequence[str]) -> str:
    fields = set(fieldnames)
    if DERIVED_FEATURE_MARKERS.intersection(fields):
        raise IdentitySourceApplyError(
            "derived_feature_csv_requires_feature_rebuild"
        )
    if {
        "group_id",
        "tracklet_id",
        "frame_index",
        "pig_id",
        "behavior",
        "x1",
        "y1",
        "x2",
        "y2",
    }.issubset(fields):
        return "dense_frame_objects"
    if {
        "group_id",
        "pig_id",
        "behavior",
        "x1",
        "y1",
        "x2",
        "y2",
    }.issubset(fields) and (
        "img_name" in fields or "image_name" in fields
    ):
        return "anchor_frame_objects"
    raise IdentitySourceApplyError(
        f"unsupported_identity_csv_schema={sorted(fields)[:20]}"
    )


def _original_track_ids(snapshot: IdentitySnapshot) -> set[str]:
    return {
        annotation.original_track_id
        for annotation in snapshot.frame_annotations.values()
        if annotation.original_track_id
    }


def _infer_group_id(
    snapshot: IdentitySnapshot,
    csv_paths: Sequence[Path],
    explicit_group_id: str,
) -> str:
    if explicit_group_id:
        return explicit_group_id
    track_ids = _original_track_ids(snapshot)
    groups: set[str] = set()
    for raw_path in csv_paths:
        path = assert_safe_identity_input_path(
            raw_path,
            role="identity_source_csv",
        )
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise IdentitySourceApplyError(f"csv_header_missing={path}")
            if "tracklet_id" not in reader.fieldnames:
                continue
            for row in reader:
                if _text(row.get("tracklet_id")) not in track_ids:
                    continue
                scope = _csv_group_and_frame(row)
                if scope is not None:
                    groups.add(scope[0])
    if len(groups) != 1:
        raise IdentitySourceApplyError(
            "identity_group_id_not_unique="
            + ",".join(sorted(groups))
        )
    return next(iter(groups))


def _actor_indexes(
    snapshot: IdentitySnapshot,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    by_track: dict[str, str] = {}
    by_object: dict[str, str] = {}
    by_original_id: dict[str, str] = {}
    for actor_id, attributes in snapshot.actor_attributes.items():
        by_original_id[attributes.original_pig_id] = actor_id
    for annotation in snapshot.frame_annotations.values():
        if annotation.original_track_id:
            by_track[annotation.original_track_id] = annotation.actor_scope_id
        if annotation.original_object_track_key:
            by_object[
                annotation.original_object_track_key
            ] = annotation.actor_scope_id
    return by_track, by_object, by_original_id


def _actor_for_row(
    row: Mapping[str, str],
    *,
    by_track: Mapping[str, str],
    by_object: Mapping[str, str],
    by_original_id: Mapping[str, str],
) -> str:
    track_id = _text(row.get("tracklet_id")) or _text(row.get("track_id"))
    if track_id in by_track:
        return by_track[track_id]
    object_key = _text(row.get("object_track_key"))
    if object_key in by_object:
        return by_object[object_key]
    return by_original_id.get(_text(row.get("pig_id")), "")


def _set_if_changed(
    row: dict[str, str],
    column: str,
    value: Any,
) -> bool:
    if column not in row:
        return False
    rendered = _text(value)
    if _text(row.get(column)) == rendered:
        return False
    row[column] = rendered
    return True


def _float_text(value: float) -> str:
    if not math.isfinite(value):
        raise IdentitySourceApplyError("nonfinite_reviewed_bbox")
    rendered = f"{value:.10f}".rstrip("0").rstrip(".")
    return rendered if rendered else "0"


def _recompute_geometry(
    row: dict[str, str],
    annotation: MiniCvatFrameAnnotation,
) -> None:
    x1, y1, x2, y2 = annotation.bbox
    width = x2 - x1
    height = y2 - y1
    if width <= 0.0 or height <= 0.0:
        raise IdentitySourceApplyError("reviewed_bbox_extent_invalid")
    for column, value in (
        ("x1", x1),
        ("y1", y1),
        ("x2", x2),
        ("y2", y2),
        ("x1_raw", x1),
        ("y1_raw", y1),
        ("x2_raw", x2),
        ("y2_raw", y2),
        ("bbox_w", width),
        ("bbox_h", height),
        ("bbox_area", width * height),
        ("cx", (x1 + x2) / 2.0),
        ("cy", (y1 + y2) / 2.0),
    ):
        if column in row:
            row[column] = _float_text(value)
    image_width = float(_text(row.get("image_width") or row.get("width") or 0))
    image_height = float(
        _text(row.get("image_height") or row.get("height") or 0)
    )
    if image_width > 0.0 and image_height > 0.0:
        normalized = (
            ("cx_n", (x1 + x2) / 2.0 / image_width),
            ("cy_n", (y1 + y2) / 2.0 / image_height),
            ("bw_n", width / image_width),
            ("bh_n", height / image_height),
            ("area_n", width * height / (image_width * image_height)),
        )
        for column, value in normalized:
            if column in row:
                row[column] = _float_text(value)
    for column in ("bbox_raw_valid", "bbox_valid", "actor_bbox_valid"):
        if column in row:
            row[column] = "True"
    if "bbox_was_clipped" in row:
        row["bbox_was_clipped"] = "False"


def _write_csv_target(
    snapshot: IdentitySnapshot,
    path: Path,
    *,
    group_id: str,
    temporary_path: Path,
) -> dict[str, Any]:
    by_track, by_object, by_original_id = _actor_indexes(snapshot)
    behavior_updates = 0
    bbox_updates = 0
    identity_updates = 0
    hidden_updates = 0
    matched_actor_rows = 0
    matched_annotations: set[tuple[str, int]] = set()
    row_count = 0
    frame_id_sets: dict[int, list[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise IdentitySourceApplyError(f"csv_header_missing={path}")
        kind = _csv_kind(reader.fieldnames)
        fieldnames = list(reader.fieldnames)
        for column in APPLY_MARKER_COLUMNS:
            if column not in fieldnames:
                fieldnames.append(column)
        with temporary_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as target:
            writer = csv.DictWriter(
                target,
                fieldnames=fieldnames,
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            for source_row in reader:
                row_count += 1
                row = dict(source_row)
                scope = _csv_group_and_frame(row)
                if scope is None or scope[0] != group_id:
                    writer.writerow(row)
                    continue
                actor_id = _actor_for_row(
                    row,
                    by_track=by_track,
                    by_object=by_object,
                    by_original_id=by_original_id,
                )
                if not actor_id:
                    writer.writerow(row)
                    continue
                matched_actor_rows += 1
                frame_index = scope[1]
                attributes = snapshot.actor_attributes[actor_id]
                if _set_if_changed(
                    row,
                    "behavior",
                    attributes.reviewed_behavior,
                ):
                    behavior_updates += 1
                if "behavior_coarse" in row:
                    row["behavior_coarse"] = behavior_to_coarse(
                        attributes.reviewed_behavior
                    )
                if "label_source" in row:
                    row["label_source"] = "human_identity_adjudication"
                if "behavior_authority_policy" in row:
                    row["behavior_authority_policy"] = (
                        "human_identity_adjudication_burst"
                    )
                if "behavior_disagrees_with_authority" in row:
                    row["behavior_disagrees_with_authority"] = "False"
                annotation = snapshot.frame_annotations.get(
                    (actor_id, frame_index)
                )
                if annotation is not None:
                    matched_annotations.add((actor_id, frame_index))
                    before_bbox = tuple(
                        _text(row.get(column))
                        for column in ("x1", "y1", "x2", "y2")
                    )
                    _recompute_geometry(row, annotation)
                    after_bbox = tuple(
                        _text(row.get(column))
                        for column in ("x1", "y1", "x2", "y2")
                    )
                    if before_bbox != after_bbox:
                        bbox_updates += 1
                    identity_changed = _set_if_changed(
                        row,
                        "pig_id",
                        annotation.reviewed_pig_id,
                    )
                    for column in ("track_label", "object_id_in_image"):
                        if (
                            column in row
                            and _text(row.get(column))
                            == annotation.original_pig_id
                        ):
                            row[column] = annotation.reviewed_pig_id
                            identity_changed = True
                    if identity_changed:
                        identity_updates += 1
                    if _set_if_changed(
                        row,
                        "hidden",
                        annotation.reviewed_hidden,
                    ):
                        hidden_updates += 1
                        if "hidden_source" in row:
                            row["hidden_source"] = (
                                "human_identity_adjudication"
                            )
                    if "bbox_source" in row and _text(
                        row.get("bbox_source")
                    ) != "gt_legacy":
                        row["bbox_source"] = (
                            "human_identity_adjudication"
                        )
                    row["identity_frame_annotation_applied"] = "YES"
                else:
                    row["identity_frame_annotation_applied"] = "NO"
                row["identity_review_status"] = "APPLIED"
                row["identity_reviewer"] = snapshot.reviewer
                row["identity_sidecar_sha256"] = snapshot.sha256
                row["identity_actor_scope_id"] = actor_id
                row["identity_source_apply_version"] = APPLY_SCHEMA
                frame_id_sets.setdefault(frame_index, []).append(
                    _text(row.get("pig_id"))
                )
                writer.writerow(row)
    expected_annotations = {
        key
        for key in snapshot.frame_annotations
        if key[1] in {
            frame
            for frame, identities in frame_id_sets.items()
            if identities
        }
    }
    missing = sorted(expected_annotations.difference(matched_annotations))
    if kind == "dense_frame_objects" and missing:
        raise IdentitySourceApplyError(
            f"csv_dense_annotations_unmatched={path}:{missing[:10]}"
        )
    duplicates = {
        frame: sorted(
            value
            for value in set(identities)
            if value and identities.count(value) > 1
        )
        for frame, identities in frame_id_sets.items()
    }
    duplicates = {
        frame: values for frame, values in duplicates.items() if values
    }
    if duplicates:
        raise IdentitySourceApplyError(
            f"csv_duplicate_reviewed_ids={path}:{duplicates}"
        )
    if matched_actor_rows == 0:
        raise IdentitySourceApplyError(
            f"csv_identity_scope_not_found={path}:{group_id}"
        )
    return {
        "kind": kind,
        "row_count": row_count,
        "behavior_updates": behavior_updates,
        "bbox_updates": bbox_updates,
        "identity_updates": identity_updates,
        "hidden_updates": hidden_updates,
        "matched_actor_rows": matched_actor_rows,
    }


def _xml_attribute(box: ET.Element, name: str) -> ET.Element | None:
    for attribute in box.findall("attribute"):
        if _text(attribute.attrib.get("name")) == name:
            return attribute
    return None


def _xml_attribute_text(box: ET.Element, name: str) -> str:
    attribute = _xml_attribute(box, name)
    return "" if attribute is None else _text(attribute.text)


def _set_xml_attribute(box: ET.Element, name: str, value: str) -> bool:
    attribute = _xml_attribute(box, name)
    if attribute is None:
        attribute = ET.SubElement(box, "attribute", {"name": name})
    if _text(attribute.text) == value:
        return False
    attribute.text = value
    return True


def _write_xml_target(
    snapshot: IdentitySnapshot,
    path: Path,
    *,
    group_id: str,
    temporary_path: Path,
) -> dict[str, Any]:
    try:
        tree = ET.parse(path)
    except (ET.ParseError, OSError) as exc:
        raise IdentitySourceApplyError(f"xml_unreadable={path}") from exc
    root = tree.getroot()
    behavior_updates = 0
    bbox_updates = 0
    identity_updates = 0
    hidden_updates = 0
    matched_actor_rows = 0
    matched_annotations: set[tuple[str, int]] = set()
    matched_frames: set[int] = set()
    for image in root.findall(".//image"):
        scope = _image_scope(_text(image.attrib.get("name")))
        if scope is None or scope[0] != group_id:
            continue
        frame_index = scope[1]
        matched_frames.add(frame_index)
        boxes_by_original_id: dict[str, ET.Element] = {}
        for box in image.findall("box"):
            pig_id = _xml_attribute_text(box, "ID")
            if pig_id in boxes_by_original_id:
                raise IdentitySourceApplyError(
                    "xml_duplicate_original_id="
                    f"{group_id}:{frame_index}:{pig_id}"
                )
            boxes_by_original_id[pig_id] = box
        reviewed_ids: list[str] = []
        for actor_id, attributes in snapshot.actor_attributes.items():
            box = boxes_by_original_id.get(attributes.original_pig_id)
            if box is None:
                continue
            matched_actor_rows += 1
            if _set_xml_attribute(
                box,
                "Behavior",
                attributes.reviewed_behavior,
            ):
                behavior_updates += 1
                box.set("source", "manual")
            annotation = snapshot.frame_annotations.get(
                (actor_id, frame_index)
            )
            if annotation is not None:
                matched_annotations.add((actor_id, frame_index))
                bbox_changed = False
                for column, value in zip(
                    ("xtl", "ytl", "xbr", "ybr"),
                    annotation.bbox,
                    strict=True,
                ):
                    rendered = _float_text(value)
                    if _text(box.attrib.get(column)) != rendered:
                        box.set(column, rendered)
                        bbox_changed = True
                if bbox_changed:
                    bbox_updates += 1
                    box.set("source", "manual")
                if _set_xml_attribute(
                    box,
                    "ID",
                    annotation.reviewed_pig_id,
                ):
                    identity_updates += 1
                    box.set("source", "manual")
                if _set_xml_attribute(
                    box,
                    "Hidden",
                    annotation.reviewed_hidden,
                ):
                    hidden_updates += 1
                    box.set("source", "manual")
                reviewed_ids.append(annotation.reviewed_pig_id)
            else:
                reviewed_ids.append(attributes.original_pig_id)
        duplicate_ids = sorted(
            value
            for value in set(reviewed_ids)
            if reviewed_ids.count(value) > 1
        )
        if duplicate_ids:
            raise IdentitySourceApplyError(
                "xml_duplicate_reviewed_ids="
                f"{group_id}:{frame_index}:{duplicate_ids}"
            )
    if not matched_frames:
        raise IdentitySourceApplyError(
            f"xml_identity_scope_not_found={path}:{group_id}"
        )
    expected_xml_annotations = {
        key
        for key in snapshot.frame_annotations
        if key[1] in matched_frames
    }
    missing = sorted(expected_xml_annotations.difference(matched_annotations))
    if missing:
        raise IdentitySourceApplyError(
            f"xml_anchor_annotations_unmatched={path}:{missing[:10]}"
        )
    ET.indent(tree, space="  ")
    tree.write(
        temporary_path,
        encoding="utf-8",
        xml_declaration=True,
        short_empty_elements=True,
    )
    return {
        "kind": "cvat_image_xml",
        "row_count": len(root.findall(".//box")),
        "behavior_updates": behavior_updates,
        "bbox_updates": bbox_updates,
        "identity_updates": identity_updates,
        "hidden_updates": hidden_updates,
        "matched_actor_rows": matched_actor_rows,
    }


def _temporary_target(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.identity-",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    return Path(name)


def _backup_path(
    backup_dir: Path,
    target: Path,
    index: int,
) -> Path:
    safe_name = target.name.replace(" ", "_")
    return backup_dir / f"{index:02d}_{safe_name}"


def _prepare_targets(
    snapshot: IdentitySnapshot,
    *,
    csv_paths: Sequence[Path],
    xml_path: Path,
    group_id: str,
    generation_dir: Path,
) -> tuple[PreparedTarget, ...]:
    backup_dir = generation_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=False)
    targets: list[PreparedTarget] = []
    all_paths = [
        *(
            (
                assert_safe_identity_input_path(
                    path,
                    role="identity_source_csv",
                ),
                "csv",
            )
            for path in csv_paths
        ),
        (
            assert_safe_identity_input_path(
                xml_path,
                role="identity_source_xml",
            ),
            "xml",
        ),
    ]
    resolved_paths = [path for path, _kind in all_paths]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise IdentitySourceApplyError("duplicate_apply_target_path")
    created_temporaries: list[Path] = []
    try:
        for index, (path, target_type) in enumerate(all_paths):
            if not path.is_file():
                raise IdentitySourceApplyError(
                    f"apply_target_not_found={path}"
                )
            temporary_path = _temporary_target(path)
            created_temporaries.append(temporary_path)
            if target_type == "csv":
                stats = _write_csv_target(
                    snapshot,
                    path,
                    group_id=group_id,
                    temporary_path=temporary_path,
                )
            else:
                stats = _write_xml_target(
                    snapshot,
                    path,
                    group_id=group_id,
                    temporary_path=temporary_path,
                )
            before_sha256 = sha256_file(path)
            after_sha256 = sha256_file(temporary_path)
            backup_path = _backup_path(backup_dir, path, index)
            targets.append(
                PreparedTarget(
                    path=path,
                    kind=stats["kind"],
                    temporary_path=temporary_path,
                    before_sha256=before_sha256,
                    after_sha256=after_sha256,
                    row_count_before=stats["row_count"],
                    row_count_after=stats["row_count"],
                    behavior_updates=stats["behavior_updates"],
                    bbox_updates=stats["bbox_updates"],
                    identity_updates=stats["identity_updates"],
                    hidden_updates=stats["hidden_updates"],
                    matched_actor_rows=stats["matched_actor_rows"],
                    backup_path=backup_path,
                )
            )
    except Exception:
        for temporary_path in created_temporaries:
            temporary_path.unlink(missing_ok=True)
        raise
    return tuple(targets)


def _manifest_payload(
    snapshot: IdentitySnapshot,
    *,
    group_id: str,
    targets: Iterable[PreparedTarget],
    status: str,
) -> dict[str, Any]:
    target_rows = []
    for target in targets:
        row = asdict(target)
        for key in ("path", "temporary_path", "backup_path"):
            row[key] = str(row[key])
        target_rows.append(row)
    return {
        "schema": APPLY_SCHEMA,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "group_id": group_id,
        "sidecar_path": str(snapshot.path),
        "sidecar_sha256": snapshot.sha256,
        "reviewer": snapshot.reviewer,
        "source_type": snapshot.source_type,
        "dataset_id": snapshot.dataset_id,
        "video_key": snapshot.video_key,
        "editable_actor_ids": list(snapshot.editable_actor_ids),
        "saved_frame_annotation_count": len(snapshot.frame_annotations),
        "actor_attributes": [
            asdict(snapshot.actor_attributes[key])
            for key in sorted(snapshot.actor_attributes)
        ],
        "targets": target_rows,
        "behavior_decision_ledger_touched": "NO",
        "review_manifest_changed": "NO",
        "source_annotations_changed": "YES",
    }


def apply_identity_adjudication(
    *,
    sidecar_path: Path,
    csv_paths: Sequence[Path],
    xml_path: Path,
    audit_root: Path,
    group_id: str = "",
) -> IdentityApplyResult:
    """Apply one sidecar to explicit sources with backup and rollback."""

    if not csv_paths:
        raise IdentitySourceApplyError("at_least_one_source_csv_required")
    snapshot = _load_snapshot(sidecar_path)
    resolved_group_id = _infer_group_id(
        snapshot,
        csv_paths,
        _text(group_id),
    )
    generation_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid.uuid4().hex[:12]
    )
    generation_dir = Path(audit_root).resolve() / generation_id
    generation_dir.mkdir(parents=True, exist_ok=False)
    prepared: tuple[PreparedTarget, ...] = ()
    committed: list[PreparedTarget] = []
    try:
        prepared = _prepare_targets(
            snapshot,
            csv_paths=csv_paths,
            xml_path=xml_path,
            group_id=resolved_group_id,
            generation_dir=generation_dir,
        )
        for target in prepared:
            shutil.copy2(target.path, target.backup_path)
        for target in prepared:
            if target.before_sha256 == target.after_sha256:
                target.temporary_path.unlink(missing_ok=True)
                continue
            current_sha256 = sha256_file(target.path)
            if current_sha256 != target.before_sha256:
                raise IdentitySourceApplyError(
                    f"target_changed_during_prepare={target.path}"
                )
            os.replace(target.temporary_path, target.path)
            committed.append(target)
        for target in prepared:
            actual_hash = sha256_file(target.path)
            if actual_hash != target.after_sha256:
                raise IdentitySourceApplyError(
                    f"post_apply_hash_mismatch={target.path}"
                )
        manifest_path = generation_dir / "identity_source_apply_manifest.json"
        payload = _manifest_payload(
            snapshot,
            group_id=resolved_group_id,
            targets=prepared,
            status="APPLIED",
        )
        manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        latest_path = Path(audit_root).resolve() / (
            "latest_identity_source_apply.json"
        )
        latest_path.write_text(
            json.dumps(
                {
                    "schema": APPLY_SCHEMA,
                    "manifest_path": str(manifest_path),
                    "group_id": resolved_group_id,
                    "sidecar_sha256": snapshot.sha256,
                    "target_after_hashes": {
                        str(target.path): target.after_sha256
                        for target in prepared
                    },
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return IdentityApplyResult(
            manifest_path=manifest_path,
            generation_dir=generation_dir,
            group_id=resolved_group_id,
            changed_target_count=len(committed),
            targets=prepared,
        )
    except Exception:
        for target in reversed(committed):
            if target.backup_path.is_file():
                shutil.copy2(target.backup_path, target.path)
        for target in prepared:
            target.temporary_path.unlink(missing_ok=True)
        failure_path = generation_dir / "identity_source_apply_failure.json"
        failure_path.write_text(
            json.dumps(
                {
                    "schema": APPLY_SCHEMA,
                    "status": "ROLLED_BACK",
                    "group_id": resolved_group_id,
                    "sidecar_path": str(snapshot.path),
                    "sidecar_sha256": snapshot.sha256,
                    "committed_before_failure": [
                        str(target.path) for target in committed
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
