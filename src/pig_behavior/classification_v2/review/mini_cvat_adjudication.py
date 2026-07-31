"""Versioned sidecar contract for local mini-CVAT corrections.

The contract is deliberately isolated from source annotations and Behavior
decision ledgers. Geometry and Hidden are frame/object attributes. Pig ID and
behavior are actor/burst attributes and therefore cannot diverge by frame.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from pig_behavior.classification_v2.review.behavior_review_contract import (
    CANONICAL_BEHAVIORS,
)

MINI_CVAT_SCHEMA = "classification_v2.mini_cvat_adjudication.v2"
LEGACY_MINI_CVAT_SCHEMA = "classification_v2.mini_cvat_adjudication.v1"
MINI_CVAT_SIDECAR_NAME = "mini_cvat_adjudication.json"
MODEL_X_FORBIDDEN = "YES"
HIDDEN_VALUES = frozenset({"Yes", "No", "Unclear"})
FRAME_BBOX_MODES = frozenset(
    {
        "SOURCE_BBOX",
        "SOURCE_BBOX_CORRECTED",
        "MISSING_BBOX_ADDED",
    }
)


class MiniCvatAdjudicationError(ValueError):
    """Raised when a mini-CVAT sidecar is unsafe or internally inconsistent."""


@dataclass(frozen=True)
class MiniCvatActorAttributes:
    """One actor-level correction shared by every frame in the burst."""

    actor_scope_id: str
    original_pig_id: str
    reviewed_pig_id: str
    original_behavior: str
    reviewed_behavior: str


@dataclass(frozen=True)
class MiniCvatFrameAnnotation:
    """One saved frame/object geometry and visibility decision."""

    actor_scope_id: str
    frame_index: int
    source_frame_index: int
    original_object_track_key: str
    original_track_id: str
    original_pig_id: str
    reviewed_pig_id: str
    bbox_mode: str
    x1: float
    y1: float
    x2: float
    y2: float
    original_hidden: str
    reviewed_hidden: str

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Return source-image xyxy coordinates."""

        return self.x1, self.y1, self.x2, self.y2


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _validate_bbox(
    annotation: MiniCvatFrameAnnotation,
) -> None:
    coordinates = annotation.bbox
    if not all(math.isfinite(value) and value >= 0.0 for value in coordinates):
        raise MiniCvatAdjudicationError(
            "mini_cvat_bbox_coordinate_invalid="
            f"{annotation.actor_scope_id}:{annotation.frame_index}"
        )
    if annotation.x2 <= annotation.x1 or annotation.y2 <= annotation.y1:
        raise MiniCvatAdjudicationError(
            "mini_cvat_bbox_extent_invalid="
            f"{annotation.actor_scope_id}:{annotation.frame_index}"
        )


def validate_mini_cvat_state(
    actor_attributes: Mapping[str, MiniCvatActorAttributes],
    frame_annotations: Mapping[tuple[str, int], MiniCvatFrameAnnotation],
    *,
    editable_actor_ids: Sequence[str],
    frame_indices: Sequence[int],
    require_complete: bool,
) -> list[str]:
    """Validate state without silently inventing actor or frame decisions."""

    errors: list[str] = []
    raw_actors = tuple(_text(value) for value in editable_actor_ids)
    expected_actors = tuple(dict.fromkeys(raw_actors))
    expected_frames = tuple(int(value) for value in frame_indices)
    expected_actor_set = set(expected_actors)
    expected_frame_set = set(expected_frames)

    if not expected_actors or any(not value for value in expected_actors):
        errors.append("mini_cvat_editable_actor_scope_invalid")
    if len(raw_actors) != len(set(raw_actors)):
        errors.append("mini_cvat_duplicate_editable_actor")
    if not expected_frames or len(expected_frames) != len(set(expected_frames)):
        errors.append("mini_cvat_frame_scope_invalid")

    for actor_id, attributes in actor_attributes.items():
        if actor_id not in expected_actor_set:
            errors.append(f"mini_cvat_unknown_actor_attributes={actor_id}")
            continue
        if attributes.actor_scope_id != actor_id:
            errors.append(f"mini_cvat_actor_attribute_key_mismatch={actor_id}")
        if not attributes.original_pig_id or not attributes.reviewed_pig_id:
            errors.append(f"mini_cvat_actor_identity_blank={actor_id}")
        if attributes.original_behavior not in CANONICAL_BEHAVIORS:
            errors.append(f"mini_cvat_original_behavior_invalid={actor_id}")
        if attributes.reviewed_behavior not in CANONICAL_BEHAVIORS:
            errors.append(f"mini_cvat_reviewed_behavior_invalid={actor_id}")

    for key, annotation in frame_annotations.items():
        actor_id, frame_index = key
        if actor_id not in expected_actor_set or frame_index not in expected_frame_set:
            errors.append(f"mini_cvat_frame_outside_scope={actor_id}:{frame_index}")
            continue
        if annotation.actor_scope_id != actor_id:
            errors.append(f"mini_cvat_frame_actor_key_mismatch={actor_id}:{frame_index}")
        if annotation.frame_index != frame_index:
            errors.append(f"mini_cvat_frame_key_mismatch={actor_id}:{frame_index}")
        if annotation.bbox_mode not in FRAME_BBOX_MODES:
            errors.append(f"mini_cvat_bbox_mode_invalid={actor_id}:{frame_index}")
        if not annotation.reviewed_pig_id:
            errors.append(f"mini_cvat_frame_identity_blank={actor_id}:{frame_index}")
        if annotation.reviewed_hidden not in HIDDEN_VALUES:
            errors.append(f"mini_cvat_hidden_invalid={actor_id}:{frame_index}")
        try:
            _validate_bbox(annotation)
        except MiniCvatAdjudicationError as exc:
            errors.append(str(exc))

    if require_complete:
        for actor_id in expected_actors:
            if actor_id not in actor_attributes:
                errors.append(f"mini_cvat_actor_attributes_pending={actor_id}")
            for frame_index in expected_frames:
                if (actor_id, frame_index) not in frame_annotations:
                    errors.append(
                        f"mini_cvat_frame_pending={actor_id}:{frame_index}"
                    )

    for frame_index in expected_frames:
        reviewed_ids = []
        for actor_id in expected_actors:
            annotation = frame_annotations.get((actor_id, frame_index))
            attributes = actor_attributes.get(actor_id)
            if annotation is not None:
                reviewed_ids.append(annotation.reviewed_pig_id)
            elif attributes is not None:
                reviewed_ids.append(attributes.reviewed_pig_id)
            else:
                reviewed_ids.append(actor_id)
        duplicates = sorted(
            {
                value
                for value in reviewed_ids
                if reviewed_ids.count(value) > 1
            }
        )
        if duplicates:
            errors.append(
                "mini_cvat_duplicate_reviewed_pig_id="
                f"{frame_index}:{','.join(duplicates)}"
            )
    return errors


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".mini-cvat-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def write_mini_cvat_sidecar(
    output_dir: Path,
    *,
    reviewer: str,
    source_type: str,
    dataset_id: str,
    video_key: str,
    editable_actor_ids: Sequence[str],
    frame_indices: Sequence[int],
    actor_attributes: Mapping[str, MiniCvatActorAttributes],
    frame_annotations: Mapping[tuple[str, int], MiniCvatFrameAnnotation],
) -> Path:
    """Write the isolated correction sidecar atomically."""

    errors = validate_mini_cvat_state(
        actor_attributes,
        frame_annotations,
        editable_actor_ids=editable_actor_ids,
        frame_indices=frame_indices,
        require_complete=False,
    )
    if errors:
        raise MiniCvatAdjudicationError(";".join(errors))
    if not reviewer.strip():
        raise MiniCvatAdjudicationError("mini_cvat_reviewer_required")

    path = Path(output_dir) / MINI_CVAT_SIDECAR_NAME
    payload = {
        "schema": MINI_CVAT_SCHEMA,
        "reviewer": reviewer.strip(),
        "source_type": source_type,
        "dataset_id": dataset_id,
        "video_key": video_key,
        "editable_actor_ids": list(editable_actor_ids),
        "frame_indices": [int(value) for value in frame_indices],
        "actor_attributes": [
            asdict(actor_attributes[key]) for key in sorted(actor_attributes)
        ],
        "frame_annotations": [
            asdict(frame_annotations[key])
            for key in sorted(frame_annotations, key=lambda value: (value[0], value[1]))
        ],
        "source_annotations_changed": "NO",
        "behavior_decision_ledger_touched": "NO",
        "model_x_forbidden": MODEL_X_FORBIDDEN,
    }
    _write_json_atomic(path, payload)
    return path


def load_mini_cvat_sidecar(
    output_dir: Path,
    *,
    source_type: str,
    dataset_id: str,
    video_key: str,
    editable_actor_ids: Sequence[str],
    frame_indices: Sequence[int],
) -> tuple[
    dict[str, MiniCvatActorAttributes],
    dict[tuple[str, int], MiniCvatFrameAnnotation],
]:
    """Load a sidecar and fail closed on stale scope or provenance."""

    path = Path(output_dir) / MINI_CVAT_SIDECAR_NAME
    if not path.exists():
        return {}, {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MiniCvatAdjudicationError("mini_cvat_sidecar_unreadable") from exc
    if not isinstance(payload, dict):
        raise MiniCvatAdjudicationError("mini_cvat_sidecar_not_object")
    payload_schema = payload.get("schema")
    if payload_schema not in {MINI_CVAT_SCHEMA, LEGACY_MINI_CVAT_SCHEMA}:
        raise MiniCvatAdjudicationError("mini_cvat_sidecar_schema_mismatch")
    if payload.get("model_x_forbidden") != MODEL_X_FORBIDDEN:
        raise MiniCvatAdjudicationError("mini_cvat_model_x_boundary_mismatch")

    expected = {
        "source_type": source_type,
        "dataset_id": dataset_id,
        "video_key": video_key,
        "editable_actor_ids": list(editable_actor_ids),
        "frame_indices": [int(value) for value in frame_indices],
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise MiniCvatAdjudicationError(
                f"mini_cvat_sidecar_scope_mismatch={field}"
            )

    try:
        actor_rows = payload.get("actor_attributes", [])
        frame_rows = payload.get("frame_annotations", [])
        attributes = {
            str(row["actor_scope_id"]): MiniCvatActorAttributes(**row)
            for row in actor_rows
        }
        annotations = {}
        for row in frame_rows:
            actor_id = str(row["actor_scope_id"])
            migrated_row = dict(row)
            if payload_schema == LEGACY_MINI_CVAT_SCHEMA:
                attributes_for_actor = attributes.get(actor_id)
                migrated_row["reviewed_pig_id"] = (
                    attributes_for_actor.reviewed_pig_id
                    if attributes_for_actor is not None
                    else actor_id
                )
            annotations[(actor_id, int(row["frame_index"]))] = (
                MiniCvatFrameAnnotation(**migrated_row)
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise MiniCvatAdjudicationError("mini_cvat_sidecar_payload_invalid") from exc
    if len(attributes) != len(actor_rows):
        raise MiniCvatAdjudicationError("mini_cvat_duplicate_actor_attributes")
    if len(annotations) != len(frame_rows):
        raise MiniCvatAdjudicationError("mini_cvat_duplicate_frame_annotation")

    errors = validate_mini_cvat_state(
        attributes,
        annotations,
        editable_actor_ids=editable_actor_ids,
        frame_indices=frame_indices,
        require_complete=False,
    )
    if errors:
        raise MiniCvatAdjudicationError(";".join(errors))
    return attributes, annotations
