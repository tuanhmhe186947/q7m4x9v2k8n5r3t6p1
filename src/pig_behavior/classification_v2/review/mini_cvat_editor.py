"""Deterministic editor state for the Classification V2 mini-CVAT GUI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from pig_behavior.classification_v2.review.behavior_review_contract import (
    CANONICAL_BEHAVIORS,
)
from pig_behavior.classification_v2.review.identity_continuity_adjudication import (
    ADDED_BBOX_MODE,
    CORRECTED_BBOX_MODE,
    SOURCE_BBOX_MODE,
    FrameCandidate,
)
from pig_behavior.classification_v2.review.mini_cvat_adjudication import (
    HIDDEN_VALUES,
    MiniCvatActorAttributes,
    MiniCvatFrameAnnotation,
    validate_mini_cvat_state,
)

BBox = tuple[float, float, float, float]
Point = tuple[float, float]
HANDLE_NAMES = ("nw", "n", "ne", "e", "se", "s", "sw", "w")
MINIMUM_BBOX_EXTENT = 3.0


class MiniCvatEditorError(ValueError):
    """Raised when a draft cannot be represented without ambiguity."""


@dataclass(frozen=True)
class FrameDraft:
    """One editable object draft that is not persisted until explicit save."""

    actor_scope_id: str
    frame_index: int
    source_frame_index: int
    original_object_track_key: str
    original_track_id: str
    original_pig_id: str
    reviewed_pig_id: str
    bbox_mode: str
    bbox: BBox | None
    original_hidden: str
    reviewed_hidden: str
    dirty: bool = False


@dataclass(frozen=True)
class FrameSaveResult:
    """Result of one atomic frame save."""

    annotation: MiniCvatFrameAnnotation
    swapped_actor_scope_id: str
    previous_reviewed_pig_id: str


@dataclass(frozen=True)
class DragIntent:
    """One stable mouse gesture captured at button press."""

    mode: str
    start_canvas: Point
    origin_bbox: BBox | None
    resize_handle: str = ""


def normalize_pig_id(value: str) -> str:
    """Normalize a bare numeric identity into the project ID form."""

    normalized = value.strip()
    if not normalized:
        return ""
    prefix, separator, suffix = normalized.partition("_")
    if separator and prefix.upper() == "ID" and suffix.strip():
        return f"ID_{suffix.strip()}"
    if normalized.isdigit():
        return f"ID_{normalized}"
    return normalized


def source_bbox_to_canvas(
    bbox: BBox,
    *,
    scale: float,
    offset: Point,
) -> BBox:
    """Transform one source-image bbox into canvas coordinates."""

    return (
        bbox[0] * scale + offset[0],
        bbox[1] * scale + offset[1],
        bbox[2] * scale + offset[0],
        bbox[3] * scale + offset[1],
    )


def bbox_handle_points(bbox: BBox) -> dict[str, Point]:
    """Return the eight resize handle centers for one bbox."""

    x1, y1, x2, y2 = bbox
    mid_x = (x1 + x2) / 2.0
    mid_y = (y1 + y2) / 2.0
    return {
        "nw": (x1, y1),
        "n": (mid_x, y1),
        "ne": (x2, y1),
        "e": (x2, mid_y),
        "se": (x2, y2),
        "s": (mid_x, y2),
        "sw": (x1, y2),
        "w": (x1, mid_y),
    }


def begin_bbox_drag(
    point: Point,
    bbox: BBox | None,
    *,
    scale: float,
    offset: Point,
    handle_radius: float = 10.0,
) -> DragIntent | None:
    """Capture resize before move so overlapping hit zones stay deterministic."""

    if bbox is None or scale <= 0.0:
        return None
    canvas_bbox = source_bbox_to_canvas(bbox, scale=scale, offset=offset)
    for handle, center in bbox_handle_points(canvas_bbox).items():
        if (
            abs(point[0] - center[0]) <= handle_radius
            and abs(point[1] - center[1]) <= handle_radius
        ):
            return DragIntent("resize", point, bbox, handle)
    if (
        canvas_bbox[0] <= point[0] <= canvas_bbox[2]
        and canvas_bbox[1] <= point[1] <= canvas_bbox[3]
    ):
        return DragIntent("move", point, bbox)
    return None


def _clamp_bbox(bbox: BBox, source_size: tuple[int, int]) -> BBox:
    width, height = source_size
    x1, y1, x2, y2 = bbox
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    return x1, y1, x2, y2


def preview_bbox_drag(
    intent: DragIntent,
    current_canvas: Point,
    *,
    scale: float,
    source_size: tuple[int, int],
) -> BBox | None:
    """Return a preview bbox without mutating editor state."""

    if intent.origin_bbox is None or scale <= 0.0:
        return None
    dx = (current_canvas[0] - intent.start_canvas[0]) / scale
    dy = (current_canvas[1] - intent.start_canvas[1]) / scale
    x1, y1, x2, y2 = intent.origin_bbox
    if intent.mode == "move":
        x1 += dx
        x2 += dx
        y1 += dy
        y2 += dy
        if x1 < 0.0:
            x2 -= x1
            x1 = 0.0
        if y1 < 0.0:
            y2 -= y1
            y1 = 0.0
        if x2 > source_size[0]:
            x1 -= x2 - source_size[0]
            x2 = float(source_size[0])
        if y2 > source_size[1]:
            y1 -= y2 - source_size[1]
            y2 = float(source_size[1])
        return x1, y1, x2, y2

    handle = intent.resize_handle
    if "w" in handle:
        x1 += dx
    if "e" in handle:
        x2 += dx
    if "n" in handle:
        y1 += dy
    if "s" in handle:
        y2 += dy
    x1, y1, x2, y2 = _clamp_bbox((x1, y1, x2, y2), source_size)
    if x2 - x1 < MINIMUM_BBOX_EXTENT:
        if "w" in handle:
            x1 = x2 - MINIMUM_BBOX_EXTENT
        else:
            x2 = x1 + MINIMUM_BBOX_EXTENT
    if y2 - y1 < MINIMUM_BBOX_EXTENT:
        if "n" in handle:
            y1 = y2 - MINIMUM_BBOX_EXTENT
        else:
            y2 = y1 + MINIMUM_BBOX_EXTENT
    return _clamp_bbox((x1, y1, x2, y2), source_size)


def bbox_from_canvas_drag(
    start: Point,
    end: Point,
    *,
    scale: float,
    offset: Point,
    source_size: tuple[int, int],
) -> BBox | None:
    """Create a normalized source bbox from a canvas draw gesture."""

    if scale <= 0.0:
        return None
    source_points = []
    for point in (start, end):
        source_points.append(
            (
                (point[0] - offset[0]) / scale,
                (point[1] - offset[1]) / scale,
            )
        )
    x1, x2 = sorted((source_points[0][0], source_points[1][0]))
    y1, y2 = sorted((source_points[0][1], source_points[1][1]))
    x1, y1, x2, y2 = _clamp_bbox((x1, y1, x2, y2), source_size)
    if (
        x2 - x1 < MINIMUM_BBOX_EXTENT
        or y2 - y1 < MINIMUM_BBOX_EXTENT
    ):
        return None
    return x1, y1, x2, y2


def smallest_candidate_at_point(
    candidates: Sequence[FrameCandidate],
    point: Point,
    *,
    scale: float,
    offset: Point,
) -> FrameCandidate | None:
    """Select the smallest source bbox containing the canvas point."""

    if scale <= 0.0:
        return None
    source_x = (point[0] - offset[0]) / scale
    source_y = (point[1] - offset[1]) / scale
    matches = [
        candidate
        for candidate in candidates
        if candidate.x1 <= source_x <= candidate.x2
        and candidate.y1 <= source_y <= candidate.y2
    ]
    if not matches:
        return None
    return min(
        matches,
        key=lambda candidate: (
            (candidate.x2 - candidate.x1)
            * (candidate.y2 - candidate.y1)
        ),
    )


class MiniCvatEditorState:
    """Single authority for frame drafts, ID swaps, and burst behavior."""

    def __init__(
        self,
        *,
        editable_actor_ids: Sequence[str],
        frame_indices: Sequence[int],
        candidates_by_frame: Mapping[int, Sequence[FrameCandidate]],
        actor_attributes: Mapping[str, MiniCvatActorAttributes],
        frame_annotations: Mapping[
            tuple[str, int],
            MiniCvatFrameAnnotation,
        ],
    ) -> None:
        self.editable_actor_ids = tuple(editable_actor_ids)
        self.frame_indices = tuple(int(value) for value in frame_indices)
        self.candidates_by_frame = {
            int(key): tuple(value)
            for key, value in candidates_by_frame.items()
        }
        self.actor_attributes = dict(actor_attributes)
        self.frame_annotations = dict(frame_annotations)
        self._ensure_default_actor_attributes()
        self._validate()

    def _source_behavior(self, actor_scope_id: str) -> str:
        for frame_index in self.frame_indices:
            candidate = self.source_candidate(actor_scope_id, frame_index)
            if (
                candidate is not None
                and candidate.behavior in CANONICAL_BEHAVIORS
            ):
                return candidate.behavior
        return ""

    def _ensure_default_actor_attributes(self) -> None:
        for actor_scope_id in self.editable_actor_ids:
            if actor_scope_id in self.actor_attributes:
                continue
            behavior = self._source_behavior(actor_scope_id)
            if behavior not in CANONICAL_BEHAVIORS:
                raise MiniCvatEditorError(
                    f"source_behavior_unavailable={actor_scope_id}"
                )
            self.actor_attributes[actor_scope_id] = MiniCvatActorAttributes(
                actor_scope_id=actor_scope_id,
                original_pig_id=actor_scope_id,
                reviewed_pig_id=actor_scope_id,
                original_behavior=behavior,
                reviewed_behavior=behavior,
            )

    def _validate(self) -> None:
        errors = validate_mini_cvat_state(
            self.actor_attributes,
            self.frame_annotations,
            editable_actor_ids=self.editable_actor_ids,
            frame_indices=self.frame_indices,
            require_complete=False,
        )
        if errors:
            raise MiniCvatEditorError(";".join(errors))

    def source_candidate(
        self,
        actor_scope_id: str,
        frame_index: int,
    ) -> FrameCandidate | None:
        candidates = self.candidates_by_frame.get(frame_index, ())
        annotation = self.frame_annotations.get(
            (actor_scope_id, frame_index)
        )
        if annotation is not None and annotation.original_object_track_key:
            matched = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.object_track_key
                    == annotation.original_object_track_key
                ),
                None,
            )
            if matched is not None:
                return matched
        matches = [
            candidate
            for candidate in candidates
            if candidate.pig_id == actor_scope_id
        ]
        return matches[0] if len(matches) == 1 else None

    def effective_reviewed_id(
        self,
        actor_scope_id: str,
        frame_index: int,
    ) -> str:
        annotation = self.frame_annotations.get(
            (actor_scope_id, frame_index)
        )
        if annotation is not None:
            return annotation.reviewed_pig_id
        return self.actor_attributes[actor_scope_id].reviewed_pig_id

    def reviewed_behavior(self, actor_scope_id: str) -> str:
        return self.actor_attributes[actor_scope_id].reviewed_behavior

    def draft(
        self,
        actor_scope_id: str,
        frame_index: int,
    ) -> FrameDraft:
        saved = self.frame_annotations.get((actor_scope_id, frame_index))
        if saved is not None:
            return FrameDraft(
                actor_scope_id=saved.actor_scope_id,
                frame_index=saved.frame_index,
                source_frame_index=saved.source_frame_index,
                original_object_track_key=saved.original_object_track_key,
                original_track_id=saved.original_track_id,
                original_pig_id=saved.original_pig_id,
                reviewed_pig_id=saved.reviewed_pig_id,
                bbox_mode=saved.bbox_mode,
                bbox=saved.bbox,
                original_hidden=saved.original_hidden,
                reviewed_hidden=saved.reviewed_hidden,
            )
        candidate = self.source_candidate(actor_scope_id, frame_index)
        if candidate is None:
            return FrameDraft(
                actor_scope_id=actor_scope_id,
                frame_index=frame_index,
                source_frame_index=frame_index,
                original_object_track_key="",
                original_track_id="",
                original_pig_id=actor_scope_id,
                reviewed_pig_id=self.effective_reviewed_id(
                    actor_scope_id,
                    frame_index,
                ),
                bbox_mode=ADDED_BBOX_MODE,
                bbox=None,
                original_hidden="",
                reviewed_hidden="",
            )
        hidden = candidate.hidden if candidate.hidden in HIDDEN_VALUES else ""
        return FrameDraft(
            actor_scope_id=actor_scope_id,
            frame_index=frame_index,
            source_frame_index=candidate.source_frame_index,
            original_object_track_key=candidate.object_track_key,
            original_track_id=candidate.track_id,
            original_pig_id=candidate.pig_id,
            reviewed_pig_id=self.effective_reviewed_id(
                actor_scope_id,
                frame_index,
            ),
            bbox_mode=SOURCE_BBOX_MODE,
            bbox=candidate.bbox,
            original_hidden=candidate.hidden,
            reviewed_hidden=hidden,
        )

    def change_draft(
        self,
        draft: FrameDraft,
        *,
        reviewed_pig_id: str | None = None,
        reviewed_hidden: str | None = None,
        bbox: BBox | None = None,
        bbox_mode: str | None = None,
    ) -> FrameDraft:
        return replace(
            draft,
            reviewed_pig_id=(
                draft.reviewed_pig_id
                if reviewed_pig_id is None
                else normalize_pig_id(reviewed_pig_id)
            ),
            reviewed_hidden=(
                draft.reviewed_hidden
                if reviewed_hidden is None
                else reviewed_hidden
            ),
            bbox=draft.bbox if bbox is None else bbox,
            bbox_mode=draft.bbox_mode if bbox_mode is None else bbox_mode,
            dirty=True,
        )

    def _annotation_from_draft(
        self,
        draft: FrameDraft,
    ) -> MiniCvatFrameAnnotation:
        if draft.bbox is None:
            raise MiniCvatEditorError("frame_bbox_required")
        if draft.reviewed_hidden not in HIDDEN_VALUES:
            raise MiniCvatEditorError("frame_hidden_required")
        reviewed_id = normalize_pig_id(draft.reviewed_pig_id)
        if reviewed_id not in self.editable_actor_ids:
            raise MiniCvatEditorError(
                f"reviewed_id_outside_editable_scope={reviewed_id}"
            )
        return MiniCvatFrameAnnotation(
            actor_scope_id=draft.actor_scope_id,
            frame_index=draft.frame_index,
            source_frame_index=draft.source_frame_index,
            original_object_track_key=draft.original_object_track_key,
            original_track_id=draft.original_track_id,
            original_pig_id=draft.original_pig_id,
            reviewed_pig_id=reviewed_id,
            bbox_mode=draft.bbox_mode,
            x1=draft.bbox[0],
            y1=draft.bbox[1],
            x2=draft.bbox[2],
            y2=draft.bbox[3],
            original_hidden=draft.original_hidden,
            reviewed_hidden=draft.reviewed_hidden,
        )

    def _source_annotation(
        self,
        actor_scope_id: str,
        frame_index: int,
        reviewed_pig_id: str,
    ) -> MiniCvatFrameAnnotation | None:
        candidate = self.source_candidate(actor_scope_id, frame_index)
        if candidate is None or candidate.hidden not in HIDDEN_VALUES:
            return None
        return MiniCvatFrameAnnotation(
            actor_scope_id=actor_scope_id,
            frame_index=frame_index,
            source_frame_index=candidate.source_frame_index,
            original_object_track_key=candidate.object_track_key,
            original_track_id=candidate.track_id,
            original_pig_id=candidate.pig_id,
            reviewed_pig_id=reviewed_pig_id,
            bbox_mode=SOURCE_BBOX_MODE,
            x1=candidate.x1,
            y1=candidate.y1,
            x2=candidate.x2,
            y2=candidate.y2,
            original_hidden=candidate.hidden,
            reviewed_hidden=candidate.hidden,
        )

    def save_frame(self, draft: FrameDraft) -> FrameSaveResult:
        """Atomically save one frame and swap an occupied reviewed ID."""

        annotation = self._annotation_from_draft(draft)
        previous_id = self.effective_reviewed_id(
            draft.actor_scope_id,
            draft.frame_index,
        )
        owner = next(
            (
                actor_scope_id
                for actor_scope_id in self.editable_actor_ids
                if actor_scope_id != draft.actor_scope_id
                and self.effective_reviewed_id(
                    actor_scope_id,
                    draft.frame_index,
                )
                == annotation.reviewed_pig_id
            ),
            "",
        )
        updated = dict(self.frame_annotations)
        updated[(draft.actor_scope_id, draft.frame_index)] = annotation
        if owner and annotation.reviewed_pig_id != previous_id:
            owner_key = (owner, draft.frame_index)
            owner_annotation = updated.get(owner_key)
            if owner_annotation is None:
                owner_annotation = self._source_annotation(
                    owner,
                    draft.frame_index,
                    previous_id,
                )
            elif owner_annotation.reviewed_pig_id != previous_id:
                owner_annotation = replace(
                    owner_annotation,
                    reviewed_pig_id=previous_id,
                )
            if owner_annotation is None:
                raise MiniCvatEditorError(
                    f"swap_owner_bbox_unavailable={owner}:{draft.frame_index}"
                )
            updated[owner_key] = owner_annotation
        errors = validate_mini_cvat_state(
            self.actor_attributes,
            updated,
            editable_actor_ids=self.editable_actor_ids,
            frame_indices=self.frame_indices,
            require_complete=False,
        )
        if errors:
            raise MiniCvatEditorError(";".join(errors))
        self.frame_annotations = updated
        return FrameSaveResult(annotation, owner, previous_id)

    def save_behavior(
        self,
        actor_scope_id: str,
        behavior: str,
    ) -> MiniCvatActorAttributes:
        """Update behavior for the actor burst without touching frame IDs."""

        if behavior not in CANONICAL_BEHAVIORS:
            raise MiniCvatEditorError(f"invalid_burst_behavior={behavior}")
        current = self.actor_attributes[actor_scope_id]
        updated = replace(current, reviewed_behavior=behavior)
        attributes = dict(self.actor_attributes)
        attributes[actor_scope_id] = updated
        errors = validate_mini_cvat_state(
            attributes,
            self.frame_annotations,
            editable_actor_ids=self.editable_actor_ids,
            frame_indices=self.frame_indices,
            require_complete=False,
        )
        if errors:
            raise MiniCvatEditorError(";".join(errors))
        self.actor_attributes = attributes
        return updated

    def saved_frame_count(self, actor_scope_id: str) -> int:
        return sum(
            actor_id == actor_scope_id
            for actor_id, _frame_index in self.frame_annotations
        )

    def corrected_bbox_mode(
        self,
        actor_scope_id: str,
        frame_index: int,
        bbox: BBox,
    ) -> str:
        candidate = self.source_candidate(actor_scope_id, frame_index)
        if candidate is None:
            return ADDED_BBOX_MODE
        return (
            SOURCE_BBOX_MODE
            if bbox == candidate.bbox
            else CORRECTED_BBOX_MODE
        )
