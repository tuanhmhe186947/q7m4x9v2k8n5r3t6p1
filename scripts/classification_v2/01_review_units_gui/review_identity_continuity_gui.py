"""Adjudicate local actor-trajectory continuity without touching behavior labels.

The GUI reads an immutable review-unit view and source actor boxes.  It writes
only versioned identity-continuity sidecars.  In particular, it never loads or
overwrites a behavior decision ledger.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tkinter as tk
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from tkinter import messagebox, simpledialog, ttk
from typing import Any

from PIL import Image, ImageDraw, ImageTk

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only in incomplete operators envs
    cv2 = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pig_behavior.classification_v2.review.behavior_review_contract import (
    CANONICAL_BEHAVIORS,
)
from pig_behavior.classification_v2.review.identity_continuity_adjudication import (
    ADDED_BBOX_MODE,
    CASE_SIDECAR_NAME,
    CORRECTED_BBOX_MODE,
    FRAME_SIDECAR_NAME,
    MANUAL_BBOX_SELECTION_KEY,
    BoundingBoxEdit,
    FrameCandidate,
    IdentityAdjudicationError,
    IdentityCase,
    assert_safe_output_dir,
    assert_single_scene,
    case_status,
    load_frame_candidates,
    load_identity_cases,
    load_session_sidecars_with_bbox_edits,
    source_frame_index_for_review_frame,
    validate_adjudication,
    write_session_sidecars,
)
from pig_behavior.classification_v2.review.mini_cvat_adjudication import (
    HIDDEN_VALUES,
    MiniCvatActorAttributes,
    MiniCvatAdjudicationError,
    MiniCvatFrameAnnotation,
    load_mini_cvat_sidecar,
    validate_mini_cvat_state,
    write_mini_cvat_sidecar,
)

WINDOW_TITLE = "Classification V2 — Hiệu chỉnh liên tục actor"
MAX_RENDERED_FRAME_CACHE = 12
FINALIZATION_FILE_NAME = "identity_continuity_finalization.json"
FINALIZATION_SCHEMA = "classification_v2.identity_continuity_finalization.v2"
FINALIZED_STATUS = "FINALIZED"
REOPENED_STATUS = "REOPENED"
BEHAVIOR_LEDGER_FILE_NAMES = frozenset(
    {
        "behavior_unit_review_decisions.csv",
        "behavior_strength_review_decisions.csv",
        "behavior_label_quality_review.csv",
    }
)
BEHAVIOR_LEDGER_PATH_PREFIX = "human_review_workspace/classification_v2/"
BOX_COLORS = (
    "#4d4d4d",
    "#0070c0",
    "#7030a0",
    "#bf9000",
    "#a61c00",
    "#38761d",
    "#134f5c",
    "#cc0000",
)
ACTIVE_CASE_COLORS = ("#00a65a", "#ff8c00", "#8e44ad", "#c0392b")


@dataclass(frozen=True)
class IdentityGuiConfig:
    review_units_csv: Path
    frame_features_csv: Path
    output_dir: Path
    reviewer: str
    review_item_ids: tuple[str, ...]
    video_path: Path | None = None
    reopen_finalized: bool = False
    editable_pig_ids: tuple[str, ...] = ()


class RenderedFrameCache:
    """Bounded RGB cache so adjacent-frame review avoids repeated decoding."""

    def __init__(self, max_items: int = MAX_RENDERED_FRAME_CACHE) -> None:
        self.max_items = max_items
        self._items: OrderedDict[int, Image.Image] = OrderedDict()

    def get(self, frame_index: int) -> Image.Image | None:
        image = self._items.get(frame_index)
        if image is None:
            return None
        self._items.move_to_end(frame_index)
        return image.copy()

    def put(self, frame_index: int, image: Image.Image) -> None:
        self._items[frame_index] = image.copy()
        self._items.move_to_end(frame_index)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)


def _normalized_path(value: Path | str) -> str:
    return "/".join(
        part.casefold()
        for part in Path(value).resolve().parts
    )


def _is_behavior_ledger_path(value: Path) -> bool:
    path = Path(value).resolve()
    normalized = _normalized_path(path)
    return (
        path.name.casefold() in BEHAVIOR_LEDGER_FILE_NAMES
        or (
            BEHAVIOR_LEDGER_PATH_PREFIX in normalized
            and "/human_decisions/behavior/" in normalized
        )
    )


def assert_safe_cli_input_paths(config: IdentityGuiConfig) -> None:
    """Reject behavior-ledger paths before the GUI reads any CLI path."""

    values: list[tuple[str, Path | None]] = [
        ("review_units_csv", config.review_units_csv),
        ("frame_features_csv", config.frame_features_csv),
        ("output_dir", config.output_dir),
        ("video_path", config.video_path),
    ]
    unsafe = [
        name
        for name, value in values
        if value is not None and _is_behavior_ledger_path(value)
    ]
    if unsafe:
        raise IdentityAdjudicationError(
            "identity_adjudication_cli_path_is_behavior_ledger="
            + ",".join(unsafe)
        )


def _declared_video_authorities(
    candidates_by_frame: Mapping[int, Sequence[FrameCandidate]],
) -> tuple[Path, ...]:
    declared: dict[str, Path] = {}
    for candidates in candidates_by_frame.values():
        for candidate in candidates:
            raw = candidate.source_video_path.strip()
            if raw:
                path = Path(raw)
                if _is_behavior_ledger_path(path):
                    raise IdentityAdjudicationError(
                        "declared_source_video_path_is_behavior_ledger="
                        f"{path}"
                    )
                declared.setdefault(_normalized_path(path), path)
    return tuple(declared[key] for key in sorted(declared))


def _explicit_path_matches_authority(
    explicit_video_path: Path,
    declared_authority: Path,
) -> bool:
    """Require the selected media path to equal the declared source path."""

    explicit = Path(explicit_video_path).resolve()
    declared = Path(declared_authority)
    return declared.is_file() and explicit == declared.resolve()


def decode_exact_source_frame(capture: Any, source_frame_index: int) -> Any:
    """Decode exactly the requested source frame or fail closed."""

    if not capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame_index):
        raise RuntimeError(f"cannot_seek_source_frame={source_frame_index}")
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(
            f"cannot_decode_source_frame={source_frame_index}"
        )
    decoded_position = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
    decoded_index = int(round(decoded_position)) - 1
    if decoded_index != source_frame_index:
        raise RuntimeError(
            "decoded_source_frame_mismatch="
            f"requested:{source_frame_index};decoded:{decoded_index}"
        )
    return frame


def assert_candidate_bounds(
    candidates: Sequence[FrameCandidate],
    *,
    review_frame_index: int,
    image_width: int,
    image_height: int,
) -> None:
    """Require every selectable source bbox to fit the decoded full frame."""

    for candidate in candidates:
        if (
            candidate.x1 < 0.0
            or candidate.y1 < 0.0
            or candidate.x2 > image_width
            or candidate.y2 > image_height
        ):
            raise RuntimeError(
                "candidate_bbox_outside_decoded_video="
                f"review:{review_frame_index};"
                f"object_track_key:{candidate.object_track_key};"
                f"image:{image_width}x{image_height}"
            )


def first_pending_case_frame(
    case: IdentityCase,
    selections: Mapping[tuple[str, int], str],
) -> int:
    """Return an in-scope frame, preferring unfinished work."""

    return next(
        (
            frame_index
            for frame_index in case.frame_indices
            if (case.review_unit_id, frame_index) not in selections
        ),
        case.frame_indices[0],
    )


def step_case_frame(
    frame_indices: Sequence[int],
    current_frame_index: int,
    delta: int,
) -> int:
    """Advance only within the active review unit's exact frame scope."""

    if not frame_indices:
        raise ValueError("active_identity_case_has_no_frames")
    try:
        current_position = list(frame_indices).index(current_frame_index)
    except ValueError:
        return int(frame_indices[0])
    return int(frame_indices[(current_position + delta) % len(frame_indices)])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(dict(payload), handle, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def load_finalization_marker(output_dir: Path) -> dict[str, str] | None:
    """Load and verify the local immutable-finalization marker, if present."""

    marker_path = Path(output_dir) / FINALIZATION_FILE_NAME
    if not marker_path.exists():
        return None
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IdentityAdjudicationError(
            "identity_finalization_marker_invalid"
        ) from exc
    if not isinstance(payload, dict):
        raise IdentityAdjudicationError("identity_finalization_marker_not_object")
    normalized = {str(key): str(value) for key, value in payload.items()}
    if normalized.get("schema") != FINALIZATION_SCHEMA:
        raise IdentityAdjudicationError("identity_finalization_marker_schema_mismatch")
    status = normalized.get("status")
    if status not in {FINALIZED_STATUS, REOPENED_STATUS}:
        raise IdentityAdjudicationError("identity_finalization_marker_status_invalid")
    if status == FINALIZED_STATUS:
        for file_name, field_name in (
            (FRAME_SIDECAR_NAME, "frame_sidecar_sha256"),
            (CASE_SIDECAR_NAME, "case_sidecar_sha256"),
        ):
            sidecar_path = Path(output_dir) / file_name
            if not sidecar_path.is_file() or (
                normalized.get(field_name) != _sha256_file(sidecar_path)
            ):
                raise IdentityAdjudicationError(
                    "identity_finalization_sidecar_hash_mismatch="
                    f"{file_name}"
                )
    return normalized


def write_finalization_marker(
    output_dir: Path,
    *,
    reviewer: str,
) -> Path:
    """Lock the current sidecars by writing their immutable local hashes."""

    output_dir = Path(output_dir)
    frame_path = output_dir / FRAME_SIDECAR_NAME
    case_path = output_dir / CASE_SIDECAR_NAME
    if not frame_path.is_file() or not case_path.is_file():
        raise IdentityAdjudicationError(
            "identity_finalization_requires_both_sidecars"
        )
    marker_path = output_dir / FINALIZATION_FILE_NAME
    _write_json_atomic(
        marker_path,
        {
            "schema": FINALIZATION_SCHEMA,
            "status": FINALIZED_STATUS,
            "reviewer": reviewer.strip(),
            "frame_sidecar_sha256": _sha256_file(frame_path),
            "case_sidecar_sha256": _sha256_file(case_path),
        },
    )
    return marker_path


def reopen_finalization_marker(output_dir: Path, *, reviewer: str) -> Path:
    """Explicitly reopen a finalized local session without deleting its marker."""

    existing = load_finalization_marker(output_dir)
    if existing is None or existing.get("status") != FINALIZED_STATUS:
        raise IdentityAdjudicationError(
            "identity_finalization_reopen_requires_finalized_marker"
        )
    marker_path = Path(output_dir) / FINALIZATION_FILE_NAME
    _write_json_atomic(
        marker_path,
        {
            "schema": FINALIZATION_SCHEMA,
            "status": REOPENED_STATUS,
            "reviewer": reviewer.strip(),
            "prior_finalization_sha256": _sha256_file(marker_path),
        },
    )
    return marker_path


def candidate_label(candidate: FrameCandidate, position: int | None = None) -> str:
    """Describe a candidate as a local box, not a global animal identity."""

    prefix = "" if position is None else f"{position}. "
    pig = candidate.pig_id or "pig?"
    track = candidate.track_id or "track?"
    return f"{prefix}{pig} · {track}"


def resolve_video_path(
    candidates_by_frame: Mapping[int, Sequence[FrameCandidate]],
    explicit_video_path: Path | None,
) -> Path:
    """Resolve a video only when it is bound to declared source authority."""

    authorities = _declared_video_authorities(candidates_by_frame)
    if len(authorities) != 1:
        raise IdentityAdjudicationError(
            "identity_adjudication_requires_one_declared_source_video="
            f"found={len(authorities)}"
        )
    authority = authorities[0]
    if explicit_video_path is not None:
        explicit_video_path = Path(explicit_video_path)
        if not explicit_video_path.is_file():
            raise IdentityAdjudicationError(
                f"explicit_video_path_not_found={explicit_video_path}"
            )
        if not _explicit_path_matches_authority(
            explicit_video_path,
            authority,
        ):
            raise IdentityAdjudicationError(
                "explicit_video_path_not_bound_to_declared_source_video="
                f"declared={authority.name};provided={explicit_video_path.name}"
            )
        return explicit_video_path.resolve()
    if not authority.is_file():
        raise IdentityAdjudicationError(
            "declared_source_video_path_not_found="
            f"{authority}"
        )
    return authority.resolve()


def render_identity_frame(
    source: Image.Image,
    candidates: Sequence[FrameCandidate],
    cases: Sequence[IdentityCase],
    selected_by_case: Mapping[str, str],
    active_case_id: str,
    bbox_edits_by_case: Mapping[str, BoundingBoxEdit] | None = None,
) -> Image.Image:
    """Render neutral candidates plus explicit original/selected box roles."""

    image = source.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    for position, candidate in enumerate(candidates, start=1):
        color = BOX_COLORS[(position - 1) % len(BOX_COLORS)]
        width = 2
        active_original = any(
            case.review_unit_id == active_case_id
            and candidate.object_track_key == case.original_object_track_key
            for case in cases
        )
        selected_case_ids = [
            case_id
            for case_id, selected_key in selected_by_case.items()
            if selected_key == candidate.object_track_key
        ]
        if active_original:
            color = "#ffd966"
            width = 3
        if selected_case_ids:
            active_selected = active_case_id in selected_case_ids
            case_position = next(
                index
                for index, case in enumerate(cases)
                if case.review_unit_id in selected_case_ids
            )
            color = ACTIVE_CASE_COLORS[case_position % len(ACTIVE_CASE_COLORS)]
            width = 5 if active_selected else 4
        box = tuple(int(value) for value in candidate.bbox)
        draw.rectangle(box, outline=color, width=width)
        label = candidate_label(candidate, position)
        if active_original:
            label = f"O {label}"
        if selected_case_ids:
            label = f"S {label}"
        draw.rectangle(
            (box[0], max(0, box[1] - 18), min(image.width, box[0] + 175), box[1]),
            fill="#111111",
        )
        draw.text((box[0] + 2, max(0, box[1] - 17)), label, fill=color)
    for case_position, case in enumerate(cases):
        edit = (bbox_edits_by_case or {}).get(case.review_unit_id)
        if edit is None:
            continue
        color = ACTIVE_CASE_COLORS[case_position % len(ACTIVE_CASE_COLORS)]
        width = 6 if case.review_unit_id == active_case_id else 4
        box = tuple(int(round(value)) for value in edit.bbox)
        draw.rectangle(box, outline=color, width=width)
        prefix = "A" if edit.mode == ADDED_BBOX_MODE else "E"
        label_y = max(0, box[1] - 18)
        draw.rectangle(
            (box[0], label_y, min(image.width, box[0] + 185), box[1]),
            fill="#401040",
        )
        draw.text(
            (box[0] + 2, label_y + 1),
            f"{prefix} bbox sidecar",
            fill="#ffffff",
        )
    return image


def candidate_at_display_point(
    candidates: Sequence[FrameCandidate],
    x: float,
    y: float,
    scale: float,
    offset: tuple[int, int],
) -> FrameCandidate | None:
    """Return the smallest rendered source box containing a canvas click."""

    if scale <= 0.0:
        return None
    source_x = (x - offset[0]) / scale
    source_y = (y - offset[1]) / scale
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
        key=lambda candidate: (candidate.x2 - candidate.x1) * (candidate.y2 - candidate.y1),
    )


def canvas_drag_to_source_bbox(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    scale: float,
    offset: tuple[int, int],
    source_size: tuple[int, int],
    minimum_extent: float = 3.0,
) -> tuple[float, float, float, float] | None:
    """Normalize and clamp one canvas drag into source-image coordinates."""

    width, height = source_size
    if scale <= 0.0 or width <= 0 or height <= 0:
        return None

    def source_point(point: tuple[float, float]) -> tuple[float, float]:
        source_x = (point[0] - offset[0]) / scale
        source_y = (point[1] - offset[1]) / scale
        return (
            min(max(source_x, 0.0), float(width)),
            min(max(source_y, 0.0), float(height)),
        )

    start_x, start_y = source_point(start)
    end_x, end_y = source_point(end)
    x1, x2 = sorted((start_x, end_x))
    y1, y2 = sorted((start_y, end_y))
    if x2 - x1 < minimum_extent or y2 - y1 < minimum_extent:
        return None
    return x1, y1, x2, y2


BBOX_HANDLE_NAMES = ("nw", "n", "ne", "e", "se", "s", "sw", "w")
BBOX_HANDLE_HIT_RADIUS = 9.0
BBOX_MINIMUM_EXTENT = 3.0


def source_bbox_to_canvas(
    bbox: tuple[float, float, float, float],
    *,
    scale: float,
    offset: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Map source-image xyxy coordinates to the current canvas view."""

    x1, y1, x2, y2 = bbox
    return (
        offset[0] + x1 * scale,
        offset[1] + y1 * scale,
        offset[0] + x2 * scale,
        offset[1] + y2 * scale,
    )


def bbox_handle_points(
    canvas_bbox: tuple[float, float, float, float],
) -> dict[str, tuple[float, float]]:
    """Return CVAT-like corner and edge handle centers."""

    x1, y1, x2, y2 = canvas_bbox
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


def hit_test_bbox_handle(
    point: tuple[float, float],
    canvas_bbox: tuple[float, float, float, float],
    *,
    radius: float = BBOX_HANDLE_HIT_RADIUS,
) -> str | None:
    """Return the nearest resize handle within a fixed canvas-pixel radius."""

    matches = [
        (
            (point[0] - handle_x) ** 2 + (point[1] - handle_y) ** 2,
            name,
        )
        for name, (handle_x, handle_y) in bbox_handle_points(canvas_bbox).items()
        if abs(point[0] - handle_x) <= radius
        and abs(point[1] - handle_y) <= radius
    ]
    return min(matches)[1] if matches else None


def canvas_point_inside_bbox(
    point: tuple[float, float],
    canvas_bbox: tuple[float, float, float, float],
) -> bool:
    """Return whether a canvas point lies inside an xyxy rectangle."""

    x1, y1, x2, y2 = canvas_bbox
    return x1 <= point[0] <= x2 and y1 <= point[1] <= y2


def transform_source_bbox(
    bbox: tuple[float, float, float, float],
    *,
    delta: tuple[float, float],
    operation: str,
    source_size: tuple[int, int],
    minimum_extent: float = BBOX_MINIMUM_EXTENT,
) -> tuple[float, float, float, float]:
    """Move or resize a source bbox while keeping it inside the image."""

    width, height = source_size
    x1, y1, x2, y2 = bbox
    delta_x, delta_y = delta
    if operation == "move":
        bounded_x = min(max(delta_x, -x1), float(width) - x2)
        bounded_y = min(max(delta_y, -y1), float(height) - y2)
        return (
            x1 + bounded_x,
            y1 + bounded_y,
            x2 + bounded_x,
            y2 + bounded_y,
        )
    if operation not in BBOX_HANDLE_NAMES:
        raise ValueError(f"invalid_bbox_transform_operation={operation}")

    if "w" in operation:
        x1 = min(max(x1 + delta_x, 0.0), x2 - minimum_extent)
    if "e" in operation:
        x2 = max(min(x2 + delta_x, float(width)), x1 + minimum_extent)
    if "n" in operation:
        y1 = min(max(y1 + delta_y, 0.0), y2 - minimum_extent)
    if "s" in operation:
        y2 = max(min(y2 + delta_y, float(height)), y1 + minimum_extent)
    return x1, y1, x2, y2


class IdentityContinuityGui:
    """Select source boxes or draw sidecar-only bbox corrections per frame."""

    def __init__(self, config: IdentityGuiConfig) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python is required for identity adjudication")
        assert_safe_cli_input_paths(config)
        self.config = config
        self.output_dir = assert_safe_output_dir(config.output_dir)
        self.cases = load_identity_cases(
            config.review_units_csv,
            config.review_item_ids,
        )
        assert_single_scene(self.cases)
        self.candidates_by_frame = load_frame_candidates(
            config.frame_features_csv,
            self.cases,
        )
        self.video_path = resolve_video_path(
            self.candidates_by_frame,
            config.video_path,
        )
        (
            self.selections,
            self.exclusions,
            self.bbox_edits,
        ) = load_session_sidecars_with_bbox_edits(
            self.output_dir,
            self.cases,
            self.candidates_by_frame,
        )
        self.editable_pig_ids = tuple(
            dict.fromkeys(
                value.strip()
                for value in config.editable_pig_ids
                if value.strip()
            )
        )
        self.mini_cvat_enabled = bool(self.editable_pig_ids)
        if getattr(self, "mini_cvat_enabled", False):
            known_pig_ids = {
                candidate.pig_id
                for candidates in self.candidates_by_frame.values()
                for candidate in candidates
                if candidate.pig_id
            }
            unknown_ids = sorted(set(self.editable_pig_ids) - known_pig_ids)
            if unknown_ids:
                raise IdentityAdjudicationError(
                    "editable_pig_id_not_found=" + ",".join(unknown_ids)
                )
        self.all_frames = tuple(
            sorted({frame for case in self.cases for frame in case.frame_indices})
        )
        if getattr(self, "mini_cvat_enabled", False):
            self.mini_actor_attributes, self.mini_frame_annotations = (
                load_mini_cvat_sidecar(
                    self.output_dir,
                    source_type=self.cases[0].source_type,
                    dataset_id=self.cases[0].dataset_id,
                    video_key=self.cases[0].video_key,
                    editable_actor_ids=self.editable_pig_ids,
                    frame_indices=self.all_frames,
                )
            )
            self.active_pig_id = self.editable_pig_ids[0]
            self.mini_selected_keys = {
                key: annotation.original_object_track_key
                for key, annotation in self.mini_frame_annotations.items()
                if annotation.original_object_track_key
            }
        else:
            self.mini_actor_attributes = {}
            self.mini_frame_annotations = {}
            self.active_pig_id = ""
            self.mini_selected_keys = {}
        self.current_frame_position, self.active_case_position = self._resume_position()
        self.resumed = bool(
            self.selections
            or self.exclusions
            or self.mini_actor_attributes
            or self.mini_frame_annotations
        )
        self.frame_cache = RenderedFrameCache()
        self.capture = cv2.VideoCapture(str(self.video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot_open_full_scene_video={self.video_path}")
        self._verify_target_frame_decoding()
        self.finalization_marker = load_finalization_marker(self.output_dir)
        if config.reopen_finalized:
            reopen_finalization_marker(
                self.output_dir,
                reviewer=config.reviewer,
            )
            self.finalization_marker = load_finalization_marker(self.output_dir)
        self.finalized = bool(
            self.finalization_marker
            and self.finalization_marker.get("status") == FINALIZED_STATUS
        )
        self._display_scale = 1.0
        self._display_offset = (0, 0)
        self._display_image_size = (0, 0)
        self._source_image_size = (0, 0)
        self._bbox_draw_mode: str | None = None
        self._bbox_drag_start: tuple[float, float] | None = None
        self._bbox_drag_rectangle: int | None = None
        self._bbox_interaction: str | None = None
        self._bbox_resize_handle: str | None = None
        self._bbox_origin: tuple[float, float, float, float] | None = None
        self._bbox_preview: tuple[float, float, float, float] | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._prefetch_after_id: str | None = None

        self.root = tk.Tk()
        self.root.title(WINDOW_TITLE)
        self.root.minsize(1280, 800)
        self.status_var = tk.StringVar(
            value=(
                "Phiên đã hoàn tất; dùng --reopen-finalized để sửa."
                if self.finalized
                else (
                    "Đã khôi phục vị trí chưa hoàn tất."
                    if self.resumed
                    else "Sẵn sàng."
                )
            )
        )
        self.info_var = tk.StringVar(value="")
        self.bbox_detail_var = tk.StringVar(value="")
        self._build_layout()
        self.root.bind_all("<Key>", self._on_keypress)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.show_current_frame()

    @property
    def active_case(self) -> IdentityCase:
        return self.cases[self.active_case_position]

    @property
    def current_frame_index(self) -> int:
        return self.all_frames[self.current_frame_position]

    def _set_current_frame(self, frame_index: int) -> None:
        self.current_frame_position = self.all_frames.index(frame_index)

    def _ensure_active_case_frame(self) -> None:
        if getattr(self, "mini_cvat_enabled", False):
            return
        if self.current_frame_index in self.active_case.frame_indices:
            return
        self._set_current_frame(
            first_pending_case_frame(self.active_case, self.selections)
        )

    def _resume_position(self) -> tuple[int, int]:
        if getattr(self, "mini_cvat_enabled", False):
            for actor_id in self.editable_pig_ids:
                for frame_index in self.all_frames:
                    if (actor_id, frame_index) not in self.mini_frame_annotations:
                        self.active_pig_id = actor_id
                        return self.all_frames.index(frame_index), 0
            return self.all_frames.index(self.all_frames[0]), 0
        for case_position, case in enumerate(self.cases):
            if case.review_unit_id in self.exclusions:
                continue
            for frame_index in case.frame_indices:
                key = (case.review_unit_id, frame_index)
                if key not in self.selections:
                    return self.all_frames.index(frame_index), case_position
        first_frame = self.cases[0].frame_indices[0]
        return self.all_frames.index(first_frame), 0

    def _build_layout(self) -> None:
        root = self.root
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=0)
        root.rowconfigure(0, weight=1)

        media = ttk.Frame(root, padding=8)
        media.grid(row=0, column=0, sticky="nsew")
        media.columnconfigure(0, weight=1)
        media.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(media, background="#202020", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Motion>", self._on_canvas_hover)

        side_host = ttk.Frame(root, padding=(0, 8, 8, 8))
        side_host.grid(row=0, column=1, sticky="nsew")
        side_host.columnconfigure(0, weight=1)
        side_host.rowconfigure(0, weight=1)
        side_canvas = tk.Canvas(
            side_host,
            width=460,
            highlightthickness=0,
            background="#f5f5f5",
        )
        side_scrollbar = ttk.Scrollbar(
            side_host,
            orient="vertical",
            command=side_canvas.yview,
        )
        side_canvas.configure(yscrollcommand=side_scrollbar.set)
        side_canvas.grid(row=0, column=0, sticky="nsew")
        side_scrollbar.grid(row=0, column=1, sticky="ns")
        side = ttk.Frame(side_canvas, padding=8)
        side_window = side_canvas.create_window((0, 0), window=side, anchor="nw")

        def update_side_scrollregion(_event: tk.Event[Any] | None = None) -> None:
            side_canvas.configure(scrollregion=side_canvas.bbox("all"))

        def resize_side_window(event: tk.Event[Any]) -> None:
            side_canvas.itemconfigure(side_window, width=max(event.width, 440))

        def scroll_side(event: tk.Event[Any]) -> None:
            side_canvas.yview_scroll(-int(event.delta / 120), "units")

        side.bind("<Configure>", update_side_scrollregion)
        side_canvas.bind("<Configure>", resize_side_window)
        side.bind("<Enter>", lambda _event: side_canvas.bind_all("<MouseWheel>", scroll_side))
        side.bind("<Leave>", lambda _event: side_canvas.unbind_all("<MouseWheel>"))

        ttk.Label(
            side,
            text="Mini-CVAT cục bộ — bbox/ID/Hidden; behavior theo burst",
            wraplength=420,
            foreground="#8b0000",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(side, textvariable=self.info_var, wraplength=420).grid(
            row=1,
            column=0,
            sticky="ew",
        )
        self.case_frame = ttk.LabelFrame(side, text="Các unit cần rà soát", padding=6)
        self.case_frame.grid(row=2, column=0, sticky="ew", pady=(10, 6))
        self.candidate_frame = ttk.LabelFrame(side, text="Tất cả bbox trong frame", padding=6)
        self.candidate_frame.grid(row=3, column=0, sticky="ew", pady=6)
        bbox_detail = ttk.LabelFrame(side, text="BBox đang chỉnh", padding=6)
        bbox_detail.grid(row=4, column=0, sticky="ew", pady=6)
        ttk.Label(
            bbox_detail,
            textvariable=self.bbox_detail_var,
            wraplength=420,
            justify="left",
        ).grid(row=0, column=0, sticky="ew")
        bbox_detail.columnconfigure(0, weight=1)

        controls = ttk.LabelFrame(side, text="Điều hướng và bbox", padding=6)
        controls.grid(row=5, column=0, sticky="ew", pady=6)
        ttk.Button(controls, text="← Frame", command=lambda: self.step_frame(-1)).grid(
            row=0, column=0, sticky="ew", padx=2, pady=2
        )
        ttk.Button(controls, text="Frame →", command=lambda: self.step_frame(1)).grid(
            row=0, column=1, sticky="ew", padx=2, pady=2
        )
        ttk.Button(controls, text="Dùng bbox gốc", command=self.use_original_box).grid(
            row=1, column=0, sticky="ew", padx=2, pady=2
        )
        ttk.Button(controls, text="Bỏ chọn bbox", command=self.clear_current_selection).grid(
            row=1, column=1, sticky="ew", padx=2, pady=2
        )
        ttk.Button(controls, text="Chỉnh bbox (E)", command=self.start_corrected_bbox).grid(
            row=2, column=0, sticky="ew", padx=2, pady=2
        )
        ttk.Button(controls, text="Thêm bbox", command=self.start_added_bbox).grid(
            row=2, column=1, sticky="ew", padx=2, pady=2
        )
        ttk.Button(controls, text="Hủy thao tác (Esc)", command=self.cancel_bbox_drawing).grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=2, pady=2
        )
        ttk.Button(controls, text="Loại unit (X)", command=self.exclude_active_case).grid(
            row=4, column=0, sticky="ew", padx=2, pady=2
        )
        ttk.Button(controls, text="Khôi phục unit (R)", command=self.restore_active_case).grid(
            row=4, column=1, sticky="ew", padx=2, pady=2
        )
        ttk.Button(controls, text="Lưu sidecar (Ctrl+S)", command=self.save).grid(
            row=5, column=0, sticky="ew", padx=2, pady=2
        )
        ttk.Button(controls, text="Hoàn tất kiểm tra", command=self.finalize).grid(
            row=5, column=1, sticky="ew", padx=2, pady=2
        )
        ttk.Button(
            controls,
            text="Loại bỏ mọi thay đổi phiên",
            command=self.reset_all_session_changes,
        ).grid(row=6, column=0, columnspan=2, sticky="ew", padx=2, pady=(8, 2))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        if getattr(self, "mini_cvat_enabled", False):
            self.mini_actor_frame = ttk.LabelFrame(
                side,
            text=(
                "Actor source scope — behavior áp dụng cả burst; "
                "ID có thể override riêng frame"
            ),
                padding=6,
            )
            self.mini_actor_frame.grid(row=6, column=0, sticky="ew", pady=6)
            self.mini_actor_id_var = tk.StringVar(value=self.active_pig_id)
            self.mini_reviewed_id_var = tk.StringVar(value="")
            self.mini_behavior_var = tk.StringVar(value="")
            self.mini_hidden_var = tk.StringVar(value="")
            self.mini_progress_var = tk.StringVar(value="")
            self.mini_actor_buttons = ttk.Frame(self.mini_actor_frame)
            self.mini_actor_buttons.grid(row=0, column=0, sticky="ew")
            ttk.Label(
                self.mini_actor_frame,
                textvariable=self.mini_actor_id_var,
                foreground="#404040",
                wraplength=420,
            ).grid(row=1, column=0, sticky="ew", pady=(6, 0))
            ttk.Label(
                self.mini_actor_frame,
                text="Reviewed ID đích (source scope giữ nguyên)",
            ).grid(
                row=2,
                column=0,
                sticky="w",
                pady=(6, 0),
            )
            ttk.Label(
                self.mini_actor_frame,
                text="Có thể nhập 5; GUI sẽ lưu thành ID_5.",
                foreground="#404040",
                wraplength=420,
            ).grid(row=3, column=0, sticky="ew")
            self.mini_reviewed_id_entry = ttk.Entry(
                self.mini_actor_frame,
                textvariable=self.mini_reviewed_id_var,
                width=22,
            )
            self.mini_reviewed_id_entry.grid(row=4, column=0, sticky="ew")
            ttk.Label(self.mini_actor_frame, text="Behavior (cả burst)").grid(
                row=5, column=0, sticky="w", pady=(6, 0)
            )
            self.mini_behavior_combo = ttk.Combobox(
                self.mini_actor_frame,
                textvariable=self.mini_behavior_var,
                values=sorted(CANONICAL_BEHAVIORS),
                state="readonly",
                width=22,
            )
            self.mini_behavior_combo.grid(row=6, column=0, sticky="ew")
            ttk.Label(self.mini_actor_frame, text="Hidden (object/frame hiện tại)").grid(
                row=7, column=0, sticky="w", pady=(6, 0)
            )
            self.mini_hidden_combo = ttk.Combobox(
                self.mini_actor_frame,
                textvariable=self.mini_hidden_var,
                values=("", *sorted(HIDDEN_VALUES - {""})),
                state="readonly",
                width=22,
            )
            self.mini_hidden_combo.grid(row=8, column=0, sticky="ew")
            ttk.Button(
                self.mini_actor_frame,
                text="Áp dụng reviewed ID + behavior cho cả burst",
                command=self.apply_mini_actor_attributes,
            ).grid(row=9, column=0, sticky="ew", pady=(8, 2))
            ttk.Button(
                self.mini_actor_frame,
                text="Hủy đổi ID (khôi phục đã lưu)",
                command=self.reset_mini_actor_identity_fields,
            ).grid(row=10, column=0, sticky="ew", pady=2)
            ttk.Button(
                self.mini_actor_frame,
                text="Lưu bbox/Hidden hiện tại",
                command=self.save_mini_current_frame,
            ).grid(row=11, column=0, sticky="ew", pady=2)
            ttk.Label(
                self.mini_actor_frame,
                textvariable=self.mini_progress_var,
                wraplength=420,
                foreground="#404040",
            ).grid(row=12, column=0, sticky="ew", pady=(6, 0))
            self.mini_actor_frame.columnconfigure(0, weight=1)

        help_row = 7 if getattr(self, "mini_cvat_enabled", False) else 6
        ttk.Label(
            side,
            text=(
                "Phím: ←/→ frame · Tab đổi unit · 1–9 chọn bbox · "
                "E chỉnh · Esc hủy · O bbox gốc · U bỏ chọn · Ctrl+S lưu"
            ),
            wraplength=420,
            foreground="#404040",
        ).grid(row=help_row, column=0, sticky="ew", pady=(8, 3))
        status_row = help_row + 1
        ttk.Label(side, textvariable=self.status_var, wraplength=420).grid(
            row=status_row,
            column=0,
            sticky="ew",
        )
        side.columnconfigure(0, weight=1)

    def _source_frame_index(self, review_frame_index: int) -> int:
        return source_frame_index_for_review_frame(
            self.candidates_by_frame,
            review_frame_index,
        )

    def _verify_target_frame_decoding(self) -> None:
        """Fail closed before UI creation when a box/video frame cannot align."""

        for review_frame_index in self.all_frames:
            source_frame_index = self._source_frame_index(review_frame_index)
            try:
                frame = decode_exact_source_frame(
                    self.capture,
                    source_frame_index,
                )
                height, width = frame.shape[:2]
                assert_candidate_bounds(
                    self.candidates_by_frame[review_frame_index],
                    review_frame_index=review_frame_index,
                    image_width=width,
                    image_height=height,
                )
            except RuntimeError:
                self.capture.release()
                raise

    def _decode_frame(self, review_frame_index: int) -> Image.Image:
        source_frame_index = self._source_frame_index(review_frame_index)
        cached = self.frame_cache.get(source_frame_index)
        if cached is not None:
            return cached
        frame = decode_exact_source_frame(
            self.capture,
            source_frame_index,
        )
        height, width = frame.shape[:2]
        assert_candidate_bounds(
            self.candidates_by_frame[review_frame_index],
            review_frame_index=review_frame_index,
            image_width=width,
            image_height=height,
        )
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
        self.frame_cache.put(source_frame_index, image)
        return image

    def _prefetch_adjacent_frame(self) -> None:
        self._prefetch_after_id = None
        next_position = self.current_frame_position + 1
        if next_position >= len(self.all_frames):
            return
        review_frame_index = self.all_frames[next_position]
        source_frame_index = self._source_frame_index(review_frame_index)
        if self.frame_cache.get(source_frame_index) is not None:
            return
        try:
            self._decode_frame(review_frame_index)
        except RuntimeError:
            return

    def _schedule_adjacent_prefetch(self) -> None:
        """Yield to Tk paint before bounded adjacent-frame decoding."""

        if self._prefetch_after_id is not None:
            self.root.after_cancel(self._prefetch_after_id)
        self._prefetch_after_id = self.root.after_idle(
            self._prefetch_adjacent_frame
        )

    def _selected_by_case(self, frame_index: int) -> dict[str, str]:
        return {
            case.review_unit_id: self.selections[(case.review_unit_id, frame_index)]
            for case in self.cases
            if (case.review_unit_id, frame_index) in self.selections
        }

    def _bbox_edits_by_case(
        self,
        frame_index: int,
    ) -> dict[str, BoundingBoxEdit]:
        return {
            case.review_unit_id: self.bbox_edits[(case.review_unit_id, frame_index)]
            for case in self.cases
            if (case.review_unit_id, frame_index) in self.bbox_edits
        }

    def _active_frame_key(self) -> tuple[str, int]:
        return self.active_case.review_unit_id, self.current_frame_index

    def _mini_frame_key(self) -> tuple[str, int]:
        return self.active_pig_id, self.current_frame_index

    def _mini_current_candidate(self) -> FrameCandidate | None:
        if not getattr(self, "mini_cvat_enabled", False) or not self.active_pig_id:
            return None
        key = getattr(self, "mini_selected_keys", {}).get(self._mini_frame_key(), "")
        candidates = self.candidates_by_frame[self.current_frame_index]
        if key:
            return next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.object_track_key == key
                ),
                None,
            )
        matches = [
            candidate
            for candidate in candidates
            if candidate.pig_id == self.active_pig_id
        ]
        return matches[0] if len(matches) == 1 else None

    def _mini_candidate_for_actor(
        self,
        actor_scope_id: str,
        frame_index: int,
    ) -> FrameCandidate | None:
        selected_key = getattr(self, "mini_selected_keys", {}).get(
            (actor_scope_id, frame_index),
            "",
        )
        candidates = self.candidates_by_frame[frame_index]
        if selected_key:
            return next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.object_track_key == selected_key
                ),
                None,
            )
        matches = [
            candidate
            for candidate in candidates
            if candidate.pig_id == actor_scope_id
        ]
        return matches[0] if len(matches) == 1 else None

    def _mini_original_behavior(self) -> str:
        candidate = self._mini_current_candidate()
        if candidate is None:
            return ""
        return (
            candidate.behavior
            if candidate.behavior in CANONICAL_BEHAVIORS
            else ""
        )

    def _mini_current_annotation(self) -> MiniCvatFrameAnnotation | None:
        if not getattr(self, "mini_cvat_enabled", False):
            return None
        return self.mini_frame_annotations.get(self._mini_frame_key())

    def _mini_effective_bbox(
        self,
    ) -> tuple[float, float, float, float] | None:
        annotation = self._mini_current_annotation()
        if annotation is not None:
            return annotation.bbox
        candidate = self._mini_current_candidate()
        return None if candidate is None else candidate.bbox

    def _mini_bbox_authority(self) -> str:
        annotation = self._mini_current_annotation()
        if annotation is None:
            return "CHƯA LƯU"
        if annotation.bbox_mode == ADDED_BBOX_MODE:
            return "ADDED — bbox thêm trong sidecar"
        if annotation.bbox_mode == CORRECTED_BBOX_MODE:
            return "CORRECTED — bbox nguồn đã chỉnh"
        return "SOURCE_BBOX — đã lưu"

    def _refresh_mini_actor_controls(self) -> None:
        if not getattr(self, "mini_cvat_enabled", False) or not hasattr(
            self, "mini_actor_frame"
        ):
            return
        for child in self.mini_actor_buttons.winfo_children():
            child.destroy()
        for position, actor_id in enumerate(self.editable_pig_ids):
            attributes = self.mini_actor_attributes.get(actor_id)
            display_id = (
                attributes.reviewed_pig_id if attributes is not None else actor_id
            )
            if attributes is None:
                text = f"nguồn {actor_id} (chưa đổi ID)"
            else:
                text = f"{display_id} ← nguồn {actor_id}"
            if actor_id == self.active_pig_id:
                text = f"✓ {text}"
            ttk.Button(
                self.mini_actor_buttons,
                text=text,
                command=lambda value=display_id: self.select_mini_display_actor(
                    value
                ),
            ).grid(row=0, column=position, sticky="ew", padx=1, pady=1)
            self.mini_actor_buttons.columnconfigure(position, weight=1)
        attributes = self.mini_actor_attributes.get(self.active_pig_id)
        candidate = self._mini_current_candidate()
        original_behavior = (
            candidate.behavior
            if candidate is not None and candidate.behavior in CANONICAL_BEHAVIORS
            else ""
        )
        self.mini_actor_id_var.set(
            f"Đang sửa: reviewed {self._mini_display_id(self.active_pig_id)} "
            f"← source scope {self.active_pig_id}"
        )
        self.mini_reviewed_id_var.set(
            self._mini_display_id(self.active_pig_id)
        )
        self.mini_behavior_var.set(
            attributes.reviewed_behavior if attributes else original_behavior
        )
        annotation = self._mini_current_annotation()
        default_hidden = candidate.hidden if candidate is not None else ""
        self.mini_hidden_var.set(
            annotation.reviewed_hidden if annotation else default_hidden
        )
        saved_count = sum(
            actor_id == self.active_pig_id
            for actor_id, _frame_index in self.mini_frame_annotations
        )
        total_count = len(self.all_frames)
        frame_state = "ĐÃ LƯU" if annotation else "CHƯA LƯU"
        self.mini_progress_var.set(
            f"Frame {self.current_frame_index}: {frame_state} · "
            f"burst {saved_count}/{total_count} frame"
        )

    def _mini_display_id(self, actor_scope_id: str) -> str:
        frame_position = getattr(self, "current_frame_position", None)
        all_frames = getattr(self, "all_frames", ())
        if frame_position is not None and all_frames:
            frame_index = all_frames[frame_position]
            annotation = self.mini_frame_annotations.get(
                (actor_scope_id, frame_index)
            )
            if annotation is not None and annotation.reviewed_pig_id:
                return annotation.reviewed_pig_id
        attributes = self.mini_actor_attributes.get(actor_scope_id)
        if attributes is None or not attributes.reviewed_pig_id:
            return actor_scope_id
        return attributes.reviewed_pig_id

    def _normalize_mini_pig_id(self, pig_id: str) -> str:
        value = pig_id.strip()
        if not value:
            return ""
        prefix, separator, suffix = value.partition("_")
        if separator and prefix.upper() == "ID" and suffix.strip():
            return f"ID_{suffix.strip()}"
        if value.isdigit():
            return f"ID_{value}"
        return value

    def select_mini_actor(self, actor_id: str) -> None:
        if actor_id not in self.editable_pig_ids:
            return
        self.cancel_bbox_drawing(silent=True)
        self.active_pig_id = actor_id
        self.status_var.set(
            f"Đang sửa source scope {actor_id} "
            f"→ reviewed {self._mini_display_id(actor_id)}"
        )
        self.show_current_frame()

    def select_mini_display_actor(self, display_id: str) -> None:
        """Select the source scope currently owning a reviewed ID."""

        display_id = self._normalize_mini_pig_id(display_id)
        matches = [
            actor_id
            for actor_id in self.editable_pig_ids
            if self._mini_display_id(actor_id) == display_id
        ]
        if len(matches) == 1:
            self.select_mini_actor(matches[0])
            return
        if display_id in self.editable_pig_ids:
            self.select_mini_actor(display_id)
            return
        self.status_var.set(
            f"Reviewed ID {display_id} chưa có một actor scope duy nhất."
        )

    def reset_mini_actor_identity_fields(self) -> None:
        if not self._ensure_mutable():
            return
        attributes = self.mini_actor_attributes.get(self.active_pig_id)
        reviewed_pig_id = (
            attributes.reviewed_pig_id if attributes is not None else self.active_pig_id
        )
        reviewed_behavior = (
            attributes.reviewed_behavior
            if attributes is not None
            else self._mini_original_behavior()
        )
        self.mini_reviewed_id_var.set(reviewed_pig_id)
        if reviewed_behavior in CANONICAL_BEHAVIORS:
            self.mini_behavior_var.set(reviewed_behavior)
        self.status_var.set(
            f"Đã khôi phục reviewed ID đã lưu cho source scope {self.active_pig_id}."
        )
        self.show_current_frame()

    def _mini_source_annotation(
        self,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        bbox_mode: str = "SOURCE_BBOX",
        reviewed_pig_id: str | None = None,
    ) -> MiniCvatFrameAnnotation | None:
        candidate = self._mini_current_candidate()
        if candidate is None and bbox is None:
            return None
        annotation_bbox = bbox if bbox is not None else candidate.bbox
        if candidate is None:
            source_frame_index = self._source_frame_index(self.current_frame_index)
            original_key = ""
            original_track_id = ""
            original_pig_id = self.active_pig_id
            original_hidden = ""
        else:
            source_frame_index = candidate.source_frame_index
            original_key = candidate.object_track_key
            original_track_id = candidate.track_id
            original_pig_id = candidate.pig_id
            original_hidden = candidate.hidden
        effective_reviewed_id = (
            reviewed_pig_id or self._mini_display_id(self.active_pig_id)
        )
        reviewed_hidden = self.mini_hidden_var.get().strip()
        return MiniCvatFrameAnnotation(
            actor_scope_id=self.active_pig_id,
            frame_index=self.current_frame_index,
            source_frame_index=source_frame_index,
            original_object_track_key=original_key,
            original_track_id=original_track_id,
            original_pig_id=original_pig_id,
            reviewed_pig_id=effective_reviewed_id,
            bbox_mode=bbox_mode,
            x1=annotation_bbox[0],
            y1=annotation_bbox[1],
            x2=annotation_bbox[2],
            y2=annotation_bbox[3],
            original_hidden=original_hidden,
            reviewed_hidden=reviewed_hidden,
        )

    def _mini_source_behavior_for_actor(self, actor_scope_id: str) -> str:
        for frame_index in self.all_frames:
            candidate = next(
                (
                    value
                    for value in self.candidates_by_frame[frame_index]
                    if value.pig_id == actor_scope_id
                    and value.behavior in CANONICAL_BEHAVIORS
                ),
                None,
            )
            if candidate is not None:
                return candidate.behavior
        return ""

    def _mini_attributes_for_actor(
        self,
        actor_scope_id: str,
        reviewed_pig_id: str,
    ) -> MiniCvatActorAttributes | None:
        existing = self.mini_actor_attributes.get(actor_scope_id)
        original_behavior = (
            existing.original_behavior
            if existing is not None
            else self._mini_source_behavior_for_actor(actor_scope_id)
        )
        reviewed_behavior = (
            existing.reviewed_behavior
            if existing is not None
            else original_behavior
        )
        if original_behavior not in CANONICAL_BEHAVIORS:
            return None
        return MiniCvatActorAttributes(
            actor_scope_id=actor_scope_id,
            original_pig_id=actor_scope_id,
            reviewed_pig_id=reviewed_pig_id,
            original_behavior=original_behavior,
            reviewed_behavior=reviewed_behavior,
        )

    def apply_mini_actor_attributes(self, *, refresh: bool = True) -> bool:
        if not self._ensure_mutable():
            return
        reviewed_pig_id = self._normalize_mini_pig_id(
            self.mini_reviewed_id_var.get()
        )
        self.mini_reviewed_id_var.set(reviewed_pig_id)
        reviewed_behavior = self.mini_behavior_var.get().strip()
        candidate = self._mini_current_candidate()
        original_behavior = candidate.behavior if candidate is not None else ""
        if not reviewed_pig_id or reviewed_behavior not in CANONICAL_BEHAVIORS:
            messagebox.showwarning(
                "Thiếu thuộc tính burst",
                "Chọn behavior chuẩn và nhập Reviewed ID trước khi lưu.",
                parent=self.root,
            )
            return
        if original_behavior not in CANONICAL_BEHAVIORS:
            original_behavior = self._mini_source_behavior_for_actor(
                self.active_pig_id
            )
        if original_behavior not in CANONICAL_BEHAVIORS:
            messagebox.showwarning(
                "Không xác định được behavior nguồn",
                "Không thể tạo sidecar burst khi behavior nguồn không chuẩn.",
                parent=self.root,
            )
            return
        prior_attributes = dict(self.mini_actor_attributes)
        self.mini_actor_attributes[self.active_pig_id] = (
            MiniCvatActorAttributes(
                actor_scope_id=self.active_pig_id,
                original_pig_id=self.active_pig_id,
                reviewed_pig_id=reviewed_pig_id,
                original_behavior=original_behavior,
                reviewed_behavior=reviewed_behavior,
            )
        )
        prior_owner = [
            actor_id
            for actor_id in self.editable_pig_ids
            if actor_id != self.active_pig_id
            and self._mini_display_id(actor_id) == reviewed_pig_id
        ]
        swapped_actor_id = ""
        if len(prior_owner) == 1:
            swapped_actor_id = prior_owner[0]
            swapped_attributes = self._mini_attributes_for_actor(
                swapped_actor_id,
                self.active_pig_id,
            )
            if swapped_attributes is None:
                self.mini_actor_attributes = prior_attributes
                messagebox.showwarning(
                    "Không thể hoán đổi ID",
                    "Actor còn lại không có behavior nguồn hợp lệ để tạo sidecar.",
                    parent=self.root,
                )
                return
            self.mini_actor_attributes[swapped_actor_id] = swapped_attributes
        elif len(prior_owner) > 1:
            self.mini_actor_attributes = prior_attributes
            messagebox.showwarning(
                "ID đích đang mơ hồ",
                "Có nhiều actor scope đang dùng Reviewed ID này; hãy sửa chúng trước.",
                parent=self.root,
            )
            return
        if not self.save(silent=False):
            self.mini_actor_attributes = prior_attributes
            self.status_var.set(
                "Không lưu được reviewed ID/behavior; đã khôi phục trạng thái trước."
            )
            return
        if swapped_actor_id:
            self.status_var.set(
                f"Đã hoán đổi reviewed ID giữa source scope {self.active_pig_id} "
                f"và {swapped_actor_id}; mọi bbox trong burst dùng mapping mới."
            )
        else:
            self.status_var.set(
                f"Đã lưu reviewed ID {reviewed_pig_id} cho source scope "
                f"{self.active_pig_id}; behavior áp dụng cho toàn burst."
            )
        if refresh:
            self.show_current_frame()
        return True

    def _mini_source_annotation_for_actor(
        self,
        actor_scope_id: str,
        frame_index: int,
        reviewed_pig_id: str,
    ) -> MiniCvatFrameAnnotation | None:
        candidate = self._mini_candidate_for_actor(actor_scope_id, frame_index)
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
            bbox_mode="SOURCE_BBOX",
            x1=candidate.x1,
            y1=candidate.y1,
            x2=candidate.x2,
            y2=candidate.y2,
            original_hidden=candidate.hidden,
            reviewed_hidden=candidate.hidden,
        )

    def save_mini_current_frame(self) -> None:
        if not self._ensure_mutable():
            return
        reviewed_pig_id = self._normalize_mini_pig_id(
            self.mini_reviewed_id_var.get()
        )
        if not reviewed_pig_id:
            messagebox.showwarning(
                "Thiếu Reviewed ID",
                "Nhập ID hợp lệ cho object/frame hiện tại.",
                parent=self.root,
            )
            return
        existing = self._mini_current_annotation()
        if existing is None:
            annotation = self._mini_source_annotation(
                reviewed_pig_id=reviewed_pig_id,
            )
        else:
            annotation = replace(
                existing,
                reviewed_pig_id=reviewed_pig_id,
                reviewed_hidden=self.mini_hidden_var.get().strip(),
            )
        if annotation is None:
            messagebox.showwarning(
                "Chưa có bbox",
                "Chọn hoặc thêm bbox trước khi lưu frame/object.",
                parent=self.root,
            )
            return
        if annotation.reviewed_hidden not in HIDDEN_VALUES:
            messagebox.showwarning(
                "Thiếu Hidden",
                "Chọn Yes, No hoặc Unclear cho object/frame hiện tại.",
                parent=self.root,
            )
            return
        key = self._mini_frame_key()
        prior_frames = dict(self.mini_frame_annotations)
        previous_id = self._mini_display_id(self.active_pig_id)
        self.mini_frame_annotations[key] = annotation
        owner = next(
            (
                actor_id
                for actor_id in self.editable_pig_ids
                if actor_id != self.active_pig_id
                and self._mini_display_id(actor_id) == reviewed_pig_id
            ),
            None,
        )
        if owner is not None and reviewed_pig_id != previous_id:
            owner_key = (owner, self.current_frame_index)
            owner_annotation = self.mini_frame_annotations.get(owner_key)
            if owner_annotation is None:
                owner_annotation = self._mini_source_annotation_for_actor(
                    owner,
                    self.current_frame_index,
                    previous_id,
                )
            if owner_annotation is None:
                self.mini_frame_annotations = prior_frames
                messagebox.showwarning(
                    "Không thể đổi ID trong frame",
                    "Không tìm thấy bbox của ID đang bị đổi chỗ ở frame này.",
                    parent=self.root,
                )
                return
            self.mini_frame_annotations[owner_key] = owner_annotation
        if not self.save(silent=False):
            self.mini_frame_annotations = prior_frames
            self.status_var.set(
                "Không lưu được object/frame; đã khôi phục trạng thái trước."
            )
            return
        self.status_var.set(
            f"Đã lưu ID {reviewed_pig_id}, bbox và Hidden của frame "
            f"{self.current_frame_index}."
        )
        self.show_current_frame()

    def _active_selected_candidate(self) -> FrameCandidate | None:
        if getattr(self, "mini_cvat_enabled", False):
            return self._mini_current_candidate()
        selected_key = self.selections.get(self._active_frame_key(), "")
        if not selected_key or selected_key == MANUAL_BBOX_SELECTION_KEY:
            return None
        return next(
            (
                candidate
                for candidate in self.candidates_by_frame[self.current_frame_index]
                if candidate.object_track_key == selected_key
            ),
            None,
        )

    def _active_effective_bbox(
        self,
    ) -> tuple[float, float, float, float] | None:
        if getattr(self, "mini_cvat_enabled", False):
            return self._mini_effective_bbox()
        edit = self.bbox_edits.get(self._active_frame_key())
        if edit is not None:
            return edit.bbox
        candidate = self._active_selected_candidate()
        return None if candidate is None else candidate.bbox

    def _active_bbox_authority(self) -> str:
        if getattr(self, "mini_cvat_enabled", False):
            return self._mini_bbox_authority()
        edit = self.bbox_edits.get(self._active_frame_key())
        if edit is not None:
            return edit.mode
        if self._active_selected_candidate() is not None:
            return "SOURCE_BBOX"
        return "CHƯA CHỌN"

    def _refresh_bbox_detail(self) -> None:
        if not hasattr(self, "bbox_detail_var"):
            return
        selected = self._active_selected_candidate()
        bbox = self._active_effective_bbox()
        if selected is not None:
            pig_id = selected.pig_id or "?"
            track_id = selected.track_id or "?"
            object_key = selected.object_track_key
        elif getattr(self, "mini_cvat_enabled", False):
            pig_id = self.active_pig_id or "?"
            track_id = "manual"
            object_key = "sidecar bbox added"
        else:
            pig_id = self.active_case.original_pig_id or "?"
            track_id = self.active_case.original_track_id or "?"
            object_key = "chưa chọn"
        lines = [
            f"Trạng thái: {self._active_bbox_authority()}",
            (
                f"Actor scope: {self.active_pig_id}"
                if getattr(self, "mini_cvat_enabled", False)
                else ""
            ),
            (
                f"Reviewed ID: {self._mini_display_id(self.active_pig_id)}"
                if getattr(self, "mini_cvat_enabled", False)
                else ""
            ),
            f"Pig ID: {pig_id} | Track ID: {track_id}",
            f"Object key: {object_key}",
        ]
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            lines.extend(
                [
                    f"xyxy: {x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}",
                    f"Kích thước: {x2 - x1:.1f} × {y2 - y1:.1f} px",
                    "Kéo bên trong bbox để di chuyển; kéo ô vuông để resize.",
                ]
            )
        else:
            lines.append("Chọn bbox nguồn hoặc bấm “Thêm bbox”.")
        self.bbox_detail_var.set("\n".join(line for line in lines if line))

    def _case_progress(self, case: IdentityCase) -> tuple[int, int, str]:
        mapped = sum(
            (case.review_unit_id, frame_index) in self.selections
            for frame_index in case.frame_indices
        )
        return mapped, len(case.frame_indices), case_status(case, self.selections, self.exclusions)

    def _refresh_case_controls(self) -> None:
        for child in self.case_frame.winfo_children():
            child.destroy()
        if getattr(self, "mini_cvat_enabled", False):
            ttk.Label(
                self.case_frame,
                text=(
                "Mini-CVAT đang chỉnh theo actor scope. "
                "Behavior là một nhãn cho toàn burst; ID và Hidden có thể "
                "lưu riêng frame/object."
                ),
                wraplength=420,
                justify="left",
            ).grid(row=0, column=0, sticky="ew")
            self.case_frame.columnconfigure(0, weight=1)
            return
        for index, case in enumerate(self.cases):
            mapped, total, status = self._case_progress(case)
            text = (
                f"{case.review_item_id}: cục bộ {case.original_pig_id or '?'} / "
                f"{case.original_track_id or '?'} — {mapped}/{total} {status}"
            )
            style = "Accent.TButton" if index == self.active_case_position else "TButton"
            ttk.Button(
                self.case_frame,
                text=text,
                style=style,
                command=lambda target=index: self.set_active_case(target),
            ).grid(row=index, column=0, sticky="ew", pady=2)
        self.case_frame.columnconfigure(0, weight=1)

    def _refresh_candidate_controls(self) -> None:
        for child in self.candidate_frame.winfo_children():
            child.destroy()
        if getattr(self, "mini_cvat_enabled", False):
            selected = self._mini_current_candidate()
            annotation = self._mini_current_annotation()
            if annotation is None:
                summary = "Chọn một bbox rồi lưu object/frame; các ID ngoài scope chỉ xem."
            else:
                summary = (
                    "✓ Object/frame đã lưu — kéo trực tiếp để di chuyển/resize."
                )
            ttk.Label(
                self.candidate_frame,
                text=summary,
                foreground="#7a1f7a" if annotation is not None else "#303030",
                wraplength=420,
            ).grid(row=0, column=0, sticky="ew", pady=(0, 4))
            candidates = self.candidates_by_frame[self.current_frame_index]
            for index, candidate in enumerate(candidates, start=1):
                text = candidate_label(candidate, index)
                if candidate.pig_id in self.editable_pig_ids:
                    mapped_id = self._mini_display_id(candidate.pig_id)
                    text = f"{text} · scope {candidate.pig_id} → {mapped_id}"
                if candidate is selected:
                    text = f"✓ {text}"
                if candidate.pig_id not in self.editable_pig_ids:
                    text = f"{text} (chỉ xem)"
                ttk.Button(
                    self.candidate_frame,
                    text=text,
                    command=lambda chosen=candidate: self.select_candidate(chosen),
                ).grid(row=index, column=0, sticky="ew", pady=1)
            self.candidate_frame.columnconfigure(0, weight=1)
            return
        frame_key = self._active_frame_key()
        selected_key = self.selections.get(frame_key, "")
        edit = self.bbox_edits.get(frame_key)
        if selected_key == MANUAL_BBOX_SELECTION_KEY:
            summary = "✓ Bbox sidecar được thêm — kéo trực tiếp để chỉnh"
        elif edit is not None:
            summary = "✓ Bbox nguồn đã chỉnh — kéo trực tiếp để chỉnh tiếp"
        elif selected_key:
            summary = "✓ Bbox nguồn — kéo trực tiếp để di chuyển/resize"
        else:
            summary = "Chọn một bbox nguồn hoặc thêm bbox bị mất"
        ttk.Label(
            self.candidate_frame,
            text=summary,
            foreground="#7a1f7a" if edit is not None else "#303030",
            wraplength=390,
        ).grid(row=0, column=0, sticky="ew", pady=(0, 4))
        candidates = self.candidates_by_frame[self.current_frame_index]
        for index, candidate in enumerate(candidates, start=1):
            selected = selected_key == candidate.object_track_key
            text = candidate_label(candidate, index)
            if selected:
                text = f"✓ {text}"
            ttk.Button(
                self.candidate_frame,
                text=text,
                command=lambda chosen=candidate: self.select_candidate(chosen),
            ).grid(row=index, column=0, sticky="ew", pady=1)
        self.candidate_frame.columnconfigure(0, weight=1)

    def _fit_to_canvas(self, image: Image.Image) -> Image.Image:
        self.canvas.update_idletasks()
        max_width = max(600, self.canvas.winfo_width())
        max_height = max(500, self.canvas.winfo_height())
        scale = min(max_width / image.width, max_height / image.height, 1.0)
        width = max(1, int(round(image.width * scale)))
        height = max(1, int(round(image.height * scale)))
        self._display_scale = scale
        self._display_image_size = (width, height)
        self._display_offset = ((max_width - width) // 2, (max_height - height) // 2)
        if (width, height) == image.size:
            return image
        return image.resize((width, height), Image.Resampling.LANCZOS)

    def _draw_bbox_editor_overlay(
        self,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> None:
        self.canvas.delete("bbox-editor")
        bbox = self._active_effective_bbox() if bbox is None else bbox
        if bbox is None:
            return
        canvas_bbox = source_bbox_to_canvas(
            bbox,
            scale=self._display_scale,
            offset=self._display_offset,
        )
        authority = self._active_bbox_authority()
        color = "#ff4dff" if authority.startswith("ADDED") else "#00e5ff"
        if authority.startswith("CORRECTED"):
            color = "#ff9f1c"
        self.canvas.create_rectangle(
            *canvas_bbox,
            outline=color,
            width=3,
            tags=("bbox-editor",),
        )
        handle_half = 5
        for handle_x, handle_y in bbox_handle_points(canvas_bbox).values():
            self.canvas.create_rectangle(
                handle_x - handle_half,
                handle_y - handle_half,
                handle_x + handle_half,
                handle_y + handle_half,
                fill="#ffffff",
                outline=color,
                width=2,
                tags=("bbox-editor",),
            )
        label = authority.split(" — ", maxsplit=1)[0]
        if getattr(self, "mini_cvat_enabled", False) and self.active_pig_id:
            label = (
                f"{self._mini_display_id(self.active_pig_id)} "
                f"← nguồn {self.active_pig_id} · {label}"
            )
        self.canvas.create_text(
            canvas_bbox[0] + 4,
            max(12.0, canvas_bbox[1] - 10),
            text=f"{label} · kéo để chỉnh",
            fill="#ffffff",
            anchor="sw",
            tags=("bbox-editor",),
        )

    def _draw_candidate_overlays(
        self,
        candidates: Sequence[FrameCandidate],
        cases: Sequence[IdentityCase],
        selected_by_case: Mapping[str, str],
        active_case_id: str,
    ) -> None:
        """Draw source candidates as independent canvas layers."""
        selected_keys = set(selected_by_case.values())
        selected_case_ids = {
            case_id for case_id, selected_key in selected_by_case.items() if selected_key
        }
        for position, candidate in enumerate(candidates, start=1):
            if getattr(self, "mini_cvat_enabled", False):
                saved = self.mini_frame_annotations.get(
                    (candidate.pig_id, self.current_frame_index)
                )
                if (
                    saved is not None
                    and saved.original_object_track_key
                    == candidate.object_track_key
                ):
                    continue
            canvas_bbox = source_bbox_to_canvas(
                candidate.bbox,
                scale=self._display_scale,
                offset=self._display_offset,
            )
            color = BOX_COLORS[(position - 1) % len(BOX_COLORS)]
            width = 2
            active_original = any(
                case.review_unit_id == active_case_id
                and candidate.object_track_key == case.original_object_track_key
                for case in cases
            )
            if active_original:
                color = "#ffd966"
                width = 3
            if candidate.object_track_key in selected_keys:
                case_position = next(
                    (
                        index
                        for index, case in enumerate(cases)
                        if case.review_unit_id in selected_case_ids
                    ),
                    0,
                )
                color = ACTIVE_CASE_COLORS[
                    case_position % len(ACTIVE_CASE_COLORS)
                ]
                width = 5 if active_case_id in selected_case_ids else 4
            self.canvas.create_rectangle(
                *canvas_bbox,
                outline=color,
                width=width,
                tags=("candidate-overlay",),
            )
            label = candidate_label(candidate, position)
            if active_original:
                label = f"O {label}"
            if candidate.object_track_key in selected_keys:
                label = f"S {label}"
            self.canvas.create_text(
                canvas_bbox[0] + 4,
                max(12.0, canvas_bbox[1] - 10),
                text=label,
                fill=color,
                anchor="sw",
                tags=("candidate-overlay",),
            )

    def _draw_mini_saved_overlays(self) -> None:
        if not getattr(self, "mini_cvat_enabled", False):
            return
        for (actor_scope_id, frame_index), annotation in (
            self.mini_frame_annotations.items()
        ):
            if frame_index != self.current_frame_index:
                continue
            canvas_bbox = source_bbox_to_canvas(
                annotation.bbox,
                scale=self._display_scale,
                offset=self._display_offset,
            )
            if annotation.bbox_mode == ADDED_BBOX_MODE:
                color = "#ff4dff"
            elif annotation.bbox_mode == CORRECTED_BBOX_MODE:
                color = "#ff9f1c"
            else:
                color = "#00e5ff"
            self.canvas.create_rectangle(
                *canvas_bbox,
                outline=color,
                width=3,
                tags=("mini-cvat-saved",),
            )
            label = (
                f"{self._mini_display_id(actor_scope_id)} ← nguồn {actor_scope_id} · "
                f"{annotation.reviewed_hidden}"
            )
            self.canvas.create_text(
                canvas_bbox[0] + 4,
                max(12.0, canvas_bbox[1] - 10),
                text=label,
                fill=color,
                anchor="sw",
                tags=("mini-cvat-saved",),
            )

    def show_current_frame(self) -> None:
        self._ensure_active_case_frame()
        frame_index = self.current_frame_index
        try:
            source = self._decode_frame(frame_index)
            self._source_image_size = source.size
            rendered = source.copy().convert("RGB")
            selected_by_case = self._selected_by_case(frame_index)
            self._source_image_size = rendered.size
        except RuntimeError as exc:
            messagebox.showerror("Lỗi video", str(exc), parent=self.root)
            return
        fitted = self._fit_to_canvas(rendered)
        self._photo = ImageTk.PhotoImage(fitted)
        self.canvas.delete("all")
        self.canvas.create_image(
            self._display_offset[0],
            self._display_offset[1],
            image=self._photo,
            anchor="nw",
        )
        self._draw_candidate_overlays(
            self.candidates_by_frame[frame_index],
            self.cases,
            selected_by_case,
            self.active_case.review_unit_id,
        )
        self._draw_mini_saved_overlays()
        self._draw_bbox_editor_overlay()
        mapped, total, status = self._case_progress(self.active_case)
        source_frame_index = self._source_frame_index(frame_index)
        frame_key = (self.active_case.review_unit_id, frame_index)
        edit = self.bbox_edits.get(frame_key)
        if frame_key not in self.selections:
            bbox_status = "BBox: chưa chọn"
        elif edit is None:
            bbox_status = "BBox: nguồn gốc, chưa sửa"
        elif edit.mode == ADDED_BBOX_MODE:
            bbox_status = "BBox: A — thêm mới vì bbox nguồn bị mất"
        else:
            bbox_status = "BBox: E — hình học nguồn đã được vẽ lại"
        self.info_var.set(
            "\n".join(
                [
                    f"Video: {self.active_case.video_key}",
                    (
                        f"Frame rà soát {frame_index} → nguồn {source_frame_index} "
                        f"({self.current_frame_position + 1}/{len(self.all_frames)})"
                    ),
                    (
                        "Unit đang rà soát: "
                        f"{self.active_case.review_item_id} | "
                        f"actor cục bộ gốc {self.active_case.original_pig_id or '?'} / "
                        f"{self.active_case.original_track_id or '?'}"
                    ),
                    f"Tiến độ unit: {mapped}/{total} · {status}",
                    bbox_status,
                    "O vàng = bbox nguồn gốc; S xanh/cam = bbox đã chọn.",
                ]
            )
        )
        self._refresh_case_controls()
        self._refresh_candidate_controls()
        self._refresh_bbox_detail()
        self._refresh_mini_actor_controls()
        self._schedule_adjacent_prefetch()

    def set_active_case(self, position: int) -> None:
        self.cancel_bbox_drawing(silent=True)
        self.active_case_position = position % len(self.cases)
        self._ensure_active_case_frame()
        self.show_current_frame()

    def step_frame(self, delta: int) -> None:
        self.cancel_bbox_drawing(silent=True)
        if getattr(self, "mini_cvat_enabled", False):
            next_position = (
                self.current_frame_position + delta
            ) % len(self.all_frames)
            self.current_frame_position = next_position
        else:
            frame_index = step_case_frame(
                self.active_case.frame_indices,
                self.current_frame_index,
                delta,
            )
            self._set_current_frame(frame_index)
        self.show_current_frame()

    def _ensure_mutable(self) -> bool:
        if not self.finalized:
            return True
        messagebox.showwarning(
                "Phiên hiệu chỉnh ID đã khóa",
                "Phiên này đã hoàn tất. Mở lại với --reopen-finalized để sửa.",
            parent=self.root,
        )
        return False

    def _selection_collision(self, candidate: FrameCandidate) -> str | None:
        for case in self.cases:
            if case.review_unit_id == self.active_case.review_unit_id:
                continue
            existing = self.selections.get((case.review_unit_id, self.current_frame_index))
            if existing == candidate.object_track_key:
                return case.review_item_id
        return None

    def select_candidate(self, candidate: FrameCandidate) -> None:
        if getattr(self, "mini_cvat_enabled", False):
            if candidate.pig_id not in self.editable_pig_ids:
                self.status_var.set(
                    "BBox này chỉ xem: actor không nằm trong --editable-pig-id."
                )
                return
            self.cancel_bbox_drawing(silent=True)
            self.active_pig_id = candidate.pig_id
            self.mini_selected_keys[self._mini_frame_key()] = (
                candidate.object_track_key
            )
            self.status_var.set(
                f"Đã chọn {candidate.pig_id}; bbox/Hidden là của frame hiện tại."
            )
            self.show_current_frame()
            return
        if not self._ensure_mutable():
            return
        if self.current_frame_index not in self.active_case.frame_indices:
            messagebox.showerror(
                "Frame ngoài phạm vi unit",
                "Chỉ được chọn bbox trong phạm vi frame của unit hiện tại.",
                parent=self.root,
            )
            return
        if self.active_case.review_unit_id in self.exclusions:
            messagebox.showwarning(
                "Unit đã loại",
                "Hãy khôi phục unit trước khi gán bbox nguồn.",
                parent=self.root,
            )
            return
        collision = self._selection_collision(candidate)
        if collision is not None:
            messagebox.showerror(
                "Một actor chỉ thuộc một unit",
                (
                    f"{candidate_label(candidate)} đã được chọn cho {collision} "
                    f"ở frame {self.current_frame_index}."
                ),
                parent=self.root,
            )
            return
        selection_key = (
            self.active_case.review_unit_id,
            self.current_frame_index,
        )
        if not hasattr(self, "bbox_edits"):
            self.bbox_edits = {}
        prior_selection = self.selections.get(selection_key)
        prior_edit = self.bbox_edits.pop(selection_key, None)
        self.selections[selection_key] = candidate.object_track_key
        if not self.save(silent=False):
            if prior_selection is None:
                self.selections.pop(selection_key, None)
            else:
                self.selections[selection_key] = prior_selection
            if prior_edit is not None:
                self.bbox_edits[selection_key] = prior_edit
            self.status_var.set("Không lưu được lựa chọn; đã khôi phục trạng thái trước đó.")
            return
        self.status_var.set(
            f"Đã lưu {candidate_label(candidate)} cho frame {self.current_frame_index}."
        )
        self.show_current_frame()

    def use_original_box(self) -> None:
        if getattr(self, "mini_cvat_enabled", False):
            candidate = next(
                (
                    item
                    for item in self.candidates_by_frame[self.current_frame_index]
                    if item.pig_id == self.active_pig_id
                ),
                None,
            )
            if candidate is None:
                messagebox.showerror(
                    "Không có bbox nguồn",
                    "Actor này không có bbox nguồn trong frame; dùng Thêm bbox.",
                    parent=self.root,
                )
                return
            self.mini_selected_keys[self._mini_frame_key()] = candidate.object_track_key
            self.mini_frame_annotations.pop(self._mini_frame_key(), None)
            if not self.save(silent=False):
                return
            self.status_var.set("Đã quay về bbox nguồn; chọn Hidden rồi lưu frame nếu cần.")
            self.show_current_frame()
            return
        original_key = self.active_case.original_object_track_key
        candidate = next(
            (
                item
                for item in self.candidates_by_frame[self.current_frame_index]
                if item.object_track_key == original_key
            ),
            None,
        )
        if candidate is None:
            messagebox.showerror(
                "Không có bbox gốc",
                "BBox actor nguồn gốc không có trong frame này.",
                parent=self.root,
            )
            return
        self.select_candidate(candidate)

    def clear_current_selection(self) -> None:
        if not self._ensure_mutable():
            return
        if getattr(self, "mini_cvat_enabled", False):
            key = self._mini_frame_key()
            prior_key = self.mini_selected_keys.pop(key, None)
            prior_annotation = self.mini_frame_annotations.pop(key, None)
            if not self.save(silent=False):
                if prior_key is not None:
                    self.mini_selected_keys[key] = prior_key
                if prior_annotation is not None:
                    self.mini_frame_annotations[key] = prior_annotation
                self.status_var.set("Không lưu được thao tác; đã khôi phục trạng thái trước.")
                return
            self.status_var.set("Đã bỏ sidecar object/frame; bbox nguồn vẫn chỉ để xem.")
            self.show_current_frame()
            return
        selection_key = (
            self.active_case.review_unit_id,
            self.current_frame_index,
        )
        if not hasattr(self, "bbox_edits"):
            self.bbox_edits = {}
        prior_selection = self.selections.pop(selection_key, None)
        prior_edit = self.bbox_edits.pop(selection_key, None)
        if not self.save(silent=False):
            if prior_selection is not None:
                self.selections[selection_key] = prior_selection
            if prior_edit is not None:
                self.bbox_edits[selection_key] = prior_edit
            self.status_var.set("Không lưu được lựa chọn; đã khôi phục trạng thái trước đó.")
            return
        self.status_var.set(f"Đã xóa lựa chọn frame {self.current_frame_index}; sidecar đã lưu.")
        self.show_current_frame()

    def exclude_active_case(self) -> None:
        if not self._ensure_mutable():
            return
        note = simpledialog.askstring(
            "Exclude identity-continuity case",
            (
                "This excludes only this unit from future training application.\n"
                "Explain why its actor trajectory cannot be adjudicated:"
            ),
            parent=self.root,
        )
        if note is None:
            return
        note = note.strip()
        if not note:
            messagebox.showerror(
                "Reason required",
                "An identity-continuity exclusion requires a short reason.",
                parent=self.root,
            )
            return
        case_id = self.active_case.review_unit_id
        if not hasattr(self, "bbox_edits"):
            self.bbox_edits = {}
        prior_selections = {
            (case_id, frame_index): self.selections[(case_id, frame_index)]
            for frame_index in self.active_case.frame_indices
            if (case_id, frame_index) in self.selections
        }
        prior_bbox_edits = {
            (case_id, frame_index): self.bbox_edits[(case_id, frame_index)]
            for frame_index in self.active_case.frame_indices
            if (case_id, frame_index) in self.bbox_edits
        }
        prior_exclusion = self.exclusions.get(case_id)
        for frame_index in self.active_case.frame_indices:
            self.selections.pop((case_id, frame_index), None)
            self.bbox_edits.pop((case_id, frame_index), None)
        self.exclusions[case_id] = note
        if not self.save(silent=False):
            self.selections.update(prior_selections)
            self.bbox_edits.update(prior_bbox_edits)
            if prior_exclusion is None:
                self.exclusions.pop(case_id, None)
            else:
                self.exclusions[case_id] = prior_exclusion
            self.status_var.set("Exclusion was not saved; prior state restored.")
            return
        self.status_var.set("Unit excluded from future training application; sidecars saved.")
        self.show_current_frame()

    def restore_active_case(self) -> None:
        if not self._ensure_mutable():
            return
        case_id = self.active_case.review_unit_id
        if case_id not in self.exclusions:
            self.status_var.set("Active unit is not excluded.")
            return
        prior_exclusion = self.exclusions.pop(case_id)
        if not self.save(silent=False):
            self.exclusions[case_id] = prior_exclusion
            self.status_var.set("Exclusion was not saved; prior state restored.")
            return
        self.status_var.set("Exclusion removed; map every frame before finalizing.")
        self.show_current_frame()

    def _completion_errors(self) -> list[str]:
        if getattr(self, "mini_cvat_enabled", False):
            return validate_mini_cvat_state(
                self.mini_actor_attributes,
                self.mini_frame_annotations,
                editable_actor_ids=self.editable_pig_ids,
                frame_indices=self.all_frames,
                require_complete=True,
            )
        return validate_adjudication(
            self.cases,
            self.candidates_by_frame,
            self.selections,
            self.exclusions,
            self.bbox_edits,
            allow_pending=False,
        )

    def finalize(self) -> None:
        if getattr(self, "mini_cvat_enabled", False):
            errors = self._completion_errors()
            if errors:
                messagebox.showerror(
                    "Mini-CVAT chưa hoàn tất",
                    "\n".join(errors),
                    parent=self.root,
                )
                return
            if not self.save(silent=False):
                return
            self.status_var.set(
                "Mini-CVAT hoàn tất: sidecar đã lưu, chưa áp dụng vào dữ liệu nguồn."
            )
            messagebox.showinfo(
                "Mini-CVAT sidecar complete",
                "ID/behavior nhất quán theo burst; bbox/Hidden đã lưu từng frame.\n"
                "Chưa có source annotation hay Behavior decision ledger nào bị thay đổi.",
                parent=self.root,
            )
            return
        if self.finalized:
            messagebox.showinfo(
                "Identity sidecars finalized",
                "Session này đã final; dùng --reopen-finalized để sửa.",
                parent=self.root,
            )
            return
        errors = self._completion_errors()
        if errors:
            messagebox.showerror(
                "Không thể hoàn tất hiệu chỉnh định danh",
                "\n".join(errors),
                parent=self.root,
            )
            return
        if not self.save(silent=False):
            return
        try:
            marker_path = write_finalization_marker(
                self.output_dir,
                reviewer=self.config.reviewer,
            )
        except (IdentityAdjudicationError, OSError) as exc:
            messagebox.showerror(
                "Không thể khóa sidecar định danh",
                str(exc),
                parent=self.root,
            )
            return
        self.finalization_marker = load_finalization_marker(self.output_dir)
        self.finalized = True
        self.status_var.set(f"Finalized and locked: {marker_path.name}")
        messagebox.showinfo(
            "Identity sidecars complete",
            (
                    "Mỗi unit không bị loại có một bbox actor có audit cho mỗi frame.\n"
                "No behavior decision, source annotation, or train artifact was changed."
            ),
            parent=self.root,
        )

    def save(self, *, silent: bool = False) -> bool:
        if self.finalized:
            if not silent:
                messagebox.showwarning(
                    "Identity session finalized",
                    "Dùng --reopen-finalized trước khi ghi thay đổi.",
                    parent=self.root,
                )
            return False
        try:
            frame_path, case_path = write_session_sidecars(
                self.output_dir,
                self.cases,
                self.candidates_by_frame,
                self.selections,
                self.exclusions,
                self.config.reviewer,
                self.bbox_edits,
            )
            mini_path = None
            if getattr(self, "mini_cvat_enabled", False):
                mini_path = write_mini_cvat_sidecar(
                    self.output_dir,
                    reviewer=self.config.reviewer,
                    source_type=self.cases[0].source_type,
                    dataset_id=self.cases[0].dataset_id,
                    video_key=self.cases[0].video_key,
                    editable_actor_ids=self.editable_pig_ids,
                    frame_indices=self.all_frames,
                    actor_attributes=self.mini_actor_attributes,
                    frame_annotations=self.mini_frame_annotations,
                )
        except (IdentityAdjudicationError, MiniCvatAdjudicationError, OSError) as exc:
            messagebox.showerror("Không thể lưu sidecar định danh", str(exc), parent=self.root)
            return False
        if not silent:
            names = [frame_path.name, case_path.name]
            if mini_path is not None:
                names.append(mini_path.name)
            self.status_var.set("Saved: " + "; ".join(names))
        return True

    def reset_all_session_changes(self) -> None:
        if not self._ensure_mutable():
            return
        confirmed = messagebox.askyesno(
            "Loại bỏ mọi thay đổi phiên?",
            (
                "Xóa toàn bộ lựa chọn, loại unit, bbox sửa và thử nghiệm ID hiện "
                "tại?\nHành động này chỉ dùng để quay về phiên sạch."
            ),
            parent=self.root,
        )
        if not confirmed:
            self.status_var.set("Đã hủy reset; phiên giữ nguyên.")
            return

        prior_selections = dict(self.selections)
        prior_exclusions = dict(self.exclusions)
        prior_bbox_edits = dict(self.bbox_edits)
        prior_active_pig_id = getattr(self, "active_pig_id", "")
        prior_current = (self.current_frame_position, self.active_case_position)
        prior_mini_attrs = dict(getattr(self, "mini_actor_attributes", {}))
        prior_mini_frames = dict(getattr(self, "mini_frame_annotations", {}))
        prior_mini_selected = dict(getattr(self, "mini_selected_keys", {}))

        self.cancel_bbox_drawing(silent=True)
        self.selections = {}
        self.exclusions = {}
        self.bbox_edits = {}
        if getattr(self, "mini_cvat_enabled", False):
            self.mini_actor_attributes = {}
            self.mini_frame_annotations = {}
            self.mini_selected_keys = {}

        self.current_frame_position, self.active_case_position = self._resume_position()
        if getattr(self, "mini_cvat_enabled", False) and self.editable_pig_ids:
            self.active_pig_id = self.editable_pig_ids[0]

        if not self.save(silent=False):
            self.selections = prior_selections
            self.exclusions = prior_exclusions
            self.bbox_edits = prior_bbox_edits
            if getattr(self, "mini_cvat_enabled", False):
                self.mini_actor_attributes = prior_mini_attrs
                self.mini_frame_annotations = prior_mini_frames
                self.mini_selected_keys = prior_mini_selected
                self.active_pig_id = prior_active_pig_id
            self.current_frame_position, self.active_case_position = prior_current
            self.status_var.set(
                "Không thể reset phiên; đã khôi phục trạng thái trước."
            )
            self.show_current_frame()
            return

        self._ensure_active_case_frame()
        self.show_current_frame()
        self.status_var.set("Đã loại bỏ mọi thay đổi phiên; sidecar đã sạch.")

    def start_corrected_bbox(self) -> None:
        if not self._ensure_mutable():
            return
        if getattr(self, "mini_cvat_enabled", False):
            has_bbox = self._active_effective_bbox() is not None
        else:
            has_bbox = bool(self.selections.get(self._active_frame_key()))
        if not has_bbox:
            messagebox.showwarning(
                "Cần chọn bbox",
                "Chọn bbox nguồn hoặc thêm bbox mới trước khi chỉnh.",
                parent=self.root,
            )
            return
        self.cancel_bbox_drawing(silent=True)
        self._bbox_draw_mode = CORRECTED_BBOX_MODE
        self.status_var.set(
            "Chế độ chỉnh: kéo bên trong bbox để di chuyển; "
            "kéo 8 ô vuông để resize."
        )

    def start_added_bbox(self) -> None:
        if not self._ensure_mutable():
            return
        if (
            not getattr(self, "mini_cvat_enabled", False)
            and self.active_case.review_unit_id in self.exclusions
        ):
            messagebox.showwarning(
                "Unit đã loại",
                "Khôi phục unit trước khi thêm bbox bị mất.",
                parent=self.root,
            )
            return
        self.cancel_bbox_drawing(silent=True)
        self._bbox_draw_mode = ADDED_BBOX_MODE
        self.status_var.set(
            "Chế độ thêm: kéo tạo bbox mới; sau khi lưu có thể kéo/resize tiếp."
        )

    def cancel_bbox_drawing(self, *, silent: bool = False) -> None:
        self._bbox_draw_mode = None
        self._bbox_drag_start = None
        self._bbox_interaction = None
        self._bbox_resize_handle = None
        self._bbox_origin = None
        self._bbox_preview = None
        if self._bbox_drag_rectangle is not None:
            self.canvas.delete(self._bbox_drag_rectangle)
            self._bbox_drag_rectangle = None
        self.canvas.delete("bbox-preview")
        self.canvas.configure(cursor="")
        self._draw_bbox_editor_overlay()
        if not silent:
            self.status_var.set("Đã hủy thao tác bbox; dữ liệu đã lưu không đổi.")

    def _begin_existing_bbox_interaction(
        self,
        point: tuple[float, float],
    ) -> bool:
        bbox = self._active_effective_bbox()
        if bbox is None:
            return False
        canvas_bbox = source_bbox_to_canvas(
            bbox,
            scale=self._display_scale,
            offset=self._display_offset,
        )
        handle = hit_test_bbox_handle(point, canvas_bbox)
        if handle is not None:
            self._bbox_interaction = "resize"
            self._bbox_resize_handle = handle
        elif canvas_point_inside_bbox(point, canvas_bbox):
            self._bbox_interaction = "move"
            self._bbox_resize_handle = None
        else:
            return False
        self._bbox_drag_start = point
        self._bbox_origin = bbox
        self._bbox_preview = bbox
        return True

    def _preview_existing_bbox(self, point: tuple[float, float]) -> None:
        if self._bbox_drag_start is None or self._bbox_origin is None:
            return
        delta = (
            (point[0] - self._bbox_drag_start[0]) / self._display_scale,
            (point[1] - self._bbox_drag_start[1]) / self._display_scale,
        )
        operation = (
            "move"
            if self._bbox_interaction == "move"
            else self._bbox_resize_handle or ""
        )
        self._bbox_preview = transform_source_bbox(
            self._bbox_origin,
            delta=delta,
            operation=operation,
            source_size=self._source_image_size,
        )
        self._draw_bbox_editor_overlay(self._bbox_preview)
        if hasattr(self, "bbox_detail_var"):
            x1, y1, x2, y2 = self._bbox_preview
            self.bbox_detail_var.set(
                f"ĐANG CHỈNH · xyxy {x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}\n"
                f"Kích thước {x2 - x1:.1f} × {y2 - y1:.1f} px"
            )

    def _persist_effective_bbox(
        self,
        bbox: tuple[float, float, float, float],
        *,
        added: bool,
    ) -> bool:
        if getattr(self, "mini_cvat_enabled", False):
            candidate = self._mini_current_candidate()
            if self.mini_hidden_var.get().strip() not in HIDDEN_VALUES:
                self.status_var.set("Chọn Hidden trước khi lưu bbox.")
                return False
            if added or candidate is None:
                bbox_mode = ADDED_BBOX_MODE
            elif bbox == candidate.bbox:
                bbox_mode = "SOURCE_BBOX"
            else:
                bbox_mode = CORRECTED_BBOX_MODE
            annotation = self._mini_source_annotation(
                bbox=bbox,
                bbox_mode=bbox_mode,
            )
            if annotation is None:
                self.status_var.set("Không thể xác định actor scope để lưu bbox.")
                return False
            key = self._mini_frame_key()
            prior = self.mini_frame_annotations.get(key)
            self.mini_frame_annotations[key] = annotation
            if not self.save(silent=False):
                if prior is None:
                    self.mini_frame_annotations.pop(key, None)
                else:
                    self.mini_frame_annotations[key] = prior
                return False
            return True
        frame_key = self._active_frame_key()
        prior_selection = self.selections.get(frame_key)
        prior_edit = self.bbox_edits.get(frame_key)
        if added:
            selected_key = MANUAL_BBOX_SELECTION_KEY
        else:
            selected_key = prior_selection
        if not selected_key:
            self.status_var.set("Chưa có actor được chọn; bbox không được lưu.")
            return False
        mode = (
            ADDED_BBOX_MODE
            if selected_key == MANUAL_BBOX_SELECTION_KEY
            else CORRECTED_BBOX_MODE
        )
        self.selections[frame_key] = selected_key
        self.bbox_edits[frame_key] = BoundingBoxEdit(
            mode=mode,
            x1=bbox[0],
            y1=bbox[1],
            x2=bbox[2],
            y2=bbox[3],
            source_object_track_key=(
                "" if mode == ADDED_BBOX_MODE else selected_key
            ),
        )
        if self.save(silent=False):
            return True
        if prior_selection is None:
            self.selections.pop(frame_key, None)
        else:
            self.selections[frame_key] = prior_selection
        if prior_edit is None:
            self.bbox_edits.pop(frame_key, None)
        else:
            self.bbox_edits[frame_key] = prior_edit
        self.status_var.set("Không lưu được bbox; đã khôi phục trạng thái trước.")
        return False

    def _on_canvas_hover(self, event: tk.Event[Any]) -> None:
        if self._bbox_interaction is not None:
            return
        if self._bbox_draw_mode == ADDED_BBOX_MODE:
            self.canvas.configure(cursor="crosshair")
            return
        bbox = self._active_effective_bbox()
        if bbox is None:
            self.canvas.configure(cursor="")
            return
        canvas_bbox = source_bbox_to_canvas(
            bbox,
            scale=self._display_scale,
            offset=self._display_offset,
        )
        point = float(event.x), float(event.y)
        if hit_test_bbox_handle(point, canvas_bbox) is not None:
            self.canvas.configure(cursor="crosshair")
        elif canvas_point_inside_bbox(point, canvas_bbox):
            self.canvas.configure(cursor="fleur")
        else:
            self.canvas.configure(cursor="")

    def _on_canvas_press(self, event: tk.Event[Any]) -> None:
        point = float(event.x), float(event.y)
        if getattr(self, "mini_cvat_enabled", False) and self._bbox_draw_mode is None:
            candidate = candidate_at_display_point(
                self.candidates_by_frame[self.current_frame_index],
                point[0],
                point[1],
                self._display_scale,
                self._display_offset,
            )
            if candidate is not None and candidate.pig_id != self.active_pig_id:
                self.select_candidate(candidate)
                return
        if self._bbox_draw_mode == ADDED_BBOX_MODE:
            if not self._ensure_mutable():
                return
            self._bbox_interaction = "add"
            self._bbox_drag_start = point
            self._bbox_origin = None
            self._bbox_preview = None
            self.canvas.configure(cursor="crosshair")
            return
        if not self._ensure_mutable():
            return
        if self._begin_existing_bbox_interaction(point):
            cursor = "fleur" if self._bbox_interaction == "move" else "crosshair"
            self.canvas.configure(cursor=cursor)
            return
        self._on_canvas_click(event)

    def _on_canvas_drag(self, event: tk.Event[Any]) -> None:
        if self._bbox_drag_start is None or self._bbox_interaction is None:
            return
        point = float(event.x), float(event.y)
        if self._bbox_interaction == "add":
            self.canvas.delete("bbox-preview")
            self.canvas.create_rectangle(
                self._bbox_drag_start[0],
                self._bbox_drag_start[1],
                point[0],
                point[1],
                outline="#ff4dff",
                width=3,
                dash=(6, 3),
                tags=("bbox-preview",),
            )
            return
        self._preview_existing_bbox(point)

    def _on_canvas_release(self, event: tk.Event[Any]) -> None:
        if self._bbox_drag_start is None or self._bbox_interaction is None:
            return
        point = float(event.x), float(event.y)
        movement = max(
            abs(point[0] - self._bbox_drag_start[0]),
            abs(point[1] - self._bbox_drag_start[1]),
        )
        interaction = self._bbox_interaction
        if interaction == "add":
            bbox = canvas_drag_to_source_bbox(
                self._bbox_drag_start,
                point,
                scale=self._display_scale,
                offset=self._display_offset,
                source_size=self._source_image_size,
            )
            self.canvas.delete("bbox-preview")
            self._bbox_interaction = None
            self._bbox_drag_start = None
            if bbox is None:
                self.status_var.set(
                    "BBox quá nhỏ hoặc ngoài ảnh; vẫn ở chế độ thêm để thử lại."
                )
                return
            saved = self._persist_effective_bbox(bbox, added=True)
            self.cancel_bbox_drawing(silent=True)
            if saved:
                self.status_var.set(
                    "Đã thêm và lưu bbox; giờ có thể kéo hoặc resize trực tiếp."
                )
            self.show_current_frame()
            return

        if movement < 2.0:
            self._bbox_interaction = None
            self._bbox_drag_start = None
            self._bbox_resize_handle = None
            self._bbox_origin = None
            self._bbox_preview = None
            self.canvas.configure(cursor="")
            self._draw_bbox_editor_overlay()
            self._refresh_bbox_detail()
            return
        self._preview_existing_bbox(point)
        bbox = self._bbox_preview
        saved = bbox is not None and self._persist_effective_bbox(
            bbox,
            added=False,
        )
        self.cancel_bbox_drawing(silent=True)
        if saved:
            action = "di chuyển" if interaction == "move" else "resize"
            self.status_var.set(f"Đã {action} và autosave bbox.")
        self.show_current_frame()

    def _on_canvas_click(self, event: tk.Event[Any]) -> None:
        candidate = candidate_at_display_point(
            self.candidates_by_frame[self.current_frame_index],
            float(event.x),
            float(event.y),
            self._display_scale,
            self._display_offset,
        )
        if candidate is not None:
            self.select_candidate(candidate)

    def _on_keypress(self, event: tk.Event[Any]) -> str | None:
        if isinstance(event.widget, (tk.Entry, ttk.Entry)):
            return None
        key = str(event.keysym).casefold()
        if key == "left":
            self.step_frame(-1)
        elif key == "right":
            self.step_frame(1)
        elif key == "tab":
            if getattr(self, "mini_cvat_enabled", False):
                position = self.editable_pig_ids.index(self.active_pig_id)
                self.select_mini_actor(
                    self.editable_pig_ids[(position + 1) % len(self.editable_pig_ids)]
                )
            else:
                self.set_active_case(self.active_case_position + 1)
        elif key == "o":
            self.use_original_box()
        elif key == "e":
            self.start_corrected_bbox()
        elif key == "a":
            self.start_added_bbox()
        elif key == "escape":
            self.cancel_bbox_drawing()
        elif key == "u":
            self.clear_current_selection()
        elif key == "x":
            self.exclude_active_case()
        elif key == "r":
            self.restore_active_case()
        elif key == "s" and bool(getattr(event, "state", 0) & 0x4):
            self.save()
        elif key == "f":
            self.finalize()
        elif key.isdigit() and 1 <= int(key) <= 9:
            candidates = self.candidates_by_frame[self.current_frame_index]
            position = int(key) - 1
            if position < len(candidates):
                self.select_candidate(candidates[position])
        else:
            return None
        return "break"

    def close(self) -> None:
        if not self.finalized and not self.save(silent=False):
            return
        if self._prefetch_after_id is not None:
            self.root.after_cancel(self._prefetch_after_id)
            self._prefetch_after_id = None
        self.capture.release()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> IdentityGuiConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-units-csv",
        required=True,
        type=Path,
        help="Immutable review-unit view; behavior decision CSVs are not accepted inputs.",
    )
    parser.add_argument(
        "--frame-features-csv",
        required=True,
        type=Path,
        help="Native source feature CSV containing full-scene actor boxes.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New identity-adjudication session root, never an existing behavior session.",
    )
    parser.add_argument("--reviewer", required=True)
    parser.add_argument(
        "--review-item-id",
        action="append",
        dest="review_item_ids",
        required=True,
        help="Repeat for each short review item, e.g. unit_review_00030931.",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help="Optional exact declared full-scene source_video_path override.",
    )
    parser.add_argument(
        "--editable-pig-id",
        action="append",
        default=[],
        help=(
            "Pig ID editable in mini-CVAT mode; repeat for every actor "
            "scope (for example ID_4 ID_5 ID_6)."
        ),
    )
    parser.add_argument(
        "--reopen-finalized",
        action="store_true",
        help="Explicitly reopen a finalized identity sidecar session for amendment.",
    )
    args = parser.parse_args()
    return IdentityGuiConfig(
        review_units_csv=args.review_units_csv,
        frame_features_csv=args.frame_features_csv,
        output_dir=args.output_dir,
        reviewer=args.reviewer.strip(),
        review_item_ids=tuple(args.review_item_ids),
        video_path=args.video_path,
        reopen_finalized=bool(args.reopen_finalized),
        editable_pig_ids=tuple(args.editable_pig_id),
    )


def main() -> None:
    config = parse_args()
    if not config.reviewer:
        raise SystemExit("--reviewer must be nonempty")
    gui = IdentityContinuityGui(config)
    gui.run()


if __name__ == "__main__":
    main()
