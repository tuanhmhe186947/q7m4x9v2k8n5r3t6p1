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
from dataclasses import dataclass
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

from pig_behavior.classification_v2.review.identity_continuity_adjudication import (
    CASE_SIDECAR_NAME,
    FRAME_SIDECAR_NAME,
    FrameCandidate,
    IdentityAdjudicationError,
    IdentityCase,
    assert_safe_output_dir,
    assert_single_scene,
    case_status,
    load_frame_candidates,
    load_identity_cases,
    load_session_sidecars,
    source_frame_index_for_review_frame,
    validate_adjudication,
    write_session_sidecars,
)

WINDOW_TITLE = "Classification V2 — Hiệu chỉnh liên tục actor"
MAX_RENDERED_FRAME_CACHE = 12
FINALIZATION_FILE_NAME = "identity_continuity_finalization.json"
FINALIZATION_SCHEMA = "classification_v2.identity_continuity_finalization.v1"
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


class IdentityContinuityGui:
    """Click or key-select existing full-frame boxes for each selected case."""

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
        self.selections, self.exclusions = load_session_sidecars(
            self.output_dir,
            self.cases,
            self.candidates_by_frame,
        )
        self.all_frames = tuple(
            sorted({frame for case in self.cases for frame in case.frame_indices})
        )
        self.current_frame_position, self.active_case_position = self._resume_position()
        self.resumed = bool(self.selections or self.exclusions)
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
        self._photo: ImageTk.PhotoImage | None = None
        self._prefetch_after_id: str | None = None

        self.root = tk.Tk()
        self.root.title(WINDOW_TITLE)
        self.root.minsize(1180, 760)
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
        if self.current_frame_index in self.active_case.frame_indices:
            return
        self._set_current_frame(
            first_pending_case_frame(self.active_case, self.selections)
        )

    def _resume_position(self) -> tuple[int, int]:
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
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        side = ttk.Frame(root, padding=8, width=400)
        side.grid(row=0, column=1, sticky="ns")
        ttk.Label(
            side,
            text="Chỉ hiệu chỉnh actor/bbox — không đổi nhãn hành vi",
            wraplength=370,
            foreground="#8b0000",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(side, textvariable=self.info_var, wraplength=370).grid(
            row=1,
            column=0,
            sticky="ew",
        )
        self.case_frame = ttk.LabelFrame(side, text="Các unit cần rà soát", padding=6)
        self.case_frame.grid(row=2, column=0, sticky="ew", pady=(10, 6))
        self.candidate_frame = ttk.LabelFrame(side, text="BBox trong frame", padding=6)
        self.candidate_frame.grid(row=3, column=0, sticky="ew", pady=6)
        controls = ttk.LabelFrame(side, text="Điều khiển", padding=6)
        controls.grid(row=4, column=0, sticky="ew", pady=6)
        ttk.Button(controls, text="← Frame trước", command=lambda: self.step_frame(-1)).grid(
            row=0,
            column=0,
            sticky="ew",
            padx=2,
            pady=2,
        )
        ttk.Button(controls, text="Frame tiếp →", command=lambda: self.step_frame(1)).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=2,
            pady=2,
        )
        ttk.Button(controls, text="Dùng bbox gốc", command=self.use_original_box).grid(
            row=1,
            column=0,
            sticky="ew",
            padx=2,
            pady=2,
        )
        ttk.Button(
            controls,
            text="Xóa chọn frame",
            command=self.clear_current_selection,
        ).grid(
            row=1,
            column=1,
            sticky="ew",
            padx=2,
            pady=2,
        )
        ttk.Button(controls, text="Loại unit hiện tại", command=self.exclude_active_case).grid(
            row=2,
            column=0,
            sticky="ew",
            padx=2,
            pady=2,
        )
        ttk.Button(controls, text="Khôi phục unit", command=self.restore_active_case).grid(
            row=2,
            column=1,
            sticky="ew",
            padx=2,
            pady=2,
        )
        ttk.Button(controls, text="Lưu sidecar", command=self.save).grid(
            row=3,
            column=0,
            sticky="ew",
            padx=2,
            pady=2,
        )
        ttk.Button(controls, text="Hoàn tất kiểm tra", command=self.finalize).grid(
            row=3,
            column=1,
            sticky="ew",
            padx=2,
            pady=2,
        )
        for column in range(2):
            controls.columnconfigure(column, weight=1)
        ttk.Label(
            side,
            text=(
                "Phím: ←/→ frame · Tab đổi unit · 1–9 chọn box · "
                "O bbox gốc · U xóa · X loại · R khôi phục · F hoàn tất · Ctrl+S lưu"
            ),
            wraplength=370,
            foreground="#404040",
        ).grid(row=5, column=0, sticky="ew", pady=(8, 3))
        ttk.Label(side, textvariable=self.status_var, wraplength=370).grid(
            row=6,
            column=0,
            sticky="ew",
        )

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

    def _case_progress(self, case: IdentityCase) -> tuple[int, int, str]:
        mapped = sum(
            (case.review_unit_id, frame_index) in self.selections
            for frame_index in case.frame_indices
        )
        return mapped, len(case.frame_indices), case_status(case, self.selections, self.exclusions)

    def _refresh_case_controls(self) -> None:
        for child in self.case_frame.winfo_children():
            child.destroy()
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
        candidates = self.candidates_by_frame[self.current_frame_index]
        for index, candidate in enumerate(candidates, start=1):
            selected = self.selections.get(
                (self.active_case.review_unit_id, self.current_frame_index)
            ) == candidate.object_track_key
            text = candidate_label(candidate, index)
            if selected:
                text = f"✓ {text}"
            ttk.Button(
                self.candidate_frame,
                text=text,
                command=lambda chosen=candidate: self.select_candidate(chosen),
            ).grid(row=index - 1, column=0, sticky="ew", pady=1)
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

    def show_current_frame(self) -> None:
        self._ensure_active_case_frame()
        frame_index = self.current_frame_index
        try:
            source = self._decode_frame(frame_index)
            rendered = render_identity_frame(
                source,
                self.candidates_by_frame[frame_index],
                self.cases,
                self._selected_by_case(frame_index),
                self.active_case.review_unit_id,
            )
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
        mapped, total, status = self._case_progress(self.active_case)
        source_frame_index = self._source_frame_index(frame_index)
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
                "O vàng = bbox nguồn gốc; S xanh/cam = bbox đã chọn.",
                ]
            )
        )
        self._refresh_case_controls()
        self._refresh_candidate_controls()
        self._schedule_adjacent_prefetch()

    def set_active_case(self, position: int) -> None:
        self.active_case_position = position % len(self.cases)
        self._ensure_active_case_frame()
        self.show_current_frame()

    def step_frame(self, delta: int) -> None:
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
        prior_selection = self.selections.get(selection_key)
        self.selections[selection_key] = candidate.object_track_key
        if not self.save(silent=True):
            if prior_selection is None:
                self.selections.pop(selection_key, None)
            else:
                self.selections[selection_key] = prior_selection
            self.status_var.set("Không lưu được lựa chọn; đã khôi phục trạng thái trước đó.")
            return
        self.status_var.set(
            f"Đã lưu {candidate_label(candidate)} cho frame {self.current_frame_index}."
        )
        self.show_current_frame()

    def use_original_box(self) -> None:
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
        selection_key = (
            self.active_case.review_unit_id,
            self.current_frame_index,
        )
        prior_selection = self.selections.pop(selection_key, None)
        if not self.save(silent=True):
            if prior_selection is not None:
                self.selections[selection_key] = prior_selection
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
        prior_selections = {
            (case_id, frame_index): self.selections[(case_id, frame_index)]
            for frame_index in self.active_case.frame_indices
            if (case_id, frame_index) in self.selections
        }
        prior_exclusion = self.exclusions.get(case_id)
        for frame_index in self.active_case.frame_indices:
            self.selections.pop((case_id, frame_index), None)
        self.exclusions[case_id] = note
        if not self.save(silent=True):
            self.selections.update(prior_selections)
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
        if not self.save(silent=True):
            self.exclusions[case_id] = prior_exclusion
            self.status_var.set("Exclusion was not saved; prior state restored.")
            return
        self.status_var.set("Exclusion removed; map every frame before finalizing.")
        self.show_current_frame()

    def _completion_errors(self) -> list[str]:
        return validate_adjudication(
            self.cases,
            self.candidates_by_frame,
            self.selections,
            self.exclusions,
            allow_pending=False,
        )

    def finalize(self) -> None:
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
        if not self.save(silent=True):
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
                "Every non-excluded unit has one existing source box per frame.\n"
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
            )
        except (IdentityAdjudicationError, OSError) as exc:
            messagebox.showerror("Không thể lưu sidecar định danh", str(exc), parent=self.root)
            return False
        if not silent:
            self.status_var.set(f"Saved: {frame_path.name}; {case_path.name}")
        return True

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
            self.set_active_case(self.active_case_position + 1)
        elif key == "o":
            self.use_original_box()
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
        if not self.finalized and not self.save(silent=True):
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
    )


def main() -> None:
    config = parse_args()
    if not config.reviewer:
        raise SystemExit("--reviewer must be nonempty")
    gui = IdentityContinuityGui(config)
    gui.run()


if __name__ == "__main__":
    main()
