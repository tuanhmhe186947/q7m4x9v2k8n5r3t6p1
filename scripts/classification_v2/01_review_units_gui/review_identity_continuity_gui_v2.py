"""Focused mini-CVAT editor for local ID, bbox, Hidden, and burst behavior."""

from __future__ import annotations

import argparse
import csv
import sys
import tkinter as tk
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from PIL import Image, ImageTk

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pig_behavior.classification_v2.review.behavior_review_contract import (  # noqa: E402
    CANONICAL_BEHAVIORS,
)
from pig_behavior.classification_v2.review.identity_continuity_adjudication import (  # noqa: E402
    SOURCE_BBOX_MODE,
    FrameCandidate,
    IdentityAdjudicationError,
    IdentityCase,
    assert_safe_output_dir,
    assert_single_scene,
    load_frame_candidates,
    load_identity_cases,
    source_frame_index_for_review_frame,
)
from pig_behavior.classification_v2.review.identity_continuity_apply import (  # noqa: E402
    IdentitySourceApplyError,
    apply_identity_adjudication,
)
from pig_behavior.classification_v2.review.mini_cvat_adjudication import (  # noqa: E402
    HIDDEN_VALUES,
    MINI_CVAT_SIDECAR_NAME,
    load_mini_cvat_sidecar,
    write_mini_cvat_sidecar,
)
from pig_behavior.classification_v2.review.mini_cvat_editor import (  # noqa: E402
    BBox,
    DragIntent,
    FrameDraft,
    MiniCvatEditorError,
    MiniCvatEditorState,
    bbox_from_canvas_drag,
    bbox_handle_points,
    begin_bbox_drag,
    preview_bbox_drag,
    smallest_candidate_at_point,
    source_bbox_to_canvas,
)

WINDOW_TITLE = "Classification V2 — Mini-CVAT Identity Editor V2"
FRAME_CACHE_SIZE = 6
ACTOR_COLORS = (
    "#00e5ff",
    "#ff9f1c",
    "#ff4dff",
    "#65d46e",
    "#ffd966",
    "#ad8cff",
)


@dataclass(frozen=True)
class GuiConfig:
    review_units_csv: Path
    frame_features_csv: Path
    output_dir: Path
    reviewer: str
    review_item_ids: tuple[str, ...]
    editable_pig_ids: tuple[str, ...]
    video_root: Path | None
    apply_source_csvs: tuple[Path, ...]
    apply_source_xml: Path | None
    apply_group_id: str


class CleanFrameCache:
    """Small bounded cache of undecorated RGB source frames."""

    def __init__(self, maximum: int = FRAME_CACHE_SIZE) -> None:
        self.maximum = maximum
        self._items: OrderedDict[int, Image.Image] = OrderedDict()

    def get(self, frame_index: int) -> Image.Image | None:
        image = self._items.get(frame_index)
        if image is None:
            return None
        self._items.move_to_end(frame_index)
        return image

    def put(self, frame_index: int, image: Image.Image) -> None:
        self._items[frame_index] = image
        self._items.move_to_end(frame_index)
        while len(self._items) > self.maximum:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()


def decode_exact_source_frame(
    capture: Any,
    source_frame_index: int,
) -> Any:
    """Decode exactly one requested source frame."""

    if not capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame_index):
        raise RuntimeError(
            f"cannot_seek_source_frame={source_frame_index}"
        )
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(
            f"cannot_decode_source_frame={source_frame_index}"
        )
    decoded_index = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
    if decoded_index != source_frame_index:
        raise RuntimeError(
            "decoded_source_frame_mismatch="
            f"requested:{source_frame_index};decoded:{decoded_index}"
        )
    return frame


def resolve_video_path(
    cases: tuple[IdentityCase, ...],
    candidates_by_frame: dict[int, tuple[FrameCandidate, ...]],
    video_root: Path | None,
) -> Path:
    """Resolve one full-scene video without altering source authority."""

    candidate_paths = {
        Path(candidate.source_video_path)
        for candidates in candidates_by_frame.values()
        for candidate in candidates
        if candidate.source_video_path
    }
    existing = sorted(
        path.resolve() for path in candidate_paths if path.is_file()
    )
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise IdentityAdjudicationError(
            "multiple_existing_source_videos="
            + ",".join(str(path) for path in existing)
        )
    if video_root is not None:
        video_key = Path(*cases[0].video_key.split("/"))
        flat_video_names = [f"{video_key.name}.mp4"]
        if not video_key.name.lower().endswith("_30fps"):
            flat_video_names.append(f"{video_key.name}_30fps.mp4")
        if video_key.name.lower().startswith("pigs"):
            flat_video_names.extend(
                f"Pigs{name[4:]}" for name in tuple(flat_video_names)
            )
        guesses = {
            video_root / video_key / "color.mp4",
            video_root / f"{cases[0].video_key}.mp4",
            video_root / video_key.with_suffix(".mp4"),
            *(video_root / name for name in flat_video_names),
        }
        matched = sorted(
            {path.resolve() for path in guesses if path.is_file()}
        )
        if len(matched) == 1:
            return matched[0]
    raise IdentityAdjudicationError(
        "source_video_not_found="
        + ",".join(str(path) for path in sorted(candidate_paths))
    )


class MiniCvatGuiV2:
    """Thin Tk layer over deterministic editor state."""

    def __init__(self, root: tk.Tk, config: GuiConfig) -> None:
        if cv2 is None:
            raise RuntimeError("opencv_python_is_required")
        self.root = root
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
        self.frames = tuple(
            sorted(
                {
                    frame_index
                    for case in self.cases
                    for frame_index in case.frame_indices
                }
            )
        )
        actor_attributes, frame_annotations = load_mini_cvat_sidecar(
            self.output_dir,
            source_type=self.cases[0].source_type,
            dataset_id=self.cases[0].dataset_id,
            video_key=self.cases[0].video_key,
            editable_actor_ids=config.editable_pig_ids,
            frame_indices=self.frames,
        )
        self.state = MiniCvatEditorState(
            editable_actor_ids=config.editable_pig_ids,
            frame_indices=self.frames,
            candidates_by_frame=self.candidates_by_frame,
            actor_attributes=actor_attributes,
            frame_annotations=frame_annotations,
        )
        self.video_path = resolve_video_path(
            self.cases,
            self.candidates_by_frame,
            config.video_root,
        )
        self.capture = cv2.VideoCapture(str(self.video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot_open_source_video={self.video_path}")
        self.frame_cache = CleanFrameCache()
        self.frame_position = self._resume_frame_position()
        self.active_actor_id = self._resume_actor_id()
        self.resume_saved_count = len(self.state.frame_annotations)
        self.resume_expected_count = len(self.frames) * len(
            self.config.editable_pig_ids
        )
        self.resumed_from_existing = self.resume_saved_count > 0
        self.draft = self.state.draft(
            self.active_actor_id,
            self.current_frame_index,
        )
        self.drag_intent: DragIntent | None = None
        self.drag_preview: BBox | None = None
        self.add_bbox_mode = False
        self.add_start: tuple[float, float] | None = None
        self.display_scale = 1.0
        self.display_offset = (0.0, 0.0)
        self.source_image_size = (0, 0)
        self.photo: ImageTk.PhotoImage | None = None

        self.root.title(WINDOW_TITLE)
        self.root.geometry("1500x900")
        self.root.minsize(1200, 720)
        self.status_var = tk.StringVar(value="Sẵn sàng")
        self.scope_var = tk.StringVar(value="")
        self.progress_var = tk.StringVar(value="")
        self.reviewed_id_var = tk.StringVar(value="")
        self.hidden_var = tk.StringVar(value="")
        self.behavior_var = tk.StringVar(value="")
        self._build_layout()
        self._load_draft_into_controls()
        self._render()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    @property
    def current_frame_index(self) -> int:
        return self.frames[self.frame_position]

    def _resume_frame_position(self) -> int:
        for frame_position, frame_index in enumerate(self.frames):
            for actor_id in self.config.editable_pig_ids:
                if (actor_id, frame_index) not in self.state.frame_annotations:
                    return frame_position
        return 0

    def _resume_actor_id(self) -> str:
        frame_index = self.frames[self.frame_position]
        for actor_id in self.config.editable_pig_ids:
            if (actor_id, frame_index) not in self.state.frame_annotations:
                return actor_id
        return self.config.editable_pig_ids[0]

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(0, weight=1)

        canvas_frame = ttk.Frame(self.root, padding=6)
        canvas_frame.grid(row=0, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            canvas_frame,
            background="#161616",
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Configure>", lambda _event: self._render())

        side = ttk.Frame(self.root, padding=10, width=400)
        side.grid(row=0, column=1, sticky="ns")
        side.grid_propagate(False)
        side.columnconfigure(0, weight=1)

        ttk.Label(
            side,
            text="Mini-CVAT V2",
            font=("Segoe UI", 16, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            side,
            textvariable=self.progress_var,
            wraplength=370,
        ).grid(row=1, column=0, sticky="ew", pady=(2, 8))

        navigation = ttk.Frame(side)
        navigation.grid(row=2, column=0, sticky="ew")
        navigation.columnconfigure((0, 1), weight=1)
        ttk.Button(
            navigation,
            text="← Frame trước",
            command=lambda: self.step_frame(-1),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(
            navigation,
            text="Frame sau →",
            command=lambda: self.step_frame(1),
        ).grid(row=0, column=1, sticky="ew", padx=(3, 0))

        self.actor_frame = ttk.LabelFrame(
            side,
            text="1. Chọn bbox/source scope",
            padding=6,
        )
        self.actor_frame.grid(row=3, column=0, sticky="ew", pady=(10, 5))
        self.actor_frame.columnconfigure(0, weight=1)

        ttk.Label(
            side,
            textvariable=self.scope_var,
            wraplength=370,
            foreground="#404040",
        ).grid(row=4, column=0, sticky="ew", pady=(2, 8))

        frame_box = ttk.LabelFrame(
            side,
            text="2. Thuộc tính frame/object",
            padding=8,
        )
        frame_box.grid(row=5, column=0, sticky="ew", pady=5)
        frame_box.columnconfigure(0, weight=1)
        ttk.Label(frame_box, text="Reviewed ID của frame hiện tại").grid(
            row=0,
            column=0,
            sticky="w",
        )
        self.reviewed_id_combo = ttk.Combobox(
            frame_box,
            textvariable=self.reviewed_id_var,
            values=self.config.editable_pig_ids,
            state="readonly",
        )
        self.reviewed_id_combo.grid(row=1, column=0, sticky="ew", pady=(2, 7))
        ttk.Label(frame_box, text="Hidden của object/frame").grid(
            row=2,
            column=0,
            sticky="w",
        )
        self.hidden_combo = ttk.Combobox(
            frame_box,
            textvariable=self.hidden_var,
            values=sorted(HIDDEN_VALUES),
            state="readonly",
        )
        self.hidden_combo.grid(row=3, column=0, sticky="ew", pady=(2, 7))
        ttk.Button(
            frame_box,
            text="LƯU FRAME: ID + bbox + Hidden",
            command=self.save_frame,
        ).grid(row=4, column=0, sticky="ew", pady=(3, 2))
        ttk.Button(
            frame_box,
            text="Hủy draft frame hiện tại (Esc)",
            command=self.restore_draft,
        ).grid(row=5, column=0, sticky="ew", pady=2)

        bbox_box = ttk.LabelFrame(
            side,
            text="3. Chỉnh bbox",
            padding=8,
        )
        bbox_box.grid(row=6, column=0, sticky="ew", pady=5)
        bbox_box.columnconfigure((0, 1), weight=1)
        ttk.Button(
            bbox_box,
            text="Thêm bbox bị mất",
            command=self.start_add_bbox,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(
            bbox_box,
            text="Khôi phục bbox nguồn",
            command=self.restore_source_bbox,
        ).grid(row=0, column=1, sticky="ew", padx=(3, 0))
        ttk.Label(
            bbox_box,
            text=(
                "Kéo trong bbox để di chuyển. Kéo ô vuông để resize. "
                "Thay đổi chỉ là draft cho đến khi bấm LƯU FRAME."
            ),
            wraplength=360,
            foreground="#404040",
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(7, 0))

        behavior_box = ttk.LabelFrame(
            side,
            text="4. Behavior của source scope — áp dụng toàn burst",
            padding=8,
        )
        behavior_box.grid(row=7, column=0, sticky="ew", pady=5)
        behavior_box.columnconfigure(0, weight=1)
        self.behavior_combo = ttk.Combobox(
            behavior_box,
            textvariable=self.behavior_var,
            values=sorted(CANONICAL_BEHAVIORS),
            state="readonly",
        )
        self.behavior_combo.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        ttk.Button(
            behavior_box,
            text="LƯU BEHAVIOR CHO TOÀN BURST",
            command=self.save_behavior,
        ).grid(row=1, column=0, sticky="ew")
        ttk.Label(
            behavior_box,
            text=(
                "Đổi behavior không đổi ID. Đổi ID frame không tự đổi "
                "behavior của bbox/source scope."
            ),
            wraplength=360,
            foreground="#404040",
        ).grid(row=2, column=0, sticky="ew", pady=(6, 0))

        ttk.Label(
            side,
            textvariable=self.status_var,
            wraplength=370,
            foreground="#154360",
        ).grid(row=8, column=0, sticky="ew", pady=(10, 0))

        self.apply_source_button = ttk.Button(
            side,
            text="ÁP DỤNG VÀO CSV + XML NGUỒN",
            command=self.apply_to_source_authority,
            state=(
                "normal"
                if self.config.apply_source_csvs
                and self.config.apply_source_xml is not None
                else "disabled"
            ),
        )
        self.apply_source_button.grid(
            row=9,
            column=0,
            sticky="ew",
            pady=(12, 0),
        )
        ttk.Button(
            side,
            text="XÓA MỌI THAY ĐỔI THỬ NGHIỆM",
            command=self.reset_all_session_changes,
        ).grid(row=10, column=0, sticky="ew", pady=(6, 0))
        self.root.bind("<Left>", lambda _event: self.step_frame(-1))
        self.root.bind("<Right>", lambda _event: self.step_frame(1))
        self.root.bind("<Escape>", lambda _event: self.restore_draft())
        self.root.bind("<Control-s>", lambda _event: self.save_frame())
        self.root.bind("a", lambda _event: self.start_add_bbox())

    def _source_frame_index(self, frame_index: int) -> int:
        return source_frame_index_for_review_frame(
            self.candidates_by_frame,
            frame_index,
        )

    def _decode_frame(self, frame_index: int) -> Image.Image:
        source_frame_index = self._source_frame_index(frame_index)
        cached = self.frame_cache.get(source_frame_index)
        if cached is not None:
            return cached
        frame = decode_exact_source_frame(
            self.capture,
            source_frame_index,
        )
        image = Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ).convert("RGB")
        self.frame_cache.put(source_frame_index, image)
        return image

    def _fit_image(self, image: Image.Image) -> Image.Image:
        self.canvas.update_idletasks()
        canvas_width = max(600, self.canvas.winfo_width())
        canvas_height = max(500, self.canvas.winfo_height())
        scale = min(
            canvas_width / image.width,
            canvas_height / image.height,
            1.0,
        )
        display_width = max(1, int(round(image.width * scale)))
        display_height = max(1, int(round(image.height * scale)))
        self.display_scale = scale
        self.display_offset = (
            (canvas_width - display_width) / 2.0,
            (canvas_height - display_height) / 2.0,
        )
        if (display_width, display_height) == image.size:
            return image
        return image.resize(
            (display_width, display_height),
            Image.Resampling.LANCZOS,
        )

    def _actor_color(self, actor_scope_id: str) -> str:
        index = self.config.editable_pig_ids.index(actor_scope_id)
        return ACTOR_COLORS[index % len(ACTOR_COLORS)]

    def _display_draft(self, actor_scope_id: str) -> FrameDraft:
        if actor_scope_id == self.active_actor_id:
            if self.drag_preview is not None:
                mode = self.state.corrected_bbox_mode(
                    actor_scope_id,
                    self.current_frame_index,
                    self.drag_preview,
                )
                return self.state.change_draft(
                    self.draft,
                    bbox=self.drag_preview,
                    bbox_mode=mode,
                )
            return self.draft
        return self.state.draft(
            actor_scope_id,
            self.current_frame_index,
        )

    def _draw_bbox(
        self,
        actor_scope_id: str,
        draft: FrameDraft,
    ) -> None:
        if draft.bbox is None:
            return
        canvas_bbox = source_bbox_to_canvas(
            draft.bbox,
            scale=self.display_scale,
            offset=self.display_offset,
        )
        color = self._actor_color(actor_scope_id)
        active = actor_scope_id == self.active_actor_id
        self.canvas.create_rectangle(
            *canvas_bbox,
            outline=color,
            width=4 if active else 2,
            tags=("bbox-object", f"actor:{actor_scope_id}"),
        )
        reviewed_id = (
            self.reviewed_id_var.get()
            if active
            else draft.reviewed_pig_id
        )
        label = (
            f"{actor_scope_id} → {reviewed_id} | "
            f"{self.state.reviewed_behavior(actor_scope_id)}"
        )
        self.canvas.create_text(
            canvas_bbox[0] + 4,
            max(14.0, canvas_bbox[1] - 8),
            text=label,
            fill=color,
            anchor="sw",
            font=("Segoe UI", 10, "bold"),
            tags=("bbox-object", f"actor:{actor_scope_id}"),
        )
        if not active:
            return
        for handle_x, handle_y in bbox_handle_points(canvas_bbox).values():
            self.canvas.create_rectangle(
                handle_x - 6,
                handle_y - 6,
                handle_x + 6,
                handle_y + 6,
                fill="#ffffff",
                outline=color,
                width=2,
                tags=("bbox-handle",),
            )

    def _draw_view_only_bbox(self, candidate: FrameCandidate) -> None:
        """Draw a non-editable candidate as a muted mini-CVAT overlay."""

        if candidate.bbox is None:
            return
        canvas_bbox = source_bbox_to_canvas(
            candidate.bbox,
            scale=self.display_scale,
            offset=self.display_offset,
        )
        self.canvas.create_rectangle(
            *canvas_bbox,
            outline="#9aa0a6",
            width=2,
            dash=(5, 3),
            tags=("bbox-view-only",),
        )
        self.canvas.create_text(
            canvas_bbox[0] + 4,
            max(14.0, canvas_bbox[1] - 8),
            text=f"{candidate.pig_id} · chỉ xem",
            fill="#d0d4d8",
            anchor="sw",
            font=("Segoe UI", 9),
            tags=("bbox-view-only",),
        )

    def _render(self) -> None:
        if not hasattr(self, "canvas"):
            return
        try:
            source = self._decode_frame(self.current_frame_index)
        except RuntimeError as exc:
            self.status_var.set(str(exc))
            return
        self.source_image_size = source.size
        fitted = self._fit_image(source)
        self.photo = ImageTk.PhotoImage(fitted)
        self.canvas.delete("all")
        self.canvas.create_image(
            self.display_offset[0],
            self.display_offset[1],
            image=self.photo,
            anchor="nw",
            tags=("source-frame",),
        )
        for candidate in self.candidates_by_frame.get(
            self.current_frame_index,
            (),
        ):
            if candidate.pig_id not in self.config.editable_pig_ids:
                self._draw_view_only_bbox(candidate)
        for actor_scope_id in self.config.editable_pig_ids:
            self._draw_bbox(
                actor_scope_id,
                self._display_draft(actor_scope_id),
            )
        if self.add_bbox_mode and self.add_start is not None:
            self.canvas.create_text(
                20,
                20,
                text="ĐANG THÊM BBOX — kéo trên frame",
                fill="#ff4dff",
                anchor="nw",
                font=("Segoe UI", 12, "bold"),
                tags=("mode-banner",),
            )
        self._refresh_actor_buttons()
        self._refresh_progress()

    def _refresh_actor_buttons(self) -> None:
        for child in self.actor_frame.winfo_children():
            child.destroy()
        for row, actor_scope_id in enumerate(self.config.editable_pig_ids):
            reviewed_id = self.state.effective_reviewed_id(
                actor_scope_id,
                self.current_frame_index,
            )
            saved = (
                actor_scope_id,
                self.current_frame_index,
            ) in self.state.frame_annotations
            marker = "✓" if saved else "·"
            text = (
                f"{marker} {actor_scope_id} → {reviewed_id} | "
                f"{self.state.reviewed_behavior(actor_scope_id)}"
            )
            ttk.Button(
                self.actor_frame,
                text=text,
                command=lambda value=actor_scope_id: self.select_actor(value),
            ).grid(row=row, column=0, sticky="ew", pady=2)

    def _refresh_progress(self) -> None:
        saved = self.state.saved_frame_count(self.active_actor_id)
        total = len(self.frames)
        draft_status = "CHƯA LƯU DRAFT" if self.draft.dirty else "đã đồng bộ"
        resume_status = (
            f"Sidecar progress: {len(self.state.frame_annotations)}/"
            f"{self.resume_expected_count} frame-object"
            if self.resumed_from_existing or self.state.frame_annotations
            else "Phiên mới: chưa có frame-object nào trong sidecar"
        )
        self.progress_var.set(
            f"Frame {self.current_frame_index} "
            f"({self.frame_position + 1}/{total}) · "
            f"{self.active_actor_id}: {saved}/{total} frame đã lưu · "
            f"{draft_status}\n{resume_status}"
        )

    def _load_draft_into_controls(self) -> None:
        self.reviewed_id_var.set(self.draft.reviewed_pig_id)
        self.hidden_var.set(self.draft.reviewed_hidden)
        self.behavior_var.set(
            self.state.reviewed_behavior(self.active_actor_id)
        )
        self.scope_var.set(
            f"Source scope: {self.active_actor_id} · "
            f"Behavior burst: "
            f"{self.state.reviewed_behavior(self.active_actor_id)}"
        )

    def _discard_dirty_if_confirmed(self) -> bool:
        if not self.draft.dirty:
            return True
        return messagebox.askyesno(
            "Draft chưa lưu",
            "Bỏ các thay đổi bbox/ID/Hidden chưa lưu của frame hiện tại?",
            parent=self.root,
        )

    def select_actor(self, actor_scope_id: str) -> None:
        if actor_scope_id == self.active_actor_id:
            return
        if not self._discard_dirty_if_confirmed():
            return
        self._cancel_drag()
        self.active_actor_id = actor_scope_id
        self.draft = self.state.draft(
            actor_scope_id,
            self.current_frame_index,
        )
        self._load_draft_into_controls()
        self.status_var.set(f"Đang sửa source scope {actor_scope_id}.")
        self._render()

    def step_frame(self, delta: int) -> None:
        if not self._discard_dirty_if_confirmed():
            return
        self._cancel_drag()
        self.frame_position = (
            self.frame_position + delta
        ) % len(self.frames)
        self.draft = self.state.draft(
            self.active_actor_id,
            self.current_frame_index,
        )
        self._load_draft_into_controls()
        self._render()

    def _persist(self) -> Path:
        return write_mini_cvat_sidecar(
            self.output_dir,
            reviewer=self.config.reviewer,
            source_type=self.cases[0].source_type,
            dataset_id=self.cases[0].dataset_id,
            video_key=self.cases[0].video_key,
            editable_actor_ids=self.config.editable_pig_ids,
            frame_indices=self.frames,
            actor_attributes=self.state.actor_attributes,
            frame_annotations=self.state.frame_annotations,
        )

    def save_frame(self) -> None:
        prior_annotations = dict(self.state.frame_annotations)
        draft = self.state.change_draft(
            self.draft,
            reviewed_pig_id=self.reviewed_id_var.get(),
            reviewed_hidden=self.hidden_var.get(),
        )
        try:
            result = self.state.save_frame(draft)
            self._persist()
        except (MiniCvatEditorError, OSError) as exc:
            self.state.frame_annotations = prior_annotations
            messagebox.showerror(
                "Không lưu được frame",
                str(exc),
                parent=self.root,
            )
            return
        self.draft = self.state.draft(
            self.active_actor_id,
            self.current_frame_index,
        )
        self._load_draft_into_controls()
        if result.swapped_actor_scope_id:
            self.status_var.set(
                f"Đã lưu frame; swap nguyên tử "
                f"{self.active_actor_id} → "
                f"{result.annotation.reviewed_pig_id}, "
                f"{result.swapped_actor_scope_id} → "
                f"{result.previous_reviewed_pig_id}."
            )
        else:
            self.status_var.set(
                f"Đã lưu frame {self.current_frame_index} cho "
                f"{self.active_actor_id}."
            )
        self._render()

    def save_behavior(self) -> None:
        prior_attributes = dict(self.state.actor_attributes)
        try:
            self.state.save_behavior(
                self.active_actor_id,
                self.behavior_var.get(),
            )
            self._persist()
        except (MiniCvatEditorError, OSError) as exc:
            self.state.actor_attributes = prior_attributes
            messagebox.showerror(
                "Không lưu được behavior burst",
                str(exc),
                parent=self.root,
            )
            return
        self.status_var.set(
            f"Đã lưu behavior {self.behavior_var.get()} cho toàn burst "
            f"source scope {self.active_actor_id}; ID không đổi."
        )
        self._load_draft_into_controls()
        self._render()

    def apply_to_source_authority(self) -> None:
        """Apply saved corrections to explicitly configured CSV/XML sources."""

        if self.draft.dirty:
            messagebox.showwarning(
                "Draft chưa lưu",
                (
                    "Hãy bấm LƯU FRAME hoặc Esc trước khi áp dụng. "
                    "Nút này không tự ghi draft chưa lưu."
                ),
                parent=self.root,
            )
            return
        if (
            not self.config.apply_source_csvs
            or self.config.apply_source_xml is None
        ):
            messagebox.showwarning(
                "Chưa cấu hình nguồn",
                (
                    "Cần truyền ít nhất một --apply-source-csv và đúng một "
                    "--apply-source-xml khi mở GUI."
                ),
                parent=self.root,
            )
            return
        target_lines = [
            *(str(path) for path in self.config.apply_source_csvs),
            str(self.config.apply_source_xml),
        ]
        confirmed = messagebox.askyesno(
            "Áp dụng vào dữ liệu nguồn?",
            (
                "Công cụ sẽ preflight, backup và cập nhật nguyên tử các file "
                "sau:\n\n"
                + "\n".join(target_lines)
                + "\n\nTiếp tục?"
            ),
            parent=self.root,
        )
        if not confirmed:
            return
        try:
            self._persist()
            self.status_var.set(
                "Đang preflight và áp dụng CSV/XML; không đóng GUI..."
            )
            self.root.update_idletasks()
            result = apply_identity_adjudication(
                sidecar_path=self.output_dir / MINI_CVAT_SIDECAR_NAME,
                csv_paths=self.config.apply_source_csvs,
                xml_path=self.config.apply_source_xml,
                audit_root=self.output_dir
                / "identity_source_apply_generations",
                group_id=self.config.apply_group_id,
            )
        except (
            IdentitySourceApplyError,
            OSError,
            csv.Error,
        ) as exc:
            self.status_var.set(f"Áp dụng thất bại: {exc}")
            messagebox.showerror(
                "Không thể áp dụng",
                str(exc),
                parent=self.root,
            )
            return
        self.status_var.set(
            "Đã áp dụng identity review vào "
            f"{result.changed_target_count} file; manifest: "
            f"{result.manifest_path}"
        )
        messagebox.showinfo(
            "Áp dụng hoàn tất",
            (
                f"Group: {result.group_id}\n"
                f"File thay đổi: {result.changed_target_count}\n"
                f"Manifest: {result.manifest_path}"
            ),
            parent=self.root,
        )

    def reset_all_session_changes(self) -> None:
        """Discard every trial annotation in this isolated mini-CVAT sidecar."""

        confirmed = messagebox.askyesno(
            "Xóa mọi thay đổi thử nghiệm?",
            (
                "Thao tác này xóa toàn bộ ID, bbox, Hidden và behavior đã lưu "
                "trong sidecar phiên này, rồi quay lại dữ liệu nguồn."
            ),
            parent=self.root,
        )
        if not confirmed:
            return
        previous_state = self.state
        previous_actor_id = self.active_actor_id
        previous_frame_position = self.frame_position
        self._cancel_drag()
        try:
            self.state = MiniCvatEditorState(
                editable_actor_ids=self.config.editable_pig_ids,
                frame_indices=self.frames,
                candidates_by_frame=self.candidates_by_frame,
                actor_attributes={},
                frame_annotations={},
            )
            self.active_actor_id = self.config.editable_pig_ids[0]
            self.frame_position = 0
            self.draft = self.state.draft(
                self.active_actor_id,
                self.current_frame_index,
            )
            self._persist()
        except (MiniCvatEditorError, OSError):
            self.state = previous_state
            self.active_actor_id = previous_actor_id
            self.frame_position = previous_frame_position
            self.draft = self.state.draft(
                self.active_actor_id,
                self.current_frame_index,
            )
            self._load_draft_into_controls()
            self.status_var.set(
                "Không thể reset phiên; sidecar cũ được giữ nguyên."
            )
            self._render()
            return
        self._load_draft_into_controls()
        self.status_var.set(
            "Đã loại bỏ mọi thay đổi phiên; đang ở frame nguồn đầu tiên."
        )
        self._render()

    def restore_draft(self) -> None:
        self._cancel_drag()
        self.draft = self.state.draft(
            self.active_actor_id,
            self.current_frame_index,
        )
        self._load_draft_into_controls()
        self.status_var.set("Đã hủy draft; sidecar không thay đổi.")
        self._render()

    def restore_source_bbox(self) -> None:
        candidate = self.state.source_candidate(
            self.active_actor_id,
            self.current_frame_index,
        )
        if candidate is None:
            messagebox.showwarning(
                "Không có bbox nguồn",
                "Source scope này không có bbox nguồn ở frame hiện tại.",
                parent=self.root,
            )
            return
        self.draft = self.state.change_draft(
            self.draft,
            bbox=candidate.bbox,
            bbox_mode=SOURCE_BBOX_MODE,
        )
        self.status_var.set(
            "Đã khôi phục bbox nguồn vào draft; bấm LƯU FRAME để ghi."
        )
        self._render()

    def start_add_bbox(self) -> None:
        self._cancel_drag()
        self.add_bbox_mode = True
        self.status_var.set(
            "Kéo trên frame để tạo bbox. Esc để hủy."
        )
        self._render()

    def _actor_at_canvas_point(
        self,
        point: tuple[float, float],
    ) -> str:
        matches = []
        for actor_scope_id in self.config.editable_pig_ids:
            draft = self._display_draft(actor_scope_id)
            if draft.bbox is None:
                continue
            canvas_bbox = source_bbox_to_canvas(
                draft.bbox,
                scale=self.display_scale,
                offset=self.display_offset,
            )
            if (
                canvas_bbox[0] <= point[0] <= canvas_bbox[2]
                and canvas_bbox[1] <= point[1] <= canvas_bbox[3]
            ):
                area = (
                    (canvas_bbox[2] - canvas_bbox[0])
                    * (canvas_bbox[3] - canvas_bbox[1])
                )
                matches.append((area, actor_scope_id))
        if matches:
            return min(matches)[1]
        candidate = smallest_candidate_at_point(
            self.candidates_by_frame[self.current_frame_index],
            point,
            scale=self.display_scale,
            offset=self.display_offset,
        )
        if (
            candidate is not None
            and candidate.pig_id in self.config.editable_pig_ids
        ):
            return candidate.pig_id
        return ""

    def _on_canvas_press(self, event: tk.Event[Any]) -> None:
        point = float(event.x), float(event.y)
        if self.add_bbox_mode:
            self.add_start = point
            self.drag_intent = DragIntent("draw", point, None)
            self.canvas.grab_set()
            return
        intent = begin_bbox_drag(
            point,
            self.draft.bbox,
            scale=self.display_scale,
            offset=self.display_offset,
        )
        if intent is not None:
            self.drag_intent = intent
            self.canvas.grab_set()
            return
        actor_scope_id = self._actor_at_canvas_point(point)
        if actor_scope_id:
            self.select_actor(actor_scope_id)

    def _on_canvas_drag(self, event: tk.Event[Any]) -> None:
        if self.drag_intent is None:
            return
        point = float(event.x), float(event.y)
        if self.drag_intent.mode == "draw":
            if self.add_start is None:
                return
            self.drag_preview = bbox_from_canvas_drag(
                self.add_start,
                point,
                scale=self.display_scale,
                offset=self.display_offset,
                source_size=self.source_image_size,
            )
        else:
            self.drag_preview = preview_bbox_drag(
                self.drag_intent,
                point,
                scale=self.display_scale,
                source_size=self.source_image_size,
            )
        self._render()

    def _on_canvas_release(self, event: tk.Event[Any]) -> None:
        if self.drag_intent is None:
            return
        point = float(event.x), float(event.y)
        if self.drag_intent.mode == "draw" and self.add_start is not None:
            final_bbox = bbox_from_canvas_drag(
                self.add_start,
                point,
                scale=self.display_scale,
                offset=self.display_offset,
                source_size=self.source_image_size,
            )
        else:
            final_bbox = preview_bbox_drag(
                self.drag_intent,
                point,
                scale=self.display_scale,
                source_size=self.source_image_size,
            )
        self._cancel_drag(clear_preview=False)
        if final_bbox is None:
            self.drag_preview = None
            self.status_var.set("Thao tác bbox quá nhỏ hoặc không hợp lệ.")
            self._render()
            return
        mode = self.state.corrected_bbox_mode(
            self.active_actor_id,
            self.current_frame_index,
            final_bbox,
        )
        self.draft = self.state.change_draft(
            self.draft,
            bbox=final_bbox,
            bbox_mode=mode,
        )
        self.drag_preview = None
        self.add_bbox_mode = False
        self.add_start = None
        self.status_var.set(
            "BBox đã đổi trong draft; bấm LƯU FRAME để ghi sidecar."
        )
        self._render()

    def _cancel_drag(self, *, clear_preview: bool = True) -> None:
        self.drag_intent = None
        self.add_start = None
        self.add_bbox_mode = False
        if clear_preview:
            self.drag_preview = None
        try:
            self.canvas.grab_release()
        except tk.TclError:
            pass

    def close(self) -> None:
        if self.draft.dirty and not messagebox.askyesno(
            "Draft chưa lưu",
            "Đóng GUI và bỏ draft chưa lưu?",
            parent=self.root,
        ):
            return
        self._cancel_drag()
        self.frame_cache.clear()
        self.capture.release()
        self.root.destroy()


def parse_args(argv: list[str] | None = None) -> GuiConfig:
    parser = argparse.ArgumentParser(
        description="Classification V2 mini-CVAT identity editor v2.",
    )
    parser.add_argument("--review-units-csv", type=Path, required=True)
    parser.add_argument("--frame-features-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument(
        "--review-item-id",
        action="append",
        dest="review_item_ids",
        required=True,
    )
    parser.add_argument(
        "--editable-pig-id",
        action="append",
        dest="editable_pig_ids",
        required=True,
    )
    parser.add_argument("--video-root", type=Path)
    parser.add_argument(
        "--apply-source-csv",
        action="append",
        dest="apply_source_csvs",
        default=[],
        help=(
            "Explicit raw/dense CSV to update from the saved sidecar. "
            "Repeat for multiple source CSVs."
        ),
    )
    parser.add_argument(
        "--apply-source-xml",
        type=Path,
        help="Explicit original CVAT image XML to update.",
    )
    parser.add_argument(
        "--apply-group-id",
        default="",
        help=(
            "Optional exact burst/group id. Otherwise infer it from a "
            "configured dense CSV and sidecar track IDs."
        ),
    )
    args = parser.parse_args(argv)
    review_item_ids = tuple(dict.fromkeys(args.review_item_ids))
    editable_pig_ids = tuple(dict.fromkeys(args.editable_pig_ids))
    if len(review_item_ids) != len(args.review_item_ids):
        parser.error("duplicate --review-item-id")
    if len(editable_pig_ids) != len(args.editable_pig_ids):
        parser.error("duplicate --editable-pig-id")
    if len(args.apply_source_csvs) != len(
        set(args.apply_source_csvs)
    ):
        parser.error("duplicate --apply-source-csv")
    if bool(args.apply_source_csvs) != bool(args.apply_source_xml):
        parser.error(
            "--apply-source-csv and --apply-source-xml must be provided "
            "together"
        )
    return GuiConfig(
        review_units_csv=args.review_units_csv,
        frame_features_csv=args.frame_features_csv,
        output_dir=args.output_dir,
        reviewer=args.reviewer.strip(),
        review_item_ids=review_item_ids,
        editable_pig_ids=editable_pig_ids,
        video_root=args.video_root,
        apply_source_csvs=tuple(
            dict.fromkeys(Path(value) for value in args.apply_source_csvs)
        ),
        apply_source_xml=args.apply_source_xml,
        apply_group_id=args.apply_group_id.strip(),
    )


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    root = tk.Tk()
    try:
        MiniCvatGuiV2(root, config)
    except (IdentityAdjudicationError, MiniCvatEditorError, RuntimeError) as exc:
        root.withdraw()
        messagebox.showerror("Không thể mở Mini-CVAT V2", str(exc))
        root.destroy()
        return 2
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
