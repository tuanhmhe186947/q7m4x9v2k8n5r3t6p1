"""Review every current Behavior candidate with source-specific context."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from tkinter import messagebox, simpledialog
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
GUI_SCRIPT_DIR = Path(__file__).resolve().parent
if str(GUI_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_SCRIPT_DIR))

from final_behavior_label_quality import (
    ERROR_PATTERNS,
    QUALITY_COLUMNS,
    QUALITY_SIDECAR_NAME,
    TECHNICAL_ERROR_PATTERN,
    build_quality_record,
    validate_quality_records,
)

from pig_behavior.classification_v2.review.source_specific_blinded_presentation_v2 import (
    local_context_identity,
    render_neutral_context_v2,
)

FINAL_PRESENTATION_VERSION = "classification_v2.final_behavior_review.v1"
FINAL_PRESENTATION_ORDER_VERSION = (
    "classification_v2.final_behavior_review_order.date_video_actor.v1"
)
CONTEXT_COLUMN = "final_context_frame_indices"
PLAYBACK_COLUMN = "final_playback_frame_indices"
PLAYBACK_SCOPE_TARGET = "TARGET only"
PLAYBACK_SCOPE_CONTEXT = "Full context"
PLAYBACK_SCOPE_VALUES = (
    PLAYBACK_SCOPE_TARGET,
    PLAYBACK_SCOPE_CONTEXT,
)
PLAYBACK_INTERVALS_MS = {
    "2 fps": 500,
    "5 fps": 200,
    "10 fps": 100,
    "15 fps": 67,
    "30 fps": 33,
}
MEDIA_DISPLAY_MAX_WIDTH = 760
MEDIA_DISPLAY_MAX_HEIGHT = 610
MEDIA_PANEL_RESERVED_WIDTH = 520
MEDIA_PANEL_RESERVED_HEIGHT = 250
ERROR_PATTERN_CHOICES = {
    1: (
        "ROI_PROXIMITY_ONLY_FALSE_POSITIVE",
        "Gán ROI chỉ vì đứng gần, nhưng hành vi thực tế là hành vi khác.",
    ),
    2: (
        "ROI_CONTACT_ABSENT_FALSE_POSITIVE",
        "Nhãn ROI sai rõ ràng vì không có tiếp xúc cần thiết.",
    ),
    3: (
        "INTERACTION_PHASE_OR_TEMPORAL_WINDOW_ERROR",
        "Fight/social bị gán sai pha hoặc sai cửa sổ thời gian.",
    ),
    4: (
        "OTHER_CLEAR_SOURCE_LABEL_ERROR",
        "Lỗi nhãn nguồn rõ ràng khác.",
    ),
}


def _load_base_module() -> Any:
    path = Path(__file__).with_name("review_temporal_unit_gui.py")
    spec = importlib.util.spec_from_file_location(
        "classification_v2_review_temporal_unit_gui",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base_module()


def final_display_frames(unit: pd.Series) -> list[int]:
    """Return chronological context plus immutable target frames."""

    targets = BASE.ReviewUnitGui._parse_frame_indices(
        unit.get("display_frame_indices", "")
    )
    if str(unit.get("source_type", "")).strip() == "legacy_recovered":
        return targets
    context = BASE.ReviewUnitGui._parse_frame_indices(
        unit.get(CONTEXT_COLUMN, "")
    )
    history = BASE.ReviewUnitGui._parse_frame_indices(
        unit.get("review_pig_history_display_frame_indices", "")
    )
    return sorted(dict.fromkeys([*context, *history, *targets]))


def final_playback_frames(unit: pd.Series) -> list[int]:
    """Return the immutable chronological playback sequence."""

    frames = BASE.ReviewUnitGui._parse_frame_indices(
        unit.get(PLAYBACK_COLUMN, "")
    )
    if frames:
        return sorted(dict.fromkeys(frames))
    return final_display_frames(unit)


def playback_frames_for_scope(
    unit: pd.Series,
    scope: str,
) -> list[int]:
    """Select target-only or full-context playback without changing scope."""

    if scope == PLAYBACK_SCOPE_CONTEXT:
        return final_playback_frames(unit)
    if scope != PLAYBACK_SCOPE_TARGET:
        raise ValueError(f"Unsupported playback scope: {scope}")
    targets = BASE.decision_scope_frames(unit)
    available = set(final_playback_frames(unit))
    selected = [frame for frame in targets if frame in available]
    return selected if selected else targets


def target_interval(unit: pd.Series) -> tuple[int, int, int] | None:
    """Return immutable decision-target bounds and frame count."""

    targets = BASE.decision_scope_frames(unit)
    if not targets:
        return None
    return targets[0], targets[-1], len(targets)


def format_playback_status(
    unit: pd.Series,
    frames: list[int],
    position: int,
    scope: str,
) -> str:
    """Keep decision and full-context bounds visible during playback."""

    if not frames:
        return "No playback frames"
    frame = frames[position]
    role = playback_frame_role(unit, frame)
    interval = target_interval(unit)
    target_text = "TARGET unavailable"
    if interval is not None:
        start, end, count = interval
        target_text = f"DECISION TARGET f{start}-f{end} ({count} frames)"
    full = final_playback_frames(unit)
    full_text = (
        f"FULL CONTEXT f{full[0]}-f{full[-1]}"
        if full
        else "FULL CONTEXT unavailable"
    )
    return (
        f"{scope} | {position + 1}/{len(frames)} | current f{frame} | "
        f"{role} | {target_text} | {full_text}"
    )


def playback_frame_role(unit: pd.Series, frame_index: int) -> str:
    """Keep target scope visible while playback adds source context."""

    if frame_index in BASE.decision_scope_frames(unit):
        return "TARGET"
    return "CONTEXT"


def compose_playback_frame(
    image: Image.Image,
    *,
    frame_index: int,
    role: str,
    target_start: int | None = None,
    target_end: int | None = None,
    playback_start: int | None = None,
    playback_end: int | None = None,
) -> Image.Image:
    """Add a clear target/context band without changing source pixels."""

    fitted = image.copy()
    fitted.thumbnail((920, 680), Image.Resampling.LANCZOS)
    band_color = "#fff2cc" if role == "TARGET" else "#d9e2f3"
    canvas = Image.new(
        "RGB",
        (fitted.width, fitted.height + 34),
        band_color,
    )
    canvas.paste(fitted, (0, 34))
    draw = ImageDraw.Draw(canvas)
    interval_text = (
        f" · DECISION TARGET f{target_start}-f{target_end}"
        if target_start is not None and target_end is not None
        else ""
    )
    draw.text(
        (10, 4),
        f"{role} · current f{frame_index}{interval_text}",
        fill="black",
    )
    if (
        fitted.width >= 80
        and playback_start is not None
        and playback_end is not None
        and target_start is not None
        and target_end is not None
        and playback_end > playback_start
    ):
        left = 10
        right = fitted.width - 10

        def timeline_x(value: int) -> int:
            ratio = (value - playback_start) / (
                playback_end - playback_start
            )
            bounded = max(0.0, min(1.0, ratio))
            return round(left + bounded * (right - left))

        draw.line((left, 27, right, 27), fill="#808080", width=4)
        draw.line(
            (timeline_x(target_start), 27, timeline_x(target_end), 27),
            fill="#f4b183",
            width=8,
        )
        current_x = timeline_x(frame_index)
        draw.line((current_x, 21, current_x, 33), fill="#c00000", width=3)
    return canvas


def format_final_summary(
    unit: pd.Series,
    diagnostics: list[str],
    matched_frame_count: int,
) -> str:
    """Show label-review instructions without selection or model hints."""

    source = str(unit.get("source_type", "")).strip()
    targets = BASE.decision_scope_frames(unit)
    context_count = len(final_display_frames(unit)) - len(targets)
    interval = target_interval(unit)
    target_scope_line = "DECISION TARGET unavailable"
    if interval is not None:
        start, end, count = interval
        target_scope_line = (
            f"DECISION TARGET: f{start}-f{end} ({count} frames). "
            "Chỉ phán quyết hành vi trong khoảng này."
        )
    if source == "legacy_recovered":
        media_note = (
            "Legacy: toàn bộ crop là actor; 16 frame đều là decision target. "
            "Không có full-frame hoặc neighbor context."
        )
    else:
        media_note = (
            "CVAT: actor được đánh dấu đỏ trong full frame. "
            "Heo khác có box xám; ROI nguồn được vẽ theo loại. "
            "T=decision target, C=context; context không đổi apply scope."
        )
    lines = [
        "FINAL BEHAVIOR REVIEW",
        (
            "Nhãn hiện tại (chưa xác nhận): "
            f"{str(unit.get('behavior_label', '')).strip()}"
        ),
        "Giữ nhãn nếu đúng; nếu sai hãy chọn trực tiếp hành vi quan sát được.",
        (
            "A = nhãn nguồn được hỗ trợ. Chọn hành vi khác = xác nhận "
            "lỗi nhãn nguồn rõ ràng và ghi loại lỗi."
        ),
        (
            "R chỉ tạm hoãn, không kết luận nhập nhằng. "
            "X chỉ dành cho lỗi media/kỹ thuật."
        ),
        media_note,
        target_scope_line,
        (
            f"Target frames: {len(targets)} · Context frames: "
            f"{context_count} · Matched media rows: {matched_frame_count}"
        ),
    ]
    if diagnostics:
        lines.append("Cảnh báo media: " + " · ".join(diagnostics[:4]))
    else:
        lines.append("Cảnh báo media: không")
    return "\n".join(lines)


def calculate_resume_index(
    review_ids: list[str],
    decided_ids: set[str],
    backtrack: int,
) -> int:
    """Return deterministic next-unreviewed index minus requested backtrack."""

    next_index = next(
        (
            index
            for index, review_id in enumerate(review_ids)
            if review_id not in decided_ids
        ),
        max(0, len(review_ids) - 1),
    )
    return max(0, next_index - max(0, backtrack))


def requested_review_index(
    review_ids: list[str],
    requested_review_unit_id: str | None,
) -> int | None:
    """Resolve an optional exact review-unit jump without fuzzy matching."""

    requested = str(requested_review_unit_id or "").strip()
    if not requested:
        return None
    matches = [
        index
        for index, review_id in enumerate(review_ids)
        if str(review_id).strip() == requested
    ]
    if len(matches) != 1:
        raise ValueError(
            "start review_unit_id must match exactly one unit; "
            f"matches={len(matches)} id={requested}"
        )
    return matches[0]


def _normalized_sort_column(
    units: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in units.columns:
        return pd.Series("", index=units.index, dtype="object")
    return (
        units[column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )


def order_final_review_units(units: pd.DataFrame) -> pd.DataFrame:
    """Group presentation by date, video, actor, then temporal position."""

    if units.empty:
        return units.copy()

    ordered = units.copy()
    date_key = _normalized_sort_column(ordered, "recording_date")
    video_key = _normalized_sort_column(ordered, "video_key")
    source_key = _normalized_sort_column(ordered, "source_type")
    dataset_key = _normalized_sort_column(ordered, "dataset_id")
    object_key = _normalized_sort_column(ordered, "object_track_key")
    track_key = _normalized_sort_column(ordered, "track_id")
    pig_key = _normalized_sort_column(ordered, "pig_id")
    review_key = _normalized_sort_column(ordered, "review_unit_id")

    actor_key = object_key.mask(object_key.eq(""), "track=" + track_key)
    actor_key = actor_key.mask(
        actor_key.eq("track="),
        "pig=" + pig_key,
    )

    ordered["_presentation_date_missing"] = date_key.eq("")
    ordered["_presentation_date"] = date_key
    ordered["_presentation_video_missing"] = video_key.eq("")
    ordered["_presentation_video"] = video_key
    ordered["_presentation_source"] = source_key
    ordered["_presentation_dataset"] = dataset_key
    ordered["_presentation_actor"] = actor_key
    ordered["_presentation_track_number"] = pd.to_numeric(
        ordered.get(
            "track_id",
            pd.Series("", index=ordered.index, dtype="object"),
        ),
        errors="coerce",
    ).fillna(float("inf"))
    ordered["_presentation_start"] = pd.to_numeric(
        ordered.get(
            "unit_start_frame",
            pd.Series("", index=ordered.index, dtype="object"),
        ),
        errors="coerce",
    ).fillna(float("inf"))
    ordered["_presentation_end"] = pd.to_numeric(
        ordered.get(
            "unit_end_frame",
            pd.Series("", index=ordered.index, dtype="object"),
        ),
        errors="coerce",
    ).fillna(float("inf"))
    ordered["_presentation_review_key"] = review_key
    ordered["_presentation_original_index"] = range(len(ordered))

    sort_columns = [
        "_presentation_date_missing",
        "_presentation_date",
        "_presentation_video_missing",
        "_presentation_video",
        "_presentation_source",
        "_presentation_dataset",
        "_presentation_track_number",
        "_presentation_actor",
        "_presentation_start",
        "_presentation_end",
        "_presentation_review_key",
        "_presentation_original_index",
    ]
    ordered = ordered.sort_values(sort_columns, kind="stable")
    return ordered.drop(columns=sort_columns).reset_index(drop=True)


def review_window_dimensions(
    screen_width: int,
    screen_height: int,
) -> tuple[int, int]:
    """Fit the review window inside the usable display."""

    width = min(1500, max(960, int(screen_width) - 32))
    height = min(940, max(700, int(screen_height) - 72))
    return width, height


def fit_media_for_display(
    image: Image.Image,
    *,
    max_width: int,
    max_height: int,
) -> Image.Image:
    """Bound media pixels so Tk layout cannot push controls over the image."""

    if max_width <= 0 or max_height <= 0:
        raise ValueError("media display bounds must be positive")
    if image.width <= max_width and image.height <= max_height:
        return image
    fitted = image.copy()
    fitted.thumbnail(
        (max_width, max_height),
        Image.Resampling.LANCZOS,
    )
    return fitted


class FinalBehaviorReviewGui(BASE.ReviewUnitGui):
    """Final all-candidate review without risk-selection disclosure."""

    def __init__(
        self,
        config: Any,
        *,
        start_review_unit_id: str | None = None,
    ) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.copy_contact_sheets:
            (self.config.output_dir / "contact_sheets").mkdir(
                parents=True,
                exist_ok=True,
            )
        self.units = self._load_units(config.review_units_csv)
        self.frames = BASE.load_gui_frame_features(config.frame_features_csv)
        self.frames["frame_index"] = pd.to_numeric(
            self.frames.get("frame_index"),
            errors="coerce",
        )
        if "relative_frame_index" in self.frames:
            self.frames["relative_frame_index"] = pd.to_numeric(
                self.frames["relative_frame_index"],
                errors="coerce",
        )
        try:
            requested_start = requested_review_index(
                self.units["review_unit_id"].astype(str).tolist(),
                start_review_unit_id,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        self.current = requested_start or 0
        self.requested_start_review_unit_id = str(
            start_review_unit_id or ""
        ).strip()
        self.decisions = self._load_existing_decisions()
        self.label_quality_records = self._load_label_quality_records()
        self._derive_supported_quality_records()
        self.current_error_pattern = ""
        self.skip_completed_on_next = False
        self.details_visible = False
        self.decision_dirty = False
        self.video_cache: dict[str, Any] = {}
        self.video_index = self._build_video_index(config.video_root)
        self.roi_overlays = self._load_roi_overlays(config.roi_coco_path)
        self.playback_running = False
        self.playback_after_id: str | None = None
        self.playback_position = 0
        self._current_playback_frames: list[int] = []
        self._current_playback_rows: dict[int, pd.Series] = {}
        self._current_scene_rows = self.frames.iloc[0:0].copy()
        self._current_actor_rows = self.frames.iloc[0:0].copy()
        self._overview_photo: Any = None

        self.root = BASE.tk.Tk()
        self.playback_repeat = BASE.tk.BooleanVar(
            master=self.root,
            value=False,
        )
        self.playback_speed = BASE.tk.StringVar(
            master=self.root,
            value="10 fps",
        )
        self.playback_scope = BASE.tk.StringVar(
            master=self.root,
            value=PLAYBACK_SCOPE_TARGET,
        )
        self.playback_status = BASE.tk.StringVar(
            master=self.root,
            value="Playback chưa sẵn sàng",
        )
        self.root.title("Final Behavior Review · classification_v2")
        window_width, window_height = review_window_dimensions(
            self.root.winfo_screenwidth(),
            self.root.winfo_screenheight(),
        )
        self.media_display_max_width = min(
            MEDIA_DISPLAY_MAX_WIDTH,
            max(480, window_width - MEDIA_PANEL_RESERVED_WIDTH),
        )
        self.media_display_max_height = min(
            MEDIA_DISPLAY_MAX_HEIGHT,
            max(360, window_height - MEDIA_PANEL_RESERVED_HEIGHT),
        )
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.minsize(
            min(1100, window_width),
            min(700, window_height),
        )
        self._build_layout()
        self._bind_shortcuts()
        if requested_start is None:
            self._offer_resume_position()
        self.show_current()
        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)
        self.root.mainloop()

    def _load_units(self, path: Path) -> pd.DataFrame:
        units = super()._load_units(path)
        return order_final_review_units(units)

    def _load_label_quality_records(self) -> dict[str, dict[str, str]]:
        path = self.config.output_dir / QUALITY_SIDECAR_NAME
        if not path.exists():
            return {}
        quality = pd.read_csv(path, dtype=str, keep_default_na=False)
        if quality.empty:
            return {}
        errors = validate_quality_records(
            quality,
            self.units,
            self.decisions,
        )
        if errors:
            raise SystemExit(
                "Existing label-quality sidecar violates contract: "
                + "; ".join(errors)
            )
        return {
            str(row["review_unit_id"]).strip(): {
                column: str(row.get(column, "")).strip()
                for column in QUALITY_COLUMNS
            }
            for row in quality.to_dict(orient="records")
        }

    def _derive_supported_quality_records(self) -> None:
        unit_rows = {
            str(row["review_unit_id"]): row
            for _, row in self.units.iterrows()
        }
        for review_id, decision in self.decisions.items():
            if review_id in self.label_quality_records:
                continue
            if str(decision.get("manual_review_decision", "")) != "accept":
                continue
            record = build_quality_record(unit_rows[review_id], decision)
            if record is not None:
                self.label_quality_records[review_id] = record

    def _quality_complete_ids(self) -> set[str]:
        return set(self.label_quality_records)

    def _current_quality_record(self) -> dict[str, str] | None:
        review_id = str(self.current_unit()["review_unit_id"])
        return self.label_quality_records.get(review_id)

    def _load_existing_decision(self, unit_id: str) -> None:
        super()._load_existing_decision(unit_id)
        record = self.label_quality_records.get(unit_id, {})
        pattern = str(record.get("error_pattern", ""))
        allowed_patterns = {*ERROR_PATTERNS, TECHNICAL_ERROR_PATTERN}
        self.current_error_pattern = (
            pattern if pattern in allowed_patterns else ""
        )

    def _ask_error_pattern(self) -> str | None:
        lines = [
            "Chọn nguyên nhân nhãn nguồn cũ sai rõ ràng:",
            "",
        ]
        lines.extend(
            f"{number}. {description}"
            for number, (_, description) in ERROR_PATTERN_CHOICES.items()
        )
        choice = simpledialog.askinteger(
            "Loại lỗi nhãn nguồn",
            "\n".join(lines),
            minvalue=1,
            maxvalue=len(ERROR_PATTERN_CHOICES),
            parent=self.root,
        )
        if choice is None:
            return None
        pattern = ERROR_PATTERN_CHOICES[choice][0]
        if pattern not in ERROR_PATTERNS:
            raise ValueError(f"unsupported error pattern={pattern}")
        return pattern

    def _confirm_technical_exclusion(self) -> bool:
        if self.current_error_pattern == TECHNICAL_ERROR_PATTERN:
            return True
        if not messagebox.askyesno(
            "Chỉ loại vì lỗi kỹ thuật",
            (
                "X chỉ dùng khi media hoặc lỗi trình bày khiến mục này "
                "không thể review. Đây có đúng là lỗi kỹ thuật không?"
            ),
            parent=self.root,
        ):
            return False
        if not self.note_var.get().strip():
            note = simpledialog.askstring(
                "Lý do lỗi kỹ thuật",
                "Mô tả ngắn lỗi media/kỹ thuật:",
                parent=self.root,
            )
            if note is None or not note.strip():
                return False
            self.note_var.set(note.strip())
        self.current_error_pattern = TECHNICAL_ERROR_PATTERN
        return True

    def _completed_count(self) -> int:
        return len(self.label_quality_records)

    def _update_decision_preview(self) -> None:
        super()._update_decision_preview()
        decision = self.decision_var.get()
        if decision == "accept":
            quality = "nhãn nguồn: SUPPORTED"
        elif decision == "corrected":
            pattern = self.current_error_pattern or "chưa chọn loại lỗi"
            quality = f"lỗi nhãn nguồn rõ ràng: {pattern}"
        elif decision == "exclude":
            quality = "chỉ loại vì lỗi media/kỹ thuật"
        else:
            quality = "tạm hoãn; chưa có kết luận chất lượng nhãn"
        self.status_var.set(f"{self.status_var.get()} · {quality}")

    def save_current(self) -> bool:
        unit = self.current_unit()
        review_id = str(unit["review_unit_id"])
        decision = self.decision_var.get()
        if decision == "corrected" and not self.current_error_pattern:
            pattern = self._ask_error_pattern()
            if pattern is None:
                return False
            self.current_error_pattern = pattern
        if decision == "exclude" and not self._confirm_technical_exclusion():
            return False

        previous = self.label_quality_records.get(review_id)
        try:
            quality_record = build_quality_record(
                unit,
                {
                    "manual_review_decision": decision,
                    "manual_corrected_behavior": self.corrected_var.get(),
                    "manual_label_strength": self.strength_var.get(),
                },
                error_pattern=self.current_error_pattern,
            )
        except ValueError as exc:
            messagebox.showerror(
                "Invalid label-quality decision",
                str(exc),
                parent=self.root,
            )
            return False

        if quality_record is None:
            self.label_quality_records.pop(review_id, None)
        else:
            self.label_quality_records[review_id] = quality_record
        if super().save_current():
            return True
        if previous is None:
            self.label_quality_records.pop(review_id, None)
        else:
            self.label_quality_records[review_id] = previous
        return False

    def write_decisions(self, show_message: bool = True) -> None:
        super().write_decisions(show_message=False)
        unit_order = {
            str(unit_id): index
            for index, unit_id in enumerate(
                self.units["review_unit_id"].astype(str)
            )
        }
        rows = sorted(
            self.label_quality_records.values(),
            key=lambda row: unit_order[str(row["review_unit_id"])],
        )
        path = self.config.output_dir / QUALITY_SIDECAR_NAME
        BASE._write_csv_atomic(path, QUALITY_COLUMNS, rows)
        if show_message:
            messagebox.showinfo(
                "Saved",
                (
                    f"Wrote {len(self.decisions)} decisions and "
                    f"{len(rows)} label-quality records\n{path}"
                ),
                parent=self.root,
            )

    def accept_current_next(self) -> None:
        self.current_error_pattern = ""
        super().accept_current_next()

    def correct_behavior_next(self, behavior: str) -> None:
        original = str(self.current_unit().get("behavior_label", "")).strip()
        if behavior == original:
            self.accept_current_next()
            return
        pattern = self._ask_error_pattern()
        if pattern is None:
            return
        self.current_error_pattern = pattern
        super().correct_behavior_next(behavior)

    def exclude_next(self) -> None:
        if self._confirm_technical_exclusion():
            super().exclude_next()

    def defer_next(self) -> None:
        self.current_error_pattern = ""
        super().defer_next()

    def next_item(self) -> None:
        review_id = str(self.current_unit()["review_unit_id"])
        decision = self.decision_var.get()
        needs_save = (
            self.decision_dirty
            or (
                decision in {"corrected", "exclude"}
                and review_id not in self._quality_complete_ids()
            )
        )
        if needs_save and not self.save_current():
            return
        if not self.skip_completed_on_next:
            if self.current < len(self.units) - 1:
                self.current += 1
                self.show_current()
            return
        for index in range(self.current + 1, len(self.units)):
            candidate_id = str(self.units.iloc[index]["review_unit_id"])
            if candidate_id not in self._quality_complete_ids():
                self.current = index
                self.show_current()
                return
        self.status_var.set(
            "Không còn mục chưa hoàn tất ở phía sau."
        )

    def prev_item(self) -> None:
        if self.decision_dirty and not self.save_current():
            return
        if self.current > 0:
            self.current -= 1
            self.show_current()

    def _offer_resume_position(self) -> None:
        if not self.decisions:
            return
        missing_quality = sorted(
            review_id
            for review_id, decision in self.decisions.items()
            if str(decision.get("manual_review_decision", ""))
            in {"corrected", "exclude"}
            and review_id not in self.label_quality_records
        )
        attribution_note = ""
        if missing_quality:
            attribution_note = (
                f"\nCó {len(missing_quality)} mục sửa/loại cũ cần bổ sung "
                "loại lỗi; các mục giữ nguyên không phải review lại."
            )
        if not messagebox.askyesno(
            "Resume final review",
            (
                f"Đã tải {len(self.decisions)} quyết định.\n"
                "Resume tại mục chưa hoàn tất tiếp theo?"
                f"{attribution_note}"
            ),
            parent=self.root,
        ):
            return
        backtrack = simpledialog.askinteger(
            "Quay lại để kiểm tra",
            "Quay lại bao nhiêu mục trước vị trí resume?",
            initialvalue=0,
            minvalue=0,
            maxvalue=max(0, len(self.units) - 1),
            parent=self.root,
        )
        if backtrack is None:
            backtrack = 0
        self.skip_completed_on_next = backtrack == 0
        self.current = calculate_resume_index(
            self.units["review_unit_id"].astype(str).tolist(),
            self._quality_complete_ids(),
            backtrack,
        )

    def _build_layout(self) -> None:
        super()._build_layout()
        self.summary_text.configure(width=44, height=6)
        self.info_text.configure(width=44)
        self.image_label.configure(
            anchor="center",
            borderwidth=1,
            relief="solid",
        )
        self.main_frame.rowconfigure(0, weight=1)
        playback = BASE.ttk.LabelFrame(
            self.main_frame,
            text="Playback · mặc định chỉ DECISION TARGET · V tổng quan",
        )
        playback.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=(0, 4),
            pady=(4, 0),
        )
        playback.columnconfigure(8, weight=1)
        BASE.ttk.Button(
            playback,
            text="V  Tổng quan",
            command=self.show_overview,
        ).grid(row=0, column=0, padx=3, pady=3)
        BASE.ttk.Button(
            playback,
            text="⏮  Đầu clip",
            command=self.restart_playback,
        ).grid(row=0, column=1, padx=3, pady=3)
        BASE.ttk.Button(
            playback,
            text="◀  Frame",
            command=lambda: self.step_playback(-1),
        ).grid(row=0, column=2, padx=3, pady=3)
        self.play_button = BASE.ttk.Button(
            playback,
            text="▶  Phát / Dừng",
            command=self.toggle_playback,
        )
        self.play_button.grid(row=0, column=3, padx=3, pady=3)
        BASE.ttk.Button(
            playback,
            text="Frame  ▶",
            command=lambda: self.step_playback(1),
        ).grid(row=0, column=4, padx=3, pady=3)
        speed = BASE.ttk.Combobox(
            playback,
            textvariable=self.playback_speed,
            values=list(PLAYBACK_INTERVALS_MS),
            width=7,
            state="readonly",
        )
        speed.grid(row=0, column=5, padx=3, pady=3)
        BASE.ttk.Label(playback, text="Phạm vi:").grid(
            row=0,
            column=6,
            padx=(6, 2),
        )
        scope = BASE.ttk.Combobox(
            playback,
            textvariable=self.playback_scope,
            values=PLAYBACK_SCOPE_VALUES,
            width=13,
            state="readonly",
        )
        scope.grid(row=0, column=7, padx=3, pady=3)
        scope.bind("<<ComboboxSelected>>", self._on_playback_scope_changed)
        BASE.ttk.Label(
            playback,
            textvariable=self.playback_status,
            anchor="center",
        ).grid(row=0, column=8, sticky="ew", padx=4)
        BASE.ttk.Checkbutton(
            playback,
            text="Lặp",
            variable=self.playback_repeat,
        ).grid(row=0, column=9, padx=4)

    def _on_keypress(self, event: Any) -> str | None:
        key = str(getattr(event, "keysym", "")).casefold()
        if BASE.shortcut_allowed_for_widget(getattr(event, "widget", None)):
            playback_actions = {
                "space": self.toggle_playback,
                "v": self.show_overview,
                "b": self.toggle_playback_scope,
                "home": self.restart_playback,
                "comma": lambda: self.step_playback(-1),
                "period": lambda: self.step_playback(1),
                "l": self.toggle_repeat,
            }
            action = playback_actions.get(key)
            if action is not None:
                action()
                return "break"
        return super()._on_keypress(event)

    def _all_display_frames(self, unit: pd.Series) -> list[int]:
        return final_display_frames(unit)

    def _display_frame_role(self, unit: pd.Series, frame_index: int) -> str:
        if frame_index in BASE.decision_scope_frames(unit):
            return "T"
        return "C"

    def _frame_rows_for_indices(
        self,
        unit: pd.Series,
        wanted: list[int],
    ) -> pd.DataFrame:
        del unit
        df = self._current_actor_rows
        return df.loc[df["frame_index"].isin(wanted)].sort_values(
            "frame_index"
        ).copy()

    def _prepare_current_media_rows(self, unit: pd.Series) -> None:
        df = self.frames
        mask = (
            df["source_type"].astype(str).eq(str(unit["source_type"]))
            & df["dataset_id"].astype(str).eq(str(unit["dataset_id"]))
            & df["video_key"].astype(str).eq(str(unit["video_key"]))
        )
        self._current_scene_rows = df.loc[mask].copy()
        actor_mask = self._current_scene_rows["pig_id"].astype(str).eq(
            str(unit["pig_id"])
        )
        if "object_track_key" in df and pd.notna(unit.get("object_track_key")):
            actor_mask &= self._current_scene_rows[
                "object_track_key"
            ].astype(str).eq(
                str(unit.get("object_track_key"))
            )
        elif "track_id" in df and pd.notna(unit.get("track_id")):
            actor_mask &= self._current_scene_rows["track_id"].astype(str).eq(
                str(unit.get("track_id"))
            )
        self._current_actor_rows = self._current_scene_rows.loc[
            actor_mask
        ].copy()

    def _frame_rows_for_unit(self, unit: pd.Series) -> pd.DataFrame:
        return self._frame_rows_for_indices(unit, self._all_display_frames(unit))

    def _scene_rows(self, actor: pd.Series) -> pd.DataFrame:
        frame_index = pd.to_numeric(actor.get("frame_index"), errors="coerce")
        return self._current_scene_rows[
            self._current_scene_rows["frame_index"].eq(frame_index)
        ].copy()

    def _cvat_full_frame(self, actor: pd.Series) -> Image.Image | None:
        if BASE.cv2 is None:
            return None
        video_path = self._resolve_video_path(actor)
        if video_path is None:
            return None
        capture = self.video_cache.get(str(video_path))
        if capture is None:
            capture = BASE.cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                return None
            self.video_cache[str(video_path)] = capture
        frame_index = int(actor["frame_index"])
        current_position = int(capture.get(BASE.cv2.CAP_PROP_POS_FRAMES))
        if current_position != frame_index:
            capture.set(BASE.cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            return None
        frame = BASE.cv2.cvtColor(frame, BASE.cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame).convert("RGB")

    def _image_for_row(
        self,
        unit: pd.Series,
        row: pd.Series,
    ) -> tuple[Image.Image, str]:
        if str(unit.get("source_type", "")).strip() == "legacy_recovered":
            return super()._image_for_row(unit, row)
        full_frame = self._cvat_full_frame(row)
        if full_frame is None:
            return self._placeholder("NO CVAT VIDEO FRAME"), "missing_video"
        image = render_neutral_context_v2(
            full_frame,
            self._scene_rows(row),
            actor_identity=local_context_identity(row),
        )
        draw = ImageDraw.Draw(image)
        self._draw_roi_overlays(
            draw,
            row,
            image.width,
            image.height,
            0,
            0,
        )
        return image, ""

    def _format_info(
        self,
        unit: pd.Series,
        diagnostics: list[str],
        frames: pd.DataFrame,
    ) -> str:
        targets = BASE.decision_scope_frames(unit)
        context = [
            frame
            for frame in final_display_frames(unit)
            if frame not in set(targets)
        ]
        return "\n".join(
            [
                f"presentation_version: {FINAL_PRESENTATION_VERSION}",
                (
                    "presentation_order_version: "
                    f"{FINAL_PRESENTATION_ORDER_VERSION}"
                ),
                f"source_type: {unit.get('source_type', '')}",
                f"target_frame_count: {len(targets)}",
                f"context_frame_count: {len(context)}",
                f"matched_media_rows: {len(frames)}",
                (
                    "actor_identity: entire crop"
                    if str(unit.get("source_type", "")).strip()
                    == "legacy_recovered"
                    else "actor_identity: red bbox in full frame"
                ),
                (
                    "media_warnings: " + " | ".join(diagnostics[:6])
                    if diagnostics
                    else "media_warnings: none"
                ),
            ]
        )

    def _configure_current_playback(
        self,
        unit: pd.Series,
        *,
        render: bool = False,
    ) -> None:
        self._current_playback_frames = playback_frames_for_scope(
            unit,
            self.playback_scope.get(),
        )
        playback_rows = self._frame_rows_for_indices(
            unit,
            self._current_playback_frames,
        )
        self._current_playback_rows = {
            int(row["frame_index"]): row
            for _, row in playback_rows.iterrows()
            if pd.notna(row.get("frame_index"))
        }
        self.playback_position = 0
        if render:
            self._render_playback_position()
        else:
            self._update_playback_status()

    def _on_playback_scope_changed(self, _event: Any = None) -> None:
        self.pause_playback()
        self._configure_current_playback(self.current_unit(), render=True)

    def toggle_playback_scope(self) -> None:
        scope = self.playback_scope.get()
        self.playback_scope.set(
            PLAYBACK_SCOPE_CONTEXT
            if scope == PLAYBACK_SCOPE_TARGET
            else PLAYBACK_SCOPE_TARGET
        )
        self._on_playback_scope_changed()

    def show_current(self) -> None:
        self.pause_playback()
        unit = self.current_unit()
        unit_id = str(unit["review_unit_id"])
        self._load_existing_decision(unit_id)
        self._prepare_current_media_rows(unit)
        frames = self._frame_rows_for_unit(unit)
        image, diagnostics = self._make_contact_sheet(unit, frames)
        self.current_diagnostics = diagnostics
        image = fit_media_for_display(
            image,
            max_width=self.media_display_max_width,
            max_height=self.media_display_max_height,
        )
        self._photo = BASE.ImageTk.PhotoImage(image)
        self._overview_photo = self._photo
        self.image_label.configure(image=self._photo)
        self._configure_current_playback(unit)

        interval = target_interval(unit)
        target_suffix = (
            f" · DECISION f{interval[0]}-f{interval[1]}"
            if interval is not None
            else " · DECISION TARGET unavailable"
        )
        scope_suffix = (
            " · ALL 16 TARGET FRAMES"
            if str(unit.get("source_type", "")).strip()
            == "legacy_recovered"
            else " · T=TARGET · C=CONTEXT"
        )
        self.header.set(
            f"{self.current + 1}/{len(self.units)} · "
            f"đã review {self._completed_count()} · "
            f"FINAL REVIEW{scope_suffix}{target_suffix} · "
            f"ngày {unit.get('recording_date', '')} · "
            f"video {unit.get('video_key', '')} · "
            f"ID {unit.get('pig_id', '')}/track {unit.get('track_id', '')}"
        )
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert(
            "1.0",
            format_final_summary(unit, diagnostics, len(frames)),
        )
        self.summary_text.configure(state="disabled")
        self.info_text.delete("1.0", "end")
        self.info_text.insert(
            "1.0",
            self._format_info(unit, diagnostics, frames),
        )
        self._update_decision_preview()

    def _update_playback_status(self) -> None:
        self.playback_status.set(
            format_playback_status(
                self.current_unit(),
                self._current_playback_frames,
                self.playback_position,
                self.playback_scope.get(),
            )
        )

    def _render_playback_position(self) -> None:
        if not self._current_playback_frames:
            return
        frame_index = self._current_playback_frames[self.playback_position]
        row = self._current_playback_rows.get(frame_index)
        if row is None:
            image = self._placeholder(f"NO ACTOR ROW\nf{frame_index}")
        else:
            image, _ = self._image_for_row(self.current_unit(), row)
        unit = self.current_unit()
        role = playback_frame_role(unit, frame_index)
        interval = target_interval(unit)
        full_frames = final_playback_frames(unit)
        image = compose_playback_frame(
            image,
            frame_index=frame_index,
            role=role,
            target_start=interval[0] if interval is not None else None,
            target_end=interval[1] if interval is not None else None,
            playback_start=full_frames[0] if full_frames else None,
            playback_end=full_frames[-1] if full_frames else None,
        )
        image = fit_media_for_display(
            image,
            max_width=self.media_display_max_width,
            max_height=self.media_display_max_height,
        )
        self._photo = BASE.ImageTk.PhotoImage(image)
        self.image_label.configure(image=self._photo)
        self._update_playback_status()

    def show_overview(self) -> None:
        self.pause_playback()
        if self._overview_photo is not None:
            self._photo = self._overview_photo
            self.image_label.configure(image=self._photo)

    def toggle_playback(self) -> None:
        if self.playback_running:
            self.pause_playback()
            return
        if not self._current_playback_frames:
            return
        if self.playback_position >= len(self._current_playback_frames) - 1:
            self.playback_position = 0
        self.playback_running = True
        self.play_button.configure(text="⏸  Dừng")
        self._render_playback_position()
        self._schedule_next_playback_frame()

    def pause_playback(self) -> None:
        self.playback_running = False
        if self.playback_after_id is not None and hasattr(self, "root"):
            self.root.after_cancel(self.playback_after_id)
        self.playback_after_id = None
        if hasattr(self, "play_button"):
            self.play_button.configure(text="▶  Phát / Dừng")

    def _schedule_next_playback_frame(self) -> None:
        interval = PLAYBACK_INTERVALS_MS.get(
            self.playback_speed.get(),
            PLAYBACK_INTERVALS_MS["10 fps"],
        )
        self.playback_after_id = self.root.after(
            interval,
            self._advance_playback,
        )

    def _advance_playback(self) -> None:
        self.playback_after_id = None
        if not self.playback_running:
            return
        next_position = self.playback_position + 1
        if next_position >= len(self._current_playback_frames):
            if self.playback_repeat.get():
                next_position = 0
            else:
                self.pause_playback()
                return
        self.playback_position = next_position
        self._render_playback_position()
        self._schedule_next_playback_frame()

    def step_playback(self, offset: int) -> None:
        self.pause_playback()
        if not self._current_playback_frames:
            return
        last = len(self._current_playback_frames) - 1
        self.playback_position = min(
            last,
            max(0, self.playback_position + offset),
        )
        self._render_playback_position()

    def restart_playback(self) -> None:
        self.pause_playback()
        if not self._current_playback_frames:
            return
        self.playback_position = 0
        self._render_playback_position()

    def toggle_repeat(self) -> None:
        self.playback_repeat.set(not self.playback_repeat.get())

    def on_quit(self) -> None:
        self.pause_playback()
        super().on_quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-units-csv", type=Path, required=True)
    parser.add_argument("--frame-features-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, default=Path("data/videos"))
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--roi-coco-json", type=Path)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--padding", type=float, default=0.8)
    parser.add_argument("--copy-contact-sheets", action="store_true")
    parser.add_argument(
        "--start-review-unit-id",
        default="",
        help=(
            "Open at one exact review_unit_id and skip the resume dialog. "
            "The review population and decision files remain unchanged."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    FinalBehaviorReviewGui(
        BASE.GuiConfig(
            review_units_csv=args.review_units_csv,
            frame_features_csv=args.frame_features_csv,
            output_dir=args.output_dir,
            video_root=args.video_root,
            raw_root=args.raw_root,
            roi_coco_path=args.roi_coco_json,
            max_items=args.max_items if args.max_items > 0 else None,
            padding=args.padding,
            copy_contact_sheets=args.copy_contact_sheets,
        ),
        start_review_unit_id=args.start_review_unit_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
