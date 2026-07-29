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

from pig_behavior.classification_v2.review.source_specific_blinded_presentation_v2 import (
    local_context_identity,
    render_neutral_context_v2,
)

FINAL_PRESENTATION_VERSION = "classification_v2.final_behavior_review.v1"
CONTEXT_COLUMN = "final_context_frame_indices"
PLAYBACK_COLUMN = "final_playback_frame_indices"
PLAYBACK_INTERVALS_MS = {
    "2 fps": 500,
    "5 fps": 200,
    "10 fps": 100,
    "15 fps": 67,
    "30 fps": 33,
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
    draw.text(
        (10, 9),
        f"{role} · frame {frame_index}",
        fill="black",
    )
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
        media_note,
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


class FinalBehaviorReviewGui(BASE.ReviewUnitGui):
    """Final all-candidate review without risk-selection disclosure."""

    def __init__(self, config: Any) -> None:
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
        self.current = 0
        self.decisions = self._load_existing_decisions()
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
        self.playback_status = BASE.tk.StringVar(
            master=self.root,
            value="Playback chưa sẵn sàng",
        )
        self.root.title("Final Behavior Review · classification_v2")
        self.root.geometry("1500x940")
        self.root.minsize(1180, 760)
        self._build_layout()
        self._bind_shortcuts()
        self._offer_resume_position()
        self.show_current()
        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)
        self.root.mainloop()

    def _offer_resume_position(self) -> None:
        if not self.decisions:
            return
        if not messagebox.askyesno(
            "Resume final review",
            (
                f"Đã tải {len(self.decisions)} quyết định.\n"
                "Resume tại mục chưa review tiếp theo?"
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
        self.current = calculate_resume_index(
            self.units["review_unit_id"].astype(str).tolist(),
            set(self.decisions),
            backtrack,
        )

    def _build_layout(self) -> None:
        super()._build_layout()
        self.main_frame.rowconfigure(0, weight=1)
        playback = BASE.ttk.LabelFrame(
            self.main_frame,
            text="Playback ngữ cảnh · Space phát/dừng · V tổng quan",
        )
        playback.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=(0, 4),
            pady=(4, 0),
        )
        playback.columnconfigure(6, weight=1)
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
        BASE.ttk.Label(
            playback,
            textvariable=self.playback_status,
            anchor="center",
        ).grid(row=0, column=6, sticky="ew", padx=4)
        BASE.ttk.Checkbutton(
            playback,
            text="Lặp",
            variable=self.playback_repeat,
        ).grid(row=0, column=7, padx=4)

    def _on_keypress(self, event: Any) -> str | None:
        key = str(getattr(event, "keysym", "")).casefold()
        if BASE.shortcut_allowed_for_widget(getattr(event, "widget", None)):
            playback_actions = {
                "space": self.toggle_playback,
                "v": self.show_overview,
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

    def show_current(self) -> None:
        self.pause_playback()
        unit = self.current_unit()
        unit_id = str(unit["review_unit_id"])
        self._load_existing_decision(unit_id)
        self._prepare_current_media_rows(unit)
        frames = self._frame_rows_for_unit(unit)
        image, diagnostics = self._make_contact_sheet(unit, frames)
        self.current_diagnostics = diagnostics
        self._photo = BASE.ImageTk.PhotoImage(image)
        self._overview_photo = self._photo
        self.image_label.configure(image=self._photo)
        self._current_playback_frames = final_playback_frames(unit)
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
        self._update_playback_status()

        scope_suffix = (
            " · ALL 16 TARGET FRAMES"
            if str(unit.get("source_type", "")).strip()
            == "legacy_recovered"
            else " · T=TARGET · C=CONTEXT"
        )
        self.header.set(
            f"{self.current + 1}/{len(self.units)} · "
            f"đã review {self._completed_count()} · "
            f"FINAL REVIEW{scope_suffix}"
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
        count = len(self._current_playback_frames)
        if not count:
            self.playback_status.set("Không có frame playback")
            return
        frame = self._current_playback_frames[self.playback_position]
        role = playback_frame_role(self.current_unit(), frame)
        self.playback_status.set(
            f"{self.playback_position + 1}/{count} · f{frame} · {role}"
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
        role = playback_frame_role(self.current_unit(), frame_index)
        image = compose_playback_frame(
            image,
            frame_index=frame_index,
            role=role,
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
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
