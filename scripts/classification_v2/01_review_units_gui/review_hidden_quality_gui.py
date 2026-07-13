"""Manifest-driven Hidden review with full-frame occlusion context.

This GUI never modifies input CSV/XML files. It writes only a resumable decision
CSV consumed by classification_v2_apply_hidden_review_decisions.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageTk

from pig_behavior.classification_v2.datasets.image_context_index import (
    resolve_legacy_crop,
)
from pig_behavior.classification_v2.review.hidden_review_builder import (
    hidden_decision_semantic_error,
)

DECISION_FILENAME = "hidden_review_decisions.csv"
REQUIRED_MANIFEST_COLUMNS = {
    "hidden_review_item_id",
    "hidden_before_review",
    "hidden_review_cohort",
    "source_type",
    "video_key",
    "frame_uid",
    "frame_index",
    "pig_id",
    "behavior",
    "x1",
    "y1",
    "x2",
    "y2",
}
DECISION_COLUMNS = [
    "hidden_review_item_id",
    "hidden_before_review",
    "hidden_after_review",
    "hidden_review_status",
    "hidden_review_confidence",
    "hidden_review_reason",
    "hidden_reviewer",
    "hidden_reviewed_at",
    "hidden_review_cohort",
    "source_type",
    "dataset_id",
    "video_key",
    "frame_uid",
    "frame_index",
    "track_id",
    "pig_id",
    "behavior",
    "hidden_false_negative_risk_score",
    "hidden_false_negative_risk_reasons",
]
REASON_OPTIONS = [
    "clearly_visible",
    "occluded_by_pig",
    "occluded_by_scene",
    "partial_bbox_or_frame_edge",
    "tracking_box_weak",
    "temporal_evidence",
    "ambiguous",
    "other",
]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--frame-features-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--video-root", type=Path, action="append", default=[])
    parser.add_argument("--crop-root", type=Path, action="append", default=[])
    parser.add_argument("--padding", type=float, default=0.12)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--include-reviewed", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate manifest and media resolution without opening Tk.",
    )
    parser.add_argument(
        "--validation-audit-json",
        type=Path,
        default=None,
        help="Optional JSON evidence path for the media-resolution gate.",
    )
    return parser.parse_args()


def bounded_window_size(
    screen_width: int,
    screen_height: int,
) -> tuple[int, int]:
    """Fit the review window inside the usable screen with taskbar margins."""
    if screen_width <= 0 or screen_height <= 0:
        raise ValueError("Screen dimensions must be positive")
    horizontal_margin = 40 if screen_width > 80 else 0
    vertical_margin = 80 if screen_height > 160 else 0
    width = min(1320, screen_width - horizontal_margin)
    height = min(920, screen_height - vertical_margin)
    return max(1, width), max(1, height)


class VideoReader:
    """Reuse video handles and a small frame cache during review."""

    def __init__(self) -> None:
        self.captures: dict[Path, cv2.VideoCapture] = {}
        self.cache: dict[tuple[Path, int], Any] = {}
        self.order: list[tuple[Path, int]] = []
        self.max_cache = 24

    def read(self, path: Path, frame_index: int):
        key = (path, frame_index)
        if key in self.cache:
            return self.cache[key].copy()
        capture = self.captures.get(path)
        if capture is None:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                return None
            self.captures[path] = capture
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            return None
        self.cache[key] = frame.copy()
        self.order.append(key)
        while len(self.order) > self.max_cache:
            self.cache.pop(self.order.pop(0), None)
        return frame

    def close(self) -> None:
        for capture in self.captures.values():
            capture.release()
        self.captures.clear()
        self.cache.clear()


class HiddenQualityReviewApp:
    """Review selected frame/object visibility without mutating annotations."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        items: pd.DataFrame,
        frame_features: pd.DataFrame,
        decisions: dict[str, dict[str, str]],
        decision_path: Path,
        reviewer: str,
        video_index: dict[str, Path],
        crop_roots: list[Path],
        padding: float,
    ) -> None:
        self.root = root
        self.items = items.reset_index(drop=True)
        self.frame_features = frame_features
        group_columns = ["source_type", "dataset_id", "video_key", "frame_uid"]
        self.frame_groups = {
            tuple(str(value) for value in key): group.copy()
            for key, group in frame_features.groupby(
                group_columns,
                dropna=False,
            )
        }
        self.decisions = decisions
        self.completed_ids = completed_decision_ids(decisions)
        self.decision_path = decision_path
        self.reviewer = reviewer
        self.video_index = video_index
        self.crop_roots = crop_roots
        self.padding = padding
        self.reader = VideoReader()
        self.index = 0
        self.undo_stack: list[tuple[str, dict[str, str] | None]] = []
        self.photo: ImageTk.PhotoImage | None = None
        window_width, window_height = bounded_window_size(
            self.root.winfo_screenwidth(),
            self.root.winfo_screenheight(),
        )
        offset_x = max(0, (self.root.winfo_screenwidth() - window_width) // 2)
        offset_y = max(0, (self.root.winfo_screenheight() - window_height) // 2)
        self.image_max_size = (
            max(1, window_width - 40),
            max(1, window_height - 260),
        )
        self.root.geometry(
            f"{window_width}x{window_height}+{offset_x}+{offset_y}"
        )

        self.info_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.confidence_var = tk.StringVar(value="high")
        self.reason_var = tk.StringVar(value="")
        self.note_var = tk.StringVar()
        self._build_ui()
        self._bind_keys()
        self._show_current()

    def _build_ui(self) -> None:
        self.root.title("Classification V2 - Two-sided Hidden Quality Review")
        tk.Label(
            self.root,
            textvariable=self.info_var,
            justify=tk.LEFT,
            anchor="w",
            font=("Consolas", 10),
        ).pack(fill=tk.X, padx=10, pady=6)

        tk.Label(
            self.root,
            textvariable=self.status_var,
            justify=tk.LEFT,
            anchor="w",
            font=("Consolas", 10),
        ).pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        controls = tk.Frame(self.root)
        controls.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=6)
        tk.Button(
            controls,
            text="Hidden = Yes [H]",
            command=lambda: self._save("Yes", "reviewed"),
            bg="#f2c96d",
            height=2,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        tk.Button(
            controls,
            text="Visible = No [V]",
            command=lambda: self._save("No", "reviewed"),
            bg="#96d59a",
            height=2,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        tk.Button(
            controls,
            text="Unclear [U]",
            command=lambda: self._save("", "unclear"),
            bg="#e0e0e0",
            height=2,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        tk.Button(
            controls,
            text="Skip pending [S]",
            command=self._skip,
            height=2,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        tk.Button(
            controls,
            text="Undo [Ctrl+Z]",
            command=self._undo,
            height=2,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        tk.Button(
            controls,
            text="Save and exit [Ctrl+S]",
            command=self._save_and_exit,
            height=2,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)

        metadata = tk.Frame(self.root)
        metadata.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=4)
        tk.Label(metadata, text="Confidence").pack(side=tk.LEFT)
        ttk.Combobox(
            metadata,
            textvariable=self.confidence_var,
            values=["high", "medium", "low"],
            state="readonly",
            width=10,
        ).pack(side=tk.LEFT, padx=5)
        tk.Label(metadata, text="Reason").pack(side=tk.LEFT)
        self.reason_combo = ttk.Combobox(
            metadata,
            textvariable=self.reason_var,
            values=REASON_OPTIONS,
            state="readonly",
            width=28,
        )
        self.reason_combo.pack(side=tk.LEFT, padx=5)
        tk.Label(metadata, text="Note").pack(side=tk.LEFT)
        tk.Entry(metadata, textvariable=self.note_var).pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            padx=5,
        )

        self.image_label = tk.Label(self.root, bg="black", fg="white")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

    def _bind_keys(self) -> None:
        self.root.bind("h", lambda _event: self._save("Yes", "reviewed"))
        self.root.bind("H", lambda _event: self._save("Yes", "reviewed"))
        self.root.bind("v", lambda _event: self._save("No", "reviewed"))
        self.root.bind("V", lambda _event: self._save("No", "reviewed"))
        self.root.bind("u", lambda _event: self._save("", "unclear"))
        self.root.bind("U", lambda _event: self._save("", "unclear"))
        self.root.bind("s", lambda _event: self._skip())
        self.root.bind("S", lambda _event: self._skip())
        self.root.bind("<Control-z>", lambda _event: self._undo())
        self.root.bind("<Control-s>", lambda _event: self._save_and_exit())

    def _show_current(self) -> None:
        if self.index >= len(self.items):
            self._save_and_exit()
            return
        row = self.items.iloc[self.index]
        risk = _text(row.get("hidden_false_negative_risk_score", ""))
        evidence = _text(row.get("hidden_false_negative_risk_reasons", ""))
        self.info_var.set(
            "\n".join(
                [
                    f"Item {self.index + 1}/{len(self.items)} | "
                    f"completed={len(self.completed_ids)} | stored={len(self.decisions)}",
                    f"cohort={row['hidden_review_cohort']} | "
                    f"before={row['hidden_before_review']} | risk={risk}",
                    f"source={row['source_type']} | video={row['video_key']} | "
                    f"frame={row['frame_index']}",
                    f"pig={row['pig_id']} | behavior={row['behavior']} | "
                    f"track={_text(row.get('track_id', ''))}",
                    f"risk_evidence={evidence or 'none'}",
                    "Yellow = reviewed actor; cyan = other annotated pigs.",
                ]
            )
        )
        self._show_image(row)
        self.status_var.set(
            f"Decision file: {self.decision_path} | remaining={len(self.items) - self.index}"
        )

    def _show_image(self, row: pd.Series) -> None:
        try:
            image = build_review_image(
                row,
                frame_groups=self.frame_groups,
                video_index=self.video_index,
                crop_roots=self.crop_roots,
                reader=self.reader,
                padding=self.padding,
            )
            image.thumbnail(self.image_max_size, Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.photo, text="")
        except Exception as exc:
            self.photo = None
            self.image_label.configure(
                image="",
                text=f"Cannot render review context:\n{exc}",
            )

    def _save(self, hidden_after: str, status: str) -> None:
        row = self.items.iloc[self.index]
        item_id = str(row["hidden_review_item_id"])
        previous = self.decisions.get(item_id)
        reason = self.reason_var.get().strip()
        if not reason and status == "reviewed" and hidden_after == "No":
            reason = "clearly_visible"
            self.reason_var.set(reason)
        if not reason and status == "unclear":
            reason = "ambiguous"
            self.reason_var.set(reason)
        note = self.note_var.get().strip()
        if note:
            reason = f"{reason};note={note}"
        semantic_error = hidden_decision_semantic_error(
            hidden_after=hidden_after,
            review_status=status,
            reason=reason,
        )
        if semantic_error is not None:
            messagebox.showerror(
                "Decision not saved",
                decision_error_message(semantic_error),
                parent=self.root,
            )
            self.reason_combo.focus_set()
            return
        record = make_decision_record(
            row,
            hidden_after=hidden_after,
            status=status,
            confidence=self.confidence_var.get(),
            reason=reason,
            reviewer=self.reviewer,
        )
        self.decisions[item_id] = record
        if is_completed_decision(record):
            self.completed_ids.add(item_id)
        else:
            self.completed_ids.discard(item_id)
        self.undo_stack.append((item_id, previous.copy() if previous else None))
        write_decisions(self.decision_path, self.decisions)
        self.index += 1
        self.reason_var.set("")
        self.note_var.set("")
        self._show_current()

    def _skip(self) -> None:
        self.index += 1
        self.reason_var.set("")
        self.note_var.set("")
        self._show_current()

    def _undo(self) -> None:
        if not self.undo_stack:
            return
        item_id, previous = self.undo_stack.pop()
        if previous is None:
            self.decisions.pop(item_id, None)
        else:
            self.decisions[item_id] = previous
        if previous is not None and is_completed_decision(previous):
            self.completed_ids.add(item_id)
        else:
            self.completed_ids.discard(item_id)
        write_decisions(self.decision_path, self.decisions)
        self.index = max(0, self.index - 1)
        self.reason_var.set("")
        self.note_var.set("")
        self._show_current()

    def _save_and_exit(self) -> None:
        write_decisions(self.decision_path, self.decisions)
        self.reader.close()
        self.root.destroy()


def load_review_inputs(
    manifest_path: Path,
    frame_features_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load review inputs and fail before GUI startup on schema errors."""
    manifest = pd.read_csv(manifest_path, low_memory=False)
    frames = pd.read_csv(frame_features_path, low_memory=False)
    missing = sorted(REQUIRED_MANIFEST_COLUMNS.difference(manifest.columns))
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    duplicate = manifest["hidden_review_item_id"].astype(str).duplicated()
    if duplicate.any():
        raise ValueError(f"Manifest has duplicate review items: {int(duplicate.sum())}")
    if "frame_uid" not in frames.columns:
        raise ValueError("frame_features missing frame_uid")
    return manifest, frames


def make_decision_record(
    row: pd.Series,
    *,
    hidden_after: str,
    status: str,
    confidence: str,
    reason: str,
    reviewer: str,
) -> dict[str, str]:
    """Create one strict decision row used by coverage and apply scripts."""
    record = {column: _text(row.get(column, "")) for column in DECISION_COLUMNS}
    record.update(
        {
            "hidden_review_item_id": _text(row["hidden_review_item_id"]),
            "hidden_before_review": _text(row["hidden_before_review"]),
            "hidden_after_review": hidden_after,
            "hidden_review_status": status,
            "hidden_review_confidence": confidence,
            "hidden_review_reason": reason,
            "hidden_reviewer": reviewer,
            "hidden_reviewed_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    return record


def load_decisions(path: Path) -> dict[str, dict[str, str]]:
    """Load resumable decisions and reject duplicate item keys."""
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    decisions: dict[str, dict[str, str]] = {}
    for row in rows:
        item_id = str(row.get("hidden_review_item_id", "")).strip()
        if not item_id:
            raise ValueError("Decision row missing hidden_review_item_id")
        if item_id in decisions:
            raise ValueError(f"Duplicate decision item: {item_id}")
        decisions[item_id] = {column: str(row.get(column, "")) for column in DECISION_COLUMNS}
    return decisions


def is_completed_decision(record: dict[str, str]) -> bool:
    """Accept only resolved, semantically coherent records for GUI resume."""
    status = _text(record.get("hidden_review_status", "")).lower()
    if status not in {"reviewed", "resolved", "complete"}:
        return False
    try:
        error = hidden_decision_semantic_error(
            hidden_after=record.get("hidden_after_review", ""),
            review_status=status,
            reason=record.get("hidden_review_reason", ""),
        )
    except ValueError:
        return False
    return error is None


def completed_decision_ids(
    decisions: dict[str, dict[str, str]],
) -> set[str]:
    """Return decision IDs that may safely be skipped during resume."""
    return {
        item_id
        for item_id, record in decisions.items()
        if is_completed_decision(record)
    }


def decision_error_message(error_code: str) -> str:
    """Translate stable audit codes into concise reviewer instructions."""
    messages = {
        "missing_hidden_review_reason": (
            "Choose why the pig is hidden before saving Hidden = Yes."
        ),
        "hidden_yes_with_clearly_visible_reason": (
            "Hidden = Yes conflicts with reason clearly_visible. "
            "Choose the observed occlusion reason."
        ),
        "visible_no_with_hidden_only_reason": (
            "Visible = No conflicts with an occlusion-only reason. "
            "Choose clearly_visible or another compatible reason."
        ),
    }
    return messages.get(error_code, f"Invalid Hidden decision: {error_code}")


def write_decisions(
    path: Path,
    decisions: dict[str, dict[str, str]],
) -> None:
    """Write sorted decision rows through a sibling file before replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DECISION_COLUMNS)
        writer.writeheader()
        for item_id in sorted(decisions):
            writer.writerow(decisions[item_id])
    temporary.replace(path)


def build_review_image(
    row: pd.Series,
    *,
    frame_groups: dict[tuple[str, ...], pd.DataFrame],
    video_index: dict[str, Path],
    crop_roots: list[Path],
    reader: VideoReader,
    padding: float,
) -> Image.Image:
    """Render full-frame context and an undistorted actor crop side by side."""
    media_mode, media_path = resolve_review_media(
        row,
        video_index=video_index,
        crop_roots=crop_roots,
    )
    if media_mode == "cvat_video_bbox" and media_path is not None:
        frame = reader.read(media_path, int(float(row["frame_index"])))
        if frame is None:
            raise OSError(f"Cannot read CVAT video frame: {media_path}")
        full = frame.copy()
        group = frame_groups.get(_frame_group_key(row), pd.DataFrame())
        draw_context_boxes(full, group, row)
        crop = crop_bbox(frame, row, padding=padding)
        return compose_context_and_crop(full, crop)

    if media_mode != "legacy_crop" or media_path is None:
        source_type = _text(row.get("source_type", ""))
        raise FileNotFoundError(f"No media for source_type={source_type}")
    crop_image = Image.open(media_path).convert("RGB")
    canvas = Image.new("RGB", (1200, 680), "black")
    crop_image.thumbnail((1150, 630), Image.Resampling.LANCZOS)
    offset = ((1200 - crop_image.width) // 2, (680 - crop_image.height) // 2)
    canvas.paste(crop_image, offset)
    ImageDraw.Draw(canvas).text((12, 12), "Legacy crop context", fill="white")
    return canvas


def draw_context_boxes(
    frame: Any,
    group: pd.DataFrame,
    target: pd.Series,
) -> None:
    """Draw all annotated pigs and emphasize the reviewed actor."""
    target_pig = _text(target.get("pig_id", ""))
    target_track = _text(target.get("track_id", ""))
    for _, other in group.iterrows():
        is_target = (
            _text(other.get("pig_id", "")) == target_pig
            and _text(other.get("track_id", "")) == target_track
        )
        color = (0, 220, 255) if is_target else (255, 220, 0)
        thickness = 4 if is_target else 2
        x1, y1, x2, y2 = _bbox_ints(other, frame.shape[1], frame.shape[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = f"{_text(other.get('pig_id', ''))}"
        cv2.putText(
            frame,
            label,
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def compose_context_and_crop(full: Any, crop: Any) -> Image.Image:
    """Letterbox both panels; never stretch the pig into a square."""
    full_rgb = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    full_pil = _letterbox(Image.fromarray(full_rgb), (850, 650))
    crop_pil = _letterbox(Image.fromarray(crop_rgb), (350, 650))
    canvas = Image.new("RGB", (1200, 680), "black")
    canvas.paste(full_pil, (0, 30))
    canvas.paste(crop_pil, (850, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), "Full-frame context", fill="white")
    draw.text((860, 8), "Actor crop (letterbox)", fill="white")
    return canvas


def build_video_index(roots: list[Path]) -> dict[str, Path]:
    """Index videos by exact and normalized names, including _30fps variants."""
    index: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(root)
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                for key in {path.name, path.stem, _normalize_video_key(path.name)}:
                    index.setdefault(key.lower(), path)
    return index


def resolve_video(row: pd.Series, index: dict[str, Path]) -> Path | None:
    direct = _text(row.get("source_video_path", ""))
    if direct and Path(direct).exists():
        return Path(direct)
    video_key = _text(row.get("video_key", ""))
    candidates = [
        video_key,
        Path(video_key).stem,
        _normalize_video_key(video_key),
        f"{_normalize_video_key(video_key)}_30fps",
    ]
    for candidate in candidates:
        path = index.get(candidate.lower())
        if path is not None:
            return path
    return None


def resolve_review_media(
    row: pd.Series,
    *,
    video_index: dict[str, Path],
    crop_roots: list[Path],
) -> tuple[str, Path | None]:
    """Resolve media strictly from the source-specific acquisition contract."""
    source_type = _text(row.get("source_type", ""))
    if source_type == "legacy_recovered":
        return "legacy_crop", resolve_crop(row, crop_roots)
    if source_type == "cvat_tracking_xml":
        return "cvat_video_bbox", resolve_video(row, video_index)
    return "unknown_source", None


def resolve_crop(row: pd.Series, roots: list[Path]) -> Path | None:
    for root in roots:
        resolved = resolve_legacy_crop(row, root)
        if resolved is not None:
            return resolved

    names = [
        _text(row.get("crop_path", "")),
        _text(row.get("image_name", "")),
    ]
    names = [Path(name).name for name in names if name]
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.exists():
                return candidate
        for name in names:
            matches = sorted(path for path in root.rglob(name) if path.is_file())
            if len(matches) == 1:
                return matches[0]
    return None


def crop_bbox(frame: Any, row: pd.Series, *, padding: float) -> Any:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _bbox_ints(row, w, h)
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad_x = int(round(box_w * padding))
    pad_y = int(round(box_h * padding))
    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(w, x2 + pad_x)
    bottom = min(h, y2 + pad_y)
    return frame[top : max(top + 1, bottom), left : max(left + 1, right)].copy()


def _bbox_ints(row: pd.Series, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, int(round(float(row.get("x1", 0))))))
    y1 = max(0, min(height - 1, int(round(float(row.get("y1", 0))))))
    x2 = max(x1 + 1, min(width, int(round(float(row.get("x2", x1 + 1))))))
    y2 = max(y1 + 1, min(height, int(round(float(row.get("y2", y1 + 1))))))
    return x1, y1, x2, y2


def _letterbox(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "black")
    offset = ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2)
    canvas.paste(copy, offset)
    return canvas


def _normalize_video_key(value: str) -> str:
    name = Path(str(value)).name
    name = re.sub(r"\.(mp4|avi|mov|mkv)$", "", name, flags=re.I)
    name = re.sub(r"_30fps$", "", name, flags=re.I)
    name = re.sub(
        r"^(test video |tracking_annotation_|tracking annotation )",
        "",
        name,
        flags=re.I,
    )
    return name.strip()


def _frame_group_key(row: pd.Series) -> tuple[str, ...]:
    columns = ["source_type", "dataset_id", "video_key", "frame_uid"]
    return tuple(_text(row.get(column, "")) for column in columns)


def _text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def validate_media_resolution(
    manifest: pd.DataFrame,
    video_index: dict[str, Path],
    crop_roots: list[Path],
) -> dict[str, Any]:
    """Audit source-specific media resolution without decoding every frame."""
    video_count = 0
    crop_count = 0
    missing_count = 0
    unknown_source_count = 0
    missing_by_source: dict[str, int] = {}
    missing_examples: list[dict[str, str]] = []
    for _, row in manifest.iterrows():
        media_mode, media_path = resolve_review_media(
            row,
            video_index=video_index,
            crop_roots=crop_roots,
        )
        if media_mode == "cvat_video_bbox" and media_path is not None:
            video_count += 1
        elif media_mode == "legacy_crop" and media_path is not None:
            crop_count += 1
        else:
            missing_count += 1
            source_type = _text(row.get("source_type", "")) or "<missing>"
            missing_by_source[source_type] = missing_by_source.get(source_type, 0) + 1
            if len(missing_examples) < 50:
                missing_examples.append(
                    {
                        "hidden_review_item_id": _text(
                            row.get("hidden_review_item_id", "")
                        ),
                        "source_type": source_type,
                        "video_key": _text(row.get("video_key", "")),
                        "crop_path": _text(row.get("crop_path", "")),
                        "source_video_path": _text(
                            row.get("source_video_path", "")
                        ),
                    }
                )
            if media_mode == "unknown_source":
                unknown_source_count += 1
    return {
        "manifest_items": int(len(manifest)),
        "video_resolved": video_count,
        "crop_resolved": crop_count,
        "unknown_source_items": unknown_source_count,
        "media_missing": missing_count,
        "missing_by_source": missing_by_source,
        "missing_examples": missing_examples,
    }


def main() -> None:
    args = parse_args()
    manifest, frames = load_review_inputs(
        args.manifest_csv,
        args.frame_features_csv,
    )
    video_index = build_video_index(args.video_root)
    resolution = validate_media_resolution(manifest, video_index, args.crop_root)
    print(resolution)
    if args.validation_audit_json is not None:
        audit = {
            "schema_version": "classification_v2_hidden_media_audit_v1",
            "manifest_csv": str(args.manifest_csv),
            "frame_features_csv": str(args.frame_features_csv),
            "video_roots": [str(path) for path in args.video_root],
            "crop_roots": [str(path) for path in args.crop_root],
            **resolution,
        }
        args.validation_audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.validation_audit_json.write_text(
            json.dumps(audit, indent=2),
            encoding="utf-8",
        )
    if args.validate_only:
        if resolution["media_missing"]:
            raise SystemExit(f"FAIL: missing media for {resolution['media_missing']} items")
        print("PASS: all Hidden review media can be resolved.")
        return

    decision_path = args.output_dir / DECISION_FILENAME
    decisions = load_decisions(decision_path)
    if args.include_reviewed:
        items = manifest.copy()
    else:
        completed_ids = completed_decision_ids(decisions)
        item_ids = manifest["hidden_review_item_id"].astype(str)
        items = manifest.loc[~item_ids.isin(completed_ids)].copy()
    if args.max_items is not None:
        if args.max_items <= 0:
            raise ValueError("--max-items must be > 0")
        items = items.head(args.max_items).copy()
    if items.empty:
        print("No pending Hidden review items.")
        return

    root = tk.Tk()
    app = HiddenQualityReviewApp(
        root,
        items=items,
        frame_features=frames,
        decisions=decisions,
        decision_path=decision_path,
        reviewer=args.reviewer,
        video_index=video_index,
        crop_roots=args.crop_root,
        padding=args.padding,
    )
    root.protocol("WM_DELETE_WINDOW", app._save_and_exit)
    root.mainloop()


if __name__ == "__main__":
    main()
