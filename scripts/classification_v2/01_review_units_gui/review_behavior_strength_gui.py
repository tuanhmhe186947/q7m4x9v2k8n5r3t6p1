from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cv2  # type: ignore

    CV2_AVAILABLE = True
except Exception:  # pragma: no cover - only used when OpenCV is missing
    cv2 = None  # type: ignore
    CV2_AVAILABLE = False

import tkinter as tk
from tkinter import messagebox, ttk

import pandas as pd
from PIL import Image, ImageTk

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

BEHAVIORS = [
    "",
    "drink",
    "eat",
    "fight",
    "social-nose",
    "explore",
    "lying",
    "stand",
    "move",
    "sitting",
    "playwithtoy",
]
STRENGTHS = ["strong", "medium", "weak", "boundary"]
DECISIONS = ["accept", "corrected", "exclude", "pending"]
GROUPS = [
    "",
    "roi_feeding_drinking_toy",
    "aggression_social",
    "motion_state",
    "posture",
    "none",
    "general",
    "unknown",
]
ACTIONS = ["", "main_train", "low_weight_train", "robust_train_only", "exclude"]


@dataclass
class ReviewItem:
    row_index: int
    review_row_index: str
    review_key: str
    image_path: Path | None
    video_path: Path | None
    image_source: str
    source_type: str
    dataset_id: str
    video_key: str
    frame_uid: str
    image_name: str
    group_id: str
    frame_index: str
    relative_frame_index: str
    track_id: str
    pig_id: str
    behavior: str
    strength_auto: str
    group_auto: str
    reason_auto: str
    priority: str
    roi_status: str
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class ReviewDecision:
    review_key: str
    review_row_index: str
    row_index: int
    image_path: str
    video_path: str
    image_source: str
    source_type: str
    dataset_id: str
    video_key: str
    frame_uid: str
    image_name: str
    group_id: str
    frame_index: str
    relative_frame_index: str
    track_id: str
    pig_id: str
    behavior: str
    manual_review_decision: str
    manual_label_strength: str
    manual_corrected_behavior: str
    manual_ambiguity_group: str
    manual_training_action: str
    manual_sample_weight: str
    manual_note: str
    auto_label_strength: str
    auto_ambiguity_group: str
    auto_review_reason: str
    roi_consistency_status_auto: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified GUI review tool for behavior label strength/ROI consistency. "
            "It supports both already-cropped legacy images and CVAT/video rows that "
            "need bbox crops from source video. It also resumes from existing decisions."
        )
    )
    parser.add_argument("--review-csv", type=Path, required=True, help="Review template CSV.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        action="append",
        default=[],
        help="Optional root containing crop/frame images. Can be repeated.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        action="append",
        default=[],
        help="Optional root containing source videos. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--decisions-csv",
        type=Path,
        default=None,
        help=("CSV storing/reusing manual decisions. Default: <output-dir>/behavior_strength_review_decisions.csv"),
    )
    parser.add_argument(
        "--merged-review-csv",
        type=Path,
        default=None,
        help=(
            "CSV copy of the review template with current manual decisions merged in. "
            "Default: <output-dir>/<review_csv_stem>__with_decisions.csv"
        ),
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.50,
        help="BBox crop padding ratio when cropping from video. Default: 0.50.",
    )
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--priority-max", type=int, default=None)
    parser.add_argument("--behavior", action="append", default=[])
    parser.add_argument("--source-type", action="append", default=[])
    parser.add_argument("--dataset-id", action="append", default=[])
    parser.add_argument(
        "--include-reviewed",
        action="store_true",
        help="Show rows that already have decisions. Default skips them.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing decisions CSV and start a new session.",
    )
    parser.add_argument(
        "--autosave-every",
        type=int,
        default=1,
        help="Autosave after N decisions. Default: 1.",
    )
    parser.add_argument("--copy-reviewed-images", action="store_true")
    return parser.parse_args()


class VideoFrameReader:
    def __init__(self) -> None:
        self.caps: dict[Path, Any] = {}
        self.frame_cache: dict[tuple[Path, int], Any] = {}
        self.cache_order: list[tuple[Path, int]] = []
        self.max_cache = 32

    def read_frame(self, video_path: Path, frame_index: int):
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV/cv2 is not available; cannot crop from video.")

        key = (video_path, frame_index)
        if key in self.frame_cache:
            return self.frame_cache[key].copy()

        cap = self.caps.get(video_path)
        if cap is None:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            self.caps[video_path] = cap

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            return None

        self.frame_cache[key] = frame.copy()
        self.cache_order.append(key)

        while len(self.cache_order) > self.max_cache:
            old = self.cache_order.pop(0)
            self.frame_cache.pop(old, None)

        return frame

    def close(self) -> None:
        for cap in self.caps.values():
            cap.release()
        self.caps.clear()


class BehaviorStrengthReviewApp:
    def __init__(
        self,
        root: tk.Tk,
        *,
        items: list[ReviewItem],
        source_df: pd.DataFrame,
        output_dir: Path,
        decisions_csv: Path,
        merged_review_csv: Path,
        existing_decisions: dict[str, ReviewDecision],
        copy_images: bool,
        padding: float,
        autosave_every: int,
    ) -> None:
        self.root = root
        self.items = items
        self.source_df = source_df
        self.output_dir = output_dir
        self.decisions_csv = decisions_csv
        self.merged_review_csv = merged_review_csv
        self.decisions_by_key: dict[str, ReviewDecision] = dict(existing_decisions)
        self.copy_images = copy_images
        self.padding = padding
        self.autosave_every = max(1, int(autosave_every))

        self.index = 0
        self.session_order: list[str] = []
        self.unsaved_count = 0
        self.reader = VideoFrameReader()
        self.current_photo: ImageTk.PhotoImage | None = None
        self.current_display_image: Image.Image | None = None

        self.strength_var = tk.StringVar(value="medium")
        self.decision_var = tk.StringVar(value="accept")
        self.behavior_var = tk.StringVar(value="")
        self.group_var = tk.StringVar(value="")
        self.action_var = tk.StringVar(value="")
        self.weight_var = tk.StringVar(value="")
        self.note_var = tk.StringVar(value="")
        self.info_var = tk.StringVar()
        self.status_var = tk.StringVar()

        self.root.title("Behavior Strength / ROI Review - Unified")
        self.root.geometry("1280x920")
        self._build_ui()
        self._bind_keys()
        self._show_current()

    def _build_ui(self) -> None:
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)
        tk.Label(top, textvariable=self.info_var, justify=tk.LEFT, anchor="w", font=("Consolas", 10)).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

        self.image_label = tk.Label(self.root, bg="black", fg="white")
        self.image_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=6)

        form = tk.Frame(self.root)
        form.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)

        self._combo(form, "Decision", self.decision_var, DECISIONS, 0)
        self._combo(form, "Strength [1-4]", self.strength_var, STRENGTHS, 1)
        self._combo(form, "Corrected behavior", self.behavior_var, BEHAVIORS, 2)
        self._combo(form, "Ambiguity group", self.group_var, GROUPS, 3)
        self._combo(form, "Training action", self.action_var, ACTIONS, 4)

        tk.Label(form, text="Weight").grid(row=0, column=10, sticky="w")
        tk.Entry(form, textvariable=self.weight_var, width=8).grid(row=1, column=10, padx=4, sticky="ew")
        tk.Label(form, text="Note").grid(row=0, column=11, sticky="w")
        tk.Entry(form, textvariable=self.note_var, width=46).grid(row=1, column=11, padx=4, sticky="ew")

        controls = tk.Frame(self.root)
        controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)
        tk.Button(controls, text="Save decision [Enter]", command=self.save_current, bg="#d7ffd7", height=2).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4
        )
        tk.Button(controls, text="Skip [S/Right]", command=self.skip, bg="#eeeeee", height=2).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4
        )
        tk.Button(controls, text="Undo current session [U]", command=self.undo, bg="#eeeeff", height=2).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4
        )
        tk.Button(controls, text="Save progress [Ctrl+S]", command=self.save_progress, bg="#d7e8ff", height=2).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4
        )
        tk.Button(controls, text="Save & Exit [Esc]", command=self.confirm_exit, bg="#ffeecf", height=2).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4
        )

        tk.Label(self.root, textvariable=self.status_var, justify=tk.LEFT, anchor="w", font=("Consolas", 10)).pack(
            side=tk.BOTTOM, fill=tk.X, padx=10, pady=4
        )

    def _combo(self, parent: tk.Frame, label: str, var: tk.StringVar, values: list[str], col: int) -> None:
        tk.Label(parent, text=label).grid(row=0, column=col * 2, sticky="w")
        ttk.Combobox(parent, textvariable=var, values=values, width=20, state="readonly").grid(
            row=1, column=col * 2, padx=4, sticky="ew"
        )

    def _bind_keys(self) -> None:
        self.root.bind("<Return>", lambda _event: self.save_current())
        self.root.bind("s", lambda _event: self.skip())
        self.root.bind("S", lambda _event: self.skip())
        self.root.bind("u", lambda _event: self.undo())
        self.root.bind("U", lambda _event: self.undo())
        self.root.bind("<Right>", lambda _event: self.skip())
        self.root.bind("<Control-s>", lambda _event: self.save_progress())
        self.root.bind("1", lambda _event: self.set_strength("strong"))
        self.root.bind("2", lambda _event: self.set_strength("medium"))
        self.root.bind("3", lambda _event: self.set_strength("weak"))
        self.root.bind("4", lambda _event: self.set_strength("boundary"))
        self.root.bind("<Escape>", lambda _event: self.confirm_exit())

    def set_strength(self, strength: str) -> None:
        self.strength_var.set(strength)
        if strength == "boundary":
            self.decision_var.set("exclude")
            self.action_var.set("exclude")
            self.weight_var.set("0")
        elif strength == "weak":
            self.action_var.set("low_weight_train")
            self.weight_var.set("0.35")
        elif strength == "medium":
            self.action_var.set("main_train")
            self.weight_var.set("0.75")
        elif strength == "strong":
            self.action_var.set("main_train")
            self.weight_var.set("1.0")

    def _show_current(self) -> None:
        if self.index >= len(self.items):
            self.save_and_exit()
            return

        item = self.items[self.index]
        self.strength_var.set(item.strength_auto if item.strength_auto in STRENGTHS else "medium")
        self.behavior_var.set("")
        self.group_var.set(default_group(item.group_auto, item.roi_status))
        self.decision_var.set("accept")
        self.action_var.set(default_action(item.strength_auto))
        self.weight_var.set(default_weight(item.strength_auto))
        self.note_var.set("")

        self.info_var.set(
            "\n".join(
                [
                    f"Item {self.index + 1}/{len(self.items)} | already saved "
                    f"decisions={len(self.decisions_by_key)} | "
                    f"source={item.image_source}",
                    f"behavior={item.behavior} | auto_strength={item.strength_auto} | group={item.group_auto}",
                    f"reason={item.reason_auto}",
                    f"roi={item.roi_status}",
                    f"source_type={item.source_type} | dataset={item.dataset_id}",
                    f"video={item.video_key} | frame={item.frame_index} | "
                    f"rel={item.relative_frame_index} | frame_uid={item.frame_uid}",
                    f"track={item.track_id} | pig={item.pig_id} | "
                    f"review_key={item.review_key} | "
                    f"review_row_index={item.review_row_index}",
                    f"bbox=({item.x1:.1f}, {item.y1:.1f}, {item.x2:.1f}, {item.y2:.1f})",
                    "Keys: 1 strong, 2 medium, 3 weak, 4 boundary, "
                    "Enter save, S skip, Ctrl+S save progress, Esc save exit",
                ]
            )
        )

        self._show_image(item)
        self.status_var.set(
            f"Session decisions: {len(self.session_order)} | Total saved-map "
            f"decisions: {len(self.decisions_by_key)} | Remaining this run: "
            f"{len(self.items) - self.index}"
        )

    def _show_image(self, item: ReviewItem) -> None:
        try:
            if item.image_path is not None and item.image_path.exists():
                img = Image.open(item.image_path).convert("RGB")
                self.current_display_image = img.copy()
                img = resize_for_display(img, 1180, 560)
                self.current_photo = ImageTk.PhotoImage(img)
                self.image_label.configure(image=self.current_photo, text="")
                return

            if item.video_path is not None and item.video_path.exists():
                frame_index = safe_int(item.frame_index, -1)
                if frame_index < 0:
                    raise ValueError(f"Invalid frame_index={item.frame_index}")

                frame = self.reader.read_frame(item.video_path, frame_index)
                if frame is None:
                    raise ValueError(f"Cannot read frame {frame_index} from {item.video_path}")

                crop = crop_bbox(frame, item.x1, item.y1, item.x2, item.y2, padding=self.padding)
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                self.current_display_image = img.copy()
                img = resize_for_display(img, 1180, 560)
                self.current_photo = ImageTk.PhotoImage(img)
                self.image_label.configure(image=self.current_photo, text="")
                return

            self.current_photo = None
            self.current_display_image = None
            self.image_label.configure(
                image="",
                text=(
                    "No image/video crop found for this review row\n\n"
                    f"source_type={item.source_type}\n"
                    f"video_key={item.video_key}\n"
                    f"frame_index={item.frame_index}\n"
                    f"bbox=({item.x1}, {item.y1}, {item.x2}, {item.y2})"
                ),
            )
        except Exception as exc:
            self.current_photo = None
            self.current_display_image = None
            self.image_label.configure(image="", text=f"Image/video error:\n{exc}")

    def save_current(self) -> None:
        item = self.items[self.index]
        decision = ReviewDecision(
            review_key=item.review_key,
            review_row_index=item.review_row_index,
            row_index=item.row_index,
            image_path=str(item.image_path or ""),
            video_path=str(item.video_path or ""),
            image_source=item.image_source,
            source_type=item.source_type,
            dataset_id=item.dataset_id,
            video_key=item.video_key,
            frame_uid=item.frame_uid,
            image_name=item.image_name,
            group_id=item.group_id,
            frame_index=item.frame_index,
            relative_frame_index=item.relative_frame_index,
            track_id=item.track_id,
            pig_id=item.pig_id,
            behavior=item.behavior,
            manual_review_decision=self.decision_var.get(),
            manual_label_strength=self.strength_var.get(),
            manual_corrected_behavior=self.behavior_var.get(),
            manual_ambiguity_group=self.group_var.get(),
            manual_training_action=self.action_var.get(),
            manual_sample_weight=self.weight_var.get(),
            manual_note=self.note_var.get(),
            auto_label_strength=item.strength_auto,
            auto_ambiguity_group=item.group_auto,
            auto_review_reason=item.reason_auto,
            roi_consistency_status_auto=item.roi_status,
        )
        self.decisions_by_key[item.review_key] = decision
        self.session_order.append(item.review_key)
        self.unsaved_count += 1

        if self.copy_images:
            self._copy_image(decision)

        if self.unsaved_count >= self.autosave_every:
            self._save_decisions(show_message=False)
            self._save_merged_review_csv()
            self.unsaved_count = 0

        self.index += 1
        self._show_current()

    def _copy_image(self, decision: ReviewDecision) -> None:
        dst_dir = (
            self.output_dir / "reviewed_images" / sanitize(decision.manual_label_strength) / sanitize(decision.behavior)
        )
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / (
            f"{sanitize(decision.video_key)}"
            f"__f{sanitize(decision.frame_index)}"
            f"__{sanitize(decision.pig_id)}"
            f"__{sanitize(decision.review_key)}.jpg"
        )
        if self.current_display_image is not None:
            self.current_display_image.save(dst)
            return
        if decision.image_path:
            src = Path(decision.image_path)
            if src.exists():
                shutil.copy2(src, dst.with_suffix(src.suffix.lower()))

    def skip(self) -> None:
        self.index += 1
        self._show_current()

    def undo(self) -> None:
        if not self.session_order:
            return
        last_key = self.session_order.pop()
        self.decisions_by_key.pop(last_key, None)
        self.index = max(0, self.index - 1)
        self._save_decisions(show_message=False)
        self._save_merged_review_csv()
        self._show_current()

    def save_progress(self) -> None:
        self._save_decisions(show_message=True)
        self._save_merged_review_csv()
        self.unsaved_count = 0

    def save_and_exit(self) -> None:
        self._save_decisions(show_message=False)
        self._save_merged_review_csv()
        self.reader.close()
        messagebox.showinfo(
            "Saved",
            f"Saved decisions to:\n{self.decisions_csv}\n\nMerged review CSV:\n{self.merged_review_csv}",
        )
        self.root.destroy()

    def confirm_exit(self) -> None:
        if messagebox.askyesno("Exit", "Save current progress and exit?"):
            self.save_and_exit()

    def _save_decisions(self, *, show_message: bool) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fields = list(ReviewDecision.__dataclass_fields__.keys())
        rows = [d.__dict__ for d in self.decisions_by_key.values()]
        rows.sort(
            key=lambda r: (
                str(r.get("video_key", "")),
                safe_int(r.get("frame_index", ""), 0),
                str(r.get("pig_id", "")),
                str(r.get("review_key", "")),
            )
        )
        with self.decisions_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        if show_message:
            messagebox.showinfo("Saved", f"Saved decisions to:\n{self.decisions_csv}")

    def _save_merged_review_csv(self) -> None:
        df = self.source_df.copy()
        for col in MANUAL_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        key_series = make_review_key_series(df)
        key_to_idx = {key: idx for idx, key in key_series.items()}
        row_index_to_idx = make_review_row_index_to_index(df)

        for decision in self.decisions_by_key.values():
            target_idx = key_to_idx.get(decision.review_key)
            if target_idx is None and decision.review_row_index:
                target_idx = row_index_to_idx.get(str(decision.review_row_index))
            if target_idx is None:
                continue
            df.loc[target_idx, "manual_review_decision"] = decision.manual_review_decision
            df.loc[target_idx, "manual_label_strength"] = decision.manual_label_strength
            df.loc[target_idx, "manual_corrected_behavior"] = decision.manual_corrected_behavior
            df.loc[target_idx, "manual_ambiguity_group"] = decision.manual_ambiguity_group
            df.loc[target_idx, "manual_training_action"] = decision.manual_training_action
            df.loc[target_idx, "manual_sample_weight"] = decision.manual_sample_weight
            df.loc[target_idx, "manual_note"] = decision.manual_note

        self.merged_review_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.merged_review_csv, index=False)


MANUAL_COLUMNS = [
    "manual_review_decision",
    "manual_label_strength",
    "manual_corrected_behavior",
    "manual_ambiguity_group",
    "manual_training_action",
    "manual_sample_weight",
    "manual_note",
]


def load_items(
    review_csv: Path,
    raw_roots: list[Path],
    video_roots: list[Path],
    args: argparse.Namespace,
) -> tuple[list[ReviewItem], pd.DataFrame, dict[str, ReviewDecision], list[dict[str, Any]]]:
    if not review_csv.exists():
        raise FileNotFoundError(review_csv)

    df = pd.read_csv(review_csv, low_memory=False)
    ensure_review_columns(df)

    decisions_csv = args.decisions_csv or (args.output_dir / "behavior_strength_review_decisions.csv")
    existing = {} if args.no_resume else load_existing_decisions(decisions_csv)

    if args.priority_max is not None and "review_priority" in df.columns:
        df = df[pd.to_numeric(df["review_priority"], errors="coerce").le(args.priority_max)].copy()
    if args.behavior:
        df = df[df["behavior"].astype(str).isin(args.behavior)].copy()
    if args.source_type and "source_type" in df.columns:
        df = df[df["source_type"].astype(str).isin(args.source_type)].copy()
    if args.dataset_id and "dataset_id" in df.columns:
        df = df[df["dataset_id"].astype(str).isin(args.dataset_id)].copy()

    image_index = build_image_index(raw_roots) if raw_roots else {}
    video_index = build_video_index(video_roots) if video_roots else {}

    items: list[ReviewItem] = []
    missing: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        review_key = get_or_make_review_key(row, int(idx))
        if not args.include_reviewed and review_key in existing:
            continue

        image_path = resolve_image(row, image_index)
        video_path: Path | None = None
        image_source = "missing"

        if image_path is not None:
            image_source = "image_or_crop"
        else:
            video_path = resolve_video_path(row, video_index)
            if video_path is not None and has_valid_bbox(row):
                image_source = "video_crop"
            else:
                missing.append(row_to_missing_record(int(idx), row, review_key))

        item = ReviewItem(
            row_index=int(idx),
            review_row_index=get_str(row, "review_row_index") or str(idx),
            review_key=review_key,
            image_path=image_path,
            video_path=video_path,
            image_source=image_source,
            source_type=get_str(row, "source_type"),
            dataset_id=get_str(row, "dataset_id"),
            video_key=get_str(row, "video_key", fallback_col="source_video_key"),
            frame_uid=get_str(row, "frame_uid"),
            image_name=get_str(row, "image_name"),
            group_id=get_str(row, "group_id"),
            frame_index=get_str(row, "frame_index"),
            relative_frame_index=get_str(row, "relative_frame_index"),
            track_id=get_str(row, "track_id", fallback_col="tracklet_id"),
            pig_id=get_str(row, "pig_id"),
            behavior=get_str(row, "behavior"),
            strength_auto=get_str(row, "label_strength_auto", fallback_col="label_strength"),
            group_auto=get_str(row, "ambiguity_group_auto", fallback_col="ambiguity_group"),
            reason_auto=get_str(row, "review_reason_auto", fallback_col="review_reason"),
            priority=get_str(row, "review_priority"),
            roi_status=get_str(row, "roi_consistency_status_auto", fallback_col="roi_consistency_status"),
            x1=safe_float(get_str(row, "x1"), 0.0),
            y1=safe_float(get_str(row, "y1"), 0.0),
            x2=safe_float(get_str(row, "x2"), 0.0),
            y2=safe_float(get_str(row, "y2"), 0.0),
        )
        items.append(item)

    items.sort(
        key=lambda item: (
            item.priority or "99",
            item.source_type,
            item.dataset_id,
            item.video_key,
            safe_int(item.frame_index, 0),
            item.pig_id,
            item.review_key,
        )
    )

    if args.max_items is not None:
        items = items[: args.max_items]

    return items, df, existing, missing


def ensure_review_columns(df: pd.DataFrame) -> None:
    if "review_key" not in df.columns:
        df["review_key"] = make_review_key_series(df)
    if "review_row_index" not in df.columns:
        df["review_row_index"] = list(df.index.astype(str))
    for col in MANUAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""


def load_existing_decisions(path: Path) -> dict[str, ReviewDecision]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return {}
    out: dict[str, ReviewDecision] = {}
    for _, row in df.iterrows():
        key = get_str(row, "review_key") or get_str(row, "review_row_index") or get_str(row, "row_index")
        if not key:
            continue
        kwargs = {field: get_str(row, field) for field in ReviewDecision.__dataclass_fields__.keys()}
        kwargs["row_index"] = safe_int(kwargs.get("row_index", ""), -1)
        out[key] = ReviewDecision(**kwargs)  # type: ignore[arg-type]
    return out


def make_review_key_series(df: pd.DataFrame) -> pd.Series:
    if "review_key" in df.columns:
        existing = df["review_key"].fillna("").astype(str).str.strip()
        if existing.ne("").all():
            return existing

    key_cols = [
        "source_type",
        "dataset_id",
        "video_key",
        "source_video_key",
        "frame_uid",
        "frame_index",
        "relative_frame_index",
        "track_id",
        "tracklet_id",
        "pig_id",
        "behavior",
        "x1",
        "y1",
        "x2",
        "y2",
    ]

    def row_hash(row: pd.Series) -> str:
        explicit = get_str(row, "review_row_index")
        if explicit:
            return explicit
        text = "|".join(str(row.get(c, "")) for c in key_cols)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    return df.apply(row_hash, axis=1)


def make_review_row_index_to_index(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "review_row_index" in df.columns:
        for idx, value in df["review_row_index"].items():
            text = str(value).strip()
            if text:
                out[text] = idx
    return out


def build_image_index(raw_roots: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for root in raw_roots:
        if not root.exists():
            raise FileNotFoundError(root)
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                for key in keys_for_path(path, root):
                    index.setdefault(key, []).append(path)
    return index


def build_video_index(video_roots: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in video_roots:
        if not root.exists():
            raise FileNotFoundError(root)
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
                continue
            keys = {path.name, path.stem, normalize_video_key(path.name), normalize_video_key(path.stem)}
            for key in keys:
                if key and key not in index:
                    index[key] = path
    return index


def resolve_image(row: pd.Series, image_index: dict[str, list[Path]]) -> Path | None:
    direct_cols = ["crop_path", "image_path", "crop_image_path", "review_image_path"]
    for col in direct_cols:
        text = get_str(row, col)
        if text:
            p = Path(text)
            if p.exists() and p.is_file():
                return p

    candidates: list[str] = []
    for col in [
        "crop_path",
        "image_path",
        "crop_image_path",
        "review_image_path",
        "image_name",
        "image_key",
        "frame_uid",
    ]:
        text = get_str(row, col)
        if text:
            candidates.extend(candidate_keys(text))

    group_id = get_str(row, "group_id")
    video_key = get_str(row, "video_key", fallback_col="source_video_key")
    pig_id = get_str(row, "pig_id")
    behavior = get_str(row, "behavior")
    rel = get_str(row, "relative_frame_index")
    frame = get_str(row, "frame_index")
    track_id = get_str(row, "track_id", fallback_col="tracklet_id")

    for text in [
        group_id,
        video_key,
        f"{group_id}_{pig_id}",
        f"{group_id}_{behavior}",
        f"{group_id}_{track_id}",
        f"{group_id}_rel{rel}",
        f"{group_id}__rel{rel}",
        f"{group_id}_frame{frame}",
        f"{group_id}__frame{frame}",
        f"{video_key}_frame{frame}",
        f"{video_key}__frame{frame}",
        f"{video_key}_{pig_id}",
    ]:
        if text:
            candidates.extend(candidate_keys(text))

    for key in candidates:
        paths = image_index.get(key)
        if paths:
            return choose_best_image(paths, row)

    return fuzzy_image_lookup(image_index, row)


def fuzzy_image_lookup(image_index: dict[str, list[Path]], row: pd.Series) -> Path | None:
    video_norm = normalize_key(get_str(row, "video_key", fallback_col="source_video_key"))
    group_norm = normalize_key(get_str(row, "group_id"))
    pig_norm = normalize_key(get_str(row, "pig_id"))
    behavior_norm = normalize_key(get_str(row, "behavior"))
    rel_int = safe_int(get_str(row, "relative_frame_index"), -1)
    frame_int = safe_int(get_str(row, "frame_index"), -1)
    rel_tokens = [
        f"rel{rel_int:02d}",
        f"rel_{rel_int:02d}",
        f"f{rel_int:06d}",
        f"frame{frame_int:06d}",
        f"frame_{frame_int:06d}",
        str(rel_int),
        str(frame_int),
    ]
    best: tuple[int, Path] | None = None
    for paths in image_index.values():
        for path in paths:
            text = normalize_key(str(path))
            score = 0
            if video_norm and video_norm in text:
                score += 22
            if group_norm and group_norm in text:
                score += 20
            if pig_norm and pig_norm in text:
                score += 8
            if behavior_norm and behavior_norm in text:
                score += 4
            if any(tok and tok in text for tok in rel_tokens):
                score += 6
            if score <= 0:
                continue
            if best is None or score > best[0] or (score == best[0] and str(path) < str(best[1])):
                best = (score, path)
    return best[1] if best is not None else None


def choose_best_image(paths: list[Path], row: pd.Series) -> Path:
    if len(paths) == 1:
        return paths[0]
    video = normalize_key(get_str(row, "video_key", fallback_col="source_video_key"))
    group = normalize_key(get_str(row, "group_id"))
    pig = normalize_key(get_str(row, "pig_id"))
    frame = safe_int(get_str(row, "frame_index"), -1)
    rel = safe_int(get_str(row, "relative_frame_index"), -1)
    tokens = [f"f{frame:06d}", f"frame{frame:06d}", f"frame_{frame:06d}", f"rel{rel:02d}", str(frame), str(rel)]
    scored = []
    for path in paths:
        text = normalize_key(str(path))
        score = 0
        if video and video in text:
            score += 20
        if group and group in text:
            score += 18
        if pig and pig in text:
            score += 8
        if any(tok in text for tok in tokens):
            score += 6
        scored.append((score, str(path), path))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][2]


def resolve_video_path(row: pd.Series, video_index: dict[str, Path]) -> Path | None:
    if not video_index:
        return None

    candidates: list[str] = []
    for col in ["video_path", "source_video_path"]:
        text = get_str(row, col)
        if text:
            p = Path(text)
            if p.exists() and p.is_file():
                return p
            candidates.append(text)

    for col in ["video_key", "source_video_key", "image_name", "task_id", "dataset_id"]:
        text = get_str(row, col)
        if text:
            candidates.append(text)
            candidates.append(Path(text).stem)
            candidates.append(Path(text).name)

    expanded: list[str] = []
    for c in candidates:
        expanded.append(c)
        expanded.append(normalize_video_key(c))
        expanded.append(strip_cvat_prefix(normalize_video_key(c)))
        if not str(c).lower().endswith(tuple(VIDEO_EXTS)):
            expanded.append(f"{c}.mp4")
            expanded.append(normalize_video_key(f"{c}.mp4"))

    for key in expanded:
        if key in video_index:
            return video_index[key]
    return None


def crop_bbox(frame: Any, x1: float, y1: float, x2: float, y2: float, *, padding: float):
    h, w = frame.shape[:2]
    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)
    bw = max(1.0, right - left)
    bh = max(1.0, bottom - top)
    pad_x = bw * padding
    pad_y = bh * padding
    x1i = max(0, int(round(left - pad_x)))
    y1i = max(0, int(round(top - pad_y)))
    x2i = min(w, int(round(right + pad_x)))
    y2i = min(h, int(round(bottom + pad_y)))
    if x2i <= x1i:
        x2i = min(w, x1i + 1)
    if y2i <= y1i:
        y2i = min(h, y1i + 1)
    return frame[y1i:y2i, x1i:x2i].copy()


def keys_for_path(path: Path, root: Path) -> set[str]:
    keys = {
        str(path),
        path.name,
        path.stem,
        normalize_key(str(path)),
        normalize_key(path.name),
        normalize_key(path.stem),
    }
    try:
        rel = path.relative_to(root)
        keys.add(str(rel))
        keys.add(normalize_key(str(rel)))
    except ValueError:
        pass
    for parent in path.parents:
        if parent == root.parent:
            break
        keys.add(parent.name)
        keys.add(normalize_key(parent.name))
    return {k for k in keys if k}


def candidate_keys(text: str) -> list[str]:
    p = Path(text)
    return [text, p.name, p.stem, normalize_key(text), normalize_key(p.name), normalize_key(p.stem)]


def resize_for_display(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    w, h = img.size
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img if size == img.size else img.resize(size, Image.Resampling.LANCZOS)


def default_action(strength: str) -> str:
    return {"strong": "main_train", "medium": "main_train", "weak": "low_weight_train", "boundary": "exclude"}.get(
        strength, ""
    )


def default_weight(strength: str) -> str:
    return {"strong": "1.0", "medium": "0.75", "weak": "0.35", "boundary": "0"}.get(strength, "")


def default_group(group_auto: str, roi_status: str) -> str:
    group = str(group_auto or "").strip()
    if group == "roi_based" or str(roi_status).startswith("target_roi"):
        return "roi_feeding_drinking_toy"
    if "+" in group:
        if "roi_based" in group:
            return "roi_feeding_drinking_toy"
        if "aggression_social" in group:
            return "aggression_social"
        if "motion_state" in group:
            return "motion_state"
        if "posture" in group:
            return "posture"
    return group if group in GROUPS else ""


def get_or_make_review_key(row: pd.Series, idx: int) -> str:
    for col in ["review_key", "review_row_index"]:
        text = get_str(row, col)
        if text:
            return text
    return make_review_key_series(pd.DataFrame([row])).iloc[0] or f"row_{idx}"


def has_valid_bbox(row: pd.Series) -> bool:
    x1 = safe_float(get_str(row, "x1"), float("nan"))
    y1 = safe_float(get_str(row, "y1"), float("nan"))
    x2 = safe_float(get_str(row, "x2"), float("nan"))
    y2 = safe_float(get_str(row, "y2"), float("nan"))
    return all(pd.notna(v) for v in [x1, y1, x2, y2]) and x2 > x1 and y2 > y1


def row_to_missing_record(idx: int, row: pd.Series, review_key: str) -> dict[str, Any]:
    return {
        "row_index": idx,
        "review_key": review_key,
        "review_row_index": get_str(row, "review_row_index"),
        "source_type": get_str(row, "source_type"),
        "dataset_id": get_str(row, "dataset_id"),
        "video_key": get_str(row, "video_key", fallback_col="source_video_key"),
        "frame_uid": get_str(row, "frame_uid"),
        "frame_index": get_str(row, "frame_index"),
        "pig_id": get_str(row, "pig_id"),
        "behavior": get_str(row, "behavior"),
        "x1": get_str(row, "x1"),
        "y1": get_str(row, "y1"),
        "x2": get_str(row, "x2"),
        "y2": get_str(row, "y2"),
        "reason": "no_image_and_no_video_crop",
    }


def save_missing(output_dir: Path, missing: list[dict[str, Any]]) -> None:
    if not missing:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(missing).to_csv(output_dir / "missing_review_images.csv", index=False)


def get_str(row: pd.Series, col: str, fallback_col: str | None = None) -> str:
    if col in row.index and not pd.isna(row[col]):
        return str(row[col]).strip()
    if fallback_col and fallback_col in row.index and not pd.isna(row[fallback_col]):
        return str(row[fallback_col]).strip()
    return ""


def normalize_key(text: str) -> str:
    text = str(text).replace("\\", "/").lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def strip_cvat_owner_suffix(value: str) -> str:
    text = str(value).strip()
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)
    return text.strip()


def strip_cvat_prefix(value: str) -> str:
    text = strip_cvat_owner_suffix(value)
    text = re.sub(r"^(Tracking_annotation_|tracking_annotation_)", "", text)
    text = re.sub(r"^(Classification_annotation_|classification_annotation_)", "", text)
    return text


def normalize_video_key(value: str) -> str:
    text = strip_cvat_prefix(value)
    text = Path(text.replace("\\", "/")).name
    text = re.sub(r"\.(mp4|avi|mov|mkv)$", "", text, flags=re.I)
    return text.strip()


def sanitize(text: str) -> str:
    text = re.sub(r'[<>:"/\\|?*\s]+', "_", str(text))
    return text.strip("_") or "empty"


def safe_int(value: Any, default: int) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value: Any, default: float) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    decisions_csv = args.decisions_csv or (args.output_dir / "behavior_strength_review_decisions.csv")
    merged_review_csv = args.merged_review_csv or (args.output_dir / f"{args.review_csv.stem}__with_decisions.csv")

    items, source_df, existing, missing = load_items(
        args.review_csv,
        args.raw_root,
        args.video_root,
        args,
    )
    save_missing(args.output_dir, missing)

    print(f"review CSV: {args.review_csv}")
    print(f"existing decisions loaded: {len(existing)}")
    print(f"review items this run: {len(items)}")
    print(f"missing images/video crops: {len(missing)}")
    print(f"decisions CSV: {decisions_csv}")
    print(f"merged review CSV: {merged_review_csv}")
    print(f"output dir: {args.output_dir}")

    if not items:
        print("No review items to show. Existing decisions were still preserved.")
        if existing:
            df = source_df.copy()
            for col in MANUAL_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            # Write a merged copy using existing decisions without opening Tk.
            key_series = make_review_key_series(df)
            key_to_idx = {key: idx for idx, key in key_series.items()}
            row_index_to_idx = make_review_row_index_to_index(df)
            for decision in existing.values():
                target_idx = key_to_idx.get(decision.review_key)
                if target_idx is None and decision.review_row_index:
                    target_idx = row_index_to_idx.get(str(decision.review_row_index))
                if target_idx is None:
                    continue
                for col in MANUAL_COLUMNS:
                    df.loc[target_idx, col] = getattr(decision, col, "")
            merged_review_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(merged_review_csv, index=False)
            print(f"merged review CSV written: {merged_review_csv}")
        return

    root = tk.Tk()
    app = BehaviorStrengthReviewApp(
        root,
        items=items,
        source_df=source_df,
        output_dir=args.output_dir,
        decisions_csv=decisions_csv,
        merged_review_csv=merged_review_csv,
        existing_decisions=existing,
        copy_images=args.copy_reviewed_images,
        padding=args.padding,
        autosave_every=args.autosave_every,
    )
    root.protocol("WM_DELETE_WINDOW", app.confirm_exit)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
