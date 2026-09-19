from __future__ import annotations

import argparse
import csv
import re
import sys
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import messagebox
from typing import Any
from xml.etree import ElementTree as ET

import cv2
import pandas as pd
from PIL import Image, ImageTk

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
DECISIONS_CSV_NAME = "hidden_review_decisions.csv"
PROGRESS_CSV_NAME = "hidden_review_progress.csv"


@dataclass
class ReviewItem:
    item_key: str
    source_kind: str  # csv_crop | cvat_xml_video
    source_path: Path
    row_index: int | None
    xml_index: int | None
    image_path: Path | None
    video_path: Path | None
    image_source: str  # crop_image | video_crop
    source_type: str
    dataset_id: str
    video_key: str
    group_id: str
    frame_uid: str
    image_name: str
    track_id: str
    track_label: str
    pig_id: str
    behavior: str
    old_hidden: str
    frame_index: int
    relative_frame_index: str
    x1: float
    y1: float
    x2: float
    y2: float
    hidden_col: str
    box_element: ET.Element | None


@dataclass
class ReviewDecision:
    item_key: str
    source_kind: str
    source_path: str
    row_index: str
    xml_index: str
    image_path: str
    video_path: str
    image_source: str
    source_type: str
    dataset_id: str
    video_key: str
    group_id: str
    frame_uid: str
    image_name: str
    track_id: str
    track_label: str
    pig_id: str
    behavior: str
    frame_index: str
    relative_frame_index: str
    old_hidden: str
    new_hidden: str
    decision: str
    note: str
    reviewed_at: str


class VideoFrameReader:
    def __init__(self) -> None:
        self.caps: dict[Path, cv2.VideoCapture] = {}
        self.frame_cache: dict[tuple[Path, int], Any] = {}
        self.cache_order: list[tuple[Path, int]] = []
        self.max_cache = 32

    def read_frame(self, video_path: Path, frame_index: int):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified hidden-label GUI. Supports crop-based CSV rows and "
            "CVAT XML/video rows in one tool, with resumable decision mapping."
        )
    )

    parser.add_argument(
        "--csv", type=Path, action="append", default=[], help="Crop/feature CSV to review. Can be repeated."
    )
    parser.add_argument(
        "--raw-root", type=Path, action="append", default=[], help="Root containing crop images. Can be repeated."
    )

    parser.add_argument("--xml", type=Path, action="append", default=[], help="Single CVAT XML file. Can be repeated.")
    parser.add_argument(
        "--xml-dir", type=Path, action="append", default=[], help="Folder containing CVAT XML files. Can be repeated."
    )
    parser.add_argument(
        "--video-root", type=Path, action="append", default=[], help="Root containing videos. Can be repeated."
    )

    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output folder for decisions/progress/corrected copies."
    )
    parser.add_argument(
        "--padding", type=float, default=0.12, help="Padding ratio around bbox for video crops. Default: 0.12."
    )
    parser.add_argument(
        "--include-hidden-no",
        action="store_true",
        help="Also review rows already marked Hidden=No. Default reviews Hidden=Yes only.",
    )
    parser.add_argument("--max-items", type=int, default=None, help="Limit remaining review items.")
    parser.add_argument(
        "--copy-reviewed-crops",
        action="store_true",
        help="Save/copy reviewed crop images to output-dir/reviewed_crops.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite XML files. Default saves corrected XML copies only."
    )

    parser.add_argument("--source-type", action="append", default=[], help="CSV source_type filter. Can be repeated.")
    parser.add_argument("--dataset-id", action="append", default=[], help="CSV dataset_id filter. Can be repeated.")
    parser.add_argument("--group-id", action="append", default=[], help="CSV group_id filter. Can be repeated.")

    parser.add_argument(
        "--include-reviewed", action="store_true", help="Show rows already present in hidden_review_decisions.csv."
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing hidden_review_decisions.csv and start a new decision map.",
    )
    parser.add_argument(
        "--autosave-corrected",
        action="store_true",
        help="Also write corrected CSV/XML copies after every decision. Slower but safest.",
    )

    return parser.parse_args()


class UnifiedHiddenReviewApp:
    def __init__(
        self,
        root: tk.Tk,
        *,
        items: list[ReviewItem],
        all_items: list[ReviewItem],
        dataframes: dict[Path, pd.DataFrame],
        csv_hidden_cols: dict[Path, str],
        trees: dict[Path, ET.ElementTree],
        decisions_by_key: dict[str, dict[str, str]],
        output_dir: Path,
        padding: float,
        copy_reviewed_crops: bool,
        overwrite_xml: bool,
        autosave_corrected: bool,
    ) -> None:
        self.root = root
        self.items = items
        self.all_items = all_items
        self.dataframes = dataframes
        self.csv_hidden_cols = csv_hidden_cols
        self.trees = trees
        self.decisions_by_key = decisions_by_key
        self.output_dir = output_dir
        self.padding = padding
        self.copy_reviewed_crops = copy_reviewed_crops
        self.overwrite_xml = overwrite_xml
        self.autosave_corrected = autosave_corrected

        self.index = 0
        self.session_stack: list[tuple[ReviewItem, dict[str, str] | None]] = []
        self.reader = VideoFrameReader()
        self.current_photo: ImageTk.PhotoImage | None = None
        self.current_image_pil: Image.Image | None = None

        self.info_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.note_var = tk.StringVar(value="manual_gui_review")

        self.root.title("Unified Hidden Label Review - CSV crops + CVAT XML video")
        self.root.geometry("1150x880")
        self._build_ui()
        self._bind_keys()
        self._show_current()

    def _build_ui(self) -> None:
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)
        tk.Label(top, textvariable=self.info_var, justify=tk.LEFT, anchor="w", font=("Consolas", 10)).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

        self.image_label = tk.Label(self.root, bg="black", fg="white")
        self.image_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)

        note_row = tk.Frame(self.root)
        note_row.pack(side=tk.TOP, fill=tk.X, padx=10, pady=4)
        tk.Label(note_row, text="Note").pack(side=tk.LEFT)
        tk.Entry(note_row, textvariable=self.note_var, width=80).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        controls = tk.Frame(self.root)
        controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        tk.Button(
            controls,
            text="✓ Hidden đúng - giữ Yes [V]",
            command=self.keep_hidden,
            bg="#d7ffd7",
            font=("Arial", 12, "bold"),
            height=2,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        tk.Button(
            controls,
            text="✗ Không hidden - đổi No [X]",
            command=self.mark_not_hidden,
            bg="#ffd7d7",
            font=("Arial", 12, "bold"),
            height=2,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        tk.Button(controls, text="Skip [S/Right]", command=self.skip, bg="#eeeeee", font=("Arial", 12), height=2).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4
        )
        tk.Button(controls, text="Undo [U]", command=self.undo, bg="#eeeeff", font=("Arial", 12), height=2).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4
        )
        tk.Button(
            controls,
            text="Save & Exit [Ctrl+S]",
            command=self.save_and_exit,
            bg="#d7e8ff",
            font=("Arial", 12, "bold"),
            height=2,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        tk.Label(self.root, textvariable=self.status_var, justify=tk.LEFT, anchor="w", font=("Consolas", 10)).pack(
            side=tk.BOTTOM, fill=tk.X, padx=10, pady=6
        )

    def _bind_keys(self) -> None:
        self.root.bind("v", lambda _event: self.keep_hidden())
        self.root.bind("V", lambda _event: self.keep_hidden())
        self.root.bind("x", lambda _event: self.mark_not_hidden())
        self.root.bind("X", lambda _event: self.mark_not_hidden())
        self.root.bind("s", lambda _event: self.skip())
        self.root.bind("S", lambda _event: self.skip())
        self.root.bind("u", lambda _event: self.undo())
        self.root.bind("U", lambda _event: self.undo())
        self.root.bind("<Right>", lambda _event: self.skip())
        self.root.bind("<Control-s>", lambda _event: self.save_and_exit())
        self.root.bind("<Escape>", lambda _event: self.confirm_exit())

    def _show_current(self) -> None:
        if self.index >= len(self.items):
            self.save_and_exit()
            return

        item = self.items[self.index]
        previous = self.decisions_by_key.get(item.item_key)
        previous_text = ""
        if previous:
            previous_text = (
                f"previous_decision={previous.get('decision', '')} -> new_hidden={previous.get('new_hidden', '')}"
            )

        self.info_var.set(
            "\n".join(
                [
                    f"Item: {self.index + 1}/{len(self.items)} | "
                    f"reviewed_map={len(self.decisions_by_key)}/"
                    f"{len(self.all_items)}",
                    f"kind={item.source_kind} | source={item.source_path}",
                    f"image_source={item.image_source} | image={item.image_path or ''} | video={item.video_path or ''}",
                    f"video_key={item.video_key} | frame={item.frame_index} | "
                    f"rel={item.relative_frame_index} | frame_uid={item.frame_uid}",
                    f"track_id={item.track_id} | label={item.track_label} | pig_id={item.pig_id}",
                    f"behavior={item.behavior} | old_hidden={item.old_hidden} | {previous_text}",
                    f"bbox=({item.x1:.1f}, {item.y1:.1f}, {item.x2:.1f}, {item.y2:.1f}) | key={item.item_key}",
                    "",
                    "V = giữ Hidden=Yes | X = đổi Hidden=No | S = bỏ qua | U = undo | Ctrl+S = lưu",
                ]
            )
        )

        self._show_image(item)
        self.status_var.set(
            f"Session decisions: {len(self.session_stack)} | Remaining this "
            f"run: {len(self.items) - self.index} | Decision file: "
            f"{self.output_dir / DECISIONS_CSV_NAME}"
        )

    def _show_image(self, item: ReviewItem) -> None:
        self.current_photo = None
        self.current_image_pil = None

        try:
            if item.image_path is not None and item.image_path.exists():
                img = Image.open(item.image_path).convert("RGB")
                self.current_image_pil = img.copy()
                img = resize_for_display(img, 1060, 650)
                self.current_photo = ImageTk.PhotoImage(img)
                self.image_label.configure(image=self.current_photo, text="")
                return

            if item.video_path is not None and item.video_path.exists():
                frame = self.reader.read_frame(item.video_path, item.frame_index)
                if frame is None:
                    raise ValueError(f"Cannot read frame {item.frame_index} from {item.video_path}")

                crop = crop_bbox(frame, item.x1, item.y1, item.x2, item.y2, padding=self.padding)
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                self.current_image_pil = img.copy()
                img = resize_for_display(img, 1060, 650)
                self.current_photo = ImageTk.PhotoImage(img)
                self.image_label.configure(image=self.current_photo, text="")
                return

            self.image_label.configure(image="", text="No crop image or video crop available for this row")
        except Exception as exc:
            self.image_label.configure(image="", text=f"Image/video error:\n{exc}")

    def keep_hidden(self) -> None:
        self._save_decision(new_hidden="Yes", decision="keep_hidden_yes")

    def mark_not_hidden(self) -> None:
        self._save_decision(new_hidden="No", decision="change_hidden_to_no")

    def _save_decision(self, *, new_hidden: str, decision: str) -> None:
        item = self.items[self.index]
        previous = self.decisions_by_key.get(item.item_key)
        record = make_decision_record(item, new_hidden=new_hidden, decision=decision, note=self.note_var.get())
        self.decisions_by_key[item.item_key] = record.__dict__.copy()
        self.session_stack.append((item, previous.copy() if previous is not None else None))
        apply_hidden_to_source(item, new_hidden, self.dataframes)

        if self.copy_reviewed_crops:
            self._save_reviewed_crop(item, record)

        self._write_decisions_and_progress()
        if self.autosave_corrected:
            self._save_corrected_outputs(show_message=False)

        self.index += 1
        self.note_var.set("manual_gui_review")
        self._show_current()

    def _save_reviewed_crop(self, item: ReviewItem, record: ReviewDecision) -> None:
        if self.current_image_pil is None:
            return
        out_dir = (
            self.output_dir
            / "reviewed_crops"
            / sanitize(record.decision)
            / sanitize(record.video_key or "no_video_key")
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = (
            f"{sanitize(record.video_key)}"
            f"__frame{sanitize(record.frame_index)}"
            f"__track{sanitize(record.track_id)}"
            f"__{sanitize(record.pig_id)}"
            f"__{sanitize(record.behavior)}"
            f"__{sanitize(record.item_key)}.jpg"
        )
        self.current_image_pil.save(out_dir / out_name)

    def skip(self) -> None:
        self.index += 1
        self._show_current()

    def undo(self) -> None:
        if not self.session_stack:
            return
        item, previous = self.session_stack.pop()
        if previous is None:
            self.decisions_by_key.pop(item.item_key, None)
            apply_hidden_to_source(item, item.old_hidden, self.dataframes)
        else:
            self.decisions_by_key[item.item_key] = previous
            apply_hidden_to_source(item, previous.get("new_hidden", item.old_hidden), self.dataframes)

        self.index = max(0, self.index - 1)
        self._write_decisions_and_progress()
        if self.autosave_corrected:
            self._save_corrected_outputs(show_message=False)
        self._show_current()

    def _write_decisions_and_progress(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_decisions_csv(self.output_dir / DECISIONS_CSV_NAME, self.decisions_by_key)
        write_progress_csv(self.output_dir / PROGRESS_CSV_NAME, self.all_items, self.decisions_by_key)

    def _save_corrected_outputs(self, *, show_message: bool) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for csv_path, df in self.dataframes.items():
            out_name = csv_path.stem + "__hidden_corrected.csv"
            out_path = self.output_dir / out_name
            df.to_csv(out_path, index=False)

        changed_xmls = {
            Path(row["source_path"])
            for row in self.decisions_by_key.values()
            if row.get("source_kind") == "cvat_xml_video"
        }
        if changed_xmls:
            xml_out_dir = self.output_dir / "corrected_xml"
            xml_out_dir.mkdir(parents=True, exist_ok=True)
            for xml_path in changed_xmls:
                tree = self.trees.get(xml_path)
                if tree is None:
                    continue
                out_path = xml_path if self.overwrite_xml else xml_out_dir / xml_path.name
                tree.write(out_path, encoding="utf-8", xml_declaration=True)

        if show_message:
            messagebox.showinfo("Saved", f"Saved decisions and corrected outputs to:\n{self.output_dir}")

    def save_and_exit(self) -> None:
        self._write_decisions_and_progress()
        self._save_corrected_outputs(show_message=True)
        self.reader.close()
        self.root.destroy()

    def confirm_exit(self) -> None:
        if messagebox.askyesno("Exit", "Save decisions/corrected outputs and exit?"):
            self.save_and_exit()


def make_decision_record(item: ReviewItem, *, new_hidden: str, decision: str, note: str) -> ReviewDecision:
    return ReviewDecision(
        item_key=item.item_key,
        source_kind=item.source_kind,
        source_path=str(item.source_path),
        row_index="" if item.row_index is None else str(item.row_index),
        xml_index="" if item.xml_index is None else str(item.xml_index),
        image_path=str(item.image_path or ""),
        video_path=str(item.video_path or ""),
        image_source=item.image_source,
        source_type=item.source_type,
        dataset_id=item.dataset_id,
        video_key=item.video_key,
        group_id=item.group_id,
        frame_uid=item.frame_uid,
        image_name=item.image_name,
        track_id=item.track_id,
        track_label=item.track_label,
        pig_id=item.pig_id,
        behavior=item.behavior,
        frame_index=str(item.frame_index),
        relative_frame_index=item.relative_frame_index,
        old_hidden=item.old_hidden,
        new_hidden=new_hidden,
        decision=decision,
        note=note,
        reviewed_at=datetime.now().isoformat(timespec="seconds"),
    )


def apply_hidden_to_source(item: ReviewItem, new_hidden: str, dataframes: dict[Path, pd.DataFrame]) -> None:
    if item.source_kind == "csv_crop":
        if item.row_index is None:
            return
        df = dataframes[item.source_path]
        hidden_col = item.hidden_col or "hidden"
        if hidden_col in df.columns:
            df.loc[item.row_index, hidden_col] = new_hidden

        if new_hidden == "No":
            # Hidden correction should not reject sample by itself.
            if "qa_status" in df.columns and str(df.loc[item.row_index, "qa_status"]).lower() == "hidden":
                df.loc[item.row_index, "qa_status"] = "ok"
            if "training_tier" in df.columns and str(df.loc[item.row_index, "training_tier"]).lower() in {
                "review",
                "warning",
            }:
                df.loc[item.row_index, "training_tier"] = "clean"
            if "include_in_training" in df.columns:
                df.loc[item.row_index, "include_in_training"] = True
            if "use_for_main_eval" in df.columns:
                df.loc[item.row_index, "use_for_main_eval"] = True

    elif item.source_kind == "cvat_xml_video" and item.box_element is not None:
        set_box_attribute(item.box_element, "Hidden", new_hidden)


def write_decisions_csv(path: Path, decisions_by_key: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(ReviewDecision.__dataclass_fields__.keys())
    rows = list(decisions_by_key.values())
    rows.sort(
        key=lambda r: (
            r.get("source_kind", ""),
            r.get("source_path", ""),
            r.get("frame_index", ""),
            r.get("item_key", ""),
        )
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_progress_csv(path: Path, items: list[ReviewItem], decisions_by_key: dict[str, dict[str, str]]) -> None:
    rows: list[dict[str, Any]] = []
    for item in items:
        dec = decisions_by_key.get(item.item_key, {})
        rows.append(
            {
                "item_key": item.item_key,
                "reviewed": bool(dec),
                "source_kind": item.source_kind,
                "source_path": str(item.source_path),
                "row_index": "" if item.row_index is None else item.row_index,
                "xml_index": "" if item.xml_index is None else item.xml_index,
                "video_key": item.video_key,
                "frame_index": item.frame_index,
                "pig_id": item.pig_id,
                "behavior": item.behavior,
                "old_hidden": item.old_hidden,
                "new_hidden": dec.get("new_hidden", ""),
                "decision": dec.get("decision", ""),
                "note": dec.get("note", ""),
                "image_source": item.image_source,
                "image_path": str(item.image_path or ""),
                "video_path": str(item.video_path or ""),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def load_existing_decisions(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, low_memory=False).fillna("")
    if "item_key" not in df.columns:
        return {}
    out: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        key = str(row.get("item_key", "")).strip()
        if not key:
            continue
        out[key] = {str(col): str(row[col]) for col in df.columns}
    return out


def collect_csv_items(
    csv_paths: list[Path],
    raw_roots: list[Path],
    args: argparse.Namespace,
) -> tuple[list[ReviewItem], dict[Path, pd.DataFrame], dict[Path, str], list[dict[str, Any]]]:
    image_index = build_image_index(raw_roots) if raw_roots else {}
    dataframes: dict[Path, pd.DataFrame] = {}
    csv_hidden_cols: dict[Path, str] = {}
    items: list[ReviewItem] = []
    missing: list[dict[str, Any]] = []

    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        df = pd.read_csv(csv_path, low_memory=False)
        dataframes[csv_path] = df

        hidden_col = find_hidden_col(df)
        if hidden_col is None:
            print(f"[SKIP] no hidden/Hidden column: {csv_path}")
            continue
        csv_hidden_cols[csv_path] = hidden_col

        mask = pd.Series(True, index=df.index)
        if not args.include_hidden_no:
            mask &= df[hidden_col].astype(str).str.lower().isin(["yes", "true", "1", "y"])
        if args.source_type and "source_type" in df.columns:
            mask &= df["source_type"].astype(str).isin(args.source_type)
        if args.dataset_id and "dataset_id" in df.columns:
            mask &= df["dataset_id"].astype(str).isin(args.dataset_id)
        if args.group_id and "group_id" in df.columns:
            mask &= df["group_id"].astype(str).isin(args.group_id)

        for row_index, row in df[mask].iterrows():
            image_path = resolve_crop_image(row, image_index)
            if image_path is None:
                missing.append(
                    row_to_missing_record("csv_crop", csv_path, int(row_index), row, reason="missing_crop_image")
                )
                continue

            item = ReviewItem(
                item_key=make_csv_item_key(csv_path, int(row_index), row),
                source_kind="csv_crop",
                source_path=csv_path,
                row_index=int(row_index),
                xml_index=None,
                image_path=image_path,
                video_path=None,
                image_source="crop_image",
                source_type=get_str(row, "source_type"),
                dataset_id=get_str(row, "dataset_id"),
                video_key=get_str(row, "video_key"),
                group_id=get_str(row, "group_id"),
                frame_uid=get_str(row, "frame_uid"),
                image_name=get_str(row, "image_name"),
                track_id=get_str(row, "track_id", fallback_col="tracklet_id"),
                track_label=get_str(row, "track_label"),
                pig_id=get_str(row, "pig_id"),
                behavior=get_str(row, "behavior"),
                old_hidden=normalize_hidden_text(get_str(row, hidden_col)),
                frame_index=safe_int(get_str(row, "frame_index"), 0),
                relative_frame_index=get_str(row, "relative_frame_index"),
                x1=safe_float(get_str(row, "x1"), 0.0),
                y1=safe_float(get_str(row, "y1"), 0.0),
                x2=safe_float(get_str(row, "x2"), 0.0),
                y2=safe_float(get_str(row, "y2"), 0.0),
                hidden_col=hidden_col,
                box_element=None,
            )
            items.append(item)

    return items, dataframes, csv_hidden_cols, missing


def collect_xml_files(args: argparse.Namespace) -> list[Path]:
    files: list[Path] = []
    for path in args.xml:
        if not path.exists():
            raise FileNotFoundError(path)
        files.append(path)
    for xml_dir in args.xml_dir:
        if not xml_dir.exists():
            raise FileNotFoundError(xml_dir)
        files.extend(sorted(xml_dir.glob("*.xml")))

    seen = set()
    unique: list[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def collect_xml_items(
    xml_files: list[Path],
    video_roots: list[Path],
    args: argparse.Namespace,
) -> tuple[list[ReviewItem], dict[Path, ET.ElementTree], list[dict[str, Any]]]:
    if not xml_files:
        return [], {}, []
    if not video_roots:
        raise ValueError("--video-root is required when using --xml or --xml-dir")

    video_index = build_video_index(video_roots)
    items: list[ReviewItem] = []
    trees: dict[Path, ET.ElementTree] = {}
    missing: list[dict[str, Any]] = []

    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        trees[xml_path] = tree

        task_name = text(root, "./meta/task/name", default=xml_path.stem)
        video_key = normalize_video_key(task_name)
        video_path = resolve_video_path(xml_path, task_name, video_index)
        if video_path is None:
            missing.append(
                {
                    "source_kind": "cvat_xml_video",
                    "source_path": str(xml_path),
                    "reason": "missing_video",
                    "task_name": task_name,
                    "video_key": video_key,
                }
            )
            continue

        xml_index = 0
        for track in root.findall("track"):
            track_id = str(track.attrib.get("id", ""))
            track_label = str(track.attrib.get("label", ""))
            fallback_pig_id = pig_id_from_label(track_label)

            for box in track.findall("box"):
                if is_true(box.attrib.get("outside", "0")):
                    continue

                attrs = box_attributes(box)
                hidden = attrs.get("Hidden", attrs.get("hidden", "No"))
                if not args.include_hidden_no and not is_hidden_yes(hidden):
                    continue

                frame_index = safe_int(box.attrib.get("frame", "0"), 0)
                pig_id = attrs.get("ID", attrs.get("id", fallback_pig_id))
                behavior = attrs.get("Behavior", attrs.get("behavior", ""))
                x1 = safe_float(box.attrib.get("xtl"), 0.0)
                y1 = safe_float(box.attrib.get("ytl"), 0.0)
                x2 = safe_float(box.attrib.get("xbr"), 0.0)
                y2 = safe_float(box.attrib.get("ybr"), 0.0)

                item = ReviewItem(
                    item_key=make_xml_item_key(xml_path, track_id, frame_index, pig_id, behavior, x1, y1, x2, y2),
                    source_kind="cvat_xml_video",
                    source_path=xml_path,
                    row_index=None,
                    xml_index=xml_index,
                    image_path=None,
                    video_path=video_path,
                    image_source="video_crop",
                    source_type="cvat_tracking_xml",
                    dataset_id="",
                    video_key=video_key,
                    group_id="",
                    frame_uid="",
                    image_name="",
                    track_id=track_id,
                    track_label=track_label,
                    pig_id=pig_id,
                    behavior=behavior,
                    old_hidden=normalize_hidden_text(hidden),
                    frame_index=frame_index,
                    relative_frame_index="",
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    hidden_col="Hidden",
                    box_element=box,
                )
                items.append(item)
                xml_index += 1

    return items, trees, missing


def apply_existing_decisions(
    all_items: list[ReviewItem],
    decisions_by_key: dict[str, dict[str, str]],
    dataframes: dict[Path, pd.DataFrame],
) -> None:
    for item in all_items:
        record = decisions_by_key.get(item.item_key)
        if not record:
            continue
        new_hidden = record.get("new_hidden", "")
        if new_hidden:
            apply_hidden_to_source(item, new_hidden, dataframes)


def build_image_index(raw_roots: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for root in raw_roots:
        if not root.exists():
            raise FileNotFoundError(root)
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            for key in keys_for_path(path, root):
                if key:
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


def resolve_crop_image(row: pd.Series, image_index: dict[str, list[Path]]) -> Path | None:
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
            p = Path(text)
            if p.exists():
                return p
            candidates.extend(candidate_keys(text))

    group_id = get_str(row, "group_id")
    pig_id = get_str(row, "pig_id")
    behavior = get_str(row, "behavior")
    rel = get_str(row, "relative_frame_index")
    frame = get_str(row, "frame_index")
    track_id = get_str(row, "track_id", fallback_col="tracklet_id")

    for text_value in [
        group_id,
        f"{group_id}_{pig_id}",
        f"{group_id}_{behavior}",
        f"{group_id}_{track_id}",
        f"{group_id}_rel{rel}",
        f"{group_id}__rel{rel}",
        f"{group_id}_frame{frame}",
        f"{group_id}__frame{frame}",
    ]:
        if text_value:
            candidates.extend(candidate_keys(text_value))

    for key in candidates:
        paths = image_index.get(key)
        if paths:
            return choose_best_image_path(paths, row)

    return fuzzy_find_image(row, image_index)


def choose_best_image_path(paths: list[Path], row: pd.Series) -> Path:
    if len(paths) == 1:
        return paths[0]

    group_norm = normalize_key(get_str(row, "group_id"))
    pig_norm = normalize_key(get_str(row, "pig_id"))
    behavior_norm = normalize_key(get_str(row, "behavior"))
    rel = safe_int(get_str(row, "relative_frame_index"), -1)
    frame = safe_int(get_str(row, "frame_index"), -1)
    tokens = [f"rel{rel:02d}", f"rel_{rel:02d}", f"f{rel:06d}", f"frame{frame:06d}", f"frame_{frame:06d}"]

    scored = []
    for path in paths:
        text_value = normalize_key(str(path))
        score = 0
        if group_norm and group_norm in text_value:
            score += 20
        if pig_norm and pig_norm in text_value:
            score += 8
        if behavior_norm and behavior_norm in text_value:
            score += 4
        if any(tok in text_value for tok in tokens):
            score += 6
        scored.append((score, str(path), path))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][2]


def fuzzy_find_image(row: pd.Series, image_index: dict[str, list[Path]]) -> Path | None:
    group_norm = normalize_key(get_str(row, "group_id"))
    pig_norm = normalize_key(get_str(row, "pig_id"))
    behavior_norm = normalize_key(get_str(row, "behavior"))
    rel = safe_int(get_str(row, "relative_frame_index"), -1)
    frame = safe_int(get_str(row, "frame_index"), -1)
    tokens = [f"rel{rel:02d}", f"rel_{rel:02d}", f"f{rel:06d}", f"frame{frame:06d}", f"frame_{frame:06d}", str(rel)]

    best: tuple[int, str, Path] | None = None
    for paths in image_index.values():
        for path in paths:
            text_value = normalize_key(str(path))
            score = 0
            if group_norm and group_norm in text_value:
                score += 20
            if pig_norm and pig_norm in text_value:
                score += 8
            if behavior_norm and behavior_norm in text_value:
                score += 4
            if any(tok in text_value for tok in tokens):
                score += 6
            if score <= 0:
                continue
            candidate = (score, str(path), path)
            if best is None or score > best[0] or (score == best[0] and str(path) < best[1]):
                best = candidate
    return best[2] if best else None


def resolve_video_path(xml_path: Path, task_name: str, video_index: dict[str, Path]) -> Path | None:
    candidates = []
    normalized_task = normalize_video_key(task_name)
    candidates.append(normalized_task)

    clean_task_name = strip_cvat_owner_suffix(task_name)
    candidates.append(clean_task_name)
    candidates.append(Path(clean_task_name).stem)
    candidates.append(Path(clean_task_name).name)

    stem = xml_path.stem
    candidates.append(stem)
    candidates.append(stem.replace("Tracking_annotation_", ""))
    candidates.append(stem.replace("tracking_annotation_", ""))
    candidates.append(stem.replace("Classification_annotation_", ""))
    candidates.append(stem.replace("classification_annotation_", ""))

    expanded = []
    for c in candidates:
        expanded.append(c)
        expanded.append(normalize_video_key(c))
        if not str(c).lower().endswith(".mp4"):
            expanded.append(f"{c}.mp4")
            expanded.append(normalize_video_key(f"{c}.mp4"))

    for key in expanded:
        if key in video_index:
            return video_index[key]
    return None


def crop_bbox(frame, x1: float, y1: float, x2: float, y2: float, *, padding: float):
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


def resize_for_display(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    w, h = img.size
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img if size == img.size else img.resize(size, Image.Resampling.LANCZOS)


def find_hidden_col(df: pd.DataFrame) -> str | None:
    for col in ["hidden", "Hidden", "HIDDEN"]:
        if col in df.columns:
            return col
    return None


def make_csv_item_key(csv_path: Path, row_index: int, row: pd.Series) -> str:
    parts = [
        "csv_crop",
        str(csv_path.resolve()),
        str(row_index),
        get_str(row, "frame_uid"),
        get_str(row, "video_key"),
        get_str(row, "group_id"),
        get_str(row, "frame_index"),
        get_str(row, "relative_frame_index"),
        get_str(row, "track_id", fallback_col="tracklet_id"),
        get_str(row, "pig_id"),
        get_str(row, "behavior"),
    ]
    return normalize_key("|".join(parts))


def make_xml_item_key(
    xml_path: Path,
    track_id: str,
    frame_index: int,
    pig_id: str,
    behavior: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> str:
    parts = [
        "cvat_xml_video",
        str(xml_path.resolve()),
        str(track_id),
        str(frame_index),
        str(pig_id),
        str(behavior),
        f"{x1:.2f}",
        f"{y1:.2f}",
        f"{x2:.2f}",
        f"{y2:.2f}",
    ]
    return normalize_key("|".join(parts))


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
    return {key for key in keys if key}


def candidate_keys(text_value: str) -> list[str]:
    p = Path(text_value)
    return [text_value, p.name, p.stem, normalize_key(text_value), normalize_key(p.name), normalize_key(p.stem)]


def row_to_missing_record(
    source_kind: str, source_path: Path, row_index: int, row: pd.Series, *, reason: str
) -> dict[str, Any]:
    return {
        "source_kind": source_kind,
        "source_path": str(source_path),
        "row_index": row_index,
        "reason": reason,
        "source_type": get_str(row, "source_type"),
        "dataset_id": get_str(row, "dataset_id"),
        "video_key": get_str(row, "video_key"),
        "group_id": get_str(row, "group_id"),
        "frame_uid": get_str(row, "frame_uid"),
        "image_name": get_str(row, "image_name"),
        "pig_id": get_str(row, "pig_id"),
        "behavior": get_str(row, "behavior"),
        "hidden": get_str(row, "hidden", fallback_col="Hidden"),
        "frame_index": get_str(row, "frame_index"),
        "relative_frame_index": get_str(row, "relative_frame_index"),
        "track_id": get_str(row, "track_id", fallback_col="tracklet_id"),
    }


def box_attributes(box: ET.Element) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for attr in box.findall("attribute"):
        name = str(attr.attrib.get("name", "")).strip()
        value = "" if attr.text is None else str(attr.text).strip()
        if name:
            attrs[name] = value
    return attrs


def set_box_attribute(box: ET.Element, name: str, value: str) -> None:
    for attr in box.findall("attribute"):
        if str(attr.attrib.get("name", "")).strip().lower() == name.lower():
            attr.text = value
            return
    new_attr = ET.SubElement(box, "attribute")
    new_attr.set("name", name)
    new_attr.text = value


def text(root: ET.Element, path: str, *, default: str) -> str:
    value = root.findtext(path)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def get_str(row: pd.Series, col: str, fallback_col: str | None = None) -> str:
    if col in row.index and not pd.isna(row[col]):
        return str(row[col]).strip()
    if fallback_col and fallback_col in row.index and not pd.isna(row[fallback_col]):
        return str(row[fallback_col]).strip()
    return ""


def normalize_key(text_value: str) -> str:
    text_value = str(text_value).replace("\\", "/").lower()
    text_value = re.sub(r"[^a-z0-9]+", "_", text_value)
    return text_value.strip("_")


def strip_cvat_owner_suffix(value: str) -> str:
    text_value = str(value).strip()
    text_value = re.sub(r"\s*\([^)]*\)\s*$", "", text_value)
    return text_value.strip()


def normalize_video_key(value: str) -> str:
    text_value = strip_cvat_owner_suffix(value)
    text_value = Path(text_value).name
    text_value = re.sub(r"\.(mp4|avi|mov|mkv)$", "", text_value, flags=re.I)
    return text_value.strip()


def pig_id_from_label(label: str) -> str:
    match = re.search(r"(\d+)$", str(label).strip())
    if match:
        return f"ID_{match.group(1)}"
    return str(label).strip()


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def is_hidden_yes(value: Any) -> bool:
    return str(value).strip().lower() in {"yes", "true", "1", "y"}


def normalize_hidden_text(value: Any) -> str:
    return "Yes" if is_hidden_yes(value) else "No"


def safe_int(value: Any, default: int) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def sanitize(value: str) -> str:
    text_value = str(value)
    text_value = re.sub(r'[<>:"/\\|?*\s]+', "_", text_value)
    return text_value.strip("_") or "empty"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.csv and not args.xml and not args.xml_dir:
        raise ValueError("Provide at least one --csv or --xml/--xml-dir input.")

    decision_path = args.output_dir / DECISIONS_CSV_NAME
    decisions_by_key = {} if args.no_resume else load_existing_decisions(decision_path)

    csv_items, dataframes, csv_hidden_cols, missing_csv = collect_csv_items(args.csv, args.raw_root, args)
    xml_files = collect_xml_files(args)
    xml_items, trees, missing_xml = collect_xml_items(xml_files, args.video_root, args)

    all_items = csv_items + xml_items
    all_items.sort(key=lambda x: (x.source_kind, str(x.source_path), x.video_key, x.frame_index, x.track_id, x.pig_id))

    apply_existing_decisions(all_items, decisions_by_key, dataframes)

    missing = missing_csv + missing_xml
    if missing:
        pd.DataFrame(missing).to_csv(args.output_dir / "missing_hidden_review_items.csv", index=False)

    if args.include_reviewed:
        remaining_items = all_items.copy()
    else:
        remaining_items = [item for item in all_items if item.item_key not in decisions_by_key]

    if args.max_items is not None:
        remaining_items = remaining_items[: args.max_items]

    write_progress_csv(args.output_dir / PROGRESS_CSV_NAME, all_items, decisions_by_key)

    print(f"CSV crop items: {len(csv_items)}")
    print(f"CVAT XML video items: {len(xml_items)}")
    print(f"existing decisions loaded: {len(decisions_by_key)}")
    print(f"remaining items this run: {len(remaining_items)}")
    print(f"missing items: {len(missing)}")
    print(f"output dir: {args.output_dir}")

    if not remaining_items:
        print("No remaining review items. Corrected outputs/progress have been written from existing decisions.")
        write_decisions_csv(decision_path, decisions_by_key)
        for csv_path, df in dataframes.items():
            df.to_csv(args.output_dir / f"{csv_path.stem}__hidden_corrected.csv", index=False)
        if trees:
            xml_out_dir = args.output_dir / "corrected_xml"
            xml_out_dir.mkdir(parents=True, exist_ok=True)
            for xml_path, tree in trees.items():
                out_path = xml_path if args.overwrite else xml_out_dir / xml_path.name
                tree.write(out_path, encoding="utf-8", xml_declaration=True)
        return

    root = tk.Tk()
    app = UnifiedHiddenReviewApp(
        root,
        items=remaining_items,
        all_items=all_items,
        dataframes=dataframes,
        csv_hidden_cols=csv_hidden_cols,
        trees=trees,
        decisions_by_key=decisions_by_key,
        output_dir=args.output_dir,
        padding=args.padding,
        copy_reviewed_crops=args.copy_reviewed_crops,
        overwrite_xml=args.overwrite,
        autosave_corrected=args.autosave_corrected,
    )
    root.protocol("WM_DELETE_WINDOW", app.confirm_exit)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
