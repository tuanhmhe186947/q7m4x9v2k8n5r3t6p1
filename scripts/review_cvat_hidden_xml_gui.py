from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import cv2
import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass
class ReviewItem:
    xml_path: Path
    xml_index: int
    track_id: str
    track_label: str
    frame_index: int
    pig_id: str
    behavior: str
    old_hidden: str
    video_key: str
    video_path: Path
    x1: float
    y1: float
    x2: float
    y2: float
    box_element: ET.Element


@dataclass
class ReviewDecision:
    xml_path: str
    xml_index: int
    video_key: str
    video_path: str
    track_id: str
    track_label: str
    frame_index: int
    pig_id: str
    behavior: str
    old_hidden: str
    new_hidden: str
    decision: str
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GUI review Hidden labels in CVAT XML using bbox crops from video."
    )

    parser.add_argument(
        "--xml-dir",
        type=Path,
        action="append",
        default=[],
        help="Folder containing CVAT XML files. Can be repeated.",
    )
    parser.add_argument(
        "--xml",
        type=Path,
        action="append",
        default=[],
        help="Single CVAT XML file. Can be repeated.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        action="append",
        required=True,
        help="Root folder for videos. Can be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output folder for corrected XML and review logs.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.12,
        help="Extra crop padding ratio around bbox. Default: 0.12.",
    )
    parser.add_argument(
        "--include-hidden-no",
        action="store_true",
        help="Review both Hidden=Yes and Hidden=No. Default only reviews Hidden=Yes.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Limit review items for quick test.",
    )
    parser.add_argument(
        "--copy-reviewed-crops",
        action="store_true",
        help="Copy reviewed crop images to output folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite original XML files. Default saves corrected copies only.",
    )

    return parser.parse_args()


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


class CvatHiddenReviewApp:
    def __init__(
        self,
        root: tk.Tk,
        *,
        items: list[ReviewItem],
        trees: dict[Path, ET.ElementTree],
        output_dir: Path,
        padding: float,
        copy_reviewed_crops: bool,
        overwrite: bool,
    ) -> None:
        self.root = root
        self.items = items
        self.trees = trees
        self.output_dir = output_dir
        self.padding = padding
        self.copy_reviewed_crops = copy_reviewed_crops
        self.overwrite = overwrite

        self.index = 0
        self.decisions: list[ReviewDecision] = []
        self.reader = VideoFrameReader()
        self.current_photo: ImageTk.PhotoImage | None = None
        self.current_crop_bgr = None

        self.info_var = tk.StringVar()
        self.status_var = tk.StringVar()

        self.root.title("CVAT XML Hidden Review")
        self.root.geometry("1120x860")

        self._build_ui()
        self._bind_keys()
        self._show_current()

    def _build_ui(self) -> None:
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        tk.Label(
            top,
            textvariable=self.info_var,
            justify=tk.LEFT,
            anchor="w",
            font=("Consolas", 11),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.image_label = tk.Label(self.root, bg="black")
        self.image_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)

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

        tk.Button(
            controls,
            text="Skip [S]",
            command=self.skip,
            bg="#eeeeee",
            font=("Arial", 12),
            height=2,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        tk.Button(
            controls,
            text="Undo [U]",
            command=self.undo,
            bg="#eeeeff",
            font=("Arial", 12),
            height=2,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        tk.Button(
            controls,
            text="Save & Exit [Ctrl+S]",
            command=self.save_and_exit,
            bg="#d7e8ff",
            font=("Arial", 12, "bold"),
            height=2,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        tk.Label(
            self.root,
            textvariable=self.status_var,
            justify=tk.LEFT,
            anchor="w",
            font=("Consolas", 10),
        ).pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=6)

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

        frame = self.reader.read_frame(item.video_path, item.frame_index)
        if frame is None:
            messagebox.showwarning(
                "Frame missing",
                f"Cannot read frame {item.frame_index} from:\n{item.video_path}",
            )
            self.skip()
            return

        crop = crop_bbox(frame, item.x1, item.y1, item.x2, item.y2, padding=self.padding)
        self.current_crop_bgr = crop.copy()

        display = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display)
        img = self._resize_for_display(img)

        self.current_photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.current_photo)

        self.info_var.set(
            "\n".join(
                [
                    f"Item: {self.index + 1}/{len(self.items)}",
                    f"XML: {item.xml_path}",
                    f"Video: {item.video_path}",
                    f"video_key={item.video_key}",
                    f"track_id={item.track_id} | label={item.track_label}",
                    f"frame={item.frame_index} | pig_id={item.pig_id}",
                    f"behavior={item.behavior} | current Hidden={item.old_hidden}",
                    f"bbox=({item.x1:.1f}, {item.y1:.1f}, {item.x2:.1f}, {item.y2:.1f})",
                    "",
                    "V = giữ Hidden=Yes | X = đổi Hidden=No | S = bỏ qua | U = undo | Ctrl+S = lưu",
                ]
            )
        )

        self.status_var.set(
            f"Reviewed decisions: {len(self.decisions)} | Remaining: {len(self.items) - self.index}"
        )

    def _resize_for_display(self, img: Image.Image) -> Image.Image:
        max_w = 1040
        max_h = 650
        w, h = img.size
        scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        if size == img.size:
            return img
        return img.resize(size, Image.Resampling.LANCZOS)

    def keep_hidden(self) -> None:
        item = self.items[self.index]
        self._apply(item, new_hidden="Yes", decision="keep_hidden_yes")
        self._next()

    def mark_not_hidden(self) -> None:
        item = self.items[self.index]
        self._apply(item, new_hidden="No", decision="change_hidden_to_no")
        self._next()

    def skip(self) -> None:
        self.index += 1
        self._show_current()

    def undo(self) -> None:
        if not self.decisions:
            return

        last = self.decisions.pop()
        item = self.items[max(0, self.index - 1)]

        set_box_attribute(item.box_element, "Hidden", last.old_hidden)
        self.index = max(0, self.index - 1)
        self._show_current()

    def _next(self) -> None:
        self.index += 1
        self._show_current()

    def _apply(self, item: ReviewItem, *, new_hidden: str, decision: str) -> None:
        set_box_attribute(item.box_element, "Hidden", new_hidden)

        record = ReviewDecision(
            xml_path=str(item.xml_path),
            xml_index=item.xml_index,
            video_key=item.video_key,
            video_path=str(item.video_path),
            track_id=item.track_id,
            track_label=item.track_label,
            frame_index=item.frame_index,
            pig_id=item.pig_id,
            behavior=item.behavior,
            old_hidden=item.old_hidden,
            new_hidden=new_hidden,
            decision=decision,
            note="manual_gui_review",
        )
        self.decisions.append(record)

        if self.copy_reviewed_crops and self.current_crop_bgr is not None:
            self._save_crop(record)

    def _save_crop(self, record: ReviewDecision) -> None:
        out_dir = (
            self.output_dir
            / "reviewed_crops"
            / sanitize(record.decision)
            / sanitize(record.video_key)
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        out_name = (
            f"{sanitize(record.video_key)}"
            f"__frame{record.frame_index:06d}"
            f"__track{sanitize(record.track_id)}"
            f"__{sanitize(record.pig_id)}"
            f"__{sanitize(record.behavior)}"
            f"__{record.old_hidden}_to_{record.new_hidden}.jpg"
        )
        cv2.imwrite(str(out_dir / out_name), self.current_crop_bgr)

    def save_and_exit(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._save_decisions()
        self._save_xmls()
        self.reader.close()
        messagebox.showinfo("Saved", f"Saved outputs to:\n{self.output_dir}")
        self.root.destroy()

    def confirm_exit(self) -> None:
        if messagebox.askyesno("Exit", "Save decisions and corrected XMLs?"):
            self.save_and_exit()

    def _save_decisions(self) -> None:
        out_path = self.output_dir / "hidden_review_decisions.csv"
        rows = [d.__dict__ for d in self.decisions]

        fields = [
            "xml_path",
            "xml_index",
            "video_key",
            "video_path",
            "track_id",
            "track_label",
            "frame_index",
            "pig_id",
            "behavior",
            "old_hidden",
            "new_hidden",
            "decision",
            "note",
        ]

        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _save_xmls(self) -> None:
        xml_out_dir = self.output_dir / "corrected_xml"
        xml_out_dir.mkdir(parents=True, exist_ok=True)

        changed_xmls = {Path(d.xml_path) for d in self.decisions}

        for xml_path, tree in self.trees.items():
            if xml_path not in changed_xmls:
                continue

            if self.overwrite:
                out_path = xml_path
            else:
                out_path = xml_out_dir / xml_path.name

            tree.write(out_path, encoding="utf-8", xml_declaration=True)


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
    unique = []
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)

    return unique


def build_video_index(video_roots: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}

    for root in video_roots:
        if not root.exists():
            raise FileNotFoundError(root)

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in VIDEO_EXTS:
                continue

            keys = {
                path.name,
                path.stem,
                normalize_video_key(path.name),
                normalize_video_key(path.stem),
            }

            for key in keys:
                if key and key not in index:
                    index[key] = path

    return index


def load_review_items(
    xml_files: list[Path],
    video_index: dict[str, Path],
    *,
    include_hidden_no: bool,
    max_items: int | None,
) -> tuple[list[ReviewItem], dict[Path, ET.ElementTree], list[dict[str, Any]]]:
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
                    "xml_path": str(xml_path),
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

                if not include_hidden_no and not is_hidden_yes(hidden):
                    continue

                frame_index = safe_int(box.attrib.get("frame", "0"), 0)
                pig_id = attrs.get("ID", attrs.get("id", fallback_pig_id))
                behavior = attrs.get("Behavior", attrs.get("behavior", ""))

                item = ReviewItem(
                    xml_path=xml_path,
                    xml_index=xml_index,
                    track_id=track_id,
                    track_label=track_label,
                    frame_index=frame_index,
                    pig_id=pig_id,
                    behavior=behavior,
                    old_hidden=normalize_hidden_text(hidden),
                    video_key=video_key,
                    video_path=video_path,
                    x1=safe_float(box.attrib.get("xtl"), 0.0),
                    y1=safe_float(box.attrib.get("ytl"), 0.0),
                    x2=safe_float(box.attrib.get("xbr"), 0.0),
                    y2=safe_float(box.attrib.get("ybr"), 0.0),
                    box_element=box,
                )
                items.append(item)
                xml_index += 1

    items.sort(
        key=lambda x: (
            x.video_key,
            x.frame_index,
            safe_int(x.track_id, 0),
            x.pig_id,
        )
    )

    if max_items is not None:
        items = items[:max_items]

    return items, trees, missing


def resolve_video_path(
    xml_path: Path,
    task_name: str,
    video_index: dict[str, Path],
) -> Path | None:
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


def save_missing(output_dir: Path, missing: list[dict[str, Any]]) -> None:
    if not missing:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(missing).to_csv(output_dir / "missing_videos_or_items.csv", index=False)


def text(root: ET.Element, path: str, *, default: str) -> str:
    value = root.findtext(path)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


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
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float) -> float:
    try:
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

    xml_files = collect_xml_files(args)
    if not xml_files:
        raise FileNotFoundError("No XML files found. Use --xml-dir or --xml.")

    print(f"xml files: {len(xml_files)}")
    print("building video index...")
    video_index = build_video_index(args.video_root)
    print(f"videos indexed: {len(video_index)}")

    items, trees, missing = load_review_items(
        xml_files,
        video_index,
        include_hidden_no=args.include_hidden_no,
        max_items=args.max_items,
    )

    save_missing(args.output_dir, missing)

    print(f"review items: {len(items)}")
    print(f"missing videos/items: {len(missing)}")
    print(f"output dir: {args.output_dir}")

    if not items:
        print("No review items found.")
        return

    root = tk.Tk()
    app = CvatHiddenReviewApp(
        root,
        items=items,
        trees=trees,
        output_dir=args.output_dir,
        padding=args.padding,
        copy_reviewed_crops=args.copy_reviewed_crops,
        overwrite=args.overwrite,
    )
    root.protocol("WM_DELETE_WINDOW", app.confirm_exit)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)