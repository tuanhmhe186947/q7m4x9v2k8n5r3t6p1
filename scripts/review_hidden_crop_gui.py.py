from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class ReviewItem:
    source_csv: Path
    row_index: int
    image_path: Path
    source_type: str
    dataset_id: str
    video_key: str
    group_id: str
    frame_uid: str
    image_name: str
    pig_id: str
    behavior: str
    hidden: str
    frame_index: str
    relative_frame_index: str
    track_id: str


@dataclass
class ReviewDecision:
    source_csv: str
    row_index: int
    image_path: str
    source_type: str
    dataset_id: str
    video_key: str
    group_id: str
    frame_uid: str
    image_name: str
    pig_id: str
    behavior: str
    old_hidden: str
    new_hidden: str
    decision: str
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GUI review tool for Hidden labels using already-cropped images. "
            "Use ✓ to keep Hidden=Yes and ✗ to change Hidden to No."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        action="append",
        required=True,
        help="Input CSV to review. Can be repeated.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        action="append",
        required=True,
        help="Root folder containing crop images. Can be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for corrected CSV and review logs.",
    )
    parser.add_argument(
        "--only-hidden",
        action="store_true",
        default=True,
        help="Review only rows where hidden is Yes/True/1. Default behavior.",
    )
    parser.add_argument(
        "--include-hidden-no",
        action="store_true",
        help="Also review Hidden=No rows. Normally not needed.",
    )
    parser.add_argument(
        "--source-type",
        action="append",
        default=[],
        help="Optional source_type filter. Can be repeated.",
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        default=[],
        help="Optional dataset_id filter. Can be repeated.",
    )
    parser.add_argument(
        "--group-id",
        action="append",
        default=[],
        help="Optional group_id filter. Can be repeated.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional limit for quick review.",
    )
    parser.add_argument(
        "--copy-reviewed-crops",
        action="store_true",
        help="Copy reviewed crop images into output-dir/reviewed_crops.",
    )
    return parser.parse_args()


class HiddenReviewApp:
    def __init__(
        self,
        root: tk.Tk,
        *,
        items: list[ReviewItem],
        dataframes: dict[Path, pd.DataFrame],
        output_dir: Path,
        copy_reviewed_crops: bool,
    ) -> None:
        self.root = root
        self.items = items
        self.dataframes = dataframes
        self.output_dir = output_dir
        self.copy_reviewed_crops = copy_reviewed_crops

        self.index = 0
        self.decisions: list[ReviewDecision] = []
        self.undo_stack: list[ReviewDecision] = []

        self.current_photo: ImageTk.PhotoImage | None = None

        self.root.title("Hidden Label Review - Crop Only")
        self.root.geometry("1100x850")

        self.info_var = tk.StringVar()
        self.status_var = tk.StringVar()

        self._build_ui()
        self._bind_keys()
        self._show_current()

    def _build_ui(self) -> None:
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        self.info_label = tk.Label(
            top,
            textvariable=self.info_var,
            justify=tk.LEFT,
            anchor="w",
            font=("Consolas", 11),
        )
        self.info_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.image_label = tk.Label(self.root, bg="black")
        self.image_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)

        controls = tk.Frame(self.root)
        controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        btn_keep = tk.Button(
            controls,
            text="✓ Hidden đúng - giữ Yes  [V]",
            command=self.keep_hidden,
            bg="#d7ffd7",
            font=("Arial", 12, "bold"),
            height=2,
        )
        btn_keep.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        btn_no = tk.Button(
            controls,
            text="✗ Không hidden - đổi No  [X]",
            command=self.mark_not_hidden,
            bg="#ffd7d7",
            font=("Arial", 12, "bold"),
            height=2,
        )
        btn_no.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        btn_skip = tk.Button(
            controls,
            text="Bỏ qua  [S]",
            command=self.skip_item,
            bg="#eeeeee",
            font=("Arial", 12),
            height=2,
        )
        btn_skip.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        btn_undo = tk.Button(
            controls,
            text="Undo  [U]",
            command=self.undo,
            bg="#eeeeff",
            font=("Arial", 12),
            height=2,
        )
        btn_undo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        btn_save = tk.Button(
            controls,
            text="Save & Exit  [Ctrl+S]",
            command=self.save_and_exit,
            bg="#d7e8ff",
            font=("Arial", 12, "bold"),
            height=2,
        )
        btn_save.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        bottom = tk.Label(
            self.root,
            textvariable=self.status_var,
            justify=tk.LEFT,
            anchor="w",
            font=("Consolas", 10),
        )
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=6)

    def _bind_keys(self) -> None:
        self.root.bind("v", lambda _event: self.keep_hidden())
        self.root.bind("V", lambda _event: self.keep_hidden())
        self.root.bind("x", lambda _event: self.mark_not_hidden())
        self.root.bind("X", lambda _event: self.mark_not_hidden())
        self.root.bind("s", lambda _event: self.skip_item())
        self.root.bind("S", lambda _event: self.skip_item())
        self.root.bind("u", lambda _event: self.undo())
        self.root.bind("U", lambda _event: self.undo())
        self.root.bind("<Control-s>", lambda _event: self.save_and_exit())
        self.root.bind("<Right>", lambda _event: self.skip_item())
        self.root.bind("<Escape>", lambda _event: self.confirm_exit())

    def _show_current(self) -> None:
        if self.index >= len(self.items):
            self.save_and_exit()
            return

        item = self.items[self.index]

        self.info_var.set(
            "\n".join(
                [
                    f"Item: {self.index + 1}/{len(self.items)}",
                    f"CSV: {item.source_csv}",
                    f"Image: {item.image_path}",
                    f"source_type={item.source_type} | dataset_id={item.dataset_id}",
                    f"video_key={item.video_key}",
                    f"group_id={item.group_id} | frame_index={item.frame_index} | relative={item.relative_frame_index}",
                    f"frame_uid={item.frame_uid}",
                    f"track_id={item.track_id} | pig_id={item.pig_id}",
                    f"behavior={item.behavior} | current hidden={item.hidden}",
                    "",
                    "V = giữ Hidden=Yes | X = đổi Hidden=No | S = bỏ qua | U = undo | Ctrl+S = lưu",
                ]
            )
        )

        try:
            img = Image.open(item.image_path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Image error", f"Cannot open image:\n{item.image_path}\n\n{exc}")
            self.skip_item()
            return

        img = self._resize_for_display(img)
        self.current_photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.current_photo)

        self.status_var.set(
            f"Reviewed: {len(self.decisions)} | Remaining: {len(self.items) - self.index}"
        )

    def _resize_for_display(self, img: Image.Image) -> Image.Image:
        max_w = 1040
        max_h = 650
        w, h = img.size

        scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))

        if new_size == img.size:
            return img
        return img.resize(new_size, Image.Resampling.LANCZOS)

    def keep_hidden(self) -> None:
        item = self.items[self.index]
        decision = self._make_decision(
            item,
            new_hidden="Yes",
            decision="keep_hidden_yes",
            note="reviewer_confirmed_hidden",
        )
        self._apply_decision(decision)
        self._next()

    def mark_not_hidden(self) -> None:
        item = self.items[self.index]
        decision = self._make_decision(
            item,
            new_hidden="No",
            decision="change_hidden_to_no",
            note="reviewer_rejected_hidden",
        )
        self._apply_decision(decision)
        self._next()

    def skip_item(self) -> None:
        self.index += 1
        self._show_current()

    def undo(self) -> None:
        if not self.decisions:
            return

        last = self.decisions.pop()
        self.undo_stack.append(last)

        df = self.dataframes[Path(last.source_csv)]
        if "hidden" in df.columns:
            df.loc[last.row_index, "hidden"] = last.old_hidden

        self.index = max(0, self.index - 1)
        self._show_current()

    def _next(self) -> None:
        self.index += 1
        self._show_current()

    def _make_decision(
        self,
        item: ReviewItem,
        *,
        new_hidden: str,
        decision: str,
        note: str,
    ) -> ReviewDecision:
        return ReviewDecision(
            source_csv=str(item.source_csv),
            row_index=item.row_index,
            image_path=str(item.image_path),
            source_type=item.source_type,
            dataset_id=item.dataset_id,
            video_key=item.video_key,
            group_id=item.group_id,
            frame_uid=item.frame_uid,
            image_name=item.image_name,
            pig_id=item.pig_id,
            behavior=item.behavior,
            old_hidden=item.hidden,
            new_hidden=new_hidden,
            decision=decision,
            note=note,
        )

    def _apply_decision(self, decision: ReviewDecision) -> None:
        source_csv = Path(decision.source_csv)
        df = self.dataframes[source_csv]

        df.loc[decision.row_index, "hidden"] = decision.new_hidden

        # Hidden correction should not cause review/reject by itself.
        # If the only reason was hidden, restore usable policy.
        if decision.new_hidden == "No":
            if "qa_status" in df.columns:
                hidden_mask = df.loc[decision.row_index, "qa_status"] == "hidden"
                if hidden_mask:
                    df.loc[decision.row_index, "qa_status"] = "ok"

            if "training_tier" in df.columns:
                tier = str(df.loc[decision.row_index, "training_tier"])
                if tier in {"review", "warning"}:
                    df.loc[decision.row_index, "training_tier"] = "clean"

            if "include_in_training" in df.columns:
                df.loc[decision.row_index, "include_in_training"] = True

            if "use_for_main_eval" in df.columns:
                df.loc[decision.row_index, "use_for_main_eval"] = True

        self.decisions.append(decision)

        if self.copy_reviewed_crops:
            self._copy_reviewed_crop(decision)

    def _copy_reviewed_crop(self, decision: ReviewDecision) -> None:
        src = Path(decision.image_path)
        if not src.exists():
            return

        dst_dir = (
            self.output_dir
            / "reviewed_crops"
            / sanitize(decision.decision)
            / sanitize(decision.group_id or "no_group")
        )
        dst_dir.mkdir(parents=True, exist_ok=True)

        dst_name = (
            f"{sanitize(decision.video_key)}"
            f"__{sanitize(decision.group_id)}"
            f"__row{decision.row_index}"
            f"__{sanitize(decision.pig_id)}"
            f"__{sanitize(decision.behavior)}"
            f"__{decision.old_hidden}_to_{decision.new_hidden}"
            f"{src.suffix.lower()}"
        )
        shutil.copy2(src, dst_dir / dst_name)

    def save_and_exit(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._save_decisions()
        self._save_corrected_csvs()
        messagebox.showinfo(
            "Saved",
            f"Saved review outputs to:\n{self.output_dir}",
        )
        self.root.destroy()

    def confirm_exit(self) -> None:
        answer = messagebox.askyesno(
            "Exit",
            "Save current decisions and exit?",
        )
        if answer:
            self.save_and_exit()

    def _save_decisions(self) -> None:
        path = self.output_dir / "hidden_review_decisions.csv"
        rows = [decision.__dict__ for decision in self.decisions]

        fieldnames = [
            "source_csv",
            "row_index",
            "image_path",
            "source_type",
            "dataset_id",
            "video_key",
            "group_id",
            "frame_uid",
            "image_name",
            "pig_id",
            "behavior",
            "old_hidden",
            "new_hidden",
            "decision",
            "note",
        ]

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _save_corrected_csvs(self) -> None:
        for csv_path, df in self.dataframes.items():
            out_name = csv_path.stem + "__hidden_corrected.csv"
            out_path = self.output_dir / out_name
            df.to_csv(out_path, index=False)


def load_items(
    *,
    csv_paths: list[Path],
    raw_roots: list[Path],
    args: argparse.Namespace,
) -> tuple[list[ReviewItem], dict[Path, pd.DataFrame], list[dict[str, Any]]]:
    image_index = build_image_index(raw_roots)

    dataframes: dict[Path, pd.DataFrame] = {}
    items: list[ReviewItem] = []
    missing: list[dict[str, Any]] = []

    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        df = pd.read_csv(csv_path, low_memory=False)
        dataframes[csv_path] = df

        if "hidden" not in df.columns:
            print(f"[SKIP] no hidden column: {csv_path}")
            continue

        mask = pd.Series(True, index=df.index)

        if not args.include_hidden_no:
            mask &= df["hidden"].astype(str).str.lower().isin(["yes", "true", "1"])

        if args.source_type and "source_type" in df.columns:
            mask &= df["source_type"].astype(str).isin(args.source_type)

        if args.dataset_id and "dataset_id" in df.columns:
            mask &= df["dataset_id"].astype(str).isin(args.dataset_id)

        if args.group_id and "group_id" in df.columns:
            mask &= df["group_id"].astype(str).isin(args.group_id)

        selected = df[mask].copy()

        for row_index, row in selected.iterrows():
            image_path = resolve_crop_image(row, image_index)

            if image_path is None:
                missing.append(row_to_missing_record(csv_path, row_index, row))
                continue

            items.append(
                ReviewItem(
                    source_csv=csv_path,
                    row_index=int(row_index),
                    image_path=image_path,
                    source_type=get_str(row, "source_type"),
                    dataset_id=get_str(row, "dataset_id"),
                    video_key=get_str(row, "video_key"),
                    group_id=get_str(row, "group_id"),
                    frame_uid=get_str(row, "frame_uid"),
                    image_name=get_str(row, "image_name"),
                    pig_id=get_str(row, "pig_id"),
                    behavior=get_str(row, "behavior"),
                    hidden=get_str(row, "hidden"),
                    frame_index=get_str(row, "frame_index"),
                    relative_frame_index=get_str(row, "relative_frame_index"),
                    track_id=get_str(row, "track_id", fallback_col="tracklet_id"),
                )
            )

    items.sort(
        key=lambda x: (
            str(x.source_csv),
            x.source_type,
            x.dataset_id,
            x.video_key,
            x.group_id,
            safe_int(x.relative_frame_index, 0),
            x.pig_id,
        )
    )

    if args.max_items is not None:
        items = items[: args.max_items]

    return items, dataframes, missing


def build_image_index(raw_roots: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}

    for root in raw_roots:
        if not root.exists():
            raise FileNotFoundError(root)

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTS:
                continue

            keys = keys_for_path(path, root)
            for key in keys:
                if key:
                    index.setdefault(key, []).append(path)

    return index


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

    return keys


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
        if col in row.index and not pd.isna(row[col]):
            text = str(row[col]).strip()
            if text:
                candidates.extend(candidate_keys(text))

    group_id = get_str(row, "group_id")
    pig_id = get_str(row, "pig_id")
    behavior = get_str(row, "behavior")
    rel = get_str(row, "relative_frame_index")
    frame = get_str(row, "frame_index")
    track_id = get_str(row, "track_id", fallback_col="tracklet_id")

    for text in [
        group_id,
        f"{group_id}_{pig_id}",
        f"{group_id}_{behavior}",
        f"{group_id}_{track_id}",
        f"{group_id}_rel{rel}",
        f"{group_id}__rel{rel}",
        f"{group_id}_frame{frame}",
        f"{group_id}__frame{frame}",
    ]:
        if text:
            candidates.extend(candidate_keys(text))

    # Exact lookup first.
    for key in candidates:
        paths = image_index.get(key)
        if paths:
            return choose_best_path(paths, row)

    # Fuzzy lookup.
    group_norm = normalize_key(group_id)
    pig_norm = normalize_key(pig_id)
    behavior_norm = normalize_key(behavior)
    rel_int = safe_int(rel, -1)
    frame_int = safe_int(frame, -1)

    rel_tokens = [
        f"rel{rel_int:02d}",
        f"rel_{rel_int:02d}",
        f"f{rel_int:06d}",
        f"frame{frame_int:06d}",
        f"frame_{frame_int:06d}",
        str(rel_int),
    ]

    best: tuple[int, Path] | None = None

    for paths in image_index.values():
        for path in paths:
            text = normalize_key(str(path))
            score = 0

            if group_norm and group_norm in text:
                score += 20
            if pig_norm and pig_norm in text:
                score += 8
            if behavior_norm and behavior_norm in text:
                score += 4
            if any(tok in text for tok in rel_tokens):
                score += 6

            if score <= 0:
                continue

            if best is None or score > best[0] or (score == best[0] and str(path) < str(best[1])):
                best = (score, path)

    if best is not None:
        return best[1]

    return None


def choose_best_path(paths: list[Path], row: pd.Series) -> Path:
    if len(paths) == 1:
        return paths[0]

    group_norm = normalize_key(get_str(row, "group_id"))
    pig_norm = normalize_key(get_str(row, "pig_id"))
    behavior_norm = normalize_key(get_str(row, "behavior"))
    rel = safe_int(get_str(row, "relative_frame_index"), -1)
    frame = safe_int(get_str(row, "frame_index"), -1)

    rel_tokens = [
        f"rel{rel:02d}",
        f"rel_{rel:02d}",
        f"f{rel:06d}",
        f"frame{frame:06d}",
        f"frame_{frame:06d}",
    ]

    scored = []
    for path in paths:
        text = normalize_key(str(path))
        score = 0

        if group_norm and group_norm in text:
            score += 20
        if pig_norm and pig_norm in text:
            score += 8
        if behavior_norm and behavior_norm in text:
            score += 4
        if any(tok in text for tok in rel_tokens):
            score += 6

        scored.append((score, path))

    scored.sort(key=lambda item: (-item[0], str(item[1])))
    return scored[0][1]


def candidate_keys(text: str) -> list[str]:
    p = Path(text)
    return [
        text,
        p.name,
        p.stem,
        normalize_key(text),
        normalize_key(p.name),
        normalize_key(p.stem),
    ]


def row_to_missing_record(csv_path: Path, row_index: int, row: pd.Series) -> dict[str, Any]:
    return {
        "source_csv": str(csv_path),
        "row_index": row_index,
        "source_type": get_str(row, "source_type"),
        "dataset_id": get_str(row, "dataset_id"),
        "video_key": get_str(row, "video_key"),
        "group_id": get_str(row, "group_id"),
        "frame_uid": get_str(row, "frame_uid"),
        "image_name": get_str(row, "image_name"),
        "pig_id": get_str(row, "pig_id"),
        "behavior": get_str(row, "behavior"),
        "hidden": get_str(row, "hidden"),
        "relative_frame_index": get_str(row, "relative_frame_index"),
        "track_id": get_str(row, "track_id", fallback_col="tracklet_id"),
    }


def save_missing(output_dir: Path, missing: list[dict[str, Any]]) -> None:
    if not missing:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(missing).to_csv(output_dir / "missing_images.csv", index=False)


def get_str(row: pd.Series, col: str, fallback_col: str | None = None) -> str:
    if col in row.index and not pd.isna(row[col]):
        return str(row[col])
    if fallback_col and fallback_col in row.index and not pd.isna(row[fallback_col]):
        return str(row[fallback_col])
    return ""


def normalize_key(text: str) -> str:
    text = str(text).replace("\\", "/")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def sanitize(text: str) -> str:
    text = str(text)
    text = re.sub(r'[<>:"/\\|?*\s]+', "_", text)
    return text.strip("_") or "empty"


def safe_int(value: Any, default: int) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    items, dataframes, missing = load_items(
        csv_paths=args.csv,
        raw_roots=args.raw_root,
        args=args,
    )

    save_missing(args.output_dir, missing)

    print(f"review items: {len(items)}")
    print(f"missing images: {len(missing)}")
    print(f"output dir: {args.output_dir}")

    if not items:
        print("No review items found.")
        return

    root = tk.Tk()
    app = HiddenReviewApp(
        root,
        items=items,
        dataframes=dataframes,
        output_dir=args.output_dir,
        copy_reviewed_crops=args.copy_reviewed_crops,
    )
    root.protocol("WM_DELETE_WINDOW", app.confirm_exit)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)