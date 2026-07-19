from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import tempfile
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageTk

from pig_behavior.classification_v2.features.roi import load_scene_rois_from_coco
from pig_behavior.classification_v2.review.behavior_review_contract import (
    CANONICAL_BEHAVIORS,
    audit_manifest_alignment,
    audit_review_unit_contract,
    canonicalize_decisions,
    validate_decision_semantics,
)

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


VALID_BEHAVIORS = sorted(CANONICAL_BEHAVIORS)


@dataclass(slots=True)
class GuiConfig:
    review_units_csv: Path
    frame_features_csv: Path
    output_dir: Path
    video_root: Path | None = None
    raw_root: Path | None = None
    roi_coco_path: Path | None = None
    source_type: str | None = None
    max_items: int | None = None
    padding: float = 0.8
    copy_contact_sheets: bool = False
    thumb_w: int = 220
    thumb_h: int = 160


def safe_filename(value: object, max_len: int = 150) -> str:
    s = str(value)
    for ch in '<>:"/\\|?*':
        s = s.replace(ch, "_")
    s = s.replace("\n", "_").replace("\r", "_").replace("\t", "_")
    s = "_".join(s.split()).strip(" ._")
    if not s:
        s = hashlib.md5(str(value).encode("utf-8", errors="ignore")).hexdigest()[:12]
    return s[:max_len]


def _write_csv_atomic(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    """Replace a decision CSV atomically so interruption cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    finally:
        if temporary_name and Path(temporary_name).exists():
            Path(temporary_name).unlink()


class ReviewUnitGui:
    def __init__(self, config: GuiConfig) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.copy_contact_sheets:
            (self.config.output_dir / "contact_sheets").mkdir(parents=True, exist_ok=True)

        self.units = self._load_units(config.review_units_csv)
        self.frames = pd.read_csv(config.frame_features_csv, low_memory=False)
        self.frames["frame_index"] = pd.to_numeric(self.frames.get("frame_index"), errors="coerce")
        if "relative_frame_index" in self.frames.columns:
            self.frames["relative_frame_index"] = pd.to_numeric(
                self.frames["relative_frame_index"],
                errors="coerce",
            )

        self.current = 0
        self.decisions = self._load_existing_decisions()
        self.video_cache: dict[str, Any] = {}
        self.video_index = self._build_video_index(config.video_root)
        self.roi_overlays = self._load_roi_overlays(config.roi_coco_path)

        self.root = tk.Tk()
        self.root.title("classification_v2 temporal review unit GUI")
        self.root.geometry("1220x880")
        self.root.minsize(1000, 720)

        self._build_layout()
        self.show_current()
        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)
        self.root.mainloop()

    def _load_units(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, low_memory=False)
        required = [
            "review_unit_id",
            "review_unit_type",
            "source_type",
            "dataset_id",
            "video_key",
            "pig_id",
            "unit_start_frame",
            "unit_end_frame",
            "display_frame_indices",
            "behavior_label",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise SystemExit(f"Review unit CSV missing required columns: {missing}")
        contract = audit_review_unit_contract(df)
        if contract["errors"]:
            raise SystemExit(
                "Review unit contract failed: " + "; ".join(contract["errors"])
            )
        if self.config.source_type:
            df = df[df["source_type"].astype(str).eq(self.config.source_type)].copy()
        if "review_priority" in df.columns:
            df = df.sort_values(["review_priority"], ascending=False)
        if self.config.max_items is not None and self.config.max_items > 0:
            df = df.head(self.config.max_items).copy()
        if df.empty:
            raise SystemExit("No review units after filtering.")
        return df.reset_index(drop=True)

    def _load_existing_decisions(self) -> dict[str, dict[str, Any]]:
        """Resume one review template without losing decisions from prior sessions."""
        path = self.config.output_dir / "behavior_unit_review_decisions.csv"
        if not path.exists():
            return {}

        existing = pd.read_csv(path, low_memory=False)
        if existing.empty:
            return {}
        if "review_unit_id" not in existing.columns:
            raise SystemExit(f"Existing decision CSV missing review_unit_id: {path}")

        ids = existing["review_unit_id"].fillna("").astype(str).str.strip()
        if ids.eq("").any():
            raise SystemExit(f"Existing decision CSV has blank review_unit_id: {path}")
        if ids.duplicated().any():
            duplicate_count = int(ids.duplicated(keep=False).sum())
            raise SystemExit(
                f"Existing decision CSV has duplicate review_unit_id rows={duplicate_count}: {path}"
            )

        expected_ids = set(self.units["review_unit_id"].astype(str))
        unexpected = sorted(set(ids) - expected_ids)
        if unexpected:
            raise SystemExit(
                "Existing decision CSV does not match the selected review template; "
                f"unexpected review_unit_id count={len(unexpected)}"
            )

        existing, normalization_warnings = canonicalize_decisions(existing)
        semantic_errors, semantic_warnings = validate_decision_semantics(
            existing,
            require_complete=False,
        )
        alignment_errors, _ = audit_manifest_alignment(
            self.units,
            existing,
            allow_blank_snapshot=False,
        )
        strength_errors = [
            warning
            for warning in semantic_warnings
            if warning.startswith("active_decision_without_strength")
        ]
        errors = semantic_errors + alignment_errors + strength_errors
        if errors:
            raise SystemExit(
                "Existing decision CSV violates review contract: "
                + "; ".join(sorted(set(errors)))
            )
        if normalization_warnings:
            print("[RESUME WARN] " + "; ".join(normalization_warnings))

        decisions: dict[str, dict[str, Any]] = {}
        for record in existing.to_dict(orient="records"):
            cleaned = {
                key: "" if pd.isna(value) else value
                for key, value in record.items()
            }
            decisions[str(cleaned["review_unit_id"]).strip()] = cleaned
        print(f"[RESUME] loaded {len(decisions)} decisions from {path}")
        return decisions

    def _build_video_index(self, root: Path | None) -> dict[str, Path]:
        """Build a tolerant video lookup index.

        CVAT rows often store video_key without the ``_30fps`` suffix,
        while the actual file on disk is named like ``Pigs291119_000231_30fps.mp4``.
        Index both the exact stem and common aliases so GUI frame loading does not
        fail for otherwise valid CVAT units.
        """
        index: dict[str, Path] = {}
        if root is None or not root.exists():
            return index

        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v"}

        def add_alias(alias: str, path: Path) -> None:
            key = str(alias).replace("\\", "/").strip().lower()
            if not key:
                return
            index.setdefault(key, path)
            # Also index extension-stripped form.
            index.setdefault(Path(key).stem.lower(), path)

        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in video_exts:
                continue

            name = p.name.lower()
            stem = p.stem.lower()
            add_alias(name, p)
            add_alias(stem, p)

            # Common CVAT mismatch: video_key=Pigs291119_000231 but file is
            # Pigs291119_000231_30fps.mp4.
            if stem.endswith("_30fps"):
                base = stem[: -len("_30fps")]
                add_alias(base, p)
                add_alias(base + p.suffix.lower(), p)
                add_alias(base + ".mp4", p)

            # Also support stems that include fps as a token with different case/name.
            for suffix in ["_30fps", "-30fps", " 30fps"]:
                if stem.endswith(suffix):
                    base = stem[: -len(suffix)]
                    add_alias(base, p)
                    add_alias(base + p.suffix.lower(), p)

        return index

    def _load_roi_overlays(self, path: Path | None) -> list[Any]:
        if path is None or not path.exists():
            return []
        try:
            return load_scene_rois_from_coco(path)
        except Exception:
            return []

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        self.header = tk.StringVar()
        header_lbl = ttk.Label(
            self.root,
            textvariable=self.header,
            font=("Segoe UI", 11, "bold"),
            wraplength=1160,
        )
        header_lbl.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 2))

        self.main_frame = ttk.Frame(self.root)
        self.main_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=2)
        self.main_frame.columnconfigure(0, weight=3)
        self.main_frame.columnconfigure(1, weight=2)
        self.main_frame.rowconfigure(0, weight=1)

        self.image_label = ttk.Label(self.main_frame, anchor="center")
        self.image_label.grid(row=0, column=0, sticky="nsew")

        right = ttk.Frame(self.main_frame)
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self.info_text = tk.Text(right, height=22, wrap="word")
        self.info_text.grid(row=0, column=0, sticky="nsew")

        form = ttk.Frame(right)
        form.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        form.columnconfigure(1, weight=1)

        self.decision_var = tk.StringVar(value="pending")
        self.strength_var = tk.StringVar(value="")
        self.corrected_var = tk.StringVar(value="")
        self.action_var = tk.StringVar(value="")
        self.weight_var = tk.StringVar(value="")
        self.note_var = tk.StringVar(value="")

        row = 0
        ttk.Label(form, text="Decision").grid(row=row, column=0, sticky="w")
        ttk.Combobox(
            form,
            textvariable=self.decision_var,
            values=["pending", "accept", "corrected", "exclude"],
            state="readonly",
        ).grid(row=row, column=1, sticky="ew")
        row += 1
        ttk.Label(form, text="Corrected behavior").grid(row=row, column=0, sticky="w")
        ttk.Combobox(
            form,
            textvariable=self.corrected_var,
            values=["", *VALID_BEHAVIORS],
            state="readonly",
        ).grid(row=row, column=1, sticky="ew")
        row += 1
        ttk.Label(form, text="Strength").grid(row=row, column=0, sticky="w")
        ttk.Combobox(
            form,
            textvariable=self.strength_var,
            values=["", "strong", "medium", "weak", "boundary"],
            state="readonly",
        ).grid(row=row, column=1, sticky="ew")
        row += 1
        ttk.Label(form, text="Training action").grid(row=row, column=0, sticky="w")
        ttk.Combobox(
            form,
            textvariable=self.action_var,
            values=[
                "",
                "main_train",
                "low_weight_train",
                "exclude",
                "review_later",
            ],
            state="readonly",
        ).grid(row=row, column=1, sticky="ew")
        row += 1
        ttk.Label(form, text="Weight").grid(row=row, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.weight_var).grid(row=row, column=1, sticky="ew")
        row += 1
        ttk.Label(form, text="Note").grid(row=row, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.note_var).grid(row=row, column=1, sticky="ew")

        bottom = ttk.Frame(self.root)
        bottom.grid(row=2, column=0, sticky="ew", padx=8, pady=8)
        for i in range(8):
            bottom.columnconfigure(i, weight=1)
        ttk.Button(bottom, text="< Prev", command=self.prev_item).grid(
            row=0, column=0, sticky="ew", padx=3
        )
        ttk.Button(bottom, text="Save", command=self.save_current).grid(
            row=0, column=1, sticky="ew", padx=3
        )
        ttk.Button(bottom, text="Save + Next", command=self.save_next).grid(
            row=0, column=2, sticky="ew", padx=3
        )
        ttk.Button(bottom, text="Accept strong + Next", command=self.accept_strong_next).grid(
            row=0, column=3, sticky="ew", padx=3
        )
        ttk.Button(bottom, text="Exclude + Next", command=self.exclude_next).grid(
            row=0, column=4, sticky="ew", padx=3
        )
        ttk.Button(bottom, text="Next >", command=self.next_item).grid(
            row=0, column=5, sticky="ew", padx=3
        )
        ttk.Button(bottom, text="Write CSV", command=self.write_decisions).grid(
            row=0, column=6, sticky="ew", padx=3
        )
        ttk.Button(bottom, text="Quit", command=self.on_quit).grid(
            row=0, column=7, sticky="ew", padx=3
        )

    def current_unit(self) -> pd.Series:
        return self.units.iloc[self.current]

    def show_current(self) -> None:
        unit = self.current_unit()
        unit_id = str(unit["review_unit_id"])
        self._load_existing_decision(unit_id)

        frames = self._frame_rows_for_unit(unit)
        image, diagnostics = self._make_contact_sheet(unit, frames)
        self.current_diagnostics = diagnostics
        self._photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self._photo)

        if self.config.copy_contact_sheets:
            safe = safe_filename(unit_id)
            output_path = (
                self.config.output_dir
                / "contact_sheets"
                / f"{self.current:05d}_{safe}.jpg"
            )
            image.save(output_path, quality=92)

        display_indices = str(unit.get("display_frame_indices", ""))
        self.header.set(
            f"{self.current + 1}/{len(self.units)} | {unit.get('review_unit_type')} | "
            f"{unit.get('source_type')} | {unit.get('behavior_label')} | "
            f"frames {unit.get('unit_start_frame')}-{unit.get('unit_end_frame')} | "
            f"shown [{display_indices}]"
        )

        self.info_text.delete("1.0", "end")
        info = self._format_info(unit, diagnostics, frames)
        self.info_text.insert("1.0", info)

    def _load_existing_decision(self, unit_id: str) -> None:
        d = self.decisions.get(unit_id, {})
        self.decision_var.set(str(d.get("manual_review_decision", "pending")))
        self.corrected_var.set(str(d.get("manual_corrected_behavior", "")))
        self.strength_var.set(str(d.get("manual_label_strength", "")))
        self.action_var.set(str(d.get("manual_training_action", "")))
        self.weight_var.set(str(d.get("manual_sample_weight", "")))
        self.note_var.set(str(d.get("manual_note", "")))

    def _format_info(self, unit: pd.Series, diagnostics: list[str], frames: pd.DataFrame) -> str:
        keys = [
            "review_item_id",
            "review_unit_id",
            "review_unit_type",
            "review_template",
            "review_reason",
            "review_priority",
            "review_evidence_available",
            "review_motion_evidence_available",
            "review_roi_evidence_available",
            "review_social_evidence_available",
            "review_posture_evidence_available",
            "review_relevant_evidence_available",
            "review_evidence_quality_score",
            "review_evidence_insufficiency_score",
            "review_motion_support_score",
            "review_roi_support_score",
            "review_social_support_score",
            "review_posture_transition_score",
            "review_evidence_conflict_score",
            "review_evidence_priority_auto",
            "review_confusion_pairs_auto",
            "review_evidence_reason_auto",
            "review_evidence_status_auto",
            "apply_scope",
            "source_type",
            "dataset_id",
            "video_key",
            "pig_id",
            "track_id",
            "object_track_key",
            "behavior_label",
            "temporal_consistency_status",
            "affected_window_count",
            "affected_window_lengths",
            "affected_main_train_windows",
            "review_reasons_window",
            "review_templates_hit",
        ]
        keys.extend(
            sorted(
                key
                for key in unit.index
                if str(key).startswith("review_pig_")
            )
        )
        keys.extend(
            [
                "behavior_review_cohort",
                "behavior_sampling_design",
                "behavior_sampling_probability",
                "behavior_sampling_weight",
            ]
        )
        lines = []
        for k in keys:
            if k in unit.index:
                lines.append(f"{k}: {unit.get(k)}")
        lines.append("")
        lines.append(f"matched frame rows: {len(frames)}")
        if diagnostics:
            lines.append("diagnostics:")
            lines.extend(f"- {x}" for x in diagnostics[:30])
        return "\n".join(lines)

    def _frame_rows_for_unit(self, unit: pd.Series) -> pd.DataFrame:
        df = self.frames
        mask = (
            df["source_type"].astype(str).eq(str(unit["source_type"]))
            & df["dataset_id"].astype(str).eq(str(unit["dataset_id"]))
            & df["video_key"].astype(str).eq(str(unit["video_key"]))
            & df["pig_id"].astype(str).eq(str(unit["pig_id"]))
        )
        if "object_track_key" in df.columns and pd.notna(unit.get("object_track_key")):
            mask &= df["object_track_key"].astype(str).eq(str(unit.get("object_track_key")))
        elif "track_id" in df.columns and pd.notna(unit.get("track_id")):
            mask &= df["track_id"].astype(str).eq(str(unit.get("track_id")))

        wanted = self._all_display_frames(unit)
        if wanted:
            mask &= df["frame_index"].isin(wanted)
        else:
            start = int(unit["unit_start_frame"])
            end = int(unit["unit_end_frame"])
            mask &= df["frame_index"].between(start, end)
        out = df.loc[mask].copy()
        return out.sort_values("frame_index")

    def _display_frames(self, unit: pd.Series) -> list[int]:
        return self._parse_frame_indices(unit.get("display_frame_indices", ""))

    def _history_display_frames(self, unit: pd.Series) -> list[int]:
        try:
            available = float(unit.get("review_pig_history_available_ratio", 0.0))
        except (TypeError, ValueError):
            available = 0.0
        if available < 1.0:
            return []
        return self._parse_frame_indices(
            unit.get("review_pig_history_display_frame_indices", "")
        )

    def _all_display_frames(self, unit: pd.Series) -> list[int]:
        return self._history_display_frames(unit) + self._display_frames(unit)

    @staticmethod
    def _parse_frame_indices(value: object) -> list[int]:
        text = str(value)
        vals = []
        for token in text.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                vals.append(int(float(token)))
            except Exception:
                pass
        return list(dict.fromkeys(vals))

    def _make_contact_sheet(
        self,
        unit: pd.Series,
        rows: pd.DataFrame,
    ) -> tuple[Image.Image, list[str]]:
        wanted = self._all_display_frames(unit)
        if not wanted:
            try:
                wanted = list(range(int(unit["unit_start_frame"]), int(unit["unit_end_frame"]) + 1))
            except Exception:
                wanted = []

        diagnostics: list[str] = []
        thumbs: list[tuple[int, Image.Image, str]] = []
        row_by_frame = {
            int(record["frame_index"]): record
            for _, record in rows.iterrows()
            if pd.notna(record.get("frame_index"))
        }
        for frame_idx in wanted:
            row = row_by_frame.get(frame_idx)
            if row is None:
                img = self._placeholder(f"NO ROW\nf{frame_idx}")
                thumbs.append((frame_idx, img, "no_row"))
                diagnostics.append(f"no frame row for frame_index={frame_idx}")
                continue
            img, msg = self._image_for_row(unit, row)
            if msg:
                diagnostics.append(f"f{frame_idx}: {msg}")
            thumbs.append((frame_idx, img, msg or "ok"))

        if not thumbs:
            thumbs.append((0, self._placeholder("NO FRAMES"), "no_frames"))

        n = len(thumbs)
        if n <= 6:
            cols = 3
        elif n <= 8:
            cols = 4
        else:
            cols = 4
        rows_n = math.ceil(n / cols)
        tw, th = self.config.thumb_w, self.config.thumb_h
        sheet = Image.new("RGB", (cols * tw, rows_n * th), "white")
        for idx, (frame_idx, img, msg) in enumerate(thumbs):
            thumb = self._fit_image(img, tw, th - 24)
            x = (idx % cols) * tw
            y = (idx // cols) * th
            sheet.paste(thumb, (x + (tw - thumb.width) // 2, y + 22))
            draw = ImageDraw.Draw(sheet)
            draw.rectangle([x, y, x + tw - 1, y + th - 1], outline="black")
            role = "H" if frame_idx in self._history_display_frames(unit) else "T"
            label = f"{role}f{frame_idx}"
            if msg and msg != "ok":
                label += f" | {msg[:22]}"
            draw.text((x + 4, y + 4), label, fill="black")

        max_w, max_h = 760, 650
        if sheet.width > max_w or sheet.height > max_h:
            sheet.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        return sheet, diagnostics

    def _image_for_row(self, unit: pd.Series, row: pd.Series) -> tuple[Image.Image, str]:
        source = str(unit.get("source_type"))
        if source == "legacy_recovered":
            img = self._legacy_crop_image(row)
            if img is not None:
                return img, ""
            return self._placeholder("NO LEGACY CROP"), "missing_legacy_crop_path"
        img = self._cvat_crop_image(unit, row)
        if img is not None:
            return img, ""
        # fallback for any source with crop_path
        img = self._legacy_crop_image(row)
        if img is not None:
            return img, ""
        return self._placeholder("NO IMAGE FOUND"), "missing_video_or_crop"

    def _legacy_crop_image(self, row: pd.Series) -> Image.Image | None:
        for col in ["crop_path", "image_path", "frame_path"]:
            if col not in row.index or pd.isna(row.get(col)):
                continue
            path = Path(str(row.get(col)))
            candidates = [path]
            # Some older CSVs pointed to outputs while the user may keep crops under data/raw.
            if self.config.raw_root and "crops" in path.parts:
                try:
                    i = list(path.parts).index("crops")
                    rel = Path(*path.parts[i + 1 :])
                    candidates.append(self.config.raw_root / rel)
                except Exception:
                    pass
            for cand in candidates:
                if cand.exists():
                    try:
                        return Image.open(cand).convert("RGB")
                    except Exception:
                        continue
        return None

    def _cvat_crop_image(
        self,
        unit: pd.Series,
        row: pd.Series,
    ) -> Image.Image | None:
        if cv2 is None:
            return None
        video_path = self._resolve_video_path(row)
        if video_path is None:
            return None
        frame_idx = int(row.get("frame_index"))
        cap = self.video_cache.get(str(video_path))
        if cap is None:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            self.video_cache[str(video_path)] = cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if str(unit.get("review_template", "")) == "interaction":
            return self._interaction_context_image(frame, row)
        h, w = frame.shape[:2]
        x1 = float(row.get("x1", 0))
        y1 = float(row.get("y1", 0))
        x2 = float(row.get("x2", w))
        y2 = float(row.get("y2", h))
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        pad = float(self.config.padding)
        xx1 = max(0, int(x1 - bw * pad))
        yy1 = max(0, int(y1 - bh * pad))
        xx2 = min(w, int(x2 + bw * pad))
        yy2 = min(h, int(y2 + bh * pad))
        crop = frame[yy1:yy2, xx1:xx2]
        if crop.size == 0:
            return None
        img = Image.fromarray(crop).convert("RGB")
        draw = ImageDraw.Draw(img)
        self._draw_roi_overlays(draw, row, w, h, xx1, yy1)
        # Actor bbox inside crop.
        ax1, ay1, ax2, ay2 = int(x1 - xx1), int(y1 - yy1), int(x2 - xx1), int(y2 - yy1)
        draw.rectangle([ax1, ay1, ax2, ay2], outline="red", width=3)
        return img

    def _interaction_context_image(self, frame: Any, actor: pd.Series) -> Image.Image:
        """Draw full-frame actor and partner context without using target labels."""
        image = Image.fromarray(frame).convert("RGB")
        draw = ImageDraw.Draw(image)
        frame_index = pd.to_numeric(actor.get("frame_index"), errors="coerce")
        candidates = self.frames[
            self.frames["source_type"].astype(str).eq(str(actor.get("source_type", "")))
            & self.frames["dataset_id"].astype(str).eq(str(actor.get("dataset_id", "")))
            & self.frames["video_key"].astype(str).eq(str(actor.get("video_key", "")))
            & self.frames["frame_index"].eq(frame_index)
        ].copy()

        actor_key = self._context_identity(actor)
        actor_center = self._bbox_center(actor)
        nearest_index: Any = None
        nearest_distance = float("inf")
        for index, candidate in candidates.iterrows():
            candidate_key = self._context_identity(candidate)
            if candidate_key == actor_key:
                continue
            center = self._bbox_center(candidate)
            if center is None or actor_center is None:
                continue
            distance = math.dist(actor_center, center)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_index = index

        for index, candidate in candidates.iterrows():
            box = self._bbox_coordinates(candidate)
            if box is None:
                continue
            candidate_key = self._context_identity(candidate)
            is_actor = candidate_key == actor_key
            color = "red" if is_actor else "#00b050" if index == nearest_index else "yellow"
            width = 4 if is_actor or index == nearest_index else 2
            draw.rectangle(box, outline=color, width=width)
            role = "actor" if is_actor else "nearest" if index == nearest_index else "other"
            identity = str(candidate.get("pig_id", candidate.get("track_id", "")))
            draw.text((box[0] + 3, max(0, box[1] - 15)), f"{role}:{identity}", fill=color)
        return image

    @staticmethod
    def _bbox_coordinates(row: pd.Series) -> tuple[int, int, int, int] | None:
        values = pd.to_numeric(
            pd.Series([row.get("x1"), row.get("y1"), row.get("x2"), row.get("y2")]),
            errors="coerce",
        )
        invalid = (
            values.isna().any()
            or values.iloc[2] <= values.iloc[0]
            or values.iloc[3] <= values.iloc[1]
        )
        if invalid:
            return None
        return tuple(int(value) for value in values.tolist())

    @staticmethod
    def _context_identity(row: pd.Series) -> str:
        """Identify an actor locally without assuming identity across videos."""
        object_key = str(row.get("object_track_key", "")).strip()
        if object_key and object_key.lower() not in {"nan", "none", "<na>"}:
            return f"object:{object_key}"
        track_id = str(row.get("track_id", "")).strip()
        pig_id = str(row.get("pig_id", "")).strip()
        return f"track:{track_id}|pig:{pig_id}"

    @classmethod
    def _bbox_center(cls, row: pd.Series) -> tuple[float, float] | None:
        box = cls._bbox_coordinates(row)
        if box is None:
            return None
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def _draw_roi_overlays(
        self,
        draw: ImageDraw.ImageDraw,
        row: pd.Series,
        frame_w: int,
        frame_h: int,
        crop_x1: int,
        crop_y1: int,
    ) -> None:
        if not self.roi_overlays:
            return
        colors = {"feeder": "#00b050", "drinker": "#0070c0", "toy": "#c000ff"}
        for roi in self.roi_overlays:
            sx = frame_w / max(1.0, float(roi.image_width))
            sy = frame_h / max(1.0, float(roi.image_height))
            rx1 = float(roi.x1) * sx - crop_x1
            ry1 = float(roi.y1) * sy - crop_y1
            rx2 = float(roi.x2) * sx - crop_x1
            ry2 = float(roi.y2) * sy - crop_y1
            # Skip ROIs that do not intersect the displayed crop.
            if rx2 < 0 or ry2 < 0:
                continue
            category = str(roi.category)
            color = colors.get(category, "yellow")
            if getattr(roi, "polygon", None) is not None and len(roi.polygon) >= 3:
                pts = [(float(x) * sx - crop_x1, float(y) * sy - crop_y1) for x, y in roi.polygon]
                draw.line([*pts, pts[0]], fill=color, width=2)
            draw.rectangle([rx1, ry1, rx2, ry2], outline=color, width=2)
            draw.text((max(0, int(rx1) + 2), max(0, int(ry1) + 2)), category, fill=color)

    def _resolve_video_path(self, row: pd.Series) -> Path | None:
        if "source_video_path" in row.index and pd.notna(row.get("source_video_path")):
            raw = str(row.get("source_video_path")).strip()
            if raw:
                p = Path(raw)
                if p.exists():
                    return p
                # Some rows may store only a filename or a relative path.
                if self.config.video_root is not None:
                    candidates = [self.config.video_root / p, self.config.video_root / p.name]
                    for cand in candidates:
                        if cand.exists():
                            return cand

        raw_keys = [str(row.get("video_key", "")).strip()]
        if "source_video_key" in row.index and pd.notna(row.get("source_video_key")):
            raw_keys.append(str(row.get("source_video_key")).strip())

        candidates = []
        for raw_key in raw_keys:
            if not raw_key:
                continue
            vk = raw_key.replace("\\", "/").lower()
            stem = Path(vk).stem.lower()
            stems = [stem]
            # Some CVAT exports store task/source names like
            # "test video Pigs291119_000302_30fps" instead of the disk stem.
            for prefix in ["test video ", "tracking_annotation_", "tracking annotation "]:
                if stem.startswith(prefix):
                    stems.append(stem[len(prefix) :])

            for candidate_stem in stems:
                candidates.extend(
                    [
                        vk,
                        candidate_stem,
                        f"{candidate_stem}.mp4",
                        f"{candidate_stem}_30fps",
                        f"{candidate_stem}_30fps.mp4",
                        f"{vk}.mp4",
                        f"{vk}_30fps",
                        f"{vk}_30fps.mp4",
                    ]
                )
                if candidate_stem.endswith("_30fps"):
                    base = candidate_stem[: -len("_30fps")]
                    candidates.extend([base, f"{base}.mp4"])

        for key in candidates:
            key = str(key).strip().lower()
            if key in self.video_index:
                return self.video_index[key]

        # Last-resort lookup already indexed recursively; avoid rglob per frame.
        return None

    def _placeholder(self, text: str) -> Image.Image:
        img = Image.new("RGB", (self.config.thumb_w, self.config.thumb_h), "#eeeeee")
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, img.width - 1, img.height - 1], outline="black")
        draw.multiline_text((10, 10), text, fill="black")
        return img

    def _fit_image(self, img: Image.Image, max_w: int, max_h: int) -> Image.Image:
        out = img.copy()
        out.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        return out

    def save_current(self) -> bool:
        unit = self.current_unit()
        unit_id = str(unit["review_unit_id"])
        record = {
            "review_item_id": unit.get("review_item_id", ""),
            "review_unit_id": unit_id,
            "review_unit_type": unit.get("review_unit_type", ""),
            "temporal_unit_key": unit.get("temporal_unit_key", unit_id),
            "source_type": unit.get("source_type", ""),
            "dataset_id": unit.get("dataset_id", ""),
            "video_key": unit.get("video_key", ""),
            "pig_id": unit.get("pig_id", ""),
            "track_id": unit.get("track_id", ""),
            "object_track_key": unit.get("object_track_key", ""),
            "unit_start_frame": unit.get("unit_start_frame", ""),
            "unit_end_frame": unit.get("unit_end_frame", ""),
            "display_frame_indices": unit.get("display_frame_indices", ""),
            "review_template": unit.get("review_template", ""),
            "behavior_label": unit.get("behavior_label", ""),
            "original_behavior": unit.get("behavior_label", ""),
            "review_reason": unit.get("review_reason", ""),
            "apply_scope": unit.get("apply_scope", ""),
            "manual_review_decision": self.decision_var.get(),
            "manual_corrected_behavior": self.corrected_var.get(),
            "manual_label_strength": self.strength_var.get(),
            "manual_training_action": self.action_var.get(),
            "manual_sample_weight": self.weight_var.get(),
            "manual_note": self.note_var.get(),
        }
        normalized, warnings = canonicalize_decisions(pd.DataFrame([record]))
        errors, semantic_warnings = validate_decision_semantics(
            normalized,
            require_complete=False,
        )
        warnings.extend(semantic_warnings)
        errors.extend(
            warning
            for warning in warnings
            if warning.startswith("active_decision_without_strength")
        )

        decision = str(normalized.iloc[0]["manual_review_decision"])
        wanted = self._all_display_frames(unit)
        frame_rows = self._frame_rows_for_unit(unit)
        observed = pd.to_numeric(frame_rows["frame_index"], errors="coerce")
        observed_frames = sorted(observed.dropna().astype(int).tolist())
        complete_scope = observed_frames == wanted
        diagnostics = getattr(self, "current_diagnostics", [])
        blocking_diagnostics = [
            message
            for message in diagnostics
            if "no frame row" in message
            or "missing_video_or_crop" in message
            or "missing_legacy_crop_path" in message
            or "no_frames" in message
        ]
        if decision in {"accept", "corrected"} and not complete_scope:
            errors.append("cannot confirm behavior: displayed frame scope is incomplete")
        if decision in {"accept", "corrected"} and blocking_diagnostics:
            errors.append("cannot confirm behavior: one or more images are unavailable")
        if decision == "exclude" and (not complete_scope or blocking_diagnostics):
            if not str(normalized.iloc[0]["manual_note"]).strip():
                errors.append("exclusion with incomplete visual evidence requires a note")
        if errors:
            messagebox.showerror("Invalid decision", "\n".join(sorted(set(errors))))
            return False

        normalized_record = normalized.iloc[0].to_dict()
        self.decisions[unit_id] = {
            key: "" if pd.isna(value) else value
            for key, value in normalized_record.items()
        }
        self.write_decisions(show_message=False)
        return True

    def write_decisions(self, show_message: bool = True) -> None:
        unit_order = {
            str(unit_id): index
            for index, unit_id in enumerate(self.units["review_unit_id"].astype(str))
        }
        rows = sorted(
            self.decisions.values(),
            key=lambda row: (
                unit_order.get(str(row.get("review_unit_id", "")), len(unit_order)),
                str(row.get("review_unit_id", "")),
            ),
        )
        out1 = self.config.output_dir / "behavior_unit_review_decisions.csv"
        out2 = self.config.output_dir / "behavior_strength_review_decisions.csv"
        if not rows:
            # Write headers anyway.
            rows = []
        cols = [
            "review_item_id",
            "review_unit_id",
            "review_unit_type",
            "temporal_unit_key",
            "source_type",
            "dataset_id",
            "video_key",
            "pig_id",
            "track_id",
            "object_track_key",
            "unit_start_frame",
            "unit_end_frame",
            "display_frame_indices",
            "review_template",
            "behavior_label",
            "original_behavior",
            "review_reason",
            "apply_scope",
            "manual_review_decision",
            "manual_corrected_behavior",
            "manual_label_strength",
            "manual_training_action",
            "manual_sample_weight",
            "manual_note",
        ]
        _write_csv_atomic(out1, cols, rows)
        # Compatibility file name; still unit-level, not expanded window-level.
        _write_csv_atomic(out2, cols, rows)
        if show_message:
            messagebox.showinfo("Saved", f"Wrote {len(rows)} decisions\n{out1}\n{out2}")

    def save_next(self) -> None:
        if self.save_current():
            self.next_item()

    def accept_strong_next(self) -> None:
        self.decision_var.set("accept")
        self.strength_var.set("strong")
        self.action_var.set("main_train")
        self.weight_var.set("1.0")
        self.save_next()

    def exclude_next(self) -> None:
        self.decision_var.set("exclude")
        self.strength_var.set("boundary")
        self.action_var.set("exclude")
        self.weight_var.set("0.0")
        self.save_next()

    def next_item(self) -> None:
        if self.current < len(self.units) - 1:
            self.current += 1
            self.show_current()

    def prev_item(self) -> None:
        if self.current > 0:
            self.current -= 1
            self.show_current()

    def on_quit(self) -> None:
        self.write_decisions(show_message=False)
        for cap in self.video_cache.values():
            try:
                cap.release()
            except Exception:
                pass
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review canonical temporal units, not training windows."
    )
    parser.add_argument("--review-units-csv", type=Path, required=True)
    parser.add_argument("--frame-features-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, default=Path("data/videos"))
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/legacy_full_multigt_masked_nodup_16f/crops"),
    )
    parser.add_argument(
        "--roi-coco-json",
        type=Path,
        default=Path("data/annotations/roi/ROI_annotations.coco.json"),
    )
    parser.add_argument("--source-type", default="")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--padding", type=float, default=0.8)
    parser.add_argument("--copy-contact-sheets", action="store_true")
    args = parser.parse_args()

    ReviewUnitGui(
        GuiConfig(
            review_units_csv=args.review_units_csv,
            frame_features_csv=args.frame_features_csv,
            output_dir=args.output_dir,
            video_root=args.video_root,
            raw_root=args.raw_root,
            roi_coco_path=args.roi_coco_json,
            source_type=args.source_type or None,
            max_items=args.max_items if args.max_items > 0 else None,
            padding=args.padding,
            copy_contact_sheets=args.copy_contact_sheets,
        )
    )


if __name__ == "__main__":
    main()
