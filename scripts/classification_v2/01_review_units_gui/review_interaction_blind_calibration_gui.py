"""Run the isolated neutral interaction-calibration presentation."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
from PIL import Image, ImageTk

from pig_behavior.classification_v2.review.blinded_calibration_presentation import (
    BEHAVIOR_VOCABULARY,
    LEGEND_TEXT,
    PRESENTATION_SEMANTIC_HASH,
    PRESENTATION_VERSION,
    REVIEW_CONFIDENCE_VALUES,
    VISUAL_REVIEWABILITY_VALUES,
    compose_blinded_contact_sheet,
    join_blinded_media_authority,
    local_context_identity,
    public_display_text,
    render_neutral_context,
    validate_calibration_decisions,
)


def _load_production_media_module() -> ModuleType:
    path = Path(__file__).with_name("review_temporal_unit_gui.py")
    spec = importlib.util.spec_from_file_location(
        "_classification_v2_production_media",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load production media helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_MEDIA = _load_production_media_module()


class NeutralMediaDelegate(_MEDIA.ReviewUnitGui):
    """Reuse immutable media resolution while replacing interaction rendering."""

    def __init__(self, config: Any, frames: pd.DataFrame) -> None:
        self.config = config
        self.frames = frames.copy()
        self.frames["frame_index"] = pd.to_numeric(
            self.frames["frame_index"],
            errors="coerce",
        )
        self.video_cache: dict[str, Any] = {}
        self.video_index = self._build_video_index(config.video_root)
        self.roi_overlays = self._load_roi_overlays(config.roi_coco_path)

    def _interaction_context_image(
        self,
        frame: Any,
        actor: pd.Series,
    ) -> Image.Image:
        image = Image.fromarray(frame).convert("RGB")
        frame_index = pd.to_numeric(actor.get("frame_index"), errors="coerce")
        scene = self.frames[
            self.frames["source_type"]
            .astype(str)
            .eq(str(actor.get("source_type", "")))
            & self.frames["dataset_id"]
            .astype(str)
            .eq(str(actor.get("dataset_id", "")))
            & self.frames["video_key"]
            .astype(str)
            .eq(str(actor.get("video_key", "")))
            & self.frames["frame_index"].eq(frame_index)
        ].copy()
        return render_neutral_context(
            image,
            scene,
            actor_identity=local_context_identity(actor),
        )

    def make_blinded_sheet(
        self,
        unit: pd.Series,
    ) -> tuple[Image.Image, list[str]]:
        rows = self._frame_rows_for_unit(unit)
        targets = set(self._display_frames(unit))
        history = set(self._history_display_frames(unit)).difference(targets)
        wanted = [*sorted(history), *sorted(targets)]
        row_by_frame = {
            int(record["frame_index"]): record
            for _, record in rows.iterrows()
            if pd.notna(record.get("frame_index"))
        }
        diagnostics: list[str] = []
        frames: list[tuple[str, int, Image.Image, str]] = []
        for frame_index in wanted:
            role = "CONTEXT" if frame_index in history else "TARGET"
            row = row_by_frame.get(frame_index)
            if row is None:
                frames.append(
                    (
                        role,
                        frame_index,
                        self._placeholder("MEDIA UNAVAILABLE"),
                        "media_unavailable",
                    )
                )
                diagnostics.append("media_unavailable")
                continue
            image, status = self._image_for_row(unit, row)
            frames.append((role, frame_index, image, status or "ok"))
            if status:
                diagnostics.append("media_unavailable")
        if not frames:
            frames.append(
                (
                    "TARGET",
                    int(unit.get("unit_start_frame", 0)),
                    self._placeholder("MEDIA UNAVAILABLE"),
                    "media_unavailable",
                )
            )
            diagnostics.append("media_unavailable")
        return compose_blinded_contact_sheet(frames), sorted(set(diagnostics))


class BlindedCalibrationGui:
    """Minimal calibration-only GUI with no production-ledger compatibility."""

    def __init__(
        self,
        *,
        manifest: pd.DataFrame,
        media: pd.DataFrame,
        frame_features: pd.DataFrame,
        output_dir: Path,
        video_root: Path,
        raw_root: Path,
        roi_coco_path: Path,
        reviewer: str,
        subset: str,
    ) -> None:
        joined = join_blinded_media_authority(manifest, media)
        self.units = joined.loc[joined["frozen_subset"].eq(subset)].copy()
        if self.units.empty:
            raise SystemExit(f"no calibration items for subset={subset}")
        self.units = self.units.sort_values("presentation_order").reset_index(
            drop=True
        )
        if "\\human_decisions\\behavior\\" in str(output_dir).replace(
            "/",
            "\\",
        ).casefold():
            raise SystemExit("production Behavior ledger output is forbidden")
        self.output_dir = output_dir
        self.reviewer = reviewer
        self.current = 0
        self.decisions: dict[str, dict[str, object]] = {}
        config = _MEDIA.GuiConfig(
            review_units_csv=Path("BLINDED_MANIFEST_ONLY"),
            frame_features_csv=Path("IMMUTABLE_FRAME_FEATURES"),
            output_dir=output_dir,
            video_root=video_root,
            raw_root=raw_root,
            roi_coco_path=roi_coco_path,
            source_type=None,
            max_items=None,
            padding=0.8,
            copy_contact_sheets=False,
        )
        self.media = NeutralMediaDelegate(config, frame_features)

        import tkinter as tk

        self.tk = tk
        self.root = tk.Tk()
        self.root.title("Blinded interaction calibration")
        self.root.geometry("1380x900")
        self.root.minsize(1100, 740)
        self.header = tk.StringVar()
        self.image_label = tk.Label(self.root)
        self.image_label.pack(fill="both", expand=True)
        tk.Label(
            self.root,
            textvariable=self.header,
            justify="left",
            font=("Segoe UI", 11),
        ).pack(fill="x")
        tk.Label(
            self.root,
            text=LEGEND_TEXT,
            fg="#404040",
        ).pack(fill="x")

        controls = tk.Frame(self.root)
        controls.pack(fill="x")
        self.behavior = tk.StringVar(value="")
        self.reviewability = tk.StringVar(value="")
        self.confidence = tk.StringVar(value="")
        self.note = tk.StringVar(value="")
        tk.OptionMenu(controls, self.behavior, "", *BEHAVIOR_VOCABULARY).pack(
            side="left"
        )
        tk.OptionMenu(
            controls,
            self.reviewability,
            "",
            *VISUAL_REVIEWABILITY_VALUES,
        ).pack(side="left")
        tk.OptionMenu(
            controls,
            self.confidence,
            "",
            *REVIEW_CONFIDENCE_VALUES,
        ).pack(side="left")
        tk.Entry(controls, textvariable=self.note, width=40).pack(side="left")
        tk.Button(controls, text="Previous", command=self.previous).pack(
            side="left"
        )
        tk.Button(controls, text="Save", command=self.save_current).pack(
            side="left"
        )
        tk.Button(controls, text="Next", command=self.next).pack(side="left")
        self.show_current()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.mainloop()

    def show_current(self) -> None:
        unit = self.units.iloc[self.current]
        sheet, _ = self.media.make_blinded_sheet(unit)
        sheet.thumbnail((1000, 650), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(sheet)
        self.image_label.configure(image=self.photo)
        targets = self.media._display_frames(unit)
        history = set(self.media._history_display_frames(unit)).difference(
            targets
        )
        self.header.set(
            public_display_text(
                item_number=self.current + 1,
                item_count=len(self.units),
                calibration_item_id=str(unit["calibration_item_id"]),
                target_count=len(targets),
                context_count=len(history),
            )
        )
        previous = self.decisions.get(str(unit["review_unit_id"]), {})
        self.behavior.set(str(previous.get("reviewed_behavior", "")))
        self.reviewability.set(
            str(previous.get("visual_reviewability", ""))
        )
        self.confidence.set(str(previous.get("review_confidence", "")))
        self.note.set(str(previous.get("optional_short_note", "")))

    def save_current(self) -> None:
        unit = self.units.iloc[self.current]
        record = {
            "review_key": str(unit["review_unit_id"]),
            "calibration_item_id": str(unit["calibration_item_id"]),
            "reviewed_behavior": self.behavior.get(),
            "visual_reviewability": self.reviewability.get(),
            "review_confidence": self.confidence.get(),
            "optional_short_note": self.note.get().strip(),
            "presentation_version": PRESENTATION_VERSION,
            "presentation_semantic_hash": PRESENTATION_SEMANTIC_HASH,
            "reviewer": self.reviewer,
            "decision_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        frame = pd.DataFrame([record])
        audit = validate_calibration_decisions(frame)
        if not audit["valid"]:
            from tkinter import messagebox

            messagebox.showerror("Invalid calibration decision", ";".join(audit["errors"]))
            return
        self.decisions[record["review_key"]] = record
        self.write_decisions()

    def write_decisions(self) -> None:
        """Write only the isolated calibration ledger."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "interaction_blind_calibration_decisions.csv"
        frame = pd.DataFrame(self.decisions.values())
        audit = validate_calibration_decisions(frame)
        if not audit["valid"]:
            raise RuntimeError(";".join(audit["errors"]))
        _MEDIA._write_csv_atomic(
            path,
            list(frame.columns),
            frame.to_dict(orient="records"),
        )

    def previous(self) -> None:
        if self.current > 0:
            self.current -= 1
            self.show_current()

    def next(self) -> None:
        if self.current + 1 < len(self.units):
            self.current += 1
            self.show_current()

    def close(self) -> None:
        for capture in self.media.video_cache.values():
            try:
                capture.release()
            except Exception:
                pass
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blinded-manifest", type=Path, required=True)
    parser.add_argument("--media-authority", type=Path, required=True)
    parser.add_argument("--frame-features-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument(
        "--subset",
        choices=(
            "CALIBRATION_DEVELOPMENT_SET",
            "BLINDED_CONFIRMATION_SET",
        ),
        default="CALIBRATION_DEVELOPMENT_SET",
    )
    parser.add_argument("--video-root", type=Path, default=Path("data/videos"))
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--roi-coco-json", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.blinded_manifest, low_memory=False)
    media = pd.read_csv(args.media_authority, low_memory=False)
    joined = join_blinded_media_authority(manifest, media)
    subset = joined.loc[joined["frozen_subset"].eq(args.subset)]
    if subset.empty:
        raise SystemExit(f"no calibration items for subset={args.subset}")
    if args.validate_only:
        print(
            "BLINDED_PRESENTATION_VALID "
            f"version={PRESENTATION_VERSION} "
            f"hash={PRESENTATION_SEMANTIC_HASH} rows={len(subset)}"
        )
        return
    frames = _MEDIA.load_gui_frame_features(args.frame_features_csv)
    BlindedCalibrationGui(
        manifest=manifest,
        media=media,
        frame_features=frames,
        output_dir=args.output_dir,
        video_root=args.video_root,
        raw_root=args.raw_root,
        roi_coco_path=args.roi_coco_json,
        reviewer=args.reviewer,
        subset=args.subset,
    )


if __name__ == "__main__":
    main()
