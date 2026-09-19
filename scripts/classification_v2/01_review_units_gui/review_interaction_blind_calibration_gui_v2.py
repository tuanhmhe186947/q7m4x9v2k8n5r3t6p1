"""Run source-specific blinded interaction calibration presentation V2."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageTk

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pig_behavior.classification_v2.review.source_specific_blinded_presentation_v2 import (
    BEHAVIOR_VOCABULARY,
    CALIBRATION_DECISION_FIELDS,
    CONTEXT_HEADING,
    CVAT_CONTEXT_MODE,
    CVAT_LEGEND_TEXT,
    CVAT_RENDER_MODE,
    FRAME_BORDER_COLOR,
    FRAME_ORDER_CONTRACT,
    LEGACY_CONTEXT_MODE,
    LEGACY_NOTICE_TEXT,
    LEGACY_RENDER_MODE,
    MEDIA_AUTHORITY_SCHEMA_VERSION,
    MISSING_MEDIA_BEHAVIOR,
    MISSING_TEMPLATE_BEHAVIOR,
    NEUTRAL_NEIGHBOR_COLOR,
    PRESENTATION_SEMANTIC_HASH,
    PRESENTATION_TEMPLATE,
    PRESENTATION_VERSION,
    REVIEW_CONFIDENCE_VALUES,
    TARGET_HEADING,
    VISUAL_REVIEWABILITY_VALUES,
    SourceSpecificPresentationError,
    canonical_presentation_contract_v2,
    compose_source_specific_contact_sheet,
    local_context_identity,
    normalized_bool,
    parse_frame_indices,
    public_display_text_v2,
    render_neutral_context_v2,
    source_dispatch,
    validate_calibration_decisions_v2,
    validate_media_authority_v2,
    visible_notice,
)


def _load_production_media_module() -> ModuleType:
    path = Path(__file__).with_name("review_temporal_unit_gui.py")
    spec = importlib.util.spec_from_file_location(
        "_classification_v2_production_media_v2",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load production media helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_MEDIA = _load_production_media_module()


def effective_runtime_presentation_contract_v2() -> dict[str, Any]:
    """Construct the effective runtime contract independently of the writer."""

    return {
        "presentation_version": PRESENTATION_VERSION,
        "media_authority_schema": {
            "schema_version": MEDIA_AUTHORITY_SCHEMA_VERSION,
            "required_fields": [
                "review_key",
                "split",
                "source_type",
                "context_mode",
                "presentation_template",
                "presentation_version",
                "presentation_semantic_hash",
                "render_mode",
                "actor_identity_semantics",
                "neighbor_context_available",
                "full_frame_context_available",
                "target_frame_indices",
                "history_frame_indices",
                "display_frame_indices",
                "target_frame_count",
                "history_frame_count",
                "frame_order_contract",
                "media_authority",
                "render_available",
                "render_failure_reason",
            ],
        },
        "source_modes": {
            CVAT_CONTEXT_MODE: {
                "source_type": "cvat_tracking_xml",
                "render_mode": CVAT_RENDER_MODE,
                "actor_identity_semantics": "red_bbox_is_reviewed_actor",
                "neighbor_context_available": True,
                "full_frame_context_available": True,
                "actor_color": "#ff0000",
                "neighbor_color": NEUTRAL_NEIGHBOR_COLOR,
                "visible_notice": CVAT_LEGEND_TEXT,
                "target_frame_count": 6,
                "fabricated_overlay_allowed": False,
            },
            LEGACY_CONTEXT_MODE: {
                "source_type": "legacy_recovered",
                "render_mode": LEGACY_RENDER_MODE,
                "actor_identity_semantics": "entire_crop_is_reviewed_actor",
                "neighbor_context_available": False,
                "full_frame_context_available": False,
                "actor_color": None,
                "neighbor_color": None,
                "visible_notice": LEGACY_NOTICE_TEXT,
                "target_frame_count": 16,
                "fabricated_overlay_allowed": False,
            },
        },
        "presentation_template": PRESENTATION_TEMPLATE,
        "frame_border_color": FRAME_BORDER_COLOR,
        "context_band_color": "#d9e2f3",
        "target_band_color": "#fff2cc",
        "ranking_visibility": "HIDDEN",
        "provisional_label_visibility": "HIDDEN",
        "machine_reason_visibility": "HIDDEN",
        "candidate_tier_visibility": "HIDDEN",
        "machine_score_visibility": "HIDDEN",
        "source_date_video_stratum_visibility": "HIDDEN",
        "target_history_headings": {
            "context": CONTEXT_HEADING,
            "target": TARGET_HEADING,
        },
        "frame_order_contract": FRAME_ORDER_CONTRACT,
        "decision_schema": canonical_presentation_contract_v2()[
            "decision_schema"
        ],
        "missing_template_behavior": MISSING_TEMPLATE_BEHAVIOR,
        "missing_media_behavior": MISSING_MEDIA_BEHAVIOR,
        "renderer_dispatch": (
            "EXPLICIT_CONTEXT_MODE_AND_RENDER_MODE_NO_FALLBACK"
        ),
    }


def runtime_contract_matches_declared() -> bool:
    """Return whether independent declared and runtime semantics agree."""

    return (
        effective_runtime_presentation_contract_v2()
        == canonical_presentation_contract_v2()
    )


class SourceSpecificMediaDelegate(_MEDIA.ReviewUnitGui):
    """Resolve immutable media with explicit V2 dispatch and no fallback."""

    def __init__(self, config: Any, frames: pd.DataFrame) -> None:
        self.config = config
        self.frames = frames.copy()
        self.frames["frame_index"] = pd.to_numeric(
            self.frames["frame_index"],
            errors="coerce",
        )
        self.video_cache: dict[str, Any] = {}
        self.video_next_frame: dict[str, int] = {}
        self.video_index = self._build_video_index(config.video_root)
        self.roi_overlays: list[Any] = []

    def close(self) -> None:
        """Release bounded video handles."""

        for capture in self.video_cache.values():
            try:
                capture.release()
            except Exception:
                pass
        self.video_cache.clear()
        self.video_next_frame.clear()

    def _display_frames(self, unit: pd.Series) -> list[int]:
        return parse_frame_indices(unit.get("target_frame_indices", ""))

    def _history_display_frames(self, unit: pd.Series) -> list[int]:
        return parse_frame_indices(unit.get("history_frame_indices", ""))

    def _all_display_frames(self, unit: pd.Series) -> list[int]:
        return [
            *self._history_display_frames(unit),
            *self._display_frames(unit),
        ]

    @staticmethod
    def _assert_dispatch(unit: pd.Series) -> dict[str, Any]:
        dispatch = source_dispatch(unit.get("source_type", ""))
        observed = {
            "context_mode": str(unit.get("context_mode", "")).strip(),
            "render_mode": str(unit.get("render_mode", "")).strip(),
            "actor_identity_semantics": str(
                unit.get("actor_identity_semantics", "")
            ).strip(),
            "neighbor_context_available": normalized_bool(
                unit.get("neighbor_context_available", "")
            ),
            "full_frame_context_available": normalized_bool(
                unit.get("full_frame_context_available", "")
            ),
        }
        expected = {
            "context_mode": dispatch["context_mode"],
            "render_mode": dispatch["render_mode"],
            "actor_identity_semantics": dispatch[
                "actor_identity_semantics"
            ],
            "neighbor_context_available": dispatch[
                "neighbor_context_available"
            ],
            "full_frame_context_available": dispatch[
                "full_frame_context_available"
            ],
        }
        if str(unit.get("presentation_template", "")).strip() != (
            PRESENTATION_TEMPLATE
        ):
            raise SourceSpecificPresentationError(
                "missing or unknown presentation_template"
            )
        if observed != expected:
            raise SourceSpecificPresentationError(
                f"source dispatch mismatch observed={observed}"
            )
        if str(unit.get("presentation_version", "")).strip() != (
            PRESENTATION_VERSION
        ):
            raise SourceSpecificPresentationError(
                "presentation_version mismatch"
            )
        if str(unit.get("presentation_semantic_hash", "")).strip() != (
            PRESENTATION_SEMANTIC_HASH
        ):
            raise SourceSpecificPresentationError(
                "presentation_semantic_hash mismatch"
            )
        return dispatch

    def _decode_cvat_full_frame(
        self,
        actor: pd.Series,
    ) -> Image.Image:
        if _MEDIA.cv2 is None:
            raise SourceSpecificPresentationError("opencv unavailable")
        video_path = self._resolve_video_path(actor)
        if video_path is None:
            raise SourceSpecificPresentationError("missing CVAT video")
        video_key = str(video_path)
        capture = self.video_cache.get(video_key)
        if capture is None:
            capture = _MEDIA.cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise SourceSpecificPresentationError(
                    f"cannot open CVAT video={video_path.name}"
                )
            self.video_cache[video_key] = capture
        frame_index = int(actor["frame_index"])
        if self.video_next_frame.get(video_key) != frame_index:
            capture.set(_MEDIA.cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            self.video_next_frame.pop(video_key, None)
            raise SourceSpecificPresentationError(
                f"cannot decode CVAT frame={frame_index}"
            )
        self.video_next_frame[video_key] = frame_index + 1
        rgb = _MEDIA.cv2.cvtColor(frame, _MEDIA.cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb).convert("RGB")

    def _scene_rows(self, actor: pd.Series) -> pd.DataFrame:
        frame_index = pd.to_numeric(
            actor.get("frame_index"),
            errors="coerce",
        )
        return self.frames[
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

    def render_source_frame(
        self,
        unit: pd.Series,
        actor: pd.Series,
    ) -> tuple[Image.Image, dict[str, Any]]:
        """Render one frame using the exact declared source path."""

        dispatch = self._assert_dispatch(unit)
        context_mode = dispatch["context_mode"]
        if context_mode == CVAT_CONTEXT_MODE:
            full_frame = self._decode_cvat_full_frame(actor)
            scene = self._scene_rows(actor)
            actor_identity = local_context_identity(actor)
            rendered = render_neutral_context_v2(
                full_frame,
                scene,
                actor_identity=actor_identity,
            )
            identities = scene.apply(local_context_identity, axis=1)
            boxes = scene[["x1", "y1", "x2", "y2"]].apply(
                pd.to_numeric,
                errors="coerce",
            )
            valid_boxes = (
                boxes.notna().all(axis=1)
                & boxes["x2"].gt(boxes["x1"])
                & boxes["y2"].gt(boxes["y1"])
            )
            valid_non_actor_count = int(
                (identities.ne(actor_identity) & valid_boxes).sum()
            )
            return rendered, {
                "runtime_renderer": CVAT_RENDER_MODE,
                "valid_non_actor_count": valid_non_actor_count,
                "legacy_direct_crop": False,
            }
        if context_mode == LEGACY_CONTEXT_MODE:
            crop = self._legacy_crop_image(actor)
            if crop is None:
                raise SourceSpecificPresentationError(
                    "missing immutable legacy actor crop"
                )
            return crop, {
                "runtime_renderer": LEGACY_RENDER_MODE,
                "valid_non_actor_count": 0,
                "legacy_direct_crop": True,
            }
        raise SourceSpecificPresentationError(
            f"undeclared context_mode={context_mode}"
        )

    def make_source_specific_sheet(
        self,
        unit: pd.Series,
        *,
        collect_preflight_audit: bool = False,
    ) -> tuple[Image.Image, list[str], list[dict[str, Any]]]:
        """Render all declared frames with explicit context/target roles."""

        self._assert_dispatch(unit)
        rows = self._frame_rows_for_unit(unit)
        targets = self._display_frames(unit)
        history = self._history_display_frames(unit)
        wanted = [*history, *targets]
        row_by_frame = {
            int(record["frame_index"]): record
            for _, record in rows.iterrows()
            if pd.notna(record.get("frame_index"))
        }
        diagnostics: list[str] = []
        rendered_frames: list[tuple[str, int, Image.Image, str]] = []
        frame_audits: list[dict[str, Any]] = []
        for frame_index in wanted:
            role = "CONTEXT" if frame_index in history else "TARGET"
            actor = row_by_frame.get(frame_index)
            if actor is None:
                diagnostics.append(f"missing_actor_row:f{frame_index}")
                continue
            try:
                image, frame_audit = self.render_source_frame(unit, actor)
            except SourceSpecificPresentationError as exc:
                diagnostics.append(f"f{frame_index}:{exc}")
                continue
            rendered_frames.append((role, frame_index, image, "ok"))
            if collect_preflight_audit:
                pixels = np.asarray(image.convert("RGB"))
                frame_audit.update(
                    {
                        "image_sha256": hashlib.sha256(
                            image.tobytes()
                        ).hexdigest(),
                        "red_pixel_count": int(
                            np.all(
                                pixels == np.array([255, 0, 0]),
                                axis=2,
                            ).sum()
                        ),
                        "neutral_pixel_count": int(
                            np.all(
                                pixels == np.array([127, 127, 127]),
                                axis=2,
                            ).sum()
                        ),
                        "ranked_green_pixel_count": int(
                            np.all(
                                pixels == np.array([0, 176, 80]),
                                axis=2,
                            ).sum()
                        ),
                    }
                )
            frame_audits.append(
                {
                    "frame_index": frame_index,
                    "role": role,
                    **frame_audit,
                }
            )
        if diagnostics or len(rendered_frames) != len(wanted):
            raise SourceSpecificPresentationError(";".join(diagnostics))
        sheet = compose_source_specific_contact_sheet(
            rendered_frames,
            context_mode=str(unit["context_mode"]),
        )
        return sheet, diagnostics, frame_audits


def load_smoke_ids(path: Path | None) -> set[str] | None:
    """Load opaque development-only smoke IDs."""

    if path is None:
        return None
    smoke = pd.read_csv(path, low_memory=False)
    required = {"calibration_item_id", "split"}
    missing = sorted(required.difference(smoke.columns))
    if missing:
        raise SystemExit(f"smoke manifest missing fields={missing}")
    if not smoke["split"].eq("CALIBRATION_DEVELOPMENT_SET").all():
        raise SystemExit("smoke manifest contains non-development items")
    ids = smoke["calibration_item_id"].fillna("").astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise SystemExit("smoke manifest has blank or duplicate IDs")
    return set(ids)


class SourceSpecificCalibrationGui:
    """Minimal isolated V2 calibration GUI."""

    def __init__(
        self,
        *,
        media: pd.DataFrame,
        frame_features: pd.DataFrame,
        output_dir: Path,
        video_root: Path,
        raw_root: Path,
        reviewer: str,
        subset: str,
        smoke_ids: set[str] | None = None,
    ) -> None:
        audit = validate_media_authority_v2(
            media,
            require_render_available=True,
        )
        if not audit["valid"]:
            raise SystemExit(";".join(audit["errors"]))
        units = media.loc[media["split"].eq(subset)].copy()
        if smoke_ids is not None:
            units = units.loc[
                units["calibration_item_id"].astype(str).isin(smoke_ids)
            ].copy()
            if set(units["calibration_item_id"].astype(str)) != smoke_ids:
                raise SystemExit("smoke manifest IDs do not match media")
        if units.empty:
            raise SystemExit(f"no calibration items for subset={subset}")
        self.units = units.sort_values(
            "presentation_order",
            kind="stable",
        ).reset_index(drop=True)
        normalized_output = str(output_dir.resolve()).replace(
            "/",
            "\\",
        ).casefold()
        if "\\human_decisions\\behavior\\" in normalized_output:
            raise SystemExit("production Behavior ledger output is forbidden")
        self.output_dir = output_dir
        self.reviewer = reviewer
        self.current = 0
        config = _MEDIA.GuiConfig(
            review_units_csv=Path("SOURCE_SPECIFIC_MEDIA_AUTHORITY_V2"),
            frame_features_csv=Path("IMMUTABLE_FRAME_FEATURES"),
            output_dir=output_dir,
            video_root=video_root,
            raw_root=raw_root,
            roi_coco_path=None,
            source_type=None,
            max_items=None,
            padding=0.0,
            copy_contact_sheets=False,
        )
        self.media = SourceSpecificMediaDelegate(config, frame_features)
        self.decisions = self._load_existing_decisions()
        self.sheet_cache = _MEDIA.RenderedImageCache(max_items=8)
        self._prefetch_after_id: str | None = None

        import tkinter as tk

        self.tk = tk
        self.root = tk.Tk()
        self.root.title("Blinded interaction calibration V2")
        self.root.geometry("1380x900")
        self.root.minsize(1100, 740)
        self.header = tk.StringVar()
        self.notice = tk.StringVar()
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
            textvariable=self.notice,
            fg="#404040",
        ).pack(fill="x")
        controls = tk.Frame(self.root)
        controls.pack(fill="x")
        self.behavior = tk.StringVar(value="")
        self.reviewability = tk.StringVar(value="")
        self.confidence = tk.StringVar(value="")
        self.note = tk.StringVar(value="")
        tk.OptionMenu(
            controls,
            self.behavior,
            "",
            *BEHAVIOR_VOCABULARY,
        ).pack(side="left")
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
        tk.Entry(
            controls,
            textvariable=self.note,
            width=40,
        ).pack(side="left")
        tk.Button(
            controls,
            text="Previous",
            command=self.previous,
        ).pack(side="left")
        tk.Button(
            controls,
            text="Save",
            command=self.save_current,
        ).pack(side="left")
        tk.Button(
            controls,
            text="Next",
            command=self.next,
        ).pack(side="left")
        self.show_current()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.mainloop()

    @property
    def decision_path(self) -> Path:
        return (
            self.output_dir
            / "interaction_blind_calibration_v2_decisions.csv"
        )

    def _load_existing_decisions(self) -> dict[str, dict[str, object]]:
        if not self.decision_path.exists():
            return {}
        frame = pd.read_csv(self.decision_path, low_memory=False)
        audit = validate_calibration_decisions_v2(frame)
        if not audit["valid"]:
            raise SystemExit(";".join(audit["errors"]))
        allowed = set(self.units["review_key"].astype(str))
        observed = set(frame["review_key"].astype(str))
        if not observed.issubset(allowed):
            raise SystemExit("existing V2 decisions do not match this view")
        return {
            str(record["review_key"]): record
            for record in frame.to_dict(orient="records")
        }

    def show_current(self) -> None:
        unit = self.units.iloc[self.current]
        sheet = self._display_sheet_for_unit(unit)
        self.photo = ImageTk.PhotoImage(sheet)
        self.image_label.configure(image=self.photo)
        targets = parse_frame_indices(unit["target_frame_indices"])
        history = parse_frame_indices(unit["history_frame_indices"])
        self.header.set(
            public_display_text_v2(
                item_number=self.current + 1,
                item_count=len(self.units),
                calibration_item_id=str(unit["calibration_item_id"]),
                target_count=len(targets),
                context_count=len(history),
                context_mode=str(unit["context_mode"]),
            )
        )
        self.notice.set(visible_notice(str(unit["context_mode"])))
        previous = self.decisions.get(str(unit["review_key"]), {})
        self.behavior.set(str(previous.get("reviewed_behavior", "")))
        self.reviewability.set(
            str(previous.get("visual_reviewability", ""))
        )
        self.confidence.set(str(previous.get("review_confidence", "")))
        self.note.set(str(previous.get("optional_short_note", "")))
        self._schedule_next_sheet()

    def _display_sheet_for_unit(self, unit: pd.Series) -> Image.Image:
        """Return a cached V2 display sheet without reading or writing decisions."""

        review_key = str(unit["review_key"])
        cached = self.sheet_cache.get(review_key)
        if cached is not None:
            image, _ = cached
            return image

        sheet, _, _ = self.media.make_source_specific_sheet(unit)
        sheet.thumbnail((1000, 650), Image.Resampling.LANCZOS)
        self.sheet_cache.put(review_key, sheet)
        return sheet

    def _cancel_pending_prefetch(self) -> None:
        """Cancel an idle preload that has not started."""

        if self._prefetch_after_id is None:
            return
        try:
            self.root.after_cancel(self._prefetch_after_id)
        except self.tk.TclError:
            pass
        self._prefetch_after_id = None

    def _schedule_next_sheet(self) -> None:
        """Warm one following media sheet while the current item is visible."""

        self._cancel_pending_prefetch()
        next_index = self.current + 1
        if next_index >= len(self.units):
            return
        self._prefetch_after_id = self.root.after_idle(
            self._prefetch_sheet,
            next_index,
        )

    def _prefetch_sheet(self, index: int) -> None:
        """Populate only the bounded derived-image cache on the GUI thread."""

        self._prefetch_after_id = None
        if 0 <= index < len(self.units):
            try:
                self._display_sheet_for_unit(self.units.iloc[index])
            except (OSError, SourceSpecificPresentationError):
                # Preserve the normal interactive error path; prefetch must
                # never change review state or make a bad item disappear.
                return

    def save_current(self) -> None:
        unit = self.units.iloc[self.current]
        record = {
            "review_key": str(unit["review_key"]),
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
        audit = validate_calibration_decisions_v2(frame)
        if not audit["valid"]:
            from tkinter import messagebox

            messagebox.showerror(
                "Invalid calibration decision",
                ";".join(audit["errors"]),
            )
            return
        self.decisions[record["review_key"]] = record
        self.write_decisions()

    def write_decisions(self) -> None:
        """Write only the isolated V2 calibration ledger."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(
            self.decisions.values(),
            columns=CALIBRATION_DECISION_FIELDS,
        )
        audit = validate_calibration_decisions_v2(frame)
        if not audit["valid"]:
            raise RuntimeError(";".join(audit["errors"]))
        _MEDIA._write_csv_atomic(
            self.decision_path,
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
        self._cancel_pending_prefetch()
        self.media.close()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--smoke-manifest", type=Path)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    media = pd.read_csv(args.media_authority, low_memory=False)
    audit = validate_media_authority_v2(
        media,
        require_render_available=True,
    )
    if not audit["valid"]:
        raise SystemExit(";".join(audit["errors"]))
    subset = media.loc[media["split"].eq(args.subset)]
    smoke_ids = load_smoke_ids(args.smoke_manifest)
    if smoke_ids is not None:
        subset = subset.loc[
            subset["calibration_item_id"].astype(str).isin(smoke_ids)
        ]
    if subset.empty:
        raise SystemExit(f"no calibration items for subset={args.subset}")
    if not runtime_contract_matches_declared():
        raise SystemExit("declared/runtime presentation contract mismatch")
    if args.validate_only:
        print(
            "SOURCE_SPECIFIC_PRESENTATION_V2_VALID "
            f"version={PRESENTATION_VERSION} "
            f"hash={PRESENTATION_SEMANTIC_HASH} rows={len(subset)}"
        )
        return
    frames = _MEDIA.load_gui_frame_features(args.frame_features_csv)
    SourceSpecificCalibrationGui(
        media=media,
        frame_features=frames,
        output_dir=args.output_dir,
        video_root=args.video_root,
        raw_root=args.raw_root,
        reviewer=args.reviewer,
        subset=args.subset,
        smoke_ids=smoke_ids,
    )


if __name__ == "__main__":
    main()
