"""Configuration objects for classification dataset v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import (
    CVAT_SEQUENCE_VIEWS,
    LABEL_POLICIES,
    SEQUENCE_VIEWS,
)


@dataclass(slots=True)
class ClassificationV2Config:
    """Runtime configuration for building the v2 classification dataset."""

    # Legacy recovered data.
    legacy_frame_csv: Path | None = None
    legacy_dense_csv: Path | None = None

    # CVAT tracking XML data: full 8-pig tracking annotations.
    cvat_tracking_xml: Path | None = None
    cvat_video_key: str | None = None
    cvat_times_txt: Path | None = None
    cvat_fps: float | None = None

    # CVAT selected/native annotations: behavior-selected bbox annotations.
    cvat_selected_task_dir: Path | None = None
    cvat_selected_annotations_json: Path | None = None
    cvat_selected_video_key: str | None = None
    cvat_selected_image_width: int | None = None
    cvat_selected_image_height: int | None = None
    cvat_selected_fps: float | None = None

    # ROI and output.
    roi_coco_json: Path | None = None
    output_root: Path = Path("outputs/classification_v2")

    # Sequence settings.
    cvat_label_stride: int = 6
    cvat_window_size: int = 6
    cvat_label_policy: str = "anchor_label"

    legacy_sequence_view: str = "legacy_dense_6_same_span_0_12"
    cvat_sequence_view: str = "cvat_window_6"

    # Context policy.
    require_full_8_for_eval: bool = False
    keep_actor_only_non_interaction: bool = True
    review_interaction_without_partner: bool = True

    # Export switches.
    write_legacy_compatible_csv: bool = True
    write_sequence_features: bool = True
    validate: bool = True

    # General.
    dry_run: bool = False
    max_rows: int | None = None
    overwrite: bool = False

    def validate_basic(self) -> None:
        """Validate config values that do not require reading data files."""
        if self.cvat_label_stride <= 0:
            raise ValueError("cvat_label_stride must be > 0.")

        if self.cvat_window_size <= 0:
            raise ValueError("cvat_window_size must be > 0.")

        if self.cvat_label_policy not in LABEL_POLICIES:
            allowed = ", ".join(sorted(LABEL_POLICIES))
            raise ValueError(
                f"Invalid cvat_label_policy={self.cvat_label_policy!r}. "
                f"Allowed values: {allowed}"
            )

        if self.legacy_sequence_view not in SEQUENCE_VIEWS:
            allowed = ", ".join(sorted(SEQUENCE_VIEWS))
            raise ValueError(
                f"Invalid legacy_sequence_view={self.legacy_sequence_view!r}. "
                f"Allowed values: {allowed}"
            )

        if self.cvat_sequence_view not in CVAT_SEQUENCE_VIEWS:
            allowed = ", ".join(sorted(CVAT_SEQUENCE_VIEWS))
            raise ValueError(
                f"Invalid cvat_sequence_view={self.cvat_sequence_view!r}. "
                f"Allowed values: {allowed}"
            )

        if self.max_rows is not None and self.max_rows <= 0:
            raise ValueError("max_rows must be None or > 0.")

        if self.cvat_fps is not None and self.cvat_fps <= 0:
            raise ValueError("cvat_fps must be None or > 0.")

        if self.cvat_selected_fps is not None and self.cvat_selected_fps <= 0:
            raise ValueError("cvat_selected_fps must be None or > 0.")

    def has_legacy_input(self) -> bool:
        """Return True when any legacy input is configured."""
        return self.legacy_frame_csv is not None or self.legacy_dense_csv is not None

    def has_cvat_tracking_input(self) -> bool:
        """Return True when CVAT tracking XML input is configured."""
        return self.cvat_tracking_xml is not None

    def has_cvat_selected_input(self) -> bool:
        """Return True when CVAT selected/native input is configured."""
        return (
            self.cvat_selected_task_dir is not None
            or self.cvat_selected_annotations_json is not None
        )

    def has_any_input(self) -> bool:
        """Return True when at least one input source is configured."""
        return (
            self.has_legacy_input()
            or self.has_cvat_tracking_input()
            or self.has_cvat_selected_input()
        )

    def ensure_has_input(self) -> None:
        """Raise if no dataset input is configured."""
        if not self.has_any_input():
            raise ValueError(
                "No input source configured. Provide at least one of: "
                "legacy_frame_csv, legacy_dense_csv, cvat_tracking_xml, "
                "cvat_selected_task_dir, or cvat_selected_annotations_json."
            )

    def ensure_output_root(self) -> None:
        """Create output root if needed."""
        self.output_root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> ClassificationV2Config:
        """Build config from a dict-like object.

        This is useful for tests and later CLI implementation.
        """
        path_fields = {
            "legacy_frame_csv",
            "legacy_dense_csv",
            "cvat_tracking_xml",
            "cvat_times_txt",
            "cvat_selected_task_dir",
            "cvat_selected_annotations_json",
            "roi_coco_json",
            "output_root",
        }

        cleaned: dict[str, Any] = {}
        for key, value in values.items():
            if value in {"", None}:
                cleaned[key] = None
                continue

            if key in path_fields:
                cleaned[key] = Path(value)
            else:
                cleaned[key] = value

        cfg = cls(**cleaned)
        cfg.validate_basic()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable config dictionary."""
        data: dict[str, Any] = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            data[field_name] = str(value) if isinstance(value, Path) else value
        return data