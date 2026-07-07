"""Helpers for concise, stable output layout."""

from __future__ import annotations

from pathlib import Path

TRACKING_MODES = frozenset(
    {
        "realtime",
        "bytetrack_raw",
        "hybrid_bytetrack",
        "gt_export",
    }
)


def prediction_root(project_root: Path, *parts: str) -> Path:
    """Build canonical prediction root under outputs/pred."""
    return project_root / "outputs" / "pred" / Path(*parts)


def evaluation_root(project_root: Path, *parts: str) -> Path:
    """Build canonical evaluation root under outputs/eval."""
    return project_root / "outputs" / "eval" / Path(*parts)


def is_mode_scoped_root(base_root: Path, mode: str) -> bool:
    """Check whether a root already contains the tracking mode segment."""
    return base_root.name == mode or base_root.parent.name == mode


def mode_scoped_video_dir(base_root: Path, mode: str, video_stem: str) -> Path:
    """Return a video output directory without duplicating mode segments."""
    if is_mode_scoped_root(base_root, mode):
        return base_root / video_stem
    return base_root / mode / video_stem


def prediction_xml_candidates(
    prediction_root_dir: Path,
    video_stem: str,
    preferred_mode: str | None = None,
) -> list[Path]:
    """Return supported legacy and canonical XML candidate paths."""
    mode_candidates: list[str] = []
    if preferred_mode:
        mode_candidates.append(preferred_mode)
        if preferred_mode == "gt_export":
            mode_candidates.append("hybrid_bytetrack")

    unique_modes = [mode for mode in dict.fromkeys(mode_candidates) if mode]
    candidates: list[Path] = []

    for mode in unique_modes:
        if is_mode_scoped_root(prediction_root_dir, mode):
            candidates.append(
                prediction_root_dir / video_stem / "annotations_cvat_video_1_1.xml"
            )
        candidates.append(
            prediction_root_dir / mode / video_stem / "annotations_cvat_video_1_1.xml"
        )
        candidates.append(
            prediction_root_dir / video_stem / mode / "annotations_cvat_video_1_1.xml"
        )

    candidates.append(prediction_root_dir / video_stem / "annotations_cvat_video_1_1.xml")
    candidates.append(
        prediction_root_dir / video_stem / f"{video_stem}_annotations_cvat_video_1_1.xml"
    )
    return candidates
