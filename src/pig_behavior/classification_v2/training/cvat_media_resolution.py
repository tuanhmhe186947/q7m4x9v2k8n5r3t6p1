"""Resolve CVAT scientific video identifiers to authority-relative media paths.

The CVAT context key is a stable scientific identifier, not a filesystem
locator.  This module accepts only the registered source-video path recorded
with the observation and never searches a media tree or infers a filename.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pandas as pd


class CvatMediaResolutionError(ValueError):
    """Raised when a CVAT scientific media identity lacks one exact mapping."""


def attach_registered_cvat_media_paths(frames: pd.DataFrame) -> pd.DataFrame:
    """Add scientific and authority-relative media fields without host paths."""

    required = {"source_type", "source_video_key", "source_video_path"}
    missing = sorted(required.difference(frames.columns))
    if missing:
        raise CvatMediaResolutionError(f"CVAT media fields missing={missing}")
    resolved = frames.copy()
    cvat = resolved["source_type"].astype(str).eq("cvat_tracking_xml")
    resolved["scientific_media_id"] = ""
    resolved["registered_relative_media_path"] = ""
    if not cvat.any():
        return resolved

    selected = resolved.loc[cvat, ["source_video_key", "source_video_path"]].copy()
    selected["scientific_media_id"] = selected["source_video_key"].map(_text)
    selected["registered_relative_media_path"] = selected["source_video_path"].map(
        registered_relative_media_path
    )
    if selected["scientific_media_id"].eq("").any():
        raise CvatMediaResolutionError("CVAT scientific media ID is blank")
    counts = selected.groupby("scientific_media_id", sort=True)[
        "registered_relative_media_path"
    ].nunique(dropna=False)
    ambiguous = sorted(counts.loc[counts.ne(1)].index.astype(str))
    if ambiguous:
        raise CvatMediaResolutionError(
            f"CVAT scientific media ID has non-unique registration={ambiguous[:5]}"
        )
    resolved.loc[cvat, "scientific_media_id"] = selected["scientific_media_id"].to_numpy()
    resolved.loc[cvat, "registered_relative_media_path"] = selected[
        "registered_relative_media_path"
    ].to_numpy()
    return resolved


def registered_relative_media_path(source_video_path: object) -> str:
    """Return the registered path below the input authority, never a host path."""

    raw = _text(source_video_path).replace("\\", "/")
    if not raw:
        raise CvatMediaResolutionError("CVAT source-video registration is blank")
    relative = _after_registered_media_root(raw)
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".mp4":
        raise CvatMediaResolutionError(f"invalid CVAT registered media path={raw}")
    return path.as_posix()


def runtime_media_path(*, input_root: Path, registered_relative_path: object) -> Path:
    """Realize one registered path under the verified input root."""

    relative = PurePosixPath(_text(registered_relative_path))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise CvatMediaResolutionError("invalid authority-relative CVAT media path")
    root = Path(input_root).resolve()
    runtime = root.joinpath(*relative.parts)
    if not runtime.is_relative_to(root):
        raise CvatMediaResolutionError("CVAT runtime path escapes verified input root")
    return runtime


def _after_registered_media_root(value: str) -> str:
    normalized = value.strip().lstrip("/")
    parts = PurePosixPath(normalized).parts
    for index in range(len(parts) - 1):
        if parts[index : index + 2] == ("data", "videos"):
            return PurePosixPath(*parts[index:]).as_posix()
    if not value.startswith("/") and not _looks_like_windows_absolute(value):
        return PurePosixPath(value).as_posix()
    raise CvatMediaResolutionError(
        "CVAT source-video path lacks registered data/videos authority root"
    )


def _looks_like_windows_absolute(value: str) -> bool:
    return len(value) >= 3 and value[1:3] == ":/"


def _text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()
