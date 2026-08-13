"""Resolve Legacy context IDs to canonical authority-relative JPEG paths."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath

import pandas as pd

LEGACY_SOURCE_RESOLUTION_VERSION = "canonical_relative_jpeg_v1"
CANONICAL_LEGACY_CROP_ROOT = PurePosixPath(
    "outputs/legacy_16f_rebuild/legacy_16f_rebuild_20260718_v2/"
    "06_full_recovery/crops"
)


class LegacyMediaResolutionError(ValueError):
    """Raised when a Legacy context lacks one canonical JPEG mapping."""


def attach_canonical_legacy_media_paths(frames: pd.DataFrame) -> pd.DataFrame:
    """Replace only Legacy loader paths with canonical relative JPEG paths."""

    required = {"source_type", "image_context_id", "resolved_media_path"}
    missing = sorted(required.difference(frames.columns))
    if missing:
        raise LegacyMediaResolutionError(f"Legacy media fields missing={missing}")
    resolved = frames.copy()
    legacy = resolved["source_type"].astype(str).eq("legacy_recovered")
    if not legacy.any():
        return resolved

    selected = resolved.loc[
        legacy,
        ["image_context_id", "resolved_media_path"],
    ].copy()
    selected["image_context_id"] = selected["image_context_id"].map(_text)
    selected["resolved_media_path"] = selected["resolved_media_path"].map(
        canonical_relative_media_path
    )
    if selected["image_context_id"].eq("").any():
        raise LegacyMediaResolutionError("Legacy image context ID is blank")
    counts = selected.groupby("image_context_id", sort=True)[
        "resolved_media_path"
    ].nunique(dropna=False)
    ambiguous = sorted(counts.loc[counts.ne(1)].index.astype(str))
    if ambiguous:
        raise LegacyMediaResolutionError(
            "Legacy image context ID has non-unique JPEG mapping="
            f"{ambiguous[:5]}"
        )
    resolved.loc[legacy, "resolved_media_path"] = selected[
        "resolved_media_path"
    ].to_numpy()
    return resolved


def canonical_relative_media_path(value: object) -> str:
    """Return one canonical JPEG path below the registered input root."""

    raw = _text(value)
    normalized = raw.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or PureWindowsPath(raw).is_absolute()
        or ".." in path.parts
        or path.suffix.lower() not in {".jpg", ".jpeg"}
        or not path.is_relative_to(CANONICAL_LEGACY_CROP_ROOT)
    ):
        raise LegacyMediaResolutionError(
            f"invalid canonical Legacy media path={raw}"
        )
    return path.as_posix()


def runtime_media_path(*, input_root: Path, canonical_relative_path: object) -> Path:
    """Realize one canonical Legacy JPEG below the verified input root."""

    relative = PurePosixPath(canonical_relative_media_path(canonical_relative_path))
    root = Path(input_root).resolve()
    runtime = root.joinpath(*relative.parts)
    if not runtime.is_relative_to(root):
        raise LegacyMediaResolutionError(
            "Legacy runtime path escapes verified input root"
        )
    return runtime


def _text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


__all__ = [
    "CANONICAL_LEGACY_CROP_ROOT",
    "LEGACY_SOURCE_RESOLUTION_VERSION",
    "LegacyMediaResolutionError",
    "attach_canonical_legacy_media_paths",
    "canonical_relative_media_path",
    "runtime_media_path",
]
