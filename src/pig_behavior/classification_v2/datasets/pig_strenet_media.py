"""Resolve real frame pixels for Pig-STRENet artifact construction.

The resolver deliberately separates actor crops from scene frames.  Actor
pixels prefer an existing crop artifact and fall back to a bbox crop from the
source video.  Scene pixels prefer the source video and only use an explicitly
frame-specific scene image as a fallback.  Static calibration/background
images are never accepted as temporal scene observations.
"""

from __future__ import annotations

import hashlib
from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
import pandas as pd

from pig_behavior.classification_v2.datasets.image_context_index import (
    build_video_index,
    resolve_legacy_crop,
    resolve_video,
)
from pig_behavior.classification_v2.datasets.pig_strenet_publication import (
    MEDIA_MANIFEST_SCHEMA,
    publish_media_manifest,
)

ACTOR_IMAGE_COLUMNS = (
    "crop_path",
    "actor_crop_path",
    "actor_image_path",
)
SCENE_IMAGE_COLUMNS = (
    "scene_image_path",
    "full_frame_path",
    "scene_frame_path",
    "frame_image_path",
)
FORBIDDEN_SCENE_BASENAMES = {"background.png"}


@dataclass(frozen=True, slots=True)
class ResolvedPixels:
    """One RGB pixel resolution result plus audit-only provenance."""

    image_rgb: np.ndarray | None
    status: str
    source_kind: str
    media_path: str
    frame_index: int | None
    media_key: str
    error: str

    @property
    def available(self) -> bool:
        return self.image_rgb is not None and self.status == "ok"

    def provenance(self) -> dict[str, Any]:
        return {
            "pixel_status": self.status,
            "pixel_source_kind": self.source_kind,
            "pixel_media_path": self.media_path,
            "pixel_frame_index": self.frame_index,
            "pixel_media_key": self.media_key,
            "pixel_error": self.error,
        }


class FrameMediaResolver:
    """Resolve actor and scene RGB pixels with bounded video caches."""

    def __init__(
        self,
        *,
        video_root: Path,
        legacy_crop_root: Path,
        max_open_videos: int = 2,
        max_cached_frames: int = 32,
    ) -> None:
        if max_open_videos <= 0:
            raise ValueError("max_open_videos must be positive")
        if max_cached_frames <= 0:
            raise ValueError("max_cached_frames must be positive")
        self.video_root = video_root
        self.legacy_crop_root = legacy_crop_root
        self.max_open_videos = max_open_videos
        self.max_cached_frames = max_cached_frames
        self.video_index = build_video_index(video_root)
        self._captures: OrderedDict[str, cv2.VideoCapture] = OrderedDict()
        self._next_frame: dict[str, int] = {}
        self._frame_cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._used_media: dict[str, dict[str, Any]] = {}
        self._status_counts: Counter[str] = Counter()
        self._runtime_counts: Counter[str] = Counter()
        self._rejected_scene_candidates: set[str] = set()

    def __enter__(self) -> FrameMediaResolver:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()
        self._next_frame.clear()

    def read_actor(
        self,
        row: pd.Series | Mapping[str, Any] | None,
        *,
        image_size: int,
    ) -> ResolvedPixels:
        """Read an actor crop, falling back to source-video bbox cropping."""

        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if row is None:
            return self._failure("missing_frame_row")
        direct_path = self._resolve_actor_image_path(row)
        if direct_path is not None:
            image = self._read_image_rgb(direct_path)
            if image is not None:
                resized = _resize_rgb(image, image_size)
                frame_index = _optional_int(_row_get(row, "frame_index"))
                self._mark_used(
                    direct_path,
                    source_kind="actor_crop_file",
                    frame_index=frame_index,
                )
                return self._success(
                    resized,
                    source_kind="actor_crop_file",
                    path=direct_path,
                    frame_index=frame_index,
                )
            self._runtime_counts["actor_crop_decode_failures"] += 1

        scene = self.read_scene(row)
        if not scene.available:
            return self._failure(
                "actor_scene_unavailable",
                source_kind=scene.source_kind,
                path=scene.media_path,
                frame_index=scene.frame_index,
                error=scene.status,
            )
        box = _bbox_from_row(row)
        crop = crop_rgb_box(scene.image_rgb, box, image_size=image_size)
        if crop is None:
            return self._failure(
                "invalid_actor_bbox",
                source_kind=scene.source_kind,
                path=scene.media_path,
                frame_index=scene.frame_index,
            )
        source_kind = "video_bbox_crop"
        if scene.source_kind == "scene_image_file":
            source_kind = "scene_image_bbox_crop"
        self._mark_used(
            Path(scene.media_path),
            source_kind=source_kind,
            frame_index=scene.frame_index,
        )
        return self._success(
            crop,
            source_kind=source_kind,
            path=Path(scene.media_path),
            frame_index=scene.frame_index,
        )

    def read_scene(
        self,
        row: pd.Series | Mapping[str, Any] | None,
    ) -> ResolvedPixels:
        """Read the actual full scene for ``row`` from video or frame image."""

        if row is None:
            return self._failure("missing_frame_row")
        series = row if isinstance(row, pd.Series) else pd.Series(dict(row))
        media_path = resolve_video(series, self.video_root, self.video_index)
        if media_path is not None:
            frame_index = _optional_int(_row_get(row, "frame_index"))
            if frame_index is None or frame_index < 0:
                return self._failure(
                    "invalid_video_frame_index",
                    source_kind="video_frame",
                    path=media_path,
                    frame_index=frame_index,
                )
            image = self._decode_video_rgb(media_path, frame_index)
            if image is None:
                return self._failure(
                    "video_decode_failed",
                    source_kind="video_frame",
                    path=media_path,
                    frame_index=frame_index,
                )
            if not _dimensions_match(row, image):
                return self._failure(
                    "video_dimension_mismatch",
                    source_kind="video_frame",
                    path=media_path,
                    frame_index=frame_index,
                )
            self._mark_used(
                media_path,
                source_kind="video_frame",
                frame_index=frame_index,
            )
            return self._success(
                image,
                source_kind="video_frame",
                path=media_path,
                frame_index=frame_index,
            )

        direct_path, rejected = self._resolve_scene_image_path(row)
        if rejected is not None:
            self._rejected_scene_candidates.add(str(rejected))
            return self._failure(
                "forbidden_static_scene_candidate",
                source_kind="scene_image_file",
                path=rejected,
                frame_index=_optional_int(_row_get(row, "frame_index")),
            )
        if direct_path is None:
            return self._failure("scene_media_unresolved")
        image = self._read_image_rgb(direct_path)
        frame_index = _optional_int(_row_get(row, "frame_index"))
        if image is None:
            return self._failure(
                "scene_image_decode_failed",
                source_kind="scene_image_file",
                path=direct_path,
                frame_index=frame_index,
            )
        if not _dimensions_match(row, image):
            return self._failure(
                "scene_image_dimension_mismatch",
                source_kind="scene_image_file",
                path=direct_path,
                frame_index=frame_index,
            )
        self._mark_used(
            direct_path,
            source_kind="scene_image_file",
            frame_index=frame_index,
        )
        return self._success(
            image,
            source_kind="scene_image_file",
            path=direct_path,
            frame_index=frame_index,
        )

    def manifest(self) -> dict[str, Any]:
        """Return hashes and runtime provenance for media actually consumed."""

        self.close()
        sources: list[dict[str, Any]] = []
        valid = True
        for path_text, usage in sorted(self._used_media.items()):
            path = Path(path_text)
            exists = path.is_file()
            digest = _sha256_file(path) if exists else None
            valid = valid and exists and digest is not None
            frame_indices = sorted(usage["frame_indices"])
            sources.append(
                {
                    "path": path_text,
                    "exists": exists,
                    "size": int(path.stat().st_size) if exists else None,
                    "sha256": digest,
                    "authority_mode": (
                        "full_file_sha256" if exists else "unavailable"
                    ),
                    "authority_sha256": digest,
                    "authority_valid": bool(exists and digest),
                    "derived_pixel_artifact_binding_required": False,
                    "source_kind_counts": dict(
                        sorted(usage["source_kind_counts"].items())
                    ),
                    "frame_index_count": len(frame_indices),
                    "frame_index_min": frame_indices[0] if frame_indices else None,
                    "frame_index_max": frame_indices[-1] if frame_indices else None,
                    "frame_indices_sha256": _ordered_values_sha256(frame_indices),
                }
            )
        return {
            "schema_version": MEDIA_MANIFEST_SCHEMA,
            "video_root": str(self.video_root),
            "legacy_crop_root": str(self.legacy_crop_root),
            "video_index_aliases": len(self.video_index),
            "source_file_count": len(sources),
            "sources": sources,
            "status_counts": dict(sorted(self._status_counts.items())),
            "runtime_counts": dict(sorted(self._runtime_counts.items())),
            "full_file_sha256_count": len(sources),
            "derived_pixel_video_authority_count": 0,
            "rejected_static_scene_candidates": sorted(
                self._rejected_scene_candidates
            ),
            "background_as_temporal_scene_used": False,
            "valid": bool(sources) and bool(valid),
        }

    def write_manifest(
        self,
        path: Path,
        *,
        checkpoint_path: Path | None = None,
        progress_callback: (
            Callable[[str, int | None, int | None], None] | None
        ) = None,
    ) -> dict[str, Any]:
        """Publish media authority with durable per-file hash checkpoints."""

        self.close()
        return publish_media_manifest(
            path,
            video_root=self.video_root,
            legacy_crop_root=self.legacy_crop_root,
            video_index_aliases=len(self.video_index),
            usage=self._used_media,
            status_counts=self._status_counts,
            runtime_counts=self._runtime_counts,
            rejected_scene_candidates=sorted(
                self._rejected_scene_candidates
            ),
            checkpoint_path=(
                checkpoint_path
                or path.parent / ".checkpoints" / "media_publication.sqlite3"
            ),
            progress_callback=progress_callback,
        )

    def _resolve_actor_image_path(
        self,
        row: pd.Series | Mapping[str, Any],
    ) -> Path | None:
        for column in ACTOR_IMAGE_COLUMNS:
            value = _row_get(row, column)
            if _is_missing(value):
                continue
            raw = Path(str(value).strip())
            for candidate in (raw, self.legacy_crop_root / raw):
                if candidate.is_file():
                    return candidate.resolve()
            narrowed = pd.Series({"crop_path": str(value).strip()})
            resolved = resolve_legacy_crop(narrowed, self.legacy_crop_root)
            if resolved is not None and resolved.is_file():
                return resolved.resolve()
        return None

    def _resolve_scene_image_path(
        self,
        row: pd.Series | Mapping[str, Any],
    ) -> tuple[Path | None, Path | None]:
        for column in SCENE_IMAGE_COLUMNS:
            value = _row_get(row, column)
            if _is_missing(value):
                continue
            raw = Path(str(value).strip())
            for candidate in (raw, self.legacy_crop_root / raw):
                if not candidate.is_file():
                    continue
                resolved = candidate.resolve()
                name = resolved.name.casefold()
                if name in FORBIDDEN_SCENE_BASENAMES or "image #1" in name:
                    return None, resolved
                return resolved, None
        return None, None

    def _read_image_rgb(self, path: Path) -> np.ndarray | None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        self._runtime_counts["direct_image_reads"] += 1
        if image is None:
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _decode_video_rgb(
        self,
        path: Path,
        frame_index: int,
    ) -> np.ndarray | None:
        path = path.resolve()
        path_text = str(path)
        cache_key = (path_text, frame_index)
        cached = self._frame_cache.pop(cache_key, None)
        if cached is not None:
            self._frame_cache[cache_key] = cached
            self._runtime_counts["video_frame_cache_hits"] += 1
            return cached

        capture = self._captures.pop(path_text, None)
        if capture is None:
            while len(self._captures) >= self.max_open_videos:
                stale_path, stale_capture = self._captures.popitem(last=False)
                stale_capture.release()
                self._next_frame.pop(stale_path, None)
            capture = cv2.VideoCapture(path_text)
            self._runtime_counts["video_open_count"] += 1
        self._captures[path_text] = capture
        if not capture.isOpened():
            return None
        if self._next_frame.get(path_text) != frame_index:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            self._runtime_counts["video_seek_count"] += 1
        ok, bgr = capture.read()
        if not ok or bgr is None:
            return None
        self._next_frame[path_text] = frame_index + 1
        self._runtime_counts["video_decode_count"] += 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._frame_cache[cache_key] = rgb
        while len(self._frame_cache) > self.max_cached_frames:
            self._frame_cache.popitem(last=False)
        return rgb

    def _mark_used(
        self,
        path: Path,
        *,
        source_kind: str,
        frame_index: int | None,
    ) -> None:
        path_text = str(path.resolve())
        usage = self._used_media.setdefault(
            path_text,
            {
                "source_kind_counts": Counter(),
                "frame_indices": set(),
            },
        )
        usage["source_kind_counts"][source_kind] += 1
        if frame_index is not None:
            usage["frame_indices"].add(int(frame_index))

    def _success(
        self,
        image: np.ndarray,
        *,
        source_kind: str,
        path: Path,
        frame_index: int | None,
    ) -> ResolvedPixels:
        self._status_counts["ok"] += 1
        path_text = str(path.resolve())
        return ResolvedPixels(
            image_rgb=image,
            status="ok",
            source_kind=source_kind,
            media_path=path_text,
            frame_index=frame_index,
            media_key=_media_key(source_kind, path_text, frame_index),
            error="",
        )

    def _failure(
        self,
        status: str,
        *,
        source_kind: str = "none",
        path: Path | str | None = None,
        frame_index: int | None = None,
        error: str = "",
    ) -> ResolvedPixels:
        self._status_counts[status] += 1
        path_text = "" if path is None else str(Path(path).resolve())
        return ResolvedPixels(
            image_rgb=None,
            status=status,
            source_kind=source_kind,
            media_path=path_text,
            frame_index=frame_index,
            media_key=(
                ""
                if not path_text
                else _media_key(source_kind, path_text, frame_index)
            ),
            error=error,
        )


def crop_rgb_box(
    image_rgb: np.ndarray | None,
    box: tuple[float, float, float, float] | None,
    *,
    image_size: int,
) -> np.ndarray | None:
    """Clip an xyxy box, crop an RGB image, and resize it to a square."""

    if image_rgb is None or box is None or image_size <= 0:
        return None
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        return None
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = box
    ix1 = max(0, min(width, int(np.floor(x1))))
    iy1 = max(0, min(height, int(np.floor(y1))))
    ix2 = max(0, min(width, int(np.ceil(x2))))
    iy2 = max(0, min(height, int(np.ceil(y2))))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return _resize_rgb(image_rgb[iy1:iy2, ix1:ix2], image_size)


def _bbox_from_row(
    row: pd.Series | Mapping[str, Any],
) -> tuple[float, float, float, float] | None:
    values = [
        pd.to_numeric(_row_get(row, column), errors="coerce")
        for column in ("x1", "y1", "x2", "y2")
    ]
    if not np.isfinite(values).all():
        return None
    x1, y1, x2, y2 = map(float, values)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _dimensions_match(
    row: pd.Series | Mapping[str, Any],
    image: np.ndarray,
) -> bool:
    expected_width = _optional_int(_row_get(row, "image_width"))
    expected_height = _optional_int(_row_get(row, "image_height"))
    if expected_width is None or expected_height is None:
        return True
    height, width = image.shape[:2]
    return width == expected_width and height == expected_height


def _resize_rgb(image: np.ndarray, image_size: int) -> np.ndarray:
    interpolation = (
        cv2.INTER_AREA
        if image.shape[0] > image_size or image.shape[1] > image_size
        else cv2.INTER_LINEAR
    )
    return cv2.resize(
        image,
        (image_size, image_size),
        interpolation=interpolation,
    )


def _row_get(
    row: pd.Series | Mapping[str, Any],
    key: str,
) -> Any:
    return row.get(key)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().casefold() in {"", "nan", "none", "<na>"}


def _optional_int(value: Any) -> int | None:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return None
    return int(number)


def _media_key(
    source_kind: str,
    path_text: str,
    frame_index: int | None,
) -> str:
    value = f"{source_kind}|{path_text}|{frame_index}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ordered_values_sha256(values: list[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "FrameMediaResolver",
    "MEDIA_MANIFEST_SCHEMA",
    "ResolvedPixels",
    "crop_rgb_box",
]
