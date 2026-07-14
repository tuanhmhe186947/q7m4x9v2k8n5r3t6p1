"""Image sequence dataset for classification_v2 multimodal smoke training.

The dataset consumes the audited image-context manifests. It returns image
tensors plus masks and metadata; path/source/review identifiers are used only
for loading and audit, not as model features.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from pig_behavior.classification_v2.datasets.image_context_index import (
    IMAGE_CONTEXT_SEQUENCE_DELIMITER,
)

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


@dataclass(frozen=True, slots=True)
class ImageSequenceDatasetConfig:
    frame_context_csv: Path = Path(
        "outputs/classification_v2/train_ready_windows/"
        "image_frame_context_manifest.csv"
    )
    window_context_csv: Path = Path(
        "outputs/classification_v2/train_ready_windows/"
        "image_window_context_manifest.csv"
    )
    image_cache_manifest_csv: Path | None = None
    packed_image_cache_npy: Path | None = None
    packed_image_cache_index_csv: Path | None = None
    image_size: int = 128
    max_windows: int | None = None
    require_complete: bool = True
    require_cached_images: bool = False
    image_cache_size: int = 8192


class ClassificationV2ImageSequenceDataset(Dataset[dict[str, Any]]):
    """Load audited crop files or video+bbox sequences for cache construction."""

    def __init__(self, config: ImageSequenceDatasetConfig) -> None:
        if config.image_size <= 0:
            raise ValueError("image_size must be positive")
        if config.image_cache_size < 0:
            raise ValueError("image_cache_size must be non-negative")
        self.config = config
        self.frames = pd.read_csv(config.frame_context_csv, low_memory=False)
        self.windows = pd.read_csv(config.window_context_csv, low_memory=False)
        self._validate_manifests()
        self.cache_by_context_id = self._load_cache_manifest(config.image_cache_manifest_csv)
        self._packed_tensor, self.packed_row_by_context_id = self._load_packed_cache(
            config.packed_image_cache_npy,
            config.packed_image_cache_index_csv,
        )
        if config.require_complete:
            complete = _to_bool(
                self.windows["window_image_context_complete"]
            )
            self.windows = self.windows[complete].copy()
        if config.max_windows is not None:
            if config.max_windows <= 0:
                raise ValueError("max_windows must be positive")
            self.windows = self.windows.head(config.max_windows).copy()
        self.windows = self.windows.reset_index(drop=True)
        self.frame_by_context_id = self.frames.set_index(
            "image_context_id",
            drop=False,
        ).to_dict("index")
        self._capture_cache: dict[str, Any] = {}
        self._capture_next_frame: dict[str, int] = {}
        self._decoded_video_frame: dict[str, tuple[int, np.ndarray]] = {}
        self._image_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.video_decode_count = 0
        self.video_seek_count = 0
        self.video_frame_reuse_count = 0
        self.memory_image_cache_hits = 0
        self.disk_image_cache_hits = 0
        self.packed_image_cache_hits = 0
        self.disk_image_cache_misses = 0
        self.source_image_loads = 0

    def __len__(self) -> int:
        return int(len(self.windows))

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows.iloc[index]
        context_ids = _split_context_id_sequence(str(window["image_context_id_sequence"]))
        expected_frames = _split_sequence(str(window["expected_frame_indices"]))
        sequence_len = len(context_ids)
        images = np.zeros(
            (
                sequence_len,
                3,
                self.config.image_size,
                self.config.image_size,
            ),
            dtype=np.float32,
        )
        length_mask = np.ones((sequence_len,), dtype=np.float32)
        observed_mask = np.zeros((sequence_len,), dtype=np.float32)
        errors: list[str] = []

        for pos, context_id in enumerate(context_ids):
            if not context_id:
                errors.append(f"missing_context_id@{pos}")
                continue
            frame = self.frame_by_context_id.get(context_id)
            if frame is None:
                errors.append(f"context_id_not_found@{pos}")
                continue
            image = self._load_context_image(context_id, frame)
            if image is None:
                errors.append(f"image_load_failed@{pos}")
                continue
            images[pos] = image
            observed_mask[pos] = 1.0

        return {
            "image": torch.from_numpy(images),
            "length_mask": torch.from_numpy(length_mask),
            "observed_mask": torch.from_numpy(observed_mask),
            "window_id": str(window["window_id"]),
            "source_type": str(window.get("source_type", "")),
            "video_key": str(window.get("video_key", "")),
            "image_context_ids": context_ids,
            "expected_frame_indices": expected_frames,
            "errors": errors,
        }

    def close(self) -> None:
        for capture in self._capture_cache.values():
            try:
                capture.release()
            except Exception:
                pass
        self._capture_cache.clear()
        self._capture_next_frame.clear()
        self._decoded_video_frame.clear()
        self._image_cache.clear()
        self._packed_tensor = None

    def _validate_manifests(self) -> None:
        frame_required = {
            "image_context_id",
            "source_type",
            "resolved_media_path",
            "image_context_loadable",
            "x1",
            "y1",
            "x2",
            "y2",
            "frame_index",
        }
        window_required = {
            "window_id",
            "source_type",
            "video_key",
            "expected_frame_indices",
            "image_context_id_sequence",
            "window_image_context_complete",
        }
        missing_frame = sorted(frame_required.difference(self.frames.columns))
        missing_window = sorted(window_required.difference(self.windows.columns))
        if missing_frame or missing_window:
            raise ValueError(
                "missing image context columns: "
                f"frames={missing_frame} windows={missing_window}"
            )
        duplicate_context = int(self.frames["image_context_id"].duplicated().sum())
        if duplicate_context:
            raise ValueError(f"duplicate image_context_id rows: {duplicate_context}")

    def _load_frame_image(self, frame: dict[str, Any]) -> np.ndarray | None:
        source_type = str(frame.get("source_type", ""))
        if source_type == "legacy_recovered":
            if str(frame.get("image_context_source", "")) == "legacy_video_bbox":
                return self._load_video_bbox_crop(frame)
            return _load_legacy_crop(
                Path(str(frame["resolved_media_path"])),
                self.config.image_size,
            )
        if source_type == "cvat_tracking_xml":
            return self._load_video_bbox_crop(frame)
        return None

    def _load_context_image(self, context_id: str, frame: dict[str, Any]) -> np.ndarray | None:
        """Load one crop with a bounded LRU cache keyed by audited context ID."""

        cache_size = int(self.config.image_cache_size)
        if cache_size > 0 and context_id in self._image_cache:
            image = self._image_cache.pop(context_id)
            self._image_cache[context_id] = image
            self.memory_image_cache_hits += 1
            return image
        image = self._load_cached_context_image(context_id)
        if image is not None:
            self.disk_image_cache_hits += 1
        elif self._cache_configured():
            self.disk_image_cache_misses += 1
            if self.config.require_cached_images:
                return None
        if image is None:
            image = self._load_frame_image(frame)
            if image is not None:
                self.source_image_loads += 1
        if image is not None and cache_size > 0:
            self._image_cache[context_id] = image
            while len(self._image_cache) > cache_size:
                self._image_cache.popitem(last=False)
        return image

    def image_load_audit(self) -> dict[str, Any]:
        """Expose cache/source counters so training cannot hide fallback I/O."""

        return {
            "cache_manifest_configured": self._cache_configured(),
            "packed_cache_configured": bool(
                self.config.packed_image_cache_npy is not None
                and self.config.packed_image_cache_index_csv is not None
            ),
            "require_cached_images": bool(self.config.require_cached_images),
            "memory_image_cache_hits": int(self.memory_image_cache_hits),
            "disk_image_cache_hits": int(self.disk_image_cache_hits),
            "packed_image_cache_hits": int(self.packed_image_cache_hits),
            "disk_image_cache_misses": int(self.disk_image_cache_misses),
            "source_image_loads": int(self.source_image_loads),
        }

    def _load_cache_manifest(self, manifest_csv: Path | None) -> dict[str, Path]:
        """Map audited context IDs to prebuilt crop files without changing labels."""

        if manifest_csv is None:
            return {}
        manifest_path = Path(manifest_csv)
        if not manifest_path.exists():
            raise FileNotFoundError(f"image cache manifest not found: {manifest_path}")
        manifest = pd.read_csv(manifest_path, low_memory=False)
        required = {"image_context_id", "cache_path", "image_size", "cache_format"}
        missing = sorted(required.difference(manifest.columns))
        if missing:
            raise ValueError(f"image cache manifest missing columns: {missing}")
        size_mismatch = manifest[
            pd.to_numeric(manifest["image_size"], errors="coerce").ne(
                self.config.image_size
            )
        ]
        if len(size_mismatch):
            raise ValueError(
                "image cache size mismatch: "
                f"expected {self.config.image_size}, "
                f"found {len(size_mismatch)} rows"
            )
        duplicate_context = int(manifest["image_context_id"].duplicated().sum())
        if duplicate_context:
            raise ValueError(f"duplicate cached image_context_id rows: {duplicate_context}")
        base = manifest_path.parent
        out: dict[str, Path] = {}
        for row in manifest.itertuples(index=False):
            cache_path = Path(str(row.cache_path))
            if not cache_path.is_absolute():
                cache_path = base / cache_path
            out[str(row.image_context_id)] = cache_path
        return out

    def _load_packed_cache(
        self,
        tensor_npy: Path | None,
        index_csv: Path | None,
    ) -> tuple[np.ndarray | None, dict[str, int]]:
        """Open a row-addressable mmap cache only when tensor and index agree."""

        if (tensor_npy is None) != (index_csv is None):
            raise ValueError(
                "packed_image_cache_npy and packed_image_cache_index_csv "
                "must be provided together"
            )
        if tensor_npy is None or index_csv is None:
            return None, {}
        tensor_path = Path(tensor_npy)
        index_path = Path(index_csv)
        if not tensor_path.exists() or not index_path.exists():
            raise FileNotFoundError(
                "packed image cache missing: "
                f"tensor={tensor_path} index={index_path}"
            )
        tensor = np.load(tensor_path, mmap_mode="r")
        expected_tail = (self.config.image_size, self.config.image_size, 3)
        if tensor.dtype != np.uint8 or tensor.ndim != 4 or tuple(tensor.shape[1:]) != expected_tail:
            raise ValueError(
                "packed image tensor contract mismatch: "
                f"dtype={tensor.dtype} shape={tensor.shape}"
            )
        index = pd.read_csv(index_path, low_memory=False)
        required = {"image_context_id", "packed_row"}
        missing = sorted(required.difference(index.columns))
        if missing:
            raise ValueError(f"packed image index missing columns: {missing}")
        if index["image_context_id"].duplicated().any():
            raise ValueError("packed image index has duplicate image_context_id rows")
        rows = pd.to_numeric(index["packed_row"], errors="coerce")
        if rows.isna().any() or (rows < 0).any() or (rows >= tensor.shape[0]).any():
            raise ValueError("packed image index contains invalid packed_row values")
        mapping = dict(zip(index["image_context_id"].astype(str), rows.astype(int), strict=True))
        return tensor, mapping

    def _load_cached_context_image(self, context_id: str) -> np.ndarray | None:
        """Read a pre-resized RGB cache item and return the model CHW float tensor."""

        packed_row = self.packed_row_by_context_id.get(context_id)
        if packed_row is not None and self._packed_tensor is not None:
            cached = np.asarray(self._packed_tensor[packed_row])
            self.packed_image_cache_hits += 1
            return _to_chw_float(cached)
        cache_path = self.cache_by_context_id.get(context_id)
        if cache_path is None or not cache_path.exists():
            return None
        try:
            cached = np.load(cache_path)
        except Exception:
            return None
        if cached.dtype == np.uint8 and cached.ndim == 3 and cached.shape[-1] == 3:
            return _to_chw_float(cached)
        if np.issubdtype(cached.dtype, np.floating) and cached.ndim == 3 and cached.shape[0] == 3:
            return cached.astype(np.float32)
        return None

    def _cache_configured(self) -> bool:
        return bool(
            self.config.image_cache_manifest_csv is not None
            or (
                self.config.packed_image_cache_npy is not None
                and self.config.packed_image_cache_index_csv is not None
            )
        )

    def _load_video_bbox_crop(self, frame: dict[str, Any]) -> np.ndarray | None:
        """Load one bbox crop while reusing sequentially decoded full frames.

        The cache builder sorts requests by video/frame. This method therefore
        reads consecutive frames without seeking and decodes a shared frame
        once when several pigs have boxes on that frame.
        """

        if cv2 is None:
            return None
        video_path = str(frame.get("resolved_media_path", ""))
        if not video_path:
            return None
        capture = self._capture_cache.get(video_path)
        if capture is None:
            capture = cv2.VideoCapture(video_path)
            self._capture_cache[video_path] = capture
        if not capture.isOpened():
            return None
        frame_index = pd.to_numeric(frame.get("frame_index"), errors="coerce")
        bbox = [pd.to_numeric(frame.get(col), errors="coerce") for col in ["x1", "y1", "x2", "y2"]]
        if pd.isna(frame_index) or any(pd.isna(value) for value in bbox):
            return None
        target_frame = int(frame_index)
        decoded = self._decoded_video_frame.get(video_path)
        if decoded is not None and decoded[0] == target_frame:
            image_bgr = decoded[1]
            self.video_frame_reuse_count += 1
        else:
            if self._capture_next_frame.get(video_path) != target_frame:
                capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                self.video_seek_count += 1
            ok, image_bgr = capture.read()
            if not ok or image_bgr is None:
                return None
            self._capture_next_frame[video_path] = target_frame + 1
            self._decoded_video_frame[video_path] = (target_frame, image_bgr)
            self.video_decode_count += 1
        height, width = image_bgr.shape[:2]
        x1, y1, x2, y2 = [float(value) for value in bbox]
        x1i = max(0, min(width, int(x1)))
        y1i = max(0, min(height, int(y1)))
        x2i = max(0, min(width, int(x2)))
        y2i = max(0, min(height, int(y2)))
        if x2i <= x1i or y2i <= y1i:
            return None
        crop_bgr = image_bgr[y1i:y2i, x1i:x2i]
        if crop_bgr.size == 0:
            return None
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_rgb = letterbox_rgb_uint8(crop_rgb, self.config.image_size)
        return _to_chw_float(crop_rgb)


def image_sequence_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate image sequence examples into padded batch tensors."""
    return {
        "image": _pad_stack([item["image"] for item in batch]),
        "length_mask": _pad_stack_1d([item["length_mask"] for item in batch]),
        "observed_mask": _pad_stack_1d([item["observed_mask"] for item in batch]),
        "window_id": [item["window_id"] for item in batch],
        "source_type": [item["source_type"] for item in batch],
        "video_key": [item["video_key"] for item in batch],
        "image_context_ids": [item["image_context_ids"] for item in batch],
        "expected_frame_indices": [item["expected_frame_indices"] for item in batch],
        "errors": [item["errors"] for item in batch],
    }


def _pad_stack(values: list[torch.Tensor]) -> torch.Tensor:
    if not values:
        raise ValueError("cannot collate empty image batch")
    max_len = max(int(value.shape[0]) for value in values)
    tail_shape = tuple(values[0].shape[1:])
    out = values[0].new_zeros((len(values), max_len, *tail_shape))
    for idx, value in enumerate(values):
        if tuple(value.shape[1:]) != tail_shape:
            raise ValueError("all image tensors must share [C,H,W] shape")
        out[idx, : value.shape[0]] = value
    return out


def _pad_stack_1d(values: list[torch.Tensor]) -> torch.Tensor:
    if not values:
        raise ValueError("cannot collate empty mask batch")
    max_len = max(int(value.shape[0]) for value in values)
    out = values[0].new_zeros((len(values), max_len))
    for idx, value in enumerate(values):
        out[idx, : value.shape[0]] = value
    return out


def _load_legacy_crop(path: Path, image_size: int) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        with Image.open(path) as image:
            image = _letterbox_pil_rgb(image.convert("RGB"), image_size)
            return _to_chw_float(np.asarray(image, dtype=np.uint8))
    except Exception:
        return None


def _letterbox_pil_rgb(image: Image.Image, image_size: int) -> Image.Image:
    """Resize without distorting pig aspect ratio, padding to a square canvas."""

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("cannot letterbox an empty image")
    scale = min(image_size / width, image_size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    left = (image_size - resized_width) // 2
    top = (image_size - resized_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def letterbox_rgb_uint8(image_rgb: np.ndarray, image_size: int) -> np.ndarray:
    """Letterbox an RGB uint8 crop to square HWC uint8 without aspect distortion."""

    if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
        raise ValueError("letterbox_rgb_uint8 expects an RGB HWC image")
    image = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
    return np.asarray(_letterbox_pil_rgb(image, image_size))


def _to_chw_float(image_rgb: np.ndarray) -> np.ndarray:
    return np.transpose(image_rgb.astype(np.float32) / 255.0, (2, 0, 1))


def context_cache_relative_path(image_context_id: str) -> Path:
    """Create a deterministic sharded cache path for one image context ID."""

    digest = hashlib.sha1(str(image_context_id).encode("utf-8")).hexdigest()
    return Path(digest[:2]) / f"{digest}.npy"


def chw_float_to_hwc_uint8(image_chw: np.ndarray) -> np.ndarray:
    """Convert loader output back to compact RGB uint8 for disk cache storage."""

    clipped = np.clip(image_chw, 0.0, 1.0)
    return np.transpose((clipped * 255.0).round().astype(np.uint8), (1, 2, 0))


def _split_sequence(value: str) -> list[str]:
    if not value or value.lower() in {"nan", "none", "<na>"}:
        return []
    return value.split("|")


def _split_context_id_sequence(value: str) -> list[str]:
    if not value or value.lower() in {"nan", "none", "<na>"}:
        return []
    if IMAGE_CONTEXT_SEQUENCE_DELIMITER in value:
        return value.split(IMAGE_CONTEXT_SEQUENCE_DELIMITER)
    starts = _context_id_starts(value)
    if not starts:
        return [value]
    out: list[str] = []
    for pos, start in enumerate(starts):
        end = starts[pos + 1] if pos + 1 < len(starts) else len(value)
        item = value[start:end].strip("|")
        if item:
            out.append(item)
    return out


def _context_id_starts(value: str) -> list[int]:
    starts: list[int] = []
    markers = ["cvat_tracking_xml|cvat_tracking_xml|", "legacy_recovered|legacy_recovered|"]
    for idx in range(len(value)):
        if idx != 0 and value[idx - 1] != "|":
            continue
        for marker in markers:
            if value.startswith(marker, idx):
                starts.append(idx)
                break
    return starts


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})
