"""Image sequence dataset for classification_v2 multimodal smoke training.

The dataset consumes the audited image-context manifests. It returns image
tensors plus masks and metadata; path/source/review identifiers are used only
for loading and audit, not as model features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from pig_behavior.classification_v2.datasets.image_context_index import IMAGE_CONTEXT_SEQUENCE_DELIMITER

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


@dataclass(frozen=True, slots=True)
class ImageSequenceDatasetConfig:
    frame_context_csv: Path = Path("outputs/classification_v2/train_ready_windows/image_frame_context_manifest.csv")
    window_context_csv: Path = Path("outputs/classification_v2/train_ready_windows/image_window_context_manifest.csv")
    image_size: int = 128
    max_windows: int | None = None
    require_complete: bool = True


class ClassificationV2ImageSequenceDataset(Dataset[dict[str, Any]]):
    """Load legacy crop sequences and CVAT video+bbox crop sequences."""

    def __init__(self, config: ImageSequenceDatasetConfig) -> None:
        if config.image_size <= 0:
            raise ValueError("image_size must be positive")
        self.config = config
        self.frames = pd.read_csv(config.frame_context_csv, low_memory=False)
        self.windows = pd.read_csv(config.window_context_csv, low_memory=False)
        self._validate_manifests()
        if config.require_complete:
            self.windows = self.windows[_to_bool(self.windows["window_image_context_complete"])].copy()
        if config.max_windows is not None:
            if config.max_windows <= 0:
                raise ValueError("max_windows must be positive")
            self.windows = self.windows.head(config.max_windows).copy()
        self.windows = self.windows.reset_index(drop=True)
        self.frame_by_context_id = self.frames.set_index("image_context_id", drop=False).to_dict("index")
        self._capture_cache: dict[str, Any] = {}

    def __len__(self) -> int:
        return int(len(self.windows))

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows.iloc[index]
        context_ids = _split_context_id_sequence(str(window["image_context_id_sequence"]))
        expected_frames = _split_sequence(str(window["expected_frame_indices"]))
        sequence_len = len(context_ids)
        images = np.zeros((sequence_len, 3, self.config.image_size, self.config.image_size), dtype=np.float32)
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
            image = self._load_frame_image(frame)
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
            raise ValueError(f"missing image context columns: frames={missing_frame} windows={missing_window}")
        duplicate_context = int(self.frames["image_context_id"].duplicated().sum())
        if duplicate_context:
            raise ValueError(f"duplicate image_context_id rows: {duplicate_context}")

    def _load_frame_image(self, frame: dict[str, Any]) -> np.ndarray | None:
        source_type = str(frame.get("source_type", ""))
        if source_type == "legacy_recovered":
            return _load_legacy_crop(Path(str(frame["resolved_media_path"])), self.config.image_size)
        if source_type == "cvat_tracking_xml":
            return self._load_cvat_crop(frame)
        return None

    def _load_cvat_crop(self, frame: dict[str, Any]) -> np.ndarray | None:
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
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, image_bgr = capture.read()
        if not ok or image_bgr is None:
            return None
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
        crop_rgb = cv2.resize(
            crop_rgb,
            (self.config.image_size, self.config.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
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
            image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
            return _to_chw_float(np.asarray(image, dtype=np.uint8))
    except Exception:
        return None


def _to_chw_float(image_rgb: np.ndarray) -> np.ndarray:
    return np.transpose(image_rgb.astype(np.float32) / 255.0, (2, 0, 1))


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
