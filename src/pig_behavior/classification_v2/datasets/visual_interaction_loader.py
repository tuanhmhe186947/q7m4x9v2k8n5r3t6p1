"""Load cached actor-partner visual context as masked window sequences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CONTEXT_SEQUENCE_DELIMITER = ";;"


@dataclass(frozen=True, slots=True)
class VisualInteractionDatasetConfig:
    cache_manifest_csv: Path
    window_context_csv: Path = Path(
        "outputs/classification_v2/train_ready_windows/image_window_context_manifest.csv"
    )
    max_windows: int | None = None
    packed_cache_npy: Path | None = None
    packed_cache_index_csv: Path | None = None
    require_packed_cache: bool = False


class VisualInteractionWindowDataset(Dataset[dict[str, Any]]):
    """Return ``[T,C,H,W]`` context tensors and explicit availability masks."""

    def __init__(self, config: VisualInteractionDatasetConfig) -> None:
        self.config = config
        cache = pd.read_csv(config.cache_manifest_csv, low_memory=False)
        windows = pd.read_csv(config.window_context_csv, low_memory=False)
        _validate_inputs(cache, windows)
        if config.max_windows is not None:
            if config.max_windows <= 0:
                raise ValueError("max_windows must be positive")
            windows = windows.head(config.max_windows).copy()
        self.windows = windows.reset_index(drop=True)
        self.cache_root = config.cache_manifest_csv.parent
        self.image_size = _single_image_size(cache)
        self.cache_by_image_context_id = {
            str(row.image_context_id): row
            for row in cache.sort_values("image_context_id").itertuples(index=False)
        }
        self.packed_cache: np.ndarray | None = None
        self.packed_row_by_context_id: dict[str, int] = {}
        self.packed_cache_hits = 0
        self.packed_cache_misses = 0
        self.individual_cache_loads = 0
        if config.packed_cache_npy is not None or config.packed_cache_index_csv is not None:
            if config.packed_cache_npy is None or config.packed_cache_index_csv is None:
                raise ValueError("packed visual cache requires both tensor and index paths")
            self.packed_cache = np.load(config.packed_cache_npy, mmap_mode="r")
            packed_index = pd.read_csv(config.packed_cache_index_csv, low_memory=False)
            _validate_packed_cache(self.packed_cache, packed_index, self.image_size)
            self.packed_row_by_context_id = dict(
                zip(
                    packed_index["image_context_id"].astype(str),
                    pd.to_numeric(packed_index["packed_row"], errors="raise").astype(int),
                    strict=True,
                )
            )
        if config.require_packed_cache and self.packed_cache is None:
            raise ValueError("require_packed_cache needs packed tensor and index paths")

    def load_audit(self) -> dict[str, Any]:
        """Summarize cache coverage without exposing metadata as model features."""

        rows = list(self.cache_by_image_context_id.values())
        available = sum(1 for row in rows if _bool_scalar(row.visual_context_available))
        return {
            "window_rows": int(len(self.windows)),
            "cache_manifest_rows": int(len(rows)),
            "cache_available_rows": int(available),
            "cache_unavailable_rows": int(len(rows) - available),
            "image_size": int(self.image_size),
            "packed_cache_configured": self.packed_cache is not None,
            "packed_index_rows": int(len(self.packed_row_by_context_id)),
            "packed_cache_hits": int(self.packed_cache_hits),
            "packed_cache_misses": int(self.packed_cache_misses),
            "individual_cache_loads": int(self.individual_cache_loads),
        }

    def __len__(self) -> int:
        return int(len(self.windows))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.windows.iloc[index]
        context_ids = _split_context_sequence(str(row["image_context_id_sequence"]))
        images = np.zeros((len(context_ids), 3, self.image_size, self.image_size), dtype=np.float32)
        observed = np.zeros(len(context_ids), dtype=np.float32)
        statuses: list[str] = []
        errors: list[str] = []
        for slot, context_id in enumerate(context_ids):
            cache_row = self.cache_by_image_context_id.get(context_id)
            if cache_row is None:
                statuses.append("missing_cache_manifest_row")
                continue
            status = str(cache_row.visual_context_status)
            statuses.append(status)
            if not _bool_scalar(cache_row.visual_context_available):
                continue
            try:
                packed_row = self.packed_row_by_context_id.get(context_id)
                if self.packed_cache is not None and packed_row is not None:
                    image = np.asarray(self.packed_cache[packed_row])
                    self.packed_cache_hits += 1
                else:
                    self.packed_cache_misses += 1
                    if self.config.require_packed_cache:
                        errors.append(f"{context_id}:missing_required_packed_cache_row")
                        continue
                    cache_path = self.cache_root / str(cache_row.cache_path)
                    image = np.load(cache_path)
                    self.individual_cache_loads += 1
            except Exception as exc:
                errors.append(f"{context_id}:cache_load_failed:{type(exc).__name__}")
                continue
            if image.shape != (self.image_size, self.image_size, 3) or image.dtype != np.uint8:
                errors.append(f"{context_id}:invalid_tensor:{image.shape}:{image.dtype}")
                continue
            images[slot] = np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))
            observed[slot] = 1.0
        return {
            "visual_context_image": torch.from_numpy(images),
            "visual_context_length_mask": torch.ones(len(context_ids), dtype=torch.float32),
            "visual_context_observed_mask": torch.from_numpy(observed),
            "window_id": str(row["window_id"]),
            "context_statuses": statuses,
            "errors": errors,
        }


def visual_interaction_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad visual context sequences while preserving metadata outside model X."""

    if not batch:
        raise ValueError("cannot collate empty visual interaction batch")
    max_length = max(int(item["visual_context_image"].shape[0]) for item in batch)
    shape = tuple(batch[0]["visual_context_image"].shape[1:])
    images = batch[0]["visual_context_image"].new_zeros((len(batch), max_length, *shape))
    length = batch[0]["visual_context_length_mask"].new_zeros((len(batch), max_length))
    observed = batch[0]["visual_context_observed_mask"].new_zeros((len(batch), max_length))
    for index, item in enumerate(batch):
        value = item["visual_context_image"]
        if tuple(value.shape[1:]) != shape:
            raise ValueError("visual context images must share [C,H,W] shape")
        count = int(value.shape[0])
        images[index, :count] = value
        length[index, :count] = item["visual_context_length_mask"]
        observed[index, :count] = item["visual_context_observed_mask"]
    return {
        "visual_context_image": images,
        "visual_context_length_mask": length,
        "visual_context_observed_mask": observed,
        "window_id": [item["window_id"] for item in batch],
        "context_statuses": [item["context_statuses"] for item in batch],
        "errors": [item["errors"] for item in batch],
    }


def _validate_inputs(cache: pd.DataFrame, windows: pd.DataFrame) -> None:
    cache_required = {
        "image_context_id",
        "visual_context_available",
        "visual_context_status",
        "cache_path",
        "image_size",
    }
    window_required = {"window_id", "image_context_id_sequence"}
    missing_cache = sorted(cache_required.difference(cache.columns))
    missing_windows = sorted(window_required.difference(windows.columns))
    if missing_cache or missing_windows:
        raise ValueError(f"missing visual loader columns: cache={missing_cache}, windows={missing_windows}")
    duplicate_cache = int(cache["image_context_id"].duplicated().sum())
    duplicate_windows = int(windows["window_id"].duplicated().sum())
    if duplicate_cache or duplicate_windows:
        raise ValueError(
            f"duplicate visual loader keys: image_context_id={duplicate_cache}, window_id={duplicate_windows}"
        )


def _single_image_size(cache: pd.DataFrame) -> int:
    sizes = pd.to_numeric(cache["image_size"], errors="coerce").dropna().astype(int).unique().tolist()
    if len(sizes) != 1 or sizes[0] <= 0:
        raise ValueError(f"cache manifest must contain one positive image_size, got {sizes}")
    return int(sizes[0])


def _validate_packed_cache(tensor: np.ndarray, index: pd.DataFrame, image_size: int) -> None:
    required = {"image_context_id", "packed_row"}
    missing = sorted(required.difference(index.columns))
    if missing:
        raise ValueError(f"packed visual cache index missing columns: {missing}")
    if index["image_context_id"].duplicated().any():
        raise ValueError("packed visual cache index has duplicate image_context_id")
    expected_shape = (len(index), image_size, image_size, 3)
    if tensor.shape != expected_shape or tensor.dtype != np.uint8:
        raise ValueError(
            f"packed visual cache contract mismatch: shape={tensor.shape}, dtype={tensor.dtype}, "
            f"expected={expected_shape}/uint8"
        )
    rows = pd.to_numeric(index["packed_row"], errors="raise").astype(int).to_numpy()
    if not np.array_equal(np.sort(rows), np.arange(len(index), dtype=int)):
        raise ValueError("packed visual cache rows must be a contiguous permutation")


def _split_context_sequence(value: str) -> list[str]:
    if not value or value.lower() in {"nan", "none", "<na>"}:
        return []
    return value.split(CONTEXT_SEQUENCE_DELIMITER)


def _bool_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}
