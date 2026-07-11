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
            cache_path = self.cache_root / str(cache_row.cache_path)
            try:
                image = np.load(cache_path)
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


def _split_context_sequence(value: str) -> list[str]:
    if not value or value.lower() in {"nan", "none", "<na>"}:
        return []
    return value.split(CONTEXT_SEQUENCE_DELIMITER)


def _bool_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}
