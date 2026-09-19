"""Interaction-context tensor loader for classification_v2.

The full-frame/partner visual branch will eventually load images, but it first
needs a strict data boundary: context availability must be derived from audited
asset/geometry readiness, not from behavior labels or review decisions. This
loader converts the interaction-context manifest into small numeric tensors and
masks that can be smoke-tested without using source, identity, label, review, or
path metadata as model input.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

INTERACTION_CONTEXT_FEATURE_COLUMNS: tuple[str, ...] = (
    "available_frame_context_ratio",
    "full_frame_context_ratio",
    "partner_context_ratio",
    "partner_count_mean",
    "partner_count_min",
)


@dataclass(frozen=True, slots=True)
class InteractionContextDatasetConfig:
    manifest_csv: Path = Path("outputs/classification_v2/train_ready_windows/interaction_window_context_manifest.csv")
    require_ready: bool = False
    max_windows: int | None = None


class InteractionContextWindowDataset(Dataset[dict[str, Any]]):
    """Return label-independent scene/partner context tensors per window."""

    def __init__(self, config: InteractionContextDatasetConfig) -> None:
        self.config = config
        self.manifest = pd.read_csv(config.manifest_csv, low_memory=False)
        _validate_manifest(self.manifest)
        self.manifest = _add_context_features(self.manifest)
        if config.require_ready:
            self.manifest = self.manifest[_to_bool(self.manifest["scene_partner_context_ready"])].copy()
        if config.max_windows is not None:
            if config.max_windows <= 0:
                raise ValueError("max_windows must be positive")
            self.manifest = self.manifest.head(config.max_windows).copy()
        self.manifest = self.manifest.reset_index(drop=True)

    def __len__(self) -> int:
        return int(len(self.manifest))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        features = row.loc[list(INTERACTION_CONTEXT_FEATURE_COLUMNS)].astype(float).to_numpy(dtype=np.float32)
        ready = bool(_bool_scalar(row["scene_partner_context_ready"]))
        scene_ready = bool(_bool_scalar(row["scene_context_ready"]))
        return {
            "interaction_context_features": torch.from_numpy(features),
            "interaction_context_available_mask": torch.tensor(1.0 if ready else 0.0, dtype=torch.float32),
            "scene_context_available_mask": torch.tensor(1.0 if scene_ready else 0.0, dtype=torch.float32),
            "window_id": str(row["window_id"]),
            "context_status": str(row["scene_partner_context_status"]),
            "errors": [],
        }


def interaction_context_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate interaction-context examples while keeping metadata out of X."""

    if not batch:
        raise ValueError("cannot collate empty interaction context batch")
    return {
        "interaction_context_features": torch.stack([item["interaction_context_features"] for item in batch]),
        "interaction_context_available_mask": torch.stack(
            [item["interaction_context_available_mask"] for item in batch]
        ),
        "scene_context_available_mask": torch.stack([item["scene_context_available_mask"] for item in batch]),
        "window_id": [item["window_id"] for item in batch],
        "context_status": [item["context_status"] for item in batch],
        "errors": [item["errors"] for item in batch],
    }


def _validate_manifest(manifest: pd.DataFrame) -> None:
    """Validate the manifest fields needed to build non-leaky context tensors."""

    required = {
        "window_id",
        "expected_frame_slots",
        "available_frame_context_rows",
        "full_frame_context_available_count",
        "partner_context_available_count",
        "partner_count_mean",
        "partner_count_min",
        "scene_context_ready",
        "scene_partner_context_ready",
        "scene_partner_context_status",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"interaction context manifest missing columns: {missing}")
    duplicate_windows = int(manifest["window_id"].duplicated().sum())
    if duplicate_windows:
        raise ValueError(f"duplicate interaction context window_id rows: {duplicate_windows}")


def _add_context_features(manifest: pd.DataFrame) -> pd.DataFrame:
    """Create bounded numeric features from label-independent context counts."""

    out = manifest.copy()
    expected = pd.to_numeric(out["expected_frame_slots"], errors="coerce").fillna(0).clip(lower=0)
    denom = expected.replace(0, np.nan)
    for source_col, out_col in [
        ("available_frame_context_rows", "available_frame_context_ratio"),
        ("full_frame_context_available_count", "full_frame_context_ratio"),
        ("partner_context_available_count", "partner_context_ratio"),
    ]:
        values = pd.to_numeric(out[source_col], errors="coerce").fillna(0).clip(lower=0)
        out[out_col] = (values / denom).fillna(0).clip(0, 1)
    for col in ["partner_count_mean", "partner_count_min"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).clip(lower=0)
    return out


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def _bool_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}
