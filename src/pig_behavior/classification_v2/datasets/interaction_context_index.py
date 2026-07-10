"""Build interaction full-frame and partner-context audit indexes."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

INTERACTION_LABELS = frozenset({"fight", "social-nose"})


@dataclass(frozen=True, slots=True)
class InteractionContextIndexConfig:
    root: Path = Path("outputs/classification_v2/train_ready_windows")
    output_dir: Path = Path("outputs/classification_v2/train_ready_windows")


@dataclass(slots=True)
class InteractionContextIndexResult:
    manifest: pd.DataFrame
    audit: dict[str, Any]
    manifest_path: Path
    audit_path: Path


def build_interaction_context_index(config: InteractionContextIndexConfig) -> InteractionContextIndexResult:
    """Build one audit row per train-ready window."""
    image_frames = pd.read_csv(config.root / "image_frame_context_manifest.csv", low_memory=False)
    image_windows = pd.read_csv(config.root / "image_window_context_manifest.csv", low_memory=False)
    split = pd.read_csv(config.root / "split_manifest.csv", low_memory=False)
    _validate_inputs(image_frames, image_windows, split)
    image_windows["window_id"] = image_windows["window_id"].astype(str)
    split["window_id"] = split["window_id"].astype(str)

    duplicate_source_frame_uid = int(image_frames["frame_uid"].duplicated().sum())
    frame_lookup_source = image_frames.sort_values(["frame_uid", "image_context_id"]).drop_duplicates(
        "frame_uid",
        keep="first",
    )
    frame_lookup = frame_lookup_source.set_index("frame_uid", drop=False)
    merged = image_windows.merge(
        split[["window_id", "behavior_window_label"]],
        on="window_id",
        how="left",
        validate="one_to_one",
    )
    manifest = merged[
        [
            "window_id",
            "source_type",
            "dataset_id",
            "video_key",
            "object_track_key",
            "pig_id",
            "track_id",
            "window_start_frame",
            "window_end_frame",
            "behavior_window_label",
        ]
    ].copy()
    manifest["is_interaction_window"] = manifest["behavior_window_label"].astype(str).isin(INTERACTION_LABELS)
    manifest["interaction_context_required"] = manifest["is_interaction_window"]
    manifest["expected_frame_slots"] = merged["frame_uid_sequence"].astype(str).map(
        lambda value: len(_split_sequence(value))
    )
    manifest["available_frame_context_rows"] = 0
    manifest["full_frame_context_available_count"] = 0
    manifest["partner_context_available_count"] = 0
    manifest["partner_count_mean"] = 0.0
    manifest["partner_count_min"] = 0.0
    manifest["partner_ids_union"] = ""
    manifest["interaction_context_ready"] = False
    manifest["interaction_context_status"] = "not_interaction"

    interaction_indices = manifest.index[manifest["is_interaction_window"]].tolist()
    for idx in interaction_indices:
        frame_uids = _split_sequence(str(merged.at[idx, "frame_uid_sequence"]))
        stats = _interaction_frame_stats(frame_lookup, frame_uids)
        status = _interaction_status(
            is_interaction=True,
            expected_slots=stats["expected_frame_slots"],
            available_rows=stats["available_frame_context_rows"],
            full_frame_count=stats["full_frame_context_available_count"],
            partner_count=stats["partner_context_available_count"],
        )
        for key, value in stats.items():
            manifest.at[idx, key] = value
        manifest.at[idx, "interaction_context_ready"] = status == "ready"
        manifest.at[idx, "interaction_context_status"] = status
    audit = _audit(manifest, duplicate_source_frame_uid=duplicate_source_frame_uid)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_dir / "interaction_window_context_manifest.csv"
    audit_path = config.output_dir / "interaction_context_audit.json"
    manifest.to_csv(manifest_path, index=False)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    if audit["errors"]:
        raise ValueError(f"interaction context index failed: {audit['errors']}")
    return InteractionContextIndexResult(
        manifest=manifest,
        audit=audit,
        manifest_path=manifest_path,
        audit_path=audit_path,
    )


def _validate_inputs(image_frames: pd.DataFrame, image_windows: pd.DataFrame, split: pd.DataFrame) -> None:
    frame_cols = {
        "frame_uid",
        "image_context_id",
        "full_frame_context_available",
        "partner_context_available",
        "interaction_partner_count",
        "interaction_partner_ids",
    }
    window_cols = {
        "window_id",
        "source_type",
        "dataset_id",
        "video_key",
        "object_track_key",
        "pig_id",
        "track_id",
        "window_start_frame",
        "window_end_frame",
        "frame_uid_sequence",
    }
    split_cols = {"window_id", "behavior_window_label"}
    missing = {
        "image_frames": sorted(frame_cols.difference(image_frames.columns)),
        "image_windows": sorted(window_cols.difference(image_windows.columns)),
        "split": sorted(split_cols.difference(split.columns)),
    }
    missing = {name: cols for name, cols in missing.items() if cols}
    if missing:
        raise ValueError(f"missing interaction context columns: {missing}")
    duplicate_split_windows = int(split["window_id"].duplicated().sum())
    if duplicate_split_windows:
        raise ValueError(f"duplicate window_id rows in split manifest: {duplicate_split_windows}")
    missing_split_windows = sorted(
        set(image_windows["window_id"].astype(str)).difference(split["window_id"].astype(str))
    )
    if missing_split_windows:
        raise ValueError(f"image context windows missing from split manifest: {len(missing_split_windows)}")


def _interaction_status(
    *,
    is_interaction: bool,
    expected_slots: int,
    available_rows: int,
    full_frame_count: int,
    partner_count: int,
) -> str:
    if not is_interaction:
        return "not_interaction"
    if expected_slots <= 0 or available_rows < expected_slots:
        return "missing_frame_context"
    if full_frame_count < expected_slots:
        return "missing_full_frame_context"
    if partner_count < expected_slots:
        return "missing_partner_context"
    return "ready"


def _interaction_frame_stats(frame_lookup: pd.DataFrame, frame_uids: list[str]) -> dict[str, Any]:
    frame_rows = frame_lookup.reindex(frame_uids)
    expected_slots = len(frame_uids)
    available = frame_rows["frame_uid"].notna()
    available_rows = int(available.sum())
    full_frame_count = int(_to_bool(frame_rows["full_frame_context_available"]).sum()) if available_rows else 0
    partner_count = int(_to_bool(frame_rows["partner_context_available"]).sum()) if available_rows else 0
    partner_counts = pd.to_numeric(frame_rows["interaction_partner_count"], errors="coerce").fillna(0)
    return {
        "expected_frame_slots": int(expected_slots),
        "available_frame_context_rows": int(available_rows),
        "full_frame_context_available_count": int(full_frame_count),
        "partner_context_available_count": int(partner_count),
        "partner_count_mean": float(partner_counts.mean()) if expected_slots else 0.0,
        "partner_count_min": float(partner_counts.min()) if expected_slots else 0.0,
        "partner_ids_union": _partner_ids_union(frame_rows["interaction_partner_ids"]) if available_rows else "",
    }


def _audit(manifest: pd.DataFrame, *, duplicate_source_frame_uid: int) -> dict[str, Any]:
    interaction = manifest[manifest["is_interaction_window"]].copy()
    duplicate_window_id = int(manifest["window_id"].duplicated().sum())
    errors: list[str] = []
    if duplicate_window_id:
        errors.append(f"duplicate_window_id={duplicate_window_id}")
    missing_labels = sorted(INTERACTION_LABELS.difference(set(interaction["behavior_window_label"].astype(str))))
    if missing_labels:
        errors.append(f"missing_interaction_labels={missing_labels}")
    return {
        "window_rows": int(len(manifest)),
        "interaction_window_rows": int(len(interaction)),
        "interaction_ready_rows": int(interaction["interaction_context_ready"].sum()) if len(interaction) else 0,
        "duplicate_window_id": duplicate_window_id,
        "duplicate_source_frame_uid_rows": int(duplicate_source_frame_uid),
        "status_counts": manifest["interaction_context_status"].value_counts(dropna=False).to_dict(),
        "interaction_status_counts": interaction["interaction_context_status"].value_counts(dropna=False).to_dict(),
        "interaction_label_counts": interaction["behavior_window_label"].value_counts(dropna=False).to_dict(),
        "interaction_source_counts": interaction["source_type"].value_counts(dropna=False).to_dict(),
        "ready_by_label": interaction.groupby("behavior_window_label")["interaction_context_ready"].sum().to_dict(),
        "ready_by_source": interaction.groupby("source_type")["interaction_context_ready"].sum().to_dict(),
        "errors": errors,
        "warnings": [
            "interaction_context_ready is an audit gate; columns from this manifest are not model inputs",
            "legacy crop-only interaction rows are expected to need separate full-frame/partner review assets",
        ],
    }


def _split_sequence(value: str) -> list[str]:
    if not value or value.lower() in {"nan", "none", "<na>"}:
        return []
    return value.split("|")


def _partner_ids_union(series: pd.Series) -> str:
    ids: set[str] = set()
    for value in series.dropna().astype(str):
        if value.lower() in {"", "nan", "none", "<na>"}:
            continue
        ids.update(part for part in re.split(r"[;,| ]+", value) if part)
    return "|".join(sorted(ids))


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})
