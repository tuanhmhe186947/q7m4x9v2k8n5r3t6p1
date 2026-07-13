"""Build label-independent scene/partner context audit indexes.

The original manifest focused on fight/social-nose windows. The S1 multimodal
contract also needs scene and partner readiness for every train-ready window so
the visual-context branch is gated by asset/geometry availability, not by the
ground-truth behavior label.
"""

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


def build_interaction_context_index(
    config: InteractionContextIndexConfig,
) -> InteractionContextIndexResult:
    """Build one audit row per train-ready window with label-independent context stats."""
    image_frames = pd.read_csv(config.root / "image_frame_context_manifest.csv", low_memory=False)
    image_windows = pd.read_csv(config.root / "image_window_context_manifest.csv", low_memory=False)
    split = pd.read_csv(config.root / "split_manifest.csv", low_memory=False)
    _validate_inputs(image_frames, image_windows, split)
    image_windows["window_id"] = image_windows["window_id"].astype(str)
    split["window_id"] = split["window_id"].astype(str)

    # frame_uid identifies a video frame, not an actor. image_context_id is the
    # actor-frame key and prevents one pig from inheriting another pig's context.
    # Input validation proves this key is unique, so no row is selected silently.
    frame_lookup = _build_frame_lookup(image_frames)
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
    manifest["is_interaction_window"] = (
        manifest["behavior_window_label"].astype(str).isin(INTERACTION_LABELS)
    )
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
    manifest["scene_context_required"] = True
    manifest["scene_partner_context_required"] = True
    manifest["scene_context_ready"] = False
    manifest["scene_partner_context_ready"] = False
    manifest["scene_partner_context_status"] = "not_evaluated"
    manifest["scene_partner_context_policy"] = "label_independent_asset_geometry_gate"
    manifest["interaction_context_ready"] = False
    manifest["interaction_context_status"] = "not_interaction"

    stats_by_key = {
        key: _interaction_frame_stats(frame_lookup, _split_context_sequence(key))
        for key in merged["image_context_id_sequence"].fillna("").astype(str).unique()
    }
    stats_records: list[dict[str, Any]] = []
    scene_statuses: list[str] = []
    context_sequences = (
        merged["image_context_id_sequence"].fillna("").astype(str).tolist()
    )
    for image_context_id_sequence in context_sequences:
        stats = stats_by_key[image_context_id_sequence]
        stats_records.append(stats)
        scene_statuses.append(
            _scene_partner_status(
                expected_slots=stats["expected_frame_slots"],
                available_rows=stats["available_frame_context_rows"],
                full_frame_count=stats["full_frame_context_available_count"],
                partner_count=stats["partner_context_available_count"],
            )
        )
    stats_df = pd.DataFrame(stats_records, index=manifest.index)
    for key in stats_df.columns:
        manifest[key] = stats_df[key]
    manifest["scene_context_ready"] = [
        status in {"ready", "missing_partner_context"}
        for status in scene_statuses
    ]
    manifest["scene_partner_context_ready"] = [status == "ready" for status in scene_statuses]
    manifest["scene_partner_context_status"] = scene_statuses

    # Vectorized assignment matters here: this manifest contains every training
    # window, and scalar DataFrame writes made a metadata-only rebuild minutes long.
    interaction_mask = manifest["is_interaction_window"].astype(bool)
    manifest.loc[interaction_mask, "interaction_context_ready"] = manifest.loc[
        interaction_mask, "scene_partner_context_status"
    ].eq("ready")
    manifest.loc[interaction_mask, "interaction_context_status"] = manifest.loc[
        interaction_mask, "scene_partner_context_status"
    ].astype(str)
    audit = _audit(manifest)
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


def _validate_inputs(
    image_frames: pd.DataFrame,
    image_windows: pd.DataFrame,
    split: pd.DataFrame,
) -> None:
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
        "image_context_id_sequence",
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
    frame_context_ids = _clean_keys(image_frames["image_context_id"])
    image_window_ids = _clean_keys(image_windows["window_id"])
    split_window_ids = _clean_keys(split["window_id"])
    errors: list[str] = []
    errors.extend(_key_errors(frame_context_ids, "image_context_id"))
    errors.extend(_key_errors(image_window_ids, "image_window_id"))
    errors.extend(_key_errors(split_window_ids, "split_window_id"))

    image_window_set = set(image_window_ids)
    split_window_set = set(split_window_ids)
    missing_split_windows = image_window_set.difference(split_window_set)
    missing_context_windows = split_window_set.difference(image_window_set)
    if missing_split_windows:
        errors.append(f"image_windows_missing_from_split={len(missing_split_windows)}")
    if missing_context_windows:
        errors.append(f"split_windows_missing_image_context={len(missing_context_windows)}")
    errors.extend(_window_sequence_errors(image_frames, image_windows))
    if errors:
        raise ValueError(f"interaction context input contract failed: {errors}")


def _clean_keys(series: pd.Series) -> pd.Series:
    """Normalize audit keys without converting missing values to valid IDs."""

    return series.fillna("").astype(str).str.strip()


def _key_errors(keys: pd.Series, name: str) -> list[str]:
    """Return blank and duplicate key violations for one manifest column."""

    errors: list[str] = []
    blank = int(keys.eq("").sum())
    duplicate = int(keys.duplicated(keep=False).sum())
    if blank:
        errors.append(f"blank_{name}={blank}")
    if duplicate:
        errors.append(f"duplicate_{name}_rows={duplicate}")
    return errors


def _window_sequence_errors(
    image_frames: pd.DataFrame,
    image_windows: pd.DataFrame,
) -> list[str]:
    """Prove frame and actor-context sequences have one aligned item per slot."""

    frame_uid_by_context = dict(
        zip(
            _clean_keys(image_frames["image_context_id"]),
            _clean_keys(image_frames["frame_uid"]),
            strict=True,
        )
    )
    blank_sequences = 0
    length_mismatches = 0
    duplicate_frame_slots = 0
    duplicate_context_slots = 0
    missing_context_rows = 0
    frame_context_mismatches = 0
    for row in image_windows.itertuples(index=False):
        frame_uids = _split_sequence(str(row.frame_uid_sequence))
        context_ids = _split_context_sequence(str(row.image_context_id_sequence))
        if not frame_uids or not context_ids:
            blank_sequences += 1
            continue
        if len(frame_uids) != len(context_ids):
            length_mismatches += 1
            continue
        duplicate_frame_slots += int(len(frame_uids) != len(set(frame_uids)))
        duplicate_context_slots += int(len(context_ids) != len(set(context_ids)))
        for frame_uid, context_id in zip(frame_uids, context_ids, strict=True):
            source_frame_uid = frame_uid_by_context.get(context_id)
            if source_frame_uid is None:
                missing_context_rows += 1
            elif source_frame_uid != frame_uid:
                frame_context_mismatches += 1

    counts = {
        "blank_window_sequences": blank_sequences,
        "window_sequence_length_mismatches": length_mismatches,
        "windows_with_duplicate_frame_slots": duplicate_frame_slots,
        "windows_with_duplicate_context_slots": duplicate_context_slots,
        "missing_context_sequence_rows": missing_context_rows,
        "frame_context_sequence_mismatches": frame_context_mismatches,
    }
    return [f"{name}={count}" for name, count in counts.items() if count]


def _scene_partner_status(
    *,
    expected_slots: int,
    available_rows: int,
    full_frame_count: int,
    partner_count: int,
) -> str:
    if expected_slots <= 0 or available_rows < expected_slots:
        return "missing_frame_context"
    if full_frame_count < expected_slots:
        return "missing_full_frame_context"
    if partner_count < expected_slots:
        return "missing_partner_context"
    return "ready"


def _build_frame_lookup(frame_lookup_source: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Precompute frame context records so every window can be audited cheaply."""
    records: dict[str, dict[str, Any]] = {}
    for row in frame_lookup_source.itertuples(index=False):
        records[str(row.image_context_id)] = {
            "full_frame_context_available": _bool_scalar(row.full_frame_context_available),
            "partner_context_available": _bool_scalar(row.partner_context_available),
            "interaction_partner_count": row.interaction_partner_count,
            "interaction_partner_ids": row.interaction_partner_ids,
        }
    return records


def _interaction_frame_stats(
    frame_lookup: dict[str, dict[str, Any]],
    frame_uids: list[str],
) -> dict[str, Any]:
    expected_slots = len(frame_uids)
    frame_rows = [frame_lookup[uid] for uid in frame_uids if uid in frame_lookup]
    available_rows = len(frame_rows)
    full_frame_count = sum(1 for row in frame_rows if row["full_frame_context_available"])
    partner_count = sum(1 for row in frame_rows if row["partner_context_available"])
    # Avoid constructing a pandas Series for every window. With tens of
    # thousands of windows this metadata calculation otherwise dominates runtime.
    partner_counts = [_nonnegative_float(row["interaction_partner_count"]) for row in frame_rows]
    return {
        "expected_frame_slots": int(expected_slots),
        "available_frame_context_rows": int(available_rows),
        "full_frame_context_available_count": int(full_frame_count),
        "partner_context_available_count": int(partner_count),
        "partner_count_mean": (
            float(sum(partner_counts) / available_rows) if available_rows else 0.0
        ),
        "partner_count_min": float(min(partner_counts)) if available_rows else 0.0,
        "partner_ids_union": (
            _partner_ids_union(
                [row["interaction_partner_ids"] for row in frame_rows]
            )
            if available_rows
            else ""
        ),
    }


def _nonnegative_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(number):
        return 0.0
    return max(0.0, number)


def _audit(manifest: pd.DataFrame) -> dict[str, Any]:
    interaction = manifest[manifest["is_interaction_window"]].copy()
    non_interaction = manifest[~manifest["is_interaction_window"]].copy()
    duplicate_window_id = int(manifest["window_id"].duplicated().sum())
    errors: list[str] = []
    if duplicate_window_id:
        errors.append(f"duplicate_window_id={duplicate_window_id}")
    missing_labels = sorted(
        INTERACTION_LABELS.difference(
            set(interaction["behavior_window_label"].astype(str))
        )
    )
    not_evaluated = int(
        manifest["scene_partner_context_status"]
        .astype(str)
        .eq("not_evaluated")
        .sum()
    )
    if not_evaluated:
        errors.append(f"scene_partner_context_not_evaluated={not_evaluated}")
    non_interaction_ready = (
        int(non_interaction["scene_partner_context_ready"].sum())
        if len(non_interaction)
        else 0
    )
    if len(non_interaction) and non_interaction_ready == 0:
        errors.append("scene_partner_context_appears_label_gated")
    return {
        "window_rows": int(len(manifest)),
        "interaction_window_rows": int(len(interaction)),
        "interaction_ready_rows": (
            int(interaction["interaction_context_ready"].sum())
            if len(interaction)
            else 0
        ),
        "non_interaction_window_rows": int(len(non_interaction)),
        "scene_context_ready_rows": int(manifest["scene_context_ready"].sum()),
        "scene_partner_context_ready_rows": int(manifest["scene_partner_context_ready"].sum()),
        "non_interaction_scene_partner_ready_rows": int(non_interaction_ready),
        "duplicate_window_id": duplicate_window_id,
        "duplicate_source_image_context_id_rows": 0,
        "missing_interaction_labels": missing_labels,
        "scene_partner_status_counts": manifest[
            "scene_partner_context_status"
        ].value_counts(dropna=False).to_dict(),
        "scene_partner_status_by_source": manifest.groupby("source_type")[
            "scene_partner_context_status"
        ]
        .value_counts(dropna=False)
        .unstack(fill_value=0)
        .to_dict(),
        "status_counts": manifest["interaction_context_status"]
        .value_counts(dropna=False)
        .to_dict(),
        "interaction_status_counts": interaction["interaction_context_status"]
        .value_counts(dropna=False)
        .to_dict(),
        "interaction_label_counts": interaction["behavior_window_label"]
        .value_counts(dropna=False)
        .to_dict(),
        "interaction_source_counts": interaction["source_type"]
        .value_counts(dropna=False)
        .to_dict(),
        "ready_by_label": interaction.groupby("behavior_window_label")[
            "interaction_context_ready"
        ].sum().to_dict(),
        "ready_by_source": interaction.groupby("source_type")[
            "interaction_context_ready"
        ].sum().to_dict(),
        "errors": errors,
        "warnings": [
            *(
                [f"interaction_labels_without_support={missing_labels}"]
                if missing_labels
                else []
            ),
            "interaction_context_ready is an audit gate; its columns are not "
            "model inputs",
            "scene_partner_context_ready is computed for every window without "
            "behavior-label gating",
            "legacy crop-only interaction rows need separate full-frame/partner "
            "review assets",
        ],
    }


def _split_sequence(value: str) -> list[str]:
    if not value or value.lower() in {"nan", "none", "<na>"}:
        return []
    return value.split("|")


def _split_context_sequence(value: str) -> list[str]:
    """Split actor-frame IDs, whose internal fields already contain pipes."""

    if not value or value.lower() in {"nan", "none", "<na>"}:
        return []
    return value.split(";;")


def _partner_ids_union(values: list[Any]) -> str:
    ids: set[str] = set()
    for value in values:
        if pd.isna(value):
            continue
        text = str(value)
        if text.lower() in {"", "nan", "none", "<na>"}:
            continue
        ids.update(part for part in re.split(r"[;,| ]+", text) if part)
    return "|".join(sorted(ids))


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def _bool_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}
