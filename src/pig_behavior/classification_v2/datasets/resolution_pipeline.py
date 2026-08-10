"""Resolution-independent actor-RGB binding and bounded audit helpers.

The scientific observation identity is the selected actor frame plus its
authoritative source, crop and temporal-window binding.  Requested spatial
resolution is deliberately excluded from that identity and is applied only by
the runtime dataset transform.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from pig_behavior.classification_v2.datasets.image_context_index import (
    IMAGE_CONTEXT_SEQUENCE_DELIMITER,
)
from pig_behavior.classification_v2.datasets.image_sequence_dataset import (
    ClassificationV2ImageSequenceDataset,
    ImageSequenceDatasetConfig,
)

SUPPORTED_RUNTIME_INPUT_RESOLUTIONS = (64, 160, 224)
RUNTIME_RGB_TRANSFORM_VERSION = "actor_crop_letterbox_pil_bilinear_v1"
INNER_ROLES = frozenset({"train", "validation"})


@dataclass(frozen=True, slots=True)
class ResolutionIndependentRGBBinding:
    """Inner-only RGB observations whose identity does not include resolution."""

    frames: pd.DataFrame
    windows: pd.DataFrame
    selection: pd.DataFrame
    media_root: Path
    identity_sha256: str
    observation_count: int
    window_count: int

    def build_dataset(
        self,
        input_resolution: int,
        *,
        image_cache_size: int = 8192,
        video_capture_cache_size: int = 8,
    ) -> ClassificationV2ImageSequenceDataset:
        """Build a source-backed runtime realization without a persistent cache."""

        validate_runtime_resolution(input_resolution)
        return ClassificationV2ImageSequenceDataset(
            ImageSequenceDatasetConfig(
                frame_context_dataframe=self.frames,
                window_context_dataframe=self.windows,
                image_size=input_resolution,
                require_complete=True,
                image_cache_size=image_cache_size,
                video_capture_cache_size=video_capture_cache_size,
                media_root=self.media_root,
            )
        )

    def runtime_realization(self, input_resolution: int) -> dict[str, Any]:
        """Return the hash-bound runtime-only fields for a proposed ablation arm."""

        validate_runtime_resolution(input_resolution)
        payload = {
            "scientific_identity_sha256": self.identity_sha256,
            "input_resolution": int(input_resolution),
            "runtime_transform_version": RUNTIME_RGB_TRANSFORM_VERSION,
            "persistent_rgb_cache": "none_source_backed_runtime",
        }
        payload["runtime_realization_sha256"] = _payload_sha256(payload)
        return payload


def build_inner_resolution_binding(
    *,
    frame_context_csv: Path,
    window_context_csv: Path,
    inner_selection_csv: Path,
    media_root: Path,
    expected_window_count: int | None = None,
    expected_observation_count: int | None = None,
) -> ResolutionIndependentRGBBinding:
    """Bind train/validation observations without opening any selected media."""

    frames = pd.read_csv(frame_context_csv, low_memory=False)
    windows = pd.read_csv(window_context_csv, low_memory=False)
    selection = pd.read_csv(inner_selection_csv, low_memory=False)
    return build_inner_resolution_binding_from_dataframes(
        frames=frames,
        windows=windows,
        selection=selection,
        media_root=media_root,
        expected_window_count=expected_window_count,
        expected_observation_count=expected_observation_count,
    )


def build_inner_resolution_binding_from_dataframes(
    *,
    frames: pd.DataFrame,
    windows: pd.DataFrame,
    selection: pd.DataFrame,
    media_root: Path,
    expected_window_count: int | None = None,
    expected_observation_count: int | None = None,
) -> ResolutionIndependentRGBBinding:
    """Build the binding from already-loaded manifests for deterministic tests."""

    _require_columns(
        frames,
        {
            "image_context_id",
            "source_type",
            "video_key",
            "object_track_key",
            "frame_index",
            "resolved_media_path",
            "image_context_source",
            "image_context_loadable",
            "image_width",
            "image_height",
            "x1",
            "y1",
            "x2",
            "y2",
        },
        "frame context",
    )
    _require_columns(
        windows,
        {
            "window_id",
            "source_type",
            "window_length_frames",
            "view_type",
            "image_context_id_sequence",
            "window_image_context_complete",
        },
        "window context",
    )
    _require_columns(
        selection,
        {
            "window_row_index",
            "window_id",
            "view_type",
            "source_type",
            "behavior_window_label",
            "window_valid_for_main_train",
            "primary_s1_role",
            "primary_s1_eligible",
        },
        "inner selection",
    )
    if frames["image_context_id"].duplicated().any():
        raise ValueError("frame context has duplicate image_context_id")

    selected = selection.loc[
        _as_bool(selection["window_valid_for_main_train"])
        & _as_bool(selection["primary_s1_eligible"])
        & selection["primary_s1_role"].astype(str).isin(INNER_ROLES)
    ].copy()
    if selected.empty:
        raise ValueError("inner selection is empty after role and validity guards")
    if not selected["primary_s1_role"].astype(str).isin(INNER_ROLES).all():
        raise ValueError("outer role survived inner selection guard")
    if not selected["view_type"].astype(str).str.contains("T6").all():
        raise ValueError("inner selection contains a non-T6 temporal view")
    if selected["window_row_index"].duplicated().any():
        raise ValueError("inner selection has duplicate window_row_index")

    positions = pd.to_numeric(selected["window_row_index"], errors="coerce")
    if positions.isna().any() or (positions < 0).any() or (positions >= len(windows)).any():
        raise ValueError("inner selection has invalid window_row_index")
    selected_windows = windows.iloc[positions.astype(int).to_numpy()].copy()
    selected_windows.index = selected.index
    if not selected_windows["window_id"].astype(str).equals(
        selected["window_id"].astype(str)
    ):
        raise ValueError("inner selection window_id does not match window manifest")
    if not selected_windows["source_type"].astype(str).equals(
        selected["source_type"].astype(str)
    ):
        raise ValueError("inner selection source_type does not match window manifest")
    if not _as_bool(selected_windows["window_image_context_complete"]).all():
        raise ValueError("inner selection contains incomplete RGB windows")

    context_to_labels: dict[str, set[str]] = {}
    context_to_roles: dict[str, set[str]] = {}
    ordered_context_ids: list[str] = []
    for window, selected_row in zip(
        selected_windows.itertuples(index=False),
        selected.itertuples(index=False),
        strict=True,
    ):
        context_ids = split_image_context_sequence(
            str(window.image_context_id_sequence)
        )
        if len(context_ids) != int(window.window_length_frames):
            raise ValueError(
                "window context cardinality mismatch: "
                f"window_id={window.window_id}"
            )
        if len(context_ids) != 6:
            raise ValueError(f"inner T6 window has {len(context_ids)} observations")
        label = str(selected_row.behavior_window_label)
        role = str(selected_row.primary_s1_role)
        for context_id in context_ids:
            ordered_context_ids.append(context_id)
            context_to_labels.setdefault(context_id, set()).add(label)
            context_to_roles.setdefault(context_id, set()).add(role)

    multi_label = sorted(
        context_id
        for context_id, labels in context_to_labels.items()
        if len(labels) != 1
    )
    multi_role = sorted(
        context_id
        for context_id, roles in context_to_roles.items()
        if len(roles) != 1
    )
    if multi_label:
        raise ValueError(
            "one observation maps to multiple labels: "
            f"count={len(multi_label)} first={multi_label[0]}"
        )
    if multi_role:
        raise ValueError(
            "one observation maps to multiple split roles: "
            f"count={len(multi_role)} first={multi_role[0]}"
        )

    unique_context_ids = set(context_to_labels)
    selected_frames = frames.loc[
        frames["image_context_id"].astype(str).isin(unique_context_ids)
    ].copy()
    if len(selected_frames) != len(unique_context_ids):
        raise ValueError(
            "selected frame accounting mismatch: "
            f"expected={len(unique_context_ids)} observed={len(selected_frames)}"
        )
    if not _as_bool(selected_frames["image_context_loadable"]).all():
        raise ValueError("inner binding includes an unloadable media observation")
    if selected_frames["resolved_media_path"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("inner binding includes a blank media path")
    selected_frames["_resolution_binding_label"] = selected_frames[
        "image_context_id"
    ].astype(str).map(lambda context_id: next(iter(context_to_labels[context_id])))
    selected_frames["_resolution_binding_role"] = selected_frames[
        "image_context_id"
    ].astype(str).map(lambda context_id: next(iter(context_to_roles[context_id])))
    selected_windows["_resolution_binding_label"] = selected[
        "behavior_window_label"
    ].astype(str)
    selected_windows["_resolution_binding_role"] = selected[
        "primary_s1_role"
    ].astype(str)

    identity_sha256 = scientific_rgb_identity_sha256(
        selected_frames,
        selected_windows,
    )
    if expected_window_count is not None and len(selected_windows) != expected_window_count:
        raise ValueError(
            "inner window count mismatch: "
            f"expected={expected_window_count} observed={len(selected_windows)}"
        )
    if (
        expected_observation_count is not None
        and len(selected_frames) != expected_observation_count
    ):
        raise ValueError(
            "inner observation count mismatch: "
            f"expected={expected_observation_count} observed={len(selected_frames)}"
        )
    return ResolutionIndependentRGBBinding(
        frames=selected_frames.reset_index(drop=True),
        windows=selected_windows.reset_index(drop=True),
        selection=selected.reset_index(drop=True),
        media_root=Path(media_root),
        identity_sha256=identity_sha256,
        observation_count=len(selected_frames),
        window_count=len(selected_windows),
    )


def validate_runtime_resolution(input_resolution: int) -> None:
    """Reject nonregistered spatial realizations before a model run starts."""

    if input_resolution not in SUPPORTED_RUNTIME_INPUT_RESOLUTIONS:
        raise ValueError(
            "unsupported input_resolution="
            f"{input_resolution}; expected={SUPPORTED_RUNTIME_INPUT_RESOLUTIONS}"
        )


def split_image_context_sequence(value: str) -> list[str]:
    """Split the registered context-id encoding without heuristic fallback."""

    if not value or value.lower() in {"nan", "none", "<na>"}:
        return []
    return [
        item
        for item in value.split(IMAGE_CONTEXT_SEQUENCE_DELIMITER)
        if item
    ]


def scientific_rgb_identity_sha256(
    frames: pd.DataFrame,
    windows: pd.DataFrame,
) -> str:
    """Hash source/crop/split/temporal identity while deliberately excluding size."""

    frame_columns = [
        "image_context_id",
        "source_type",
        "video_key",
        "object_track_key",
        "frame_index",
        "resolved_media_path",
        "image_context_source",
        "x1",
        "y1",
        "x2",
        "y2",
        "_resolution_binding_role",
    ]
    window_columns = [
        "window_id",
        "source_type",
        "view_type",
        "window_length_frames",
        "image_context_id_sequence",
        "_resolution_binding_role",
    ]
    _require_columns(frames, set(frame_columns), "identity frames")
    _require_columns(windows, set(window_columns), "identity windows")
    payload = {
        "identity_schema_version": "classification_v2_scientific_rgb_identity_v1",
        "frames": _stable_records(frames, frame_columns, "image_context_id"),
        "windows": _stable_records(windows, window_columns, "window_id"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def resolve_audited_media_path(value: object, media_root: Path) -> Path:
    """Resolve only a manifest-declared relative media path against its root."""

    path = Path(str(value))
    return path if path.is_absolute() else Path(media_root) / path


def scan_legacy_jpeg_headers(
    frames: pd.DataFrame,
    *,
    media_root: Path,
    output_csv: Path,
    checkpoint_json: Path,
    workers: int = 4,
    checkpoint_every: int = 1000,
) -> dict[str, Any]:
    """Read only selected JPEG metadata with resumable deterministic checkpoints."""

    if workers <= 0 or workers > 8:
        raise ValueError("workers must be between 1 and 8")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    _require_columns(
        frames,
        {"image_context_id", "source_type", "resolved_media_path"},
        "header scan frames",
    )
    legacy = frames.loc[
        frames["source_type"].astype(str).eq("legacy_recovered")
    ].copy()
    legacy = legacy.sort_values("image_context_id", kind="mergesort").reset_index(drop=True)
    selection_sha256 = _stable_table_sha256(
        legacy,
        ["image_context_id", "resolved_media_path"],
        "image_context_id",
    )
    existing = _load_header_checkpoint(
        output_csv=output_csv,
        checkpoint_json=checkpoint_json,
        selection_sha256=selection_sha256,
    )
    expected_ids = legacy["image_context_id"].astype(str).tolist()
    existing_ids = set(existing["image_context_id"].astype(str))
    unknown_ids = existing_ids.difference(expected_ids)
    if unknown_ids:
        raise ValueError("header checkpoint has observations outside inner binding")
    todo = legacy.loc[
        ~legacy["image_context_id"].astype(str).isin(existing_ids)
    ].copy()
    records = existing.to_dict("records")
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="jpeg-header") as pool:
        for index, record in enumerate(
            pool.map(
                _read_legacy_jpeg_header,
                [
                    (
                        str(row.image_context_id),
                        resolve_audited_media_path(
                            row.resolved_media_path,
                            media_root,
                        ),
                    )
                    for row in todo.itertuples(index=False)
                ],
            ),
            start=1,
        ):
            records.append(record)
            if index % checkpoint_every == 0:
                _write_header_checkpoint(
                    records,
                    output_csv=output_csv,
                    checkpoint_json=checkpoint_json,
                    selection_sha256=selection_sha256,
                    expected_count=len(legacy),
                )
    _write_header_checkpoint(
        records,
        output_csv=output_csv,
        checkpoint_json=checkpoint_json,
        selection_sha256=selection_sha256,
        expected_count=len(legacy),
    )
    result = pd.DataFrame(records)
    completed = int(len(result))
    failed = int(result["status"].ne("ok").sum()) if completed else 0
    return {
        "schema_version": "classification_v2_legacy_jpeg_header_scan_v1",
        "expected": int(len(legacy)),
        "completed": completed,
        "failed": failed,
        "workers": int(workers),
        "method": "PIL.Image.open(...).size header metadata only",
        "selection_sha256": selection_sha256,
        "output_csv": str(output_csv),
        "checkpoint_json": str(checkpoint_json),
        "complete": completed == len(legacy),
    }


def native_crop_pixel_audit(
    binding: ResolutionIndependentRGBBinding,
    legacy_header_csv: Path,
) -> dict[str, Any]:
    """Describe crop pixels before runtime resize without decoding CVAT video frames."""

    legacy_headers = pd.read_csv(legacy_header_csv, low_memory=False)
    _require_columns(
        legacy_headers,
        {"image_context_id", "width", "height", "status"},
        "legacy header scan",
    )
    legacy = binding.frames.loc[
        binding.frames["source_type"].astype(str).eq("legacy_recovered")
    ].copy()
    cvat = binding.frames.loc[
        binding.frames["source_type"].astype(str).eq("cvat_tracking_xml")
    ].copy()
    failed_headers = legacy_headers.loc[legacy_headers["status"].ne("ok")]
    if len(failed_headers):
        raise ValueError(f"legacy header scan failures={len(failed_headers)}")
    if legacy_headers["image_context_id"].duplicated().any():
        raise ValueError("legacy header scan has duplicate observation identifiers")
    legacy = legacy.merge(
        legacy_headers[["image_context_id", "width", "height"]],
        on="image_context_id",
        how="left",
        validate="one_to_one",
    )
    if legacy[["width", "height"]].isna().any().any():
        raise ValueError("legacy header scan is incomplete for inner observations")
    cvat_dimensions = _cvat_native_dimensions(cvat)
    legacy_dimensions = legacy[
        ["image_context_id", "source_type", "_resolution_binding_label", "width", "height"]
    ].copy()
    dimensions = pd.concat(
        [cvat_dimensions, legacy_dimensions],
        ignore_index=True,
    )
    dimensions["width"] = dimensions["width"].astype(int)
    dimensions["height"] = dimensions["height"].astype(int)
    dimensions["min_dimension"] = dimensions[["width", "height"]].min(axis=1)
    dimensions["max_dimension"] = dimensions[["width", "height"]].max(axis=1)
    dimensions["aspect_ratio"] = dimensions["width"] / dimensions["height"]
    if len(dimensions) != binding.observation_count:
        raise ValueError("native pixel audit lost observations")
    class_reports = {
        label: _dimension_summary(
            dimensions.loc[
                dimensions["_resolution_binding_label"].astype(str).eq(label)
            ]
        )
        for label in ["fight", "social-nose", "move", "playwithtoy"]
    }
    all_summary = _dimension_summary(dimensions)
    return {
        "schema_version": "classification_v2_native_crop_pixel_audit_v1",
        "observation_count": int(len(dimensions)),
        "source_counts": dimensions["source_type"].value_counts().to_dict(),
        "all": all_summary,
        "class_specific": class_reports,
        "native_information_support_160": _information_support_band(
            all_summary["min_dimension"],
            160,
        ),
        "native_information_support_224": _information_support_band(
            all_summary["min_dimension"],
            224,
        ),
        "support_band_rule": (
            "HIGH when P25(min_dimension) is at least the requested size; "
            "LOW when P75(min_dimension) is below it; otherwise MIXED. "
            "This is a descriptive source-detail category, not a performance threshold."
        ),
    }


def storage_projection(
    observation_count: int,
    *,
    resolutions: Iterable[int] = SUPPORTED_RUNTIME_INPUT_RESOLUTIONS,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Calculate uncompressed HWC RGB payloads without generating a cache."""

    if observation_count < 0:
        raise ValueError("observation_count must be non-negative")
    result: dict[str, dict[str, dict[str, float | int]]] = {}
    for dtype, bytes_per_channel in (("uint8", 1), ("float32", 4)):
        result[dtype] = {}
        for resolution in resolutions:
            if resolution <= 0:
                raise ValueError("resolution must be positive")
            byte_count = observation_count * resolution * resolution * 3 * bytes_per_channel
            result[dtype][str(resolution)] = {
                "bytes": int(byte_count),
                "decimal_gb": round(byte_count / 1_000_000_000, 6),
                "gib": round(byte_count / (1024**3), 6),
            }
    return result


def _read_legacy_jpeg_header(item: tuple[str, Path]) -> dict[str, Any]:
    context_id, path = item
    try:
        with Image.open(path) as image:
            width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("nonpositive_image_size")
        return {
            "image_context_id": context_id,
            "width": int(width),
            "height": int(height),
            "status": "ok",
            "error": "",
        }
    except Exception as exc:
        return {
            "image_context_id": context_id,
            "width": np.nan,
            "height": np.nan,
            "status": "error",
            "error": f"{type(exc).__name__}:{exc}",
        }


def _load_header_checkpoint(
    *,
    output_csv: Path,
    checkpoint_json: Path,
    selection_sha256: str,
) -> pd.DataFrame:
    columns = ["image_context_id", "width", "height", "status", "error"]
    if not output_csv.exists() and not checkpoint_json.exists():
        return pd.DataFrame(columns=columns)
    if not output_csv.exists() or not checkpoint_json.exists():
        raise ValueError("legacy header checkpoint is incomplete")
    checkpoint = json.loads(checkpoint_json.read_text(encoding="utf-8"))
    if checkpoint.get("selection_sha256") != selection_sha256:
        raise ValueError("legacy header checkpoint selection mismatch")
    existing = pd.read_csv(output_csv, low_memory=False)
    _require_columns(existing, set(columns), "legacy header checkpoint")
    if existing["image_context_id"].duplicated().any():
        raise ValueError("legacy header checkpoint has duplicate observation identifiers")
    if int(checkpoint.get("processed_count", -1)) != len(existing):
        raise ValueError("legacy header checkpoint processed count mismatch")
    return existing[columns].copy()


def _write_header_checkpoint(
    records: list[dict[str, Any]],
    *,
    output_csv: Path,
    checkpoint_json: Path,
    selection_sha256: str,
    expected_count: int,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_json.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(records)
    result = result.sort_values("image_context_id", kind="mergesort")
    tmp_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")
    result.to_csv(tmp_csv, index=False)
    tmp_csv.replace(output_csv)
    payload = {
        "schema_version": "classification_v2_legacy_jpeg_header_checkpoint_v1",
        "selection_sha256": selection_sha256,
        "expected_count": int(expected_count),
        "processed_count": int(len(result)),
        "failed_count": int(result["status"].ne("ok").sum()),
        "complete": bool(len(result) == expected_count),
    }
    tmp_json = checkpoint_json.with_suffix(checkpoint_json.suffix + ".tmp")
    tmp_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_json.replace(checkpoint_json)


def _cvat_native_dimensions(cvat: pd.DataFrame) -> pd.DataFrame:
    values: list[dict[str, Any]] = []
    columns = [
        "image_context_id",
        "source_type",
        "_resolution_binding_label",
        "image_width",
        "image_height",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    for (
        context_id,
        source_type,
        label,
        raw_width,
        raw_height,
        raw_x1,
        raw_y1,
        raw_x2,
        raw_y2,
    ) in cvat.loc[:, columns].itertuples(index=False, name=None):
        image_width = _required_int(raw_width, "image_width")
        image_height = _required_int(raw_height, "image_height")
        x1 = max(0, min(image_width, _required_int(raw_x1, "x1")))
        y1 = max(0, min(image_height, _required_int(raw_y1, "y1")))
        x2 = max(0, min(image_width, _required_int(raw_x2, "x2")))
        y2 = max(0, min(image_height, _required_int(raw_y2, "y2")))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                "invalid CVAT crop after current loader integer clamp: "
                f"context_id={context_id}"
            )
        values.append(
            {
                "image_context_id": str(context_id),
                "source_type": str(source_type),
                "_resolution_binding_label": str(label),
                "width": x2 - x1,
                "height": y2 - y1,
            }
        )
    return pd.DataFrame(values)


def _dimension_summary(dimensions: pd.DataFrame) -> dict[str, Any]:
    if dimensions.empty:
        return {
            "count": 0,
            "width": _describe(pd.Series(dtype=float)),
            "height": _describe(pd.Series(dtype=float)),
            "area": _describe(pd.Series(dtype=float)),
            "min_dimension": _describe(pd.Series(dtype=float)),
            "max_dimension": _describe(pd.Series(dtype=float)),
            "min_dimension_support": {
                f"ge{threshold}": 0.0
                for threshold in (64, 96, 128, 160, 224)
            },
            "max_dimension_support": {
                f"ge{threshold}": 0.0
                for threshold in (160, 224)
            },
            "aspect_ratio": _describe(pd.Series(dtype=float)),
        }
    return {
        "count": int(len(dimensions)),
        "width": _describe(dimensions["width"]),
        "height": _describe(dimensions["height"]),
        "area": _describe(dimensions["width"] * dimensions["height"]),
        "min_dimension": _describe(dimensions["min_dimension"]),
        "max_dimension": _describe(dimensions["max_dimension"]),
        "min_dimension_support": {
            f"ge{threshold}": round(
                float((dimensions["min_dimension"] >= threshold).mean()),
                6,
            )
            for threshold in (64, 96, 128, 160, 224)
        },
        "max_dimension_support": {
            f"ge{threshold}": round(
                float((dimensions["max_dimension"] >= threshold).mean()),
                6,
            )
            for threshold in (160, 224)
        },
        "aspect_ratio": _describe(dimensions["aspect_ratio"]),
    }


def _describe(values: pd.Series) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return {key: 0 for key in _QUANTILE_KEYS}
    quantiles = np.quantile(array, _QUANTILES, method="linear")
    return {
        key: _number(value)
        for key, value in zip(_QUANTILE_KEYS, quantiles, strict=True)
    }


def _information_support_band(
    min_dimension_summary: dict[str, float | int],
    target_resolution: int,
) -> str:
    if float(min_dimension_summary["p25"]) >= target_resolution:
        return "HIGH"
    if float(min_dimension_summary["p75"]) < target_resolution:
        return "LOW"
    return "MIXED"


def _stable_table_sha256(
    frame: pd.DataFrame,
    columns: list[str],
    sort_column: str,
) -> str:
    payload = _stable_records(frame, columns, sort_column)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _stable_records(
    frame: pd.DataFrame,
    columns: list[str],
    sort_column: str,
) -> list[dict[str, Any]]:
    sorted_frame = frame.loc[:, columns].sort_values(sort_column, kind="mergesort")
    return [
        {
            key: _json_scalar(value)
            for key, value in row.items()
        }
        for row in sorted_frame.to_dict("records")
    ]


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _required_int(value: object, field: str) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        raise ValueError(f"missing numeric {field}")
    return int(float(numeric))


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y", "t"}
    )


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _number(value: float) -> float | int:
    rounded = round(float(value), 6)
    return int(rounded) if rounded.is_integer() else rounded


_QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
_QUANTILE_KEYS = ("min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max")


__all__ = [
    "INNER_ROLES",
    "RUNTIME_RGB_TRANSFORM_VERSION",
    "SUPPORTED_RUNTIME_INPUT_RESOLUTIONS",
    "ResolutionIndependentRGBBinding",
    "build_inner_resolution_binding",
    "build_inner_resolution_binding_from_dataframes",
    "native_crop_pixel_audit",
    "resolve_audited_media_path",
    "scan_legacy_jpeg_headers",
    "scientific_rgb_identity_sha256",
    "split_image_context_sequence",
    "storage_projection",
    "validate_runtime_resolution",
]
