"""Build the governed DATA+ train-only supplemental T6 sidecar.

This builder never edits the frozen FULL-T6 population or VG manifests.  It
unitizes authoritative XML actor tracks, reconstructs optional partner context
from each short clip on CPU, and exports the current canonical 46D spatial
tensor with explicit missingness masks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import pandas as pd

from legacy_burst_recovery.check_duplicate_videos import (
    normalize_source_video_key,
)
from pig_behavior.classification_v2.contracts.identifiers import (
    ensure_frame_object_identifiers,
    ensure_object_track_keys,
)
from pig_behavior.classification_v2.features.geometry import (
    build_geometry_features,
)
from pig_behavior.classification_v2.features.partner_tokens import (
    PARTNER_TOKEN_COLUMNS,
    FrameObservation,
    extract_frame_partner_tokens,
)
from pig_behavior.classification_v2.features.roi import build_roi_features
from pig_behavior.classification_v2.features.spatiotemporal import (
    build_enhanced_spatiotemporal_features,
)
from pig_behavior.classification_v2.sources.cvat_tracking_xml import (
    load_cvat_tracking_xml,
)
from pig_behavior.classification_v2.spatial_sequence_export import (
    export_spatial_sequences,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PLUS_ROOT = PROJECT_ROOT / "data" / "data_plus"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "classification_v2"
    / "data_plus_supplemental_t6_v2_20260825"
)
DEFAULT_FOLD_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "classification_v2"
    / "video_group_stratified_5fold_authority_20260821"
    / "full_t6_video_group_stratified_5fold_manifest.csv"
)
DEFAULT_CANONICAL_ROW_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "classification_v2"
    / "full_t6_canonical_46d_20260816"
    / "full_t6_row_manifest.csv"
)
DEFAULT_CANONICAL_NPZ = (
    PROJECT_ROOT
    / "outputs"
    / "classification_v2"
    / "full_t6_canonical_46d_20260816"
    / "full_t6_canonical_46d.npz"
)
DEFAULT_ROI_COCO = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "roi"
    / "ROI_annotations.toy_adjusted.coco.json"
)
DEFAULT_DETECTOR = (
    PROJECT_ROOT / "models" / "detector" / "pig_detector_yolov8.pt"
)
DEFAULT_TRACKING_MASK = (
    PROJECT_ROOT / "data" / "annotations" / "scene" / "mask.png"
)

SOURCE_TYPE = "data_plus_supplemental"
T6_LENGTH = 6
FOLDS = tuple(f"VG{index}" for index in range(1, 6))
VALID_BEHAVIORS = {
    "drink",
    "eat",
    "fight",
    "social-nose",
    "explore",
    "lying",
    "stand",
    "move",
    "sitting",
    "playwithtoy",
}
PARTNER_K = 2
PARTNER_TRACKING_MODE = "hybrid_bytetrack"
PARTNER_TRACKING_PROFILE = "hybrid_bytetrack_best"
PARTNER_TRACKING_LINEAGE_FILES = (
    PROJECT_ROOT / "src" / "pig_behavior" / "tracking" / "runner.py",
    PROJECT_ROOT / "src" / "pig_behavior" / "tracking" / "config.py",
    PROJECT_ROOT / "src" / "pig_behavior" / "tracking" / "association.py",
    PROJECT_ROOT
    / "src"
    / "pig_behavior"
    / "tracking"
    / "profiles"
    / "hybrid_bytetrack.py",
)
FIXED_EXCLUSION_KEYS: frozenset[str] = frozenset()
EXPECTED_CLASS_COUNTS = {
    "explore": 47,
    "fight": 6,
    "lying": 2,
    "move": 65,
    "playwithtoy": 337,
    "sitting": 102,
    "social-nose": 366,
    "stand": 327,
}
XML_IDENTITY_EXACT_FIELDS = (
    "frame_index",
    "pig_id",
    "behavior",
    "hidden",
    "track_label",
    "x1_raw",
    "y1_raw",
    "x2_raw",
    "y2_raw",
    "bbox_valid",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-plus-root", type=Path, default=DEFAULT_DATA_PLUS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument(
        "--canonical-row-manifest",
        type=Path,
        default=DEFAULT_CANONICAL_ROW_MANIFEST,
    )
    parser.add_argument("--canonical-npz", type=Path, default=DEFAULT_CANONICAL_NPZ)
    parser.add_argument("--roi-coco", type=Path, default=DEFAULT_ROI_COCO)
    parser.add_argument("--detector-weights", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--expected-units", type=int, default=1252)
    parser.add_argument("--reuse-partner-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def resolve_video_key(value: object) -> str:
    """Resolve canonical key while preserving production normalization."""
    resolved = normalize_source_video_key(value)
    if resolved:
        return resolved
    text = str(value).strip()
    if not text:
        return ""
    resolved = normalize_source_video_key(text + ".mp4")
    if resolved:
        return resolved
    match = re.fullmatch(
        r"(?:pigs)?(?P<day>\d{6}[a-z]?)[_-](?P<clip>\d{1,6})"
        r"(?:[_-]30fps)?(?:\.[^.]+)?",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return ""
    day = match.group("day").lower()
    clip = match.group("clip").zfill(6)
    return f"pigs{day}/{clip}"


def _task_name(xml_path: Path) -> str:
    root = ET.parse(xml_path).getroot()
    value = root.findtext("./meta/task/name")
    return (value or xml_path.stem).strip()


def discover_clip_pairs(root: Path) -> list[dict[str, Any]]:
    media = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov"}
    )
    xml_paths = sorted(root.rglob("*.xml"))
    xml_by_key: dict[str, list[Path]] = {}
    for xml_path in xml_paths:
        key = resolve_video_key(_task_name(xml_path))
        if not key:
            key = resolve_video_key(xml_path.stem)
        xml_by_key.setdefault(key, []).append(xml_path)

    pairs: list[dict[str, Any]] = []
    used_xml: set[Path] = set()
    for media_path in media:
        clean_stem = media_path.stem.strip()
        video_key = resolve_video_key(clean_stem + media_path.suffix.lower())
        if not video_key:
            raise ValueError(f"Unresolved DATA+ media video_key: {media_path}")
        candidates = [
            path for path in xml_by_key.get(video_key, []) if path not in used_xml
        ]
        if len(candidates) != 1:
            raise ValueError(
                "DATA+ media/XML pairing is not one-to-one: "
                f"media={media_path}, video_key={video_key}, xml={candidates}"
            )
        xml_path = candidates[0]
        used_xml.add(xml_path)
        pairs.append(
            {
                "folder": media_path.parent.name,
                "clip_name": media_path.name,
                "clip_id": clean_stem,
                "media_path": media_path,
                "xml_path": xml_path,
                "xml_task_name": _task_name(xml_path),
                "video_key": video_key,
            }
        )
    unused_xml = sorted(set(xml_paths).difference(used_xml))
    if unused_xml:
        raise ValueError(f"Unpaired DATA+ XML files: {unused_xml}")
    return pairs


def probe_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Unreadable DATA+ media: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid media metadata: {path}, fps={fps}, frames={frame_count}, "
            f"size={width}x{height}"
        )
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": frame_count / fps,
    }


def normalize_fragmented_xml_identity_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Relink one-frame CVAT fragments only when XML ID authority is unique."""
    normalized = frame.copy()
    normalized["xml_fragment_track_id"] = normalized["track_id"].astype(str)
    normalized["xml_fragment_track_ids"] = normalized["track_id"].astype(str)
    normalized["xml_exact_duplicate_count"] = 0
    normalized["identity_authority"] = "cvat_track_id"
    normalized["identity_relink_applied"] = False

    frames_per_track = normalized.groupby("track_id")["frame_index"].nunique()
    if normalized.empty or not frames_per_track.eq(1).all():
        return normalized
    pig_ids = normalized["pig_id"].fillna("").astype(str).str.strip()
    if pig_ids.eq("").any():
        raise ValueError("Fragmented XML rows have blank ID attributes")

    signature = list(XML_IDENTITY_EXACT_FIELDS)
    grouped = normalized.groupby(signature, dropna=False, sort=False)
    normalized["xml_fragment_track_ids"] = grouped["track_id"].transform(
        lambda values: ",".join(sorted(values.astype(str).unique()))
    )
    normalized["xml_exact_duplicate_count"] = (
        grouped["track_id"].transform("size").astype(int) - 1
    )
    normalized = normalized.drop_duplicates(signature, keep="first").copy()

    conflicts = normalized.duplicated(["frame_index", "pig_id"], keep=False)
    if conflicts.any():
        sample = normalized.loc[
            conflicts,
            ["video_key", "frame_index", "pig_id", "track_id"],
        ].head(10)
        raise ValueError(
            "Fragmented XML requires Mini-CVAT identity review: "
            + sample.to_dict(orient="records").__repr__()
        )

    normalized["track_id"] = "xml_id:" + pig_ids.loc[normalized.index]
    normalized["identity_authority"] = "xml_ID_attribute_relinked"
    normalized["identity_relink_applied"] = True
    return normalized


def load_actor_rows(pair: dict[str, Any], media: dict[str, Any]) -> pd.DataFrame:
    video_key = str(pair["video_key"])
    dataset_id = "data_plus_" + video_key.replace("/", "_")
    frame = load_cvat_tracking_xml(
        pair["xml_path"],
        video_key=video_key,
        dataset_id=dataset_id,
        fps=float(media["fps"]),
    ).copy()
    frame = normalize_fragmented_xml_identity_rows(frame)
    for column, media_key in (
        ("image_width", "width"),
        ("image_height", "height"),
    ):
        values = pd.to_numeric(
            frame.get(column, pd.Series(np.nan, index=frame.index)),
            errors="coerce",
        )
        frame[column] = values.where(values.gt(0), int(media[media_key])).astype(int)
    frame["source_type"] = SOURCE_TYPE
    frame["dataset_id"] = dataset_id
    frame["video_key"] = video_key
    frame["source_video_key"] = video_key
    frame["clip_id"] = str(pair["clip_id"])
    frame["source_video_path"] = str(
        Path(pair["media_path"]).relative_to(PROJECT_ROOT).as_posix()
    )
    frame["annotation_xml_path"] = str(
        Path(pair["xml_path"]).relative_to(PROJECT_ROOT).as_posix()
    )
    frame["clip_name"] = str(pair["clip_name"])
    frame["folder"] = str(pair["folder"])
    frame["observed_mask"] = True
    frame["object_track_key"] = ""
    frame["identifier_schema_version"] = ""
    frame["scene_frame_uid"] = (
        str(pair["clip_id"])
        + "::f"
        + pd.to_numeric(frame["frame_index"], errors="raise")
        .astype(int)
        .astype(str)
        .str.zfill(6)
    )
    frame["frame_uid"] = ""
    frame = ensure_object_track_keys(frame, source_name="DATA+ actor XML")
    frame = ensure_frame_object_identifiers(frame, source_name="DATA+ actor XML")
    return frame


def unitize_actor_rows(
    actor_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    manifest_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    dropped = Counter()
    eligible = actor_rows[
        actor_rows["behavior"].isin(VALID_BEHAVIORS)
        & actor_rows["bbox_valid"].fillna(False).astype(bool)
    ].copy()
    group_columns = ["object_track_key", "behavior"]
    for (_, behavior), group in eligible.groupby(group_columns, sort=True):
        group = group.sort_values("frame_index", kind="mergesort")
        duplicate = group.duplicated("frame_index", keep=False)
        if duplicate.any():
            raise ValueError(
                "Duplicate XML target track/frame rows: "
                f"count={int(duplicate.sum())}"
            )
        frames = pd.to_numeric(group["frame_index"], errors="raise").astype(int)
        run_ids = frames.diff().ne(1).cumsum()
        for _, run in group.groupby(run_ids, sort=False):
            run = run.sort_values("frame_index", kind="mergesort")
            complete = len(run) // T6_LENGTH
            dropped[str(behavior)] += len(run) % T6_LENGTH
            run_start = int(run["frame_index"].iloc[0])
            run_end = int(run["frame_index"].iloc[-1])
            for ordinal in range(complete):
                chunk = run.iloc[
                    ordinal * T6_LENGTH : (ordinal + 1) * T6_LENGTH
                ].copy()
                indices = [int(value) for value in chunk["frame_index"]]
                if np.diff(indices).tolist() != [1] * (T6_LENGTH - 1):
                    raise RuntimeError("DATA+ unitization emitted nonconsecutive frames")
                first = chunk.iloc[0]
                identity = _json(
                    {
                        "video_key": first["video_key"],
                        "clip_id": first["clip_id"],
                        "object_track_key": first["object_track_key"],
                        "behavior": behavior,
                        "frames": indices,
                    }
                )
                unit_id = "data_plus_t6_" + hashlib.sha256(
                    identity.encode("utf-8")
                ).hexdigest()[:24]
                timestamps = [float(value) for value in chunk["timestamp_sec"]]
                pair_seconds = [
                    timestamps[index] - timestamps[index - 1]
                    for index in range(1, T6_LENGTH)
                ]
                manifest_rows.append(
                    {
                        "supplemental_unit_id": unit_id,
                        "window_id": unit_id,
                        "class_label": str(behavior),
                        "behavior": str(behavior),
                        "source_type": SOURCE_TYPE,
                        "dataset_id": str(first["dataset_id"]),
                        "video_key": str(first["video_key"]),
                        "source_clip_id": str(first["clip_id"]),
                        "clip_name": str(first["clip_name"]),
                        "folder": str(first["folder"]),
                        "source_video_path": str(first["source_video_path"]),
                        "annotation_xml_path": str(first["annotation_xml_path"]),
                        "object_track_key": str(first["object_track_key"]),
                        "target_object_track_key": str(first["object_track_key"]),
                        "target_track_id": str(first["track_id"]),
                        "target_pig_id": str(first["pig_id"]),
                        "source_frame_indices": _json(indices),
                        "physical_frame_ids_json": _json(indices),
                        "observed_mask_json": _json([True] * T6_LENGTH),
                        "window_start_frame": indices[0],
                        "window_end_frame": indices[-1],
                        "window_length_frames": T6_LENGTH,
                        "run_start_frame": run_start,
                        "run_end_frame": run_end,
                        "run_chunk_ordinal": ordinal,
                        "annotation_authority": (
                            "cvat_tracking_xml_box_behavior_attribute"
                        ),
                        "provenance": (
                            "DATA+ short clip; human-confirmed non-overlap; "
                            "clip-local frame lineage"
                        ),
                        "view_type": "T6_contiguous",
                        "sampling_pattern": "contiguous",
                        "unit_stride_frames": T6_LENGTH,
                        "overlap_frames": 0,
                        "padding_policy": "none",
                        "selected_frame_offsets": _json(list(range(T6_LENGTH))),
                        "selected_frame_indices": _json(indices),
                        "selected_timestamps_seconds": _json(timestamps),
                        "pair_delta_frames": _json([1] * (T6_LENGTH - 1)),
                        "pair_delta_seconds": _json(pair_seconds),
                        "feature_computation_grain": "FINAL_VIEW_FEATURES",
                        "pair_scope_key": unit_id,
                        "pair_recomputed_for_view": True,
                        "aggregate_recomputed_for_view": True,
                    }
                )
                chunk["supplemental_unit_id"] = unit_id
                chunk["temporal_unit_key"] = unit_id
                chunk["annotation_role"] = "target_actor"
                selected_rows.extend(chunk.to_dict("records"))
    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["video_key", "source_clip_id", "object_track_key", "window_start_frame"],
        kind="mergesort",
    ).reset_index(drop=True)
    selected = pd.DataFrame(selected_rows).sort_values(
        ["supplemental_unit_id", "frame_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    return manifest, selected, dict(sorted(dropped.items()))


def build_fold_eligibility(
    units: pd.DataFrame,
    fold_manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    fold_manifest = pd.read_csv(fold_manifest_path, low_memory=False)
    fold_manifest["normalized_video_key"] = fold_manifest["video_key"].map(
        resolve_video_key
    )
    role_map: dict[str, dict[str, str]] = {}
    for video_key, group in fold_manifest.groupby("normalized_video_key"):
        if not video_key:
            continue
        roles: dict[str, str] = {}
        for fold in FOLDS:
            values = sorted(set(group[f"split_{fold}"].dropna().astype(str)))
            if len(values) != 1 or values[0] not in {"train", "test"}:
                raise ValueError(
                    f"Ambiguous canonical role: key={video_key}, fold={fold}, "
                    f"roles={values}"
                )
            roles[fold] = values[0]
        role_map[str(video_key)] = roles

    rows: list[dict[str, Any]] = []
    for unit in units.itertuples(index=False):
        key = str(unit.video_key)
        existing = key in role_map
        for fold in FOLDS:
            canonical_role = role_map[key][fold] if existing else "new_video"
            eligible = (not existing) or canonical_role == "train"
            rows.append(
                {
                    "supplemental_unit_id": unit.supplemental_unit_id,
                    "video_key": key,
                    "fold": fold,
                    "existing_in_canonical": existing,
                    "canonical_role": canonical_role,
                    "supplemental_role": "train" if eligible else "excluded",
                    "eligible_for_training": eligible,
                    "eligible_for_inner_validation": False,
                    "eligible_for_outer_test": False,
                    "append_target": "inner_train_only",
                    "group_guard_key": key,
                }
            )
    eligibility = pd.DataFrame(rows)
    return eligibility, role_map


def canonical_duplicate_audit(
    units: pd.DataFrame,
    canonical_manifest_path: Path,
) -> tuple[set[str], dict[str, Any]]:
    canonical = pd.read_csv(
        canonical_manifest_path,
        usecols=["video_key", "behavior", "physical_frame_ids_json"],
        low_memory=False,
    )
    signatures: set[tuple[str, str, tuple[int, ...]]] = set()
    canonical_keys: set[str] = set()
    for row in canonical.itertuples(index=False):
        key = resolve_video_key(row.video_key)
        if not key:
            continue
        canonical_keys.add(key)
        try:
            frames = tuple(int(value) for value in json.loads(row.physical_frame_ids_json))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        signatures.add((key, str(row.behavior), frames))
    collisions: set[str] = set()
    for row in units.itertuples(index=False):
        frames = tuple(int(value) for value in json.loads(row.source_frame_indices))
        if (str(row.video_key), str(row.behavior), frames) in signatures:
            collisions.add(str(row.supplemental_unit_id))
    existing_units = int(units["video_key"].isin(canonical_keys).sum())
    return collisions, {
        "canonical_video_keys": len(canonical_keys),
        "data_plus_units_with_existing_video_key": existing_units,
        "strict_key_label_frame_signature_collisions": len(collisions),
        "strict_collision_unit_ids": sorted(collisions),
        "interpretation": (
            "Potential-only comparison because DATA+ frame indices are local to "
            "short clips. Human non-overlap authority remains controlling."
        ),
    }


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def partner_tracking_lineage_hash() -> str:
    payload = {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(path)
        for path in PARTNER_TRACKING_LINEAGE_FILES
    }
    return _canonical_hash(payload)


def partner_tracking_profile_hash() -> str:
    from pig_behavior.tracking.profiles.hybrid_bytetrack import EVAL_CONFIGS

    return _canonical_hash(EVAL_CONFIGS[PARTNER_TRACKING_PROFILE])


def build_partner_tracking_config(
    media_path: Path,
    detector_weights: Path,
    output_dir: Path,
) -> tuple[Any, str]:
    from pig_behavior.tracking.config import TrackingConfig, validate_config
    from pig_behavior.tracking.profiles.hybrid_bytetrack import EVAL_CONFIGS

    overrides = dict(EVAL_CONFIGS[PARTNER_TRACKING_PROFILE])
    profile_hash = _canonical_hash(overrides)
    overrides.pop("mode", None)
    cfg = TrackingConfig(
        mode=PARTNER_TRACKING_MODE,
        video_path=media_path,
        weights_path=detector_weights,
        mask_path=DEFAULT_TRACKING_MASK,
        output_dir=output_dir,
        device="cpu",
        half=False,
        write_output_video=False,
        show=False,
        start_frame=0,
        max_frames=None,
        overrides=set(overrides),
        **overrides,
    )
    cfg.association_debug = False
    validate_config(cfg)
    return cfg, profile_hash


def _clean_coco_to_detections(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in payload.get("images", [])}
    rows: list[dict[str, Any]] = []
    for annotation in payload.get("annotations", []):
        image = images[int(annotation["image_id"])]
        x, y, width, height = [float(value) for value in annotation["bbox"]]
        attributes = annotation.get("attributes", {})
        track_source = str(attributes.get("TrackSource", "unknown"))
        hidden = str(attributes.get("Hidden", "No"))
        needs_review = str(attributes.get("NeedsReview", "No"))
        if track_source != "detected" or hidden != "No" or needs_review != "No":
            continue
        rows.append(
            {
                "frame_index": int(image["frame"]),
                "tracker_id": int(annotation["track_id"]),
                "x1": x,
                "y1": y,
                "x2": x + width,
                "y2": y + height,
                "confidence": float(annotation.get("score", 1.0)),
                "image_width": int(image["width"]),
                "image_height": int(image["height"]),
                "track_source": track_source,
                "hidden": hidden,
                "needs_review": needs_review,
            }
        )
    return pd.DataFrame(rows)


def run_cpu_hybrid_bytetrack(
    media_path: Path,
    detector_weights: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, str]:
    from pig_behavior.tracking.runner import run_tracking

    cfg, profile_hash = build_partner_tracking_config(
        media_path,
        detector_weights,
        output_dir,
    )
    print(f"CPU hybrid_bytetrack start: {media_path.name}", flush=True)
    summary = run_tracking(cfg)
    detections = _clean_coco_to_detections(summary.clean_coco_annotations_json)
    print(
        "CPU hybrid_bytetrack done: "
        f"{media_path.name} frames={summary.frames_written} "
        f"clean_detected_rows={len(detections)}",
        flush=True,
    )
    return detections, profile_hash


def load_or_build_partner_detections(
    clip_pairs: list[dict[str, Any]],
    output_root: Path,
    detector_weights: Path,
    reuse_cache: bool,
) -> pd.DataFrame:
    cache_root = output_root / "partner_cache_hybrid_bytetrack_best"
    cache_root.mkdir(parents=True, exist_ok=True)
    detector_sha = sha256_file(detector_weights)
    lineage_sha = partner_tracking_lineage_hash()
    expected_profile_sha = partner_tracking_profile_hash()
    outputs: list[pd.DataFrame] = []
    for pair in clip_pairs:
        clip_id = str(pair["clip_id"])
        token = hashlib.sha256(clip_id.encode("utf-8")).hexdigest()[:16]
        csv_path = cache_root / f"{token}.csv"
        json_path = cache_root / f"{token}.json"
        tracking_output_dir = cache_root / token / "production_tracking"
        media_sha = sha256_file(pair["media_path"])
        cached_valid = False
        if reuse_cache and csv_path.exists() and json_path.exists():
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
            cached_valid = (
                metadata.get("media_sha256") == media_sha
                and metadata.get("detector_sha256") == detector_sha
                and metadata.get("tracking_lineage_sha256") == lineage_sha
                and metadata.get("tracker_mode") == PARTNER_TRACKING_MODE
                and metadata.get("tracker_profile") == PARTNER_TRACKING_PROFILE
                and metadata.get("tracker_profile_sha256")
                == expected_profile_sha
            )
        if cached_valid:
            detections = pd.read_csv(csv_path)
            clean_mask = (
                detections["track_source"].eq("detected")
                & detections["hidden"].eq("No")
                & detections["needs_review"].eq("No")
            )
            if not clean_mask.all():
                detections = detections[clean_mask].copy()
                _atomic_csv(detections, csv_path)
            print(
                f"CPU hybrid_bytetrack cache reused: {pair['clip_name']}",
                flush=True,
            )
        else:
            detections, profile_hash = run_cpu_hybrid_bytetrack(
                pair["media_path"],
                detector_weights,
                tracking_output_dir,
            )
            if profile_hash != expected_profile_sha:
                raise RuntimeError("Partner tracker profile hash changed during run")
            _atomic_csv(detections, csv_path)
            _atomic_json(
                {
                    "clip_id": clip_id,
                    "clip_name": pair["clip_name"],
                    "video_key": pair["video_key"],
                    "media_sha256": media_sha,
                    "detector_sha256": detector_sha,
                    "device": "cpu",
                    "tracker_mode": PARTNER_TRACKING_MODE,
                    "tracker_profile": PARTNER_TRACKING_PROFILE,
                    "tracker_profile_sha256": profile_hash,
                    "tracking_lineage_sha256": lineage_sha,
                    "partner_evidence_policy": (
                        "production clean_training_shapes: detected, "
                        "non-hidden, score>=review_conf"
                    ),
                    "rows": len(detections),
                },
                json_path,
            )
        detections["clip_id"] = clip_id
        detections["clip_name"] = str(pair["clip_name"])
        detections["video_key"] = str(pair["video_key"])
        outputs.append(detections)
    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True)


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    if len(others) == 0:
        return np.zeros(0, dtype=float)
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, others[:, 2] - others[:, 0]) * np.maximum(
        0.0,
        others[:, 3] - others[:, 1],
    )
    return inter / np.maximum(area_a + area_b - inter, 1e-12)


def _context_dataset_id(unit_id: str) -> str:
    return "data_plus_context_" + hashlib.sha256(
        unit_id.encode("utf-8")
    ).hexdigest()[:16]


def _partner_object_key(
    original_dataset: str,
    video_key: str,
    tracker_id: str,
) -> str:
    return (
        f"source={SOURCE_TYPE}|dataset={quote(original_dataset, safe='-_.~')}"
        f"|video={quote(video_key, safe='-_.~')}"
        f"|track_id={quote(tracker_id, safe='-_.~')}"
    )


def _bbox_is_valid(row: pd.Series | dict[str, Any]) -> bool:
    values = np.asarray(
        [row["x1"], row["y1"], row["x2"], row["y2"]],
        dtype=float,
    )
    return bool(
        np.isfinite(values).all()
        and values[2] > values[0]
        and values[3] > values[1]
    )


def _as_frame_observation(
    row: pd.Series | dict[str, Any],
    *,
    dataset_id: str,
    video_key: str,
    scene_frame_uid: str,
    object_track_key: str,
) -> FrameObservation:
    width = float(row["image_width"])
    height = float(row["image_height"])
    if width <= 0 or height <= 0 or not _bbox_is_valid(row):
        raise ValueError("Invalid bbox or media dimensions in partner extraction")
    x1, y1, x2, y2 = (
        float(row["x1"]),
        float(row["y1"]),
        float(row["x2"]),
        float(row["y2"]),
    )
    return FrameObservation(
        source_type=SOURCE_TYPE,
        dataset_id=dataset_id,
        video_key=video_key,
        scene_frame_uid=scene_frame_uid,
        frame_index=int(row["frame_index"]),
        object_track_key=object_track_key,
        cx_n=((x1 + x2) / 2.0) / width,
        cy_n=((y1 + y2) / 2.0) / height,
        bw_n=(x2 - x1) / width,
        bh_n=(y2 - y1) / height,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        bbox_valid=True,
    )


def _exclude_current_target_detection(
    candidates: pd.DataFrame,
    target_row: pd.Series,
) -> tuple[pd.DataFrame, bool]:
    """Remove at most one tracker row corresponding to the current target."""
    if candidates.empty:
        return candidates.copy(), False
    boxes = candidates[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
    target_box = target_row[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
    overlaps = _iou(target_box, boxes)
    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
    center_inside = (
        (centers_x >= target_box[0])
        & (centers_x <= target_box[2])
        & (centers_y >= target_box[1])
        & (centers_y <= target_box[3])
    )
    possible = np.flatnonzero((overlaps >= 0.10) | center_inside)
    if not len(possible):
        return candidates.copy(), False
    target_cx = (target_box[0] + target_box[2]) / 2.0
    target_cy = (target_box[1] + target_box[3]) / 2.0
    ranked = sorted(
        possible.tolist(),
        key=lambda index: (
            -float(overlaps[index]),
            float(
                np.hypot(
                    centers_x[index] - target_cx,
                    centers_y[index] - target_cy,
                )
            ),
            str(candidates.iloc[index]["tracker_id"]),
        ),
    )
    return candidates.drop(candidates.index[ranked[0]]).copy(), True


def build_partner_context_v2(
    units: pd.DataFrame,
    selected_actors: pd.DataFrame,
    detections: pd.DataFrame,
    processed_clip_ids: set[str],
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    """Build observation-only K=2 partner evidence for every DATA+ unit."""
    unit_columns = {
        "supplemental_unit_id",
        "target_object_track_key",
        "source_clip_id",
        "video_key",
        "source_frame_indices",
    }
    actor_columns = {
        "supplemental_unit_id",
        "clip_id",
        "dataset_id",
        "video_key",
        "object_track_key",
        "frame_index",
        "x1",
        "y1",
        "x2",
        "y2",
        "image_width",
        "image_height",
    }
    detection_columns = {
        "clip_id",
        "frame_index",
        "tracker_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "confidence",
        "image_width",
        "image_height",
        "track_source",
        "hidden",
        "needs_review",
    }
    missing_units = unit_columns.difference(units.columns)
    missing_actors = actor_columns.difference(selected_actors.columns)
    missing_detections = detection_columns.difference(detections.columns)
    if missing_units or missing_actors or (not detections.empty and missing_detections):
        raise ValueError(
            "Partner v2 input schema is incomplete: "
            f"units={sorted(missing_units)}, actors={sorted(missing_actors)}, "
            f"detections={sorted(missing_detections)}"
        )
    if units["supplemental_unit_id"].duplicated().any():
        raise ValueError("Partner v2 requires unique supplemental_unit_id rows")

    clean = detections.copy()
    if not clean.empty:
        numeric_columns = (
            "frame_index",
            "x1",
            "y1",
            "x2",
            "y2",
            "confidence",
            "image_width",
            "image_height",
        )
        for column in numeric_columns:
            clean[column] = pd.to_numeric(clean[column], errors="coerce")
        finite = np.isfinite(clean[list(numeric_columns)].to_numpy(dtype=float)).all(
            axis=1
        )
        observation_only = (
            clean["track_source"].astype(str).eq("detected")
            & clean["hidden"].astype(str).eq("No")
            & clean["needs_review"].astype(str).eq("No")
            & clean["tracker_id"].notna()
            & clean["image_width"].gt(0)
            & clean["image_height"].gt(0)
            & clean["x2"].gt(clean["x1"])
            & clean["y2"].gt(clean["y1"])
            & finite
        )
        clean = clean.loc[observation_only].copy()
    if clean.empty:
        detections_by_frame: dict[tuple[str, int], pd.DataFrame] = {}
    else:
        detections_by_frame = {
            (str(clip_id), int(frame_index)): group.copy()
            for (clip_id, frame_index), group in clean.groupby(
                ["clip_id", "frame_index"],
                sort=False,
            )
        }

    ordered_units = units.reset_index(drop=True)
    unit_count = len(ordered_units)
    source_frames = np.zeros((unit_count, T6_LENGTH), dtype=np.int64)
    partner_keys = np.full(
        (unit_count, T6_LENGTH, PARTNER_K),
        "",
        dtype=object,
    )
    partner_mask = np.zeros(
        (unit_count, T6_LENGTH, PARTNER_K),
        dtype=bool,
    )
    partner_bbox = np.zeros(
        (unit_count, T6_LENGTH, PARTNER_K, 4),
        dtype=np.float32,
    )
    partner_confidence = np.zeros(
        (unit_count, T6_LENGTH, PARTNER_K),
        dtype=np.float32,
    )
    partner_geometry = np.zeros(
        (unit_count, T6_LENGTH, PARTNER_K, len(PARTNER_TOKEN_COLUMNS)),
        dtype=np.float32,
    )
    extraction_complete = np.zeros(
        (unit_count, T6_LENGTH),
        dtype=bool,
    )
    slot_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []

    for unit_index, unit in enumerate(ordered_units.itertuples(index=False)):
        unit_id = str(unit.supplemental_unit_id)
        target_key = str(unit.target_object_track_key)
        clip_id = str(unit.source_clip_id)
        video_key = str(unit.video_key)
        raw_frames = unit.source_frame_indices
        frames = json.loads(raw_frames) if isinstance(raw_frames, str) else raw_frames
        frames = [int(value) for value in frames]
        if len(frames) != T6_LENGTH:
            raise ValueError(f"Partner v2 unit is not T6: {unit_id}")
        source_frames[unit_index] = np.asarray(frames, dtype=np.int64)
        target = selected_actors[
            selected_actors["supplemental_unit_id"].eq(unit_id)
            & selected_actors["object_track_key"].astype(str).eq(target_key)
        ].copy()
        if len(target) != T6_LENGTH or target["frame_index"].duplicated().any():
            raise ValueError(f"Partner v2 target binding is not unique T6: {unit_id}")
        target["frame_index"] = pd.to_numeric(
            target["frame_index"],
            errors="raise",
        ).astype(int)
        target_by_frame = target.set_index("frame_index")
        if set(target_by_frame.index) != set(frames):
            raise ValueError(f"Partner v2 target frames differ from unit: {unit_id}")
        tracker_complete = clip_id in processed_clip_ids

        for step_index, frame_index in enumerate(frames):
            target_row = target_by_frame.loc[frame_index].copy()
            target_row["frame_index"] = frame_index
            if str(target_row["clip_id"]) != clip_id:
                raise ValueError(f"Partner v2 target clip differs from unit: {unit_id}")
            if not _bbox_is_valid(target_row):
                raise ValueError(f"Partner v2 target bbox is invalid: {unit_id}")
            scene_uid = f"partner-v2::{clip_id}::f{frame_index:06d}"
            actor = _as_frame_observation(
                target_row,
                dataset_id=str(target_row["dataset_id"]),
                video_key=video_key,
                scene_frame_uid=scene_uid,
                object_track_key=target_key,
            )
            candidates = detections_by_frame.get(
                (clip_id, frame_index),
                pd.DataFrame(columns=sorted(detection_columns)),
            )
            target_excluded = False
            if tracker_complete:
                candidates, target_excluded = _exclude_current_target_detection(
                    candidates,
                    target_row,
                )
            else:
                candidates = candidates.iloc[0:0].copy()

            observations: list[FrameObservation] = []
            candidate_by_key: dict[str, dict[str, Any]] = {}
            for candidate in candidates.to_dict("records"):
                tracker_id = f"detector_{candidate['tracker_id']}"
                partner_key = _partner_object_key(
                    str(target_row["dataset_id"]),
                    video_key,
                    tracker_id,
                )
                if partner_key in candidate_by_key:
                    raise ValueError(
                        "Duplicate partner tracker ID in one frame: "
                        f"unit={unit_id}, frame={frame_index}, key={partner_key}"
                    )
                observation = _as_frame_observation(
                    candidate,
                    dataset_id=str(target_row["dataset_id"]),
                    video_key=video_key,
                    scene_frame_uid=scene_uid,
                    object_track_key=partner_key,
                )
                observations.append(observation)
                candidate_by_key[partner_key] = candidate

            geometry, mask, keys, _ = extract_frame_partner_tokens(
                actor,
                observations,
                k=PARTNER_K,
            )
            extraction_complete[unit_index, step_index] = tracker_complete
            partner_mask[unit_index, step_index] = mask
            partner_geometry[unit_index, step_index] = geometry
            for slot_index in range(PARTNER_K):
                key = keys[slot_index]
                bbox = np.zeros(4, dtype=np.float32)
                confidence = 0.0
                tracker_id = ""
                if mask[slot_index]:
                    candidate = candidate_by_key[key]
                    bbox = np.asarray(
                        [
                            candidate["x1"],
                            candidate["y1"],
                            candidate["x2"],
                            candidate["y2"],
                        ],
                        dtype=np.float32,
                    )
                    confidence = float(candidate["confidence"])
                    tracker_id = str(candidate["tracker_id"])
                    partner_keys[unit_index, step_index, slot_index] = key
                    partner_bbox[unit_index, step_index, slot_index] = bbox
                    partner_confidence[unit_index, step_index, slot_index] = confidence
                slot_row: dict[str, Any] = {
                    "supplemental_unit_id": unit_id,
                    "target_object_track_key": target_key,
                    "source_clip_id": clip_id,
                    "video_key": video_key,
                    "frame_index": frame_index,
                    "t6_step": step_index,
                    "partner_slot": slot_index,
                    "partner_track_key": key,
                    "partner_tracker_id": tracker_id,
                    "partner_mask": bool(mask[slot_index]),
                    "partner_bbox_xyxy": _json(bbox.tolist()),
                    "partner_confidence": confidence,
                    "partner_geometry_6d": _json(geometry[slot_index].tolist()),
                    "x1": float(bbox[0]),
                    "y1": float(bbox[1]),
                    "x2": float(bbox[2]),
                    "y2": float(bbox[3]),
                    "image_width": int(target_row["image_width"]),
                    "image_height": int(target_row["image_height"]),
                }
                slot_row.update(
                    {
                        name: float(geometry[slot_index, feature_index])
                        for feature_index, name in enumerate(PARTNER_TOKEN_COLUMNS)
                    }
                )
                slot_rows.append(slot_row)
            frame_rows.append(
                {
                    "supplemental_unit_id": unit_id,
                    "target_object_track_key": target_key,
                    "source_clip_id": clip_id,
                    "video_key": video_key,
                    "frame_index": frame_index,
                    "t6_step": step_index,
                    "partner_extraction_complete": tracker_complete,
                    "target_detection_excluded": target_excluded,
                    "candidate_count_after_target_exclusion": len(observations),
                    "partner_available": bool(mask.any()),
                }
            )

    arrays = {
        "supplemental_unit_id": ordered_units["supplemental_unit_id"].to_numpy(
            dtype=str
        ),
        "target_object_track_key": ordered_units[
            "target_object_track_key"
        ].to_numpy(dtype=str),
        "source_frame_indices": source_frames,
        "partner_track_key": np.asarray(partner_keys, dtype=str),
        "partner_mask": partner_mask,
        "partner_bbox_xyxy": partner_bbox,
        "partner_confidence": partner_confidence,
        "partner_geometry_6d": partner_geometry,
        "partner_extraction_complete": extraction_complete,
    }
    return arrays, pd.DataFrame(slot_rows), pd.DataFrame(frame_rows)


def build_target_frame_features(
    units: pd.DataFrame,
    selected_actors: pd.DataFrame,
    partner_slots: pd.DataFrame,
    roi_coco_path: Path,
) -> pd.DataFrame:
    observed_slots = partner_slots[
        partner_slots["partner_mask"].fillna(False).astype(bool)
    ].copy()
    slots_by_frame = {
        (str(unit_id), int(frame_index)): group
        for (unit_id, frame_index), group in observed_slots.groupby(
            ["supplemental_unit_id", "frame_index"],
            sort=False,
        )
    }
    exploded: list[dict[str, Any]] = []
    for unit in units.itertuples(index=False):
        unit_id = str(unit.supplemental_unit_id)
        target = selected_actors[
            selected_actors["supplemental_unit_id"].eq(unit_id)
        ].copy()
        if len(target) != T6_LENGTH:
            raise RuntimeError(f"Target unit frame count is not T6: {unit_id}")
        context_dataset = _context_dataset_id(unit_id)
        for target_row in target.to_dict("records"):
            frame_index = int(target_row["frame_index"])
            clip_id = str(target_row["clip_id"])
            original_dataset = str(target_row["dataset_id"])
            scene_uid = f"{context_dataset}::scene={clip_id}::f{frame_index:06d}"
            target_record = dict(target_row)
            target_record["_original_dataset_id"] = original_dataset
            target_record["_original_scene_frame_uid"] = target_row["scene_frame_uid"]
            target_record["_original_frame_uid"] = target_row["frame_uid"]
            target_record["dataset_id"] = context_dataset
            target_record["scene_frame_uid"] = scene_uid
            target_record["frame_uid"] = (
                scene_uid + "::target=" + quote(unit_id, safe="-_.~")
            )
            target_record["temporal_unit_key"] = unit_id
            target_record["feature_computation_grain"] = "FRAME_LOCAL_PRIMITIVES"
            target_record["annotation_role"] = "target_actor"
            exploded.append(target_record)

            detected = slots_by_frame.get((unit_id, frame_index))
            if detected is None or detected.empty:
                continue
            for partner in detected.to_dict("records"):
                tracker_id = f"detector_{partner['partner_tracker_id']}"
                partner_key = str(partner["partner_track_key"])
                detected_record = dict(target_record)
                detected_record.update(
                    {
                        "object_track_key": partner_key,
                        "track_id": tracker_id,
                        "pig_id": "DETECTED_PARTNER",
                        "behavior": "",
                        "x1_raw": float(partner["x1"]),
                        "y1_raw": float(partner["y1"]),
                        "x2_raw": float(partner["x2"]),
                        "y2_raw": float(partner["y2"]),
                        "x1": float(partner["x1"]),
                        "y1": float(partner["y1"]),
                        "x2": float(partner["x2"]),
                        "y2": float(partner["y2"]),
                        "image_width": int(partner["image_width"]),
                        "image_height": int(partner["image_height"]),
                        "bbox_valid": True,
                        "observed_mask": True,
                        "include_in_training": False,
                        "annotation_role": "observed_partner",
                        "confidence": float(partner["partner_confidence"]),
                        "bbox_source": (
                            "pig_detector_yolov8_"
                            "hybrid_bytetrack_best_cpu_clean"
                        ),
                        "temporal_unit_key": (
                            unit_id + "::observed_partner=" + partner_key
                        ),
                        "frame_uid": (
                            scene_uid
                            + "::observed_partner="
                            + quote(partner_key, safe="-_.~")
                        ),
                    }
                )
                exploded.append(detected_record)

    context = pd.DataFrame(exploded)
    context = build_geometry_features(context)
    context = build_roi_features(context, roi_coco_path=roi_coco_path)
    enhanced = build_enhanced_spatiotemporal_features(context)
    target = enhanced[enhanced["annotation_role"].eq("target_actor")].copy()
    target["dataset_id"] = target.pop("_original_dataset_id")
    target["scene_frame_uid"] = target.pop("_original_scene_frame_uid")
    target["frame_uid"] = target.pop("_original_frame_uid")
    target = target.sort_values(
        ["supplemental_unit_id", "frame_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(target) != len(units) * T6_LENGTH:
        raise RuntimeError("Target feature construction changed the T6 population")
    return target


def attach_readiness(
    units: pd.DataFrame,
    target_features: pd.DataFrame,
    partner_frames: pd.DataFrame,
) -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []
    for unit_id, group in target_features.groupby("supplemental_unit_id"):
        extraction = partner_frames[
            partner_frames["supplemental_unit_id"].eq(unit_id)
        ]
        extraction_complete_frames = int(
            extraction["partner_extraction_complete"]
            .fillna(False)
            .astype(bool)
            .sum()
        )
        partner_available_frames = int(
            extraction["partner_available"].fillna(False).astype(bool).sum()
        )
        roi_frames = int(
            (
                group["roi_feeder_available"].fillna(False).astype(bool)
                & group["roi_drinker_available"].fillna(False).astype(bool)
                & group["roi_toy_available"].fillna(False).astype(bool)
            ).sum()
        )
        bbox_frames = int(group["bbox_valid"].fillna(False).astype(bool).sum())
        partner_ready = extraction_complete_frames == T6_LENGTH
        roi_ready = roi_frames == T6_LENGTH
        if bbox_frames != T6_LENGTH or not partner_ready:
            tier = "REVIEW_REQUIRED"
        elif roi_ready:
            tier = "FULL_MULTIMODAL_READY"
        else:
            tier = "ACTOR_ONLY_READY"
        summaries.append(
            {
                "supplemental_unit_id": unit_id,
                "partner_extraction_complete_frames": (
                    extraction_complete_frames
                ),
                "partner_available_frames": partner_available_frames,
                "partner_context_reconstructable": partner_ready,
                "roi_available_frames": roi_frames,
                "roi_context_reconstructable": roi_ready,
                "actor_bbox_valid_frames": bbox_frames,
                "quality_tier": tier,
            }
        )
    return units.merge(pd.DataFrame(summaries), on="supplemental_unit_id")


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(arrays: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_outputs_available(paths: list[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite DATA+ outputs without --overwrite: "
            f"{existing}"
        )


def _clip_inventory(
    pairs: list[dict[str, Any]],
    probes: dict[str, dict[str, Any]],
    role_map: dict[str, dict[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        key = str(pair["video_key"])
        probe = probes[str(pair["clip_id"])]
        row: dict[str, Any] = {
            "folder": pair["folder"],
            "clip_name": pair["clip_name"],
            "xml_name": Path(pair["xml_path"]).name,
            "resolved_video_key": key,
            "existing_in_canonical": key in role_map,
            "fixed_exclusion": key in FIXED_EXCLUSION_KEYS,
            "media_sha256": sha256_file(pair["media_path"]),
            "xml_sha256": sha256_file(pair["xml_path"]),
            "file_size_bytes": Path(pair["media_path"]).stat().st_size,
            **probe,
        }
        for fold in FOLDS:
            row[f"{fold}_role"] = (
                role_map[key][fold] if key in role_map else "new_train_only"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    final_paths = [
        args.output_root / "clip_inventory.csv",
        args.output_root / "supplemental_t6_manifest.csv",
        args.output_root / "supplemental_fold_eligibility.csv",
        args.output_root / "supplemental_target_frame_features.csv",
        args.output_root / "partner_detections.csv",
        args.output_root / "supplemental_partner_slots.csv",
        args.output_root / "supplemental_partner_extraction_frames.csv",
        args.output_root / "data_plus_partner_context_v2.npz",
        args.output_root / "data_plus_spatial_46d.npz",
        args.output_root / "spatial_sequence_audit.json",
        args.output_root / "data_plus_build_audit.json",
        args.output_root / "artifact_hashes.json",
    ]
    _require_outputs_available(final_paths, args.overwrite)
    required = [
        args.data_plus_root,
        args.fold_manifest,
        args.canonical_row_manifest,
        args.canonical_npz,
        args.roi_coco,
        args.detector_weights,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing DATA+ build authorities: {missing}")

    protected_paths = {
        "fold_manifest": args.fold_manifest,
        "canonical_row_manifest": args.canonical_row_manifest,
        "canonical_npz": args.canonical_npz,
    }
    protected_before = {
        name: sha256_file(path) for name, path in protected_paths.items()
    }
    pairs = discover_clip_pairs(args.data_plus_root)
    if len(pairs) != 20:
        raise ValueError(f"Expected 20 DATA+ media/XML pairs, found {len(pairs)}")
    probes = {
        str(pair["clip_id"]): probe_video(pair["media_path"]) for pair in pairs
    }
    non_30fps = {
        clip: values["fps"]
        for clip, values in probes.items()
        if not np.isclose(float(values["fps"]), 30.0, atol=0.01)
    }
    if non_30fps:
        raise ValueError(f"DATA+ contains non-30fps clips: {non_30fps}")

    valid_pairs = [
        pair for pair in pairs if str(pair["video_key"]) not in FIXED_EXCLUSION_KEYS
    ]
    actor_frames = pd.concat(
        [
            load_actor_rows(pair, probes[str(pair["clip_id"])])
            for pair in valid_pairs
        ],
        ignore_index=True,
    )
    relinked_rows = actor_frames["identity_relink_applied"].astype(bool)
    xml_identity_normalization = {
        "relinked_video_keys": sorted(
            actor_frames.loc[relinked_rows, "video_key"].astype(str).unique()
        ),
        "exact_duplicate_rows_removed": int(
            actor_frames["xml_exact_duplicate_count"].astype(int).sum()
        ),
        "remaining_same_id_frame_conflicts": int(
            actor_frames.duplicated(["video_key", "frame_index", "pig_id"]).sum()
        ),
        "identity_authority": "XML ID attribute after exact-record deduplication",
        "gui_review_required": False,
        "gui_fallback": (
            "review_identity_continuity_gui_v2.py for non-identical "
            "same-ID/frame candidates"
        ),
    }
    if xml_identity_normalization["remaining_same_id_frame_conflicts"]:
        raise ValueError("DATA+ XML identity normalization left duplicate IDs")
    units, selected_actors, dropped = unitize_actor_rows(actor_frames)
    class_counts = {
        str(key): int(value)
        for key, value in units["behavior"].value_counts().sort_index().items()
    }
    if len(units) != args.expected_units:
        raise ValueError(
            f"DATA+ T6 count mismatch: expected={args.expected_units}, actual={len(units)}"
        )
    if class_counts != dict(sorted(EXPECTED_CLASS_COUNTS.items())):
        raise ValueError(
            "DATA+ T6 class counts changed: "
            f"expected={EXPECTED_CLASS_COUNTS}, actual={class_counts}"
        )

    eligibility, role_map = build_fold_eligibility(units, args.fold_manifest)
    collisions, duplicate_audit = canonical_duplicate_audit(
        units,
        args.canonical_row_manifest,
    )
    units["potential_canonical_signature_collision"] = units[
        "supplemental_unit_id"
    ].isin(collisions)
    detections = load_or_build_partner_detections(
        valid_pairs,
        args.output_root,
        args.detector_weights,
        args.reuse_partner_cache,
    )
    if detections.empty:
        clean_detection_contract = pd.Series(dtype=bool)
    else:
        clean_detection_contract = (
            detections["track_source"].eq("detected")
            & detections["hidden"].eq("No")
            & detections["needs_review"].eq("No")
        )
    if not bool(clean_detection_contract.all()):
        raise ValueError(
            "Partner cache contains predicted, hidden, or review-required rows"
        )
    processed_clip_ids = {str(pair["clip_id"]) for pair in valid_pairs}
    partner_arrays, partner_slots, partner_frames = build_partner_context_v2(
        units,
        selected_actors,
        detections,
        processed_clip_ids,
    )
    if len(processed_clip_ids) != len(valid_pairs):
        raise ValueError("DATA+ clip_id values are not unique for partner tracking")
    if not bool(partner_arrays["partner_extraction_complete"].all()):
        raise ValueError("Partner extraction did not complete for every DATA+ frame")
    missing_slots = ~partner_arrays["partner_mask"]
    invalid_padding = (
        bool((partner_arrays["partner_track_key"][missing_slots] != "").any())
        or bool((partner_arrays["partner_bbox_xyxy"][missing_slots] != 0.0).any())
        or bool((partner_arrays["partner_confidence"][missing_slots] != 0.0).any())
        or bool((partner_arrays["partner_geometry_6d"][missing_slots] != 0.0).any())
    )
    if invalid_padding:
        raise ValueError("Partner v2 missing slots violate masked-zero padding")
    target_features = build_target_frame_features(
        units,
        selected_actors,
        partner_slots,
        args.roi_coco,
    )
    units = attach_readiness(units, target_features, partner_frames)
    spatial = export_spatial_sequences(units, target_features)
    if spatial.audit.get("errors"):
        raise ValueError(f"DATA+ spatial export failed: {spatial.audit['errors']}")

    leakage_errors: list[str] = []
    invalid_roles = eligibility[~eligibility["supplemental_role"].isin({"train", "excluded"})]
    if not invalid_roles.empty:
        leakage_errors.append(f"invalid_supplemental_roles={len(invalid_roles)}")
    leaked_test = eligibility[
        eligibility["canonical_role"].eq("test")
        & eligibility["eligible_for_training"].astype(bool)
    ]
    if not leaked_test.empty:
        leakage_errors.append(f"canonical_test_group_in_train={len(leaked_test)}")
    if eligibility["eligible_for_inner_validation"].astype(bool).any():
        leakage_errors.append("supplemental_inner_validation_eligibility_nonzero")
    if eligibility["eligible_for_outer_test"].astype(bool).any():
        leakage_errors.append("supplemental_outer_test_eligibility_nonzero")
    if leakage_errors:
        raise ValueError(f"DATA+ leakage guard failed: {leakage_errors}")

    clip_inventory = _clip_inventory(pairs, probes, role_map)
    protected_after = {
        name: sha256_file(path) for name, path in protected_paths.items()
    }
    protected_unchanged = protected_before == protected_after
    if not protected_unchanged:
        raise RuntimeError("A frozen canonical or fold authority changed during build")

    args.output_root.mkdir(parents=True, exist_ok=True)
    _atomic_csv(clip_inventory, final_paths[0])
    _atomic_csv(units, final_paths[1])
    _atomic_csv(eligibility, final_paths[2])
    _atomic_csv(target_features, final_paths[3])
    _atomic_csv(detections, final_paths[4])
    _atomic_csv(partner_slots, final_paths[5])
    _atomic_csv(partner_frames, final_paths[6])
    _atomic_npz(partner_arrays, final_paths[7])
    _atomic_npz(spatial.arrays, final_paths[8])
    spatial_audit = {
        **spatial.audit,
        "feature_names": spatial.feature_names,
        "unit_manifest": str(final_paths[1]),
        "target_frame_features": str(final_paths[3]),
        "device": "cpu",
        "training_runs": 0,
    }
    _atomic_json(spatial_audit, final_paths[9])

    quality_counts = {
        str(key): int(value)
        for key, value in units["quality_tier"].value_counts().sort_index().items()
    }
    partner_ready_count = int(
        units["partner_context_reconstructable"].astype(bool).sum()
    )
    partner_available_units = int(
        partner_frames.groupby("supplemental_unit_id")["partner_available"]
        .any()
        .sum()
    )
    existing_keys = sorted(
        {str(pair["video_key"]) for pair in pairs if str(pair["video_key"]) in role_map}
    )
    new_keys = sorted(
        {
            str(pair["video_key"])
            for pair in pairs
            if str(pair["video_key"]) not in role_map
        }
    )
    audit = {
        "schema_version": "classification_v2.data_plus_supplemental_t6.v2",
        "status": "PASS",
        "data_plus_root": str(args.data_plus_root),
        "output_root": str(args.output_root),
        "clips_total": len(pairs),
        "clips_excluded_fixed": len(pairs) - len(valid_pairs),
        "fixed_exclusion_video_keys": sorted(FIXED_EXCLUSION_KEYS),
        "clips_valid": len(valid_pairs),
        "resolved_video_keys": len({str(pair["video_key"]) for pair in pairs}),
        "unresolved_video_keys": 0,
        "existing_video_keys": existing_keys,
        "new_video_keys": new_keys,
        "total_valid_t6": len(units),
        "class_counts": class_counts,
        "xml_identity_normalization": xml_identity_normalization,
        "dropped_remainder_observations_by_class": dropped,
        "quality_tier_counts": quality_counts,
        "partner_context_reconstructable": partner_ready_count,
        "partner_context_not_reconstructable": (
            len(units) - partner_ready_count
        ),
        "partner_available_units": partner_available_units,
        "partner_tracking": {
            "mode": PARTNER_TRACKING_MODE,
            "profile": PARTNER_TRACKING_PROFILE,
            "profile_sha256": partner_tracking_profile_hash(),
            "lineage_sha256": partner_tracking_lineage_hash(),
            "device": "cpu",
            "clips_processed": len(processed_clip_ids),
            "clips_expected": len(valid_pairs),
            "clean_detected_rows": len(detections),
            "candidate_slots_k": PARTNER_K,
            "selection_authority": "observation_only",
            "target_label_fields_used": False,
            "partner_behavior_fields_used": False,
            "non_target_xml_annotations_used": False,
            "evidence_policy": (
                "detected and non-hidden and not review-required; predicted "
                "or occluded tracker boxes are excluded"
            ),
        },
        "partner_context_v2": {
            "path": str(final_paths[7]),
            "unit_count": len(partner_arrays["supplemental_unit_id"]),
            "valid_partner_slots": int(partner_arrays["partner_mask"].sum()),
            "extraction_complete_frames": int(
                partner_arrays["partner_extraction_complete"].sum()
            ),
            "array_shapes": {
                key: list(value.shape) for key, value in partner_arrays.items()
            },
            "padding_contract": "zeros and empty key only where mask=false",
        },
        "unitization_contract": {
            "target_length": T6_LENGTH,
            "sampling_pattern": "contiguous",
            "unit_stride_frames": T6_LENGTH,
            "overlap_frames": 0,
            "padding_policy": "none",
            "selected_target_observations": len(selected_actors),
            "reused_target_observations": int(
                selected_actors.duplicated(
                    ["clip_id", "object_track_key", "frame_index"],
                    keep=False,
                ).sum()
            ),
        },
        "roi_context_reconstructable": int(
            units["roi_context_reconstructable"].astype(bool).sum()
        ),
        "duplicate_audit": duplicate_audit,
        "fold_eligibility": {
            "rows": len(eligibility),
            "training_rows": int(
                eligibility["eligible_for_training"].astype(bool).sum()
            ),
            "excluded_rows": int(
                (~eligibility["eligible_for_training"].astype(bool)).sum()
            ),
            "inner_validation_rows": 0,
            "outer_test_rows": 0,
            "errors": leakage_errors,
            "method": (
                "existing video_key follows frozen split_VG1..VG5 role; new "
                "video_key is isolated train-only; consumer appends only after "
                "inner_train selection and excludes matching inner_val groups"
            ),
        },
        "authorities": {
            "fold_manifest": {
                "path": str(args.fold_manifest),
                "sha256": protected_before["fold_manifest"],
            },
            "canonical_row_manifest": {
                "path": str(args.canonical_row_manifest),
                "sha256": protected_before["canonical_row_manifest"],
            },
            "canonical_npz": {
                "path": str(args.canonical_npz),
                "sha256": protected_before["canonical_npz"],
            },
            "roi_coco": {
                "path": str(args.roi_coco),
                "sha256": sha256_file(args.roi_coco),
            },
            "detector_weights": {
                "path": str(args.detector_weights),
                "sha256": sha256_file(args.detector_weights),
            },
        },
        "protected_authorities_unchanged": protected_unchanged,
        "human_non_overlap_assumption": True,
        "gpu_used": False,
        "gui_used": False,
        "training_runs": 0,
        "canonical_population_modified": False,
        "frozen_fold_manifests_modified": False,
    }
    _atomic_json(audit, final_paths[10])
    artifact_hashes = {
        str(path.relative_to(args.output_root)): sha256_file(path)
        for path in final_paths[:-1]
    }
    _atomic_json(
        {
            "schema_version": "classification_v2.artifact_hashes.v2",
            "artifacts": artifact_hashes,
        },
        final_paths[11],
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
