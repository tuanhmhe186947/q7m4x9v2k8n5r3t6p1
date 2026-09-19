"""Deterministic 224px full-frame cache for the legacy L6 S4 short gate."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from pig_behavior.classification_v2.datasets.image_sequence_dataset import (
    letterbox_rgb_uint8,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

LINEAGE_SCOPE = "legacy-only-unreviewed-development"
CANONICAL_SOURCE_NAME = "legacy_16f"
SOURCE_TYPE = "legacy_recovered"
DATASET_ID = "legacy_recovered_16f"
IMAGE_SIZE = 224
SEQUENCE_LENGTH = 6
EXPECTED_WINDOWS = 1_300
EXPECTED_SCENE_FRAMES = 1_545
RESIZE_POLICY = "full_frame_letterbox_rgb_pad_black_v1"
CONFIG_SCHEMA = (
    "classification_v2.legacy_development_l6."
    "full_frame_short_cache_config.v1"
)
AUDIT_SCHEMA = (
    "classification_v2.legacy_development_l6."
    "full_frame_short_cache.v1"
)
REPEAT_SCHEMA = (
    "classification_v2.legacy_development_l6."
    "full_frame_short_cache_repeat.v1"
)


@dataclass(frozen=True, slots=True)
class LegacyL6FullFrameCacheConfig:
    path: Path
    payload: dict[str, Any]
    repo_root: Path

    @property
    def sha256(self) -> str:
        return file_sha256(self.path)

    def output_root(self, replica: str) -> Path:
        if replica not in {"primary", "repeat"}:
            raise ValueError(f"unknown full-frame cache replica={replica}")
        value = self.payload["outputs"][f"{replica}_root_relative_path"]
        return _resolve_inside(self.repo_root, value)


def load_full_frame_cache_config(path: Path) -> LegacyL6FullFrameCacheConfig:
    resolved = path.resolve()
    payload = _read_json(resolved)
    _validate_config(payload)
    config = LegacyL6FullFrameCacheConfig(
        path=resolved,
        payload=payload,
        repo_root=resolved.parents[2],
    )
    for section in ("inputs", "implementation"):
        for name, spec_value in payload[section].items():
            spec = _object(spec_value, f"{section}.{name}")
            bound = _resolve_inside(config.repo_root, spec["path"])
            _require_bound_file(bound, spec["sha256"], f"{section}.{name}")
    return config


def build_full_frame_cache(
    config: LegacyL6FullFrameCacheConfig,
    *,
    replica: str,
) -> dict[str, Any]:
    """Decode each selected scene frame once into a packed letterbox cache."""

    guard = _git_guard(config)
    if not guard["valid"]:
        raise ValueError(f"full-frame cache git guard failed={guard['errors']}")
    output_root = config.output_root(replica)
    if output_root.exists():
        raise FileExistsError(f"full-frame cache output exists={output_root}")
    selected = _selected_scene_frames(config)
    _require_media_access(selected["resolved_media_path"])
    output_root.mkdir(parents=True)
    tensor_path = output_root / "full_frame_rgb_224_letterbox.npy"
    index_path = output_root / "full_frame_index.csv"
    audit_path = output_root / "full_frame_cache_audit.json"
    start = time.perf_counter()
    try:
        index, runtime = _decode_full_frames(selected, tensor_path)
        index.to_csv(index_path, index=False)
        audit = _cache_audit(
            config,
            replica=replica,
            output_root=output_root,
            tensor_path=tensor_path,
            index_path=index_path,
            index=index,
            runtime=runtime,
            runtime_seconds=time.perf_counter() - start,
            git_guard=guard,
        )
        audit_path.write_text(
            json.dumps(audit, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return audit
    except Exception as error:
        failure = {
            "schema_version": f"{AUDIT_SCHEMA}.failure",
            "replica": replica,
            "config_sha256": config.sha256,
            "error_type": type(error).__name__,
            "error": str(error),
            "valid": False,
        }
        (output_root / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        raise


def write_full_frame_cache_repeat_gate(
    config: LegacyL6FullFrameCacheConfig,
) -> tuple[Path, dict[str, Any]]:
    """Require byte-identical primary and independent-repeat cache artifacts."""

    audits: dict[str, dict[str, Any]] = {}
    for replica in ("primary", "repeat"):
        root = config.output_root(replica)
        audit_path = root / "full_frame_cache_audit.json"
        audit = _read_json(audit_path)
        if audit.get("valid") is not True or audit.get("replica") != replica:
            raise ValueError(f"invalid full-frame cache audit replica={replica}")
        audits[replica] = audit
    exact_fields = (
        "scene_frame_order_sha256",
        "tensor_sha256",
        "index_sha256",
        "packed_rows",
        "tensor_shape",
        "tensor_dtype",
        "resize_policy",
    )
    equality = {
        field: audits["primary"][field] == audits["repeat"][field]
        for field in exact_fields
    }
    errors = [field for field, equal in equality.items() if not equal]
    payload = {
        "schema_version": REPEAT_SCHEMA,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_FULL_FRAME_CACHE_REPEAT"
            if not errors
            else "FAIL_LEGACY_DEVELOPMENT_L6_FULL_FRAME_CACHE_REPEAT"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "canonical_source_name": CANONICAL_SOURCE_NAME,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "config_sha256": config.sha256,
        "equality": equality,
        "primary_audit_sha256": file_sha256(
            config.output_root("primary") / "full_frame_cache_audit.json"
        ),
        "repeat_audit_sha256": file_sha256(
            config.output_root("repeat") / "full_frame_cache_audit.json"
        ),
        "source_media_fallback_during_training": 0,
        "outer_holdout_rows": 0,
        "errors": errors,
        "valid": not errors,
    }
    output = _resolve_inside(
        config.repo_root,
        config.payload["outputs"]["repeat_gate_relative_path"],
    )
    if output.exists():
        raise FileExistsError(f"full-frame repeat gate exists={output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output, payload


def _selected_scene_frames(
    config: LegacyL6FullFrameCacheConfig,
) -> pd.DataFrame:
    inputs = config.payload["inputs"]
    subset_audit = _read_json(
        _resolve_inside(
            config.repo_root,
            inputs["window_subset_audit"]["path"],
        )
    )
    expected_subset = {
        "selected_windows": EXPECTED_WINDOWS,
        "missing_context_ids": 0,
        "outer_holdout_windows": 0,
        "output_manifest_sha256": inputs["selected_window_context"][
            "sha256"
        ],
        "valid": True,
    }
    for field, value in expected_subset.items():
        if subset_audit.get(field) != value:
            raise ValueError(f"full-frame subset audit drift={field}")
    windows = pd.read_csv(
        _resolve_inside(
            config.repo_root,
            inputs["selected_window_context"]["path"],
        ),
        low_memory=False,
    )
    required_window = {
        "window_id",
        "scene_frame_uid_sequence",
        "lineage_scope",
        "human_review_complete",
    }
    if not required_window.issubset(windows.columns):
        raise ValueError("full-frame selected-window columns are incomplete")
    if len(windows) != EXPECTED_WINDOWS:
        raise ValueError(f"full-frame selected windows={len(windows)}")
    if windows["window_id"].astype(str).duplicated().any():
        raise ValueError("full-frame selected windows contain duplicates")
    _require_claim_columns(windows, "selected windows")
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for value in windows["scene_frame_uid_sequence"].astype(str):
        ids = [item for item in value.split("|") if item]
        if len(ids) != SEQUENCE_LENGTH:
            raise ValueError("full-frame scene sequence length drift")
        for scene_id in ids:
            if scene_id not in seen:
                seen.add(scene_id)
                ordered_ids.append(scene_id)
    if len(ordered_ids) != EXPECTED_SCENE_FRAMES:
        raise ValueError(f"full-frame unique scene count={len(ordered_ids)}")
    frames = pd.read_csv(
        _resolve_inside(
            config.repo_root,
            inputs["image_frame_context_manifest"]["path"],
        ),
        usecols=[
            "scene_frame_uid",
            "source_type",
            "dataset_id",
            "video_key",
            "frame_index",
            "resolved_media_path",
            "image_width",
            "image_height",
            "lineage_scope",
            "human_review_complete",
            "full_frame_context_available",
        ],
        low_memory=False,
    )
    selected = frames[frames["scene_frame_uid"].astype(str).isin(seen)].copy()
    if selected["scene_frame_uid"].astype(str).nunique() != len(ordered_ids):
        raise ValueError("full-frame scene selection is incomplete")
    _validate_scene_metadata(selected)
    selected = selected.drop_duplicates("scene_frame_uid", keep="first")
    order = {scene_id: index for index, scene_id in enumerate(ordered_ids)}
    selected["selection_order"] = selected["scene_frame_uid"].map(order)
    return selected.sort_values(
        ["resolved_media_path", "frame_index", "scene_frame_uid"],
        kind="mergesort",
    ).reset_index(drop=True)


def _validate_scene_metadata(frame: pd.DataFrame) -> None:
    _require_claim_columns(frame, "scene frame manifest")
    expected = {
        "source_type": SOURCE_TYPE,
        "dataset_id": DATASET_ID,
        "full_frame_context_available": True,
    }
    for column, value in expected.items():
        if not frame[column].eq(value).all():
            raise ValueError(f"full-frame scene {column} drift")
    metadata = [
        "video_key",
        "frame_index",
        "resolved_media_path",
        "image_width",
        "image_height",
        "source_type",
        "dataset_id",
        "lineage_scope",
        "human_review_complete",
    ]
    unique = frame.groupby("scene_frame_uid", sort=False)[metadata].nunique(
        dropna=False
    )
    if int(unique.to_numpy().max()) != 1:
        raise ValueError("full-frame duplicate scene metadata conflicts")


def _decode_full_frames(
    selected: pd.DataFrame,
    tensor_path: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    tensor = np.lib.format.open_memmap(
        tensor_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(selected), IMAGE_SIZE, IMAGE_SIZE, 3),
    )
    rows: list[dict[str, Any]] = []
    decode_count = 0
    seek_count = 0
    try:
        for media_path, group in selected.groupby(
            "resolved_media_path",
            sort=False,
        ):
            capture = cv2.VideoCapture(str(media_path))
            if not capture.isOpened():
                raise OSError(f"cannot open full-frame media={media_path}")
            next_frame: int | None = None
            try:
                for item in group.itertuples(index=False):
                    target = int(item.frame_index)
                    if next_frame != target:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, target)
                        seek_count += 1
                    ok, bgr = capture.read()
                    if not ok or bgr is None:
                        raise OSError(
                            f"cannot decode full frame={media_path}:{target}"
                        )
                    next_frame = target + 1
                    decode_count += 1
                    height, width = bgr.shape[:2]
                    if (width, height) != (
                        int(item.image_width),
                        int(item.image_height),
                    ):
                        raise ValueError("full-frame decoded dimension drift")
                    packed_row = len(rows)
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    tensor[packed_row] = letterbox_rgb_uint8(rgb, IMAGE_SIZE)
                    rows.append(
                        _index_row(
                            item,
                            packed_row=packed_row,
                            width=width,
                            height=height,
                        )
                    )
            finally:
                capture.release()
        tensor.flush()
    finally:
        mmap = getattr(tensor, "_mmap", None)
        if mmap is not None:
            mmap.close()
    return pd.DataFrame.from_records(rows), {
        "video_decode_count": decode_count,
        "video_seek_count": seek_count,
        "peak_open_videos": 1,
    }


def _index_row(
    item: Any,
    *,
    packed_row: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    pad_left = (IMAGE_SIZE - resized_width) // 2
    pad_top = (IMAGE_SIZE - resized_height) // 2
    return {
        "scene_frame_uid": str(item.scene_frame_uid),
        "packed_row": packed_row,
        "selection_order": int(item.selection_order),
        "video_key": str(item.video_key),
        "frame_index": int(item.frame_index),
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "source_width": width,
        "source_height": height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "pad_left": pad_left,
        "pad_right": IMAGE_SIZE - resized_width - pad_left,
        "pad_top": pad_top,
        "pad_bottom": IMAGE_SIZE - resized_height - pad_top,
        "resize_policy": RESIZE_POLICY,
    }


def _cache_audit(
    config: LegacyL6FullFrameCacheConfig,
    **values: Any,
) -> dict[str, Any]:
    index = values["index"]
    runtime = values["runtime"]
    tensor_path = values["tensor_path"]
    index_path = values["index_path"]
    tensor = np.load(tensor_path, mmap_mode="r")
    try:
        shape = list(tensor.shape)
        dtype = str(tensor.dtype)
    finally:
        mmap = getattr(tensor, "_mmap", None)
        if mmap is not None:
            mmap.close()
    return {
        "schema_version": AUDIT_SCHEMA,
        "status": "PASS_LEGACY_DEVELOPMENT_L6_FULL_FRAME_SHORT_CACHE",
        "replica": values["replica"],
        "lineage_scope": LINEAGE_SCOPE,
        "canonical_source_name": CANONICAL_SOURCE_NAME,
        "source_type": SOURCE_TYPE,
        "dataset_id": DATASET_ID,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "config_sha256": config.sha256,
        "code_sha": values["git_guard"]["code_sha"],
        "packed_rows": len(index),
        "selected_windows": EXPECTED_WINDOWS,
        "selected_slots": EXPECTED_WINDOWS * SEQUENCE_LENGTH,
        "unique_scene_frames": EXPECTED_SCENE_FRAMES,
        "scene_frame_order_sha256": _ordered_sha256(
            index["scene_frame_uid"]
        ),
        "tensor_path": str(tensor_path),
        "tensor_sha256": file_sha256(tensor_path),
        "tensor_shape": shape,
        "tensor_dtype": dtype,
        "index_path": str(index_path),
        "index_sha256": file_sha256(index_path),
        "resize_policy": RESIZE_POLICY,
        "aspect_distortion_used": False,
        "letterbox_metadata_complete": True,
        "source_media_reads": runtime["video_decode_count"],
        "source_media_fallback_during_training": 0,
        "outer_holdout_rows": 0,
        "video_decode_count": runtime["video_decode_count"],
        "video_seek_count": runtime["video_seek_count"],
        "peak_open_videos": runtime["peak_open_videos"],
        "runtime_seconds": values["runtime_seconds"],
        "errors": [],
        "valid": True,
    }


def _validate_config(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("full-frame cache config schema drift")
    identity = {
        "lineage_scope": LINEAGE_SCOPE,
        "canonical_source_name": CANONICAL_SOURCE_NAME,
        "source_type": SOURCE_TYPE,
        "dataset_id": DATASET_ID,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
    }
    for field, value in identity.items():
        if payload.get(field) != value:
            raise ValueError(f"full-frame cache identity drift={field}")
    for section in ("inputs", "implementation", "outputs", "execution_guard"):
        if not isinstance(payload.get(section), dict):
            raise ValueError(f"full-frame cache missing section={section}")


def _git_guard(config: LegacyL6FullFrameCacheConfig) -> dict[str, Any]:
    guard = config.payload["execution_guard"]
    completed = subprocess.run(
        ["git", "-C", str(config.repo_root), "status", "--porcelain"],
        capture_output=True,
        check=True,
        text=True,
    )
    observed = sorted(
        line[3:].strip().replace("\\", "/")
        for line in completed.stdout.splitlines()
        if line.strip()
    )
    allowed = sorted(str(value).replace("\\", "/") for value in guard["allowed_dirty_paths"])
    unexpected = sorted(set(observed) - set(allowed))
    required = [str(value).replace("\\", "/") for value in guard["required_tracked_paths"]]
    untracked = []
    for value in required:
        check = subprocess.run(
            ["git", "-C", str(config.repo_root), "ls-files", "--error-unmatch", "--", value],
            capture_output=True,
            check=False,
            text=True,
        )
        if check.returncode != 0:
            untracked.append(value)
    errors = []
    if unexpected:
        errors.append(f"unexpected_dirty_paths={unexpected}")
    if untracked:
        errors.append(f"required_paths_untracked={untracked}")
    code_sha = subprocess.run(
        ["git", "-C", str(config.repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    return {
        "code_sha": code_sha,
        "observed_dirty_paths": observed,
        "allowed_dirty_paths": allowed,
        "unexpected_dirty_paths": unexpected,
        "untracked_required_paths": untracked,
        "errors": errors,
        "valid": not errors,
    }


def _require_claim_columns(frame: pd.DataFrame, name: str) -> None:
    if not frame["lineage_scope"].eq(LINEAGE_SCOPE).all():
        raise ValueError(f"{name} lineage drift")
    if not frame["human_review_complete"].eq(False).all():
        raise ValueError(f"{name} review claim drift")


def _require_media_access(values: pd.Series) -> None:
    missing = [value for value in sorted(set(values.astype(str))) if not Path(value).is_file()]
    if missing:
        raise FileNotFoundError(f"full-frame media unavailable={missing[:3]}")


def _ordered_sha256(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values.astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _require_bound_file(path: Path, expected: object, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing full-frame {name}={path}")
    if file_sha256(path) != str(expected):
        raise ValueError(f"full-frame bound hash drift={name}")


def _resolve_inside(root: Path, value: object) -> Path:
    path = (root.resolve() / str(value)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"full-frame path escapes repository={value}") from error
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return _object(json.loads(path.read_text(encoding="utf-8")), str(path))


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"full-frame {name} must be an object")
    return value
