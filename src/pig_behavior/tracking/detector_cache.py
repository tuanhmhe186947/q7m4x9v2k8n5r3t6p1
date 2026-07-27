"""Fail-closed detector-evidence recording and replay.

The cache binds detector output to one video, detector, semantic detector
configuration, and producer revision.  It is an opt-in transport for paired
tracking work; importing this module does not select it or change tracking
defaults.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DETECTOR_CACHE_SCHEMA_VERSION = "tracking.detector_evidence_cache.v2"
_ARRAY_FIELDS = ("xyxy", "conf", "cls", "id", "masks", "appearance")
_SHA256_LENGTH = 64


class DetectorCacheError(RuntimeError):
    """Raised when detector evidence cannot be recorded or replayed safely."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        return cpu().numpy()
    return np.asarray(value)


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class DetectorCacheIdentity:
    """Immutable authority bound into every detector cache artifact."""

    video_key: str
    source_video_sha256: str
    detector_weight_sha256: str
    detector_semantic_config_sha256: str
    producer_code_sha: str
    creation_authority: str
    requires_masks: bool = False
    requires_appearance_descriptors: bool = False
    uses_frame_timestamps: bool = False

    def validate(self) -> None:
        if not self.video_key.strip():
            raise DetectorCacheError("video_key must be non-empty")
        if not self.creation_authority.strip():
            raise DetectorCacheError("creation_authority must be non-empty")
        for field_name in (
            "source_video_sha256",
            "detector_weight_sha256",
            "detector_semantic_config_sha256",
        ):
            value = str(getattr(self, field_name)).lower()
            if not _is_sha256(value):
                raise DetectorCacheError(
                    f"{field_name} must be a lowercase SHA-256"
                )
        producer = self.producer_code_sha.lower()
        if len(producer) not in {40, 64} or not all(
            character in "0123456789abcdef" for character in producer
        ):
            raise DetectorCacheError(
                "producer_code_sha must be a 40- or 64-character hex digest"
            )


@dataclass(slots=True)
class _CachedBoxes:
    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray
    id: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])


@dataclass(slots=True)
class _CachedMasks:
    data: np.ndarray


@dataclass(slots=True)
class _CachedResult:
    boxes: _CachedBoxes
    names: dict[int, str]
    masks: _CachedMasks | None
    orig_shape: tuple[int, int]
    appearance_descriptors: np.ndarray | None = None
    frame_timestamp: float | None = None


@dataclass(slots=True)
class DetectorEvidenceCache:
    """Detector results for a strictly increasing sequence of frame indices."""

    identity: DetectorCacheIdentity
    names: dict[int, str] = field(default_factory=dict)
    frames: dict[int, dict[str, Any]] = field(default_factory=dict)
    cache_artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        self.identity.validate()

    def record(
        self,
        frame_index: int,
        result: Any,
        *,
        original_frame_dimensions: tuple[int, int],
        frame_timestamp: float | None = None,
        appearance_descriptors: np.ndarray | None = None,
    ) -> None:
        """Record one result, rejecting duplicates and non-monotonic frames."""

        index = int(frame_index)
        if index < 0:
            raise DetectorCacheError("frame_index must be non-negative")
        if self.frames and index <= max(self.frames):
            raise DetectorCacheError(
                "frame indices must be unique and strictly increasing"
            )
        height, width = (int(value) for value in original_frame_dimensions)
        if height <= 0 or width <= 0:
            raise DetectorCacheError("original frame dimensions must be positive")
        if self.identity.uses_frame_timestamps:
            if frame_timestamp is None or not math.isfinite(frame_timestamp):
                raise DetectorCacheError("a finite frame timestamp is required")
        elif frame_timestamp is not None:
            raise DetectorCacheError(
                "frame timestamp supplied when identity declares it unused"
            )

        boxes = getattr(result, "boxes", None)
        raw_xyxy = None if boxes is None else _to_numpy(boxes.xyxy)
        count = 0 if raw_xyxy is None else int(np.asarray(raw_xyxy).shape[0])
        xyxy = np.empty((0, 4), dtype=np.float32)
        conf = np.empty((0,), dtype=np.float32)
        classes = np.empty((0,), dtype=np.int64)
        raw_ids: np.ndarray | None = None
        masks: np.ndarray | None = None
        if count:
            xyxy = np.asarray(raw_xyxy, dtype=np.float32)
            conf = np.asarray(_to_numpy(boxes.conf), dtype=np.float32)
            classes = np.asarray(_to_numpy(boxes.cls), dtype=np.int64)
            raw_id = _to_numpy(getattr(boxes, "id", None))
            if raw_id is not None:
                raw_ids = np.asarray(raw_id, dtype=np.float32)
            raw_masks = getattr(getattr(result, "masks", None), "data", None)
            if raw_masks is not None:
                masks = np.asarray(_to_numpy(raw_masks), dtype=np.float32)
        appearance = (
            None
            if appearance_descriptors is None
            else np.asarray(appearance_descriptors, dtype=np.float32)
        )
        entry = {
            "xyxy": xyxy,
            "conf": conf,
            "cls": classes,
            "id": raw_ids,
            "masks": masks,
            "appearance": appearance,
            "original_frame_dimensions": (height, width),
            "frame_timestamp": frame_timestamp,
        }
        self._validate_frame(index, entry)
        if not self.names:
            raw_names = getattr(result, "names", {}) or {}
            self.names = {
                int(key): str(value)
                for key, value in dict(raw_names).items()
            }
        self.frames[index] = entry
        self.cache_artifact_sha256 = None

    def build_result(self, frame_index: int) -> _CachedResult:
        """Build an isolated result object for one exact cached frame."""

        index = int(frame_index)
        try:
            entry = self.frames[index]
        except KeyError as exc:
            raise DetectorCacheError(
                f"no detector evidence for frame_index={index}"
            ) from exc
        boxes = _CachedBoxes(
            xyxy=entry["xyxy"].copy(),
            conf=entry["conf"].copy(),
            cls=entry["cls"].copy(),
            id=None if entry["id"] is None else entry["id"].copy(),
        )
        masks = (
            None
            if entry["masks"] is None
            else _CachedMasks(entry["masks"].copy())
        )
        appearance = (
            None
            if entry["appearance"] is None
            else entry["appearance"].copy()
        )
        return _CachedResult(
            boxes=boxes,
            names=dict(self.names),
            masks=masks,
            orig_shape=entry["original_frame_dimensions"],
            appearance_descriptors=appearance,
            frame_timestamp=entry["frame_timestamp"],
        )

    def save(self, path: Path) -> str:
        """Serialize the cache and return the SHA-256 of the NPZ artifact."""

        if not self.frames:
            raise DetectorCacheError("refusing to save an empty detector cache")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = self._metadata()
        payload: dict[str, np.ndarray] = {
            "__metadata_json__": np.frombuffer(
                _canonical_json(metadata),
                dtype=np.uint8,
            )
        }
        for frame_index, entry in self.frames.items():
            for key in _ARRAY_FIELDS:
                array = entry[key]
                if array is not None:
                    payload[f"f{frame_index}__{key}"] = array
        np.savez_compressed(path, **payload)
        artifact_sha = _sha256_file(path)
        sidecar = {
            "schema_version": DETECTOR_CACHE_SCHEMA_VERSION,
            "cache_artifact_sha256": artifact_sha,
        }
        path.with_suffix(".sha256.json").write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.cache_artifact_sha256 = artifact_sha
        return artifact_sha

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_identity: DetectorCacheIdentity,
    ) -> DetectorEvidenceCache:
        """Load a cache only when artifact and semantic identity both match."""

        path = Path(path)
        sidecar_path = path.with_suffix(".sha256.json")
        if not path.is_file() or not sidecar_path.is_file():
            raise DetectorCacheError("cache artifact and SHA sidecar are required")
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DetectorCacheError("malformed detector cache SHA sidecar") from exc
        if sidecar.get("schema_version") != DETECTOR_CACHE_SCHEMA_VERSION:
            raise DetectorCacheError("unsupported detector cache schema version")
        actual_artifact_sha = _sha256_file(path)
        if sidecar.get("cache_artifact_sha256") != actual_artifact_sha:
            raise DetectorCacheError("corrupted detector cache artifact hash")

        try:
            with np.load(path, allow_pickle=False) as payload:
                if "__metadata_json__" not in payload.files:
                    raise DetectorCacheError("cache metadata is missing")
                metadata = json.loads(
                    np.asarray(payload["__metadata_json__"], dtype=np.uint8)
                    .tobytes()
                    .decode("utf-8")
                )
                cache = cls._from_payload(
                    payload,
                    metadata,
                    expected_identity=expected_identity,
                )
        except DetectorCacheError:
            raise
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise DetectorCacheError("malformed detector cache artifact") from exc
        cache.cache_artifact_sha256 = actual_artifact_sha
        return cache

    @classmethod
    def _from_payload(
        cls,
        payload: Any,
        metadata: dict[str, Any],
        *,
        expected_identity: DetectorCacheIdentity,
    ) -> DetectorEvidenceCache:
        if metadata.get("schema_version") != DETECTOR_CACHE_SCHEMA_VERSION:
            raise DetectorCacheError("unsupported embedded cache schema version")
        try:
            actual_identity = DetectorCacheIdentity(**metadata["identity"])
        except (KeyError, TypeError) as exc:
            raise DetectorCacheError("cache identity is missing or malformed") from exc
        actual_identity.validate()
        expected_identity.validate()
        if actual_identity != expected_identity:
            raise DetectorCacheError("detector cache identity mismatch")

        raw_frames = metadata.get("frames")
        if not isinstance(raw_frames, list) or not raw_frames:
            raise DetectorCacheError("cache frame metadata is missing")
        indices = [record.get("frame_index") for record in raw_frames]
        if any(not isinstance(index, int) for index in indices):
            raise DetectorCacheError("frame indices must be integers")
        if indices != sorted(set(indices)):
            raise DetectorCacheError(
                "duplicate or non-monotonic cache frame indices"
            )
        if metadata.get("frame_count") != len(indices):
            raise DetectorCacheError("cache frame count mismatch")

        names_payload = metadata.get("names")
        if not isinstance(names_payload, dict):
            raise DetectorCacheError("cache class names are missing")
        cache = cls(
            identity=actual_identity,
            names={
                int(key): str(value)
                for key, value in names_payload.items()
            },
        )
        expected_keys = {"__metadata_json__"}
        for record in raw_frames:
            frame_index = int(record["frame_index"])
            presence = record.get("arrays")
            if not isinstance(presence, dict):
                raise DetectorCacheError("frame array manifest is missing")
            entry: dict[str, Any] = {}
            for key in _ARRAY_FIELDS:
                payload_key = f"f{frame_index}__{key}"
                is_present = presence.get(key)
                if not isinstance(is_present, bool):
                    raise DetectorCacheError("frame array presence is malformed")
                if is_present:
                    if payload_key not in payload.files:
                        raise DetectorCacheError(
                            f"required cache array is missing: {payload_key}"
                        )
                    entry[key] = np.asarray(payload[payload_key])
                    expected_keys.add(payload_key)
                else:
                    if payload_key in payload.files:
                        raise DetectorCacheError(
                            f"undeclared cache array is present: {payload_key}"
                        )
                    entry[key] = None
            dimensions = record.get("original_frame_dimensions")
            if (
                not isinstance(dimensions, list)
                or len(dimensions) != 2
                or any(not isinstance(value, int) for value in dimensions)
            ):
                raise DetectorCacheError("original frame dimensions are malformed")
            entry["original_frame_dimensions"] = tuple(dimensions)
            entry["frame_timestamp"] = record.get("frame_timestamp")
            cache._validate_frame(frame_index, entry)
            cache.frames[frame_index] = entry
        if set(payload.files) != expected_keys:
            raise DetectorCacheError("cache contains undeclared array entries")
        return cache

    def _metadata(self) -> dict[str, Any]:
        frame_records = []
        for frame_index, entry in self.frames.items():
            self._validate_frame(frame_index, entry)
            frame_records.append(
                {
                    "frame_index": frame_index,
                    "original_frame_dimensions": list(
                        entry["original_frame_dimensions"]
                    ),
                    "frame_timestamp": entry["frame_timestamp"],
                    "arrays": {
                        key: entry[key] is not None for key in _ARRAY_FIELDS
                    },
                }
            )
        return {
            "schema_version": DETECTOR_CACHE_SCHEMA_VERSION,
            "identity": asdict(self.identity),
            "names": {
                str(key): value for key, value in sorted(self.names.items())
            },
            "frame_count": len(frame_records),
            "frames": frame_records,
        }

    def _validate_frame(
        self,
        frame_index: int,
        entry: dict[str, Any],
    ) -> None:
        xyxy = entry.get("xyxy")
        conf = entry.get("conf")
        classes = entry.get("cls")
        if not isinstance(xyxy, np.ndarray) or xyxy.dtype != np.float32:
            raise DetectorCacheError(f"frame {frame_index}: xyxy must be float32")
        if xyxy.ndim != 2 or xyxy.shape[1:] != (4,):
            raise DetectorCacheError(f"frame {frame_index}: malformed xyxy")
        count = xyxy.shape[0]
        for key, array, dtype in (
            ("conf", conf, np.float32),
            ("cls", classes, np.int64),
        ):
            if (
                not isinstance(array, np.ndarray)
                or array.dtype != dtype
                or array.shape != (count,)
            ):
                raise DetectorCacheError(
                    f"frame {frame_index}: malformed {key}"
                )
        for key in ("id",):
            array = entry.get(key)
            if array is not None and (
                not isinstance(array, np.ndarray)
                or array.dtype != np.float32
                or array.shape != (count,)
            ):
                raise DetectorCacheError(
                    f"frame {frame_index}: malformed {key}"
                )
        for key in ("masks", "appearance"):
            array = entry.get(key)
            if array is not None and (
                not isinstance(array, np.ndarray)
                or array.dtype != np.float32
                or array.ndim < 2
                or array.shape[0] != count
            ):
                raise DetectorCacheError(
                    f"frame {frame_index}: malformed {key}"
                )
        numeric_arrays = [
            entry[key]
            for key in _ARRAY_FIELDS
            if entry.get(key) is not None
        ]
        if any(not np.isfinite(array).all() for array in numeric_arrays):
            raise DetectorCacheError(
                f"frame {frame_index}: non-finite detector evidence"
            )
        dimensions = entry.get("original_frame_dimensions")
        if (
            not isinstance(dimensions, tuple)
            or len(dimensions) != 2
            or any(not isinstance(value, int) or value <= 0 for value in dimensions)
        ):
            raise DetectorCacheError(
                f"frame {frame_index}: invalid original frame dimensions"
            )
        timestamp = entry.get("frame_timestamp")
        if self.identity.uses_frame_timestamps:
            if not isinstance(timestamp, (int, float)) or not math.isfinite(
                timestamp
            ):
                raise DetectorCacheError(
                    f"frame {frame_index}: frame timestamp is required"
                )
        elif timestamp is not None:
            raise DetectorCacheError(
                f"frame {frame_index}: unexpected frame timestamp"
            )
        if self.identity.requires_masks and entry.get("masks") is None:
            raise DetectorCacheError(f"frame {frame_index}: masks are required")
        if (
            self.identity.requires_appearance_descriptors
            and entry.get("appearance") is None
        ):
            raise DetectorCacheError(
                f"frame {frame_index}: appearance descriptors are required"
            )


class RecordingDetector:
    """Forward model calls while recording exact result evidence."""

    detector_cache_mode = "record"

    def __init__(self, model: Any, cache: DetectorEvidenceCache) -> None:
        self._model = model
        self._cache = cache
        self._frame_index: int | None = None
        self._frame_dimensions: tuple[int, int] | None = None
        self.invocations = 0

    def set_frame_context(
        self,
        frame_index: int,
        original_frame_dimensions: tuple[int, int],
    ) -> None:
        self._frame_index = int(frame_index)
        self._frame_dimensions = tuple(
            int(value) for value in original_frame_dimensions
        )

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke("predict", *args, **kwargs)

    def track(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke("track", *args, **kwargs)

    def _invoke(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if self._frame_index is None or self._frame_dimensions is None:
            raise DetectorCacheError(
                "set_frame_context must be called before detector invocation"
            )
        results = getattr(self._model, method)(*args, **kwargs)
        if not isinstance(results, (list, tuple)) or len(results) != 1:
            raise DetectorCacheError("detector must return exactly one result")
        self._cache.record(
            self._frame_index,
            results[0],
            original_frame_dimensions=self._frame_dimensions,
        )
        self.invocations += 1
        return results

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


class ReplayDetector:
    """Serve cached results without holding or invoking a detector model."""

    detector_cache_mode = "replay"

    def __init__(self, cache: DetectorEvidenceCache) -> None:
        self._cache = cache
        self._frame_index: int | None = None
        self._frame_dimensions: tuple[int, int] | None = None
        self.invocations = 0

    def set_frame_context(
        self,
        frame_index: int,
        original_frame_dimensions: tuple[int, int],
    ) -> None:
        dimensions = tuple(int(value) for value in original_frame_dimensions)
        try:
            cached_dimensions = self._cache.frames[int(frame_index)][
                "original_frame_dimensions"
            ]
        except KeyError as exc:
            raise DetectorCacheError(
                f"no detector evidence for frame_index={frame_index}"
            ) from exc
        if dimensions != cached_dimensions:
            raise DetectorCacheError(
                "replay frame dimensions differ from cached dimensions"
            )
        self._frame_index = int(frame_index)
        self._frame_dimensions = dimensions

    def predict(self, *args: Any, **kwargs: Any) -> list[_CachedResult]:
        del args, kwargs
        return self._replay()

    def track(self, *args: Any, **kwargs: Any) -> list[_CachedResult]:
        del args, kwargs
        return self._replay()

    def _replay(self) -> list[_CachedResult]:
        if self._frame_index is None or self._frame_dimensions is None:
            raise DetectorCacheError(
                "set_frame_context must be called before replay"
            )
        result = self._cache.build_result(self._frame_index)
        self.invocations += 1
        return [result]

    def to(self, *args: Any, **kwargs: Any) -> ReplayDetector:
        del args, kwargs
        return self

    def fuse(self, *args: Any, **kwargs: Any) -> ReplayDetector:
        del args, kwargs
        return self


__all__ = [
    "DETECTOR_CACHE_SCHEMA_VERSION",
    "DetectorCacheError",
    "DetectorCacheIdentity",
    "DetectorEvidenceCache",
    "RecordingDetector",
    "ReplayDetector",
]
