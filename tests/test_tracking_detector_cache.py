"""Focused tests for generic fail-closed detector cache replay."""

from __future__ import annotations

import builtins
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pig_behavior.tracking.detector_cache import (
    DETECTOR_CACHE_SCHEMA_VERSION,
    DetectorCacheError,
    DetectorCacheIdentity,
    DetectorEvidenceCache,
    RecordingDetector,
    ReplayDetector,
)
from pig_behavior.tracking.runner import _resolve_detector

HEX_A = "a" * 64
HEX_B = "b" * 64
CODE_SHA = "c" * 40


def _identity(**overrides: object) -> DetectorCacheIdentity:
    values: dict[str, object] = {
        "video_key": "video-a",
        "source_video_sha256": HEX_A,
        "detector_weight_sha256": HEX_B,
        "detector_semantic_config_sha256": "d" * 64,
        "producer_code_sha": CODE_SHA,
        "creation_authority": "bounded-fixture",
    }
    values.update(overrides)
    return DetectorCacheIdentity(**values)


def _result() -> SimpleNamespace:
    boxes = SimpleNamespace(
        xyxy=np.asarray([[1.0, 2.0, 11.0, 12.0]], dtype=np.float32),
        conf=np.asarray([0.75], dtype=np.float32),
        cls=np.asarray([0.0], dtype=np.float32),
        id=np.asarray([4.0], dtype=np.float32),
    )
    masks = SimpleNamespace(
        data=np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    )
    return SimpleNamespace(boxes=boxes, masks=masks, names={0: "pig"})


def _cache() -> DetectorEvidenceCache:
    cache = DetectorEvidenceCache(identity=_identity())
    cache.record(
        3,
        _result(),
        original_frame_dimensions=(720, 1280),
    )
    cache.record(
        5,
        _result(),
        original_frame_dimensions=(720, 1280),
    )
    return cache


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_sidecar_hash(path: Path) -> None:
    sidecar_path = path.with_suffix(".sha256.json")
    sidecar = {
        "schema_version": DETECTOR_CACHE_SCHEMA_VERSION,
        "cache_artifact_sha256": _sha256(path),
    }
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_cache_serialization_and_artifact_hash_are_deterministic(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"

    first_sha = _cache().save(first)
    second_sha = _cache().save(second)

    assert first.read_bytes() == second.read_bytes()
    assert first_sha == second_sha == _sha256(first)


def test_record_load_replay_preserves_population_and_frame_indices(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evidence.npz"
    expected = _cache()
    artifact_sha = expected.save(path)

    actual = DetectorEvidenceCache.load(
        path,
        expected_identity=_identity(),
    )
    replay = ReplayDetector(actual)
    replay.set_frame_context(3, (720, 1280))
    result = replay.predict(source=np.zeros((1,), dtype=np.uint8))[0]

    assert list(actual.frames) == [3, 5]
    assert actual.cache_artifact_sha256 == artifact_sha
    assert replay.invocations == 1
    assert result.orig_shape == (720, 1280)
    assert result.names == {0: "pig"}
    np.testing.assert_array_equal(
        result.boxes.xyxy,
        np.asarray([[1.0, 2.0, 11.0, 12.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        result.masks.data,
        np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
    )


def test_recording_and_replay_population_parity_without_reinference() -> None:
    class FixtureDetector:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, **kwargs: object) -> list[SimpleNamespace]:
            assert "source" in kwargs
            self.calls += 1
            return [_result()]

    fixture = FixtureDetector()
    cache = DetectorEvidenceCache(identity=_identity())
    recorder = RecordingDetector(fixture, cache)
    recorder.set_frame_context(8, (100, 200))
    direct = recorder.predict(source=np.zeros((100, 200, 3), dtype=np.uint8))
    replay = ReplayDetector(cache)
    replay.set_frame_context(8, (100, 200))
    replayed = replay.predict()

    assert fixture.calls == 1
    assert recorder.invocations == replay.invocations == 1
    np.testing.assert_array_equal(
        direct[0].boxes.xyxy,
        replayed[0].boxes.xyxy,
    )
    assert not hasattr(replay, "_model")


def test_duplicate_or_non_monotonic_frame_is_rejected() -> None:
    cache = DetectorEvidenceCache(identity=_identity())
    cache.record(4, _result(), original_frame_dimensions=(100, 200))

    with pytest.raises(DetectorCacheError, match="strictly increasing"):
        cache.record(4, _result(), original_frame_dimensions=(100, 200))
    with pytest.raises(DetectorCacheError, match="strictly increasing"):
        cache.record(2, _result(), original_frame_dimensions=(100, 200))


def test_load_rejects_corrupted_artifact_hash(tmp_path: Path) -> None:
    path = tmp_path / "evidence.npz"
    _cache().save(path)
    path.write_bytes(path.read_bytes() + b"corruption")

    with pytest.raises(DetectorCacheError, match="artifact hash"):
        DetectorEvidenceCache.load(path, expected_identity=_identity())


def test_load_rejects_unsupported_schema(tmp_path: Path) -> None:
    path = tmp_path / "evidence.npz"
    _cache().save(path)
    sidecar_path = path.with_suffix(".sha256.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["schema_version"] = "unsupported.v0"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(DetectorCacheError, match="schema version"):
        DetectorEvidenceCache.load(path, expected_identity=_identity())


@pytest.mark.parametrize(
    "changed_identity",
    [
        replace(_identity(), source_video_sha256="e" * 64),
        replace(_identity(), detector_weight_sha256="e" * 64),
        replace(_identity(), detector_semantic_config_sha256="e" * 64),
        replace(_identity(), video_key="video-b"),
    ],
)
def test_load_rejects_identity_mismatch(
    tmp_path: Path,
    changed_identity: DetectorCacheIdentity,
) -> None:
    path = tmp_path / "evidence.npz"
    _cache().save(path)

    with pytest.raises(DetectorCacheError, match="identity mismatch"):
        DetectorEvidenceCache.load(
            path,
            expected_identity=changed_identity,
        )


def test_load_rejects_malformed_detection_arrays(tmp_path: Path) -> None:
    path = tmp_path / "evidence.npz"
    _cache().save(path)
    with np.load(path, allow_pickle=False) as stored:
        payload = {key: np.asarray(stored[key]) for key in stored.files}
    payload["f3__xyxy"] = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    np.savez_compressed(path, **payload)
    _rewrite_sidecar_hash(path)

    with pytest.raises(DetectorCacheError, match="malformed xyxy"):
        DetectorEvidenceCache.load(path, expected_identity=_identity())


def test_required_masks_and_appearance_fail_closed() -> None:
    no_masks = _result()
    no_masks.masks = None
    cache = DetectorEvidenceCache(
        identity=_identity(
            requires_masks=True,
            requires_appearance_descriptors=True,
        )
    )

    with pytest.raises(DetectorCacheError, match="masks are required"):
        cache.record(
            0,
            no_masks,
            original_frame_dimensions=(100, 200),
        )


def test_replay_rejects_frame_dimension_mismatch() -> None:
    replay = ReplayDetector(_cache())

    with pytest.raises(DetectorCacheError, match="dimensions differ"):
        replay.set_frame_context(3, (721, 1280))


def test_injected_detector_does_not_import_ultralytics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "ultralytics":
            raise AssertionError("injected detector must not import Ultralytics")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert _resolve_detector(sentinel, Path("weights.pt")) is sentinel


def test_default_detector_path_constructs_original_yolo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[str] = []
    sentinel = object()

    def fake_yolo(weights: str) -> object:
        created.append(weights)
        return sentinel

    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        SimpleNamespace(YOLO=fake_yolo),
    )

    assert _resolve_detector(None, Path("weights.pt")) is sentinel
    assert created == ["weights.pt"]


def test_bounded_cache_validation_root_contains_no_mp4(tmp_path: Path) -> None:
    _cache().save(tmp_path / "evidence.npz")

    assert not list(tmp_path.rglob("*.mp4"))
