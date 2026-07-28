"""Synthetic gates for odd-only full-frame detector cache completion."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pig_behavior.tracking.detector_cache import (
    DetectorCacheIdentity,
    DetectorEvidenceCache,
)

REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "scripts"
    / "tracking"
    / "complete_full_frame_detector_cache.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("full_frame_cache_tool", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _identity(authority: str) -> DetectorCacheIdentity:
    return DetectorCacheIdentity(
        video_key="video-a",
        source_video_sha256="a" * 64,
        detector_weight_sha256="b" * 64,
        detector_semantic_config_sha256="c" * 64,
        producer_code_sha="d" * 40,
        creation_authority=authority,
    )


def _result(value: float = 1.0) -> SimpleNamespace:
    boxes = SimpleNamespace(
        xyxy=np.asarray(
            [[value, value, value + 10.0, value + 12.0]],
            dtype=np.float32,
        ),
        conf=np.asarray([0.75], dtype=np.float32),
        cls=np.asarray([0.0], dtype=np.float32),
        id=None,
    )
    return SimpleNamespace(boxes=boxes, masks=None, names={0: "pig"})


def _cache(
    indices: tuple[int, ...],
    *,
    authority: str,
    value: float,
) -> DetectorEvidenceCache:
    cache = DetectorEvidenceCache(identity=_identity(authority))
    for frame_index in indices:
        cache.record(
            frame_index,
            _result(value + frame_index / 10000.0),
            original_frame_dimensions=(720, 1280),
        )
    return cache


def _even_cache():
    module = _load_module()
    return _cache(
        module.expected_even_indices(),
        authority="even-fixture",
        value=1.0,
    )


def _odd_cache():
    module = _load_module()
    return _cache(
        module.expected_odd_indices(),
        authority="odd-fixture",
        value=2.0,
    )


def test_missing_frame_manifest_is_exact_odd_complement() -> None:
    module = _load_module()

    missing = module.derive_missing_indices(module.expected_even_indices())

    assert missing == module.expected_odd_indices()
    assert len(missing) == 900
    assert missing[0] == 1
    assert missing[-1] == 1799
    assert all(index % 2 == 1 for index in missing)


@pytest.mark.parametrize(
    "invalid",
    [
        tuple(range(1800)),
        tuple(range(0, 1798, 2)),
        tuple(range(1, 1800, 2)),
        (0, 0, *range(2, 1800, 2)),
    ],
)
def test_missing_frame_derivation_rejects_wrong_existing_population(
    invalid: tuple[int, ...],
) -> None:
    module = _load_module()

    with pytest.raises(
        module.FullFrameCacheError,
        match="exact even subset",
    ):
        module.derive_missing_indices(invalid)


def test_combine_caches_preserves_even_and_odd_evidence() -> None:
    module = _load_module()
    even = _even_cache()
    odd = _odd_cache()

    full = module.combine_caches(
        even,
        odd,
        _identity("full-fixture"),
    )

    assert tuple(full.frames) == tuple(range(1800))
    assert module.assert_even_subset_parity(even, full) == (
        module.cache_content_hash(even)
    )
    assert module.cache_content_hash(
        full,
        module.expected_odd_indices(),
    ) == module.cache_content_hash(odd)


def test_combined_cache_does_not_alias_source_arrays() -> None:
    module = _load_module()
    even = _even_cache()
    odd = _odd_cache()
    full = module.combine_caches(even, odd, _identity("full-fixture"))

    full.frames[0]["xyxy"][0, 0] = 99.0
    full.frames[1]["xyxy"][0, 0] = 98.0

    assert even.frames[0]["xyxy"][0, 0] != 99.0
    assert odd.frames[1]["xyxy"][0, 0] != 98.0


def test_even_subset_parity_detects_one_value_change() -> None:
    module = _load_module()
    even = _even_cache()
    full = module.combine_caches(
        even,
        _odd_cache(),
        _identity("full-fixture"),
    )
    full.frames[100]["conf"][0] = 0.5

    with pytest.raises(
        module.FullFrameCacheError,
        match="even detector evidence changed",
    ):
        module.assert_even_subset_parity(even, full)


def test_combine_rejects_non_odd_completion_population() -> None:
    module = _load_module()
    wrong = _cache(
        module.expected_even_indices(),
        authority="wrong-fixture",
        value=2.0,
    )

    with pytest.raises(
        module.FullFrameCacheError,
        match="odd cache coverage",
    ):
        module.combine_caches(
            _even_cache(),
            wrong,
            _identity("full-fixture"),
        )


def test_content_hash_canonicalizes_record_order() -> None:
    module = _load_module()
    cache = _even_cache()
    expected = module.cache_content_hash(cache)
    cache.frames = dict(reversed(tuple(cache.frames.items())))

    assert module.cache_content_hash(cache) == expected


def test_content_hash_changes_with_detection_content() -> None:
    module = _load_module()
    cache = _even_cache()
    expected = module.cache_content_hash(cache)
    cache.frames[0]["xyxy"][0, 0] += 1.0

    assert module.cache_content_hash(cache) != expected


def test_full_cache_serialization_and_replay_are_deterministic(
    tmp_path: Path,
) -> None:
    module = _load_module()
    full = module.combine_caches(
        _even_cache(),
        _odd_cache(),
        _identity("full-fixture"),
    )
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"

    assert full.save(first) == full.save(second)
    assert first.read_bytes() == second.read_bytes()
    loaded = DetectorEvidenceCache.load(
        first,
        expected_identity=_identity("full-fixture"),
    )
    assert module.replay_cache(loaded) == 1800
    assert module.cache_content_hash(loaded) == module.cache_content_hash(full)


def test_record_odd_video_rejects_even_manifest_before_inference(
    tmp_path: Path,
) -> None:
    module = _load_module()
    video = SimpleNamespace(
        video_key="video-a",
        video_path=tmp_path / "unused.mp4",
        video_sha256="a" * 64,
        frame_count=1800,
        width=1280,
        height=720,
    )

    with pytest.raises(
        module.FullFrameCacheError,
        match="odd manifest changed",
    ):
        module.record_odd_video(
            video,
            SimpleNamespace(),
            object(),
            tmp_path,
            "d" * 40,
            module.expected_even_indices(),
            0,
        )


def test_preflight_refuses_existing_output_root(tmp_path: Path) -> None:
    module = _load_module()
    output_root = tmp_path / "existing"
    output_root.mkdir()

    with pytest.raises(
        module.FullFrameCacheError,
        match="refusing existing output root",
    ):
        module.preflight(
            tmp_path,
            tmp_path / "lineage.json",
            tmp_path / "r0",
            output_root,
        )


def test_profile_consumption_contract_declares_cadence_fairness() -> None:
    module = _load_module()
    contract = module.profile_consumption_contract()

    assert contract["B0_CACHE_CONSUMPTION"] == "FULL_FRAME_CACHE"
    assert contract["B1_CACHE_CONSUMPTION"] == "FULL_FRAME_CACHE"
    assert contract["R0_CACHE_CONSUMPTION"] == "FROZEN_EVEN_SUBSET"
    assert contract["R1_CACHE_CONSUMPTION"] == "FROZEN_EVEN_SUBSET"
    assert contract["B1_MINUS_B0_DETECTOR_CADENCE_MATCHED"] == "YES"
    assert contract["R1_MINUS_R0_DETECTOR_CADENCE_MATCHED"] == "YES"
    assert contract["PURE_ASSOCIATION_CORE_EFFECT_CLAIM_AUTHORIZED"] == "NO"


def test_artifact_inventory_is_path_ordered_and_hash_bound(
    tmp_path: Path,
) -> None:
    module = _load_module()
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")

    inventory = module.artifact_inventory(
        tmp_path,
        excluded_names=set(),
    )

    assert [row["relative_path"] for row in inventory] == ["a.txt", "b.txt"]
    assert all(len(row["sha256"]) == 64 for row in inventory)


def test_atomic_json_retries_transient_windows_reader_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    destination = tmp_path / "state.json"
    original_replace = module.os.replace
    attempts = 0

    def flaky_replace(source: Path, target: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("transient reader lock")
        original_replace(source, target)

    monkeypatch.setattr(module.os, "replace", flaky_replace)

    module.atomic_write_json(destination, {"status": "PASS"})

    assert module.load_json(destination) == {"status": "PASS"}
    assert attempts == 3


def test_prior_attempt_audit_counts_one_failed_heartbeat_batch(
    tmp_path: Path,
) -> None:
    module = _load_module()
    producer = "e" * 40
    video = SimpleNamespace(
        video_key="video-a",
        video_sha256="a" * 64,
    )
    prior_root = tmp_path / "prior"
    cache = DetectorEvidenceCache(
        identity=module.generated_identity(
            video,
            producer,
            module.ODD_CREATION_AUTHORITY,
        )
    )
    for frame_index in module.expected_odd_indices():
        cache.record(
            frame_index,
            _result(),
            original_frame_dimensions=(720, 1280),
        )
    cache_path = module.odd_cache_path(prior_root, video.video_key)
    cache_sha = cache.save(cache_path)
    module.atomic_write_json(
        prior_root / "FULL_FRAME_CACHE_PREFLIGHT.json",
        {"producer_code_sha": producer},
    )
    module.atomic_write_json(
        prior_root / "FULL_FRAME_CACHE_RUN_STATE.json",
        {
            "status": "RUNNING",
            "completed_odd_frames": 1250,
        },
    )
    (prior_root / "generation.stderr.log").write_text(
        "PermissionError in record_odd_video\n",
        encoding="utf-8",
    )
    module.atomic_write_json(
        module.checkpoint_path(
            prior_root,
            "odd_inference",
            video.video_key,
        ),
        {
            "status": "COMMITTED",
            "video_key": video.video_key,
            "detector_inference_calls": 900,
            "cache_artifact_sha256": cache_sha,
            "canonical_content_hash": module.cache_content_hash(cache),
        },
    )

    authority, imported = module.inspect_prior_attempt(
        prior_root,
        [video],
    )

    assert authority is not None
    assert authority["prior_attempt_physical_odd_calls"] == 1300
    assert authority["prior_committed_unique_odd_records"] == 900
    assert authority["prior_uncommitted_retry_calls"] == 400
    assert len(imported) == 1


def test_completion_tool_has_no_tracker_evaluator_or_mp4_path() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert ".track(" not in source
    assert "evaluate_tracking" not in source
    assert "write_output_video=True" not in source
    assert "VideoWriter" not in source
