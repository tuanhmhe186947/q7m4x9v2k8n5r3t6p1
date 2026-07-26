from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from pig_behavior.classification_v2 import lineage_config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_source_bundle(
    tmp_path: Path,
) -> tuple[Path, dict[str, object], Path]:
    root = tmp_path / "repository"
    crops = root / "crops"
    videos = root / "videos"
    cvat = root / "cvat"
    run_root = tmp_path / "run"
    crops.mkdir(parents=True)
    videos.mkdir()
    cvat.mkdir()
    (root / "legacy.csv").write_text("id\n1\n2\n", encoding="utf-8")
    (root / "audit.json").write_text(
        json.dumps({"status": "PASS"}),
        encoding="utf-8",
    )
    (root / "roi.json").write_text('{"roi":1}', encoding="utf-8")
    (root / "pen.png").write_bytes(b"synthetic-pen-mask")
    (crops / "crop-a.jpg").write_bytes(b"crop-a")
    (crops / "crop-b.jpg").write_bytes(b"crop-b")
    (videos / "video.mp4").write_bytes(b"synthetic-video")
    (cvat / "a.xml").write_text(
        "<annotations><track><box/><box/></track></annotations>",
        encoding="utf-8",
    )
    config: dict[str, object] = {
        "run_root_env": "CLASSIFICATION_V2_TEST_RUN_ROOT",
        "run_root_default": str(run_root),
        "source": {
            "legacy_export": "legacy.csv",
            "legacy_completion_audit": "audit.json",
            "legacy_crop_root": "crops",
            "cvat_behavior_root": "cvat",
            "cvat_behavior_xml": ["cvat/a.xml"],
            "roi": "roi.json",
            "video_root": "videos",
            "pen_mask": "pen.png",
            "bundle_id": "synthetic-source-bundle-v1",
            "expected_bundle_fingerprint": "pending",
            "expected_legacy_sha256": "pending",
            "expected_legacy_rows": 2,
            "expected_legacy_crop_files": 2,
            "expected_cvat_xml_count": 1,
            "expected_cvat_box_rows": 2,
            "expected_mixed_rows": 4,
            "expected_roi_sha256": "pending",
            "expected_pen_mask_sha256": "pending",
            "expected_completion_audit_sha256": "pending",
            "expected_cvat_xml_fingerprint": "pending",
            "expected_crop_fingerprint": "pending",
            "expected_video_fingerprint": "pending",
        },
    }
    config_path = root / "lineage.yaml"
    config_path.write_text("synthetic: true\n", encoding="utf-8")
    probe = lineage_config.source_bundle_report(
        root,
        config,
        verification_mode="full",
        config_path=config_path,
    )
    source = config["source"]
    assert isinstance(source, dict)
    source.update(
        {
            "expected_bundle_fingerprint": probe["bundle_fingerprint"],
            "expected_legacy_sha256": probe["legacy_csv_sha256"],
            "expected_roi_sha256": probe["roi_sha256"],
            "expected_pen_mask_sha256": probe["pen_mask_sha256"],
            "expected_completion_audit_sha256": probe[
                "completion_audit_sha256"
            ],
            "expected_cvat_xml_fingerprint": probe[
                "cvat_xml_fingerprint"
            ],
            "expected_crop_fingerprint": probe["crop_fingerprint"],
            "expected_video_fingerprint": probe["video_fingerprint"],
        }
    )
    return root, config, config_path


def _write_valid_cache(
    root: Path,
    config: dict[str, object],
    config_path: Path,
) -> dict[str, object]:
    report = lineage_config.source_bundle_report(
        root,
        config,
        verification_mode="full",
        write_cache=True,
        config_path=config_path,
    )
    assert report["valid"] is True
    assert report["verification_cache_written"] is True
    return report


def test_full_verification_writes_atomic_bound_cache(tmp_path: Path) -> None:
    root, config, config_path = _synthetic_source_bundle(tmp_path)
    source_files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path != config_path
    ]
    before = {str(path): _sha256(path) for path in source_files}

    report = _write_valid_cache(root, config, config_path)

    cache_path = Path(str(report["verification_cache_path"]))
    assert cache_path.is_file()
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    assert cache["schema_version"] == (
        lineage_config.SOURCE_VERIFICATION_CACHE_SCHEMA
    )
    assert cache["verified_report"]["bundle_fingerprint"] == (
        report["bundle_fingerprint"]
    )
    assert not list(cache_path.parent.glob("*.tmp"))
    assert before == {str(path): _sha256(path) for path in source_files}


def test_fast_verification_reuses_exact_heavy_fingerprints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config, config_path = _synthetic_source_bundle(tmp_path)
    full = _write_valid_cache(root, config, config_path)
    original_sha256 = lineage_config._sha256

    def reject_heavy_payload_hash(path: Path) -> str:
        if root / "crops" in path.parents or root / "videos" in path.parents:
            pytest.fail(f"fast verification hashed heavy payload: {path}")
        return original_sha256(path)

    monkeypatch.setattr(lineage_config, "_sha256", reject_heavy_payload_hash)
    fast = lineage_config.source_bundle_report(
        root,
        config,
        verification_mode="fast",
        config_path=config_path,
    )

    assert fast["valid"] is True
    assert fast["verification_cache_used"] is True
    assert fast["bundle_fingerprint"] == full["bundle_fingerprint"]
    assert fast["crop_fingerprint"] == full["crop_fingerprint"]
    assert fast["video_fingerprint"] == full["video_fingerprint"]


@pytest.mark.parametrize("change", ("path", "size", "mtime"))
def test_fast_verification_fails_closed_on_heavy_metadata_change(
    tmp_path: Path,
    change: str,
) -> None:
    root, config, config_path = _synthetic_source_bundle(tmp_path)
    _write_valid_cache(root, config, config_path)
    crop = root / "crops" / "crop-a.jpg"
    if change == "path":
        crop.rename(crop.with_name("crop-renamed.jpg"))
    elif change == "size":
        crop.write_bytes(crop.read_bytes() + b"x")
    else:
        stat = crop.stat()
        os.utime(
            crop,
            ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000),
        )

    report = lineage_config.source_bundle_report(
        root,
        config,
        verification_mode="fast",
        config_path=config_path,
    )

    assert report["valid"] is False
    assert "SOURCE_VERIFICATION_CACHE_CROP_METADATA_MISMATCH" in (
        report["verification_errors"]
    )


def test_full_verification_still_detects_source_hash_mismatch(
    tmp_path: Path,
) -> None:
    root, config, config_path = _synthetic_source_bundle(tmp_path)
    _write_valid_cache(root, config, config_path)
    (root / "roi.json").write_text('{"roi":2}', encoding="utf-8")

    report = lineage_config.source_bundle_report(
        root,
        config,
        verification_mode="full",
        config_path=config_path,
    )

    assert report["valid"] is False
    assert report["verification_cache_written"] is False


def test_fast_verification_rejects_config_and_bundle_drift(
    tmp_path: Path,
) -> None:
    root, config, config_path = _synthetic_source_bundle(tmp_path)
    _write_valid_cache(root, config, config_path)
    changed = copy.deepcopy(config)
    source = changed["source"]
    assert isinstance(source, dict)
    source["bundle_id"] = "different-bundle"

    report = lineage_config.source_bundle_report(
        root,
        changed,
        verification_mode="fast",
        config_path=config_path,
    )

    assert report["valid"] is False
    assert "SOURCE_VERIFICATION_CACHE_BUNDLE_ID_MISMATCH" in (
        report["verification_errors"]
    )
    assert "SOURCE_VERIFICATION_CACHE_CONFIG_MISMATCH" in (
        report["verification_errors"]
    )


def test_fast_verification_rejects_missing_or_malformed_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config, config_path = _synthetic_source_bundle(tmp_path)
    monkeypatch.setattr(
        lineage_config,
        "_tree_snapshot",
        lambda *_: pytest.fail("invalid cache must fail before tree traversal"),
    )
    missing = lineage_config.source_bundle_report(
        root,
        config,
        verification_mode="fast",
        config_path=config_path,
    )
    assert missing["valid"] is False
    assert missing["verification_errors"] == [
        "SOURCE_VERIFICATION_CACHE_MISSING:RUN_FULL_SOURCE_VERIFY"
    ]

    cache_path = lineage_config.source_verification_cache_path(root, config)
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("{not-json", encoding="utf-8")
    malformed = lineage_config.source_bundle_report(
        root,
        config,
        verification_mode="fast",
        config_path=config_path,
    )
    assert malformed["valid"] is False
    assert malformed["verification_errors"] == [
        "SOURCE_VERIFICATION_CACHE_MALFORMED:RUN_FULL_SOURCE_VERIFY"
    ]
