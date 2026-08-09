"""Focused contracts for the PRE-S1 inner-only RGB binding layer."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.training import pre_s1_rgb_binding as rgb_binding
from pig_behavior.classification_v2.training.pre_s1_rgb_binding import (
    RgbBindingError,
    audit_inner_rgb_binding,
    build_rgb_source_integrity_evidence,
    materialize_inner_rgb_binding,
    resolve_execution_rgb_binding,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _requested_roles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["train-window", "validation-window"],
            "primary_s1_role": ["train", "validation"],
        }
    )


def _provenance_hashes() -> dict[str, str]:
    return {
        "authority": "a" * 64,
        "primary_eligibility": "b" * 64,
        "event_weight": "c" * 64,
        "split_role": "d" * 64,
    }


def _source_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "reviewed_rgb_v1"
    context_dir = root / "image_context_v2"
    cache_dir = root / "actor_rgb_64_full"
    context_dir.mkdir(parents=True)
    cache_dir.mkdir()
    windows: list[dict[str, object]] = []
    frames: list[dict[str, object]] = []
    index: list[dict[str, object]] = []
    cache_rows: list[np.ndarray] = []
    for window_id, video_key, actor, prefix in (
        ("train-window", "video-train", "actor-train", "train"),
        ("validation-window", "video-validation", "actor-validation", "validation"),
        ("outer-window", "video-outer", "actor-outer", "outer"),
    ):
        context_ids = [f"{prefix}-context-{index}" for index in range(6)]
        windows.append(
            {
                "window_id": window_id,
                "source_type": "cvat_tracking_xml",
                "object_track_key": actor,
                "window_length_frames": 6,
                "window_start_frame": 0,
                "window_end_frame": 5,
                "selected_frame_indices": "|".join(str(index) for index in range(6)),
                "view_type": "T6_contiguous",
                "window_valid_for_main_train": True,
                "lineage_scope": "reviewed",
                "human_review_complete": True,
                "dataset_id": "dataset-a",
                "video_key": video_key,
                "pig_id": actor,
                "track_id": actor,
                "expected_frame_indices": "|".join(str(index) for index in range(6)),
                "scene_frame_uid_sequence": ";;".join(
                    f"{video_key}:{index}" for index in range(6)
                ),
                "frame_uid_sequence": ";;".join(
                    f"{video_key}:{actor}:{index}" for index in range(6)
                ),
                "image_context_id_sequence": ";;".join(context_ids),
                "observed_image_context_rows": 6,
                "loadable_image_context_rows": 6,
                "missing_image_context_slots": 0,
                "window_image_context_complete": True,
            }
        )
        for frame_index, context_id in enumerate(context_ids):
            frames.append(
                {
                    "identifier_schema_version": "frame_object_v2",
                    "scene_frame_uid": f"{video_key}:{frame_index}",
                    "frame_uid": f"{video_key}:{actor}:{frame_index}",
                    "source_type": "cvat_tracking_xml",
                    "dataset_id": "dataset-a",
                    "video_key": video_key,
                    "source_video_key": video_key,
                    "object_track_key": actor,
                    "pig_id": actor,
                    "track_id": actor,
                    "frame_index": frame_index,
                    "temporal_unit_key": f"native-{prefix}",
                    "image_width": 64,
                    "image_height": 64,
                    "x1": 0,
                    "y1": 0,
                    "x2": 64,
                    "y2": 64,
                    "bbox_valid": True,
                    "lineage_scope": "reviewed",
                    "human_review_complete": True,
                    "image_context_id": context_id,
                    "image_context_source": "cvat_video_bbox",
                    "resolved_media_path": f"C:/source/{video_key}.mp4",
                    "resolved_media_exists": True,
                    "bbox_context_valid": True,
                    "full_frame_context_available": True,
                    "partner_context_available": False,
                    "image_context_loadable": True,
                    "image_context_error": "",
                }
            )
            index.append(
                {
                    "image_context_id": context_id,
                    "packed_row": len(cache_rows),
                    "lineage_scope": "reviewed",
                    "human_review_complete": True,
                }
            )
            cache_rows.append(np.full((64, 64, 3), len(cache_rows), dtype=np.uint8))
    frame_path = context_dir / "image_frame_context_manifest.csv"
    window_path = context_dir / "image_window_context_manifest.csv"
    index_path = cache_dir / "packed_image_cache_index.csv"
    manifest_path = cache_dir / "manifest.csv"
    pd.DataFrame(frames).to_csv(frame_path, index=False)
    pd.DataFrame(windows).to_csv(window_path, index=False)
    pd.DataFrame(index).to_csv(index_path, index=False)
    pd.DataFrame(index).to_csv(manifest_path, index=False)
    np.save(cache_dir / "packed_rgb_64_letterbox.npy", np.stack(cache_rows))
    (cache_dir / "cache_audit.json").write_text("{}", encoding="utf-8")
    (cache_dir / "packed_image_cache_audit.json").write_text("{}", encoding="utf-8")
    parity = {
        "source_authorities": {
            "rgb_cache_index_sha256": _sha256(index_path),
            "rgb_cache_manifest_sha256": _sha256(manifest_path),
        }
    }
    return root, parity


def _materialize(tmp_path: Path) -> tuple[dict[str, object], pd.DataFrame]:
    source, parity = _source_fixture(tmp_path)
    requested = _requested_roles()
    source_integrity = build_rgb_source_integrity_evidence(
        rgb_source_root=source,
        output_path=tmp_path / "source_integrity.json",
        input_parity_evidence=parity,
    )
    report = materialize_inner_rgb_binding(
        output_dir=tmp_path / "binding",
        rgb_source_root=source,
        requested_roles=requested,
        authority_sha256="a" * 64,
        provenance_hashes=_provenance_hashes(),
        expected_train_windows=1,
        expected_validation_windows=1,
        input_parity_evidence=parity,
        source_integrity_evidence=json.loads(
            Path(str(source_integrity["path"])).read_text(encoding="utf-8")
        ),
    )
    return report, requested


def test_materialized_binding_is_inner_only_hash_bound_and_portable(tmp_path: Path) -> None:
    report, requested = _materialize(tmp_path)
    coverage = report["coverage"]
    assert coverage == {
        "train_windows_bound": 1,
        "validation_windows_bound": 1,
        "missing_windows": 0,
        "duplicate_windows": 0,
        "bad_sequence_length": 0,
        "role_violations": 0,
        "cross_video_violations": 0,
        "unexpected_windows": 0,
        "temporal_violations": 0,
        "incomplete_window_violations": 0,
        "missing_context_ids": 0,
        "missing_packed_index_ids": 0,
        "duplicate_context_ids": 0,
        "duplicate_packed_index_ids": 0,
        "sequence_order_violations": 0,
        "actor_identity_violations": 0,
        "media_reference_violations": 0,
    }
    binding_root = Path(str(report["scientific_binding_path"])).parent
    scientific = json.loads(
        Path(str(report["scientific_binding_path"])).read_text(encoding="utf-8")
    )
    assert str(tmp_path / "reviewed_rgb_v1") not in json.dumps(scientific)
    inner_windows = pd.read_csv(binding_root / "inner_window_context.csv")
    assert set(inner_windows["window_id"]) == {"train-window", "validation-window"}
    assert set(inner_windows["calibration_role"]) == {"train", "validation"}
    resolved = resolve_execution_rgb_binding(
        data_bindings_path=Path(str(report["data_bindings_path"])),
        requested_roles=requested,
        authority_sha256="a" * 64,
        provenance_hashes=_provenance_hashes(),
    )
    assert resolved.audit["valid"] is True
    assert resolved.coverage["missing_windows"] == 0
    assert resolved.window_context_path == binding_root / "inner_window_context.csv"


def test_binding_audit_rejects_outer_bad_sequence_and_cross_video(tmp_path: Path) -> None:
    report, requested = _materialize(tmp_path)
    binding_root = Path(str(report["scientific_binding_path"])).parent
    windows = pd.read_csv(binding_root / "inner_window_context.csv")
    frames = pd.read_csv(binding_root / "inner_frame_context.csv")
    index = pd.read_csv(binding_root / "inner_packed_image_cache_index.csv")

    outer = windows.copy()
    outer.loc[0, "calibration_role"] = "outer"
    assert audit_inner_rgb_binding(
        windows=outer,
        frames=frames,
        packed_index=index,
        requested_roles=requested,
    )["coverage"]["role_violations"] > 0

    malformed = windows.copy()
    malformed.loc[0, "image_context_id_sequence"] = "one;;two"
    assert audit_inner_rgb_binding(
        windows=malformed,
        frames=frames,
        packed_index=index,
        requested_roles=requested,
    )["coverage"]["bad_sequence_length"] == 1

    json_selected = windows.copy()
    json_selected.loc[0, "selected_frame_indices"] = "[0,1,2,3,4,5]"
    assert audit_inner_rgb_binding(
        windows=json_selected,
        frames=frames,
        packed_index=index,
        requested_roles=requested,
    )["valid"] is True

    cross_video = frames.copy()
    cross_video.loc[0, "video_key"] = "different-video"
    assert audit_inner_rgb_binding(
        windows=windows,
        frames=cross_video,
        packed_index=index,
        requested_roles=requested,
    )["coverage"]["cross_video_violations"] == 1


def test_binding_audit_bulk_checks_repeated_six_slot_windows(tmp_path: Path) -> None:
    report, requested = _materialize(tmp_path)
    binding_root = Path(str(report["scientific_binding_path"])).parent
    windows = pd.read_csv(binding_root / "inner_window_context.csv")
    frames = pd.read_csv(binding_root / "inner_frame_context.csv")
    index = pd.read_csv(binding_root / "inner_packed_image_cache_index.csv")
    repeats = 256
    expanded = pd.concat([windows.iloc[[0]]] * repeats, ignore_index=True)
    expanded["window_id"] = [f"bulk-window-{value:03d}" for value in range(repeats)]
    expanded["calibration_role"] = "train"
    roles = pd.DataFrame(
        {
            "window_id": expanded["window_id"],
            "primary_s1_role": "train",
        }
    )
    audit = audit_inner_rgb_binding(
        windows=expanded,
        frames=frames,
        packed_index=index,
        requested_roles=roles,
    )
    assert audit["valid"] is True
    assert audit["coverage"]["train_windows_bound"] == repeats


def test_binding_rejects_one_byte_packed_cache_change(tmp_path: Path) -> None:
    report, requested = _materialize(tmp_path)
    bindings = json.loads(Path(str(report["data_bindings_path"])).read_text(encoding="utf-8"))
    packed_cache = Path(bindings["execution_path_realization"]["packed_cache_path"])
    with packed_cache.open("r+b") as handle:
        handle.seek(-1, 2)
        original = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([original[0] ^ 1]))
    cache_stat = packed_cache.stat()
    os.utime(
        packed_cache,
        ns=(cache_stat.st_atime_ns, cache_stat.st_mtime_ns + 1_000_000_000),
    )
    with pytest.raises(RgbBindingError, match="packed RGB cache hash mismatch"):
        resolve_execution_rgb_binding(
            data_bindings_path=Path(str(report["data_bindings_path"])),
            requested_roles=requested,
            authority_sha256="a" * 64,
            provenance_hashes=_provenance_hashes(),
        )


def test_attested_binding_reuses_current_verified_cache_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, requested = _materialize(tmp_path)
    bindings = json.loads(Path(str(report["data_bindings_path"])).read_text(encoding="utf-8"))
    packed_cache = Path(bindings["execution_path_realization"]["packed_cache_path"])
    original_sha256 = rgb_binding._sha256_file

    def forbid_rehash(path: Path) -> str:
        if Path(path).resolve() == packed_cache.resolve():
            raise AssertionError("packed cache was rehashed despite matching attestation")
        return original_sha256(path)

    monkeypatch.setattr(rgb_binding, "_sha256_file", forbid_rehash)
    resolved = resolve_execution_rgb_binding(
        data_bindings_path=Path(str(report["data_bindings_path"])),
        requested_roles=requested,
        authority_sha256="a" * 64,
        provenance_hashes=_provenance_hashes(),
    )
    assert resolved.hashes["rgb_packed_cache"] == bindings[
        "execution_path_realization"
    ]["packed_cache_identity_attestation"]["sha256"]


def test_source_integrity_evidence_rejects_file_identity_drift(tmp_path: Path) -> None:
    source, parity = _source_fixture(tmp_path)
    evidence = build_rgb_source_integrity_evidence(
        rgb_source_root=source,
        output_path=tmp_path / "source_integrity.json",
        input_parity_evidence=parity,
    )
    cache = source / "actor_rgb_64_full" / "packed_rgb_64_letterbox.npy"
    cache_stat = cache.stat()
    os.utime(
        cache,
        ns=(cache_stat.st_atime_ns, cache_stat.st_mtime_ns + 1_000_000_000),
    )
    with pytest.raises(RgbBindingError, match="source integrity evidence is stale"):
        materialize_inner_rgb_binding(
            output_dir=tmp_path / "binding",
            rgb_source_root=source,
            requested_roles=_requested_roles(),
            authority_sha256="a" * 64,
            provenance_hashes=_provenance_hashes(),
            expected_train_windows=1,
            expected_validation_windows=1,
            input_parity_evidence=parity,
            source_integrity_evidence=json.loads(
                Path(str(evidence["path"])).read_text(encoding="utf-8")
            ),
        )
