"""Focused contracts for Stage-1 variable-length RGB bindings."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.training import pre_s1_rgb_binding
from pig_behavior.classification_v2.training import stage1_rgb_binding as binding


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _roles(view: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": [f"{view}-train", f"{view}-validation"],
            "primary_s1_role": ["train", "validation"],
        }
    )


def _provenance() -> dict[str, str]:
    return {
        "s1_authority": "a" * 64,
        "event_weight": "b" * 64,
        "primary_windows": "c" * 64,
        "common_cohort": "d" * 64,
    }


def _source_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "reviewed_rgb_v1"
    context_dir = root / "image_context_v2"
    cache_dir = root / "actor_rgb_64_full"
    context_dir.mkdir(parents=True)
    cache_dir.mkdir()
    windows: list[dict[str, object]] = []
    frames: list[dict[str, object]] = []
    packed_index: list[dict[str, object]] = []
    cache_rows: list[np.ndarray] = []
    for view, length in (("T6", 6), ("T8", 8), ("T12", 12), ("T16", 16)):
        for role in ("train", "validation"):
            window_id = f"{view}-{role}"
            video_key = f"video-{view}-{role}"
            actor = f"actor-{view}-{role}"
            context_ids = [f"{window_id}-context-{index}" for index in range(length)]
            sequence = ";;".join(context_ids)
            frame_sequence = "|".join(str(index) for index in range(length))
            windows.append(
                {
                    "window_id": window_id,
                    "source_type": "cvat_tracking_xml",
                    "object_track_key": actor,
                    "window_length_frames": length,
                    "window_start_frame": 0,
                    "window_end_frame": length - 1,
                    "selected_frame_indices": frame_sequence,
                    "view_type": f"{view}_contiguous",
                    "window_valid_for_main_train": True,
                    "lineage_scope": "reviewed",
                    "human_review_complete": True,
                    "dataset_id": "dataset-a",
                    "video_key": video_key,
                    "pig_id": actor,
                    "track_id": actor,
                    "expected_frame_indices": frame_sequence,
                    "scene_frame_uid_sequence": ";;".join(
                        f"{video_key}:{index}" for index in range(length)
                    ),
                    "frame_uid_sequence": ";;".join(
                        f"{video_key}:{actor}:{index}" for index in range(length)
                    ),
                    "image_context_id_sequence": sequence,
                    "observed_image_context_rows": length,
                    "loadable_image_context_rows": length,
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
                        "temporal_unit_key": f"native-{window_id}",
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
                packed_index.append(
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
    pd.DataFrame(packed_index).to_csv(index_path, index=False)
    pd.DataFrame(packed_index).to_csv(manifest_path, index=False)
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


def _materialize(
    tmp_path: Path,
    view: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    source, parity = _source_fixture(tmp_path)
    roles = _roles(view)
    source_integrity = pre_s1_rgb_binding.build_rgb_source_integrity_evidence(
        rgb_source_root=source,
        output_path=tmp_path / "source_integrity.json",
        input_parity_evidence=parity,
    )
    length = int(view[1:])
    report = binding.materialize_stage1_rgb_binding(
        output_dir=tmp_path / f"binding-{view}",
        rgb_source_root=source,
        requested_roles=roles,
        authority_sha256="a" * 64,
        provenance_hashes=_provenance(),
        view=view,
        sequence_length=length,
        expected_train_windows=1,
        expected_validation_windows=1,
        input_parity_evidence=parity,
        source_integrity_evidence=json.loads(
            Path(str(source_integrity["path"])).read_text(encoding="utf-8")
        ),
    )
    return report, roles


@pytest.mark.parametrize("view", ["T6", "T8", "T12", "T16"])
def test_stage1_binding_is_hash_bound_inner_only_and_view_specific(
    tmp_path: Path,
    view: str,
) -> None:
    report, roles = _materialize(tmp_path, view)
    root = Path(str(report["scientific_binding_path"])).parent
    scientific = json.loads(
        Path(str(report["scientific_binding_path"])).read_text(encoding="utf-8")
    )
    assert scientific["stage1"]["temporal_view"] == view
    assert scientific["stage1"]["sequence_length"] == int(view[1:])
    assert str(tmp_path / "reviewed_rgb_v1") not in json.dumps(scientific)
    assert report["coverage"]["train_windows_bound"] == 1
    assert report["coverage"]["validation_windows_bound"] == 1
    assert report["coverage"]["bad_sequence_length"] == 0
    windows = pd.read_csv(root / "stage1_window_context.csv")
    assert set(windows["window_id"]) == set(roles["window_id"])
    resolved = binding.resolve_stage1_execution_rgb_binding(
        data_bindings_path=Path(str(report["data_bindings_path"])),
        requested_roles=roles,
        authority_sha256="a" * 64,
        provenance_hashes=_provenance(),
        view=view,
        sequence_length=int(view[1:]),
    )
    assert resolved.audit["valid"] is True
    assert resolved.coverage["missing_windows"] == 0


def test_stage1_binding_audit_rejects_outer_wrong_length_and_cross_video(
    tmp_path: Path,
) -> None:
    report, roles = _materialize(tmp_path, "T8")
    root = Path(str(report["scientific_binding_path"])).parent
    windows = pd.read_csv(root / "stage1_window_context.csv")
    frames = pd.read_csv(root / "stage1_frame_context.csv")
    index = pd.read_csv(root / "stage1_packed_image_cache_index.csv")
    outer = windows.copy()
    outer.loc[0, "stage1_role"] = "outer"
    assert binding.audit_stage1_rgb_binding(
        windows=outer,
        frames=frames,
        packed_index=index,
        requested_roles=roles,
        view="T8",
        sequence_length=8,
    )["coverage"]["role_violations"] > 0
    malformed = windows.copy()
    malformed.loc[0, "image_context_id_sequence"] = "one;;two"
    assert binding.audit_stage1_rgb_binding(
        windows=malformed,
        frames=frames,
        packed_index=index,
        requested_roles=roles,
        view="T8",
        sequence_length=8,
    )["coverage"]["bad_sequence_length"] == 1
    cross_video = frames.copy()
    cross_video.loc[0, "video_key"] = "wrong-video"
    assert binding.audit_stage1_rgb_binding(
        windows=windows,
        frames=cross_video,
        packed_index=index,
        requested_roles=roles,
        view="T8",
        sequence_length=8,
    )["coverage"]["cross_video_violations"] == 1


def test_stage1_binding_rejects_wrong_identity_and_cache_byte_change(tmp_path: Path) -> None:
    report, roles = _materialize(tmp_path, "T12")
    with pytest.raises(binding.Stage1RgbBindingError, match="identity drifted"):
        binding.resolve_stage1_execution_rgb_binding(
            data_bindings_path=Path(str(report["data_bindings_path"])),
            requested_roles=roles,
            authority_sha256="a" * 64,
            provenance_hashes=_provenance(),
            view="T8",
            sequence_length=8,
        )
    payload = json.loads(Path(str(report["data_bindings_path"])).read_text(encoding="utf-8"))
    cache = Path(payload["execution_path_realization"]["packed_cache_path"])
    with cache.open("r+b") as handle:
        handle.seek(-1, 2)
        original = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([original[0] ^ 1]))
    stat = cache.stat()
    os.utime(cache, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    with pytest.raises(binding.Stage1RgbBindingError, match="cache hash mismatch"):
        binding.resolve_stage1_execution_rgb_binding(
            data_bindings_path=Path(str(report["data_bindings_path"])),
            requested_roles=roles,
            authority_sha256="a" * 64,
            provenance_hashes=_provenance(),
            view="T12",
            sequence_length=12,
        )
