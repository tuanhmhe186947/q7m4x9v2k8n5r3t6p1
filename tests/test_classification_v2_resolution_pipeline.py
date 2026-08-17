from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from pig_behavior.classification_v2.datasets import (
    image_sequence_dataset as image_sequence_dataset_module,
)
from pig_behavior.classification_v2.datasets.image_sequence_dataset import (
    ClassificationV2ImageSequenceDataset,
    ImageSequenceDatasetConfig,
)
from pig_behavior.classification_v2.datasets.resolution_pipeline import (
    RUNTIME_RGB_TRANSFORM_VERSION,
    build_inner_resolution_binding_from_dataframes,
    native_crop_pixel_audit,
    scan_legacy_jpeg_headers,
    storage_projection,
)
from pig_behavior.classification_v2.training import (
    post_s1_resolution_screening as post_s1,
)
from pig_behavior.classification_v2.training.post_s1_resolution_screening import (
    PostS1ResolutionError,
    _R128PackedArrayReader,
    _validate_runtime_input_binding,
)
from pig_behavior.classification_v2.training.remote_input_resolution import (
    RemoteInputAuthority,
    RemoteInputResolutionError,
    resolve_remote_input_root,
)


def _remote_input_authority(
    preferred: Path,
    *registered: Path,
) -> RemoteInputAuthority:
    return RemoteInputAuthority(
        authority_id="synthetic-input-authority",
        expected_file_count=2,
        expected_total_bytes=2,
        preferred_runtime_locator=preferred,
        registered_runtime_locators=(preferred, *registered),
        sentinel_sha256={
            "sentinel.txt": "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"
        },
        historical_parity_evidence={
            "relative_path": "synthetic.json",
            "sha256": "0" * 64,
        },
        parity_report_locator=Path("does-not-exist.json"),
    )


def _remote_input_root(path: Path, *, second_byte: bytes = b"b") -> Path:
    path.mkdir(parents=True)
    (path / "sentinel.txt").write_bytes(b"a")
    (path / "payload.bin").write_bytes(second_byte)
    return path


def _binding_inputs(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    legacy_ids = [f"legacy-{index}" for index in range(6)]
    cvat_ids = [f"cvat-{index}" for index in range(6)]
    crop_root = tmp_path / "crops"
    crop_root.mkdir()
    for index, context_id in enumerate(legacy_ids):
        image = Image.new("RGB", (80 + index, 50 + index), (index, 10, 20))
        image.save(crop_root / f"{context_id}.jpg")
    legacy_frames = pd.DataFrame(
        {
            "image_context_id": legacy_ids,
            "source_type": "legacy_recovered",
            "video_key": "legacy-video",
            "object_track_key": "legacy-track",
            "frame_index": range(6),
            "resolved_media_path": [f"crops/{context_id}.jpg" for context_id in legacy_ids],
            "image_context_source": "legacy_crop",
            "image_context_loadable": True,
            "image_width": 1280,
            "image_height": 720,
            "x1": 10.0,
            "y1": 20.0,
            "x2": 90.0,
            "y2": 70.0,
        }
    )
    cvat_frames = pd.DataFrame(
        {
            "image_context_id": cvat_ids,
            "source_type": "cvat_tracking_xml",
            "video_key": "cvat-video",
            "object_track_key": "cvat-track",
            "frame_index": range(6),
            "resolved_media_path": "videos/cvat.mp4",
            "image_context_source": "cvat_video_bbox",
            "image_context_loadable": True,
            "image_width": 100,
            "image_height": 100,
            "x1": -10.2,
            "y1": 80.1,
            "x2": 20.9,
            "y2": 110.4,
        }
    )
    windows = pd.DataFrame(
        {
            "window_id": ["legacy-window", "cvat-window"],
            "source_type": ["legacy_recovered", "cvat_tracking_xml"],
            "video_key": ["legacy-video", "cvat-video"],
            "window_length_frames": [6, 6],
            "view_type": ["T6_contiguous", "T6_contiguous"],
            "expected_frame_indices": ["0|1|2|3|4|5", "0|1|2|3|4|5"],
            "image_context_id_sequence": [
                ";;".join(legacy_ids),
                ";;".join(cvat_ids),
            ],
            "window_image_context_complete": [True, True],
        }
    )
    selection = pd.DataFrame(
        {
            "window_row_index": [0, 1],
            "window_id": ["legacy-window", "cvat-window"],
            "view_type": ["T6_contiguous", "T6_contiguous"],
            "source_type": ["legacy_recovered", "cvat_tracking_xml"],
            "behavior_window_label": ["fight", "social-nose"],
            "window_valid_for_main_train": [True, True],
            "primary_s1_role": ["train", "validation"],
            "primary_s1_eligible": [True, True],
        }
    )
    return pd.concat([legacy_frames, cvat_frames], ignore_index=True), windows, selection


def _binding(tmp_path: Path):
    frames, windows, selection = _binding_inputs(tmp_path)
    return build_inner_resolution_binding_from_dataframes(
        frames=frames,
        windows=windows,
        selection=selection,
        media_root=tmp_path,
        expected_window_count=2,
        expected_observation_count=12,
    )


def test_r128_packed_reader_matches_mmap_rows(tmp_path: Path) -> None:
    packed = np.arange(7 * 2 * 3 * 3, dtype=np.uint8).reshape(7, 2, 3, 3)
    packed_path = tmp_path / "packed.npy"
    np.save(packed_path, packed, allow_pickle=False)
    mmap = np.load(packed_path, allow_pickle=False, mmap_mode="r")
    reader = _R128PackedArrayReader(packed_path)
    try:
        for requested_rows in ([6, 2, 3, 0, 3], [0, 1, 2, 3], []):
            actual = reader.read_rows(requested_rows)
            expected = np.asarray(mmap[requested_rows])
            np.testing.assert_array_equal(actual, expected)
    finally:
        reader.close()
        del mmap


def test_runtime_resolution_changes_only_spatial_realization(tmp_path: Path) -> None:
    binding = _binding(tmp_path)
    fingerprints = []
    for resolution in (64, 128, 160, 224):
        dataset = binding.build_dataset(resolution, image_cache_size=0)
        first = dataset[0]
        repeated = dataset[0]
        assert tuple(first["image"].shape) == (6, 3, resolution, resolution)
        assert np.array_equal(first["image"].numpy(), repeated["image"].numpy())
        assert first["image_context_ids"] == repeated["image_context_ids"]
        assert first["expected_frame_indices"] == repeated["expected_frame_indices"]
        assert first["errors"] == []
        realization = binding.runtime_realization(resolution)
        assert realization["scientific_identity_sha256"] == binding.identity_sha256
        assert realization["runtime_transform_version"] == RUNTIME_RGB_TRANSFORM_VERSION
        assert len(realization["runtime_realization_sha256"]) == 64
        fingerprints.append(realization)
        dataset.close()
    assert [item["input_resolution"] for item in fingerprints] == [64, 128, 160, 224]
    assert len({item["runtime_realization_sha256"] for item in fingerprints}) == 4


def test_binding_rejects_outer_selection_without_opening_media(tmp_path: Path) -> None:
    frames, windows, selection = _binding_inputs(tmp_path)
    selection["primary_s1_role"] = "outer"
    with pytest.raises(ValueError, match="inner selection is empty"):
        build_inner_resolution_binding_from_dataframes(
            frames=frames,
            windows=windows,
            selection=selection,
            media_root=tmp_path,
        )


def test_missing_legacy_media_is_masked_not_substituted(tmp_path: Path) -> None:
    binding = _binding(tmp_path)
    frames = binding.frames.copy()
    frames.loc[0, "resolved_media_path"] = "crops/missing.jpg"
    broken = build_inner_resolution_binding_from_dataframes(
        frames=frames,
        windows=binding.windows,
        selection=binding.selection,
        media_root=tmp_path,
    )
    dataset = broken.build_dataset(64, image_cache_size=0)
    item = dataset[0]
    assert item["observed_mask"][0].item() == 0.0
    assert item["observed_mask"][1:].sum().item() == 5.0
    assert item["errors"] == ["image_load_failed@0"]
    dataset.close()


def test_native_header_audit_is_resumable_and_uses_clamped_cvat_geometry(
    tmp_path: Path,
) -> None:
    binding = _binding(tmp_path)
    output_csv = tmp_path / "headers.csv"
    checkpoint_json = tmp_path / "headers.checkpoint.json"
    first = scan_legacy_jpeg_headers(
        binding.frames,
        media_root=tmp_path,
        output_csv=output_csv,
        checkpoint_json=checkpoint_json,
        workers=1,
        checkpoint_every=1,
    )
    resumed = scan_legacy_jpeg_headers(
        binding.frames,
        media_root=tmp_path,
        output_csv=output_csv,
        checkpoint_json=checkpoint_json,
        workers=1,
        checkpoint_every=1,
    )
    audit = native_crop_pixel_audit(binding, output_csv)
    assert first["complete"] is True
    assert resumed["completed"] == 6
    assert audit["observation_count"] == 12
    assert audit["all"]["min_dimension"]["min"] == 20
    assert audit["class_specific"]["fight"]["count"] == 6
    assert audit["class_specific"]["social-nose"]["count"] == 6


def test_storage_projection_is_exact_and_nonmaterializing() -> None:
    audit = storage_projection(201792)
    assert audit["uint8"]["224"]["bytes"] == 30375346176
    assert audit["float32"]["224"]["bytes"] == 121501384704


def test_partially_clipped_cvat_crop_preserves_current_clamp_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCapture:
        def isOpened(self) -> bool:
            return True

        def set(self, _property: int, _value: int) -> None:
            return None

        def read(self) -> tuple[bool, np.ndarray]:
            return True, np.full((10, 20, 3), 128, dtype=np.uint8)

        def release(self) -> None:
            return None

    monkeypatch.setattr(
        image_sequence_dataset_module,
        "cv2",
        SimpleNamespace(
            VideoCapture=lambda _path: FakeCapture(),
            CAP_PROP_POS_FRAMES=1,
            COLOR_BGR2RGB=2,
            cvtColor=lambda image, _code: image[..., ::-1],
        ),
    )
    dataset = object.__new__(ClassificationV2ImageSequenceDataset)
    dataset.config = ImageSequenceDatasetConfig(image_size=64)
    dataset._capture_cache = OrderedDict()
    dataset._capture_next_frame = {}
    dataset._decoded_video_frame = {}
    dataset.video_decode_count = 0
    dataset.video_seek_count = 0
    dataset.video_frame_reuse_count = 0
    dataset.video_capture_open_count = 0
    dataset.video_capture_eviction_count = 0
    dataset.peak_open_video_captures = 0
    dataset.video_decode_seconds = 0.0
    dataset.video_crop_resize_seconds = 0.0

    image = dataset._load_video_bbox_crop(
        {
            "resolved_media_path": "synthetic.mp4",
            "frame_index": 0,
            "x1": -5.4,
            "y1": 2.2,
            "x2": 25.9,
            "y2": 11.8,
        }
    )

    assert image is not None
    assert image.shape == (3, 64, 64)
    assert dataset.video_seek_count == 1


def test_remote_input_resolver_prefers_registered_matching_locator(tmp_path: Path) -> None:
    preferred = _remote_input_root(tmp_path / "inputs")

    binding = resolve_remote_input_root(_remote_input_authority(preferred))

    assert binding.effective_remote_input_root == preferred.resolve()
    assert binding.scientific_input_authority_id == "synthetic-input-authority"


def test_remote_input_resolver_uses_registered_fallback_after_preferred_absence(
    tmp_path: Path,
) -> None:
    preferred = tmp_path / "inputs"
    fallback = _remote_input_root(tmp_path / "project" / "inputs")

    binding = resolve_remote_input_root(_remote_input_authority(preferred, fallback))

    assert binding.effective_remote_input_root == fallback.resolve()
    assert binding.preferred_remote_input_root == preferred


def test_remote_input_resolver_rejects_registered_fallback_parity_mismatch(
    tmp_path: Path,
) -> None:
    preferred = tmp_path / "inputs"
    fallback = _remote_input_root(tmp_path / "project" / "inputs", second_byte=b"wrong")

    with pytest.raises(RemoteInputResolutionError, match="locator conflict"):
        resolve_remote_input_root(_remote_input_authority(preferred, fallback))


def test_remote_input_resolver_never_discovers_unregistered_inputs(tmp_path: Path) -> None:
    preferred = tmp_path / "inputs"
    _remote_input_root(tmp_path / "unregistered" / "inputs")

    with pytest.raises(RemoteInputResolutionError, match="no registered"):
        resolve_remote_input_root(_remote_input_authority(preferred))


def test_remote_input_resolver_selects_equivalent_registered_alias_deterministically(
    tmp_path: Path,
) -> None:
    preferred = _remote_input_root(tmp_path / "inputs")
    equivalent = preferred / "."

    binding = resolve_remote_input_root(_remote_input_authority(preferred, equivalent))

    assert binding.effective_remote_input_root == preferred.resolve()
    assert binding.resolved_candidates == (preferred.resolve(),)


def test_locator_change_preserves_scientific_input_identity(tmp_path: Path) -> None:
    first = _remote_input_root(tmp_path / "first")
    second = _remote_input_root(tmp_path / "second")

    first_binding = resolve_remote_input_root(_remote_input_authority(first))
    second_binding = resolve_remote_input_root(_remote_input_authority(second))

    assert (
        first_binding.scientific_input_authority_id
        == second_binding.scientific_input_authority_id
    )
    assert first_binding.effective_remote_input_root != second_binding.effective_remote_input_root


def test_input_resolution_leaves_existing_outer_protection_unchanged(tmp_path: Path) -> None:
    root = _remote_input_root(tmp_path / "inputs")
    binding = resolve_remote_input_root(_remote_input_authority(root))

    assert binding.scientific_input_authority_id == "synthetic-input-authority"
    outer_root = tmp_path / "outer-protection"
    outer_root.mkdir()
    frames, windows, selection = _binding_inputs(outer_root)
    selection["primary_s1_role"] = "outer"
    with pytest.raises(ValueError, match="inner selection is empty"):
        build_inner_resolution_binding_from_dataframes(
            frames=frames,
            windows=windows,
            selection=selection,
            media_root=outer_root,
        )


def test_resolution_runner_requires_verified_root_binding_for_media_root(
    tmp_path: Path,
) -> None:
    media_root = tmp_path / "inputs"
    media_root.mkdir()
    contract_path = tmp_path / "remote-input-contract.json"
    contract = {
        "schema_version": "classification_v2.remote_input_root.v1",
        "status": "ACTIVE_RUNTIME_LOCATOR_CONTRACT",
        "scientific_input_authority": {
            "authority_id": "synthetic-input-authority",
            "historical_parity_evidence": {
                "relative_path": "synthetic.json",
                "sha256": "0" * 64,
            },
            "expected_population": {
                "physical_file_count": 1,
                "total_bytes": 1,
            },
            "registered_sentinels": {},
        },
        "runtime_input_locators": {
            "preferred": str(media_root),
            "registered": [str(media_root)],
            "parity_report_locator": "parity.json",
        },
    }
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    contract_sha256 = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    binding_path = tmp_path / "runtime-input-binding.json"
    binding_path.write_text(
        json.dumps(
            {
                "scientific_input_authority_id": "synthetic-input-authority",
                "effective_remote_input_root": str(media_root),
                "expected_file_count": 1,
                "expected_total_bytes": 1,
            }
        ),
        encoding="utf-8",
    )
    resolution_authority = {
        "runtime_input_authority": {
            "relative_segments": ["."],
            "filename": contract_path.name,
            "sha256": contract_sha256,
        }
    }

    verified = _validate_runtime_input_binding(
        resolution_authority,
        repository_root=tmp_path,
        authority_path=contract_path,
        binding_path=binding_path,
        media_root=media_root,
    )

    assert verified["effective_remote_input_root"] == str(media_root)
    with pytest.raises(PostS1ResolutionError, match="media root"):
        _validate_runtime_input_binding(
            resolution_authority,
            repository_root=tmp_path,
            authority_path=contract_path,
            binding_path=binding_path,
            media_root=tmp_path / "other-inputs",
        )


def test_full_resolution_arm_delegates_to_inherited_stage1_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inherited_plan = object()
    rows = SimpleNamespace(
        train=pd.DataFrame({"window_id": ["train"]}),
        validation=pd.DataFrame({"window_id": ["validation"]}),
        expected_native_units=pd.DataFrame({"native_unit_id": ["native"]}),
        common_cohort_native_units=pd.DataFrame({"native_unit_id": ["native"]}),
    )
    population = SimpleNamespace(
        rows=rows,
        stage1_plan=inherited_plan,
        load_batch=object(),
        close=object(),
        data_hashes={"post_s1_host_binding": "binding"},
    )
    plan = SimpleNamespace(device_name="cuda")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        post_s1,
        "_assert_l4",
        lambda value: captured.setdefault("l4", value),
    )

    def run_inherited(plan_value: object, population_value: object) -> dict[str, object]:
        captured["plan"] = plan_value
        captured["population"] = population_value
        return {"status": "PASS", "executor": "inherited"}

    monkeypatch.setattr(
        post_s1.stage1,
        "run_stage1_temporal_screening",
        run_inherited,
    )

    assert post_s1.run_resolution_arm(plan, population, steps=4164) == {
        "status": "PASS",
        "executor": "inherited",
    }
    assert captured["l4"] is plan
    assert captured["plan"] is inherited_plan
    inherited_population = captured["population"]
    assert inherited_population.load_batch is population.load_batch
    assert inherited_population.data_hashes == population.data_hashes
