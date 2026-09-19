from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.training.legacy_development_l5 import (
    CANONICAL_RARE_CLASSES,
    LINEAGE_SCOPE,
    SAMPLING_PROTOCOLS,
    TEMPORAL_ENCODERS,
    TEMPORAL_LENGTHS,
    VISUAL_CONTROL_IDS,
    _expected_cache_root,
    _expected_cache_source_paths,
    _letterbox_audit,
    _packed_equivalence_audit,
    _role_overlap_audit,
    _role_support_audit,
    _view_name,
    load_legacy_l5_config,
)

CONFIG_PATH = Path(
    "configs/classification_v2/legacy_development_l5_v1.json"
)


def test_legacy_l5_config_freezes_scientific_matrix() -> None:
    config = load_legacy_l5_config(CONFIG_PATH)

    assert config.payload["lineage_scope"] == LINEAGE_SCOPE
    assert config.payload["outer_holdout_predictions_authorized"] is False
    assert tuple(
        row["control_id"] for row in config.payload["visual_controls"]
    ) == VISUAL_CONTROL_IDS
    assert tuple(
        config.payload["temporal_matrix"]["sequence_lengths"]
    ) == TEMPORAL_LENGTHS
    assert tuple(
        config.payload["temporal_matrix"]["sampling_protocols"]
    ) == SAMPLING_PROTOCOLS
    assert tuple(
        config.payload["temporal_matrix"]["temporal_encoders"]
    ) == TEMPORAL_ENCODERS
    assert len(config.payload["optimization"]["seeds"]) == 3
    assert tuple(
        config.payload["promotion_contract"]["rare_classes"]
    ) == CANONICAL_RARE_CLASSES
    optimization = config.payload["optimization"]
    assert optimization["declared_local_gpu_vram_gib"] == 4
    assert optimization["maximum_peak_vram_fraction"] == 0.7
    assert optimization["short_vram_probe_required"] is True
    assert optimization["oom_retry_allowed"] is False
    assert config.payload["feature_cache"]["resnet18_frame_batch_size"] == 16
    assert config.payload["feature_cache"]["resnet34_frame_batch_size"] == 8
    expected_short_cache = Path(
        "outputs/classification_v2/legacy_only_unreviewed_development"
    ) / Path(
        "short_temporal_tiers_v3_20260714/13_c224"
    )
    assert config.short_cache_224_root == expected_short_cache
    assert _expected_cache_root(config, "short") == expected_short_cache
    expected_short_reference = Path(
        "outputs/classification_v2/legacy_only_unreviewed_development"
    ) / Path(
        "short_temporal_tiers_v3_20260714/09_actor_cache_224"
    )
    assert config.short_cache_224_reference_root == expected_short_reference
    expected_short_context = Path(
        "outputs/classification_v2/legacy_only_unreviewed_development"
    ) / Path("short_temporal_tiers_v3_20260714/08_image_context")
    assert config.short_image_context_root == expected_short_context
    short_sources = _expected_cache_source_paths(config, "short")
    assert short_sources["image_frames"].parent == expected_short_context


def test_legacy_l5_config_rejects_outer_prediction_authorization(
    tmp_path: Path,
) -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["outer_holdout_predictions_authorized"] = True
    path = tmp_path / "invalid_l5.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="claim boundary"):
        load_legacy_l5_config(path)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("optimization", "maximum_peak_vram_fraction", 0.75),
        ("feature_cache", "resnet18_frame_batch_size", 64),
    ],
)
def test_legacy_l5_config_rejects_unsafe_four_gib_gpu_settings(
    tmp_path: Path,
    section: str,
    field: str,
    value: float | int,
) -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload[section][field] = value
    path = tmp_path / "unsafe_l5.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_legacy_l5_config(path)


def test_legacy_l5_group_roles_fail_on_video_or_recording_overlap() -> None:
    roles = {
        "train": _role_frame("train", "date_train", "video_train"),
        "validation": _role_frame(
            "validation",
            "date_validation",
            "video_validation",
        ),
        "outer_holdout": _role_frame(
            "outer",
            "date_outer",
            "video_outer",
        ),
    }
    clean = _role_overlap_audit(roles)
    assert clean["valid"] is True

    roles["validation"].loc[:, "video_key"] = "video_train"
    roles["outer_holdout"].loc[:, "recording_group_id"] = "date_train"
    overlap = _role_overlap_audit(roles)

    assert overlap["valid"] is False
    assert any("video_key" in error for error in overlap["errors"])
    assert any("recording_group_id" in error for error in overlap["errors"])


def test_legacy_l5_support_warns_without_dropping_rare_classes() -> None:
    labels = [
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
    ]
    roles = {
        role: pd.DataFrame(
            {
                "behavior_label": labels,
                "video_key": [f"{role}_video"] * len(labels),
                "recording_group_id": [f"{role}_date"] * len(labels),
            }
        )
        for role in ("train", "validation", "outer_holdout")
    }

    support = _role_support_audit(
        roles,
        rare_classes=CANONICAL_RARE_CLASSES,
        warning_threshold=2,
    )

    assert support["valid"] is True
    assert support["rare_classes"] == list(CANONICAL_RARE_CLASSES)
    assert len(support["warnings"]) == len(CANONICAL_RARE_CLASSES)


def test_legacy_l5_letterbox_audit_rejects_square_stretch() -> None:
    manifest = pd.DataFrame(
        [
            {
                "source_crop_width": 200.0,
                "source_crop_height": 100.0,
                "source_crop_aspect_ratio": 2.0,
                "letterbox_resized_width": 224,
                "letterbox_resized_height": 112,
                "letterbox_pad_left": 0,
                "letterbox_pad_top": 56,
                "letterbox_pad_right": 0,
                "letterbox_pad_bottom": 56,
            }
        ]
    )
    passing = _letterbox_audit(manifest, image_size=224)
    assert passing["valid"] is True
    assert passing["padded_canvas_rows"] == 1

    manifest.loc[0, "letterbox_resized_height"] = 224
    failing = _letterbox_audit(manifest, image_size=224)
    assert failing["valid"] is False
    assert failing["invalid_rows"] == 1


def test_legacy_l5_packed_equivalence_is_exact(tmp_path: Path) -> None:
    cache_dir = tmp_path / "actor_rgb_224_letterbox"
    cache_dir.mkdir(parents=True)
    images = [
        np.full((224, 224, 3), fill_value, dtype=np.uint8)
        for fill_value in (17, 91)
    ]
    paths = []
    for index, image in enumerate(images):
        relative = Path("actor_rgb_224_letterbox") / f"image_{index}.npy"
        np.save(tmp_path / relative, image)
        paths.append(str(relative))
    manifest = pd.DataFrame(
        {
            "image_context_id": ["context_b", "context_a"],
            "cache_path": [paths[1], paths[0]],
        }
    )
    ordered = manifest.sort_values("image_context_id", kind="mergesort")
    tensor = np.stack(
        [np.load(tmp_path / path) for path in ordered["cache_path"]]
    )
    tensor_path = tmp_path / "packed_rgb_224_letterbox.npy"
    np.save(tensor_path, tensor)
    index = pd.DataFrame(
        {
            "image_context_id": ordered["image_context_id"].tolist(),
            "packed_row": [0, 1],
        }
    )

    passing = _packed_equivalence_audit(
        cache_root=tmp_path,
        manifest=manifest,
        index=index,
        packed_tensor_path=tensor_path,
        reopen_every_rows=1,
    )
    assert passing["valid"] is True
    assert passing["pixel_mismatches"] == 0
    assert passing["mapping_open_count"] == 2

    corrupted = tensor.copy()
    corrupted[0, 0, 0, 0] ^= 1
    np.save(tensor_path, corrupted)
    failing = _packed_equivalence_audit(
        cache_root=tmp_path,
        manifest=manifest,
        index=index,
        packed_tensor_path=tensor_path,
        reopen_every_rows=1,
    )
    assert failing["valid"] is False
    assert failing["pixel_mismatches"] == 1


def test_legacy_l5_view_names_are_exact() -> None:
    assert _view_name(6, "all_sliding_event_balanced") == (
        "legacy_t6_all_sliding_observed_time"
    )
    assert _view_name(16, "one_centered_window_matched") == (
        "legacy_t16_centered_matched_observed_time"
    )


def _role_frame(prefix: str, group: str, video: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "temporal_unit_key": [f"{prefix}_unit"],
            "recording_group_id": [group],
            "video_key": [video],
        }
    )
