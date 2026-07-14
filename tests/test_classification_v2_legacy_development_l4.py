from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.legacy_development_l4 import (
    LINEAGE_SCOPE,
    load_legacy_l4_config,
    load_legacy_l4_data,
    run_legacy_l4_fold_epoch,
    run_legacy_l4_short,
    state_sha256,
)


def test_legacy_l4_short_and_fold_epoch_are_cache_only(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    config = load_legacy_l4_config(config_path)
    checkpoint = tmp_path / "audit" / "resume_probe.pt"

    short = run_legacy_l4_short(config, checkpoint_path=checkpoint)

    assert short["valid"] is True
    assert short["status"] == "PASS_LEGACY_DEVELOPMENT_L4_SHORT"
    assert short["input_contract_audit"]["metadata_separated_from_x"] is True
    assert short["mask_and_order_audit"]["valid"] is True
    assert short["one_batch_gradient_audit"]["valid"] is True
    assert short["deterministic_repeat_audit"]["valid"] is True
    assert short["checkpoint_resume_audit"]["valid"] is True
    assert short["tiny_overfit_audit"]["valid"] is True
    assert short["cache_only_audit"]["source_image_loads"] == 0
    assert short["cache_only_audit"]["video_decode_count"] == 0
    short_path = tmp_path / "audit" / "short.json"
    short_path.write_text(json.dumps(short, indent=2), encoding="utf-8")

    full = run_legacy_l4_fold_epoch(config, short_audit_path=short_path)

    assert full["valid"] is True
    assert full["status"] == "PASS_LEGACY_DEVELOPMENT_L4"
    assert full["one_fold_one_epoch_audit"]["epochs"] == 1
    assert full["one_fold_one_epoch_audit"]["native_event_rows"] == 10
    assert full["optimizer_support_audit"]["all_classes_supported"] is True
    assert full["held_out_predictions_computed"] is False
    assert full["held_out_accuracy_f1_computed"] is False
    assert full["cache_only_audit"]["packed_image_cache_hits"] == 20

    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="resume_checkpoint_sha256"):
        run_legacy_l4_fold_epoch(config, short_audit_path=short_path)


def test_legacy_l4_config_rejects_claim_expansion(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["accuracy_f1_comparison_authorized"] = True
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="claim boundary"):
        load_legacy_l4_config(config_path)


def test_legacy_l4_data_rejects_duplicate_native_units(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    config = load_legacy_l4_config(config_path)
    fold_path = config.primary_root / "11_folds" / "window_oof_fold_manifest.csv"
    folds = pd.read_csv(fold_path)
    folds.loc[1, "temporal_unit_key"] = folds.loc[0, "temporal_unit_key"]
    folds.to_csv(fold_path, index=False)

    with pytest.raises(ValueError, match="one-to-one|not unique|duplicate"):
        load_legacy_l4_data(config)


def test_legacy_l4_state_hash_is_order_stable() -> None:
    first = {"b": np.array([2, 3]), "a": {"x": 1}}
    second = {"a": {"x": 1}, "b": np.array([2, 3])}

    assert state_sha256(first) == state_sha256(second)


def _write_fixture(tmp_path: Path) -> Path:
    development_root = tmp_path / "development"
    run_id = "primary"
    root = development_root / run_id
    for directory in (
        root / "06_temporal_tier_contract",
        root / "09_image_context",
        root / "10_actor_cache_32",
        root / "11_folds",
        root / "13_l3_audit",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    fold_rows, native_rows, frame_rows, window_rows, view_rows, packed = (
        _fixture_rows()
    )
    pd.DataFrame(fold_rows).to_csv(
        root / "11_folds" / "window_oof_fold_manifest.csv",
        index=False,
    )
    pd.DataFrame(native_rows).to_csv(
        root
        / "06_temporal_tier_contract"
        / "native_temporal_unit_manifest.csv",
        index=False,
    )
    pd.DataFrame(frame_rows).to_csv(
        root / "09_image_context" / "image_frame_context_manifest.csv",
        index=False,
    )
    pd.DataFrame(window_rows).to_csv(
        root / "09_image_context" / "image_window_context_manifest.csv",
        index=False,
    )
    pd.DataFrame(view_rows).to_csv(
        root
        / "06_temporal_tier_contract"
        / "legacy_t16_centered_matched_observed_time_manifest.csv",
        index=False,
    )
    np.save(root / "10_actor_cache_32" / "packed_rgb_32_letterbox.npy", packed)
    pd.DataFrame(
        {
            "image_context_id": [row["image_context_id"] for row in frame_rows],
            "packed_row": np.arange(len(frame_rows)),
        }
    ).to_csv(
        root / "10_actor_cache_32" / "packed_image_cache_index.csv",
        index=False,
    )
    _write_support(root)
    _write_l3_audit(root)
    config_path = tmp_path / "legacy_l4.json"
    config_path.write_text(
        json.dumps(_config_payload(development_root, run_id), indent=2),
        encoding="utf-8",
    )
    return config_path


def _fixture_rows() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    np.ndarray,
]:
    fold_rows: list[dict[str, object]] = []
    native_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    view_rows: list[dict[str, object]] = []
    packed_rows: list[np.ndarray] = []
    for side in ("train", "test"):
        for class_index, label in enumerate(VALID_BEHAVIORS):
            unit = f"unit_{side}_{class_index:02d}"
            window = f"window_{side}_{class_index:02d}"
            fold = "fold_other" if side == "train" else "fold_000"
            contexts = []
            for slot in range(2):
                context = f"context_{side}_{class_index:02d}_{slot}"
                contexts.append(context)
                frame_rows.append(
                    {
                        "image_context_id": context,
                        "source_type": "legacy_recovered",
                        "resolved_media_path": "unused.mp4",
                        "image_context_loadable": True,
                        "image_context_source": "legacy_video_bbox",
                        "x1": 0,
                        "y1": 0,
                        "x2": 32,
                        "y2": 32,
                        "frame_index": slot,
                    }
                )
                packed_rows.append(_class_image(class_index, slot, side))
                view_rows.append(
                    {
                        "temporal_view_name": (
                            "legacy_t16_centered_matched_observed_time"
                        ),
                        "parent_window_id": window,
                        "slot_index": slot,
                        "declared_sequence_length": 2,
                        "time_delta": 0.0 if slot == 0 else 0.2,
                        "length_mask": True,
                        "observed_mask": True,
                        "timing_valid_mask": True,
                        "padding_mask": False,
                        "lineage_scope": LINEAGE_SCOPE,
                        "human_review_complete": False,
                    }
                )
            fold_rows.append(
                {
                    "window_id": window,
                    "temporal_unit_key": unit,
                    "oof_fold_id": fold,
                    "behavior_label": label,
                    "source_type": "legacy_recovered",
                    "video_key": f"video_{side}_{class_index:02d}",
                    "recording_group_id": f"date_{side}_{class_index:02d}",
                    "lineage_scope": LINEAGE_SCOPE,
                    "human_review_complete": False,
                    "legacy_t16_centered_matched_keep": True,
                }
            )
            native_rows.append(
                {
                    "temporal_unit_key": unit,
                    "native_unit_valid_for_development": True,
                    "lineage_scope": LINEAGE_SCOPE,
                    "human_review_complete": False,
                }
            )
            window_rows.append(
                {
                    "window_id": window,
                    "source_type": "legacy_recovered",
                    "video_key": f"video_{side}_{class_index:02d}",
                    "expected_frame_indices": "0|1",
                    "image_context_id_sequence": ";;".join(contexts),
                    "window_image_context_complete": True,
                }
            )
    return (
        fold_rows,
        native_rows,
        frame_rows,
        window_rows,
        view_rows,
        np.stack(packed_rows),
    )


def _class_image(class_index: int, slot: int, side: str) -> np.ndarray:
    image = np.full((32, 32, 3), 16, dtype=np.uint8)
    channel = class_index % 3
    start = 2 + (class_index * 2) % 20
    image[:, start : start + 4, channel] = 220
    row = 2 + ((9 - class_index) * 2) % 20
    image[row : row + 4, :, (channel + 1) % 3] = 180
    image[slot * 4 : slot * 4 + 3, :3, :] = 255
    if side == "test":
        image[-2:, -2:, :] = 8
    return image


def _write_support(root: Path) -> None:
    class_rows = [
        {
            "oof_fold_id": "fold_000",
            "behavior_label": label,
            "test_native_units": 1,
            "train_native_units": 1,
            "test_supported": True,
            "train_supported": True,
            "lineage_scope": LINEAGE_SCOPE,
            "human_review_complete": False,
        }
        for label in VALID_BEHAVIORS
    ]
    pd.DataFrame(class_rows).to_csv(
        root / "11_folds" / "class_by_fold_support.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "oof_fold_id": "fold_000",
                "source_type": "legacy_recovered",
                "test_native_units": 10,
                "train_native_units": 10,
                "test_supported": True,
                "train_supported": True,
                "lineage_scope": LINEAGE_SCOPE,
                "human_review_complete": False,
            }
        ]
    ).to_csv(
        root / "11_folds" / "source_by_fold_support.csv",
        index=False,
    )


def _write_l3_audit(root: Path) -> None:
    payload = {
        "status": "PASS_LEGACY_DEVELOPMENT_L3",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "accuracy_f1_comparison_authorized": False,
        "l4_model_correctness_authorized": True,
        "bounded_model_correctness_training_authorized": True,
        "primary_root": str(root).replace("\\", "/"),
        "errors": [],
        "valid": True,
    }
    path = root / "13_l3_audit" / "legacy_development_l3_audit.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _config_payload(development_root: Path, run_id: str) -> dict[str, object]:
    return {
        "schema_version": "classification_v2.legacy_development_l4.config.v1",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "accuracy_f1_comparison_authorized": False,
        "development_root": str(development_root),
        "primary_run_id": run_id,
        "l3_audit_relative_path": (
            "13_l3_audit/legacy_development_l3_audit.json"
        ),
        "fold_id": "fold_000",
        "temporal_view_name": (
            "legacy_t16_centered_matched_observed_time"
        ),
        "temporal_selection_column": "legacy_t16_centered_matched_keep",
        "expected_selected_native_units": 20,
        "expected_development_valid_native_units": 20,
        "model": {
            "model_mode": "actor_temporal",
            "backbone_name": "smoke_cnn",
            "pretrained_weight_enum": "NONE_RANDOM_INIT",
            "image_size": 32,
            "sequence_length": 2,
            "hidden_dim": 16,
            "temporal_encoder_name": "masked_tcn",
            "dropout": 0.0,
        },
        "short_gate": {
            "seed": 20260714,
            "device": "cpu",
            "gradient_learning_rate": 0.003,
            "tiny_events_per_class": 1,
            "tiny_steps": 80,
            "tiny_learning_rate": 0.02,
            "tiny_minimum_accuracy": 0.9,
            "tiny_maximum_loss_ratio": 0.4,
            "frame_batch_events": 2,
            "maximum_peak_vram_fraction": 0.75,
        },
        "fold_epoch": {
            "epochs": 1,
            "frame_batch_events": 2,
            "train_batch_size": 10,
            "learning_rate": 0.003,
            "weight_decay": 0.0,
            "maximum_peak_vram_fraction": 0.75,
            "visual_policy": "frozen_random_resnet_correctness_only",
        },
    }
