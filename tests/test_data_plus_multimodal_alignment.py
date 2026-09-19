"""Keyed DATA+ multimodal sidecar and Joint-forward checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pig_behavior.classification_v2.models.joint_representation_model import (
    JointRepresentationClassifier,
)
from pig_behavior.classification_v2.models.multimodal_fusion import (
    MultimodalFusionConfig,
)
from pig_behavior.classification_v2.models.visual_backbones import (
    NO_PRETRAINED_WEIGHTS,
)
from pig_behavior.classification_v2.training.data_plus_multimodal import (
    DataPlusMultimodalStore,
    DataPlusSidecarPaths,
    make_data_plus_sample_key,
)


def _write_sidecars(root, *, duplicate=False, omit_h5=False, pb_subset=False):
    n = 2
    units = np.asarray(["u0", "u1"])
    keys = np.asarray([make_data_plus_sample_key(v) for v in units])
    if duplicate:
        keys[1] = keys[0]
    target = np.asarray(["clip:actor0", "clip:actor1"])
    identity = {
        "sample_key": keys,
        "supplemental_unit_id": units,
        "target_object_track_key": target,
    }
    np.savez(
        root / "data_plus_rgb_sidecar.npz",
        **identity,
        feature_tensor=np.zeros((n, 2, 6, 8, 8, 3), np.uint8),
        validity_mask=np.ones((n, 2, 6), bool),
        time_delta_seconds=np.zeros((n, 6), np.float32),
    )
    structured = np.zeros((n, 6, 46), np.float32)
    np.savez(
        root / "data_plus_spatial_46d_sidecar.npz",
        **identity,
        class_label=np.asarray(["stand", "eat"]),
        feature_tensor=structured,
        validity_mask=np.ones((n, 6), bool),
        roi_validity_mask=np.ones((n, 6, 3), bool),
        social_feature_validity_mask=np.ones((n, 6, 10), bool),
        motion_feature_validity_mask=np.ones((n, 6, 12), bool),
    )
    if not omit_h5:
        np.savez(
            root / "data_plus_h5_sidecar.npz",
            **identity,
            feature_tensor=np.zeros((n, 5, 46), np.float32),
            validity_mask=np.ones((n, 5), bool),
        )
    np.savez(
        root / "data_plus_roi_sidecar.npz",
        **identity,
        feature_tensor=np.zeros((n, 6, 18), np.float32),
        validity_mask=np.ones((n, 6, 3), bool),
    )
    np.savez(
        root / "data_plus_posture_sidecar.npz",
        **identity,
        feature_tensor=np.full(n, -1, np.int64),
        validity_mask=np.zeros(n, bool),
    )
    pb_rows = 1 if pb_subset else n
    torch.save(
        {
            "sample_key": keys[:pb_rows],
            "partner_probs": torch.zeros(pb_rows, 2, 10),
            "partner_hidden": torch.zeros(pb_rows, 2, 256),
            "partner_mask": torch.zeros(pb_rows, 2, dtype=torch.bool),
        },
        root / "data_plus_pb_sidecar.pt",
    )


def test_all_modalities_resolve_by_one_key(tmp_path):
    _write_sidecars(tmp_path)
    store = DataPlusMultimodalStore(DataPlusSidecarPaths.from_root(tmp_path))
    batch = store.batch(["data_plus_u1"], device=torch.device("cpu"))
    assert batch.sample_keys == ("data_plus_u1",)
    assert all(values == batch.sample_keys for values in batch.source_keys.values())
    assert batch.training_batch.metadata["supplemental_unit_id"] == ["u1"]
    assert batch.context_inputs["h5_structured"].shape == (1, 5, 46)


def test_unknown_key_fails_closed_without_row_fallback(tmp_path):
    _write_sidecars(tmp_path)
    store = DataPlusMultimodalStore(DataPlusSidecarPaths.from_root(tmp_path))
    with pytest.raises(KeyError, match=r"unknown DATA\+ sample_key"):
        store.batch(["canonical_row_0"], device=torch.device("cpu"))


def test_duplicate_identity_fails_closed(tmp_path):
    _write_sidecars(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="duplicate sample_key"):
        DataPlusMultimodalStore(DataPlusSidecarPaths.from_root(tmp_path))


def test_missing_modality_fails_closed(tmp_path):
    _write_sidecars(tmp_path, omit_h5=True)
    with pytest.raises(FileNotFoundError, match=r"missing DATA\+ H5"):
        DataPlusMultimodalStore(DataPlusSidecarPaths.from_root(tmp_path))


def test_fold_specific_pb_subset_fails_only_for_unavailable_key(tmp_path):
    _write_sidecars(tmp_path, pb_subset=True)
    store = DataPlusMultimodalStore(DataPlusSidecarPaths.from_root(tmp_path))
    assert store.batch(
        ["data_plus_u0"], device=torch.device("cpu")
    ).sample_keys == ("data_plus_u0",)
    with pytest.raises(KeyError, match=r"no fold-eligible DATA\+ entry"):
        store.batch(["data_plus_u1"], device=torch.device("cpu"))


def test_one_keyed_data_plus_batch_reaches_joint_forward(tmp_path):
    _write_sidecars(tmp_path)
    store = DataPlusMultimodalStore(DataPlusSidecarPaths.from_root(tmp_path))
    batch = store.batch(["data_plus_u0"], device=torch.device("cpu"))
    config = MultimodalFusionConfig(
        backbone_name="smoke_cnn",
        pretrained_weight_enum=NO_PRETRAINED_WEIGHTS,
        temporal_encoder_name="masked_tcn",
        image_embedding_dim=128,
        spatial_embedding_dim=128,
        interaction_embedding_dim=64,
        visual_context_embedding_dim=128,
        fusion_hidden_dim=256,
        num_classes=10,
        enable_image=True,
        enable_spatial=True,
        enable_interaction_context=True,
        enable_visual_context=True,
        interaction_context_dim=5,
        spatial_input_dims={
            "bbox_xywh_n": 4,
            "bbox_shape_n": 2,
            "motion_delta": 12,
            "roi_class_relation": 18,
            "social_relation": 10,
        },
        dropout=0.0,
    )
    model = JointRepresentationClassifier(config).eval()
    with torch.inference_mode():
        output = model(
            **batch.training_batch.model_inputs,
            **batch.context_inputs,
        )
    assert output.behavior_logits.shape == (1, 10)
    assert torch.isfinite(output.behavior_logits).all()


def test_joint_runner_has_no_silent_positional_fallback():
    source = Path(
        "src/pig_behavior/classification_v2/training/"
        "run_joint_representation_5fold.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "train_idx_map.get",
        "canonical_row_{idx}",
        ".get(index, 0)",
        ".get(idx, 0)",
    ):
        assert forbidden not in source
