"""The model must derive the motion dimension from the canonical schema."""

from __future__ import annotations

import ast
from pathlib import Path

import torch

from pig_behavior.classification_v2.features.motion_schema import (
    MOTION_FEATURE_NAMES,
    MOTION_SCHEMA_DIMENSION,
    MOTION_SCHEMA_HASH,
    MOTION_SCHEMA_VERSION,
)
from pig_behavior.classification_v2.models.balanced import contracts as contracts_module
from pig_behavior.classification_v2.models.balanced.baselines import baseline_config
from pig_behavior.classification_v2.models.balanced.contracts import (
    ModelBatch,
    numeric_group_feature_names,
    validate_batch,
)
from pig_behavior.classification_v2.models.balanced.synthetic import (
    SyntheticBatchSpec,
    synthetic_batch,
)
from pig_behavior.classification_v2.spatial_sequence_export import (
    SPATIAL_FRAME_FEATURES,
)

B3 = "B3_ACTOR_T6_PLUS_GEOMETRY_MOTION"


def test_motion_names_are_the_single_authority() -> None:
    names = numeric_group_feature_names()["motion_delta"]
    assert names == tuple(MOTION_FEATURE_NAMES)
    assert len(names) == MOTION_SCHEMA_DIMENSION
    assert names == tuple(SPATIAL_FRAME_FEATURES["motion_delta"])


def test_motion_dimension_is_derived_not_hard_coded() -> None:
    source = Path(contracts_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    literal_twelves = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value == 12
    ]
    assert not literal_twelves, (
        "the balanced contract must derive the motion dimension from "
        "len(motion_names); found a hard-coded 12"
    )


def test_schema_hash_and_version_mismatch_fail_closed() -> None:
    config = baseline_config(B3)
    contract = config.batch_contract
    batch = synthetic_batch(SyntheticBatchSpec(contract=contract, batch_size=2))
    assert batch.motion_schema_hash == MOTION_SCHEMA_HASH
    assert batch.motion_schema_version == MOTION_SCHEMA_VERSION
    assert validate_batch(batch, contract).passed

    for field, value in (
        ("motion_schema_hash", "0" * 64),
        ("motion_schema_version", "classification_v2.motion_tensor.v1"),
    ):
        drifted = ModelBatch(
            target=batch.target,
            numeric_feature_names=batch.numeric_feature_names,
            quality_mask_names=batch.quality_mask_names,
            modality_availability=batch.modality_availability,
            labels=batch.labels,
            native_unit_id=batch.native_unit_id,
            window_id=batch.window_id,
            **{
                "motion_schema_hash": batch.motion_schema_hash,
                "motion_schema_version": batch.motion_schema_version,
                field: value,
            },
        )
        check = validate_batch(drifted, contract).check("MOTION_DIMENSION_CONTRACT")
        assert not check.passed
        assert any(field in error for error in check.errors)


def test_dynamic_motion_width_flows_through_the_model() -> None:
    from pig_behavior.classification_v2.models.balanced.balanced_model import (
        BalancedCausalModel,
    )

    config = baseline_config(B3, hidden_dim=16)
    model = BalancedCausalModel(config)
    linear = model.numeric_encoder.encoders["motion_delta"][0]
    assert linear.in_features == len(MOTION_FEATURE_NAMES)

    batch = synthetic_batch(
        SyntheticBatchSpec(contract=config.batch_contract, batch_size=2, image_size=16)
    )
    assert batch.target.numeric_groups["motion_delta"].shape[-1] == len(
        MOTION_FEATURE_NAMES
    )
    assert bool(torch.isfinite(model(batch)["logits"]).all())
