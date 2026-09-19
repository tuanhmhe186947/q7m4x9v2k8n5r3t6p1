"""Tiny synthetic overfit smoke test.

This proves the optimization path works end to end. It says nothing whatsoever
about model quality: the data is synthetic and separable by construction, and no
production label, media or run root is touched.
"""

from __future__ import annotations

import pytest
import torch

from pig_behavior.classification_v2.models.balanced.balanced_model import (
    BalancedCausalModel,
)
from pig_behavior.classification_v2.models.balanced.baselines import baseline_config
from pig_behavior.classification_v2.models.balanced.synthetic import (
    synthetic_overfit_dataset,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.balanced.class_priors import (
    compute_class_priors,
)
from pig_behavior.classification_v2.training.balanced.losses import (
    LossConfig,
    build_loss,
)

MAX_STEPS = 60
REQUIRED_LOSS_REDUCTION = 0.5


def _tiny_model() -> BalancedCausalModel:
    config = baseline_config(
        "B3_ACTOR_T6_PLUS_GEOMETRY_MOTION",
        hidden_dim=32,
        target_length=4,
    )
    torch.manual_seed(0)
    return BalancedCausalModel(config)


def test_tiny_synthetic_overfit_reduces_loss() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    batch, provenance = synthetic_overfit_dataset(
        model.config.batch_contract,
        batch_size=8,
        image_size=12,
    )
    assert provenance == {
        "production_data_used": False,
        "production_labels_used": False,
        "production_media_used": False,
        "claims_model_quality": False,
    }

    loss_fn = build_loss(LossConfig(name="L0_STANDARD_CROSS_ENTROPY"))
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    model.train()
    first_loss: float | None = None
    last_loss = float("inf")
    for step in range(MAX_STEPS):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch, validate=step == 0)["logits"]
        components = loss_fn(logits, batch.labels)
        components.reduced_loss.backward()
        optimizer.step()
        value = float(components.reduced_loss.detach())
        assert value == value, "loss became NaN"
        if first_loss is None:
            first_loss = value
        last_loss = value

    assert first_loss is not None
    assert last_loss < first_loss * REQUIRED_LOSS_REDUCTION, (
        f"tiny synthetic overfit did not reduce loss: first={first_loss} "
        f"last={last_loss}"
    )


def test_tiny_overfit_uses_train_fold_priors_when_weighted() -> None:
    torch.manual_seed(1)
    model = _tiny_model()
    # One synthetic native unit per class: a prior with a zero count is
    # undefined and is rejected upstream by design.
    batch, _ = synthetic_overfit_dataset(
        model.config.batch_contract,
        batch_size=len(VALID_BEHAVIORS),
        image_size=12,
    )
    priors = compute_class_priors(
        native_unit_ids=list(batch.native_unit_id),
        native_unit_labels=batch.labels.tolist(),
        fold_id="synthetic",
    )
    loss_fn = build_loss(
        LossConfig(name="L3_BALANCED_SOFTMAX", priors=priors)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    model.train()
    losses: list[float] = []
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        components = loss_fn(model(batch, validate=False)["logits"], batch.labels)
        components.reduced_loss.backward()
        optimizer.step()
        losses.append(float(components.reduced_loss.detach()))
    assert losses[-1] < losses[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_optional_small_cuda_smoke() -> None:
    torch.manual_seed(2)
    model = _tiny_model().to("cuda")
    batch, _ = synthetic_overfit_dataset(
        model.config.batch_contract,
        batch_size=4,
        image_size=12,
    )
    moved = _to_device(batch, "cuda")
    logits = model(moved, validate=False)["logits"]
    assert logits.device.type == "cuda"
    assert bool(torch.isfinite(logits).all())


def _to_device(batch, device: str):
    from pig_behavior.classification_v2.models.balanced.contracts import (
        ModelBatch,
        SequenceSegment,
    )

    segment = batch.target
    return ModelBatch(
        target=SequenceSegment(
            valid_mask=segment.valid_mask.to(device),
            frame_offsets=segment.frame_offsets.to(device),
            images=None if segment.images is None else segment.images.to(device),
            numeric_groups={
                name: tensor.to(device)
                for name, tensor in segment.numeric_groups.items()
            },
            quality_mask=(
                None if segment.quality_mask is None else segment.quality_mask.to(device)
            ),
        ),
        numeric_feature_names=batch.numeric_feature_names,
        quality_mask_names=batch.quality_mask_names,
        modality_availability={
            name: tensor.to(device)
            for name, tensor in batch.modality_availability.items()
        },
        labels=None if batch.labels is None else batch.labels.to(device),
        native_unit_id=batch.native_unit_id,
        window_id=batch.window_id,
        motion_schema_hash=batch.motion_schema_hash,
        motion_schema_version=batch.motion_schema_version,
    )
