"""M1-DG1 gradient, isolation, and authoritative CPU fast-path proofs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from pig_behavior.classification_v2.models.date_adversarial import (
    DateDomainHead,
    gradient_reverse,
)
from pig_behavior.classification_v2.training.config import load_training_config
from pig_behavior.classification_v2.training.data_module import (
    StrictTrainingDataModule,
)
from pig_behavior.classification_v2.training.trainer import (
    _behavior_class_weights,
    _build_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DG1_CONFIG = (
    PROJECT_ROOT
    / "configs/classification_v2/m1_dg1_date_adversarial_v1.json"
)


def test_grl_reverses_hidden_gradient_but_not_head_parameter_gradients() -> None:
    hidden_values = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 10.0
    targets = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    head = DateDomainHead(
        6,
        hidden_dim=8,
        num_domains=3,
        dropout=0.0,
    )

    plain_hidden = hidden_values.clone().requires_grad_(True)
    plain_loss = F.cross_entropy(head(plain_hidden), targets)
    plain_loss.backward()
    plain_hidden_gradient = plain_hidden.grad.detach().clone()
    plain_parameter_gradients = [
        parameter.grad.detach().clone() for parameter in head.parameters()
    ]

    head.zero_grad(set_to_none=True)
    reversed_hidden = hidden_values.clone().requires_grad_(True)
    reversed_loss = F.cross_entropy(
        head(gradient_reverse(reversed_hidden)),
        targets,
    )
    reversed_loss.backward()

    torch.testing.assert_close(
        reversed_hidden.grad,
        -plain_hidden_gradient,
        rtol=1e-6,
        atol=1e-7,
    )
    for plain_gradient, parameter in zip(
        plain_parameter_gradients,
        head.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            parameter.grad,
            plain_gradient,
            rtol=1e-6,
            atol=1e-7,
        )


def test_authoritative_cpu_fast_path_and_validation_inference() -> None:
    config = load_training_config(DG1_CONFIG)
    m0_model_config = replace(
        config.model,
        model_mode="full_multimodal",
        enable_date_adversarial=False,
    )
    m0_config = replace(config, model=m0_model_config)
    device = torch.device("cpu")

    with StrictTrainingDataModule(config, device=device) as dg1_data:
        with StrictTrainingDataModule(m0_config, device=device) as m0_data:
            assert dg1_data.window_major_reader is not None
            assert m0_data.window_major_reader is not None
            assert dg1_data.audit()["rows"] == 33287
            assert len(dg1_data.split_indices("train")) == 27834
            assert len(dg1_data.split_indices("validation")) == 5453
            assert len(dg1_data.recording_date_to_index) == 12
            assert "281119" not in dg1_data.recording_date_to_index
            assert dg1_data.date_domain_audit["train_rows"] == 27834

            dg1_data.fit_fold_preprocessor()
            m0_data.fit_fold_preprocessor()
            train_indices = dg1_data.split_indices("train")[:1]
            validation_indices = dg1_data.split_indices("validation")[:1]
            dg1_train = dg1_data.batch(train_indices)
            m0_train = m0_data.batch(train_indices)
            _assert_nested_tensor_parity(
                dg1_train.model_inputs,
                m0_train.model_inputs,
            )
            torch.testing.assert_close(
                dg1_train.behavior_target,
                m0_train.behavior_target,
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                dg1_train.sample_weight,
                m0_train.sample_weight,
                rtol=0.0,
                atol=0.0,
            )
            assert dg1_train.recording_date_target is not None
            assert dg1_train.metadata["recording_date_target_source"] == (
                "FULL_T6_train_manifest_video_key"
            )

            model = _build_model(config, dg1_train, dg1_data).to(device)
            model.train()
            output = model(**dg1_train.model_inputs)
            assert output.behavior.shape == (1, 10)
            assert output.domain is not None
            assert output.domain.shape == (1, 12)
            assert torch.isfinite(output.behavior).all()
            assert torch.isfinite(output.domain).all()

            behavior_weights = _behavior_class_weights(
                dg1_data,
                dg1_data.split_indices("train"),
                config,
                device,
            )
            m0_behavior_weights = _behavior_class_weights(
                m0_data,
                m0_data.split_indices("train"),
                m0_config,
                device,
            )
            torch.testing.assert_close(
                behavior_weights,
                m0_behavior_weights,
                rtol=0.0,
                atol=0.0,
            )
            behavior_per_row = F.cross_entropy(
                output.behavior,
                dg1_train.behavior_target,
                weight=behavior_weights,
                reduction="none",
            )
            behavior_loss = (
                behavior_per_row * dg1_train.sample_weight
            ).sum() / dg1_train.sample_weight.sum().clamp_min(1e-8)
            domain_loss = F.cross_entropy(
                output.domain,
                dg1_train.recording_date_target,
            )
            total_loss = behavior_loss + 0.10 * domain_loss
            assert torch.isfinite(behavior_loss)
            assert torch.isfinite(domain_loss)
            assert torch.isfinite(total_loss)
            total_loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            assert gradients
            assert all(torch.isfinite(gradient).all() for gradient in gradients)
            assert all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                for parameter in model.domain_head.parameters()
            )

            model.eval()
            validation_batch = dg1_data.batch(validation_indices)
            assert validation_batch.recording_date_target is None
            assert validation_batch.metadata["recording_date_token"] == [""]
            assert validation_batch.metadata[
                "recording_date_target_source"
            ] is None
            with torch.no_grad():
                validation_output = model(**validation_batch.model_inputs)
            assert validation_output.behavior.shape == (1, 10)
            assert torch.isfinite(validation_output.behavior).all()
            assert validation_output.domain is None


def _assert_nested_tensor_parity(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_tensor_parity(left[key], right[key])
        return
    assert left == right
