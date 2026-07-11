from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pig_behavior.classification_v2.models.multimodal_fusion import MultimodalFusionConfig
from pig_behavior.classification_v2.models.multitask_fusion import MultitaskFusionClassifier
from pig_behavior.classification_v2.models.multitask_heads import AUXILIARY_LABEL_ORDER


def main() -> None:
    parser = argparse.ArgumentParser(description="Check multitask fusion shapes, gradients, and behavior parity.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_smoke/multitask_forward_audit.json"),
    )
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    batch_size, sequence_length = 4, 6
    model = MultitaskFusionClassifier(
        MultimodalFusionConfig(
            spatial_input_dims={"bbox_xywh_n": 4, "motion_delta": 3},
            num_classes=10,
            interaction_context_dim=5,
            image_embedding_dim=16,
            spatial_embedding_dim=16,
            interaction_embedding_dim=8,
            visual_context_embedding_dim=16,
            fusion_hidden_dim=24,
            dropout=0.0,
            enable_visual_context=True,
        )
    )
    model_inputs = _synthetic_inputs(batch_size, sequence_length)
    model.eval()
    output = model(**model_inputs)
    behavior_direct = model.backbone(**model_inputs)
    parity_delta = float((output.behavior - behavior_direct).abs().max().item())
    shape_report = {
        "behavior": list(output.behavior.shape),
        **{name: list(logits.shape) for name, logits in output.auxiliary_logits().items()},
    }
    expected_shapes = {
        "behavior": [batch_size, 10],
        **{name: [batch_size, len(labels)] for name, labels in AUXILIARY_LABEL_ORDER.items()},
    }

    model.train()
    train_output = model(**model_inputs)
    objective = train_output.behavior.square().mean()
    objective = objective + sum(value.square().mean() for value in train_output.auxiliary_logits().values())
    objective.backward()
    shared_gradient = _gradient_sum(model.backbone.image_encoder)
    head_gradients = {
        name: _gradient_sum(head) for name, head in model.auxiliary_heads.heads.items()
    }
    errors: list[str] = []
    if shape_report != expected_shapes:
        errors.append(f"shape_contract_mismatch={shape_report}")
    if parity_delta > 1e-7:
        errors.append(f"behavior_only_parity_delta={parity_delta}")
    if shared_gradient <= 0.0:
        errors.append("no_gradient_to_shared_image_encoder")
    for name, value in head_gradients.items():
        if value <= 0.0:
            errors.append(f"no_gradient_to_auxiliary_head={name}")
    audit = {
        "schema_version": "classification_v2_multitask_forward_audit_v1",
        "shape_report": shape_report,
        "expected_shapes": expected_shapes,
        "behavior_only_parity_max_abs_delta": parity_delta,
        "shared_image_encoder_gradient_abs_sum": shared_gradient,
        "auxiliary_head_gradient_abs_sum": head_gradients,
        "auxiliary_targets_used_as_model_inputs": False,
        "errors": errors,
        "valid": not errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if errors:
        raise SystemExit(1)


def _synthetic_inputs(batch_size: int, sequence_length: int) -> dict[str, object]:
    length_mask = torch.ones(batch_size, sequence_length)
    observed_mask = length_mask.clone()
    observed_mask[-1, -2:] = 0.0
    return {
        "image": torch.rand(batch_size, sequence_length, 3, 32, 32),
        "spatial_features": {
            "bbox_xywh_n": torch.rand(batch_size, sequence_length, 4),
            "motion_delta": torch.rand(batch_size, sequence_length, 3),
        },
        "length_mask": length_mask,
        "observed_mask": observed_mask,
        "interaction_context_features": torch.rand(batch_size, 5),
        "interaction_context_available_mask": torch.tensor([1.0, 1.0, 0.0, 1.0]),
        "visual_context_image": torch.rand(batch_size, sequence_length, 3, 32, 32),
        "visual_context_length_mask": length_mask,
        "visual_context_observed_mask": observed_mask,
    }


def _gradient_sum(module: torch.nn.Module | None) -> float:
    if module is None:
        return 0.0
    return float(
        sum(
            parameter.grad.detach().abs().sum().item()
            for parameter in module.parameters()
            if parameter.grad is not None
        )
    )


if __name__ == "__main__":
    main()
