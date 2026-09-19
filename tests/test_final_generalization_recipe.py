from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pig_behavior.classification_v2.training.config import load_training_config
from pig_behavior.classification_v2.training.generalization import (
    apply_locked_recipe,
    build_locked_scheduler,
    load_locked_generalization_recipe,
    video_safe_photometric_augment,
)
from pig_behavior.classification_v2.training.run_joint_representation_5fold import (
    _select_preflight_keys,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = REPO_ROOT / "configs/classification_v2/final_generalization_recipe_v1.json"


def test_locked_recipe_overlays_only_generalization_fields() -> None:
    config = load_training_config(
        REPO_ROOT / "configs/classification_v2/g2_cclsa_earlystop_vg1_scientific_v1.json"
    )
    recipe = load_locked_generalization_recipe(
        RECIPE_PATH,
        expected_control_commit="b6a9fdaeb17db50b939fc6cc45af3c9d528c093d",
    )
    refined = apply_locked_recipe(config, recipe)

    assert refined.dataset.augmentation_policy == "video_safe_photometric_v1"
    assert refined.optimization.weight_decay == 1e-4
    assert refined.optimization.scheduler == "cosine_floor_0.1"
    assert refined.optimization.seed == config.optimization.seed
    assert refined.model == config.model
    assert refined.loss == config.loss
    assert refined.dataset.native_oof_fold_manifest == config.dataset.native_oof_fold_manifest


def test_augmentation_is_shared_across_rgb_streams_and_preserves_masked_slots() -> None:
    recipe = load_locked_generalization_recipe(
        RECIPE_PATH,
        expected_control_commit="b6a9fdaeb17db50b939fc6cc45af3c9d528c093d",
    )
    rgb = torch.tensor(
        [[[[[0.2, 0.4], [0.6, 0.8]], [[0.3, 0.5], [0.7, 0.9]], [[0.1, 0.2], [0.4, 0.6]]]]],
        dtype=torch.float32,
    )
    inputs = {
        "image": rgb.clone(),
        "visual_context_image": rgb.clone(),
        "image_available_mask": torch.tensor([[1.0]]),
        "visual_context_available_mask": torch.tensor([[1.0]]),
    }

    torch.manual_seed(7)
    augmented = video_safe_photometric_augment(inputs, recipe)
    assert torch.allclose(augmented["image"], augmented["visual_context_image"])
    assert not torch.equal(augmented["image"], inputs["image"])

    masked_inputs = {
        **inputs,
        "image": torch.cat([rgb, torch.zeros_like(rgb)], dim=1),
        "visual_context_image": torch.cat([rgb, torch.zeros_like(rgb)], dim=1),
        "image_available_mask": torch.tensor([[1.0, 0.0]]),
        "visual_context_available_mask": torch.tensor([[1.0, 0.0]]),
    }
    torch.manual_seed(7)
    masked_augmented = video_safe_photometric_augment(masked_inputs, recipe)
    assert torch.equal(masked_augmented["image"][:, 1], masked_inputs["image"][:, 1])
    assert torch.equal(
        masked_augmented["visual_context_image"][:, 1],
        masked_inputs["visual_context_image"][:, 1],
    )


def test_scheduler_reaches_floor_for_each_parameter_group() -> None:
    recipe = load_locked_generalization_recipe(
        RECIPE_PATH,
        expected_control_commit="b6a9fdaeb17db50b939fc6cc45af3c9d528c093d",
    )
    first = torch.nn.Parameter(torch.tensor(1.0))
    second = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW(
        [
            {"params": [first], "lr": 0.003, "weight_decay": 1e-4},
            {"params": [second], "lr": 0.0003, "weight_decay": 1e-4},
        ]
    )
    scheduler = build_locked_scheduler(optimizer, recipe)
    initial = [group["lr"] for group in optimizer.param_groups]
    for _ in range(recipe.max_epochs):
        optimizer.step()
        scheduler.step()
    final = [group["lr"] for group in optimizer.param_groups]
    assert final == pytest.approx([value * 0.1 for value in initial])
    assert final[0] / final[1] == pytest.approx(initial[0] / initial[1])


def test_preflight_spread_selection_is_deterministic_and_label_independent() -> None:
    keys = tuple(f"key-{index}" for index in range(10))

    selected = _select_preflight_keys(keys, limit=4, selection="spread")

    assert selected == ("key-0", "key-3", "key-6", "key-9")
    assert selected != keys[:4]
