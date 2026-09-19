from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from pig_behavior.classification_v2.contracts.behavior_posture import (
    BEHAVIOR_POSTURE_CONTRACT_VERSION,
    POSTURE_LABEL_ORDER,
    build_burst_posture_authority,
    derive_safe_posture,
    expand_posture_authority_to_windows,
)
from pig_behavior.classification_v2.models.multitask_heads import (
    AUXILIARY_LABEL_ORDER,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.data_module import MODEL_INPUT_KEYS
from pig_behavior.classification_v2.training.multitask_loss import (
    build_auxiliary_label_maps,
    encode_auxiliary_batch,
    hierarchy_consistency_loss,
)


def test_safe_posture_derivations_are_bounded() -> None:
    assert derive_safe_posture("lying") == "lying"
    assert derive_safe_posture("sitting") == "sitting"
    assert derive_safe_posture("stand") == "upright"
    assert derive_safe_posture("eat") == "upright"
    assert derive_safe_posture("social-nose") is None
    assert derive_safe_posture("fight") is None
    assert derive_safe_posture("drink") is None


def test_original_provisional_behavior_cannot_create_posture_authority() -> None:
    bursts = pd.DataFrame(
        {
            "native_temporal_unit_key": ["eat-unit"],
            "behavior_target": ["eat"],
        }
    )

    with pytest.raises(ValueError, match="requires frozen reviewed or synthetic"):
        build_burst_posture_authority(
            bursts,
            behavior_label_authority="ORIGINAL_PROVISIONAL",
        )


def test_burst_authority_masks_active_behavior_posture_until_resolved() -> None:
    bursts = pd.DataFrame(
        {
            "native_temporal_unit_key": [
                "lying-unit",
                "sitting-unit",
                "stand-unit",
                "eat-unit",
                "social-unit",
                "fight-unit",
            ],
            "behavior_target": [
                "lying",
                "sitting",
                "stand",
                "eat",
                "social-nose",
                "fight",
            ],
        }
    )

    authority, audit = build_burst_posture_authority(
        bursts,
        behavior_label_authority="SYNTHETIC_TEST",
    )
    indexed = authority.set_index("native_temporal_unit_key")

    assert indexed.at["lying-unit", "posture_target"] == "lying"
    assert indexed.at["sitting-unit", "posture_target"] == "sitting"
    assert indexed.at["stand-unit", "posture_target"] == "upright"
    assert indexed.at["eat-unit", "posture_target"] == "upright"
    assert indexed.at["social-unit", "posture_target"] == ""
    assert not bool(indexed.at["social-unit", "posture_valid_mask"])
    assert indexed.at["social-unit", "posture_authority"] == "UNRESOLVED"
    assert audit["valid_rows"] == 4
    assert audit["unresolved_rows"] == 2


def test_reviewed_social_posture_can_be_sitting_without_changing_behavior() -> None:
    bursts = pd.DataFrame(
        {
            "native_temporal_unit_key": ["social-unit"],
            "behavior_target": ["social-nose"],
        }
    )
    overrides = pd.DataFrame(
        {
            "native_temporal_unit_key": ["social-unit"],
            "posture_target": ["sitting"],
            "posture_valid_mask": [True],
            "posture_transition_flag": [False],
            "posture_authority": ["HUMAN_REVIEWED"],
        }
    )

    authority, _ = build_burst_posture_authority(
        bursts,
        overrides,
        behavior_label_authority="SYNTHETIC_TEST",
    )

    assert authority.loc[0, "behavior_target"] == "social-nose"
    assert authority.loc[0, "posture_target"] == "sitting"
    assert bool(authority.loc[0, "posture_valid_mask"])


def test_override_cannot_contradict_fixed_feeder_eat_posture() -> None:
    bursts = pd.DataFrame(
        {
            "native_temporal_unit_key": ["eat-unit"],
            "behavior_target": ["eat"],
        }
    )
    overrides = pd.DataFrame(
        {
            "native_temporal_unit_key": ["eat-unit"],
            "posture_target": ["sitting"],
            "posture_valid_mask": [True],
            "posture_transition_flag": [False],
            "posture_authority": ["HUMAN_REVIEWED"],
        }
    )

    with pytest.raises(ValueError, match="contradicts safe posture"):
        build_burst_posture_authority(
            bursts,
            overrides,
            behavior_label_authority="SYNTHETIC_TEST",
        )


def test_burst_authority_expands_only_through_explicit_anchor_key() -> None:
    bursts = pd.DataFrame(
        {
            "native_temporal_unit_key": ["unit-a", "unit-b"],
            "behavior_target": ["eat", "social-nose"],
        }
    )
    authority, _ = build_burst_posture_authority(
        bursts,
        behavior_label_authority="SYNTHETIC_TEST",
    )
    windows = pd.DataFrame(
        {
            "window_id": ["window-a1", "window-a2", "window-b1"],
            "anchor_native_temporal_unit_key": ["unit-a", "unit-a", "unit-b"],
            "behavior_target": ["eat", "eat", "social-nose"],
        }
    )

    expanded = expand_posture_authority_to_windows(windows, authority)

    assert expanded.loc[:1, "posture_target"].eq("upright").all()
    assert expanded.loc[2, "posture_target"] == ""
    assert not bool(expanded.loc[2, "posture_valid_mask"])


def test_masked_unknown_posture_encodes_placeholder_without_supervision() -> None:
    targets = _complete_auxiliary_targets()

    label_maps = build_auxiliary_label_maps(targets)
    encoded, masks = encode_auxiliary_batch(targets, label_maps)

    assert label_maps["posture"] == list(POSTURE_LABEL_ORDER)
    assert encoded["posture"].shape == (4,)
    assert masks["posture"].tolist() == [True, True, True, False]
    assert int(encoded["posture"][3]) == 0


def test_hierarchy_does_not_force_social_posture() -> None:
    behavior_targets = torch.tensor(
        [VALID_BEHAVIORS.index("social-nose"), VALID_BEHAVIORS.index("eat")]
    )
    behavior_logits = _behavior_logits(["social-nose", "eat"])
    masks = _all_task_masks(batch_size=2)
    logits_a = _auxiliary_logits(batch_size=2)
    logits_b = {name: value.clone() for name, value in logits_a.items()}
    logits_a["posture"][0] = torch.tensor([12.0, -12.0, -12.0])
    logits_b["posture"][0] = torch.tensor([-12.0, 12.0, -12.0])

    loss_a = hierarchy_consistency_loss(
        behavior_logits,
        logits_a,
        behavior_targets,
        masks,
    )
    loss_b = hierarchy_consistency_loss(
        behavior_logits,
        logits_b,
        behavior_targets,
        masks,
    )

    assert torch.allclose(loss_a, loss_b, atol=1e-8, rtol=0.0)


def test_hierarchy_enforces_fixed_feeder_eat_upright_relation() -> None:
    behavior_targets = torch.tensor([VALID_BEHAVIORS.index("eat")])
    behavior_logits = _behavior_logits(["eat"])
    masks = _all_task_masks(batch_size=1)
    correct = _auxiliary_logits(batch_size=1)
    incorrect = {name: value.clone() for name, value in correct.items()}
    upright_index = POSTURE_LABEL_ORDER.index("upright")
    lying_index = POSTURE_LABEL_ORDER.index("lying")
    correct["posture"][0, upright_index] = 12.0
    correct["posture"][0, lying_index] = -12.0
    incorrect["posture"][0, upright_index] = -12.0
    incorrect["posture"][0, lying_index] = 12.0

    correct_loss = hierarchy_consistency_loss(
        behavior_logits,
        correct,
        behavior_targets,
        masks,
    )
    incorrect_loss = hierarchy_consistency_loss(
        behavior_logits,
        incorrect,
        behavior_targets,
        masks,
    )

    assert float(correct_loss) < float(incorrect_loss)


def test_posture_contract_matches_machine_json_and_stays_out_of_model_x() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "configs/classification_v2/behavior_posture_contract_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["schema_version"] == BEHAVIOR_POSTURE_CONTRACT_VERSION
    assert tuple(payload["posture_target"]["class_order"]) == POSTURE_LABEL_ORDER
    assert payload["safe_derivations"]["eat"] == "upright"
    assert "posture_target" not in MODEL_INPUT_KEYS
    assert "posture_authority" not in MODEL_INPUT_KEYS


def test_auxiliary_target_cli_exports_masked_independent_posture(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    train_ready = tmp_path / "train_ready"
    train_ready.mkdir()
    labels = pd.Series(VALID_BEHAVIORS, name="behavior_target")
    labels.to_csv(train_ready / "y_behavior.csv", index=False)
    pd.DataFrame(
        {
            "window_id": [f"window-{index}" for index in range(len(labels))],
            "split": ["train"] * len(labels),
            "split_group_key": [f"group-{index}" for index in range(len(labels))],
        }
    ).to_csv(train_ready / "split_manifest.csv", index=False)
    pd.Series([True] * len(labels), name="train_mask").to_csv(
        train_ready / "train_mask.csv",
        index=False,
    )

    builder = root / (
        "scripts/classification_v2/02_train_ready_exports/"
        "classification_v2_build_auxiliary_targets.py"
    )
    checker = root / (
        "scripts/classification_v2/02_train_ready_exports/"
        "check_classification_v2_auxiliary_targets.py"
    )
    loss_checker = root / (
        "scripts/classification_v2/04_baselines_smokes/"
        "check_classification_v2_multitask_loss.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(builder),
            "--root",
            str(train_ready),
            "--behavior-label-authority",
            "SYNTHETIC_TEST",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(loss_checker),
            "--root",
            str(train_ready),
            "--output-json",
            str(train_ready / "multitask_loss_audit.json"),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(checker),
            "--csv",
            str(train_ready / "y_auxiliary_targets.csv"),
            "--audit-json",
            str(train_ready / "auxiliary_targets_audit.json"),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    targets = pd.read_csv(train_ready / "y_auxiliary_targets.csv").fillna("")
    indexed = targets.set_index("behavior_target")

    assert indexed.at["eat", "posture_target"] == "upright"
    assert bool(indexed.at["eat", "has_posture_aux_target"])
    assert indexed.at["social-nose", "posture_target"] == ""
    assert not bool(indexed.at["social-nose", "has_posture_aux_target"])


def _complete_auxiliary_targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "posture_target": ["lying", "sitting", "upright", ""],
            "has_posture_aux_target": [True, True, True, False],
            "motion_context_target": ["move", "explore", "stand", "other"],
            "has_motion_context_aux_target": [True] * 4,
            "roi_intent_target": ["eat", "drink", "playwithtoy", "none"],
            "has_roi_intent_aux_target": [True] * 4,
            "interaction_target": ["fight", "social-nose", "none", "none"],
            "has_interaction_aux_target": [True] * 4,
        }
    )


def _behavior_logits(labels: list[str]) -> torch.Tensor:
    logits = torch.full((len(labels), len(VALID_BEHAVIORS)), -8.0)
    for row, label in enumerate(labels):
        logits[row, VALID_BEHAVIORS.index(label)] = 8.0
    return logits


def _all_task_masks(*, batch_size: int) -> dict[str, torch.Tensor]:
    return {
        name: torch.ones(batch_size, dtype=torch.bool)
        for name in AUXILIARY_LABEL_ORDER
    }


def _auxiliary_logits(*, batch_size: int) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros((batch_size, len(labels)), dtype=torch.float32)
        for name, labels in AUXILIARY_LABEL_ORDER.items()
    }
