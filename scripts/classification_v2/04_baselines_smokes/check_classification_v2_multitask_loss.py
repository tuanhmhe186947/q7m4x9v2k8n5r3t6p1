from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from pig_behavior.classification_v2.models.multitask_heads import (
    AuxiliaryHeadConfig,
    AuxiliaryPredictionHeads,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.multitask_loss import (
    DEFAULT_AUXILIARY_TASKS,
    build_auxiliary_label_maps,
    build_fold_auxiliary_class_weights,
    encode_auxiliary_batch,
    hierarchy_consistency_loss,
    masked_cross_entropy,
    masked_multitask_loss,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check classification_v2 masked auxiliary multitask loss contract."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows/multitask_loss_audit.json"),
    )
    parser.add_argument("--batch-rows-per-task", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    targets = pd.read_csv(args.root / "y_auxiliary_targets.csv", low_memory=False)
    batch = _sample_auxiliary_batch(targets, args.batch_rows_per_task)
    label_maps = build_auxiliary_label_maps(targets)
    training_targets = targets[_to_bool(targets["aux_include_in_training"])].copy()
    class_weights = build_fold_auxiliary_class_weights(training_targets, label_maps)
    encoded_targets, masks = encode_auxiliary_batch(batch, label_maps)

    heads = AuxiliaryPredictionHeads(
        input_dim=args.embedding_dim,
        heads=[
            AuxiliaryHeadConfig(name=name, num_classes=len(labels))
            for name, labels in label_maps.items()
        ],
    )
    embedding = torch.randn(len(batch), args.embedding_dim)
    logits = heads(embedding)
    total_loss, loss_audit = masked_multitask_loss(
        logits,
        encoded_targets,
        masks,
        class_weights_by_task=class_weights,
    )
    all_masked_zero = _check_all_masked_zero(logits, encoded_targets)
    inactive_shuffle_delta = _inactive_target_shuffle_delta(logits, encoded_targets, masks)
    behavior_logits = torch.randn(len(batch), 10)
    consistency_loss = hierarchy_consistency_loss(
        behavior_logits,
        logits,
        torch.tensor(
            [VALID_BEHAVIORS.index(value) for value in batch["behavior_target"]],
            dtype=torch.long,
        ),
        masks,
    )

    errors: list[str] = []
    if not torch.isfinite(total_loss):
        errors.append("total_loss_nonfinite")
    if float(total_loss.detach().cpu().item()) <= 0.0:
        errors.append("total_loss_not_positive")
    if not all_masked_zero:
        errors.append("all_masked_loss_not_zero")
    if inactive_shuffle_delta > 1e-7:
        errors.append(f"inactive_target_shuffle_changed_loss={inactive_shuffle_delta}")
    if not torch.isfinite(consistency_loss):
        errors.append("hierarchy_consistency_loss_nonfinite")
    for spec in DEFAULT_AUXILIARY_TASKS:
        if int(masks[spec.name].sum()) <= 0:
            errors.append(f"no_positive_mask_support:{spec.name}")

    audit = {
        "root": str(args.root),
        "rows": int(len(targets)),
        "batch_rows": int(len(batch)),
        "label_maps": label_maps,
        "loss": loss_audit,
        "fold_local_class_weights": {
            name: value.detach().cpu().tolist()
            for name, value in class_weights.items()
        },
        "class_weights_derived_from_training_rows_only": True,
        "class_weight_training_rows": int(len(training_targets)),
        "all_masked_loss_zero": bool(all_masked_zero),
        "inactive_target_shuffle_max_loss_delta": inactive_shuffle_delta,
        "hierarchy_consistency_loss": float(consistency_loss.detach().cpu().item()),
        "auxiliary_targets_are_y_only": True,
        "forbidden_as_model_input": [
            "behavior_target",
            "posture_target",
            "motion_context_target",
            "roi_intent_target",
            "interaction_target",
            "has_*_aux_target",
        ],
        "errors": errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


def _sample_auxiliary_batch(targets: pd.DataFrame, per_task: int) -> pd.DataFrame:
    indices: list[int] = []
    for spec in DEFAULT_AUXILIARY_TASKS:
        mask = _to_bool(targets[spec.mask_column])
        sample = targets[mask].sort_values(["window_id"]).head(per_task)
        indices.extend(sample.index.tolist())
    return targets.loc[sorted(set(indices))].reset_index(drop=True)


def _check_all_masked_zero(
    logits: dict[str, torch.Tensor],
    encoded_targets: dict[str, torch.Tensor],
) -> bool:
    for name, value in logits.items():
        mask = torch.zeros_like(encoded_targets[name], dtype=torch.bool)
        loss = masked_cross_entropy(value, encoded_targets[name], mask)
        if float(loss.detach().cpu().item()) != 0.0:
            return False
    return True


def _inactive_target_shuffle_delta(
    logits: dict[str, torch.Tensor],
    encoded_targets: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
) -> float:
    deltas: list[float] = []
    for name, value in logits.items():
        original = masked_cross_entropy(value, encoded_targets[name], masks[name])
        shuffled = encoded_targets[name].clone()
        inactive = ~masks[name]
        if inactive.any():
            shuffled[inactive] = torch.flip(shuffled[inactive], dims=[0])
        changed = masked_cross_entropy(value, shuffled, masks[name])
        deltas.append(abs(float((original - changed).detach().cpu().item())))
    return max(deltas, default=0.0)


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


if __name__ == "__main__":
    main()
