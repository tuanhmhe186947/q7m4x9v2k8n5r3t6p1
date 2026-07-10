"""Masked multitask loss helpers for classification_v2.

Auxiliary targets are deterministic decompositions of the main behavior label,
so they are supervised outputs only. They must never be fed into X. The masked
loss below lets posture/motion/ROI/interaction heads learn only from samples
where that auxiliary label is meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class AuxiliaryTaskSpec:
    """Column and weight contract for one auxiliary task."""

    name: str
    target_column: str
    mask_column: str
    loss_weight: float = 1.0


DEFAULT_AUXILIARY_TASKS = (
    AuxiliaryTaskSpec("posture", "posture_target", "has_posture_aux_target", 0.25),
    AuxiliaryTaskSpec("motion_context", "motion_context_target", "has_motion_context_aux_target", 0.25),
    AuxiliaryTaskSpec("roi_intent", "roi_intent_target", "has_roi_intent_aux_target", 0.25),
    AuxiliaryTaskSpec("interaction", "interaction_target", "has_interaction_aux_target", 0.25),
)


def build_auxiliary_label_maps(
    targets: pd.DataFrame,
    task_specs: tuple[AuxiliaryTaskSpec, ...] = DEFAULT_AUXILIARY_TASKS,
) -> dict[str, list[str]]:
    """Build deterministic per-task label order from the auxiliary target table."""
    label_maps: dict[str, list[str]] = {}
    for spec in task_specs:
        _require_columns(targets, [spec.target_column, spec.mask_column])
        labels = targets[spec.target_column].fillna("").astype(str)
        masked_labels = labels[_to_bool(targets[spec.mask_column])]
        observed = sorted(set(masked_labels.tolist()) | set(labels.tolist()))
        if len(observed) <= 1:
            raise ValueError(f"auxiliary task {spec.name} has <=1 class: {observed}")
        label_maps[spec.name] = observed
    return label_maps


def encode_auxiliary_batch(
    targets: pd.DataFrame,
    label_maps: dict[str, list[str]],
    task_specs: tuple[AuxiliaryTaskSpec, ...] = DEFAULT_AUXILIARY_TASKS,
    *,
    device: torch.device | str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Encode target labels and masks for a batch of auxiliary rows."""
    encoded_targets: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    for spec in task_specs:
        labels = label_maps[spec.name]
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        raw = targets[spec.target_column].fillna("").astype(str)
        unknown = sorted(set(raw).difference(label_to_idx))
        if unknown:
            raise ValueError(f"unknown labels for {spec.name}: {unknown}")
        encoded_values = [label_to_idx[value] for value in raw]
        encoded_targets[spec.name] = torch.tensor(encoded_values, dtype=torch.long, device=device)
        mask_values = _to_bool(targets[spec.mask_column]).to_numpy()
        masks[spec.name] = torch.tensor(mask_values, dtype=torch.bool, device=device)
    return encoded_targets, masks


def masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross entropy over valid samples only; all-masked batches return zero."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [B, C]")
    if target.shape != mask.shape or target.ndim != 1:
        raise ValueError("target and mask must both have shape [B]")
    if logits.shape[0] != target.shape[0]:
        raise ValueError("logits batch dimension must match target")
    valid = mask.bool()
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], target[valid])


def masked_multitask_loss(
    logits_by_task: dict[str, torch.Tensor],
    targets_by_task: dict[str, torch.Tensor],
    masks_by_task: dict[str, torch.Tensor],
    task_specs: tuple[AuxiliaryTaskSpec, ...] = DEFAULT_AUXILIARY_TASKS,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return weighted auxiliary loss and a small audit dictionary."""
    losses: dict[str, torch.Tensor] = {}
    support: dict[str, int] = {}
    weighted_terms: list[torch.Tensor] = []
    for spec in task_specs:
        missing = [
            name
            for name, source in [
                (spec.name, logits_by_task),
                (spec.name, targets_by_task),
                (spec.name, masks_by_task),
            ]
            if name not in source
        ]
        if missing:
            raise ValueError(f"missing multitask tensors for {spec.name}")
        loss = masked_cross_entropy(logits_by_task[spec.name], targets_by_task[spec.name], masks_by_task[spec.name])
        losses[spec.name] = loss
        support[spec.name] = int(masks_by_task[spec.name].bool().sum().detach().cpu().item())
        weighted_terms.append(loss * float(spec.loss_weight))
    total = torch.stack(weighted_terms).sum()
    audit = {
        "task_names": [spec.name for spec in task_specs],
        "loss_weights": {spec.name: float(spec.loss_weight) for spec in task_specs},
        "support": support,
        "loss_values": {name: float(value.detach().cpu().item()) for name, value in losses.items()},
        "total_auxiliary_loss": float(total.detach().cpu().item()),
    }
    return total, audit


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"missing auxiliary target columns: {missing}")


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})
