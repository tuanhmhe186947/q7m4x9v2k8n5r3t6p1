"""Masked multitask loss helpers for Classification V2.

Posture may be independent of the main behavior label. Auxiliary targets are
supervised outputs only and must never be fed into X. Per-task masks decide
whether a target and a hierarchy relation are authoritative for each sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from pig_behavior.classification_v2.contracts.behavior_posture import (
    SAFE_POSTURE_BY_BEHAVIOR,
)
from pig_behavior.classification_v2.models.multitask_heads import AUXILIARY_LABEL_ORDER
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS


@dataclass(frozen=True, slots=True)
class AuxiliaryTaskSpec:
    """Column and weight contract for one auxiliary task."""

    name: str
    target_column: str
    mask_column: str
    loss_weight: float = 1.0


DEFAULT_AUXILIARY_TASKS = (
    AuxiliaryTaskSpec("posture", "posture_target", "has_posture_aux_target", 0.25),
    AuxiliaryTaskSpec(
        "motion_context",
        "motion_context_target",
        "has_motion_context_aux_target",
        0.25,
    ),
    AuxiliaryTaskSpec(
        "roi_intent",
        "roi_intent_target",
        "has_roi_intent_aux_target",
        0.25,
    ),
    AuxiliaryTaskSpec(
        "interaction",
        "interaction_target",
        "has_interaction_aux_target",
        0.25,
    ),
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
        masks = _to_bool(targets[spec.mask_column])
        expected = AUXILIARY_LABEL_ORDER[spec.name]
        observed = set(labels[labels.ne("")].tolist())
        observed_active = set(labels[masks].tolist())
        unexpected = sorted(observed.difference(expected))
        missing = sorted(set(expected).difference(observed_active))
        masked_empty = int((masks & labels.eq("")).sum())
        if unexpected or missing or masked_empty:
            raise ValueError(
                f"auxiliary label contract mismatch for {spec.name}: "
                f"unexpected={unexpected}, missing={missing}, "
                f"masked_empty={masked_empty}"
            )
        label_maps[spec.name] = list(expected)
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
        mask_values = _to_bool(targets[spec.mask_column])
        unknown = sorted(set(raw[raw.ne("")]).difference(label_to_idx))
        if unknown:
            raise ValueError(f"unknown labels for {spec.name}: {unknown}")
        masked_empty = mask_values & raw.eq("")
        if masked_empty.any():
            raise ValueError(
                f"masked auxiliary target is empty for {spec.name}: "
                f"count={int(masked_empty.sum())}"
            )
        placeholder = labels[0]
        encoded_values = [label_to_idx[value or placeholder] for value in raw]
        encoded_targets[spec.name] = torch.tensor(
            encoded_values,
            dtype=torch.long,
            device=device,
        )
        masks[spec.name] = torch.tensor(
            mask_values.to_numpy(),
            dtype=torch.bool,
            device=device,
        )
    return encoded_targets, masks


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
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
    return F.cross_entropy(logits[valid], target[valid], weight=class_weight)


def masked_multitask_loss(
    logits_by_task: dict[str, torch.Tensor],
    targets_by_task: dict[str, torch.Tensor],
    masks_by_task: dict[str, torch.Tensor],
    task_specs: tuple[AuxiliaryTaskSpec, ...] = DEFAULT_AUXILIARY_TASKS,
    class_weights_by_task: dict[str, torch.Tensor] | None = None,
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
        loss = masked_cross_entropy(
            logits_by_task[spec.name],
            targets_by_task[spec.name],
            masks_by_task[spec.name],
            class_weight=(class_weights_by_task or {}).get(spec.name),
        )
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


def hierarchy_consistency_loss(
    behavior_logits: torch.Tensor,
    auxiliary_logits: dict[str, torch.Tensor],
    behavior_targets: torch.Tensor,
    auxiliary_masks: dict[str, torch.Tensor],
    *,
    behavior_label_order: tuple[str, ...] = tuple(VALID_BEHAVIORS),
) -> torch.Tensor:
    """Align only behavior-to-auxiliary relations that are authoritative."""

    if behavior_logits.ndim != 2 or behavior_logits.shape[1] != len(behavior_label_order):
        raise ValueError("behavior logits do not match behavior label order")
    if behavior_targets.shape != (behavior_logits.shape[0],):
        raise ValueError("behavior targets must have shape [B]")
    behavior_prob = torch.softmax(behavior_logits.float(), dim=1)
    terms: list[torch.Tensor] = []
    for task_name, labels in AUXILIARY_LABEL_ORDER.items():
        logits = auxiliary_logits.get(task_name)
        if logits is None or logits.shape != (behavior_logits.shape[0], len(labels)):
            raise ValueError(f"auxiliary logits contract mismatch for {task_name}")
        derived = _aggregate_behavior_probabilities(
            behavior_prob,
            behavior_label_order=behavior_label_order,
            task_name=task_name,
            auxiliary_labels=labels,
        )
        auxiliary_prob = torch.softmax(logits.float(), dim=1)
        task_mask = auxiliary_masks.get(task_name)
        if task_mask is None or task_mask.shape != behavior_targets.shape:
            raise ValueError(f"auxiliary hierarchy mask mismatch for {task_name}")
        valid = task_mask.bool()
        if task_name == "posture":
            valid = valid & _safe_posture_behavior_mask(
                behavior_targets,
                behavior_label_order=behavior_label_order,
            )
        if bool(valid.any()):
            terms.append(F.mse_loss(auxiliary_prob[valid], derived[valid]))
    if not terms:
        return behavior_logits.sum() * 0.0
    return torch.stack(terms).mean()


def build_fold_auxiliary_class_weights(
    targets: pd.DataFrame,
    label_maps: dict[str, list[str]],
    task_specs: tuple[AuxiliaryTaskSpec, ...] = DEFAULT_AUXILIARY_TASKS,
    *,
    power: float = 0.5,
    max_weight: float = 5.0,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Derive deterministic class weights from training-fold auxiliary rows only."""

    if power < 0.0 or max_weight <= 0.0:
        raise ValueError("power must be non-negative and max_weight positive")
    weights: dict[str, torch.Tensor] = {}
    for spec in task_specs:
        labels = label_maps[spec.name]
        active = targets.loc[_to_bool(targets[spec.mask_column]), spec.target_column].astype(str)
        counts = active.value_counts().reindex(labels, fill_value=0).astype(float)
        if (counts <= 0).any():
            missing = counts[counts <= 0].index.tolist()
            raise ValueError(f"training fold missing auxiliary classes for {spec.name}: {missing}")
        inverse = (float(counts.max()) / counts) ** power
        normalized = inverse / float(inverse.mean())
        clipped = normalized.clip(upper=max_weight)
        weights[spec.name] = torch.tensor(clipped.to_numpy(), dtype=torch.float32, device=device)
    return weights


def _aggregate_behavior_probabilities(
    behavior_prob: torch.Tensor,
    *,
    behavior_label_order: tuple[str, ...],
    task_name: str,
    auxiliary_labels: tuple[str, ...],
) -> torch.Tensor:
    mapping = {
        "posture": SAFE_POSTURE_BY_BEHAVIOR,
        "motion_context": {"move": "move", "explore": "explore", "stand": "stand"},
        "roi_intent": {"eat": "eat", "drink": "drink", "playwithtoy": "playwithtoy"},
        "interaction": {"fight": "fight", "social-nose": "social-nose"},
    }[task_name]
    default_label = {
        "posture": None,
        "motion_context": "other",
        "roi_intent": "none",
        "interaction": "none",
    }[task_name]
    target_index = {label: index for index, label in enumerate(auxiliary_labels)}
    out = behavior_prob.new_zeros((behavior_prob.shape[0], len(auxiliary_labels)))
    for behavior_index, behavior_label in enumerate(behavior_label_order):
        auxiliary_label = mapping.get(behavior_label, default_label)
        if auxiliary_label is None:
            continue
        out[:, target_index[auxiliary_label]] += behavior_prob[:, behavior_index]
    return out


def _safe_posture_behavior_mask(
    behavior_targets: torch.Tensor,
    *,
    behavior_label_order: tuple[str, ...],
) -> torch.Tensor:
    safe_indices = [
        index
        for index, label in enumerate(behavior_label_order)
        if label in SAFE_POSTURE_BY_BEHAVIOR
    ]
    mask = torch.zeros_like(behavior_targets, dtype=torch.bool)
    for index in safe_indices:
        mask |= behavior_targets.eq(index)
    return mask


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"missing auxiliary target columns: {missing}")


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})
