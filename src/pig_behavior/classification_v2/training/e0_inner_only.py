"""Fail-closed local execution path for the frozen E0 B3 engineering pilot.

This module deliberately does not call the generic OOF runner.  E0 holds the
registered outer fold inaccessible and uses only that fold's predeclared
``train`` and ``validation`` roles.  It reuses the current B3 model, RGB cache
loader, and immutable spatial shards while accepting only the B3 numerical
groups.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from pig_behavior.classification_v2.datasets.image_sequence_dataset import (
    ClassificationV2ImageSequenceDataset,
    ImageSequenceDatasetConfig,
    image_sequence_collate,
)
from pig_behavior.classification_v2.features.motion_schema import (
    MOTION_SCHEMA_HASH,
    MOTION_SCHEMA_VERSION,
)
from pig_behavior.classification_v2.models.balanced.baselines import (
    baseline_config,
)
from pig_behavior.classification_v2.models.balanced.contracts import (
    ModelBatch,
    SequenceSegment,
    numeric_group_feature_names,
)
from pig_behavior.classification_v2.models.balanced.registry import build_model
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

AUTHORITY_SCHEMA_VERSION = "classification_v2.e0_execution_authority.v1"
CHECKPOINT_SCHEMA_VERSION = "classification_v2.e0_inner_only_checkpoint.v1"
PREDICTION_SCHEMA_VERSION = "classification_v2.e0_inner_only_prediction.v1"
E0_MODEL = "B3_ACTOR_T6_PLUS_GEOMETRY_MOTION"
E0_TEMPORAL_VIEW = "T6"
E0_OUTER_FOLD = "FOLD_3"
E0_SEED = 20260804
E0_NUMERIC_GROUPS = ("bbox_xywh_n", "bbox_shape_n", "motion_delta")
E0_REQUIRED_DATA_REFERENCES = (
    "effective_window_index",
    "spatial_memmap_dir",
    "rgb_root",
    "grouped_fold_roles",
    "event_weight_manifest",
)


class E0ContractError(ValueError):
    """Raised when an E0 authority, path binding, or role is unsafe."""


@dataclass(slots=True)
class E0DataPopulation:
    """Locally resolved E0 records plus read-only RGB and spatial inputs."""

    frame: pd.DataFrame
    image_dataset: ClassificationV2ImageSequenceDataset
    spatial_arrays: dict[str, np.ndarray]
    spatial_manifest: dict[str, Any]

    def close(self) -> None:
        self.image_dataset.close()
        for array in self.spatial_arrays.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()


def load_e0_execution_authority(path: Path) -> dict[str, Any]:
    """Load one authority and reject any drift from the frozen E0 contract."""

    if not path.is_file():
        raise FileNotFoundError(path)
    payload = _read_json_object(path)
    if payload.get("schema_version") != AUTHORITY_SCHEMA_VERSION:
        raise E0ContractError("unsupported E0 execution authority schema")
    _require(payload.get("model") == E0_MODEL, "E0 model identity mismatch")
    _require(
        payload.get("temporal_view") == E0_TEMPORAL_VIEW,
        "E0 temporal view must be T6",
    )
    _require(
        payload.get("outer_fold") == E0_OUTER_FOLD,
        "E0 outer fold must be FOLD_3",
    )
    _require(payload.get("seed") == E0_SEED, "E0 seed mismatch")
    modalities = payload.get("modalities")
    _require(isinstance(modalities, dict), "E0 modalities must be an object")
    expected_modalities = {
        "actor_rgb": True,
        "geometry": True,
        "geometry_dim": 6,
        "motion": True,
        "motion_dim": 12,
        "roi": False,
        "social": False,
        "interaction_context": False,
        "visual_context": False,
        "history": "none",
        "posture_auxiliary": False,
        "availability_controls": False,
        "quality_controls": False,
    }
    _require(
        modalities == expected_modalities,
        "E0 modalities differ from the frozen B3 contract",
    )
    model_config = payload.get("model_config")
    _require(isinstance(model_config, dict), "E0 model_config must be an object")
    _require(
        model_config.get("numeric_groups") == list(E0_NUMERIC_GROUPS),
        "E0 numeric groups must be B3 geometry plus motion only",
    )
    _require(
        model_config.get("include_controls") is False,
        "E0 must not encode ROI/social availability or quality controls",
    )
    expected_model_config = {
        "target_length": 6,
        "hidden_dim": 16,
        "temporal_encoder": "causal_tcn",
        "backbone_name": "smoke_cnn",
        "pretrained_weight_enum": "NONE_RANDOM_INIT",
        "image_size": 64,
        "dropout": 0.0,
    }
    for name, value in expected_model_config.items():
        _require(
            model_config.get(name) == value,
            f"E0 model_config {name} differs from the frozen authority",
        )
    optimization = payload.get("optimization")
    _require(isinstance(optimization, dict), "E0 optimization must be an object")
    for key in (
        "optimizer",
        "learning_rate",
        "weight_decay",
        "train_batch_size",
        "validation_batch_size",
        "training_budget",
        "scheduler",
        "precision",
        "gradient_clip_norm",
        "sample_weight_policy",
        "weight_source",
        "class_weight_power",
        "class_weight_max",
        "checkpoint_every_steps",
        "train_per_class",
        "validation_per_class",
        "smoke_per_class",
        "adamw_betas",
        "adamw_eps",
        "adamw_amsgrad",
    ):
        _require(key in optimization, f"E0 optimization is missing {key}")
    _require(
        optimization["training_budget"] == {"type": "steps", "value": 16},
        "E0 training budget must be the frozen 16-step engineering budget",
    )
    _require(
        optimization["scheduler"] == "none",
        "E0 must not introduce a scheduler",
    )
    expected_optimization = {
        "optimizer": "adamw",
        "learning_rate": 0.003,
        "weight_decay": 0.0,
        "train_batch_size": 16,
        "validation_batch_size": 16,
        "precision": "fp32",
        "gradient_clip_norm": 1.0,
        "sample_weight_policy": "event_class",
        "weight_source": "fold_event_class_sample_weight",
        "class_weight_power": 0.5,
        "class_weight_max": 5.0,
        "checkpoint_every_steps": 4,
        "train_per_class": 8,
        "validation_per_class": 8,
        "smoke_per_class": 1,
        "adamw_betas": [0.9, 0.999],
        "adamw_eps": 1e-08,
        "adamw_amsgrad": False,
    }
    for name, value in expected_optimization.items():
        _require(
            optimization.get(name) == value,
            f"E0 optimization {name} differs from the frozen authority",
        )
    _require(
        payload.get("checkpoint_selection_rule") == "final_registered_endpoint",
        "E0 checkpoint selection must be non-selective",
    )
    outer = payload.get("outer_test_prohibition")
    _require(isinstance(outer, dict), "E0 outer-test prohibition must be an object")
    _require(
        all(outer.get(key) is False for key in ("training", "validation", "metrics", "export")),
        "E0 outer-test prohibition is incomplete",
    )
    data_references = payload.get("data_references")
    _require(isinstance(data_references, dict), "E0 data references must be an object")
    _require(
        set(E0_REQUIRED_DATA_REFERENCES).issubset(data_references),
        "E0 data references are incomplete",
    )
    return payload


def authority_sha256(path: Path) -> str:
    """Return the byte hash used to bind a checkpoint to one authority."""

    return _sha256_file(path)


def verify_e0_execution_sources(
    authority: Mapping[str, Any],
    *,
    repository_root: Path,
) -> None:
    """Reject a launcher whose bounded execution source bytes drifted."""

    declared = authority.get("execution_source_hashes")
    _require(
        isinstance(declared, dict) and declared,
        "E0 execution source hashes are absent",
    )
    for relative_path, expected_hash in declared.items():
        _require(
            isinstance(relative_path, str) and isinstance(expected_hash, str),
            "E0 execution source hash declaration is invalid",
        )
        candidate = repository_root / relative_path
        _require(
            candidate.is_file() and candidate.resolve().is_relative_to(repository_root.resolve()),
            f"E0 execution source path is invalid: {relative_path}",
        )
        _require(
            _sha256_file(candidate) == expected_hash,
            f"E0 execution source hash mismatch: {relative_path}",
        )


def resolve_e0_data_paths(
    authority: Mapping[str, Any],
    *,
    bindings_path: Path | None = None,
    use_authority_local_paths: bool = False,
) -> dict[str, Path]:
    """Resolve portable bindings, or explicit local authority paths for a smoke."""

    bindings: Mapping[str, Any] = {}
    if bindings_path is not None:
        bindings = _read_json_object(bindings_path).get("paths", {})
        _require(isinstance(bindings, dict), "E0 data bindings paths must be an object")
    if bindings_path is None and not use_authority_local_paths:
        raise E0ContractError(
            "E0 needs --data-bindings remotely or --use-authority-local-paths locally"
        )
    result: dict[str, Path] = {}
    references = authority["data_references"]
    for name in E0_REQUIRED_DATA_REFERENCES:
        declared = references[name]
        _require(isinstance(declared, dict), f"invalid E0 data reference={name}")
        raw = bindings.get(name)
        if raw is None and use_authority_local_paths:
            raw = declared.get("local_path")
        _require(isinstance(raw, str) and raw.strip(), f"missing E0 path binding={name}")
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"E0 path binding does not exist: {name}={path}")
        result[name] = path
    return result


def inspect_e0_execution_authority(path: Path) -> dict[str, Any]:
    """Report the real, fail-closed execution settings without opening data."""

    authority = load_e0_execution_authority(path)
    optimization = authority["optimization"]
    return {
        "schema_version": "classification_v2.e0_execution_resolution.v1",
        "authority_sha256": authority_sha256(path),
        "model": authority["model"],
        "temporal_view": authority["temporal_view"],
        "outer_fold": authority["outer_fold"],
        "seed": authority["seed"],
        "modalities": authority["modalities"],
        "optimizer": optimization["optimizer"],
        "learning_rate": optimization["learning_rate"],
        "weight_decay": optimization["weight_decay"],
        "training_budget": optimization["training_budget"],
        "train_batch_size": optimization["train_batch_size"],
        "validation_batch_size": optimization["validation_batch_size"],
        "scheduler": optimization["scheduler"],
        "precision": optimization["precision"],
        "gradient_clip_norm": optimization["gradient_clip_norm"],
        "sample_weight_policy": optimization["sample_weight_policy"],
        "checkpoint_selection_rule": authority["checkpoint_selection_rule"],
        "checkpoint_contract": authority["checkpoint_contract"],
        "prediction_export_contract": authority["prediction_export_contract"],
        "outer_test_access": "BLOCKED",
    }


def load_e0_data_population(
    authority: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> E0DataPopulation:
    """Open only E0 inputs and attach the predeclared FOLD_3 roles."""

    frame = _load_e0_window_frame(paths)
    role_map = _load_fold_three_role_map(paths["grouped_fold_roles"])
    frame["e0_role"] = frame["temporal_unit_keys_json"].map(
        lambda value: _resolve_window_role(value, role_map)
    )
    _require(
        set(frame["e0_role"].unique()).issubset({"train", "validation", "test"}),
        "E0 role resolution produced an unsupported role",
    )
    image_dataset = _build_image_dataset(paths["rgb_root"], authority["model_config"]["image_size"])
    image_lookup = pd.DataFrame(
        {
            "window_id": image_dataset.windows["window_id"].astype(str),
            "dataset_index": np.arange(len(image_dataset.windows), dtype=np.int64),
        }
    )
    frame = frame.merge(image_lookup, on="window_id", how="inner", validate="one_to_one")
    if len(frame) == 0:
        image_dataset.close()
        raise E0ContractError("E0 has no RGB-resolved T6 rows")
    _attach_event_weights(frame, paths["event_weight_manifest"])
    arrays, manifest = _load_b3_spatial_arrays(paths["spatial_memmap_dir"])
    max_row = int(frame["tensor_row_index"].max())
    for group, array in arrays.items():
        if max_row >= len(array):
            image_dataset.close()
            _close_arrays(arrays)
            raise E0ContractError(f"E0 tensor row exceeds {group} spatial array")
    return E0DataPopulation(
        frame=frame.reset_index(drop=True),
        image_dataset=image_dataset,
        spatial_arrays=arrays,
        spatial_manifest=manifest,
    )


def select_e0_role_rows(
    population: E0DataPopulation,
    *,
    role: str,
    per_class: int,
    seed: int,
) -> pd.DataFrame:
    """Choose a deterministic balanced E0 development subset, never test rows."""

    assert_e0_role_permitted(role)
    if per_class <= 0:
        raise E0ContractError("E0 per-class selection must be positive")
    eligible = population.frame.loc[population.frame["e0_role"].eq(role)].copy()
    if eligible.empty:
        raise E0ContractError(f"E0 has no rows for role={role}")
    parts: list[pd.DataFrame] = []
    for offset, label in enumerate(VALID_BEHAVIORS):
        group = eligible.loc[eligible["behavior_label"].eq(label)].sort_values(
            "window_id",
            kind="mergesort",
        )
        if len(group) < per_class:
            raise E0ContractError(
                f"E0 {role} lacks class support for {label}: {len(group)} < {per_class}"
            )
        rng = np.random.default_rng(seed + offset)
        selected_positions = np.sort(rng.choice(len(group), size=per_class, replace=False))
        parts.append(group.iloc[selected_positions])
    selected = pd.concat(parts, ignore_index=True).sort_values(
        "window_id",
        kind="mergesort",
    )
    if selected["e0_role"].eq("test").any():
        raise E0ContractError("E0 selected an outer-test row")
    return selected.reset_index(drop=True)


def assert_e0_role_permitted(role: str) -> None:
    """Fail closed before an E0 path can request a held-out outer-test role."""

    if role not in {"train", "validation"}:
        raise E0ContractError(
            "E0 may select only train or validation roles; outer test is blocked"
        )


def build_e0_model_batch(
    population: E0DataPopulation,
    selected: pd.DataFrame,
    *,
    device: torch.device,
) -> tuple[ModelBatch, dict[str, list[int]]]:
    """Create a B3-only batch from actor RGB, geometry 6D, and motion 12D."""

    loader = DataLoader(
        Subset(population.image_dataset, selected["dataset_index"].tolist()),
        batch_size=len(selected),
        shuffle=False,
        collate_fn=image_sequence_collate,
    )
    image_batch = next(iter(loader))
    errors = [error for item in image_batch["errors"] for error in item]
    if errors:
        raise E0ContractError(f"E0 RGB cache load errors={errors[:8]}")
    expected_ids = selected["window_id"].astype(str).tolist()
    if image_batch["window_id"] != expected_ids:
        raise E0ContractError("E0 RGB window order differs from selected rows")
    row_indices = selected["tensor_row_index"].to_numpy(dtype=np.int64)
    groups = {
        name: torch.from_numpy(np.asarray(population.spatial_arrays[name][row_indices, :6]))
        .float()
        .to(device)
        for name in E0_NUMERIC_GROUPS
    }
    _require(tuple(groups) == E0_NUMERIC_GROUPS, "E0 loaded a non-B3 spatial group")
    expected_dimensions = {"bbox_xywh_n": 4, "bbox_shape_n": 2, "motion_delta": 12}
    observed_dimensions = {name: int(value.shape[-1]) for name, value in groups.items()}
    _require(observed_dimensions == expected_dimensions, "E0 B3 numeric dimensions drifted")
    image = image_batch["image"].to(device)
    observed = image_batch["observed_mask"].to(device)
    labels = torch.tensor(
        [VALID_BEHAVIORS.index(value) for value in selected["behavior_label"]],
        dtype=torch.long,
        device=device,
    )
    target_length = int(image.shape[1])
    _require(target_length == 6, "E0 RGB batch must contain exactly six T6 frames")
    batch = ModelBatch(
        target=SequenceSegment(
            valid_mask=observed,
            frame_offsets=torch.arange(-5, 1, device=device).repeat(len(selected), 1),
            images=image,
            numeric_groups=groups,
            quality_mask=None,
        ),
        numeric_feature_names={
            group: numeric_group_feature_names()[group] for group in E0_NUMERIC_GROUPS
        },
        quality_mask_names=(),
        modality_availability={},
        labels=labels,
        native_unit_id=selected["temporal_unit_keys_json"].astype(str).tolist(),
        window_id=expected_ids,
        motion_schema_hash=MOTION_SCHEMA_HASH,
        motion_schema_version=MOTION_SCHEMA_VERSION,
    )
    shapes = {
        "rgb": list(image.shape),
        "geometry": [len(selected), target_length, 6],
        "motion": [len(selected), target_length, 12],
        "target": list(labels.shape),
        "masks": {"actor_observed_mask": list(observed.shape)},
    }
    return batch, shapes


def run_e0_local_smoke(
    authority_path: Path,
    *,
    paths: Mapping[str, Path],
    output_dir: Path,
    device_name: str = "cpu",
    train_steps: int = 1,
) -> dict[str, Any]:
    """Run one bounded inner-only loader/forward/backward/export smoke."""

    authority = load_e0_execution_authority(authority_path)
    if train_steps <= 0 or train_steps > int(authority["optimization"]["training_budget"]["value"]):
        raise E0ContractError("local E0 smoke train_steps is outside the frozen budget")
    device = _resolve_device(device_name)
    if device.type == "cuda":
        raise E0ContractError("Phase 2B local E0 smoke must be CPU-only")
    _set_seed(int(authority["seed"]))
    population = load_e0_data_population(authority, paths)
    try:
        smoke_per_class = int(authority["optimization"]["smoke_per_class"])
        train_rows = select_e0_role_rows(
            population,
            role="train",
            per_class=smoke_per_class,
            seed=int(authority["seed"]),
        )
        validation_rows = select_e0_role_rows(
            population,
            role="validation",
            per_class=smoke_per_class,
            seed=int(authority["seed"]) + 10_000,
        )
        model = _build_e0_model(authority).to(device)
        optimizer = _build_e0_optimizer(model, authority)
        train_batch, train_shapes = build_e0_model_batch(population, train_rows, device=device)
        validation_batch, validation_shapes = build_e0_model_batch(
            population,
            validation_rows,
            device=device,
        )
        sample_weights = _event_class_weights(
            train_rows,
            authority,
            device=device,
        )
        model.train()
        loss_values: list[float] = []
        for _ in range(train_steps):
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_batch)["logits"]
            raw_loss = nn.functional.cross_entropy(logits, train_batch.labels, reduction="none")
            loss = (raw_loss * sample_weights).sum() / sample_weights.sum()
            if not bool(torch.isfinite(loss)):
                raise E0ContractError("E0 local smoke loss is nonfinite")
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(authority["optimization"]["gradient_clip_norm"]),
            )
            optimizer.step()
            loss_values.append(float(loss.detach().cpu()))
        checkpoint_path = output_dir / "checkpoints" / "e0_local_smoke.pt"
        checkpoint_audit = save_e0_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            authority_path=authority_path,
            authority=authority,
            train_rows=train_rows,
            validation_rows=validation_rows,
            global_step=train_steps,
        )
        restored_model = _build_e0_model(authority).to(device)
        restored_optimizer = _build_e0_optimizer(restored_model, authority)
        resume = load_e0_checkpoint(
            checkpoint_path,
            model=restored_model,
            optimizer=restored_optimizer,
            authority_path=authority_path,
            train_rows=train_rows,
            validation_rows=validation_rows,
            device=device,
        )
        predictions, prediction_audit = export_e0_validation_predictions(
            restored_model,
            validation_batch,
            validation_rows,
            authority_path=authority_path,
            checkpoint_path=checkpoint_path,
            output_path=output_dir / "predictions.csv",
            device=device,
        )
        report = {
            "schema_version": "classification_v2.e0_local_smoke.v1",
            "contract": inspect_e0_execution_authority(authority_path),
            "local_e0_dataset_open": "PASS",
            "local_e0_batch_collation": "PASS",
            "local_e0_forward_pass": "PASS",
            "local_e0_loss_finite": True,
            "local_e0_backward_pass": "PASS",
            "local_e0_optimizer_step": "PASS",
            "local_e0_checkpoint_serialization_smoke": "PASS",
            "local_e0_prediction_export_smoke": "PASS",
            "outer_test_access": "BLOCKED",
            "train_shapes": train_shapes,
            "validation_shapes": validation_shapes,
            "train_rows": int(len(train_rows)),
            "validation_rows": int(len(validation_rows)),
            "loss_values": loss_values,
            "checkpoint": checkpoint_audit,
            "resume": resume,
            "prediction_export": prediction_audit,
            "prediction_rows": int(len(predictions)),
        }
        _write_json_atomic(output_dir / "e0_local_smoke.json", report)
        return report
    finally:
        population.close()


def run_e0_inner_only_training(
    authority_path: Path,
    *,
    paths: Mapping[str, Path],
    output_dir: Path,
    device_name: str,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Execute the frozen inner-only E0 route after a separate authorization.

    The caller is responsible for the external paid-execution authorization.
    This function enforces only scientific and lineage constraints: B3 inputs,
    FOLD_3 development roles, bounded steps, resumable checkpoints, and
    inner-validation prediction export.
    """

    authority = load_e0_execution_authority(authority_path)
    device = _resolve_device(device_name)
    _set_seed(int(authority["seed"]))
    optimization = authority["optimization"]
    budget = int(optimization["training_budget"]["value"])
    checkpoint_every = int(optimization["checkpoint_every_steps"])
    population = load_e0_data_population(authority, paths)
    try:
        train_rows = select_e0_role_rows(
            population,
            role="train",
            per_class=int(optimization["train_per_class"]),
            seed=int(authority["seed"]),
        )
        validation_rows = select_e0_role_rows(
            population,
            role="validation",
            per_class=int(optimization["validation_per_class"]),
            seed=int(authority["seed"]) + 10_000,
        )
        model = _build_e0_model(authority).to(device)
        optimizer = _build_e0_optimizer(model, authority)
        global_step = 0
        if resume_checkpoint is not None:
            resume = load_e0_checkpoint(
                resume_checkpoint,
                model=model,
                optimizer=optimizer,
                authority_path=authority_path,
                train_rows=train_rows,
                validation_rows=validation_rows,
                device=device,
            )
            global_step = int(resume["global_step"])
            if global_step >= budget:
                raise E0ContractError("E0 resume checkpoint already reached the frozen budget")
        else:
            resume = None
        loss_values: list[float] = []
        checkpoint_audits: list[dict[str, Any]] = []
        batch_size = int(optimization["train_batch_size"])
        for next_step in range(global_step + 1, budget + 1):
            batch_rows = _rows_for_training_step(
                train_rows,
                batch_size=batch_size,
                step=next_step,
                seed=int(authority["seed"]),
            )
            batch, _ = build_e0_model_batch(population, batch_rows, device=device)
            sample_weights = _event_class_weights(
                batch_rows,
                authority,
                device=device,
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)["logits"]
            raw_loss = nn.functional.cross_entropy(logits, batch.labels, reduction="none")
            loss = (raw_loss * sample_weights).sum() / sample_weights.sum()
            if not bool(torch.isfinite(loss)):
                raise E0ContractError("E0 training loss is nonfinite")
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(optimization["gradient_clip_norm"]),
            )
            optimizer.step()
            loss_values.append(float(loss.detach().cpu()))
            if next_step % checkpoint_every == 0:
                checkpoint_path = output_dir / "checkpoints" / f"step_{next_step:04d}.pt"
                checkpoint_audits.append(
                    save_e0_checkpoint(
                        checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        authority_path=authority_path,
                        authority=authority,
                        train_rows=train_rows,
                        validation_rows=validation_rows,
                        global_step=next_step,
                    )
                )
        final_checkpoint = output_dir / "checkpoints" / "final_registered_endpoint.pt"
        final_audit = save_e0_checkpoint(
            final_checkpoint,
            model=model,
            optimizer=optimizer,
            authority_path=authority_path,
            authority=authority,
            train_rows=train_rows,
            validation_rows=validation_rows,
            global_step=budget,
        )
        predictions, prediction_audit = _export_e0_validation_prediction_batches(
            model,
            population,
            validation_rows,
            authority_path=authority_path,
            checkpoint_path=final_checkpoint,
            output_path=output_dir / "predictions.csv",
            device=device,
            batch_size=int(optimization["validation_batch_size"]),
        )
        report = {
            "schema_version": "classification_v2.e0_inner_only_run.v1",
            "contract": inspect_e0_execution_authority(authority_path),
            "execution_mode": "inner_only_e0",
            "outer_test_access": "BLOCKED",
            "resumed": resume is not None,
            "resume": resume,
            "completed_training_steps": budget,
            "loss_values": loss_values,
            "checkpoint_cadence_steps": checkpoint_every,
            "checkpoint_audits": checkpoint_audits,
            "final_checkpoint": final_audit,
            "prediction_export": prediction_audit,
            "prediction_rows": int(len(predictions)),
        }
        _write_json_atomic(output_dir / "e0_inner_only_run.json", report)
        return report
    finally:
        population.close()


def save_e0_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    authority_path: Path,
    authority: Mapping[str, Any],
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    global_step: int,
) -> dict[str, Any]:
    """Atomically persist state needed for a lineage-checked E0 resume."""

    if global_step <= 0:
        raise E0ContractError("E0 checkpoint requires at least one completed step")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "authority_sha256": authority_sha256(authority_path),
        "resolved_contract": inspect_e0_execution_authority(authority_path),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": int(global_step),
        "epoch": 0,
        "rng_state": _rng_state(),
        "train_selection_sha256": _selection_sha256(train_rows),
        "validation_selection_sha256": _selection_sha256(validation_rows),
        "checkpoint_selection_rule": authority["checkpoint_selection_rule"],
        "code_sha": _git_sha(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    audit = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_path": str(path),
        "checkpoint_sha256": _sha256_file(path),
        "global_step": int(global_step),
        "has_model_state": True,
        "has_optimizer_state": True,
        "has_scheduler_state": False,
        "has_amp_scaler_state": False,
        "has_rng_state": True,
        "authority_sha256": payload["authority_sha256"],
        "valid": True,
    }
    _write_json_atomic(path.with_suffix(path.suffix + ".audit.json"), audit)
    return audit


def load_e0_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    authority_path: Path,
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    device: torch.device,
) -> dict[str, Any]:
    """Load only an E0 checkpoint from the exact frozen contract and samples."""

    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise E0ContractError("E0 checkpoint schema mismatch")
    expected = {
        "authority_sha256": authority_sha256(authority_path),
        "train_selection_sha256": _selection_sha256(train_rows),
        "validation_selection_sha256": _selection_sha256(validation_rows),
    }
    mismatches = {
        name: {"expected": value, "observed": payload.get(name)}
        for name, value in expected.items()
        if payload.get(name) != value
    }
    if mismatches:
        raise E0ContractError(f"E0 checkpoint lineage mismatch={mismatches}")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    _restore_rng_state(payload["rng_state"])
    return {
        "global_step": int(payload["global_step"]),
        "epoch": int(payload["epoch"]),
        "rng_restored": True,
        "authority_sha256": payload["authority_sha256"],
    }


def export_e0_validation_predictions(
    model: nn.Module,
    batch: ModelBatch,
    rows: pd.DataFrame,
    *,
    authority_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Export one checked prediction per bounded inner-validation target."""

    if rows["e0_role"].ne("validation").any():
        raise E0ContractError("E0 may export only inner-validation predictions")
    records = _e0_prediction_records(
        model,
        batch,
        rows,
        authority_path=authority_path,
        checkpoint_path=checkpoint_path,
    )
    return _write_e0_prediction_records(records, rows, output_path=output_path)


def _export_e0_validation_prediction_batches(
    model: nn.Module,
    population: E0DataPopulation,
    rows: pd.DataFrame,
    *,
    authority_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Export inner-validation predictions in the frozen evaluation batch size."""

    if batch_size <= 0:
        raise E0ContractError("E0 validation batch size must be positive")
    records: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows.iloc[start : start + batch_size].reset_index(drop=True)
        batch, _ = build_e0_model_batch(population, batch_rows, device=device)
        records.extend(
            _e0_prediction_records(
                model,
                batch,
                batch_rows,
                authority_path=authority_path,
                checkpoint_path=checkpoint_path,
            )
        )
    return _write_e0_prediction_records(records, rows, output_path=output_path)


def _e0_prediction_records(
    model: nn.Module,
    batch: ModelBatch,
    rows: pd.DataFrame,
    *,
    authority_path: Path,
    checkpoint_path: Path,
) -> list[dict[str, Any]]:
    """Predict one inner-validation batch without writing partial exports."""

    model.eval()
    with torch.no_grad():
        logits = model(batch)["logits"]
        probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
    if not np.isfinite(probabilities).all():
        raise E0ContractError("E0 prediction probabilities are nonfinite")
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows.itertuples(index=False)):
        predicted_index = int(probabilities[index].argmax())
        record = {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "window_id": str(row.window_id),
            "temporal_unit_key": str(row.temporal_unit_keys_json),
            "true_behavior": str(row.behavior_label),
            "predicted_behavior": VALID_BEHAVIORS[predicted_index],
            "outer_fold": E0_OUTER_FOLD,
            "permitted_role": "validation",
            "authority_sha256": authority_sha256(authority_path),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
        }
        record.update(
            {
                f"prob_{label}": float(probabilities[index, column])
                for column, label in enumerate(VALID_BEHAVIORS)
            }
        )
        records.append(record)
    return records


def _write_e0_prediction_records(
    records: Sequence[Mapping[str, Any]],
    expected_rows: pd.DataFrame,
    *,
    output_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Write the one validated E0 prediction artifact without silent drops."""

    predictions = pd.DataFrame.from_records(records)
    expected_keys = expected_rows["window_id"].astype(str).tolist()
    observed_keys = predictions["window_id"].astype(str).tolist()
    duplicates = int(predictions["window_id"].duplicated().sum())
    missing = sorted(set(expected_keys).difference(observed_keys))
    unexpected = sorted(set(observed_keys).difference(expected_keys))
    if duplicates or missing or unexpected or len(predictions) != len(expected_keys):
        raise E0ContractError(
            "E0 prediction export coverage mismatch="
            f"duplicates:{duplicates},missing:{len(missing)},unexpected:{len(unexpected)}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    audit = {
        "schema_version": "classification_v2.e0_prediction_export_audit.v1",
        "expected_count": int(len(expected_keys)),
        "exported_count": int(len(predictions)),
        "duplicate_keys": duplicates,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "probability_vector_dimension": len(VALID_BEHAVIORS),
        "outer_test_access": "BLOCKED",
        "valid": True,
    }
    _write_json_atomic(output_path.with_suffix(".audit.json"), audit)
    return predictions, audit


def _load_e0_window_frame(paths: Mapping[str, Path]) -> pd.DataFrame:
    columns = [
        "window_id",
        "window_row_index",
        "view_type",
        "window_length_frames",
        "temporal_unit_keys_json",
        "behavior_window_label",
        "window_valid_for_main_train",
        "window_sample_weight",
    ]
    frame = pd.read_csv(paths["effective_window_index"], usecols=columns, low_memory=False)
    selected = frame.loc[
        frame["view_type"].eq("T6_contiguous")
        & pd.to_numeric(frame["window_length_frames"], errors="coerce").eq(6)
        & _to_bool(frame["window_valid_for_main_train"])
        & pd.to_numeric(frame["window_sample_weight"], errors="coerce").gt(0.0)
    ].copy()
    selected["window_id"] = selected["window_id"].astype(str)
    if selected["window_id"].duplicated().any():
        raise E0ContractError("E0 effective T6 rows have duplicate window_id")
    selected["behavior_label"] = selected["behavior_window_label"].astype(str)
    if not selected["behavior_label"].isin(VALID_BEHAVIORS).all():
        raise E0ContractError("E0 effective T6 rows have unsupported behavior labels")
    selected["tensor_row_index"] = pd.to_numeric(
        selected["window_row_index"],
        errors="raise",
    ).astype(np.int64)
    return selected


def _load_fold_three_role_map(path: Path) -> dict[str, str]:
    roles = pd.read_csv(
        path,
        usecols=["temporal_unit_key", "outer_fold_id", "role"],
        low_memory=False,
    )
    roles = roles.loc[roles["outer_fold_id"].astype(str).eq(E0_OUTER_FOLD)].copy()
    if roles.empty or roles["temporal_unit_key"].duplicated().any():
        raise E0ContractError("E0 FOLD_3 role authority is absent or duplicated")
    role_map = roles.set_index("temporal_unit_key")["role"].astype(str).to_dict()
    if "test" not in set(role_map.values()):
        raise E0ContractError("E0 FOLD_3 authority lacks its held-out test role")
    return role_map


def _resolve_window_role(raw_keys: object, role_map: Mapping[str, str]) -> str:
    try:
        keys = json.loads(str(raw_keys))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise E0ContractError("E0 window has invalid temporal-unit key JSON") from exc
    if not isinstance(keys, list) or not keys:
        raise E0ContractError("E0 window has no temporal-unit keys")
    values = {role_map.get(str(key)) for key in keys}
    if None in values or len(values) != 1:
        raise E0ContractError("E0 window does not map to exactly one FOLD_3 role")
    return next(iter(values))


def _build_image_dataset(root: Path, image_size: int) -> ClassificationV2ImageSequenceDataset:
    cache = root / "actor_rgb_64_full"
    return ClassificationV2ImageSequenceDataset(
        ImageSequenceDatasetConfig(
            frame_context_csv=root / "image_context_v2" / "image_frame_context_manifest.csv",
            window_context_csv=root / "image_context_v2" / "image_window_context_manifest.csv",
            image_size=int(image_size),
            packed_image_cache_npy=cache / "packed_rgb_64_letterbox.npy",
            packed_image_cache_index_csv=cache / "packed_image_cache_index.csv",
            require_complete=True,
            require_cached_images=True,
        )
    )


def _attach_event_weights(frame: pd.DataFrame, path: Path) -> None:
    weights = pd.read_csv(
        path,
        usecols=[
            "outer_fold_id",
            "window_id",
            "role",
            "fold_event_class_sample_weight",
            "window_valid_for_fold_training_weight",
        ],
        low_memory=False,
    )
    weights = weights.loc[
        weights["outer_fold_id"].astype(str).eq(E0_OUTER_FOLD)
        & weights["role"].astype(str).eq("train")
    ].copy()
    weights["window_id"] = weights["window_id"].astype(str)
    if weights["window_id"].duplicated().any():
        raise E0ContractError("E0 FOLD_3 event-weight authority has duplicate window_id")
    lookup = weights.set_index("window_id")
    train_rows = frame.loc[frame["e0_role"].eq("train"), "window_id"]
    if train_rows.empty:
        raise E0ContractError("E0 has no FOLD_3 train rows for event-class weighting")
    aligned = lookup.reindex(train_rows)
    if aligned["fold_event_class_sample_weight"].isna().any():
        raise E0ContractError("E0 event-weight authority is missing FOLD_3 train rows")
    if not _to_bool(aligned["window_valid_for_fold_training_weight"]).all():
        raise E0ContractError("E0 selected a fold-training-weight-ineligible row")
    values = pd.to_numeric(aligned["fold_event_class_sample_weight"], errors="raise")
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise E0ContractError("E0 selected an invalid FOLD_3 event-class weight")
    frame["event_sample_weight"] = np.float32(1.0)
    frame.loc[frame["e0_role"].eq("train"), "event_sample_weight"] = (
        values.to_numpy(dtype=np.float32)
    )


def _load_b3_spatial_arrays(root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest_path = root / "spatial_memmap_manifest.json"
    manifest = _read_json_object(manifest_path)
    if manifest.get("spatial_schema_hash") is None:
        raise E0ContractError("E0 spatial manifest lacks a schema hash")
    declared = manifest.get("arrays")
    if not isinstance(declared, dict):
        raise E0ContractError("E0 spatial manifest lacks array declarations")
    arrays: dict[str, np.ndarray] = {}
    try:
        for group in E0_NUMERIC_GROUPS:
            entry = declared.get(group)
            if not isinstance(entry, dict):
                raise E0ContractError(f"E0 spatial manifest is missing {group}")
            path = root / str(entry.get("path", ""))
            if not path.is_file():
                raise FileNotFoundError(path)
            if _sha256_file(path) != entry.get("sha256"):
                raise E0ContractError(f"E0 spatial shard hash mismatch for {group}")
            array = np.load(path, allow_pickle=False, mmap_mode="r")
            arrays[group] = array
        expected = {"bbox_xywh_n": 4, "bbox_shape_n": 2, "motion_delta": 12}
        observed = {group: int(array.shape[-1]) for group, array in arrays.items()}
        if observed != expected:
            raise E0ContractError(f"E0 spatial group dimensions mismatch={observed}")
        return arrays, manifest
    except Exception:
        _close_arrays(arrays)
        raise


def _build_e0_model(authority: Mapping[str, Any]) -> nn.Module:
    settings = authority["model_config"]
    config = baseline_config(
        E0_MODEL,
        target_length=6,
        hidden_dim=int(settings["hidden_dim"]),
        temporal_encoder=str(settings["temporal_encoder"]),
        backbone_name=str(settings["backbone_name"]),
        pretrained_weight_enum=str(settings["pretrained_weight_enum"]),
        image_size=int(settings["image_size"]),
        dropout=float(settings["dropout"]),
        include_controls=False,
    )
    if tuple(config.numeric.groups if config.numeric is not None else ()) != E0_NUMERIC_GROUPS:
        raise E0ContractError("E0 model config expanded or removed B3 numeric groups")
    if config.control_names or config.availability_names:
        raise E0ContractError("E0 model config unexpectedly enables controls")
    return build_model(
        E0_MODEL,
        target_length=6,
        hidden_dim=int(settings["hidden_dim"]),
        temporal_encoder=str(settings["temporal_encoder"]),
        backbone_name=str(settings["backbone_name"]),
        pretrained_weight_enum=str(settings["pretrained_weight_enum"]),
        image_size=int(settings["image_size"]),
        dropout=float(settings["dropout"]),
        include_controls=False,
    )


def _build_e0_optimizer(model: nn.Module, authority: Mapping[str, Any]) -> torch.optim.Optimizer:
    config = authority["optimization"]
    if config["optimizer"] != "adamw":
        raise E0ContractError("E0 supports only the frozen AdamW optimizer")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        betas=tuple(float(value) for value in config["adamw_betas"]),
        eps=float(config["adamw_eps"]),
        weight_decay=float(config["weight_decay"]),
        amsgrad=bool(config["adamw_amsgrad"]),
    )


def _event_class_weights(
    rows: pd.DataFrame,
    authority: Mapping[str, Any],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Read the once-applied FOLD_3 event-class weight for one train batch."""

    config = authority["optimization"]
    if config["sample_weight_policy"] != "event_class":
        raise E0ContractError("E0 imbalance treatment must be event_class")
    values = np.array(
        rows["event_sample_weight"],
        dtype=np.float32,
        copy=True,
    )
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise E0ContractError("E0 event-class weights are invalid")
    return torch.from_numpy(values).to(device)


def _rows_for_training_step(
    rows: pd.DataFrame,
    *,
    batch_size: int,
    step: int,
    seed: int,
) -> pd.DataFrame:
    """Return the deterministic fixed-size mini-batch for one E0 update."""

    if batch_size <= 0 or step <= 0:
        raise E0ContractError("E0 batch size and training step must be positive")
    if len(rows) < batch_size:
        raise E0ContractError("E0 train selection is smaller than the frozen batch size")
    order = np.random.default_rng(seed).permutation(len(rows))
    epoch_offset = ((step - 1) * batch_size) % len(rows)
    positions = np.concatenate(
        (
            order[epoch_offset : epoch_offset + batch_size],
            order[: max(0, epoch_offset + batch_size - len(rows))],
        )
    )
    if len(positions) != batch_size:
        raise E0ContractError("E0 deterministic batch construction drifted")
    return rows.iloc[positions].reset_index(drop=True)


def _selection_sha256(rows: pd.DataFrame) -> str:
    values = rows["window_id"].astype(str).tolist()
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str) -> torch.device:
    if name not in {"cpu", "cuda"}:
        raise E0ContractError(f"unsupported E0 device={name}")
    if name == "cuda" and not torch.cuda.is_available():
        raise E0ContractError("E0 CUDA requested but unavailable")
    return torch.device(name)


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return None


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise E0ContractError(f"E0 JSON must be an object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    for array in arrays.values():
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise E0ContractError(message)


__all__ = [
    "AUTHORITY_SCHEMA_VERSION",
    "CHECKPOINT_SCHEMA_VERSION",
    "E0ContractError",
    "E0DataPopulation",
    "E0_MODEL",
    "E0_OUTER_FOLD",
    "E0_SEED",
    "E0_TEMPORAL_VIEW",
    "PREDICTION_SCHEMA_VERSION",
    "authority_sha256",
    "assert_e0_role_permitted",
    "build_e0_model_batch",
    "export_e0_validation_predictions",
    "inspect_e0_execution_authority",
    "load_e0_checkpoint",
    "load_e0_data_population",
    "load_e0_execution_authority",
    "resolve_e0_data_paths",
    "run_e0_local_smoke",
    "run_e0_inner_only_training",
    "save_e0_checkpoint",
    "select_e0_role_rows",
    "verify_e0_execution_sources",
]
