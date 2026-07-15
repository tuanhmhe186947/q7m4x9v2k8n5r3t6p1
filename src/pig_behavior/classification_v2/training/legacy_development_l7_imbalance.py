"""One-variable imbalance-loss adapter for the retained legacy T6 model."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training import (
    legacy_development_l5_cached_training as frozen_engine,
)
from pig_behavior.classification_v2.training.imbalance_losses import (
    DEFAULT_EFFECTIVE_NUMBER_BETA,
    ImbalanceLossState,
    fit_imbalance_loss_state,
    weighted_imbalance_loss,
)
from pig_behavior.classification_v2.training.legacy_development_l5_cached_data import (
    LegacyL5CachedFeatureClassifier,
    LegacyL5CachedFeatureView,
)
from pig_behavior.classification_v2.training.legacy_development_l5_temporal_ladder import (
    CANONICAL_VIEWS,
    LINEAGE_SCOPE,
    TemporalLadderConfig,
    TemporalLadderSelection,
    aggregate_temporal_ladder_predictions,
    build_window_prediction_frame,
    load_temporal_ladder_batch,
)

VIEW_ID = "t6_sliding"
EXPECTED_PARAMETER_COUNT = 68_234
EXPECTED_FULL_TRAIN_WINDOWS = 14_608
EXPECTED_FULL_TRAIN_NATIVE_UNITS = 3_652
EXPECTED_WINDOWS_PER_NATIVE_UNIT = 4


@dataclass(frozen=True, slots=True)
class LegacyL7LossFitAudit:
    """Hash-bound proof that priors use the complete training role only."""

    state: ImbalanceLossState
    train_windows: int
    train_native_units: int
    event_mass: float
    class_native_units: tuple[int, ...]
    ordered_window_id_sha256: str
    ordered_native_unit_sha256: str
    fit_audit_sha256: str

    @property
    def state_sha256(self) -> str:
        return self.state.state_sha256

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": (
                "classification_v2.legacy_development_l7.loss_fit_audit.v1"
            ),
            "lineage_scope": LINEAGE_SCOPE,
            "view_id": VIEW_ID,
            "loss_state": self.state.to_payload(),
            "train_windows": self.train_windows,
            "train_native_units": self.train_native_units,
            "event_mass": self.event_mass,
            "class_order": list(VALID_BEHAVIORS),
            "class_native_units": list(self.class_native_units),
            "ordered_window_id_sha256": self.ordered_window_id_sha256,
            "ordered_native_unit_sha256": self.ordered_native_unit_sha256,
            "fit_contract": {
                "complete_training_role_used": True,
                "short_optimizer_subset_used_for_fit": False,
                "validation_rows_read_for_fit": 0,
                "outer_holdout_rows_read_for_fit": 0,
                "one_total_mass_per_native_unit": True,
            },
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            **self.semantic_payload(),
            "fit_audit_sha256": self.fit_audit_sha256,
        }


@dataclass(frozen=True, slots=True)
class LegacyL7ImbalanceOutcome:
    """Selected checkpoint plus explicit fold-local loss-fit evidence."""

    epoch_metrics: pd.DataFrame
    window_predictions: pd.DataFrame
    native_predictions: pd.DataFrame
    validation_metrics: dict[str, Any]
    per_class_metrics: pd.DataFrame
    confusion: pd.DataFrame
    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any]
    loss_fit: LegacyL7LossFitAudit
    best_epoch: int
    optimizer_steps: int
    parameter_sha256: str
    window_prediction_sha256: str
    native_prediction_sha256: str
    epoch_metrics_sha256: str
    maximum_loaded_batch_bytes: int


def fit_full_training_loss(
    view: LegacyL5CachedFeatureView,
    *,
    policy: str,
    effective_number_beta: float = DEFAULT_EFFECTIVE_NUMBER_BETA,
) -> LegacyL7LossFitAudit:
    """Fit priors from all 3,652 training events, never the short subset."""

    positions = view.indices_for_role("train")
    if len(positions) != EXPECTED_FULL_TRAIN_WINDOWS:
        raise ValueError(
            "L7 full training-role windows="
            f"{len(positions)}!={EXPECTED_FULL_TRAIN_WINDOWS}"
        )
    rows = view.windows.iloc[positions].reset_index(drop=True)
    if set(rows["l5_role"].astype(str)) != {"train"}:
        raise ValueError("L7 loss fit includes a non-training role")
    native = rows["temporal_unit_key"].astype(str)
    if native.eq("").any():
        raise ValueError("L7 loss fit has a blank native unit")
    native_counts = native.value_counts(sort=False)
    if len(native_counts) != EXPECTED_FULL_TRAIN_NATIVE_UNITS:
        raise ValueError("L7 loss fit training native-unit count drift")
    if not native_counts.eq(EXPECTED_WINDOWS_PER_NATIVE_UNIT).all():
        raise ValueError("L7 loss fit windows per native unit drift")
    targets = view.targets[positions].astype(np.int64, copy=True)
    weights = view.sample_weights[positions].astype(np.float64, copy=True)
    event_mass = pd.Series(weights).groupby(native, sort=False).sum()
    if not np.allclose(event_mass.to_numpy(), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("L7 loss fit native event mass is not one")
    state = fit_imbalance_loss_state(
        targets,
        weights,
        policy=policy,
        effective_number_beta=effective_number_beta,
    )
    native_labels = rows[["temporal_unit_key", "behavior_label"]].drop_duplicates(
        "temporal_unit_key",
        keep="first",
    )
    if len(native_labels) != EXPECTED_FULL_TRAIN_NATIVE_UNITS:
        raise ValueError("L7 native-label collapse lost training units")
    conflicts = rows.groupby("temporal_unit_key", sort=False)[
        "behavior_label"
    ].nunique()
    if not conflicts.eq(1).all():
        raise ValueError("L7 loss fit native unit has conflicting labels")
    class_counts = (
        native_labels["behavior_label"]
        .astype(str)
        .value_counts()
        .reindex(VALID_BEHAVIORS, fill_value=0)
    )
    if not np.allclose(
        np.asarray(state.class_mass),
        class_counts.to_numpy(dtype=np.float64),
        atol=1e-10,
        rtol=0.0,
    ):
        raise ValueError("L7 loss-state class mass differs from native counts")
    semantic = {
        "schema_version": (
            "classification_v2.legacy_development_l7.loss_fit_audit.v1"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "view_id": VIEW_ID,
        "loss_state": state.to_payload(),
        "train_windows": int(len(positions)),
        "train_native_units": int(len(native_counts)),
        "event_mass": float(weights.sum()),
        "class_order": list(VALID_BEHAVIORS),
        "class_native_units": class_counts.astype(int).tolist(),
        "ordered_window_id_sha256": _ordered_hash(rows["window_id"]),
        "ordered_native_unit_sha256": _ordered_hash(
            native_labels["temporal_unit_key"]
        ),
        "fit_contract": {
            "complete_training_role_used": True,
            "short_optimizer_subset_used_for_fit": False,
            "validation_rows_read_for_fit": 0,
            "outer_holdout_rows_read_for_fit": 0,
            "one_total_mass_per_native_unit": True,
        },
    }
    return LegacyL7LossFitAudit(
        state=state,
        train_windows=int(len(positions)),
        train_native_units=int(len(native_counts)),
        event_mass=float(weights.sum()),
        class_native_units=tuple(int(value) for value in class_counts),
        ordered_window_id_sha256=semantic["ordered_window_id_sha256"],
        ordered_native_unit_sha256=semantic["ordered_native_unit_sha256"],
        fit_audit_sha256=_payload_sha256(semantic),
    )


def build_l7_model(
    config: TemporalLadderConfig,
) -> LegacyL5CachedFeatureClassifier:
    """Build the unchanged retained L5 T6 actor-only architecture."""

    model = config.payload["model"]
    classifier = LegacyL5CachedFeatureClassifier(
        temporal_encoder_name=str(model["temporal_encoder_name"]),
        hidden_dim=int(model["hidden_dim"]),
        dropout=float(model["dropout"]),
        transformer_layers=int(model["transformer_layers"]),
        transformer_heads=int(model["transformer_heads"]),
    )
    observed = sum(parameter.numel() for parameter in classifier.parameters())
    if observed != EXPECTED_PARAMETER_COUNT:
        raise ValueError(
            f"L7 model parameters={observed}!={EXPECTED_PARAMETER_COUNT}"
        )
    return classifier


def imbalance_training_step(
    model: LegacyL5CachedFeatureClassifier,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, np.ndarray],
    *,
    loss_state: ImbalanceLossState,
    device: torch.device,
    gradient_clip_norm: float,
) -> tuple[float, float, float]:
    """Run one finite custom-loss step without changing sampler semantics."""

    features = torch.from_numpy(batch["features"]).to(device=device)
    observed_mask = torch.from_numpy(batch["observed_mask"]).to(
        device=device,
        dtype=torch.float32,
    )
    time_delta = torch.from_numpy(batch["time_delta"]).to(
        device=device,
        dtype=torch.float32,
    )
    targets = torch.from_numpy(batch["targets"]).to(
        device=device,
        dtype=torch.long,
    )
    weights = torch.from_numpy(batch["sample_weights"]).to(
        device=device,
        dtype=torch.float32,
    )
    optimizer.zero_grad(set_to_none=True)
    logits = model(features, observed_mask, time_delta=time_delta)
    loss, effective_mass = weighted_imbalance_loss(
        logits,
        targets,
        weights,
        loss_state,
    )
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=gradient_clip_norm,
        error_if_nonfinite=True,
    )
    if not torch.isfinite(gradient_norm) or float(gradient_norm) <= 0.0:
        raise FloatingPointError("L7 imbalance gradients are invalid")
    optimizer.step()
    return (
        float(loss.detach().cpu()),
        float(effective_mass.detach().cpu()),
        float(gradient_norm.detach().cpu()),
    )


def train_l7_imbalance_core(
    view: LegacyL5CachedFeatureView,
    selection: TemporalLadderSelection,
    config: TemporalLadderConfig,
    *,
    policy: str,
    device: torch.device | str,
) -> LegacyL7ImbalanceOutcome:
    """Train one loss policy while keeping model, data and sampler fixed."""

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("L7 imbalance requested unavailable CUDA")
    _validate_selection(view, selection, config)
    loss_fit = fit_full_training_loss(view, policy=policy)
    optimization = config.payload["optimization"]
    seed = int(optimization["seed"])
    frozen_engine._seed_all(seed, seed_cuda=resolved_device.type == "cuda")
    model: LegacyL5CachedFeatureClassifier | None = None
    optimizer: torch.optim.Optimizer | None = None
    try:
        model = build_l7_model(config).to(resolved_device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimization["learning_rate"]),
            weight_decay=float(optimization["weight_decay"]),
        )
        best: dict[str, Any] | None = None
        best_score: tuple[float, float] | None = None
        epoch_rows: list[dict[str, Any]] = []
        optimizer_steps = 0
        maximum_batch_bytes = 0
        for epoch in range(1, int(optimization["epochs"]) + 1):
            positions = selection.train_positions.copy()
            np.random.default_rng(seed + epoch).shuffle(positions)
            loss_mass = 0.0
            effective_mass = 0.0
            model.train()
            for batch_positions in frozen_engine._position_batches(
                positions,
                batch_size=int(optimization["batch_size"]),
            ):
                batch, loaded = load_temporal_ladder_batch(
                    view,
                    batch_positions,
                    maximum_batch_bytes=int(
                        optimization["maximum_loaded_batch_bytes"]
                    ),
                )
                maximum_batch_bytes = max(maximum_batch_bytes, loaded)
                loss_value, batch_mass, _ = imbalance_training_step(
                    model,
                    optimizer,
                    batch,
                    loss_state=loss_fit.state,
                    device=resolved_device,
                    gradient_clip_norm=float(
                        optimization["gradient_clip_norm"]
                    ),
                )
                optimizer_steps += 1
                loss_mass += loss_value * batch_mass
                effective_mass += batch_mass
            if effective_mass <= 0.0:
                raise RuntimeError("L7 imbalance train mass is empty")
            evaluation = _evaluate(
                model,
                view,
                selection,
                config,
                policy=policy,
                device=resolved_device,
            )
            maximum_batch_bytes = max(
                maximum_batch_bytes,
                int(evaluation["maximum_loaded_batch_bytes"]),
            )
            metrics = evaluation["validation_metrics"]
            parameter_sha = frozen_engine._state_dict_sha256(
                model.state_dict()
            )
            score = (
                float(metrics["macro_f1_global_10_class"]),
                -float(metrics["nll"]),
            )
            if best_score is None or score > best_score:
                best_score = score
                best = {
                    **evaluation,
                    "model_state": frozen_engine._clone_state_dict(
                        model.state_dict()
                    ),
                    "optimizer_state": frozen_engine._clone_to_cpu(
                        optimizer.state_dict()
                    ),
                    "best_epoch": epoch,
                }
            epoch_rows.append(
                _epoch_row(
                    config,
                    selection,
                    policy=policy,
                    epoch=epoch,
                    optimizer_steps=optimizer_steps,
                    train_loss=loss_mass / effective_mass,
                    metrics=metrics,
                    parameter_sha=parameter_sha,
                    window_sha=frozen_engine._dataframe_sha256(
                        evaluation["window_predictions"]
                    ),
                    native_sha=frozen_engine._dataframe_sha256(
                        evaluation["native_predictions"]
                    ),
                    loss_fit_sha=loss_fit.fit_audit_sha256,
                )
            )
        expected_steps = int(CANONICAL_VIEWS[VIEW_ID][
            "optimizer_steps_short"
            if config.training_scope == "short_repeat_gate"
            else "optimizer_steps_full"
        ])
        if optimizer_steps != expected_steps:
            raise RuntimeError(
                f"L7 optimizer steps={optimizer_steps}!={expected_steps}"
            )
        if best is None:
            raise RuntimeError("L7 checkpoint selection is empty")
        epoch_rows[int(best["best_epoch"]) - 1]["selected_checkpoint"] = True
        epoch_metrics = pd.DataFrame.from_records(epoch_rows)
        return LegacyL7ImbalanceOutcome(
            epoch_metrics=epoch_metrics,
            window_predictions=best["window_predictions"],
            native_predictions=best["native_predictions"],
            validation_metrics=best["validation_metrics"],
            per_class_metrics=best["per_class_metrics"],
            confusion=best["confusion"],
            model_state=best["model_state"],
            optimizer_state=best["optimizer_state"],
            loss_fit=loss_fit,
            best_epoch=int(best["best_epoch"]),
            optimizer_steps=optimizer_steps,
            parameter_sha256=frozen_engine._state_dict_sha256(
                best["model_state"]
            ),
            window_prediction_sha256=frozen_engine._dataframe_sha256(
                best["window_predictions"]
            ),
            native_prediction_sha256=frozen_engine._dataframe_sha256(
                best["native_predictions"]
            ),
            epoch_metrics_sha256=frozen_engine._dataframe_sha256(
                epoch_metrics
            ),
            maximum_loaded_batch_bytes=maximum_batch_bytes,
        )
    finally:
        if model is not None:
            model.to("cpu")
        del model, optimizer


def _evaluate(
    model: LegacyL5CachedFeatureClassifier,
    view: LegacyL5CachedFeatureView,
    selection: TemporalLadderSelection,
    config: TemporalLadderConfig,
    *,
    policy: str,
    device: torch.device,
) -> dict[str, Any]:
    optimization = config.payload["optimization"]
    evaluation = frozen_engine._evaluate_cached_classifier(
        model,
        view,
        selection.validation_positions,
        batch_size=int(optimization["evaluation_batch_size"]),
        maximum_batch_bytes=int(optimization["maximum_loaded_batch_bytes"]),
        device=device,
    )
    windows = build_window_prediction_frame(
        view,
        selection.validation_positions,
        probabilities=evaluation["probabilities"],
        targets=evaluation["targets"],
        config=config,
        view_id=VIEW_ID,
    )
    native, metrics, per_class, confusion = (
        aggregate_temporal_ladder_predictions(
            windows,
            expected_windows_per_native=EXPECTED_WINDOWS_PER_NATIVE_UNIT,
            training_scope=config.training_scope,
        )
    )
    for frame in (windows, native, per_class, confusion):
        frame["loss_policy"] = policy
    metrics = {**metrics, "loss_policy": policy}
    return {
        "window_predictions": windows,
        "native_predictions": native,
        "validation_metrics": metrics,
        "per_class_metrics": per_class,
        "confusion": confusion,
        "maximum_loaded_batch_bytes": int(
            evaluation["maximum_loaded_batch_bytes"]
        ),
    }


def _validate_selection(
    view: LegacyL5CachedFeatureView,
    selection: TemporalLadderSelection,
    config: TemporalLadderConfig,
) -> None:
    expected_train = (
        320 if config.training_scope == "short_repeat_gate" else 14_608
    )
    expected = {
        "valid": True,
        "training_scope": config.training_scope,
        "view_id": VIEW_ID,
        "sampling_protocol": "all_sliding_event_balanced",
        "sequence_length": 6,
        "windows_per_native_unit": EXPECTED_WINDOWS_PER_NATIVE_UNIT,
        "train_windows": expected_train,
        "validation_windows": 980,
        "validation_native_units": 245,
        "outer_holdout_rows": 0,
        "source_media_reads": 0,
    }
    for field, value in expected.items():
        if selection.audit.get(field) != value:
            raise ValueError(
                f"L7 selection {field}={selection.audit.get(field)!r}!={value!r}"
            )
    for role, positions in (
        ("train", selection.train_positions),
        ("validation", selection.validation_positions),
    ):
        observed = set(view.windows.iloc[positions]["l5_role"].astype(str))
        if observed != {role}:
            raise ValueError(f"L7 {role} selection role drift={observed}")


def _epoch_row(
    config: TemporalLadderConfig,
    selection: TemporalLadderSelection,
    *,
    policy: str,
    epoch: int,
    optimizer_steps: int,
    train_loss: float,
    metrics: dict[str, Any],
    parameter_sha: str,
    window_sha: str,
    native_sha: str,
    loss_fit_sha: str,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "optimizer_steps_cumulative": optimizer_steps,
        "train_native_units": selection.audit["train_native_units"],
        "train_windows": selection.audit["train_windows"],
        "train_loss": train_loss,
        "validation_native_units": selection.audit["validation_native_units"],
        "validation_windows": selection.audit["validation_windows"],
        "validation_macro_f1_global_10_class": metrics[
            "macro_f1_global_10_class"
        ],
        "validation_accuracy": metrics["accuracy"],
        "validation_nll": metrics["nll"],
        "parameter_sha256": parameter_sha,
        "window_prediction_sha256": window_sha,
        "native_prediction_sha256": native_sha,
        "loss_fit_audit_sha256": loss_fit_sha,
        "selected_checkpoint": False,
        "training_scope": config.training_scope,
        "view_id": VIEW_ID,
        "loss_policy": policy,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
    }


def _ordered_hash(values: pd.Series) -> str:
    payload = "\n".join(values.fillna("").astype(str).tolist())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
