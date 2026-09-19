"""Fail-closed, inner-only executor for the authorized PRE-S1 calibration.

This module is intentionally separate from the generic and E0 trainers.  It
accepts no scientific CLI overrides: all model, temporal, role, optimizer and
evaluation choices are resolved from the frozen S1 authority before an image
payload can be opened.  ``engineering_smoke`` is the same route with a bounded
in-memory inner fixture; it is never a calibration result.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
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
from pig_behavior.classification_v2.evaluation.s1_primary_native_evaluator import (
    evaluate_primary_s1_validation,
)
from pig_behavior.classification_v2.models.balanced.contracts import (
    ModelBatch,
    SequenceSegment,
)
from pig_behavior.classification_v2.models.balanced.registry import (
    build_model,
    model_spec_contract,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.pre_s1_rgb_binding import (
    DATA_BINDINGS_SCHEMA,
    RgbBindingError,
    resolve_execution_rgb_binding,
)

AUTHORITY_SCHEMA = "classification_v2.s1_control_and_pre_s1_calibration_authority.v1"
CHECKPOINT_SCHEMA = "classification_v2.pre_s1_calibration_checkpoint.v1"
PREDICTION_SCHEMA = "classification_v2.pre_s1_calibration_window_prediction.v1"
RUN_KIND = "PRE_S1_CALIBRATION"
MODEL_ID = "B1_ACTOR_T6_SEQUENCE"
VIEW = "T6"
FOLD = "FOLD_3"
SEED = 20260804
MAX_STEPS = 6246
EVAL_STEPS = (2082, 4164, 6246)
EXPECTED_TRAIN_WINDOWS = 33300
EXPECTED_VALIDATION_WINDOWS = 6154
EXPECTED_EVENT_WEIGHT_SHA256 = (
    "600a45bea5bc4c9bbd8889ecda6024a0cd24417974d738cb88aded3657c38c72"
)
CANONICAL_AUTHORITY_SHA256 = (
    "948d242201a4df86e2a2a11861e41a9d928d67b695c454357dd74b39fdd77766"
)
FORBIDDEN_SCOPE_TOKENS = frozenset(
    {"test", "outer", "q2_outer_00", "oof_test", "outer_prediction"}
)


class PreS1CalibrationError(ValueError):
    """Raised before an unsafe calibration path can create optimizer state."""


@dataclass(frozen=True, slots=True)
class CalibrationPlan:
    """Resolved immutable calibration contract and non-scientific run identity."""

    repository_root: Path
    outputs_root: Path
    authority_path: Path
    authority_sha256: str
    authority: Mapping[str, Any]
    output_dir: Path
    run_id: str
    engineering_smoke: bool
    device_name: str
    data_bindings_path: Path | None

    @property
    def max_steps(self) -> int:
        return 2 if self.engineering_smoke else MAX_STEPS

    @property
    def evaluation_steps(self) -> tuple[int, ...]:
        return () if self.engineering_smoke else EVAL_STEPS


@dataclass(slots=True)
class CalibrationPopulation:
    """Selected inner rows and the only payload loader the executor may call."""

    train: pd.DataFrame
    validation: pd.DataFrame
    expected_native_units: pd.DataFrame
    load_batch: Callable[[pd.DataFrame, torch.device], ModelBatch]
    close: Callable[[], None]
    data_hashes: Mapping[str, str]
    binding_audit: Mapping[str, Any] | None = None
    image_load_audit: Callable[[], Mapping[str, Any]] | None = None


def create_calibration_plan(
    authority_path: Path,
    *,
    repository_root: Path | None = None,
    outputs_root: Path | None = None,
    output_dir: Path | None = None,
    run_id: str | None = None,
    device_name: str = "cuda",
    data_bindings_path: Path | None = None,
    engineering_smoke: bool = False,
    frozen_overrides: Mapping[str, object] | None = None,
    allow_existing_output: bool = False,
) -> CalibrationPlan:
    """Resolve the frozen authority before opening metadata or image payloads."""

    _reject_frozen_overrides(frozen_overrides)
    _assert_permitted_scope("train")
    _assert_permitted_scope("validation")
    if device_name not in {"cpu", "cuda"}:
        raise PreS1CalibrationError(f"unsupported calibration device={device_name}")
    if not engineering_smoke and device_name != "cuda":
        raise PreS1CalibrationError("real calibration requires the authorized CUDA route")
    authority_path = authority_path.resolve()
    root = (repository_root or _repository_root(authority_path)).resolve()
    resolved_outputs_root = (outputs_root or root / "outputs").resolve()
    if not resolved_outputs_root.is_dir():
        raise PreS1CalibrationError("calibration outputs root is unavailable")
    authority = _read_json(authority_path)
    authority_hash = _sha256_file(authority_path)
    _validate_authority(authority)
    if not engineering_smoke and (
        authority["pre_s1_calibration"].get("status")
        == "COMPLETED_VALID_HORIZON_FROZEN"
    ):
        raise PreS1CalibrationError(
            "new PRE-S1 calibration is forbidden after the valid trajectory "
            "was frozen"
        )
    if not engineering_smoke and authority_hash != CANONICAL_AUTHORITY_SHA256:
        raise PreS1CalibrationError("canonical S1 authority hash mismatch")
    chosen_run_id = run_id or _new_run_id(engineering_smoke)
    _validate_run_id(chosen_run_id)
    default_parent = (
        resolved_outputs_root
        / "classification_v2"
        / "s1_post_temporal_closure_20260809"
    )
    namespace = "engineering_smoke" if engineering_smoke else "pre_s1_calibration"
    target = (output_dir or default_parent / namespace / chosen_run_id).resolve()
    _assert_safe_output_root(
        target,
        outputs_root=resolved_outputs_root,
        engineering_smoke=engineering_smoke,
    )
    if target.exists() and not allow_existing_output:
        raise PreS1CalibrationError(f"immutable calibration output already exists={target}")
    return CalibrationPlan(
        repository_root=root,
        outputs_root=resolved_outputs_root,
        authority_path=authority_path,
        authority_sha256=authority_hash,
        authority=authority,
        output_dir=target,
        run_id=chosen_run_id,
        engineering_smoke=engineering_smoke,
        device_name=device_name,
        data_bindings_path=data_bindings_path.resolve() if data_bindings_path else None,
    )


def preflight_calibration(plan: CalibrationPlan) -> dict[str, str]:
    """Verify every canonical source hash before a dataset or optimizer exists."""

    if plan.authority_sha256 != _sha256_file(plan.authority_path):
        raise PreS1CalibrationError("authority bytes changed after plan resolution")
    authority = plan.authority
    roots = authority["path_roots"]
    derived = _resolve_root(
        plan.repository_root,
        roots["S1_DERIVED"],
        outputs_root=plan.outputs_root,
    )
    route = _resolve_root(
        plan.repository_root,
        roots["S1_ROUTE"],
        outputs_root=plan.outputs_root,
    )
    route_base = _resolve_root(
        plan.repository_root,
        roots["S1_ROUTE_BASE"],
        outputs_root=plan.outputs_root,
    )
    grouped = _resolve_root(
        plan.repository_root,
        roots["GROUPED_ROLE_AUTHORITY"],
        outputs_root=plan.outputs_root,
    )
    hashes = {
        "s1_authority": plan.authority_sha256,
        "temporal_semantic": _verify_artifact(
            route / "s1_temporal_semantic_contract.json",
            authority["created_from"]["temporal_semantic_closure"]["sha256"],
            "temporal semantic authority",
        ),
        "temporal_view": _verify_artifact(
            route_base / "temporal_view_screening_manifest.json",
            authority["created_from"]["temporal_view_manifest"]["sha256"],
            "temporal view authority",
        ),
        "eligibility": _verify_artifact(
            derived / authority["derived_population"]["eligibility_artifact"]["relative_path"],
            authority["derived_population"]["eligibility_artifact"]["sha256"],
            "primary eligibility authority",
        ),
        "event_weight": _verify_artifact(
            derived
            / authority["derived_population"]["per_view"]["T6"]
            ["event_weight_artifact"]["relative_path"],
            EXPECTED_EVENT_WEIGHT_SHA256,
            "corrected T6 event-weight authority",
        ),
        "split_role": _verify_artifact(
            grouped
            / authority["derived_population"]["source_bindings"]
            ["grouped_role_authority"]["relative_path"],
            authority["derived_population"]["source_bindings"]["grouped_role_authority"]["sha256"],
            "FOLD_3 role authority",
        ),
        "effective_window": _verify_artifact(
            _resolve_root(
                plan.repository_root,
                roots["EFFECTIVE_WINDOW_INDEX"],
                outputs_root=plan.outputs_root,
            )
            / authority["derived_population"]["source_bindings"]
            ["effective_window_index"]["relative_path"],
            authority["derived_population"]["source_bindings"]["effective_window_index"]["sha256"],
            "effective-window authority",
        ),
        "model_registry": _verify_artifact(
            route_base
            / authority["pre_s1_calibration"]["reference_configuration"]
            ["registry_relative_path"],
            authority["pre_s1_calibration"]["reference_configuration"]["registry_sha256"],
            "B1 model registry authority",
        ),
    }
    _assert_b1_contract()
    return hashes


def load_canonical_inner_rows(
    plan: CalibrationPlan,
    hashes: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select exactly T6/FOLD_3 inner rows; image payload remains unopened."""

    _assert_permitted_scope("train")
    _assert_permitted_scope("validation")
    derived = _resolve_root(
        plan.repository_root,
        plan.authority["path_roots"]["S1_DERIVED"],
        outputs_root=plan.outputs_root,
    )
    eligibility_path = (
        derived
        / plan.authority["derived_population"]["eligibility_artifact"]
        ["relative_path"]
    )
    weights_path = (
        derived
        / plan.authority["derived_population"]["per_view"]["T6"]
        ["event_weight_artifact"]["relative_path"]
    )
    expected_path = derived / "fold3_t6_validation_native_units.csv"
    eligibility = pd.read_csv(eligibility_path, low_memory=False)
    _require_columns(
        eligibility,
        {
            "window_id", "view_type", "window_length_frames", "temporal_unit_keys_json",
            "behavior_window_label", "primary_s1_role", "primary_s1_eligible",
            "primary_s1_eligibility_status",
        },
        "primary eligibility",
    )
    inner = eligibility.loc[
        eligibility["view_type"].astype(str).eq("T6_contiguous")
        & pd.to_numeric(eligibility["window_length_frames"], errors="coerce").eq(6)
        & _strict_bool(eligibility["primary_s1_eligible"])
        & eligibility["primary_s1_role"].astype(str).isin({"train", "validation"})
    ].copy()
    if inner["primary_s1_eligibility_status"].astype(str).eq("MIXED_LABEL").any():
        raise PreS1CalibrationError("mixed-label row entered the primary calibration population")
    inner["window_id"] = inner["window_id"].astype(str)
    if inner["window_id"].duplicated().any():
        raise PreS1CalibrationError("primary eligibility has duplicate T6 inner window_id")
    train = inner.loc[inner["primary_s1_role"].astype(str).eq("train")].copy()
    validation = inner.loc[inner["primary_s1_role"].astype(str).eq("validation")].copy()
    if len(train) != EXPECTED_TRAIN_WINDOWS:
        raise PreS1CalibrationError(
            f"authorized T6 train count mismatch={len(train)} expected={EXPECTED_TRAIN_WINDOWS}"
        )
    if train["behavior_window_label"].astype(str).isin(VALID_BEHAVIORS).all() is False:
        raise PreS1CalibrationError("training population has unsupported behavior labels")
    weights = pd.read_csv(weights_path, low_memory=False)
    _require_columns(
        weights,
        {
            "outer_fold_id", "window_id", "role", "fold_event_class_sample_weight",
            "window_valid_for_fold_training_weight",
        },
        "event weights",
    )
    weights = weights.loc[
        weights["outer_fold_id"].astype(str).eq(FOLD)
        & weights["role"].astype(str).eq("train")
    ].copy()
    weights["window_id"] = weights["window_id"].astype(str)
    if weights["window_id"].duplicated().any():
        raise PreS1CalibrationError("event-weight artifact has duplicate T6 train window_id")
    weighted = train.merge(
        weights[
            [
                "window_id",
                "fold_event_class_sample_weight",
                "window_valid_for_fold_training_weight",
            ]
        ],
        on="window_id",
        how="left",
        validate="one_to_one",
    )
    if weighted["fold_event_class_sample_weight"].isna().any():
        raise PreS1CalibrationError("event weights do not cover every authorized train window")
    if not _strict_bool(weighted["window_valid_for_fold_training_weight"]).all():
        raise PreS1CalibrationError("non-train-only event weight reached calibration")
    sample_weight = pd.to_numeric(weighted["fold_event_class_sample_weight"], errors="coerce")
    if not np.isfinite(sample_weight).all() or (sample_weight <= 0.0).any():
        raise PreS1CalibrationError("corrected event weights are non-finite or non-positive")
    weighted["event_sample_weight"] = sample_weight.astype(np.float32)
    expected = pd.read_csv(expected_path, low_memory=False)
    _require_columns(expected, {"temporal_unit_key", "behavior_label"}, "expected natives")
    if expected["temporal_unit_key"].astype(str).duplicated().any():
        raise PreS1CalibrationError("expected validation natives are duplicated")
    return weighted, validation.reset_index(drop=True), expected.reset_index(drop=True)


def load_canonical_population(
    plan: CalibrationPlan,
    hashes: Mapping[str, str],
) -> CalibrationPopulation:
    """Attach only an already-proven inner RGB binding to canonical rows."""

    train, validation, expected = load_canonical_inner_rows(plan, hashes)
    return _population_from_rows(train, validation, expected, hashes, plan)


def run_pre_s1_calibration(
    plan: CalibrationPlan,
    population: CalibrationPopulation,
    *,
    resume_checkpoint: Path | None = None,
    stop_after_steps: int | None = None,
) -> dict[str, Any]:
    """Run the one fixed, inner-only route (CPU smoke or later authorized CUDA)."""

    if not plan.engineering_smoke and plan.device_name != "cuda":
        raise PreS1CalibrationError("real calibration may not use CPU")
    if plan.device_name == "cuda" and not torch.cuda.is_available():
        raise PreS1CalibrationError("authorized CUDA calibration requested but CUDA is unavailable")
    if resume_checkpoint is None:
        plan.output_dir.mkdir(parents=True, exist_ok=False)
        for name in (
            "manifest",
            "checkpoints",
            "predictions",
            "metrics",
            "runtime",
            "logs",
        ):
            (plan.output_dir / name).mkdir(exist_ok=False)
    elif not plan.output_dir.is_dir():
        raise PreS1CalibrationError("resume root is absent")
    if stop_after_steps is not None and (
        not plan.engineering_smoke or stop_after_steps < 1 or stop_after_steps >= plan.max_steps
    ):
        raise PreS1CalibrationError("only engineering smoke may stop before its final step")
    device = torch.device(plan.device_name)
    _set_seed(SEED)
    started = time.perf_counter()
    started_at = _utc_now()
    model = _build_b1_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    state = _initial_state(plan, population)
    if resume_checkpoint is not None:
        state = _load_checkpoint(resume_checkpoint, plan, model, optimizer, population)
    _write_json_atomic(plan.output_dir / "manifest" / "run_manifest.json", state["manifest"])
    losses: list[float] = list(state.get("losses", []))
    completed = int(state["completed_steps"])
    snapshots: list[dict[str, Any]] = list(state.get("snapshots", []))
    try:
        for step in range(completed + 1, plan.max_steps + 1):
            batch_rows = _rows_for_step(population.train, step=step, batch_size=16, seed=SEED)
            batch = population.load_batch(batch_rows, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)["logits"]
            per_row = nn.functional.cross_entropy(logits, batch.labels, reduction="none")
            weights = torch.tensor(
                batch_rows["event_sample_weight"].to_numpy(np.float32),
                device=device,
            )
            loss = (per_row * weights).sum() / weights.sum()
            if not bool(torch.isfinite(loss)):
                raise PreS1CalibrationError("non-finite training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if plan.engineering_smoke:
                _save_checkpoint(plan, model, optimizer, population, step, losses, snapshots)
            if step in plan.evaluation_steps or (
                plan.engineering_smoke and step == plan.max_steps
            ):
                snapshot = _evaluate_snapshot(plan, population, model, device, step, losses)
                snapshots.append(snapshot)
                _save_checkpoint(plan, model, optimizer, population, step, losses, snapshots)
            if stop_after_steps == step:
                return {
                    "status": "INTERRUPTED_ENGINEERING_SMOKE",
                    "completed_steps": step,
                    "checkpoint": str(plan.output_dir / "checkpoints" / f"step_{step:06d}.pt"),
                }
        elapsed = time.perf_counter() - started
        telemetry = _runtime_telemetry(
            plan,
            completed_steps=plan.max_steps,
            elapsed=elapsed,
            started_at=started_at,
        )
        report = {
            "status": "PASS",
            "run_kind": RUN_KIND,
            "engineering_smoke": plan.engineering_smoke,
            "claim_grade_result": False,
            "scientific_trial": False,
            "completed_steps": plan.max_steps,
            "losses": losses,
            "snapshots": snapshots,
            "telemetry": telemetry,
        }
        _write_json_atomic(plan.output_dir / "runtime" / "runtime.json", telemetry)
        _write_json_atomic(plan.output_dir / "manifest" / "result.json", report)
        return report
    except Exception as exc:
        _write_json_atomic(
            plan.output_dir / "runtime" / "failure.json",
            {"status": "FAIL", "reason": f"{type(exc).__name__}: {exc}"},
        )
        raise
    finally:
        population.close()


def _population_from_rows(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    expected: pd.DataFrame,
    hashes: Mapping[str, str],
    plan: CalibrationPlan,
) -> CalibrationPopulation:
    """Build the RGB payload loader only after role/hash/output validation completed."""

    if plan.data_bindings_path is None:
        raise PreS1CalibrationError("real calibration requires a hash-bound RGB data-bindings file")
    requested_roles = pd.concat(
        [
            train.loc[:, ["window_id", "primary_s1_role"]],
            validation.loc[:, ["window_id", "primary_s1_role"]],
        ],
        ignore_index=True,
    )
    try:
        rgb_binding = resolve_execution_rgb_binding(
            data_bindings_path=plan.data_bindings_path,
            requested_roles=requested_roles,
            authority_sha256=plan.authority_sha256,
            provenance_hashes=hashes,
        )
    except (FileNotFoundError, RgbBindingError) as exc:
        raise PreS1CalibrationError(str(exc)) from exc
    dataset = ClassificationV2ImageSequenceDataset(
        ImageSequenceDatasetConfig(
            frame_context_csv=rgb_binding.frame_context_path,
            window_context_csv=rgb_binding.window_context_path,
            packed_image_cache_npy=rgb_binding.packed_cache_path,
            packed_image_cache_index_csv=rgb_binding.packed_index_path,
            image_size=64,
            require_complete=True,
            require_cached_images=True,
        )
    )
    requested = set(train["window_id"].astype(str)) | set(validation["window_id"].astype(str))
    lookup = {str(value): index for index, value in enumerate(dataset.windows["window_id"])}
    if set(lookup) != requested:
        dataset.close()
        missing = len(requested.difference(lookup))
        raise PreS1CalibrationError(
            "RGB loader population differs from authority; "
            f"missing={missing}"
        )
    def load_batch(rows: pd.DataFrame, device: torch.device) -> ModelBatch:
        subset = Subset(dataset, [lookup[str(value)] for value in rows["window_id"]])
        image = next(
            iter(
                DataLoader(
                    subset,
                    batch_size=len(rows),
                    shuffle=False,
                    collate_fn=image_sequence_collate,
                )
            )
        )
        errors = [error for group in image["errors"] for error in group]
        if errors:
            raise PreS1CalibrationError(f"inner RGB payload failures={errors[:5]}")
        ids = rows["window_id"].astype(str).tolist()
        if image["window_id"] != ids:
            raise PreS1CalibrationError("inner RGB payload order differs from selected windows")
        return _make_b1_batch(
            image["image"].to(device),
            image["observed_mask"].to(device),
            rows,
            device,
        )
    return CalibrationPopulation(
        train,
        validation,
        expected,
        load_batch,
        dataset.close,
        {**hashes, **rgb_binding.hashes},
        binding_audit=rgb_binding.audit,
        image_load_audit=dataset.image_load_audit,
    )


def run_real_data_cpu_preflight(
    plan: CalibrationPlan,
    population: CalibrationPopulation,
    *,
    sample_size: int = 16,
) -> dict[str, Any]:
    """Prove real inner population and RGB loading without optimizer steps."""

    if plan.engineering_smoke:
        raise PreS1CalibrationError("real-data preflight refuses engineering-smoke plans")
    if sample_size <= 0:
        raise PreS1CalibrationError("real-data preflight sample size must be positive")
    if len(population.train) != EXPECTED_TRAIN_WINDOWS:
        raise PreS1CalibrationError("real-data preflight train population count mismatch")
    if len(population.validation) != EXPECTED_VALIDATION_WINDOWS:
        raise PreS1CalibrationError("real-data preflight validation population count mismatch")
    _require_columns(
        population.train,
        {"primary_s1_role", "primary_s1_eligibility_status", "window_id"},
        "real-data train population",
    )
    _require_columns(
        population.validation,
        {"primary_s1_role", "window_id"},
        "real-data validation population",
    )
    if not population.train["primary_s1_role"].astype(str).eq("train").all():
        raise PreS1CalibrationError("non-train role entered real-data preflight")
    if not population.validation["primary_s1_role"].astype(str).eq("validation").all():
        raise PreS1CalibrationError("non-validation role entered real-data preflight")
    mixed_rows = int(
        population.train["primary_s1_eligibility_status"].astype(str).eq("MIXED_LABEL").sum()
    )
    if mixed_rows:
        raise PreS1CalibrationError("mixed-label row entered real-data preflight")
    if not population.binding_audit:
        raise PreS1CalibrationError("real-data preflight requires RGB binding audit")
    coverage = population.binding_audit.get("coverage", {})
    required_coverage = {
        "train_windows_bound": EXPECTED_TRAIN_WINDOWS,
        "validation_windows_bound": EXPECTED_VALIDATION_WINDOWS,
        "missing_windows": 0,
        "duplicate_windows": 0,
        "bad_sequence_length": 0,
        "role_violations": 0,
        "cross_video_violations": 0,
    }
    for key, expected_value in required_coverage.items():
        if coverage.get(key) != expected_value:
            raise PreS1CalibrationError(
                f"real-data RGB binding coverage mismatch={key}:{coverage.get(key)}"
            )

    _assert_b1_contract()
    cpu = torch.device("cpu")
    decoded_windows = 0
    for rows in (population.train, population.validation):
        sample = _deterministic_preflight_rows(rows, sample_size=sample_size)
        batch = population.load_batch(sample, cpu)
        if tuple(batch.target.images.shape[1:]) != (6, 3, 64, 64):
            raise PreS1CalibrationError("real-data B1 RGB tensor shape drifted")
        if batch.target.valid_mask.shape != (len(sample), 6):
            raise PreS1CalibrationError("real-data B1 observed-mask shape drifted")
        decoded_windows += len(sample)
    image_audit = population.image_load_audit() if population.image_load_audit else {}
    if image_audit:
        if image_audit.get("source_image_loads") != 0:
            raise PreS1CalibrationError("real-data preflight fell back to source media")
        minimum_cache_hits = decoded_windows * 6
        if image_audit.get("packed_image_cache_hits", 0) < minimum_cache_hits:
            raise PreS1CalibrationError("real-data preflight did not decode packed RGB cache")
    return {
        "status": "PASS",
        "primary_train_windows_expected": EXPECTED_TRAIN_WINDOWS,
        "primary_train_windows_loaded": len(population.train),
        "primary_validation_windows_expected": EXPECTED_VALIDATION_WINDOWS,
        "primary_validation_windows_loaded": len(population.validation),
        "mixed_label_training_rows": mixed_rows,
        "outer_windows_loaded": 0,
        "rgb_binding_coverage": coverage,
        "real_rgb_decode_sample": {
            "status": "PASS",
            "windows": decoded_windows,
            "cache_only": True,
        },
        "real_data_b1_input_audit": "PASS",
        "b1_effective_inputs": _b1_effective_inputs(),
        "image_load_audit": image_audit,
    }


def _deterministic_preflight_rows(rows: pd.DataFrame, *, sample_size: int) -> pd.DataFrame:
    count = min(len(rows), sample_size)
    indexes = np.linspace(0, len(rows) - 1, num=count, dtype=int)
    return rows.iloc[indexes].reset_index(drop=True)


def _make_b1_batch(
    images: torch.Tensor,
    observed: torch.Tensor,
    rows: pd.DataFrame,
    device: torch.device,
) -> ModelBatch:
    if tuple(images.shape[1:]) != (6, 3, 64, 64):
        raise PreS1CalibrationError(f"B1 RGB tensor shape drifted={tuple(images.shape)}")
    labels = torch.tensor(
        [VALID_BEHAVIORS.index(str(value)) for value in rows["behavior_window_label"]],
        dtype=torch.long,
        device=device,
    )
    return ModelBatch(
        target=SequenceSegment(
            valid_mask=observed,
            frame_offsets=torch.arange(-5, 1, device=device).repeat(len(rows), 1),
            images=images,
        ),
        labels=labels,
        native_unit_id=rows["temporal_unit_keys_json"].astype(str).tolist(),
        window_id=rows["window_id"].astype(str).tolist(),
    )


def _build_b1_model() -> nn.Module:
    _assert_b1_contract()
    return build_model(MODEL_ID)


def _assert_b1_contract() -> None:
    contract = model_spec_contract(MODEL_ID)
    if contract["temporal_view"] != "T6_TARGET_CONTIGUOUS":
        raise PreS1CalibrationError("B1 registered temporal view drifted")
    if contract["numeric_groups"] != []:
        raise PreS1CalibrationError("B1 must not use geometry, motion, ROI or social inputs")
    config = contract["model_config"]
    if config["target_length"] != 6:
        raise PreS1CalibrationError("B1 model target length must remain T6")


def _evaluate_snapshot(
    plan: CalibrationPlan,
    population: CalibrationPopulation,
    model: nn.Module,
    device: torch.device,
    step: int,
    losses: Sequence[float],
) -> dict[str, Any]:
    predictions: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(population.validation), 16):
            rows = population.validation.iloc[start : start + 16].reset_index(drop=True)
            batch = population.load_batch(rows, device)
            probabilities = torch.softmax(model(batch)["logits"].float(), dim=1).cpu().numpy()
            for index, window_id in enumerate(rows["window_id"].astype(str)):
                label_index = int(probabilities[index].argmax())
                predictions.append(
                    {
                        "window_id": window_id,
                        "y_pred": VALID_BEHAVIORS[label_index],
                        "confidence": float(probabilities[index, label_index]),
                    }
                )
    window = pd.DataFrame(predictions)
    result = evaluate_primary_s1_validation(
        window,
        population.validation,
        population.expected_native_units,
    )
    audit = result.audit
    if (
        not audit["valid"]
        or audit["native_units_unpredicted"]
        or audit["duplicate_collapsed_native_predictions"]
    ):
        raise PreS1CalibrationError("native prediction coverage gate failed")
    prefix = f"step_{step:06d}"
    window_path = plan.output_dir / "predictions" / f"{prefix}_windows.csv"
    native_path = plan.output_dir / "predictions" / f"{prefix}_native.csv"
    window.to_csv(window_path, index=False)
    result.predictions.to_csv(native_path, index=False)
    metric = audit["metrics_on_predicted_units"]
    snapshot = {
        "step": step,
        "training_loss": float(losses[-1]),
        "native_macro_f1": metric.get("macro_f1"),
        "per_class": metric.get("per_class", {}),
        "rare_class_guardrails": {
            "fight": metric.get("per_class", {}).get("fight", {}),
            "social_nose": metric.get("per_class", {}).get("social-nose", {}),
            "weak_rare_support": {
                str(label): int(count)
                for label, count in result.predictions["behavior_label"]
                .astype(str)
                .value_counts()
                .items()
            },
            "decision": "OBSERVATION_ONLY_NO_THRESHOLD_IN_CALIBRATION_AUTHORITY",
        },
        "native_prediction_coverage": audit,
        "window_prediction_sha256": _sha256_file(window_path),
        "native_prediction_sha256": _sha256_file(native_path),
        "composite_key_primary_path_used": False,
    }
    _write_json_atomic(plan.output_dir / "metrics" / f"{prefix}.json", snapshot)
    return snapshot


def _save_checkpoint(
    plan: CalibrationPlan,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    population: CalibrationPopulation,
    step: int,
    losses: Sequence[float],
    snapshots: Sequence[Mapping[str, Any]],
) -> None:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "fingerprint": _fingerprint(plan, population),
        "completed_steps": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng": _rng_state(),
        "losses": list(losses),
        "snapshots": list(snapshots),
    }
    path = plan.output_dir / "checkpoints" / f"step_{step:06d}.pt"
    torch.save(payload, path)
    _write_json_atomic(path.with_suffix(".json"), {"sha256": _sha256_file(path), "step": step})


def _load_checkpoint(
    path: Path,
    plan: CalibrationPlan,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    population: CalibrationPopulation,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise PreS1CalibrationError("resume checkpoint schema mismatch")
    if payload.get("fingerprint") != _fingerprint(plan, population):
        raise PreS1CalibrationError("RESUME_REFUSED=YES fingerprint mismatch")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    _restore_rng_state(payload["rng"])
    return {
        "completed_steps": int(payload["completed_steps"]),
        "losses": list(payload["losses"]),
        "snapshots": list(payload["snapshots"]),
        "manifest": _manifest(plan, population),
    }


def _initial_state(plan: CalibrationPlan, population: CalibrationPopulation) -> dict[str, Any]:
    return {
        "completed_steps": 0,
        "losses": [],
        "snapshots": [],
        "manifest": _manifest(plan, population),
    }


def _manifest(plan: CalibrationPlan, population: CalibrationPopulation) -> dict[str, Any]:
    return {
        "run_id": plan.run_id,
        "run_kind": RUN_KIND,
        "engineering_smoke": plan.engineering_smoke,
        "claim_grade_result": False,
        "scientific_trial": False,
        "model_promotion_allowed": False,
        "temporal_selection_allowed": False,
        "feature_selection_allowed": False,
        "outer_access_allowed": False,
        "authority_sha256": plan.authority_sha256,
        "fingerprint": _fingerprint(plan, population),
        "model": MODEL_ID,
        "temporal_view": VIEW,
        "fold": FOLD,
        "roles": ["train", "validation"],
        "seed": SEED,
        "optimizer": "AdamW",
        "learning_rate": 0.003,
        "weight_decay": 0.0,
        "batch_size": 16,
        "precision": "FP32",
        "scheduler": "none",
        "max_steps": plan.max_steps,
        "evaluation_steps": list(plan.evaluation_steps),
        "git_sha": _git_sha(),
        "data_hashes": dict(population.data_hashes),
        "b1_effective_inputs": _b1_effective_inputs(),
    }


def _fingerprint(plan: CalibrationPlan, population: CalibrationPopulation) -> dict[str, object]:
    return {
        "run_id": plan.run_id,
        "authority_sha256": plan.authority_sha256,
        "model_config_sha256": _json_sha256(model_spec_contract(MODEL_ID)),
        "seed": SEED,
        "temporal_view": VIEW,
        "inner_roles": ["train", "validation"],
        "data_hashes": dict(sorted(population.data_hashes.items())),
        "optimizer": {
            "name": "AdamW",
            "lr": 0.003,
            "weight_decay": 0.0,
            "batch_size": 16,
            "precision": "FP32",
            "scheduler": "none",
        },
        "max_steps": plan.max_steps,
        "evaluation_steps": list(plan.evaluation_steps),
        "hardware": {"family": "CPU_SMOKE" if plan.engineering_smoke else "NVIDIA_L4", "count": 1},
    }


def _rows_for_step(rows: pd.DataFrame, *, step: int, batch_size: int, seed: int) -> pd.DataFrame:
    if len(rows) < batch_size:
        raise PreS1CalibrationError("training population is smaller than frozen batch size")
    order = np.random.default_rng(seed).permutation(len(rows))
    start = ((step - 1) * batch_size) % len(rows)
    indexes = np.concatenate(
        (
            order[start : start + batch_size],
            order[: max(0, start + batch_size - len(rows))],
        )
    )
    return rows.iloc[indexes].reset_index(drop=True)


def _runtime_telemetry(
    plan: CalibrationPlan,
    *,
    completed_steps: int,
    elapsed: float,
    started_at: str,
) -> dict[str, object]:
    cuda = torch.cuda if torch.cuda.is_available() else None
    telemetry: dict[str, object] = {
        "run_id": plan.run_id,
        "start_timestamp": started_at,
        "end_timestamp": _utc_now(),
        "wall_clock_seconds": elapsed,
        "training_seconds": elapsed,
        "completed_steps": completed_steps,
        "seconds_per_step": elapsed / max(1, completed_steps),
        "gpu_name": cuda.get_device_name(0) if cuda else None,
        "gpu_count": cuda.device_count() if cuda else 0,
        "peak_vram_mb": (
            float(cuda.max_memory_allocated(0)) / (1024 * 1024) if cuda else 0.0
        ),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if cuda else None,
        "gpu_hour_accounting": {
            "ceiling_gpu_hours": 48,
            "consumed_gpu_hours": elapsed / 3600 if cuda else 0.0,
            "automatic_ceiling_expansion": False,
        },
    }
    if not plan.engineering_smoke and any(
        telemetry[key] is None
        for key in (
            "cuda_version",
            "wall_clock_seconds",
            "seconds_per_step",
            "peak_vram_mb",
        )
    ):
        raise PreS1CalibrationError("real calibration telemetry is incomplete")
    return telemetry


def _validate_authority(authority: Mapping[str, Any]) -> None:
    if authority.get("schema_version") != AUTHORITY_SCHEMA:
        raise PreS1CalibrationError("unsupported S1 calibration authority schema")
    calibration = authority.get("pre_s1_calibration", {})
    controls = authority.get("fixed_stage_1_to_4_controls", {})
    allowed_calibration_statuses = {
        "AUTHORIZED_NOT_EXECUTED",
        "COMPLETED_VALID_HORIZON_FROZEN",
    }
    if calibration.get("status") not in allowed_calibration_statuses:
        raise PreS1CalibrationError("PRE-S1 calibration status authority drifted")
    if (
        calibration.get("reference_configuration", {}).get("id") != MODEL_ID
        or calibration.get("temporal_view") != VIEW
        or calibration.get("seed") != SEED
    ):
        raise PreS1CalibrationError("calibration model/view/seed authority mismatch")
    if (
        calibration.get("max_steps") != MAX_STEPS
        or tuple(calibration.get("event_snapshots_at_steps", ())) != EVAL_STEPS
    ):
        raise PreS1CalibrationError("calibration step envelope or snapshots drifted")
    expected_controls = {
        **controls,
        "optimizer": "AdamW",
        "learning_rate": 0.003,
        "weight_decay": 0,
        "batch_size": 16,
        "precision": "FP32",
        "scheduler": "none",
    }
    if controls != expected_controls:
        raise PreS1CalibrationError("frozen S1 optimizer controls mismatch")
    fixed_steps = authority.get("matched_training_policy", {}).get(
        "fixed_training_steps"
    )
    if fixed_steps not in {"UNRESOLVED_PENDING_PRE_S1_CALIBRATION", 4164}:
        raise PreS1CalibrationError("unsupported final S1 training horizon authority")
    if calibration.get("status") == "COMPLETED_VALID_HORIZON_FROZEN" and fixed_steps != 4164:
        raise PreS1CalibrationError("completed PRE-S1 authority must bind 4164 steps")
    if authority.get("inner_role_binding", {}).get("forbidden_roles") != [
        "test",
        "outer",
        "q2_outer_00",
    ]:
        raise PreS1CalibrationError("outer refusal authority drifted")
    primary = authority.get("primary_evaluation", {})
    if (
        primary.get("required_path") != "EXPLICIT_WINDOW_TO_NATIVE_COLLAPSE"
        or primary.get("composite_key_direct_primary_metric_allowed") is not False
    ):
        raise PreS1CalibrationError("primary native evaluator authority drifted")


def _assert_permitted_scope(value: str) -> None:
    normalized = str(value).strip().lower()
    if normalized not in {"train", "validation"} or normalized in FORBIDDEN_SCOPE_TOKENS:
        raise PreS1CalibrationError("outer/test role refused before dataset payload access")


def _reject_frozen_overrides(overrides: Mapping[str, object] | None) -> None:
    if overrides:
        raise PreS1CalibrationError(
            f"frozen calibration field override refused={sorted(overrides)}"
        )


def _assert_safe_output_root(
    path: Path,
    *,
    outputs_root: Path,
    engineering_smoke: bool,
) -> None:
    leaf = path.name.lower()
    forbidden_leaf = {
        token
        for token in FORBIDDEN_SCOPE_TOKENS
        if token in leaf
    }
    if forbidden_leaf:
        raise PreS1CalibrationError(
            "outer/test prediction or export root refused before dataset payload access"
        )
    real_namespace = (
        outputs_root
        / "classification_v2"
        / "s1_post_temporal_closure_20260809"
        / "pre_s1_calibration"
    ).resolve()
    if not engineering_smoke and not path.is_relative_to(real_namespace):
        raise PreS1CalibrationError(
            "real calibration output root is outside the isolated namespace"
        )


def _validate_run_id(run_id: str) -> None:
    if not run_id or any(token in run_id.lower() for token in FORBIDDEN_SCOPE_TOKENS):
        raise PreS1CalibrationError("invalid or outer-scoped calibration run ID")


def _resolve_root(
    root: Path,
    parts: Sequence[str],
    *,
    outputs_root: Path,
) -> Path:
    if parts and parts[0] == "outputs":
        return outputs_root.joinpath(*parts[1:])
    return root.joinpath(*parts)


def _verify_artifact(path: Path, expected: str, label: str) -> str:
    observed = _sha256_file(path)
    if observed != expected:
        raise PreS1CalibrationError(f"{label} hash mismatch")
    return observed


def _repository_root(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if (parent / ".git").exists():
            return parent
    raise PreS1CalibrationError("could not resolve repository root for calibration authority")


def _b1_effective_inputs() -> list[str]:
    contract = model_spec_contract(MODEL_ID)["model_config"]
    inputs = ["actor_rgb_T6", "causal_frame_offsets", "actor_observed_mask"]
    if contract["control_names"]:
        inputs.append("registered_zero_default_quality_controls")
    if contract["availability_names"]:
        inputs.append("registered_zero_default_availability_controls")
    return inputs


def _strict_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise PreS1CalibrationError(f"{label} missing columns={missing}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PreS1CalibrationError(f"JSON object required={path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def _restore_rng_state(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def _new_run_id(engineering_smoke: bool) -> str:
    prefix = "engineering_smoke" if engineering_smoke else "pre_s1_calibration"
    return f"{prefix}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:10]}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return None


__all__ = [
    "CalibrationPlan",
    "CalibrationPopulation",
    "DATA_BINDINGS_SCHEMA",
    "PreS1CalibrationError",
    "create_calibration_plan",
    "load_canonical_inner_rows",
    "load_canonical_population",
    "preflight_calibration",
    "run_real_data_cpu_preflight",
    "run_pre_s1_calibration",
]
