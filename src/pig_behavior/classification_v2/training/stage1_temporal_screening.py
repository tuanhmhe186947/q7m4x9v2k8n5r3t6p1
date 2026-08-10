"""Fail-closed, inner-only Stage-1 temporal-screening executor.

The executor is deliberately distinct from PRE-S1 calibration.  It consumes a
user-frozen 4164-step authority, admits only T6/T8/T12/T16 B1 actor-RGB arms,
and keeps primary validation and common-cohort diagnostics separate.  The
current authority permits only CPU preflight; real CUDA execution remains
blocked until a later authorization changes that authority.
"""

from __future__ import annotations

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
from pig_behavior.classification_v2.evaluation.native_temporal_collapse import (
    collapse_window_predictions_to_native_units,
    parse_temporal_unit_keys,
)
from pig_behavior.classification_v2.evaluation.s1_primary_native_evaluator import (
    evaluate_primary_s1_validation,
)
from pig_behavior.classification_v2.models.balanced.baselines import baseline_config
from pig_behavior.classification_v2.models.balanced.contracts import (
    ModelBatch,
    SequenceSegment,
)
from pig_behavior.classification_v2.models.balanced.registry import build_model
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.stage1_rgb_binding import (
    DATA_BINDINGS_SCHEMA,
    Stage1RgbBindingError,
    resolve_stage1_execution_rgb_binding,
)

AUTHORITY_SCHEMA = "classification_v2.s1_control_and_pre_s1_calibration_authority.v1"
DECISION_SCHEMA = "classification_v2.s1_pre_s1_calibration_horizon_decision.v1"
CHECKPOINT_SCHEMA = "classification_v2.s1_stage1_temporal_checkpoint.v1"
RUN_KIND = "S1_STAGE1_TEMPORAL_SCREENING"
MODEL_ID = "B1_ACTOR_T6_SEQUENCE"
FOLD = "FOLD_3"
SEED = 20260804
MAX_STEPS = 4164
EVAL_STEPS = (4164,)
BATCH_SIZE = 16
GPU_HOUR_CEILING = 48
FORBIDDEN_SCOPE_TOKENS = frozenset(
    {"test", "outer", "q2_outer_00", "oof_test", "outer_prediction"}
)
VIEW_SPECS = {
    "T6": {
        "length": 6,
        "view_type": "T6_contiguous",
        "train_windows": 33300,
        "validation_windows": 6154,
        "validation_native_units": 3335,
    },
    "T8": {
        "length": 8,
        "view_type": "T8_contiguous",
        "train_windows": 29383,
        "validation_windows": 5142,
        "validation_native_units": 3331,
    },
    "T12": {
        "length": 12,
        "view_type": "T12_contiguous",
        "train_windows": 26442,
        "validation_windows": 4206,
        "validation_native_units": 3332,
    },
    "T16": {
        "length": 16,
        "view_type": "T16_contiguous",
        "train_windows": 22693,
        "validation_windows": 3199,
        "validation_native_units": 3305,
    },
}


class Stage1TemporalScreeningError(ValueError):
    """Raised before unsafe Stage-1 work can create optimizer state."""


@dataclass(frozen=True, slots=True)
class Stage1Plan:
    """Resolved Stage-1 contract and non-scientific arm identity."""

    repository_root: Path
    outputs_root: Path
    authority_path: Path
    authority_sha256: str
    authority: Mapping[str, Any]
    view: str
    output_dir: Path
    trial_id: str
    engineering_smoke: bool
    device_name: str
    data_bindings_path: Path | None

    @property
    def sequence_length(self) -> int:
        return int(VIEW_SPECS[self.view]["length"])

    @property
    def max_steps(self) -> int:
        return 2 if self.engineering_smoke else MAX_STEPS

    @property
    def evaluation_steps(self) -> tuple[int, ...]:
        return () if self.engineering_smoke else EVAL_STEPS


@dataclass(frozen=True, slots=True)
class Stage1Rows:
    """Hash-bound Stage-1 metadata selected before any RGB payload is opened."""

    train: pd.DataFrame
    validation: pd.DataFrame
    expected_native_units: pd.DataFrame
    common_cohort_native_units: pd.DataFrame
    data_hashes: Mapping[str, str]


@dataclass(slots=True)
class Stage1Population:
    """The only inner-only image loader admitted to the Stage-1 executor."""

    train: pd.DataFrame
    validation: pd.DataFrame
    expected_native_units: pd.DataFrame
    common_cohort_native_units: pd.DataFrame
    load_batch: Callable[[pd.DataFrame, torch.device], ModelBatch]
    close: Callable[[], None]
    data_hashes: Mapping[str, str]
    binding_audit: Mapping[str, Any] | None = None
    image_load_audit: Callable[[], Mapping[str, Any]] | None = None


def create_stage1_plan(
    authority_path: Path,
    *,
    view: str,
    repository_root: Path | None = None,
    outputs_root: Path | None = None,
    output_dir: Path | None = None,
    trial_id: str | None = None,
    device_name: str = "cpu",
    data_bindings_path: Path | None = None,
    engineering_smoke: bool = False,
    frozen_overrides: Mapping[str, object] | None = None,
    allow_existing_output: bool = False,
) -> Stage1Plan:
    """Resolve one frozen Stage-1 arm before metadata or RGB payload access."""

    _validate_view(view)
    _reject_frozen_overrides(frozen_overrides)
    _assert_permitted_scope("train")
    _assert_permitted_scope("validation")
    if device_name not in {"cpu", "cuda"}:
        raise Stage1TemporalScreeningError(f"unsupported Stage-1 device={device_name}")
    authority_path = authority_path.resolve()
    root = (repository_root or _repository_root(authority_path)).resolve()
    resolved_outputs_root = (outputs_root or root / "outputs").resolve()
    if not resolved_outputs_root.is_dir():
        raise Stage1TemporalScreeningError("Stage-1 outputs root is unavailable")
    authority = _read_json(authority_path)
    authority_hash = _sha256_file(authority_path)
    _validate_authority(authority)
    route = _resolve_root(
        root,
        authority["path_roots"]["S1_ROUTE"],
        outputs_root=resolved_outputs_root,
    )
    _validate_calibration_decision(
        route / authority["calibration_decision_authority"]["relative_path"],
        authority["calibration_decision_authority"],
    )
    stage1 = authority["stage_1_temporal_screening"]
    if not engineering_smoke and device_name == "cuda" and not bool(
        stage1["gpu_execution_authorized"]
    ):
        raise Stage1TemporalScreeningError(
            "Stage-1 GPU execution is not authorized by the current authority"
        )
    chosen_trial_id = trial_id or _new_trial_id(view, engineering_smoke)
    _validate_trial_id(chosen_trial_id)
    default_parent = (
        resolved_outputs_root
        / "classification_v2"
        / "s1_post_temporal_closure_20260809"
    )
    namespace = "s1_stage1_engineering_smoke" if engineering_smoke else "s1_trials"
    target = (output_dir or default_parent / namespace / chosen_trial_id).resolve()
    _assert_safe_output_root(
        target,
        outputs_root=resolved_outputs_root,
        engineering_smoke=engineering_smoke,
    )
    if target.exists() and not allow_existing_output:
        raise Stage1TemporalScreeningError(
            f"immutable Stage-1 output already exists={target}"
        )
    return Stage1Plan(
        repository_root=root,
        outputs_root=resolved_outputs_root,
        authority_path=authority_path,
        authority_sha256=authority_hash,
        authority=authority,
        view=view,
        output_dir=target,
        trial_id=chosen_trial_id,
        engineering_smoke=engineering_smoke,
        device_name=device_name,
        data_bindings_path=data_bindings_path.resolve() if data_bindings_path else None,
    )


def preflight_stage1(plan: Stage1Plan) -> dict[str, str]:
    """Verify every authority input before a loader or optimizer exists."""

    if plan.authority_sha256 != _sha256_file(plan.authority_path):
        raise Stage1TemporalScreeningError("authority bytes changed after plan resolution")
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
    view_authority = authority["derived_population"]["per_view"][plan.view]
    label = plan.view.lower()
    hashes = {
        "s1_authority": plan.authority_sha256,
        "calibration_decision": _verify_artifact(
            route / authority["calibration_decision_authority"]["relative_path"],
            authority["calibration_decision_authority"]["sha256"],
            "calibration horizon decision",
        ),
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
        "derivation_audit": _verify_artifact(
            derived / authority["derived_population"]["derivation_audit"]["relative_path"],
            authority["derived_population"]["derivation_audit"]["sha256"],
            "primary derivation audit",
        ),
        "event_weight": _verify_artifact(
            derived / view_authority["event_weight_artifact"]["relative_path"],
            view_authority["event_weight_artifact"]["sha256"],
            f"{plan.view} train-only event-weight authority",
        ),
        "validation_native": _verify_artifact(
            derived / f"fold3_{label}_validation_native_units.csv",
            view_authority["validation_native_population"]["sha256"],
            f"{plan.view} validation native authority",
        ),
        "common_cohort": _verify_artifact(
            derived
            / authority["derived_population"]["common_t6_t8_t12_t16_cohort"][
                "relative_path"
            ],
            authority["derived_population"]["common_t6_t8_t12_t16_cohort"]["sha256"],
            "common-cohort authority",
        ),
        "split_role": _verify_artifact(
            grouped
            / authority["derived_population"]["source_bindings"][
                "grouped_role_authority"
            ]["relative_path"],
            authority["derived_population"]["source_bindings"][
                "grouped_role_authority"
            ]["sha256"],
            "FOLD_3 role authority",
        ),
        "effective_window": _verify_artifact(
            _resolve_root(
                plan.repository_root,
                roots["EFFECTIVE_WINDOW_INDEX"],
                outputs_root=plan.outputs_root,
            )
            / authority["derived_population"]["source_bindings"][
                "effective_window_index"
            ]["relative_path"],
            authority["derived_population"]["source_bindings"][
                "effective_window_index"
            ]["sha256"],
            "effective-window authority",
        ),
        "model_registry": _verify_artifact(
            route_base
            / authority["pre_s1_calibration"]["reference_configuration"][
                "registry_relative_path"
            ],
            authority["pre_s1_calibration"]["reference_configuration"]["registry_sha256"],
            "B1 model registry authority",
        ),
    }
    _assert_b1_contract(plan.view)
    return hashes


def load_stage1_inner_rows(
    plan: Stage1Plan,
    hashes: Mapping[str, str],
) -> Stage1Rows:
    """Select exactly one view's inner rows, without opening RGB payloads."""

    _assert_permitted_scope("train")
    _assert_permitted_scope("validation")
    derived = _resolve_root(
        plan.repository_root,
        plan.authority["path_roots"]["S1_DERIVED"],
        outputs_root=plan.outputs_root,
    )
    audit_path = (
        derived
        / plan.authority["derived_population"]["derivation_audit"]["relative_path"]
    )
    audit = _read_json(audit_path)
    view_type = str(VIEW_SPECS[plan.view]["view_type"])
    audit_view = audit.get("views", {}).get(view_type)
    if not isinstance(audit_view, Mapping):
        raise Stage1TemporalScreeningError(f"derivation audit lacks {view_type}")
    event_weights = audit_view.get("event_weights")
    if not isinstance(event_weights, Mapping):
        raise Stage1TemporalScreeningError("derivation audit lacks event-weight source")
    primary_path = derived / f"fold3_{plan.view.lower()}_primary_windows.csv"
    primary_hash = _verify_artifact(
        primary_path,
        str(event_weights.get("window_input_sha256", "")),
        f"{plan.view} primary windows",
    )
    primary = pd.read_csv(primary_path, low_memory=False)
    _require_columns(
        primary,
        {
            "window_id",
            "view_type",
            "window_length_frames",
            "temporal_unit_keys_json",
            "behavior_window_label",
            "primary_s1_role",
            "primary_s1_eligible",
            "primary_s1_eligibility_status",
        },
        "Stage-1 primary windows",
    )
    if primary["window_id"].astype(str).duplicated().any():
        raise Stage1TemporalScreeningError("Stage-1 primary windows have duplicate window_id")
    if not primary["view_type"].astype(str).eq(view_type).all():
        raise Stage1TemporalScreeningError("primary windows contain a non-selected temporal view")
    if not pd.to_numeric(primary["window_length_frames"], errors="coerce").eq(
        plan.sequence_length
    ).all():
        raise Stage1TemporalScreeningError("primary windows have wrong temporal sequence length")
    roles = primary["primary_s1_role"].astype(str)
    if not roles.isin({"train", "validation"}).all():
        raise Stage1TemporalScreeningError("outer/test role reached Stage-1 primary windows")
    selected = primary.loc[_strict_bool(primary["primary_s1_eligible"])].copy()
    if selected["primary_s1_eligibility_status"].astype(str).eq("MIXED_LABEL").any():
        raise Stage1TemporalScreeningError("mixed-label row entered Stage-1 population")
    selected["window_id"] = selected["window_id"].astype(str)
    train = selected.loc[selected["primary_s1_role"].astype(str).eq("train")].copy()
    validation = selected.loc[
        selected["primary_s1_role"].astype(str).eq("validation")
    ].copy()
    spec = VIEW_SPECS[plan.view]
    if len(train) != int(spec["train_windows"]):
        raise Stage1TemporalScreeningError(
            f"{plan.view} train count mismatch={len(train)} expected={spec['train_windows']}"
        )
    if len(validation) != int(spec["validation_windows"]):
        raise Stage1TemporalScreeningError(
            "{view} validation count mismatch={actual} expected={expected}".format(
                view=plan.view,
                actual=len(validation),
                expected=spec["validation_windows"],
            )
        )
    if not train["behavior_window_label"].astype(str).isin(VALID_BEHAVIORS).all():
        raise Stage1TemporalScreeningError("training population has unsupported behavior labels")

    weights_path = (
        derived
        / plan.authority["derived_population"]["per_view"][plan.view][
            "event_weight_artifact"
        ]["relative_path"]
    )
    weights = pd.read_csv(weights_path, low_memory=False)
    _require_columns(
        weights,
        {
            "outer_fold_id",
            "window_id",
            "role",
            "fold_event_class_sample_weight",
            "window_valid_for_fold_training_weight",
        },
        "Stage-1 event weights",
    )
    weights = weights.loc[
        weights["outer_fold_id"].astype(str).eq(FOLD)
        & weights["role"].astype(str).eq("train")
    ].copy()
    weights["window_id"] = weights["window_id"].astype(str)
    if weights["window_id"].duplicated().any():
        raise Stage1TemporalScreeningError("Stage-1 event weights have duplicate train window_id")
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
        raise Stage1TemporalScreeningError("event weights do not cover every train window")
    if not _strict_bool(weighted["window_valid_for_fold_training_weight"]).all():
        raise Stage1TemporalScreeningError("non-train-only event weight reached Stage-1")
    sample_weight = pd.to_numeric(
        weighted["fold_event_class_sample_weight"],
        errors="coerce",
    )
    if not np.isfinite(sample_weight).all() or (sample_weight <= 0.0).any():
        raise Stage1TemporalScreeningError("event weights are non-finite or non-positive")
    weighted["event_sample_weight"] = sample_weight.astype(np.float32)

    expected_path = derived / f"fold3_{plan.view.lower()}_validation_native_units.csv"
    expected = pd.read_csv(expected_path, low_memory=False)
    _require_columns(expected, {"temporal_unit_key", "behavior_label"}, "expected natives")
    if expected["temporal_unit_key"].astype(str).duplicated().any():
        raise Stage1TemporalScreeningError("expected validation natives are duplicated")
    if len(expected) != int(spec["validation_native_units"]):
        raise Stage1TemporalScreeningError("expected validation native count mismatch")

    common_path = (
        derived
        / plan.authority["derived_population"]["common_t6_t8_t12_t16_cohort"][
            "relative_path"
        ]
    )
    common = pd.read_csv(common_path, low_memory=False)
    _require_columns(
        common,
        {"temporal_unit_key", "role", "behavior_label"},
        "common native cohort",
    )
    if common["temporal_unit_key"].astype(str).duplicated().any():
        raise Stage1TemporalScreeningError("common cohort has duplicate native units")
    if not common["role"].astype(str).isin({"train", "validation"}).all():
        raise Stage1TemporalScreeningError("outer/test role reached common cohort")
    expected_common = int(
        plan.authority["derived_population"]["common_t6_t8_t12_t16_cohort"][
            "native_units"
        ]
    )
    if len(common) != expected_common:
        raise Stage1TemporalScreeningError("common cohort native count mismatch")
    return Stage1Rows(
        train=weighted.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        expected_native_units=expected.reset_index(drop=True),
        common_cohort_native_units=common.reset_index(drop=True),
        data_hashes={**hashes, "primary_windows": primary_hash},
    )


def load_stage1_population(
    plan: Stage1Plan,
    hashes: Mapping[str, str],
) -> Stage1Population:
    """Attach one already-proven Stage-1 RGB binding to the selected rows."""

    if plan.data_bindings_path is None:
        raise Stage1TemporalScreeningError("Stage-1 requires a hash-bound RGB binding")
    rows = load_stage1_inner_rows(plan, hashes)
    requested_roles = pd.concat(
        [
            rows.train.loc[:, ["window_id", "primary_s1_role"]],
            rows.validation.loc[:, ["window_id", "primary_s1_role"]],
        ],
        ignore_index=True,
    )
    try:
        rgb_binding = resolve_stage1_execution_rgb_binding(
            data_bindings_path=plan.data_bindings_path,
            requested_roles=requested_roles,
            authority_sha256=plan.authority_sha256,
            provenance_hashes=rows.data_hashes,
            view=plan.view,
            sequence_length=plan.sequence_length,
        )
    except (FileNotFoundError, Stage1RgbBindingError) as exc:
        raise Stage1TemporalScreeningError(str(exc)) from exc
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
    requested = set(rows.train["window_id"].astype(str)) | set(
        rows.validation["window_id"].astype(str)
    )
    lookup = {str(value): index for index, value in enumerate(dataset.windows["window_id"])}
    if set(lookup) != requested:
        dataset.close()
        raise Stage1TemporalScreeningError(
            "RGB loader population differs from authority; "
            f"missing={len(requested.difference(lookup))}"
        )

    def load_batch(selected_rows: pd.DataFrame, device: torch.device) -> ModelBatch:
        subset = Subset(dataset, [lookup[str(value)] for value in selected_rows["window_id"]])
        image = next(
            iter(
                DataLoader(
                    subset,
                    batch_size=len(selected_rows),
                    shuffle=False,
                    collate_fn=image_sequence_collate,
                )
            )
        )
        errors = [error for group in image["errors"] for error in group]
        if errors:
            raise Stage1TemporalScreeningError(f"Stage-1 RGB payload failures={errors[:5]}")
        ids = selected_rows["window_id"].astype(str).tolist()
        if image["window_id"] != ids:
            raise Stage1TemporalScreeningError("Stage-1 RGB payload order drifted")
        return _make_b1_batch(
            image["image"].to(device),
            image["observed_mask"].to(device),
            selected_rows,
            device,
            sequence_length=plan.sequence_length,
        )

    return Stage1Population(
        train=rows.train,
        validation=rows.validation,
        expected_native_units=rows.expected_native_units,
        common_cohort_native_units=rows.common_cohort_native_units,
        load_batch=load_batch,
        close=dataset.close,
        data_hashes={**rows.data_hashes, **rgb_binding.hashes},
        binding_audit=rgb_binding.audit,
        image_load_audit=dataset.image_load_audit,
    )


def run_real_data_cpu_preflight(
    plan: Stage1Plan,
    population: Stage1Population,
    *,
    sample_size: int = 8,
) -> dict[str, Any]:
    """Prove one real inner-only view and its packed RGB cache on CPU."""

    if plan.engineering_smoke:
        raise Stage1TemporalScreeningError("real-data preflight refuses engineering-smoke plans")
    if sample_size <= 0:
        raise Stage1TemporalScreeningError("real-data preflight sample size must be positive")
    spec = VIEW_SPECS[plan.view]
    if len(population.train) != int(spec["train_windows"]):
        raise Stage1TemporalScreeningError("real-data preflight train count mismatch")
    if len(population.validation) != int(spec["validation_windows"]):
        raise Stage1TemporalScreeningError("real-data preflight validation count mismatch")
    if not population.train["primary_s1_role"].astype(str).eq("train").all():
        raise Stage1TemporalScreeningError("non-train role entered Stage-1 preflight")
    if not population.validation["primary_s1_role"].astype(str).eq("validation").all():
        raise Stage1TemporalScreeningError("non-validation role entered Stage-1 preflight")
    mixed_rows = int(
        population.train["primary_s1_eligibility_status"].astype(str).eq("MIXED_LABEL").sum()
    )
    if mixed_rows:
        raise Stage1TemporalScreeningError("mixed-label row entered Stage-1 preflight")
    if not population.binding_audit:
        raise Stage1TemporalScreeningError("real-data preflight requires a binding audit")
    coverage = population.binding_audit.get("coverage", {})
    required_coverage = {
        "train_windows_bound": int(spec["train_windows"]),
        "validation_windows_bound": int(spec["validation_windows"]),
        "missing_windows": 0,
        "duplicate_windows": 0,
        "bad_sequence_length": 0,
        "role_violations": 0,
        "cross_video_violations": 0,
    }
    for key, expected in required_coverage.items():
        if coverage.get(key) != expected:
            raise Stage1TemporalScreeningError(
                f"Stage-1 RGB coverage mismatch={key}:{coverage.get(key)}"
            )
    _assert_b1_contract(plan.view)
    cpu = torch.device("cpu")
    decoded_windows = 0
    for selected_rows in (population.train, population.validation):
        sample = _deterministic_rows(selected_rows, sample_size=sample_size)
        batch = population.load_batch(sample, cpu)
        expected_shape = (plan.sequence_length, 3, 64, 64)
        if tuple(batch.target.images.shape[1:]) != expected_shape:
            raise Stage1TemporalScreeningError("real-data B1 RGB tensor shape drifted")
        if batch.target.valid_mask.shape != (len(sample), plan.sequence_length):
            raise Stage1TemporalScreeningError("real-data B1 observed-mask shape drifted")
        decoded_windows += len(sample)
    image_audit = population.image_load_audit() if population.image_load_audit else {}
    if image_audit:
        if image_audit.get("source_image_loads") != 0:
            raise Stage1TemporalScreeningError("Stage-1 preflight fell back to source media")
        minimum_hits = decoded_windows * plan.sequence_length
        if image_audit.get("packed_image_cache_hits", 0) < minimum_hits:
            raise Stage1TemporalScreeningError("Stage-1 preflight did not decode packed cache")
    return {
        "status": "PASS",
        "view": plan.view,
        "primary_train_windows_expected": int(spec["train_windows"]),
        "primary_train_windows_loaded": len(population.train),
        "primary_validation_windows_expected": int(spec["validation_windows"]),
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
        "b1_effective_inputs": _b1_effective_inputs(plan.view),
        "image_load_audit": image_audit,
    }


def run_real_data_cpu_engineering_smoke(
    plan: Stage1Plan,
    population: Stage1Population,
    *,
    steps: int = 1,
) -> dict[str, Any]:
    """Run at most two CPU optimizer steps against real bound inner data."""

    if plan.engineering_smoke or plan.device_name != "cpu":
        raise Stage1TemporalScreeningError("real-data CPU smoke requires a non-smoke CPU plan")
    if steps not in {1, 2}:
        raise Stage1TemporalScreeningError("real-data CPU smoke is bounded to one or two steps")
    _set_seed(SEED)
    device = torch.device("cpu")
    model = _build_b1_model(plan.view).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    losses: list[float] = []
    for step in range(1, steps + 1):
        rows = _rows_for_step(population.train, step=step, batch_size=BATCH_SIZE, seed=SEED)
        batch = population.load_batch(rows, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)["logits"]
        per_row = nn.functional.cross_entropy(logits, batch.labels, reduction="none")
        weights = torch.tensor(
            rows["event_sample_weight"].to_numpy(np.float32),
            device=device,
        )
        loss = (per_row * weights).sum() / weights.sum()
        if not bool(torch.isfinite(loss)):
            raise Stage1TemporalScreeningError("non-finite real-data CPU smoke loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {
        "status": "PASS",
        "view": plan.view,
        "cpu_smoke_steps": steps,
        "losses": losses,
        "gpu_used": False,
    }


def run_stage1_temporal_screening(
    plan: Stage1Plan,
    population: Stage1Population,
    *,
    resume_checkpoint: Path | None = None,
    stop_after_steps: int | None = None,
) -> dict[str, Any]:
    """Run a CPU engineering smoke or a separately authorized real L4 arm."""

    _assert_execution_hardware(plan)
    if resume_checkpoint is None:
        plan.output_dir.mkdir(parents=True, exist_ok=False)
        for name in ("manifest", "checkpoints", "predictions", "metrics", "runtime", "logs"):
            (plan.output_dir / name).mkdir(exist_ok=False)
    elif not plan.output_dir.is_dir():
        raise Stage1TemporalScreeningError("resume root is absent")
    if stop_after_steps is not None and (
        not plan.engineering_smoke or stop_after_steps < 1 or stop_after_steps >= plan.max_steps
    ):
        raise Stage1TemporalScreeningError("only engineering smoke may stop before its endpoint")
    device = torch.device(plan.device_name)
    if plan.device_name == "cuda":
        torch.cuda.reset_peak_memory_stats(0)
    _set_seed(SEED)
    started = time.perf_counter()
    started_at = _utc_now()
    model = _build_b1_model(plan.view).to(device)
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
            rows = _rows_for_step(population.train, step=step, batch_size=BATCH_SIZE, seed=SEED)
            batch = population.load_batch(rows, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)["logits"]
            per_row = nn.functional.cross_entropy(logits, batch.labels, reduction="none")
            weights = torch.tensor(
                rows["event_sample_weight"].to_numpy(np.float32),
                device=device,
            )
            loss = (per_row * weights).sum() / weights.sum()
            if not bool(torch.isfinite(loss)):
                raise Stage1TemporalScreeningError("non-finite Stage-1 training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            recovery_due = (
                plan.engineering_smoke
                or step == plan.max_steps
                or (not plan.engineering_smoke and step % 2082 == 0)
            )
            if recovery_due:
                _save_checkpoint(plan, model, optimizer, population, step, losses, snapshots)
            if step in plan.evaluation_steps or (plan.engineering_smoke and step == plan.max_steps):
                snapshot = _evaluate_endpoint(plan, population, model, device, step, losses)
                snapshots.append(snapshot)
                _save_checkpoint(plan, model, optimizer, population, step, losses, snapshots)
                checkpoint = plan.output_dir / "checkpoints" / f"step_{step:06d}.pt"
                snapshot["checkpoint_path"] = str(checkpoint)
                snapshot["checkpoint_sha256"] = _sha256_file(checkpoint)
                _write_json_atomic(plan.output_dir / "metrics" / f"step_{step:06d}.json", snapshot)
            if stop_after_steps == step:
                return {
                    "status": "INTERRUPTED_ENGINEERING_SMOKE",
                    "completed_steps": step,
                    "checkpoint": str(plan.output_dir / "checkpoints" / f"step_{step:06d}.pt"),
                }
        elapsed = time.perf_counter() - started
        telemetry = _runtime_telemetry(
            plan,
            population,
            completed_steps=plan.max_steps,
            elapsed=elapsed,
            started_at=started_at,
        )
        report = {
            "status": "PASS",
            "run_kind": RUN_KIND,
            "view": plan.view,
            "engineering_smoke": plan.engineering_smoke,
            "claim_grade_result": False,
            "scientific_trial": not plan.engineering_smoke,
            "completed_steps": plan.max_steps,
            "losses": losses,
            "snapshots": snapshots,
            "telemetry": telemetry,
        }
        _write_json_atomic(plan.output_dir / "runtime" / "runtime.json", telemetry)
        _write_json_atomic(plan.output_dir / "manifest" / "result.json", report)
        artifact_manifest = _write_artifact_manifest(plan.output_dir)
        report["artifact_manifest"] = artifact_manifest
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


def _evaluate_endpoint(
    plan: Stage1Plan,
    population: Stage1Population,
    model: nn.Module,
    device: torch.device,
    step: int,
    losses: Sequence[float],
) -> dict[str, Any]:
    validation_window = _predict_windows(population.validation, population, model, device)
    primary = evaluate_primary_s1_validation(
        validation_window,
        population.validation,
        population.expected_native_units,
    )
    _assert_native_coverage(primary.audit, label="primary validation")
    inner_rows = pd.concat([population.train, population.validation], ignore_index=True)
    common_rows = _common_cohort_window_manifest(
        inner_rows,
        population.common_cohort_native_units,
    )
    common_window = _predict_windows(common_rows, population, model, device)
    common = collapse_window_predictions_to_native_units(
        common_window,
        common_rows[["window_id", "temporal_unit_keys_json"]],
        population.common_cohort_native_units[["temporal_unit_key", "behavior_label"]],
    )
    _assert_native_coverage(common.audit, label="common cohort")
    prefix = f"step_{step:06d}"
    paths = {
        "validation_window": plan.output_dir / "predictions" / f"{prefix}_validation_windows.csv",
        "validation_native": plan.output_dir / "predictions" / f"{prefix}_validation_native.csv",
        "common_window": plan.output_dir / "predictions" / f"{prefix}_common_windows.csv",
        "common_native": plan.output_dir / "predictions" / f"{prefix}_common_native.csv",
    }
    validation_window.to_csv(paths["validation_window"], index=False)
    primary.predictions.to_csv(paths["validation_native"], index=False)
    common_window.to_csv(paths["common_window"], index=False)
    common.predictions.to_csv(paths["common_native"], index=False)
    primary_metric = primary.audit["metrics_on_predicted_units"]
    common_metric = common.audit["metrics_on_predicted_units"]
    snapshot = {
        "step": step,
        "training_loss": float(losses[-1]),
        "native_macro_f1": primary_metric.get("macro_f1"),
        "per_class": primary_metric.get("per_class", {}),
        "rare_class_guardrails": {
            "fight": primary_metric.get("per_class", {}).get("fight", {}),
            "social_nose": primary_metric.get("per_class", {}).get("social-nose", {}),
            "weak_rare_support": {
                str(label): int(count)
                for label, count in primary.predictions["behavior_label"]
                .astype(str)
                .value_counts()
                .items()
            },
            "decision": "OBSERVATION_ONLY_REGISTERED_STAGE1_GUARDRAILS",
        },
        "native_prediction_coverage": primary.audit,
        "common_cohort": {
            "native_units": len(population.common_cohort_native_units),
            "native_macro_f1": common_metric.get("macro_f1"),
            "per_class": common_metric.get("per_class", {}),
            "prediction_coverage": common.audit,
            "diagnostic_only": True,
        },
        "validation_window_prediction_sha256": _sha256_file(paths["validation_window"]),
        "validation_native_prediction_sha256": _sha256_file(paths["validation_native"]),
        "common_window_prediction_sha256": _sha256_file(paths["common_window"]),
        "common_native_prediction_sha256": _sha256_file(paths["common_native"]),
        "composite_key_primary_path_used": False,
    }
    _write_json_atomic(plan.output_dir / "metrics" / f"{prefix}.json", snapshot)
    return snapshot


def _predict_windows(
    rows: pd.DataFrame,
    population: Stage1Population,
    model: nn.Module,
    device: torch.device,
) -> pd.DataFrame:
    predictions: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_SIZE):
            batch_rows = rows.iloc[start : start + BATCH_SIZE].reset_index(drop=True)
            batch = population.load_batch(batch_rows, device)
            probabilities = torch.softmax(model(batch)["logits"].float(), dim=1).cpu().numpy()
            for index, window_id in enumerate(batch_rows["window_id"].astype(str)):
                label_index = int(probabilities[index].argmax())
                predictions.append(
                    {
                        "window_id": window_id,
                        "y_pred": VALID_BEHAVIORS[label_index],
                        "confidence": float(probabilities[index, label_index]),
                    }
                )
    return pd.DataFrame(predictions)


def _common_cohort_window_manifest(
    rows: pd.DataFrame,
    common_native_units: pd.DataFrame,
) -> pd.DataFrame:
    common_keys = set(common_native_units["temporal_unit_key"].astype(str))
    selected_rows: list[dict[str, object]] = []
    for row in rows.itertuples(index=False):
        keys = [
            key
            for key in parse_temporal_unit_keys(row.temporal_unit_keys_json)
            if key in common_keys
        ]
        if keys:
            selected = row._asdict()
            selected["temporal_unit_keys_json"] = json.dumps(keys)
            selected_rows.append(selected)
    manifest = pd.DataFrame(selected_rows)
    if manifest.empty:
        raise Stage1TemporalScreeningError("common cohort has no represented Stage-1 windows")
    if manifest["window_id"].astype(str).duplicated().any():
        raise Stage1TemporalScreeningError("common-cohort window manifest has duplicate window_id")
    return manifest.reset_index(drop=True)


def _assert_native_coverage(audit: Mapping[str, Any], *, label: str) -> None:
    if (
        not bool(audit.get("valid"))
        or int(audit.get("native_units_unpredicted", 0))
        or int(audit.get("duplicate_collapsed_native_predictions", 0))
        or int(len(audit.get("unexpected_native_prediction_examples", [])))
    ):
        raise Stage1TemporalScreeningError(f"{label} native prediction coverage gate failed")


def _save_checkpoint(
    plan: Stage1Plan,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    population: Stage1Population,
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
    plan: Stage1Plan,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    population: Stage1Population,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise Stage1TemporalScreeningError("resume checkpoint schema mismatch")
    if payload.get("fingerprint") != _fingerprint(plan, population):
        raise Stage1TemporalScreeningError("RESUME_REFUSED=YES fingerprint mismatch")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    _restore_rng_state(payload["rng"])
    return {
        "completed_steps": int(payload["completed_steps"]),
        "losses": list(payload["losses"]),
        "snapshots": list(payload["snapshots"]),
        "manifest": _manifest(plan, population),
    }


def _initial_state(plan: Stage1Plan, population: Stage1Population) -> dict[str, Any]:
    return {
        "completed_steps": 0,
        "losses": [],
        "snapshots": [],
        "manifest": _manifest(plan, population),
    }


def _manifest(plan: Stage1Plan, population: Stage1Population) -> dict[str, Any]:
    return {
        "trial_id": plan.trial_id,
        "run_kind": RUN_KIND,
        "view": plan.view,
        "engineering_smoke": plan.engineering_smoke,
        "claim_grade_result": False,
        "outer_access_allowed": False,
        "authority_sha256": plan.authority_sha256,
        "fingerprint": _fingerprint(plan, population),
        "model": MODEL_ID,
        "temporal_view": plan.view,
        "fold": FOLD,
        "roles": ["train", "validation"],
        "seed": SEED,
        "optimizer": "AdamW",
        "learning_rate": 0.003,
        "weight_decay": 0.0,
        "batch_size": BATCH_SIZE,
        "precision": "FP32",
        "scheduler": "none",
        "max_steps": plan.max_steps,
        "evaluation_steps": list(plan.evaluation_steps),
        "scientific_ranking_checkpoint": "FIXED_STEP_ENDPOINT",
        "primary_metric": "NATIVE_UNIT_10_CLASS_MACRO_F1",
        "common_cohort_native_units": len(population.common_cohort_native_units),
        "git_sha": _git_sha(),
        "data_hashes": dict(population.data_hashes),
        "b1_effective_inputs": _b1_effective_inputs(plan.view),
    }


def _fingerprint(plan: Stage1Plan, population: Stage1Population) -> dict[str, object]:
    return {
        "trial_id": plan.trial_id,
        "run_kind": RUN_KIND,
        "authority_sha256": plan.authority_sha256,
        "model_config_sha256": _json_sha256(_model_config_payload(plan.view)),
        "seed": SEED,
        "temporal_view": plan.view,
        "inner_roles": ["train", "validation"],
        "data_hashes": dict(sorted(population.data_hashes.items())),
        "optimizer": {
            "name": "AdamW",
            "lr": 0.003,
            "weight_decay": 0.0,
            "batch_size": BATCH_SIZE,
            "precision": "FP32",
            "scheduler": "none",
        },
        "max_steps": plan.max_steps,
        "evaluation_steps": list(plan.evaluation_steps),
        "hardware": {
            "family": "CPU_SMOKE" if plan.engineering_smoke else "NVIDIA_L4",
            "count": 1,
        },
    }


def _rows_for_step(
    rows: pd.DataFrame,
    *,
    step: int,
    batch_size: int,
    seed: int,
) -> pd.DataFrame:
    if len(rows) < batch_size:
        raise Stage1TemporalScreeningError("training population is smaller than batch size")
    order = np.random.default_rng(seed).permutation(len(rows))
    start = ((step - 1) * batch_size) % len(rows)
    indexes = np.concatenate(
        (order[start : start + batch_size], order[: max(0, start + batch_size - len(rows))])
    )
    return rows.iloc[indexes].reset_index(drop=True)


def _runtime_telemetry(
    plan: Stage1Plan,
    population: Stage1Population,
    *,
    completed_steps: int,
    elapsed: float,
    started_at: str,
) -> dict[str, object]:
    cuda = torch.cuda if torch.cuda.is_available() else None
    telemetry: dict[str, object] = {
        "trial_id": plan.trial_id,
        "run_kind": RUN_KIND,
        "view": plan.view,
        "start_timestamp": started_at,
        "end_timestamp": _utc_now(),
        "wall_clock_seconds": elapsed,
        "training_seconds": elapsed,
        "completed_steps": completed_steps,
        "seconds_per_step": elapsed / max(1, completed_steps),
        "gpu_name": cuda.get_device_name(0) if cuda else None,
        "gpu_count": cuda.device_count() if cuda else 0,
        "peak_vram_mb": float(cuda.max_memory_allocated(0)) / (1024 * 1024) if cuda else 0.0,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if cuda else None,
        "authority_hashes": dict(population.data_hashes),
        "configuration_hash": _json_sha256(_model_config_payload(plan.view)),
        "optimizer": "AdamW",
        "learning_rate": 0.003,
        "weight_decay": 0.0,
        "batch_size": BATCH_SIZE,
        "precision": "FP32",
        "scheduler": "none",
        "gpu_hour_accounting": {
            "ceiling_gpu_hours": GPU_HOUR_CEILING,
            "consumed_gpu_hours": elapsed / 3600 if cuda else 0.0,
            "automatic_ceiling_expansion": False,
        },
    }
    if not plan.engineering_smoke and any(
        telemetry[key] is None
        for key in ("cuda_version", "wall_clock_seconds", "seconds_per_step", "peak_vram_mb")
    ):
        raise Stage1TemporalScreeningError("real Stage-1 telemetry is incomplete")
    return telemetry


def _write_artifact_manifest(output_dir: Path) -> dict[str, object]:
    paths = [
        path
        for folder in ("checkpoints", "predictions", "metrics", "runtime", "manifest")
        for path in (output_dir / folder).glob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    entries = [
        {
            "relative_path": str(path.relative_to(output_dir)).replace("\\", "/"),
            "sha256": _sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for path in sorted(paths)
    ]
    payload = {
        "schema_version": "classification_v2.s1_stage1_artifact_manifest.v1",
        "artifacts": entries,
    }
    path = output_dir / "manifest" / "artifact_manifest.json"
    _write_json_atomic(path, payload)
    return {"path": str(path), "sha256": _sha256_file(path), "artifacts": len(entries)}


def _deterministic_rows(rows: pd.DataFrame, *, sample_size: int) -> pd.DataFrame:
    count = min(len(rows), sample_size)
    indexes = np.linspace(0, len(rows) - 1, num=count, dtype=int)
    return rows.iloc[indexes].reset_index(drop=True)


def _make_b1_batch(
    images: torch.Tensor,
    observed: torch.Tensor,
    rows: pd.DataFrame,
    device: torch.device,
    *,
    sequence_length: int,
) -> ModelBatch:
    if tuple(images.shape[1:]) != (sequence_length, 3, 64, 64):
        raise Stage1TemporalScreeningError(
            f"B1 RGB tensor shape drifted={tuple(images.shape)}"
        )
    labels = torch.tensor(
        [VALID_BEHAVIORS.index(str(value)) for value in rows["behavior_window_label"]],
        dtype=torch.long,
        device=device,
    )
    return ModelBatch(
        target=SequenceSegment(
            valid_mask=observed,
            frame_offsets=torch.arange(
                -(sequence_length - 1),
                1,
                device=device,
            ).repeat(len(rows), 1),
            images=images,
        ),
        labels=labels,
        native_unit_id=rows["temporal_unit_keys_json"].astype(str).tolist(),
        window_id=rows["window_id"].astype(str).tolist(),
    )


def _build_b1_model(view: str) -> nn.Module:
    _assert_b1_contract(view)
    return build_model(MODEL_ID, target_length=int(VIEW_SPECS[view]["length"]))


def _assert_b1_contract(view: str) -> None:
    _validate_view(view)
    config = baseline_config(MODEL_ID, target_length=int(VIEW_SPECS[view]["length"]))
    if config.batch_contract.target_length != int(VIEW_SPECS[view]["length"]):
        raise Stage1TemporalScreeningError("B1 target length drifted")
    if config.numeric is not None or config.batch_contract.required_modalities != ("actor_images",):
        raise Stage1TemporalScreeningError("B1 must not activate geometry, motion, ROI or social")
    if config.temporal is None or config.temporal.name != "causal_tcn":
        raise Stage1TemporalScreeningError("B1 temporal encoder drifted")


def _b1_effective_inputs(view: str) -> list[str]:
    config = baseline_config(MODEL_ID, target_length=int(VIEW_SPECS[view]["length"]))
    inputs = [f"actor_rgb_{view}", "causal_frame_offsets", "actor_observed_mask"]
    if config.control_names:
        inputs.append("registered_zero_default_quality_controls")
    if config.availability_names:
        inputs.append("registered_zero_default_availability_controls")
    return inputs


def _model_config_payload(view: str) -> dict[str, object]:
    return baseline_config(MODEL_ID, target_length=int(VIEW_SPECS[view]["length"])).to_payload()


def _assert_execution_hardware(plan: Stage1Plan) -> None:
    if plan.engineering_smoke:
        if plan.device_name != "cpu":
            raise Stage1TemporalScreeningError("engineering smoke is CPU-only")
        return
    if plan.device_name != "cuda":
        raise Stage1TemporalScreeningError("real Stage-1 execution requires CUDA")
    if not bool(plan.authority["stage_1_temporal_screening"]["gpu_execution_authorized"]):
        raise Stage1TemporalScreeningError("Stage-1 GPU execution remains unauthorized")
    if not torch.cuda.is_available():
        raise Stage1TemporalScreeningError("authorized Stage-1 CUDA route is unavailable")
    if torch.cuda.device_count() != 1:
        raise Stage1TemporalScreeningError("Stage-1 requires exactly one GPU")
    if torch.cuda.get_device_name(0) != "NVIDIA L4":
        raise Stage1TemporalScreeningError("Stage-1 requires exactly one NVIDIA L4")


def _validate_authority(authority: Mapping[str, Any]) -> None:
    if authority.get("schema_version") != AUTHORITY_SCHEMA:
        raise Stage1TemporalScreeningError("unsupported S1 authority schema")
    if authority.get("status") != "POST_CALIBRATION_HORIZON_FROZEN_STAGE1_CPU_PREFLIGHT_PENDING":
        raise Stage1TemporalScreeningError("Stage-1 authority status drifted")
    decision = authority.get("calibration_decision_authority")
    if not isinstance(decision, Mapping) or decision.get("selected_horizon") != MAX_STEPS:
        raise Stage1TemporalScreeningError("frozen 4164-step decision binding drifted")
    controls = authority.get("fixed_stage_1_to_4_controls", {})
    expected_controls = {
        **controls,
        "optimizer": "AdamW",
        "learning_rate": 0.003,
        "weight_decay": 0,
        "batch_size": BATCH_SIZE,
        "precision": "FP32",
        "scheduler": "none",
    }
    if controls != expected_controls:
        raise Stage1TemporalScreeningError("fixed Stage-1 controls drifted")
    policy = authority.get("matched_training_policy", {})
    if (
        policy.get("fixed_training_steps") != MAX_STEPS
        or policy.get("matched_stage_1_to_4_max_steps") != MAX_STEPS
        or policy.get("training_budget_unit") != "OPTIMIZER_STEPS"
        or policy.get("fixed_epochs_allowed") is not False
    ):
        raise Stage1TemporalScreeningError("common optimizer-step policy drifted")
    stage1 = authority.get("stage_1_temporal_screening", {})
    if (
        stage1.get("run_kind") != RUN_KIND
        or stage1.get("temporal_views") != list(VIEW_SPECS)
        or stage1.get("initial_seed") != SEED
        or stage1.get("max_steps") != MAX_STEPS
        or tuple(stage1.get("evaluation_steps", ())) != EVAL_STEPS
        or stage1.get("training_budget_unit") != "OPTIMIZER_STEPS"
        or stage1.get("early_stopping") != "DISABLED"
        or stage1.get("scientific_ranking_checkpoint") != "FIXED_STEP_ENDPOINT"
        or stage1.get("primary_metric") != "NATIVE_UNIT_10_CLASS_MACRO_F1"
        or stage1.get("primary_evaluation_path") != "EXPLICIT_WINDOW_TO_NATIVE_COLLAPSE"
        or stage1.get("outer_access_allowed") is not False
    ):
        raise Stage1TemporalScreeningError("Stage-1 temporal-screening authority drifted")
    common = stage1.get("common_cohort", {})
    if common.get("native_units") != 27378 or common.get("diagnostic_only", True) is False:
        raise Stage1TemporalScreeningError("Stage-1 common-cohort authority drifted")
    inner = authority.get("inner_role_binding", {})
    if inner.get("forbidden_roles") != ["test", "outer", "q2_outer_00"]:
        raise Stage1TemporalScreeningError("outer refusal authority drifted")
    primary = authority.get("primary_evaluation", {})
    if (
        primary.get("required_path") != "EXPLICIT_WINDOW_TO_NATIVE_COLLAPSE"
        or primary.get("composite_key_direct_primary_metric_allowed") is not False
    ):
        raise Stage1TemporalScreeningError("primary evaluator authority drifted")


def _validate_calibration_decision(path: Path, authority_ref: Mapping[str, object]) -> None:
    expected_hash = str(authority_ref.get("sha256", ""))
    if _sha256_file(path) != expected_hash:
        raise Stage1TemporalScreeningError("calibration horizon decision hash mismatch")
    decision = _read_json(path)
    if (
        decision.get("schema_version") != DECISION_SCHEMA
        or decision.get("status") != "FROZEN"
        or decision.get("selection", {}).get("selected_horizon") != MAX_STEPS
        or decision.get("selection", {}).get("training_budget_unit") != "OPTIMIZER_STEPS"
        or decision.get("selection", {}).get("selection_rule")
        != "REGISTERED_TRAJECTORY_REVIEW"
    ):
        raise Stage1TemporalScreeningError("calibration horizon decision content drifted")


def _assert_permitted_scope(value: str) -> None:
    normalized = str(value).strip().lower()
    if normalized not in {"train", "validation"} or normalized in FORBIDDEN_SCOPE_TOKENS:
        raise Stage1TemporalScreeningError("outer/test role refused before dataset payload access")


def _reject_frozen_overrides(overrides: Mapping[str, object] | None) -> None:
    if overrides:
        raise Stage1TemporalScreeningError(
            f"frozen Stage-1 field override refused={sorted(overrides)}"
        )


def _assert_safe_output_root(
    path: Path,
    *,
    outputs_root: Path,
    engineering_smoke: bool,
) -> None:
    leaf = path.name.lower()
    if any(token in leaf for token in FORBIDDEN_SCOPE_TOKENS):
        raise Stage1TemporalScreeningError(
            "outer/test prediction or export root refused before dataset payload access"
        )
    real_namespace = (
        outputs_root
        / "classification_v2"
        / "s1_post_temporal_closure_20260809"
        / "s1_trials"
    ).resolve()
    if not engineering_smoke and not path.is_relative_to(real_namespace):
        raise Stage1TemporalScreeningError("Stage-1 output root is outside isolated namespace")


def _validate_view(view: str) -> None:
    if view not in VIEW_SPECS:
        raise Stage1TemporalScreeningError(f"unregistered Stage-1 temporal view={view}")


def _validate_trial_id(value: str) -> None:
    lowered = value.lower()
    if not value or any(token in lowered for token in FORBIDDEN_SCOPE_TOKENS):
        raise Stage1TemporalScreeningError("invalid or outer-scoped Stage-1 trial ID")
    if "pre_s1" in lowered or "calibration" in lowered:
        raise Stage1TemporalScreeningError("Stage-1 must not reuse a calibration run identity")


def _resolve_root(root: Path, parts: Sequence[str], *, outputs_root: Path) -> Path:
    if parts and parts[0] == "outputs":
        return outputs_root.joinpath(*parts[1:])
    return root.joinpath(*parts)


def _verify_artifact(path: Path, expected: str, label: str) -> str:
    observed = _sha256_file(path)
    if observed != expected:
        raise Stage1TemporalScreeningError(f"{label} hash mismatch")
    return observed


def _repository_root(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if (parent / ".git").exists():
            return parent
    raise Stage1TemporalScreeningError("could not resolve repository root for Stage-1")


def _strict_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise Stage1TemporalScreeningError(f"{label} missing columns={missing}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Stage1TemporalScreeningError(f"JSON object required={path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Mapping[str, Any]) -> str:
    import hashlib

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


def _new_trial_id(view: str, engineering_smoke: bool) -> str:
    prefix = "s1_stage1_engineering" if engineering_smoke else "s1_stage1"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{view.lower()}_seed{SEED}_{timestamp}_{uuid.uuid4().hex[:8]}"


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
    "BATCH_SIZE",
    "DATA_BINDINGS_SCHEMA",
    "EVAL_STEPS",
    "MAX_STEPS",
    "RUN_KIND",
    "SEED",
    "Stage1Plan",
    "Stage1Population",
    "Stage1Rows",
    "Stage1TemporalScreeningError",
    "VIEW_SPECS",
    "create_stage1_plan",
    "load_stage1_inner_rows",
    "load_stage1_population",
    "preflight_stage1",
    "run_real_data_cpu_engineering_smoke",
    "run_real_data_cpu_preflight",
    "run_stage1_temporal_screening",
]
