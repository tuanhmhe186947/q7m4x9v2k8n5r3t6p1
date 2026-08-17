"""Fail-closed T6-only post-S1 pure-resolution executor.

This module deliberately does not alter the frozen 64x64 Stage-1 executor.
It reuses its hash-bound inner T6 rows and native evaluator while applying the
already-audited source crop, aspect-preserving letterbox, and normalization at
one explicitly declared runtime spatial resolution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from pig_behavior.classification_v2.datasets.image_sequence_dataset import (
    image_sequence_collate,
)
from pig_behavior.classification_v2.datasets.resolution_pipeline import (
    ResolutionIndependentRGBBinding,
    build_inner_resolution_binding_from_dataframes,
    validate_runtime_resolution,
)
from pig_behavior.classification_v2.models.balanced.contracts import (
    ModelBatch,
    SequenceSegment,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training import stage1_temporal_screening as stage1
from pig_behavior.classification_v2.training.post_s1_host_binding import (
    PostS1HostBindingError,
    ensure_post_s1_t6_host_binding,
)
from pig_behavior.classification_v2.training.remote_input_resolution import (
    RemoteInputResolutionError,
    load_remote_input_authority,
)
from pig_behavior.classification_v2.training.temporal_v2_consumer import (
    TemporalV2ConsumerInput,
    build_resolution_temporal_v2_input,
)

AUTHORITY_SCHEMA = "classification_v2.post_s1_resolution_screen.v1"
RUN_KIND = "POST_S1_T6_PURE_RESOLUTION_SCREEN"
TEMPORAL_VIEW = "T6"
SEED = 20260804
MAX_STEPS = 4164
BATCH_SIZE = 16
RESOLUTION_ARMS = frozenset({64, 128, 160})

# The historical post-S1 resolver above remains available for its existing
# local regression tests.  The cloud recovery command below is a separate,
# fail-closed production route with the newly registered trial seed.  It never
# calls the historical media resolver.
R128_RUNTIME_SCHEMA = "classification_v2.cloud_r128_runtime.v1"
R128_TEMPORAL_VIEW = "T6"
R128_RESOLUTION = 128
R128_SEED = 20260814
R128_MATCHED_SEEDS = frozenset({20260804, 20260805, 20260806})
R128_MAX_STEPS = 4164
R128_BATCH_SIZE = 16
R128_TRAIN_ROWS = 12421
R128_VALIDATION_ROWS = 2285
R128_CACHE_ROWS = 245680
R128_CACHE_BYTES = 12075663488
R128_INDEX_BYTES = 47781243
R128_CACHE_SHA256 = "c352a74cade4587e9dcbb8c3eead0c095c992306549b53da6d8b2a361691f5ee"
R128_INDEX_SHA256 = "9ccef8607973cfb8c8377474665af5d62874b5beea39ad716872b187f8d29d68"
R128_FORBIDDEN_FALLBACKS = {
    "raw_video_fallback_used": False,
    "loose_crop_fallback_used": False,
    "cache_build_on_studio": False,
}


def load_canonical_resolution_temporal_target(
    *, input_resolution: int, **kwargs: Any
) -> TemporalV2ConsumerInput:
    """Build an R64/R128/R160 temporal input from emitted membership only."""

    return build_resolution_temporal_v2_input(
        input_resolution=input_resolution,
        **kwargs,
    )


class PostS1ResolutionError(ValueError):
    """Raised before an unsafe post-S1 resolution action creates model state."""


@dataclass(frozen=True, slots=True)
class ResolutionPlan:
    """One immutable post-S1 pure-resolution arm."""

    repository_root: Path
    outputs_root: Path
    authority_path: Path
    authority_sha256: str
    authority: Mapping[str, Any]
    base_stage1_authority_path: Path
    host_binding_path: Path
    canonical_code_sha: str
    cvat_source_registration_path: Path
    rgb_source_root: Path
    runtime_input_authority_path: Path
    runtime_input_binding: Mapping[str, Any]
    media_root: Path
    output_dir: Path
    stage1_data_bindings_path: Path
    stage1_binding_bundle_path: Path
    execution_permit_path: Path
    trial_id: str
    input_resolution: int
    device_name: str


@dataclass(slots=True)
class ResolutionPopulation:
    """Inner-only B1 data with an on-the-fly resolution realization."""

    rows: stage1.Stage1Rows
    stage1_plan: stage1.Stage1Plan
    binding: ResolutionIndependentRGBBinding
    data_hashes: Mapping[str, str]
    load_batch: Any
    close: Any


def create_resolution_plan(
    authority_path: Path,
    *,
    repository_root: Path,
    outputs_root: Path,
    stage1_data_bindings_path: Path,
    stage1_binding_bundle_path: Path,
    execution_permit_path: Path,
    base_stage1_authority_path: Path,
    host_binding_path: Path,
    canonical_code_sha: str,
    rgb_source_root: Path,
    runtime_input_authority_path: Path,
    runtime_input_binding_path: Path,
    media_root: Path,
    output_dir: Path,
    trial_id: str,
    input_resolution: int,
    device_name: str,
) -> ResolutionPlan:
    """Resolve one registered R64/R128/R160 arm before opening RGB media."""

    if device_name not in {"cpu", "cuda"}:
        raise PostS1ResolutionError(f"unsupported device={device_name}")
    if input_resolution not in RESOLUTION_ARMS:
        raise PostS1ResolutionError(f"unregistered resolution={input_resolution}")
    validate_runtime_resolution(input_resolution)
    authority_path = Path(authority_path).resolve()
    authority = _read_json(authority_path)
    _validate_authority(authority)
    expected_trial = f"post_s1_t6_r{input_resolution}_seed{SEED}_steps{MAX_STEPS}"
    if trial_id != expected_trial:
        raise PostS1ResolutionError(f"trial_id must equal {expected_trial}")
    resolved_output = Path(output_dir).resolve()
    if resolved_output.exists():
        raise PostS1ResolutionError(f"output already exists={resolved_output}")
    if "outer" in str(resolved_output).lower():
        raise PostS1ResolutionError("outer token in output path")
    resolved_repository = Path(repository_root).resolve()
    resolved_media_root = Path(media_root).resolve()
    resolved_input_authority = Path(runtime_input_authority_path).resolve()
    runtime_input_binding = _validate_runtime_input_binding(
        authority,
        repository_root=resolved_repository,
        authority_path=resolved_input_authority,
        binding_path=Path(runtime_input_binding_path).resolve(),
        media_root=resolved_media_root,
    )
    registration_path = _validate_cvat_source_registration(
        authority,
        repository_root=resolved_repository,
    )
    return ResolutionPlan(
        repository_root=resolved_repository,
        outputs_root=Path(outputs_root).resolve(),
        authority_path=authority_path,
        authority_sha256=_sha256_file(authority_path),
        authority=authority,
        base_stage1_authority_path=Path(base_stage1_authority_path).resolve(),
        host_binding_path=Path(host_binding_path).resolve(),
        canonical_code_sha=canonical_code_sha,
        cvat_source_registration_path=registration_path,
        rgb_source_root=Path(rgb_source_root).resolve(),
        runtime_input_authority_path=resolved_input_authority,
        runtime_input_binding=runtime_input_binding,
        media_root=resolved_media_root,
        output_dir=resolved_output,
        stage1_data_bindings_path=Path(stage1_data_bindings_path).resolve(),
        stage1_binding_bundle_path=Path(stage1_binding_bundle_path).resolve(),
        execution_permit_path=Path(execution_permit_path).resolve(),
        trial_id=trial_id,
        input_resolution=input_resolution,
        device_name=device_name,
    )


def load_resolution_population(plan: ResolutionPlan) -> ResolutionPopulation:
    """Bind the frozen T6 inner rows to the high-fidelity source realization."""

    base = stage1.create_stage1_plan(
        plan.base_stage1_authority_path,
        view=TEMPORAL_VIEW,
        seed=SEED,
        repository_root=plan.repository_root,
        outputs_root=plan.outputs_root,
        output_dir=plan.output_dir,
        data_bindings_path=plan.stage1_data_bindings_path,
        execution_permit_path=plan.execution_permit_path,
        binding_bundle_path=plan.stage1_binding_bundle_path,
        trial_id="s1_stage1_t6_seed20260804_steps4164",
        device_name=plan.device_name,
    )
    hashes = stage1.preflight_stage1(base)
    rows = stage1.load_stage1_inner_rows(base, hashes)
    requested_roles = pd.concat(
        [
            rows.train[["window_id", "primary_s1_role"]],
            rows.validation[["window_id", "primary_s1_role"]],
        ],
        ignore_index=True,
    )
    try:
        host_binding = ensure_post_s1_t6_host_binding(
            binding_path=plan.host_binding_path,
            canonical_code_sha=plan.canonical_code_sha,
            input_authority=load_remote_input_authority(plan.runtime_input_authority_path),
            runtime_input_binding=plan.runtime_input_binding,
            media_root=plan.media_root,
            rgb_source_root=plan.rgb_source_root,
            t6_population_authority_sha256=base.authority_sha256,
            t6_population_provenance_hashes=rows.data_hashes,
            requested_roles=requested_roles,
            input_resolution=plan.input_resolution,
            cvat_source_registration_path=plan.cvat_source_registration_path,
        )
    except (PostS1HostBindingError, RemoteInputResolutionError) as error:
        raise PostS1ResolutionError(str(error)) from error
    rgb = host_binding.rgb
    frames = pd.read_csv(rgb.frame_context_path, low_memory=False)
    windows = pd.read_csv(rgb.window_context_path, low_memory=False)
    selected = pd.concat([rows.train, rows.validation], ignore_index=True).copy()
    index_by_window = {
        str(window_id): index
        for index, window_id in enumerate(windows["window_id"])
    }
    selected["window_row_index"] = selected["window_id"].astype(str).map(index_by_window)
    if selected["window_row_index"].isna().any():
        raise PostS1ResolutionError("frozen T6 row is absent from RGB binding")
    selected["window_row_index"] = selected["window_row_index"].astype(int)
    selected["window_valid_for_main_train"] = True
    selected["primary_s1_eligible"] = True
    binding = build_inner_resolution_binding_from_dataframes(
        frames=frames,
        windows=windows,
        selection=selected,
        media_root=plan.media_root,
        expected_window_count=len(selected),
        expected_observation_count=201792,
    )
    packed_npy = (
        plan.rgb_source_root
        / f"actor_rgb_{plan.input_resolution}_full"
        / f"packed_rgb_{plan.input_resolution}_letterbox.npy"
    )
    packed_idx = (
        plan.rgb_source_root
        / f"actor_rgb_{plan.input_resolution}_full"
        / "packed_image_cache_index.csv"
    )
    if not (packed_npy.exists() and packed_idx.exists()):
        candidate_npy = (
            plan.rgb_source_root
            / f"packed_rgb_{plan.input_resolution}_letterbox.npy"
        )
        candidate_idx = plan.rgb_source_root / "packed_image_cache_index.csv"
        if candidate_npy.exists() and candidate_idx.exists():
            packed_npy, packed_idx = candidate_npy, candidate_idx
        else:
            packed_npy, packed_idx = None, None
    if packed_npy is None or packed_idx is None:
        raise PostS1ResolutionError(
            "prepared packed RGB cache is unavailable; source-media fallback is forbidden"
        )

    dataset = binding.build_dataset(
        plan.input_resolution,
        image_cache_size=8192,
        packed_image_cache_npy=packed_npy,
        packed_image_cache_index_csv=packed_idx,
        require_packed_cache=True,
    )
    lookup = {str(window_id): index for index, window_id in enumerate(dataset.windows["window_id"])}

    def load_batch(selected_rows: pd.DataFrame, device: torch.device) -> ModelBatch:
        subset = Subset(
            dataset,
            [lookup[str(value)] for value in selected_rows["window_id"]],
        )
        payload = next(
            iter(
                DataLoader(
                    subset,
                    batch_size=len(selected_rows),
                    shuffle=False,
                    collate_fn=image_sequence_collate,
                )
            )
        )
        errors = [error for group in payload["errors"] for error in group]
        if errors:
            raise PostS1ResolutionError(f"source RGB payload failures={errors[:5]}")
        if payload["window_id"] != selected_rows["window_id"].astype(str).tolist():
            raise PostS1ResolutionError("source RGB payload order drifted")
        return _make_batch(
            payload["image"].to(device),
            payload["observed_mask"].to(device),
            selected_rows,
            device,
            plan.input_resolution,
        )

    return ResolutionPopulation(
        rows=rows,
        stage1_plan=base,
        binding=binding,
        data_hashes={
            **rows.data_hashes,
            **rgb.hashes,
            "resolution_authority": plan.authority_sha256,
            "post_s1_host_binding": host_binding.binding_sha256,
            "rgb_identity": binding.identity_sha256,
            "runtime_realization": binding.runtime_realization(
                plan.input_resolution
            )["runtime_realization_sha256"],
        },
        load_batch=load_batch,
        close=dataset.close,
    )


def run_resolution_arm(
    plan: ResolutionPlan,
    population: ResolutionPopulation,
    *,
    steps: int = MAX_STEPS,
) -> dict[str, Any]:
    """Run one short gate or one exact 4,164-step L4 arm without resuming."""

    if steps <= 0 or steps > MAX_STEPS:
        raise PostS1ResolutionError("steps must be in 1..4164")
    if steps == MAX_STEPS:
        _assert_l4(plan)
        inherited_population = stage1.Stage1Population(
            train=population.rows.train,
            validation=population.rows.validation,
            expected_native_units=population.rows.expected_native_units,
            common_cohort_native_units=population.rows.common_cohort_native_units,
            load_batch=population.load_batch,
            close=population.close,
            data_hashes=population.data_hashes,
        )
        return stage1.run_stage1_temporal_screening(
            population.stage1_plan,
            inherited_population,
        )
    elif plan.device_name != "cpu":
        raise PostS1ResolutionError("representative short gate is CPU-only")
    plan.output_dir.mkdir(parents=True, exist_ok=False)
    for folder in ("manifest", "checkpoints", "predictions", "metrics", "runtime"):
        (plan.output_dir / folder).mkdir(exist_ok=False)
    device = torch.device(plan.device_name)
    if plan.device_name == "cuda":
        torch.cuda.reset_peak_memory_stats(0)
    stage1._set_seed(SEED)
    model = stage1._build_b1_model(TEMPORAL_VIEW).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    losses: list[float] = []
    started = time.perf_counter()
    try:
        for step in range(1, steps + 1):
            selected = stage1._rows_for_step(
                population.rows.train,
                step=step,
                batch_size=BATCH_SIZE,
                seed=SEED,
            )
            batch = population.load_batch(selected, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)["logits"]
            weights = torch.tensor(
                selected["event_sample_weight"].to_numpy(np.float32),
                device=device,
            )
            loss = (
                nn.functional.cross_entropy(logits, batch.labels, reduction="none")
                * weights
            ).sum() / weights.sum()
            if not bool(torch.isfinite(loss)):
                raise PostS1ResolutionError("non-finite training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        checkpoint = plan.output_dir / "checkpoints" / f"step_{steps:06d}.pt"
        torch.save(
            {
                "trial_id": plan.trial_id,
                "steps": steps,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint,
        )
        result: dict[str, Any] = {
            "status": "PASS",
            "run_kind": RUN_KIND,
            "trial_id": plan.trial_id,
            "input_resolution": plan.input_resolution,
            "completed_steps": steps,
            "losses": losses,
            "claim_grade_result": False,
            "outer_examples_accessed": False,
        }
        if steps == MAX_STEPS:
            stage_population = stage1.Stage1Population(
                train=population.rows.train,
                validation=population.rows.validation,
                expected_native_units=population.rows.expected_native_units,
                common_cohort_native_units=population.rows.common_cohort_native_units,
                load_batch=population.load_batch,
                close=population.close,
                data_hashes=population.data_hashes,
            )
            result["endpoint"] = stage1._evaluate_endpoint(
                plan,
                stage_population,
                model,
                device,
                steps,
                losses,
            )
        elapsed = time.perf_counter() - started
        result["runtime"] = _runtime(plan, population, elapsed, steps)
        _write_json(
            plan.output_dir / "manifest" / "run_manifest.json",
            _manifest(plan, population, steps),
        )
        _write_json(plan.output_dir / "manifest" / "result.json", result)
        _write_json(plan.output_dir / "runtime" / "runtime.json", result["runtime"])
        result["artifact_manifest"] = stage1._write_artifact_manifest(plan.output_dir)
        return result
    finally:
        population.close()


def _make_batch(
    images: torch.Tensor,
    observed: torch.Tensor,
    rows: pd.DataFrame,
    device: torch.device,
    resolution: int,
) -> ModelBatch:
    if tuple(images.shape[1:]) != (6, 3, resolution, resolution):
        raise PostS1ResolutionError(f"RGB tensor shape drifted={tuple(images.shape)}")
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


def _validate_authority(authority: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "status",
        "temporal_reference",
        "arms",
        "fixed_controls",
        "bound_stage1_authorities",
        "runtime_input_authority",
        "cvat_source_registration",
        "outer_access_allowed",
        "forbidden_families",
    }
    if set(authority) != required or authority.get("schema_version") != AUTHORITY_SCHEMA:
        raise PostS1ResolutionError("unsupported post-S1 resolution authority")
    if (
        authority.get("status") != "USER_APPROVED_PRE_GPU_SCREEN"
        or authority.get("temporal_reference") != TEMPORAL_VIEW
    ):
        raise PostS1ResolutionError("post-S1 temporal reference drifted")
    if (
        authority.get("arms") != [64, 128, 160]
        or authority.get("outer_access_allowed") is not False
    ):
        raise PostS1ResolutionError("post-S1 resolution-arm or outer boundary drifted")
    if authority.get("forbidden_families") != [
        "backbone",
        "augmentation",
        "crop_margin",
        "geometry",
        "motion",
        "roi",
        "social",
        "h5",
        "posture",
    ]:
        raise PostS1ResolutionError("post-S1 forbidden-family boundary drifted")


def _validate_runtime_input_binding(
    authority: Mapping[str, Any],
    *,
    repository_root: Path,
    authority_path: Path,
    binding_path: Path,
    media_root: Path,
) -> Mapping[str, Any]:
    """Require a verified runtime path without making it dataset identity."""

    expected = authority["runtime_input_authority"]
    if not isinstance(expected, Mapping):
        raise PostS1ResolutionError("runtime input authority binding is invalid")
    relative_segments = expected.get("relative_segments")
    filename = expected.get("filename")
    expected_sha256 = expected.get("sha256")
    if (
        not isinstance(relative_segments, list)
        or not all(isinstance(segment, str) and segment for segment in relative_segments)
        or not isinstance(filename, str)
        or not isinstance(expected_sha256, str)
    ):
        raise PostS1ResolutionError("runtime input authority binding is incomplete")
    if authority_path != (repository_root.joinpath(*relative_segments) / filename).resolve():
        raise PostS1ResolutionError("runtime input authority path drifted")
    if _sha256_file(authority_path) != expected_sha256:
        raise PostS1ResolutionError("runtime input authority hash drifted")
    try:
        input_authority = load_remote_input_authority(authority_path)
    except RemoteInputResolutionError as error:
        raise PostS1ResolutionError(str(error)) from error
    binding = _read_json(binding_path)
    effective_root = binding.get("effective_remote_input_root")
    if binding.get("scientific_input_authority_id") != input_authority.authority_id:
        raise PostS1ResolutionError("runtime input scientific authority drifted")
    if not isinstance(effective_root, str) or Path(effective_root).resolve() != media_root:
        raise PostS1ResolutionError("media root does not match verified runtime input root")
    if (
        binding.get("expected_file_count") != input_authority.expected_file_count
        or binding.get("expected_total_bytes") != input_authority.expected_total_bytes
    ):
        raise PostS1ResolutionError("runtime input population parity drifted")
    return binding


def _validate_cvat_source_registration(
    authority: Mapping[str, Any],
    *,
    repository_root: Path,
) -> Path:
    """Resolve only the hash-bound source-registration authority in the bundle."""

    reference = authority.get("cvat_source_registration")
    if not isinstance(reference, Mapping) or set(reference) != {
        "relative_segments",
        "filename",
        "sha256",
    }:
        raise PostS1ResolutionError("CVAT source-registration reference is invalid")
    segments = reference.get("relative_segments")
    filename = reference.get("filename")
    expected_sha256 = reference.get("sha256")
    if (
        not isinstance(segments, list)
        or not segments
        or any(not isinstance(segment, str) or segment in {"", ".", ".."} for segment in segments)
        or not isinstance(filename, str)
        or filename in {"", ".", ".."}
        or "/" in filename
        or "\\" in filename
        or not isinstance(expected_sha256, str)
    ):
        raise PostS1ResolutionError("CVAT source-registration reference is incomplete")
    candidate = repository_root.joinpath(*segments, filename).resolve()
    if not candidate.is_relative_to(repository_root) or not candidate.is_file():
        raise PostS1ResolutionError("CVAT source-registration authority is unavailable")
    if _sha256_file(candidate) != expected_sha256:
        raise PostS1ResolutionError("CVAT source-registration authority hash drifted")
    return candidate


def _assert_l4(plan: ResolutionPlan) -> None:
    if (
        plan.device_name != "cuda"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise PostS1ResolutionError("full resolution arm requires exactly one CUDA device")
    if torch.cuda.get_device_name(0) != "NVIDIA L4":
        raise PostS1ResolutionError("full resolution arm requires NVIDIA L4")


def _manifest(
    plan: ResolutionPlan,
    population: ResolutionPopulation,
    steps: int,
) -> dict[str, Any]:
    return {
        "trial_id": plan.trial_id,
        "run_kind": RUN_KIND,
        "temporal_view": TEMPORAL_VIEW,
        "input_resolution": plan.input_resolution,
        "seed": SEED,
        "max_steps": steps,
        "optimizer": "AdamW",
        "learning_rate": 0.003,
        "weight_decay": 0.0,
        "batch_size": BATCH_SIZE,
        "precision": "FP32",
        "scheduler": "none",
        "outer_examples_accessed": False,
        "authority_sha256": plan.authority_sha256,
        "data_hashes": dict(population.data_hashes),
        "runtime_input": dict(plan.runtime_input_binding),
        "cvat_source_registration": {
            "path": str(plan.cvat_source_registration_path),
            "sha256": _sha256_file(plan.cvat_source_registration_path),
        },
        "host_binding": {
            "path": str(plan.host_binding_path),
            "canonical_code_sha": plan.canonical_code_sha,
        },
    }


def _runtime(
    plan: ResolutionPlan,
    population: ResolutionPopulation,
    elapsed: float,
    steps: int,
) -> dict[str, Any]:
    cuda = torch.cuda if torch.cuda.is_available() else None
    return {
        "trial_id": plan.trial_id,
        "run_kind": RUN_KIND,
        "input_resolution": plan.input_resolution,
        "wall_clock_seconds": elapsed,
        "training_seconds": elapsed,
        "completed_steps": steps,
        "seconds_per_step": elapsed / steps,
        "gpu_name": cuda.get_device_name(0) if cuda else None,
        "gpu_count": cuda.device_count() if cuda else 0,
        "peak_vram_allocated_mb": (
            float(cuda.max_memory_allocated(0)) / (1024 * 1024) if cuda else 0.0
        ),
        "data_hashes": dict(population.data_hashes),
    }


def _read_json(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class R128RuntimeError(RuntimeError):
    """Raised when the cloud-only R128 route cannot prove its contract."""


def _r128_split_sequence(value: object, delimiter: str) -> list[str]:
    text = str(value)
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return []
    return [item for item in text.split(delimiter) if item]


def _r128_strict_bool(series: pd.Series, name: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0"}
    if not normalized.isin(allowed).all():
        raise R128RuntimeError(f"{name} contains non-boolean values")
    return normalized.isin({"true", "1"})


def _r128_bool_sequence(value: object, name: str) -> list[bool]:
    values = _r128_split_sequence(value, ";;")
    if len(values) != 6:
        raise R128RuntimeError(f"{name} must contain six frame masks")
    normalized = [item.strip().lower() for item in values]
    allowed = {"true", "false", "1", "0"}
    if not set(normalized).issubset(allowed):
        raise R128RuntimeError(f"{name} contains non-boolean values")
    return [item in {"true", "1"} for item in normalized]


def _r128_load_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise R128RuntimeError(f"runtime manifest is unavailable: {path}")
    rows = pd.read_csv(path, low_memory=False)
    required = {
        "window_id",
        "view_type",
        "window_length_frames",
        "temporal_unit_keys_json",
        "temporal_unit_key",
        "behavior_window_label",
        "primary_s1_role",
        "primary_s1_eligible",
        "window_valid_for_main_train",
        "event_sample_weight",
        "image_context_id_sequence",
        "expected_frame_indices",
        "observed_mask_sequence",
        "source_type",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise R128RuntimeError(f"runtime manifest columns missing: {missing}")
    if rows.empty or rows["window_id"].astype(str).duplicated().any():
        raise R128RuntimeError("runtime manifest is empty or has duplicate windows")
    if not rows["view_type"].astype(str).eq("T6_contiguous").all():
        raise R128RuntimeError("runtime manifest contains a non-T6 view")
    lengths = pd.to_numeric(rows["window_length_frames"], errors="coerce")
    if not lengths.eq(6).all():
        raise R128RuntimeError("runtime manifest is not six frames per window")
    rows["primary_s1_eligible"] = _r128_strict_bool(
        rows["primary_s1_eligible"], "primary_s1_eligible"
    )
    rows["window_valid_for_main_train"] = _r128_strict_bool(
        rows["window_valid_for_main_train"], "window_valid_for_main_train"
    )
    if not rows["primary_s1_eligible"].all():
        raise R128RuntimeError("ineligible rows entered the runtime manifest")
    if not rows["window_valid_for_main_train"].all():
        raise R128RuntimeError("invalid rows entered the runtime manifest")
    roles = rows["primary_s1_role"].astype(str)
    if not roles.isin({"train", "validation"}).all():
        raise R128RuntimeError("runtime manifest contains an outer/test role")
    role_counts = roles.value_counts().to_dict()
    if role_counts != {"train": R128_TRAIN_ROWS, "validation": R128_VALIDATION_ROWS}:
        raise R128RuntimeError(f"matched-T6 role counts drifted: {role_counts}")
    labels = rows["behavior_window_label"].astype(str)
    if not labels.isin(VALID_BEHAVIORS).all():
        raise R128RuntimeError("runtime manifest contains an unsupported label")
    weights = pd.to_numeric(rows["event_sample_weight"], errors="coerce")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise R128RuntimeError("runtime manifest contains invalid event weights")
    for row in rows.itertuples(index=False):
        context_ids = _r128_split_sequence(row.image_context_id_sequence, ";;")
        frame_indices = _r128_split_sequence(row.expected_frame_indices, "|")
        if len(context_ids) != 6 or len(frame_indices) != 6:
            raise R128RuntimeError(f"T6 context/frame order is incomplete: {row.window_id}")
        try:
            numeric_frames = [int(value) for value in frame_indices]
        except ValueError as error:
            raise R128RuntimeError("runtime frame indices are not integers") from error
        if numeric_frames != list(range(numeric_frames[0], numeric_frames[0] + 6)):
            raise R128RuntimeError(f"T6 frame ordering drifted: {row.window_id}")
        _r128_bool_sequence(row.observed_mask_sequence, "observed_mask_sequence")
    return rows.reset_index(drop=True)


class DriveR128Dataset(Dataset[dict[str, object]]):
    """Read T6 sequences exclusively from the registered packed Drive cache."""

    def __init__(self, rows: pd.DataFrame, packed_npy: Path, packed_index: Path) -> None:
        self.rows = rows.reset_index(drop=True)
        self.packed_npy = Path(packed_npy)
        self.packed_index = Path(packed_index)
        if self.packed_npy.stat().st_size != R128_CACHE_BYTES:
            raise R128RuntimeError("packed R128 tensor byte size drifted")
        if self.packed_index.stat().st_size != R128_INDEX_BYTES:
            raise R128RuntimeError("packed R128 index byte size drifted")
        # Keep the authoritative file on Drive while avoiding random-read
        # latency for every six-frame batch over the shared filesystem.
        self.tensor = np.load(self.packed_npy, allow_pickle=False)
        expected_shape = (R128_CACHE_ROWS, R128_RESOLUTION, R128_RESOLUTION, 3)
        if self.tensor.dtype != np.uint8 or tuple(self.tensor.shape) != expected_shape:
            raise R128RuntimeError(
                f"packed R128 tensor contract drifted: {self.tensor.dtype} {self.tensor.shape}"
            )
        index = pd.read_csv(self.packed_index, low_memory=False)
        if not {"image_context_id", "packed_row"}.issubset(index.columns):
            raise R128RuntimeError("packed R128 index required columns are missing")
        if len(index) != R128_CACHE_ROWS:
            raise R128RuntimeError("packed R128 index row count drifted")
        context_ids = index["image_context_id"].astype(str)
        if context_ids.duplicated().any():
            raise R128RuntimeError("packed R128 index has duplicate context IDs")
        packed_rows = pd.to_numeric(index["packed_row"], errors="coerce")
        if packed_rows.isna().any() or (packed_rows < 0).any():
            raise R128RuntimeError("packed R128 index has invalid packed rows")
        if (packed_rows >= R128_CACHE_ROWS).any():
            raise R128RuntimeError("packed R128 index points outside the tensor")
        self.context_to_row = dict(zip(context_ids, packed_rows.astype(int), strict=True))
        for row in self.rows.itertuples(index=False):
            context_ids = _r128_split_sequence(row.image_context_id_sequence, ";;")
            if any(context_id not in self.context_to_row for context_id in context_ids):
                raise R128RuntimeError(f"Drive cache lacks a T6 context: {row.window_id}")
        self.packed_reads = 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows.iloc[index]
        context_ids = _r128_split_sequence(row["image_context_id_sequence"], ";;")
        packed_rows = [self.context_to_row[context_id] for context_id in context_ids]
        cached = np.asarray(self.tensor[packed_rows], dtype=np.uint8)
        if tuple(cached.shape) != (6, R128_RESOLUTION, R128_RESOLUTION, 3):
            raise R128RuntimeError(f"Drive cache batch shape drifted for {row['window_id']}")
        self.packed_reads += len(packed_rows)
        images = np.transpose(cached.astype(np.float32) / 255.0, (0, 3, 1, 2)).copy()
        label = VALID_BEHAVIORS.index(str(row["behavior_window_label"]))
        observed_mask = torch.tensor(
            _r128_bool_sequence(
                row["observed_mask_sequence"],
                "observed_mask_sequence",
            ),
            dtype=torch.float32,
        )
        return {
            "image": torch.from_numpy(images),
            "observed_mask": observed_mask,
            "label": label,
            "weight": float(row["event_sample_weight"]),
            "window_id": str(row["window_id"]),
            "native_unit_id": str(row["temporal_unit_key"]),
        }

    def audit(self) -> dict[str, object]:
        return {
            "packed_cache_configured": True,
            "packed_cache_hits": self.packed_reads,
            "source_image_loads": 0,
            "observed_mask_source": "canonical_sequence_frame_features",
            **R128_FORBIDDEN_FALLBACKS,
        }

    def close(self) -> None:
        self.tensor = None


@dataclass(slots=True)
class R128Population:
    rows: pd.DataFrame
    dataset: DriveR128Dataset
    row_by_window_id: dict[str, int]


def _r128_population(
    manifest: Path,
    packed_npy: Path,
    packed_index: Path,
) -> R128Population:
    rows = _r128_load_manifest(manifest)
    dataset = DriveR128Dataset(rows, packed_npy, packed_index)
    lookup = {
        str(window_id): int(index)
        for index, window_id in enumerate(rows["window_id"].astype(str))
    }
    return R128Population(rows, dataset, lookup)


def _r128_model_batch(
    selected: pd.DataFrame,
    population: R128Population,
    device: torch.device,
) -> tuple[ModelBatch, torch.Tensor]:
    window_ids = selected["window_id"].astype(str).tolist()
    try:
        indexes = [population.row_by_window_id[window_id] for window_id in window_ids]
    except KeyError as error:
        raise R128RuntimeError(
            f"selected window is absent from the Drive manifest: {error}"
        ) from error
    loader = DataLoader(
        Subset(population.dataset, indexes),
        batch_size=len(indexes),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    payload = next(iter(loader))
    images = payload["image"]
    if tuple(images.shape[1:]) != (6, 3, R128_RESOLUTION, R128_RESOLUTION):
        raise R128RuntimeError(f"ModelBatch image shape drifted: {tuple(images.shape)}")
    observed = payload["observed_mask"].to(device)
    batch_size = int(images.shape[0])
    frame_offsets = torch.arange(-5, 1, device=device).repeat(batch_size, 1)
    labels = payload["label"].to(device=device, dtype=torch.long)
    batch = ModelBatch(
        target=SequenceSegment(
            images=images.to(device),
            frame_offsets=frame_offsets,
            valid_mask=observed,
        ),
        labels=labels,
        window_id=payload["window_id"],
        native_unit_id=payload["native_unit_id"],
    )
    weights = payload["weight"].to(device=device, dtype=torch.float32)
    return batch, weights


def _r128_logits(model: nn.Module, batch: ModelBatch) -> torch.Tensor:
    outputs = model(batch)
    if not isinstance(outputs, Mapping) or "logits" not in outputs:
        raise R128RuntimeError("production model output must be a dict containing logits")
    logits = outputs["logits"]
    expected = (batch.target.batch_size, len(VALID_BEHAVIORS))
    if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != expected:
        raise R128RuntimeError(
            f"production model output contract drifted: {getattr(logits, 'shape', None)}"
        )
    return logits.float()


def _r128_loss(
    logits: torch.Tensor,
    batch: ModelBatch,
    weights: torch.Tensor,
) -> torch.Tensor:
    per_row = nn.functional.cross_entropy(logits, batch.labels, reduction="none")
    if tuple(weights.shape) != tuple(per_row.shape):
        raise R128RuntimeError("event-weight shape does not match the real batch")
    loss = (per_row * weights).sum() / weights.sum()
    if not bool(torch.isfinite(loss)):
        raise R128RuntimeError("real-data loss is non-finite")
    return loss


def _r128_model_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _r128_environment() -> dict[str, object]:
    import importlib.metadata

    packages: dict[str, object] = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "packages": packages,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
    }


def _r128_gpu_sample() -> dict[str, object]:
    if not torch.cuda.is_available():
        return {"gpu_utilization_percent": None, "allocated_vram_mb": 0.0}
    utilization: int | None = None
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
        utilization = int(completed.stdout.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        utilization = None
    return {
        "gpu_utilization_percent": utilization,
        "allocated_vram_mb": round(torch.cuda.memory_allocated(0) / 2**20, 3),
        "reserved_vram_mb": round(torch.cuda.memory_reserved(0) / 2**20, 3),
        "gpu_device": torch.cuda.get_device_name(0),
    }


def _r128_cache_descriptor(
    packed_npy: Path,
    packed_index: Path,
    manifest: Path,
    cache_uri: str,
) -> dict[str, object]:
    return {
        "source": "teamspace_drive_shared_training_storage",
        "cache_uri": cache_uri,
        "packed_npy_path": str(packed_npy),
        "packed_index_path": str(packed_index),
        "packed_npy_bytes": R128_CACHE_BYTES,
        "packed_index_bytes": R128_INDEX_BYTES,
        "packed_npy_sha256": R128_CACHE_SHA256,
        "packed_index_sha256": R128_INDEX_SHA256,
        "manifest_path": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        **R128_FORBIDDEN_FALLBACKS,
    }


def _r128_prepare_output(output_dir: Path) -> None:
    if output_dir.exists():
        raise R128RuntimeError(f"refusing to overwrite output directory: {output_dir}")
    for name in ("checkpoint", "predictions", "metrics", "logs", "runtime"):
        (output_dir / name).mkdir(parents=True, exist_ok=False)


def run_r128_cpu_preflight(
    *,
    manifest: Path,
    packed_npy: Path,
    packed_index: Path,
    output_dir: Path,
    cache_uri: str,
) -> dict[str, object]:
    """Run one disposable real-data CPU step; never save its model."""
    _r128_prepare_output(output_dir)
    population = _r128_population(manifest, packed_npy, packed_index)
    try:
        train = population.rows.loc[
            population.rows["primary_s1_role"].eq("train")
        ].head(R128_BATCH_SIZE)
        device = torch.device("cpu")
        model = stage1._build_b1_model(R128_TEMPORAL_VIEW).to(device)
        batch, weights = _r128_model_batch(train, population, device)
        batch_shape = tuple(batch.target.images.shape)
        if batch_shape != (
            R128_BATCH_SIZE,
            6,
            3,
            R128_RESOLUTION,
            R128_RESOLUTION,
        ):
            raise R128RuntimeError(f"real batch shape failed: {batch_shape}")
        logits = _r128_logits(model, batch)
        loss = _r128_loss(logits, batch, weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            raise R128RuntimeError("CPU backward produced a non-finite gradient")
        optimizer.step()
        report = {
            "schema": R128_RUNTIME_SCHEMA,
            "status": "PASS",
            "cpu_preflight": "PASS",
            "real_data_batch": "PASS",
            "real_labels": "PASS",
            "model_forward": "PASS",
            "loss_finite": "PASS",
            "backward": "PASS",
            "throwaway_optimizer_step": "PASS",
            "optimizer_steps": 1,
            "model_output_contract": {
                "type": "dict",
                "key": "logits",
                "shape": list(logits.shape),
            },
            "batch_shape": list(batch_shape),
            "counts": {
                "total": len(population.rows),
                "train": int(population.rows["primary_s1_role"].eq("train").sum()),
                "validation": int(
                    population.rows["primary_s1_role"].eq("validation").sum()
                ),
            },
            "cache": _r128_cache_descriptor(
                packed_npy, packed_index, manifest, cache_uri
            ),
            "cache_audit": population.dataset.audit(),
            "environment": _r128_environment(),
            "throwaway_model_discarded": True,
            "synthetic_result_path_used": False,
        }
        _write_json(output_dir / "runtime" / "cpu_preflight.json", report)
        return report
    finally:
        population.dataset.close()


def _r128_predict_validation(
    model: nn.Module,
    population: R128Population,
    device: torch.device,
) -> pd.DataFrame:
    validation = population.rows.loc[
        population.rows["primary_s1_role"].eq("validation")
    ].reset_index(drop=True)
    records: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(validation), R128_BATCH_SIZE):
            selected = validation.iloc[start : start + R128_BATCH_SIZE]
            batch, _ = _r128_model_batch(selected, population, device)
            probabilities = torch.softmax(_r128_logits(model, batch), dim=1)
            values = probabilities.cpu().numpy()
            for offset, row in enumerate(selected.itertuples(index=False)):
                predicted_index = int(values[offset].argmax())
                records.append(
                    {
                        "window_id": str(row.window_id),
                        "temporal_unit_key": str(row.temporal_unit_key),
                        "true_behavior": str(row.behavior_window_label),
                        "predicted_behavior": VALID_BEHAVIORS[predicted_index],
                        "confidence": float(values[offset, predicted_index]),
                    }
                )
    predictions = pd.DataFrame(records)
    expected_ids = set(validation["window_id"].astype(str))
    actual_ids = set(predictions["window_id"].astype(str))
    if len(predictions) != R128_VALIDATION_ROWS:
        raise R128RuntimeError(f"validation prediction count drifted: {len(predictions)}")
    if expected_ids != actual_ids or predictions["window_id"].duplicated().any():
        raise R128RuntimeError("validation prediction coverage is incomplete")
    return predictions


def _r128_per_class_metrics(prediction_path: Path, output_path: Path) -> dict[str, object]:
    from sklearn.metrics import precision_recall_fscore_support

    persisted = pd.read_csv(prediction_path, low_memory=False)
    if len(persisted) != R128_VALIDATION_ROWS:
        raise R128RuntimeError("persisted validation prediction count drifted")
    precision, recall, f1, support = precision_recall_fscore_support(
        persisted["true_behavior"],
        persisted["predicted_behavior"],
        labels=list(VALID_BEHAVIORS),
        zero_division=0,
    )
    metrics = pd.DataFrame(
        {
            "behavior": list(VALID_BEHAVIORS),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
    metrics.to_csv(output_path, index=False)
    return {
        "macro_f1": float(metrics["f1"].mean()),
        "per_class_metrics_path": str(output_path),
        "per_class_metrics_sha256": _sha256_file(output_path),
    }


def _r128_artifact_manifest(output_dir: Path) -> dict[str, object]:
    artifacts: list[dict[str, object]] = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append(
                {
                    "relative_path": str(path.relative_to(output_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    manifest = {
        "schema": "classification-v2.r128.artifact-manifest.v1",
        "status": "PASS",
        "artifacts": artifacts,
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)
    return manifest


def run_r128_trial(
    *,
    manifest: Path,
    packed_npy: Path,
    packed_index: Path,
    output_dir: Path,
    cache_uri: str,
    code_sha: str,
    runtime_bundle_sha: str,
    seed: int = R128_SEED,
) -> dict[str, object]:
    """Run one production T6/R128 trial for a registered matched seed."""
    if seed not in R128_MATCHED_SEEDS:
        raise R128RuntimeError(f"unregistered matched R128 seed: {seed}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise R128RuntimeError("R128 proof run requires exactly one visible CUDA device")
    if torch.cuda.get_device_name(0) != "NVIDIA L4":
        raise R128RuntimeError("R128 proof run requires NVIDIA L4")
    _r128_prepare_output(output_dir)
    population = _r128_population(manifest, packed_npy, packed_index)
    device = torch.device("cuda")
    log_path = output_dir / "logs" / "training_log.jsonl"
    gpu_samples: list[int] = []
    try:
        stage1._set_seed(seed)
        torch.cuda.reset_peak_memory_stats(0)
        model = stage1._build_b1_model(R128_TEMPORAL_VIEW).to(device)
        initial_hash = _r128_model_hash(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
        started = time.perf_counter()
        completed_steps = 0
        train = population.rows.loc[
            population.rows["primary_s1_role"].eq("train")
        ]
        with log_path.open("w", encoding="utf-8") as log:
            for step in range(1, R128_MAX_STEPS + 1):
                selected = stage1._rows_for_step(
                    train,
                    step=step,
                    batch_size=R128_BATCH_SIZE,
                    seed=seed,
                )
                batch, weights = _r128_model_batch(selected, population, device)
                optimizer.zero_grad(set_to_none=True)
                loss = _r128_loss(_r128_logits(model, batch), batch, weights)
                loss.backward()
                if not all(
                    parameter.grad is None
                    or bool(torch.isfinite(parameter.grad).all())
                    for parameter in model.parameters()
                ):
                    raise R128RuntimeError(f"non-finite gradient at step {step}")
                optimizer.step()
                completed_steps += 1
                sample = None
                if step == 1 or step % 25 == 0 or step == R128_MAX_STEPS:
                    sample = _r128_gpu_sample()
                    if sample["gpu_utilization_percent"] is not None:
                        gpu_samples.append(int(sample["gpu_utilization_percent"]))
                    print(
                        json.dumps(
                            {
                                "event": "optimizer_step",
                                "step": step,
                                "total_steps": R128_MAX_STEPS,
                                "loss": float(loss.detach().cpu()),
                                **sample,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                entry: dict[str, object] = {
                    "step": step,
                    "total_steps": R128_MAX_STEPS,
                    "loss": float(loss.detach().cpu()),
                }
                if sample is not None:
                    entry["gpu"] = sample
                log.write(json.dumps(entry, sort_keys=True) + "\n")
                log.flush()
        elapsed = time.perf_counter() - started
        final_hash = _r128_model_hash(model)
        checkpoint_path = output_dir / "checkpoint" / "r128_step_004164.pt"
        torch.save(
            {
                "schema": R128_RUNTIME_SCHEMA,
                "temporal_view": R128_TEMPORAL_VIEW,
                "resolution": R128_RESOLUTION,
                "seed": seed,
                "optimizer_steps": completed_steps,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "initial_model_sha256": initial_hash,
                "final_model_sha256": final_hash,
            },
            checkpoint_path,
        )
        checkpoint_sha = _sha256_file(checkpoint_path)
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
        reload_cpu = stage1._build_b1_model(R128_TEMPORAL_VIEW).cpu()
        reload_cpu.load_state_dict(checkpoint_payload["model_state_dict"])
        loaded_hash = _r128_model_hash(reload_cpu)
        checkpoint_loadable = loaded_hash == final_hash
        if not checkpoint_loadable:
            raise R128RuntimeError("checkpoint reload hash differs from final model")
        del reload_cpu, checkpoint_payload, model, optimizer
        torch.cuda.empty_cache()
        loaded_model = stage1._build_b1_model(R128_TEMPORAL_VIEW).to(device)
        loaded_model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)["model_state_dict"]
        )
        predictions_path = output_dir / "predictions" / "validation_predictions.csv"
        predictions = _r128_predict_validation(loaded_model, population, device)
        predictions.to_csv(predictions_path, index=False)
        prediction_sha = _sha256_file(predictions_path)
        metrics_path = output_dir / "metrics" / "per_class_metrics.csv"
        metrics = _r128_per_class_metrics(predictions_path, metrics_path)
        mean_gpu = float(sum(gpu_samples) / len(gpu_samples)) if gpu_samples else None
        peak_vram_mb = round(torch.cuda.max_memory_allocated(0) / 2**20, 3)
        descriptor = {
            "schema": R128_RUNTIME_SCHEMA,
            "trial_id": f"T6_R128_seed{seed}_steps{R128_MAX_STEPS}",
            "temporal_view": R128_TEMPORAL_VIEW,
            "resolution": R128_RESOLUTION,
            "seed": seed,
            "optimizer": "AdamW",
            "learning_rate": 0.003,
            "weight_decay": 0.0,
            "scheduler": "none",
            "batch_size": R128_BATCH_SIZE,
            "precision": "FP32",
            "optimizer_steps_required": R128_MAX_STEPS,
            "optimizer_steps_completed": completed_steps,
            "architecture": "B1_STAGE1_UNCHANGED",
            "loss": "event_weighted_cross_entropy",
            "outer_examples_accessed": False,
            "code_sha": code_sha,
            "runtime_bundle_sha256": runtime_bundle_sha,
            "cache": _r128_cache_descriptor(
                packed_npy, packed_index, manifest, cache_uri
            ),
            "initial_model_sha256": initial_hash,
            "final_model_sha256": final_hash,
            "weights_changed": initial_hash != final_hash,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_loadable": checkpoint_loadable,
            "prediction_path": str(predictions_path),
            "prediction_sha256": prediction_sha,
            "validation_coverage": f"{len(predictions)}/{R128_VALIDATION_ROWS}",
            "synthetic_result_path_used": False,
            "environment": _r128_environment(),
        }
        descriptor_path = output_dir / "runtime" / "descriptor.json"
        _write_json(descriptor_path, descriptor)
        result = {
            **descriptor,
            **metrics,
            "status": "PASS",
            "result_computed_from_persisted_predictions": True,
            "training_log_path": str(log_path),
            "training_wall_seconds": elapsed,
            "mean_gpu_utilization_during_optimization": mean_gpu,
            "peak_gpu_vram_mb": peak_vram_mb,
            "gpu_device": torch.cuda.get_device_name(0),
            "per_class_metrics_path": str(metrics_path),
            "artifact_manifest_path": str(output_dir / "artifact_manifest.json"),
        }
        _write_json(output_dir / "runtime" / "result.json", result)
        _r128_artifact_manifest(output_dir)
        return result
    finally:
        population.dataset.close()


def _r128_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fail-closed R128 route.")
    parser.add_argument("--mode", choices=("cpu_preflight", "trial"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--packed-npy", type=Path, required=True)
    parser.add_argument("--packed-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-uri", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--code-sha", default="UNSPECIFIED")
    parser.add_argument("--runtime-bundle-sha", default="UNSPECIFIED")
    parser.add_argument("--seed", type=int, choices=sorted(R128_MATCHED_SEEDS))
    return parser.parse_args()


def main() -> None:
    args = _r128_cli()
    if args.mode == "cpu_preflight":
        if args.device != "cpu":
            raise R128RuntimeError("CPU preflight must use --device cpu")
        result = run_r128_cpu_preflight(
            manifest=args.manifest,
            packed_npy=args.packed_npy,
            packed_index=args.packed_index,
            output_dir=args.output_dir,
            cache_uri=args.cache_uri,
        )
    else:
        if args.device != "cuda":
            raise R128RuntimeError("R128 proof trial must use --device cuda")
        if args.seed is None:
            raise R128RuntimeError("R128 proof trial requires --seed")
        result = run_r128_trial(
            manifest=args.manifest,
            packed_npy=args.packed_npy,
            packed_index=args.packed_index,
            output_dir=args.output_dir,
            cache_uri=args.cache_uri,
            code_sha=args.code_sha,
            runtime_bundle_sha=args.runtime_bundle_sha,
            seed=args.seed,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
