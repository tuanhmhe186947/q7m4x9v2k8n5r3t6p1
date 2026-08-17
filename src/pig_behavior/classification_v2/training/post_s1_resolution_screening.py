"""Fail-closed T6-only post-S1 pure-resolution executor.

This module deliberately does not alter the frozen 64x64 Stage-1 executor.
It reuses its hash-bound inner T6 rows and native evaluator while applying the
already-audited source crop, aspect-preserving letterbox, and normalization at
one explicitly declared runtime spatial resolution.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

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
R128_MATCHED_SEEDS = frozenset({20260804, 20260805, 20260806})


class _R128PackedArrayReader:
    """Read exact packed rows without mmap random reads on Teamspace Drive."""

    def __init__(self, packed_npy: Path) -> None:
        self.path = Path(packed_npy)
        self._handle = self.path.open("rb", buffering=0)
        try:
            version = np.lib.format.read_magic(self._handle)
            header_reader = {
                (1, 0): np.lib.format.read_array_header_1_0,
                (2, 0): np.lib.format.read_array_header_2_0,
            }.get(version)
            if version == (3, 0):
                header_reader = getattr(
                    np.lib.format,
                    "read_array_header_3_0",
                    None,
                )
            if header_reader is None:
                raise PostS1ResolutionError(f"unsupported packed NPY version: {version}")
            shape, fortran_order, dtype = header_reader(self._handle)
            self.shape = tuple(int(value) for value in shape)
            self.fortran_order = bool(fortran_order)
            self.dtype = np.dtype(dtype)
            self.data_offset = int(self._handle.tell())
            if self.fortran_order:
                raise PostS1ResolutionError("packed R128 tensor must be C-contiguous")
            if self.dtype != np.dtype(np.uint8):
                raise PostS1ResolutionError(
                    f"packed R128 tensor dtype drifted: {self.dtype}"
                )
            if len(self.shape) != 4:
                raise PostS1ResolutionError("packed R128 tensor rank drifted")
            self.row_bytes = int(np.prod(self.shape[1:], dtype=np.int64))
            self._fd = self._handle.fileno()
            self._seek_lock = threading.Lock()
        except Exception:
            self._handle.close()
            raise

    def _read_bytes(self, offset: int, count: int) -> bytes:
        if hasattr(os, "pread"):
            payload = os.pread(self._fd, count, offset)
        else:
            with self._seek_lock:
                self._handle.seek(offset)
                payload = self._handle.read(count)
        if len(payload) != count:
            raise PostS1ResolutionError(
                f"short packed R128 read: expected {count}, got {len(payload)}"
            )
        return payload

    def read_rows(self, packed_rows: list[int]) -> np.ndarray:
        """Return rows in requested order while coalescing contiguous reads."""

        if not packed_rows:
            return np.empty((0, *self.shape[1:]), dtype=self.dtype)
        row_ids = [int(row) for row in packed_rows]
        if any(row < 0 or row >= self.shape[0] for row in row_ids):
            raise PostS1ResolutionError("packed R128 row is outside the tensor")
        output = np.empty((len(row_ids), *self.shape[1:]), dtype=self.dtype)
        ordered = sorted(enumerate(row_ids), key=lambda item: item[1])
        group_start = 0
        while group_start < len(ordered):
            group_end = group_start + 1
            while (
                group_end < len(ordered)
                and ordered[group_end][1] == ordered[group_end - 1][1] + 1
            ):
                group_end += 1
            first_row = ordered[group_start][1]
            group_count = group_end - group_start
            payload = self._read_bytes(
                self.data_offset + first_row * self.row_bytes,
                group_count * self.row_bytes,
            )
            block = np.frombuffer(payload, dtype=self.dtype).reshape(
                group_count, *self.shape[1:]
            )
            for output_index, row_id in ordered[group_start:group_end]:
                output[output_index] = block[row_id - first_row]
            group_start = group_end
        return output

    def close(self) -> None:
        self._handle.close()



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
    seed: int = SEED
    confirmation_authority_path: Path | None = None


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
    seed: int = SEED,
    confirmation_authority_path: Path | None = None,
) -> ResolutionPlan:
    """Resolve one registered R64/R128/R160 arm before opening RGB media."""

    if device_name not in {"cpu", "cuda"}:
        raise PostS1ResolutionError(f"unsupported device={device_name}")
    if input_resolution not in RESOLUTION_ARMS:
        raise PostS1ResolutionError(f"unregistered resolution={input_resolution}")
    if seed not in R128_MATCHED_SEEDS:
        raise PostS1ResolutionError(f"unregistered seed={seed}")
    validate_runtime_resolution(input_resolution)
    authority_path = Path(authority_path).resolve()
    authority = _read_json(authority_path)
    _validate_authority(authority)
    expected_trial = f"post_s1_t6_r{input_resolution}_seed{seed}_steps{MAX_STEPS}"
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
        seed=seed,
        confirmation_authority_path=(
            Path(confirmation_authority_path).resolve()
            if confirmation_authority_path is not None
            else None
        ),
    )


def load_resolution_population(plan: ResolutionPlan) -> ResolutionPopulation:
    """Bind the frozen T6 inner rows to the high-fidelity source realization."""

    base = stage1.create_stage1_plan(
        plan.base_stage1_authority_path,
        view=TEMPORAL_VIEW,
        seed=plan.seed,
        repository_root=plan.repository_root,
        outputs_root=plan.outputs_root,
        output_dir=plan.output_dir,
        data_bindings_path=plan.stage1_data_bindings_path,
        execution_permit_path=plan.execution_permit_path,
        binding_bundle_path=plan.stage1_binding_bundle_path,
        confirmation_authority_path=plan.confirmation_authority_path,
        trial_id=f"s1_stage1_t6_seed{plan.seed}_steps{MAX_STEPS}",
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
    stage1._set_seed(plan.seed)
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
                seed=plan.seed,
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
        "seed": plan.seed,
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
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = [
    "AUTHORITY_SCHEMA",
    "BATCH_SIZE",
    "MAX_STEPS",
    "PostS1ResolutionError",
    "R128_MATCHED_SEEDS",
    "RESOLUTION_ARMS",
    "RUN_KIND",
    "ResolutionPlan",
    "ResolutionPopulation",
    "SEED",
    "TEMPORAL_VIEW",
    "_R128PackedArrayReader",
    "_validate_runtime_input_binding",
    "create_resolution_plan",
    "load_canonical_resolution_temporal_target",
    "load_resolution_population",
    "run_resolution_arm",
]
