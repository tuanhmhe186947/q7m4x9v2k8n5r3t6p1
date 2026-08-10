"""Single-use execution permits for frozen Stage-1 temporal arms.

The canonical Stage-1 scientific authority deliberately keeps its GPU flag
false.  This module therefore records user-approved paid execution as four
external, per-arm transactions without changing the authority bytes or RGB
binding identities.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

PERMIT_SCHEMA = "classification_v2.s1_stage1_execution_permit.v1"
RETENTION_AUTHORITY_SCHEMA = "classification_v2.s1_stage1_temporal_retention_authority.v1"
SUPERSESSION_SCHEMA = "classification_v2.s1_stage1_execution_permit_supersession.v1"
BINDING_BUNDLE_SCHEMA = "classification_v2.s1_stage1_rgb_binding_bundle.v1"
DATA_BINDINGS_SCHEMA = "classification_v2.s1_stage1_temporal_data_bindings.v1"
RUN_KIND = "S1_STAGE1_TEMPORAL_SCREENING"
PERMIT_NAMESPACE = (
    "classification_v2",
    "s1_post_temporal_closure_20260809",
    "s1_stage1_execution_authorizations",
)
TRIAL_NAMESPACE = (
    "classification_v2",
    "s1_post_temporal_closure_20260809",
    "s1_trials",
)
VIEWS = ("T6", "T8", "T12", "T16")
SEED = 20260804
MAX_STEPS = 4164
DEFAULT_TTL_HOURS = 24
INITIAL_AUTHORIZATION_SOURCE = "USER_STAGE1_INITIAL_TEMPORAL_SCREEN_20260810"
CONFIRMATION_AUTHORIZATION_SOURCE = (
    "USER_APPROVED_STAGE1_TEMPORAL_RETENTION_CONFIRMATION_20260810"
)


class Stage1ExecutionAuthorizationError(ValueError):
    """Raised when a Stage-1 GPU permit is absent, stale, or mismatched."""


@dataclass(frozen=True, slots=True)
class Stage1ExecutionPermit:
    """One verified permit bound to a single Stage-1 arm."""

    path: Path
    payload: Mapping[str, Any]
    sha256: str

    @property
    def permit_id(self) -> str:
        return str(self.payload["permit_id"])

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    def manifest_record(self) -> dict[str, object]:
        """Return stable permit lineage fields without host-specific paths."""

        return {
            "permit_id": self.permit_id,
            "permit_sha256": self.sha256,
            "authority_sha256": str(self.payload["authority_sha256"]),
            "code_sha": str(self.payload["code_sha"]),
            "view": str(self.payload["view"]),
            "trial_id": str(self.payload["trial_id"]),
            "seed": int(self.payload["seed"]),
            "authorization_source": str(self.payload["authorization_source"]),
            "confirmation_authority_sha256": self.payload.get(
                "confirmation_authority_sha256"
            ),
            "rgb_binding_bundle_sha256": str(
                self.payload["rgb_binding_bundle_sha256"]
            ),
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class Stage1ExecutionPermitRotation:
    """Immutable lineage for one code-stale permit replacement."""

    previous: Stage1ExecutionPermit
    replacement: Stage1ExecutionPermit
    superseded_path: Path
    record_path: Path


@dataclass(frozen=True, slots=True)
class Stage1SeedAuthorization:
    """One initial or retention-approved Stage-1 seed authorization."""

    seed: int
    authorization_source: str
    confirmation_authority_path: Path | None
    confirmation_authority_sha256: str | None
    confirmation_candidates: tuple[str, ...]

    def manifest_record(self) -> dict[str, object]:
        """Return path-independent seed-authorization lineage."""

        return {
            "seed": self.seed,
            "authorization_source": self.authorization_source,
            "confirmation_authority_sha256": self.confirmation_authority_sha256,
            "confirmation_candidates": list(self.confirmation_candidates),
        }


def validate_stage1_seed_authorization(
    *,
    authority_path: Path,
    authority: Mapping[str, Any],
    authority_sha256: str,
    seed: int,
    view: str | None,
    confirmation_authority_path: Path | None,
) -> Stage1SeedAuthorization:
    """Resolve an initial or user-retained confirmation seed before data access."""

    _validate_seed(seed)
    _validate_frozen_authority(authority)
    if view is not None:
        _validate_view(view)
    stage = authority["stage_1_temporal_screening"]
    initial_seed = int(stage["initial_seed"])
    if seed == initial_seed:
        if confirmation_authority_path is not None:
            raise Stage1ExecutionAuthorizationError(
                "initial Stage-1 seed must not use a confirmation authority"
            )
        return Stage1SeedAuthorization(
            seed=seed,
            authorization_source=INITIAL_AUTHORIZATION_SOURCE,
            confirmation_authority_path=None,
            confirmation_authority_sha256=None,
            confirmation_candidates=(),
        )
    confirmation_seeds = tuple(stage.get("confirmation_seeds", ()))
    if seed not in confirmation_seeds:
        raise Stage1ExecutionAuthorizationError(f"unregistered Stage-1 seed={seed}")
    if confirmation_authority_path is None:
        raise Stage1ExecutionAuthorizationError(
            "confirmation seed requires a retention authority"
        )
    retention_path = Path(confirmation_authority_path).resolve()
    retention = _read_json(retention_path)
    candidates = _validate_confirmation_authority(
        retention_path=retention_path,
        retention=retention,
        authority_path=Path(authority_path).resolve(),
        authority_sha256=authority_sha256,
        confirmation_seeds=confirmation_seeds,
    )
    if view is not None and view not in candidates:
        raise Stage1ExecutionAuthorizationError(
            f"confirmation view is not retained={view}"
        )
    return Stage1SeedAuthorization(
        seed=seed,
        authorization_source=CONFIRMATION_AUTHORIZATION_SOURCE,
        confirmation_authority_path=retention_path,
        confirmation_authority_sha256=_sha256_file(retention_path),
        confirmation_candidates=candidates,
    )


def canonical_trial_id(view: str, *, seed: int = SEED) -> str:
    """Return the immutable identity for one registered Stage-1 seed and view."""

    _validate_view(view)
    _validate_seed(seed)
    return f"s1_stage1_{view.lower()}_seed{seed}_steps{MAX_STEPS}"


def permit_directory(outputs_root: Path) -> Path:
    """Resolve the external lineage root reserved for Stage-1 permits."""

    return Path(outputs_root).resolve().joinpath(*PERMIT_NAMESPACE)


def create_stage1_execution_permits(
    *,
    repository_root: Path,
    outputs_root: Path,
    authority_path: Path,
    binding_bundle_path: Path,
    seed: int = SEED,
    confirmation_authority_path: Path | None = None,
    views: Sequence[str] | None = None,
    ttl_hours: int = DEFAULT_TTL_HOURS,
) -> dict[str, Stage1ExecutionPermit]:
    """Issue only the registered, immutable permits for one Stage-1 seed."""

    if ttl_hours <= 0 or ttl_hours > DEFAULT_TTL_HOURS:
        raise Stage1ExecutionAuthorizationError("permit TTL must be in 1..24 hours")
    root = Path(repository_root).resolve()
    _assert_clean_canonical_repository(root)
    authority_path = Path(authority_path).resolve()
    authority = _read_json(authority_path)
    authority_sha256 = _sha256_file(authority_path)
    _validate_frozen_authority(authority)
    seed_authorization = validate_stage1_seed_authorization(
        authority_path=authority_path,
        authority=authority,
        authority_sha256=authority_sha256,
        seed=seed,
        view=None,
        confirmation_authority_path=confirmation_authority_path,
    )
    selected_views = _issuance_views(views, seed_authorization)
    bundle_path = Path(binding_bundle_path).resolve()
    bundle = _read_json(bundle_path)
    _validate_binding_bundle(
        bundle,
        authority_sha256,
        expected_views=seed_authorization.confirmation_candidates or VIEWS,
    )
    code_sha = _git_sha(root)
    created_at = datetime.now(UTC)
    expires_at = created_at + timedelta(hours=ttl_hours)
    directory = permit_directory(outputs_root)
    directory.mkdir(parents=True, exist_ok=True)
    permits: dict[str, Stage1ExecutionPermit] = {}
    for view in selected_views:
        trial_id = canonical_trial_id(view, seed=seed_authorization.seed)
        payload = _permit_payload(
            authority=authority,
            authority_sha256=authority_sha256,
            binding_bundle_sha256=_sha256_file(bundle_path),
            bundle_view=bundle["views"][view],
            code_sha=code_sha,
            view=view,
            trial_id=trial_id,
            seed_authorization=seed_authorization,
            created_at=created_at,
            expires_at=expires_at,
        )
        path = directory / f"{trial_id}.authorization.json"
        _write_new_json(path, payload)
        permits[view] = Stage1ExecutionPermit(
            path=path,
            payload=payload,
            sha256=_sha256_file(path),
        )
    return permits


def rotate_stage1_execution_permits(
    *,
    repository_root: Path,
    outputs_root: Path,
    authority_path: Path,
    binding_bundle_path: Path,
    views: Sequence[str],
    reason: str,
    seed: int = SEED,
    confirmation_authority_path: Path | None = None,
    ttl_hours: int = DEFAULT_TTL_HOURS,
) -> dict[str, Stage1ExecutionPermitRotation]:
    """Replace selected unconsumed permits after a code-only repair.

    Normal issuance intentionally creates all four permits together.  Rotation
    is the only path that can issue a selected subset, and it preserves each
    stale predecessor byte-for-byte under a superseded lineage filename.
    """

    if ttl_hours <= 0 or ttl_hours > DEFAULT_TTL_HOURS:
        raise Stage1ExecutionAuthorizationError("permit TTL must be in 1..24 hours")
    selected_views = _canonical_rotation_views(views)
    supersession_reason = _validate_supersession_reason(reason)
    root = Path(repository_root).resolve()
    _assert_clean_canonical_repository(root)
    authority_path = Path(authority_path).resolve()
    authority = _read_json(authority_path)
    authority_sha256 = _sha256_file(authority_path)
    _validate_frozen_authority(authority)
    seed_authorization = validate_stage1_seed_authorization(
        authority_path=authority_path,
        authority=authority,
        authority_sha256=authority_sha256,
        seed=seed,
        view=None,
        confirmation_authority_path=confirmation_authority_path,
    )
    bundle_path = Path(binding_bundle_path).resolve()
    bundle = _read_json(bundle_path)
    _validate_binding_bundle(
        bundle,
        authority_sha256,
        expected_views=seed_authorization.confirmation_candidates or VIEWS,
    )
    binding_bundle_sha256 = _sha256_file(bundle_path)
    code_sha = _git_sha(root)
    created_at = datetime.now(UTC)
    expires_at = created_at + timedelta(hours=ttl_hours)
    directory = permit_directory(outputs_root)
    if not directory.is_dir():
        raise Stage1ExecutionAuthorizationError("Stage-1 permit lineage root is missing")

    pending: list[
        tuple[
            str,
            Stage1ExecutionPermit,
            dict[str, object],
            Path,
            Path,
            dict[str, object],
        ]
    ] = []
    for view in selected_views:
        _validate_seed_view(view, seed_authorization)
        trial_id = canonical_trial_id(view, seed=seed_authorization.seed)
        path = directory / f"{trial_id}.authorization.json"
        if not path.is_file():
            raise Stage1ExecutionAuthorizationError(
                f"active Stage-1 permit is missing={path}"
            )
        previous_payload = _read_json(path)
        previous = Stage1ExecutionPermit(
            path=path,
            payload=previous_payload,
            sha256=_sha256_file(path),
        )
        bundle_view = bundle["views"].get(view)
        if not isinstance(bundle_view, Mapping):
            raise Stage1ExecutionAuthorizationError(
                "Stage-1 binding bundle view is invalid"
            )
        replacement_payload = _permit_payload(
            authority=authority,
            authority_sha256=authority_sha256,
            binding_bundle_sha256=binding_bundle_sha256,
            bundle_view=bundle_view,
            code_sha=code_sha,
            view=view,
            trial_id=trial_id,
            seed_authorization=seed_authorization,
            created_at=created_at,
            expires_at=expires_at,
        )
        expected = _expected_permit_fields(
            repository_root=root,
            outputs_root=Path(outputs_root).resolve(),
            authority=authority,
            authority_sha256=authority_sha256,
            view=view,
            trial_id=trial_id,
            seed_authorization=seed_authorization,
            scientific_rgb_binding_sha256=str(
                bundle_view["scientific_binding_sha256"]
            ),
        )
        _validate_rotation_predecessor(
            previous_payload,
            expected=expected,
            binding_bundle_sha256=binding_bundle_sha256,
            current_code_sha=code_sha,
            expected_field_names=frozenset(replacement_payload),
        )
        superseded_path = directory / f"superseded-{previous.permit_id}.json"
        record_path = directory / f"supersession-{previous.permit_id}.json"
        if superseded_path.exists() or record_path.exists():
            raise Stage1ExecutionAuthorizationError(
                f"Stage-1 permit supersession lineage already exists={previous.permit_id}"
            )
        pending.append(
            (
                view,
                previous,
                replacement_payload,
                superseded_path,
                record_path,
                expected,
            )
        )

    rotations: dict[str, Stage1ExecutionPermitRotation] = {}
    for view, previous, replacement_payload, superseded_path, record_path, expected in pending:
        if _sha256_file(previous.path) != previous.sha256:
            raise Stage1ExecutionAuthorizationError(
                "Stage-1 permit changed before supersession"
            )
        try:
            previous.path.replace(superseded_path)
        except FileNotFoundError as error:
            raise Stage1ExecutionAuthorizationError(
                "Stage-1 permit disappeared before supersession"
            ) from error
        superseded = Stage1ExecutionPermit(
            path=superseded_path,
            payload=previous.payload,
            sha256=_sha256_file(superseded_path),
        )
        _write_new_json(previous.path, replacement_payload)
        replacement = Stage1ExecutionPermit(
            path=previous.path,
            payload=replacement_payload,
            sha256=_sha256_file(previous.path),
        )
        record = _supersession_record(
            previous=superseded,
            replacement=replacement,
            reason=supersession_reason,
            expected=expected,
        )
        _write_new_json(record_path, record)
        rotations[view] = Stage1ExecutionPermitRotation(
            previous=superseded,
            replacement=replacement,
            superseded_path=superseded_path,
            record_path=record_path,
        )
    return rotations


def validate_stage1_execution_permit(
    *,
    permit_path: Path,
    repository_root: Path,
    outputs_root: Path,
    authority_path: Path,
    authority: Mapping[str, Any],
    authority_sha256: str,
    view: str,
    trial_id: str,
    data_bindings_path: Path,
    binding_bundle_path: Path,
    seed: int = SEED,
    confirmation_authority_path: Path | None = None,
    allow_consumed: bool = False,
) -> Stage1ExecutionPermit:
    """Verify one permit before Stage-1 RGB data or optimizer state is opened."""

    _validate_view(view)
    if _sha256_file(Path(authority_path).resolve()) != authority_sha256:
        raise Stage1ExecutionAuthorizationError("Stage-1 authority hash changed")
    _validate_frozen_authority(authority)
    seed_authorization = validate_stage1_seed_authorization(
        authority_path=Path(authority_path).resolve(),
        authority=authority,
        authority_sha256=authority_sha256,
        seed=seed,
        view=view,
        confirmation_authority_path=confirmation_authority_path,
    )
    if trial_id != canonical_trial_id(view, seed=seed_authorization.seed):
        raise Stage1ExecutionAuthorizationError(
            "Stage-1 permit trial identity does not match seed and view"
        )
    path = Path(permit_path).resolve()
    directory = permit_directory(outputs_root)
    if path.parent != directory:
        raise Stage1ExecutionAuthorizationError("Stage-1 permit escapes lineage root")
    payload = _read_json(path)
    status = str(payload.get("status", ""))
    if status == "AUTHORIZED":
        expected_path = directory / f"{trial_id}.authorization.json"
        if path != expected_path:
            raise Stage1ExecutionAuthorizationError("Stage-1 permit path mismatches trial")
    elif status == "CONSUMED" and allow_consumed:
        if not path.name.startswith("used-"):
            raise Stage1ExecutionAuthorizationError("consumed Stage-1 permit mismatches trial")
        if not payload.get("consumed_at_utc"):
            raise Stage1ExecutionAuthorizationError("consumed Stage-1 permit lacks timestamp")
    else:
        raise Stage1ExecutionAuthorizationError("Stage-1 permit is not available for execution")
    _assert_not_expired(payload)
    bundle_path = Path(binding_bundle_path).resolve()
    if _sha256_file(bundle_path) != str(payload.get("rgb_binding_bundle_sha256", "")):
        raise Stage1ExecutionAuthorizationError("Stage-1 RGB binding bundle hash changed")
    bundle = _read_json(bundle_path)
    _validate_binding_bundle(
        bundle,
        authority_sha256,
        expected_views=seed_authorization.confirmation_candidates or VIEWS,
    )
    bundle_view = bundle["views"].get(view)
    if not isinstance(bundle_view, Mapping):
        raise Stage1ExecutionAuthorizationError("Stage-1 binding bundle view is invalid")
    expected = _expected_permit_fields(
        repository_root=Path(repository_root).resolve(),
        outputs_root=Path(outputs_root).resolve(),
        authority=authority,
        authority_sha256=authority_sha256,
        view=view,
        trial_id=trial_id,
        seed_authorization=seed_authorization,
        scientific_rgb_binding_sha256=_scientific_binding_sha256(data_bindings_path),
    )
    for key, value in expected.items():
        if payload.get(key) != value:
            raise Stage1ExecutionAuthorizationError(
                f"Stage-1 permit mismatch={key}"
            )
    if (
        bundle_view.get("scientific_binding_sha256")
        != expected["scientific_rgb_binding_sha256"]
        or bundle_view.get("provenance_hashes", {}).get("event_weight")
        != expected["event_weight_sha256"]
    ):
        raise Stage1ExecutionAuthorizationError("Stage-1 binding bundle view drifted")
    return Stage1ExecutionPermit(path=path, payload=payload, sha256=_sha256_file(path))


def consume_stage1_execution_permit(
    permit: Stage1ExecutionPermit,
    *,
    allow_consumed_resume: bool,
) -> Stage1ExecutionPermit:
    """Atomically consume a fresh permit, or admit it only for exact resume."""

    if permit.status == "CONSUMED":
        if allow_consumed_resume:
            return permit
        raise Stage1ExecutionAuthorizationError("consumed Stage-1 permit cannot start new work")
    if permit.status != "AUTHORIZED":
        raise Stage1ExecutionAuthorizationError("Stage-1 permit status is invalid")
    current = _read_json(permit.path)
    if current != dict(permit.payload):
        raise Stage1ExecutionAuthorizationError("Stage-1 permit changed before consumption")
    consumed = permit.path.with_name(f"used-{permit.permit_id[:12]}.json")
    try:
        permit.path.replace(consumed)
    except FileNotFoundError as error:
        raise Stage1ExecutionAuthorizationError(
            "Stage-1 permit was consumed by another process"
        ) from error
    current["status"] = "CONSUMED"
    current["consumed_at_utc"] = datetime.now(UTC).isoformat()
    _write_json_atomic(consumed, current)
    return Stage1ExecutionPermit(
        path=consumed,
        payload=current,
        sha256=_sha256_file(consumed),
    )


def _permit_payload(
    *,
    authority: Mapping[str, Any],
    authority_sha256: str,
    binding_bundle_sha256: str,
    bundle_view: Mapping[str, Any],
    code_sha: str,
    view: str,
    trial_id: str,
    seed_authorization: Stage1SeedAuthorization,
    created_at: datetime,
    expires_at: datetime,
) -> dict[str, object]:
    event_weight = authority["derived_population"]["per_view"][view][
        "event_weight_artifact"
    ]["sha256"]
    if bundle_view.get("provenance_hashes", {}).get("event_weight") != event_weight:
        raise Stage1ExecutionAuthorizationError("binding bundle event-weight mismatch")
    scientific_hash = str(bundle_view.get("scientific_binding_sha256", ""))
    if len(scientific_hash) != 64:
        raise Stage1ExecutionAuthorizationError("binding bundle lacks scientific RGB hash")
    return {
        "schema_version": PERMIT_SCHEMA,
        "permit_id": uuid4().hex,
        "status": "AUTHORIZED",
        "authorization_source": seed_authorization.authorization_source,
        "confirmation_authority_sha256": (
            seed_authorization.confirmation_authority_sha256
        ),
        "single_use": True,
        "automatic_downstream_execution": False,
        "automatic_promotion": False,
        "release_authority": False,
        "run_kind": RUN_KIND,
        "authority_sha256": authority_sha256,
        "code_sha": code_sha,
        "view": view,
        "trial_id": trial_id,
        "seed": seed_authorization.seed,
        "max_steps": MAX_STEPS,
        "evaluation_steps": [MAX_STEPS],
        "training_budget_unit": "OPTIMIZER_STEPS",
        "optimizer": "AdamW",
        "learning_rate": 0.003,
        "weight_decay": 0,
        "batch_size": 16,
        "precision": "FP32",
        "scheduler": "none",
        "early_stopping": "DISABLED",
        "scientific_ranking_checkpoint": "FIXED_STEP_ENDPOINT",
        "event_weight_sha256": event_weight,
        "scientific_rgb_binding_sha256": scientific_hash,
        "rgb_binding_bundle_sha256": binding_bundle_sha256,
        "hardware": {"gpu_count": 1, "gpu_name": "NVIDIA L4"},
        "outer_access_allowed": False,
        "output_relative_path": "/".join((*TRIAL_NAMESPACE, trial_id)),
        "created_at_utc": created_at.isoformat(),
        "expires_at_utc": expires_at.isoformat(),
    }


def _expected_permit_fields(
    *,
    repository_root: Path,
    outputs_root: Path,
    authority: Mapping[str, Any],
    authority_sha256: str,
    view: str,
    trial_id: str,
    seed_authorization: Stage1SeedAuthorization,
    scientific_rgb_binding_sha256: str,
) -> dict[str, object]:
    stage = authority["stage_1_temporal_screening"]
    controls = authority["fixed_stage_1_to_4_controls"]
    event_weight = authority["derived_population"]["per_view"][view][
        "event_weight_artifact"
    ]["sha256"]
    expected_output = outputs_root.joinpath(*TRIAL_NAMESPACE, trial_id).resolve()
    return {
        "schema_version": PERMIT_SCHEMA,
        "authorization_source": seed_authorization.authorization_source,
        "confirmation_authority_sha256": (
            seed_authorization.confirmation_authority_sha256
        ),
        "single_use": True,
        "automatic_downstream_execution": False,
        "automatic_promotion": False,
        "release_authority": False,
        "run_kind": RUN_KIND,
        "authority_sha256": authority_sha256,
        "code_sha": _git_sha(repository_root),
        "view": view,
        "trial_id": trial_id,
        "seed": seed_authorization.seed,
        "max_steps": stage["max_steps"],
        "evaluation_steps": stage["evaluation_steps"],
        "training_budget_unit": stage["training_budget_unit"],
        "optimizer": controls["optimizer"],
        "learning_rate": controls["learning_rate"],
        "weight_decay": controls["weight_decay"],
        "batch_size": controls["batch_size"],
        "precision": controls["precision"],
        "scheduler": controls["scheduler"],
        "early_stopping": stage["early_stopping"],
        "scientific_ranking_checkpoint": stage["scientific_ranking_checkpoint"],
        "event_weight_sha256": event_weight,
        "scientific_rgb_binding_sha256": scientific_rgb_binding_sha256,
        "hardware": {"gpu_count": 1, "gpu_name": "NVIDIA L4"},
        "outer_access_allowed": False,
        "output_relative_path": str(expected_output.relative_to(outputs_root)).replace(
            "\\", "/"
        ),
    }


def _validate_frozen_authority(authority: Mapping[str, Any]) -> None:
    stage = authority.get("stage_1_temporal_screening")
    if not isinstance(stage, Mapping):
        raise Stage1ExecutionAuthorizationError("Stage-1 authority is malformed")
    if stage.get("gpu_execution_authorized") is not False:
        raise Stage1ExecutionAuthorizationError(
            "canonical Stage-1 GPU flag must remain false"
        )
    if (
        stage.get("run_kind") != RUN_KIND
        or tuple(stage.get("temporal_views", ())) != VIEWS
        or stage.get("initial_seed") != SEED
        or tuple(stage.get("confirmation_seeds", ())) != (20260805, 20260806)
        or stage.get("max_steps") != MAX_STEPS
        or stage.get("evaluation_steps") != [MAX_STEPS]
        or stage.get("training_budget_unit") != "OPTIMIZER_STEPS"
        or stage.get("early_stopping") != "DISABLED"
        or stage.get("scientific_ranking_checkpoint") != "FIXED_STEP_ENDPOINT"
        or stage.get("outer_access_allowed") is not False
    ):
        raise Stage1ExecutionAuthorizationError("frozen Stage-1 authority drifted")
    controls = authority.get("fixed_stage_1_to_4_controls")
    expected_controls = {
        "optimizer": "AdamW",
        "learning_rate": 0.003,
        "weight_decay": 0,
        "batch_size": 16,
        "precision": "FP32",
        "scheduler": "none",
    }
    if not isinstance(controls, Mapping) or any(
        controls.get(key) != value for key, value in expected_controls.items()
    ):
        raise Stage1ExecutionAuthorizationError("fixed Stage-1 controls drifted")


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0:
        raise Stage1ExecutionAuthorizationError("Stage-1 seed must be a positive integer")


def _issuance_views(
    requested: Sequence[str] | None,
    seed_authorization: Stage1SeedAuthorization,
) -> tuple[str, ...]:
    if requested is None:
        selected = seed_authorization.confirmation_candidates or VIEWS
    else:
        selected = _canonical_rotation_views(requested)
    if seed_authorization.confirmation_authority_sha256 is None:
        if selected != VIEWS:
            raise Stage1ExecutionAuthorizationError(
                "initial Stage-1 permit issuance requires all registered views"
            )
        return selected
    if selected != seed_authorization.confirmation_candidates:
        raise Stage1ExecutionAuthorizationError(
            "confirmation permit issuance must contain exactly the retained views"
        )
    return selected


def _validate_seed_view(
    view: str,
    seed_authorization: Stage1SeedAuthorization,
) -> None:
    _validate_view(view)
    if (
        seed_authorization.confirmation_authority_sha256 is not None
        and view not in seed_authorization.confirmation_candidates
    ):
        raise Stage1ExecutionAuthorizationError(
            f"confirmation view is not retained={view}"
        )


def _validate_confirmation_authority(
    *,
    retention_path: Path,
    retention: Mapping[str, Any],
    authority_path: Path,
    authority_sha256: str,
    confirmation_seeds: tuple[object, ...],
) -> tuple[str, ...]:
    if retention.get("schema_version") != RETENTION_AUTHORITY_SCHEMA:
        raise Stage1ExecutionAuthorizationError("unsupported retention authority schema")
    if retention.get("status") != "FROZEN_USER_APPROVED_STAGE1_CONFIRMATION_RETENTION":
        raise Stage1ExecutionAuthorizationError("retention authority status is not frozen")
    if (
        retention.get("authority_classification")
        != "USER_APPROVED_STAGE1_CONFIRMATION_RETENTION_DECISION"
        or retention.get("scope")
        != "behavior_only_S1_inner_development_confirmation_only"
    ):
        raise Stage1ExecutionAuthorizationError("retention authority scope drifted")
    base_path, base_sha256 = _retention_artifact(
        retention_path,
        retention,
        "base_s1_authority",
    )
    if base_path != authority_path or base_sha256 != authority_sha256:
        raise Stage1ExecutionAuthorizationError("retention base authority hash mismatch")
    if _sha256_file(base_path) != authority_sha256:
        raise Stage1ExecutionAuthorizationError("retention base authority bytes changed")
    evidence_path, evidence_sha256 = _retention_artifact(
        retention_path,
        retention,
        "initial_screen_evidence",
    )
    if _sha256_file(evidence_path) != evidence_sha256:
        raise Stage1ExecutionAuthorizationError("retention initial-screen evidence hash mismatch")
    _validate_initial_screen_evidence(_read_json(evidence_path))
    binding = retention.get("initial_screen_binding")
    if not isinstance(binding, Mapping) or binding != {
        "run_kind": RUN_KIND,
        "initial_seed": SEED,
        "fixed_optimizer_steps": MAX_STEPS,
        "training_budget_unit": "OPTIMIZER_STEPS",
        "primary_metric": "NATIVE_UNIT_10_CLASS_MACRO_F1",
        "common_cohort_metric": "COMMON_COHORT_NATIVE_MACRO_F1",
        "outer_examples_accessed": False,
    }:
        raise Stage1ExecutionAuthorizationError("retention initial-screen binding drifted")
    rule = retention.get("retention_rule")
    expected_conditions = {
        "t16_valid": True,
        "t16_exceeds_t6_primary_metric": True,
        "t16_exceeds_t6_common_cohort_metric": True,
        "registered_rare_class_disqualification": "NONE_REGISTERED",
    }
    if not isinstance(rule, Mapping) or (
        rule.get("retain_reference_control") != "T6"
        or rule.get("retain_provisional_candidate") != "T16"
        or rule.get("rejected_views") != ["T8", "T12"]
        or rule.get("candidate_not_final") is not True
        or rule.get("conditions") != expected_conditions
    ):
        raise Stage1ExecutionAuthorizationError("retention rule drifted")
    confirmation = retention.get("confirmation")
    candidates = ("T6", "T16")
    if not isinstance(confirmation, Mapping) or (
        tuple(confirmation.get("candidates", ())) != candidates
        or tuple(confirmation.get("seeds", ())) != confirmation_seeds
        or confirmation.get("fixed_optimizer_steps") != MAX_STEPS
        or confirmation.get("training_budget_unit") != "OPTIMIZER_STEPS"
        or confirmation.get("same_fixed_controls_required") is not True
        or confirmation.get("gpu_execution_authorized") is not False
        or confirmation.get("external_single_use_per_arm_permit_required") is not True
        or confirmation.get("outer_access_allowed") is not False
    ):
        raise Stage1ExecutionAuthorizationError("retention confirmation contract drifted")
    return candidates


def _retention_artifact(
    retention_path: Path,
    retention: Mapping[str, Any],
    key: str,
) -> tuple[Path, str]:
    reference = retention.get(key)
    if not isinstance(reference, Mapping):
        raise Stage1ExecutionAuthorizationError(f"retention artifact missing={key}")
    relative = Path(str(reference.get("relative_path", "")))
    expected = str(reference.get("sha256", ""))
    if relative.is_absolute() or ".." in relative.parts or len(expected) != 64:
        raise Stage1ExecutionAuthorizationError(f"retention artifact invalid={key}")
    base = retention_path.parent.resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise Stage1ExecutionAuthorizationError(f"retention artifact escapes root={key}")
    return path, expected


def _validate_initial_screen_evidence(evidence: Mapping[str, Any]) -> None:
    if (
        evidence.get("schema_version")
        != "classification_v2.s1_stage1_initial_temporal_screen_evidence.v1"
        or evidence.get("status") != "ARTIFACT_VERIFIED_INNER_DEVELOPMENT_EVIDENCE"
    ):
        raise Stage1ExecutionAuthorizationError("initial-screen evidence schema drifted")
    scope = evidence.get("scope")
    if not isinstance(scope, Mapping) or (
        scope.get("run_kind") != RUN_KIND
        or scope.get("initial_seed") != SEED
        or scope.get("fixed_optimizer_steps") != MAX_STEPS
        or scope.get("training_budget_unit") != "OPTIMIZER_STEPS"
        or scope.get("outer_examples_accessed") is not False
        or scope.get("claim_grade_result") is not False
        or scope.get("scientific_promotion_allowed") is not False
    ):
        raise Stage1ExecutionAuthorizationError("initial-screen evidence scope drifted")
    views = evidence.get("views")
    if not isinstance(views, Mapping) or set(views) != set(VIEWS):
        raise Stage1ExecutionAuthorizationError("initial-screen evidence view set drifted")
    try:
        t6_primary = float(views["T6"]["primary_native_macro_f1"])
        t6_common = float(views["T6"]["common_cohort_native_macro_f1"])
        t16_primary = float(views["T16"]["primary_native_macro_f1"])
        t16_common = float(views["T16"]["common_cohort_native_macro_f1"])
        t8_primary = float(views["T8"]["primary_native_macro_f1"])
        t8_common = float(views["T8"]["common_cohort_native_macro_f1"])
        t12_primary = float(views["T12"]["primary_native_macro_f1"])
        t12_common = float(views["T12"]["common_cohort_native_macro_f1"])
    except (KeyError, TypeError, ValueError) as error:
        raise Stage1ExecutionAuthorizationError(
            "initial-screen evidence metrics are malformed"
        ) from error
    if not (
        t16_primary > t6_primary
        and t16_common > t6_common
        and t8_primary < t6_primary
        and t8_common < t6_common
        and t12_primary < t6_primary
        and t12_common < t6_common
    ):
        raise Stage1ExecutionAuthorizationError("initial-screen retention evidence drifted")
    guardrails = evidence.get("rare_class_guardrails")
    if not isinstance(guardrails, Mapping) or (
        guardrails.get("numerical_disqualification_threshold_authority") != "MISSING"
        or guardrails.get("t16_explicit_disqualification") is not False
    ):
        raise Stage1ExecutionAuthorizationError("retention rare-class boundary drifted")
    comparison = evidence.get("comparison_notes")
    if not isinstance(comparison, Mapping) or (
        comparison.get("fixed_update_comparison") is not True
        or comparison.get("matched_epoch_comparison") is not False
    ):
        raise Stage1ExecutionAuthorizationError(
            "initial-screen comparison contract drifted"
        )


def _validate_binding_bundle(
    bundle: Mapping[str, Any],
    authority_sha256: str,
    *,
    expected_views: Sequence[str],
) -> None:
    if bundle.get("schema_version") != BINDING_BUNDLE_SCHEMA:
        raise Stage1ExecutionAuthorizationError("unsupported Stage-1 binding bundle")
    authority = bundle.get("authority")
    if not isinstance(authority, Mapping) or authority.get("sha256") != authority_sha256:
        raise Stage1ExecutionAuthorizationError("binding bundle authority mismatch")
    views = bundle.get("views")
    if not isinstance(views, Mapping) or set(views) != set(expected_views):
        raise Stage1ExecutionAuthorizationError("binding bundle view set drifted")


def _scientific_binding_sha256(data_bindings_path: Path) -> str:
    payload = _read_json(Path(data_bindings_path).resolve())
    if payload.get("schema_version") != DATA_BINDINGS_SCHEMA:
        raise Stage1ExecutionAuthorizationError("unsupported Stage-1 data bindings")
    scientific = payload.get("scientific_binding")
    if not isinstance(scientific, Mapping):
        raise Stage1ExecutionAuthorizationError("Stage-1 data bindings lack science ref")
    value = str(scientific.get("sha256", ""))
    if len(value) != 64:
        raise Stage1ExecutionAuthorizationError("Stage-1 scientific RGB hash is invalid")
    return value


def _assert_not_expired(payload: Mapping[str, Any]) -> None:
    try:
        expires = datetime.fromisoformat(str(payload["expires_at_utc"]))
    except (KeyError, TypeError, ValueError) as error:
        raise Stage1ExecutionAuthorizationError("Stage-1 permit expiry is invalid") from error
    if expires.tzinfo is None or expires <= datetime.now(UTC):
        raise Stage1ExecutionAuthorizationError("Stage-1 permit is expired")


def _canonical_rotation_views(views: Sequence[str]) -> tuple[str, ...]:
    """Return a nonempty registered subset in canonical view order."""

    selected = tuple(views)
    if not selected:
        raise Stage1ExecutionAuthorizationError("at least one Stage-1 view is required")
    if any(not isinstance(view, str) for view in selected):
        raise Stage1ExecutionAuthorizationError("Stage-1 rotation views must be strings")
    if len(set(selected)) != len(selected):
        raise Stage1ExecutionAuthorizationError("Stage-1 rotation views must be unique")
    for view in selected:
        _validate_view(view)
    return tuple(view for view in VIEWS if view in selected)


def _validate_supersession_reason(reason: str) -> str:
    """Reject empty or unbounded free-text lineage reasons."""

    if not isinstance(reason, str):
        raise Stage1ExecutionAuthorizationError("Stage-1 supersession reason must be text")
    normalized = reason.strip()
    if not normalized or len(normalized) > 500:
        raise Stage1ExecutionAuthorizationError(
            "Stage-1 supersession reason must contain 1..500 characters"
        )
    return normalized


def _validate_rotation_predecessor(
    payload: Mapping[str, Any],
    *,
    expected: Mapping[str, object],
    binding_bundle_sha256: str,
    current_code_sha: str,
    expected_field_names: frozenset[str],
) -> None:
    """Prove a selected predecessor differs only by its stale code SHA."""

    if set(payload) != expected_field_names:
        raise Stage1ExecutionAuthorizationError("Stage-1 rotation predecessor is malformed")
    if payload.get("status") != "AUTHORIZED":
        raise Stage1ExecutionAuthorizationError(
            "Stage-1 rotation predecessor is not an active permit"
        )
    permit_id = str(payload.get("permit_id", ""))
    if len(permit_id) != 32 or any(char not in "0123456789abcdef" for char in permit_id):
        raise Stage1ExecutionAuthorizationError("Stage-1 rotation permit ID is malformed")
    previous_code_sha = str(payload.get("code_sha", ""))
    if len(previous_code_sha) != 40 or any(
        char not in "0123456789abcdef" for char in previous_code_sha
    ):
        raise Stage1ExecutionAuthorizationError("Stage-1 rotation code SHA is malformed")
    if previous_code_sha == current_code_sha:
        raise Stage1ExecutionAuthorizationError(
            "Stage-1 rotation predecessor already binds current code"
        )
    created_at = _parse_permit_timestamp(payload, "created_at_utc")
    expires_at = _parse_permit_timestamp(payload, "expires_at_utc")
    _assert_not_expired(payload)
    if expires_at <= created_at:
        raise Stage1ExecutionAuthorizationError("Stage-1 rotation permit lifetime is invalid")
    if payload.get("rgb_binding_bundle_sha256") != binding_bundle_sha256:
        raise Stage1ExecutionAuthorizationError(
            "Stage-1 rotation binding bundle hash changed"
        )
    for key, value in expected.items():
        if key != "code_sha" and payload.get(key) != value:
            raise Stage1ExecutionAuthorizationError(
                f"Stage-1 rotation frozen field mismatch={key}"
            )


def _parse_permit_timestamp(payload: Mapping[str, Any], key: str) -> datetime:
    """Read one timezone-aware permit timestamp, or fail closed."""

    try:
        value = datetime.fromisoformat(str(payload[key]))
    except (KeyError, TypeError, ValueError) as error:
        raise Stage1ExecutionAuthorizationError(
            f"Stage-1 rotation timestamp is invalid={key}"
        ) from error
    if value.tzinfo is None:
        raise Stage1ExecutionAuthorizationError(
            f"Stage-1 rotation timestamp lacks timezone={key}"
        )
    return value


def _supersession_record(
    *,
    previous: Stage1ExecutionPermit,
    replacement: Stage1ExecutionPermit,
    reason: str,
    expected: Mapping[str, object],
) -> dict[str, object]:
    """Record immutable predecessor/replacement lineage outside permit bytes."""

    previous_code_sha = str(previous.payload["code_sha"])
    replacement_code_sha = str(replacement.payload["code_sha"])
    if previous_code_sha == replacement_code_sha:
        raise Stage1ExecutionAuthorizationError(
            "Stage-1 supersession requires a changed code SHA"
        )
    return {
        "schema_version": SUPERSESSION_SCHEMA,
        "status": "SUPERSEDED",
        "reason": reason,
        "superseded_at_utc": replacement.payload["created_at_utc"],
        "view": replacement.payload["view"],
        "trial_id": replacement.payload["trial_id"],
        "previous_permit_id": previous.permit_id,
        "previous_permit_sha256": previous.sha256,
        "previous_code_sha": previous_code_sha,
        "superseded_permit_filename": previous.path.name,
        "replacement_permit_id": replacement.permit_id,
        "replacement_permit_sha256": replacement.sha256,
        "replacement_code_sha": replacement_code_sha,
        "replacement_permit_filename": replacement.path.name,
        "code_sha_changed": True,
        "frozen_fields_verified": sorted(
            [
                *[key for key in expected if key != "code_sha"],
                "rgb_binding_bundle_sha256",
            ]
        ),
    }


def _assert_clean_canonical_repository(root: Path) -> None:
    if _git_status(root):
        raise Stage1ExecutionAuthorizationError("permit issuer requires a clean repository")


def _git_sha(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def _git_status(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip()


def _validate_view(view: str) -> None:
    if view not in VIEWS:
        raise Stage1ExecutionAuthorizationError(f"unregistered Stage-1 view={view}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Stage1ExecutionAuthorizationError(f"Stage-1 JSON unreadable={path}") from error
    if not isinstance(value, dict):
        raise Stage1ExecutionAuthorizationError(f"Stage-1 JSON object required={path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise Stage1ExecutionAuthorizationError(
            f"active Stage-1 permit already exists={path}"
        ) from error


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "CONFIRMATION_AUTHORIZATION_SOURCE",
    "DEFAULT_TTL_HOURS",
    "INITIAL_AUTHORIZATION_SOURCE",
    "PERMIT_SCHEMA",
    "RETENTION_AUTHORITY_SCHEMA",
    "SUPERSESSION_SCHEMA",
    "Stage1ExecutionAuthorizationError",
    "Stage1ExecutionPermit",
    "Stage1ExecutionPermitRotation",
    "Stage1SeedAuthorization",
    "canonical_trial_id",
    "consume_stage1_execution_permit",
    "create_stage1_execution_permits",
    "permit_directory",
    "rotate_stage1_execution_permits",
    "validate_stage1_seed_authorization",
    "validate_stage1_execution_permit",
]
