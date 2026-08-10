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
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

PERMIT_SCHEMA = "classification_v2.s1_stage1_execution_permit.v1"
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
            "rgb_binding_bundle_sha256": str(
                self.payload["rgb_binding_bundle_sha256"]
            ),
            "status": self.status,
        }


def canonical_trial_id(view: str) -> str:
    """Return the only user-authorized initial-screen trial identity."""

    _validate_view(view)
    return f"s1_stage1_{view.lower()}_seed{SEED}_steps{MAX_STEPS}"


def permit_directory(outputs_root: Path) -> Path:
    """Resolve the external lineage root reserved for Stage-1 permits."""

    return Path(outputs_root).resolve().joinpath(*PERMIT_NAMESPACE)


def create_stage1_execution_permits(
    *,
    repository_root: Path,
    outputs_root: Path,
    authority_path: Path,
    binding_bundle_path: Path,
    ttl_hours: int = DEFAULT_TTL_HOURS,
) -> dict[str, Stage1ExecutionPermit]:
    """Issue four immutable, per-arm permits after strict local verification."""

    if ttl_hours <= 0 or ttl_hours > DEFAULT_TTL_HOURS:
        raise Stage1ExecutionAuthorizationError("permit TTL must be in 1..24 hours")
    root = Path(repository_root).resolve()
    _assert_clean_canonical_repository(root)
    authority_path = Path(authority_path).resolve()
    authority = _read_json(authority_path)
    authority_sha256 = _sha256_file(authority_path)
    _validate_frozen_authority(authority)
    bundle_path = Path(binding_bundle_path).resolve()
    bundle = _read_json(bundle_path)
    _validate_binding_bundle(bundle, authority_sha256)
    code_sha = _git_sha(root)
    created_at = datetime.now(UTC)
    expires_at = created_at + timedelta(hours=ttl_hours)
    directory = permit_directory(outputs_root)
    directory.mkdir(parents=True, exist_ok=True)
    permits: dict[str, Stage1ExecutionPermit] = {}
    for view in VIEWS:
        trial_id = canonical_trial_id(view)
        payload = _permit_payload(
            authority=authority,
            authority_sha256=authority_sha256,
            binding_bundle_sha256=_sha256_file(bundle_path),
            bundle_view=bundle["views"][view],
            code_sha=code_sha,
            view=view,
            trial_id=trial_id,
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
    allow_consumed: bool = False,
) -> Stage1ExecutionPermit:
    """Verify one permit before Stage-1 RGB data or optimizer state is opened."""

    _validate_view(view)
    if _sha256_file(Path(authority_path).resolve()) != authority_sha256:
        raise Stage1ExecutionAuthorizationError("Stage-1 authority hash changed")
    _validate_frozen_authority(authority)
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
    _validate_binding_bundle(bundle, authority_sha256)
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
        "authorization_source": "USER_STAGE1_INITIAL_TEMPORAL_SCREEN_20260810",
        "single_use": True,
        "automatic_downstream_execution": False,
        "automatic_promotion": False,
        "release_authority": False,
        "run_kind": RUN_KIND,
        "authority_sha256": authority_sha256,
        "code_sha": code_sha,
        "view": view,
        "trial_id": trial_id,
        "seed": SEED,
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
        "authorization_source": "USER_STAGE1_INITIAL_TEMPORAL_SCREEN_20260810",
        "single_use": True,
        "automatic_downstream_execution": False,
        "automatic_promotion": False,
        "release_authority": False,
        "run_kind": RUN_KIND,
        "authority_sha256": authority_sha256,
        "code_sha": _git_sha(repository_root),
        "view": view,
        "trial_id": trial_id,
        "seed": stage["initial_seed"],
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


def _validate_binding_bundle(bundle: Mapping[str, Any], authority_sha256: str) -> None:
    if bundle.get("schema_version") != BINDING_BUNDLE_SCHEMA:
        raise Stage1ExecutionAuthorizationError("unsupported Stage-1 binding bundle")
    authority = bundle.get("authority")
    if not isinstance(authority, Mapping) or authority.get("sha256") != authority_sha256:
        raise Stage1ExecutionAuthorizationError("binding bundle authority mismatch")
    views = bundle.get("views")
    if not isinstance(views, Mapping) or set(views) != set(VIEWS):
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
    "DEFAULT_TTL_HOURS",
    "PERMIT_SCHEMA",
    "Stage1ExecutionAuthorizationError",
    "Stage1ExecutionPermit",
    "canonical_trial_id",
    "consume_stage1_execution_permit",
    "create_stage1_execution_permits",
    "permit_directory",
    "validate_stage1_execution_permit",
]
