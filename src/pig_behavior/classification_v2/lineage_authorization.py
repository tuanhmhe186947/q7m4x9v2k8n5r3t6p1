"""Single-use, run-local authorization for one lineage stage."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from pig_behavior.classification_v2.lineage_config import (
    current_git_sha,
    resolve_run_root,
)

SCHEMA_VERSION = "classification_v2.stage_authorization.v1"
DEFAULT_TTL_HOURS = 24


def authorization_path(
    root: Path,
    config: dict[str, Any],
    stage_id: str,
) -> Path:
    return (
        resolve_run_root(root, config)
        / "authorizations"
        / f"{stage_id}.authorization.json"
    )


def create_stage_authorization(
    *,
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    stage_id: str,
    ttl_hours: int = DEFAULT_TTL_HOURS,
) -> tuple[Path, dict[str, Any]]:
    if ttl_hours <= 0:
        raise ValueError("AUTHORIZATION_TTL_MUST_BE_POSITIVE")
    if any(value is not False for value in config["authorization"].values()):
        raise ValueError("CANONICAL_AUTHORIZATION_FLAGS_MUST_REMAIN_FALSE")
    path = authorization_path(root, config, stage_id)
    if path.exists():
        raise FileExistsError(f"ACTIVE_AUTHORIZATION_EXISTS:{path}")
    now = datetime.now(UTC)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "authorization_id": uuid4().hex,
        "status": "AUTHORIZED",
        "lineage_id": str(config["lineage_id"]),
        "stage_id": stage_id,
        "authorization_flag": str(
            config["stages"][stage_id]["authorization_flag"]
        ),
        "config_sha256": _sha256(config_path),
        "code_sha": current_git_sha(root),
        "source_bundle_id": str(config["source"]["bundle_id"]),
        "source_bundle_fingerprint": str(
            config["source"]["expected_bundle_fingerprint"]
        ),
        "created_at_utc": now.isoformat(),
        "expires_at_utc": (now + timedelta(hours=ttl_hours)).isoformat(),
        "single_use": True,
        "automatic_downstream_execution": False,
        "automatic_promotion": False,
        "release_authority": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(path, payload)
    return path, payload


def validate_stage_authorization(
    *,
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    stage_id: str,
) -> tuple[bool, str, Path | None]:
    path = authorization_path(root, config, stage_id)
    if not path.is_file():
        return False, "RUN_LOCAL_AUTHORIZATION_MISSING", None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "RUN_LOCAL_AUTHORIZATION_MALFORMED", path
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "AUTHORIZED",
        "lineage_id": str(config["lineage_id"]),
        "stage_id": stage_id,
        "authorization_flag": str(
            config["stages"][stage_id]["authorization_flag"]
        ),
        "config_sha256": _sha256(config_path),
        "code_sha": current_git_sha(root),
        "source_bundle_id": str(config["source"]["bundle_id"]),
        "source_bundle_fingerprint": str(
            config["source"]["expected_bundle_fingerprint"]
        ),
        "single_use": True,
        "automatic_downstream_execution": False,
        "automatic_promotion": False,
        "release_authority": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            return False, f"RUN_LOCAL_AUTHORIZATION_MISMATCH:{key}", path
    try:
        expires = datetime.fromisoformat(str(payload["expires_at_utc"]))
    except (KeyError, ValueError):
        return False, "RUN_LOCAL_AUTHORIZATION_EXPIRY_INVALID", path
    if expires <= datetime.now(UTC):
        return False, "RUN_LOCAL_AUTHORIZATION_EXPIRED", path
    return True, "RUN_LOCAL_AUTHORIZATION_VALID", path


def consume_stage_authorization(path: Path) -> Path:
    payload = json.loads(path.read_text(encoding="utf-8"))
    consumed = path.with_name(
        f"{path.stem}.consumed.{payload['authorization_id']}.json"
    )
    path.replace(consumed)
    payload["status"] = "CONSUMED"
    payload["consumed_at_utc"] = datetime.now(UTC).isoformat()
    _write_json_atomic(consumed, payload)
    return consumed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
