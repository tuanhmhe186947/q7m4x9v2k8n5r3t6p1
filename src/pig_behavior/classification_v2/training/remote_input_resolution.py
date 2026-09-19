"""Resolve registered remote input locators without changing input identity."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AUTHORITY_SCHEMA = "classification_v2.remote_input_root.v1"


class RemoteInputResolutionError(ValueError):
    """Raised when no registered runtime locator proves the input authority."""


@dataclass(frozen=True, slots=True)
class RemoteInputAuthority:
    """Immutable scientific identity plus ordered runtime realizations."""

    authority_id: str
    expected_file_count: int
    expected_total_bytes: int
    preferred_runtime_locator: Path
    registered_runtime_locators: tuple[Path, ...]
    sentinel_sha256: Mapping[str, str]
    historical_parity_evidence: Mapping[str, str]
    parity_report_locator: Path


@dataclass(frozen=True, slots=True)
class RuntimeInputBinding:
    """A verified environment-specific locator for an immutable authority."""

    scientific_input_authority_id: str
    effective_remote_input_root: Path
    preferred_remote_input_root: Path
    expected_file_count: int
    expected_total_bytes: int
    resolved_candidates: tuple[Path, ...]
    parity_report_sha256: str | None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe execution-provenance record."""

        return {
            "scientific_input_authority_id": self.scientific_input_authority_id,
            "effective_remote_input_root": str(self.effective_remote_input_root),
            "preferred_remote_input_root": str(self.preferred_remote_input_root),
            "expected_file_count": self.expected_file_count,
            "expected_total_bytes": self.expected_total_bytes,
            "resolved_candidates": [str(path) for path in self.resolved_candidates],
            "parity_report_sha256": self.parity_report_sha256,
        }


def load_remote_input_authority(path: Path) -> RemoteInputAuthority:
    """Load an explicit locator contract; never discover arbitrary directories."""

    payload = _read_json(path)
    if payload.get("schema_version") != AUTHORITY_SCHEMA:
        raise RemoteInputResolutionError("unsupported remote input authority schema")
    if payload.get("status") != "ACTIVE_RUNTIME_LOCATOR_CONTRACT":
        raise RemoteInputResolutionError("remote input authority is not active")
    scientific = _mapping(payload, "scientific_input_authority")
    runtime = _mapping(payload, "runtime_input_locators")
    expected = _mapping(scientific, "expected_population")
    sentinels = _mapping(scientific, "registered_sentinels")
    parity_evidence = _mapping(scientific, "historical_parity_evidence")
    candidates = tuple(Path(value) for value in _strings(runtime, "registered"))
    preferred = Path(_string(runtime, "preferred"))
    if not candidates or candidates[0] != preferred or preferred not in candidates:
        raise RemoteInputResolutionError("registered input locators must start with preferred")
    if len(set(candidates)) != len(candidates):
        raise RemoteInputResolutionError("registered input locators must be unique")
    expected_files = _integer(expected, "physical_file_count")
    expected_bytes = _integer(expected, "total_bytes")
    if expected_files <= 0 or expected_bytes <= 0:
        raise RemoteInputResolutionError("registered input population must be positive")
    return RemoteInputAuthority(
        authority_id=_string(scientific, "authority_id"),
        expected_file_count=expected_files,
        expected_total_bytes=expected_bytes,
        preferred_runtime_locator=preferred,
        registered_runtime_locators=candidates,
        sentinel_sha256={key: _sha256(value) for key, value in sentinels.items()},
        historical_parity_evidence={
            "relative_path": _string(parity_evidence, "relative_path"),
            "sha256": _sha256(_string(parity_evidence, "sha256")),
        },
        parity_report_locator=Path(_string(runtime, "parity_report_locator")),
    )


def resolve_remote_input_root(
    authority: RemoteInputAuthority,
    *,
    parity_report_path: Path | None = None,
) -> RuntimeInputBinding:
    """Select exactly one registered, parity-proven input realization."""

    report_sha256 = _validate_parity_report(
        parity_report_path or authority.parity_report_locator,
        authority,
    )
    valid: list[Path] = []
    invalid: list[str] = []
    for locator in authority.registered_runtime_locators:
        if not locator.exists():
            continue
        if not locator.is_dir():
            invalid.append(f"not_directory={locator}")
            continue
        resolved = locator.resolve(strict=True)
        try:
            _validate_candidate(resolved, authority)
        except RemoteInputResolutionError as error:
            invalid.append(f"{locator}:{error}")
        else:
            valid.append(resolved)
    if invalid:
        raise RemoteInputResolutionError(
            "registered input locator conflict: " + "; ".join(invalid)
        )
    if not valid:
        raise RemoteInputResolutionError("no registered input locator passed parity")
    unique = tuple(dict.fromkeys(valid))
    if len(unique) != 1:
        raise RemoteInputResolutionError("multiple registered locators are not equivalent")
    selected = next(
        locator
        for locator in authority.registered_runtime_locators
        if locator.exists() and locator.resolve(strict=True) == unique[0]
    )
    return RuntimeInputBinding(
        scientific_input_authority_id=authority.authority_id,
        effective_remote_input_root=selected.resolve(strict=True),
        preferred_remote_input_root=authority.preferred_runtime_locator,
        expected_file_count=authority.expected_file_count,
        expected_total_bytes=authority.expected_total_bytes,
        resolved_candidates=unique,
        parity_report_sha256=report_sha256,
    )


def _validate_candidate(path: Path, authority: RemoteInputAuthority) -> None:
    count, total_bytes = _inventory(path)
    if count != authority.expected_file_count or total_bytes != authority.expected_total_bytes:
        raise RemoteInputResolutionError(
            f"inventory={count}/{total_bytes}, expected="
            f"{authority.expected_file_count}/{authority.expected_total_bytes}"
        )
    for relative, expected_sha256 in authority.sentinel_sha256.items():
        sentinel = path / relative
        if not sentinel.is_file():
            raise RemoteInputResolutionError(f"missing_sentinel={relative}")
        if _sha256_file(sentinel) != expected_sha256:
            raise RemoteInputResolutionError(f"sentinel_hash_mismatch={relative}")


def _validate_parity_report(path: Path, authority: RemoteInputAuthority) -> str | None:
    """Accept an optional historical report only when its numeric parity agrees."""

    if not path.exists():
        return None
    report = _read_json(path)
    report_values = _find_integer_values(report)
    expected_values = {authority.expected_file_count, authority.expected_total_bytes}
    if not expected_values.issubset(report_values):
        raise RemoteInputResolutionError("registered parity report disagrees with authority")
    return _sha256_file(path)


def _inventory(root: Path) -> tuple[int, int]:
    count = 0
    total_bytes = 0
    for directory, _, files in os.walk(root, followlinks=False):
        for filename in files:
            item = Path(directory) / filename
            if item.is_symlink():
                raise RemoteInputResolutionError(f"symlinked_input_file={item}")
            count += 1
            total_bytes += item.stat().st_size
    return count, total_bytes


def _find_integer_values(value: object) -> set[int]:
    if isinstance(value, bool):
        return set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, Mapping):
        return set().union(*(_find_integer_values(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_find_integer_values(item) for item in value))
    return set()


def _read_json(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RemoteInputResolutionError(f"JSON object required: {path}")
    return payload


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise RemoteInputResolutionError(f"mapping required: {key}")
    return value


def _string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RemoteInputResolutionError(f"nonempty string required: {key}")
    return value


def _strings(payload: Mapping[str, Any], key: str) -> Iterable[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise RemoteInputResolutionError(f"nonempty string list required: {key}")
    return value


def _integer(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RemoteInputResolutionError(f"integer required: {key}")
    return value


def _sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RemoteInputResolutionError("SHA256 values must be lowercase hexadecimal")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
