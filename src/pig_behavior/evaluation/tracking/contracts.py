"""Version contracts for scientifically comparable tracking metrics."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Iterable, Mapping
from typing import Any

LEGACY_EVALUATOR_CONTRACT_ID = "TRACKING_EVALUATOR_LEGACY_V1"
EVALUATOR_CONTRACT_ID = "TRACKING_EVALUATOR_STANDARD_V2"
MATCHING_CONTRACT_ID = "TRACKING_MATCHING_STANDARD_V2"
IDENTITY_EPISODE_CONTRACT_ID = "IDENTITY_ERROR_EPISODES_V2"
IDENTITY_AUTHORITY_POLICY = "IDENTITY_AUTHORITY_FIRST_OBSERVATION_V2"
SEQUENCE_BOUNDARY_POLICY = "VIDEO_SESSION_ISOLATED_V2"
IDSW_POLICY = "TRACKEVAL_CLEAR_LAST_MATCH_WITHIN_SEQUENCE"
REFERENCE_PARITY_PASS = "PASS"
HOTA_ALPHAS = tuple(round(value / 100, 2) for value in range(5, 100, 5))

_REQUIRED_METADATA = (
    "evaluator_contract_id",
    "identity_episode_contract_id",
    "matching_contract_id",
    "hota_threshold_set",
    "include_hidden",
    "sequence_boundary_policy",
    "idsw_policy",
    "identity_authority_policy",
    "reference_parity_status",
    "evaluator_code_sha",
    "metric_config_sha256",
)
_FORBIDDEN_V2_FIELDS = (
    "permanent_swap",
    "terminal_swap",
    "remapped_hota",
    "remapped_assa",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class MetricContractError(ValueError):
    """Raised when metric rows mix or omit evaluator contracts."""


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a mapping with deterministic JSON serialization."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_evaluator_code_sha() -> str:
    """Resolve the checked-out Git commit without modifying the repository."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MetricContractError("Unable to resolve evaluator Git SHA") from exc
    value = completed.stdout.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise MetricContractError(f"Invalid evaluator Git SHA: {value!r}")
    return value


def build_metric_metadata(
    *,
    include_hidden: bool,
    evaluator_code_sha: str,
    detection_iou_threshold: float = 0.5,
    identity_iou_threshold: float = 0.5,
    identity_authority_policy: str = IDENTITY_AUTHORITY_POLICY,
    reference_parity_status: str = REFERENCE_PARITY_PASS,
) -> dict[str, Any]:
    """Build required V2 metadata and its semantic configuration hash."""
    semantic_config = {
        "evaluator_contract_id": EVALUATOR_CONTRACT_ID,
        "matching_contract_id": MATCHING_CONTRACT_ID,
        "identity_episode_contract_id": IDENTITY_EPISODE_CONTRACT_ID,
        "identity_authority_policy": identity_authority_policy,
        "hota_threshold_set": list(HOTA_ALPHAS),
        "detection_iou_threshold": float(detection_iou_threshold),
        "identity_iou_threshold": float(identity_iou_threshold),
        "include_hidden": bool(include_hidden),
        "sequence_boundary_policy": SEQUENCE_BOUNDARY_POLICY,
        "idsw_policy": IDSW_POLICY,
        "episode_max_frame_delta": 15,
        "pairwise_persistence_observations": 60,
    }
    return {
        **semantic_config,
        "reference_parity_status": reference_parity_status,
        "evaluator_code_sha": evaluator_code_sha,
        "metric_config_sha256": canonical_sha256(semantic_config),
    }


def validate_report_contract(
    rows: Iterable[Mapping[str, Any]],
    *,
    allow_historical_legacy: bool = False,
) -> str:
    """Validate one homogeneous report and return its evaluator contract ID."""
    materialized = list(rows)
    if not materialized:
        raise MetricContractError("A report must contain at least one metric row")

    declared = {
        str(row.get("evaluator_contract_id", "")).strip() for row in materialized
    }
    if declared == {""}:
        if allow_historical_legacy:
            return LEGACY_EVALUATOR_CONTRACT_ID
        raise MetricContractError("Unversioned reports are historical-only")
    if "" in declared or len(declared) != 1:
        raise MetricContractError("Mixed or partially versioned report")

    contract_id = next(iter(declared))
    if contract_id == LEGACY_EVALUATOR_CONTRACT_ID:
        raise MetricContractError("Legacy V1 is read-only for historical reports")
    if contract_id != EVALUATOR_CONTRACT_ID:
        raise MetricContractError(f"Unknown evaluator contract: {contract_id}")

    first = materialized[0]
    for field in _REQUIRED_METADATA:
        if field not in first:
            raise MetricContractError(f"V2 report is missing metadata: {field}")
        expected = first[field]
        if any(row.get(field) != expected for row in materialized[1:]):
            raise MetricContractError(f"V2 metadata differs across rows: {field}")

    if tuple(first["hota_threshold_set"]) != HOTA_ALPHAS:
        raise MetricContractError("V2 report uses a non-canonical HOTA alpha set")
    if first["matching_contract_id"] != MATCHING_CONTRACT_ID:
        raise MetricContractError("V2 report uses the wrong matching contract")
    if first["identity_episode_contract_id"] != IDENTITY_EPISODE_CONTRACT_ID:
        raise MetricContractError("V2 report uses the wrong episode contract")
    if first["reference_parity_status"] != REFERENCE_PARITY_PASS:
        raise MetricContractError("V2 headline report lacks reference parity")
    if not _SHA256_PATTERN.fullmatch(str(first["metric_config_sha256"])):
        raise MetricContractError("V2 report has an invalid metric config hash")

    for row in materialized:
        present = [field for field in _FORBIDDEN_V2_FIELDS if field in row]
        if present:
            fields = ", ".join(sorted(present))
            raise MetricContractError(f"Legacy fields are forbidden in V2: {fields}")
    return EVALUATOR_CONTRACT_ID
