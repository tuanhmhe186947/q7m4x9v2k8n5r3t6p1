"""Immutable authority contract for Classification V2 behavior review.

The manifest produced here binds only review-critical Group A artifacts.  It
deliberately excludes final T6/T8/T12/T16 outputs, folds, weights, and model
configuration because those are post-review concerns.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.features.frame_local import (
    forbidden_frame_local_columns,
)
from pig_behavior.classification_v2.training.lineage_hashing import (
    file_sha256,
    payload_sha256,
)

REVIEW_AUTHORITY_SCHEMA_VERSION = (
    "classification_v2.behavior_review_authority.v1"
)
OFFICIAL_SCOPE = "official_v4_pre_behavior_review"
SMOKE_SCOPE = "representative_smoke_only"
VALID_SCOPES = frozenset({OFFICIAL_SCOPE, SMOKE_SCOPE})
GIT_COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
STOPPED_V3 = "c2v2_human_review_20260721_reviewer01_v3"

PAIR_DERIVED_COLUMN_PATTERNS = (
    re.compile(r"^(prev|next|delta)_"),
    re.compile(r"(^|_)displacement(_|$)"),
    re.compile(r"(^|_)speed(_|$)"),
    re.compile(r"(^|_)accel(eration)?(_|$)"),
    re.compile(r"(^|_)path_length(_|$)"),
    re.compile(r"(^|_)direction_change(_|$)"),
    re.compile(r"(^|_)motion_energy(_|$)"),
    re.compile(r"(^|_)(entry|exit)_event(_|$)"),
    re.compile(r"_(unit|window)$"),
)

REQUIRED_ARTIFACT_KEYS = (
    "frame_local",
    "hidden_reviewed_frames",
    "harmonized_frames",
    "temporal_native_units",
    "pig_strenet_evidence",
    "behavior_review_units",
    "media_authority",
)

OFFICIAL_REQUIRED_ARTIFACT_KEYS = (
    "timestamp_fps_contract",
    "evidence_semantics",
)

REQUIRED_COMPONENT_GATE_KEYS = (
    "frame_local",
    "hidden_coverage",
    "hidden_scientific",
    "hidden_apply",
    "temporal_harmonization",
    "native_evidence",
    "pig_strenet",
    "native_review_unit_coverage",
    "timestamp_fps",
    "evidence_semantics",
    "media_authority",
)


def build_review_authority_manifest(
    *,
    code_authority_sha: str,
    code_dirty: bool,
    lineage_id: str,
    authority_scope: str,
    source_artifacts: Mapping[str, Path],
    artifacts: Mapping[str, Path],
    timestamp_fps_contract: Mapping[str, Any],
    evidence_semantics: Mapping[str, Any],
    component_gates: Mapping[str, Path] | None = None,
    actual_head_sha: str | None = None,
    tracked_code_clean: bool | None = None,
    require_full_component_gates: bool = False,
) -> dict[str, Any]:
    """Build one deterministic, fail-closed review-authority manifest."""

    errors: list[str] = []
    code_sha = str(code_authority_sha).strip().lower()
    if not GIT_COMMIT_SHA_PATTERN.fullmatch(code_sha):
        errors.append("invalid_code_authority_sha")
    if authority_scope not in VALID_SCOPES:
        errors.append(f"invalid_authority_scope={authority_scope}")
    if not str(lineage_id).strip():
        errors.append("blank_lineage_id")

    missing_artifact_keys = sorted(
        set(REQUIRED_ARTIFACT_KEYS).difference(artifacts)
    )
    if missing_artifact_keys:
        errors.append(
            f"missing_review_authority_artifact_keys={missing_artifact_keys}"
        )

    source_profiles = {
        name: _artifact_profile(Path(path), errors, f"source:{name}")
        for name, path in sorted(source_artifacts.items())
    }
    artifact_profiles = {
        name: _artifact_profile(Path(path), errors, name)
        for name, path in sorted(artifacts.items())
    }
    if not source_profiles:
        errors.append("source_artifacts_empty")

    official = authority_scope == OFFICIAL_SCOPE
    observed_head = str(actual_head_sha or code_sha).strip().lower()
    observed_clean = (
        not code_dirty if tracked_code_clean is None else tracked_code_clean
    )
    all_paths = [
        Path(path)
        for path in [*source_artifacts.values(), *artifacts.values()]
    ]
    if official:
        missing_official_artifacts = sorted(
            set(OFFICIAL_REQUIRED_ARTIFACT_KEYS).difference(artifacts)
        )
        if missing_official_artifacts:
            errors.append(
                "missing_official_contract_artifacts="
                f"{missing_official_artifacts}"
            )
        if code_dirty or not observed_clean:
            errors.append("official_authority_requires_clean_code")
        if observed_head != code_sha:
            errors.append(
                "actual_head_mismatch="
                f"supplied:{code_sha},actual:{observed_head}"
            )
        if not str(lineage_id).strip().endswith("_v4"):
            errors.append("official_authority_requires_v4_lineage_id")
        stopped_v3_paths = [
            str(path)
            for path in all_paths
            if "c2v2_human_review_20260721_reviewer01_v3"
            in str(path).replace("\\", "/")
        ]
        if stopped_v3_paths:
            errors.append(
                "official_authority_references_stopped_v3="
                f"{stopped_v3_paths}"
            )
        stopped_v3_payloads = [
            str(path)
            for path in all_paths
            if _file_contains_stopped_v3(Path(path))
        ]
        if stopped_v3_payloads:
            errors.append(
                "official_authority_payload_references_stopped_v3="
                f"{stopped_v3_payloads}"
            )

    gate_paths = dict(component_gates or {})
    gate_profiles, gate_payloads = _component_gate_profiles(
        gate_paths,
        errors,
        required=require_full_component_gates,
        lineage_id=str(lineage_id),
        official=official,
    )

    frame_local_path = artifacts.get("frame_local")
    frame_local_schema: dict[str, Any] = {}
    if frame_local_path is not None and Path(frame_local_path).is_file():
        frame_local_schema = _csv_schema(Path(frame_local_path), errors)
        errors.extend(
            audit_frame_local_schema(frame_local_schema.get("columns", []))
        )

    temporal_identity = _manifest_identity(
        artifacts.get("temporal_native_units"),
        manifest_kind="temporal_native_units",
        errors=errors,
    )
    review_identity = _manifest_identity(
        artifacts.get("behavior_review_units"),
        manifest_kind="behavior_review_units",
        errors=errors,
    )
    evidence_version = str(
        evidence_semantics.get("evidence_column_semantic_version", "")
    ).strip()
    if not evidence_version:
        errors.append("missing_evidence_column_semantic_version")
    for name, payload in (
        ("timestamp_fps_contract", timestamp_fps_contract),
        ("evidence_semantics", evidence_semantics),
    ):
        if payload.get("valid") is not True or payload.get("errors") != []:
            errors.append(f"invalid_embedded_contract={name}")
        declared_lineage = str(payload.get("lineage_id", "")).strip()
        if official and declared_lineage and declared_lineage != str(lineage_id):
            errors.append(f"embedded_contract_lineage_mismatch={name}")

    components = {
        "source_artifact_hashes": {
            name: profile.get("sha256")
            for name, profile in source_profiles.items()
        },
        "frame_local_artifact_sha256": _profile_sha(
            artifact_profiles,
            "frame_local",
        ),
        "frame_local_schema_sha256": payload_sha256(frame_local_schema),
        "timestamp_fps_contract_file_sha256": _profile_sha(
            artifact_profiles,
            "timestamp_fps_contract",
        ),
        "timestamp_fps_contract_payload_sha256": payload_sha256(
            dict(timestamp_fps_contract)
        ),
        "hidden_reviewed_frame_artifact_sha256": _profile_sha(
            artifact_profiles,
            "hidden_reviewed_frames",
        ),
        "harmonized_frame_artifact_sha256": _profile_sha(
            artifact_profiles,
            "harmonized_frames",
        ),
        "temporal_native_unit_artifact_sha256": _profile_sha(
            artifact_profiles,
            "temporal_native_units",
        ),
        "temporal_native_unit_key_span_sha256": temporal_identity.get(
            "key_span_sha256"
        ),
        "pig_strenet_evidence_artifact_sha256": _profile_sha(
            artifact_profiles,
            "pig_strenet_evidence",
        ),
        "pig_strenet_evidence_schema_sha256": _profile_schema_sha(
            artifact_profiles,
            "pig_strenet_evidence",
        ),
        "behavior_review_unit_manifest_sha256": _profile_sha(
            artifact_profiles,
            "behavior_review_units",
        ),
        "behavior_review_unit_key_span_sha256": review_identity.get(
            "key_span_sha256"
        ),
        "media_crop_video_authority_sha256": _profile_sha(
            artifact_profiles,
            "media_authority",
        ),
        "evidence_semantics_file_sha256": _profile_sha(
            artifact_profiles,
            "evidence_semantics",
        ),
        "evidence_column_semantics_payload_sha256": payload_sha256(
            dict(evidence_semantics)
        ),
        "component_gate_hashes": {
            name: profile.get("sha256")
            for name, profile in gate_profiles.items()
        },
    }
    core = {
        "schema_version": REVIEW_AUTHORITY_SCHEMA_VERSION,
        "authority_scope": authority_scope,
        "lineage_id": str(lineage_id).strip(),
        "code_authority_sha": code_sha,
        "code_dirty": bool(code_dirty),
        "actual_head_sha": observed_head,
        "tracked_code_clean": bool(observed_clean),
        "components": components,
    }
    valid = not errors
    authority_sha = payload_sha256(core) if valid else None
    return {
        **core,
        "official_review_authority": official,
        "authorizes_behavior_gui": bool(valid and official),
        "authorizes_final_view_build": False,
        "authorizes_training": False,
        "valid": valid,
        "errors": errors,
        "source_artifacts": source_profiles,
        "artifacts": artifact_profiles,
        "frame_local_schema": frame_local_schema,
        "timestamp_fps_contract": dict(timestamp_fps_contract),
        "temporal_native_unit_identity": temporal_identity,
        "behavior_review_unit_identity": review_identity,
        "evidence_semantics": dict(evidence_semantics),
        "component_gates": gate_profiles,
        "component_gate_payloads": gate_payloads,
        "review_authority_sha256": authority_sha,
    }


def audit_frame_local_schema(columns: Sequence[str]) -> list[str]:
    """Reject pair-derived or aggregate columns from frame-local primitives."""

    normalized = [str(column).strip() for column in columns]
    forbidden = forbidden_frame_local_columns(normalized)
    pattern_only = sorted(
        column
        for column in normalized
        if any(pattern.search(column) for pattern in PAIR_DERIVED_COLUMN_PATTERNS)
    )
    forbidden = sorted(set(forbidden) | set(pattern_only))
    errors: list[str] = []
    if forbidden:
        errors.append(f"frame_local_contains_pair_or_aggregate={forbidden}")
    if "feature_computation_grain" in normalized:
        # Grain values are checked from data rows by the producing stage.  The
        # schema itself may retain the explicit audit field.
        return errors
    return errors


def _component_gate_profiles(
    paths: Mapping[str, Path],
    errors: list[str],
    *,
    required: bool,
    lineage_id: str,
    official: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    payloads: dict[str, Any] = {}
    if required:
        missing = sorted(set(REQUIRED_COMPONENT_GATE_KEYS).difference(paths))
        if missing:
            errors.append(f"missing_component_gates={missing}")
    for name, raw_path in sorted(paths.items()):
        path = Path(raw_path)
        profile = _artifact_profile(path, errors, f"component_gate:{name}")
        profiles[name] = profile
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid_component_gate_json={name}:{exc}")
            continue
        payloads[name] = payload
        if not isinstance(payload, dict):
            errors.append(f"component_gate_not_object={name}")
            continue
        if "errors" not in payload and "status" not in payload:
            errors.append(f"component_gate_missing_result_fields={name}")
        gate_errors = payload.get("errors", [])
        if gate_errors != []:
            errors.append(f"component_gate_errors={name}:{gate_errors}")
        valid = payload.get("valid")
        passed = payload.get("pass")
        status = str(payload.get("status", "")).strip().upper()
        if valid is False or passed is False or (status and status != "PASS"):
            errors.append(f"component_gate_not_pass={name}")
        normalized = json.dumps(payload, ensure_ascii=False).casefold()
        if STOPPED_V3.casefold() in normalized:
            errors.append(f"component_gate_references_stopped_v3={name}")
        declared_lineage = str(payload.get("lineage_id", "")).strip()
        path_contains_lineage = lineage_id in str(path)
        if official and declared_lineage != lineage_id and not path_contains_lineage:
            errors.append(f"component_gate_lineage_mismatch={name}")
    return profiles, payloads


def _file_contains_stopped_v3(path: Path) -> bool:
    if not path.is_file() or path.suffix.casefold() not in {".json", ".csv"}:
        return False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            while chunk := stream.read(1024 * 1024):
                if STOPPED_V3.casefold() in chunk.casefold():
                    return True
    except OSError:
        return False
    return False


def _artifact_profile(
    path: Path,
    errors: list[str],
    name: str,
) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": None,
        "sha256": None,
        "schema": {},
        "schema_sha256": None,
    }
    if not path.is_file():
        errors.append(f"missing_artifact={name}:{path}")
        return profile
    profile["size_bytes"] = int(path.stat().st_size)
    profile["sha256"] = file_sha256(path)
    schema = _artifact_schema(path, errors)
    profile["schema"] = schema
    profile["schema_sha256"] = payload_sha256(schema)
    return profile


def _artifact_schema(path: Path, errors: list[str]) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _csv_schema(path, errors)
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid_json_artifact={path}:{exc}")
            return {"format": "json", "top_level_keys": []}
        keys = sorted(payload) if isinstance(payload, dict) else []
        return {"format": "json", "top_level_keys": keys}
    return {"format": suffix.lstrip(".") or "unknown"}


def _csv_schema(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        columns = pd.read_csv(path, nrows=0).columns.astype(str).tolist()
    except Exception as exc:  # pandas parser errors vary by engine/version.
        errors.append(f"invalid_csv_artifact={path}:{exc}")
        columns = []
    return {
        "format": "csv",
        "columns": columns,
        "column_count": len(columns),
    }


def _manifest_identity(
    path: Path | None,
    *,
    manifest_kind: str,
    errors: list[str],
) -> dict[str, Any]:
    if path is None or not Path(path).is_file():
        return {"rows": 0, "units": 0, "key_span_sha256": None}
    frame = pd.read_csv(Path(path), low_memory=False)
    if manifest_kind == "behavior_review_units":
        key = "review_unit_id"
        start_candidates = ("unit_start_frame", "label_window_start")
        end_candidates = ("unit_end_frame", "label_window_end")
    else:
        key = "temporal_unit_key"
        start_candidates = ("label_window_start", "unit_start_frame")
        end_candidates = ("label_window_end", "unit_end_frame")
    if key not in frame.columns:
        errors.append(f"{manifest_kind}_missing_key={key}")
        return {"rows": len(frame), "units": 0, "key_span_sha256": None}
    start = _first_present(frame, start_candidates)
    end = _first_present(frame, end_candidates)
    if start is None or end is None:
        if "frame_index" not in frame.columns:
            errors.append(f"{manifest_kind}_missing_frame_span")
            return {
                "rows": len(frame),
                "units": int(frame[key].nunique()),
                "key_span_sha256": None,
            }
        start = end = "frame_index"

    identities = [
        column
        for column in (
            "source_type",
            "dataset_id",
            "video_key",
            "object_track_key",
            "pig_id",
            "track_id",
        )
        if column in frame.columns
    ]
    records: list[dict[str, Any]] = []
    blank_keys = frame[key].fillna("").astype(str).str.strip().eq("")
    if blank_keys.any():
        errors.append(f"{manifest_kind}_blank_keys={int(blank_keys.sum())}")
    for value, group in frame.loc[~blank_keys].groupby(
        key,
        dropna=False,
        sort=True,
    ):
        record: dict[str, Any] = {
            key: str(value),
            "start_frame": int(pd.to_numeric(group[start]).min()),
            "end_frame": int(pd.to_numeric(group[end]).max()),
        }
        for column in identities:
            values = sorted(
                set(group[column].fillna("").astype(str).tolist())
            )
            if len(values) != 1:
                errors.append(
                    f"{manifest_kind}_identity_mismatch="
                    f"{value}:{column}:{values}"
                )
            record[column] = values[0] if values else ""
        records.append(record)
    records.sort(key=lambda item: str(item[key]))
    return {
        "rows": int(len(frame)),
        "units": int(len(records)),
        "key_column": key,
        "start_column": start,
        "end_column": end,
        "key_span_sha256": payload_sha256(records),
    }


def _first_present(
    frame: pd.DataFrame,
    candidates: Sequence[str],
) -> str | None:
    return next((column for column in candidates if column in frame), None)


def _profile_sha(
    profiles: Mapping[str, Mapping[str, Any]],
    name: str,
) -> str | None:
    return profiles.get(name, {}).get("sha256")


def _profile_schema_sha(
    profiles: Mapping[str, Mapping[str, Any]],
    name: str,
) -> str | None:
    return profiles.get(name, {}).get("schema_sha256")


__all__ = [
    "OFFICIAL_SCOPE",
    "REVIEW_AUTHORITY_SCHEMA_VERSION",
    "SMOKE_SCOPE",
    "audit_frame_local_schema",
    "build_review_authority_manifest",
]
