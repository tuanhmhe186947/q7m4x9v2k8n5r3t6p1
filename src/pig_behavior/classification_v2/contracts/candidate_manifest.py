"""Production-owned candidate artifact manifest construction."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.contracts.model_io import (
    validate_model_input_columns,
)
from pig_behavior.classification_v2.contracts.runtime_dependencies import (
    assert_stage_runtime_dependencies_complete,
)
from pig_behavior.classification_v2.contracts.semantic_lineage import (
    ARTIFACT_MANIFEST_VERSION,
    AXIS_DISTANCE_METRIC_ID,
    AXIS_DISTANCE_METRIC_VERSION,
    CANDIDATE_AUTHORITY_STATE,
    DIAGONAL_DISTANCE_METRIC_ID,
    DIAGONAL_DISTANCE_METRIC_VERSION,
    MANIFEST_BUILDER_ID,
    MANIFEST_BUILDER_VERSION,
    MOTION_SCHEMA_HASH,
    MOTION_SCHEMA_ID,
    MOTION_SCHEMA_VERSION,
    ROI_AGGREGATION_VERSION,
    SOCIAL_IDENTITY_VERSION,
    SOCIAL_TIE_BREAK_VERSION,
    build_semantic_bundle,
    build_semantic_domain_registry,
    canonical_json_bytes,
    canonical_sha256,
    compute_stage_code_hash,
    compute_stage_execution_fingerprint,
    compute_stage_input_fingerprint,
    compute_stage_semantics_hash,
    file_sha256,
    git_code_authority,
    load_code_contract_mapping,
    load_scientific_contract,
    validate_artifact_manifest,
)
from pig_behavior.classification_v2.features.motion_schema import (
    MOTION_FEATURE_NAMES,
)

CANDIDATE_MANIFEST_DEVELOPMENT_CONTRACT_VERSION = (
    "classification_v2.candidate_manifest_builder.development.v1"
)
OUTPUT_INSPECTOR_REGISTRY_ID = (
    "registry.classification_v2.candidate_output_inspectors"
)
OUTPUT_INSPECTOR_REGISTRY_VERSION = (
    "classification_v2.candidate_output_inspectors.v1"
)

_SCIENTIFIC_AUTHORITY_PATH_PREFIXES = (
    "src/",
    "scripts/classification_v2/",
    "docs/classification_v2/scientific_contract_v1/",
)
_OFFICIAL_ARTIFACT_CLASSES = frozenset(
    {"OFFICIAL", "OFFICIAL_SCIENTIFIC"}
)
_SUPPORTED_ARTIFACT_CLASSES = frozenset(
    {
        "NON_OFFICIAL_AUDIT",
        "SYNTHETIC_INTEGRATION_TEST_ONLY",
        "SCIENTIFIC_CANDIDATE",
        *_OFFICIAL_ARTIFACT_CLASSES,
    }
)
_INVALID_UPSTREAM_CLASS_TOKENS = (
    "FAILED",
    "STOPPED",
    "DIAGNOSTIC",
)
_OBJECT_TRACK_KEY_STAGES = frozenset(
    {
        "stage.legacy_cvat_source_merge",
        "stage.frame_local_primitives",
        "stage.hidden_review_design",
        "stage.hidden_apply",
        "stage.temporal_harmonization",
        "stage.native_review_evidence",
        "stage.pig_strenet_evidence",
        "stage.behavior_review_unit_construction",
        "stage.behavior_decision_apply",
    }
)
_MODEL_INPUT_STAGES = frozenset(
    {
        "stage.train_ready_export",
        "stage.tensor_export",
        "stage.model_input",
    }
)
_MOTION_SCHEMA_STAGES = frozenset(
    {
        "stage.native_review_evidence",
        "stage.tensor_export",
    }
)
_LOAD_BEARING_FIELDS = frozenset(
    {
        "artifact_manifest_version",
        "stage_version",
        "code_authority_vcs",
        "code_authority_object_format",
        "created_by_code_authority_sha",
        "stage_code_hash",
        "stage_semantics_hash",
        "semantic_bundle_hash",
        "contract_manifest_hash",
        "input_artifact_ids",
        "input_artifact_fingerprints",
        "stage_input_fingerprint",
        "stage_execution_fingerprint",
        "input_file_sha256",
        "output_file_sha256",
        "output_schema_hash",
        "row_count",
        "column_count",
        "output_byte_size",
        "manifest_builder_code_hash",
    }
)
_EPHEMERAL_METADATA_KEY_TOKENS = (
    "absolute_path",
    "created_at",
    "generated_at",
    "timestamp",
    "username",
)


@dataclass(frozen=True, slots=True)
class OutputInspection:
    """Byte and schema authority derived from one actual output."""

    inspector_id: str
    inspector_version: str
    output_file_sha256: str
    output_byte_size: int
    row_count: int
    column_count: int
    ordered_columns: tuple[str, ...]
    schema_payload: Mapping[str, Any]
    model_feature_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LoadedUpstreamManifest:
    """One validated upstream manifest and its resolved output."""

    manifest_path: Path
    output_path: Path
    manifest: Mapping[str, Any]
    manifest_file_sha256: str


@dataclass(frozen=True, slots=True)
class CandidateManifestBuild:
    """Result returned only after atomic write and independent revalidation."""

    manifest: Mapping[str, Any]
    manifest_path: Path
    output_inspection: OutputInspection
    validation: Mapping[str, Any]
    production_builder_owned: bool


OutputInspector = Callable[[Path], OutputInspection]


def _digest(path: Path) -> str:
    return file_sha256(path)


def _inspection(
    path: Path,
    *,
    inspector_id: str,
    rows: int,
    columns: Sequence[str],
    schema_payload: Mapping[str, Any],
    model_feature_names: Sequence[str] = (),
) -> OutputInspection:
    return OutputInspection(
        inspector_id=inspector_id,
        inspector_version=OUTPUT_INSPECTOR_REGISTRY_VERSION,
        output_file_sha256=_digest(path),
        output_byte_size=path.stat().st_size,
        row_count=int(rows),
        column_count=len(columns),
        ordered_columns=tuple(str(value) for value in columns),
        schema_payload=dict(schema_payload),
        model_feature_names=tuple(str(value) for value in model_feature_names),
    )


def _inspect_csv(path: Path) -> OutputInspection:
    frame = pd.read_csv(path, low_memory=False)
    columns = [str(value) for value in frame.columns]
    return _inspection(
        path,
        inspector_id="inspector.csv.v1",
        rows=len(frame),
        columns=columns,
        schema_payload={
            "format": "csv",
            "ordered_columns": columns,
            "dtypes": [str(value) for value in frame.dtypes],
        },
    )


def _model_feature_names_from_json(value: Any) -> list[str]:
    names: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in {
                "feature_names",
                "features",
                "model_feature_names",
                "ordered_feature_names",
                "requested_features",
            } and isinstance(child, list):
                names.extend(
                    str(item)
                    for item in child
                    if isinstance(item, str)
                )
            elif normalized != "forbidden_model_inputs":
                names.extend(_model_feature_names_from_json(child))
    elif isinstance(value, list):
        for child in value:
            names.extend(_model_feature_names_from_json(child))
    return names


def _inspect_json(path: Path) -> OutputInspection:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        columns = [str(value) for value in payload]
        rows = 1
        shape = "object"
    elif isinstance(payload, list):
        rows = len(payload)
        columns = []
        for row in payload:
            if not isinstance(row, Mapping):
                raise ValueError("JSON_ROWS_MUST_BE_OBJECTS")
            for key in row:
                if str(key) not in columns:
                    columns.append(str(key))
        shape = "array"
    else:
        raise ValueError("JSON_OUTPUT_MUST_BE_OBJECT_OR_ARRAY")
    return _inspection(
        path,
        inspector_id="inspector.json.v1",
        rows=rows,
        columns=columns,
        schema_payload={
            "format": "json",
            "json_shape": shape,
            "ordered_columns": columns,
        },
        model_feature_names=sorted(
            set(_model_feature_names_from_json(payload))
        ),
    )


def _inspect_jsonl(path: Path) -> OutputInspection:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"JSONL_ROW_NOT_OBJECT:{line_number}")
        rows.append(value)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in columns:
                columns.append(str(key))
    return _inspection(
        path,
        inspector_id="inspector.jsonl.v1",
        rows=len(rows),
        columns=columns,
        schema_payload={
            "format": "jsonl",
            "ordered_columns": columns,
        },
    )


def _inspect_npz(path: Path) -> OutputInspection:
    with np.load(path, allow_pickle=False) as arrays:
        names = sorted(str(value) for value in arrays.files)
        metadata = []
        row_counts = set()
        for name in names:
            array = arrays[name]
            metadata.append(
                {
                    "name": name,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                }
            )
            if array.ndim:
                row_counts.add(int(array.shape[0]))
        if len(row_counts) > 1:
            raise ValueError(f"NPZ_ROW_COUNT_MISMATCH:{sorted(row_counts)}")
        rows = next(iter(row_counts), 0)
    return _inspection(
        path,
        inspector_id="inspector.npz.v1",
        rows=rows,
        columns=names,
        schema_payload={"format": "npz", "arrays": metadata},
    )


def _inspect_npy(path: Path) -> OutputInspection:
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    columns = [f"axis_{index}" for index in range(array.ndim)]
    rows = int(array.shape[0]) if array.ndim else 1
    return _inspection(
        path,
        inspector_id="inspector.npy.v1",
        rows=rows,
        columns=columns,
        schema_payload={
            "format": "npy",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        },
    )


_OUTPUT_INSPECTORS: dict[str, OutputInspector] = {
    ".csv": _inspect_csv,
    ".json": _inspect_json,
    ".jsonl": _inspect_jsonl,
    ".npy": _inspect_npy,
    ".npz": _inspect_npz,
}


def output_inspector_registry() -> dict[str, Any]:
    """Return the immutable typed output-inspector authority."""

    return {
        "registry_id": OUTPUT_INSPECTOR_REGISTRY_ID,
        "registry_version": OUTPUT_INSPECTOR_REGISTRY_VERSION,
        "inspectors": {
            suffix: inspector.__name__
            for suffix, inspector in sorted(_OUTPUT_INSPECTORS.items())
        },
        "unsupported_output_policy": "UNSUPPORTED_OUTPUT_INSPECTOR",
    }


def inspect_candidate_output(path: Path) -> OutputInspection:
    """Inspect actual output bytes through the exact registered type."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    inspector = _OUTPUT_INSPECTORS.get(resolved.suffix.casefold())
    if inspector is None:
        raise ValueError(
            f"UNSUPPORTED_OUTPUT_INSPECTOR:{resolved.suffix.casefold()}"
        )
    return inspector(resolved)


def _run_git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root.resolve()), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tracked_scientific_changes(repo_root: Path) -> list[str]:
    records = _run_git(
        repo_root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    ).splitlines()
    changed = []
    for record in records:
        relative = record[3:].replace("\\", "/")
        if relative.startswith(_SCIENTIFIC_AUTHORITY_PATH_PREFIXES):
            changed.append(relative)
    return sorted(changed)


def _validate_development_contract(
    repo_root: Path,
    development_contract_version: str | None,
) -> list[str]:
    changed = _tracked_scientific_changes(repo_root)
    if not changed:
        return []
    if (
        development_contract_version
        != CANDIDATE_MANIFEST_DEVELOPMENT_CONTRACT_VERSION
    ):
        raise ValueError(f"DIRTY_SCIENTIFIC_WORKTREE:{changed}")
    return changed


def _contains_forbidden_metadata_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in _LOAD_BEARING_FIELDS:
                return str(key)
            if any(token in normalized for token in _EPHEMERAL_METADATA_KEY_TOKENS):
                return str(key)
            nested = _contains_forbidden_metadata_key(child)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for child in value:
            nested = _contains_forbidden_metadata_key(child)
            if nested is not None:
                return nested
    return None


def _canonical_artifact_relative_path(
    output_path: Path,
    manifest_path: Path,
) -> str:
    relative = os.path.relpath(
        output_path.resolve(),
        manifest_path.resolve().parent,
    )
    return Path(relative).as_posix()


def _resolve_manifest_output(
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> Path:
    relative = str(manifest.get("output_path", "")).strip()
    if not relative or Path(relative).is_absolute():
        raise ValueError("INVALID_ARTIFACT_RELATIVE_OUTPUT_PATH")
    return (manifest_path.resolve().parent / Path(relative)).resolve()


def load_validated_upstream_manifest(
    manifest_path: Path,
) -> LoadedUpstreamManifest:
    """Load one upstream candidate while independently verifying its bytes."""

    resolved = manifest_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"UPSTREAM_MANIFEST_MISSING:{resolved}")
    try:
        manifest = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"UPSTREAM_MANIFEST_UNREADABLE:{type(exc).__name__}"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("UPSTREAM_MANIFEST_NOT_OBJECT")
    output_path = _resolve_manifest_output(resolved, manifest)
    result = validate_artifact_manifest(
        manifest,
        output_path=output_path,
    )
    if not result["valid"]:
        raise ValueError(
            f"UPSTREAM_MANIFEST_INVALID:{result['errors']}"
        )
    return LoadedUpstreamManifest(
        manifest_path=resolved,
        output_path=output_path,
        manifest=dict(manifest),
        manifest_file_sha256=file_sha256(resolved),
    )


def _validate_upstreams(
    *,
    stage_id: str,
    artifact_id: str,
    artifact_class: str,
    upstream_manifest_paths: Sequence[Path],
    stage_order: Sequence[str],
) -> list[LoadedUpstreamManifest]:
    loaded = [
        load_validated_upstream_manifest(path)
        for path in upstream_manifest_paths
    ]
    ids = [str(item.manifest["artifact_id"]) for item in loaded]
    if len(ids) != len(set(ids)):
        raise ValueError(f"DUPLICATE_UPSTREAM_ARTIFACT_ID:{ids}")
    if artifact_id in ids:
        raise ValueError(f"SELF_DEPENDENCY:{artifact_id}")
    order = {value: index for index, value in enumerate(stage_order)}
    for item in loaded:
        manifest = item.manifest
        upstream_stage = str(manifest["stage_id"])
        upstream_class = str(manifest["artifact_class"])
        if upstream_stage == stage_id:
            raise ValueError(f"SELF_STAGE_DEPENDENCY:{upstream_stage}")
        if (
            upstream_stage not in order
            or order[upstream_stage] >= order[stage_id]
        ):
            raise ValueError(
                f"DOWNSTREAM_TO_UPSTREAM_CYCLE:{upstream_stage}->{stage_id}"
            )
        if any(
            token in upstream_class.upper()
            for token in _INVALID_UPSTREAM_CLASS_TOKENS
        ):
            raise ValueError(
                f"INVALID_UPSTREAM_ARTIFACT_CLASS:{upstream_class}"
            )
        if artifact_class in _OFFICIAL_ARTIFACT_CLASSES:
            if upstream_class not in _OFFICIAL_ARTIFACT_CLASSES:
                raise ValueError(
                    f"AUDIT_ONLY_UPSTREAM_FOR_OFFICIAL:{upstream_class}"
                )
            if manifest.get("authority_state") != "OFFICIAL_PROMOTED":
                raise ValueError("OFFICIAL_UPSTREAM_NOT_PROMOTED")
    return sorted(loaded, key=lambda item: str(item.manifest["artifact_id"]))


def _stage_contract(
    contract: Mapping[str, Any],
    stage_id: str,
) -> Mapping[str, Any]:
    matches = [
        stage
        for stage in contract.get("stages", [])
        if stage.get("stage_id") == stage_id
    ]
    if len(matches) != 1:
        raise ValueError(f"UNKNOWN_OR_DUPLICATE_STAGE:{stage_id}")
    return matches[0]


def _validate_schema_authority(
    *,
    stage: Mapping[str, Any],
    intended_schema_id: str | None,
    intended_schema_version: str | None,
    inspection: OutputInspection,
    stage_specific_metadata: Mapping[str, Any],
) -> tuple[str, str, str]:
    stage_id = str(stage["stage_id"])
    permitted = [str(value) for value in stage["output_artifacts"]]
    if intended_schema_id is None:
        if len(permitted) != 1:
            raise ValueError(
                f"OUTPUT_SCHEMA_ID_REQUIRED:{stage_id}:{permitted}"
            )
        schema_id = permitted[0]
    else:
        schema_id = intended_schema_id
    if schema_id not in permitted:
        raise ValueError(
            f"OUTPUT_SCHEMA_NOT_PERMITTED:{stage_id}:{schema_id}"
        )
    schema_version = str(stage["schema_version"])
    if (
        intended_schema_version is not None
        and intended_schema_version != schema_version
    ):
        raise ValueError(
            "OUTPUT_SCHEMA_VERSION_MISMATCH:"
            f"{intended_schema_version}!={schema_version}"
        )
    columns = list(inspection.ordered_columns)
    produced = [
        str(value)
        for value in stage.get("produced_columns", [])
        if isinstance(value, str)
        and value
        and " " not in value
        and "*" not in value
    ]
    missing_produced = sorted(set(produced) - set(columns))
    if missing_produced:
        raise ValueError(
            f"MISSING_CONTRACT_OUTPUT_COLUMNS:{missing_produced}"
        )
    if (
        stage_id in _OBJECT_TRACK_KEY_STAGES
        and inspection.inspector_id == "inspector.csv.v1"
        and "object_track_key" not in columns
    ):
        raise ValueError("MISSING_OBJECT_TRACK_KEY")
    metadata_features = stage_specific_metadata.get(
        "model_feature_names",
        [],
    )
    model_features = sorted(
        set(inspection.model_feature_names)
        | {
            str(value)
            for value in metadata_features
            if isinstance(value, str)
        }
    )
    if stage_id in _MODEL_INPUT_STAGES and model_features:
        leakage = validate_model_input_columns(model_features)
        if not leakage["valid"]:
            raise ValueError(
                f"FORBIDDEN_MODEL_INPUT_COLUMNS:{leakage['forbidden_columns']}"
            )
    if stage_id in _MOTION_SCHEMA_STAGES:
        declared_motion = stage_specific_metadata.get(
            "motion_feature_names",
        )
        if declared_motion is None:
            relative = [
                value for value in columns if value in MOTION_FEATURE_NAMES
            ]
            declared_motion = relative
        if list(declared_motion) != list(MOTION_FEATURE_NAMES):
            raise ValueError("MOTION_SCHEMA_FEATURE_ORDER_MISMATCH")
    schema_payload = {
        "stage_id": stage_id,
        "schema_id": schema_id,
        "schema_version": schema_version,
        "output_grain": stage["output_grain"],
        "canonical_identity_keys": stage.get(
            "canonical_identity_keys",
            [],
        ),
        "inspection": inspection.schema_payload,
        "model_feature_names": model_features,
        "motion_feature_names": (
            list(MOTION_FEATURE_NAMES)
            if stage_id in _MOTION_SCHEMA_STAGES
            else []
        ),
    }
    return schema_id, schema_version, canonical_sha256(schema_payload)


def _builder_code_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def candidate_manifest_builder_contract() -> dict[str, Any]:
    """Return the machine-readable production builder contract."""

    return {
        "manifest_builder_id": MANIFEST_BUILDER_ID,
        "manifest_builder_version": MANIFEST_BUILDER_VERSION,
        "artifact_manifest_version": ARTIFACT_MANIFEST_VERSION,
        "development_contract_version": (
            CANDIDATE_MANIFEST_DEVELOPMENT_CONTRACT_VERSION
        ),
        "load_bearing_caller_hashes_trusted": False,
        "candidate_authority_state": CANDIDATE_AUTHORITY_STATE,
        "output_inspector_registry": output_inspector_registry(),
        "atomic_write_sequence": [
            "inspect_output",
            "validate_upstreams",
            "derive_code_semantic_schema_authority",
            "validate_in_memory",
            "write_fsync_temporary",
            "atomic_replace",
            "reread_revalidate",
        ],
    }


def _assert_expected_authority(
    manifest: Mapping[str, Any],
    expected_authority: Mapping[str, Any] | None,
) -> None:
    for field, expected in (expected_authority or {}).items():
        if field not in manifest:
            raise ValueError(f"UNKNOWN_EXPECTED_AUTHORITY_FIELD:{field}")
        if manifest[field] != expected:
            raise ValueError(
                f"EXPECTED_AUTHORITY_MISMATCH:{field}:"
                f"{expected!r}!={manifest[field]!r}"
            )


def _write_candidate_atomically(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    before_atomic_replace: Callable[[], None] | None,
) -> None:
    destination = manifest_path.resolve()
    if destination.exists():
        raise FileExistsError(
            f"candidate manifest already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".candidate-staging",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(manifest) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if before_atomic_replace is not None:
            before_atomic_replace()
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def build_candidate_artifact_manifest(
    *,
    repo_root: Path,
    contract_root: Path,
    stage_id: str,
    artifact_id: str,
    artifact_class: str,
    output_path: Path,
    candidate_manifest_path: Path,
    upstream_manifest_paths: Sequence[Path] = (),
    output_schema_id: str | None = None,
    output_schema_version: str | None = None,
    stage_specific_metadata: Mapping[str, Any] | None = None,
    expected_authority: Mapping[str, Any] | None = None,
    development_contract_version: str | None = None,
    before_atomic_replace: Callable[[], None] | None = None,
) -> CandidateManifestBuild:
    """Derive, write, reread, and validate one candidate manifest."""

    repo_root = repo_root.resolve()
    contract_root = contract_root.resolve()
    output_path = output_path.resolve()
    candidate_manifest_path = candidate_manifest_path.resolve()
    if not artifact_id.strip():
        raise ValueError("BLANK_ARTIFACT_ID")
    if artifact_class not in _SUPPORTED_ARTIFACT_CLASSES:
        raise ValueError(f"UNSUPPORTED_ARTIFACT_CLASS:{artifact_class}")
    metadata = dict(stage_specific_metadata or {})
    forbidden_metadata = _contains_forbidden_metadata_key(metadata)
    if forbidden_metadata is not None:
        raise ValueError(
            f"LOAD_BEARING_OR_EPHEMERAL_METADATA:{forbidden_metadata}"
        )
    dirty_files = _validate_development_contract(
        repo_root,
        development_contract_version,
    )
    contract_path = contract_root / "00_pipeline_contract.yaml"
    mapping_path = contract_root / "10_code_contract_mapping.csv"
    contract_manifest_path = contract_root / "contract_manifest.json"
    for authority_path in (
        contract_path,
        mapping_path,
        contract_manifest_path,
    ):
        if not authority_path.is_file():
            raise FileNotFoundError(authority_path)
    contract = load_scientific_contract(contract_path)
    stage = _stage_contract(contract, stage_id)
    stage_order = [
        str(value["stage_id"]) for value in contract["stages"]
    ]
    upstreams = _validate_upstreams(
        stage_id=stage_id,
        artifact_id=artifact_id,
        artifact_class=artifact_class,
        upstream_manifest_paths=upstream_manifest_paths,
        stage_order=stage_order,
    )
    inspection = inspect_candidate_output(output_path)
    schema_id, schema_version, schema_hash = _validate_schema_authority(
        stage=stage,
        intended_schema_id=output_schema_id,
        intended_schema_version=output_schema_version,
        inspection=inspection,
        stage_specific_metadata=metadata,
    )
    mapping_rows = load_code_contract_mapping(mapping_path)
    runtime_audit = assert_stage_runtime_dependencies_complete(
        repo_root,
        stage_id,
        mapping_rows,
    )
    if (
        runtime_audit["runtime_dependency_closure"]
        != runtime_audit["hashed_code_files"]
    ):
        raise ValueError(
            "STAGE_RUNTIME_AUTHORITY_NOT_EXACT:"
            f"{stage_id}:closure={runtime_audit['runtime_dependency_closure']}:"
            f"hashed={runtime_audit['hashed_code_files']}"
        )
    semantic_registry = build_semantic_domain_registry(contract)
    semantic_bundle = build_semantic_bundle(semantic_registry)
    stage_code_hash = compute_stage_code_hash(
        repo_root,
        stage_id,
        mapping_rows,
    )
    stage_semantics_hash = compute_stage_semantics_hash(
        stage_id,
        semantic_registry,
    )
    input_fingerprints = {
        str(item.manifest["artifact_id"]): str(
            item.manifest["stage_execution_fingerprint"]
        )
        for item in upstreams
    }
    input_manifest_hashes = {
        str(item.manifest["artifact_id"]): item.manifest_file_sha256
        for item in upstreams
    }
    input_artifact_ids = sorted(input_fingerprints)
    stage_input_fingerprint = compute_stage_input_fingerprint(
        input_fingerprints
    )
    execution_parameters_hash = canonical_sha256(metadata)
    execution_fingerprint = compute_stage_execution_fingerprint(
        stage_id=stage_id,
        stage_version=str(stage["schema_version"]),
        stage_code_hash=stage_code_hash,
        stage_semantics_hash=stage_semantics_hash,
        stage_input_fingerprint=stage_input_fingerprint,
        schema_hashes={
            "execution_parameters_hash": execution_parameters_hash,
            "output_schema_hash": schema_hash,
        },
    )
    authority = git_code_authority(repo_root)
    warnings = (
        [f"DEVELOPMENT_CONTRACT_DIRTY_FILES:{dirty_files}"]
        if dirty_files
        else []
    )
    manifest = {
        "artifact_manifest_version": ARTIFACT_MANIFEST_VERSION,
        "manifest_builder_id": MANIFEST_BUILDER_ID,
        "manifest_builder_version": MANIFEST_BUILDER_VERSION,
        "manifest_builder_code_hash": _builder_code_hash(),
        "authority_state": CANDIDATE_AUTHORITY_STATE,
        "artifact_id": artifact_id,
        "artifact_class": artifact_class,
        "stage_id": stage_id,
        "stage_version": str(stage["schema_version"]),
        **authority,
        "stage_code_hash": stage_code_hash,
        "stage_semantics_hash": stage_semantics_hash,
        "stage_input_fingerprint": stage_input_fingerprint,
        "stage_execution_fingerprint": execution_fingerprint,
        "execution_parameters_hash": execution_parameters_hash,
        "semantic_bundle_hash": semantic_bundle["semantic_bundle_hash"],
        "contract_manifest_hash": file_sha256(contract_manifest_path),
        "input_artifact_ids": input_artifact_ids,
        "input_artifact_fingerprints": input_fingerprints,
        "input_file_sha256": canonical_sha256(
            {"upstream_manifest_file_hashes": input_manifest_hashes}
        ),
        "output_path": _canonical_artifact_relative_path(
            output_path,
            candidate_manifest_path,
        ),
        "output_file_sha256": inspection.output_file_sha256,
        "output_byte_size": inspection.output_byte_size,
        "output_inspector_id": inspection.inspector_id,
        "output_inspector_version": inspection.inspector_version,
        "output_schema_id": schema_id,
        "output_schema_version": schema_version,
        "output_schema_hash": schema_hash,
        "row_count": inspection.row_count,
        "column_count": inspection.column_count,
        "ordered_columns": list(inspection.ordered_columns),
        "stage_specific_metadata": metadata,
        "feature_computation_grain": str(stage["output_grain"]),
        "pair_scope_key": str(
            stage.get("pair_reset_key", stage["output_grain"])
        ),
        "distance_metric_ids": [
            AXIS_DISTANCE_METRIC_ID,
            DIAGONAL_DISTANCE_METRIC_ID,
        ],
        "distance_metric_versions": [
            AXIS_DISTANCE_METRIC_VERSION,
            DIAGONAL_DISTANCE_METRIC_VERSION,
        ],
        "social_identity_version": SOCIAL_IDENTITY_VERSION,
        "social_tie_break_version": SOCIAL_TIE_BREAK_VERSION,
        "roi_aggregation_version": ROI_AGGREGATION_VERSION,
        "motion_schema_id": MOTION_SCHEMA_ID,
        "motion_schema_version": MOTION_SCHEMA_VERSION,
        "motion_schema_hash": MOTION_SCHEMA_HASH,
        "human_decision_authority": (
            "SYNTHETIC_TEST_ONLY"
            if artifact_class == "SYNTHETIC_INTEGRATION_TEST_ONLY"
            else "NONE_OR_SEPARATELY_VALIDATED"
        ),
        "review_key_schema_version": (
            "classification_v2.review_key.v1"
        ),
        "status": "VALIDATED",
        "validation_errors": [],
        "validation_warnings": warnings,
    }
    _assert_expected_authority(manifest, expected_authority)
    upstream_map = {
        str(item.manifest["artifact_id"]): item.manifest
        for item in upstreams
    }
    validation = validate_artifact_manifest(
        manifest,
        output_path=output_path,
        upstream_manifests=upstream_map,
        expected_stage_execution_fingerprint=execution_fingerprint,
        expected_schema=(schema_id, schema_version, schema_hash),
    )
    if not validation["valid"]:
        raise ValueError(
            f"CANDIDATE_MANIFEST_INVALID:{validation['errors']}"
        )
    _write_candidate_atomically(
        manifest_path=candidate_manifest_path,
        manifest=manifest,
        before_atomic_replace=before_atomic_replace,
    )
    reread = load_validated_upstream_manifest(candidate_manifest_path)
    if dict(reread.manifest) != manifest:
        candidate_manifest_path.unlink(missing_ok=True)
        raise ValueError("CANDIDATE_MANIFEST_REREAD_MISMATCH")
    return CandidateManifestBuild(
        manifest=manifest,
        manifest_path=candidate_manifest_path,
        output_inspection=inspection,
        validation=validation,
        production_builder_owned=True,
    )


__all__ = [
    "CANDIDATE_MANIFEST_DEVELOPMENT_CONTRACT_VERSION",
    "CandidateManifestBuild",
    "LoadedUpstreamManifest",
    "OUTPUT_INSPECTOR_REGISTRY_ID",
    "OUTPUT_INSPECTOR_REGISTRY_VERSION",
    "OutputInspection",
    "build_candidate_artifact_manifest",
    "candidate_manifest_builder_contract",
    "inspect_candidate_output",
    "load_validated_upstream_manifest",
    "output_inspector_registry",
]
