"""Production-owned candidate artifact manifest construction."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from csv import DictReader
from dataclasses import dataclass
from io import StringIO
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
    CANDIDATE_TRANSACTION_STATE_COMMITTED,
    CANDIDATE_TRANSACTION_STATE_PENDING,
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
    candidate_transaction_provenance_hash,
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
    stage_code_files,
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
UPSTREAM_CURRENT_AUTHORITY_CONTRACT_VERSION = (
    "classification_v2.upstream_current_authority.v1"
)
CANDIDATE_TRANSACTION_CONTRACT_VERSION = (
    "classification_v2.candidate_transaction.v1"
)
VALID_HISTORICAL = "VALID_HISTORICAL"
CURRENT_AUTHORITATIVE = "CURRENT_AUTHORITATIVE"
STALE_CODE_AUTHORITY = "STALE_CODE_AUTHORITY"
STALE_SEMANTIC_AUTHORITY = "STALE_SEMANTIC_AUTHORITY"
STALE_SCHEMA_AUTHORITY = "STALE_SCHEMA_AUTHORITY"
INVALID_UPSTREAM_INTEGRITY = "INVALID_UPSTREAM_INTEGRITY"
INELIGIBLE_ARTIFACT_CLASS = "INELIGIBLE_ARTIFACT_CLASS"

_SCIENTIFIC_AUTHORITY_PATH_PREFIXES = (
    "src/",
    "scripts/classification_v2/",
    "docs/classification_v2/scientific_contract_v1/",
)
_PUBLICATION_ONLY_STAGE_CODE_PATHS = frozenset(
    {
        "scripts/classification_v2/run_lineage_stage.py",
        "src/pig_behavior/classification_v2/contracts/candidate_manifest.py",
        "src/pig_behavior/classification_v2/contracts/semantic_lineage.py",
    }
)
_REGISTRY_ONLY_STAGE_CODE_PATHS = frozenset(
    {
        "src/pig_behavior/classification_v2/contracts/semantic_lineage.py",
    }
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
    authority_classification: str = VALID_HISTORICAL
    authority_reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class UpstreamAuthorityValidation:
    """Historical-integrity and current-authority result for one upstream."""

    classification: str
    reason_codes: tuple[str, ...]
    historical_integrity_valid: bool
    current_authoritative: bool
    loaded_manifest: LoadedUpstreamManifest | None
    expected_authority: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CandidateTransactionRecord:
    """Auditable state of one candidate-manifest filesystem transaction."""

    transaction_id: str
    initial_final_path_state: str
    prior_manifest_sha256: str | None
    temporary_path: str | None
    backup_path: str | None
    last_completed_state: str
    rollback_attempted: bool
    rollback_succeeded: bool
    final_path_exists: bool
    final_path_sha256: str | None
    new_candidate_survived: bool
    official_promotion_occurred: bool
    classification: str


class CandidateManifestTransactionError(RuntimeError):
    """Raised when candidate writing or rollback does not commit cleanly."""

    def __init__(
        self,
        message: str,
        *,
        transaction: CandidateTransactionRecord,
    ) -> None:
        super().__init__(message)
        self.transaction = transaction


@dataclass(frozen=True, slots=True)
class CandidateManifestBuild:
    """Result returned only after atomic write and independent revalidation."""

    manifest: Mapping[str, Any]
    manifest_path: Path
    output_inspection: OutputInspection
    validation: Mapping[str, Any]
    production_builder_owned: bool
    transaction: CandidateTransactionRecord


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
    chunks = pd.read_csv(path, chunksize=1_000, low_memory=False)
    rows = 0
    columns: list[str] | None = None
    dtype_names: list[str | None] | None = None
    fallback_dtype_names: list[str] | None = None
    for frame in chunks:
        observed_columns = [str(value) for value in frame.columns]
        observed_dtypes = [str(value) for value in frame.dtypes]
        observed_non_null = [
            bool(frame[column].notna().any()) for column in frame.columns
        ]
        if columns is None:
            columns = observed_columns
            dtype_names = [
                dtype_name if has_value else None
                for dtype_name, has_value in zip(
                    observed_dtypes,
                    observed_non_null,
                    strict=True,
                )
            ]
            fallback_dtype_names = observed_dtypes
        elif observed_columns != columns:
            raise ValueError("CSV_COLUMN_ORDER_CHANGED_BETWEEN_CHUNKS")
        else:
            assert dtype_names is not None
            dtype_names = [
                (
                    current
                    if not has_value
                    else (
                        observed
                        if current is None
                        else _merge_csv_dtype_names(current, observed)
                    )
                )
                for current, observed, has_value in zip(
                    dtype_names,
                    observed_dtypes,
                    observed_non_null,
                    strict=True,
                )
            ]
        rows += len(frame)
    if columns is None:
        empty = pd.read_csv(path, nrows=0)
        columns = [str(value) for value in empty.columns]
        dtype_names = [str(value) for value in empty.dtypes]
        fallback_dtype_names = list(dtype_names)
    assert dtype_names is not None
    assert fallback_dtype_names is not None
    resolved_dtype_names = [
        current if current is not None else fallback
        for current, fallback in zip(
            dtype_names,
            fallback_dtype_names,
            strict=True,
        )
    ]
    return _inspection(
        path,
        inspector_id="inspector.csv.v1",
        rows=rows,
        columns=columns,
        schema_payload={
            "format": "csv",
            "ordered_columns": columns,
            "dtypes": resolved_dtype_names,
        },
    )


def _merge_csv_dtype_names(current: str, observed: str) -> str:
    """Match whole-file CSV inference without retaining every parsed row."""

    if current == observed:
        return current
    if "str" in {current, observed}:
        return "str"
    numeric = {"int64", "float64"}
    if {current, observed}.issubset(numeric):
        return "float64"
    return "object"


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


def _mapping_symbols_by_path(
    stage_id: str,
    mapping_rows: Sequence[Mapping[str, str]],
) -> dict[str, set[str]]:
    symbols_by_path: dict[str, set[str]] = {}
    for row in mapping_rows:
        if str(row.get("contract_item_id", "")).strip() != stage_id:
            continue
        path = str(row.get("source_file", "")).strip().replace("\\", "/")
        symbol = str(row.get("symbol", "")).strip()
        if path and symbol:
            symbols_by_path.setdefault(path, set()).add(symbol)
    return symbols_by_path


def _python_symbol_fingerprints(
    source: bytes,
    symbols: Sequence[str],
) -> dict[str, str]:
    tree = ast.parse(source.decode("utf-8-sig"))
    fingerprints: dict[str, str] = {}
    for symbol in symbols:
        body = tree.body
        node: ast.AST | None = None
        for component in symbol.split("."):
            node = next(
                (
                    candidate
                    for candidate in body
                    if getattr(candidate, "name", None) == component
                ),
                None,
            )
            if node is None:
                raise ValueError(f"MAPPED_SYMBOL_NOT_FOUND:{symbol}")
            body = list(getattr(node, "body", ()))
        canonical = ast.dump(
            node,
            annotate_fields=True,
            include_attributes=False,
        )
        fingerprints[symbol] = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
    return fingerprints


def _mapped_stage_symbols_unchanged(
    *,
    stage_id: str,
    changed_paths: set[str],
    historical_mapping_rows: Sequence[Mapping[str, str]],
    current_mapping_rows: Sequence[Mapping[str, str]],
    historical_sources: Mapping[str, bytes],
    repo_root: Path,
) -> bool:
    historical_symbols = _mapping_symbols_by_path(
        stage_id,
        historical_mapping_rows,
    )
    current_symbols = _mapping_symbols_by_path(stage_id, current_mapping_rows)
    for relative in sorted(changed_paths):
        symbols = sorted(
            historical_symbols.get(relative, set())
            | current_symbols.get(relative, set())
        )
        if not symbols:
            if relative not in _REGISTRY_ONLY_STAGE_CODE_PATHS:
                return False
            continue
        historical = historical_sources.get(relative)
        current_path = repo_root / relative
        if historical is None or not current_path.is_file():
            return False
        try:
            historical_fingerprints = _python_symbol_fingerprints(
                historical,
                symbols,
            )
            current_fingerprints = _python_symbol_fingerprints(
                current_path.read_bytes(),
                symbols,
            )
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
            return False
        if historical_fingerprints != current_fingerprints:
            return False
    return True


def _publication_only_stage_code_drift(
    *,
    repo_root: Path,
    stage_id: str,
    mapping_rows: Sequence[Mapping[str, str]],
    manifest: Mapping[str, Any],
) -> bool:
    """Recognize scoped infrastructure drift without accepting runtime drift."""

    code_authority_sha = str(
        manifest.get("created_by_code_authority_sha", "")
    ).strip()
    recorded_hash = str(manifest.get("stage_code_hash", "")).strip()
    if not code_authority_sha or not recorded_hash:
        return False
    historical_sources: dict[str, bytes] = {}
    changed_paths: set[str] = set()
    try:
        mapping_relative = (
            "docs/classification_v2/scientific_contract_v1/"
            "10_code_contract_mapping.csv"
        )
        historical_mapping_bytes = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root.resolve()),
                "cat-file",
                "--filters",
                f"--path={mapping_relative}",
                f"{code_authority_sha}:{mapping_relative}",
            ],
            check=True,
            capture_output=True,
        ).stdout
        historical_mapping_rows = [
            dict(row)
            for row in DictReader(
                StringIO(historical_mapping_bytes.decode("utf-8-sig"))
            )
        ]
        historical_files = set(
            stage_code_files(stage_id, historical_mapping_rows)
        )
        current_files = set(stage_code_files(stage_id, mapping_rows))
        committed_changes = set(
            _run_git(
                repo_root,
                "diff",
                "--name-only",
                f"{code_authority_sha}..HEAD",
                "--",
                *sorted(historical_files | current_files),
            ).splitlines()
        )
        worktree_changes = set(
            _run_git(
                repo_root,
                "diff",
                "--name-only",
                "--",
                *sorted(historical_files | current_files),
            ).splitlines()
        )
        staged_changes = set(
            _run_git(
                repo_root,
                "diff",
                "--cached",
                "--name-only",
                "--",
                *sorted(historical_files | current_files),
            ).splitlines()
        )
        changed_paths.update(
            committed_changes | worktree_changes | staged_changes
        )
        for relative in sorted(historical_files | current_files):
            if relative not in historical_files:
                changed_paths.add(relative)
                continue
            historical = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo_root.resolve()),
                    "cat-file",
                    "--filters",
                    f"--path={relative}",
                    f"{code_authority_sha}:{relative}",
                ],
                check=True,
                capture_output=True,
            ).stdout
            historical_sources[relative] = historical
            current_path = repo_root / relative
            if relative not in current_files or not current_path.is_file():
                changed_paths.add(relative)
        historical_hash = compute_stage_code_hash(
            repo_root,
            stage_id,
            historical_mapping_rows,
            file_overrides=historical_sources,
        )
    except (OSError, subprocess.CalledProcessError, ValueError):
        return False
    recorded_hash_valid = (
        len(recorded_hash) == 64
        and set(recorded_hash) <= set("0123456789abcdef")
        and recorded_hash != "0" * 64
    )
    validated_transaction = (
        manifest.get("candidate_transaction_state")
        == CANDIDATE_TRANSACTION_STATE_COMMITTED
        and str(manifest.get("manifest_builder_code_hash", "")).strip() != ""
    )
    recorded_authority_valid = (
        historical_hash == recorded_hash
        or (recorded_hash_valid and validated_transaction)
    )
    return (
        recorded_authority_valid
        and bool(changed_paths)
        and changed_paths <= _PUBLICATION_ONLY_STAGE_CODE_PATHS
        and _mapped_stage_symbols_unchanged(
            stage_id=stage_id,
            changed_paths=changed_paths,
            historical_mapping_rows=historical_mapping_rows,
            current_mapping_rows=mapping_rows,
            historical_sources=historical_sources,
            repo_root=repo_root,
        )
    )


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


def _current_semantic_identifiers() -> dict[str, Any]:
    return {
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
        "review_key_schema_version": "classification_v2.review_key.v1",
    }


def validate_upstream_manifest_for_current_authority(
    *,
    manifest_path: Path,
    repo_root: Path,
    contract_root: Path,
    intended_downstream_stage_id: str,
) -> UpstreamAuthorityValidation:
    """Classify historical integrity separately from current eligibility."""

    try:
        loaded = load_validated_upstream_manifest(manifest_path)
        inspection = inspect_candidate_output(loaded.output_path)
    except Exception as exc:
        return UpstreamAuthorityValidation(
            classification=INVALID_UPSTREAM_INTEGRITY,
            reason_codes=(
                f"INVALID_UPSTREAM_INTEGRITY:{type(exc).__name__}:{exc}",
            ),
            historical_integrity_valid=False,
            current_authoritative=False,
            loaded_manifest=None,
            expected_authority={},
        )

    manifest = loaded.manifest
    integrity_mismatches = [
        name
        for name, actual, expected in (
            (
                "output_file_sha256",
                manifest.get("output_file_sha256"),
                inspection.output_file_sha256,
            ),
            (
                "output_byte_size",
                manifest.get("output_byte_size"),
                inspection.output_byte_size,
            ),
            ("row_count", manifest.get("row_count"), inspection.row_count),
            (
                "column_count",
                manifest.get("column_count"),
                inspection.column_count,
            ),
            (
                "ordered_columns",
                manifest.get("ordered_columns"),
                list(inspection.ordered_columns),
            ),
        )
        if actual != expected
    ]
    if integrity_mismatches:
        return UpstreamAuthorityValidation(
            classification=INVALID_UPSTREAM_INTEGRITY,
            reason_codes=tuple(
                f"HISTORICAL_METADATA_MISMATCH:{name}"
                for name in integrity_mismatches
            ),
            historical_integrity_valid=False,
            current_authoritative=False,
            loaded_manifest=loaded,
            expected_authority={},
        )

    artifact_class = str(manifest.get("artifact_class", ""))
    artifact_status = str(manifest.get("status", ""))
    authority_state = str(manifest.get("authority_state", ""))
    eligible_classes = {
        "SYNTHETIC_INTEGRATION_TEST_ONLY",
        "SCIENTIFIC_CANDIDATE",
        *_OFFICIAL_ARTIFACT_CLASSES,
    }
    ineligible = (
        artifact_class not in eligible_classes
        or any(
            token in artifact_class.upper()
            for token in _INVALID_UPSTREAM_CLASS_TOKENS
        )
        or artifact_status != "VALIDATED"
        or authority_state
        not in {CANDIDATE_AUTHORITY_STATE, "OFFICIAL_PROMOTED"}
    )
    if ineligible:
        return UpstreamAuthorityValidation(
            classification=INELIGIBLE_ARTIFACT_CLASS,
            reason_codes=(
                "INELIGIBLE_ARTIFACT_CLASS:"
                f"{artifact_class}:{authority_state}:{artifact_status}",
            ),
            historical_integrity_valid=True,
            current_authoritative=False,
            loaded_manifest=loaded,
            expected_authority={},
        )

    contract_path = contract_root.resolve() / "00_pipeline_contract.yaml"
    mapping_path = contract_root.resolve() / "10_code_contract_mapping.csv"
    try:
        contract = load_scientific_contract(contract_path)
        _validate_manifest_authority_contract(contract)
        _stage_contract(contract, intended_downstream_stage_id)
        stage_id = str(manifest.get("stage_id", ""))
        stage = _stage_contract(contract, stage_id)
        mapping_rows = load_code_contract_mapping(mapping_path)
        runtime_audit = assert_stage_runtime_dependencies_complete(
            repo_root.resolve(),
            stage_id,
            mapping_rows,
        )
        if (
            runtime_audit["runtime_dependency_closure"]
            != runtime_audit["hashed_code_files"]
        ):
            raise ValueError("STAGE_RUNTIME_AUTHORITY_NOT_EXACT")
        semantic_registry = build_semantic_domain_registry(contract)
        current_code_hash = compute_stage_code_hash(
            repo_root.resolve(),
            stage_id,
            mapping_rows,
        )
        current_semantics_hash = compute_stage_semantics_hash(
            stage_id,
            semantic_registry,
        )
        schema_id, schema_version, schema_hash = _validate_schema_authority(
            stage=stage,
            intended_schema_id=str(manifest.get("output_schema_id", "")),
            intended_schema_version=str(
                manifest.get("output_schema_version", "")
            ),
            inspection=inspection,
            stage_specific_metadata=dict(
                manifest.get("stage_specific_metadata", {})
            ),
        )
        bundle = validate_candidate_output_bundle(
            manifest,
            manifest_path=loaded.manifest_path,
            stage=stage,
        )
        if not bundle["valid"]:
            raise ValueError(f"OUTPUT_ARTIFACT_BUNDLE_INVALID:{bundle['errors']}")
    except Exception as exc:
        return UpstreamAuthorityValidation(
            classification=STALE_SCHEMA_AUTHORITY,
            reason_codes=(
                f"CURRENT_STAGE_OR_SCHEMA_UNRECOGNIZED:{type(exc).__name__}:"
                f"{exc}",
            ),
            historical_integrity_valid=True,
            current_authoritative=False,
            loaded_manifest=loaded,
            expected_authority={},
        )

    expected_authority = {
        "stage_id": stage_id,
        "stage_version": str(stage["schema_version"]),
        "stage_code_hash": current_code_hash,
        "stage_semantics_hash": current_semantics_hash,
        "output_schema_id": schema_id,
        "output_schema_version": schema_version,
        "output_schema_hash": schema_hash,
        **_current_semantic_identifiers(),
    }
    code_hash_drift = manifest.get("stage_code_hash") != current_code_hash
    publication_only_drift = code_hash_drift and (
        _publication_only_stage_code_drift(
            repo_root=repo_root.resolve(),
            stage_id=stage_id,
            mapping_rows=mapping_rows,
            manifest=manifest,
        )
    )
    code_reasons = (
        ("STAGE_CODE_HASH_NOT_CURRENT",)
        if code_hash_drift and not publication_only_drift
        else ()
    )
    semantic_reasons = tuple(
        reason
        for name, expected in {
            "stage_semantics_hash": current_semantics_hash,
            **_current_semantic_identifiers(),
        }.items()
        if manifest.get(name) != expected
        for reason in (f"SEMANTIC_AUTHORITY_NOT_CURRENT:{name}",)
    )
    schema_reasons = tuple(
        reason
        for name, expected in {
            "stage_id": stage_id,
            "stage_version": str(stage["schema_version"]),
            "output_schema_id": schema_id,
            "output_schema_version": schema_version,
            "output_schema_hash": schema_hash,
        }.items()
        if manifest.get(name) != expected
        for reason in (f"SCHEMA_AUTHORITY_NOT_CURRENT:{name}",)
    )
    if code_reasons:
        classification = STALE_CODE_AUTHORITY
        reasons = code_reasons
    elif semantic_reasons:
        classification = STALE_SEMANTIC_AUTHORITY
        reasons = semantic_reasons
    elif schema_reasons:
        classification = STALE_SCHEMA_AUTHORITY
        reasons = schema_reasons
    else:
        classification = CURRENT_AUTHORITATIVE
        reasons = (
            ("CURRENT_AUTHORITY_MATCH_PUBLICATION_ONLY_DRIFT",)
            if publication_only_drift
            else ("CURRENT_AUTHORITY_MATCH",)
        )
    current = classification == CURRENT_AUTHORITATIVE
    classified_loaded = LoadedUpstreamManifest(
        manifest_path=loaded.manifest_path,
        output_path=loaded.output_path,
        manifest=loaded.manifest,
        manifest_file_sha256=loaded.manifest_file_sha256,
        authority_classification=classification,
        authority_reason_codes=reasons,
    )
    return UpstreamAuthorityValidation(
        classification=classification,
        reason_codes=reasons,
        historical_integrity_valid=True,
        current_authoritative=current,
        loaded_manifest=classified_loaded,
        expected_authority=expected_authority,
    )


def _validate_upstreams(
    *,
    repo_root: Path,
    contract_root: Path,
    stage_id: str,
    artifact_id: str,
    artifact_class: str,
    upstream_manifest_paths: Sequence[Path],
    stage_order: Sequence[str],
) -> list[LoadedUpstreamManifest]:
    authority_results = [
        validate_upstream_manifest_for_current_authority(
            manifest_path=path,
            repo_root=repo_root,
            contract_root=contract_root,
            intended_downstream_stage_id=stage_id,
        )
        for path in upstream_manifest_paths
    ]
    noncurrent = [
        result
        for result in authority_results
        if not result.current_authoritative
    ]
    if noncurrent:
        detail = "|".join(
            f"{result.classification}:{','.join(result.reason_codes)}"
            for result in noncurrent
        )
        raise ValueError(f"UPSTREAM_NOT_CURRENT_AUTHORITATIVE:{detail}")
    loaded = [
        result.loaded_manifest
        for result in authority_results
        if result.loaded_manifest is not None
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


def _validate_manifest_authority_contract(
    contract: Mapping[str, Any],
) -> None:
    metadata = contract.get("contract_metadata", {})
    upstream = metadata.get("upstream_current_authority", {})
    transaction = metadata.get("candidate_transaction", {})
    if upstream.get("contract_version") != (
        UPSTREAM_CURRENT_AUTHORITY_CONTRACT_VERSION
    ):
        raise ValueError("UPSTREAM_CURRENT_AUTHORITY_CONTRACT_MISMATCH")
    if upstream.get("required_classification") != CURRENT_AUTHORITATIVE:
        raise ValueError("UPSTREAM_REQUIRED_CLASSIFICATION_MISMATCH")
    if upstream.get("compatibility_exceptions") != []:
        raise ValueError("UNSUPPORTED_UPSTREAM_COMPATIBILITY_EXCEPTION")
    if transaction.get("contract_version") != (
        CANDIDATE_TRANSACTION_CONTRACT_VERSION
    ):
        raise ValueError("CANDIDATE_TRANSACTION_CONTRACT_MISMATCH")
    if transaction.get("committed_state") != (
        CANDIDATE_TRANSACTION_STATE_COMMITTED
    ):
        raise ValueError("CANDIDATE_COMMITTED_STATE_MISMATCH")


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
    registry = stage.get("output_schema_registry", {})
    registry_entry = (
        registry.get(schema_id, {})
        if isinstance(registry, Mapping)
        else {}
    )
    schema_version = str(
        registry_entry.get("schema_version", stage["schema_version"])
    )
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
    missing_registry_columns = sorted(
        set(registry_entry.get("required_columns", [])) - set(columns)
    )
    if missing_registry_columns:
        raise ValueError(
            "MISSING_OUTPUT_SCHEMA_REGISTRY_COLUMNS:"
            f"{schema_id}:{missing_registry_columns}"
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
    if registry_entry:
        schema_payload["schema_registry_entry"] = dict(registry_entry)
    return schema_id, schema_version, canonical_sha256(schema_payload)


def _output_schema_hash(
    *,
    stage: Mapping[str, Any],
    schema_id: str,
    schema_version: str,
    inspection: OutputInspection,
    schema_registry_entry: Mapping[str, Any] | None = None,
) -> str:
    """Hash one declared output schema against its observed bytes."""

    registry = stage.get("output_schema_registry", {})
    resolved_registry_entry = dict(schema_registry_entry or {})
    if not resolved_registry_entry and isinstance(registry, Mapping):
        resolved_registry_entry = dict(registry.get(schema_id, {}))
    expected_version = str(
        resolved_registry_entry.get("schema_version", schema_version)
    )
    if schema_version != expected_version:
        raise ValueError(
            "OUTPUT_SCHEMA_VERSION_MISMATCH:"
            f"{schema_version}!={expected_version}"
        )
    missing_columns = sorted(
        set(resolved_registry_entry.get("required_columns", []))
        - set(inspection.ordered_columns)
    )
    if missing_columns:
        raise ValueError(
            "MISSING_OUTPUT_SCHEMA_REGISTRY_COLUMNS:"
            f"{schema_id}:{missing_columns}"
        )
    return canonical_sha256(
        {
            "stage_id": str(stage["stage_id"]),
            "schema_id": schema_id,
            "schema_version": schema_version,
            "output_grain": stage["output_grain"],
            "canonical_identity_keys": stage.get(
                "canonical_identity_keys",
                [],
            ),
            "schema_registry_entry": resolved_registry_entry,
            "inspection": inspection.schema_payload,
        }
    )


def validate_candidate_output_bundle(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate every committed output declared by a candidate manifest."""

    declared = manifest.get("output_artifacts")
    declared_stage_outputs = [
        str(value) for value in stage.get("output_artifacts", [])
    ]
    if not declared_stage_outputs:
        output_schemas = stage.get("output_schemas", [])
        if isinstance(output_schemas, list):
            declared_stage_outputs = [
                str(value.get("artifact_id", ""))
                for value in output_schemas
                if isinstance(value, Mapping)
            ]
    primary_artifact_id = str(manifest.get("artifact_id", "")).strip()
    expected_ids = (
        [primary_artifact_id, *declared_stage_outputs[1:]]
        if primary_artifact_id
        else declared_stage_outputs
    )
    if len(expected_ids) > 1 and not isinstance(declared, list):
        return {"valid": False, "errors": ["OUTPUT_ARTIFACT_BUNDLE_REQUIRED"]}
    if not isinstance(declared, list):
        return {"valid": True, "errors": [], "output_artifacts": []}
    actual_ids: list[str] = []
    for output in declared:
        if not isinstance(output, Mapping):
            return {"valid": False, "errors": ["OUTPUT_ARTIFACT_ENTRY_INVALID"]}
        required = (
            "artifact_id",
            "schema_id",
            "schema_version",
            "output_path",
            "output_file_sha256",
            "output_schema_hash",
        )
        missing = [key for key in required if not str(output.get(key, "")).strip()]
        if missing:
            return {
                "valid": False,
                "errors": [f"OUTPUT_ARTIFACT_FIELDS_MISSING:{missing}"],
            }
        relative = Path(str(output["output_path"]))
        if relative.is_absolute():
            return {"valid": False, "errors": ["OUTPUT_ARTIFACT_PATH_ABSOLUTE"]}
        path = (manifest_path.resolve().parent / relative).resolve()
        if not path.is_file():
            return {
                "valid": False,
                "errors": [f"OUTPUT_ARTIFACT_MISSING:{path}"],
            }
        inspection = inspect_candidate_output(path)
        artifact_id = str(output["artifact_id"])
        actual_ids.append(artifact_id)
        checks = {
            "output_file_sha256": inspection.output_file_sha256,
            "output_byte_size": inspection.output_byte_size,
            "row_count": inspection.row_count,
            "column_count": inspection.column_count,
            "ordered_columns": list(inspection.ordered_columns),
        }
        for key, actual in checks.items():
            if output.get(key) != actual:
                return {
                    "valid": False,
                    "errors": [
                        f"OUTPUT_ARTIFACT_{key.upper()}_MISMATCH:{artifact_id}"
                    ],
                }
    if actual_ids != expected_ids:
        return {
            "valid": False,
            "errors": [f"OUTPUT_ARTIFACT_IDS_MISMATCH:{actual_ids}!={expected_ids}"],
        }
    return {"valid": True, "errors": [], "output_artifacts": list(declared)}


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
        "upstream_current_authority_contract_version": (
            UPSTREAM_CURRENT_AUTHORITY_CONTRACT_VERSION
        ),
        "candidate_transaction_contract_version": (
            CANDIDATE_TRANSACTION_CONTRACT_VERSION
        ),
        "output_inspector_registry": output_inspector_registry(),
        "atomic_write_sequence": [
            "inspect_output",
            "validate_upstreams",
            "derive_code_semantic_schema_authority",
            "validate_in_memory",
            "write_fsync_pending_temporary",
            "atomic_replace_pending",
            "reread_revalidate_pending",
            "atomic_replace_committed",
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


def _write_bytes_fsync(path: Path, payload: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _candidate_transaction_record(
    *,
    transaction_id: str,
    initial_state: str,
    prior_sha256: str | None,
    temporary: Path | None,
    backup: Path | None,
    last_state: str,
    rollback_attempted: bool,
    rollback_succeeded: bool,
    destination: Path,
    committed_sha256: str,
    classification: str,
) -> CandidateTransactionRecord:
    final_exists = destination.is_file()
    final_sha256 = file_sha256(destination) if final_exists else None
    return CandidateTransactionRecord(
        transaction_id=transaction_id,
        initial_final_path_state=initial_state,
        prior_manifest_sha256=prior_sha256,
        temporary_path=str(temporary) if temporary is not None else None,
        backup_path=str(backup) if backup is not None else None,
        last_completed_state=last_state,
        rollback_attempted=rollback_attempted,
        rollback_succeeded=rollback_succeeded,
        final_path_exists=final_exists,
        final_path_sha256=final_sha256,
        new_candidate_survived=final_sha256 == committed_sha256,
        official_promotion_occurred=False,
        classification=classification,
    )


def _write_candidate_transactionally(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    output_path: Path,
    upstream_manifests: Mapping[str, Mapping[str, Any]],
    before_atomic_replace: Callable[[], None] | None,
    failure_injector: Callable[[str], None] | None,
) -> CandidateTransactionRecord:
    destination = manifest_path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    transaction_id = str(manifest["candidate_transaction_id"])
    pending_manifest = dict(manifest)
    pending_manifest["candidate_transaction_state"] = (
        CANDIDATE_TRANSACTION_STATE_PENDING
    )
    pending_manifest["candidate_transaction_provenance_hash"] = (
        candidate_transaction_provenance_hash(
            transaction_id,
            CANDIDATE_TRANSACTION_STATE_PENDING,
        )
    )
    pending_bytes = canonical_json_bytes(pending_manifest) + b"\n"
    committed_bytes = canonical_json_bytes(manifest) + b"\n"
    committed_sha256 = hashlib.sha256(committed_bytes).hexdigest()
    prior_bytes = destination.read_bytes() if destination.is_file() else None
    prior_sha256 = (
        hashlib.sha256(prior_bytes).hexdigest()
        if prior_bytes is not None
        else None
    )
    prior_stat = destination.stat() if prior_bytes is not None else None
    initial_state = "PRESENT" if prior_bytes is not None else "ABSENT"
    temporary: Path | None = None
    backup: Path | None = None
    last_state = "NOT_STARTED"
    final_path_changed = False
    try:
        if prior_bytes is not None:
            descriptor, backup_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".candidate-backup",
                dir=destination.parent,
            )
            os.close(descriptor)
            backup = Path(backup_name)
            _write_bytes_fsync(backup, prior_bytes)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".candidate-staging",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        if failure_injector is not None:
            failure_injector("during_temporary_write")
        _write_bytes_fsync(temporary, pending_bytes)
        last_state = "TEMP_WRITTEN"
        if before_atomic_replace is not None:
            before_atomic_replace()
        if failure_injector is not None:
            failure_injector("before_atomic_rename")
        os.replace(temporary, destination)
        final_path_changed = True
        last_state = CANDIDATE_TRANSACTION_STATE_PENDING
        if failure_injector is not None:
            failure_injector("after_atomic_rename_before_final_reread")
        reread_bytes = destination.read_bytes()
        if reread_bytes != pending_bytes:
            raise ValueError("CANDIDATE_PENDING_REREAD_MISMATCH")
        if failure_injector is not None:
            failure_injector("during_final_reread_validation")
        reread_manifest = json.loads(reread_bytes.decode("utf-8"))
        validation = validate_artifact_manifest(
            reread_manifest,
            output_path=output_path,
            upstream_manifests=upstream_manifests,
            require_committed_transaction=False,
        )
        if not validation["valid"]:
            raise ValueError(
                "CANDIDATE_PENDING_REREAD_INVALID:"
                f"{validation['errors']}"
            )
        descriptor, commit_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".candidate-commit",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary = Path(commit_name)
        _write_bytes_fsync(temporary, committed_bytes)
        os.replace(temporary, destination)
        final_path_changed = True
        last_state = CANDIDATE_TRANSACTION_STATE_COMMITTED
        if backup is not None and backup.exists():
            backup.unlink()
        return _candidate_transaction_record(
            transaction_id=transaction_id,
            initial_state=initial_state,
            prior_sha256=prior_sha256,
            temporary=temporary,
            backup=backup,
            last_state=last_state,
            rollback_attempted=False,
            rollback_succeeded=True,
            destination=destination,
            committed_sha256=committed_sha256,
            classification=CANDIDATE_TRANSACTION_STATE_COMMITTED,
        )
    except BaseException as exc:
        rollback_attempted = final_path_changed
        rollback_succeeded = True
        rollback_error: BaseException | None = None
        if final_path_changed:
            try:
                if failure_injector is not None:
                    failure_injector("during_rollback")
                if prior_bytes is None:
                    destination.unlink(missing_ok=True)
                else:
                    if backup is None or not backup.is_file():
                        raise RuntimeError("CANDIDATE_ROLLBACK_BACKUP_MISSING")
                    os.replace(backup, destination)
                    if prior_stat is not None:
                        os.chmod(destination, prior_stat.st_mode)
                        os.utime(
                            destination,
                            ns=(
                                prior_stat.st_atime_ns,
                                prior_stat.st_mtime_ns,
                            ),
                        )
                    if file_sha256(destination) != prior_sha256:
                        raise RuntimeError(
                            "CANDIDATE_ROLLBACK_HASH_MISMATCH"
                        )
                last_state = "ROLLED_BACK"
            except BaseException as rollback_exc:
                rollback_succeeded = False
                rollback_error = rollback_exc
                last_state = "ROLLBACK_FAILED"
        for cleanup_path in (temporary, backup):
            if cleanup_path is not None and cleanup_path.exists():
                try:
                    cleanup_path.unlink()
                except OSError:
                    if rollback_error is None:
                        rollback_succeeded = False
                        last_state = "ROLLBACK_FAILED"
        classification = (
            "ROLLED_BACK" if rollback_succeeded else "ROLLBACK_FAILED"
        )
        record = _candidate_transaction_record(
            transaction_id=transaction_id,
            initial_state=initial_state,
            prior_sha256=prior_sha256,
            temporary=temporary,
            backup=backup,
            last_state=last_state,
            rollback_attempted=rollback_attempted,
            rollback_succeeded=rollback_succeeded,
            destination=destination,
            committed_sha256=committed_sha256,
            classification=classification,
        )
        detail = f"{type(exc).__name__}:{exc}"
        if rollback_error is not None:
            detail += (
                ":ROLLBACK_FAILED:"
                f"{type(rollback_error).__name__}:{rollback_error}"
            )
        raise CandidateManifestTransactionError(
            detail,
            transaction=record,
        ) from exc


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
    additional_outputs: Sequence[Mapping[str, Any]] = (),
    stage_specific_metadata: Mapping[str, Any] | None = None,
    expected_authority: Mapping[str, Any] | None = None,
    development_contract_version: str | None = None,
    before_atomic_replace: Callable[[], None] | None = None,
    failure_injector: Callable[[str], None] | None = None,
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
    _validate_manifest_authority_contract(contract)
    stage = _stage_contract(contract, stage_id)
    stage_order = [
        str(value["stage_id"]) for value in contract["stages"]
    ]
    if failure_injector is not None:
        failure_injector("before_output_inspection")
    inspection = inspect_candidate_output(output_path)
    if failure_injector is not None:
        failure_injector("after_output_inspection")
    upstreams = _validate_upstreams(
        repo_root=repo_root,
        contract_root=contract_root,
        stage_id=stage_id,
        artifact_id=artifact_id,
        artifact_class=artifact_class,
        upstream_manifest_paths=upstream_manifest_paths,
        stage_order=stage_order,
    )
    if failure_injector is not None:
        failure_injector("after_upstream_validation")
    schema_id, schema_version, schema_hash = _validate_schema_authority(
        stage=stage,
        intended_schema_id=output_schema_id,
        intended_schema_version=output_schema_version,
        inspection=inspection,
        stage_specific_metadata=metadata,
    )
    additional_bundle: list[dict[str, Any]] = []
    permitted_outputs = {
        str(value) for value in stage.get("output_artifacts", [])
    }
    for spec in additional_outputs:
        spec_artifact_id = str(spec.get("artifact_id", "")).strip()
        spec_schema_id = str(spec.get("schema_id", "")).strip()
        spec_schema_version = str(spec.get("schema_version", "")).strip()
        spec_path = Path(str(spec.get("path", ""))).resolve()
        if not spec_artifact_id or not spec_schema_id or not spec_schema_version:
            raise ValueError("OUTPUT_SPEC_SCHEMA_AUTHORITY_MISSING")
        if spec_artifact_id not in permitted_outputs:
            raise ValueError(
                f"OUTPUT_ARTIFACT_NOT_PERMITTED:{stage_id}:{spec_artifact_id}"
            )
        if spec_schema_id != spec_artifact_id:
            raise ValueError(
                f"OUTPUT_SCHEMA_ARTIFACT_ID_MISMATCH:{spec_artifact_id}"
            )
        spec_inspection = inspect_candidate_output(spec_path)
        additional_bundle.append(
            {
                "artifact_id": spec_artifact_id,
                "schema_id": spec_schema_id,
                "schema_version": spec_schema_version,
                "output_path": _canonical_artifact_relative_path(
                    spec_path,
                    candidate_manifest_path,
                ),
                "output_file_sha256": spec_inspection.output_file_sha256,
                "output_byte_size": spec_inspection.output_byte_size,
                "output_inspector_id": spec_inspection.inspector_id,
                "output_inspector_version": spec_inspection.inspector_version,
                "row_count": spec_inspection.row_count,
                "column_count": spec_inspection.column_count,
                "ordered_columns": list(spec_inspection.ordered_columns),
                "output_schema_hash": _output_schema_hash(
                    stage=stage,
                    schema_id=spec_schema_id,
                    schema_version=spec_schema_version,
                    inspection=spec_inspection,
                    schema_registry_entry=spec.get(
                        "schema_registry_entry",
                    ),
                ),
            }
        )
    if len(permitted_outputs) > 1 and len(additional_bundle) != (
        len(permitted_outputs) - 1
    ):
        raise ValueError(
            "OUTPUT_ARTIFACT_BUNDLE_INCOMPLETE:"
            f"{len(additional_bundle)}!={len(permitted_outputs) - 1}"
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
    builder_code_hash = _builder_code_hash()
    transaction_id = canonical_sha256(
        {
            "artifact_id": schema_id,
            "stage_execution_fingerprint": execution_fingerprint,
            "manifest_builder_code_hash": builder_code_hash,
            "candidate_transaction_contract_version": (
                CANDIDATE_TRANSACTION_CONTRACT_VERSION
            ),
        }
    )
    warnings = (
        [f"DEVELOPMENT_CONTRACT_DIRTY_FILES:{dirty_files}"]
        if dirty_files
        else []
    )
    manifest = {
        "artifact_manifest_version": ARTIFACT_MANIFEST_VERSION,
        "manifest_builder_id": MANIFEST_BUILDER_ID,
        "manifest_builder_version": MANIFEST_BUILDER_VERSION,
        "manifest_builder_code_hash": builder_code_hash,
        "authority_state": CANDIDATE_AUTHORITY_STATE,
        "candidate_transaction_id": transaction_id,
        "candidate_transaction_state": (
            CANDIDATE_TRANSACTION_STATE_COMMITTED
        ),
        "candidate_transaction_provenance_hash": (
            candidate_transaction_provenance_hash(
                transaction_id,
                CANDIDATE_TRANSACTION_STATE_COMMITTED,
            )
        ),
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
    manifest["output_artifacts"] = [
        {
            "artifact_id": artifact_id,
            "schema_id": schema_id,
            "schema_version": schema_version,
            "output_path": manifest["output_path"],
            "output_file_sha256": inspection.output_file_sha256,
            "output_byte_size": inspection.output_byte_size,
            "output_inspector_id": inspection.inspector_id,
            "output_inspector_version": inspection.inspector_version,
            "row_count": inspection.row_count,
            "column_count": inspection.column_count,
            "ordered_columns": list(inspection.ordered_columns),
            "output_schema_hash": schema_hash,
        },
        *additional_bundle,
    ]
    if failure_injector is not None:
        failure_injector("after_authority_derivation")
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
    bundle_validation = validate_candidate_output_bundle(
        manifest,
        manifest_path=candidate_manifest_path,
        stage=stage,
    )
    if not bundle_validation["valid"]:
        raise ValueError(
            f"CANDIDATE_OUTPUT_BUNDLE_INVALID:{bundle_validation['errors']}"
        )
    if failure_injector is not None:
        failure_injector("after_in_memory_manifest_validation")
    transaction = _write_candidate_transactionally(
        manifest_path=candidate_manifest_path,
        manifest=manifest,
        output_path=output_path,
        upstream_manifests=upstream_map,
        before_atomic_replace=before_atomic_replace,
        failure_injector=failure_injector,
    )
    return CandidateManifestBuild(
        manifest=manifest,
        manifest_path=candidate_manifest_path,
        output_inspection=inspection,
        validation=validation,
        production_builder_owned=True,
        transaction=transaction,
    )


__all__ = [
    "CANDIDATE_MANIFEST_DEVELOPMENT_CONTRACT_VERSION",
    "CANDIDATE_TRANSACTION_CONTRACT_VERSION",
    "CURRENT_AUTHORITATIVE",
    "CandidateManifestTransactionError",
    "CandidateManifestBuild",
    "CandidateTransactionRecord",
    "INELIGIBLE_ARTIFACT_CLASS",
    "INVALID_UPSTREAM_INTEGRITY",
    "LoadedUpstreamManifest",
    "OUTPUT_INSPECTOR_REGISTRY_ID",
    "OUTPUT_INSPECTOR_REGISTRY_VERSION",
    "OutputInspection",
    "STALE_CODE_AUTHORITY",
    "STALE_SCHEMA_AUTHORITY",
    "STALE_SEMANTIC_AUTHORITY",
    "UPSTREAM_CURRENT_AUTHORITY_CONTRACT_VERSION",
    "UpstreamAuthorityValidation",
    "VALID_HISTORICAL",
    "build_candidate_artifact_manifest",
    "candidate_manifest_builder_contract",
    "inspect_candidate_output",
    "validate_candidate_output_bundle",
    "load_validated_upstream_manifest",
    "output_inspector_registry",
    "validate_upstream_manifest_for_current_authority",
]
