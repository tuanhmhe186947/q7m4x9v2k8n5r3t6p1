"""Consolidate the deterministic full legacy L2 lineage gate."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.contracts.lineage_claims import (
    LINEAGE_CLAIM_COLUMNS,
    resolve_optional_lineage_claims,
)
from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.contracts.temporal_tier_contract import (
    LEGACY_TEMPORAL_MODEL_VIEW_SPECS,
)
from pig_behavior.classification_v2.datasets.legacy_unreviewed_development import (
    LEGACY_DEVELOPMENT_SCOPE,
    LEGACY_SOURCE,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

SCHEMA_VERSION = "classification_v2.legacy_development_l2_gate.v1"
TIER_SCHEMA_VERSION = "classification_v2.legacy_unreviewed_development.v3"
LOADER_SCHEMA_VERSION = "classification_v2.temporal_loader_audit.v1"
REAL_LOADER_SCHEMA_VERSION = "classification_v2.legacy_tier_loader_audit.v2"
EXPECTED_FRAME_ROWS = 72_864
EXPECTED_NATIVE_UNITS = 4_554
EXPECTED_WINDOW_ROWS = 45_540
EXPECTED_MATCHED_ROWS = 18_216
EXPECTED_SLIDING_BY_LENGTH = {
    6: 18_216,
    8: 13_662,
    12: 9_108,
    16: 4_554,
}

CSV_EXPECTED_ROWS: dict[str, int] = {
    "00_scope/merged_frame_objects_legacy_complete_units.csv": (
        EXPECTED_FRAME_ROWS
    ),
    "01_context/frame_context.csv": EXPECTED_FRAME_ROWS,
    "02_geometry/frame_geometry.csv": EXPECTED_FRAME_ROWS,
    "03_roi/frame_roi.csv": EXPECTED_FRAME_ROWS,
    "04_enhanced/spatiotemporal_frame_features_enhanced.csv": (
        EXPECTED_FRAME_ROWS
    ),
    "05_sequence/harmonized_frames.csv": EXPECTED_FRAME_ROWS,
    "05_sequence/temporal_intervals_standalone.csv": EXPECTED_NATIVE_UNITS,
    "05_windows/training_ready_frame_features_harmonized_preview.csv": (
        EXPECTED_FRAME_ROWS
    ),
    "05_windows/temporal_label_intervals.csv": EXPECTED_NATIVE_UNITS,
    "05_windows/sequence_window_manifest.csv": EXPECTED_WINDOW_ROWS,
    "05_windows/sequence_window_features.csv": EXPECTED_WINDOW_ROWS,
    "06_temporal_tier_contract/source_unit_manifest.csv": (
        EXPECTED_NATIVE_UNITS
    ),
    "06_temporal_tier_contract/native_temporal_unit_manifest.csv": (
        EXPECTED_NATIVE_UNITS
    ),
    "06_temporal_tier_contract/temporal_tier_selection_manifest.csv": (
        EXPECTED_WINDOW_ROWS
    ),
    "06_temporal_tier_contract/temporal_tier_all_sliding_manifest.csv": (
        EXPECTED_WINDOW_ROWS
    ),
    "06_temporal_tier_contract/temporal_tier_matched_manifest.csv": (
        EXPECTED_MATCHED_ROWS
    ),
}

UPSTREAM_AUDITS = {
    "source_selection": "00_scope/source_selection_audit.json",
    "context": "01_context/frame_context_audit.json",
    "geometry": "02_geometry/frame_geometry_audit.json",
    "roi": "03_roi/frame_roi_audit.json",
    "enhanced": (
        "04_enhanced/spatiotemporal_frame_features_enhanced_audit.json"
    ),
    "harmonization": "05_sequence/temporal_harmonization_audit.json",
    "windows": "05_windows/sequence_window_audit.json",
    "tier": (
        "06_temporal_tier_contract/legacy_unreviewed_development_audit.json"
    ),
    "loader": "07_loader_audit/temporal_view_loader_audit.json",
}

TIER_INPUT_ARTIFACTS = {
    "source_reference_csv": (
        "00_scope/merged_frame_objects_legacy_complete_units.csv"
    ),
    "harmonized_frame_csv": "05_sequence/harmonized_frames.csv",
    "intervals_csv": "05_sequence/temporal_intervals_standalone.csv",
    "window_manifest_csv": "05_windows/sequence_window_manifest.csv",
}
TIER_OUTPUT_ARTIFACTS = {
    "source_units": "06_temporal_tier_contract/source_unit_manifest.csv",
    "native_units": (
        "06_temporal_tier_contract/native_temporal_unit_manifest.csv"
    ),
    "selection": (
        "06_temporal_tier_contract/temporal_tier_selection_manifest.csv"
    ),
    "all_sliding": (
        "06_temporal_tier_contract/temporal_tier_all_sliding_manifest.csv"
    ),
    "matched": (
        "06_temporal_tier_contract/temporal_tier_matched_manifest.csv"
    ),
}


def _selected_windows(view_name: str, sequence_length: int) -> int:
    if "all_sliding" in view_name:
        return EXPECTED_SLIDING_BY_LENGTH[sequence_length]
    return EXPECTED_NATIVE_UNITS


for _view_name, _view_spec in LEGACY_TEMPORAL_MODEL_VIEW_SPECS.items():
    _length = int(_view_spec["sequence_length"])
    _selected = _selected_windows(_view_name, _length)
    _filename = str(_view_spec["slot_manifest_filename"])
    _relative = f"06_temporal_tier_contract/{_filename}"
    CSV_EXPECTED_ROWS[_relative] = _selected * _length
    TIER_OUTPUT_ARTIFACTS[f"slots:{_view_name}"] = _relative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--repeat-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _evidence_expectations() -> dict[str, dict[tuple[str, ...], Any]]:
    tier: dict[tuple[str, ...], Any] = {
        ("schema_version",): TIER_SCHEMA_VERSION,
        ("status",): "PASS_LEGACY_UNREVIEWED_BOUNDED_DEVELOPMENT",
        ("lineage_scope",): LEGACY_DEVELOPMENT_SCOPE,
        ("human_review_complete",): False,
        ("source_scope",): LEGACY_SOURCE,
        ("full_oof_authorized",): False,
        ("q2_claim_allowed",): False,
        ("valid_for_bounded_development",): True,
        ("training_evidence_status",): "UNREVIEWED_DEVELOPMENT_ONLY",
        ("errors",): [],
        ("source_audit", "rows"): EXPECTED_FRAME_ROWS,
        ("source_audit", "native_units"): EXPECTED_NATIVE_UNITS,
        ("source_audit", "duplicate_source_unit_frame_rows"): 0,
        ("source_audit", "invalid_relative_frame_rows"): 0,
        ("source_audit", "incomplete_native_units"): 0,
        ("source_audit", "errors"): [],
        ("harmonized_frame_audit", "rows"): EXPECTED_FRAME_ROWS,
        ("harmonized_frame_audit", "native_units"): EXPECTED_NATIVE_UNITS,
        ("harmonized_frame_audit", "duplicate_temporal_unit_frame_rows"): 0,
        ("harmonized_frame_audit", "invalid_frame_index_rows"): 0,
        ("harmonized_frame_audit", "incomplete_native_units"): 0,
        ("harmonized_frame_audit", "errors"): [],
        ("native_unit_audit", "rows"): EXPECTED_NATIVE_UNITS,
        ("native_unit_audit", "duplicate_temporal_unit_key"): 0,
        ("native_unit_audit", "missing_source_units"): 0,
        ("native_unit_audit", "missing_harmonized_units"): 0,
        ("native_unit_audit", "invalid_interval_geometry_units"): 0,
        ("native_unit_audit", "harmonized_interval_bound_mismatch_units"): 0,
        ("native_unit_audit", "hidden_used_as_exclusion"): False,
        ("native_unit_audit", "errors"): [],
        ("temporal_tier_audit", "native_units"): EXPECTED_NATIVE_UNITS,
        ("temporal_tier_audit", "all_sliding_rows"): EXPECTED_WINDOW_ROWS,
        ("temporal_tier_audit", "matched_rows"): EXPECTED_MATCHED_ROWS,
        ("temporal_tier_audit", "rows_dropped"): 0,
        ("temporal_tier_audit", "labels_changed"): 0,
        ("temporal_tier_audit", "missing_native_window_rows"): 0,
        ("temporal_tier_audit", "source_mismatch_rows"): 0,
        ("temporal_tier_audit", "dataset_mismatch_rows"): 0,
        ("temporal_tier_audit", "video_mismatch_rows"): 0,
        ("temporal_tier_audit", "track_mismatch_rows"): 0,
        ("temporal_tier_audit", "behavior_mismatch_rows"): 0,
        ("temporal_tier_audit", "outside_native_unit_rows"): 0,
        ("temporal_tier_audit", "tier_support_count_mismatches"): 0,
        ("temporal_tier_audit", "tier_window_lattice_mismatches"): 0,
        ("temporal_tier_audit", "missing_native_tier_pairs"): 0,
        ("temporal_tier_audit", "matched_duplicate_pairs"): 0,
        ("temporal_tier_audit", "input_window_order_preserved"): True,
        ("temporal_tier_audit", "tier_event_mass_max_abs_error"): 0.0,
        ("temporal_tier_audit", "errors"): [],
        ("temporal_model_input_audit", "selection_rows"): (
            EXPECTED_WINDOW_ROWS
        ),
        ("temporal_model_input_audit", "selection_order_matches_window_universe"): True,
        ("temporal_model_input_audit", "duplicate_temporal_slot_source_frames"): 0,
        ("temporal_model_input_audit", "invalid_temporal_slot_source_frame_indices"): 0,
        ("temporal_model_input_audit", "rows_dropped"): 0,
        ("temporal_model_input_audit", "labels_changed"): 0,
        ("temporal_model_input_audit", "errors"): [],
    }
    loader: dict[tuple[str, ...], Any] = {
        ("schema_version",): LOADER_SCHEMA_VERSION,
        ("lineage_scope",): LEGACY_DEVELOPMENT_SCOPE,
        ("human_review_complete",): False,
        ("optimizer_steps",): 0,
        ("full_dataset_read",): False,
        ("errors",): [],
        ("valid",): True,
        ("legacy_tier_real_packet", "window_universe_rows"): (
            EXPECTED_WINDOW_ROWS
        ),
        ("legacy_tier_real_packet", "schema_version"): (
            REAL_LOADER_SCHEMA_VERSION
        ),
        ("legacy_tier_real_packet", "expected_view_count"): 8,
        ("legacy_tier_real_packet", "loaded_view_count"): 8,
        ("legacy_tier_real_packet", "lineage_scope"): (
            LEGACY_DEVELOPMENT_SCOPE
        ),
        ("legacy_tier_real_packet", "human_review_complete"): False,
        ("legacy_tier_real_packet", "optimizer_steps"): 0,
        ("legacy_tier_real_packet", "errors"): [],
        ("legacy_tier_real_packet", "valid"): True,
    }
    for view_name, spec in LEGACY_TEMPORAL_MODEL_VIEW_SPECS.items():
        length = int(spec["sequence_length"])
        selected = _selected_windows(view_name, length)
        tier_prefix = ("temporal_model_input_audit", "view_audits", view_name)
        loader_prefix = ("legacy_tier_real_packet", "views", view_name)
        tier.update(
            {
                (*tier_prefix, "windows"): selected,
                (*tier_prefix, "slot_rows"): selected * length,
                (*tier_prefix, "sequence_length"): length,
                (*tier_prefix, "missing_observed_slots"): 0,
                (*tier_prefix, "invalid_timing_slots"): 0,
                (*tier_prefix, "nonpositive_time_delta_slots"): 0,
                (*tier_prefix, "duplicate_slot_key_rows"): 0,
                (*tier_prefix, "rows_dropped"): 0,
                (*tier_prefix, "errors"): [],
            }
        )
        loader.update(
            {
                (*loader_prefix, "sequence_length"): length,
                (*loader_prefix, "selected_windows"): selected,
                (*loader_prefix, "tensor_shape"): [
                    EXPECTED_WINDOW_ROWS,
                    length,
                ],
                (*loader_prefix, "shape_valid"): True,
                (*loader_prefix, "selected_nonempty"): True,
                (*loader_prefix, "selected_time_delta_finite"): True,
                (*loader_prefix, "unselected_time_delta_nan"): True,
                (*loader_prefix, "selected_timing_valid"): True,
                (*loader_prefix, "selected_observed"): True,
                (*loader_prefix, "unselected_masks_clear"): True,
                (*loader_prefix, "lineage_scope"): LEGACY_DEVELOPMENT_SCOPE,
                (*loader_prefix, "human_review_complete"): False,
            }
        )
    for length, count in EXPECTED_SLIDING_BY_LENGTH.items():
        tier[("temporal_tier_audit", "all_sliding_rows_by_tier", f"T{length}")] = count
        tier[("temporal_tier_audit", "matched_rows_by_tier", f"T{length}")] = (
            EXPECTED_NATIVE_UNITS
        )
    return {"tier": tier, "loader": loader}


_MISSING = object()


def _lookup(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _same_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    return actual == expected


def audit_legacy_l2_evidence(
    tier_audit: dict[str, Any],
    loader_audit: dict[str, Any],
    *,
    artifact_name: str,
) -> dict[str, Any]:
    """Validate counts, no-loss invariants, loader views, and claim boundaries."""

    errors: list[str] = []
    payloads = {"tier": tier_audit, "loader": loader_audit}
    for payload_name, expectations in _evidence_expectations().items():
        payload = payloads[payload_name]
        for path, expected in expectations.items():
            actual = _lookup(payload, path)
            if actual is _MISSING or not _same_value(actual, expected):
                rendered = ".".join(path)
                errors.append(
                    f"{artifact_name}:{payload_name}.{rendered}="
                    f"{actual!r}:expected={expected!r}"
                )

    native = tier_audit.get("native_unit_audit", {})
    valid_units = native.get("valid_development_units")
    invalid_units = native.get("invalid_development_units")
    if not isinstance(valid_units, int) or not isinstance(invalid_units, int):
        errors.append(f"{artifact_name}:invalid native validity counts")
    elif valid_units + invalid_units != EXPECTED_NATIVE_UNITS:
        errors.append(
            f"{artifact_name}:native validity total="
            f"{valid_units + invalid_units}:expected={EXPECTED_NATIVE_UNITS}"
        )
    labels = native.get("behavior_counts", {})
    if sorted(labels) != sorted(VALID_BEHAVIORS):
        errors.append(f"{artifact_name}:native labels={sorted(labels)}")
    elif any(int(value) <= 0 for value in labels.values()):
        errors.append(f"{artifact_name}:native label has zero support")
    elif sum(int(value) for value in labels.values()) != EXPECTED_NATIVE_UNITS:
        errors.append(f"{artifact_name}:native label support does not reconcile")
    return {
        "valid_development_units": valid_units,
        "invalid_development_units": invalid_units,
        "behavior_counts": labels,
        "errors": errors,
        "valid": not errors,
    }


def _walk_audit_failures(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, dict):
        if "errors" in value and value["errors"] != []:
            errors.append(f"{path}.errors={value['errors']!r}")
        if value.get("valid") is False:
            errors.append(f"{path}.valid=false")
        for key, nested in value.items():
            _walk_audit_failures(nested, f"{path}.{key}", errors)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _walk_audit_failures(nested, f"{path}[{index}]", errors)


def _read_upstream_audits(root: Path) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    payloads: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    errors: list[str] = []
    for name, relative in UPSTREAM_AUDITS.items():
        path = root / relative
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            hashes[name] = file_sha256(path)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{root}:{name}:{type(exc).__name__}:{exc}")
            continue
        payloads[name] = payload
        if payload.get("lineage_scope") != LEGACY_DEVELOPMENT_SCOPE:
            errors.append(f"{root}:{name}:invalid lineage_scope")
        if payload.get("human_review_complete") is not False:
            errors.append(f"{root}:{name}:invalid human_review_complete")
        _walk_audit_failures(payload, f"{root}:{name}", errors)
    return payloads, hashes, errors


def _audit_csv_artifacts(
    root: Path,
    expected_rows: dict[str, int] | None = None,
) -> dict[str, Any]:
    expected_rows = expected_rows or CSV_EXPECTED_ROWS
    rows: dict[str, int] = {}
    hashes: dict[str, str] = {}
    errors: list[str] = []
    for relative, expected in expected_rows.items():
        path = root / relative
        try:
            claims_frame = pd.read_csv(
                path,
                usecols=list(LINEAGE_CLAIM_COLUMNS),
                low_memory=False,
            )
            claims = resolve_optional_lineage_claims(
                claims_frame,
                artifact_name=str(path),
            )
            hashes[relative] = file_sha256(path)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            errors.append(f"{root}:{relative}:{type(exc).__name__}:{exc}")
            continue
        rows[relative] = int(len(claims_frame))
        if len(claims_frame) != expected:
            errors.append(
                f"{root}:{relative}:rows={len(claims_frame)}:expected={expected}"
            )
        if (
            claims is None
            or claims.lineage_scope != LEGACY_DEVELOPMENT_SCOPE
            or claims.human_review_complete
        ):
            errors.append(f"{root}:{relative}:invalid lineage claims")
    return {"rows": rows, "hashes": hashes, "errors": errors, "valid": not errors}


def _verify_bound_hashes(
    root: Path,
    tier_audit: dict[str, Any],
    actual_hashes: dict[str, str],
) -> dict[str, Any]:
    errors: list[str] = []
    checked: dict[str, str] = {}
    sections = (
        ("input_artifacts", TIER_INPUT_ARTIFACTS),
        ("output_artifacts", TIER_OUTPUT_ARTIFACTS),
    )
    for section_name, mapping in sections:
        section = tier_audit.get(section_name, {})
        for artifact_name, relative in mapping.items():
            record = section.get(artifact_name, {})
            bound_hash = record.get("sha256") if isinstance(record, dict) else None
            bound_path = record.get("path") if isinstance(record, dict) else None
            actual_hash = actual_hashes.get(relative)
            checked[artifact_name] = str(actual_hash or "")
            if not actual_hash or bound_hash != actual_hash:
                errors.append(
                    f"{root}:{artifact_name}:bound={bound_hash}:actual={actual_hash}"
                )
            expected_path = str(root / relative).replace("\\", "/").casefold()
            observed_path = str(bound_path or "").replace("\\", "/").casefold()
            if observed_path != expected_path:
                errors.append(
                    f"{root}:{artifact_name}:path={bound_path}:"
                    f"expected={root / relative}"
                )
    return {"hashes": checked, "errors": errors, "valid": not errors}


def _audit_root(root: Path) -> dict[str, Any]:
    payloads, audit_hashes, audit_errors = _read_upstream_audits(root)
    csv_audit = _audit_csv_artifacts(root)
    tier = payloads.get("tier", {})
    loader = payloads.get("loader", {})
    evidence = audit_legacy_l2_evidence(
        tier,
        loader,
        artifact_name=str(root),
    )
    bound = _verify_bound_hashes(root, tier, csv_audit["hashes"])
    errors = [
        *audit_errors,
        *csv_audit["errors"],
        *evidence["errors"],
        *bound["errors"],
    ]
    return {
        "root": str(root),
        "csv_audit": csv_audit,
        "upstream_audit_hashes": audit_hashes,
        "evidence": evidence,
        "bound_hash_audit": bound,
        "errors": errors,
        "valid": not errors,
    }


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "dirty_entries": status}


def _audit_repeat_hashes(
    primary_hashes: dict[str, str],
    repeat_hashes: dict[str, str],
    relatives: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    relatives = relatives or tuple(CSV_EXPECTED_ROWS)
    errors: list[str] = []
    pairs: dict[str, Any] = {}
    for relative in relatives:
        primary_hash = primary_hashes.get(relative)
        repeat_hash = repeat_hashes.get(relative)
        identical = bool(primary_hash and primary_hash == repeat_hash)
        if not identical:
            errors.append(f"repeat_hash_mismatch={relative}")
        pairs[relative] = {
            "primary_sha256": primary_hash,
            "repeat_sha256": repeat_hash,
            "byte_identical": identical,
        }
    return {"pairs": pairs, "errors": errors, "valid": not errors}


def run_legacy_development_l2_audit(
    primary_root: Path,
    repeat_root: Path,
) -> dict[str, Any]:
    """Audit both full roots and require byte-identical derived CSV artifacts."""

    same_root = primary_root.resolve() == repeat_root.resolve()
    primary = _audit_root(primary_root)
    repeat = _audit_root(repeat_root)
    repeat_hash_audit = _audit_repeat_hashes(
        primary["csv_audit"]["hashes"],
        repeat["csv_audit"]["hashes"],
    )
    errors = [
        *(["primary_and_repeat_roots_must_differ"] if same_root else []),
        *primary["errors"],
        *repeat["errors"],
        *repeat_hash_audit["errors"],
    ]
    valid = not errors
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_LEGACY_DEVELOPMENT_L2" if valid else "FAIL_L2",
        "lineage_scope": LEGACY_DEVELOPMENT_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "model_training_authorized": False,
        "l3_immutable_input_freeze_authorized": valid,
        "expected_counts": {
            "frame_rows": EXPECTED_FRAME_ROWS,
            "native_units": EXPECTED_NATIVE_UNITS,
            "all_sliding_rows": EXPECTED_WINDOW_ROWS,
            "centered_matched_rows": EXPECTED_MATCHED_ROWS,
            "all_sliding_by_length": EXPECTED_SLIDING_BY_LENGTH,
        },
        "git_state": _git_state(),
        "primary": primary,
        "repeat": repeat,
        "repeat_hash_audit": repeat_hash_audit,
        "errors": errors,
        "valid": valid,
    }


def main() -> None:
    args = parse_args()
    output_json = args.output_json or (
        args.primary_root / "08_l2_audit" / "legacy_development_l2_audit.json"
    )
    require_output_paths_available([output_json], overwrite=args.overwrite)
    audit = run_legacy_development_l2_audit(
        args.primary_root,
        args.repeat_root,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "valid": audit["valid"],
                "primary_root": str(args.primary_root),
                "repeat_root": str(args.repeat_root),
                "csv_pairs": len(audit["repeat_hash_audit"]["pairs"]),
                "errors": audit["errors"],
                "output_json": str(output_json),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not audit["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
