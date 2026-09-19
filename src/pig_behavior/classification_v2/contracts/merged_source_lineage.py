"""Validate the exact source set used by a mixed classification merge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_CLASSIFICATION_XML_NAMES = frozenset(
    {
        "Pigs281119_000085.xml",
        "Pigs281119_000114.xml",
        "Pigs291119_000216.xml",
        "Pigs291119_000225.xml",
        "Pigs291119_000226.xml",
        "Pigs291119_000231.xml",
        "Pigs291119_000233.xml",
        "Pigs291119_000302.xml",
        "Pigs301119_000327.xml",
        "Pigs301119_000328.xml",
        "Pigs301119_000329.xml",
        "Pigs301119_000330.xml",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_mixed_source_lineage(
    lineage_json: Path,
    *,
    legacy_export: Path,
    classification_dir: Path,
    expected_xml_count: int = 12,
    expected_xml_names: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """Check source identity, hashes, counts and output hash for a merge."""

    errors: list[str] = []
    try:
        payload = json.loads(lineage_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _result(errors=[f"invalid_lineage_json:{exc}"])

    if payload.get("schema_version") != "classification_v2.merged_source_lineage.v1":
        errors.append("invalid_merged_source_lineage_schema")
    entries = payload.get("source_files", [])
    if not isinstance(entries, list):
        errors.append("source_files_not_a_list")
        entries = []

    expected_legacy = legacy_export.resolve()
    expected_dir = classification_dir.resolve()
    resolved_entries: list[Path] = []
    xml_entries: list[Path] = []
    for entry in entries:
        path = Path(str(entry.get("path", ""))).resolve()
        resolved_entries.append(path)
        if path.suffix.casefold() == ".xml":
            xml_entries.append(path)
        if not path.is_file():
            errors.append(f"source_file_missing:{path}")
            continue
        if int(entry.get("size", -1)) != path.stat().st_size:
            errors.append(f"source_file_size_mismatch:{path}")
        if entry.get("sha256") != sha256_file(path):
            errors.append(f"source_file_sha256_mismatch:{path}")

    if resolved_entries.count(expected_legacy) != 1:
        errors.append("legacy_export_not_bound_exactly_once")
    if len(xml_entries) != expected_xml_count:
        errors.append(
            f"classification_xml_count={len(xml_entries)}!={expected_xml_count}"
        )
    expected_names = (
        EXPECTED_CLASSIFICATION_XML_NAMES
        if expected_xml_names is None
        else frozenset(expected_xml_names)
    )
    actual_names = {path.name for path in xml_entries}
    if actual_names != expected_names:
        errors.append(
            "classification_xml_name_set_mismatch:"
            f"missing={sorted(expected_names - actual_names)}"
            f":unexpected={sorted(actual_names - expected_names)}"
        )
    for path in xml_entries:
        if path.parent != expected_dir:
            errors.append(f"xml_outside_classification_dir:{path}")
        if "tracking" in {part.casefold() for part in path.parts}:
            errors.append(f"tracking_xml_forbidden:{path}")

    source_counts = payload.get("source_type_counts", {})
    if int(source_counts.get("legacy_recovered", 0)) <= 0:
        errors.append("legacy_recovered_source_missing")
    if int(source_counts.get("cvat_tracking_xml", 0)) <= 0:
        errors.append("cvat_tracking_xml_source_missing")
    if int(payload.get("rows", 0)) <= 0:
        errors.append("merged_output_empty")

    output = payload.get("output", {})
    output_path = Path(str(output.get("path", ""))).resolve()
    if not output_path.is_file():
        errors.append(f"merged_output_missing:{output_path}")
    else:
        if int(output.get("size", -1)) != output_path.stat().st_size:
            errors.append("merged_output_size_mismatch")
        if output.get("sha256") != sha256_file(output_path):
            errors.append("merged_output_sha256_mismatch")

    return _result(errors=errors)


def _result(*, errors: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "classification_v2.mixed_source_lineage_gate.v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }


__all__ = ["audit_mixed_source_lineage", "sha256_file"]
