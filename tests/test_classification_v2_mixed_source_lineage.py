from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pig_behavior.classification_v2.contracts.merged_source_lineage import (
    audit_mixed_source_lineage,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_lineage(tmp_path: Path, *, use_tracking: bool = False) -> tuple[Path, Path, Path]:
    legacy = tmp_path / "legacy_16f_rebuild" / "legacy_frame_object_annotations.csv"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("frame_uid\nlegacy\n", encoding="utf-8")
    folder_name = "tracking" if use_tracking else "classification"
    xml_dir = tmp_path / "data" / "annotations" / folder_name
    xml_dir.mkdir(parents=True)
    xml_paths = []
    for index in range(2):
        path = xml_dir / f"video_{index}.xml"
        path.write_text(f"xml-{index}\n", encoding="utf-8")
        xml_paths.append(path)
    output = tmp_path / "merged.csv"
    output.write_text("frame_uid\nlegacy\nxml\n", encoding="utf-8")
    entries = [legacy, *xml_paths]
    lineage = {
        "schema_version": "classification_v2.merged_source_lineage.v1",
        "source_files": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in entries
        ],
        "output": {
            "path": str(output.resolve()),
            "size": output.stat().st_size,
            "sha256": _sha256(output),
        },
        "source_type_counts": {
            "legacy_recovered": 1,
            "cvat_tracking_xml": 1,
        },
        "rows": 2,
    }
    lineage_path = tmp_path / "lineage.json"
    lineage_path.write_text(json.dumps(lineage), encoding="utf-8")
    return lineage_path, legacy, xml_dir


def test_mixed_source_lineage_passes(tmp_path: Path) -> None:
    lineage, legacy, xml_dir = _make_lineage(tmp_path)

    audit = audit_mixed_source_lineage(
        lineage,
        legacy_export=legacy,
        classification_dir=xml_dir,
        expected_xml_count=2,
        expected_xml_names={"video_0.xml", "video_1.xml"},
    )

    assert audit["status"] == "PASS"


def test_tracking_directory_is_rejected(tmp_path: Path) -> None:
    lineage, legacy, xml_dir = _make_lineage(tmp_path, use_tracking=True)

    audit = audit_mixed_source_lineage(
        lineage,
        legacy_export=legacy,
        classification_dir=tmp_path / "data" / "annotations" / "classification",
        expected_xml_count=2,
        expected_xml_names={"video_0.xml", "video_1.xml"},
    )

    assert audit["status"] == "FAIL"
    assert any("xml_outside_classification_dir" in error for error in audit["errors"])
