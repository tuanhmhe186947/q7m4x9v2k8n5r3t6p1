#!/usr/bin/env python3
"""Audit that future frames do not change already-flushed tracking XML."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_prefix(xml_path: Path, frame_exclusive: int) -> list[dict[str, Any]]:
    root = ElementTree.parse(xml_path).getroot()
    payload: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for track in root.findall("track"):
        label = str(track.get("label", ""))
        track_id = str(track.get("id", ""))
        for box in track.findall("box"):
            frame = int(box.get("frame", "-1"))
            if frame < 0 or frame >= frame_exclusive:
                continue
            key = (label, track_id, frame)
            if key in seen:
                raise ValueError(f"Duplicate track/frame payload: {key}")
            seen.add(key)
            attributes = sorted(
                (
                    str(attribute.get("name", "")),
                    str(attribute.text or ""),
                )
                for attribute in box.findall("attribute")
            )
            payload.append(
                {
                    "label": label,
                    "track_id": track_id,
                    "frame": frame,
                    "box_attributes": {
                        key: str(value)
                        for key, value in sorted(box.attrib.items())
                    },
                    "attributes": attributes,
                }
            )
    payload.sort(key=lambda item: (item["frame"], item["label"], item["track_id"]))
    return payload


def _payload_sha256(payload: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_report(xml_path: Path) -> tuple[Path, dict[str, Any]]:
    report_path = xml_path.with_name("tracking_quality_report.json")
    return report_path, json.loads(report_path.read_text(encoding="utf-8"))


def _timing_contract(report: dict[str, Any]) -> tuple[str, int]:
    telemetry = report.get("telemetry") or {}
    return (
        str(telemetry.get("output_timing_contract", "")),
        int(telemetry.get("declared_delay_frames", -999)),
    )


def audit_prefix_invariance(
    prefix_xml: Path,
    extended_xml: Path,
    frame_exclusive: int,
    expected_timing_contract: str,
    expected_delay_frames: int,
    artifact_roots: list[Path],
) -> dict[str, Any]:
    prefix_payload = _canonical_prefix(prefix_xml, frame_exclusive)
    extended_payload = _canonical_prefix(extended_xml, frame_exclusive)
    prefix_report_path, prefix_report = _load_report(prefix_xml)
    extended_report_path, extended_report = _load_report(extended_xml)
    prefix_timing = _timing_contract(prefix_report)
    extended_timing = _timing_contract(extended_report)
    mp4_paths = sorted(
        str(path)
        for root in artifact_roots
        for path in root.rglob("*.mp4")
        if path.is_file()
    )
    processed_prefix = int(prefix_report.get("processed_frames", -1))
    processed_extended = int(extended_report.get("processed_frames", -1))
    errors: list[str] = []
    if prefix_payload != extended_payload:
        errors.append("flushed_xml_payload_changed_with_future_frames")
    if prefix_timing != (expected_timing_contract, expected_delay_frames):
        errors.append("prefix_timing_contract_mismatch")
    if extended_timing != (expected_timing_contract, expected_delay_frames):
        errors.append("extended_timing_contract_mismatch")
    if processed_prefix != frame_exclusive:
        errors.append("prefix_processed_frames_mismatch")
    if processed_extended <= frame_exclusive:
        errors.append("extended_run_has_no_future_frames")
    if mp4_paths:
        errors.append("generated_mp4_found")
    return {
        "schema_version": "tracking_prefix_invariance_audit_v1",
        "status": "PASS" if not errors else "FAIL",
        "frame_exclusive": frame_exclusive,
        "compared_payload_count": len(prefix_payload),
        "prefix_payload_sha256": _payload_sha256(prefix_payload),
        "extended_prefix_payload_sha256": _payload_sha256(extended_payload),
        "payloads_equal": prefix_payload == extended_payload,
        "expected_timing_contract": expected_timing_contract,
        "expected_delay_frames": expected_delay_frames,
        "prefix_timing": list(prefix_timing),
        "extended_timing": list(extended_timing),
        "processed_frames": [processed_prefix, processed_extended],
        "mp4_count": len(mp4_paths),
        "mp4_paths": mp4_paths,
        "artifacts": {
            "prefix_xml": str(prefix_xml),
            "prefix_xml_sha256": _file_sha256(prefix_xml),
            "extended_xml": str(extended_xml),
            "extended_xml_sha256": _file_sha256(extended_xml),
            "prefix_report": str(prefix_report_path),
            "prefix_report_sha256": _file_sha256(prefix_report_path),
            "extended_report": str(extended_report_path),
            "extended_report_sha256": _file_sha256(extended_report_path),
        },
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-xml", type=Path, required=True)
    parser.add_argument("--extended-xml", type=Path, required=True)
    parser.add_argument("--frame-exclusive", type=int, required=True)
    parser.add_argument("--expected-timing-contract", required=True)
    parser.add_argument("--expected-delay-frames", type=int, required=True)
    parser.add_argument(
        "--artifact-root",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {args.output}")
    audit = audit_prefix_invariance(
        prefix_xml=args.prefix_xml,
        extended_xml=args.extended_xml,
        frame_exclusive=args.frame_exclusive,
        expected_timing_contract=args.expected_timing_contract,
        expected_delay_frames=args.expected_delay_frames,
        artifact_roots=args.artifact_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
