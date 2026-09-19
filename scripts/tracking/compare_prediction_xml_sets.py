"""Compare two CVAT prediction sets without invoking tracking or evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

METADATA_TAGS = ("created", "updated", "dumped")
BBOX_NAMES = ("xtl", "ytl", "xbr", "ybr")


def canonical_hash(payload: Any) -> str:
    """Return a deterministic SHA-256 for JSON-compatible content."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the byte SHA-256 for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attributes(box: ET.Element) -> dict[str, str]:
    return {
        str(item.attrib.get("name", "")): str(item.text or "")
        for item in box.findall("attribute")
    }


def _confidence(attributes: dict[str, str]) -> str | None:
    for name, value in attributes.items():
        if name.lower() in {"confidence", "score"}:
            return value
    return None


def prediction_rows(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Parse scientific prediction rows and retain their serialized form."""

    root = ET.parse(path).getroot()
    video_key = str(root.findtext("./meta/task/name") or path.stem)
    rows: list[dict[str, Any]] = []
    for track_position, track in enumerate(root.findall("./track")):
        track_id = str(track.attrib.get("id", ""))
        label = str(track.attrib.get("label", ""))
        for row_position, box in enumerate(track.findall("./box")):
            attributes = _attributes(box)
            bbox_text = tuple(str(box.attrib[name]) for name in BBOX_NAMES)
            rows.append(
                {
                    "key": (track_id, int(box.attrib["frame"])),
                    "track_id": track_id,
                    "label": label,
                    "frame": int(box.attrib["frame"]),
                    "identity": attributes.get("ID", ""),
                    "hidden": attributes.get("Hidden", ""),
                    "outside": str(box.attrib.get("outside", "0")),
                    "occluded": str(box.attrib.get("occluded", "0")),
                    "bbox": tuple(float(value) for value in bbox_text),
                    "bbox_text": bbox_text,
                    "confidence": _confidence(attributes),
                    "attributes": tuple(sorted(attributes.items())),
                    "order": (track_position, row_position),
                }
            )
    return video_key, rows


def prediction_paths(root: Path) -> dict[str, Path]:
    """Resolve one XML per task name from flat or nested prediction roots."""

    resolved: dict[str, Path] = {}
    for path in sorted(root.rglob("*.xml")):
        video_key, _ = prediction_rows(path)
        if video_key in resolved:
            raise ValueError(f"duplicate prediction for {video_key}: {root}")
        resolved[video_key] = path
    return resolved


def bbox_iou(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    """Return IoU for two xyxy boxes."""

    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0, first[3] - first[1]
    )
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 1.0


def scientific_row(row: dict[str, Any]) -> tuple[Any, ...]:
    """Return metadata- and order-independent scientific row content."""

    return (
        row["track_id"],
        row["label"],
        row["frame"],
        row["identity"],
        row["hidden"],
        row["outside"],
        row["occluded"],
        row["bbox"],
        row["confidence"],
    )


def compare_xml(
    first_path: Path,
    second_path: Path,
    *,
    bbox_abs_tolerance: float,
    bbox_iou_threshold: float,
) -> dict[str, Any]:
    """Compare one prediction pair at byte, representation, and row levels."""

    first_key, first_rows = prediction_rows(first_path)
    second_key, second_rows = prediction_rows(second_path)
    if first_key != second_key:
        raise ValueError(f"task-name mismatch: {first_path} vs {second_path}")
    first_by_key = {row["key"]: row for row in first_rows}
    second_by_key = {row["key"]: row for row in second_rows}
    if len(first_by_key) != len(first_rows) or len(second_by_key) != len(
        second_rows
    ):
        raise ValueError(f"duplicate track/frame key: {first_key}")
    first_keys = set(first_by_key)
    second_keys = set(second_by_key)
    shared = sorted(first_keys & second_keys)
    removed = sorted(first_keys - second_keys)
    added = sorted(second_keys - first_keys)
    identity = 0
    hidden = 0
    bbox = 0
    bbox_serialization = 0
    bbox_tolerance = 0
    confidence = 0
    minimum_iou = 1.0
    maximum_delta = 0.0
    first_divergence: dict[str, Any] | None = None
    for key in shared:
        first = first_by_key[key]
        second = second_by_key[key]
        row_identity = first["identity"] != second["identity"]
        row_hidden = (
            first["hidden"],
            first["outside"],
            first["occluded"],
        ) != (
            second["hidden"],
            second["outside"],
            second["occluded"],
        )
        deltas = [
            abs(left - right)
            for left, right in zip(
                first["bbox"],
                second["bbox"],
                strict=True,
            )
        ]
        row_bbox = any(delta != 0.0 for delta in deltas)
        row_serialization = (
            not row_bbox and first["bbox_text"] != second["bbox_text"]
        )
        iou = bbox_iou(first["bbox"], second["bbox"])
        row_tolerance = (
            max(deltas, default=0.0) > bbox_abs_tolerance
            or iou < bbox_iou_threshold
        )
        row_confidence = first["confidence"] != second["confidence"]
        identity += int(row_identity)
        hidden += int(row_hidden)
        bbox += int(row_bbox)
        bbox_serialization += int(row_serialization)
        bbox_tolerance += int(row_tolerance)
        confidence += int(row_confidence)
        minimum_iou = min(minimum_iou, iou)
        maximum_delta = max(maximum_delta, *deltas)
        if first_divergence is None and any(
            (
                row_identity,
                row_hidden,
                row_bbox,
                row_serialization,
                row_confidence,
            )
        ):
            first_divergence = {
                "video_key": first_key,
                "track_id": key[0],
                "frame": key[1],
                "first": scientific_row(first),
                "second": scientific_row(second),
            }
    if first_divergence is None and (removed or added):
        first_divergence = {
            "video_key": first_key,
            "removed_key": removed[0] if removed else None,
            "added_key": added[0] if added else None,
        }
    first_canonical = canonical_hash(
        sorted(scientific_row(row) for row in first_rows)
    )
    second_canonical = canonical_hash(
        sorted(scientific_row(row) for row in second_rows)
    )
    first_order = [row["key"] for row in first_rows]
    second_order = [row["key"] for row in second_rows]
    return {
        "video_key": first_key,
        "first_path": str(first_path),
        "second_path": str(second_path),
        "first_file_sha256": sha256_file(first_path),
        "second_file_sha256": sha256_file(second_path),
        "byte_equal": sha256_file(first_path) == sha256_file(second_path),
        "first_row_count": len(first_rows),
        "second_row_count": len(second_rows),
        "row_removals": len(removed),
        "row_additions": len(added),
        "frame_index_differences": len(removed) + len(added),
        "identity_differences": identity,
        "hidden_state_differences": hidden,
        "bbox_exact_differences": bbox,
        "bbox_numeric_serialization_differences": bbox_serialization,
        "bbox_tolerance_violations": bbox_tolerance,
        "minimum_paired_bbox_iou": minimum_iou,
        "maximum_absolute_bbox_coordinate_difference": maximum_delta,
        "confidence_differences": confidence,
        "ordering_differences": int(first_order != second_order),
        "first_canonical_content_sha256": first_canonical,
        "second_canonical_content_sha256": second_canonical,
        "canonical_content_equal": first_canonical == second_canonical,
        "metadata_or_non_scientific_only_difference": (
            sha256_file(first_path) != sha256_file(second_path)
            and first_canonical == second_canonical
        ),
        "first_divergence": first_divergence,
    }


def compare_prediction_sets(
    first_root: Path,
    second_root: Path,
    *,
    bbox_abs_tolerance: float = 0.01,
    bbox_iou_threshold: float = 0.9999,
) -> dict[str, Any]:
    """Compare two complete prediction roots with one canonical contract."""

    first_paths = prediction_paths(first_root)
    second_paths = prediction_paths(second_root)
    if set(first_paths) != set(second_paths):
        raise ValueError("prediction video populations differ")
    records = [
        compare_xml(
            first_paths[key],
            second_paths[key],
            bbox_abs_tolerance=bbox_abs_tolerance,
            bbox_iou_threshold=bbox_iou_threshold,
        )
        for key in sorted(first_paths)
    ]
    sum_fields = (
        "first_row_count",
        "second_row_count",
        "row_removals",
        "row_additions",
        "frame_index_differences",
        "identity_differences",
        "hidden_state_differences",
        "bbox_exact_differences",
        "bbox_numeric_serialization_differences",
        "bbox_tolerance_violations",
        "confidence_differences",
        "ordering_differences",
    )
    aggregate = {
        field: sum(int(record[field]) for record in records)
        for field in sum_fields
    }
    aggregate.update(
        {
            "videos_compared": len(records),
            "minimum_paired_bbox_iou": min(
                float(record["minimum_paired_bbox_iou"])
                for record in records
            ),
            "maximum_absolute_bbox_coordinate_difference": max(
                float(record["maximum_absolute_bbox_coordinate_difference"])
                for record in records
            ),
            "byte_equal": all(record["byte_equal"] for record in records),
            "canonical_content_equal": all(
                record["canonical_content_equal"] for record in records
            ),
            "metadata_or_non_scientific_only_difference": all(
                record["byte_equal"]
                or record["metadata_or_non_scientific_only_difference"]
                for record in records
            ),
            "first_divergence": next(
                (
                    record["first_divergence"]
                    for record in records
                    if record["first_divergence"] is not None
                ),
                None,
            ),
        }
    )
    if not math.isfinite(aggregate["minimum_paired_bbox_iou"]):
        raise ValueError("non-finite IoU")
    return {
        "schema_version": "tracking.prediction_set_comparison.v1",
        "first_root": str(first_root),
        "second_root": str(second_root),
        "bbox_abs_tolerance": bbox_abs_tolerance,
        "bbox_iou_threshold": bbox_iou_threshold,
        "aggregate": aggregate,
        "videos": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-root", required=True, type=Path)
    parser.add_argument("--second-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bbox-abs-tolerance", type=float, default=0.01)
    parser.add_argument("--bbox-iou-threshold", type=float, default=0.9999)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing overwrite: {args.output}")
    result = compare_prediction_sets(
        args.first_root.resolve(),
        args.second_root.resolve(),
        bbox_abs_tolerance=args.bbox_abs_tolerance,
        bbox_iou_threshold=args.bbox_iou_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
