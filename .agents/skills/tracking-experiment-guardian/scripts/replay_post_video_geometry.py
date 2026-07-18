#!/usr/bin/env python3
"""Replay one geometry-only post-video candidate from immutable parent shapes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pig_behavior.evaluation.tracking.lineage import (  # noqa: E402
    cvat_prediction_semantic_sha256,
)
from pig_behavior.tracking.config import TrackingConfig  # noqa: E402
from pig_behavior.tracking.exporters.annotation import (  # noqa: E402
    strip_internal_shape_keys,
    write_annotation_json,
)
from pig_behavior.tracking.exporters.cvat_xml import (  # noqa: E402
    write_cvat_video_xml,
)
from pig_behavior.tracking.masks import load_mask  # noqa: E402
from pig_behavior.tracking.refinement import (  # noqa: E402
    refine_near_wall_hidden_geometry,
)

SELECTED_SKILLS = [
    "tracking-experiment-guardian",
    "computer-vision-opencv",
    "safe-refactor-test-guardian",
    "scientific-ablation-controller",
    "experiment-lineage-reproducibility",
]
GEOMETRY_CANDIDATE = "near_wall_hidden_geometry_v1"
DELTA_FIELDS = [
    "frame",
    "label",
    "id_value",
    "parent_points",
    "candidate_points",
    "width_excess",
    "previous_anchor_frame",
    "following_anchor_frame",
]


def file_sha256(path: Path) -> str:
    """Return the SHA256 of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bounded_unit_float(raw: str) -> float:
    """Parse a float in the closed unit interval."""

    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return value


def positive_int(raw: str) -> int:
    """Parse a positive integer."""

    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def nonnegative_int(raw: str) -> int:
    """Parse a non-negative integer."""

    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=[GEOMETRY_CANDIDATE], required=True)
    parser.add_argument("--parent-shapes-json", type=Path, required=True)
    parser.add_argument("--parent-xml", type=Path, required=True)
    parser.add_argument("--parent-run-manifest", type=Path, required=True)
    parser.add_argument("--experiment-plan", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-start-frame", type=nonnegative_int, default=None)
    parser.add_argument("--score-end-frame", type=nonnegative_int, default=None)
    parser.add_argument("--max-gap-frames", type=positive_int, default=30)
    parser.add_argument(
        "--distance-bbox-scale",
        type=bounded_unit_float,
        default=0.25,
    )
    parser.add_argument(
        "--min-width-excess",
        type=bounded_unit_float,
        default=0.08,
    )
    parser.add_argument(
        "--max-center-shift",
        type=bounded_unit_float,
        default=0.04,
    )
    parser.add_argument(
        "--original-weight",
        type=bounded_unit_float,
        default=0.50,
    )
    parser.add_argument(
        "--allow-no-change",
        action="store_true",
        help="Permit a replay with zero changed boxes for diagnostic use only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize without creating the output directory.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    """Resolve and require one regular file."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def read_parent_shapes(path: Path) -> list[dict[str, Any]]:
    """Read the canonical CVAT shapes JSON emitted by the tracker."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, list)
        or len(payload) != 1
        or not isinstance(payload[0], dict)
        or not isinstance(payload[0].get("shapes"), list)
    ):
        raise ValueError("parent shapes JSON does not match the CVAT shapes schema")
    shapes = payload[0]["shapes"]
    if not shapes:
        raise ValueError("parent shapes JSON contains no shapes")
    return shapes


def shape_attributes(shape: dict[str, Any]) -> dict[str, str]:
    """Return shape attributes keyed by name."""

    return {
        str(item["name"]): str(item.get("value", ""))
        for item in shape.get("attributes", [])
    }


def shape_key(shape: dict[str, Any]) -> tuple[str, int, str]:
    """Return the immutable geometry replay key."""

    attributes = shape_attributes(shape)
    return (
        str(shape.get("label", "")),
        int(shape["frame"]),
        attributes.get("ID", ""),
    )


def exported_shape_record(shape: dict[str, Any]) -> dict[str, Any]:
    """Build the exact semantic payload represented in CVAT XML."""

    attributes = shape_attributes(shape)
    return {
        "label": str(shape["label"]),
        "frame": int(shape["frame"]),
        "points": [round(float(value), 2) for value in shape["points"]],
        "outside": bool(shape.get("outside", False)),
        "occluded": bool(shape.get("occluded", False)),
        "ID": attributes.get("ID", ""),
        "Behavior": attributes.get("Behavior", ""),
        "Hidden": attributes.get("Hidden", ""),
    }


def read_xml_shape_records(path: Path) -> list[dict[str, Any]]:
    """Read ordered CVAT box records for parent-integrity validation."""

    root = ET.parse(path).getroot()
    records: list[dict[str, Any]] = []
    for track in root.findall("track"):
        label = str(track.get("label", ""))
        for box in track.findall("box"):
            attributes = {
                str(item.get("name", "")): str(item.text or "")
                for item in box.findall("attribute")
            }
            records.append(
                {
                    "label": label,
                    "frame": int(box.get("frame", "0")),
                    "points": [
                        round(float(box.get(name, "0")), 2)
                        for name in ("xtl", "ytl", "xbr", "ybr")
                    ],
                    "outside": box.get("outside", "0") == "1",
                    "occluded": box.get("occluded", "0") == "1",
                    "ID": attributes.get("ID", ""),
                    "Behavior": attributes.get("Behavior", ""),
                    "Hidden": attributes.get("Hidden", ""),
                }
            )
    records.sort(key=lambda item: (item["label"], item["frame"]))
    return records


def parent_geometry_contract(
    shapes: list[dict[str, Any]],
    parent_xml: Path,
) -> None:
    """Require the parent shapes JSON and XML to describe identical boxes."""

    json_records = [exported_shape_record(shape) for shape in shapes]
    json_records.sort(key=lambda item: (item["label"], item["frame"]))
    xml_records = read_xml_shape_records(parent_xml)
    if json_records != xml_records:
        raise ValueError("parent shapes JSON does not match parent CVAT XML")


def non_geometry_payload(shape: dict[str, Any]) -> dict[str, Any]:
    """Return every exported shape field except bbox coordinates."""

    clean = strip_internal_shape_keys(shape)
    return {key: value for key, value in clean.items() if key != "points"}


def stable_json_sha256(payload: Any) -> str:
    """Hash a deterministic JSON representation."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_state() -> dict[str, Any]:
    """Capture the repository commit and dirty inventory."""

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status_lines = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "commit": commit,
        "dirty": bool(status_lines),
        "dirty_entries": status_lines,
    }


def frame_size_from_parent_xml(path: Path) -> tuple[int, int]:
    """Read the original frame size from parent CVAT metadata."""

    root = ET.parse(path).getroot()
    width_text = root.findtext("./meta/task/original_size/width")
    height_text = root.findtext("./meta/task/original_size/height")
    if width_text is None or height_text is None:
        raise ValueError("parent XML is missing original frame dimensions")
    width = int(width_text)
    height = int(height_text)
    if width < 1 or height < 1:
        raise ValueError("parent XML frame dimensions must be positive")
    return width, height


def validate_frame_window(args: argparse.Namespace) -> None:
    """Require score-window arguments to be absent or a valid pair."""

    values = (args.score_start_frame, args.score_end_frame)
    if (values[0] is None) != (values[1] is None):
        raise ValueError("score start and end frames must be provided together")
    if values[0] is not None and values[0] > values[1]:
        raise ValueError("score start frame must be <= score end frame")


def replay_config(args: argparse.Namespace, mask_path: Path) -> TrackingConfig:
    """Build the exact isolated geometry candidate config."""

    return TrackingConfig(
        mask_path=mask_path,
        use_mask=True,
        near_wall_hidden_geometry_refine=True,
        near_wall_hidden_geometry_max_gap_frames=args.max_gap_frames,
        near_wall_hidden_geometry_distance_bbox_scale=(
            args.distance_bbox_scale
        ),
        near_wall_hidden_geometry_min_width_excess=args.min_width_excess,
        near_wall_hidden_geometry_max_center_shift=args.max_center_shift,
        near_wall_hidden_geometry_original_weight=args.original_weight,
    )


def build_delta_rows(
    parent_shapes: list[dict[str, Any]],
    candidate_shapes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one audit row per changed bbox."""

    rows: list[dict[str, Any]] = []
    for parent, candidate in zip(parent_shapes, candidate_shapes, strict=True):
        if parent["points"] == candidate["points"]:
            continue
        rows.append(
            {
                "frame": int(candidate["frame"]),
                "label": str(candidate["label"]),
                "id_value": shape_attributes(candidate).get("ID", ""),
                "parent_points": json.dumps(parent["points"]),
                "candidate_points": json.dumps(candidate["points"]),
                "width_excess": candidate.get(
                    "_near_wall_hidden_geometry_width_excess",
                    "",
                ),
                "previous_anchor_frame": candidate.get(
                    "_near_wall_hidden_geometry_anchor_frames",
                    ["", ""],
                )[0],
                "following_anchor_frame": candidate.get(
                    "_near_wall_hidden_geometry_anchor_frames",
                    ["", ""],
                )[1],
            }
        )
    return rows


def write_delta_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write deterministic bbox delta evidence."""

    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DELTA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def input_record(path: Path) -> dict[str, Any]:
    """Build a hashed input record."""

    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def output_record(path: Path) -> dict[str, Any]:
    """Build a hashed output record."""

    record = input_record(path)
    if path.suffix.lower() == ".xml":
        record["semantic_sha256"] = cvat_prediction_semantic_sha256(path)
    return record


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Validate, replay and optionally write the candidate artifacts."""

    validate_frame_window(args)
    parent_shapes_json = require_file(
        args.parent_shapes_json,
        "parent shapes JSON",
    )
    parent_xml = require_file(args.parent_xml, "parent XML")
    parent_run_manifest = require_file(
        args.parent_run_manifest,
        "parent run manifest",
    )
    experiment_plan = require_file(args.experiment_plan, "experiment plan")
    mask_path = require_file(args.mask, "mask")
    source_video = require_file(args.source_video, "source video")
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to reuse output directory: {args.output_dir}"
        )

    parent_manifest_payload = json.loads(
        parent_run_manifest.read_text(encoding="utf-8")
    )
    if parent_manifest_payload.get("status") != "completed":
        raise ValueError("parent run manifest status must be completed")
    if not parent_manifest_payload.get("config", {}).get(
        "include_hidden",
        False,
    ):
        raise ValueError("parent run must use include_hidden=true")

    parent_shapes = read_parent_shapes(parent_shapes_json)
    parent_geometry_contract(parent_shapes, parent_xml)
    width, height = frame_size_from_parent_xml(parent_xml)
    cfg = replay_config(args, mask_path)
    mask = load_mask(mask_path, width, height, cfg)
    candidate_shapes = refine_near_wall_hidden_geometry(
        parent_shapes,
        width,
        height,
        mask,
        cfg,
    )
    if len(parent_shapes) != len(candidate_shapes):
        raise ValueError("geometry replay changed the shape count")

    parent_keys = [shape_key(shape) for shape in parent_shapes]
    candidate_keys = [shape_key(shape) for shape in candidate_shapes]
    if parent_keys != candidate_keys or len(set(parent_keys)) != len(parent_keys):
        raise ValueError("geometry replay changed or duplicated shape keys")

    parent_payload = [non_geometry_payload(shape) for shape in parent_shapes]
    candidate_payload = [
        non_geometry_payload(shape) for shape in candidate_shapes
    ]
    if parent_payload != candidate_payload:
        raise ValueError("geometry replay changed non-geometry payload")

    delta_rows = build_delta_rows(parent_shapes, candidate_shapes)
    if not delta_rows and not args.allow_no_change:
        raise ValueError("geometry replay changed zero boxes")
    score_delta_rows = delta_rows
    if args.score_start_frame is not None:
        score_delta_rows = [
            row
            for row in delta_rows
            if args.score_start_frame <= row["frame"] <= args.score_end_frame
        ]
        if not score_delta_rows and not args.allow_no_change:
            raise ValueError("geometry replay changed zero boxes in score window")

    summary = {
        "shape_count": len(parent_shapes),
        "changed_bbox_count": len(delta_rows),
        "score_window_changed_bbox_count": len(score_delta_rows),
        "changed_ids": sorted({row["id_value"] for row in delta_rows}),
        "non_geometry_payload_sha256": stable_json_sha256(parent_payload),
    }
    if args.dry_run:
        return {
            "status": "dry_run_pass",
            "summary": summary,
        }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    candidate_json = args.output_dir / "annotations_cvat_shapes.json"
    candidate_xml = args.output_dir / "annotations_cvat_video_1_1.xml"
    delta_csv = args.output_dir / "geometry_delta.csv"
    manifest_path = args.output_dir / "geometry_replay_manifest.json"
    write_annotation_json(candidate_json, candidate_shapes)
    frame_count = max(int(shape["frame"]) for shape in candidate_shapes) + 1
    write_cvat_video_xml(
        candidate_xml,
        candidate_shapes,
        source_video,
        width,
        height,
        frame_count,
    )
    write_delta_csv(delta_csv, delta_rows)
    if list(args.output_dir.rglob("*.mp4")):
        raise RuntimeError("geometry replay output contains an MP4")

    manifest = {
        "schema_version": 1,
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": args.candidate,
        "command": list(sys.argv),
        "selected_skills": SELECTED_SKILLS,
        "git": git_state(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "config": {
            "include_hidden": True,
            "max_gap_frames": args.max_gap_frames,
            "distance_bbox_scale": args.distance_bbox_scale,
            "min_width_excess": args.min_width_excess,
            "max_center_shift": args.max_center_shift,
            "original_weight": args.original_weight,
            "score_start_frame": args.score_start_frame,
            "score_end_frame": args.score_end_frame,
        },
        "inputs": {
            "parent_shapes_json": input_record(parent_shapes_json),
            "parent_xml": input_record(parent_xml),
            "parent_run_manifest": input_record(parent_run_manifest),
            "experiment_plan": input_record(experiment_plan),
            "mask": input_record(mask_path),
            "source_video": input_record(source_video),
        },
        "parent_prediction_semantic_sha256": (
            cvat_prediction_semantic_sha256(parent_xml)
        ),
        "summary": summary,
        "payload_integrity": {
            "status": "PASS",
            "shape_keys_equal": True,
            "non_geometry_payload_equal": True,
            "non_geometry_payload_sha256": stable_json_sha256(parent_payload),
        },
        "outputs": {
            "candidate_shapes_json": output_record(candidate_json),
            "candidate_xml": output_record(candidate_xml),
            "geometry_delta_csv": output_record(delta_csv),
        },
        "recursive_no_mp4": {
            "status": "PASS",
            "mp4_count": 0,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    return {
        "status": "completed",
        "manifest": str(manifest_path),
        "summary": summary,
    }


def main() -> int:
    """Run the geometry replay."""

    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
