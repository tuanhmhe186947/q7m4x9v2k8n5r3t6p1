from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pig_behavior.tracking.exporters.cvat_xml import write_cvat_video_xml  # noqa: E402
from pig_behavior.tracking.refinement import _shape_attributes_dict  # noqa: E402


def shape_id(shape: dict[str, Any]) -> str | None:
    value = _shape_attributes_dict(shape).get("ID")
    return str(value) if value else None


def set_shape_id(shape: dict[str, Any], value: str) -> None:
    for attr in shape.get("attributes", []):
        if attr.get("name") == "ID":
            attr["value"] = value
            return
    shape.setdefault("attributes", []).append({"name": "ID", "value": value})


def center(shape: dict[str, Any]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in shape["points"]]
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a["points"]]
    bx1, by1, bx2, by2 = [float(v) for v in b["points"]]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def center_distance(a: dict[str, Any], b: dict[str, Any], diagonal: float) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return math.hypot(ax - bx, ay - by) / diagonal


def hidden(shape: dict[str, Any]) -> bool:
    return bool(shape.get("outside", False)) or _shape_attributes_dict(shape).get(
        "Hidden",
        "No",
    ) == "Yes"


def relabel_motion(
    shapes: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    max_jump: float,
    min_gain: float,
    memory_frames: int,
    min_current_iou: float,
    max_current_center_dist: float,
    require_current_proposed_id: bool,
    require_mutual_swap: bool,
    allowed_edges: set[tuple[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    relabeled = [deepcopy(shape) for shape in shapes]
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for shape in relabeled:
        if hidden(shape):
            continue
        by_frame.setdefault(int(shape["frame"]), []).append(shape)

    diagonal = max(math.hypot(width, height), 1.0)
    prev_by_id: dict[str, tuple[float, float]] = {}
    prev_frame_by_id: dict[str, int] = {}
    changes = 0
    change_rows: list[dict[str, Any]] = []
    for frame in sorted(by_frame):
        frame_shapes = by_frame[frame]
        current_ids = [shape_id(shape) for shape in frame_shapes]
        shape_by_current_id = {
            shape_id(shape): shape
            for shape in frame_shapes
            if shape_id(shape) is not None
        }
        active_prev_by_id = {
            id_value: xy
            for id_value, xy in prev_by_id.items()
            if frame - prev_frame_by_id.get(id_value, frame) <= memory_frames
        }
        if not active_prev_by_id or any(value is None for value in current_ids):
            prev_by_id = dict(active_prev_by_id)
            prev_by_id.update(
                {
                    shape_id(shape): center(shape)
                    for shape in frame_shapes
                    if shape_id(shape) is not None
                }
            )
            prev_frame_by_id.update(
                {
                    shape_id(shape): frame
                    for shape in frame_shapes
                    if shape_id(shape) is not None
                }
            )
            continue

        candidates: list[tuple[float, str, int]] = []
        for idx, shape in enumerate(frame_shapes):
            cx, cy = center(shape)
            for id_value, (px, py) in active_prev_by_id.items():
                cost = math.hypot(cx - px, cy - py) / diagonal
                if cost <= max_jump:
                    candidates.append((cost, id_value, idx))
        candidates.sort()

        used_ids: set[str] = set()
        used_indexes: set[int] = set()
        assigned: dict[int, str] = {}
        for cost, id_value, idx in candidates:
            _ = cost
            if id_value in used_ids or idx in used_indexes:
                continue
            used_ids.add(id_value)
            used_indexes.add(idx)
            assigned[idx] = id_value

        # Require a real motion advantage over keeping the current ID, otherwise
        # leave the tracker output untouched.
        for idx, proposed_id in assigned.items():
            shape = frame_shapes[idx]
            current_id = shape_id(shape)
            if current_id == proposed_id or current_id is None:
                continue
            if current_id in active_prev_by_id:
                px, py = active_prev_by_id[current_id]
                cx, cy = center(shape)
                keep_cost = math.hypot(cx - px, cy - py) / diagonal
            else:
                keep_cost = max_jump + min_gain
            proposed_px, proposed_py = active_prev_by_id[proposed_id]
            cx, cy = center(shape)
            proposed_cost = math.hypot(cx - proposed_px, cy - proposed_py) / diagonal
            if keep_cost - proposed_cost < min_gain:
                continue
            partner_shape = shape_by_current_id.get(proposed_id)
            if require_current_proposed_id and partner_shape is None:
                continue
            current_iou = iou(shape, partner_shape) if partner_shape is not None else 0.0
            current_center_dist = (
                center_distance(shape, partner_shape, diagonal)
                if partner_shape is not None
                else float("inf")
            )
            if min_current_iou > 0.0 and current_iou < min_current_iou:
                continue
            if (
                max_current_center_dist > 0.0
                and current_center_dist > max_current_center_dist
            ):
                continue
            partner_idx = (
                frame_shapes.index(partner_shape)
                if partner_shape is not None
                else None
            )
            if (
                require_mutual_swap
                and partner_idx is not None
                and assigned.get(partner_idx) != current_id
            ):
                continue
            if require_mutual_swap and partner_idx is None:
                continue
            edge = tuple(sorted((current_id, proposed_id)))
            if allowed_edges is not None and edge not in allowed_edges:
                continue
            set_shape_id(shape, proposed_id)
            shape["_realtime_motion_relabel"] = True
            changes += 1
            change_rows.append(
                {
                    "frame": frame,
                    "old_id": current_id,
                    "new_id": proposed_id,
                    "keep_cost": round(keep_cost, 6),
                    "proposed_cost": round(proposed_cost, 6),
                    "gain": round(keep_cost - proposed_cost, 6),
                    "current_iou": round(current_iou, 6),
                    "current_center_dist": round(current_center_dist, 6)
                    if math.isfinite(current_center_dist)
                    else "",
                    "mutual": bool(
                        partner_idx is not None and assigned.get(partner_idx) == current_id
                    ),
                }
            )

        prev_by_id = dict(active_prev_by_id)
        prev_by_id.update(
            {
                shape_id(shape): center(shape)
                for shape in frame_shapes
                if shape_id(shape) is not None
            }
        )
        prev_frame_by_id.update(
            {
                shape_id(shape): frame
                for shape in frame_shapes
                if shape_id(shape) is not None
            }
        )
    return relabeled, changes, change_rows


def copy_and_relabel_video(
    source_video_dir: Path,
    target_video_dir: Path,
    *,
    video_path: Path,
    max_jump: float,
    min_gain: float,
    memory_frames: int,
    min_current_iou: float,
    max_current_center_dist: float,
    require_current_proposed_id: bool,
    require_mutual_swap: bool,
    max_component_size: int,
    max_component_edges: int,
    dense_fallback_max_edges: int,
    dense_fallback_max_support_ratio: float,
) -> int:
    target_video_dir.mkdir(parents=True, exist_ok=True)
    source_json = source_video_dir / "annotations_cvat_shapes.json"
    payload = json.loads(source_json.read_text(encoding="utf-8"))
    shapes = payload[0]["shapes"]
    width = 1280
    height = 720
    frame_count = max(int(shape["frame"]) for shape in shapes) + 1
    allowed_edges: set[tuple[str, str]] | None = None
    if max_component_size > 0:
        _, _, planned_rows = relabel_motion(
            shapes,
            width=width,
            height=height,
            max_jump=max_jump,
            min_gain=min_gain,
            memory_frames=memory_frames,
            min_current_iou=min_current_iou,
            max_current_center_dist=max_current_center_dist,
            require_current_proposed_id=require_current_proposed_id,
            require_mutual_swap=require_mutual_swap,
        )
        neighbors: dict[str, set[str]] = {}
        edge_support: dict[tuple[str, str], int] = {}
        for row in planned_rows:
            old_id = str(row["old_id"])
            new_id = str(row["new_id"])
            neighbors.setdefault(old_id, set()).add(new_id)
            neighbors.setdefault(new_id, set()).add(old_id)
            edge = tuple(sorted((old_id, new_id)))
            edge_support[edge] = edge_support.get(edge, 0) + 1
        allowed_edges = set()
        seen: set[str] = set()
        for start in neighbors:
            if start in seen:
                continue
            stack = [start]
            component: set[str] = set()
            while stack:
                node = stack.pop()
                if node in component:
                    continue
                component.add(node)
                stack.extend(neighbors.get(node, set()) - component)
            seen.update(component)
            component_edges = {
                tuple(sorted((old_id, new_id)))
                for old_id in component
                for new_id in neighbors.get(old_id, set()) & component
            }
            component_too_large = len(component) > max_component_size
            component_too_dense = max_component_edges > 0 and len(component_edges) > max_component_edges
            if component_too_large or component_too_dense:
                if dense_fallback_max_edges <= 0 or dense_fallback_max_support_ratio <= 0.0:
                    continue
                max_support = max(edge_support[edge] for edge in component_edges)
                rare_edges = [
                    edge
                    for edge in component_edges
                    if edge_support[edge] <= max_support * dense_fallback_max_support_ratio
                ]
                rare_edges.sort(key=lambda edge: (edge_support[edge], edge))
                allowed_edges.update(rare_edges[:dense_fallback_max_edges])
                continue
            allowed_edges.update(component_edges)

    relabeled, changes, change_rows = relabel_motion(
        shapes,
        width=width,
        height=height,
        max_jump=max_jump,
        min_gain=min_gain,
        memory_frames=memory_frames,
        min_current_iou=min_current_iou,
        max_current_center_dist=max_current_center_dist,
        require_current_proposed_id=require_current_proposed_id,
        require_mutual_swap=require_mutual_swap,
        allowed_edges=allowed_edges,
    )
    payload[0]["shapes"] = relabeled
    (target_video_dir / "annotations_cvat_shapes.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    write_cvat_video_xml(
        target_video_dir / "annotations_cvat_video_1_1.xml",
        relabeled,
        video_path,
        width,
        height,
        frame_count,
    )
    for name in ("annotations_coco.json", "annotations_coco_clean_train.json", "labels.json"):
        source = source_video_dir / name
        if source.exists():
            shutil.copy2(source, target_video_dir / name)
    if change_rows:
        import csv

        with (target_video_dir / "motion_relabel_changes.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(change_rows[0]))
            writer.writeheader()
            writer.writerows(change_rows)
    return changes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pred-root", required=True, type=Path)
    parser.add_argument("--target-pred-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--video", required=True, help="Video stem or comma-separated stems.")
    parser.add_argument("--max-jump", type=float, default=0.08)
    parser.add_argument("--min-gain", type=float, default=0.02)
    parser.add_argument("--memory-frames", type=int, default=15)
    parser.add_argument("--min-current-iou", type=float, default=0.0)
    parser.add_argument("--max-current-center-dist", type=float, default=0.0)
    parser.add_argument("--require-current-proposed-id", action="store_true")
    parser.add_argument("--require-mutual-swap", action="store_true")
    parser.add_argument("--max-component-size", type=int, default=0)
    parser.add_argument("--max-component-edges", type=int, default=0)
    parser.add_argument("--dense-fallback-max-edges", type=int, default=0)
    parser.add_argument("--dense-fallback-max-support-ratio", type=float, default=0.0)
    args = parser.parse_args()

    videos = [video.strip() for video in args.video.split(",") if video.strip()]
    video_paths: list[Path] = []
    total_changes = 0
    for video in videos:
        source_video_dir = next(args.source_pred_root.glob(f"**/{video}"))
        video_path = PROJECT_ROOT / "data" / "videos" / f"{video}.mp4"
        video_paths.append(video_path)
        target_video_dir = (
            args.target_pred_root
            / "iou0_area0_condarea0_merge0"
            / "realtime"
            / video
        )
        changes = copy_and_relabel_video(
            source_video_dir,
            target_video_dir,
            video_path=video_path,
            max_jump=args.max_jump,
            min_gain=args.min_gain,
            memory_frames=args.memory_frames,
            min_current_iou=args.min_current_iou,
            max_current_center_dist=args.max_current_center_dist,
            require_current_proposed_id=args.require_current_proposed_id,
            require_mutual_swap=args.require_mutual_swap,
            max_component_size=args.max_component_size,
            max_component_edges=args.max_component_edges,
            dense_fallback_max_edges=args.dense_fallback_max_edges,
            dense_fallback_max_support_ratio=args.dense_fallback_max_support_ratio,
        )
        total_changes += changes
        print(f"{video}: motion_relabel_changes={changes}")
    print(f"motion_relabel_changes={total_changes}")

    cmd = [
        sys.executable,
        "src/pig_behavior/evaluation/tracking_pipeline.py",
        "--video",
        ",".join(str(video_path) for video_path in video_paths),
        "--tracking-mode",
        "realtime",
        "--prediction-root",
        str(args.target_pred_root),
        "--output-root",
        str(args.output_root),
        "--no-run-missing-tracker",
        "--no-benchmark-rules",
        "--rule-combo",
        "iou0_area0_condarea0_merge0",
    ]
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
