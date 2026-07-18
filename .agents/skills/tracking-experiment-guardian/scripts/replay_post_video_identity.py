#!/usr/bin/env python3
"""Replay one identity-payload boundary candidate from a hashed parent."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
for search_root in (SCRIPT_ROOT, SRC_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from replay_post_video_geometry import (  # noqa: E402
    exported_shape_record,
    file_sha256,
    frame_size_from_parent_xml,
    git_state,
    input_record,
    nonnegative_int,
    output_record,
    parent_geometry_contract,
    positive_int,
    read_parent_shapes,
    read_xml_shape_records,
    require_file,
    shape_attributes,
    stable_json_sha256,
    validate_frame_window,
)

from pig_behavior.evaluation.tracking.lineage import (  # noqa: E402
    cvat_prediction_semantic_sha256,
)
from pig_behavior.tracking.exporters.annotation import (  # noqa: E402
    strip_internal_shape_keys,
    write_annotation_json,
)
from pig_behavior.tracking.exporters.cvat_xml import (  # noqa: E402
    write_cvat_video_xml,
)
from pig_behavior.tracking.geometry import bbox_iou  # noqa: E402

SELECTED_SKILLS = [
    "tracking-experiment-guardian",
    "computer-vision-opencv",
    "safe-refactor-test-guardian",
    "scientific-ablation-controller",
    "experiment-lineage-reproducibility",
    "skill-creator",
]
CANDIDATE_AFTER_RUN = "hidden_suffix_commit_after_run_v1"
CANDIDATE_PERSISTENT_OVERLAP = (
    "hidden_suffix_commit_after_overlap_persistence_v1"
)
CANDIDATES = (CANDIDATE_AFTER_RUN, CANDIDATE_PERSISTENT_OVERLAP)
# Keep the original public name for callers and fixtures that target H5a.
CANDIDATE = CANDIDATE_AFTER_RUN
RUNTIME_CONTRACT = "post_video_identity_payload_replay"
DELTA_FIELDS = [
    "frame",
    "label",
    "parent_id",
    "candidate_id",
    "hidden",
    "score",
    "hidden_label",
    "partner_label",
    "run_start",
    "run_end",
    "old_commit_start",
    "candidate_commit_start",
]


@dataclass(frozen=True)
class ReplayConfig:
    """Thresholds matching the existing hidden-suffix repair trigger."""

    min_hidden_frames: int = 8
    max_hidden_frames: int = 15
    min_overlap_iou: float = 0.70
    max_hidden_median_score: float = 0.50
    start_back_frames: int = 7
    min_suffix_frames: int = 600


@dataclass(frozen=True)
class PersistenceReplayConfig(ReplayConfig):
    """Thresholds for a geometry-persistent hidden-overlap commit."""

    min_overlap_persistence_frames: int = 2


def bounded_unit_float(raw: str) -> float:
    """Parse a float in the closed unit interval."""

    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return value


def parse_args() -> argparse.Namespace:
    """Parse the fail-closed identity replay command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--parent-shapes-json", type=Path, required=True)
    parser.add_argument("--parent-xml", type=Path, required=True)
    parser.add_argument("--parent-run-manifest", type=Path, required=True)
    parser.add_argument("--experiment-plan", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-start-frame", type=nonnegative_int)
    parser.add_argument("--score-end-frame", type=nonnegative_int)
    parser.add_argument("--min-hidden-frames", type=positive_int)
    parser.add_argument("--max-hidden-frames", type=positive_int)
    parser.add_argument(
        "--min-overlap-iou",
        type=bounded_unit_float,
    )
    parser.add_argument(
        "--max-hidden-median-score",
        type=bounded_unit_float,
    )
    parser.add_argument(
        "--start-back-frames",
        type=nonnegative_int,
    )
    parser.add_argument("--min-suffix-frames", type=positive_int)
    parser.add_argument(
        "--min-overlap-persistence-frames",
        type=positive_int,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def label_identity(shape: dict[str, Any]) -> int:
    """Return the fixed numeric identity encoded by the tracker label."""

    label = str(shape.get("label", ""))
    prefix = "Pig_"
    if not label.startswith(prefix):
        raise ValueError(f"shape label does not encode an identity: {label}")
    return int(label.removeprefix(prefix))


def id_value(shape: dict[str, Any]) -> str:
    """Return the exported ID attribute."""

    value = shape_attributes(shape).get("ID")
    if value is None or not value:
        raise ValueError("shape is missing a non-empty ID attribute")
    return value


def hidden_value(shape: dict[str, Any]) -> str:
    """Return the exported Hidden attribute."""

    return shape_attributes(shape).get("Hidden", "No")


def set_id_value(shape: dict[str, Any], value: str) -> None:
    """Set an existing ID attribute without changing any other payload."""

    for attribute in shape.get("attributes", []):
        if attribute.get("name") == "ID":
            attribute["value"] = value
            return
    raise ValueError("shape is missing the ID attribute")


def contiguous_runs(frames: list[int]) -> list[tuple[int, int]]:
    """Return inclusive contiguous frame runs."""

    if not frames:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = frames[0]
    for frame in frames[1:]:
        if frame == previous + 1:
            previous = frame
            continue
        runs.append((start, previous))
        start = previous = frame
    runs.append((start, previous))
    return runs


def identity_shape_key(shape: dict[str, Any]) -> tuple[str, int]:
    """Return the immutable key for an identity-payload replay."""

    return str(shape["label"]), int(shape["frame"])


def payload_without_id(shape: dict[str, Any]) -> dict[str, Any]:
    """Return every exported payload field except the allowed ID attribute."""

    clean = strip_internal_shape_keys(shape)
    attributes = [
        item
        for item in clean.get("attributes", [])
        if str(item.get("name")) != "ID"
    ]
    return {**clean, "attributes": attributes}


def parent_has_old_repair(
    hidden_frames: dict[int, dict[str, Any]],
    partner_frames: dict[int, dict[str, Any]],
    hidden_id: int,
    partner_id: int,
    old_start: int,
    suffix_frames: list[int],
) -> bool:
    """Recognize a fully applied old hidden-suffix ID repair."""

    before = old_start - 1
    if before in hidden_frames and before in partner_frames:
        if id_value(hidden_frames[before]) != f"ID_{hidden_id}":
            return False
        if id_value(partner_frames[before]) != f"ID_{partner_id}":
            return False
    return all(
        id_value(hidden_frames[frame]) == f"ID_{partner_id}"
        and id_value(partner_frames[frame]) == f"ID_{hidden_id}"
        for frame in suffix_frames
    )


def first_persistent_overlap_boundary(
    overlap_by_frame: dict[int, float],
    min_overlap_iou: float,
    persistence_frames: int,
) -> int | None:
    """Return the last frame of the first qualifying overlap run."""

    if persistence_frames <= 0:
        raise ValueError("overlap persistence must be positive")
    qualifying = sorted(
        frame
        for frame, overlap in overlap_by_frame.items()
        if overlap >= min_overlap_iou
    )
    for run_start, run_end in contiguous_runs(qualifying):
        if run_end - run_start + 1 >= persistence_frames:
            return run_start + persistence_frames - 1
    return None


def replay_hidden_suffix_commit_boundary(
    shapes: list[dict[str, Any]],
    cfg: ReplayConfig,
    *,
    candidate_name: str = CANDIDATE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Move an applied hidden-suffix ID commit to a frozen boundary."""

    candidate_shapes = copy.deepcopy(shapes)
    by_label_frame: dict[int, dict[int, dict[str, Any]]] = {}
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for shape in candidate_shapes:
        if bool(shape.get("outside", False)):
            continue
        fixed_id = label_identity(shape)
        frame = int(shape["frame"])
        if frame in by_label_frame.setdefault(fixed_id, {}):
            raise ValueError("identity replay found duplicate label/frame keys")
        by_label_frame[fixed_id][frame] = shape
        by_frame.setdefault(frame, []).append(shape)

    events: list[dict[str, Any]] = []
    consumed_pairs: set[tuple[int, int]] = set()
    for hidden_id, hidden_frames in sorted(by_label_frame.items()):
        hidden_frame_numbers = sorted(
            frame
            for frame, shape in hidden_frames.items()
            if hidden_value(shape) == "Yes"
        )
        for run_start, run_end in contiguous_runs(hidden_frame_numbers):
            run_length = run_end - run_start + 1
            if run_length < cfg.min_hidden_frames:
                continue
            if cfg.max_hidden_frames > 0 and run_length > cfg.max_hidden_frames:
                continue
            persistence_frames = getattr(
                cfg,
                "min_overlap_persistence_frames",
                None,
            )
            reappearance_frame = run_end + 1
            reappeared = hidden_frames.get(reappearance_frame)
            if (
                persistence_frames is None
                and (
                    reappeared is None
                    or hidden_value(reappeared) == "Yes"
                )
            ):
                continue
            hidden_run_shapes = [hidden_frames[f] for f in range(run_start, run_end + 1)]
            median_score = statistics.median(
                float(shape.get("score", 0.0)) for shape in hidden_run_shapes
            )
            if median_score > cfg.max_hidden_median_score:
                continue

            partner_overlaps: dict[int, list[float]] = {}
            partner_overlap_by_frame: dict[int, dict[int, float]] = {}
            for hidden_shape in hidden_run_shapes:
                frame = int(hidden_shape["frame"])
                for other in by_frame.get(frame, []):
                    partner_id = label_identity(other)
                    if partner_id == hidden_id or hidden_value(other) == "Yes":
                        continue
                    overlap = bbox_iou(
                        hidden_shape["points"],
                        other["points"],
                    )
                    partner_overlaps.setdefault(partner_id, []).append(
                        overlap
                    )
                    partner_overlap_by_frame.setdefault(partner_id, {})[
                        frame
                    ] = overlap
            if not partner_overlaps:
                continue
            partner_id, overlaps = max(
                partner_overlaps.items(),
                key=lambda item: max(item[1]) if item[1] else 0.0,
            )
            if max(overlaps) < cfg.min_overlap_iou:
                continue
            pair = tuple(sorted((hidden_id, partner_id)))
            if pair in consumed_pairs:
                continue
            partner_frames = by_label_frame[partner_id]

            old_start = max(run_start, run_end - cfg.start_back_frames)
            if persistence_frames is None:
                new_start = reappearance_frame
                partner_reappeared = partner_frames.get(new_start)
                if partner_reappeared is None:
                    continue
                if hidden_value(partner_reappeared) == "Yes":
                    continue
            else:
                new_start = first_persistent_overlap_boundary(
                    partner_overlap_by_frame[partner_id],
                    cfg.min_overlap_iou,
                    persistence_frames,
                )
                if new_start is None or new_start <= old_start:
                    continue
            suffix_frames = sorted(
                frame
                for frame in set(hidden_frames) & set(partner_frames)
                if frame >= old_start
            )
            if len(suffix_frames) < cfg.min_suffix_frames:
                continue
            if not parent_has_old_repair(
                hidden_frames,
                partner_frames,
                hidden_id,
                partner_id,
                old_start,
                suffix_frames,
            ):
                continue

            changed_frames = [
                frame
                for frame in suffix_frames
                if old_start <= frame < new_start
            ]
            if not changed_frames:
                continue
            for frame in changed_frames:
                set_id_value(hidden_frames[frame], f"ID_{hidden_id}")
                set_id_value(partner_frames[frame], f"ID_{partner_id}")
                for shape in (hidden_frames[frame], partner_frames[frame]):
                    shape["_identity_payload_replay"] = candidate_name
            event = {
                "hidden_label": f"Pig_{hidden_id}",
                "partner_label": f"Pig_{partner_id}",
                "run_start": run_start,
                "run_end": run_end,
                "run_length": run_length,
                "hidden_median_score": round(float(median_score), 6),
                "max_partner_iou": round(float(max(overlaps)), 6),
                "old_commit_start": old_start,
                "candidate_commit_start": new_start,
                "common_suffix_frames": len(suffix_frames),
                "changed_frames": changed_frames,
            }
            if persistence_frames is not None:
                event["overlap_persistence_frames"] = persistence_frames
                event["overlap_persistence_start"] = (
                    new_start - persistence_frames + 1
                )
            events.append(event)
            consumed_pairs.add(pair)
    return candidate_shapes, events


def build_delta_rows(
    parents: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one audit row for each changed ID attribute."""

    event_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for event in events:
        for frame in event["changed_frames"]:
            event_by_key[(event["hidden_label"], frame)] = event
            event_by_key[(event["partner_label"], frame)] = event
    rows: list[dict[str, Any]] = []
    for parent, candidate in zip(parents, candidates, strict=True):
        if id_value(parent) == id_value(candidate):
            continue
        key = identity_shape_key(candidate)
        event = event_by_key.get(key)
        if event is None:
            raise ValueError("changed ID row is not linked to a replay event")
        rows.append(
            {
                "frame": int(candidate["frame"]),
                "label": str(candidate["label"]),
                "parent_id": id_value(parent),
                "candidate_id": id_value(candidate),
                "hidden": hidden_value(candidate),
                "score": float(candidate.get("score", 0.0)),
                "hidden_label": event["hidden_label"],
                "partner_label": event["partner_label"],
                "run_start": event["run_start"],
                "run_end": event["run_end"],
                "old_commit_start": event["old_commit_start"],
                "candidate_commit_start": event["candidate_commit_start"],
            }
        )
    return rows


def write_delta_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write deterministic identity-payload delta evidence."""

    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DELTA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def resolve_plan_path(raw_path: str, label: str) -> Path:
    """Resolve a plan-owned path relative to the repository root."""

    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return require_file(path, label)


def require_same_path(actual: Path, expected: Path, label: str) -> None:
    """Require two paths to resolve to the same file."""

    if actual.resolve() != expected.resolve():
        raise ValueError(f"{label} path does not match the frozen plan")


def require_sha256(path: Path, expected: str, label: str) -> None:
    """Require an exact lowercase SHA256 from the frozen plan."""

    if len(expected) != 64 or expected.lower() != expected:
        raise ValueError(f"{label} plan SHA256 is malformed")
    if file_sha256(path) != expected:
        raise ValueError(f"{label} SHA256 does not match the frozen plan")


def require_git_ancestor(ancestor: str, descendant: str, label: str) -> None:
    """Require one frozen commit to be an ancestor of the current commit."""

    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"{label} is not an ancestor of the current commit")


def validate_current_lineage(plan: dict[str, Any]) -> dict[str, Any]:
    """Require a clean commit descended from the frozen experiment start."""

    state = git_state()
    if state["dirty"]:
        raise ValueError("identity replay requires a clean worktree")
    starting_commit = str(plan.get("starting_commit", ""))
    parent_commit = str(plan.get("parent", {}).get("git_commit", ""))
    require_git_ancestor(starting_commit, state["commit"], "starting commit")
    require_git_ancestor(parent_commit, state["commit"], "parent commit")
    return state


def validate_frozen_window(
    plan: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Path, dict[str, str], tuple[int, int]]:
    """Bind scoring to the parent-derived event row frozen in the plan."""

    window = plan.get("frozen_window", {})
    window_path = resolve_plan_path(
        str(window.get("path", "")),
        "frozen event-window manifest",
    )
    legacy_path = resolve_plan_path(
        str(Path(window.get("directory", "")) / window.get("filename", "")),
        "frozen event-window manifest",
    )
    require_same_path(window_path, legacy_path, "event-window manifest")
    require_sha256(
        window_path,
        str(window.get("manifest_sha256", "")),
        "event-window manifest",
    )
    with window_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    video = str(plan.get("parent", {}).get("video", ""))
    event_id = str(window.get("event_id", ""))
    matches = [
        row
        for row in rows
        if row.get("video_stem") == video
        and row.get("episode_id") == event_id
    ]
    if len(matches) != 1:
        raise ValueError("frozen event-window row is missing or duplicated")
    row = matches[0]
    score_frames = window.get("score_frames", [])
    if not isinstance(score_frames, list) or len(score_frames) != 2:
        raise ValueError("frozen score_frames must contain start and end")
    score_window = (int(score_frames[0]), int(score_frames[1]))
    csv_score_window = (
        int(row["score_start_frame"]),
        int(row["score_end_frame"]),
    )
    if score_window != csv_score_window:
        raise ValueError("plan score window does not match frozen manifest")
    validate_frame_window(args)
    cli_window = (args.score_start_frame, args.score_end_frame)
    if cli_window[0] is not None and cli_window != score_window:
        raise ValueError("CLI score window cannot override the frozen plan")
    switch_frames = window.get("switch_frames", [])
    if switch_frames != [
        int(row["first_switch_frame"]),
        int(row["last_switch_frame"]),
    ]:
        raise ValueError("plan switch frames do not match frozen manifest")
    if int(window.get("parent_remapped_idsw", -1)) != int(
        row["switch_event_rows"]
    ):
        raise ValueError("plan parent IDSW does not match frozen manifest")
    return window_path, row, score_window


def validated_replay_config(
    plan: dict[str, Any],
    args: argparse.Namespace,
) -> ReplayConfig:
    """Build the candidate config and reject every CLI parameter drift."""

    candidate_name = str(plan.get("candidate", {}).get("name", ""))
    config_type: type[ReplayConfig]
    if candidate_name == CANDIDATE_PERSISTENT_OVERLAP:
        config_type = PersistenceReplayConfig
    elif candidate_name == CANDIDATE_AFTER_RUN:
        config_type = ReplayConfig
    else:
        raise ValueError("candidate does not have a supported config")
    parameters = plan.get("candidate", {}).get("parameters", {})
    expected_fields = set(config_type.__dataclass_fields__)
    if set(parameters) != expected_fields:
        raise ValueError("candidate parameters do not match its config")
    cfg = config_type(**parameters)
    if cfg.max_hidden_frames < cfg.min_hidden_frames:
        raise ValueError("max_hidden_frames must be >= min_hidden_frames")
    for field, expected in asdict(cfg).items():
        supplied = getattr(args, field, None)
        if supplied is not None and supplied != expected:
            raise ValueError(f"CLI parameter {field} differs from frozen plan")
    return cfg


def validate_parent_manifest(
    plan: dict[str, Any],
    parent_manifest: Path,
    parent_json: Path,
    parent_xml: Path,
    source_video: Path,
) -> dict[str, Any]:
    """Bind parent files, config, GT lineage, and source video to the plan."""

    parent = plan.get("parent", {})
    require_same_path(
        parent_manifest,
        resolve_plan_path(parent.get("run_manifest_path", ""), "parent manifest"),
        "parent manifest",
    )
    require_same_path(
        parent_json,
        resolve_plan_path(
            parent.get("prediction_shapes_path", ""),
            "parent shapes JSON",
        ),
        "parent shapes JSON",
    )
    require_same_path(
        parent_xml,
        resolve_plan_path(parent.get("prediction_xml_path", ""), "parent XML"),
        "parent XML",
    )
    require_sha256(
        parent_manifest,
        str(parent.get("run_manifest_sha256", "")),
        "parent manifest",
    )
    require_sha256(
        parent_json,
        str(parent.get("shapes_sha256", "")),
        "parent shapes JSON",
    )
    require_sha256(parent_xml, str(parent.get("xml_sha256", "")), "parent XML")

    payload = json.loads(parent_manifest.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise ValueError("parent run manifest status must be completed")
    parent_git = payload.get("git", {})
    if parent_git.get("dirty") or parent_git.get("dirty_entries"):
        raise ValueError("parent run manifest must record a clean worktree")
    if parent_git.get("commit") != parent.get("git_commit"):
        raise ValueError("parent run commit does not match the frozen plan")

    semantic = payload.get("semantic_config", {})
    semantic_hash = stable_json_sha256(semantic)
    if semantic_hash != payload.get("semantic_config_sha256"):
        raise ValueError("parent semantic config hash is invalid")
    contract = plan.get("evaluation_contract", {})
    required_semantics = {
        "include_hidden": True,
        "tracking_mode": parent.get("mode"),
        "iou_threshold": contract.get("iou_threshold"),
        "gap_tolerance_frames": contract.get("gap_tolerance_frames"),
        "USE_IOU_FALLBACK": False,
        "USE_AREA_OCCLUSION_FREEZE": False,
        "USE_CONDITIONAL_AREA_OCCLUSION_FREEZE": False,
        "USE_MERGED_BOX_SPLIT": False,
    }
    for key, expected in required_semantics.items():
        if semantic.get(key) != expected:
            raise ValueError(f"parent semantic config mismatch: {key}")
    if parent.get("rule_combo") != contract.get("rule_combo"):
        raise ValueError("parent rule combo differs from evaluation contract")

    pairs = payload.get("inputs", {}).get("pairs", [])
    matches = [
        pair for pair in pairs if pair.get("video_stem") == parent.get("video")
    ]
    if len(matches) != 1:
        raise ValueError("parent video pair is missing or duplicated")
    pair = matches[0]
    prediction_record = pair.get("prediction_xml", {})
    if prediction_record.get("sha256") != parent.get("xml_sha256"):
        raise ValueError("parent manifest prediction hash differs from plan")
    require_same_path(
        parent_xml,
        require_file(Path(prediction_record.get("path", "")), "manifest XML"),
        "parent manifest prediction",
    )
    video_record = pair.get("video", {})
    require_same_path(
        source_video,
        require_file(Path(video_record.get("path", "")), "manifest video"),
        "source video",
    )
    require_sha256(
        source_video,
        str(video_record.get("sha256", "")),
        "source video",
    )
    if pair.get("gt_xml", {}).get("sha256") != parent.get("gt_sha256"):
        raise ValueError("parent GT hash differs from the frozen plan")
    return payload


def validate_written_candidate(
    candidate_json: Path,
    candidate_xml: Path,
    expected_shapes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reparse candidate files and require exact exported payload integrity."""

    written_shapes = read_parent_shapes(candidate_json)
    expected_clean = [strip_internal_shape_keys(shape) for shape in expected_shapes]
    if written_shapes != expected_clean:
        raise ValueError("written candidate JSON differs from in-memory payload")
    expected_xml = [exported_shape_record(shape) for shape in expected_shapes]
    expected_xml.sort(key=lambda item: (item["label"], item["frame"]))
    if read_xml_shape_records(candidate_xml) != expected_xml:
        raise ValueError("written candidate XML differs from candidate JSON")
    return written_shapes


def validate_plan(path: Path, candidate: str) -> dict[str, Any]:
    """Require a frozen H5 plan that names this replay candidate."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_to_candidate = {
        "tracking_h5_hidden_suffix_commit_plan_v1": CANDIDATE_AFTER_RUN,
        "tracking_h5_hidden_suffix_persistence_plan_v1": (
            CANDIDATE_PERSISTENT_OVERLAP
        ),
    }
    schema = payload.get("schema_version")
    if schema not in schema_to_candidate:
        raise ValueError("experiment plan schema is not supported")
    if payload.get("status") != "frozen_before_replay":
        raise ValueError("experiment plan must be frozen_before_replay")
    if payload.get("priority") != "hybrid_bytetrack_residual_first":
        raise ValueError("experiment plan does not keep hybrid first")
    if payload.get("candidate", {}).get("name") != candidate:
        raise ValueError("experiment plan candidate does not match replay")
    if schema_to_candidate[schema] != candidate:
        raise ValueError("experiment plan schema does not match replay candidate")
    candidate_contract = payload.get("candidate", {})
    if candidate_contract.get("gt_used_to_generate_prediction") is not False:
        raise ValueError("experiment plan must forbid GT-generated prediction")
    if candidate_contract.get("video_or_frame_hardcode_allowed") is not False:
        raise ValueError("experiment plan must forbid video/frame hardcoding")
    contract = payload.get("identity_replay_contract", {})
    if contract.get("allowed_payload_change") != ["ID attribute"]:
        raise ValueError("experiment plan does not isolate the ID attribute")
    if contract.get("geometry_replay_contract_allowed") is not False:
        raise ValueError("identity replay cannot use the geometry contract")
    evaluation = payload.get("evaluation_contract", {})
    expected_evaluation = {
        "include_hidden": True,
        "hidden_is_optimization_target": False,
        "rule_combo": "iou0_area0_condarea0_merge0",
        "generated_mp4_allowed": False,
        "classification_scope_allowed": False,
    }
    for key, expected in expected_evaluation.items():
        if evaluation.get(key) != expected:
            raise ValueError(f"experiment evaluation contract mismatch: {key}")
    if candidate == CANDIDATE_PERSISTENT_OVERLAP:
        parameters = candidate_contract.get("parameters", {})
        persistence = parameters.get("min_overlap_persistence_frames")
        if not isinstance(persistence, int) or persistence < 2:
            raise ValueError("persistent-overlap plan requires at least two frames")
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Validate, replay and optionally write candidate artifacts."""

    parent_json = require_file(args.parent_shapes_json, "parent shapes JSON")
    parent_xml = require_file(args.parent_xml, "parent XML")
    parent_manifest = require_file(
        args.parent_run_manifest,
        "parent run manifest",
    )
    experiment_plan = require_file(args.experiment_plan, "experiment plan")
    source_video = require_file(args.source_video, "source video")
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to reuse output directory: {args.output_dir}"
        )

    plan = validate_plan(experiment_plan, args.candidate)
    current_git = validate_current_lineage(plan)
    window_path, window_row, score_window = validate_frozen_window(plan, args)
    parent_manifest_payload = validate_parent_manifest(
        plan,
        parent_manifest,
        parent_json,
        parent_xml,
        source_video,
    )

    parent_shapes = read_parent_shapes(parent_json)
    parent_geometry_contract(parent_shapes, parent_xml)
    cfg = validated_replay_config(plan, args)
    candidate_shapes, events = replay_hidden_suffix_commit_boundary(
        parent_shapes,
        cfg,
        candidate_name=args.candidate,
    )
    if len(parent_shapes) != len(candidate_shapes):
        raise ValueError("identity replay changed the shape count")
    parent_keys = [identity_shape_key(shape) for shape in parent_shapes]
    candidate_keys = [identity_shape_key(shape) for shape in candidate_shapes]
    if parent_keys != candidate_keys or len(set(parent_keys)) != len(parent_keys):
        raise ValueError("identity replay changed or duplicated shape keys")
    parent_payload = [payload_without_id(shape) for shape in parent_shapes]
    candidate_payload = [payload_without_id(shape) for shape in candidate_shapes]
    if parent_payload != candidate_payload:
        raise ValueError("identity replay changed payload outside ID")

    delta_rows = build_delta_rows(parent_shapes, candidate_shapes, events)
    if not delta_rows:
        raise ValueError("identity replay changed zero ID rows")
    score_rows = [
        row
        for row in delta_rows
        if score_window[0] <= row["frame"] <= score_window[1]
    ]
    if not score_rows:
        raise ValueError("identity replay changed zero IDs in score window")

    summary = {
        "shape_count": len(parent_shapes),
        "event_count": len(events),
        "changed_id_row_count": len(delta_rows),
        "score_window_changed_id_row_count": len(score_rows),
        "changed_labels": sorted({row["label"] for row in delta_rows}),
        "non_id_payload_sha256": stable_json_sha256(parent_payload),
        "frozen_event_id": window_row["episode_id"],
        "score_start_frame": score_window[0],
        "score_end_frame": score_window[1],
        "events": events,
    }
    if args.dry_run:
        return {
            "status": "dry_run_pass",
            "git": current_git,
            "inputs": {
                "parent_shapes_json": input_record(parent_json),
                "parent_xml": input_record(parent_xml),
                "parent_run_manifest": input_record(parent_manifest),
                "experiment_plan": input_record(experiment_plan),
                "frozen_window_manifest": input_record(window_path),
                "source_video": input_record(source_video),
            },
            "summary": summary,
        }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    candidate_json = args.output_dir / "annotations_cvat_shapes.json"
    candidate_xml = args.output_dir / "annotations_cvat_video_1_1.xml"
    delta_csv = args.output_dir / "identity_delta.csv"
    manifest_path = args.output_dir / "identity_replay_manifest.json"
    write_annotation_json(candidate_json, candidate_shapes)
    width, height = frame_size_from_parent_xml(parent_xml)
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
    written_shapes = validate_written_candidate(
        candidate_json,
        candidate_xml,
        candidate_shapes,
    )
    written_non_id_payload = [
        payload_without_id(shape) for shape in written_shapes
    ]
    if written_non_id_payload != parent_payload:
        raise ValueError("written candidate changed payload outside ID")
    if list(args.output_dir.rglob("*.mp4")):
        raise RuntimeError("identity replay output contains an MP4")

    manifest = {
        "schema_version": 1,
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": args.candidate,
        "command": list(sys.argv),
        "selected_skills": SELECTED_SKILLS,
        "git": current_git,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "runtime_contract": RUNTIME_CONTRACT,
        "tracker_runtime_status": "NOT_APPLICABLE",
        "gt_used_to_generate_prediction": False,
        "config": {
            "include_hidden": True,
            **asdict(cfg),
            "score_start_frame": score_window[0],
            "score_end_frame": score_window[1],
        },
        "inputs": {
            "parent_shapes_json": input_record(parent_json),
            "parent_xml": input_record(parent_xml),
            "parent_run_manifest": input_record(parent_manifest),
            "experiment_plan": input_record(experiment_plan),
            "frozen_window_manifest": input_record(window_path),
            "source_video": input_record(source_video),
        },
        "parent_lineage": {
            "run_commit": parent_manifest_payload["git"]["commit"],
            "semantic_config_sha256": parent_manifest_payload[
                "semantic_config_sha256"
            ],
            "event_id": window_row["episode_id"],
        },
        "parent_prediction_semantic_sha256": (
            cvat_prediction_semantic_sha256(parent_xml)
        ),
        "summary": summary,
        "payload_integrity": {
            "status": "PASS",
            "shape_keys_equal": True,
            "non_id_payload_equal": True,
            "candidate_json_reparsed": True,
            "candidate_xml_reparsed": True,
            "candidate_json_xml_equal": True,
            "non_id_payload_sha256": stable_json_sha256(parent_payload),
            "allowed_change": ["ID attribute"],
        },
        "outputs": {
            "candidate_shapes_json": output_record(candidate_json),
            "candidate_xml": output_record(candidate_xml),
            "identity_delta_csv": output_record(delta_csv),
        },
        "recursive_no_mp4": {"status": "PASS", "mp4_count": 0},
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
    """Run the identity-payload replay."""

    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
