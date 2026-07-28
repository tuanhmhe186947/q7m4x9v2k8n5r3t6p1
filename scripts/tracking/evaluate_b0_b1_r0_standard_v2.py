"""Freeze Standard-V2 metric authority for immutable B0, B1, and R0 XMLs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pig_behavior.evaluation.tracking.assets import video_metadata  # noqa: E402
from pig_behavior.evaluation.tracking.contracts import (  # noqa: E402
    EVALUATOR_CONTRACT_ID,
    HOTA_ALPHAS,
    IDENTITY_EPISODE_CONTRACT_ID,
    IDSW_POLICY,
    MATCHING_CONTRACT_ID,
    REFERENCE_PARITY_PASS,
    SEQUENCE_BOUNDARY_POLICY,
    resolve_evaluator_code_sha,
)
from pig_behavior.evaluation.tracking.cvat_io import (  # noqa: E402
    box_hidden,
    box_id,
    is_outside,
    parse_cvat_video_xml,
)
from pig_behavior.evaluation.tracking.evaluator_standard_v2 import (  # noqa: E402
    StandardV2Evaluation,
    aggregate_tracking_standard_v2,
    evaluate_tracking_standard_v2,
)
from pig_behavior.evaluation.tracking.lineage import (  # noqa: E402
    cvat_prediction_semantic_sha256,
)
from pig_behavior.evaluation.tracking.reporting_standard_v2 import (  # noqa: E402
    hota_alpha_dataframe,
    identity_ambiguity_dataframe,
    identity_authority_dataframe,
    identity_episode_dataframe,
    pairwise_swap_dataframe,
)

DATE = "20260728"
STARTING_MAIN_SHA = "3b0311774e80c066e9e517a9b3d5a0a6acb1d0a7"
EXPECTED_FRAMES = 1800
EXPECTED_VIDEOS = 13
PREDICTION_HASH_CONTRACT = "tracking_prediction_xml_set_sha256_v1"
SOURCE_MANIFEST_SHA256 = (
    "91289c9acb40958e59c17e98872714904f8df7e4c49682a4649e7fcc84bab9be"
)
GT_AUTHORITY_SHA256 = (
    "675cf37c4f924e391ffa457ba6c6e9453b967af318f37ecd2bc1ab1190a1d9dd"
)
FULL_CACHE_AUTHORITY_SHA256 = (
    "494d17ddb9d592dcf2105fd89a7204181f99a49353614315216d30ea43716e00"
)
DETECTOR_WEIGHTS_SHA256 = (
    "6b57d95b82f8715ab7525efe7524feab6d55a50bc0376355dc7ea208ada49fed"
)
DETECTOR_CONFIG_SHA256 = (
    "2b50d8afa950626e2bed6b41807cb602a01a90e66baf7529fa08945d3d676ef8"
)
REQUIRED_REPEAT_FILES = (
    "B0_B1_R0_STANDARD_V2_AGGREGATE_METRICS.csv",
    "B0_B1_R0_STANDARD_V2_PER_VIDEO_METRICS.csv",
    "B0_B1_R0_STANDARD_V2_PER_ALPHA_METRICS.csv",
    "B0_B1_R0_IDENTITY_ERROR_EPISODES.csv",
    "B0_B1_R0_PERSISTENT_PAIRWISE_SWAPS.csv",
    "B0_B1_R0_EXPOSURE_NORMALIZED_METRICS.csv",
)
METRIC_COLUMNS = (
    "hota",
    "deta",
    "assa",
    "loca",
    "idf1",
    "id_precision",
    "id_recall",
    "idsw_standard",
    "fp",
    "fn",
    "fragments",
    "wrong_id_matched_frames",
    "wrong_id_matched_seconds",
    "identity_error_episode_count",
    "recovered_identity_error_episode_count",
    "terminal_identity_error_episode_count",
    "persistent_pairwise_identity_swap_count",
)
EXPOSURE_COLUMNS = (
    "authoritative_matched_gt_frames",
    "idsw_standard_per_1000_authoritative_matched_gt_frames",
    "wrong_id_matched_frames",
    "wrong_id_matched_seconds",
    "wrong_id_matched_frames_per_1000_authoritative_matched_gt_frames",
    "authoritative_gt_trajectories",
    "gt_trajectories_with_identity_error_count",
    "gt_trajectories_with_identity_error_pct",
    "videos_with_terminal_episode_count",
    "videos_with_terminal_episode_pct",
    "videos_with_persistent_pairwise_swap_count",
    "videos_with_persistent_pairwise_swap_pct",
)


class EvaluationAuthorityError(RuntimeError):
    """Fail-closed error for frozen-input or authority mismatches."""


@dataclass(frozen=True)
class ArmSpec:
    """Immutable inputs and scientific metadata for one arm."""

    arm: str
    profile: str
    prediction_root: Path
    authority_path: Path
    artifact_sha256: str
    config_sha256: str
    detector_cadence: str
    detector_authority_sha256: str


def sha256_file(path: Path) -> str:
    """Hash one file without changing it."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    """Hash JSON-compatible content with the repository authority contract."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and fail on any other top-level type."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EvaluationAuthorityError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    """Write deterministic, newline-terminated JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_csv(dataframe: pd.DataFrame, path: Path) -> None:
    """Write a deterministic CSV after canonical row ordering."""

    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="",
    )


def git_sha(repo: Path) -> str:
    """Resolve the clean evaluation-tool commit."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def arm_specs(source_repo: Path, worktree_repo: Path) -> tuple[ArmSpec, ...]:
    """Return the three frozen arm definitions."""

    frozen_root = (
        source_repo
        / "outputs"
        / "tracking"
        / "frozen_predictions_standard_v2_20260728_retry1"
    )
    docs = worktree_repo / "docs" / "tracking"
    return (
        ArmSpec(
            arm="B0",
            profile="bytetrack_raw",
            prediction_root=frozen_root / "B0_bytetrack_raw" / "predictions",
            authority_path=(
                docs
                / "b0_b1_prediction_authority"
                / "B0_BYTETRACK_RAW_PREDICTION_AUTHORITY_20260728.json"
            ),
            artifact_sha256=(
                "13d9226c36141264cc33e4b498d38e5f3eaa9891cf32bc4c8"
                "fb87b01fd27d576"
            ),
            config_sha256=(
                "547ae86e3be26671a9a148cb0e613ea1c602a0ff842a977ce9b7"
                "f1d217c10e41"
            ),
            detector_cadence="EVERY_FRAME",
            detector_authority_sha256=FULL_CACHE_AUTHORITY_SHA256,
        ),
        ArmSpec(
            arm="B1",
            profile="hybrid_bytetrack",
            prediction_root=(
                frozen_root / "B1_hybrid_bytetrack" / "predictions"
            ),
            authority_path=(
                docs
                / "b0_b1_prediction_authority"
                / "B1_HYBRID_BYTETRACK_PREDICTION_AUTHORITY_20260728.json"
            ),
            artifact_sha256=(
                "569c49e00905add068fac70c919fe21c10127e3ab773528a4ac44"
                "199fcb4835b"
            ),
            config_sha256=(
                "4eb3d4e2262485d48d425be06fd8a6b3adfd8a01a27b28e76b5"
                "a8d55958d1d55"
            ),
            detector_cadence="EVERY_FRAME",
            detector_authority_sha256=FULL_CACHE_AUTHORITY_SHA256,
        ),
        ArmSpec(
            arm="R0",
            profile="realtime_fast",
            prediction_root=(
                source_repo
                / "outputs"
                / "tracking"
                / "current_main_baseline_20260728"
                / "predictions"
            ),
            authority_path=(
                docs / "CURRENT_MAIN_R0_BASELINE_AUTHORITY_20260728.json"
            ),
            artifact_sha256=(
                "fd2d4f3dec0710d1c9eecba9308247a7b226dd34a4a02a9cb89"
                "f17acb22bbbfe"
            ),
            config_sha256=(
                "9bf4ce6d07423ab517b4705c716e3eb012349b756b7c0591cc345"
                "8eac207808d"
            ),
            detector_cadence="EVERY_2_FRAMES",
            detector_authority_sha256=(
                "795df7732393e4e258a82db58e29101b068cf8ac3583acf7702e0"
                "afdaeec6e7a"
            ),
        ),
    )


def prediction_path(arm: ArmSpec, video_key: str) -> Path:
    """Resolve the authority-bound prediction XML for one arm/video."""

    if arm.arm in {"B0", "B1"}:
        return arm.prediction_root / f"{video_key}.xml"
    return (
        arm.prediction_root
        / video_key
        / "realtime"
        / video_key
        / "annotations_cvat_video_1_1.xml"
    )


def _raw_active_rows(path: Path) -> list[tuple[Any, ...]]:
    """Read active CVAT rows without representation mutation."""

    root = ET.parse(path).getroot()
    rows: list[tuple[Any, ...]] = []
    for track in root.findall("./track"):
        track_id = str(track.attrib.get("id", ""))
        label = str(track.attrib.get("label", ""))
        for box in track.findall("./box"):
            if is_outside(box):
                continue
            frame = int(box.attrib["frame"])
            rows.append(
                (
                    frame,
                    box_id(box, label, track_id),
                    float(box.attrib["xtl"]),
                    float(box.attrib["ytl"]),
                    float(box.attrib["xbr"]),
                    float(box.attrib["ybr"]),
                    box_hidden(box),
                    track_id,
                    label,
                )
            )
    return sorted(rows)


def adapter_audit(path: Path) -> dict[str, Any]:
    """Prove the CVAT adapter only parses and canonicalizes representation."""

    raw_rows = _raw_active_rows(path)
    parsed = parse_cvat_video_xml(
        path,
        include_hidden=True,
        start_frame=0,
        end_frame=EXPECTED_FRAMES - 1,
    )
    parsed_rows = sorted(
        (
            frame,
            obj.obj_id,
            *obj.bbox,
            obj.hidden,
            obj.source_track_id,
            obj.label,
        )
        for frame, objects in parsed.items()
        for obj in objects
    )
    if raw_rows != parsed_rows:
        raise EvaluationAuthorityError(f"Prediction adapter changed rows: {path}")
    return {
        "active_xml_rows": len(raw_rows),
        "parsed_rows": len(parsed_rows),
        "adapter_bbox_changes": 0,
        "adapter_id_changes": 0,
        "adapter_row_additions": 0,
        "adapter_row_removals": 0,
        "adapter_frame_index_changes": 0,
        "adapter_timestamp_changes": 0,
        "adapter_prediction_hidden_changes": 0,
        "canonical_row_sha256": canonical_hash(parsed_rows),
    }


def prediction_structural_record(
    path: Path,
    *,
    video_key: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Reproduce the frozen B0/B1 prediction-set hash contract."""

    root = ET.parse(path).getroot()
    if root.findtext("./meta/task/name") != video_key:
        raise EvaluationAuthorityError(f"Prediction task-name mismatch: {path}")
    if root.findtext("./meta/task/size") != str(EXPECTED_FRAMES):
        raise EvaluationAuthorityError(f"Prediction task-size mismatch: {path}")
    rows: list[tuple[Any, ...]] = []
    frames: set[int] = set()
    for track in root.findall("./track"):
        track_id = int(track.attrib["id"])
        label = str(track.attrib.get("label", ""))
        for box in track.findall("./box"):
            frame = int(box.attrib["frame"])
            coords = tuple(
                float(box.attrib[name])
                for name in ("xtl", "ytl", "xbr", "ybr")
            )
            if frame < 0 or frame >= EXPECTED_FRAMES:
                raise EvaluationAuthorityError(
                    f"Prediction frame outside authority: {path}"
                )
            if not all(math.isfinite(value) for value in coords):
                raise EvaluationAuthorityError(f"Non-finite bbox: {path}")
            xtl, ytl, xbr, ybr = coords
            if not (
                0 <= xtl <= xbr <= width
                and 0 <= ytl <= ybr <= height
            ):
                raise EvaluationAuthorityError(f"Bbox outside frame: {path}")
            attributes = tuple(
                sorted(
                    (
                        str(attribute.attrib.get("name", "")),
                        str(attribute.text or ""),
                    )
                    for attribute in box.findall("./attribute")
                )
            )
            rows.append(
                (
                    track_id,
                    label,
                    frame,
                    xtl,
                    ytl,
                    xbr,
                    ybr,
                    attributes,
                )
            )
            frames.add(frame)
    canonical_rows = sorted(rows)
    return {
        "video_key": video_key,
        "prediction_xml_sha256": sha256_file(path),
        "prediction_semantic_sha256": cvat_prediction_semantic_sha256(path),
        "canonical_row_sha256": canonical_hash(canonical_rows),
        "prediction_object_count": len(canonical_rows),
        "processed_frame_count": EXPECTED_FRAMES,
        "minimum_prediction_frame": min(frames) if frames else None,
        "maximum_prediction_frame": max(frames) if frames else None,
    }


def prediction_set_hash(records: list[dict[str, Any]]) -> str:
    """Hash the semantic prediction population independently of input order."""

    payload = [
        {
            "video_key": record["video_key"],
            "prediction_xml_sha256": record["prediction_xml_sha256"],
            "prediction_semantic_sha256": record[
                "prediction_semantic_sha256"
            ],
            "canonical_row_sha256": record["canonical_row_sha256"],
            "prediction_object_count": record["prediction_object_count"],
            "processed_frame_count": record["processed_frame_count"],
        }
        for record in sorted(records, key=lambda item: item["video_key"])
    ]
    return canonical_hash(
        {
            "contract": PREDICTION_HASH_CONTRACT,
            "predictions": payload,
        }
    )


def _r0_manifest_records(source_repo: Path) -> dict[str, dict[str, Any]]:
    manifest = (
        source_repo
        / "outputs"
        / "tracking"
        / "current_main_baseline_20260728"
        / "ARTIFACT_SHA256.json"
    )
    if sha256_file(manifest) != (
        "461f2300318ab26134bb36beb78957a6642cbd361de9d74492f9cd09db688223"
    ):
        raise EvaluationAuthorityError("R0 artifact manifest hash mismatch")
    payload = load_json(manifest)
    return {
        str(row["relative_path"]): row
        for row in payload["artifacts"]
        if str(row["relative_path"]).endswith(".xml")
    }


def _validate_arm_authority(
    arm: ArmSpec,
    authority: dict[str, Any],
) -> None:
    if arm.arm in {"B0", "B1"}:
        if authority.get("status") != "ESTABLISHED":
            raise EvaluationAuthorityError(f"{arm.arm} authority not established")
        if authority.get("canonical_prediction_content_sha256") != (
            arm.artifact_sha256
        ):
            raise EvaluationAuthorityError(f"{arm.arm} artifact hash mismatch")
        if authority.get("profile_config_sha256") != arm.config_sha256:
            raise EvaluationAuthorityError(f"{arm.arm} config hash mismatch")
        marker = arm.prediction_root.parent / (
            "FROZEN_SCIENTIFIC_AUTHORITY_DO_NOT_DELETE.txt"
        )
        if not marker.is_file():
            raise EvaluationAuthorityError(f"{arm.arm} retention marker missing")
        return
    if authority.get("r0_baseline_authority") != "ESTABLISHED":
        raise EvaluationAuthorityError("R0 authority not established")
    if authority.get("r0_config_sha256") != arm.config_sha256:
        raise EvaluationAuthorityError("R0 config hash mismatch")


def preflight(
    source_repo: Path,
    worktree_repo: Path,
) -> tuple[
    list[dict[str, Any]],
    tuple[ArmSpec, ...],
    dict[str, Any],
    dict[str, Any],
]:
    """Revalidate the complete immutable input authority before evaluation."""

    manifest_path = (
        worktree_repo
        / "docs"
        / "tracking"
        / "b0_b1_prediction_authority"
        / "B0_B1_LOCKED_EXECUTION_MANIFEST_20260728.json"
    )
    manifest = load_json(manifest_path)
    if manifest.get("video_count") != EXPECTED_VIDEOS:
        raise EvaluationAuthorityError("Locked video count mismatch")
    if manifest.get("source_video_manifest_sha256") != SOURCE_MANIFEST_SHA256:
        raise EvaluationAuthorityError("Source manifest authority mismatch")
    if manifest.get("gt_authority_sha256") != GT_AUTHORITY_SHA256:
        raise EvaluationAuthorityError("GT authority mismatch")
    arms = arm_specs(source_repo, worktree_repo)
    authorities = {
        arm.arm: load_json(arm.authority_path) for arm in arms
    }
    for arm in arms:
        _validate_arm_authority(arm, authorities[arm.arm])
    r0_records = _r0_manifest_records(source_repo)
    r0_root = (
        source_repo / "outputs" / "tracking" / "current_main_baseline_20260728"
    )
    videos: list[dict[str, Any]] = []
    adapter_rows: list[dict[str, Any]] = []
    b0_b1_structural: dict[str, list[dict[str, Any]]] = {
        "B0": [],
        "B1": [],
    }
    before_prediction_hashes: dict[str, str] = {}
    expected_keys = {str(row["video_key"]) for row in manifest["videos"]}
    if len(expected_keys) != EXPECTED_VIDEOS:
        raise EvaluationAuthorityError("Duplicate locked video key")
    for source_row in manifest["videos"]:
        row = dict(source_row)
        video_key = str(row["video_key"])
        video_path = source_repo / Path(row["source_video_path"]).name
        if not video_path.is_file():
            video_path = Path(row["source_video_path"])
        gt_path = source_repo / "data" / "annotations" / "tracking" / (
            Path(row["gt_path"]).name
        )
        if not gt_path.is_file():
            gt_path = Path(row["gt_path"])
        if sha256_file(video_path) != row["source_video_sha256"]:
            raise EvaluationAuthorityError(f"Source hash mismatch: {video_key}")
        if sha256_file(gt_path) != row["gt_sha256"]:
            raise EvaluationAuthorityError(f"GT hash mismatch: {video_key}")
        metadata = video_metadata(video_path)
        if int(metadata.get("video_frame_count", 0)) < EXPECTED_FRAMES:
            raise EvaluationAuthorityError(f"Video coverage mismatch: {video_key}")
        fps = float(metadata.get("video_fps", 0.0))
        width = int(metadata.get("video_width", 0))
        height = int(metadata.get("video_height", 0))
        if fps <= 0 or width <= 0 or height <= 0:
            raise EvaluationAuthorityError(
                f"Invalid video metadata authority: {video_key}"
            )
        gt_rows = _raw_active_rows(gt_path)
        visible_gt = sum(not item[6] for item in gt_rows)
        hidden_gt = sum(bool(item[6]) for item in gt_rows)
        prediction_paths: dict[str, str] = {}
        for arm in arms:
            path = prediction_path(arm, video_key)
            if not path.is_file():
                raise EvaluationAuthorityError(
                    f"Missing {arm.arm} prediction: {video_key}"
                )
            actual_sha = sha256_file(path)
            before_prediction_hashes[str(path)] = actual_sha
            if arm.arm in {"B0", "B1"}:
                expected = authorities[arm.arm]["per_video_prediction_hashes"][
                    video_key
                ]
                if actual_sha != expected["file_sha256"]:
                    raise EvaluationAuthorityError(
                        f"{arm.arm} prediction hash mismatch: {video_key}"
                    )
                record = prediction_structural_record(
                    path,
                    video_key=video_key,
                    width=width,
                    height=height,
                )
                if (
                    record["prediction_semantic_sha256"]
                    != expected["semantic_sha256"]
                    or record["canonical_row_sha256"]
                    != expected["canonical_row_sha256"]
                ):
                    raise EvaluationAuthorityError(
                        f"{arm.arm} semantic hash mismatch: {video_key}"
                    )
                b0_b1_structural[arm.arm].append(record)
            else:
                relative = path.relative_to(r0_root).as_posix()
                expected = r0_records.get(relative)
                if expected is None or actual_sha != expected["sha256"]:
                    raise EvaluationAuthorityError(
                        f"R0 prediction hash mismatch: {video_key}"
                    )
                root = ET.parse(path).getroot()
                if root.findtext("./meta/task/size") != str(EXPECTED_FRAMES):
                    raise EvaluationAuthorityError(
                        f"R0 task-size mismatch: {video_key}"
                    )
            audit = adapter_audit(path)
            adapter_rows.append(
                {
                    "arm": arm.arm,
                    "profile": arm.profile,
                    "video_key": video_key,
                    **audit,
                }
            )
            prediction_paths[arm.arm] = str(path)
        videos.append(
            {
                "video_key": video_key,
                "source_video_path": str(video_path),
                "source_video_sha256": row["source_video_sha256"],
                "gt_path": str(gt_path),
                "gt_sha256": row["gt_sha256"],
                "frame_start": 0,
                "frame_end": EXPECTED_FRAMES - 1,
                "frame_count": EXPECTED_FRAMES,
                "frames_per_second": fps,
                "visible_gt_rows": visible_gt,
                "hidden_gt_rows": hidden_gt,
                "sequence_boundary": video_key,
                "aggregate_inclusion_status": row[
                    "aggregate_inclusion_role"
                ],
                "gt_authority_status": row["gt_authority_status"],
                "mechanism_ranking_eligibility": row[
                    "mechanism_ranking_eligibility"
                ],
                "prediction_paths": prediction_paths,
            }
        )
    for arm_key in ("B0", "B1"):
        actual = prediction_set_hash(b0_b1_structural[arm_key])
        expected = next(
            arm.artifact_sha256 for arm in arms if arm.arm == arm_key
        )
        if actual != expected:
            raise EvaluationAuthorityError(
                f"{arm_key} prediction-set authority mismatch"
            )
    return (
        sorted(videos, key=lambda item: item["video_key"]),
        arms,
        {
            "schema_version": "tracking.standard_v2.adapter_conservation.v1",
            "date": DATE,
            "status": "PASS",
            "rows": sorted(
                adapter_rows,
                key=lambda item: (item["arm"], item["video_key"]),
            ),
            "adapter_bbox_changes": 0,
            "adapter_id_changes": 0,
            "adapter_row_additions": 0,
            "adapter_row_removals": 0,
            "adapter_frame_index_changes": 0,
            "adapter_timestamp_changes": 0,
            "adapter_prediction_hidden_changes": 0,
        },
        {
            "prediction_hashes": before_prediction_hashes,
            "artifact_authorities": {
                arm.arm: arm.artifact_sha256 for arm in arms
            },
        },
    )


def _permuted_frames(
    rows: dict[int, list[Any]],
) -> dict[int, list[Any]]:
    """Reverse frame insertion and object ordering for invariance testing."""

    return {
        frame: list(reversed(rows[frame]))
        for frame in sorted(rows, reverse=True)
    }


def _maximum_recovery_seconds(
    evaluation: StandardV2Evaluation,
) -> float | None:
    values = [
        float(episode.recovery_latency_seconds)
        for episode in evaluation.episode_result.episodes
        if episode.recovery_latency_seconds is not None
    ]
    return max(values) if values else None


def _metric_row(
    arm: ArmSpec,
    evaluation: StandardV2Evaluation,
    *,
    mechanism_eligible: bool | None,
) -> dict[str, Any]:
    row = asdict(evaluation.metrics)
    row["hota_threshold_set"] = json.dumps(
        list(row["hota_threshold_set"]),
        separators=(",", ":"),
    )
    for key in (
        "hota_by_alpha",
        "deta_by_alpha",
        "assa_by_alpha",
        "loca_by_alpha",
        "hota_tp_by_alpha",
        "hota_fp_by_alpha",
        "hota_fn_by_alpha",
    ):
        row.pop(key, None)
    row.update(
        {
            "arm": arm.arm,
            "profile": arm.profile,
            "detector_cadence": arm.detector_cadence,
            "mechanism_ranking_eligibility": mechanism_eligible,
            "maximum_recovery_latency_seconds": _maximum_recovery_seconds(
                evaluation
            ),
        }
    )
    return row


def _sorted_frame(dataframe: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe
    available = [key for key in keys if key in dataframe.columns]
    return dataframe.sort_values(available, kind="stable").reset_index(drop=True)


def _descriptive_summary(
    per_video: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm, arm_rows in per_video.groupby("arm", sort=True):
        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(arm_rows[metric], errors="coerce")
            valid = arm_rows.loc[values.notna()].copy()
            numeric = values.dropna()
            if numeric.empty:
                continue
            minimum_index = numeric.idxmin()
            maximum_index = numeric.idxmax()
            rows.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "video_count": len(numeric),
                    "median": float(numeric.median()),
                    "q1": float(numeric.quantile(0.25)),
                    "q3": float(numeric.quantile(0.75)),
                    "iqr": float(
                        numeric.quantile(0.75) - numeric.quantile(0.25)
                    ),
                    "minimum": float(numeric.loc[minimum_index]),
                    "minimum_video": arm_rows.loc[
                        minimum_index, "video_stem"
                    ],
                    "maximum": float(numeric.loc[maximum_index]),
                    "maximum_video": arm_rows.loc[
                        maximum_index, "video_stem"
                    ],
                    "non_null_video_count": len(valid),
                }
            )
    return pd.DataFrame(rows)


def _conservation(
    evaluations: dict[str, list[StandardV2Evaluation]],
) -> dict[str, Any]:
    wrong_input = 0
    wrong_classified = 0
    wrong_double = 0
    pairwise_ids: set[str] = set()
    pairwise_double = 0
    tp_fp_fn_pass = True
    boundary_pass = True
    for arm_evaluations in evaluations.values():
        for evaluation in arm_evaluations:
            result = evaluation.episode_result
            wrong_input += result.wrong_id_rows_input
            wrong_classified += result.wrong_id_rows_classified
            wrong_double += result.wrong_id_rows_double_counted
            gt_count = evaluation.metrics.gt_detections
            pred_count = evaluation.metrics.pred_detections
            for index in range(len(HOTA_ALPHAS)):
                tp_fp_fn_pass &= (
                    evaluation.hota_result.tp[index]
                    + evaluation.hota_result.fn[index]
                    == gt_count
                )
                tp_fp_fn_pass &= (
                    evaluation.hota_result.tp[index]
                    + evaluation.hota_result.fp[index]
                    == pred_count
                )
            for episode in result.episodes:
                boundary_pass &= (
                    episode.sequence_key == evaluation.metrics.video_stem
                )
            for event in result.pairwise_events:
                if event.event_id in pairwise_ids:
                    pairwise_double += 1
                pairwise_ids.add(event.event_id)
                pairwise_double += int(
                    len(set(event.gt_ids)) != 2
                    or tuple(sorted(event.gt_ids)) != event.gt_ids
                )
                boundary_pass &= (
                    event.sequence_key == evaluation.metrics.video_stem
                )
    wrong_pass = (
        wrong_input == wrong_classified
        and wrong_double == 0
    )
    return {
        "schema_version": "tracking.standard_v2.identity_conservation.v1",
        "date": DATE,
        "wrong_id_rows_input": wrong_input,
        "wrong_id_rows_classified": wrong_classified,
        "wrong_id_rows_unclassified": wrong_input - wrong_classified,
        "wrong_id_row_conservation": "PASS" if wrong_pass else "FAIL",
        "identity_episode_double_count": wrong_double,
        "pairwise_swap_double_count": pairwise_double,
        "tp_fp_fn_conservation": "PASS" if tp_fp_fn_pass else "FAIL",
        "multi_video_boundary_status": "PASS" if boundary_pass else "FAIL",
    }


def evaluate_pass(
    pass_root: Path,
    arms: tuple[ArmSpec, ...],
    videos: list[dict[str, Any]],
    *,
    evaluator_code_sha: str,
    reverse_inputs: bool,
) -> dict[str, Any]:
    """Run one complete Standard-V2 pass and write canonical tables."""

    if pass_root.exists():
        raise EvaluationAuthorityError(f"Refusing overwrite: {pass_root}")
    pass_root.mkdir(parents=True)
    video_lookup = {row["video_key"]: row for row in videos}
    ordered_videos = list(reversed(videos)) if reverse_inputs else videos
    ordered_arms = list(reversed(arms)) if reverse_inputs else list(arms)
    evaluations: dict[str, list[StandardV2Evaluation]] = {}
    aggregates: dict[str, StandardV2Evaluation] = {}
    for arm in ordered_arms:
        arm_results: list[StandardV2Evaluation] = []
        for video in ordered_videos:
            gt = parse_cvat_video_xml(
                Path(video["gt_path"]),
                include_hidden=True,
                start_frame=0,
                end_frame=EXPECTED_FRAMES - 1,
            )
            pred = parse_cvat_video_xml(
                Path(video["prediction_paths"][arm.arm]),
                include_hidden=True,
                start_frame=0,
                end_frame=EXPECTED_FRAMES - 1,
            )
            if reverse_inputs:
                pred = _permuted_frames(pred)
            arm_results.append(
                evaluate_tracking_standard_v2(
                    gt,
                    pred,
                    video_stem=video["video_key"],
                    include_hidden=True,
                    detection_iou_threshold=0.5,
                    frames_per_second=float(video["frames_per_second"]),
                    evaluator_code_sha=evaluator_code_sha,
                )
            )
        arm_results = sorted(
            arm_results,
            key=lambda item: item.metrics.video_stem,
        )
        evaluations[arm.arm] = arm_results
        aggregates[arm.arm] = aggregate_tracking_standard_v2(arm_results)

    aggregate_rows: list[dict[str, Any]] = []
    per_video_rows: list[dict[str, Any]] = []
    alpha_frames: list[pd.DataFrame] = []
    episode_frames: list[pd.DataFrame] = []
    pairwise_frames: list[pd.DataFrame] = []
    all_pairwise_frames: list[pd.DataFrame] = []
    exposure_rows: list[dict[str, Any]] = []
    authority_frames: list[pd.DataFrame] = []
    ambiguity_frames: list[pd.DataFrame] = []
    for arm in arms:
        aggregate = aggregates[arm.arm]
        aggregate_row = _metric_row(
            arm,
            aggregate,
            mechanism_eligible=None,
        )
        aggregate_row["videos_with_any_identity_error_episode"] = sum(
            bool(item.episode_result.episodes)
            for item in evaluations[arm.arm]
        )
        aggregate_rows.append(aggregate_row)
        for evaluation in evaluations[arm.arm]:
            video = video_lookup[evaluation.metrics.video_stem]
            per_video_rows.append(
                _metric_row(
                    arm,
                    evaluation,
                    mechanism_eligible=bool(
                        video["mechanism_ranking_eligibility"]
                    ),
                )
            )
        alpha = hota_alpha_dataframe(
            [*evaluations[arm.arm], aggregate]
        )
        alpha.insert(0, "profile", arm.profile)
        alpha.insert(0, "arm", arm.arm)
        alpha_frames.append(alpha)
        episodes = identity_episode_dataframe(evaluations[arm.arm])
        episodes.insert(0, "profile", arm.profile)
        episodes.insert(0, "arm", arm.arm)
        if not episodes.empty:
            episodes["mechanism_ranking_eligibility"] = episodes[
                "sequence_key"
            ].map(
                lambda key: bool(
                    video_lookup[str(key)]["mechanism_ranking_eligibility"]
                )
            )
        episode_frames.append(episodes)
        pairwise = pairwise_swap_dataframe(evaluations[arm.arm])
        pairwise.insert(0, "profile", arm.profile)
        pairwise.insert(0, "arm", arm.arm)
        if not pairwise.empty:
            pairwise["mechanism_ranking_eligibility"] = pairwise[
                "sequence_key"
            ].map(
                lambda key: bool(
                    video_lookup[str(key)]["mechanism_ranking_eligibility"]
                )
            )
        all_pairwise_frames.append(pairwise)
        if not pairwise.empty:
            pairwise_frames.append(
                pairwise.loc[pairwise["persistent"].astype(bool)].copy()
            )
        authority = identity_authority_dataframe(evaluations[arm.arm])
        authority.insert(0, "profile", arm.profile)
        authority.insert(0, "arm", arm.arm)
        authority_frames.append(authority)
        ambiguity = identity_ambiguity_dataframe(evaluations[arm.arm])
        ambiguity.insert(0, "profile", arm.profile)
        ambiguity.insert(0, "arm", arm.arm)
        ambiguity_frames.append(ambiguity)
        for row in [
            *(
                _metric_row(
                    arm,
                    evaluation,
                    mechanism_eligible=bool(
                        video_lookup[evaluation.metrics.video_stem][
                            "mechanism_ranking_eligibility"
                        ]
                    ),
                )
                for evaluation in evaluations[arm.arm]
            ),
            aggregate_row,
        ]:
            exposure_rows.append(
                {
                    "arm": row["arm"],
                    "profile": row["profile"],
                    "video_stem": row["video_stem"],
                    "mechanism_ranking_eligibility": row[
                        "mechanism_ranking_eligibility"
                    ],
                    **{key: row[key] for key in EXPOSURE_COLUMNS},
                }
            )

    aggregate_df = _sorted_frame(pd.DataFrame(aggregate_rows), ["arm"])
    per_video_df = _sorted_frame(
        pd.DataFrame(per_video_rows),
        ["arm", "video_stem"],
    )
    alpha_df = _sorted_frame(
        pd.concat(alpha_frames, ignore_index=True),
        ["arm", "video_stem", "alpha"],
    )
    episode_df = _sorted_frame(
        pd.concat(episode_frames, ignore_index=True),
        ["arm", "sequence_key", "gt_id", "start_frame", "event_id"],
    )
    pairwise_df = _sorted_frame(
        (
            pd.concat(pairwise_frames, ignore_index=True)
            if pairwise_frames
            else pd.DataFrame(
                columns=[
                    "arm",
                    "profile",
                    "event_id",
                    "sequence_key",
                    "gt_ids",
                    "start_frame",
                    "end_frame",
                    "direct_joint_frames",
                    "direct_joint_observations",
                    "linked_episode_ids",
                    "persistent",
                    "persistence_basis",
                    "mechanism_ranking_eligibility",
                ]
            )
        ),
        ["arm", "sequence_key", "start_frame", "event_id"],
    )
    all_pairwise_df = _sorted_frame(
        pd.concat(all_pairwise_frames, ignore_index=True),
        ["arm", "sequence_key", "start_frame", "event_id"],
    )
    exposure_df = _sorted_frame(
        pd.DataFrame(exposure_rows),
        ["arm", "video_stem"],
    )
    authority_df = _sorted_frame(
        pd.concat(authority_frames, ignore_index=True),
        ["arm", "sequence_key", "gt_id"],
    )
    ambiguity_df = _sorted_frame(
        pd.concat(ambiguity_frames, ignore_index=True),
        ["arm", "sequence_key", "frame", "gt_id", "pred_id"],
    )
    outputs = {
        REQUIRED_REPEAT_FILES[0]: aggregate_df,
        REQUIRED_REPEAT_FILES[1]: per_video_df,
        REQUIRED_REPEAT_FILES[2]: alpha_df,
        REQUIRED_REPEAT_FILES[3]: episode_df,
        REQUIRED_REPEAT_FILES[4]: pairwise_df,
        REQUIRED_REPEAT_FILES[5]: exposure_df,
        "B0_B1_R0_ALL_PAIRWISE_SWAP_EVENTS.csv": all_pairwise_df,
        "B0_B1_R0_IDENTITY_AUTHORITIES.csv": authority_df,
        "B0_B1_R0_IDENTITY_AMBIGUITIES.csv": ambiguity_df,
        "B0_B1_R0_STANDARD_V2_VIDEO_DESCRIPTIVE_SUMMARY.csv": (
            _descriptive_summary(per_video_df)
        ),
    }
    for name, dataframe in outputs.items():
        write_csv(dataframe, pass_root / name)
    conservation = _conservation(evaluations)
    write_json(pass_root / "IDENTITY_EVENT_CONSERVATION.json", conservation)
    return {
        "evaluations": evaluations,
        "aggregates": aggregates,
        "aggregate_dataframe": aggregate_df,
        "per_video_dataframe": per_video_df,
        "conservation": conservation,
        "output_hashes": {
            name: sha256_file(pass_root / name) for name in outputs
        },
    }


def compare_passes(
    pass1: dict[str, Any],
    pass2: dict[str, Any],
) -> dict[str, Any]:
    """Require byte-identical canonical outputs from both complete passes."""

    file_checks = {
        name: {
            "pass1_sha256": pass1["output_hashes"][name],
            "pass2_sha256": pass2["output_hashes"][name],
            "equal": (
                pass1["output_hashes"][name]
                == pass2["output_hashes"][name]
            ),
        }
        for name in REQUIRED_REPEAT_FILES
    }
    repeatability = all(item["equal"] for item in file_checks.values())
    return {
        "schema_version": "tracking.standard_v2.determinism.v1",
        "date": DATE,
        "complete_evaluation_passes": 2,
        "pass2_prediction_file_order": "REVERSED",
        "pass2_prediction_row_order": "REVERSED_WITHIN_FRAME",
        "required_output_files": file_checks,
        "reevaluation_repeatability": (
            "PASS" if repeatability else "FAIL"
        ),
        "input_order_invariance": "PASS" if repeatability else "FAIL",
        "deterministic_metric_ordering": (
            "PASS" if repeatability else "FAIL"
        ),
        "deterministic_floating_point_serialization": (
            "PASS" if repeatability else "FAIL"
        ),
    }


def _legacy_rows(
    source_repo: Path,
    aggregate: pd.DataFrame,
) -> pd.DataFrame:
    old_b0_path = (
        source_repo
        / "outputs"
        / "eval"
        / "mode_compare"
        / "20260709_040751"
        / "bytetrack_raw"
        / "iou0_area0_condarea0_merge0"
        / "tracking_metrics.csv"
    )
    old_b0 = pd.read_csv(old_b0_path).loc[
        lambda frame: frame["video_stem"] == "ALL"
    ].iloc[0]
    legacy: dict[str, dict[str, float]] = {
        "B0": {
            "hota": float(old_b0["hota"]),
            "idf1": float(old_b0["idf1"]),
            "idsw_standard": float(old_b0["idsw"]),
            "fragments": float(old_b0["fragments"]),
            "fp": float(old_b0["fp"]),
            "fn": float(old_b0["fn"]),
        },
        "B1": {
            "hota": 0.9835062270290739,
            "idf1": 0.9914903846153846,
            "idsw_standard": 0.0,
            "fragments": 426.0,
            "fp": 1593.0,
            "fn": 1593.0,
        },
        "R0": {
            "hota": 0.9704398315450558,
            "idf1": 0.9707702337312571,
            "idsw_standard": 53.0,
            "fragments": 107.0,
            "fp": 486.0,
            "fn": 610.0,
        },
    }
    authorities = {
        "B0": str(old_b0_path),
        "B1": "docs/TRACKING_PROMOTION_DECISION_20260719_HYBRID_H5B_H4.json",
        "R0": "outputs/tracking/current_main_baseline_20260728/CURRENT_MAIN_R0_AUTHORITY.json",
    }
    rows: list[dict[str, Any]] = []
    for arm in ("B0", "B1", "R0"):
        corrected_row = aggregate.loc[aggregate["arm"] == arm].iloc[0]
        for metric, old_value in legacy[arm].items():
            corrected_value = float(corrected_row[metric])
            comparable = metric not in {"hota"}
            if arm == "B0":
                comparable = False
            absolute = corrected_value - old_value
            relative = absolute / abs(old_value) if old_value else None
            rows.append(
                {
                    "arm": arm,
                    "legacy_authority": authorities[arm],
                    "legacy_contract": "TRACKING_EVALUATOR_LEGACY_V1",
                    "legacy_metric_name": (
                        "idsw" if metric == "idsw_standard" else metric
                    ),
                    "legacy_value": old_value,
                    "corrected_v2_metric_name": metric,
                    "corrected_value": corrected_value,
                    "absolute_difference": absolute,
                    "relative_difference": relative,
                    "comparison_validity": (
                        "DIRECTIONAL_ONLY" if comparable else "NOT_COMPARABLE"
                    ),
                    "likely_cause_of_difference": (
                        "Standard-V2 formula and matching contract; frozen "
                        "prediction bytes are unchanged by evaluation."
                    ),
                    "status": (
                        "LEGACY_DIRECTION_PRESERVED_VALUE_CHANGED"
                        if comparable
                        else "LEGACY_METRIC_NOT_COMPARABLE"
                    ),
                }
            )
    return _sorted_frame(
        pd.DataFrame(rows),
        ["arm", "corrected_v2_metric_name"],
    )


def _ranking(aggregate: pd.DataFrame, metric: str) -> list[str]:
    return (
        aggregate.sort_values(metric, ascending=False, kind="stable")["arm"]
        .astype(str)
        .tolist()
    )


def _comparison_rows(aggregate: pd.DataFrame) -> pd.DataFrame:
    indexed = aggregate.set_index("arm")
    rows: list[dict[str, Any]] = []
    for label, left, right, cadence_matched, interpretation in (
        (
            "B1_MINUS_B0",
            "B1",
            "B0",
            True,
            "ByteTrack offline-repair effect under every-frame cadence",
        ),
        (
            "R0_MINUS_B0",
            "R0",
            "B0",
            False,
            "Whole-pipeline difference including detector cadence",
        ),
        (
            "R0_MINUS_B1",
            "R0",
            "B1",
            False,
            "Whole-pipeline difference including cadence and offline processing",
        ),
    ):
        for metric in METRIC_COLUMNS:
            left_value = indexed.loc[left, metric]
            right_value = indexed.loc[right, metric]
            if pd.isna(left_value) or pd.isna(right_value):
                difference = None
            else:
                difference = float(left_value) - float(right_value)
            rows.append(
                {
                    "comparison": label,
                    "metric": metric,
                    "difference": difference,
                    "detector_cadence_matched": cadence_matched,
                    "interpretation": interpretation,
                }
            )
    return pd.DataFrame(rows)


def _population_document(videos: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for video in videos:
        rows.append(
            {
                key: value
                for key, value in video.items()
                if key != "prediction_paths"
            }
            | {
                "B0_prediction_coverage": "0-1799",
                "B1_prediction_coverage": "0-1799",
                "R0_prediction_coverage": "0-1799",
            }
        )
    return {
        "schema_version": "tracking.standard_v2.population_manifest.v1",
        "date": DATE,
        "video_count": len(rows),
        "primary_include_hidden": True,
        "common_video_authority": "PASS",
        "common_frame_authority": "PASS",
        "common_gt_authority": "PASS",
        "common_sequence_boundary_authority": "PASS",
        "source_video_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "gt_authority_sha256": GT_AUTHORITY_SHA256,
        "videos": rows,
    }


def _metric_config_document(
    evaluator_code_sha: str,
    metric_config_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "tracking.standard_v2.metric_config.v1",
        "date": DATE,
        "evaluator_contract_id": EVALUATOR_CONTRACT_ID,
        "identity_episode_contract_id": IDENTITY_EPISODE_CONTRACT_ID,
        "matching_contract_id": MATCHING_CONTRACT_ID,
        "include_hidden": True,
        "hota_alpha_set": list(HOTA_ALPHAS),
        "matching_policy": "PRE_ASSIGNMENT_ELIGIBILITY",
        "sequence_boundary": SEQUENCE_BOUNDARY_POLICY,
        "idsw_policy": IDSW_POLICY,
        "aggregation_policy": (
            "TrackEval per-alpha sufficient statistics; sequence-local identity "
            "counts; pooled count/rate recomputation"
        ),
        "reference_parity_status": REFERENCE_PARITY_PASS,
        "evaluator_code_sha": evaluator_code_sha,
        "metric_config_sha256": metric_config_sha256,
        "profile_specific_evaluator_branches": 0,
    }


def artifact_inventory(root: Path) -> list[dict[str, Any]]:
    """Inventory result files while excluding the self-referential inventory."""

    excluded = {"B0_B1_R0_METRIC_ARTIFACT_INVENTORY.json"}
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name in excluded:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _post_prediction_hashes(
    before: dict[str, str],
) -> dict[str, Any]:
    modified = []
    for raw_path, expected in sorted(before.items()):
        path = Path(raw_path)
        actual = sha256_file(path)
        if actual != expected:
            modified.append(
                {
                    "path": raw_path,
                    "before": expected,
                    "after": actual,
                }
            )
    return {
        "prediction_artifacts_checked": len(before),
        "prediction_artifacts_modified": len(modified),
        "modified": modified,
    }


def _headline_markdown(
    aggregate: pd.DataFrame,
    ranking_preserved: bool,
) -> str:
    indexed = aggregate.set_index("arm")
    lines = [
        "# B0/B1/R0 Standard-V2 headline update",
        "",
        f"Date: {DATE}",
        "",
        "The frozen prediction bytes were re-evaluated under",
        "`TRACKING_EVALUATOR_STANDARD_V2`; no tracker or detector ran.",
        "",
        "| Arm | HOTA | DetA | AssA | LocA | IDF1 | IDSW_STANDARD |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ("B0", "B1", "R0"):
        row = indexed.loc[arm]
        lines.append(
            f"| {arm} | {row['hota']:.9f} | {row['deta']:.9f} | "
            f"{row['assa']:.9f} | {row['loca']:.9f} | "
            f"{row['idf1']:.9f} | {int(row['idsw_standard'])} |"
        )
    lines.extend(
        [
            "",
            "Legacy HOTA/DetA/AssA values remain historical non-standard",
            "diagnostics and must be replaced in current headline reporting.",
            "",
            (
                "The old B1 > R0 > B0 headline ordering is preserved."
                if ranking_preserved
                else "The old B1 > R0 > B0 headline ordering is not preserved."
            ),
            "",
            "B1−B0 is the matched-cadence offline-repair comparison.",
            "R0 comparisons are whole-pipeline effects including detector cadence;",
            "they are not pure association-core estimates.",
            "",
            "`000216` remains aggregate-only and is excluded from authoritative",
            "mechanism ranking because its GT authority is unresolved.",
        ]
    )
    return "\n".join(lines) + "\n"


def _arm_authority(
    arm: ArmSpec,
    aggregate_row: pd.Series,
    result_root: Path,
    hashes: dict[str, str],
    evaluator_code_sha: str,
    metric_config_sha256: str,
    prediction_manifest_hash: str,
) -> dict[str, Any]:
    metrics = {
        key: (
            None
            if pd.isna(aggregate_row[key])
            else aggregate_row[key].item()
            if hasattr(aggregate_row[key], "item")
            else aggregate_row[key]
        )
        for key in aggregate_row.index
        if key not in {"hota_threshold_set"}
    }
    return {
        "schema_version": "tracking.standard_v2.arm_authority.v2",
        "date": DATE,
        "arm": arm.arm,
        "profile": arm.profile,
        "status": "ESTABLISHED",
        "prediction_root": str(arm.prediction_root),
        "prediction_artifact_sha256": arm.artifact_sha256,
        "prediction_manifest_hash": prediction_manifest_hash,
        "source_video_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "gt_authority_sha256": GT_AUTHORITY_SHA256,
        "detector_authority_sha256": arm.detector_authority_sha256,
        "detector_weights_sha256": DETECTOR_WEIGHTS_SHA256,
        "detector_semantic_config_sha256": DETECTOR_CONFIG_SHA256,
        "detector_cadence": arm.detector_cadence,
        "evaluator_contract_id": EVALUATOR_CONTRACT_ID,
        "identity_episode_contract_id": IDENTITY_EPISODE_CONTRACT_ID,
        "include_hidden": True,
        "evaluator_code_sha": evaluator_code_sha,
        "metric_config_sha256": metric_config_sha256,
        "aggregate_metrics": metrics,
        "per_video_metric_table_sha256": hashes[REQUIRED_REPEAT_FILES[1]],
        "per_alpha_table_sha256": hashes[REQUIRED_REPEAT_FILES[2]],
        "identity_episode_table_sha256": hashes[REQUIRED_REPEAT_FILES[3]],
        "pairwise_swap_table_sha256": hashes[REQUIRED_REPEAT_FILES[4]],
        "exposure_metric_table_sha256": hashes[REQUIRED_REPEAT_FILES[5]],
        "result_root": str(result_root),
        "determinism": "PASS",
        "known_scientific_limitations": [
            "Development population only; no unseen evidence.",
            "000216 is aggregate-only and ineligible for mechanism ranking.",
            (
                "Cross-core comparisons are whole-pipeline effects including "
                "detector cadence."
            ),
            "No promotion is authorized by this authority.",
        ],
    }


def freeze(
    source_repo: Path,
    worktree_repo: Path,
    result_root: Path,
    docs_root: Path,
) -> dict[str, Any]:
    """Run both passes, prove conservation, and freeze result authorities."""

    if result_root.exists():
        raise EvaluationAuthorityError(f"Refusing overwrite: {result_root}")
    if git_sha(worktree_repo) == STARTING_MAIN_SHA:
        raise EvaluationAuthorityError(
            "Commit evaluation orchestration before running real evaluation"
        )
    videos, arms, adapter, prediction_state = preflight(
        source_repo,
        worktree_repo,
    )
    result_root.mkdir(parents=True)
    code_sha = resolve_evaluator_code_sha()
    if code_sha != git_sha(worktree_repo):
        raise EvaluationAuthorityError("Evaluator code SHA is not current HEAD")
    pass1 = evaluate_pass(
        result_root / "pass1",
        arms,
        videos,
        evaluator_code_sha=code_sha,
        reverse_inputs=False,
    )
    pass2 = evaluate_pass(
        result_root / "pass2",
        arms,
        videos,
        evaluator_code_sha=code_sha,
        reverse_inputs=True,
    )
    determinism = compare_passes(pass1, pass2)
    if determinism["reevaluation_repeatability"] != "PASS":
        raise EvaluationAuthorityError("Complete reevaluation is not repeatable")
    conservation = pass1["conservation"]
    if (
        conservation["wrong_id_row_conservation"] != "PASS"
        or conservation["tp_fp_fn_conservation"] != "PASS"
        or conservation["identity_episode_double_count"] != 0
        or conservation["pairwise_swap_double_count"] != 0
    ):
        raise EvaluationAuthorityError("Metric conservation failed")
    for path in sorted((result_root / "pass1").iterdir()):
        if path.is_file():
            shutil.copy2(path, result_root / path.name)
    aggregate = pass1["aggregate_dataframe"]
    comparisons = _comparison_rows(aggregate)
    write_csv(
        comparisons,
        result_root / "B0_B1_R0_STANDARD_V2_COMPARISONS.csv",
    )
    legacy = _legacy_rows(source_repo, aggregate)
    legacy_name = (
        "B0_B1_R0_LEGACY_STANDARD_V2_RECONCILIATION_20260728.csv"
    )
    write_csv(legacy, result_root / legacy_name)
    old_order = ["B1", "R0", "B0"]
    ranking_preserved = (
        _ranking(aggregate, "hota") == old_order
        and _ranking(aggregate, "idf1") == old_order
    )
    headline_name = "B0_B1_R0_HEADLINE_UPDATE_DECISION_20260728.md"
    headline = _headline_markdown(aggregate, ranking_preserved)
    (result_root / headline_name).write_text(
        headline,
        encoding="utf-8",
        newline="\n",
    )
    metric_hashes = {
        name: sha256_file(result_root / name)
        for name in REQUIRED_REPEAT_FILES
    }
    metric_config_sha = str(aggregate.iloc[0]["metric_config_sha256"])
    if set(aggregate["metric_config_sha256"]) != {metric_config_sha}:
        raise EvaluationAuthorityError("Arm metric-config hashes differ")
    population = _population_document(videos)
    metric_config = _metric_config_document(code_sha, metric_config_sha)
    post_hashes = _post_prediction_hashes(
        prediction_state["prediction_hashes"]
    )
    if post_hashes["prediction_artifacts_modified"] != 0:
        raise EvaluationAuthorityError("Prediction artifacts changed")
    mp4_count = sum(
        1 for path in result_root.rglob("*") if path.suffix.lower() == ".mp4"
    )
    if mp4_count:
        raise EvaluationAuthorityError("Metric result root contains MP4")
    run_manifest = {
        "schema_version": "tracking.standard_v2.metric_run_manifest.v1",
        "date": DATE,
        "starting_main_sha": STARTING_MAIN_SHA,
        "evaluation_code_sha": code_sha,
        "evaluator_contract_id": EVALUATOR_CONTRACT_ID,
        "identity_episode_contract_id": IDENTITY_EPISODE_CONTRACT_ID,
        "primary_include_hidden": True,
        "metric_config_sha256": metric_config_sha,
        "result_root": str(result_root),
        "complete_evaluation_passes": 2,
        "prediction_artifacts_before_and_after": post_hashes,
        "execution_counts": {
            "tracker_executions": 0,
            "detector_inference_calls": 0,
            "prediction_artifacts_modified": 0,
            "prediction_artifacts_regenerated": 0,
            "run_root_mp4_count": 0,
            "unseen_videos_accessed": False,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
        },
        "scientific_interpretation": {
            "cross_core_comparison_scope": (
                "WHOLE_PIPELINE_EFFECT_INCLUDING_DETECTOR_CADENCE"
            ),
            "pure_association_core_effect_claim_authorized": False,
            "B1_minus_B0_detector_cadence_matched": True,
            "R0_minus_B0_detector_cadence_matched": False,
            "R0_minus_B1_detector_cadence_matched": False,
        },
    }
    write_json(
        result_root / "B0_B1_R0_METRIC_RUN_MANIFEST.json",
        run_manifest,
    )
    write_json(
        result_root / "B0_B1_R0_STANDARD_V2_DETERMINISM.json",
        determinism,
    )
    write_json(
        result_root / "B0_B1_R0_IDENTITY_EVENT_CONSERVATION.json",
        conservation,
    )
    write_json(
        result_root / "B0_B1_R0_PREDICTION_ADAPTER_CONSERVATION.json",
        adapter,
    )
    inventory = artifact_inventory(result_root)
    inventory_document = {
        "schema_version": "tracking.standard_v2.metric_inventory.v1",
        "date": DATE,
        "inventory_excludes_itself": True,
        "artifact_count": len(inventory),
        "artifacts": inventory,
        "canonical_inventory_sha256": canonical_hash(inventory),
    }
    write_json(
        result_root / "B0_B1_R0_METRIC_ARTIFACT_INVENTORY.json",
        inventory_document,
    )

    docs_root.mkdir(parents=True, exist_ok=True)
    write_json(
        docs_root
        / "B0_B1_R0_STANDARD_V2_POPULATION_MANIFEST_20260728.json",
        population,
    )
    write_json(
        docs_root / "B0_B1_R0_STANDARD_V2_METRIC_CONFIG_20260728.json",
        metric_config,
    )
    write_json(
        docs_root
        / "B0_B1_R0_PREDICTION_ADAPTER_CONSERVATION_20260728.json",
        adapter,
    )
    write_json(
        docs_root / "B0_B1_R0_IDENTITY_EVENT_CONSERVATION_20260728.json",
        conservation,
    )
    write_json(
        docs_root / "B0_B1_R0_STANDARD_V2_DETERMINISM_20260728.json",
        determinism,
    )
    write_csv(legacy, docs_root / legacy_name)
    (docs_root / headline_name).write_text(
        headline,
        encoding="utf-8",
        newline="\n",
    )
    prediction_manifest_hashes = {
        "B0": load_json(
            next(arm.authority_path for arm in arms if arm.arm == "B0")
        )["recursive_artifact_inventory_sha256"],
        "B1": load_json(
            next(arm.authority_path for arm in arms if arm.arm == "B1")
        )["recursive_artifact_inventory_sha256"],
        "R0": (
            "461f2300318ab26134bb36beb78957a6642cbd361de9d74492f9cd09db688223"
        ),
    }
    authority_names = {
        "B0": "B0_BYTETRACK_RAW_STANDARD_V2_AUTHORITY_20260728.json",
        "B1": "B1_HYBRID_BYTETRACK_STANDARD_V2_AUTHORITY_20260728.json",
        "R0": "R0_REALTIME_FAST_STANDARD_V2_AUTHORITY_20260728.json",
    }
    for arm in arms:
        aggregate_row = aggregate.loc[aggregate["arm"] == arm.arm].iloc[0]
        authority = _arm_authority(
            arm,
            aggregate_row,
            result_root,
            metric_hashes,
            code_sha,
            metric_config_sha,
            prediction_manifest_hashes[arm.arm],
        )
        write_json(docs_root / authority_names[arm.arm], authority)
    decision = {
        "schema_version": "tracking.standard_v2.reevaluation_decision.v1",
        "date": DATE,
        "decision": "PASS_B0_B1_R0_STANDARD_V2_AUTHORITY_ESTABLISHED",
        "B0_standard_v2_authority": "ESTABLISHED",
        "B1_standard_v2_authority": "ESTABLISHED",
        "R0_standard_v2_authority": "ESTABLISHED",
        "common_video_authority": "PASS",
        "common_frame_authority": "PASS",
        "common_gt_authority": "PASS",
        "common_evaluator_authority": "PASS",
        "common_hidden_policy": "PASS",
        "prediction_artifacts_modified": 0,
        "tracker_executions": 0,
        "detector_inference_calls": 0,
        "wrong_id_row_conservation": "PASS",
        "tp_fp_fn_conservation": "PASS",
        "identity_episode_double_count": 0,
        "pairwise_swap_double_count": 0,
        "reevaluation_repeatability": "PASS",
        "input_order_invariance": "PASS",
        "legacy_ranking_preserved": ranking_preserved,
        "headline_results_require_update": True,
        "ready_for_R1_prediction_generation": True,
        "ready_for_complete_2x2_evaluation": False,
        "ready_for_unseen_evaluation": False,
        "ready_to_promote": False,
        "blockers": [
            "R1 predictions and metrics do not yet exist.",
            "Unseen evaluation and promotion remain unauthorized.",
        ],
    }
    write_json(
        docs_root
        / "B0_B1_R0_STANDARD_V2_REEVALUATION_DECISION_20260728.json",
        decision,
    )
    marker = result_root / "FROZEN_SCIENTIFIC_METRIC_AUTHORITY_DO_NOT_DELETE.txt"
    marker.write_text(
        "NON_DISPOSABLE_FROZEN_STANDARD_V2_METRIC_AUTHORITY\n"
        "Deletion requires explicit authority retirement.\n",
        encoding="utf-8",
        newline="\n",
    )
    final_inventory = artifact_inventory(result_root)
    inventory_document["artifact_count"] = len(final_inventory)
    inventory_document["artifacts"] = final_inventory
    inventory_document["canonical_inventory_sha256"] = canonical_hash(
        final_inventory
    )
    write_json(
        result_root / "B0_B1_R0_METRIC_ARTIFACT_INVENTORY.json",
        inventory_document,
    )
    for path in result_root.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode & ~stat.S_IWRITE)
    return {
        "decision": decision,
        "aggregate_metrics": aggregate.to_dict(orient="records"),
        "metric_hashes": metric_hashes,
        "result_root": str(result_root),
        "inventory_sha256": inventory_document[
            "canonical_inventory_sha256"
        ],
        "code_sha": code_sha,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--worktree-repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--docs-root",
        type=Path,
        default=(
            REPO_ROOT / "docs" / "tracking" / "b0_b1_r0_standard_v2"
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Rehash and validate all immutable inputs without evaluation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.preflight_only:
        videos, arms, adapter, prediction_state = preflight(
            args.source_repo.resolve(),
            args.worktree_repo.resolve(),
        )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "video_count": len(videos),
                    "arms": [arm.arm for arm in arms],
                    "adapter_status": adapter["status"],
                    "prediction_file_count": len(
                        prediction_state["prediction_hashes"]
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = freeze(
        args.source_repo.resolve(),
        args.worktree_repo.resolve(),
        args.result_root.resolve(),
        args.docs_root.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
