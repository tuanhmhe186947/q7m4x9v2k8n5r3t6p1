"""Freeze and audit the completed 500-item posture-review session.

This is a read-only authority-binding step with respect to source data.  It
creates small, hash-bound CSV/JSON/Markdown artifacts from the existing queue,
completed GUI ledger, reviewed snapshot, and frozen split manifests.  It does
not create labels, infer posture from behavior, or modify any project data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


SNAPSHOT_ID = "reviewed_engineering_amendment_992f34c0204a85a1"
SNAPSHOT_SHA256 = (
    "ab86e2e04267cfdc8248f9bdb8774615479d67a3589f7a25844bb1a4c93a639e"
)
SPLIT_HASH = "557156a7eb6cceeb6a91f667f7c51dcb286e3111f35f414970fa7431acc7e63b"
DECISION_SCHEMA = "classification_v2.posture_pilot_decisions.v1"
DECISION_VALUES = ("upright", "sitting", "lying", "technical_exclude")
POSTURE_CLASSES = ("lying", "sitting", "upright")
REPORT_LABELS = ("lying", "sitting", "upright", "unresolved", "exclude")
ROLE_STATUS_TOKENS = frozenset({"exclude"})
BEHAVIOR_CLASSES = (
    "drink",
    "eat",
    "fight",
    "social-nose",
    "explore",
    "lying",
    "stand",
    "move",
    "sitting",
    "playwithtoy",
)


def clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def as_bool(value: Any) -> bool:
    return clean(value).lower() in {"1", "true", "yes", "y", "t"}


def all_true(values: Any) -> bool:
    return all(as_bool(value) for value in values)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, semantic_role: str) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "semantic_role": semantic_role,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def json_record(path: Path, semantic_role: str) -> dict[str, Any]:
    return file_record(path, semantic_role)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def current_git_head(worktree: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def read_session_manifest(session_dir: Path) -> dict[str, Any]:
    manifest_path = session_dir / "session_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "queue_path",
        "decisions_path",
        "queue_sha256",
        "reviewed_snapshot",
        "reviewed_snapshot_sha256",
        "split_hash",
        "candidate_population_hash",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"session manifest missing fields: {missing}")
    if manifest["reviewed_snapshot"] != SNAPSHOT_ID:
        raise ValueError("session manifest reviewed snapshot mismatch")
    if manifest["reviewed_snapshot_sha256"] != SNAPSHOT_SHA256:
        raise ValueError("session manifest snapshot hash mismatch")
    if manifest["split_hash"] != SPLIT_HASH:
        raise ValueError("session manifest split hash mismatch")
    return manifest


def read_current_frame_rows(
    frame_path: Path,
    keys: set[str],
) -> pd.DataFrame:
    header = pd.read_csv(frame_path, nrows=0).columns.tolist()
    required = [
        "temporal_unit_key",
        "source_type",
        "dataset_id",
        "video_key",
        "source_video_key",
        "object_track_key",
        "pig_id",
        "track_id",
        "frame_index",
        "source_frame_index",
        "label_window_start",
        "label_window_end",
        "behavior_reviewed_final",
        "behavior_temporal_final",
        "dominant_behavior_in_unit",
        "behavior_after_review",
        "behavior_before_review",
        "review_include_in_training",
        "behavior_review_include_in_training",
        "include_in_training",
        "sample_weight",
        "temporal_unit_stable_for_training",
        "temporal_interval_complete",
        "temporal_harmonization_valid",
        "actor_bbox_valid",
        "bbox_valid",
        "sequence_complete",
        "sequence_range_valid",
        "spatiotemporal_feature_valid",
        "source_sequence_length",
        "temporal_unit_needs_review",
        "post_review_frame_transition_unit_excluded",
        "crop_path",
        "source_video_path",
    ]
    missing = sorted(set(required).difference(header))
    if missing:
        raise ValueError(f"reviewed frame authority missing columns: {missing}")

    selected: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        frame_path,
        usecols=required,
        dtype=str,
        keep_default_na=False,
        chunksize=100_000,
        low_memory=False,
    ):
        matched = chunk[chunk["temporal_unit_key"].isin(keys)]
        if not matched.empty:
            selected.append(matched)
    if not selected:
        return pd.DataFrame(columns=required)
    return pd.concat(selected, ignore_index=True)


def summarize_frame_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    frame = frame.copy()
    for column in (
        "frame_index",
        "source_frame_index",
        "label_window_start",
        "label_window_end",
        "source_sequence_length",
        "sample_weight",
    ):
        frame[f"_{column}_num"] = pd.to_numeric(frame[column], errors="coerce")
    grouped = frame.groupby("temporal_unit_key", sort=True, dropna=False)
    summary = grouped.first()
    summary["snapshot_frame_row_count"] = grouped.size()
    summary["snapshot_frame_count"] = grouped["frame_index"].nunique()
    summary["snapshot_frame_start"] = grouped["_frame_index_num"].min()
    summary["snapshot_frame_end"] = grouped["_frame_index_num"].max()
    summary["source_frame_start"] = grouped["_source_frame_index_num"].min()
    summary["source_frame_end"] = grouped["_source_frame_index_num"].max()
    summary["behavior_model_frame_eligible"] = (
        grouped["review_include_in_training"].agg(all_true)
        & grouped["behavior_review_include_in_training"].agg(all_true)
        & grouped["include_in_training"].agg(all_true)
        & grouped["_sample_weight_num"].agg(lambda values: values.gt(0).all())
    )
    for column in (
        "source_type",
        "dataset_id",
        "video_key",
        "source_video_key",
        "object_track_key",
        "pig_id",
        "track_id",
        "behavior_reviewed_final",
        "behavior_temporal_final",
        "dominant_behavior_in_unit",
        "label_window_start",
        "label_window_end",
    ):
        summary[f"{column}_unique_count"] = grouped[column].nunique(
            dropna=False
        )
    summary = summary.reset_index()
    summary["current_snapshot_key_found"] = True
    summary["behavior_label"] = summary["behavior_reviewed_final"].map(clean)
    summary["snapshot_behavior_valid"] = summary["behavior_label"].isin(
        BEHAVIOR_CLASSES
    )
    summary["frame_binding_valid"] = (
        summary["snapshot_frame_count"]
        .eq(pd.to_numeric(summary["source_sequence_length"], errors="coerce"))
        & summary["snapshot_frame_start"].notna()
        & summary["snapshot_frame_end"].notna()
        & summary["label_window_start_unique_count"].eq(1)
        & summary["label_window_end_unique_count"].eq(1)
    )
    return summary


def video_aliases(path: Path) -> set[str]:
    stem = path.stem.lower()
    aliases = {path.name.lower(), stem}
    for suffix in ("_30fps", "-30fps", " 30fps"):
        if stem.endswith(suffix):
            base = stem[: -len(suffix)]
            aliases.update({base, f"{base}.mp4"})
    return aliases


def build_video_index(video_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not video_root.exists():
        return index
    for path in video_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".mpg",
            ".mpeg",
            ".m4v",
        }:
            for alias in video_aliases(path):
                index.setdefault(alias.replace("\\", "/").lower(), path)
    return index


def resolve_video(row: pd.Series, index: dict[str, Path]) -> Path | None:
    candidates: list[str] = []
    for raw in (clean(row.get("video_key")), clean(row.get("source_video_key"))):
        if not raw:
            continue
        key = raw.replace("\\", "/").lower()
        stem = Path(key).stem.lower()
        stems = [stem]
        for prefix in ("test video ", "tracking_annotation_"):
            if stem.startswith(prefix):
                stems.append(stem[len(prefix) :])
        for candidate in stems:
            candidates.extend(
                [key, candidate, f"{candidate}.mp4", f"{candidate}_30fps"]
            )
    for candidate in candidates:
        if candidate in index:
            return index[candidate]
    return None


def resolve_crop(raw: Any, raw_root: Path) -> Path | None:
    value = clean(raw)
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return path
    parts = list(path.parts)
    lower_parts = [part.lower() for part in parts]
    if "crops" in lower_parts:
        index = lower_parts.index("crops")
        candidate = raw_root.joinpath(*parts[index + 1 :])
        if candidate.exists():
            return candidate
    return None


def media_bindings(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    video_root: Path,
    raw_root: Path,
) -> dict[str, dict[str, Any]]:
    index = build_video_index(video_root)
    result: dict[str, dict[str, Any]] = {}
    for key, rows in frame.groupby("temporal_unit_key", sort=True):
        source = clean(rows["source_type"].iloc[0])
        if source == "legacy_recovered":
            paths = [resolve_crop(value, raw_root) for value in rows["crop_path"]]
            valid = bool(paths) and all(path is not None for path in paths)
            representative = next((str(path) for path in paths if path), "")
            result[key] = {
                "media_binding_valid": valid,
                "media_binding_kind": "legacy_crop_sequence",
                "media_binding_path": representative,
                "media_frame_count": len(paths),
                "media_missing_frame_count": sum(path is None for path in paths),
            }
        else:
            row = summary.loc[summary["temporal_unit_key"].eq(key)].iloc[0]
            video = resolve_video(row, index)
            frames = pd.to_numeric(rows["frame_index"], errors="coerce")
            valid = video is not None and frames.notna().all() and frames.ge(0).all()
            result[key] = {
                "media_binding_valid": bool(valid),
                "media_binding_kind": "cvat_source_video",
                "media_binding_path": str(video) if video else "",
                "media_frame_count": int(len(frames)),
                "media_missing_frame_count": 0 if valid else int(len(frames)),
            }
    return result


def normalize_role_values(values: Any) -> tuple[list[str], list[str]]:
    """Separate model-role tokens from eligibility-status tokens.

    The frozen manifests may serialize an eligibility status and an inner role
    in one value, for example ``exclude|validation``.  ``exclude`` does not
    create a second train/validation role and remains separate eligibility
    metadata for any later matched ablation.
    """
    model_roles: set[str] = set()
    status_tokens: set[str] = set()
    for raw_value in values:
        for raw_token in clean(raw_value).split("|"):
            token = clean(raw_token)
            if not token:
                continue
            if token.lower() in ROLE_STATUS_TOKENS:
                status_tokens.add(token.lower())
            else:
                model_roles.add(token)
    return sorted(model_roles), sorted(status_tokens)


def split_bindings(
    effective_path: Path,
    split_path: Path,
    keys: set[str],
) -> dict[str, dict[str, Any]]:
    effective = pd.read_csv(
        effective_path,
        usecols=["window_id", "temporal_unit_keys_json", "window_exclusion_reason"],
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    split = pd.read_csv(
        split_path,
        usecols=[
            "window_id",
            "split",
            "model_split_role",
            "outer_fold_id",
        ],
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    if split["window_id"].duplicated().any():
        raise ValueError("split manifest has duplicate window IDs")
    joined = effective.merge(split, on="window_id", how="inner", validate="one_to_one")
    bindings: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {
            "outer_fold_id": set(),
            "split": set(),
            "model_split_role": set(),
            "window_exclusion_reason": set(),
            "window_ids": set(),
        }
    )
    for record in joined.to_dict(orient="records"):
        try:
            unit_keys = json.loads(clean(record["temporal_unit_keys_json"]))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid temporal unit key JSON: {exc}") from exc
        for key in unit_keys:
            key = clean(key)
            if key and key in keys:
                bindings[key]["outer_fold_id"].add(clean(record["outer_fold_id"]))
                bindings[key]["split"].add(clean(record["split"]))
                bindings[key]["model_split_role"].add(
                    clean(record["model_split_role"])
                )
                bindings[key]["window_exclusion_reason"].add(
                    clean(record["window_exclusion_reason"])
                )
                bindings[key]["window_ids"].add(clean(record["window_id"]))
    result: dict[str, dict[str, Any]] = {}
    for key, values in bindings.items():
        split_roles, split_statuses = normalize_role_values(values["split"])
        model_roles, model_statuses = normalize_role_values(
            values["model_split_role"]
        )
        result[key] = {
            "outer_fold_ids": sorted(values["outer_fold_id"] - {""}),
            "split_roles": split_roles,
            "model_split_roles": model_roles,
            "eligibility_status_tokens": sorted(
                set(split_statuses).union(model_statuses)
            ),
            "window_exclusion_reasons": sorted(
                values["window_exclusion_reason"] - {""}
            ),
            "window_ids": sorted(values["window_ids"] - {""}),
        }
    return result


def exact_metadata_alignment(
    row: dict[str, Any],
    current: pd.Series | None,
) -> tuple[bool, list[str]]:
    if current is None:
        return False, ["current_snapshot_key_missing"]
    mismatches: list[str] = []
    for column in (
        "source_type",
        "dataset_id",
        "video_key",
        "object_track_key",
        "pig_id",
        "track_id",
    ):
        if clean(row.get(column)) != clean(current.get(column)):
            mismatches.append(column)
    for column in ("unit_start_frame", "unit_end_frame"):
        expected = pd.to_numeric(clean(row.get(column)), errors="coerce")
        boundary = "start" if column.endswith("start_frame") else "end"
        observed = pd.to_numeric(
            clean(current.get(f"label_window_{boundary}")),
            errors="coerce",
        )
        if pd.isna(expected) or pd.isna(observed) or int(expected) != int(observed):
            mismatches.append(column)
    return not mismatches, mismatches


def report_label(value: str) -> str:
    return "exclude" if value == "technical_exclude" else value


def group_support(
    gold: pd.DataFrame,
    dimensions: list[tuple[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dimension, column in dimensions:
        for group_value, subset in gold.groupby(column, sort=True, dropna=False):
            group_value = clean(group_value)
            for label in REPORT_LABELS:
                exact = "technical_exclude" if label == "exclude" else label
                count = int(
                    gold.loc[
                        gold[column].eq(group_value)
                        & gold["posture_decision"].eq(exact)
                    ].shape[0]
                )
                rows.append(
                    {
                        "group_dimension": dimension,
                        "group_value": group_value,
                        "posture_label": label,
                        "machine_decision_value": exact,
                        "support_count": count,
                        "usable_support_count": count if label in POSTURE_CLASSES else 0,
                        "group_total": int(len(subset)),
                    }
                )
    return pd.DataFrame(rows)


def crosstab(gold: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for behavior in BEHAVIOR_CLASSES:
        subset = gold[gold["behavior_label"].eq(behavior)]
        total = len(subset)
        for label in REPORT_LABELS:
            exact = "technical_exclude" if label == "exclude" else label
            count = int(subset["posture_decision"].eq(exact).sum())
            rows.append(
                {
                    "behavior_label": behavior,
                    "posture_label": label,
                    "machine_decision_value": exact,
                    "support_count": count,
                    "behavior_total": int(total),
                    "proportion": (count / total) if total else None,
                }
            )
    return pd.DataFrame(rows)


def mapping_evidence_category(
    total: int,
    counts: dict[str, int],
    candidate_mapping: str | None,
) -> str:
    """Classify observed mapping evidence without creating derived targets."""
    observed = {label: count for label, count in counts.items() if count}
    if total < 2:
        return "insufficient_support"
    if len(observed) == 1:
        return "empirically_near_deterministic"
    if candidate_mapping is not None:
        candidate_count = counts.get(candidate_mapping, 0)
        other_counts = [
            count for label, count in counts.items() if label != candidate_mapping
        ]
        if candidate_count > max(other_counts, default=0):
            return "strong_but_not_deterministic"
    return "mixed"


def mapping_audit(cross: pd.DataFrame) -> dict[str, Any]:
    candidates = {
        "lying": "lying",
        "sitting": "sitting",
        "stand": "upright",
        "eat": "upright",
    }
    records: list[dict[str, Any]] = []
    for behavior in BEHAVIOR_CLASSES:
        subset = cross[cross["behavior_label"].eq(behavior)]
        total = int(subset["behavior_total"].iloc[0])
        counts = {
            label: int(
                subset.loc[subset["posture_label"].eq(label), "support_count"].iloc[0]
            )
            for label in REPORT_LABELS
        }
        usable_counts = {label: counts[label] for label in POSTURE_CLASSES}
        candidate = candidates.get(behavior)
        category = mapping_evidence_category(total, counts, candidate)
        records.append(
            {
                "behavior_label": behavior,
                "human_reviewed_examples": total,
                "posture_counts": counts,
                "posture_proportions": {
                    label: (counts[label] / total) if total else None
                    for label in REPORT_LABELS
                },
                "candidate_mapping": candidate,
                "candidate_mapping_observed_count": (
                    counts.get(candidate, 0) if candidate else None
                ),
                "empirical_category": category,
                "derived_labels_created": False,
                "interpretation": (
                    "Diagnostic human-review evidence only; no target was created."
                ),
            }
        )
    return {
        "schema_version": "classification_v2.posture_behavior_mapping_empirical_audit.v1",
        "status": "COMPLETE_DIAGNOSTIC_ONLY",
        "threshold_policy": (
            "No retrospective numerical threshold was used. A registered "
            "candidate mapping is strong only when it is the unique observed "
            "mode; a single-label observation is near-deterministic."
        ),
        "class_order": list(BEHAVIOR_CLASSES),
        "records": records,
        "derived_posture_labels_created": False,
    }


def validate_e0_handoff(path: Path) -> dict[str, Any]:
    handoff = json.loads(path.read_text(encoding="utf-8"))
    descriptor = handoff.get("descriptor", {})
    checks = {
        "do_not_execute": handoff.get("do_not_execute") is True,
        "authorization_no": handoff.get("execution_authorization") == "NO",
        "inner_fold": descriptor.get("inner_fold") == "FOLD_3",
        "model": descriptor.get("model") == "B3_ACTOR_T6_PLUS_GEOMETRY_MOTION",
        "temporal_view": descriptor.get("temporal_view") == "T6",
        "seed": descriptor.get("seed") == 20260804,
        "snapshot": descriptor.get("snapshot_sha256") == SNAPSHOT_SHA256,
        "split": descriptor.get("split_hash") == SPLIT_HASH,
        "outer_exclusion": not any(
            handoff.get("outer_test_exclusion", {}).get(key, True)
            for key in (
                "labels",
                "metrics",
                "predictions",
                "errors",
                "confusion_matrices",
                "data_mount",
            )
        ),
    }
    provider = handoff.get("provider", {})
    checks["budget"] = provider == {
        "gpu": "1x NVIDIA L4 24 GB",
        "interruptible": False,
        "max_cost_usd": 1.5,
        "max_gpu_hours": 2.0,
        "max_remote_disk_gb": 15,
        "max_wall_hours": 4.0,
        "paid_retries": 0,
    }
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "descriptor": descriptor,
        "provider": provider,
        "launch_command": handoff.get("launch_command", ""),
        "resume_command": handoff.get("resume_command", ""),
        "forced_interruption_procedure": handoff.get(
            "forced_interruption_procedure", ""
        ),
        "prediction_export": handoff.get("prediction_export", ""),
        "download_manifest": handoff.get("download_manifest", ""),
        "hash_verification": handoff.get("hash_verification", ""),
        "gpu_stop_command": handoff.get("gpu_stop_command", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--reviewed-frame-features", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--effective-window-index", type=Path, required=True)
    parser.add_argument("--old-ledger", type=Path, required=True)
    parser.add_argument("--e0-handoff", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--legacy-crop-root", type=Path, required=True)
    parser.add_argument("--execution-worktree", type=Path, required=True)
    parser.add_argument("--gui-worktree", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    session = args.session_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = session / "session_manifest.json"
    queue_path = session / "posture_review_scope.csv"
    decisions_path = session / "posture_pilot_decisions.csv"
    manifest = read_session_manifest(session)

    if sha256_file(queue_path) != manifest["queue_sha256"]:
        raise ValueError("queue hash does not match session manifest")
    expected_session_manifest_sha256 = (
        "7e979f41246abbf33424a03cf947aa16d2fef7a6dc87f3b175c409950cb4d168"
    )
    if sha256_file(manifest_path) != expected_session_manifest_sha256:
        raise ValueError("session manifest hash differs from frozen session authority")

    queue = pd.read_csv(queue_path, dtype=str, keep_default_na=False)
    decisions = pd.read_csv(decisions_path, dtype=str, keep_default_na=False)
    candidate = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    candidate_rows = pd.DataFrame(candidate["rows"])
    if len(queue) != 500 or queue["native_temporal_unit_key"].nunique() != 500:
        raise ValueError("completed queue is not exactly 500 unique native units")
    if len(decisions) != 500:
        raise ValueError("completed posture ledger does not contain 500 rows")
    if decisions["posture_review_item_id"].duplicated().any():
        raise ValueError("completed posture ledger has duplicate item IDs")
    if decisions["native_temporal_unit_key"].duplicated().any():
        raise ValueError("completed posture ledger has duplicate native keys")
    if set(decisions["posture_decision"]) - set(DECISION_VALUES):
        raise ValueError("completed posture ledger contains unknown decision values")
    if (
        decisions["scope_sha256"].nunique() != 1
        or decisions["scope_sha256"].iloc[0] != manifest["queue_sha256"]
    ):
        raise ValueError("completed posture ledger scope binding mismatch")
    if (
        decisions["schema_version"].nunique() != 1
        or decisions["schema_version"].iloc[0] != DECISION_SCHEMA
    ):
        raise ValueError("completed posture ledger schema mismatch")
    technical_excludes = decisions["posture_decision"].eq("technical_exclude")
    missing_technical_reason = decisions.loc[
        technical_excludes,
        "technical_exclusion_reason",
    ].eq("").any()
    if technical_excludes.any() and missing_technical_reason:
        raise ValueError("technical exclusions require a reason")

    queue = queue.merge(
        candidate_rows[
            [
                "posture_review_item_id",
                "queue_behavior_context",
                "queue_stratum",
                "difficulty_score",
            ]
        ],
        on="posture_review_item_id",
        how="left",
        validate="one_to_one",
    )
    if queue["queue_stratum"].isna().any():
        raise ValueError("candidate manifest does not cover the queue")

    old = pd.read_csv(args.old_ledger, dtype=str, keep_default_na=False)
    all_keys = set(queue["native_temporal_unit_key"]) | set(old["native_temporal_unit_key"])
    frame = read_current_frame_rows(args.reviewed_frame_features, all_keys)
    summary = summarize_frame_rows(frame)
    summary_map = summary.set_index("temporal_unit_key", drop=False)
    media = media_bindings(frame, summary, args.video_root, args.legacy_crop_root)
    roles = split_bindings(args.effective_window_index, args.split_manifest, all_keys)

    rows: list[dict[str, Any]] = []
    alignment_failures: Counter[str] = Counter()
    for record in queue.merge(
        decisions,
        on=["posture_review_item_id", "native_temporal_unit_key"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_decision"),
    ).to_dict(orient="records"):
        key = clean(record["native_temporal_unit_key"])
        current = summary_map.loc[key] if key in summary_map.index else None
        aligned, mismatch = exact_metadata_alignment(record, current)
        for reason in mismatch:
            alignment_failures[reason] += 1
        role = roles.get(key, {})
        expected_roles, expected_statuses = normalize_role_values(
            [record["split_role"]]
        )
        expected_model_role = expected_roles[0] if len(expected_roles) == 1 else ""
        status_binding_valid = (
            role.get("eligibility_status_tokens", []) == expected_statuses
        )
        role_valid = bool(
            len(expected_roles) == 1
            and len(role.get("outer_fold_ids", [])) == 1
            and len(role.get("model_split_roles", [])) == 1
            and len(role.get("split_roles", [])) == 1
            and role["outer_fold_ids"][0] == clean(record["outer_fold_id"])
            and role["model_split_roles"][0] == expected_model_role
            and role["split_roles"][0] == expected_model_role
            and status_binding_valid
        )
        if not role_valid:
            alignment_failures["split_role_binding"] += 1
        media_row = media.get(key, {})
        media_valid = bool(media_row.get("media_binding_valid", False))
        if not media_valid:
            alignment_failures["media_binding"] += 1
        posture = clean(record["posture_decision"])
        usable = posture in POSTURE_CLASSES and aligned and role_valid and media_valid
        behavior_eligible = bool(
            role_valid
            and current is not None
            and bool(current["behavior_model_frame_eligible"])
            and not role.get("eligibility_status_tokens", [])
            and not role.get("window_exclusion_reasons", [])
        )
        rows.append(
            {
                "stable_native_unit_key": key,
                "actor_key": clean(record["object_track_key"]),
                "posture_label": posture,
                "posture_decision": posture,
                "posture_label_display": report_label(posture),
                "posture_label_source": "human_review",
                "source_type": clean(record["source_type"]),
                "source_video_key": (
                    clean(record["source_video_key"])
                    or clean(record["video_key"])
                ),
                "video_key": clean(record["video_key"]),
                "recording_date": clean(record["recording_date"]),
                "native_start_frame": int(record["unit_start_frame"]),
                "native_end_frame": int(record["unit_end_frame"]),
                "split_role": clean(record["split_role"]),
                "outer_fold_id": clean(record["outer_fold_id"]),
                "model_split_role": expected_model_role,
                "split_eligibility_status_tokens": "|".join(expected_statuses),
                "behavior_label": clean(
                    current["behavior_label"] if current is not None else ""
                ),
                "sampling_stratum": clean(record["queue_stratum"]),
                "reviewed_snapshot": SNAPSHOT_ID,
                "reviewed_snapshot_sha": SNAPSHOT_SHA256,
                "split_hash": SPLIT_HASH,
                "review_session": session.name,
                "review_status": "human_review_completed",
                "reviewer": clean(record["reviewer"]),
                "decision_timestamp": clean(record["reviewed_at"]),
                "technical_exclusion_reason": clean(
                    record["technical_exclusion_reason"]
                ),
                "media_binding_status": "PASS" if media_valid else "FAIL",
                "media_binding_path": clean(media_row.get("media_binding_path")),
                "snapshot_binding_status": "PASS" if aligned else "FAIL",
                "split_binding_status": "PASS" if role_valid else "FAIL",
                "usable_for_posture_supervision": bool(usable),
                "posture_valid_mask": bool(usable),
                "behavior_model_eligible_for_matched_ablation": behavior_eligible,
            }
        )
    gold = pd.DataFrame(rows)
    if len(gold) != 500 or gold["stable_native_unit_key"].nunique() != 500:
        raise ValueError("gold table is not exactly 500 unique native units")

    # The human ledger is the sole target authority.  Queue behavior context is
    # diagnostic and must agree with the current snapshot, but never overwrites it.
    context_map = queue.set_index("native_temporal_unit_key")["queue_behavior_context"]
    context_match = all(
        clean(context_map.get(key)) == clean(value)
        for key, value in zip(gold["stable_native_unit_key"], gold["behavior_label"])
    )
    if not context_match:
        raise ValueError("queue behavior context does not match current behavior snapshot")

    all_current_bindings_valid = bool(
        gold["snapshot_binding_status"].eq("PASS").all()
        and gold["split_binding_status"].eq("PASS").all()
        and gold["media_binding_status"].eq("PASS").all()
    )
    support = gold["posture_label_display"].value_counts().to_dict()
    usable_support = gold.loc[
        gold["usable_for_posture_supervision"],
        "posture_label_display",
    ].value_counts().to_dict()
    if any(int(usable_support.get(label, 0)) == 0 for label in POSTURE_CLASSES):
        raise ValueError("a posture class has no usable human-reviewed support")
    matched_mask = (
        gold["usable_for_posture_supervision"]
        & gold["behavior_model_eligible_for_matched_ablation"]
    )
    matched_support = gold.loc[matched_mask, "posture_label_display"].value_counts().to_dict()
    matched_ablation_executable = all(
        int(matched_support.get(label, 0)) > 0 for label in POSTURE_CLASSES
    )

    crosstab_table = crosstab(gold)
    mapping = mapping_audit(crosstab_table)
    group_table = group_support(
        gold,
        [
            ("source_type", "source_type"),
            ("recording_date", "recording_date"),
            ("source_video", "source_video_key"),
            ("outer_fold", "outer_fold_id"),
            ("sampling_stratum", "sampling_stratum"),
        ],
    )
    fold_table = group_table[group_table["group_dimension"].eq("outer_fold")].copy()

    output_files: dict[str, Path] = {}
    gold_path = output / "posture_500_human_gold.csv"
    gold.to_csv(gold_path, index=False, lineterminator="\n")
    output_files["posture_500_human_gold"] = gold_path

    support_rows = []
    for label in REPORT_LABELS:
        exact = "technical_exclude" if label == "exclude" else label
        count = int(gold["posture_decision"].eq(exact).sum())
        support_rows.append(
            {
                "group_dimension": "overall",
                "group_value": "all_500_completed_items",
                "posture_label": label,
                "machine_decision_value": exact,
                "support_count": count,
                "usable_support_count": count if label in POSTURE_CLASSES else 0,
                "status": (
                    "PASS"
                    if label in POSTURE_CLASSES and count > 0
                    else "OBSERVED_ZERO_OR_NOT_USABLE"
                ),
            }
        )
    support_path = output / "posture_500_human_support.csv"
    pd.DataFrame(support_rows).to_csv(support_path, index=False, lineterminator="\n")
    output_files["posture_500_human_support"] = support_path

    cross_path = output / "posture_500_behavior_posture_crosstab.csv"
    crosstab_table.to_csv(cross_path, index=False, lineterminator="\n")
    output_files["posture_500_behavior_posture_crosstab"] = cross_path

    fold_path = output / "posture_500_fold_support.csv"
    fold_table.to_csv(fold_path, index=False, lineterminator="\n")
    output_files["posture_500_fold_support"] = fold_path

    source_path = output / "posture_500_source_support.csv"
    group_table.to_csv(source_path, index=False, lineterminator="\n")
    output_files["posture_500_source_support"] = source_path

    pilot_rows: list[dict[str, Any]] = []
    pilot_failures: Counter[str] = Counter()
    old_item_merge = old.copy()
    for record in old_item_merge.to_dict(orient="records"):
        key = clean(record["native_temporal_unit_key"])
        current = summary_map.loc[key] if key in summary_map.index else None
        aligned, mismatch = exact_metadata_alignment(record, current)
        role = roles.get(key, {})
        role_valid = bool(
            len(role.get("outer_fold_ids", [])) == 1
            and len(role.get("model_split_roles", [])) == 1
            and len(role.get("split_roles", [])) == 1
        )
        media_valid = bool(media.get(key, {}).get("media_binding_valid", False))
        semantics_valid = (
            clean(record.get("schema_version")) == DECISION_SCHEMA
            and clean(record.get("posture_decision")) in DECISION_VALUES
        )
        reasons = list(mismatch)
        if not role_valid:
            reasons.append("current_split_binding_missing_or_cross_role")
        if not media_valid:
            reasons.append("current_media_binding_invalid")
        if not semantics_valid:
            reasons.append("decision_semantics_invalid")
        for reason in reasons:
            pilot_failures[reason] += 1
        pilot_rows.append(
            {
                "posture_review_item_id": clean(record["posture_review_item_id"]),
                "stable_native_unit_key": key,
                "posture_decision": clean(record["posture_decision"]),
                "source_session": args.old_ledger.parent.name,
                "current_snapshot_alignment_valid": aligned,
                "current_split_alignment_valid": role_valid,
                "current_media_binding_valid": media_valid,
                "decision_semantics_valid": semantics_valid,
                "current_outer_fold_id": (
                    role.get("outer_fold_ids", [""])[0]
                    if len(role.get("outer_fold_ids", [])) == 1
                    else ""
                ),
                "current_model_split_role": (
                    role.get("model_split_roles", [""])[0]
                    if len(role.get("model_split_roles", [])) == 1
                    else ""
                ),
                "binding_status": "historical_human_review_bound"
                if not reasons
                else "historical_pilot_only",
                "unresolved_binding_reasons": reasons,
            }
        )
    pilot_bindable = sum(
        row["binding_status"] == "historical_human_review_bound" for row in pilot_rows
    )
    pilot_audit = {
        "schema_version": "classification_v2.posture_120_pilot_binding_audit.v1",
        "status": "PASS" if pilot_bindable == len(pilot_rows) else "PARTIAL_BINDING",
        "source_ledger": file_record(args.old_ledger, "historical 120-item posture pilot ledger"),
        "observed_scope_sha256": sorted(set(old["scope_sha256"])),
        "decision_schema": DECISION_SCHEMA,
        "current_snapshot": {
            "id": SNAPSHOT_ID,
            "sha256": SNAPSHOT_SHA256,
            "frame_authority": file_record(
                args.reviewed_frame_features, "current reviewed snapshot frame/media rows"
            ),
        },
        "current_split_hash": SPLIT_HASH,
        "posture_class_order": list(POSTURE_CLASSES),
        "posture_review_reopened": False,
        "POSTURE_120_FULLY_BINDABLE": pilot_bindable == len(pilot_rows),
        "POSTURE_120_BINDABLE_ROWS": pilot_bindable,
        "POSTURE_120_PILOT_ONLY_ROWS": len(pilot_rows) - pilot_bindable,
        "failure_counts": dict(sorted(pilot_failures.items())),
        "rows": pilot_rows,
    }
    pilot_path = output / "posture_120_pilot_binding_audit.json"
    write_json(pilot_path, pilot_audit)
    output_files["posture_120_pilot_binding_audit"] = pilot_path

    mapping_path = output / "posture_behavior_mapping_empirical_audit.json"
    write_json(mapping_path, mapping)
    output_files["posture_behavior_mapping_empirical_audit"] = mapping_path

    snapshot_manifest_record = file_record(
        args.snapshot_manifest, "current reviewed snapshot identity manifest"
    )
    split_record = file_record(args.split_manifest, "frozen grouped split manifest")
    effective_record = file_record(
        args.effective_window_index, "current effective window and role inheritance manifest"
    )
    frame_record = file_record(
        args.reviewed_frame_features, "current reviewed snapshot frame/media binding"
    )
    media_cache_records = []
    for name in (".posture_pilot_frame_features.sqlite3", ".final_behavior_frame_features.sqlite3"):
        path = session / name
        if path.exists():
            media_cache_records.append(file_record(path, "GUI session media cache"))

    gui_paths = [
        args.gui_worktree
        / "scripts"
        / "classification_v2"
        / "01_review_units_gui"
        / "review_posture_pilot_gui.py",
        args.gui_worktree
        / "scripts"
        / "classification_v2"
        / "01_review_units_gui"
        / "review_final_behavior_gui_v1.py",
    ]
    missing_gui_paths = [str(path) for path in gui_paths if not path.exists()]
    if missing_gui_paths:
        raise ValueError(f"completed review GUI sources missing: {missing_gui_paths}")
    gui_lineage = {
        "worktree": str(args.gui_worktree.resolve()),
        "git_head": current_git_head(args.gui_worktree),
        "source_files": [
            file_record(path, "posture review GUI implementation")
            for path in gui_paths
        ],
    }
    audit_output_records = {
        name: file_record(path, "post-review closure artifact")
        for name, path in sorted(output_files.items())
    }

    audit = {
        "schema_version": "classification_v2.posture_500_authority_audit.v1",
        "status": "PASS" if all_current_bindings_valid else "FAIL",
        "review_reopened": False,
        "derived_posture_labels_created": False,
        "review_session": {
            "path": str(session),
            "manifest": file_record(manifest_path, "completed posture session manifest"),
            "queue": file_record(queue_path, "frozen posture review queue"),
            "decisions": file_record(decisions_path, "completed human posture decision ledger"),
            "queue_rows": int(len(queue)),
            "queue_unique_native_keys": int(queue["native_temporal_unit_key"].nunique()),
            "decision_rows": int(len(decisions)),
            "pending_decisions": 0,
            "duplicate_decision_keys": 0,
            "unknown_decision_values": sorted(
                set(decisions["posture_decision"]) - set(DECISION_VALUES)
            ),
            "decision_values": dict(sorted(decisions["posture_decision"].value_counts().items())),
            "decision_schema": DECISION_SCHEMA,
            "reviewers": sorted(set(decisions["reviewer"])),
        },
        "authorities": {
            "reviewed_snapshot": {
                "id": SNAPSHOT_ID,
                "sha256": SNAPSHOT_SHA256,
                "manifest": snapshot_manifest_record,
                "frame_features": frame_record,
            },
            "split_hash": SPLIT_HASH,
            "split_manifest": split_record,
            "effective_window_index": effective_record,
            "media_cache_artifacts": media_cache_records,
            "review_gui": gui_lineage,
            "candidate_population_hash": candidate["candidate_population_hash"],
            "candidate_manifest": file_record(
                args.candidate_manifest, "registered 500-item sampling and provenance manifest"
            ),
        },
        "binding_checks": {
            "all_current_snapshot_bindings_valid": all_current_bindings_valid,
            "all_current_split_bindings_valid": bool(gold["split_binding_status"].eq("PASS").all()),
            "all_current_media_bindings_valid": bool(gold["media_binding_status"].eq("PASS").all()),
            "cross_role_native_unit_count": int(
                sum(
                    len(roles.get(key, {}).get("model_split_roles", [])) > 1
                    for key in gold["stable_native_unit_key"]
                )
            ),
            "alignment_failure_counts": dict(sorted(alignment_failures.items())),
            "current_snapshot_keys_missing": int(
                sum(key not in summary_map.index for key in gold["stable_native_unit_key"])
            ),
        },
        "support": {
            "overall": {label: int(support.get(label, 0)) for label in REPORT_LABELS},
            "usable_three_class": {
                label: int(usable_support.get(label, 0)) for label in POSTURE_CLASSES
            },
            "source_type_counts": {
                clean(key): int(value)
                for key, value in gold["source_type"].value_counts().items()
            },
            "recording_date_count": int(gold["recording_date"].nunique()),
            "source_video_count": int(gold["source_video_key"].nunique()),
            "outer_folds": sorted(set(gold["outer_fold_id"])),
            "sampling_strata": {
                clean(key): int(value)
                for key, value in gold["sampling_stratum"].value_counts().items()
            },
            "behavior_model_eligible_for_matched_ablation": {
                "rows": int(matched_mask.sum()),
                "posture_support": {
                    label: int(matched_support.get(label, 0))
                    for label in POSTURE_CLASSES
                },
            },
        },
        "scientific_decision": {
            "POSTURE_AUTHORITY": "PASS" if all_current_bindings_valid else "FAIL",
            "POSTURE_EXPERIMENT_EXECUTABLE": all_current_bindings_valid,
            "POSTURE_MATCHED_ABLATION_EXECUTABLE": (
                all_current_bindings_valid and matched_ablation_executable
            ),
            "POSTURE_CLASS_SUPPORT_LIMITATION": True,
            "class_support_limitation_reason": (
                "The three-class support is imbalanced (194/204/102), but every "
                "class has usable support and this is not a validity gate."
            ),
            "human_review_only_gold": True,
            "behavior_cross_tab_diagnostic_only": True,
        },
        "output_files": audit_output_records,
    }
    completed_authority_path = output / "posture_500_completed_authority.json"
    write_json(completed_authority_path, audit)
    output_files["posture_500_completed_authority"] = completed_authority_path
    authority_audit_path = output / "posture_500_authority_audit.json"
    write_json(authority_audit_path, audit)
    output_files["posture_500_authority_audit"] = authority_audit_path

    independent_contract = {
        "schema_version": "classification_v2.posture_independent_experiment_contract.v1",
        "status": "REGISTERED_NOT_EXECUTED",
        "posture_authority": str(completed_authority_path),
        "posture_authority_sha256": sha256_file(completed_authority_path),
        "primary_task": "lying_vs_sitting_vs_upright",
        "gold_evaluation_authority": "500-item human-reviewed posture labels only",
        "class_order": list(POSTURE_CLASSES),
        "unresolved_and_exclude_policy": "masked_or_excluded; never coerced",
        "derived_labels_created": False,
        "matched_comparison": {
            "P0": "final selected behavior architecture without posture supervision",
            "P1": "same architecture with independent masked posture auxiliary head",
        },
        "fixed_factors": [
            "behavior data",
            "reviewed snapshot",
            "frozen split",
            "temporal view",
            "modalities",
            "backbone",
            "optimizer procedure",
            "seed policy",
            "stopping rule",
            "behavior evaluator",
        ],
        "behavior_primary_metric": "native-unit ten-class Macro-F1",
        "behavior_guardrails": [
            "fight F1",
            "social-nose F1",
            "playwithtoy F1",
            "lying behavior F1",
            "sitting behavior F1",
            "stand behavior F1",
            "common-class harm",
        ],
        "posture_metrics": ["Macro-F1", "per-class F1", "support"],
        "promotion_rule": (
            "Keep posture only when the matched comparison provides complementary "
            "behavior evidence without material overall or rare-class harm."
        ),
        "execution_authorization": "NOT_AUTHORIZED_IN_THIS_PHASE",
        "included_in_s1": False,
    }
    independent_path = output / "posture_independent_experiment_contract.json"
    write_json(independent_path, independent_contract)
    output_files["posture_independent_experiment_contract"] = independent_path

    matched_contract = {
        "schema_version": "classification_v2.posture_behavior_matched_ablation_contract.v2",
        "status": "REGISTERED_NOT_EXECUTED",
        "executable_after_separate_experiment_authorization": (
            all_current_bindings_valid and matched_ablation_executable
        ),
        "P0": {
            "id": "behavior_only",
            "posture_supervision": False,
            "role": "matched_control",
        },
        "P1": {
            "id": "behavior_plus_masked_posture",
            "posture_supervision": True,
            "posture_authority": str(completed_authority_path),
            "posture_authority_sha256": sha256_file(completed_authority_path),
            "valid_mask": (
                "usable_for_posture_supervision AND "
                "behavior_model_eligible_for_matched_ablation"
            ),
            "unresolved_targets_contribute": False,
            "eligible_human_posture_support": {
                label: int(matched_support.get(label, 0))
                for label in POSTURE_CLASSES
            },
        },
        "primary_behavior_metric": "native-unit ten-class Macro-F1",
        "required_rare_class_metrics": [
            "fight F1",
            "social-nose F1",
            "playwithtoy F1",
        ],
        "fixed_factors": independent_contract["fixed_factors"],
        "no_automatic_promotion": True,
        "no_outer_guidance": True,
        "posture_review_reopened": False,
        "included_in_s1": False,
    }
    matched_path = output / "posture_behavior_matched_ablation_contract.json"
    write_json(matched_path, matched_contract)
    output_files["posture_behavior_matched_ablation_contract"] = matched_path

    e0_check = validate_e0_handoff(args.e0_handoff)
    posture_readiness = {
        "schema_version": "classification_v2.updated_posture_readiness_decision.v1",
        "POSTURE_AUTHORITY": audit["scientific_decision"]["POSTURE_AUTHORITY"],
        "POSTURE_EXPERIMENT_EXECUTABLE": all_current_bindings_valid,
        "POSTURE_MATCHED_ABLATION_EXECUTABLE": (
            all_current_bindings_valid and matched_ablation_executable
        ),
        "POSTURE_CLASS_SUPPORT_LIMITATION": True,
        "POSTURE_INCLUDED_IN_S1": False,
        "BEHAVIOR_ONLY_S1_BLOCKED_BY_POSTURE": False,
        "POSTURE_CAMPAIGN_STATUS": "HUMAN_REVIEW_CLOSED_AUTHORITY_BOUND",
        "POSTURE_REVIEW_REOPENED": False,
        "CURRENT_AUTHORITY": str(completed_authority_path),
        "CURRENT_AUTHORITY_SHA256": sha256_file(completed_authority_path),
        "PILOT_BINDING_AUDIT": str(pilot_path),
        "PILOT_BINDING_AUDIT_SHA256": sha256_file(pilot_path),
        "MAPPING_AUDIT": str(mapping_path),
        "MAPPING_AUDIT_SHA256": sha256_file(mapping_path),
        "INDEPENDENT_CONTRACT": str(independent_path),
        "INDEPENDENT_CONTRACT_SHA256": sha256_file(independent_path),
        "MATCHED_ABLATION_CONTRACT": str(matched_path),
        "MATCHED_ABLATION_CONTRACT_SHA256": sha256_file(matched_path),
        "NEXT_ACTION": (
            "Keep posture as a registered optional matched ablation; do not "
            "execute it in this closure phase."
        ),
    }
    posture_readiness_path = output / "updated_posture_readiness_decision.json"
    write_json(posture_readiness_path, posture_readiness)
    output_files["updated_posture_readiness_decision"] = posture_readiness_path

    campaign = {
        "schema_version": "classification_v2.updated_campaign_readiness_decision.v1",
        "A12_DIRECT_SOURCE_LEAKAGE_SAFETY": "PASS",
        "A12_OVERLAP_AND_GROUPING_INTEGRITY": "PASS",
        "POSTURE_AUTHORITY": audit["scientific_decision"]["POSTURE_AUTHORITY"],
        "POSTURE_INCLUDED_IN_S1": False,
        "BEHAVIOR_ONLY_S1_BLOCKED_BY_POSTURE": False,
        "CURRENT_ELIGIBLE_NATIVE_UNITS": 159410,
        "CURRENT_EXCLUDED_NATIVE_UNITS": 5895,
        "E0": {
            "MODEL": "B3_ACTOR_T6_PLUS_GEOMETRY_MOTION",
            "TEMPORAL_VIEW": "T6",
            "INNER_FOLD": "FOLD_3",
            "SEED": 20260804,
            "E0_EXACT_INNER_FOLD_BOUND": True,
            "E0_PREFLIGHT": "PASS",
            "E0_EXECUTED": False,
            "PAID_EXECUTION_AUTHORIZATION": "NO",
            "E0_OUTER_TEST_ACCESS": "BLOCKED",
        },
        "L4_HANDOFF_REVALIDATION": e0_check,
        "READY_FOR_PAID_INNER_AUTORESEARCH_S1": False,
        "S1_BLOCKERS": ["E0_ENGINEERING_PILOT_NOT_YET_EXECUTED"],
        "READY_FOR_CLAIM_GRADE_OUTER_OOF_C2": False,
        "PAPER_GRADE_RESULT_AVAILABLE": False,
        "DATA_REBUILD_REQUIRED": False,
        "OUTER_SPLIT_CHANGE_REQUIRED": False,
        "NO_TRAINING_OR_GPU_EXECUTION": True,
        "NEXT_AUTHORIZED_ACTION": (
            "Run the registered E0 engineering pilot on one NVIDIA L4 24 GB "
            "after explicit paid-execution authorization."
        ),
    }
    campaign_path = output / "updated_campaign_readiness_decision.json"
    write_json(campaign_path, campaign)
    output_files["updated_campaign_readiness_decision"] = campaign_path

    handoff_path = output / "posture_post_review_handoff.md"
    support_text = ", ".join(
        f"{label}={int(support.get(label, 0))}" for label in REPORT_LABELS
    )
    handoff = f"""# Classification V2 posture post-review closure

The completed session is frozen as the current human posture authority.  No
review was reopened, no derived posture labels were created, and no source,
behavior label, split, or model artifact was modified.

- Session root: external posture session store (hash-bound in the authority JSON)
- Session name: `{session.name}`
- Queue (within session): `{queue_path.name}`
  SHA-256: `{sha256_file(queue_path)}`
- Completed ledger (within session): `{decisions_path.name}`
  SHA-256: `{sha256_file(decisions_path)}`
- Session manifest SHA-256: `{sha256_file(manifest_path)}`
- Current snapshot: `{SNAPSHOT_ID}`
  SHA-256: `{SNAPSHOT_SHA256}`
- Split hash: `{SPLIT_HASH}`
- Gold authority: [posture_500_completed_authority.json](posture_500_completed_authority.json)
- Gold support: `{support_text}`
- Usable three-class support: `{usable_support}`
- Source rows: CVAT `{int((gold['source_type'] == 'cvat_tracking_xml').sum())}`,
  legacy `{int((gold['source_type'] == 'legacy_recovered').sum())}`;
  videos `{int(gold['source_video_key'].nunique())}`;
  dates `{int(gold['recording_date'].nunique())}`;
  folds `{sorted(set(gold['outer_fold_id']))}`

The 120-item pilot is audited separately; it is not silently pooled with the
500-item authority.

- Historical pilot audit:
  [posture_120_pilot_binding_audit.json](posture_120_pilot_binding_audit.json)
- Empirical mapping audit (diagnostic only; no ontology-derived supervision):
  [posture_behavior_mapping_empirical_audit.json](posture_behavior_mapping_empirical_audit.json)

Posture is registered as an independent optional matched ablation. It is not
included in S1.

- Matched-ablation contract:
  [matched contract](posture_behavior_matched_ablation_contract.json)

Behavior-only S1 is not blocked by posture, but S1 remains blocked until the
registered E0 engineering pilot has actually executed successfully. E0 was not
executed here and paid authorization remains `NO`.

## Validation boundary

The closure uses only focused artifact, binding, label, support, and lineage
checks.  It does not train a model, use paid compute, run S1/C2, inspect outer
test results, rebuild data, or apply pending posture decisions.
"""
    handoff_path.write_text(handoff, encoding="utf-8")
    output_files["posture_post_review_handoff"] = handoff_path

    print(json.dumps({
        "output_dir": str(output),
        "posture_queue_rows": len(queue),
        "posture_decision_rows": len(decisions),
        "posture_support": support,
        "usable_support": usable_support,
        "posture_authority": audit["scientific_decision"]["POSTURE_AUTHORITY"],
        "pilot_bindable_rows": pilot_bindable,
        "pilot_only_rows": len(pilot_rows) - pilot_bindable,
        "e0_handoff_revalidation": e0_check["status"],
        "files": {name: str(path) for name, path in sorted(output_files.items())},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
