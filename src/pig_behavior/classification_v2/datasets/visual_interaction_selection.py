"""Freeze the exact short visual-interaction cache target universe.

The output contains image-context identifiers only. Behavior labels and split
metadata remain audit-only, while partner geometry is selected without labels.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.datasets.visual_interaction_context import (
    _canonical_id,
    _context_lookup_key,
    _same_frame_actor_lookup,
    _valid_box,
)

SCHEMA_VERSION = "classification_v2.legacy_development_l6.union_context_short_selection_config.v1"
AUDIT_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l6."
    "union_context_short_selection_audit.v1"
)
LINEAGE_SCOPE = "legacy-only-unreviewed-development"
CANONICAL_SOURCE_NAME = "legacy_16f"
SOURCE_TYPE = "legacy_recovered"
DATASET_ID = "legacy_recovered_16f"

INPUT_NAMES = (
    "training_selection_manifest",
    "training_selection_audit",
    "image_window_context_manifest",
    "image_frame_context_manifest",
    "window_fold_manifest",
)


@dataclass(frozen=True, slots=True)
class VisualInteractionSelectionConfig:
    path: Path
    repo_root: Path
    sha256: str
    payload: dict[str, Any]

    @property
    def output_dir(self) -> Path:
        return _bound_repo_path(
            self.repo_root,
            self.payload["output"]["root_relative_path"],
        )


def load_visual_interaction_selection_config(
    path: Path,
) -> VisualInteractionSelectionConfig:
    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _validate_config_payload(payload)
    repo_root = resolved.parents[2]
    return VisualInteractionSelectionConfig(
        path=resolved,
        repo_root=repo_root,
        sha256=_file_sha256(resolved),
        payload=payload,
    )


def build_visual_interaction_short_selection(
    config: VisualInteractionSelectionConfig,
) -> dict[str, Any]:
    """Write a hash-bound short selection without reading source media."""

    inputs = _validate_inputs(config)
    guard = _git_guard(config)
    if not guard["valid"]:
        raise ValueError(f"union-context selection git guard failed: {guard['errors']}")

    training = pd.read_csv(
        inputs["training_selection_manifest"]["path"],
        low_memory=False,
    )
    parent_audit = json.loads(
        inputs["training_selection_audit"]["path"].read_text(
            encoding="utf-8"
        )
    )
    image_windows = pd.read_csv(
        inputs["image_window_context_manifest"]["path"],
        usecols=[
            "window_id",
            "source_type",
            "dataset_id",
            "window_length_frames",
            "image_context_id_sequence",
            "observed_image_context_rows",
            "loadable_image_context_rows",
            "missing_image_context_slots",
            "window_image_context_complete",
            "lineage_scope",
            "human_review_complete",
        ],
        low_memory=False,
    )
    folds = pd.read_csv(
        inputs["window_fold_manifest"]["path"],
        usecols=[
            "window_id",
            "temporal_unit_key",
            "recording_group_id",
            "video_key",
            "oof_fold_id",
            "source_type",
            "dataset_id",
            "legacy_t6_all_sliding_keep",
        ],
        low_memory=False,
    )
    frame_context = pd.read_csv(
        inputs["image_frame_context_manifest"]["path"],
        usecols=[
            "image_context_id",
            "source_type",
            "dataset_id",
            "video_key",
            "clip_id",
            "resolved_media_path",
            "resolved_media_exists",
            "frame_index",
            "track_id",
            "nearest_track_id",
            "x1",
            "y1",
            "x2",
            "y2",
            "partner_context_available",
            "lineage_scope",
            "human_review_complete",
        ],
        low_memory=False,
    )
    selection, evidence = derive_visual_interaction_short_selection(
        training=training,
        parent_audit=parent_audit,
        image_windows=image_windows,
        folds=folds,
        frame_context=frame_context,
        contract=dict(config.payload["contract"]),
    )

    selection_path = config.output_dir / "visual_context_selection.csv"
    audit_path = config.output_dir / "visual_context_selection_audit.json"
    if selection_path.exists() or audit_path.exists():
        raise FileExistsError(
            "union-context short selection output already exists; "
            "use a new version"
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    selection.to_csv(selection_path, index=False, lineterminator="\n")
    selection_sha256 = _file_sha256(selection_path)

    audit: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "PASS_LEGACY_DEVELOPMENT_L6_UNION_CONTEXT_SHORT_SELECTION",
        "lineage_scope": LINEAGE_SCOPE,
        "canonical_source_name": CANONICAL_SOURCE_NAME,
        "source_type": SOURCE_TYPE,
        "dataset_id": DATASET_ID,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "implementation_sha256": _file_sha256(Path(__file__)),
        "git_guard": guard,
        "inputs": {
            name: {
                "path": str(entry["path"]),
                "sha256": entry["sha256"],
                "size_bytes": entry["size_bytes"],
            }
            for name, entry in inputs.items()
        },
        "selection_csv": str(selection_path),
        "selection_csv_sha256": selection_sha256,
        "selection_columns": list(selection.columns),
        "behavior_labels_in_selection_csv": False,
        "split_fields_in_selection_csv": False,
        "partner_selection_label_gated": False,
        "cache_scope_derived_from_model_selection_universe": True,
        "source_media_reads": 0,
        "cuda_initialized": False,
        **evidence,
        "errors": [],
        "valid": True,
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def derive_visual_interaction_short_selection(
    *,
    training: pd.DataFrame,
    parent_audit: dict[str, Any],
    image_windows: pd.DataFrame,
    folds: pd.DataFrame,
    frame_context: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Derive target IDs and independently prove split and geometry scope."""

    _validate_parent_selection(training, parent_audit, contract)
    routed = _join_fold_authority(training, folds, contract)
    selected_windows, context_ids = _join_image_windows(
        routed,
        image_windows,
        contract,
    )
    selected_frames, static_status = _select_frame_context(
        context_ids,
        frame_context,
        contract,
    )
    ordered_ids = sorted(selected_frames["image_context_id"].astype(str))
    selection = pd.DataFrame({"image_context_id": ordered_ids})

    train = routed.loc[routed["l5_role"].eq("train")]
    validation = routed.loc[routed["l5_role"].eq("validation")]
    train_groups = set(
        zip(
            train["recording_group_id"].astype(str),
            train["video_key"].astype(str),
            strict=True,
        )
    )
    validation_groups = set(
        zip(
            validation["recording_group_id"].astype(str),
            validation["video_key"].astype(str),
            strict=True,
        )
    )
    group_overlap = sorted(train_groups.intersection(validation_groups))
    if group_overlap:
        raise ValueError("union-context short selection has train/validation overlap")

    declared_partner = _strict_bool(
        selected_frames["partner_context_available"],
        name="partner_context_available",
    )
    static_ready = selected_frames["_static_status"].eq("geometry_ready")
    if not declared_partner.equals(static_ready):
        raise ValueError("declared partner availability differs from static geometry")

    evidence = {
        "training_selection_content_sha256": _dataframe_sha256(training),
        "selected_windows": int(len(routed)),
        "train_windows": int(len(train)),
        "validation_windows": int(len(validation)),
        "selected_native_units": int(routed["temporal_unit_key"].nunique()),
        "train_native_units": int(train["temporal_unit_key"].nunique()),
        "validation_native_units": int(validation["temporal_unit_key"].nunique()),
        "selected_image_context_ids": int(len(selection)),
        "selected_image_context_id_sha256": _ordered_sha256(selection["image_context_id"]),
        "static_union_geometry_status_counts": dict(sorted(static_status.items())),
        "static_union_geometry_ready": int(static_ready.sum()),
        "static_union_geometry_unavailable": int((~static_ready).sum()),
        "declared_partner_context_available": int(declared_partner.sum()),
        "train_validation_group_overlap": [],
        "outer_holdout_windows": int(
            routed["oof_fold_id"].astype(str).eq(str(contract["outer_holdout_fold_id"])).sum()
        ),
        "outer_holdout_image_context_ids": 0,
        "selected_window_class_support": {
            role: {
                str(label): int(count)
                for label, count in subset["behavior_label"]
                .astype(str)
                .value_counts()
                .sort_index()
                .items()
            }
            for role, subset in (("train", train), ("validation", validation))
        },
        "selected_window_context_complete": bool(
            _strict_bool(
                selected_windows["window_image_context_complete"],
                name="window_image_context_complete",
            ).all()
        ),
    }
    if evidence["outer_holdout_windows"] != 0:
        raise ValueError("outer-holdout windows entered union-context selection")
    return selection, evidence


def _validate_parent_selection(
    training: pd.DataFrame,
    parent_audit: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    required = {
        "selection_order",
        "position",
        "window_id",
        "temporal_unit_key",
        "recording_group_id",
        "video_key",
        "source_type",
        "dataset_id",
        "behavior_label",
        "l5_role",
        "view_id",
        "sampling_protocol",
        "sequence_length",
        "training_scope",
        "lineage_scope",
        "human_review_complete",
    }
    _require_columns(training, required, "training selection")
    if not parent_audit.get("valid") or parent_audit.get("outer_holdout_rows") != 0:
        raise ValueError("parent training selection audit is not valid and holdout-free")
    observed_hash = _dataframe_sha256(training)
    if parent_audit.get("selection_content_sha256") != observed_hash:
        raise ValueError("parent training selection semantic hash mismatch")
    if training["window_id"].astype(str).duplicated().any():
        raise ValueError("training selection has duplicate window_id")
    expected_order = np.arange(len(training), dtype=np.int64)
    if not np.array_equal(
        training["selection_order"].to_numpy(dtype=np.int64),
        expected_order,
    ):
        raise ValueError("training selection order is not contiguous")
    if set(training["l5_role"].astype(str)) != {"train", "validation"}:
        raise ValueError("training selection exposes a forbidden role")
    _require_constant(training, "source_type", SOURCE_TYPE)
    _require_constant(training, "dataset_id", DATASET_ID)
    _require_constant(training, "view_id", str(contract["view_id"]))
    _require_constant(
        training,
        "sampling_protocol",
        str(contract["sampling_protocol"]),
    )
    _require_constant(
        training,
        "sequence_length",
        int(contract["sequence_length"]),
    )
    _require_constant(
        training,
        "training_scope",
        str(contract["training_scope"]),
    )
    _require_constant(training, "lineage_scope", LINEAGE_SCOPE)
    if _strict_bool(
        training["human_review_complete"],
        name="human_review_complete",
    ).any():
        raise ValueError("unreviewed selection contains reviewed claim flags")
    _require_count(len(training), contract["selected_windows"], "selected windows")
    for role in ("train", "validation"):
        subset = training.loc[training["l5_role"].eq(role)]
        _require_count(
            len(subset),
            contract[f"{role}_windows"],
            f"{role} windows",
        )
        _require_count(
            subset["temporal_unit_key"].nunique(),
            contract[f"{role}_native_units"],
            f"{role} native units",
        )
        per_unit = subset.groupby("temporal_unit_key", sort=False).size()
        if not per_unit.eq(int(contract["windows_per_native_unit"])).all():
            raise ValueError(f"{role} windows-per-native-unit drift")


def _join_fold_authority(
    training: pd.DataFrame,
    folds: pd.DataFrame,
    contract: dict[str, Any],
) -> pd.DataFrame:
    required = {
        "window_id",
        "temporal_unit_key",
        "recording_group_id",
        "video_key",
        "oof_fold_id",
        "source_type",
        "dataset_id",
        "legacy_t6_all_sliding_keep",
    }
    _require_columns(folds, required, "fold manifest")
    wanted = set(training["window_id"].astype(str))
    subset = folds.loc[folds["window_id"].astype(str).isin(wanted)].copy()
    if len(subset) != len(training) or subset["window_id"].duplicated().any():
        raise ValueError("fold authority does not cover selected windows one-to-one")
    subset = subset.rename(
        columns={
            "temporal_unit_key": "fold_temporal_unit_key",
            "recording_group_id": "fold_recording_group_id",
            "video_key": "fold_video_key",
            "source_type": "fold_source_type",
            "dataset_id": "fold_dataset_id",
        }
    )
    joined = training.merge(
        subset,
        on="window_id",
        how="left",
        validate="one_to_one",
    )
    comparisons = {
        "temporal_unit_key": "fold_temporal_unit_key",
        "recording_group_id": "fold_recording_group_id",
        "video_key": "fold_video_key",
        "source_type": "fold_source_type",
        "dataset_id": "fold_dataset_id",
    }
    for left, right in comparisons.items():
        if not joined[left].astype(str).eq(joined[right].astype(str)).all():
            raise ValueError(f"fold authority metadata mismatch: {left}")
    if not _strict_bool(
        joined["legacy_t6_all_sliding_keep"],
        name="legacy_t6_all_sliding_keep",
    ).all():
        raise ValueError("selected fold rows are not valid T6 sliding windows")
    outer_fold = str(contract["outer_holdout_fold_id"])
    validation_fold = str(contract["development_validation_fold_id"])
    validation = joined["l5_role"].eq("validation")
    if not joined.loc[validation, "oof_fold_id"].astype(str).eq(validation_fold).all():
        raise ValueError("validation role differs from validation-fold authority")
    train_folds = set(joined.loc[~validation, "oof_fold_id"].astype(str))
    if outer_fold in train_folds or validation_fold in train_folds:
        raise ValueError("held-out fold entered union-context training scope")
    return joined


def _join_image_windows(
    routed: pd.DataFrame,
    image_windows: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    required = {
        "window_id",
        "source_type",
        "dataset_id",
        "window_length_frames",
        "image_context_id_sequence",
        "observed_image_context_rows",
        "loadable_image_context_rows",
        "missing_image_context_slots",
        "window_image_context_complete",
        "lineage_scope",
        "human_review_complete",
    }
    _require_columns(image_windows, required, "image-window manifest")
    wanted = set(routed["window_id"].astype(str))
    selected = image_windows.loc[image_windows["window_id"].astype(str).isin(wanted)].copy()
    if len(selected) != len(routed) or selected["window_id"].duplicated().any():
        raise ValueError("image-window authority does not cover selection one-to-one")
    selected = routed[["selection_order", "window_id"]].merge(
        selected,
        on="window_id",
        how="left",
        validate="one_to_one",
    )
    selected = selected.sort_values("selection_order", kind="mergesort")
    sequence_length = int(contract["sequence_length"])
    _require_constant(selected, "source_type", SOURCE_TYPE)
    _require_constant(selected, "dataset_id", DATASET_ID)
    _require_constant(selected, "window_length_frames", sequence_length)
    _require_constant(selected, "observed_image_context_rows", sequence_length)
    _require_constant(selected, "loadable_image_context_rows", sequence_length)
    _require_constant(selected, "missing_image_context_slots", 0)
    _require_constant(selected, "lineage_scope", LINEAGE_SCOPE)
    if not _strict_bool(
        selected["window_image_context_complete"],
        name="window_image_context_complete",
    ).all():
        raise ValueError("selected image windows contain incomplete context")
    if _strict_bool(
        selected["human_review_complete"],
        name="human_review_complete",
    ).any():
        raise ValueError("image-window authority has reviewed claim drift")
    context_ids: list[str] = []
    for value in selected["image_context_id_sequence"].astype(str):
        identifiers = value.split(";;")
        if len(identifiers) != sequence_length or any(not item for item in identifiers):
            raise ValueError("image-context sequence length or blank-ID drift")
        context_ids.extend(identifiers)
    unique_ids = sorted(set(context_ids))
    _require_count(
        len(unique_ids),
        contract["selected_image_context_ids"],
        "selected image-context IDs",
    )
    return selected, unique_ids


def _select_frame_context(
    context_ids: list[str],
    frames: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, Counter[str]]:
    required = {
        "image_context_id",
        "source_type",
        "dataset_id",
        "video_key",
        "clip_id",
        "resolved_media_path",
        "resolved_media_exists",
        "frame_index",
        "track_id",
        "nearest_track_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "partner_context_available",
        "lineage_scope",
        "human_review_complete",
    }
    _require_columns(frames, required, "frame-context manifest")
    _require_count(
        len(frames),
        contract["frame_context_rows"],
        "frame-context rows",
    )
    if frames["image_context_id"].astype(str).duplicated().any():
        raise ValueError("frame-context manifest has duplicate image_context_id")
    frame_ids = set(frames["image_context_id"].astype(str))
    missing = sorted(set(context_ids) - frame_ids)
    if missing:
        raise ValueError(f"selected image-context IDs missing from frames: {missing[:5]}")
    lookup = _same_frame_actor_lookup(frames)
    indexed = frames.set_index(frames["image_context_id"].astype(str), drop=False)
    selected = indexed.loc[context_ids].copy().reset_index(drop=True)
    _require_constant(selected, "source_type", SOURCE_TYPE)
    _require_constant(selected, "dataset_id", DATASET_ID)
    _require_constant(selected, "lineage_scope", LINEAGE_SCOPE)
    if _strict_bool(
        selected["human_review_complete"],
        name="human_review_complete",
    ).any():
        raise ValueError("frame-context authority has reviewed claim drift")
    if not _strict_bool(
        selected["resolved_media_exists"],
        name="resolved_media_exists",
    ).all():
        raise ValueError("selected context contains unresolved source media")
    statuses: list[str] = []
    for row in selected.to_dict("records"):
        partner_id = _canonical_id(row.get("nearest_track_id"))
        if not partner_id:
            statuses.append("missing_nearest_partner_bbox")
            continue
        frame_index = int(pd.to_numeric(row["frame_index"], errors="raise"))
        key = _context_lookup_key(
            row,
            frame_index=frame_index,
            track_id=partner_id,
        )
        partner = lookup.get(key)
        if partner is None:
            statuses.append("missing_nearest_partner_bbox")
        elif _valid_box(row) is None or _valid_box(partner) is None:
            statuses.append("invalid_actor_or_partner_bbox")
        else:
            statuses.append("geometry_ready")
    selected["_static_status"] = statuses
    return selected, Counter(statuses)


def _validate_inputs(
    config: VisualInteractionSelectionConfig,
) -> dict[str, dict[str, Any]]:
    payload = config.payload["inputs"]
    resolved: dict[str, dict[str, Any]] = {}
    for name in INPUT_NAMES:
        entry = payload[name]
        path = _bound_repo_path(config.repo_root, entry["path"])
        if not path.is_file():
            raise FileNotFoundError(f"union-context selection input missing: {path}")
        observed = _file_sha256(path)
        if observed != entry["sha256"]:
            raise ValueError(f"union-context selection input hash mismatch: {name}")
        resolved[name] = {
            "path": path,
            "sha256": observed,
            "size_bytes": int(path.stat().st_size),
        }
    return resolved


def _validate_config_payload(payload: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "lineage_scope",
        "canonical_source_name",
        "source_type",
        "dataset_id",
        "human_review_complete",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        "canonical_full_oof_authorized",
        "outer_holdout_predictions_authorized",
        "inputs",
        "contract",
        "execution_guard",
        "output",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"union-context selection config missing keys: {missing}")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "lineage_scope": LINEAGE_SCOPE,
        "canonical_source_name": CANONICAL_SOURCE_NAME,
        "source_type": SOURCE_TYPE,
        "dataset_id": DATASET_ID,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
    }
    mismatches = [name for name, value in expected.items() if payload.get(name) != value]
    if mismatches:
        raise ValueError(f"union-context selection config drift: {mismatches}")
    if set(payload["inputs"]) != set(INPUT_NAMES):
        raise ValueError("union-context selection config input set drift")
    for name in INPUT_NAMES:
        if set(payload["inputs"][name]) != {"path", "sha256"}:
            raise ValueError(f"union-context selection input contract drift: {name}")
    if set(payload["execution_guard"]) != {
        "allowed_dirty_paths",
        "required_tracked_paths",
    }:
        raise ValueError("union-context selection execution-guard drift")


def _git_guard(config: VisualInteractionSelectionConfig) -> dict[str, Any]:
    guard = config.payload["execution_guard"]
    status = _git(
        config.repo_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    entries = [line for line in status.splitlines() if line.strip()]
    observed = sorted(_status_path(line) for line in entries)
    allowed = sorted(str(path).replace("\\", "/") for path in guard["allowed_dirty_paths"])
    unexpected = sorted(set(observed) - set(allowed))
    required_paths = [str(path).replace("\\", "/") for path in guard["required_tracked_paths"]]
    untracked = [
        path
        for path in required_paths
        if subprocess.run(
            [
                "git",
                "-C",
                str(config.repo_root),
                "ls-files",
                "--error-unmatch",
                "--",
                path,
            ],
            capture_output=True,
            check=False,
            text=True,
        ).returncode
        != 0
    ]
    errors: list[str] = []
    if unexpected:
        errors.append(f"unexpected_dirty_paths={unexpected}")
    if untracked:
        errors.append(f"required_paths_untracked={untracked}")
    return {
        "code_sha": _git(config.repo_root, "rev-parse", "HEAD").strip(),
        "dirty_entries": entries,
        "allowed_dirty_paths": allowed,
        "observed_dirty_paths": observed,
        "unexpected_dirty_paths": unexpected,
        "required_tracked_paths": required_paths,
        "untracked_required_paths": untracked,
        "errors": errors,
        "valid": not errors,
    }


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _require_constant(frame: pd.DataFrame, column: str, expected: Any) -> None:
    values = frame[column]
    if isinstance(expected, int):
        valid = pd.to_numeric(values, errors="coerce").eq(expected)
    else:
        valid = values.astype(str).eq(str(expected))
    if not valid.all():
        raise ValueError(f"{column} differs from expected value={expected}")


def _require_count(observed: int, expected: Any, name: str) -> None:
    if int(observed) != int(expected):
        raise ValueError(f"{name}={observed}!={int(expected)}")


def _strict_bool(values: pd.Series, *, name: str) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower()
    allowed = {"true": True, "false": False, "1": True, "0": False}
    invalid = ~normalized.isin(allowed)
    if invalid.any():
        raise ValueError(f"{name} contains invalid boolean values")
    return normalized.map(allowed).astype(bool)


def _dataframe_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ordered_sha256(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values.astype(str):
        if not value.strip():
            raise ValueError("ordered image-context hash contains a blank ID")
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_repo_path(repo_root: Path, value: str) -> Path:
    path = (repo_root / Path(str(value))).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as error:
        raise ValueError(f"path escapes repository: {value}") from error
    return path


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _status_path(line: str) -> str:
    value = line[3:].strip()
    if " -> " in value:
        value = value.rsplit(" -> ", maxsplit=1)[1]
    return value.replace("\\", "/")


__all__ = [
    "VisualInteractionSelectionConfig",
    "build_visual_interaction_short_selection",
    "derive_visual_interaction_short_selection",
    "load_visual_interaction_selection_config",
]
