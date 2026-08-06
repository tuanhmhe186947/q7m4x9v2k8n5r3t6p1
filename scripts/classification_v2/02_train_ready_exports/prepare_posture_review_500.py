"""Prepare a bounded, blinded posture-review session.

The helper consumes the current reviewed frame authority and writes only
derived queue/audit artifacts.  It does not write labels, alter split roles,
or change the existing GUI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


SNAPSHOT_ID = "reviewed_engineering_amendment_992f34c0204a85a1"
SNAPSHOT_SHA256 = "ab86e2e04267cfdc8248f9bdb8774615479d67a3589f7a25844bb1a4c93a639e"
SPLIT_HASH = "557156a7eb6cceeb6a91f667f7c51dcb286e3111f35f414970fa7431acc7e63b"
QUEUE_SEED = 20260806
QUEUE_TARGETS = {
    "lying_enriched": 200,
    "sitting_enriched": 200,
    "upright_control": 100,
}
DECISION_VALUES = ("upright", "sitting", "lying", "technical_exclude")
SCOPE_COLUMNS = (
    "posture_review_item_id",
    "native_temporal_unit_key",
    "source_type",
    "dataset_id",
    "video_key",
    "source_video_key",
    "object_track_key",
    "pig_id",
    "track_id",
    "recording_date",
    "outer_fold_id",
    "split_role",
    "unit_start_frame",
    "unit_end_frame",
    "label_window_start",
    "label_window_end",
    "observed_frame_start",
    "observed_frame_end",
)
DECISION_COLUMNS = (
    "schema_version",
    "scope_sha256",
    "posture_review_item_id",
    "native_temporal_unit_key",
    "source_type",
    "dataset_id",
    "video_key",
    "object_track_key",
    "pig_id",
    "track_id",
    "unit_start_frame",
    "unit_end_frame",
    "posture_decision",
    "technical_exclusion_reason",
    "reviewer",
    "reviewed_at",
)
DECISION_SCHEMA_VERSION = "classification_v2.posture_pilot_decisions.v1"
DATE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def clean(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def date_token(video_key: Any, source_video_key: Any) -> str:
    for value in (source_video_key, video_key):
        match = DATE_RE.search(clean(value))
        if match:
            return match.group(1)
    return "UNKNOWN"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return clean(value).lower() in {"true", "1", "yes", "y", "t"}


def _all_true(values: pd.Series) -> bool:
    return all(_as_bool(value) for value in values)


def _all_false(values: pd.Series) -> bool:
    return all(not _as_bool(value) for value in values)


def _first_nonempty(values: pd.Series) -> str:
    for value in values:
        text = clean(value)
        if text:
            return text
    return ""


def read_current_units(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "temporal_unit_key",
        "source_type",
        "dataset_id",
        "video_key",
        "object_track_key",
        "pig_id",
        "track_id",
        "frame_index",
        "label_window_start",
        "label_window_end",
        "behavior_reviewed_final",
        "source_sequence_length",
        "sample_weight",
        "temporal_unit_stable_for_training",
        "temporal_interval_complete",
        "temporal_harmonization_valid",
        "actor_bbox_valid",
        "bbox_valid",
        "sequence_complete",
        "sequence_range_valid",
        "spatiotemporal_feature_valid",
        "include_in_training",
        "behavior_review_include_in_training",
        "post_review_frame_transition_unit_excluded",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"reviewed authority missing columns: {missing}")

    frame["temporal_unit_key"] = frame["temporal_unit_key"].map(clean)
    if frame["temporal_unit_key"].eq("").any():
        raise ValueError("reviewed authority has blank temporal unit keys")

    grouped = frame.groupby("temporal_unit_key", sort=True, dropna=False)
    unit = grouped.first().reset_index()
    summary = grouped.agg(
        unit_frame_count=("frame_index", "nunique"),
        unit_frame_start=("frame_index", "min"),
        unit_frame_end=("frame_index", "max"),
        unit_row_count=("frame_index", "size"),
        bbox_valid_all=("bbox_valid", _all_true),
        actor_bbox_valid_all=("actor_bbox_valid", _all_true),
        sequence_complete_all=("sequence_complete", _all_true),
        sequence_range_valid_all=("sequence_range_valid", _all_true),
        temporal_interval_complete_all=("temporal_interval_complete", _all_true),
        temporal_harmonization_valid_all=("temporal_harmonization_valid", _all_true),
        spatiotemporal_feature_valid_all=("spatiotemporal_feature_valid", _all_true),
        stable_training_all=("temporal_unit_stable_for_training", _all_true),
        include_training_all=("include_in_training", _all_true),
        behavior_include_training_all=("behavior_review_include_in_training", _all_true),
        transition_excluded_any=(
            "post_review_frame_transition_unit_excluded",
            lambda values: any(_as_bool(value) for value in values),
        ),
        behavior_consistency_all=("behavior_consistency_in_unit", _all_true),
        behavior_needs_review_any=(
            "temporal_unit_needs_review",
            lambda values: any(_as_bool(value) for value in values),
        ),
    ).reset_index()
    unit = unit.merge(
        summary,
        on="temporal_unit_key",
        how="left",
        validate="one_to_one",
    )
    unit["native_temporal_unit_key"] = unit["temporal_unit_key"]
    unit["observed_frame_start"] = unit["unit_frame_start"].astype(int)
    unit["observed_frame_end"] = unit["unit_frame_end"].astype(int)
    unit["unit_start_frame"] = unit["unit_frame_start"].astype(int)
    unit["unit_end_frame"] = unit["unit_frame_end"].astype(int)
    unit["label_window_start"] = pd.to_numeric(
        unit["label_window_start"], errors="raise"
    ).astype(int)
    unit["label_window_end"] = pd.to_numeric(
        unit["label_window_end"], errors="raise"
    ).astype(int)
    unit["observed_frame_start"] = unit["observed_frame_start"].astype(int)
    unit["observed_frame_end"] = unit["observed_frame_end"].astype(int)
    unit["recording_date"] = [
        date_token(video, source)
        for video, source in zip(
            unit["video_key"],
            unit.get("source_video_key", pd.Series(index=unit.index)),
            strict=True,
        )
    ]
    unit["queue_behavior_context"] = unit["behavior_reviewed_final"].map(clean)
    fallback = unit.get("dominant_behavior_in_unit", pd.Series(index=unit.index)).map(clean)
    unit.loc[unit["queue_behavior_context"].eq(""), "queue_behavior_context"] = fallback
    unit["queue_stratum"] = "upright_control"
    unit.loc[unit["queue_behavior_context"].eq("lying"), "queue_stratum"] = (
        "lying_enriched"
    )
    unit.loc[unit["queue_behavior_context"].eq("sitting"), "queue_stratum"] = (
        "sitting_enriched"
    )
    unit["source_sequence_length"] = pd.to_numeric(
        unit["source_sequence_length"], errors="raise"
    ).astype(int)
    unit["sample_weight"] = pd.to_numeric(unit["sample_weight"], errors="coerce")
    unit["media_contract_valid"] = (
        unit["unit_frame_count"].eq(unit["source_sequence_length"])
        & unit["unit_end_frame"].ge(unit["unit_start_frame"])
        & unit["bbox_valid_all"]
        & unit["actor_bbox_valid_all"]
        & unit["temporal_interval_complete_all"]
        & unit["temporal_harmonization_valid_all"]
        & unit["stable_training_all"]
        & unit["include_training_all"]
        & unit["behavior_include_training_all"]
        & ~unit["transition_excluded_any"]
        & unit["sample_weight"].gt(0)
    )
    return frame, unit


def discover_decision_ledgers(roots: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for root in roots:
        if root.exists():
            paths.update(root.rglob("posture_pilot_decisions.csv"))
    return sorted(path.resolve() for path in paths)


def inventory_decisions(
    paths: list[Path],
    units: pd.DataFrame,
    *,
    reviewed_snapshot_sha: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    unit_map = units.set_index("native_temporal_unit_key", drop=False)
    rows: list[dict[str, Any]] = []
    key_records: dict[str, dict[str, Any]] = {}
    for path in paths:
        ledger_sha = sha256_file(path)
        decisions = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = sorted(set(DECISION_COLUMNS).difference(decisions.columns))
        if missing:
            raise ValueError(f"decision ledger missing columns: {path}: {missing}")
        for record in decisions.to_dict(orient="records"):
            key = clean(record["native_temporal_unit_key"])
            decision = clean(record["posture_decision"])
            valid = bool(key and decision in DECISION_VALUES)
            current = unit_map.loc[key] if key in unit_map.index else None
            metadata_match = False
            if current is not None:
                metadata_match = all(
                    clean(record[field]) == clean(current[field])
                    for field in (
                        "source_type",
                        "dataset_id",
                        "video_key",
                        "object_track_key",
                        "pig_id",
                        "track_id",
                    )
                ) and int(record["unit_start_frame"]) == int(current["unit_start_frame"])
                metadata_match = metadata_match and int(record["unit_end_frame"]) == int(
                    current["unit_end_frame"]
                )
            aligned = bool(valid and current is not None and metadata_match)
            rows.append(
                {
                    "ledger_path": str(path),
                    "ledger_sha256": ledger_sha,
                    "source_session": path.parent.name,
                    "posture_review_item_id": clean(record["posture_review_item_id"]),
                    "native_temporal_unit_key": key,
                    "posture_decision": decision,
                    "scope_sha256": clean(record["scope_sha256"]),
                    "scope_hash_matches_current_snapshot": clean(record["scope_sha256"])
                    == reviewed_snapshot_sha,
                    "current_key_found": current is not None,
                    "current_snapshot_alignment_valid": aligned,
                    "valid_decision": valid,
                }
            )
            if valid:
                item = key_records.setdefault(
                    key,
                    {
                        "native_temporal_unit_key": key,
                        "decisions": set(),
                        "source_ledgers": set(),
                        "current_snapshot_alignment_valid": True,
                    },
                )
                item["decisions"].add(decision)
                item["source_ledgers"].add(str(path))
                item["current_snapshot_alignment_valid"] = bool(
                    item["current_snapshot_alignment_valid"] and aligned
                )

    inventory = pd.DataFrame(rows)
    serialized_records = []
    for key in sorted(key_records):
        item = key_records[key]
        serialized_records.append(
            {
                "native_temporal_unit_key": key,
                "decisions": sorted(item["decisions"]),
                "source_ledgers": sorted(item["source_ledgers"]),
                "current_snapshot_alignment_valid": item[
                    "current_snapshot_alignment_valid"
                ],
            }
        )
    exclusion = {
        "schema_version": "classification_v2.existing_posture_exclusion.v1",
        "reviewed_snapshot_sha256": reviewed_snapshot_sha,
        "decision_ledger_count": len(paths),
        "unique_decided_key_count": len(serialized_records),
        "keys": [record["native_temporal_unit_key"] for record in serialized_records],
        "key_records": serialized_records,
        "ledger_hashes": [
            {"path": str(path), "sha256": sha256_file(path)} for path in paths
        ],
    }
    return inventory, exclusion


def build_fold_bindings(
    effective_path: Path | None,
    split_path: Path | None,
) -> dict[str, dict[str, str]]:
    if effective_path is None or split_path is None:
        return {}
    if not effective_path.exists() or not split_path.exists():
        return {}
    split = pd.read_csv(
        split_path,
        usecols=["window_id", "split", "model_split_role", "outer_fold_id"],
        low_memory=False,
    )
    split = split.drop_duplicates("window_id", keep="first")
    effective = pd.read_csv(
        effective_path,
        usecols=[
            "window_id",
            "temporal_unit_keys_json",
            "window_valid_for_main_train",
            "window_exclusion_reason",
        ],
        low_memory=False,
    )
    joined = effective.merge(split, on="window_id", how="inner", validate="one_to_one")
    bindings: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"outer_fold_id": set(), "split_role": set()}
    )
    for record in joined.to_dict(orient="records"):
        try:
            keys = json.loads(clean(record["temporal_unit_keys_json"]))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid temporal_unit_keys_json: {exc}") from exc
        for key in keys:
            key = clean(key)
            if key:
                bindings[key]["outer_fold_id"].add(clean(record["outer_fold_id"]))
                role = clean(record["model_split_role"]) or clean(record["split"])
                bindings[key]["split_role"].add(role)
    return {
        key: {
            "outer_fold_id": "|".join(sorted(value["outer_fold_id"])),
            "split_role": "|".join(sorted(value["split_role"])),
        }
        for key, value in bindings.items()
    }


def _video_aliases(path: Path) -> set[str]:
    stem = path.stem.lower()
    aliases = {path.name.lower(), stem}
    if stem.endswith("_30fps"):
        base = stem[: -len("_30fps")]
        aliases.update({base, f"{base}.mp4"})
    for suffix in ("_30fps", "-30fps", " 30fps"):
        if stem.endswith(suffix):
            base = stem[: -len(suffix)]
            aliases.update({base, f"{base}.mp4"})
    return aliases


def build_video_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not root.exists():
        return index
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".mpg",
            ".mpeg",
            ".m4v",
        }:
            for alias in _video_aliases(path):
                index.setdefault(alias.replace("\\", "/").lower(), path)
    return index


def resolve_video(row: pd.Series, index: dict[str, Path]) -> Path | None:
    raw_keys = [clean(row.get("video_key")), clean(row.get("source_video_key"))]
    candidates: list[str] = []
    for raw_key in raw_keys:
        if not raw_key:
            continue
        key = raw_key.replace("\\", "/").lower()
        stem = Path(key).stem.lower()
        stems = [stem]
        for prefix in ("test video ", "tracking_annotation_", "tracking annotation "):
            if stem.startswith(prefix):
                stems.append(stem[len(prefix) :])
        for candidate in stems:
            candidates.extend(
                [
                    key,
                    candidate,
                    f"{candidate}.mp4",
                    f"{candidate}_30fps",
                    f"{candidate}_30fps.mp4",
                ]
            )
    for candidate in candidates:
        if candidate in index:
            return index[candidate]
    return None


def resolve_crop(raw: Any, raw_root: Path) -> Path | None:
    text = clean(raw)
    if not text:
        return None
    path = Path(text)
    candidates = [path]
    if "crops" in path.parts:
        crop_index = list(path.parts).index("crops")
        candidates.append(raw_root.joinpath(*path.parts[crop_index + 1 :]))
    return next((candidate for candidate in candidates if candidate.exists()), None)


def mark_media_valid(
    frame: pd.DataFrame,
    units: pd.DataFrame,
    *,
    video_root: Path,
    raw_root: Path,
) -> pd.DataFrame:
    video_index = build_video_index(video_root)
    video_paths: dict[str, Path | None] = {}
    for record in units.to_dict(orient="records"):
        if clean(record["source_type"]) != "legacy_recovered":
            key = clean(record["video_key"])
            video_paths.setdefault(key, resolve_video(pd.Series(record), video_index))

    valid_by_key: dict[str, bool] = {}
    for key, rows in frame.groupby("temporal_unit_key", sort=True):
        source = clean(rows["source_type"].iloc[0])
        if source == "legacy_recovered":
            valid_by_key[key] = all(
                resolve_crop(value, raw_root) is not None
                for value in rows["crop_path"]
            )
        else:
            path = video_paths.get(clean(rows["video_key"].iloc[0]))
            valid_by_key[key] = path is not None and all(
                int(value) >= 0 for value in rows["frame_index"]
            )
    units = units.copy()
    units["media_path_contract_valid"] = units["temporal_unit_key"].map(
        valid_by_key
    ).fillna(False)
    return units


def difficulty_score(record: pd.Series) -> int:
    score = 0
    if _as_bool(record.get("behavior_needs_review_any")):
        score += 4
    if not _as_bool(record.get("behavior_consistency_all")):
        score += 3
    if clean(record.get("temporal_consistency_status")) not in {"", "stable"}:
        score += 2
    if clean(record.get("context_quality")) not in {"", "good", "high"}:
        score += 1
    return score


def select_queue(
    units: pd.DataFrame,
    excluded_keys: set[str],
    *,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    eligible = units.loc[
        units["media_contract_valid"]
        & units["media_path_contract_valid"]
        & ~units["native_temporal_unit_key"].isin(excluded_keys)
    ].copy()
    eligible["difficulty_score"] = eligible.apply(difficulty_score, axis=1)
    eligible["selection_hash"] = eligible.apply(
        lambda row: hashlib.sha256(
            f"{seed}|{row['queue_stratum']}|{row['native_temporal_unit_key']}".encode(
                "utf-8"
            )
        ).hexdigest(),
        axis=1,
    )
    capacity_by_stratum = {}
    for stratum, target in QUEUE_TARGETS.items():
        population = eligible.loc[eligible["queue_stratum"].eq(stratum)]
        video_capacity = int(population["video_key"].value_counts().clip(upper=25).sum())
        date_capacity = int(
            population["recording_date"].value_counts().clip(upper=75).sum()
        )
        capacity_by_stratum[stratum] = {
            "eligible": len(population),
            "target": target,
            "video_cap_feasible": video_capacity >= target,
            "date_cap_feasible": date_capacity >= target,
            "video_capacity": video_capacity,
            "date_capacity": date_capacity,
        }
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    video_counts: Counter[str] = Counter()
    date_counts: Counter[str] = Counter()
    for stratum, target in QUEUE_TARGETS.items():
        population = eligible.loc[eligible["queue_stratum"].eq(stratum)].copy()
        report = capacity_by_stratum[stratum]
        if len(population) < target:
            raise ValueError(f"insufficient eligible {stratum}: {len(population)} < {target}")
        enforce_video = bool(report["video_cap_feasible"])
        enforce_date = bool(report["date_cap_feasible"])
        population = population.sort_values(
            ["difficulty_score", "selection_hash", "native_temporal_unit_key"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        count = 0
        for record in population.to_dict(orient="records"):
            key = clean(record["native_temporal_unit_key"])
            video = clean(record["video_key"])
            date = clean(record["recording_date"])
            if key in selected_keys:
                continue
            if enforce_video and video_counts[video] >= 25:
                continue
            if enforce_date and date_counts[date] >= 75:
                continue
            selected.append(record)
            selected_keys.add(key)
            video_counts[video] += 1
            date_counts[date] += 1
            count += 1
            if count == target:
                break
        if count != target:
            raise ValueError(f"could not select {target} valid {stratum} rows; selected={count}")

    queue = pd.DataFrame(selected)
    queue = queue.sort_values(
        ["queue_stratum", "difficulty_score", "selection_hash", "native_temporal_unit_key"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    queue.insert(
        0,
        "posture_review_item_id",
        [f"posture_review_{index:07d}" for index in range(1, len(queue) + 1)],
    )
    missing_sources = sorted(
        set(units["source_type"].map(clean))
        - set(queue["source_type"].map(clean))
    )
    missing_folds = sorted(
        set(units["outer_fold_id"].map(clean))
        - set(queue["outer_fold_id"].map(clean))
        - {""}
    )
    audit = {
        "schema_version": "classification_v2.posture_500_selection_audit.v1",
        "queue_seed": seed,
        "target_rows": 500,
        "eligible_population_rows": len(eligible),
        "eligible_population_hash": canonical_sha256(
            eligible["native_temporal_unit_key"].astype(str).tolist()
        ),
        "stratum_targets": QUEUE_TARGETS,
        "actual_stratum_counts": queue["queue_stratum"].value_counts().to_dict(),
        "video_counts": queue["video_key"].map(clean).value_counts().to_dict(),
        "date_counts": queue["recording_date"].map(clean).value_counts().to_dict(),
        "source_counts": queue["source_type"].map(clean).value_counts().to_dict(),
        "outer_fold_counts": queue["outer_fold_id"].map(clean).value_counts().to_dict(),
        "missing_sources": missing_sources,
        "missing_outer_folds": missing_folds,
        "capacity_by_stratum": capacity_by_stratum,
        "video_cap": 25,
        "date_cap": 75,
        "difficulty_score_mean": float(queue["difficulty_score"].mean()),
    }
    return queue, audit


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed-csv", type=Path, required=True)
    parser.add_argument("--decision-root", type=Path, action="append", required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--effective-window-index", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--queue-seed", type=int, default=QUEUE_SEED)
    parser.add_argument("--code-sha", default="364c36e05018a3a213374c51d11e4e77148a6ca5")
    args = parser.parse_args()

    if args.queue_seed < 0:
        raise SystemExit("queue seed must be non-negative")
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    frame, units = read_current_units(args.reviewed_csv)
    fold_bindings = build_fold_bindings(args.effective_window_index, args.split_manifest)
    units["outer_fold_id"] = units["native_temporal_unit_key"].map(
        lambda key: fold_bindings.get(key, {}).get("outer_fold_id", "")
    )
    units["split_role"] = units["native_temporal_unit_key"].map(
        lambda key: fold_bindings.get(key, {}).get("split_role", "")
    )
    if units["outer_fold_id"].eq("").any():
        missing_count = int(units["outer_fold_id"].eq("").sum())
        raise SystemExit(f"unbound current outer-fold units: {missing_count}")
    units = mark_media_valid(
        frame,
        units,
        video_root=args.video_root,
        raw_root=args.raw_root,
    )
    print(
        "FILTER_COUNTS",
        units.groupby("queue_stratum")[
            [
                "media_contract_valid",
                "media_path_contract_valid",
                "bbox_valid_all",
                "actor_bbox_valid_all",
                "sequence_complete_all",
                "sequence_range_valid_all",
                "temporal_interval_complete_all",
                "temporal_harmonization_valid_all",
                "stable_training_all",
                "include_training_all",
                "behavior_include_training_all",
                "sample_weight",
            ]
        ].agg(
            media_contract_valid=("media_contract_valid", "sum"),
            media_path_contract_valid=("media_path_contract_valid", "sum"),
            bbox_valid_all=("bbox_valid_all", "sum"),
            actor_bbox_valid_all=("actor_bbox_valid_all", "sum"),
            sequence_complete_all=("sequence_complete_all", "sum"),
            sequence_range_valid_all=("sequence_range_valid_all", "sum"),
            temporal_interval_complete_all=("temporal_interval_complete_all", "sum"),
            temporal_harmonization_valid_all=(
                "temporal_harmonization_valid_all",
                "sum",
            ),
            stable_training_all=("stable_training_all", "sum"),
            include_training_all=("include_training_all", "sum"),
            behavior_include_training_all=("behavior_include_training_all", "sum"),
            nonzero_weight=("sample_weight", lambda values: int(values.gt(0).sum())),
        ).to_dict(orient="index"),
    )

    ledgers = discover_decision_ledgers(args.decision_root)
    if not ledgers:
        raise SystemExit("no posture decision ledger found")
    inventory, exclusion = inventory_decisions(
        ledgers,
        units,
        reviewed_snapshot_sha=SNAPSHOT_SHA256,
    )
    exclusion_path = args.audit_dir / "existing_posture_review_exclusion_set.json"
    inventory_path = args.audit_dir / "existing_posture_review_inventory.csv"
    inventory.to_csv(inventory_path, index=False, lineterminator="\n")
    write_json(exclusion_path, exclusion)
    excluded_keys = set(exclusion["keys"])

    queue, selection_audit = select_queue(
        units,
        excluded_keys,
        seed=args.queue_seed,
    )
    queue_path_local = args.audit_dir / "posture_review_session" / "posture_review_scope.csv"
    decisions_path_local = (
        args.audit_dir / "posture_review_session" / "posture_pilot_decisions.csv"
    )
    manifest_path_local = args.audit_dir / "posture_review_session" / "session_manifest.json"
    queue_path_local.parent.mkdir(parents=True, exist_ok=True)
    queue[list(SCOPE_COLUMNS)].to_csv(
        queue_path_local,
        index=False,
        lineterminator="\n",
    )
    empty_decisions = pd.DataFrame(columns=DECISION_COLUMNS)
    empty_decisions.to_csv(decisions_path_local, index=False, lineterminator="\n")
    queue_hash = sha256_file(queue_path_local)
    session_name = f"posture_500_{queue_hash[:16]}"
    session_path = args.session_root / session_name
    manifest = {
        "schema_version": "classification_v2.posture_review_session.v1",
        "session_name": session_name,
        "session_path": str(session_path),
        "queue_path": str(session_path / "posture_review_scope.csv"),
        "decisions_path": str(session_path / "posture_pilot_decisions.csv"),
        "staging_queue_path": str(queue_path_local.resolve()),
        "staging_decisions_path": str(decisions_path_local.resolve()),
        "reviewed_snapshot": SNAPSHOT_ID,
        "reviewed_snapshot_sha256": SNAPSHOT_SHA256,
        "split_hash": SPLIT_HASH,
        "reviewed_frame_features_path": str(args.reviewed_csv.resolve()),
        "reviewed_frame_features_sha256": sha256_file(args.reviewed_csv),
        "queue_sha256": queue_hash,
        "queue_construction_seed": args.queue_seed,
        "candidate_population_hash": selection_audit["eligible_population_hash"],
        "exclusion_set_hash": sha256_file(exclusion_path),
        "source_code_sha": args.code_sha,
        "decision_schema_version": DECISION_SCHEMA_VERSION,
        "decision_values": DECISION_VALUES,
        "initial_decision_rows": 0,
        "prefilled_human_decisions": 0,
        "queue_order_frozen": True,
        "stratum_counts": selection_audit["actual_stratum_counts"],
        "selection_audit": selection_audit,
    }
    write_json(manifest_path_local, manifest)
    selection_audit["session_name"] = session_name
    selection_audit["session_path"] = str(session_path)
    selection_audit["queue_path"] = str(session_path / "posture_review_scope.csv")
    selection_audit["decisions_path"] = str(session_path / "posture_pilot_decisions.csv")
    selection_audit["queue_sha256"] = queue_hash
    selection_audit["session_manifest_sha256"] = sha256_file(manifest_path_local)
    selection_audit["reviewed_snapshot"] = SNAPSHOT_ID
    selection_audit["reviewed_snapshot_sha256"] = SNAPSHOT_SHA256
    selection_audit["split_hash"] = SPLIT_HASH
    write_json(args.audit_dir / "posture_500_queue_audit.json", selection_audit)
    selection_manifest = {
        "schema_version": "classification_v2.posture_500_candidate_manifest.v1",
        "queue_seed": args.queue_seed,
        "selection_audit_path": str((args.audit_dir / "posture_500_queue_audit.json").resolve()),
        "candidate_population_hash": selection_audit["eligible_population_hash"],
        "rows": queue[
            [
                "posture_review_item_id",
                "native_temporal_unit_key",
                "queue_stratum",
                "queue_behavior_context",
                "difficulty_score",
                "source_type",
                "video_key",
                "recording_date",
                "outer_fold_id",
            ]
        ].to_dict(orient="records"),
    }
    write_json(args.audit_dir / "posture_500_candidate_manifest.json", selection_manifest)
    print(json.dumps(selection_audit, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
