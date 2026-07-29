"""Build a source-specific final review view for a frozen review scope."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

FINAL_REVIEW_SCHEMA_VERSION = "classification_v2.final_behavior_review.v1"
CONTEXT_PER_SIDE = 6
CONTEXT_RADIUS_FRAMES = 90
BASE_KEY_COLUMNS = ("source_type", "dataset_id", "video_key", "pig_id")
FrameLookup = dict[tuple[str, ...], list[int]]


def _parse_indices(value: object) -> list[int]:
    if pd.isna(value):
        return []
    values: list[int] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(float(token)))
        except (TypeError, ValueError):
            raise ValueError(f"invalid frame index token: {token}") from None
    return values


def _sample_evenly(values: list[int], limit: int) -> list[int]:
    if len(values) <= limit:
        return values
    positions = [
        round(index * (len(values) - 1) / (limit - 1))
        for index in range(limit)
    ]
    return [values[position] for position in positions]


def _lookup_key(unit: pd.Series, actor_column: str) -> tuple[str, ...]:
    return tuple(
        [str(unit[column]) for column in BASE_KEY_COLUMNS]
        + [str(unit.get(actor_column, ""))]
    )


def _build_frame_lookups(
    frames: pd.DataFrame,
) -> tuple[FrameLookup, FrameLookup]:
    lookups: list[FrameLookup] = []
    for actor_column in ("object_track_key", "track_id"):
        if actor_column not in frames:
            lookups.append({})
            continue
        columns = [*BASE_KEY_COLUMNS, actor_column]
        grouped = (
            frames.dropna(subset=["frame_index"])
            .assign(
                **{
                    column: frames[column].fillna("").astype(str)
                    for column in columns
                }
            )
            .groupby(columns, sort=False)["frame_index"]
            .agg(lambda values: sorted(set(values.astype(int))))
        )
        lookups.append(
            {
                tuple(str(value) for value in key): list(values)
                for key, values in grouped.items()
            }
        )
    return lookups[0], lookups[1]


def _available_frames(
    unit: pd.Series,
    object_lookup: FrameLookup,
    track_lookup: FrameLookup,
) -> list[int]:
    if pd.notna(unit.get("object_track_key")):
        values = object_lookup.get(_lookup_key(unit, "object_track_key"))
        if values is not None:
            return values
    if pd.notna(unit.get("track_id")):
        return track_lookup.get(_lookup_key(unit, "track_id"), [])
    return []


def _playback_indices(
    unit: pd.Series,
    available: list[int],
) -> list[int]:
    if str(unit["source_type"]).strip() == "legacy_recovered":
        return _parse_indices(unit["display_frame_indices"])
    targets = _parse_indices(unit["display_frame_indices"])
    if not targets:
        return []
    start = min(targets)
    end = max(targets)
    return [
        frame
        for frame in available
        if start - CONTEXT_RADIUS_FRAMES
        <= frame
        <= end + CONTEXT_RADIUS_FRAMES
    ]


def _context_indices(
    unit: pd.Series,
    playback: list[int],
) -> list[int]:
    if str(unit["source_type"]).strip() == "legacy_recovered":
        return []
    targets = _parse_indices(unit["display_frame_indices"])
    if not targets:
        return []
    start = min(targets)
    end = max(targets)
    before = [
        frame
        for frame in playback
        if frame < start
    ]
    after = [
        frame
        for frame in playback
        if frame > end
    ]
    return _sample_evenly(before, CONTEXT_PER_SIDE) + _sample_evenly(
        after,
        CONTEXT_PER_SIDE,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest-csv", type=Path, required=True)
    parser.add_argument("--frame-features-csv", type=Path, required=True)
    parser.add_argument("--output-view-csv", type=Path, required=True)
    parser.add_argument("--output-audit-json", type=Path, required=True)
    parser.add_argument("--existing-decisions-csv", type=Path)
    parser.add_argument(
        "--authority-role",
        default="FROZEN_FINAL_BEHAVIOR_HUMAN_REVIEW_VIEW",
    )
    parser.add_argument("--expected-item-count", type=int, default=0)
    return parser.parse_args()


def _preserved_keys(path: Path | None) -> set[str]:
    if path is None:
        return set()
    decisions = pd.read_csv(path, usecols=["review_unit_id"])
    ids = decisions["review_unit_id"].fillna("").astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("existing decisions require unique nonblank keys")
    return set(ids)


def main() -> int:
    args = parse_args()
    if args.output_view_csv.exists() or args.output_audit_json.exists():
        raise FileExistsError("refusing to overwrite final review artifacts")

    candidates = pd.read_csv(args.candidate_manifest_csv, low_memory=False)
    if args.expected_item_count and len(candidates) != args.expected_item_count:
        raise ValueError(
            "review scope count mismatch "
            f"expected={args.expected_item_count} actual={len(candidates)}"
        )
    available_columns = set(
        pd.read_csv(args.frame_features_csv, nrows=0).columns
    )
    frame_columns = [
        column
        for column in [
            *BASE_KEY_COLUMNS,
            "object_track_key",
            "track_id",
            "frame_index",
        ]
        if column in available_columns
    ]
    frames = pd.read_csv(
        args.frame_features_csv,
        usecols=frame_columns,
        low_memory=False,
    )
    frames["frame_index"] = pd.to_numeric(
        frames["frame_index"],
        errors="coerce",
    )
    object_lookup, track_lookup = _build_frame_lookups(frames)
    required = {
        "review_unit_id",
        "include_in_review",
        "candidate_tier",
        "source_type",
        "display_frame_indices",
        "behavior_label",
    }
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"candidate manifest missing columns: {missing}")
    ids = candidates["review_unit_id"].astype(str)
    if ids.duplicated().any():
        raise ValueError("candidate manifest has duplicate review keys")
    included = candidates["include_in_review"].astype(str).str.lower().isin(
        {"true", "1", "yes", "y"}
    )
    if not included.all():
        raise ValueError("final review requires every candidate row included")
    auto = candidates["candidate_tier"].astype(str).eq("AUTO_CARRY_LOW_RISK")
    if auto.any():
        raise ValueError("final review cannot include auto-carry rows")

    preserved = _preserved_keys(args.existing_decisions_csv)
    unknown = sorted(preserved.difference(ids))
    if unknown:
        raise ValueError(f"existing decisions outside candidate view: {len(unknown)}")

    view = candidates.copy()
    contexts: list[str] = []
    playback_sequences: list[str] = []
    target_counts: list[int] = []
    context_counts: list[int] = []
    playback_counts: list[int] = []
    target_failures: list[str] = []
    for _, unit in view.iterrows():
        targets = _parse_indices(unit["display_frame_indices"])
        available = _available_frames(unit, object_lookup, track_lookup)
        playback = _playback_indices(unit, available)
        context = _context_indices(unit, playback)
        contexts.append(",".join(str(frame) for frame in context))
        playback_sequences.append(
            ",".join(str(frame) for frame in playback)
        )
        target_counts.append(len(targets))
        context_counts.append(len(context))
        playback_counts.append(len(playback))
        if not set(targets).issubset(set(available)):
            target_failures.append(str(unit["review_unit_id"]))

    view["final_context_frame_indices"] = contexts
    view["final_playback_frame_indices"] = playback_sequences
    view["final_review_schema_version"] = FINAL_REVIEW_SCHEMA_VERSION
    view["final_review_context_contract"] = view["source_type"].map(
        lambda source: (
            "actor_crop_only_no_context"
            if str(source).strip() == "legacy_recovered"
            else "cvat_full_frame_context_extended"
        )
    )
    view["final_target_frame_count"] = target_counts
    view["final_context_frame_count"] = context_counts
    view["final_playback_frame_count"] = playback_counts

    args.output_view_csv.parent.mkdir(parents=True, exist_ok=True)
    view.to_csv(args.output_view_csv, index=False, lineterminator="\n")
    audit = {
        "schema_version": FINAL_REVIEW_SCHEMA_VERSION,
        "authority_role": args.authority_role,
        "candidate_count": int(len(candidates)),
        "view_count": int(len(view)),
        "preserved_existing_review_key_count": int(len(preserved)),
        "new_keys_added": 0,
        "candidate_membership_changed": False,
        "auto_carry_membership_changed": False,
        "decision_schema_changed": False,
        "target_scope_failures": len(target_failures),
        "target_scope_failure_keys": target_failures[:20],
        "target_count_distribution": view[
            "final_target_frame_count"
        ].value_counts().to_dict(),
        "context_count_distribution": view[
            "final_context_frame_count"
        ].value_counts().to_dict(),
        "playback_count_distribution": view[
            "final_playback_frame_count"
        ].value_counts().to_dict(),
        "context_radius_frames_each_side": CONTEXT_RADIUS_FRAMES,
        "contact_sheet_context_samples_each_side": CONTEXT_PER_SIDE,
        "source_distribution": view["source_type"].value_counts().to_dict(),
        "presentation_contracts": view[
            "final_review_context_contract"
        ].value_counts().to_dict(),
        "candidate_manifest_sha256": _sha256(args.candidate_manifest_csv),
        "frame_features_sha256": _sha256(args.frame_features_csv),
        "output_view_sha256": _sha256(args.output_view_csv),
        "review_key_set_sha256": hashlib.sha256(
            "\n".join(view["review_unit_id"].astype(str)).encode()
        ).hexdigest(),
        "presentation_label_policy": (
            "show_current_label_without_machine_reason_or_score"
        ),
        "confirmation_authorized": False,
        "gui_opened": False,
        "decisions_written": False,
    }
    args.output_audit_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        "FINAL_BEHAVIOR_REVIEW_VIEW "
        f"rows={len(view)} "
        f"target_failures={len(target_failures)} "
        f"preserved={len(preserved)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
