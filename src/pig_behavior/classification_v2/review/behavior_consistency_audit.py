"""Read-only consistency audit for completed burst-level Behavior review."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.review.behavior_review_contract import (
    BEHAVIOR_REVIEW_TEMPLATE,
    SOURCE_UNIT_CONTRACTS,
)

INTERACTION_BEHAVIORS = frozenset({"fight", "social-nose"})
TERMINAL_DECISIONS = frozenset({"accept", "corrected", "exclude"})

FRAME_REQUIRED_COLUMNS = (
    "source_type",
    "dataset_id",
    "video_key",
    "object_track_key",
    "frame_index",
    "pig_id",
    "track_id",
    "temporal_unit_key",
    "behavior_temporal_final",
    "nearest_partner_key",
)

DECISION_REQUIRED_COLUMNS = (
    "review_item_id",
    "review_unit_id",
    "temporal_unit_key",
    "behavior_label",
    "manual_review_decision",
    "manual_corrected_behavior",
    "manual_note",
)

NOTE_RISK_PATTERN = re.compile(
    r"fight|social|hung|cắn|nhầm|nửa|đoạn đầu|đoạn sau",
    flags=re.IGNORECASE,
)


def build_effective_behavior_tables(
    frame_features: pd.DataFrame,
    decisions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Resolve reviewed labels over the immutable full temporal-unit table."""
    _require_columns(frame_features, FRAME_REQUIRED_COLUMNS, table="frame_features")
    _require_columns(decisions, DECISION_REQUIRED_COLUMNS, table="decisions")

    frames = frame_features.copy()
    reviewed = decisions.copy()
    for column in FRAME_REQUIRED_COLUMNS:
        if column != "frame_index":
            frames[column] = frames[column].map(_text)
    for column in DECISION_REQUIRED_COLUMNS:
        reviewed[column] = reviewed[column].map(_text)

    if frames["temporal_unit_key"].eq("").any():
        raise ValueError("frame_features contains blank temporal_unit_key")
    if reviewed["temporal_unit_key"].eq("").any():
        raise ValueError("decisions contains blank temporal_unit_key")
    duplicate_decisions = reviewed["temporal_unit_key"].duplicated(keep=False)
    if duplicate_decisions.any():
        raise ValueError(
            "duplicate decision temporal units="
            f"{int(duplicate_decisions.sum())}"
        )

    invalid_decisions = ~reviewed["manual_review_decision"].isin(
        TERMINAL_DECISIONS
    )
    if invalid_decisions.any():
        values = sorted(reviewed.loc[invalid_decisions, "manual_review_decision"].unique())
        raise ValueError(f"non-terminal completed-review decisions={values}")

    corrected = reviewed["manual_review_decision"].eq("corrected")
    invalid_correction = corrected & reviewed["manual_corrected_behavior"].eq("")
    unexpected_correction = ~corrected & reviewed["manual_corrected_behavior"].ne("")
    if invalid_correction.any() or unexpected_correction.any():
        raise ValueError(
            "invalid correction payloads="
            f"{int(invalid_correction.sum() + unexpected_correction.sum())}"
        )

    unit_label_counts = frames.groupby("temporal_unit_key", sort=False)[
        "behavior_temporal_final"
    ].nunique(dropna=False)
    unstable_units = unit_label_counts.gt(1)
    if unstable_units.any():
        raise ValueError(
            "behavior_temporal_final varies inside temporal units="
            f"{int(unstable_units.sum())}"
        )

    numeric_frames = pd.to_numeric(frames["frame_index"], errors="coerce")
    if numeric_frames.isna().any():
        raise ValueError(
            f"invalid frame_index rows={int(numeric_frames.isna().sum())}"
        )
    frames["frame_index"] = numeric_frames.astype(int)

    grouped = frames.groupby("temporal_unit_key", sort=False)
    units = grouped.agg(
        source_type=("source_type", "first"),
        dataset_id=("dataset_id", "first"),
        video_key=("video_key", "first"),
        object_track_key=("object_track_key", "first"),
        pig_id=("pig_id", "first"),
        track_id=("track_id", "first"),
        unit_start_frame=("frame_index", "min"),
        unit_end_frame=("frame_index", "max"),
        frame_count=("frame_index", "nunique"),
        original_behavior=("behavior_temporal_final", "first"),
    ).reset_index()

    decision_columns = [
        "review_item_id",
        "review_unit_id",
        "temporal_unit_key",
        "behavior_label",
        "manual_review_decision",
        "manual_corrected_behavior",
        "manual_note",
    ]
    units = units.merge(
        reviewed[decision_columns],
        on="temporal_unit_key",
        how="left",
        validate="one_to_one",
    )
    for column in decision_columns:
        if column != "temporal_unit_key":
            units[column] = units[column].fillna("").map(_text)

    units["reviewed"] = units["manual_review_decision"].ne("")
    source_mismatch = units["reviewed"] & units["behavior_label"].ne(
        units["original_behavior"]
    )
    if source_mismatch.any():
        raise ValueError(
            "decision/source original label mismatch="
            f"{int(source_mismatch.sum())}"
        )

    units["effective_behavior"] = units["original_behavior"]
    correction_mask = units["manual_review_decision"].eq("corrected")
    units.loc[correction_mask, "effective_behavior"] = units.loc[
        correction_mask, "manual_corrected_behavior"
    ]
    units.loc[units["manual_review_decision"].eq("exclude"), "effective_behavior"] = ""

    effective = frames.merge(
        units[["temporal_unit_key", "effective_behavior", "reviewed"]],
        on="temporal_unit_key",
        how="left",
        validate="many_to_one",
    )
    audit = {
        "frame_rows": int(len(frames)),
        "temporal_units": int(len(units)),
        "decision_rows": int(len(reviewed)),
        "matched_decision_units": int(units["reviewed"].sum()),
        "corrected_units": int(correction_mask.sum()),
        "excluded_units": int(
            units["manual_review_decision"].eq("exclude").sum()
        ),
        "errors": [],
    }
    return units, effective, audit


def build_interaction_pair_findings(
    effective_frames: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find suspicious nearest-pair disagreement under project semantics."""
    required = (*FRAME_REQUIRED_COLUMNS, "effective_behavior", "reviewed")
    _require_columns(effective_frames, required, table="effective_frames")

    keys = ["source_type", "dataset_id", "video_key", "frame_index"]
    lookup_columns = [
        *keys,
        "object_track_key",
        "temporal_unit_key",
        "pig_id",
        "effective_behavior",
        "reviewed",
    ]
    lookup = effective_frames[lookup_columns].copy()
    duplicate_lookup = lookup.duplicated([*keys, "object_track_key"], keep=False)
    if duplicate_lookup.any():
        raise ValueError(
            f"duplicate frame/object lookup rows={int(duplicate_lookup.sum())}"
        )
    lookup = lookup.rename(
        columns={
            "object_track_key": "partner_object_track_key",
            "temporal_unit_key": "partner_temporal_unit_key",
            "pig_id": "partner_pig_id",
            "effective_behavior": "partner_effective_behavior",
            "reviewed": "partner_reviewed",
        }
    )

    actors = effective_frames.loc[
        effective_frames["effective_behavior"].isin(INTERACTION_BEHAVIORS)
    ].copy()
    pairs = actors.merge(
        lookup,
        left_on=[*keys, "nearest_partner_key"],
        right_on=[*keys, "partner_object_track_key"],
        how="left",
        validate="many_to_one",
    )
    pairs["partner_matched"] = pairs["partner_temporal_unit_key"].notna()
    pairs["partner_is_fight"] = pairs["partner_effective_behavior"].eq("fight")
    pairs["partner_is_social_nose"] = pairs[
        "partner_effective_behavior"
    ].eq("social-nose")

    stats_rows: list[dict[str, Any]] = []
    finding_rows: list[dict[str, Any]] = []
    for temporal_unit_key, rows in pairs.groupby("temporal_unit_key", sort=True):
        total = len(rows)
        matched = int(rows["partner_matched"].sum())
        partner_fight = int(rows["partner_is_fight"].sum())
        partner_social = int(rows["partner_is_social_nose"].sum())
        actor_behavior = _text(rows["effective_behavior"].iloc[0])
        related = _joined_unique(rows["partner_temporal_unit_key"])
        partner_ids = _joined_unique(rows["partner_pig_id"])
        matched_ratio = matched / total if total else 0.0
        fight_ratio = partner_fight / matched if matched else 0.0
        social_ratio = partner_social / matched if matched else 0.0
        stats = {
            "temporal_unit_key": temporal_unit_key,
            "effective_behavior": actor_behavior,
            "frame_count": total,
            "partner_matched_count": matched,
            "partner_matched_ratio": matched_ratio,
            "partner_fight_count": partner_fight,
            "partner_fight_ratio": fight_ratio,
            "partner_social_nose_count": partner_social,
            "partner_social_nose_ratio": social_ratio,
            "related_temporal_unit_keys": related,
            "partner_pig_ids": partner_ids,
        }
        stats_rows.append(stats)

        reasons: list[tuple[str, str]] = []
        if actor_behavior == "fight":
            if matched_ratio < 0.5:
                reasons.append(("FIGHT_PARTNER_CONTEXT_INCOMPLETE", "MEDIUM"))
            elif partner_social > 0:
                reasons.append(("FIGHT_SOCIAL_PAIR_CONFLICT", "HIGH"))
            elif partner_fight == 0:
                reasons.append(("FIGHT_GROUP_PARTNER_NOT_FIGHT", "HIGH"))
            elif fight_ratio < 0.5:
                reasons.append(("FIGHT_GROUP_PARTNER_PARTIAL", "MEDIUM"))
        elif actor_behavior == "social-nose" and fight_ratio >= 0.5:
            reasons.append(("SOCIAL_ACTOR_WITH_FIGHT_PARTNER", "HIGH"))

        for reason, severity in reasons:
            finding_rows.append(
                {
                    **stats,
                    "finding_family": "INTERACTION_PAIR",
                    "finding_reason": reason,
                    "severity": severity,
                }
            )

    return pd.DataFrame(finding_rows), pd.DataFrame(stats_rows)


def add_temporal_encounter_partner_context(
    selected_findings: pd.DataFrame,
    effective_frames: pd.DataFrame,
    units: pd.DataFrame,
    *,
    context_radius_frames: int = 90,
    min_partner_support_frames: int = 2,
    max_temporal_partners: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prioritize persistent bidirectional partners around each finding."""
    _require_columns(
        selected_findings,
        ("temporal_unit_key", "finding_reason", "related_temporal_unit_keys"),
        table="selected_findings",
    )
    _require_columns(
        effective_frames,
        (
            "source_type",
            "dataset_id",
            "video_key",
            "object_track_key",
            "frame_index",
            "nearest_partner_key",
        ),
        table="effective_frames",
    )
    _require_columns(
        units,
        (
            "temporal_unit_key",
            "source_type",
            "dataset_id",
            "video_key",
            "object_track_key",
            "pig_id",
            "unit_start_frame",
            "unit_end_frame",
        ),
        table="units",
    )
    if context_radius_frames < 0:
        raise ValueError("context_radius_frames must be non-negative")
    if min_partner_support_frames < 1:
        raise ValueError("min_partner_support_frames must be positive")
    if max_temporal_partners < 1:
        raise ValueError("max_temporal_partners must be positive")

    expanded = selected_findings.copy()
    expanded["temporal_episode_partner_unit_keys"] = "[]"
    expanded["temporal_episode_partner_pig_ids"] = "[]"
    expanded["temporal_partner_selection_contract"] = (
        "classification_v2.bidirectional_temporal_encounter.v1"
    )
    if expanded.empty:
        return expanded, pd.DataFrame()

    frame_rows = effective_frames.copy()
    frame_rows["frame_index"] = pd.to_numeric(
        frame_rows["frame_index"], errors="raise"
    ).astype(int)
    unit_rows = units.copy()
    for column in ("unit_start_frame", "unit_end_frame"):
        unit_rows[column] = pd.to_numeric(
            unit_rows[column], errors="raise"
        ).astype(int)
    unit_index = unit_rows.set_index("temporal_unit_key", drop=False)
    if unit_index.index.duplicated().any():
        raise ValueError("units contains duplicate temporal_unit_key")

    video_columns = ["source_type", "dataset_id", "video_key"]
    frame_groups = {
        key: rows.copy()
        for key, rows in frame_rows.groupby(video_columns, sort=False)
    }
    unit_groups = {
        key: rows.copy()
        for key, rows in unit_rows.groupby(video_columns, sort=False)
    }
    trace_rows: list[dict[str, Any]] = []

    for finding_index, finding in expanded.iterrows():
        actor_key = _text(finding["temporal_unit_key"])
        if actor_key not in unit_index.index:
            raise ValueError(f"finding references unknown temporal unit={actor_key}")
        actor = unit_index.loc[actor_key]
        video_key = tuple(_text(actor[column]) for column in video_columns)
        video_frames = frame_groups[video_key]
        video_units = unit_groups[video_key]
        actor_track = _text(actor["object_track_key"])
        actor_start = int(actor["unit_start_frame"])
        actor_end = int(actor["unit_end_frame"])
        window_start = actor_start - context_radius_frames
        window_end = actor_end + context_radius_frames
        window = video_frames.loc[
            video_frames["frame_index"].between(window_start, window_end)
        ]

        support_frames: dict[str, set[int]] = {}
        support_directions: dict[str, set[str]] = {}
        actor_frames = window.loc[window["object_track_key"].eq(actor_track)]
        for row in actor_frames.itertuples(index=False):
            partner = _text(row.nearest_partner_key)
            if not partner or partner == actor_track:
                continue
            support_frames.setdefault(partner, set()).add(int(row.frame_index))
            support_directions.setdefault(partner, set()).add("ACTOR_TO_PARTNER")

        reverse = window.loc[
            window["nearest_partner_key"].eq(actor_track)
            & window["object_track_key"].ne(actor_track)
        ]
        for row in reverse.itertuples(index=False):
            partner = _text(row.object_track_key)
            support_frames.setdefault(partner, set()).add(int(row.frame_index))
            support_directions.setdefault(partner, set()).add("PARTNER_TO_ACTOR")

        actor_center = (actor_start + actor_end) / 2.0
        ranked_partners = sorted(
            support_frames,
            key=lambda partner: (
                -len(support_frames[partner]),
                min(abs(frame - actor_center) for frame in support_frames[partner]),
                partner,
            ),
        )
        ranked_partners = [
            partner
            for partner in ranked_partners
            if len(support_frames[partner]) >= min_partner_support_frames
        ][:max_temporal_partners]

        temporal_unit_keys: list[str] = []
        temporal_pig_ids: list[str] = []
        for rank, partner in enumerate(ranked_partners, start=1):
            candidate_units = video_units.loc[
                video_units["object_track_key"].eq(partner)
            ]
            synchronized = candidate_units.loc[
                candidate_units["unit_start_frame"].le(actor_end)
                & candidate_units["unit_end_frame"].ge(actor_start)
            ]
            selection_status = "SYNCHRONIZED_TARGET_INTERVAL"
            if synchronized.empty:
                closest_frame = min(
                    support_frames[partner],
                    key=lambda frame: (abs(frame - actor_center), -frame),
                )
                synchronized = candidate_units.loc[
                    candidate_units["unit_start_frame"].le(closest_frame)
                    & candidate_units["unit_end_frame"].ge(closest_frame)
                ]
                selection_status = "CLOSEST_SUPPORTED_INTERVAL"

            selected_keys = synchronized.sort_values(
                ["unit_start_frame", "temporal_unit_key"], kind="mergesort"
            )["temporal_unit_key"].map(_text).tolist()
            selected_pig_ids = synchronized["pig_id"].map(_text).tolist()
            temporal_unit_keys.extend(selected_keys)
            temporal_pig_ids.extend(selected_pig_ids)
            trace_rows.append(
                {
                    "temporal_unit_key": actor_key,
                    "finding_reason": _text(finding["finding_reason"]),
                    "actor_object_track_key": actor_track,
                    "partner_object_track_key": partner,
                    "partner_rank": rank,
                    "support_frame_count": len(support_frames[partner]),
                    "support_first_frame": min(support_frames[partner]),
                    "support_last_frame": max(support_frames[partner]),
                    "support_directions": ";".join(
                        sorted(support_directions[partner])
                    ),
                    "selected_partner_unit_keys": _encode_ordered_key_list(
                        selected_keys
                    ),
                    "selection_status": selection_status,
                }
            )

        temporal_unit_keys = list(dict.fromkeys(temporal_unit_keys))
        temporal_pig_ids = list(dict.fromkeys(temporal_pig_ids))
        existing_keys = decode_related_temporal_unit_keys(
            finding["related_temporal_unit_keys"]
        )
        all_related = list(dict.fromkeys([*temporal_unit_keys, *existing_keys]))
        expanded.at[finding_index, "related_temporal_unit_keys"] = (
            _encode_ordered_key_list(all_related)
        )
        expanded.at[finding_index, "temporal_episode_partner_unit_keys"] = (
            _encode_ordered_key_list(temporal_unit_keys)
        )
        expanded.at[finding_index, "temporal_episode_partner_pig_ids"] = (
            _encode_ordered_key_list(temporal_pig_ids)
        )

    return expanded, pd.DataFrame(trace_rows)


def build_temporal_continuity_findings(units: pd.DataFrame) -> pd.DataFrame:
    """Find one-burst label islands with special handling for interaction."""
    required = (
        "temporal_unit_key",
        "source_type",
        "dataset_id",
        "video_key",
        "object_track_key",
        "pig_id",
        "unit_start_frame",
        "unit_end_frame",
        "effective_behavior",
    )
    _require_columns(units, required, table="units")
    findings: list[dict[str, Any]] = []
    group_columns = [
        "source_type",
        "dataset_id",
        "video_key",
        "object_track_key",
    ]
    for _, rows in units.groupby(group_columns, sort=True, dropna=False):
        ordered = rows.sort_values(
            ["unit_start_frame", "unit_end_frame", "temporal_unit_key"],
            kind="mergesort",
        ).reset_index(drop=True)
        for index in range(1, len(ordered) - 1):
            previous = ordered.iloc[index - 1]
            current = ordered.iloc[index]
            following = ordered.iloc[index + 1]
            if int(current["unit_start_frame"]) > int(previous["unit_end_frame"]) + 1:
                continue
            if int(following["unit_start_frame"]) > int(current["unit_end_frame"]) + 1:
                continue
            previous_label = _text(previous["effective_behavior"])
            current_label = _text(current["effective_behavior"])
            following_label = _text(following["effective_behavior"])
            if previous_label != following_label or current_label == previous_label:
                continue

            if previous_label == "fight":
                reason = "NON_FIGHT_BURST_BETWEEN_FIGHT"
                severity = "HIGH"
            elif current_label == "fight" and previous_label == "social-nose":
                reason = "FIGHT_BURST_BETWEEN_SOCIAL_NOSE"
                severity = "MEDIUM"
            elif INTERACTION_BEHAVIORS.intersection(
                {previous_label, current_label}
            ):
                reason = "INTERACTION_SINGLE_BURST_ISLAND"
                severity = "MEDIUM"
            else:
                reason = "GENERAL_SINGLE_BURST_LABEL_ISLAND"
                severity = "LOW"

            findings.append(
                {
                    "temporal_unit_key": current["temporal_unit_key"],
                    "effective_behavior": current_label,
                    "finding_family": "TEMPORAL_CONTINUITY",
                    "finding_reason": reason,
                    "severity": severity,
                    "previous_temporal_unit_key": previous["temporal_unit_key"],
                    "previous_behavior": previous_label,
                    "next_temporal_unit_key": following["temporal_unit_key"],
                    "next_behavior": following_label,
                    "related_temporal_unit_keys": _encode_key_list(
                        [
                            previous["temporal_unit_key"],
                            following["temporal_unit_key"],
                        ]
                    ),
                    "source_type": current["source_type"],
                    "dataset_id": current["dataset_id"],
                    "video_key": current["video_key"],
                    "object_track_key": current["object_track_key"],
                    "pig_id": current["pig_id"],
                    "unit_start_frame": int(current["unit_start_frame"]),
                    "unit_end_frame": int(current["unit_end_frame"]),
                }
            )
    return pd.DataFrame(findings)


def build_note_findings(decisions: pd.DataFrame) -> pd.DataFrame:
    """Surface reviewer-authored interaction and temporal-boundary notes."""
    _require_columns(decisions, DECISION_REQUIRED_COLUMNS, table="decisions")
    notes = decisions["manual_note"].fillna("").astype(str).str.strip()
    flagged = notes.str.contains(NOTE_RISK_PATTERN, na=False)
    rows: list[dict[str, Any]] = []
    for index in decisions.index[flagged]:
        row = decisions.loc[index]
        rows.append(
            {
                "temporal_unit_key": _text(row["temporal_unit_key"]),
                "effective_behavior": (
                    _text(row["manual_corrected_behavior"])
                    if _text(row["manual_review_decision"]) == "corrected"
                    else _text(row["behavior_label"])
                ),
                "finding_family": "REVIEWER_NOTE",
                "finding_reason": "INTERACTION_OR_BOUNDARY_NOTE",
                "severity": "MEDIUM",
                "related_temporal_unit_keys": "",
                "manual_note": _text(row["manual_note"]),
                "review_item_id": _text(row["review_item_id"]),
                "review_unit_id": _text(row["review_unit_id"]),
            }
        )
    return pd.DataFrame(rows)


def build_interaction_correction_findings(
    decisions: pd.DataFrame,
    pair_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Require a paired recheck for every correction involving interaction."""
    _require_columns(decisions, DECISION_REQUIRED_COLUMNS, table="decisions")
    required_stats = ("temporal_unit_key", "related_temporal_unit_keys")
    _require_columns(pair_stats, required_stats, table="pair_stats")
    stats_by_key = pair_stats.set_index("temporal_unit_key")
    corrected = decisions.loc[
        decisions["manual_review_decision"].fillna("").astype(str).eq("corrected")
    ].copy()
    interaction = corrected["behavior_label"].isin(INTERACTION_BEHAVIORS) | corrected[
        "manual_corrected_behavior"
    ].isin(INTERACTION_BEHAVIORS)
    rows: list[dict[str, Any]] = []
    for _, decision in corrected.loc[interaction].iterrows():
        key = _text(decision["temporal_unit_key"])
        related = "[]"
        if key in stats_by_key.index:
            related = _text(stats_by_key.at[key, "related_temporal_unit_keys"])
        corrected_behavior = _text(decision["manual_corrected_behavior"])
        if corrected_behavior == "fight":
            reason = "RECHECK_CORRECTED_TO_FIGHT_WITH_PARTNER"
            severity = "HIGH"
        elif corrected_behavior == "social-nose":
            reason = "RECHECK_CORRECTED_TO_SOCIAL_NOSE_WITH_PARTNER"
            severity = "HIGH"
        else:
            reason = "RECHECK_CORRECTED_FROM_INTERACTION"
            severity = "MEDIUM"
        rows.append(
            {
                "temporal_unit_key": key,
                "effective_behavior": corrected_behavior,
                "finding_family": "INTERACTION_CORRECTION",
                "finding_reason": reason,
                "severity": severity,
                "related_temporal_unit_keys": related,
                "review_item_id": _text(decision["review_item_id"]),
                "review_unit_id": _text(decision["review_unit_id"]),
                "manual_note": _text(decision["manual_note"]),
            }
        )
    return pd.DataFrame(rows)


def build_consistency_review_scope(
    units: pd.DataFrame,
    reference_review_view: pd.DataFrame,
    selected_findings: pd.DataFrame,
    *,
    selection_config_hash: str,
    context_radius_frames: int = 90,
) -> pd.DataFrame:
    """Build a GUI-compatible scope with paired units kept adjacent."""
    required_units = (
        "temporal_unit_key",
        "source_type",
        "dataset_id",
        "video_key",
        "object_track_key",
        "pig_id",
        "track_id",
        "unit_start_frame",
        "unit_end_frame",
        "effective_behavior",
        "review_item_id",
        "manual_review_decision",
        "manual_corrected_behavior",
        "manual_note",
    )
    _require_columns(units, required_units, table="units")
    _require_columns(
        selected_findings,
        (
            "temporal_unit_key",
            "finding_reason",
            "severity",
            "related_temporal_unit_keys",
        ),
        table="selected_findings",
    )
    if context_radius_frames < 0:
        raise ValueError("context_radius_frames must be non-negative")
    if not selection_config_hash:
        raise ValueError("selection_config_hash is required")

    unit_index = units.set_index("temporal_unit_key", drop=False)
    if unit_index.index.duplicated().any():
        raise ValueError("units contains duplicate temporal_unit_key")
    reference = reference_review_view.copy()
    if not reference.empty:
        _require_columns(reference, ("temporal_unit_key",), table="reference_view")
        reference["temporal_unit_key"] = reference["temporal_unit_key"].map(_text)
        reference = reference.drop_duplicates("temporal_unit_key", keep="first")
        reference_index = reference.set_index("temporal_unit_key", drop=False)
    else:
        reference_index = pd.DataFrame().set_index(pd.Index([], dtype=object))

    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    findings = selected_findings.copy()
    findings["_severity_order"] = findings["severity"].map(severity_order).fillna(9)
    findings = findings.sort_values(
        ["_severity_order", "finding_reason", "temporal_unit_key"],
        kind="mergesort",
    ).reset_index(drop=True)

    ordered_keys: list[str] = []
    reasons_by_key: dict[str, set[str]] = {}
    groups_by_key: dict[str, list[str]] = {}
    severity_by_key: dict[str, str] = {}
    role_by_key: dict[str, set[str]] = {}
    seen: set[str] = set()
    for finding_index, finding in findings.iterrows():
        group_id = f"consistency_group_{finding_index + 1:06d}"
        actor_key = _text(finding["temporal_unit_key"])
        related_keys = decode_related_temporal_unit_keys(
            finding["related_temporal_unit_keys"]
        )
        temporal_partner_keys: set[str] = set()
        if "temporal_episode_partner_unit_keys" in findings.columns:
            temporal_partner_keys.update(
                decode_related_temporal_unit_keys(
                    finding["temporal_episode_partner_unit_keys"]
                )
            )
        related_members = [
            (
                key,
                "EPISODE_PARTNER_CANDIDATE"
                if key in temporal_partner_keys
                else "CURRENT_NEAREST_OR_BOUNDARY",
            )
            for key in related_keys
        ]
        for key, role in [(actor_key, "ACTOR"), *related_members]:
            if key not in unit_index.index:
                raise ValueError(f"finding references unknown temporal unit={key}")
            reasons_by_key.setdefault(key, set()).add(_text(finding["finding_reason"]))
            groups_by_key.setdefault(key, []).append(group_id)
            role_by_key.setdefault(key, set()).add(role)
            current_severity = severity_by_key.get(key, "LOW")
            proposed_severity = _text(finding["severity"])
            if severity_order.get(proposed_severity, 9) < severity_order.get(
                current_severity, 9
            ):
                severity_by_key[key] = proposed_severity
            else:
                severity_by_key.setdefault(key, current_severity)
            if key not in seen:
                seen.add(key)
                ordered_keys.append(key)

    video_max_frame = units.groupby(
        ["source_type", "dataset_id", "video_key"],
        dropna=False,
    )["unit_end_frame"].max()
    rows: list[dict[str, Any]] = []
    for order, key in enumerate(ordered_keys, start=1):
        unit = unit_index.loc[key]
        if key in reference_index.index:
            record = reference_index.loc[key].to_dict()
        else:
            record = {}
        source_type = _text(unit["source_type"])
        contract = SOURCE_UNIT_CONTRACTS.get(source_type)
        if contract is None:
            raise ValueError(f"unsupported source_type={source_type}")
        start = int(unit["unit_start_frame"])
        end = int(unit["unit_end_frame"])
        target_frames = list(range(start, end + 1))
        video_key = (
            source_type,
            _text(unit["dataset_id"]),
            _text(unit["video_key"]),
        )
        max_frame = int(video_max_frame.loc[video_key])
        if source_type == "cvat_tracking_xml":
            playback_start = max(0, start - context_radius_frames)
            playback_end = min(max_frame, end + context_radius_frames)
            playback_frames = list(range(playback_start, playback_end + 1))
            context_frames = _evenly_spaced_frames(
                playback_start,
                playback_end,
                count=12,
            )
        else:
            playback_frames = target_frames
            context_frames = target_frames
        behavior = _text(unit["effective_behavior"])
        if behavior not in BEHAVIOR_REVIEW_TEMPLATE:
            raise ValueError(f"invalid effective behavior for scope={behavior}")

        record.update(
            {
                "review_item_id": f"consistency_review_{order:07d}",
                "review_unit_id": key,
                "temporal_unit_key": key,
                "review_unit_type": contract["review_unit_type"],
                "source_type": source_type,
                "dataset_id": _text(unit["dataset_id"]),
                "video_key": _text(unit["video_key"]),
                "object_track_key": _text(unit["object_track_key"]),
                "pig_id": _text(unit["pig_id"]),
                "track_id": _text(unit["track_id"]),
                "unit_start_frame": start,
                "unit_end_frame": end,
                "unit_frame_count": end - start + 1,
                "display_frame_indices": _join_frames(target_frames),
                "display_frame_count": len(target_frames),
                "behavior_label": behavior,
                "original_behavior": behavior,
                "review_template": BEHAVIOR_REVIEW_TEMPLATE[behavior],
                "apply_scope": contract["apply_scope"],
                "review_reason": "POST_REVIEW_BEHAVIOR_CONSISTENCY",
                "review_reason_codes": ";".join(sorted(reasons_by_key[key])),
                "review_priority": order,
                "recommended_gui": "review_final_behavior_gui_v1",
                "candidate_tier": "POST_REVIEW_CONSISTENCY_AUDIT",
                "include_in_review": True,
                "selection_predicate_version": (
                    "classification_v2.post_review_behavior_consistency.v1"
                ),
                "selection_config_hash": selection_config_hash,
                "global_mandatory_review": False,
                "final_scope_component": "POST_REVIEW_CONSISTENCY_REREVIEW",
                "final_context_frame_indices": _join_frames(context_frames),
                "final_context_frame_count": len(context_frames),
                "final_playback_frame_indices": _join_frames(playback_frames),
                "final_playback_frame_count": len(playback_frames),
                "final_review_context_contract": (
                    "classification_v2.consistency_extended_context.v1"
                ),
                "consistency_review_order": order,
                "consistency_group_ids": _encode_key_list(groups_by_key[key]),
                "consistency_roles": ";".join(sorted(role_by_key[key])),
                "consistency_severity": severity_by_key[key],
                "consistency_previous_review_item_id": _text(
                    unit["review_item_id"]
                ),
                "consistency_previous_decision": _text(
                    unit["manual_review_decision"]
                ),
                "consistency_previous_correction": _text(
                    unit["manual_corrected_behavior"]
                ),
                "consistency_previous_note": _text(unit["manual_note"]),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def combine_findings(*tables: pd.DataFrame) -> pd.DataFrame:
    """Create one deterministic, de-duplicated consistency finding ledger."""
    nonempty = [table.copy() for table in tables if not table.empty]
    if not nonempty:
        return pd.DataFrame(
            columns=[
                "temporal_unit_key",
                "finding_family",
                "finding_reason",
                "severity",
                "related_temporal_unit_keys",
            ]
        )
    combined = pd.concat(nonempty, ignore_index=True, sort=False)
    for column in combined.columns:
        if combined[column].dtype == object:
            combined[column] = combined[column].fillna("")
    combined = combined.drop_duplicates(
        ["temporal_unit_key", "finding_reason", "related_temporal_unit_keys"]
    )
    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    combined["_severity_order"] = combined["severity"].map(severity_order).fillna(9)
    combined = combined.sort_values(
        ["_severity_order", "finding_family", "finding_reason", "temporal_unit_key"],
        kind="mergesort",
    ).drop(columns="_severity_order")
    return combined.reset_index(drop=True)


def review_scope_keys(findings: pd.DataFrame, *, include_low: bool) -> list[str]:
    """Return actor and related unit keys for bounded re-review."""
    if findings.empty:
        return []
    selected = findings.copy()
    if not include_low:
        selected = selected.loc[~selected["severity"].eq("LOW")]
    keys: set[str] = set(selected["temporal_unit_key"].map(_text))
    for value in selected["related_temporal_unit_keys"].map(_text):
        keys.update(decode_related_temporal_unit_keys(value))
    return sorted(key for key in keys if key)


def _joined_unique(series: pd.Series) -> str:
    return _encode_key_list(series)


def decode_related_temporal_unit_keys(value: Any) -> list[str]:
    """Decode a JSON key array without treating key punctuation as delimiters."""
    text = _text(value)
    if not text:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("related_temporal_unit_keys must be a JSON list")
    keys = [_text(item) for item in parsed]
    if any(not key for key in keys):
        raise ValueError("related_temporal_unit_keys contains blank keys")
    return keys


def _encode_key_list(values: Any) -> str:
    keys = sorted({_text(value) for value in values if _text(value)})
    return json.dumps(keys, ensure_ascii=False, separators=(",", ":"))


def _encode_ordered_key_list(values: Any) -> str:
    keys = list(dict.fromkeys(_text(value) for value in values if _text(value)))
    return json.dumps(keys, ensure_ascii=False, separators=(",", ":"))


def _join_frames(frames: list[int]) -> str:
    return ",".join(str(frame) for frame in frames)


def select_targeted_consistency_scope_v3(
    scope: pd.DataFrame,
    completed_decisions: pd.DataFrame,
    temporal_partner_trace: pd.DataFrame,
    *,
    fight_neighbor_radius_frames: int = 24,
    min_overlapping_partner_support_frames: int = 48,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep actors and evidence-backed non-fight partners for v3 review."""
    if fight_neighbor_radius_frames < 0:
        raise ValueError("fight_neighbor_radius_frames must be non-negative")
    if min_overlapping_partner_support_frames < 1:
        raise ValueError(
            "min_overlapping_partner_support_frames must be positive"
        )
    _require_columns(
        scope,
        (
            "temporal_unit_key",
            "object_track_key",
            "unit_start_frame",
            "unit_end_frame",
            "behavior_label",
            "consistency_roles",
            "consistency_review_order",
        ),
        table="scope",
    )
    _require_columns(
        completed_decisions,
        (
            "temporal_unit_key",
            "object_track_key",
            "unit_start_frame",
            "unit_end_frame",
            "original_behavior",
            "manual_review_decision",
            "manual_corrected_behavior",
        ),
        table="completed_decisions",
    )
    trace_columns = (
        "support_frame_count",
        "support_first_frame",
        "support_last_frame",
        "support_directions",
        "selected_partner_unit_keys",
    )
    if not temporal_partner_trace.empty:
        _require_columns(
            temporal_partner_trace,
            trace_columns,
            table="temporal_partner_trace",
        )

    decisions = completed_decisions.copy()
    decisions["_effective_behavior"] = decisions["original_behavior"].map(_text)
    corrected = decisions["manual_review_decision"].map(_text).eq("corrected")
    decisions.loc[corrected, "_effective_behavior"] = decisions.loc[
        corrected, "manual_corrected_behavior"
    ].map(_text)
    decisions = decisions.loc[
        decisions["manual_review_decision"].map(_text).isin({"accept", "corrected"})
    ].copy()
    decisions["unit_start_frame"] = pd.to_numeric(
        decisions["unit_start_frame"], errors="raise"
    ).astype(int)
    decisions["unit_end_frame"] = pd.to_numeric(
        decisions["unit_end_frame"], errors="raise"
    ).astype(int)

    fight_intervals_by_track: dict[str, list[tuple[str, int, int]]] = {}
    fight_rows = decisions.loc[decisions["_effective_behavior"].eq("fight")]
    for row in fight_rows.itertuples(index=False):
        track_key = _text(row.object_track_key)
        fight_intervals_by_track.setdefault(track_key, []).append(
            (
                _text(row.temporal_unit_key),
                int(row.unit_start_frame),
                int(row.unit_end_frame),
            )
        )

    traces_by_partner: dict[str, list[dict[str, object]]] = {}
    for row in temporal_partner_trace.itertuples(index=False):
        directions = set(
            filter(None, _text(row.support_directions).split(";"))
        )
        trace_record = {
            "bidirectional": {
                "ACTOR_TO_PARTNER",
                "PARTNER_TO_ACTOR",
            }.issubset(directions),
            "support_frame_count": int(row.support_frame_count),
            "support_first_frame": int(row.support_first_frame),
            "support_last_frame": int(row.support_last_frame),
        }
        for partner_key in decode_related_temporal_unit_keys(
            row.selected_partner_unit_keys
        ):
            traces_by_partner.setdefault(partner_key, []).append(trace_record)

    audit_rows: list[dict[str, object]] = []
    keep_keys: list[str] = []
    for row in scope.itertuples(index=False):
        unit_key = _text(row.temporal_unit_key)
        track_key = _text(row.object_track_key)
        start_frame = int(row.unit_start_frame)
        end_frame = int(row.unit_end_frame)
        roles = set(filter(None, _text(row.consistency_roles).split(";")))
        is_actor = "ACTOR" in roles
        is_candidate = "EPISODE_PARTNER_CANDIDATE" in roles
        behavior = _text(row.behavior_label)

        fight_gaps = []
        for fight_key, fight_start, fight_end in fight_intervals_by_track.get(
            track_key, []
        ):
            if fight_key == unit_key:
                continue
            fight_gaps.append(
                max(start_frame - fight_end, fight_start - end_frame, 0)
            )
        nearest_fight_gap = min(fight_gaps) if fight_gaps else None
        has_nearby_fight = (
            nearest_fight_gap is not None
            and nearest_fight_gap <= fight_neighbor_radius_frames
        )

        partner_traces = traces_by_partner.get(unit_key, [])
        bidirectional = any(
            bool(trace["bidirectional"]) for trace in partner_traces
        )
        max_support = max(
            (
                int(trace["support_frame_count"])
                for trace in partner_traces
                if bool(trace["bidirectional"])
            ),
            default=0,
        )
        overlapping_strong_support = any(
            bool(trace["bidirectional"])
            and int(trace["support_frame_count"])
            >= min_overlapping_partner_support_frames
            and int(trace["support_first_frame"]) <= end_frame
            and int(trace["support_last_frame"]) >= start_frame
            for trace in partner_traces
        )

        if is_actor:
            keep = True
            reason = "ACTOR_FINDING"
        elif not is_candidate:
            keep = False
            reason = "DROP_PROXIMITY_ONLY_CONTEXT"
        elif behavior == "fight":
            keep = False
            reason = "DROP_ALREADY_FIGHT_PARTNER"
        elif not bidirectional:
            keep = False
            reason = "DROP_UNIDIRECTIONAL_PARTNER"
        elif has_nearby_fight:
            keep = True
            reason = "TARGETED_PARTNER_NEAR_SAME_TRACK_FIGHT"
        elif overlapping_strong_support:
            keep = True
            reason = "TARGETED_PARTNER_STRONG_TARGET_OVERLAP"
        else:
            keep = False
            reason = "DROP_WEAK_OR_DISTANT_PARTNER"

        if keep:
            keep_keys.append(unit_key)
        audit_rows.append(
            {
                "temporal_unit_key": unit_key,
                "prior_consistency_review_order": int(
                    row.consistency_review_order
                ),
                "consistency_roles": _text(row.consistency_roles),
                "effective_behavior": behavior,
                "bidirectional_partner_support": bidirectional,
                "max_bidirectional_partner_support_frames": max_support,
                "partner_support_overlaps_target": overlapping_strong_support,
                "nearest_same_track_fight_gap_frames": nearest_fight_gap,
                "keep_in_v3": keep,
                "v3_selection_reason": reason,
            }
        )

    keep_set = set(keep_keys)
    filtered = scope.loc[
        scope["temporal_unit_key"].map(_text).isin(keep_set)
    ].copy()
    filtered = filtered.sort_values(
        "consistency_review_order", kind="mergesort"
    ).reset_index(drop=True)
    filtered["consistency_review_order"] = range(1, len(filtered) + 1)
    order_by_key = dict(
        zip(
            filtered["temporal_unit_key"].map(_text),
            filtered["consistency_review_order"],
            strict=True,
        )
    )
    audit = pd.DataFrame(audit_rows)
    audit["v3_consistency_review_order"] = audit["temporal_unit_key"].map(
        order_by_key
    )
    return filtered, audit


def _evenly_spaced_frames(start: int, end: int, *, count: int) -> list[int]:
    if end < start:
        raise ValueError("invalid frame range")
    if count <= 1 or start == end:
        return [start]
    span = end - start
    return sorted(
        {
            int(round(start + index * span / (count - 1)))
            for index in range(count)
        }
    )


def _require_columns(
    frame: pd.DataFrame,
    required: tuple[str, ...] | list[str],
    *,
    table: str,
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{table} missing columns={missing}")


def _text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.casefold() == "nan" else text


__all__ = [
    "add_temporal_encounter_partner_context",
    "build_effective_behavior_tables",
    "build_consistency_review_scope",
    "build_interaction_correction_findings",
    "build_interaction_pair_findings",
    "build_note_findings",
    "build_temporal_continuity_findings",
    "combine_findings",
    "decode_related_temporal_unit_keys",
    "review_scope_keys",
    "select_targeted_consistency_scope_v3",
]
