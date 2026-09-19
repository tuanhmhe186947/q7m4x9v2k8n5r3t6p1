from __future__ import annotations

import pandas as pd

from pig_behavior.classification_v2.review.behavior_consistency_audit import (
    add_temporal_encounter_partner_context,
    build_consistency_review_scope,
    build_effective_behavior_tables,
    build_interaction_correction_findings,
    build_interaction_pair_findings,
    build_note_findings,
    build_temporal_continuity_findings,
    combine_findings,
    decode_related_temporal_unit_keys,
    review_scope_keys,
    select_targeted_consistency_scope_v3,
)
from pig_behavior.classification_v2.review.behavior_review_contract import (
    audit_review_unit_contract,
)


def test_reviewed_labels_drive_pair_and_temporal_consistency_findings() -> None:
    frames = _frames()
    decisions = _decisions()

    units, effective_frames, audit = build_effective_behavior_tables(
        frames,
        decisions,
    )
    pair_findings, _ = build_interaction_pair_findings(effective_frames)
    temporal_findings = build_temporal_continuity_findings(units)

    assert audit["matched_decision_units"] == 3
    labels = units.set_index("temporal_unit_key")["effective_behavior"]
    assert labels["actor-a-0"] == "fight"
    assert "FIGHT_GROUP_PARTNER_NOT_FIGHT" in set(
        pair_findings["finding_reason"]
    )
    island = temporal_findings.loc[
        temporal_findings["finding_reason"].eq(
            "NON_FIGHT_BURST_BETWEEN_FIGHT"
        )
    ]
    assert island["temporal_unit_key"].tolist() == ["actor-a-1"]


def test_actor_only_social_nose_does_not_require_partner_social_label() -> None:
    _, effective_frames, _ = build_effective_behavior_tables(
        _frames(),
        _decisions(),
    )

    findings, stats = build_interaction_pair_findings(effective_frames)

    social_stats = stats.loc[stats["temporal_unit_key"].eq("actor-c-social")]
    assert social_stats["partner_fight_count"].iloc[0] == 0
    assert not findings["temporal_unit_key"].eq("actor-c-social").any()


def test_note_findings_and_related_units_enter_bounded_scope() -> None:
    units, effective_frames, _ = build_effective_behavior_tables(
        _frames(),
        _decisions(),
    )
    pair_findings, pair_stats = build_interaction_pair_findings(effective_frames)
    temporal_findings = build_temporal_continuity_findings(units)
    note_findings = build_note_findings(_decisions())
    correction_findings = build_interaction_correction_findings(
        _decisions(),
        pair_stats,
    )
    findings = combine_findings(
        correction_findings,
        pair_findings,
        temporal_findings,
        note_findings,
    )

    assert "INTERACTION_OR_BOUNDARY_NOTE" in set(findings["finding_reason"])
    assert "RECHECK_CORRECTED_TO_FIGHT_WITH_PARTNER" in set(
        findings["finding_reason"]
    )
    keys = review_scope_keys(findings, include_low=False)
    assert "actor-a-0" in keys
    assert "actor-a-1" in keys
    assert "partner-b" in keys


def test_completed_review_audit_rejects_pending_decisions() -> None:
    decisions = _decisions()
    decisions.loc[0, "manual_review_decision"] = "pending"

    try:
        build_effective_behavior_tables(_frames(), decisions)
    except ValueError as exc:
        assert "non-terminal" in str(exc)
    else:
        raise AssertionError("pending decision was accepted")


def test_related_keys_preserve_pipe_characters_inside_authoritative_keys() -> None:
    key = "source=cvat|video=one|track=2|anchor=6"
    findings = pd.DataFrame(
        [
            {
                "temporal_unit_key": "source=cvat|video=one|track=1|anchor=6",
                "finding_family": "INTERACTION_PAIR",
                "finding_reason": "FIGHT_GROUP_PARTNER_NOT_FIGHT",
                "severity": "HIGH",
                "related_temporal_unit_keys": f'["{key}"]',
            }
        ]
    )

    assert decode_related_temporal_unit_keys(
        findings.loc[0, "related_temporal_unit_keys"]
    ) == [key]
    assert key in review_scope_keys(findings, include_low=False)


def test_consistency_scope_builds_missing_reference_rows_in_pair_order() -> None:
    actor_key = "source=cvat|video=one|track=1|anchor=0"
    partner_key = "source=cvat|video=one|track=2|anchor=0"
    units = pd.DataFrame(
        [
            _scope_unit(actor_key, "actor-1", "ID_1", "fight"),
            _scope_unit(partner_key, "actor-2", "ID_2", "stand"),
        ]
    )
    findings = pd.DataFrame(
        [
            {
                "temporal_unit_key": actor_key,
                "finding_family": "INTERACTION_PAIR",
                "finding_reason": "FIGHT_GROUP_PARTNER_NOT_FIGHT",
                "severity": "HIGH",
                "related_temporal_unit_keys": f'["{partner_key}"]',
            }
        ]
    )

    scope = build_consistency_review_scope(
        units,
        pd.DataFrame(),
        findings,
        selection_config_hash="a" * 64,
    )

    assert scope["temporal_unit_key"].tolist() == [actor_key, partner_key]
    assert scope["behavior_label"].tolist() == ["fight", "stand"]
    assert scope["consistency_review_order"].tolist() == [1, 2]
    assert audit_review_unit_contract(scope)["errors"] == []


def test_temporal_partner_context_survives_partner_separation() -> None:
    actor_key = "source=cvat|video=one|track=1|anchor=10"
    persistent_key = "source=cvat|video=one|track=2|anchor=10"
    current_key = "source=cvat|video=one|track=3|anchor=10"
    units = pd.DataFrame(
        [
            _scope_unit(actor_key, "actor", "ID_1", "fight", start=10),
            _scope_unit(
                persistent_key,
                "persistent-partner",
                "ID_2",
                "explore",
                start=10,
            ),
            _scope_unit(
                current_key,
                "current-nearest",
                "ID_3",
                "sitting",
                start=10,
            ),
        ]
    )
    frames: list[dict[str, object]] = []
    for frame_index in range(4, 10):
        frames.extend(
            [
                _frame(
                    "actor-history",
                    frame_index,
                    "actor",
                    "ID_1",
                    "fight",
                    "persistent-partner",
                ),
                _frame(
                    "persistent-history",
                    frame_index,
                    "persistent-partner",
                    "ID_2",
                    "fight",
                    "actor",
                ),
            ]
        )
    for frame_index in range(10, 12):
        frames.extend(
            [
                _frame(
                    actor_key,
                    frame_index,
                    "actor",
                    "ID_1",
                    "fight",
                    "current-nearest",
                ),
                _frame(
                    persistent_key,
                    frame_index,
                    "persistent-partner",
                    "ID_2",
                    "explore",
                    "current-nearest",
                ),
                _frame(
                    current_key,
                    frame_index,
                    "current-nearest",
                    "ID_3",
                    "sitting",
                    "actor",
                ),
            ]
        )
    findings = pd.DataFrame(
        [
            {
                "temporal_unit_key": actor_key,
                "finding_family": "INTERACTION_PAIR",
                "finding_reason": "FIGHT_GROUP_PARTNER_NOT_FIGHT",
                "severity": "HIGH",
                "related_temporal_unit_keys": f'["{current_key}"]',
            }
        ]
    )

    expanded, trace = add_temporal_encounter_partner_context(
        findings,
        pd.DataFrame(frames),
        units,
        context_radius_frames=10,
    )

    related = decode_related_temporal_unit_keys(
        expanded.loc[0, "related_temporal_unit_keys"]
    )
    assert related[:2] == [persistent_key, current_key]
    assert trace.iloc[0]["partner_object_track_key"] == "persistent-partner"
    scope = build_consistency_review_scope(
        units,
        pd.DataFrame(),
        expanded,
        selection_config_hash="b" * 64,
        context_radius_frames=10,
    )
    assert scope["temporal_unit_key"].tolist() == [
        actor_key,
        persistent_key,
        current_key,
    ]
    assert scope.loc[1, "consistency_roles"] == "EPISODE_PARTNER_CANDIDATE"


def test_v3_scope_drops_proximity_only_rows_and_keeps_targeted_partners() -> None:
    scope = pd.DataFrame(
        [
            _v3_scope_row("actor", "track-actor", "fight", "ACTOR", 1),
            _v3_scope_row(
                "strong-partner",
                "track-strong",
                "explore",
                "EPISODE_PARTNER_CANDIDATE",
                2,
            ),
            _v3_scope_row(
                "near-fight-partner",
                "track-near-fight",
                "stand",
                "EPISODE_PARTNER_CANDIDATE",
                3,
            ),
            _v3_scope_row(
                "weak-partner",
                "track-weak",
                "sitting",
                "EPISODE_PARTNER_CANDIDATE",
                4,
            ),
            _v3_scope_row(
                "already-fight",
                "track-fight",
                "fight",
                "EPISODE_PARTNER_CANDIDATE",
                5,
            ),
            _v3_scope_row(
                "proximity-only",
                "track-nearest",
                "explore",
                "CURRENT_NEAREST_OR_BOUNDARY",
                6,
            ),
        ]
    )
    decisions = pd.DataFrame(
        [
            _v3_decision("actor", "track-actor", "fight", 100),
            _v3_decision("strong-partner", "track-strong", "explore", 100),
            _v3_decision(
                "near-fight-partner",
                "track-near-fight",
                "stand",
                100,
            ),
            _v3_decision(
                "near-fight-history",
                "track-near-fight",
                "fight",
                80,
            ),
            _v3_decision("weak-partner", "track-weak", "sitting", 100),
            _v3_decision("already-fight", "track-fight", "fight", 100),
            _v3_decision(
                "proximity-only", "track-nearest", "explore", 100
            ),
        ]
    )
    traces = pd.DataFrame(
        [
            _v3_trace("strong-partner", support=50, first=90, last=105),
            _v3_trace("near-fight-partner", support=2, first=10, last=11),
            _v3_trace("weak-partner", support=10, first=100, last=105),
            _v3_trace("already-fight", support=60, first=100, last=105),
        ]
    )

    filtered, audit = select_targeted_consistency_scope_v3(
        scope,
        decisions,
        traces,
    )

    assert filtered["temporal_unit_key"].tolist() == [
        "actor",
        "strong-partner",
        "near-fight-partner",
    ]
    assert filtered["consistency_review_order"].tolist() == [1, 2, 3]
    reason_by_key = audit.set_index("temporal_unit_key")["v3_selection_reason"]
    assert reason_by_key["strong-partner"] == (
        "TARGETED_PARTNER_STRONG_TARGET_OVERLAP"
    )
    assert reason_by_key["near-fight-partner"] == (
        "TARGETED_PARTNER_NEAR_SAME_TRACK_FIGHT"
    )
    assert reason_by_key["weak-partner"] == "DROP_WEAK_OR_DISTANT_PARTNER"
    assert reason_by_key["already-fight"] == "DROP_ALREADY_FIGHT_PARTNER"
    assert reason_by_key["proximity-only"] == "DROP_PROXIMITY_ONLY_CONTEXT"


def _v3_scope_row(
    unit_key: str,
    track_key: str,
    behavior: str,
    roles: str,
    order: int,
) -> dict[str, object]:
    return {
        "temporal_unit_key": unit_key,
        "object_track_key": track_key,
        "unit_start_frame": 100,
        "unit_end_frame": 105,
        "behavior_label": behavior,
        "consistency_roles": roles,
        "consistency_review_order": order,
    }


def _v3_decision(
    unit_key: str,
    track_key: str,
    behavior: str,
    start: int,
) -> dict[str, object]:
    return {
        "temporal_unit_key": unit_key,
        "object_track_key": track_key,
        "unit_start_frame": start,
        "unit_end_frame": start + 5,
        "original_behavior": behavior,
        "manual_review_decision": "accept",
        "manual_corrected_behavior": "",
    }


def _v3_trace(
    partner_key: str,
    *,
    support: int,
    first: int,
    last: int,
) -> dict[str, object]:
    return {
        "support_frame_count": support,
        "support_first_frame": first,
        "support_last_frame": last,
        "support_directions": "ACTOR_TO_PARTNER;PARTNER_TO_ACTOR",
        "selected_partner_unit_keys": f'["{partner_key}"]',
    }


def _frames() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    actor_units = [
        ("actor-a-0", 0, 1, "social-nose"),
        ("actor-a-1", 2, 3, "stand"),
        ("actor-a-2", 4, 5, "fight"),
    ]
    for unit_key, start, end, behavior in actor_units:
        for frame_index in range(start, end + 1):
            rows.append(
                _frame(
                    unit_key,
                    frame_index,
                    "actor-a",
                    "ID_1",
                    behavior,
                    "partner-b",
                )
            )
    for frame_index in range(6):
        rows.append(
            _frame(
                "partner-b",
                frame_index,
                "partner-b",
                "ID_2",
                "stand",
                "actor-a",
            )
        )
    for frame_index in range(2):
        rows.extend(
            [
                _frame(
                    "actor-c-social",
                    frame_index,
                    "actor-c",
                    "ID_3",
                    "social-nose",
                    "partner-d",
                ),
                _frame(
                    "partner-d",
                    frame_index,
                    "partner-d",
                    "ID_4",
                    "stand",
                    "actor-c",
                ),
            ]
        )
    return pd.DataFrame(rows)


def _frame(
    temporal_unit_key: str,
    frame_index: int,
    object_track_key: str,
    pig_id: str,
    behavior: str,
    nearest_partner_key: str,
) -> dict[str, object]:
    return {
        "source_type": "cvat_tracking_xml",
        "dataset_id": "dataset-a",
        "video_key": "video-a",
        "object_track_key": object_track_key,
        "frame_index": frame_index,
        "pig_id": pig_id,
        "track_id": object_track_key,
        "temporal_unit_key": temporal_unit_key,
        "behavior_temporal_final": behavior,
        "nearest_partner_key": nearest_partner_key,
    }


def _decisions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "review_item_id": "review-0",
                "review_unit_id": "actor-a-0",
                "temporal_unit_key": "actor-a-0",
                "behavior_label": "social-nose",
                "manual_review_decision": "corrected",
                "manual_corrected_behavior": "fight",
                "manual_note": "đoạn sau phản ứng fight",
            },
            {
                "review_item_id": "review-1",
                "review_unit_id": "actor-a-1",
                "temporal_unit_key": "actor-a-1",
                "behavior_label": "stand",
                "manual_review_decision": "accept",
                "manual_corrected_behavior": "",
                "manual_note": "",
            },
            {
                "review_item_id": "review-2",
                "review_unit_id": "actor-a-2",
                "temporal_unit_key": "actor-a-2",
                "behavior_label": "fight",
                "manual_review_decision": "accept",
                "manual_corrected_behavior": "",
                "manual_note": "",
            },
        ]
    )


def _scope_unit(
    temporal_unit_key: str,
    object_track_key: str,
    pig_id: str,
    behavior: str,
    *,
    start: int = 0,
) -> dict[str, object]:
    return {
        "temporal_unit_key": temporal_unit_key,
        "source_type": "cvat_tracking_xml",
        "dataset_id": "dataset-a",
        "video_key": "video-a",
        "object_track_key": object_track_key,
        "pig_id": pig_id,
        "track_id": object_track_key,
        "unit_start_frame": start,
        "unit_end_frame": start + 5,
        "effective_behavior": behavior,
        "review_item_id": "",
        "manual_review_decision": "",
        "manual_corrected_behavior": "",
        "manual_note": "",
    }
