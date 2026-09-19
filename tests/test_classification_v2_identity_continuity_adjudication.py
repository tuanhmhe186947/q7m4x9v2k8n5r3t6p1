from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from pig_behavior.classification_v2.review.identity_continuity_adjudication import (
    ADDED_BBOX_MODE,
    CASE_SIDECAR_NAME,
    CORRECTED_BBOX_MODE,
    EXCLUDED_STATUS,
    FRAME_SIDECAR_NAME,
    MANUAL_BBOX_SELECTION_KEY,
    MAPPED_STATUS,
    BoundingBoxEdit,
    FrameCandidate,
    IdentityAdjudicationError,
    IdentityCase,
    assert_safe_identity_input_path,
    assert_safe_output_dir,
    assert_safe_review_units_csv,
    case_status,
    load_frame_candidates,
    load_identity_cases,
    load_session_sidecars,
    load_session_sidecars_with_bbox_edits,
    source_frame_index_for_review_frame,
    validate_adjudication,
    write_csv_atomic,
    write_session_sidecars,
)


def _cases() -> tuple[IdentityCase, IdentityCase]:
    shared = {
        "source_type": "legacy_recovered",
        "dataset_id": "legacy",
        "video_key": "scene/001",
        "frame_indices": (3, 4),
    }
    return (
        IdentityCase(
            review_item_id="item-a",
            review_unit_id="unit-a",
            original_pig_id="ID_1",
            original_track_id="track-a",
            original_object_track_key="actor-a",
            **shared,
        ),
        IdentityCase(
            review_item_id="item-b",
            review_unit_id="unit-b",
            original_pig_id="ID_2",
            original_track_id="track-b",
            original_object_track_key="actor-b",
            **shared,
        ),
    )


def _candidate(frame_index: int, key: str, source_frame_index: int) -> FrameCandidate:
    x1 = 10.0 if key == "actor-a" else 50.0
    return FrameCandidate(
        frame_index=frame_index,
        source_frame_index=source_frame_index,
        object_track_key=key,
        track_id=key.replace("actor-", "track-"),
        pig_id=key.replace("actor-", "ID_"),
        x1=x1,
        y1=10.0,
        x2=x1 + 20.0,
        y2=30.0,
        source_video_path="C:/source/scene.mp4",
    )


def _candidates() -> dict[int, tuple[FrameCandidate, ...]]:
    return {
        3: (_candidate(3, "actor-a", 103), _candidate(3, "actor-b", 103)),
        4: (_candidate(4, "actor-a", 104), _candidate(4, "actor-b", 104)),
    }


def _complete_selection() -> dict[tuple[str, int], str]:
    return {
        ("unit-a", 3): "actor-a",
        ("unit-a", 4): "actor-b",
        ("unit-b", 3): "actor-b",
        ("unit-b", 4): "actor-a",
    }


def _all_sidecar_paths(output_dir: Path, sidecar_name: str) -> list[Path]:
    return [output_dir / sidecar_name, *output_dir.rglob(sidecar_name)]


def _rewrite_all_sidecars(
    output_dir: Path,
    sidecar_name: str,
    mutate: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    for path in _all_sidecar_paths(output_dir, sidecar_name):
        mutate(pd.read_csv(path)).to_csv(path, index=False)


def test_complete_frame_mapping_is_valid_and_local() -> None:
    cases = _cases()
    selections = _complete_selection()

    assert validate_adjudication(cases, _candidates(), selections, {}, allow_pending=False) == []
    assert case_status(cases[0], selections, {}) == MAPPED_STATUS
    assert source_frame_index_for_review_frame(_candidates(), 3) == 103


def test_same_box_cannot_be_assigned_to_two_cases_in_one_frame() -> None:
    cases = _cases()
    selections = {
        ("unit-a", 3): "actor-a",
        ("unit-b", 3): "actor-a",
    }

    errors = validate_adjudication(cases, _candidates(), selections, {}, allow_pending=True)

    assert errors == ["duplicate_actor_selection_same_frame=frame:3;cases:unit-a,unit-b"]


def test_pending_case_cannot_pass_finish_validation() -> None:
    errors = validate_adjudication(
        _cases(),
        _candidates(),
        {("unit-a", 3): "actor-a"},
        {},
        allow_pending=False,
    )

    assert errors == ["incomplete_identity_case=unit-a", "incomplete_identity_case=unit-b"]


def test_review_frame_requires_one_authoritative_source_frame() -> None:
    mismatched = _candidates()
    first, second = mismatched[3]
    mismatched[3] = (
        first,
        FrameCandidate(
            frame_index=second.frame_index,
            source_frame_index=104,
            object_track_key=second.object_track_key,
            track_id=second.track_id,
            pig_id=second.pig_id,
            x1=second.x1,
            y1=second.y1,
            x2=second.x2,
            y2=second.y2,
            source_video_path=second.source_video_path,
        ),
    )

    with pytest.raises(IdentityAdjudicationError, match="exactly_one_source_frame"):
        source_frame_index_for_review_frame(mismatched, 3)


def test_excluded_case_requires_note_and_cannot_keep_selections() -> None:
    cases = _cases()
    selections = {("unit-a", 3): "actor-a"}

    errors = validate_adjudication(
        cases,
        _candidates(),
        selections,
        {"unit-a": "unresolvable swap"},
        allow_pending=True,
    )

    assert errors == ["excluded_case_has_selected_frames=item-a:3"]
    assert case_status(cases[0], {}, {"unit-a": "unresolvable swap"}) == EXCLUDED_STATUS


def test_sidecars_preserve_source_frame_and_resume_exact_scope(tmp_path: Path) -> None:
    cases = _cases()
    selections = _complete_selection()

    frame_path, case_path = write_session_sidecars(
        tmp_path,
        cases,
        _candidates(),
        selections,
        {},
        "reviewer-1",
    )

    assert frame_path.name == FRAME_SIDECAR_NAME
    assert case_path.name == CASE_SIDECAR_NAME
    frame_rows = pd.read_csv(frame_path)
    assert len(frame_rows) == 4
    assert set(frame_rows["source_frame_index"]) == {103, 104}
    assert set(frame_rows["model_x_forbidden"]) == {"YES"}
    resumed, exclusions = load_session_sidecars(tmp_path, cases, _candidates())
    assert resumed == selections
    assert exclusions == {}


def test_sidecar_resume_rejects_duplicate_case_frame(tmp_path: Path) -> None:
    cases = _cases()
    frame_path, _ = write_session_sidecars(
        tmp_path,
        cases,
        _candidates(),
        _complete_selection(),
        {},
        "reviewer-1",
    )
    def add_duplicate(rows: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([rows, rows.iloc[[0]]], ignore_index=True)

    _rewrite_all_sidecars(tmp_path, frame_path.name, add_duplicate)

    with pytest.raises(IdentityAdjudicationError, match="sidecar_duplicate_case_frame"):
        load_session_sidecars(tmp_path, cases, _candidates())


def test_sidecar_resume_rejects_stale_case_status(tmp_path: Path) -> None:
    cases = _cases()
    _, case_path = write_session_sidecars(
        tmp_path,
        cases,
        _candidates(),
        _complete_selection(),
        {},
        "reviewer-1",
    )
    def mark_pending(rows: pd.DataFrame) -> pd.DataFrame:
        rows.loc[0, "case_status"] = "PENDING"
        return rows

    _rewrite_all_sidecars(tmp_path, case_path.name, mark_pending)

    with pytest.raises(IdentityAdjudicationError, match="sidecar_case_status_mismatch"):
        load_session_sidecars(tmp_path, cases, _candidates())


def test_identity_input_and_output_cannot_be_behavior_decision_paths(tmp_path: Path) -> None:
    behavior_csv = tmp_path / "behavior_unit_review_decisions.csv"
    behavior_csv.write_text("review_item_id\nitem-a\n", encoding="utf-8")

    with pytest.raises(IdentityAdjudicationError, match="not_behavior_ledger"):
        assert_safe_review_units_csv(behavior_csv)
    forbidden_output = (
        tmp_path
        / "human_review_workspace"
        / "classification_v2"
        / "run"
        / "human_decisions"
        / "behavior"
    )
    with pytest.raises(IdentityAdjudicationError, match="must_not_be_a_behavior"):
        assert_safe_output_dir(forbidden_output)


def test_loads_short_review_item_id_and_exact_source_frame(tmp_path: Path) -> None:
    units_path = tmp_path / "review_view.csv"
    pd.DataFrame(
        [
            {
                "review_item_id": "unit_review_00030931",
                "review_unit_id": "stable-unit-a",
                "source_type": "legacy_recovered",
                "dataset_id": "legacy",
                "video_key": "scene/001",
                "pig_id": "ID_1",
                "track_id": "track-a",
                "object_track_key": "actor-a",
                "unit_start_frame": "3",
                "unit_end_frame": "4",
                "display_frame_indices": "3,4",
                "manual_review_decision": "must_not_be_loaded",
            }
        ]
    ).to_csv(units_path, index=False)
    features_path = tmp_path / "frames.csv"
    rows = []
    for frame_index, source_frame_index in ((3, 103), (4, 104)):
        for key in ("actor-a", "actor-b"):
            rows.append(
                {
                    "source_type": "legacy_recovered",
                    "dataset_id": "legacy",
                    "video_key": "scene/001",
                    "frame_index": str(frame_index),
                    "source_frame_index": str(source_frame_index),
                    "pig_id": key.replace("actor-", "ID_"),
                    "track_id": key.replace("actor-", "track-"),
                    "object_track_key": key,
                    "x1": "1",
                    "y1": "2",
                    "x2": "11",
                    "y2": "12",
                    "bbox_valid": "True",
                    "source_video_path": "C:/source/scene.mp4",
                }
            )
    pd.DataFrame(rows).to_csv(features_path, index=False)

    cases = load_identity_cases(units_path, ["unit_review_00030931"])
    candidates = load_frame_candidates(features_path, cases)

    assert cases[0].review_unit_id == "stable-unit-a"
    assert source_frame_index_for_review_frame(candidates, 4) == 104


def test_v2_sidecar_round_trips_corrected_and_added_bbox(tmp_path: Path) -> None:
    selections = _complete_selection()
    selections[("unit-a", 4)] = MANUAL_BBOX_SELECTION_KEY
    bbox_edits = {
        ("unit-a", 3): BoundingBoxEdit(
            mode=CORRECTED_BBOX_MODE,
            x1=12.0,
            y1=11.0,
            x2=31.0,
            y2=32.0,
            source_object_track_key="actor-a",
        ),
        ("unit-a", 4): BoundingBoxEdit(
            mode=ADDED_BBOX_MODE,
            x1=15.0,
            y1=12.0,
            x2=36.0,
            y2=34.0,
        ),
    }

    write_session_sidecars(
        tmp_path,
        _cases(),
        _candidates(),
        selections,
        {},
        "reviewer",
        bbox_edits,
    )
    loaded = load_session_sidecars_with_bbox_edits(
        tmp_path,
        _cases(),
        _candidates(),
    )

    assert loaded == (selections, {}, bbox_edits)
    with pytest.raises(
        IdentityAdjudicationError,
        match="contains_bbox_edits_use_v2_loader",
    ):
        load_session_sidecars(tmp_path, _cases(), _candidates())


def test_atomic_sidecar_temp_name_stays_below_windows_path_limit(
    tmp_path: Path,
) -> None:
    parent = tmp_path
    while len(str(parent)) < 205:
        remaining = 205 - len(str(parent)) - 1
        if remaining <= 0:
            break
        parent /= "x" * min(20, remaining)
    path = parent / FRAME_SIDECAR_NAME

    assert len(str(path)) < 260
    old_temp_name = f".{path.name}.12345678.tmp"
    assert len(str(parent / old_temp_name)) >= 260
    write_csv_atomic(path, ("value",), ({"value": "saved"},))

    assert path.read_text(encoding="utf-8").splitlines() == ["value", "saved"]


@pytest.mark.parametrize(
    "edit",
    [
        BoundingBoxEdit(CORRECTED_BBOX_MODE, 1.0, 1.0, 1.0, 2.0, "actor-a"),
        BoundingBoxEdit(CORRECTED_BBOX_MODE, float("nan"), 1.0, 2.0, 2.0, "actor-a"),
        BoundingBoxEdit(CORRECTED_BBOX_MODE, -1.0, 1.0, 2.0, 2.0, "actor-a"),
    ],
)
def test_invalid_bbox_edit_geometry_fails_closed(edit: BoundingBoxEdit) -> None:
    errors = validate_adjudication(
        _cases(),
        _candidates(),
        _complete_selection(),
        {},
        {("unit-a", 3): edit},
        allow_pending=False,
    )

    assert "bbox_edit_geometry_invalid=unit-a:3" in errors


def test_corrected_bbox_must_match_selected_source_key() -> None:
    errors = validate_adjudication(
        _cases(),
        _candidates(),
        _complete_selection(),
        {},
        {
            ("unit-a", 3): BoundingBoxEdit(
                CORRECTED_BBOX_MODE,
                12.0,
                11.0,
                31.0,
                32.0,
                "actor-b",
            )
        },
        allow_pending=False,
    )

    assert "corrected_bbox_source_selection_mismatch=unit-a:3" in errors


def _write_minimal_source_tables(
    tmp_path: Path,
    *,
    display_frame_indices: str = "3,4",
    original_object_track_key: str = "actor-a",
    candidate_keys: tuple[str, ...] = ("actor-a", "actor-b"),
) -> tuple[Path, Path]:
    units_path = tmp_path / "review_view.csv"
    pd.DataFrame(
        [
            {
                "review_item_id": "item-a",
                "review_unit_id": "unit-a",
                "source_type": "legacy_recovered",
                "dataset_id": "legacy",
                "video_key": "scene/001",
                "pig_id": "ID_1",
                "track_id": "track-a",
                "object_track_key": original_object_track_key,
                "unit_start_frame": "3",
                "unit_end_frame": "4",
                "display_frame_indices": display_frame_indices,
            }
        ]
    ).to_csv(units_path, index=False)
    feature_rows: list[dict[str, str]] = []
    for frame_index, source_frame_index in ((3, 103), (4, 104)):
        for key in candidate_keys:
            feature_rows.append(
                {
                    "source_type": "legacy_recovered",
                    "dataset_id": "legacy",
                    "video_key": "scene/001",
                    "frame_index": str(frame_index),
                    "source_frame_index": str(source_frame_index),
                    "pig_id": key.replace("actor-", "ID_"),
                    "track_id": key.replace("actor-", "track-"),
                    "object_track_key": key,
                    "x1": "1",
                    "y1": "2",
                    "x2": "11",
                    "y2": "12",
                    "bbox_valid": "True",
                    "source_video_path": "C:/source/scene.mp4",
                }
            )
    features_path = tmp_path / "frames.csv"
    pd.DataFrame(feature_rows).to_csv(features_path, index=False)
    return units_path, features_path


def test_safe_input_path_rejects_nonexistent_behavior_ledger(tmp_path: Path) -> None:
    forbidden_path = (
        tmp_path
        / "human_review_workspace"
        / "classification_v2"
        / "run"
        / "human_decisions"
        / "behavior"
        / "not_created_yet.csv"
    )

    with pytest.raises(IdentityAdjudicationError, match="must_not_read_active"):
        assert_safe_identity_input_path(forbidden_path, role="frame_features_csv")


def test_display_frame_scope_must_equal_immutable_unit_bounds(tmp_path: Path) -> None:
    units_path, _ = _write_minimal_source_tables(
        tmp_path,
        display_frame_indices="3",
    )

    with pytest.raises(IdentityAdjudicationError, match="display_frame_indices_must"):
        load_identity_cases(units_path, ["item-a"])


def test_candidate_loader_requires_original_actor_on_every_target_frame(
    tmp_path: Path,
) -> None:
    units_path, features_path = _write_minimal_source_tables(
        tmp_path,
        candidate_keys=("actor-b",),
    )
    cases = load_identity_cases(units_path, ["item-a"])

    with pytest.raises(IdentityAdjudicationError, match="original_actor_candidate_missing"):
        load_frame_candidates(features_path, cases)


def test_candidate_loader_streams_frame_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    units_path, features_path = _write_minimal_source_tables(tmp_path)
    cases = load_identity_cases(units_path, ["item-a"])
    original_read_csv = pd.read_csv
    chunksizes: list[int] = []

    def observed_read_csv(*args: object, **kwargs: object) -> object:
        chunksize = kwargs.get("chunksize")
        if chunksize is not None:
            chunksizes.append(int(chunksize))
        return original_read_csv(*args, **kwargs)

    module_path = (
        "pig_behavior.classification_v2.review."
        "identity_continuity_adjudication.pd.read_csv"
    )
    monkeypatch.setattr(module_path, observed_read_csv)

    load_frame_candidates(features_path, cases)

    assert len(chunksizes) == 1
    assert chunksizes[0] > 0


@pytest.mark.parametrize(
    ("sidecar_name", "mutate", "match"),
    [
        (
            FRAME_SIDECAR_NAME,
            lambda rows: rows.assign(identity_adjudication_version="stale"),
            "sidecar_version_mismatch",
        ),
        (
            FRAME_SIDECAR_NAME,
            lambda rows: rows.assign(model_x_forbidden="NO"),
            "sidecar_model_x_forbidden_mismatch",
        ),
        (
            CASE_SIDECAR_NAME,
            lambda rows: rows.assign(source_type="wrong_source"),
            "sidecar_provenance_mismatch",
        ),
        (
            CASE_SIDECAR_NAME,
            lambda rows: rows.assign(required_frame_count=99),
            "sidecar_required_frame_count_mismatch",
        ),
    ],
)
def test_sidecar_resume_rejects_stale_immutable_provenance(
    tmp_path: Path,
    sidecar_name: str,
    mutate: Callable[[pd.DataFrame], pd.DataFrame],
    match: str,
) -> None:
    cases = _cases()
    write_session_sidecars(
        tmp_path,
        cases,
        _candidates(),
        _complete_selection(),
        {},
        "reviewer-1",
    )
    _rewrite_all_sidecars(tmp_path, sidecar_name, mutate)

    with pytest.raises(IdentityAdjudicationError, match=match):
        load_session_sidecars(tmp_path, cases, _candidates())


def test_sidecar_resume_rejects_mismatched_generation_sessions(tmp_path: Path) -> None:
    cases = _cases()
    write_session_sidecars(
        tmp_path,
        cases,
        _candidates(),
        _complete_selection(),
        {},
        "reviewer-1",
    )
    _rewrite_all_sidecars(
        tmp_path,
        CASE_SIDECAR_NAME,
        lambda rows: rows.assign(identity_adjudication_session_id="other-session"),
    )

    with pytest.raises(IdentityAdjudicationError, match="sidecar_session_id_mismatch"):
        load_session_sidecars(tmp_path, cases, _candidates())


def test_sidecar_resume_recovers_completed_generation_after_current_pair_failure(
    tmp_path: Path,
) -> None:
    cases = _cases()
    selections = _complete_selection()
    _, current_case_path = write_session_sidecars(
        tmp_path,
        cases,
        _candidates(),
        selections,
        {},
        "reviewer-1",
    )
    pd.read_csv(current_case_path).assign(
        identity_adjudication_session_id="partial-save"
    ).to_csv(current_case_path, index=False)

    resumed, exclusions = load_session_sidecars(tmp_path, cases, _candidates())

    assert resumed == selections
    assert exclusions == {}
