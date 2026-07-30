"""Sidecar contract for local actor-trajectory identity adjudication.

This contract deliberately does not alter source annotations, behavior-review
decisions, or model input features.  It records a reviewer-selected existing
actor box for each affected frame, or a defensible exclusion when continuity
cannot be established.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

import pandas as pd

IDENTITY_ADJUDICATION_VERSION = "classification_v2.identity_continuity_adjudication.v1"
MAPPED_STATUS = "MAPPED"
EXCLUDED_STATUS = "EXCLUDE_IDENTITY_CONTINUITY_DEFECT"
PENDING_STATUS = "PENDING"
FRAME_FEATURE_CHUNK_ROWS = 100_000
SIDECAR_SESSION_COLUMN = "identity_adjudication_session_id"
PROTECTED_BEHAVIOR_DECISION_FILENAMES = frozenset(
    {
        "behavior_unit_review_decisions.csv",
        "behavior_strength_review_decisions.csv",
        "behavior_label_quality_review.csv",
    }
)

REVIEW_UNIT_COLUMNS = (
    "review_item_id",
    "review_unit_id",
    "source_type",
    "dataset_id",
    "video_key",
    "pig_id",
    "track_id",
    "object_track_key",
    "unit_start_frame",
    "unit_end_frame",
    "display_frame_indices",
)
FRAME_FEATURE_COLUMNS = (
    "source_type",
    "dataset_id",
    "video_key",
    "frame_index",
    "source_frame_index",
    "pig_id",
    "track_id",
    "object_track_key",
    "x1",
    "y1",
    "x2",
    "y2",
    "bbox_valid",
    "source_video_path",
)
FRAME_SIDECAR_NAME = "identity_continuity_frame_adjudications.csv"
CASE_SIDECAR_NAME = "identity_continuity_case_adjudications.csv"
SIDECAR_GENERATIONS_DIR_NAME = "identity_continuity_sidecar_generations"
FRAME_SIDECAR_COLUMNS = (
    "identity_adjudication_version",
    SIDECAR_SESSION_COLUMN,
    "review_item_id",
    "review_unit_id",
    "source_type",
    "dataset_id",
    "video_key",
    "frame_index",
    "source_frame_index",
    "original_object_track_key",
    "original_track_id",
    "original_pig_id",
    "selected_object_track_key",
    "selected_track_id",
    "selected_pig_id",
    "selection_status",
    "reviewer",
    "adjudication_note",
    "model_x_forbidden",
)
CASE_SIDECAR_COLUMNS = (
    "identity_adjudication_version",
    SIDECAR_SESSION_COLUMN,
    "review_item_id",
    "review_unit_id",
    "source_type",
    "dataset_id",
    "video_key",
    "original_object_track_key",
    "original_track_id",
    "original_pig_id",
    "required_frame_count",
    "mapped_frame_count",
    "case_status",
    "reviewer",
    "adjudication_note",
    "model_x_forbidden",
)
MODEL_X_FORBIDDEN_COLUMNS = FRAME_SIDECAR_COLUMNS + CASE_SIDECAR_COLUMNS


class IdentityAdjudicationError(ValueError):
    """Raised when an identity-adjudication session is unsafe or inconsistent."""


@dataclass(frozen=True)
class IdentityCase:
    """One existing review unit whose actor trajectory needs adjudication."""

    review_item_id: str
    review_unit_id: str
    source_type: str
    dataset_id: str
    video_key: str
    original_pig_id: str
    original_track_id: str
    original_object_track_key: str
    frame_indices: tuple[int, ...]


@dataclass(frozen=True)
class FrameCandidate:
    """An existing source actor box eligible for selection at one frame."""

    frame_index: int
    source_frame_index: int
    object_track_key: str
    track_id: str
    pig_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    source_video_path: str

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _frame_number(value: Any, *, field: str) -> int:
    try:
        numeric = float(_text(value))
    except ValueError as exc:
        raise IdentityAdjudicationError(f"invalid_{field}={value!r}") from exc
    if not numeric.is_integer():
        raise IdentityAdjudicationError(f"nonintegral_{field}={value!r}")
    return int(numeric)


def _required_columns(path: Path, required: Sequence[str]) -> None:
    available = set(pd.read_csv(path, nrows=0).columns)
    missing = sorted(set(required) - available)
    if missing:
        raise IdentityAdjudicationError(
            f"missing_required_columns={','.join(missing)} path={path}"
        )


def parse_frame_indices(value: Any) -> tuple[int, ...]:
    """Parse a finite, strictly increasing frame sequence from a CSV cell."""

    text = _text(value)
    if not text:
        return ()
    try:
        frames = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise IdentityAdjudicationError(f"invalid_display_frame_indices={text!r}") from exc
    if not frames:
        return ()
    if len(set(frames)) != len(frames):
        raise IdentityAdjudicationError(f"duplicate_display_frame_indices={text!r}")
    if tuple(sorted(frames)) != frames:
        raise IdentityAdjudicationError(f"nonmonotonic_display_frame_indices={text!r}")
    return frames


def _has_path_segment_sequence(parts: Sequence[str], sequence: Sequence[str]) -> bool:
    return any(
        tuple(parts[index : index + len(sequence)]) == tuple(sequence)
        for index in range(len(parts) - len(sequence) + 1)
    )


def assert_safe_identity_input_path(path: Path, *, role: str) -> Path:
    """Reject Behavior decision ledgers before attempting to inspect an input."""

    candidate = Path(path).resolve()
    if candidate.name.casefold() in PROTECTED_BEHAVIOR_DECISION_FILENAMES:
        if role == "review_units_csv":
            raise IdentityAdjudicationError(
                "identity_adjudication_requires_review_unit_view_not_behavior_ledger"
            )
        raise IdentityAdjudicationError(
            f"identity_adjudication_{role}_must_not_be_behavior_ledger"
        )
    parts = tuple(part.casefold() for part in candidate.parts)
    is_classification_workspace = _has_path_segment_sequence(
        parts,
        ("human_review_workspace", "classification_v2"),
    )
    is_behavior_decision_area = _has_path_segment_sequence(
        parts,
        ("human_decisions", "behavior"),
    )
    if is_classification_workspace and is_behavior_decision_area:
        raise IdentityAdjudicationError(
            f"identity_adjudication_{role}_must_not_read_active_behavior_decisions"
        )
    return candidate


def assert_safe_review_units_csv(review_units_csv: Path) -> Path:
    """Compatibility wrapper for the immutable review-unit-view input."""

    return assert_safe_identity_input_path(
        review_units_csv,
        role="review_units_csv",
    )


def load_identity_cases(
    review_units_csv: Path,
    review_item_ids: Sequence[str],
) -> tuple[IdentityCase, ...]:
    """Load only identity-safe fields from an immutable review-unit view."""

    requested = tuple(_text(value) for value in review_item_ids if _text(value))
    if not requested:
        raise IdentityAdjudicationError("at_least_one_review_item_id_is_required")
    if len(set(requested)) != len(requested):
        raise IdentityAdjudicationError("duplicate_requested_review_item_id")

    review_units_csv = assert_safe_review_units_csv(review_units_csv)
    _required_columns(review_units_csv, REVIEW_UNIT_COLUMNS)
    units = pd.read_csv(
        review_units_csv,
        usecols=lambda column: column in set(REVIEW_UNIT_COLUMNS),
        dtype=str,
    )
    cases: list[IdentityCase] = []
    for review_item_id in requested:
        matches = units.loc[units["review_item_id"].map(_text).eq(review_item_id)]
        if len(matches) != 1:
            raise IdentityAdjudicationError(
                "review_item_id_must_match_once="
                f"{review_item_id!r};matches={len(matches)}"
            )
        row = matches.iloc[0]
        start = _frame_number(row.get("unit_start_frame"), field="unit_start_frame")
        end = _frame_number(row.get("unit_end_frame"), field="unit_end_frame")
        if end < start:
            raise IdentityAdjudicationError(
                f"invalid_frame_bounds={review_item_id}:{start}>{end}"
            )
        expected_frames = tuple(range(start, end + 1))
        frames = parse_frame_indices(row.get("display_frame_indices"))
        if frames != expected_frames:
            raise IdentityAdjudicationError(
                "display_frame_indices_must_match_unit_bounds="
                f"{review_item_id}:{frames!r}!={expected_frames!r}"
            )
        case = IdentityCase(
            review_item_id=review_item_id,
            review_unit_id=_text(row.get("review_unit_id")),
            source_type=_text(row.get("source_type")),
            dataset_id=_text(row.get("dataset_id")),
            video_key=_text(row.get("video_key")),
            original_pig_id=_text(row.get("pig_id")),
            original_track_id=_text(row.get("track_id")),
            original_object_track_key=_text(row.get("object_track_key")),
            frame_indices=frames,
        )
        required_values = {
            "review_unit_id": case.review_unit_id,
            "source_type": case.source_type,
            "dataset_id": case.dataset_id,
            "video_key": case.video_key,
            "object_track_key": case.original_object_track_key,
        }
        missing_values = sorted(name for name, value in required_values.items() if not value)
        if missing_values:
            raise IdentityAdjudicationError(
                "identity_case_has_blank_required_value="
                f"{review_item_id}:{','.join(missing_values)}"
            )
        if any(existing.review_unit_id == case.review_unit_id for existing in cases):
            raise IdentityAdjudicationError(
                f"duplicate_review_unit_id={case.review_unit_id}"
            )
        cases.append(case)
    return tuple(cases)


def assert_single_scene(cases: Sequence[IdentityCase]) -> None:
    """Require one scene so overlapping cases can be collision-checked together."""

    scenes = {(case.source_type, case.dataset_id, case.video_key) for case in cases}
    if len(scenes) != 1:
        raise IdentityAdjudicationError(
            "selected_review_units_must_share_one_source_dataset_video_scene"
        )


def _valid_bbox(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    if _text(row.get("bbox_valid")).casefold() not in {"true", "1", "yes"}:
        raise IdentityAdjudicationError(
            "invalid_candidate_bbox_valid="
            f"frame={_text(row.get('frame_index'))} "
            f"object_track_key={_text(row.get('object_track_key'))}"
        )
    try:
        x1 = float(_text(row.get("x1")))
        y1 = float(_text(row.get("y1")))
        x2 = float(_text(row.get("x2")))
        y2 = float(_text(row.get("y2")))
    except ValueError as exc:
        raise IdentityAdjudicationError("candidate_bbox_has_non_numeric_coordinate") from exc
    if x2 <= x1 or y2 <= y1:
        raise IdentityAdjudicationError(
            "candidate_bbox_has_nonpositive_extent="
            f"object_track_key={_text(row.get('object_track_key'))}"
        )
    return x1, y1, x2, y2


def load_frame_candidates(
    frame_features_csv: Path,
    cases: Sequence[IdentityCase],
) -> dict[int, tuple[FrameCandidate, ...]]:
    """Load only source actor boxes required for the selected, single scene."""

    if not cases:
        raise IdentityAdjudicationError("at_least_one_identity_case_is_required")
    assert_single_scene(cases)
    frame_features_csv = assert_safe_identity_input_path(
        frame_features_csv,
        role="frame_features_csv",
    )
    _required_columns(frame_features_csv, FRAME_FEATURE_COLUMNS)
    source_type, dataset_id, video_key = next(
        iter({(case.source_type, case.dataset_id, case.video_key) for case in cases})
    )
    wanted_frames = {frame for case in cases for frame in case.frame_indices}
    filtered_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        frame_features_csv,
        usecols=lambda column: column in set(FRAME_FEATURE_COLUMNS),
        dtype=str,
        chunksize=FRAME_FEATURE_CHUNK_ROWS,
    ):
        source_rows = chunk.loc[
            chunk["source_type"].map(_text).eq(source_type)
            & chunk["dataset_id"].map(_text).eq(dataset_id)
            & chunk["video_key"].map(_text).eq(video_key)
        ].copy()
        if source_rows.empty:
            continue
        source_rows["_frame_index"] = source_rows["frame_index"].map(
            lambda value: _frame_number(value, field="frame_index")
        )
        wanted_rows = source_rows.loc[
            source_rows["_frame_index"].isin(wanted_frames)
        ].copy()
        if not wanted_rows.empty:
            filtered_chunks.append(wanted_rows)
    if filtered_chunks:
        filtered = pd.concat(filtered_chunks, ignore_index=True)
    else:
        filtered = pd.DataFrame(columns=(*FRAME_FEATURE_COLUMNS, "_frame_index"))
    candidates: dict[int, list[FrameCandidate]] = {frame: [] for frame in wanted_frames}
    seen: set[tuple[int, str]] = set()
    for row in filtered.to_dict(orient="records"):
        frame_index = int(row["_frame_index"])
        source_frame_index = _frame_number(
            row.get("source_frame_index"),
            field="source_frame_index",
        )
        object_track_key = _text(row.get("object_track_key"))
        if not object_track_key:
            raise IdentityAdjudicationError(
                f"blank_candidate_object_track_key=frame:{frame_index}"
            )
        duplicate_key = (frame_index, object_track_key)
        if duplicate_key in seen:
            raise IdentityAdjudicationError(
                "duplicate_candidate_object_track_key="
                f"frame:{frame_index};key:{object_track_key}"
            )
        seen.add(duplicate_key)
        x1, y1, x2, y2 = _valid_bbox(row)
        candidates[frame_index].append(
            FrameCandidate(
                frame_index=frame_index,
                source_frame_index=source_frame_index,
                object_track_key=object_track_key,
                track_id=_text(row.get("track_id")),
                pig_id=_text(row.get("pig_id")),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                source_video_path=_text(row.get("source_video_path")),
            )
        )
    missing_frames = sorted(frame for frame, values in candidates.items() if not values)
    if missing_frames:
        raise IdentityAdjudicationError(
            "identity_candidate_frame_missing="
            + ",".join(str(frame) for frame in missing_frames)
        )
    ordered_candidates = {
        frame: tuple(
            sorted(
                values,
                key=lambda candidate: (
                    candidate.pig_id.casefold(),
                    candidate.track_id.casefold(),
                    candidate.object_track_key.casefold(),
                ),
            )
        )
        for frame, values in candidates.items()
    }
    for frame_index in ordered_candidates:
        source_frame_index_for_review_frame(ordered_candidates, frame_index)
    available_candidates = candidate_lookup(ordered_candidates)
    for case in cases:
        missing_original_frames = [
            frame_index
            for frame_index in case.frame_indices
            if (frame_index, case.original_object_track_key)
            not in available_candidates
        ]
        if missing_original_frames:
            raise IdentityAdjudicationError(
                "original_actor_candidate_missing="
                f"{case.review_unit_id}:"
                f"{','.join(map(str, missing_original_frames))}"
            )
    return ordered_candidates


def source_frame_index_for_review_frame(
    candidates_by_frame: Mapping[int, Sequence[FrameCandidate]],
    frame_index: int,
) -> int:
    """Resolve the one authoritative decode index for a review-frame overlay."""

    candidates = candidates_by_frame.get(frame_index, ())
    source_indices = {candidate.source_frame_index for candidate in candidates}
    if len(source_indices) != 1:
        raise IdentityAdjudicationError(
            "review_frame_requires_exactly_one_source_frame_index="
            f"review_frame:{frame_index};count:{len(source_indices)}"
        )
    return next(iter(source_indices))


def candidate_lookup(
    candidates_by_frame: Mapping[int, Sequence[FrameCandidate]],
) -> dict[tuple[int, str], FrameCandidate]:
    """Index available source boxes by their local actor key and frame."""

    return {
        (frame_index, candidate.object_track_key): candidate
        for frame_index, candidates in candidates_by_frame.items()
        for candidate in candidates
    }


def case_status(
    case: IdentityCase,
    selections: Mapping[tuple[str, int], str],
    exclusions: Mapping[str, str],
) -> str:
    if case.review_unit_id in exclusions:
        return EXCLUDED_STATUS
    selected_count = sum(
        (case.review_unit_id, frame_index) in selections
        for frame_index in case.frame_indices
    )
    return MAPPED_STATUS if selected_count == len(case.frame_indices) else PENDING_STATUS


def validate_adjudication(
    cases: Sequence[IdentityCase],
    candidates_by_frame: Mapping[int, Sequence[FrameCandidate]],
    selections: Mapping[tuple[str, int], str],
    exclusions: Mapping[str, str],
    *,
    allow_pending: bool,
) -> list[str]:
    """Validate exact box membership, collision freedom, and completion state."""

    errors: list[str] = []
    case_by_id = {case.review_unit_id: case for case in cases}
    candidate_by_key = candidate_lookup(candidates_by_frame)
    for frame_index in candidates_by_frame:
        try:
            source_frame_index_for_review_frame(candidates_by_frame, frame_index)
        except IdentityAdjudicationError as exc:
            errors.append(str(exc))
    allowed_case_frames = {
        (case.review_unit_id, frame_index)
        for case in cases
        for frame_index in case.frame_indices
    }
    for key, candidate_key in selections.items():
        if key not in allowed_case_frames:
            errors.append(f"selection_outside_case_frame_scope={key[0]}:{key[1]}")
            continue
        if (key[1], candidate_key) not in candidate_by_key:
            errors.append(f"selection_not_an_available_box={key[0]}:{key[1]}")
    for review_unit_id, note in exclusions.items():
        if review_unit_id not in case_by_id:
            errors.append(f"exclusion_for_unknown_review_unit={review_unit_id}")
        elif not _text(note):
            errors.append(f"exclusion_requires_note={review_unit_id}")
    for case in cases:
        if case.review_unit_id not in exclusions:
            continue
        selected_frames = [
            frame_index
            for frame_index in case.frame_indices
            if (case.review_unit_id, frame_index) in selections
        ]
        if selected_frames:
            errors.append(
                "excluded_case_has_selected_frames="
                f"{case.review_item_id}:{','.join(map(str, selected_frames))}"
            )

    for frame_index in candidates_by_frame:
        assigned: dict[str, str] = {}
        for case in cases:
            if case.review_unit_id in exclusions:
                continue
            candidate_key = selections.get((case.review_unit_id, frame_index))
            if not candidate_key:
                continue
            prior_case = assigned.get(candidate_key)
            if prior_case is not None:
                errors.append(
                    "duplicate_actor_selection_same_frame="
                    f"frame:{frame_index};cases:{prior_case},{case.review_unit_id}"
                )
            assigned[candidate_key] = case.review_unit_id

    if not allow_pending:
        for case in cases:
            if case_status(case, selections, exclusions) == PENDING_STATUS:
                errors.append(f"incomplete_identity_case={case.review_unit_id}")
    return sorted(set(errors))


def frame_adjudication_records(
    cases: Sequence[IdentityCase],
    candidates_by_frame: Mapping[int, Sequence[FrameCandidate]],
    selections: Mapping[tuple[str, int], str],
    exclusions: Mapping[str, str],
    reviewer: str,
    *,
    session_id: str,
) -> list[dict[str, str | int]]:
    """Build non-model sidecar rows without changing any source identity field."""

    by_key = candidate_lookup(candidates_by_frame)
    records: list[dict[str, str | int]] = []
    for case in cases:
        status = case_status(case, selections, exclusions)
        note = _text(exclusions.get(case.review_unit_id, ""))
        for frame_index in case.frame_indices:
            selected_key = selections.get((case.review_unit_id, frame_index), "")
            selected = by_key.get((frame_index, selected_key))
            records.append(
                {
                    "identity_adjudication_version": IDENTITY_ADJUDICATION_VERSION,
                    SIDECAR_SESSION_COLUMN: session_id,
                    "review_item_id": case.review_item_id,
                    "review_unit_id": case.review_unit_id,
                    "source_type": case.source_type,
                    "dataset_id": case.dataset_id,
                    "video_key": case.video_key,
                    "frame_index": frame_index,
                    "source_frame_index": (
                        candidates_by_frame[frame_index][0].source_frame_index
                    ),
                    "original_object_track_key": case.original_object_track_key,
                    "original_track_id": case.original_track_id,
                    "original_pig_id": case.original_pig_id,
                    "selected_object_track_key": (
                        "" if selected is None else selected.object_track_key
                    ),
                    "selected_track_id": "" if selected is None else selected.track_id,
                    "selected_pig_id": "" if selected is None else selected.pig_id,
                    "selection_status": status,
                    "reviewer": _text(reviewer),
                    "adjudication_note": note,
                    "model_x_forbidden": "YES",
                }
            )
    return records


def case_adjudication_records(
    cases: Sequence[IdentityCase],
    selections: Mapping[tuple[str, int], str],
    exclusions: Mapping[str, str],
    reviewer: str,
    *,
    session_id: str,
) -> list[dict[str, str | int]]:
    """Summarize each case while retaining a separate frame-level mapping."""

    records: list[dict[str, str | int]] = []
    for case in cases:
        mapped_frame_count = sum(
            (case.review_unit_id, frame_index) in selections
            for frame_index in case.frame_indices
        )
        records.append(
            {
                "identity_adjudication_version": IDENTITY_ADJUDICATION_VERSION,
                SIDECAR_SESSION_COLUMN: session_id,
                "review_item_id": case.review_item_id,
                "review_unit_id": case.review_unit_id,
                "source_type": case.source_type,
                "dataset_id": case.dataset_id,
                "video_key": case.video_key,
                "original_object_track_key": case.original_object_track_key,
                "original_track_id": case.original_track_id,
                "original_pig_id": case.original_pig_id,
                "required_frame_count": len(case.frame_indices),
                "mapped_frame_count": mapped_frame_count,
                "case_status": case_status(case, selections, exclusions),
                "reviewer": _text(reviewer),
                "adjudication_note": _text(exclusions.get(case.review_unit_id, "")),
                "model_x_forbidden": "YES",
            }
        )
    return records


def assert_safe_output_dir(output_dir: Path) -> Path:
    """Refuse any output root that could overwrite a behavior decision ledger."""

    output_dir = Path(output_dir).resolve()
    normalized = "/".join(part.casefold() for part in output_dir.parts)
    forbidden = "human_review_workspace/classification_v2/"
    if forbidden in normalized and "/human_decisions/behavior" in normalized:
        raise IdentityAdjudicationError(
            "identity_adjudication_output_must_not_be_a_behavior_decision_root"
        )
    protected_files = (
        "behavior_unit_review_decisions.csv",
        "behavior_strength_review_decisions.csv",
        "behavior_label_quality_review.csv",
    )
    existing = [name for name in protected_files if (output_dir / name).exists()]
    if existing:
        raise IdentityAdjudicationError(
            "identity_adjudication_output_contains_behavior_decision_file="
            + ",".join(existing)
        )
    return output_dir


def write_csv_atomic(
    path: Path,
    columns: Sequence[str],
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Write an auditable sidecar atomically, including a valid empty header."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            writer = csv.DictWriter(
                handle,
                fieldnames=list(columns),
                extrasaction="raise",
            )
            writer.writeheader()
            for record in records:
                writer.writerow({column: record.get(column, "") for column in columns})
        os.replace(temp_path, path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def _assert_sidecar_case_provenance(
    row: Mapping[str, Any],
    case: IdentityCase,
    *,
    sidecar_name: str,
) -> None:
    if _text(row.get("identity_adjudication_version")) != IDENTITY_ADJUDICATION_VERSION:
        raise IdentityAdjudicationError(
            f"sidecar_version_mismatch={sidecar_name}:{case.review_unit_id}"
        )
    if _text(row.get("model_x_forbidden")) != "YES":
        raise IdentityAdjudicationError(
            "sidecar_model_x_forbidden_mismatch="
            f"{sidecar_name}:{case.review_unit_id}"
        )
    expected_values = {
        "review_item_id": case.review_item_id,
        "review_unit_id": case.review_unit_id,
        "source_type": case.source_type,
        "dataset_id": case.dataset_id,
        "video_key": case.video_key,
        "original_object_track_key": case.original_object_track_key,
        "original_track_id": case.original_track_id,
        "original_pig_id": case.original_pig_id,
    }
    for column, expected in expected_values.items():
        if _text(row.get(column)) != expected:
            raise IdentityAdjudicationError(
                "sidecar_provenance_mismatch="
                f"{sidecar_name}:{case.review_unit_id}:{column}"
            )


def _single_nonblank_sidecar_value(
    rows: Sequence[Mapping[str, Any]],
    column: str,
    *,
    sidecar_name: str,
) -> str:
    values = {_text(row.get(column)) for row in rows}
    if not values or "" in values or len(values) != 1:
        raise IdentityAdjudicationError(
            "sidecar_value_must_be_single_nonblank="
            f"{sidecar_name}:{column}"
        )
    return next(iter(values))


def _read_sidecar_rows(path: Path) -> list[dict[str, str]]:
    try:
        return pd.read_csv(path, dtype=str).to_dict(orient="records")
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise IdentityAdjudicationError(f"sidecar_unreadable={path.name}") from exc


def _load_validated_sidecar_pair(
    frame_path: Path,
    case_path: Path,
    cases: Sequence[IdentityCase],
    candidates_by_frame: Mapping[int, Sequence[FrameCandidate]],
) -> tuple[dict[tuple[str, int], str], dict[str, str]]:
    _required_columns(frame_path, FRAME_SIDECAR_COLUMNS)
    _required_columns(case_path, CASE_SIDECAR_COLUMNS)
    frame_rows = _read_sidecar_rows(frame_path)
    case_rows = _read_sidecar_rows(case_path)
    frame_session_id = _single_nonblank_sidecar_value(
        frame_rows,
        SIDECAR_SESSION_COLUMN,
        sidecar_name=FRAME_SIDECAR_NAME,
    )
    case_session_id = _single_nonblank_sidecar_value(
        case_rows,
        SIDECAR_SESSION_COLUMN,
        sidecar_name=CASE_SIDECAR_NAME,
    )
    if frame_session_id != case_session_id:
        raise IdentityAdjudicationError("sidecar_session_id_mismatch")
    frame_reviewer = _single_nonblank_sidecar_value(
        frame_rows,
        "reviewer",
        sidecar_name=FRAME_SIDECAR_NAME,
    )
    case_reviewer = _single_nonblank_sidecar_value(
        case_rows,
        "reviewer",
        sidecar_name=CASE_SIDECAR_NAME,
    )
    if frame_reviewer != case_reviewer:
        raise IdentityAdjudicationError("sidecar_reviewer_mismatch")

    case_by_id = {case.review_unit_id: case for case in cases}
    known_ids = set(case_by_id)
    expected_frame_keys = {
        (case.review_unit_id, frame_index)
        for case in cases
        for frame_index in case.frame_indices
    }
    candidates_by_key = candidate_lookup(candidates_by_frame)
    selections: dict[tuple[str, int], str] = {}
    seen_frame_keys: set[tuple[str, int]] = set()
    for row in frame_rows:
        review_unit_id = _text(row.get("review_unit_id"))
        if review_unit_id not in known_ids:
            raise IdentityAdjudicationError(
                f"sidecar_contains_unknown_review_unit={review_unit_id}"
            )
        case = case_by_id[review_unit_id]
        _assert_sidecar_case_provenance(
            row,
            case,
            sidecar_name=FRAME_SIDECAR_NAME,
        )
        frame_index = _frame_number(row.get("frame_index"), field="frame_index")
        frame_key = (review_unit_id, frame_index)
        if frame_key not in expected_frame_keys:
            raise IdentityAdjudicationError(
                f"sidecar_frame_outside_case_scope={review_unit_id}:{frame_index}"
            )
        if frame_key in seen_frame_keys:
            raise IdentityAdjudicationError(
                f"sidecar_duplicate_case_frame={review_unit_id}:{frame_index}"
            )
        seen_frame_keys.add(frame_key)
        expected_source_frame = source_frame_index_for_review_frame(
            candidates_by_frame,
            frame_index,
        )
        actual_source_frame = _frame_number(
            row.get("source_frame_index"),
            field="source_frame_index",
        )
        if actual_source_frame != expected_source_frame:
            raise IdentityAdjudicationError(
                "sidecar_source_frame_mismatch="
                f"review:{frame_index};expected:{expected_source_frame};"
                f"actual:{actual_source_frame}"
            )
        selected_key = _text(row.get("selected_object_track_key"))
        selected_track_id = _text(row.get("selected_track_id"))
        selected_pig_id = _text(row.get("selected_pig_id"))
        if not selected_key:
            if selected_track_id or selected_pig_id:
                raise IdentityAdjudicationError(
                    "sidecar_blank_selection_has_metadata="
                    f"{review_unit_id}:{frame_index}"
                )
            continue
        selected = candidates_by_key.get((frame_index, selected_key))
        if selected is None:
            raise IdentityAdjudicationError(
                "sidecar_selection_not_an_available_box="
                f"{review_unit_id}:{frame_index}"
            )
        if (
            selected_track_id != selected.track_id
            or selected_pig_id != selected.pig_id
        ):
            raise IdentityAdjudicationError(
                "sidecar_selected_metadata_mismatch="
                f"{review_unit_id}:{frame_index}"
            )
        selections[frame_key] = selected_key
    if seen_frame_keys != expected_frame_keys:
        raise IdentityAdjudicationError("sidecar_frame_scope_is_incomplete")

    exclusions: dict[str, str] = {}
    case_rows_by_id: dict[str, Mapping[str, Any]] = {}
    for row in case_rows:
        review_unit_id = _text(row.get("review_unit_id"))
        if review_unit_id not in known_ids:
            raise IdentityAdjudicationError(
                f"sidecar_case_contains_unknown_review_unit={review_unit_id}"
            )
        if review_unit_id in case_rows_by_id:
            raise IdentityAdjudicationError(
                f"sidecar_duplicate_case={review_unit_id}"
            )
        case = case_by_id[review_unit_id]
        _assert_sidecar_case_provenance(
            row,
            case,
            sidecar_name=CASE_SIDECAR_NAME,
        )
        required_frame_count = _frame_number(
            row.get("required_frame_count"),
            field="required_frame_count",
        )
        if required_frame_count != len(case.frame_indices):
            raise IdentityAdjudicationError(
                f"sidecar_required_frame_count_mismatch={review_unit_id}"
            )
        case_rows_by_id[review_unit_id] = row
        if _text(row.get("case_status")) == EXCLUDED_STATUS:
            exclusions[review_unit_id] = _text(row.get("adjudication_note"))
    if set(case_rows_by_id) != known_ids:
        raise IdentityAdjudicationError("sidecar_case_scope_is_incomplete")

    errors = validate_adjudication(
        cases,
        candidates_by_frame,
        selections,
        exclusions,
        allow_pending=True,
    )
    if errors:
        raise IdentityAdjudicationError(";".join(errors))
    expected_status = {
        case.review_unit_id: case_status(case, selections, exclusions)
        for case in cases
    }
    for row in frame_rows:
        review_unit_id = _text(row.get("review_unit_id"))
        expected_note = _text(exclusions.get(review_unit_id, ""))
        if _text(row.get("selection_status")) != expected_status[review_unit_id]:
            raise IdentityAdjudicationError(
                f"sidecar_frame_status_mismatch={review_unit_id}"
            )
        if _text(row.get("adjudication_note")) != expected_note:
            raise IdentityAdjudicationError(
                f"sidecar_frame_note_mismatch={review_unit_id}"
            )
    for case in cases:
        row = case_rows_by_id[case.review_unit_id]
        actual_mapped_count = _frame_number(
            row.get("mapped_frame_count"),
            field="mapped_frame_count",
        )
        expected_mapped_count = sum(
            (case.review_unit_id, frame_index) in selections
            for frame_index in case.frame_indices
        )
        if actual_mapped_count != expected_mapped_count:
            raise IdentityAdjudicationError(
                f"sidecar_mapped_frame_count_mismatch={case.review_unit_id}"
            )
        if _text(row.get("case_status")) != expected_status[case.review_unit_id]:
            raise IdentityAdjudicationError(
                f"sidecar_case_status_mismatch={case.review_unit_id}"
            )
        expected_note = _text(exclusions.get(case.review_unit_id, ""))
        if _text(row.get("adjudication_note")) != expected_note:
            raise IdentityAdjudicationError(
                f"sidecar_case_note_mismatch={case.review_unit_id}"
            )
    return selections, exclusions


def _generation_paths(output_dir: Path, session_id: str) -> tuple[Path, Path]:
    generation_dir = (
        output_dir / SIDECAR_GENERATIONS_DIR_NAME / session_id
    )
    return (
        generation_dir / FRAME_SIDECAR_NAME,
        generation_dir / CASE_SIDECAR_NAME,
    )


def _recover_latest_generation(
    output_dir: Path,
    cases: Sequence[IdentityCase],
    candidates_by_frame: Mapping[int, Sequence[FrameCandidate]],
) -> tuple[dict[tuple[str, int], str], dict[str, str]] | None:
    generation_root = output_dir / SIDECAR_GENERATIONS_DIR_NAME
    if not generation_root.is_dir():
        return None
    generation_dirs = sorted(
        (path for path in generation_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for generation_dir in generation_dirs:
        frame_path = generation_dir / FRAME_SIDECAR_NAME
        case_path = generation_dir / CASE_SIDECAR_NAME
        if not frame_path.is_file() or not case_path.is_file():
            continue
        try:
            return _load_validated_sidecar_pair(
                frame_path,
                case_path,
                cases,
                candidates_by_frame,
            )
        except IdentityAdjudicationError:
            continue
    return None


def write_session_sidecars(
    output_dir: Path,
    cases: Sequence[IdentityCase],
    candidates_by_frame: Mapping[int, Sequence[FrameCandidate]],
    selections: Mapping[tuple[str, int], str],
    exclusions: Mapping[str, str],
    reviewer: str,
) -> tuple[Path, Path]:
    """Persist only the identity-adjudication sidecars after validation."""

    output_dir = assert_safe_output_dir(output_dir)
    reviewer = _text(reviewer)
    if not reviewer:
        raise IdentityAdjudicationError("identity_adjudication_reviewer_is_required")
    errors = validate_adjudication(
        cases,
        candidates_by_frame,
        selections,
        exclusions,
        allow_pending=True,
    )
    if errors:
        raise IdentityAdjudicationError(";".join(errors))
    frame_path = output_dir / FRAME_SIDECAR_NAME
    case_path = output_dir / CASE_SIDECAR_NAME
    session_id = uuid4().hex
    frame_records = frame_adjudication_records(
        cases,
        candidates_by_frame,
        selections,
        exclusions,
        reviewer,
        session_id=session_id,
    )
    case_records = case_adjudication_records(
        cases,
        selections,
        exclusions,
        reviewer,
        session_id=session_id,
    )
    generation_frame_path, generation_case_path = _generation_paths(
        output_dir,
        session_id,
    )
    generation_frame_path.parent.mkdir(parents=True, exist_ok=False)
    write_csv_atomic(
        generation_frame_path,
        FRAME_SIDECAR_COLUMNS,
        frame_records,
    )
    write_csv_atomic(
        generation_case_path,
        CASE_SIDECAR_COLUMNS,
        case_records,
    )
    _load_validated_sidecar_pair(
        generation_frame_path,
        generation_case_path,
        cases,
        candidates_by_frame,
    )
    write_csv_atomic(
        frame_path,
        FRAME_SIDECAR_COLUMNS,
        frame_records,
    )
    write_csv_atomic(
        case_path,
        CASE_SIDECAR_COLUMNS,
        case_records,
    )
    return frame_path, case_path


def load_session_sidecars(
    output_dir: Path,
    cases: Sequence[IdentityCase],
    candidates_by_frame: Mapping[int, Sequence[FrameCandidate]],
) -> tuple[dict[tuple[str, int], str], dict[str, str]]:
    """Resume a consistent current or completed-generation sidecar pair."""

    output_dir = assert_safe_output_dir(output_dir)
    frame_path = output_dir / FRAME_SIDECAR_NAME
    case_path = output_dir / CASE_SIDECAR_NAME
    current_error: IdentityAdjudicationError | None = None
    if frame_path.exists() and case_path.exists():
        try:
            return _load_validated_sidecar_pair(
                frame_path,
                case_path,
                cases,
                candidates_by_frame,
            )
        except IdentityAdjudicationError as exc:
            current_error = exc
    elif frame_path.exists() or case_path.exists():
        current_error = IdentityAdjudicationError(
            "identity_session_sidecars_must_exist_together"
        )

    recovered = _recover_latest_generation(
        output_dir,
        cases,
        candidates_by_frame,
    )
    if recovered is not None:
        return recovered
    if current_error is not None:
        raise current_error
    return {}, {}
