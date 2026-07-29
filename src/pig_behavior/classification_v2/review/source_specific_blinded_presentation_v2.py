"""Source-specific blinded interaction-calibration presentation V2."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw

PRESENTATION_VERSION = "interaction_blind_calibration_presentation.v2"
OLD_PRESENTATION_HASH = (
    "9eba97958b100e18bef7e8a216e0bd890e4f7cb2e6777c1eda90c6641b76fd3f"
)
MEDIA_AUTHORITY_SCHEMA_VERSION = (
    "classification_v2.calibration_media_authority.v2"
)
PRESENTATION_TEMPLATE = "source_specific_interaction_blind"
CVAT_CONTEXT_MODE = "cvat_full_frame_context"
LEGACY_CONTEXT_MODE = "legacy_actor_crop_only"
CVAT_RENDER_MODE = "full_frame_neutral_context"
LEGACY_RENDER_MODE = "actor_crop_only"
CVAT_ACTOR_SEMANTICS = "red_bbox_is_reviewed_actor"
LEGACY_ACTOR_SEMANTICS = "entire_crop_is_reviewed_actor"
ACTOR_COLOR = "#ff0000"
NEUTRAL_NEIGHBOR_COLOR = "#7f7f7f"
FRAME_BORDER_COLOR = "#202020"
CONTEXT_BAND_COLOR = "#d9e2f3"
TARGET_BAND_COLOR = "#fff2cc"
CVAT_LEGEND_TEXT = (
    "Neutral outlines show geometric context only. "
    "No interaction partner has been verified."
)
LEGACY_NOTICE_TEXT = (
    "Actor crop only. Full interaction context may be unavailable."
)
CONTEXT_HEADING = "CONTEXT — NOT DECISION TARGET"
TARGET_HEADING = "DECISION TARGET"
FRAME_ORDER_CONTRACT = (
    "HISTORY_ASCENDING_THEN_TARGET_ASCENDING_NO_DUPLICATES"
)
MISSING_TEMPLATE_BEHAVIOR = "FAIL_CLOSED_NO_RENDER_FALLBACK"
MISSING_MEDIA_BEHAVIOR = (
    "SHOW_MEDIA_UNAVAILABLE_AND_REQUIRE_UNRESOLVED_OR_TECHNICAL_DEFECT"
)
BEHAVIOR_VOCABULARY = (
    "drink",
    "eat",
    "explore",
    "fight",
    "lying",
    "move",
    "playwithtoy",
    "sitting",
    "social-nose",
    "stand",
    "unclear",
    "unreviewable",
)
REVIEWABLE_BEHAVIORS = frozenset(BEHAVIOR_VOCABULARY[:10])
VISUAL_REVIEWABILITY_VALUES = (
    "reviewable",
    "visually_unresolved",
    "technical_authority_defect",
)
CALIBRATION_OUTCOME_VALUES = (
    "CORRECTION_REQUIRED",
    "LABEL_SUPPORTED",
    "VISUALLY_UNRESOLVED",
    "TECHNICAL_AUTHORITY_DEFECT",
)
REVIEW_CONFIDENCE_VALUES = ("high", "medium", "low")
CALIBRATION_DECISION_FIELDS = (
    "review_key",
    "calibration_item_id",
    "reviewed_behavior",
    "visual_reviewability",
    "review_confidence",
    "optional_short_note",
    "presentation_version",
    "presentation_semantic_hash",
    "reviewer",
    "decision_timestamp",
)
MEDIA_AUTHORITY_REQUIRED_FIELDS = (
    "review_key",
    "split",
    "source_type",
    "context_mode",
    "presentation_template",
    "presentation_version",
    "presentation_semantic_hash",
    "render_mode",
    "actor_identity_semantics",
    "neighbor_context_available",
    "full_frame_context_available",
    "target_frame_indices",
    "history_frame_indices",
    "display_frame_indices",
    "target_frame_count",
    "history_frame_count",
    "frame_order_contract",
    "media_authority",
    "render_available",
    "render_failure_reason",
)


class SourceSpecificPresentationError(ValueError):
    """Raised when V2 source or presentation semantics cannot be proven."""


def source_mode_contracts() -> dict[str, dict[str, Any]]:
    """Return source-conditioned render semantics."""

    return {
        CVAT_CONTEXT_MODE: {
            "source_type": "cvat_tracking_xml",
            "render_mode": CVAT_RENDER_MODE,
            "actor_identity_semantics": CVAT_ACTOR_SEMANTICS,
            "neighbor_context_available": True,
            "full_frame_context_available": True,
            "actor_color": ACTOR_COLOR,
            "neighbor_color": NEUTRAL_NEIGHBOR_COLOR,
            "visible_notice": CVAT_LEGEND_TEXT,
            "target_frame_count": 6,
            "fabricated_overlay_allowed": False,
        },
        LEGACY_CONTEXT_MODE: {
            "source_type": "legacy_recovered",
            "render_mode": LEGACY_RENDER_MODE,
            "actor_identity_semantics": LEGACY_ACTOR_SEMANTICS,
            "neighbor_context_available": False,
            "full_frame_context_available": False,
            "actor_color": None,
            "neighbor_color": None,
            "visible_notice": LEGACY_NOTICE_TEXT,
            "target_frame_count": 16,
            "fabricated_overlay_allowed": False,
        },
    }


def canonical_presentation_contract_v2() -> dict[str, Any]:
    """Return every declared semantic input bound by the V2 hash."""

    return {
        "presentation_version": PRESENTATION_VERSION,
        "media_authority_schema": {
            "schema_version": MEDIA_AUTHORITY_SCHEMA_VERSION,
            "required_fields": list(MEDIA_AUTHORITY_REQUIRED_FIELDS),
        },
        "source_modes": source_mode_contracts(),
        "presentation_template": PRESENTATION_TEMPLATE,
        "frame_border_color": FRAME_BORDER_COLOR,
        "context_band_color": CONTEXT_BAND_COLOR,
        "target_band_color": TARGET_BAND_COLOR,
        "ranking_visibility": "HIDDEN",
        "provisional_label_visibility": "HIDDEN",
        "machine_reason_visibility": "HIDDEN",
        "candidate_tier_visibility": "HIDDEN",
        "machine_score_visibility": "HIDDEN",
        "source_date_video_stratum_visibility": "HIDDEN",
        "target_history_headings": {
            "context": CONTEXT_HEADING,
            "target": TARGET_HEADING,
        },
        "frame_order_contract": FRAME_ORDER_CONTRACT,
        "decision_schema": {
            "fields": list(CALIBRATION_DECISION_FIELDS),
            "reviewed_behavior_values": list(BEHAVIOR_VOCABULARY),
            "visual_reviewability_values": list(
                VISUAL_REVIEWABILITY_VALUES
            ),
            "review_confidence_values": list(REVIEW_CONFIDENCE_VALUES),
            "derived_outcomes": list(CALIBRATION_OUTCOME_VALUES),
            "visually_unresolved_is_label_supported": False,
            "visually_unresolved_is_auto_carry": False,
            "partner_annotation_required": False,
            "role": "CALIBRATION_ONLY_NOT_PRODUCTION_BEHAVIOR_LEDGER",
        },
        "missing_template_behavior": MISSING_TEMPLATE_BEHAVIOR,
        "missing_media_behavior": MISSING_MEDIA_BEHAVIOR,
        "renderer_dispatch": (
            "EXPLICIT_CONTEXT_MODE_AND_RENDER_MODE_NO_FALLBACK"
        ),
    }


def presentation_semantic_hash_v2(
    contract: dict[str, Any] | None = None,
) -> str:
    """Hash the complete canonical V2 presentation contract."""

    encoded = json.dumps(
        contract or canonical_presentation_contract_v2(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


PRESENTATION_SEMANTIC_HASH = presentation_semantic_hash_v2()


def parse_frame_indices(value: object) -> list[int]:
    """Parse a comma-separated frame contract without reordering."""

    if pd.isna(value):
        return []
    frames: list[int] = []
    for token in str(value).split(","):
        normalized = token.strip()
        if normalized:
            frames.append(int(float(normalized)))
    return frames


def format_frame_indices(frames: Iterable[int]) -> str:
    """Serialize frame indices canonically."""

    return ",".join(str(int(frame)) for frame in frames)


def _normalized_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalized_bool(value: object) -> bool:
    """Parse a serialized Boolean without Python's truthy-string defect."""

    normalized = _normalized_text(value).casefold()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise SourceSpecificPresentationError(
        f"invalid Boolean value={normalized or 'blank'}"
    )


def source_dispatch(source_type: object) -> dict[str, Any]:
    """Resolve a supported source to one exact render contract."""

    source = _normalized_text(source_type)
    for context_mode, contract in source_mode_contracts().items():
        if source == contract["source_type"]:
            return {"context_mode": context_mode, **contract}
    raise SourceSpecificPresentationError(
        f"unsupported calibration source_type={source or 'blank'}"
    )


def _valid_order(frames: list[int]) -> bool:
    return frames == sorted(frames) and len(frames) == len(set(frames))


def build_media_authority_v2(
    blinded_manifest_v1: pd.DataFrame,
    media_authority_v1: pd.DataFrame,
    *,
    producer_sha: str,
    input_hashes: dict[str, str],
) -> pd.DataFrame:
    """Transform the frozen 480 rows without changing keys, split, or order."""

    if len(producer_sha) != 40:
        raise SourceSpecificPresentationError(
            "producer_sha must be a full Git SHA"
        )
    required_blinded = {
        "calibration_item_id",
        "media_authority_key",
        "frozen_subset",
        "presentation_order",
        "sampling_config_hash",
    }
    required_media = {
        "calibration_item_id",
        "media_authority_key",
        "review_unit_id",
        "temporal_unit_key",
        "source_type",
        "dataset_id",
        "video_key",
        "recording_date",
        "object_track_key",
        "pig_id",
        "track_id",
        "unit_start_frame",
        "unit_end_frame",
        "display_frame_indices",
        "review_pig_history_display_frame_indices",
    }
    missing_blinded = sorted(required_blinded.difference(blinded_manifest_v1))
    missing_media = sorted(required_media.difference(media_authority_v1))
    if missing_blinded or missing_media:
        raise SourceSpecificPresentationError(
            "missing frozen inputs: "
            f"blinded={missing_blinded} media={missing_media}"
        )
    if len(blinded_manifest_v1) != len(media_authority_v1):
        raise SourceSpecificPresentationError("frozen row count mismatch")
    joined = blinded_manifest_v1.merge(
        media_authority_v1,
        on=["calibration_item_id", "media_authority_key"],
        how="left",
        validate="one_to_one",
        suffixes=("_blind", ""),
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise SourceSpecificPresentationError(
            "missing frozen private media rows"
        )

    records: list[dict[str, Any]] = []
    input_hashes_json = json.dumps(
        dict(sorted(input_hashes.items())),
        sort_keys=True,
        separators=(",", ":"),
    )
    for row in joined.itertuples(index=False):
        dispatch = source_dispatch(row.source_type)
        targets = parse_frame_indices(row.display_frame_indices)
        raw_history = parse_frame_indices(
            row.review_pig_history_display_frame_indices
        )
        target_set = set(targets)
        history = [
            frame for frame in raw_history if frame not in target_set
        ]
        display = [*history, *targets]
        failures: list[str] = []
        expected_target_count = int(dispatch["target_frame_count"])
        if len(targets) != expected_target_count:
            failures.append(
                "target_frame_count="
                f"{len(targets)} expected={expected_target_count}"
            )
        if not _valid_order(targets):
            failures.append("invalid_target_frame_order")
        if not _valid_order(history):
            failures.append("invalid_history_frame_order")
        if set(history).intersection(targets):
            failures.append("history_target_overlap")
        if history and targets and max(history) >= min(targets):
            failures.append("history_not_before_target")
        if len(display) != len(set(display)):
            failures.append("duplicate_display_frames")
        records.append(
            {
                "calibration_item_id": row.calibration_item_id,
                "media_authority_key": row.media_authority_key,
                "review_key": row.review_unit_id,
                "temporal_unit_key": row.temporal_unit_key,
                "split": row.frozen_subset,
                "presentation_order": int(row.presentation_order),
                "sampling_config_hash": row.sampling_config_hash,
                "source_type": row.source_type,
                "dataset_id": row.dataset_id,
                "video_key": row.video_key,
                "recording_date": row.recording_date,
                "object_track_key": row.object_track_key,
                "pig_id": row.pig_id,
                "track_id": row.track_id,
                "unit_start_frame": int(row.unit_start_frame),
                "unit_end_frame": int(row.unit_end_frame),
                "context_mode": dispatch["context_mode"],
                "presentation_template": PRESENTATION_TEMPLATE,
                "presentation_version": PRESENTATION_VERSION,
                "presentation_semantic_hash": (
                    PRESENTATION_SEMANTIC_HASH
                ),
                "render_mode": dispatch["render_mode"],
                "actor_identity_semantics": dispatch[
                    "actor_identity_semantics"
                ],
                "neighbor_context_available": bool(
                    dispatch["neighbor_context_available"]
                ),
                "full_frame_context_available": bool(
                    dispatch["full_frame_context_available"]
                ),
                "target_frame_indices": format_frame_indices(targets),
                "history_frame_indices": format_frame_indices(history),
                "raw_history_frame_indices": format_frame_indices(
                    raw_history
                ),
                "display_frame_indices": format_frame_indices(display),
                "target_frame_count": len(targets),
                "history_frame_count": len(history),
                "display_frame_count": len(display),
                "frame_order_contract": FRAME_ORDER_CONTRACT,
                "media_authority": row.media_authority_key,
                "render_available": not failures,
                "render_failure_reason": ";".join(failures),
                "semantic_status": (
                    "PRE_REVIEW_SOURCE_SPECIFIC_PRESENTATION_V2"
                ),
                "producer_sha": producer_sha,
                "input_hashes_json": input_hashes_json,
            }
        )
    return pd.DataFrame.from_records(records).reset_index(drop=True)


def validate_media_authority_v2(
    media: pd.DataFrame,
    *,
    require_render_available: bool = False,
) -> dict[str, Any]:
    """Fail closed on missing, unknown, or source-inconsistent dispatch."""

    errors: list[str] = []
    missing = sorted(
        set(MEDIA_AUTHORITY_REQUIRED_FIELDS).difference(media.columns)
    )
    if missing:
        return {
            "valid": False,
            "errors": [f"missing_media_authority_fields={missing}"],
        }
    keys = media["review_key"].fillna("").astype(str).str.strip()
    if keys.eq("").any():
        errors.append("blank_review_keys")
    if keys.duplicated().any():
        errors.append("duplicate_review_keys")
    if not media["presentation_version"].eq(PRESENTATION_VERSION).all():
        errors.append("presentation_version_mismatch")
    if not media["presentation_semantic_hash"].eq(
        PRESENTATION_SEMANTIC_HASH
    ).all():
        errors.append("presentation_semantic_hash_mismatch")
    if not media["presentation_template"].eq(
        PRESENTATION_TEMPLATE
    ).all():
        errors.append("missing_or_unknown_presentation_template")

    row_failures = 0
    for row in media.itertuples(index=False):
        try:
            dispatch = source_dispatch(row.source_type)
        except SourceSpecificPresentationError:
            row_failures += 1
            continue
        expected = {
            "context_mode": dispatch["context_mode"],
            "render_mode": dispatch["render_mode"],
            "actor_identity_semantics": dispatch[
                "actor_identity_semantics"
            ],
            "neighbor_context_available": bool(
                dispatch["neighbor_context_available"]
            ),
            "full_frame_context_available": bool(
                dispatch["full_frame_context_available"]
            ),
        }
        try:
            neighbor_available = normalized_bool(
                row.neighbor_context_available
            )
            full_frame_available = normalized_bool(
                row.full_frame_context_available
            )
        except SourceSpecificPresentationError:
            row_failures += 1
            continue
        observed = {
            "context_mode": row.context_mode,
            "render_mode": row.render_mode,
            "actor_identity_semantics": row.actor_identity_semantics,
            "neighbor_context_available": neighbor_available,
            "full_frame_context_available": full_frame_available,
        }
        targets = parse_frame_indices(row.target_frame_indices)
        history = parse_frame_indices(row.history_frame_indices)
        display = parse_frame_indices(row.display_frame_indices)
        valid = (
            observed == expected
            and display == [*history, *targets]
            and len(display) == len(set(display))
            and _valid_order(targets)
            and _valid_order(history)
            and int(row.target_frame_count) == len(targets)
            and int(row.history_frame_count) == len(history)
            and row.frame_order_contract == FRAME_ORDER_CONTRACT
            and len(targets) == int(dispatch["target_frame_count"])
        )
        if not valid:
            row_failures += 1
    if row_failures:
        errors.append(f"invalid_source_dispatch_rows={row_failures}")
    render_available = media["render_available"].map(
        lambda value: str(value).strip().casefold()
        in {"1", "true", "yes", "y"}
    )
    if require_render_available and not render_available.all():
        errors.append(
            "render_unavailable_rows="
            f"{int((~render_available).sum())}"
        )
    blank_failure = (
        media["render_failure_reason"].fillna("").astype(str).str.strip().eq("")
    )
    inconsistent_failure = render_available.eq(blank_failure)
    if not inconsistent_failure.all():
        errors.append(
            "render_failure_semantics_rows="
            f"{int((~inconsistent_failure).sum())}"
        )
    return {
        "valid": not errors,
        "errors": errors,
        "row_count": int(len(media)),
        "duplicate_review_keys": int(keys.duplicated().sum()),
        "missing_presentation_template_rows": int(
            media["presentation_template"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        ),
        "missing_context_mode_rows": int(
            media["context_mode"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        ),
        "missing_render_mode_rows": int(
            media["render_mode"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        ),
        "blank_presentation_hash_rows": int(
            media["presentation_semantic_hash"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        ),
        "invalid_source_dispatch_rows": row_failures,
    }


def frozen_identity_check(
    blinded_manifest_v1: pd.DataFrame,
    media_v2: pd.DataFrame,
) -> dict[str, Any]:
    """Prove V2 changes presentation only, never the frozen sample."""

    old = blinded_manifest_v1[
        [
            "calibration_item_id",
            "frozen_subset",
            "presentation_order",
        ]
    ].copy()
    old = old.rename(columns={"frozen_subset": "split"})
    new = media_v2[
        ["calibration_item_id", "split", "presentation_order"]
    ].copy()
    joined = old.merge(
        new,
        on="calibration_item_id",
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True,
    )
    missing = int(joined["_merge"].ne("both").sum())
    split_changed = int(
        joined["split_old"].fillna("").ne(
            joined["split_new"].fillna("")
        ).sum()
    )
    order_changed = int(
        pd.to_numeric(joined["presentation_order_old"], errors="coerce")
        .ne(
            pd.to_numeric(
                joined["presentation_order_new"],
                errors="coerce",
            )
        )
        .sum()
    )
    development_name = "CALIBRATION_DEVELOPMENT_SET"
    confirmation_name = "BLINDED_CONFIRMATION_SET"
    development = joined.loc[joined["split_old"].eq(development_name)]
    confirmation = joined.loc[joined["split_old"].eq(confirmation_name)]
    development_key_changes = int(
        development["_merge"].ne("both").sum()
    )
    confirmation_key_changes = int(
        confirmation["_merge"].ne("both").sum()
    )
    unmatched_new = joined.loc[joined["_merge"].eq("right_only")]
    development_key_changes += int(
        unmatched_new["split_new"].eq(development_name).sum()
    )
    confirmation_key_changes += int(
        unmatched_new["split_new"].eq(confirmation_name).sum()
    )
    return {
        "valid": not (missing or split_changed or order_changed),
        "development_count": int(len(development)),
        "confirmation_count": int(len(confirmation)),
        "development_keys_changed": development_key_changes,
        "confirmation_keys_changed": confirmation_key_changes,
        "split_membership_changed": split_changed,
        "presentation_order_changed": order_changed,
    }


def apply_preflight_availability(
    media: pd.DataFrame,
    preflight: pd.DataFrame,
) -> pd.DataFrame:
    """Bind actual preflight availability without changing frozen identity."""

    required = {"review_key", "reviewable", "failure_reason"}
    missing = sorted(required.difference(preflight.columns))
    if missing:
        raise SourceSpecificPresentationError(
            f"preflight missing fields={missing}"
        )
    availability = preflight[list(required)].copy()
    if availability["review_key"].duplicated().any():
        raise SourceSpecificPresentationError(
            "preflight has duplicate review keys"
        )
    output = media.drop(
        columns=["render_available", "render_failure_reason"]
    ).merge(
        availability,
        on="review_key",
        how="left",
        validate="one_to_one",
    )
    if output["reviewable"].isna().any():
        raise SourceSpecificPresentationError(
            "preflight missing media-authority keys"
        )
    output = output.rename(
        columns={
            "reviewable": "render_available",
            "failure_reason": "render_failure_reason",
        }
    )
    return output[media.columns].copy()


def _box(row: pd.Series) -> tuple[int, int, int, int] | None:
    values = pd.to_numeric(
        pd.Series(
            [row.get("x1"), row.get("y1"), row.get("x2"), row.get("y2")]
        ),
        errors="coerce",
    )
    if (
        values.isna().any()
        or values.iloc[2] <= values.iloc[0]
        or values.iloc[3] <= values.iloc[1]
    ):
        return None
    return tuple(int(round(value)) for value in values.tolist())


def local_context_identity(row: pd.Series) -> str:
    """Return a local actor join key that is never displayed or modeled."""

    object_key = _normalized_text(row.get("object_track_key", ""))
    if object_key.casefold() not in {"", "nan", "none", "<na>"}:
        return f"object:{object_key}"
    track = _normalized_text(row.get("track_id", ""))
    pig = _normalized_text(row.get("pig_id", ""))
    return f"track:{track}|pig:{pig}"


def render_neutral_context_v2(
    image: Image.Image,
    scene_rows: pd.DataFrame,
    *,
    actor_identity: str,
) -> Image.Image:
    """Draw one red actor and every valid non-actor identically."""

    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    rows = scene_rows.copy()
    rows["_identity"] = rows.apply(local_context_identity, axis=1)
    actor_rows = rows.loc[rows["_identity"].eq(actor_identity)]
    if len(actor_rows) != 1:
        raise SourceSpecificPresentationError(
            f"expected one actor row, observed={len(actor_rows)}"
        )
    if _box(actor_rows.iloc[0]) is None:
        raise SourceSpecificPresentationError("actor bbox is invalid")
    rows = rows.sort_values("_identity", kind="stable")
    for _, row in rows.iterrows():
        coordinates = _box(row)
        if coordinates is None:
            continue
        is_actor = str(row["_identity"]) == actor_identity
        color = ACTOR_COLOR if is_actor else NEUTRAL_NEIGHBOR_COLOR
        width = 4 if is_actor else 2
        draw.rectangle(coordinates, outline=color, width=width)
        role = "ACTOR" if is_actor else "NEARBY"
        draw.text(
            (coordinates[0] + 3, max(0, coordinates[1] - 13)),
            role,
            fill=color,
        )
    return output


def visible_notice(context_mode: str) -> str:
    """Return only the source-specific limitation visible to reviewers."""

    contracts = source_mode_contracts()
    if context_mode not in contracts:
        raise SourceSpecificPresentationError(
            f"unknown context_mode={context_mode or 'blank'}"
        )
    return str(contracts[context_mode]["visible_notice"])


def compose_source_specific_contact_sheet(
    frames: Iterable[tuple[str, int, Image.Image, str]],
    *,
    context_mode: str,
    thumb_width: int = 220,
    thumb_height: int = 160,
) -> Image.Image:
    """Compose frames with source notice and explicit decision boundaries."""

    items = list(frames)
    if not items:
        raise SourceSpecificPresentationError(
            "contact sheet requires at least one frame"
        )
    roles = [item[0] for item in items]
    if any(role not in {"CONTEXT", "TARGET"} for role in roles):
        raise SourceSpecificPresentationError("invalid frame role")
    role_values = [0 if role == "CONTEXT" else 1 for role in roles]
    if role_values != sorted(role_values):
        raise SourceSpecificPresentationError(
            "context frames must precede target frames"
        )
    for role in ("CONTEXT", "TARGET"):
        indices = [
            frame_index
            for item_role, frame_index, _, _ in items
            if item_role == role
        ]
        if not _valid_order(indices):
            raise SourceSpecificPresentationError(
                f"invalid {role.casefold()} frame order"
            )
    all_indices = [item[1] for item in items]
    if len(all_indices) != len(set(all_indices)):
        raise SourceSpecificPresentationError(
            "duplicate display frame indices"
        )

    columns = 3 if len(items) <= 6 else 4
    row_count = math.ceil(len(items) / columns)
    notice_height = 42
    sheet = Image.new(
        "RGB",
        (
            columns * thumb_width,
            notice_height + row_count * thumb_height,
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    notice = visible_notice(context_mode)
    draw.rectangle(
        [0, 0, sheet.width - 1, notice_height - 1],
        fill="#f2f2f2",
        outline=FRAME_BORDER_COLOR,
    )
    draw.text((8, 13), notice, fill=FRAME_BORDER_COLOR)
    for index, (role, frame_index, image, status) in enumerate(items):
        x = (index % columns) * thumb_width
        y = notice_height + (index // columns) * thumb_height
        band = (
            CONTEXT_BAND_COLOR if role == "CONTEXT" else TARGET_BAND_COLOR
        )
        draw.rectangle(
            [x, y, x + thumb_width - 1, y + 25],
            fill=band,
            outline=FRAME_BORDER_COLOR,
        )
        heading = CONTEXT_HEADING if role == "CONTEXT" else TARGET_HEADING
        label = f"{heading} · f{frame_index}"
        if status and status != "ok":
            label += " · MEDIA CHECK"
        draw.text((x + 4, y + 5), label, fill=FRAME_BORDER_COLOR)
        fitted = image.copy()
        fitted.thumbnail(
            (thumb_width - 4, thumb_height - 31),
            Image.Resampling.LANCZOS,
        )
        sheet.paste(
            fitted,
            (
                x + (thumb_width - fitted.width) // 2,
                y + 28,
            ),
        )
        draw.rectangle(
            [x, y, x + thumb_width - 1, y + thumb_height - 1],
            outline=FRAME_BORDER_COLOR,
        )
    return sheet


def public_display_text_v2(
    *,
    item_number: int,
    item_count: int,
    calibration_item_id: str,
    target_count: int,
    context_count: int,
    context_mode: str,
) -> str:
    """Return visible text without labels, reasons, ranks, or strata."""

    return "\n".join(
        [
            f"Item {item_number}/{item_count} · {calibration_item_id}",
            f"Decision targets: {target_count}",
            f"Context frames: {context_count}",
            visible_notice(context_mode),
        ]
    )


def validate_calibration_decisions_v2(
    decisions: pd.DataFrame,
) -> dict[str, Any]:
    """Validate isolated V2 decisions without producing human outcomes."""

    errors: list[str] = []
    missing = sorted(
        set(CALIBRATION_DECISION_FIELDS).difference(decisions.columns)
    )
    if missing:
        return {
            "valid": False,
            "errors": [f"missing_calibration_decision_fields={missing}"],
        }
    keys = decisions["review_key"].fillna("").astype(str).str.strip()
    if keys.eq("").any():
        errors.append("blank_review_key")
    if keys.duplicated().any():
        errors.append("duplicate_review_key")
    behavior = decisions["reviewed_behavior"].fillna("").astype(str)
    reviewability = (
        decisions["visual_reviewability"].fillna("").astype(str)
    )
    confidence = decisions["review_confidence"].fillna("").astype(str)
    if not behavior.isin(BEHAVIOR_VOCABULARY).all():
        errors.append("invalid_reviewed_behavior")
    if not reviewability.isin(VISUAL_REVIEWABILITY_VALUES).all():
        errors.append("invalid_visual_reviewability")
    if not confidence.isin(REVIEW_CONFIDENCE_VALUES).all():
        errors.append("invalid_review_confidence")
    reviewable_invalid = reviewability.eq("reviewable") & ~behavior.isin(
        REVIEWABLE_BEHAVIORS
    )
    unresolved_invalid = reviewability.eq(
        "visually_unresolved"
    ) & ~behavior.isin({"unclear", "unreviewable"})
    defect_invalid = reviewability.eq(
        "technical_authority_defect"
    ) & ~behavior.eq("unreviewable")
    if reviewable_invalid.any():
        errors.append("reviewable_item_requires_supported_behavior")
    if unresolved_invalid.any():
        errors.append("visually_unresolved_requires_unclear_behavior")
    if defect_invalid.any():
        errors.append("technical_defect_requires_unreviewable_behavior")
    if not decisions["presentation_version"].eq(PRESENTATION_VERSION).all():
        errors.append("decision_presentation_version_mismatch")
    if not decisions["presentation_semantic_hash"].eq(
        PRESENTATION_SEMANTIC_HASH
    ).all():
        errors.append("decision_presentation_hash_mismatch")
    return {"valid": not errors, "errors": errors}


def derive_calibration_outcome(
    *,
    provisional_behavior: str,
    reviewed_behavior: str,
    visual_reviewability: str,
) -> str:
    """Derive one frozen outcome only after the blinded ledger is closed."""

    if visual_reviewability == "visually_unresolved":
        return "VISUALLY_UNRESOLVED"
    if visual_reviewability == "technical_authority_defect":
        return "TECHNICAL_AUTHORITY_DEFECT"
    if visual_reviewability != "reviewable":
        raise SourceSpecificPresentationError(
            "unknown visual_reviewability"
        )
    if reviewed_behavior not in REVIEWABLE_BEHAVIORS:
        raise SourceSpecificPresentationError(
            "reviewable outcome requires a canonical behavior"
        )
    return (
        "LABEL_SUPPORTED"
        if reviewed_behavior == provisional_behavior
        else "CORRECTION_REQUIRED"
    )


def write_declared_contract_v2(path: Path) -> None:
    """Write the immutable declared V2 contract, never a decision file."""

    payload = {
        "semantic_status": "PRE_REVIEW_SOURCE_SPECIFIC_PRESENTATION_V2",
        "authoritative_for_candidate_membership": False,
        "presentation": canonical_presentation_contract_v2(),
        "presentation_semantic_hash": PRESENTATION_SEMANTIC_HASH,
        "old_presentation_hash": OLD_PRESENTATION_HASH,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ACTOR_COLOR",
    "BEHAVIOR_VOCABULARY",
    "CALIBRATION_DECISION_FIELDS",
    "CALIBRATION_OUTCOME_VALUES",
    "CONTEXT_HEADING",
    "CVAT_CONTEXT_MODE",
    "CVAT_LEGEND_TEXT",
    "CVAT_RENDER_MODE",
    "FRAME_ORDER_CONTRACT",
    "LEGACY_CONTEXT_MODE",
    "LEGACY_NOTICE_TEXT",
    "LEGACY_RENDER_MODE",
    "MEDIA_AUTHORITY_REQUIRED_FIELDS",
    "MEDIA_AUTHORITY_SCHEMA_VERSION",
    "NEUTRAL_NEIGHBOR_COLOR",
    "OLD_PRESENTATION_HASH",
    "PRESENTATION_SEMANTIC_HASH",
    "PRESENTATION_TEMPLATE",
    "PRESENTATION_VERSION",
    "REVIEW_CONFIDENCE_VALUES",
    "SourceSpecificPresentationError",
    "TARGET_HEADING",
    "VISUAL_REVIEWABILITY_VALUES",
    "apply_preflight_availability",
    "build_media_authority_v2",
    "canonical_presentation_contract_v2",
    "compose_source_specific_contact_sheet",
    "derive_calibration_outcome",
    "format_frame_indices",
    "frozen_identity_check",
    "local_context_identity",
    "normalized_bool",
    "parse_frame_indices",
    "presentation_semantic_hash_v2",
    "public_display_text_v2",
    "render_neutral_context_v2",
    "source_dispatch",
    "source_mode_contracts",
    "validate_calibration_decisions_v2",
    "validate_media_authority_v2",
    "visible_notice",
    "write_declared_contract_v2",
]
