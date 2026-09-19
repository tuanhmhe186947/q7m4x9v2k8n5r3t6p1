"""Blinded, neutral presentation primitives for interaction calibration."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw

PRESENTATION_VERSION = "interaction_blind_calibration_presentation.v1"
ACTOR_COLOR = "#ff0000"
NEUTRAL_NEIGHBOR_COLOR = "#7f7f7f"
FRAME_BORDER_COLOR = "#202020"
CONTEXT_BAND_COLOR = "#d9e2f3"
TARGET_BAND_COLOR = "#fff2cc"
LEGEND_TEXT = (
    "Neutral outlines show nearby geometric context only. "
    "No interaction partner has been verified."
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
VISUAL_REVIEWABILITY_VALUES = (
    "reviewable",
    "visually_unresolved",
    "technical_authority_defect",
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
PUBLIC_MANIFEST_REQUIRED_FIELDS = (
    "calibration_item_id",
    "media_authority_key",
    "frozen_subset",
    "presentation_order",
    "presentation_version",
    "presentation_semantic_hash",
    "sampling_config_hash",
    "semantic_status",
    "producer_sha",
    "input_hashes_json",
)
PUBLIC_MANIFEST_FORBIDDEN_TOKENS = (
    "behavior",
    "candidate",
    "contact",
    "crowd",
    "date",
    "label",
    "partner",
    "reason",
    "score",
    "selector",
    "source",
    "stratum",
    "video",
)


class BlindedPresentationError(ValueError):
    """Raised when presentation or decision semantics fail closed."""


def canonical_presentation_payload() -> dict[str, Any]:
    """Return every visible semantic field bound by the presentation hash."""

    return {
        "presentation_version": PRESENTATION_VERSION,
        "actor_color": ACTOR_COLOR,
        "neutral_neighbor_color": NEUTRAL_NEIGHBOR_COLOR,
        "frame_border_color": FRAME_BORDER_COLOR,
        "context_band_color": CONTEXT_BAND_COLOR,
        "target_band_color": TARGET_BAND_COLOR,
        "ranking_visibility": "HIDDEN",
        "machine_reason_visibility": "HIDDEN",
        "provisional_label_visibility": "HIDDEN",
        "technical_detail_visibility": "HIDDEN",
        "source_video_date_visibility": "HIDDEN",
        "history_target_separation": (
            "CONTEXT_NOT_DECISION_TARGET_VS_DECISION_TARGET"
        ),
        "legend_text": LEGEND_TEXT,
        "default_zoom_layout": {
            "window": "1380x900",
            "minimum": "1100x740",
            "thumbnail": [220, 160],
            "columns_up_to_6": 3,
            "columns_over_6": 4,
        },
        "frame_ordering": "HISTORY_ASCENDING_THEN_TARGET_ASCENDING",
        "decision_field_schema": list(CALIBRATION_DECISION_FIELDS),
        "reviewed_behavior_values": list(BEHAVIOR_VOCABULARY),
        "visual_reviewability_values": list(VISUAL_REVIEWABILITY_VALUES),
        "review_confidence_values": list(REVIEW_CONFIDENCE_VALUES),
        "partner_annotation_required": False,
        "decision_schema_role": (
            "CALIBRATION_ONLY_NOT_PRODUCTION_BEHAVIOR_LEDGER"
        ),
    }


def presentation_semantic_hash(
    payload: dict[str, Any] | None = None,
) -> str:
    """Hash canonical visible semantics."""

    encoded = json.dumps(
        payload or canonical_presentation_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


PRESENTATION_SEMANTIC_HASH = presentation_semantic_hash()


def validate_blinded_manifest(manifest: pd.DataFrame) -> dict[str, Any]:
    """Reject machine-hypothesis fields and semantic-hash drift."""

    errors: list[str] = []
    missing = sorted(
        set(PUBLIC_MANIFEST_REQUIRED_FIELDS).difference(manifest.columns)
    )
    if missing:
        errors.append(f"missing_public_manifest_fields={missing}")
    forbidden = sorted(
        column
        for column in manifest.columns
        if any(
            token in column.casefold()
            for token in PUBLIC_MANIFEST_FORBIDDEN_TOKENS
        )
    )
    if forbidden:
        errors.append(f"machine_hypothesis_columns_visible={forbidden}")
    if "calibration_item_id" in manifest.columns:
        ids = manifest["calibration_item_id"].fillna("").astype(str)
        if ids.eq("").any():
            errors.append("blank_calibration_item_id")
        if ids.duplicated().any():
            errors.append("duplicate_calibration_item_id")
    if "presentation_version" in manifest.columns:
        if not manifest["presentation_version"].eq(PRESENTATION_VERSION).all():
            errors.append("presentation_version_mismatch")
    if "presentation_semantic_hash" in manifest.columns:
        if not manifest["presentation_semantic_hash"].eq(
            PRESENTATION_SEMANTIC_HASH
        ).all():
            errors.append("presentation_semantic_hash_mismatch")
    return {
        "valid": not errors,
        "errors": errors,
        "machine_hypothesis_visible": bool(forbidden),
        "provisional_label_visible": any(
            "label" in column.casefold() for column in manifest.columns
        ),
        "ranked_neighbor_visible": any(
            "rank" in column.casefold() for column in manifest.columns
        ),
        "stratum_visible": any(
            "stratum" in column.casefold() for column in manifest.columns
        ),
    }


def validate_media_authority(media: pd.DataFrame) -> dict[str, Any]:
    """Validate private media joins without interpreting their content."""

    required = {
        "calibration_item_id",
        "media_authority_key",
        "review_unit_id",
        "temporal_unit_key",
        "source_type",
        "dataset_id",
        "video_key",
        "object_track_key",
        "pig_id",
        "track_id",
        "unit_start_frame",
        "unit_end_frame",
        "display_frame_indices",
        "review_pig_history_display_frame_indices",
        "presentation_version",
        "presentation_semantic_hash",
    }
    errors: list[str] = []
    missing = sorted(required.difference(media.columns))
    if missing:
        errors.append(f"missing_media_authority_fields={missing}")
    if not missing:
        ids = media["calibration_item_id"].fillna("").astype(str)
        if ids.eq("").any() or ids.duplicated().any():
            errors.append("invalid_media_authority_item_ids")
        if not media["presentation_version"].eq(PRESENTATION_VERSION).all():
            errors.append("media_presentation_version_mismatch")
        if not media["presentation_semantic_hash"].eq(
            PRESENTATION_SEMANTIC_HASH
        ).all():
            errors.append("media_presentation_hash_mismatch")
    return {"valid": not errors, "errors": errors}


def join_blinded_media_authority(
    manifest: pd.DataFrame,
    media: pd.DataFrame,
) -> pd.DataFrame:
    """Join opaque public order to private media authority one-to-one."""

    manifest_audit = validate_blinded_manifest(manifest)
    media_audit = validate_media_authority(media)
    errors = manifest_audit["errors"] + media_audit["errors"]
    if errors:
        raise BlindedPresentationError(";".join(errors))
    joined = manifest.merge(
        media,
        on=["calibration_item_id", "media_authority_key"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_media"),
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise BlindedPresentationError("missing private media authority rows")
    joined = joined.drop(columns=["_merge"])
    return joined.sort_values(
        ["frozen_subset", "presentation_order"]
    ).reset_index(drop=True)


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
    """Return local join identity; this value is never drawn or modeled."""

    object_key = str(row.get("object_track_key", "")).strip()
    if object_key.casefold() not in {"", "nan", "none", "<na>"}:
        return f"object:{object_key}"
    track = str(row.get("track_id", "")).strip()
    pig = str(row.get("pig_id", "")).strip()
    return f"track:{track}|pig:{pig}"


def render_neutral_context(
    image: Image.Image,
    scene_rows: pd.DataFrame,
    *,
    actor_identity: str,
) -> Image.Image:
    """Draw one red actor and all valid non-self neighbors identically."""

    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    rows = scene_rows.copy()
    rows["_identity"] = rows.apply(local_context_identity, axis=1)
    actor_rows = rows.loc[rows["_identity"].eq(actor_identity)]
    if len(actor_rows) != 1:
        raise BlindedPresentationError(
            f"expected one actor row, observed={len(actor_rows)}"
        )
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


def _fit(image: Image.Image, width: int, height: int) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail((width, height), Image.Resampling.LANCZOS)
    return fitted


def compose_blinded_contact_sheet(
    frames: Iterable[tuple[str, int, Image.Image, str]],
    *,
    thumb_width: int = 220,
    thumb_height: int = 160,
) -> Image.Image:
    """Compose context first, targets second, with explicit semantic bands."""

    items = list(frames)
    role_order = {"CONTEXT": 0, "TARGET": 1}
    items.sort(key=lambda item: (role_order[item[0]], item[1]))
    if not items:
        raise BlindedPresentationError("contact sheet requires at least one frame")
    columns = 3 if len(items) <= 6 else 4
    row_count = math.ceil(len(items) / columns)
    sheet = Image.new(
        "RGB",
        (columns * thumb_width, row_count * thumb_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, (role, frame_index, image, status) in enumerate(items):
        if role not in role_order:
            raise BlindedPresentationError(f"invalid frame role: {role}")
        x = (index % columns) * thumb_width
        y = (index // columns) * thumb_height
        band = CONTEXT_BAND_COLOR if role == "CONTEXT" else TARGET_BAND_COLOR
        draw.rectangle(
            [x, y, x + thumb_width - 1, y + 25],
            fill=band,
            outline=FRAME_BORDER_COLOR,
        )
        role_text = (
            "CONTEXT — NOT DECISION TARGET"
            if role == "CONTEXT"
            else "DECISION TARGET"
        )
        label = f"{role_text} · f{frame_index}"
        if status and status != "ok":
            label += " · MEDIA CHECK"
        draw.text((x + 4, y + 5), label, fill=FRAME_BORDER_COLOR)
        fitted = _fit(image, thumb_width - 4, thumb_height - 31)
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


def public_display_text(
    *,
    item_number: int,
    item_count: int,
    calibration_item_id: str,
    target_count: int,
    context_count: int,
) -> str:
    """Return reviewer-visible text without a machine or source hypothesis."""

    return "\n".join(
        [
            f"Item {item_number}/{item_count} · {calibration_item_id}",
            f"Decision targets: {target_count}",
            f"Context frames: {context_count}",
            LEGEND_TEXT,
        ]
    )


def validate_calibration_decisions(decisions: pd.DataFrame) -> dict[str, Any]:
    """Validate the isolated calibration ledger schema."""

    errors: list[str] = []
    missing = sorted(set(CALIBRATION_DECISION_FIELDS).difference(decisions.columns))
    if missing:
        errors.append(f"missing_calibration_decision_fields={missing}")
        return {"valid": False, "errors": errors}
    keys = decisions["review_key"].fillna("").astype(str).str.strip()
    if keys.eq("").any():
        errors.append("blank_review_key")
    if keys.duplicated().any():
        errors.append("duplicate_review_key")
    behavior = decisions["reviewed_behavior"].fillna("").astype(str)
    invalid_behavior = ~behavior.isin(BEHAVIOR_VOCABULARY)
    if invalid_behavior.any():
        errors.append(
            f"invalid_reviewed_behavior={int(invalid_behavior.sum())}"
        )
    reviewability = (
        decisions["visual_reviewability"].fillna("").astype(str)
    )
    invalid_reviewability = ~reviewability.isin(VISUAL_REVIEWABILITY_VALUES)
    if invalid_reviewability.any():
        errors.append(
            f"invalid_visual_reviewability={int(invalid_reviewability.sum())}"
        )
    confidence = decisions["review_confidence"].fillna("").astype(str)
    invalid_confidence = ~confidence.isin(REVIEW_CONFIDENCE_VALUES)
    if invalid_confidence.any():
        errors.append(
            f"invalid_review_confidence={int(invalid_confidence.sum())}"
        )
    if not decisions["presentation_version"].eq(PRESENTATION_VERSION).all():
        errors.append("decision_presentation_version_mismatch")
    if not decisions["presentation_semantic_hash"].eq(
        PRESENTATION_SEMANTIC_HASH
    ).all():
        errors.append("decision_presentation_hash_mismatch")
    unsupported_missing = reviewability.eq("reviewable") & behavior.isin(
        {"unclear", "unreviewable"}
    )
    if unsupported_missing.any():
        errors.append(
            "reviewable_item_has_unresolved_behavior="
            f"{int(unsupported_missing.sum())}"
        )
    return {"valid": not errors, "errors": errors}


def write_presentation_schema(path: Path) -> None:
    """Write the stable presentation schema, never a decision ledger."""

    payload = {
        "semantic_status": "PRE_REVIEW_CALIBRATION_INFRASTRUCTURE",
        "authoritative_for_candidate_membership": False,
        "presentation": canonical_presentation_payload(),
        "presentation_semantic_hash": PRESENTATION_SEMANTIC_HASH,
        "public_manifest_required_fields": list(
            PUBLIC_MANIFEST_REQUIRED_FIELDS
        ),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ACTOR_COLOR",
    "BEHAVIOR_VOCABULARY",
    "BlindedPresentationError",
    "CALIBRATION_DECISION_FIELDS",
    "LEGEND_TEXT",
    "NEUTRAL_NEIGHBOR_COLOR",
    "PRESENTATION_SEMANTIC_HASH",
    "PRESENTATION_VERSION",
    "REVIEW_CONFIDENCE_VALUES",
    "VISUAL_REVIEWABILITY_VALUES",
    "canonical_presentation_payload",
    "compose_blinded_contact_sheet",
    "join_blinded_media_authority",
    "local_context_identity",
    "presentation_semantic_hash",
    "public_display_text",
    "render_neutral_context",
    "validate_blinded_manifest",
    "validate_calibration_decisions",
    "validate_media_authority",
    "write_presentation_schema",
]
