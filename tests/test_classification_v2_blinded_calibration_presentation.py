from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
from PIL import Image

from pig_behavior.classification_v2.review.blinded_calibration_presentation import (
    ACTOR_COLOR,
    CALIBRATION_DECISION_FIELDS,
    NEUTRAL_NEIGHBOR_COLOR,
    PRESENTATION_SEMANTIC_HASH,
    PRESENTATION_VERSION,
    canonical_presentation_payload,
    compose_blinded_contact_sheet,
    presentation_semantic_hash,
    public_display_text,
    render_neutral_context,
    validate_blinded_manifest,
    validate_calibration_decisions,
)


def _public_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "calibration_item_id": "calibration_item_000001",
                "media_authority_key": "media_1",
                "frozen_subset": "CALIBRATION_DEVELOPMENT_SET",
                "presentation_order": 1,
                "presentation_version": PRESENTATION_VERSION,
                "presentation_semantic_hash": PRESENTATION_SEMANTIC_HASH,
                "sampling_config_hash": "a" * 64,
                "semantic_status": "PRE_REVIEW_CALIBRATION_INFRASTRUCTURE",
                "producer_sha": "b" * 40,
                "input_hashes_json": "{}",
            }
        ]
    )


def test_public_manifest_hides_all_machine_hypotheses() -> None:
    audit = validate_blinded_manifest(_public_manifest())
    assert audit["valid"]
    assert not audit["machine_hypothesis_visible"]
    assert not audit["provisional_label_visible"]
    assert not audit["ranked_neighbor_visible"]
    assert not audit["stratum_visible"]


def test_manifest_rejects_provisional_label() -> None:
    manifest = _public_manifest()
    manifest["behavior_label"] = "fight"
    audit = validate_blinded_manifest(manifest)
    assert not audit["valid"]
    assert audit["provisional_label_visible"]


def test_actor_is_red_and_every_neighbor_is_neutral() -> None:
    image = Image.new("RGB", (100, 80), "white")
    rows = pd.DataFrame(
        [
            {
                "object_track_key": "actor",
                "x1": 10,
                "y1": 20,
                "x2": 30,
                "y2": 50,
            },
            {
                "object_track_key": "neighbor_b",
                "x1": 60,
                "y1": 20,
                "x2": 80,
                "y2": 50,
            },
            {
                "object_track_key": "neighbor_a",
                "x1": 35,
                "y1": 20,
                "x2": 55,
                "y2": 50,
            },
        ]
    )
    rendered = render_neutral_context(
        image,
        rows,
        actor_identity="object:actor",
    )
    assert rendered.getpixel((10, 20)) == (255, 0, 0)
    assert rendered.getpixel((35, 20)) == (127, 127, 127)
    assert rendered.getpixel((60, 20)) == (127, 127, 127)
    colors = rendered.getcolors(maxcolors=rendered.width * rendered.height)
    assert colors is not None
    assert (0, 176, 80) not in {color for _, color in colors}
    assert ACTOR_COLOR == "#ff0000"
    assert NEUTRAL_NEIGHBOR_COLOR == "#7f7f7f"


def test_neutral_render_is_deterministic_under_row_permutation() -> None:
    image = Image.new("RGB", (100, 80), "white")
    rows = pd.DataFrame(
        [
            {
                "object_track_key": "actor",
                "x1": 10,
                "y1": 20,
                "x2": 30,
                "y2": 50,
            },
            {
                "object_track_key": "neighbor",
                "x1": 60,
                "y1": 20,
                "x2": 80,
                "y2": 50,
            },
        ]
    )
    first = render_neutral_context(
        image,
        rows,
        actor_identity="object:actor",
    )
    second = render_neutral_context(
        image,
        rows.sample(frac=1.0, random_state=4),
        actor_identity="object:actor",
    )
    assert first.tobytes() == second.tobytes()


def test_contact_sheet_orders_context_before_targets() -> None:
    context = Image.new("RGB", (40, 40), "blue")
    target = Image.new("RGB", (40, 40), "orange")
    sheet = compose_blinded_contact_sheet(
        [
            ("TARGET", 11, target, "ok"),
            ("CONTEXT", 4, context, "ok"),
        ],
        thumb_width=120,
        thumb_height=80,
    )
    assert sheet.getpixel((60, 40)) == (0, 0, 255)
    assert sheet.getpixel((180, 40)) == (255, 165, 0)


def test_public_text_has_no_source_or_machine_fields() -> None:
    text = public_display_text(
        item_number=1,
        item_count=10,
        calibration_item_id="calibration_item_000001",
        target_count=6,
        context_count=4,
    ).casefold()
    for token in (
        "fight",
        "social-nose",
        "candidate",
        "reason",
        "score",
        "video",
        "source",
        "partner rank",
    ):
        assert token not in text
    assert "not a verified" not in text
    assert "no interaction partner has been verified" in text


def test_every_visible_semantic_change_changes_hash() -> None:
    payload = canonical_presentation_payload()
    original = presentation_semantic_hash(payload)
    for field in payload:
        changed = deepcopy(payload)
        value = changed[field]
        if isinstance(value, bool):
            changed[field] = not value
        elif isinstance(value, dict):
            changed[field] = {**value, "_test_change": True}
        elif isinstance(value, list):
            changed[field] = [*value, "_test_change"]
        else:
            changed[field] = f"{value}_test_change"
        assert presentation_semantic_hash(changed) != original


def test_calibration_decision_schema_is_isolated_and_valid() -> None:
    record = {
        "review_key": "review_1",
        "calibration_item_id": "calibration_item_000001",
        "reviewed_behavior": "fight",
        "visual_reviewability": "reviewable",
        "review_confidence": "high",
        "optional_short_note": "",
        "presentation_version": PRESENTATION_VERSION,
        "presentation_semantic_hash": PRESENTATION_SEMANTIC_HASH,
        "reviewer": "reviewer01",
        "decision_timestamp": "2026-07-29T00:00:00+00:00",
    }
    assert tuple(record) == CALIBRATION_DECISION_FIELDS
    assert validate_calibration_decisions(pd.DataFrame([record]))["valid"]
    assert "manual_review_decision" not in record


def test_noninteractive_render_never_writes_decisions(tmp_path: Path) -> None:
    image = Image.new("RGB", (20, 20), "white")
    compose_blinded_contact_sheet(
        [("TARGET", 1, image, "ok")],
        thumb_width=40,
        thumb_height=40,
    )
    assert list(tmp_path.iterdir()) == []
