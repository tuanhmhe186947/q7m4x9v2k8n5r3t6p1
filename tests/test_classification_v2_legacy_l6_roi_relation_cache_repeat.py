from __future__ import annotations

import copy
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation import (
    legacy_development_l6_roi_relation_cache_repeat as repeat_gate,
)

PRIMARY_CONFIG = Path(
    "configs/classification_v2/legacy_development_l6_roi_relation_cache_v1.json"
)
REPEAT_CONFIG = Path(
    "configs/classification_v2/"
    "legacy_development_l6_roi_relation_cache_repeat_v1.json"
)
PRIMARY_MANIFEST = Path(
    "outputs/classification_v2/legacy_only_unreviewed_development/"
    "full_legacy_lineage_v2_20260714/16_l6_input_context/"
    "roi_relation_cache_v1/roi_relation_cache_manifest.json"
)
REPEAT_MANIFEST = Path(
    "outputs/classification_v2/legacy_only_unreviewed_development/"
    "full_legacy_lineage_v2_20260714/16_l6_input_context/"
    "roi_relation_cache_repeat_v1/roi_relation_cache_manifest.json"
)
GATE_CONFIG = Path(
    "configs/classification_v2/"
    "legacy_development_l6_roi_relation_cache_repeat_gate_v1.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_roi_relation_cache_repeat_semantics_include_order_authority() -> None:
    primary = _load(PRIMARY_CONFIG)
    repeat = _load(REPEAT_CONFIG)

    comparison = repeat_gate._semantic_config_comparison(primary, repeat)

    assert comparison["valid"]
    assert "order_authority" in comparison["compared_sections"]
    changed = copy.deepcopy(repeat)
    changed["order_authority"]["geometry_values_used"] = True
    assert not repeat_gate._semantic_config_comparison(primary, changed)["valid"]


def test_roi_relation_cache_repeat_is_byte_and_content_identical() -> None:
    primary = _load(PRIMARY_MANIFEST)
    repeat = _load(REPEAT_MANIFEST)

    artifacts = repeat_gate._artifact_comparison(primary, repeat)
    content = repeat_gate._content_comparison(primary, repeat)

    assert artifacts["valid"]
    assert artifacts["artifact_count"] == len(repeat_gate.ARTIFACT_NAMES)
    assert content["valid"]
    changed = copy.deepcopy(repeat)
    changed["artifacts"]["roi_relation"]["sha256"] = "0" * 64
    assert not repeat_gate._artifact_comparison(primary, changed)["valid"]


def test_roi_relation_cache_repeat_gate_config_is_strict() -> None:
    payload = _load(GATE_CONFIG)

    repeat_gate._validate_config(payload)

    assert payload["lineage_scope"] == "legacy-only-unreviewed-development"
    assert payload["human_review_complete"] is False
    assert payload["primary"]["config_sha256"] != payload["repeat"][
        "config_sha256"
    ]
