from __future__ import annotations

import copy
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation.legacy_development_l6_motion_cache_repeat import (
    ARTIFACT_NAMES,
    _artifact_comparison,
    _content_comparison,
    _validate_config,
)

CONFIG_PATH = Path(
    "configs/classification_v2/"
    "legacy_development_l6_motion_cache_repeat_gate_v1.json"
)


def test_motion_cache_repeat_config_is_strict() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    _validate_config(payload)

    assert payload["lineage_scope"] == "legacy-only-unreviewed-development"
    assert payload["human_review_complete"] is False


def test_motion_cache_repeat_comparison_detects_drift() -> None:
    manifest_path = Path(
        "outputs/classification_v2/legacy_only_unreviewed_development/"
        "full_legacy_lineage_v2_20260714/16_l6_input_context/"
        "motion_cache_v2/motion_cache_manifest.json"
    )
    primary = json.loads(manifest_path.read_text(encoding="utf-8"))
    repeat = copy.deepcopy(primary)

    artifacts = _artifact_comparison(primary, repeat)
    content = _content_comparison(primary, repeat)

    assert artifacts["valid"]
    assert artifacts["artifact_count"] == len(ARTIFACT_NAMES)
    assert content["valid"]
    repeat["artifacts"]["motion"]["sha256"] = "0" * 64
    assert not _artifact_comparison(primary, repeat)["valid"]
