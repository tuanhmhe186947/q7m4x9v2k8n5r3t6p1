from __future__ import annotations

import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation.legacy_development_l6_geometry_cache_repeat import (
    _artifact_comparison,
    _content_comparison,
    _semantic_config_comparison,
    _validate_config,
)

PRIMARY_CONFIG = Path(
    "configs/classification_v2/legacy_development_l6_geometry_cache_v1.json"
)
REPEAT_CONFIG = Path(
    "configs/classification_v2/legacy_development_l6_geometry_cache_repeat_v1.json"
)
GATE_CONFIG = Path(
    "configs/classification_v2/"
    "legacy_development_l6_geometry_cache_repeat_gate_v1.json"
)


def test_repeat_configs_change_only_output_and_config_tracking() -> None:
    primary = json.loads(PRIMARY_CONFIG.read_text(encoding="utf-8"))
    repeat = json.loads(REPEAT_CONFIG.read_text(encoding="utf-8"))

    comparison = _semantic_config_comparison(primary, repeat)

    assert comparison["valid"] is True
    assert comparison["different_sections"] == []
    assert comparison["primary_output_root"] != comparison["repeat_output_root"]


def test_artifact_comparison_requires_exact_hash_and_size() -> None:
    primary = _manifest("a")
    repeat = _manifest("a")

    assert _artifact_comparison(primary, repeat)["valid"] is True

    repeat["artifacts"]["geometry"]["sha256"] = "b" * 64
    comparison = _artifact_comparison(primary, repeat)
    assert comparison["valid"] is False
    assert comparison["errors"] == ["cache_artifact_repeat_mismatch=geometry"]


def test_content_comparison_rejects_source_probe_drift() -> None:
    primary = _manifest("a")
    repeat = _manifest("a")

    assert _content_comparison(primary, repeat)["valid"] is True

    repeat["content_audit"]["source_probe"]["status"] = "ESTIMABLE"
    comparison = _content_comparison(primary, repeat)
    assert comparison["valid"] is False
    assert comparison["different_fields"] == ["source_probe"]


def test_repeat_gate_config_preserves_claim_boundary() -> None:
    payload = json.loads(GATE_CONFIG.read_text(encoding="utf-8"))

    _validate_config(payload)

    assert payload["lineage_scope"] == "legacy-only-unreviewed-development"
    assert payload["human_review_complete"] is False
    assert payload["q2_claim_allowed"] is False
    assert payload["canonical_full_oof_authorized"] is False


def _manifest(sha_character: str) -> dict[str, object]:
    artifacts = {
        name: {
            "sha256": sha_character * 64,
            "size_bytes": 100 + index,
        }
        for index, name in enumerate(
            ("geometry", "availability", "window_index", "slot_index")
        )
    }
    content = {
        "model_window_rows": 15_588,
        "model_slot_rows": 93_528,
        "role_window_counts": {"train": 14_608, "validation": 980},
        "available_slots": 93_528,
        "unavailable_slots": 0,
        "geometry_shape": [15_588, 6, 8],
        "availability_shape": [15_588, 6],
        "geometry_dtype": "float32",
        "availability_dtype": "bool",
        "geometry_statistics": {"cx_n": {"mean": 0.5}},
        "ordered_window_id_sha256": "1" * 64,
        "window_index_content_sha256": "2" * 64,
        "slot_index_content_sha256": "3" * 64,
        "reference_audit": {"reference_match": True},
        "source_probe": {
            "status": "NOT_ESTIMABLE_SINGLE_LEGACY_SOURCE",
        },
    }
    return {"artifacts": artifacts, "content_audit": content}
