from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.tracking import generate_r1_rf_hybrid_predictions as r1


def test_raw_core_guard_accepts_representation_only_internal_fields() -> None:
    expected = [
        {
            "frame": 4,
            "label": "Pig_1",
            "points": [1.0, 2.0, 3.0, 4.0],
            "score": 0.9,
            "attributes": [{"name": "ID", "value": "ID_1"}],
        }
    ]
    rows: list[dict[str, object]] = []
    observed = [{**expected[0], "_raw_track_id": 7, "_track_state": "VISIBLE"}]

    r1.raw_core_guard(expected, "video", rows)(observed)

    assert rows[0]["status"] == "PASS"
    assert rows[0]["bbox_value_changes"] == 0
    assert rows[0]["id_value_changes"] == 0


def test_raw_core_guard_rejects_before_repair_on_public_change() -> None:
    expected = [{"frame": 4, "points": [1.0, 2.0, 3.0, 4.0]}]
    observed = [{"frame": 4, "points": [1.0, 2.0, 3.0, 5.0]}]

    with pytest.raises(r1.R1AuthorityError, match="FAIL_R0_RAW_CORE_PARITY"):
        r1.raw_core_guard(expected, "video", [])(observed)


def test_r1_artifact_inventory_includes_raw_core_and_excludes_audits(
    tmp_path: Path,
) -> None:
    for name in (
        "predictions",
        "machine_readable",
        "raw_core_snapshots",
        "repair_ledgers",
        "commands",
        "audits",
        "manifests",
    ):
        (tmp_path / name).mkdir()
    (tmp_path / "predictions" / "video.xml").write_text("<x/>")
    (tmp_path / "raw_core_snapshots" / "video.json").write_text("{}")
    (tmp_path / "audits" / "self.json").write_text("{}")

    relative = {
        row["relative_path"] for row in r1.r1_artifact_inventory(tmp_path)
    }

    assert "predictions/video.xml" in relative
    assert "raw_core_snapshots/video.json" in relative
    assert "audits/self.json" not in relative


def test_create_output_root_refuses_existing_directory(tmp_path: Path) -> None:
    existing = tmp_path / "authority"
    existing.mkdir()

    with pytest.raises(r1.R1AuthorityError, match="refusing existing"):
        r1.create_output_root(existing)


def test_ledger_summary_is_gt_free_and_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    payload = {
        "repair_config_hash": r1.REPAIR_SEMANTIC_SHA256,
        "events": [
            {
                "repair_stage": "suffix_pair_swap",
                "frames_modified": [10, 11],
                "future_frames_used": True,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    first = r1.ledger_summary(path)
    second = r1.ledger_summary(path)

    assert first == second
    assert first["event_count"] == 1
    assert first["frames_modified"] == 2
    assert first["future_frame_events"] == 1


def test_runner_raw_core_guard_precedes_adapter_and_repair() -> None:
    source = (
        Path("src/pig_behavior/tracking/runner.py")
        .read_text(encoding="utf-8")
    )
    guard_call = source.index("rf_raw_core_guard(raw_snapshot)")
    adapter_call = source.index(
        "adapt_rf_shapes_for_offline_repair(",
        guard_call,
    )
    repair_call = source.index("apply_offline_repair_stack(", adapter_call)

    assert guard_call < adapter_call < repair_call
