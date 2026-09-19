from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pig_behavior.classification_v2.review import reviewed_rebuild as rebuild_module
from pig_behavior.classification_v2.review.post_review_learning import (
    build_review_close_authority,
)
from pig_behavior.classification_v2.review.reviewed_rebuild import (
    ReviewedRebuildContractError,
    audit_reviewed_label_overlay,
    build_final_review_autocarry,
    build_reviewed_application_views,
    derive_reviewed_lineage_config,
    freeze_reviewed_training_application_authority,
)


def _scope(prefix: str, count: int) -> pd.DataFrame:
    rows = []
    for index in range(count):
        rows.append(
            {
                "review_unit_id": f"{prefix}-{index}",
                "temporal_unit_key": f"time-{prefix}-{index}",
                "source_type": "cvat_tracking_xml",
                "video_key": f"video-{index % 3}",
                "unit_start_frame": index * 2,
                "unit_end_frame": index * 2 + 1,
                "unit_frame_count": 2,
                "behavior_label": "stand",
            }
        )
    return pd.DataFrame(rows)


def _decisions(scope: pd.DataFrame) -> pd.DataFrame:
    changed = [index % 4 == 0 for index in range(len(scope))]
    return pd.DataFrame(
        {
            "review_unit_id": scope["review_unit_id"],
            "behavior_label": scope["behavior_label"],
            "manual_review_decision": [
                "corrected" if value else "accept" for value in changed
            ],
            "manual_corrected_behavior": [
                "move" if value else "" for value in changed
            ],
            "manual_label_strength": "strong",
        }
    )


def _quality(scope: pd.DataFrame) -> pd.DataFrame:
    changed = [index % 4 == 0 for index in range(len(scope))]
    return pd.DataFrame(
        {
            "review_unit_id": scope["review_unit_id"],
            "original_behavior": "stand",
            "reviewed_behavior": ["move" if value else "stand" for value in changed],
            "label_status": [
                "SOURCE_LABEL_ERROR_CONFIRMED" if value else "SUPPORTED"
                for value in changed
            ],
            "source_label_error_confirmed": [
                "YES" if value else "NO" for value in changed
            ],
            "error_pattern": [
                "OTHER_CLEAR_SOURCE_LABEL_ERROR" if value else "NONE"
                for value in changed
            ],
        }
    )


def _bindings() -> dict[str, dict[str, str]]:
    names = {
        "primary_scope",
        "primary_decisions",
        "primary_quality",
        "control_scope",
        "control_decisions",
        "control_quality",
    }
    return {
        name: {"path": f"outputs/{name}.csv", "sha256": f"{index:064x}"}
        for index, name in enumerate(sorted(names), start=1)
    }


def test_application_authority_is_exact_union_of_primary_and_control() -> None:
    primary = _scope("primary", 4)
    control = _scope("control", 120)
    primary_decisions = _decisions(primary)
    control_decisions = _decisions(control)
    primary_quality = _quality(primary)
    control_quality = _quality(control)
    bindings = _bindings()
    close = build_review_close_authority(
        primary_scope=primary,
        primary_decisions=primary_decisions,
        primary_quality=primary_quality,
        control_scope=control,
        control_decisions=control_decisions,
        control_quality=control_quality,
        artifact_bindings=bindings,
        expected_primary_count=4,
        minimum_control_count=120,
    )
    full_scope = pd.concat([control, primary], ignore_index=True)
    full_decisions = pd.concat(
        [primary_decisions, control_decisions],
        ignore_index=True,
    )
    full_quality = pd.concat([control_quality, primary_quality], ignore_index=True)
    authority = freeze_reviewed_training_application_authority(
        review_close_authority=close,
        primary_scope=primary,
        primary_decisions=primary_decisions,
        primary_quality=primary_quality,
        control_scope=control,
        control_decisions=control_decisions,
        control_quality=control_quality,
        composite_scope=full_scope,
        composite_decisions=full_decisions,
        composite_quality=full_quality,
        corrected_source_authority={"status": "FROZEN"},
        fixed_point_audit={"high_target_rows": 0, "automatic_label_changes": 0},
        artifact_bindings=bindings,
    )
    assert authority["status"] == "FROZEN"
    assert authority["human_reviewed_units"] == 124
    assert authority["candidate_selection_recomputed"] is False
    assert authority["source_label_corrections"] == 31


def _small_overlay_fixture() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    frames = pd.DataFrame(
        {
            "temporal_unit_key": ["u1", "u1", "u2", "u2", "u3", "u3"],
            "source_type": ["cvat"] * 6,
            "video_key": ["video"] * 6,
            "frame_index": [0, 1, 2, 3, 4, 5],
            "behavior": ["stand", "stand", "eat", "eat", "move", "move"],
        }
    )
    scope = pd.DataFrame(
        {
            "review_unit_id": ["r1", "r2"],
            "temporal_unit_key": ["u1", "u2"],
            "source_type": ["cvat", "cvat"],
            "video_key": ["video", "video"],
            "unit_start_frame": [0, 2],
            "unit_end_frame": [1, 3],
            "unit_frame_count": [2, 2],
            "behavior_label": ["social-nose", "eat"],
            "behavior": ["social-nose", "eat"],
        }
    )
    decisions = pd.DataFrame(
        {
            "review_unit_id": ["r1", "r2"],
            "behavior_label": ["social-nose", "eat"],
            "manual_review_decision": ["corrected", "accept"],
            "manual_corrected_behavior": ["fight", ""],
        }
    )
    quality = pd.DataFrame(
        {
            "review_unit_id": ["r1", "r2"],
            "original_behavior": ["social-nose", "eat"],
            "reviewed_behavior": ["fight", "eat"],
        }
    )
    return frames, scope, decisions, quality


def test_derive_config_refreshes_all_source_hash_authorities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = tmp_path / "repository"
    roi_path = (
        repository_root
        / "data"
        / "annotations"
        / "roi"
        / "ROI_annotations.toy_adjusted.coco.json"
    )
    roi_path.parent.mkdir(parents=True)
    roi_path.write_text("{}", encoding="utf-8")
    base_config_path = repository_root / "base.yaml"
    base_config_path.write_text("base: true\n", encoding="utf-8")
    base_config = {
        "source": {
            "legacy_crop_root": "crops",
            "video_root": "videos",
            "expected_crop_fingerprint": "crop-fingerprint",
            "expected_video_fingerprint": "video-fingerprint",
        },
        "authorization": {"source_merge": False},
    }
    observed = {
        "legacy_csv_sha256": "legacy-sha",
        "legacy_csv_rows": 10,
        "completion_audit_sha256": "completion-sha",
        "completion_audit_status": "PASS",
        "crop_file_count": 2,
        "crop_fingerprint": "crop-fingerprint",
        "cvat_xml_count": 3,
        "cvat_xml_fingerprint": "xml-fingerprint",
        "cvat_box_rows": 12,
        "roi_sha256": "roi-sha",
        "pen_mask_sha256": "pen-sha",
        "video_fingerprint": "video-fingerprint",
        "projected_mixed_rows": 22,
        "bundle_fingerprint": "bundle-fingerprint",
    }

    def _fake_source_report(**kwargs: object) -> dict[str, object]:
        config = kwargs["config"]
        assert isinstance(config, dict)
        source = config["source"]
        assert isinstance(source, dict)
        report = dict(observed)
        report["valid"] = (
            source.get("expected_legacy_sha256") == observed["legacy_csv_sha256"]
            and source.get("expected_roi_sha256") == observed["roi_sha256"]
            and source.get("expected_completion_audit_sha256")
            == observed["completion_audit_sha256"]
        )
        return report

    monkeypatch.setattr(
        rebuild_module,
        "_tree_snapshot",
        lambda _path: ([], "unused"),
    )
    monkeypatch.setattr(rebuild_module, "_build_source_report", _fake_source_report)
    derived, manifest = derive_reviewed_lineage_config(
        repository_root=repository_root,
        base_config=base_config,
        base_config_path=base_config_path,
        lineage_id="reviewed-v1",
        run_root=tmp_path / "run",
        scientific_accepted_sha="a" * 40,
        adjusted_roi_path=roi_path,
    )

    source = derived["source"]
    assert source["expected_legacy_sha256"] == "legacy-sha"
    assert source["expected_roi_sha256"] == "roi-sha"
    assert source["expected_completion_audit_sha256"] == "completion-sha"
    assert manifest["source_report"]["roi_sha256"] == "roi-sha"


def test_application_views_adapt_action_but_keep_final_reviewed_label() -> None:
    frames, scope, decisions, quality = _small_overlay_fixture()
    app_scope, app_decisions, audit = build_reviewed_application_views(
        frame_features=frames,
        composite_scope=scope,
        composite_decisions=decisions,
        composite_quality=quality,
    )
    assert app_scope["behavior_label"].tolist() == ["stand", "eat"]
    assert app_decisions["manual_review_decision"].tolist() == [
        "corrected",
        "accept",
    ]
    assert app_decisions["manual_corrected_behavior"].tolist() == ["fight", ""]
    assert audit["fresh_source_snapshot_changed_units"] == 1
    carry, carry_audit = build_final_review_autocarry(frames, app_scope)
    assert carry["temporal_unit_key"].tolist() == ["u3"]
    assert carry_audit["auto_carry_units"] == 1


def test_overlay_audit_rejects_any_unreviewed_label_change() -> None:
    frames, scope, _, quality = _small_overlay_fixture()
    after = frames.copy()
    expected = {"u1": "fight", "u2": "eat", "u3": "move"}
    after["behavior_after_review"] = after["temporal_unit_key"].map(expected)
    after["behavior_reviewed_final"] = after["behavior_after_review"]
    after["behavior_review_label_resolved"] = True
    after["behavior_review_auto_carried"] = after["temporal_unit_key"].eq("u3")
    audit = audit_reviewed_label_overlay(
        before_frames=frames,
        after_frames=after,
        composite_scope=scope,
        composite_quality=quality,
        apply_audit={"missing_review_unit_count": 0},
    )
    assert audit["status"] == "PASS"
    broken = after.copy()
    broken.loc[broken["temporal_unit_key"].eq("u3"), "behavior_after_review"] = "stand"
    with pytest.raises(
        ReviewedRebuildContractError,
        match="unreviewed_label_changed",
    ):
        audit_reviewed_label_overlay(
            before_frames=frames,
            after_frames=broken,
            composite_scope=scope,
            composite_quality=quality,
            apply_audit={"missing_review_unit_count": 0},
        )
